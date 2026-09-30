"""Canonical classifier-only reliability-aware graph test-time adaptation.

The implementation follows one differentiable objective exactly:

    sum_i q_i H(p_i)
    + lambda_cal * sum_k (C_k(W) - C_k(W0)) ** 2
    + 0.5 * lambda_af * ||W - W0||_F ** 2,

where ``q_i = r_i / sum_j r_j`` is a detached reliability weight and the three
degree groups are fixed before adaptation. Candidate updates are checked after
the gradient step. Rejected candidates are never deployed; two consecutive
rejections restore the immutable source classifier.
"""

from __future__ import annotations

import copy
from typing import Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from data_adapter import DEVICE, UnlabeledGraphView, adj_to_edge_index
from detector import detector_should_halt
from reliability import (
    degree_group_masks,
    entropy,
    group_confidence,
    neighborhood_agreement,
    reliability_scores,
    source_consistency,
)

_NO_RELIABILITY = {"source_only", "entropy_all_nodes", "tent_entropy"}
_GROUP_NAMES = ("low", "mid", "high")
_SUPPORTED_METHODS = {
    "source_only",
    "entropy_all_nodes",
    "tent_entropy",
    "confidence_source_entropy",
    "full_method",
    "no_neighborhood_agreement",
    "no_structural_stability",
    "no_confidence",
    "no_source_consistency",
    "no_degree_prior",
    "no_calibration_loss",
    "no_anti_forgetting",
    "eata_filter",
    "graph_tta_consistency",
    "matcha_reliable",
}


def _make_predict_fn(model):
    """Return ``(x_np, adj_np) -> probs_np`` for reliability diagnostics."""

    def predict_fn(x_np, adj_np):
        xt = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=DEVICE)
        ei = adj_to_edge_index(adj_np)
        return model.predict_probs(xt, ei).cpu().numpy()

    return predict_fn


def normalized_reliability_weights(r: np.ndarray) -> np.ndarray:
    """Linearly normalize reliability scores into a probability vector."""
    clipped = np.clip(np.asarray(r, dtype=float), 1e-6, 1.0)
    return clipped / max(float(clipped.sum()), 1e-12)


def _fail_closed_nonfinite(weight, weight_source, detector, step, reason):
    """Restore the immutable source checkpoint after a numerical failure."""
    with torch.no_grad():
        weight.copy_(weight_source)
    if detector is not None:
        detector.triggered = True
        detector.trigger_step = (
            step if detector.trigger_step is None else detector.trigger_step
        )
        detector.trigger_reason = (
            reason if detector.trigger_reason is None else detector.trigger_reason
        )
        detector.strikes = max(detector.strikes, 2)
        detector.rejected_candidates += 1
        detector.state = "SOURCE_HALTED"
        detector.decision_history.append("REJECT_NONFINITE_SOURCE_HALT")


def _torch_group_masks(
    groups: Mapping[str, np.ndarray], device
) -> Dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(np.asarray(groups[name]), dtype=torch.bool, device=device)
        for name in _GROUP_NAMES
    }


def differentiable_objective(
    probs_t: torch.Tensor,
    reliability_weights_t: torch.Tensor,
    group_masks_t: Mapping[str, torch.Tensor],
    source_group_conf_t: Mapping[str, torch.Tensor],
    weight: torch.Tensor,
    weight_source: torch.Tensor,
    lambda_cal: float,
    lambda_af: float,
    use_cal: bool = True,
    use_af: bool = True,
):
    """Return the exact deployed objective and its differentiable components."""
    ent_t = -(probs_t.clamp_min(1e-12).log() * probs_t).sum(dim=1)
    rel_loss = (reliability_weights_t * ent_t).sum()

    cal_loss = probs_t.new_zeros(())
    if use_cal:
        for name in _GROUP_NAMES:
            mask = group_masks_t[name]
            if bool(mask.any()):
                current = probs_t[mask].amax(dim=1).mean()
                cal_loss = cal_loss + (current - source_group_conf_t[name]) ** 2

    af_loss = probs_t.new_zeros(())
    if use_af:
        af_loss = 0.5 * ((weight - weight_source) ** 2).sum()

    total = rel_loss + lambda_cal * cal_loss + lambda_af * af_loss
    return total, {
        "reliability": rel_loss,
        "calibration": cal_loss,
        "anti_forgetting": af_loss,
    }


def _source_only_result(detector=None):
    return {
        "steps": 0,
        "loss_history": [],
        "mean_reliability": 1.0,
        "selected_fraction": 1.0,
        "delta_trace": [],
        "phi_trace": [],
        "drift_trace": {"low": [], "mid": [], "high": []},
        "detector": detector.to_dict() if detector is not None else None,
        "detector_halted_step": None,
    }


def _validated_source_anchor(model, weight: torch.Tensor) -> torch.Tensor:
    """Return a detached source anchor without mutating model state."""

    source = getattr(model, "source_classifier_weight", None)
    valid_flag = getattr(model, "source_classifier_anchor_valid", None)
    registered_buffers = getattr(model, "_buffers", {})
    nonpersistent_buffers = getattr(model, "_non_persistent_buffers_set", set())
    if (
        "source_classifier_weight" not in registered_buffers
        or "source_classifier_anchor_valid" not in registered_buffers
        or "source_classifier_weight" in nonpersistent_buffers
        or "source_classifier_anchor_valid" in nonpersistent_buffers
    ):
        raise RuntimeError("source classifier anchor is not registered persistently")
    if (
        not isinstance(valid_flag, torch.Tensor)
        or valid_flag.dtype is not torch.bool
        or valid_flag.numel() != 1
    ):
        raise RuntimeError("source classifier anchor validity flag is invalid")
    if not bool(valid_flag.item()):
        raise RuntimeError("source classifier anchor is not initialized")
    if not isinstance(source, torch.Tensor):
        raise RuntimeError("source classifier anchor is missing")
    if source.shape != weight.shape:
        raise RuntimeError(
            "source classifier anchor shape mismatch: "
            f"expected {tuple(weight.shape)}, got {tuple(source.shape)}"
        )
    if source.dtype != weight.dtype:
        raise RuntimeError(
            "source classifier anchor dtype mismatch: "
            f"expected {weight.dtype}, got {source.dtype}"
        )
    if source.device != weight.device:
        raise RuntimeError(
            "source classifier anchor device mismatch: "
            f"expected {weight.device}, got {source.device}"
        )
    if not torch.isfinite(source).all():
        raise RuntimeError("source classifier anchor is non-finite")
    return source.detach().clone()


def _adapt_classifier_impl(
    model,
    sb,
    method,
    seed=0,
    steps=80,
    lr=0.05,
    lambda_cal=0.5,
    lambda_af=0.01,
    tol=1e-8,
    detector=None,
    rel_kwargs=None,
    candidate_observer=None,
    transaction_context=None,
):
    """Implement final-classifier adaptation on an unlabeled graph view.

    Reliability scores are recomputed at every attempt and detached. With a
    detector, an update is a candidate and becomes deployable only after it
    satisfies both source-relative proxy tolerances.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported adaptation method: {method!r}")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    if not np.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    rel_kwargs = dict(rel_kwargs or {})
    transaction_context = transaction_context if transaction_context is not None else {}
    transaction_context.update(step=None, phase="source_setup", candidate_mutated=False)
    use_reliability = method not in _NO_RELIABILITY
    lite_method = method == "confidence_source_entropy"
    use_agreement = method != "no_neighborhood_agreement" and not lite_method
    use_stability = method != "no_structural_stability" and not lite_method
    use_confidence = method != "no_confidence"
    use_source = method != "no_source_consistency"
    use_degree = method != "no_degree_prior" and not lite_method
    use_cal = (
        method not in {"no_calibration_loss", "tent_entropy", "eata_filter"}
        and not lite_method
    )
    use_af = method not in {"no_anti_forgetting", "tent_entropy"}
    use_eata_filter = method == "eata_filter"
    use_graph_consistency = method == "graph_tta_consistency"
    use_matcha_mask = method == "matcha_reliable"

    weight = model.classifier_weight()
    call_start_weight = weight.detach().clone()
    weight_source = _validated_source_anchor(model, weight)

    if method == "source_only":
        with torch.no_grad():
            weight.copy_(weight_source)
        return _source_only_result(detector)

    if detector is not None and detector.state == "SOURCE_HALTED":
        with torch.no_grad():
            weight.copy_(weight_source)
        return _source_only_result(detector)

    x_t, edge_index, adj, x_np = sb.x, sb.edge_index, sb.adj, sb.x_np
    predict_fn = _make_predict_fn(model)

    model.eval()
    model.freeze_all()
    weight.requires_grad_(True)

    # Source quantities and structural groups remain immutable during the call.
    # Computing them must not erase a checkpoint carried over from an earlier
    # graph in a stream, so restore the call-start checkpoint afterwards.
    with torch.no_grad():
        weight.copy_(weight_source)
    source_probs = predict_fn(x_np, adj)
    if not np.isfinite(source_probs).all() or not torch.isfinite(weight_source).all():
        with torch.no_grad():
            weight.copy_(weight_source)
        raise FloatingPointError("non-finite immutable source checkpoint or prediction")
    groups = degree_group_masks(adj)
    groups_t = _torch_group_masks(groups, weight.device)
    source_group_conf = group_confidence(adj, source_probs, groups=groups)
    source_group_conf_t = {
        name: torch.tensor(
            source_group_conf[name], dtype=torch.float32, device=weight.device
        )
        for name in _GROUP_NAMES
    }
    source_argmax = np.argmax(source_probs, axis=1)

    with torch.no_grad():
        weight.copy_(call_start_weight)
    accepted_weight = call_start_weight.detach().clone()
    losses, reliability_trace, selected_trace = [], [], []
    delta_trace, phi_trace = [], []
    drift_trace = {name: [] for name in _GROUP_NAMES}
    prev_loss = None
    stable = 0
    detector_halted_step = None
    attempts = 0
    components = {}
    numerical_failure = None
    empty_selection_skipped = False

    for step in range(int(steps)):
        transaction_context.update(
            step=step, phase="preupdate_reliability", candidate_mutated=False
        )
        attempts += 1
        # Reliability is evaluated at the latest accepted checkpoint and then
        # detached from the gradient graph for this candidate.
        with torch.no_grad():
            weight.copy_(accepted_weight)
        probs_before = predict_fn(x_np, adj)
        if not np.isfinite(probs_before).all():
            numerical_failure = "nonfinite_preupdate_probability"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        if use_reliability:
            r, _ = reliability_scores(
                predict_fn,
                x_np,
                adj,
                seed=seed + step,
                use_agreement=use_agreement,
                use_stability=use_stability,
                reference_probs=source_probs,
                use_confidence=use_confidence,
                use_source=use_source,
                use_degree=use_degree,
                **rel_kwargs,
            )
        else:
            r = np.ones(probs_before.shape[0], dtype=float)
        if use_eata_filter:
            ent = entropy(probs_before) / np.log(probs_before.shape[1])
            src_sim = source_consistency(probs_before, source_probs)
            r = (
                (ent <= np.quantile(ent, 0.6)) & (src_sim >= np.quantile(src_sim, 0.4))
            ).astype(float)
        if use_matcha_mask:
            r = (r >= np.quantile(r, 0.5)).astype(float)

        if not np.isfinite(r).all():
            numerical_failure = "nonfinite_reliability"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        if not np.any(r > 0):
            empty_selection_skipped = True
            if detector is not None:
                detector.decision_history.append("SKIP_EMPTY_SELECTION")
            break

        q = normalized_reliability_weights(r)
        if not np.isfinite(q).all():
            numerical_failure = "nonfinite_normalized_reliability"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        reliability_trace.append(float(np.mean(r)))
        selected_trace.append(float(np.mean(r >= 0.5)))
        q_t = torch.tensor(q, dtype=torch.float32, device=weight.device)

        logits = model(x_t, edge_index)
        probs_t = F.softmax(logits, dim=1)
        obj, components = differentiable_objective(
            probs_t,
            q_t,
            groups_t,
            source_group_conf_t,
            weight,
            weight_source,
            lambda_cal,
            lambda_af,
            use_cal=use_cal,
            use_af=use_af,
        )
        if not torch.isfinite(probs_t).all() or not torch.isfinite(obj):
            numerical_failure = "nonfinite_objective_or_probability"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        model.zero_grad(set_to_none=True)
        obj.backward()
        if weight.grad is None or not torch.isfinite(weight.grad).all():
            numerical_failure = "nonfinite_gradient"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        grad = weight.grad.detach().clone()
        loss = float(obj.detach())

        # This legacy local comparator is retained under its explicit surrogate
        # name; it is not part of the proposed full method.
        if use_graph_consistency:
            probs_np = probs_t.detach().cpu().numpy()
            neigh_agree, _ = neighborhood_agreement(adj, probs_np)
            penalty = float(np.mean((1.0 - neigh_agree) * np.max(probs_np, axis=1)))
            grad = grad + 0.05 * penalty * weight.detach()
            loss += 0.05 * penalty

        grad_norm = float(torch.linalg.vector_norm(grad))
        scale = min(1.0, 2.0 / max(grad_norm, 1e-12))
        with torch.no_grad():
            weight.add_(grad, alpha=-lr * scale)
        transaction_context.update(
            phase="postupdate_validation", candidate_mutated=True
        )
        if not torch.isfinite(weight).all():
            numerical_failure = "nonfinite_candidate_weight"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        losses.append(loss)

        # The detector evaluates the post-update candidate.
        candidate_probs = predict_fn(x_np, adj)
        if not np.isfinite(candidate_probs).all():
            numerical_failure = "nonfinite_candidate_probability"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        candidate_group_conf = group_confidence(adj, candidate_probs, groups=groups)
        delta_t = float(
            np.mean(
                [
                    abs(candidate_group_conf[name] - source_group_conf[name])
                    for name in _GROUP_NAMES
                ]
            )
        )
        phi_t = float(np.mean(np.argmax(candidate_probs, axis=1) != source_argmax))
        if not np.isfinite(delta_t) or not np.isfinite(phi_t):
            numerical_failure = "nonfinite_proxy"
            _fail_closed_nonfinite(
                weight, weight_source, detector, step, numerical_failure
            )
            accepted_weight = weight_source.detach().clone()
            detector_halted_step = step
            break
        delta_trace.append(delta_t)
        phi_trace.append(phi_t)
        for name in _GROUP_NAMES:
            drift_trace[name].append(
                float(candidate_group_conf[name] - source_group_conf[name])
            )

        rejected = False
        reason = None
        if detector is not None:
            detector.delta_history.append(delta_t)
            detector.phi_history.append(phi_t)
            transaction_context["phase"] = "detector_decision"
            rejected, reason = detector_should_halt(detector, delta_t, phi_t)

        if candidate_observer is not None:
            transaction_context["phase"] = "candidate_observer"
            source_snapshot = np.array(source_probs, copy=True)
            candidate_snapshot = np.array(candidate_probs, copy=True)
            source_snapshot.setflags(write=False)
            candidate_snapshot.setflags(write=False)
            observer_result = candidate_observer(
                step=step,
                source_probs=source_snapshot,
                candidate_probs=candidate_snapshot,
                delta=delta_t,
                phi=phi_t,
            )
            if observer_result is not None:
                raise TypeError("candidate_observer must return None")

        if rejected:
            detector.triggered = True
            if detector.trigger_step is None:
                detector.trigger_step = step
                detector.trigger_reason = reason
            detector.strikes += 1
            detector.rejected_candidates += 1
            with torch.no_grad():
                weight.copy_(accepted_weight)
            if detector.strikes >= 2:
                with torch.no_grad():
                    weight.copy_(weight_source)
                accepted_weight = weight_source.detach().clone()
                detector.state = "SOURCE_HALTED"
                detector.decision_history.append("REJECT_SOURCE_HALT")
                detector_halted_step = step
                break
            detector.decision_history.append("REJECT_RETAIN_CHECKPOINT")
            continue

        accepted_weight = weight.detach().clone()
        if detector is not None:
            detector.strikes = 0
            detector.accepted_candidates += 1
            detector.state = "ADAPTING"
            detector.decision_history.append("ACCEPT")

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            stable += 1
        else:
            stable = 0
        prev_loss = loss
        if stable >= 5:
            break

    with torch.no_grad():
        weight.copy_(accepted_weight)
    if detector is not None and detector.state != "SOURCE_HALTED":
        detector.state = "FINISHED"

    return {
        "steps": attempts,
        "loss_history": losses,
        "mean_reliability": (
            float(np.mean(reliability_trace)) if reliability_trace else 1.0
        ),
        "selected_fraction": float(np.mean(selected_trace)) if selected_trace else 1.0,
        "delta_trace": delta_trace,
        "phi_trace": phi_trace,
        "drift_trace": drift_trace,
        "objective_components_last": {
            name: float(value.detach()) for name, value in components.items()
        },
        "detector": detector.to_dict() if detector is not None else None,
        "detector_halted_step": detector_halted_step,
        "numerical_failure": numerical_failure,
        "empty_selection_skipped": empty_selection_skipped,
    }


def adapt_classifier(
    model,
    sb,
    method,
    seed=0,
    steps=80,
    lr=0.05,
    lambda_cal=0.5,
    lambda_af=0.01,
    tol=1e-8,
    detector=None,
    rel_kwargs=None,
    candidate_observer=None,
):
    """Adapt from a structurally label-free graph view and fail closed.

    Ordinary ``Exception`` subclasses raised after entering the adaptation call
    restore the validated source classifier before they are re-raised.  The
    call also restores entry-time training mode, gradient flags, and gradients.
    The model and detector must be exclusively owned by the caller.  This
    guarantee is limited to the current process; it is not durable against
    ``BaseException`` subclasses, process, OS, or hardware failure.
    """
    if type(sb) is not UnlabeledGraphView:
        raise TypeError(
            "adapt_classifier requires the exact UnlabeledGraphView type; "
            "call bundle.unlabeled()"
        )
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported adaptation method: {method!r}")
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    if not np.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and non-negative")
    if candidate_observer is not None and not callable(candidate_observer):
        raise TypeError("candidate_observer must be callable or None")

    weight = model.classifier_weight()
    source = _validated_source_anchor(model, weight)
    modules = list(model.modules())
    entry_training = [module.training for module in modules]
    parameters = list(model.parameters())
    entry_requires_grad = [parameter.requires_grad for parameter in parameters]
    entry_gradients = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    transaction_context = {
        "step": None,
        "phase": "entry",
        "candidate_mutated": False,
    }
    detector_snapshot = (
        copy.deepcopy(detector.__dict__) if detector is not None else None
    )
    try:
        return _adapt_classifier_impl(
            model,
            sb,
            method,
            seed=seed,
            steps=steps,
            lr=lr,
            lambda_cal=lambda_cal,
            lambda_af=lambda_af,
            tol=tol,
            detector=detector,
            rel_kwargs=rel_kwargs,
            candidate_observer=candidate_observer,
            transaction_context=transaction_context,
        )
    except Exception as exc:
        try:
            with torch.no_grad():
                weight.copy_(source)
        except Exception as restore_error:
            exc.add_note(
                "source classifier restoration also failed: "
                f"{type(restore_error).__name__}: {restore_error}"
            )
        if detector is not None:
            detector.__dict__.clear()
            detector.__dict__.update(copy.deepcopy(detector_snapshot))
            detector.triggered = True
            detector.trigger_step = transaction_context["step"]
            detector.trigger_reason = f"exception:{type(exc).__name__}"
            detector.strikes = max(detector.strikes, 2)
            detector.rejected_candidates += 1
            detector.state = "SOURCE_HALTED"
            detector.decision_history.append("EXCEPTION_SOURCE_RESTORE")
        raise
    finally:
        for module, training in zip(modules, entry_training):
            module.training = training
        for parameter, requires_grad, gradient in zip(
            parameters, entry_requires_grad, entry_gradients
        ):
            parameter.requires_grad_(requires_grad)
            parameter.grad = None if gradient is None else gradient.clone()
