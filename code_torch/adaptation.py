"""Reliability-aware classifier-only test-time adaptation (PyTorch port).

Faithful port of ``code/adaptation.py::adapt_classifier``.  The proposed method
and its controlled ablations adapt **only the final linear classifier weight**
(``model.classifier_weight()`` == the NumPy ``W1``), preserving the closed-form
drift bound.  The per-step update mirrors the NumPy dynamics exactly:

* manual gradient step (lr=0.05) with gradient-norm clipping to 2.0,
* reliability weights / quantile masks / convergence test / detector rollback
  computed by the *unchanged* NumPy logic (``reliability.py`` / ``detector.py``).

Fidelity of the gradient (see plan):
* **Entropy term** and **anti-forgetting L2** are differentiable -> computed by
  real autograd through the PyG forward (more credible than the hand-derived
  NumPy gradient, and numerically equivalent).
* **Calibration** and **graph-consistency** terms use ``group_confidence`` /
  neighborhood agreement, which involve argmax + quantile binning and are not
  differentiable.  The NumPy code therefore uses a conservative *shrinkage
  proxy* (``dw1 += coef * W``); that exact proxy is replicated here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data_adapter import adj_to_edge_index, DEVICE
from reliability import (
    entropy,
    group_confidence,
    neighborhood_agreement,
    reliability_scores,
    source_consistency,
)

# Methods that perform no reliability weighting (uniform over all nodes).
_NO_RELIABILITY = {"source_only", "entropy_all_nodes", "tent_entropy"}


def _make_predict_fn(model):
    """Closure (x_np, adj_np) -> probs_np using the current model state."""

    def predict_fn(x_np, adj_np):
        xt = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=DEVICE)
        ei = adj_to_edge_index(adj_np)
        return model.predict_probs(xt, ei).cpu().numpy()

    return predict_fn


def adapt_classifier(model, sb, method, seed=0, steps=80, lr=0.05, lambda_cal=0.5,
                     lambda_af=0.01, tol=1e-8, detector=None, rel_kwargs=None):
    """Run reliability-aware test-time adaptation on the shifted graph ``sb``.

    ``sb`` is a :class:`data_adapter.GraphBundle` for the (already shifted) target
    graph: it carries torch tensors (``sb.x``, ``sb.edge_index``) for the forward
    pass and NumPy views (``sb.x_np``, ``sb.adj``) for reliability/detector scoring.
    ``detector`` (optional) is a :class:`detector.DetectorState`.
    ``rel_kwargs`` (optional) forwards extra keyword arguments to
    ``reliability_scores`` (e.g. ``homophily_delta`` for the misestimation study).
    """
    rel_kwargs = dict(rel_kwargs or {})
    use_reliability = method not in _NO_RELIABILITY
    use_agreement = method != "no_neighborhood_agreement"
    use_stability = method != "no_structural_stability"
    use_confidence = method != "no_confidence"
    use_source = method != "no_source_consistency"
    use_degree = method != "no_degree_prior"
    use_cal = method not in {"no_calibration_loss", "tent_entropy", "eata_filter"}
    use_af = method not in {"no_anti_forgetting", "tent_entropy"}
    use_eata_filter = method == "eata_filter"
    use_graph_consistency = method == "graph_tta_consistency"
    use_matcha_mask = method == "matcha_reliable"

    if method == "source_only":
        return {"steps": 0, "loss_history": [], "mean_reliability": 1.0, "selected_fraction": 1.0,
                "detector": detector.to_dict() if detector is not None else None,
                "detector_halted_step": None}

    x_t, edge_index, adj, x_np = sb.x, sb.edge_index, sb.adj, sb.x_np
    num_classes = sb.num_classes
    predict_fn = _make_predict_fn(model)

    model.eval()                     # BN uses running (source) stats; dropout off
    model.enable_classifier_grad()   # freeze everything except the classifier weight
    weight = model.classifier_weight()
    # Anti-forgetting anchor = the ORIGINAL trained source classifier (NumPy
    # ``w1_source``).  Falls back to the call-time weight for models that were
    # not trained through ``train_model`` (then the two coincide anyway).
    weight_source = getattr(model, "source_classifier_weight", None)
    if weight_source is None:
        weight_source = weight.detach().clone()
    else:
        weight_source = weight_source.to(weight.device)

    source_probs = predict_fn(x_np, adj)
    source_group_conf = group_confidence(adj, source_probs)
    source_argmax = np.argmax(source_probs, axis=1)

    losses, reliability_trace, selected_trace = [], [], []
    delta_trace, phi_trace = [], []
    drift_trace = {"low": [], "mid": [], "high": []}
    prev_weight = weight.detach().clone()
    prev_loss = None
    stable = 0
    detector_halted_step = None
    last_step = 0

    for step in range(steps):
        last_step = step
        # --- grad-enabled forward on the current classifier ---
        logits = model(x_t, edge_index)
        probs_t = F.softmax(logits, dim=1)
        probs = probs_t.detach().cpu().numpy()

        # --- detector signals (computed even when detector is None, for auditing) ---
        current_group_conf = group_confidence(adj, probs)
        delta_t = float(np.mean([abs(current_group_conf[k] - source_group_conf[k]) for k in source_group_conf]))
        phi_t = float(np.mean(np.argmax(probs, axis=1) != source_argmax))
        delta_trace.append(delta_t)
        phi_trace.append(phi_t)
        for k in drift_trace:
            drift_trace[k].append(float(current_group_conf[k] - source_group_conf[k]))
        if detector is not None and step > 0:
            if delta_t > detector.delta_tolerance:
                detector.triggered = True
                detector.trigger_step = step
                detector.trigger_reason = f"delta>{detector.delta_tolerance:.3f}"
                with torch.no_grad():
                    weight.copy_(prev_weight)
                detector_halted_step = step
                break
            if phi_t > detector.phi_tolerance:
                detector.triggered = True
                detector.trigger_step = step
                detector.trigger_reason = f"phi>{detector.phi_tolerance:.3f}"
                with torch.no_grad():
                    weight.copy_(prev_weight)
                detector_halted_step = step
                break
        prev_weight = weight.detach().clone()
        if detector is not None:
            detector.delta_history.append(delta_t)
            detector.phi_history.append(phi_t)

        # --- reliability weights (unchanged NumPy logic) ---
        if use_reliability:
            r, _ = reliability_scores(predict_fn, x_np, adj, seed=seed + step,
                                      use_agreement=use_agreement, use_stability=use_stability,
                                      reference_probs=source_probs, use_confidence=use_confidence,
                                      use_source=use_source, use_degree=use_degree, **rel_kwargs)
        else:
            r = np.ones(probs.shape[0], dtype=float)
        if use_eata_filter:
            ent = entropy(probs) / np.log(probs.shape[1])
            src_sim = source_consistency(probs, source_probs)
            r = ((ent <= np.quantile(ent, 0.6)) & (src_sim >= np.quantile(src_sim, 0.4))).astype(float)
        if use_matcha_mask:
            r = (r >= np.quantile(r, 0.5)).astype(float)
        reliability_trace.append(float(np.mean(r)))
        selected_trace.append(float(np.mean(r >= 0.5)))
        weights = np.power(np.clip(r, 1e-3, 1.0), 1.5)
        weights = weights / max(np.sum(weights), 1e-12)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        # --- differentiable objective: reliability-weighted entropy + anti-forgetting ---
        ent_t = -(probs_t.clamp_min(1e-12).log() * probs_t).sum(dim=1)
        obj = (weights_t * ent_t).sum()
        loss = float((weights * entropy(probs)).sum())
        if use_af:
            diff = weight - weight_source
            obj = obj + 0.5 * lambda_af * (diff ** 2).sum()
            loss += 0.5 * lambda_af * float((diff.detach() ** 2).sum())
        if weight.grad is not None:
            weight.grad = None
        obj.backward()
        grad = weight.grad.detach().clone()

        # --- non-differentiable shrinkage proxies (replicated verbatim from NumPy) ---
        if use_graph_consistency:
            neigh_agree, _ = neighborhood_agreement(adj, probs)
            consistency_penalty = float(np.mean((1.0 - neigh_agree) * np.max(probs, axis=1)))
            grad = grad + 0.05 * consistency_penalty * weight.detach()
            loss += 0.05 * consistency_penalty
        if use_cal:
            cal_loss = sum((current_group_conf[k] - source_group_conf[k]) ** 2 for k in source_group_conf)
            grad = grad + lambda_cal * cal_loss * weight.detach()
            loss += lambda_cal * float(cal_loss)

        # --- manual step with gradient-norm clipping (matches NumPy dynamics) ---
        grad_norm = float(torch.sqrt((grad ** 2).sum()))
        scale = min(1.0, 2.0 / max(grad_norm, 1e-12))
        with torch.no_grad():
            weight.add_(grad, alpha=-lr * scale)
        losses.append(loss)

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            stable += 1
        else:
            stable = 0
        if stable >= 5:
            break
        prev_loss = loss

    return {
        "steps": last_step + 1,
        "loss_history": losses,
        "mean_reliability": float(np.mean(reliability_trace)) if reliability_trace else 1.0,
        "selected_fraction": float(np.mean(selected_trace)) if selected_trace else 1.0,
        "delta_trace": delta_trace,
        "phi_trace": phi_trace,
        "drift_trace": drift_trace,
        "detector": detector.to_dict() if detector is not None else None,
        "detector_halted_step": detector_halted_step,
    }
