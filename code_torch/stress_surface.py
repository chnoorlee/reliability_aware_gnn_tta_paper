"""Run a controlled real-graph stress surface for negative adaptation.

The caller must set ``evidence_status`` before execution.  Development scans
and frozen held-out audits use the same implementation, but their outputs must
remain explicitly separated and must never be relabelled after observing the
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch_geometric

from adaptation import adapt_classifier
from detector import DetectorState
from exp_common import train_source, webkb_bundle
from proxy_scope_audit import ScopeTraceRecorder, audit_proxy_scopes
from reliability import degree_group_masks
from safety_holdout_audit import (
    HIDDEN,
    MAX_NODES,
    WEBKB,
    _metrics,
    graph_homophily,
    make_target,
)

STRESS_GRID = (
    ("feature_noise_rms", 1.0),
    ("feature_noise_rms", 2.0),
    ("feature_noise_rms", 4.0),
    ("edge_drop", 0.30),
    ("edge_drop", 0.60),
    ("edge_drop", 0.90),
    ("edge_add", 0.20),
    ("edge_add", 0.50),
    ("edge_add", 1.00),
    ("homophily_shift", 0.50),
    ("homophily_shift", 0.75),
    ("homophily_shift", 1.00),
)
ADAPTERS = (
    ("confidence_source_entropy", "confidence_source_entropy"),
    ("uniform_entropy", "tent_entropy"),
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path, payload):
    """Replace a JSON checkpoint without exposing a partially written file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _update_array_hash(digest, name, value):
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.asarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())


def _bundle_sha256(bundle):
    """Bind provenance to every parallel representation consumed by the run."""
    digest = hashlib.sha256()
    _update_array_hash(digest, "x_np", bundle.x_np)
    if scipy.sparse.issparse(bundle.adj):
        adjacency = bundle.adj.tocsr()
        _update_array_hash(digest, "adj_indptr", adjacency.indptr)
        _update_array_hash(digest, "adj_indices", adjacency.indices)
        _update_array_hash(digest, "adj_data", adjacency.data)
    else:
        _update_array_hash(digest, "adj_dense", bundle.adj)
    for name in ("y_np", "train_idx", "val_idx", "test_idx"):
        _update_array_hash(digest, name, getattr(bundle, name))
    for name in (
        "x",
        "edge_index",
        "y",
        "train_mask",
        "val_mask",
        "test_mask",
    ):
        _update_array_hash(digest, name, getattr(bundle, name))
    _update_array_hash(
        digest, "num_classes", np.asarray(bundle.num_classes, dtype=np.int64)
    )
    return digest.hexdigest()


def _materialize_run_inputs(datasets, seeds, stress_grid):
    """Consume one-shot iterables once and reject vacuous completed runs."""
    datasets = tuple(datasets)
    seeds = tuple(seeds)
    raw_grid = tuple(stress_grid)
    if not datasets:
        raise ValueError("datasets must not be empty")
    if not seeds:
        raise ValueError("seeds must not be empty")
    if not raw_grid:
        raise ValueError("stress_grid must not be empty")
    normalized_grid = []
    for index, item in enumerate(raw_grid):
        try:
            shift, intensity = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"stress_grid item {index} must be a (shift, intensity) pair"
            ) from exc
        normalized_grid.append((shift, intensity))
    return datasets, seeds, tuple(normalized_grid)


def _code_provenance():
    root = Path(__file__).resolve().parent
    names = (
        "stress_surface.py",
        "adaptation.py",
        "_np_bridge.py",
        "data_adapter.py",
        "detector.py",
        "exp_common.py",
        "models.py",
        "proxy_scope_audit.py",
        "reliability.py",
        "safety_holdout_audit.py",
        "../code/data.py",
        "../code/utils.py",
        "../code/webkb_loader.py",
    )
    return {name: _sha256_file(root / name) for name in names}


def _run_impl(
    out_path,
    datasets,
    seeds,
    train_epochs=300,
    adapt_steps=300,
    include_guard=False,
    stress_grid=STRESS_GRID,
    evidence_status="exploratory_development_scan",
    guard_mode=None,
):
    datasets, seeds, stress_grid = _materialize_run_inputs(datasets, seeds, stress_grid)
    if guard_mode is None:
        guard_mode = "both" if include_guard else "unguarded"
    if guard_mode not in {"unguarded", "both", "guarded-only"}:
        raise ValueError(f"Unsupported guard mode: {guard_mode!r}")
    out_path = Path(out_path)
    started = time.perf_counter()
    records = []
    metadata = {
        "evidence_status": evidence_status,
        "datasets": list(datasets),
        "seeds": list(seeds),
        "train_epochs": train_epochs,
        "adapt_steps": adapt_steps,
        "include_guard": guard_mode == "both",
        "guard_mode": guard_mode,
        "controller": {
            "policy": "per-update two-strike source-relative rollback",
            "delta_tolerance": 0.05,
            "phi_tolerance": 0.20,
            "exception_atomicity": "source restored on Python exception; process-crash durability not claimed",
        },
        "proxy_scope_audit": {
            "status": "diagnostic_only_not_used_by_adaptation_or_guard",
            "timing": "each post-update candidate and the final deployed endpoint",
            "scopes": ["target", "evaluation", "non_evaluation"],
            "degree_groups": (
                "fixed on the full target graph, then intersected with each scope"
            ),
            "evaluation_scope": "the pre-existing benchmark test-index set without labels",
        },
        "stress_grid": [list(item) for item in stress_grid],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "invocation": list(sys.argv),
        "code_sha256": _code_provenance(),
    }
    for dataset in datasets:
        for seed in seeds:
            bundle = webkb_bundle(dataset, seed) if dataset in WEBKB else None
            model, base, train_info = train_source(
                dataset,
                seed,
                hidden=HIDDEN[dataset],
                epochs=train_epochs,
                max_nodes=MAX_NODES.get(dataset),
                bundle=bundle,
            )
            source_checkpoint_sha256 = _model_sha256(model)
            source_bundle_sha256 = _bundle_sha256(base)
            for shift, intensity in stress_grid:
                target = make_target(base, seed, shift, intensity)
                target_bundle_sha256 = _bundle_sha256(target)
                source_metrics = _metrics(model, target)
                source_probs = (
                    model.predict_probs(target.x, target.edge_index)
                    .detach()
                    .cpu()
                    .numpy()
                )
                evaluation_mask = np.zeros(target.num_nodes, dtype=bool)
                evaluation_mask[np.asarray(target.test_idx, dtype=np.int64)] = True
                diagnostic_scopes = {
                    "target": np.ones(target.num_nodes, dtype=bool),
                    "evaluation": evaluation_mask,
                    "non_evaluation": ~evaluation_mask,
                }
                degree_groups = degree_group_masks(target.adj)
                for adapter_label, method_name in ADAPTERS:
                    variants = []
                    if guard_mode in {"unguarded", "both"}:
                        variants.append(("unguarded", None))
                    if guard_mode in {"guarded-only", "both"}:
                        variants.append(
                            (
                                "guarded",
                                DetectorState(delta_tolerance=0.05, phi_tolerance=0.20),
                            )
                        )
                    for guard_label, detector in variants:
                        current = model.clone()
                        scope_recorder = ScopeTraceRecorder(
                            degree_groups, diagnostic_scopes
                        )
                        tic = time.perf_counter()
                        info = adapt_classifier(
                            current,
                            target.unlabeled(),
                            method=method_name,
                            seed=seed,
                            steps=adapt_steps,
                            detector=detector,
                            candidate_observer=scope_recorder,
                        )
                        scope_traces = scope_recorder.to_dict()
                        result = _metrics(current, target)
                        candidate_probs = (
                            current.predict_probs(target.x, target.edge_index)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        endpoint_proxy_scopes = audit_proxy_scopes(
                            degree_groups,
                            source_probs,
                            candidate_probs,
                            diagnostic_scopes,
                        )
                        gain = result["accuracy"] - source_metrics["accuracy"]
                        evaluation_phi = endpoint_proxy_scopes["evaluation"]["phi"]
                        if abs(gain) > evaluation_phi + 1e-6:
                            raise AssertionError(
                                "same-scope prediction turnover failed to bound "
                                "the absolute accuracy change"
                            )
                        if info["delta_trace"]:
                            target_trace = scope_traces["target"]
                        else:
                            target_trace = {"delta_trace": [], "phi_trace": []}
                        if (
                            target_trace["delta_trace"] != info["delta_trace"]
                            or target_trace["phi_trace"] != info["phi_trace"]
                        ):
                            raise AssertionError(
                                "target-scope diagnostic traces do not reproduce "
                                "the operational all-node traces"
                            )
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "shift": shift,
                            "intensity": intensity,
                            "adapter": adapter_label,
                            "guard": guard_label,
                            "source_accuracy": source_metrics["accuracy"],
                            "accuracy": result["accuracy"],
                            "source_relative_accuracy": gain,
                            "source_relative_downside": max(
                                0.0, source_metrics["accuracy"] - result["accuracy"]
                            ),
                            "source_nll": source_metrics["nll"],
                            "nll": result["nll"],
                            "source_ece": source_metrics["ece"],
                            "ece": result["ece"],
                            "source_homophily": graph_homophily(base.adj, base.y_np),
                            "target_homophily": graph_homophily(
                                target.adj, target.y_np
                            ),
                            "runtime_seconds": time.perf_counter() - tic,
                            "adaptation_attempts": info["steps"],
                            "triggered": (
                                bool(detector.triggered) if detector else False
                            ),
                            "trigger_step": detector.trigger_step if detector else None,
                            "detector_state": detector.state if detector else None,
                            "trigger_reason": (
                                detector.trigger_reason if detector else None
                            ),
                            "accepted_candidates": (
                                detector.accepted_candidates
                                if detector
                                else info["steps"]
                            ),
                            "rejected_candidates": (
                                detector.rejected_candidates if detector else 0
                            ),
                            "max_delta": max(info["delta_trace"], default=0.0),
                            "max_phi": max(info["phi_trace"], default=0.0),
                            "delta_trace": info["delta_trace"],
                            "phi_trace": info["phi_trace"],
                            "diagnostic_scope_traces": scope_traces,
                            "endpoint_proxy_scopes": endpoint_proxy_scopes,
                            "evaluation_turnover_bound_slack": evaluation_phi
                            - abs(gain),
                            "train_epochs_completed": train_info["epochs"],
                            "source_checkpoint_sha256": source_checkpoint_sha256,
                            "source_bundle_sha256": source_bundle_sha256,
                            "target_bundle_sha256": target_bundle_sha256,
                        }
                        records.append(row)
                        print(
                            f"[stress] {dataset} seed={seed} {shift}/{intensity} "
                            f"{adapter_label}/{guard_label}: source={source_metrics['accuracy']:.4f} "
                            f"adapt={result['accuracy']:.4f} diff={row['source_relative_accuracy']:+.4f} "
                            f"trigger={row['triggered']}"
                        )
            payload = dict(metadata)
            payload.update(
                {
                    "status": "running",
                    "elapsed_seconds": time.perf_counter() - started,
                    "records": records,
                }
            )
            _write_json_atomic(out_path, payload)

    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.perf_counter() - started
    _write_json_atomic(out_path, payload)
    return records


def run(
    out_path,
    datasets,
    seeds,
    train_epochs=300,
    adapt_steps=300,
    include_guard=False,
    stress_grid=STRESS_GRID,
    evidence_status="exploratory_development_scan",
    guard_mode=None,
):
    """Persist an unambiguous lifecycle record around the long replay."""
    out_path = Path(out_path)
    initial = {
        "status": "running",
        "evidence_status": evidence_status,
        "train_epochs": train_epochs,
        "adapt_steps": adapt_steps,
        "include_guard": include_guard,
        "guard_mode": guard_mode,
        "records": [],
    }
    _write_json_atomic(out_path, initial)
    try:
        datasets, seeds, stress_grid = _materialize_run_inputs(
            datasets, seeds, stress_grid
        )
        initial.update(
            {
                "datasets": list(datasets),
                "seeds": list(seeds),
                "stress_grid": [list(item) for item in stress_grid],
            }
        )
        _write_json_atomic(out_path, initial)
        return _run_impl(
            out_path=out_path,
            datasets=datasets,
            seeds=seeds,
            train_epochs=train_epochs,
            adapt_steps=adapt_steps,
            include_guard=include_guard,
            stress_grid=stress_grid,
            evidence_status=evidence_status,
            guard_mode=guard_mode,
        )
    except BaseException as error:
        try:
            failed = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failed = initial
        failed.update(
            {
                "status": "failed",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_json_atomic(out_path, failed)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default="cora,texas")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--adapt-steps", type=int, default=300)
    parser.add_argument("--include-guard", action="store_true")
    parser.add_argument(
        "--guard-mode",
        choices=("unguarded", "both", "guarded-only"),
        default=None,
        help="Run only unguarded, both variants, or only the online guarded controller.",
    )
    parser.add_argument(
        "--evidence-status",
        default="exploratory_development_scan",
        help="Provenance label stored verbatim in the output metadata.",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Optional comma-separated shift:intensity pairs; defaults to the full grid.",
    )
    args = parser.parse_args()
    datasets = tuple(
        item.strip().lower() for item in args.datasets.split(",") if item.strip()
    )
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    stress_grid = STRESS_GRID
    if args.conditions:
        stress_grid = tuple(
            (shift.strip(), float(intensity))
            for shift, intensity in (
                item.strip().split(":", maxsplit=1)
                for item in args.conditions.split(",")
                if item.strip()
            )
        )
    run(
        args.out,
        datasets,
        seeds,
        args.train_epochs,
        args.adapt_steps,
        args.include_guard,
        stress_grid,
        args.evidence_status,
        args.guard_mode,
    )


if __name__ == "__main__":
    main()
