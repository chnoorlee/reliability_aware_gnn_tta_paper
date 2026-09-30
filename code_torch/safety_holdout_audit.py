"""Internal held-out real-graph audit for the corrected proxy guard.

The protocol is frozen in ``revision_2026-08-19/HELDOUT_AUDIT_PROTOCOL.md``.
This runner writes progress after every completed dataset so interrupted runs
remain auditable.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import adapt_classifier
from detector import DetectorState
from data_adapter import build_bundle
from exp_common import shift_bundle, train_source, webkb_bundle


DEFAULT_DATASETS = (
    "cora",
    "citeseer",
    "pubmed",
    "amazon_computers",
    "amazon_photo",
    "coauthor_cs",
    "texas",
    "cornell",
    "wisconsin",
)
WEBKB = {"texas", "cornell", "wisconsin"}
CONDITIONS = (
    ("feature_noise_rms", 1.00),
    ("edge_drop", 0.30),
    ("edge_add", 0.20),
    ("homophily_shift", 0.50),
)
MAX_NODES = {"amazon_computers": 5000, "amazon_photo": 5000, "coauthor_cs": 6000}
HIDDEN = {
    "cora": 32,
    "citeseer": 32,
    "pubmed": 48,
    "amazon_computers": 48,
    "amazon_photo": 48,
    "coauthor_cs": 64,
    "texas": 24,
    "cornell": 24,
    "wisconsin": 24,
}


def graph_homophily(adj, labels):
    if hasattr(adj, "tocoo"):
        coo = adj.tocoo()
        row, col = coo.row, coo.col
    else:
        row, col = np.where(np.asarray(adj) > 0)
    keep = row < col
    return float(np.mean(labels[row[keep]] == labels[col[keep]])) if bool(keep.any()) else 0.0


def _metrics(model, bundle):
    probs = model.predict_probs(bundle.x, bundle.edge_index).cpu().numpy()
    return evaluate(probs[bundle.test_idx], bundle.y_np[bundle.test_idx], bundle.num_classes)


def make_target(base, seed, shift, intensity):
    if shift != "feature_noise_rms":
        return shift_bundle(base, seed, shift, intensity)
    rng = np.random.default_rng(seed + 7919)
    feature_rms = max(float(np.sqrt(np.mean(np.square(base.x_np)))), 1e-12)
    shifted_x = base.x_np + intensity * feature_rms * rng.normal(size=base.x_np.shape)
    return build_bundle(
        shifted_x,
        base.adj.copy(),
        base.y_np,
        base.train_idx,
        base.val_idx,
        base.test_idx,
    )


def _write(out_path, records, metadata, status):
    payload = dict(metadata)
    payload.update({"status": status, "records": records})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(out_path, datasets=DEFAULT_DATASETS, seeds=(0, 1, 2, 3, 4),
        train_epochs=300, adapt_steps=90):
    out_path = Path(out_path)
    started = time.perf_counter()
    records = []
    metadata = {
        "evidence_status": "internal_heldout_audit",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "conditions": [list(item) for item in CONDITIONS],
        "train_epochs": train_epochs,
        "adapt_steps": adapt_steps,
        "guard": {"delta_tolerance": 0.05, "phi_tolerance": 0.20},
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    _write(out_path, records, metadata, "running")

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
            source_h = graph_homophily(base.adj, base.y_np)
            for shift, intensity in CONDITIONS:
                target = make_target(base, seed, shift, intensity)
                target_h = graph_homophily(target.adj, target.y_np)
                variants = (
                    ("source", "source", "source_only", None),
                    (
                        "confidence_source_unguarded",
                        "confidence_source_entropy",
                        "confidence_source_entropy",
                        None,
                    ),
                    (
                        "confidence_source_guarded",
                        "confidence_source_entropy",
                        "confidence_source_entropy",
                        DetectorState(delta_tolerance=0.05, phi_tolerance=0.20),
                    ),
                    ("uniform_entropy_unguarded", "uniform_entropy", "tent_entropy", None),
                    (
                        "uniform_entropy_guarded",
                        "uniform_entropy",
                        "tent_entropy",
                        DetectorState(delta_tolerance=0.05, phi_tolerance=0.20),
                    ),
                )
                condition_rows = []
                for method_name, adapter_family, adapter_name, detector in variants:
                    current = model.clone()
                    tic = time.perf_counter()
                    info = adapt_classifier(
                        current,
                        target.unlabeled(),
                        method=adapter_name,
                        seed=seed,
                        steps=adapt_steps,
                        detector=detector,
                    )
                    runtime = time.perf_counter() - tic
                    result = _metrics(current, target)
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "shift": shift,
                        "intensity": intensity,
                        "method": method_name,
                        "adapter_family": adapter_family,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "nll": result["nll"],
                        "brier": result["brier"],
                        "ece": result["ece"],
                        "runtime_seconds": runtime,
                        "source_homophily": source_h,
                        "target_homophily": target_h,
                        "num_nodes": target.num_nodes,
                        "num_edges": int(target.edge_index.shape[1] // 2),
                        "train_epochs_completed": train_info["epochs"],
                        "adaptation_attempts": info["steps"],
                        "triggered": bool(detector.triggered) if detector else False,
                        "trigger_step": detector.trigger_step if detector else None,
                        "detector_state": detector.state if detector else None,
                        "accepted_candidates": detector.accepted_candidates if detector else None,
                        "rejected_candidates": detector.rejected_candidates if detector else None,
                    }
                    records.append(row)
                    condition_rows.append(row)

                source_accuracy = next(
                    row["accuracy"] for row in condition_rows if row["method"] == "source"
                )
                for row in condition_rows:
                    row["source_relative_accuracy"] = row["accuracy"] - source_accuracy
                    row["source_relative_downside"] = max(0.0, source_accuracy - row["accuracy"])
                    print(
                        f"[holdout] {dataset} seed={seed} {shift}/{intensity} {row['method']}: "
                        f"acc={row['accuracy']:.4f} diff={row['source_relative_accuracy']:+.4f} "
                        f"trigger={row['triggered']}"
                    )
            metadata["elapsed_seconds"] = time.perf_counter() - started
            _write(out_path, records, metadata, "running")

    metadata["elapsed_seconds"] = time.perf_counter() - started
    _write(out_path, records, metadata, "complete")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--adapt-steps", type=int, default=90)
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    run(args.out, datasets, seeds, args.train_epochs, args.adapt_steps)


if __name__ == "__main__":
    main()
