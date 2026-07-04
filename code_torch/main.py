"""PyTorch Geometric experiment runner (mirror of ``code/main.py``).

Same seeds x conditions x methods design, the same record schema, and the same
metric definitions (via ``_np_bridge.evaluate``) as the NumPy runner, so the two
result sets are directly comparable.  The primary backbone is a **no-BN GCN**,
which reproduces the published source-only numbers (Cora/Citeseer/Pubmed ~=
0.81/0.71/0.79); BatchNorm is reserved for the fair-comparison experiment, where
the official Tent/EATA baselines adapt it.

Examples:
    python main.py --quick --datasets public_core
    python main.py --quick --dataset synthetic --backbone gat
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import adapt_classifier
from data_adapter import load_bundle, shift_bundle
from models import make_model, train_model

METHODS = [
    "source_only", "entropy_all_nodes", "tent_entropy", "eata_filter",
    "graph_tta_consistency", "matcha_reliable", "reliable_entropy",
    "no_neighborhood_agreement", "no_structural_stability", "no_calibration_loss",
    "no_anti_forgetting", "full_method",
]
PUBLIC_BENCHMARK_METHODS = [
    "source_only", "entropy_all_nodes", "tent_entropy", "eata_filter",
    "graph_tta_consistency", "matcha_reliable", "full_method",
]
HETEROPHILY_DATASETS = {"texas", "cornell", "wisconsin", "actor", "film"}
BENCHMARK_DATASETS = {
    "public_core": ["cora", "citeseer", "pubmed"],
    "public_extended": ["cora", "citeseer", "pubmed", "amazon_computers", "amazon_photo", "coauthor_cs"],
    "heterophily_core": ["texas", "cornell", "wisconsin"],
    "heterophily_extended": ["texas", "cornell", "wisconsin", "actor", "film"],
}
DEFAULT_HIDDEN = {
    "synthetic": 24, "cora": 32, "citeseer": 32, "pubmed": 48,
    "amazon_computers": 48, "amazon_photo": 48, "coauthor_cs": 64,
    "texas": 24, "cornell": 24, "wisconsin": 24, "actor": 24, "film": 24,
}
DEFAULT_MAX_NODES = {"amazon_computers": 5000, "amazon_photo": 5000, "coauthor_cs": 6000}


def methods_for_dataset(dataset):
    return METHODS if dataset == "synthetic" or dataset in HETEROPHILY_DATASETS else PUBLIC_BENCHMARK_METHODS


def resolve_dataset_list(dataset_arg, datasets_arg):
    raw = []
    if datasets_arg:
        raw.extend([s.strip().lower().replace("-", "_") for s in datasets_arg.split(",") if s.strip()])
    else:
        raw.append(dataset_arg.lower().replace("-", "_"))
    out = []
    for item in raw:
        out.extend(BENCHMARK_DATASETS.get(item, [item]))
    ordered, seen = [], set()
    for item in out:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def resolve_setup(dataset, quick, train_epochs=None, adapt_steps=None):
    if dataset == "synthetic":
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.35), ("edge_drop", 0.25)]
            n, hidden, dte, das = 240, 16, 200, 25
        else:
            seeds = [0, 1, 2, 3, 4]
            conditions = [("clean", 0.0), ("feature_noise", 0.20), ("feature_noise", 0.45),
                          ("edge_drop", 0.15), ("edge_drop", 0.35), ("edge_add", 0.15),
                          ("edge_add", 0.35), ("degree_shift", 0.35), ("homophily_shift", 0.25),
                          ("homophily_shift", 0.50)]
            n, hidden, dte, das = 360, 24, 300, 90
    elif dataset in HETEROPHILY_DATASETS:
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.10), ("edge_add", 0.10)]
            dte, das = 250, 20
        else:
            seeds = [0, 1, 2, 3, 4]
            conditions = [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.10),
                          ("edge_add", 0.10), ("homophily_shift", 0.25)]
            dte, das = 300, 35
        n, hidden = 180, DEFAULT_HIDDEN.get(dataset, 24)
    else:
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.08), ("edge_drop", 0.10), ("edge_add", 0.05)]
            dte, das = 200, 20
        else:
            seeds = [0, 1, 2]
            conditions = [("clean", 0.0), ("feature_noise", 0.08), ("edge_drop", 0.10),
                          ("edge_add", 0.05), ("degree_shift", 0.15)]
            dte, das = 250, 25
        n, hidden = 180, DEFAULT_HIDDEN.get(dataset, 32)
    return {
        "seeds": seeds, "conditions": conditions, "n": n, "hidden": hidden,
        "train_epochs": train_epochs if train_epochs is not None else dte,
        "adapt_steps": adapt_steps if adapt_steps is not None else das,
    }


def _model_kwargs(backbone, hidden):
    if backbone == "gat":
        return {"hidden_dim": 8, "heads": 8}
    return {"hidden_dim": hidden}


def run_one(seed, shift, intensity, hidden, dataset, backbone, train_epochs, adapt_steps,
            n=None, max_nodes=None, use_bn=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = load_bundle(dataset, seed=seed, n=n, max_nodes=max_nodes)
    kw = _model_kwargs(backbone, hidden)
    model = make_model(backbone, base.x.shape[1], out_dim=base.num_classes, use_bn=use_bn, seed=seed, **kw)
    train_info = train_model(model, base.x, base.edge_index, base.y, base.train_mask, base.val_mask,
                             epochs=train_epochs, lr=0.01, weight_decay=5e-4, patience=max(40, train_epochs // 4))

    sb = shift_bundle(base, seed, shift, intensity)
    num_edges = float(sb.edge_index.shape[1] // 2)

    records = []
    for method in methods_for_dataset(dataset):
        m = model.clone()
        # Protocol parity with code/main.py: the main benchmark runs WITHOUT the
        # negative-adaptation detector, so out-of-envelope failure modes remain
        # visible in the main tables.  The detector is evaluated in its dedicated
        # experiments (supplementary detector_sensitivity, fair_comparison,
        # detector_calibration).
        detector = None
        start = time.perf_counter()
        adapt_info = adapt_classifier(m, sb, method=method, seed=seed, steps=adapt_steps, detector=detector)
        runtime = time.perf_counter() - start
        probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
        metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
        metrics.update({
            "seed": seed, "dataset": dataset, "backbone": backbone, "shift": shift,
            "intensity": intensity, "method": method, "runtime_seconds": runtime,
            "adapted_steps": adapt_info["steps"], "mean_reliability": adapt_info["mean_reliability"],
            "selected_fraction": adapt_info["selected_fraction"], "train_epochs": train_info["epochs"],
            "best_val_loss": train_info["best_val_loss"], "num_nodes": float(sb.num_nodes),
            "num_edges": num_edges,
            "detector_triggered": bool(detector.triggered) if detector is not None else None,
        })
        records.append(metrics)
    return records


def summarize(records):
    grouped = {}
    for r in records:
        key = (r.get("dataset", "synthetic"), r.get("backbone", "gcn"), r["shift"], r["intensity"], r["method"])
        grouped.setdefault(key, []).append(r)
    metric_keys = ["accuracy", "macro_f1", "nll", "ece", "brier", "runtime_seconds",
                   "adapted_steps", "mean_reliability", "selected_fraction", "num_nodes", "num_edges"]
    summary = []
    for key, rows in grouped.items():
        out = {"dataset": key[0], "backbone": key[1], "shift": key[2], "intensity": key[3],
               "method": key[4], "count": len(rows)}
        for mk in metric_keys:
            vals = np.array([row[mk] for row in rows], dtype=float)
            out[mk + "_mean"] = float(np.mean(vals))
            out[mk + "_std"] = float(np.std(vals))
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--backbone", default="gcn", choices=["gcn", "gat", "graphsage", "appnp"])
    parser.add_argument("--use-bn", action="store_true", help="enable BatchNorm in the backbone (off by default)")
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results_torch"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_list = resolve_dataset_list(args.dataset, args.datasets)
    setups = {d: resolve_setup(d, args.quick, args.train_epochs, args.adapt_steps) for d in dataset_list}

    all_records = []
    start_all = time.perf_counter()
    for dataset in dataset_list:
        setup = setups[dataset]
        max_nodes = DEFAULT_MAX_NODES.get(dataset)
        for seed in setup["seeds"]:
            for shift, intensity in setup["conditions"]:
                print(f"running dataset={dataset} backbone={args.backbone} seed={seed} shift={shift} intensity={intensity}")
                all_records.extend(run_one(
                    seed, shift, intensity, hidden=setup["hidden"], dataset=dataset,
                    backbone=args.backbone, train_epochs=setup["train_epochs"],
                    adapt_steps=setup["adapt_steps"], n=setup["n"], max_nodes=max_nodes, use_bn=args.use_bn,
                ))

    summary = summarize(all_records)
    payload = {"records": all_records, "summary": summary, "quick": args.quick,
               "datasets": dataset_list, "backbone": args.backbone,
               "elapsed_seconds": time.perf_counter() - start_all,
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if summary:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    for row in summary[:24]:
        print(f"{row['dataset']:10s} {row['backbone']:5s} {row['shift']:14s} {row['intensity']} "
              f"{row['method']:18s} acc={row['accuracy_mean']:.4f} ece={row['ece_mean']:.4f}")
    print(f"results: {out_dir / 'results.json'}  ({payload['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
