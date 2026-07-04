"""Scalability experiment on ogbn-arxiv-like subsets (PyG port of code/run_scalability.py).

Same scales / seeds / conditions / methods / budgets; writes
``results_torch/scalability_study.json`` + ``scalability_summary.csv``.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import adapt_classifier
from exp_common import arxiv_bundle, shift_bundle
from models import make_model, train_model

SCALES = [1000, 2000, 5000]
METHODS = ["source_only", "entropy_all_nodes", "tent_entropy", "eata_filter", "full_method"]
CONDITIONS = [("clean", 0.0), ("feature_noise", 0.20), ("edge_drop", 0.15)]
SEEDS = [0, 1, 2]
TRAIN_EPOCHS = 400
ADAPT_STEPS = 50
HIDDEN = 64


def run_scalability():
    results = []
    output_dir = Path(__file__).resolve().parents[1] / "results_torch"
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in SCALES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            base = arxiv_bundle(seed, n=n)
            model = make_model("gcn", base.x.shape[1], HIDDEN, base.num_classes, use_bn=False, seed=seed)
            train_model(model, base.x, base.edge_index, base.y, base.train_mask, base.val_mask,
                        epochs=TRAIN_EPOCHS, lr=0.01, weight_decay=5e-4, patience=100)

            for shift, intensity in CONDITIONS:
                sb = shift_bundle(base, seed, shift, intensity)
                for method in METHODS:
                    m = model.clone()
                    t0 = time.time()
                    if method != "source_only":
                        adapt_classifier(m, sb, method=method, seed=seed, steps=ADAPT_STEPS,
                                         lr=0.01, lambda_cal=0.1, lambda_af=0.01)
                    runtime = time.time() - t0
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    results.append({
                        "scale": n, "seed": seed, "shift": shift, "intensity": intensity,
                        "method": method, "accuracy": metrics["accuracy"], "ece": metrics["ece"],
                        "nll": metrics["nll"], "runtime_seconds": runtime,
                    })
                    print(f"n={n} seed={seed} {shift}({intensity}) {method}: "
                          f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f} time={runtime:.3f}s")
            # persist incrementally
            (output_dir / "scalability_study.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    (output_dir / "scalability_study.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    agg = defaultdict(list)
    for r in results:
        agg[(r["scale"], r["method"])].append(r)
    with (output_dir / "scalability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "method", "mean_accuracy", "mean_ece", "mean_runtime"])
        for (scale, method), recs in sorted(agg.items()):
            writer.writerow([scale, method,
                             f"{np.mean([r['accuracy'] for r in recs]):.4f}",
                             f"{np.mean([r['ece'] for r in recs]):.4f}",
                             f"{np.mean([r['runtime_seconds'] for r in recs]):.4f}"])
    print(f"\nResults saved to {output_dir / 'scalability_study.json'}")


if __name__ == "__main__":
    run_scalability()
