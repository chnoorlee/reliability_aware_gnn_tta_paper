"""Scalability experiment on ogbn-arxiv-like subsets at varying scales."""
import csv
import json
import time
from pathlib import Path

import numpy as np

from adaptation import adapt_classifier
from data import make_arxiv_subset, apply_shift, split_indices
from models import TwoLayerGCN
from utils import degree_vector, evaluate, upper_triangle_edges


SCALES = [1000, 2000, 5000]
METHODS = ["source_only", "entropy_all_nodes", "tent_entropy", "eata_filter", "full_method"]
CONDITIONS = [("clean", 0.0), ("feature_noise", 0.20), ("edge_drop", 0.15)]
SEEDS = [0, 1, 2]
TRAIN_EPOCHS = 400
ADAPT_STEPS = 50
HIDDEN = 64


def run_scalability():
    results = []
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for n in SCALES:
        for seed in SEEDS:
            x, adj, y, train_idx, val_idx, test_idx = make_arxiv_subset(seed, n=n)
            num_classes = int(y.max()) + 1
            model = TwoLayerGCN(x.shape[1], HIDDEN, num_classes)
            model.train(x, adj, y, train_idx, val_idx, epochs=TRAIN_EPOCHS, lr=0.01, weight_decay=5e-4)
            w0_src, w1_src = model.w0.copy(), model.w1.copy()

            for shift, intensity in CONDITIONS:
                x_shifted, adj_shifted = apply_shift(seed, x, adj, y, shift, intensity)
                deg = degree_vector(adj_shifted)

                for method in METHODS:
                    model.w0 = w0_src.copy()
                    model.w1 = w1_src.copy()
                    model.w0_source = w0_src.copy()
                    model.w1_source = w1_src.copy()
                    t0 = time.time()

                    if method == "source_only":
                        pass
                    else:
                        adapt_classifier(
                            model, x_shifted, adj_shifted, method,
                            steps=ADAPT_STEPS,
                            lr=0.01,
                            lambda_cal=0.1,
                            lambda_af=0.01,
                        )

                    runtime = time.time() - t0
                    probs, _ = model.forward(x_shifted, adj_shifted)
                    metrics = evaluate(probs[test_idx], y[test_idx], num_classes)

                    record = {
                        "scale": n,
                        "seed": seed,
                        "shift": shift,
                        "intensity": intensity,
                        "method": method,
                        "accuracy": metrics["accuracy"],
                        "ece": metrics["ece"],
                        "nll": metrics["nll"],
                        "runtime_seconds": runtime,
                    }
                    results.append(record)
                    print(f"n={n} seed={seed} {shift}({intensity}) {method}: acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f} time={runtime:.3f}s")

    # Save results
    json_path = output_dir / "scalability_study.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate by scale and method
    csv_path = output_dir / "scalability_summary.csv"
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        key = (r["scale"], r["method"])
        agg[key].append(r)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "method", "mean_accuracy", "mean_ece", "mean_runtime"])
        for (scale, method), recs in sorted(agg.items()):
            acc = np.mean([r["accuracy"] for r in recs])
            ece = np.mean([r["ece"] for r in recs])
            rt = np.mean([r["runtime_seconds"] for r in recs])
            writer.writerow([scale, method, f"{acc:.4f}", f"{ece:.4f}", f"{rt:.4f}"])

    print(f"\nResults saved to {json_path} and {csv_path}")


if __name__ == "__main__":
    run_scalability()
