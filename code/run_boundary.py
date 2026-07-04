"""Boundary-region experiment: Actor (h=0.38) and Film (h=0.45) analogues.

These sit just below the predicted operating-envelope threshold h>=0.5.  If the
boundary is tight, no TTA variant should significantly beat source-only here.
Persists results so the boundary table in the paper is reproducible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from adaptation import adapt_classifier
from data import apply_shift, make_heterophily_benchmark, split_indices
from models import TwoLayerGCN
from utils import evaluate

OUT = Path(__file__).resolve().parents[1] / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_MAP = {
    "source_only": "source_only",
    "tent_entropy": "Tent",
    "eata_filter": "EATA",
    "matcha_reliable": "Matcha",
    "full_method": "Full method",
}


def run(seeds=(0, 1, 2, 3, 4)):
    records = []
    for name in ("actor", "film"):
        for seed in seeds:
            x, adj, y = make_heterophily_benchmark(seed=seed, name=name)
            tr, va, te = split_indices(seed, y, train_per_class=12, val_per_class=12)
            classes = int(y.max() + 1)
            model = TwoLayerGCN(x.shape[1], 24, classes, seed=seed)
            model.train(x, adj, y, tr, va, epochs=300)
            clean_p, _ = model.forward(x, adj)
            clean_acc = float(np.mean(np.argmax(clean_p[te], axis=1) == y[te]))
            x_t, adj_t = apply_shift(seed, x, adj, y, "feature_noise", 0.05)
            if len(x_t) != len(y):
                x_t, adj_t = x.copy(), adj.copy()
            for method in METHOD_MAP:
                m = model.clone()
                adapt_classifier(m, x_t, adj_t, method=method, seed=seed, steps=35)
                p, _ = m.forward(x_t, adj_t)
                metrics = evaluate(p[te], y[te], classes)
                records.append({
                    "dataset": name, "seed": seed, "method": method,
                    "clean_acc": clean_acc, "shifted_acc": metrics["accuracy"], "ece": metrics["ece"],
                })
            print(f"[boundary] {name} seed={seed} clean_acc={clean_acc:.4f}")
    OUT.joinpath("boundary_homophily.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["dataset"]][r["method"]].append(r)
    print("\n=== Boundary region (mean over 5 seeds) ===")
    for ds in ("actor", "film"):
        for method in METHOD_MAP:
            rows = by[ds][method]
            print(f"{ds:<7} {METHOD_MAP[method]:<14} clean={mean(r['clean_acc'] for r in rows):.4f} "
                  f"shifted={mean(r['shifted_acc'] for r in rows):.4f} ece={mean(r['ece'] for r in rows):.4f}")
    return records


if __name__ == "__main__":
    run()
