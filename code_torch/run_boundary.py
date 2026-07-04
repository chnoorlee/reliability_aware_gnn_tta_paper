"""Boundary-region experiment (PyG port of code/run_boundary.py).

Actor (h~0.38) and Film (h~0.45) analogues sit just below the predicted
operating-envelope threshold; if the boundary is tight, no TTA variant should
significantly beat source-only.  Writes results_torch/extended/boundary_homophily.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from _np_bridge import evaluate
from adaptation import adapt_classifier
from exp_common import shift_bundle, train_source

OUT = Path(__file__).resolve().parents[1] / "results_torch" / "extended"

METHOD_MAP = {
    "source_only": "source_only",
    "tent_entropy": "Tent",
    "eata_filter": "EATA",
    "matcha_reliable": "Matcha",
    "full_method": "Full method",
}


def run(seeds=(0, 1, 2, 3, 4)):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("actor", "film"):
        for seed in seeds:
            model, base, _ = train_source(name, seed, hidden=24, epochs=300,
                                          train_per_class=12, val_per_class=12)
            clean_p = model.predict_probs(base.x, base.edge_index).cpu().numpy()
            clean_acc = float(np.mean(np.argmax(clean_p[base.test_idx], axis=1) == base.y_np[base.test_idx]))
            sb = shift_bundle(base, seed, "feature_noise", 0.05)
            for method in METHOD_MAP:
                m = model.clone()
                adapt_classifier(m, sb, method=method, seed=seed, steps=35)
                p = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(p[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "dataset": name, "seed": seed, "method": method,
                    "clean_acc": clean_acc, "shifted_acc": metrics["accuracy"], "ece": metrics["ece"],
                })
            print(f"[boundary] {name} seed={seed} clean_acc={clean_acc:.4f}")
    OUT.joinpath("boundary_homophily.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["dataset"]][r["method"]].append(r)
    print("\n=== Boundary region (mean over seeds) ===")
    for ds in ("actor", "film"):
        for method in METHOD_MAP:
            rows = by[ds][method]
            print(f"{ds:<7} {METHOD_MAP[method]:<14} clean={mean(r['clean_acc'] for r in rows):.4f} "
                  f"shifted={mean(r['shifted_acc'] for r in rows):.4f} ece={mean(r['ece'] for r in rows):.4f}")
    return records


if __name__ == "__main__":
    run()
