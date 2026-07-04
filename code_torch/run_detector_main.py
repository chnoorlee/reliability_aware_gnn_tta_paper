"""Detector-enabled full method on the MAIN synthetic configuration.

Produces the stored numbers behind the ``Full method + detector'' row of the
operating-envelope table (homophily shift 0.50, the out-of-envelope condition),
under exactly the same protocol as ``main.py`` (n=360, 5 seeds, 90 adaptation
steps) but with the negative-adaptation detector active at its default
tolerances.  Writes results_torch/detector_main.json + .csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from _np_bridge import evaluate
from adaptation import adapt_classifier
from detector import DetectorState
from exp_common import shift_bundle, train_source

OUT = Path(__file__).resolve().parents[1] / "results_torch"
CONDITIONS = [("homophily_shift", 0.50), ("homophily_shift", 0.25)]
SEEDS = [0, 1, 2, 3, 4]


def run(k_auto=2.0):
    from detector_calibration import _self_drift
    records = []
    for seed in SEEDS:
        model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=360)
        delta_self = _self_drift(model, base, seed)
        # Two detector variants: the fixed legacy default (Delta*=0.05) and the
        # paper's source-side auto-calibration (Delta* = k * delta_self).
        variants = [
            ("full_method_detector_fixed", DetectorState()),
            ("full_method_detector_auto", DetectorState(delta_tolerance=k_auto * delta_self)),
        ]
        for shift, intensity in CONDITIONS:
            sb = shift_bundle(base, seed, shift, intensity)
            for mname, det_proto in variants:
                m = model.clone()
                det = DetectorState(delta_tolerance=det_proto.delta_tolerance,
                                    phi_tolerance=det_proto.phi_tolerance)
                info = adapt_classifier(m, sb, method="full_method", seed=seed, steps=90, detector=det)
                probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "seed": seed, "shift": shift, "intensity": intensity,
                    "method": mname, "delta_tolerance": det.delta_tolerance,
                    "delta_self": delta_self, "k_auto": k_auto if mname.endswith("auto") else None,
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "ece": metrics["ece"], "nll": metrics["nll"], "brier": metrics["brier"],
                    "triggered": bool(det.triggered), "trigger_step": det.trigger_step,
                    "trigger_reason": det.trigger_reason, "steps": info["steps"],
                })
                print(f"[det-main] seed={seed} {shift}/{intensity} {mname}: acc={metrics['accuracy']:.4f} "
                      f"ece={metrics['ece']:.4f} trig={det.triggered}@{det.trigger_step}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector_main.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    with (OUT / "detector_main.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        w.writeheader()
        w.writerows(records)
    for shift, intensity in CONDITIONS:
        for mname in ("full_method_detector_fixed", "full_method_detector_auto"):
            rows = [r for r in records if r["shift"] == shift and abs(r["intensity"] - intensity) < 1e-9
                    and r["method"] == mname]
            if rows:
                print(f"{shift}/{intensity} {mname}: acc={np.mean([r['accuracy'] for r in rows]):.4f}"
                      f"+-{np.std([r['accuracy'] for r in rows]):.4f} "
                      f"ece={np.mean([r['ece'] for r in rows]):.4f} "
                      f"trig={sum(r['triggered'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    run()
