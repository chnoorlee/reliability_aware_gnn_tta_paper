"""Detector threshold auto-calibration + dual-checkpoint streaming
(PyG port of code/detector_calibration.py).

Part A (threshold_calibration):
    Delta* = k * delta_self where delta_self is the label-free confidence drift
    measured when adapting on the CLEAN target (source-side anchor), for
    k in {0.5, 1, 2, 3, 4}.

Part B (dual_checkpoint_streaming):
    Homophily-drift stream with three rollback policies: none /
    single_checkpoint (previous step) / dual_checkpoint (source on two
    consecutive triggers).

Writes results_torch/extended/{threshold_calibration,dual_checkpoint_streaming}.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import _make_predict_fn, adapt_classifier
from detector import DetectorState
from exp_common import shift_bundle, train_source
from reliability import group_confidence

OUT = Path(__file__).resolve().parents[1] / "results_torch" / "extended"


def _self_drift(model, base, seed, steps=20):
    """Label-free 'no-shift' drift floor: confidence drift when adapting on the
    clean target.  Anchors Delta* to the source model's natural volatility."""
    predict_fn = _make_predict_fn(model)
    src_probs = predict_fn(base.x_np, base.adj)
    src_conf = group_confidence(base.adj, src_probs)
    m = model.clone()
    adapt_classifier(m, base, method="full_method", seed=seed, steps=steps, detector=None)
    probs = _make_predict_fn(m)(base.x_np, base.adj)
    conf = group_confidence(base.adj, probs)
    return float(np.mean([abs(conf[k] - src_conf[k]) for k in src_conf])) + 1e-4


# ----------------------------------------------------------------- Part A
def threshold_calibration(seeds=(0, 1, 2, 3, 4)):
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = [("feature_noise", 0.45), ("edge_add", 0.35),
                  ("homophily_shift", 0.25), ("homophily_shift", 0.50)]
    ks = [0.5, 1.0, 2.0, 3.0, 4.0]
    records = []
    for seed in seeds:
        model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=300)
        delta_self = _self_drift(model, base, seed)
        for shift, intensity in conditions:
            sb = shift_bundle(base, seed, shift, intensity)
            for k in ks:
                delta_star = k * delta_self
                m = model.clone()
                det = DetectorState(delta_tolerance=delta_star, phi_tolerance=0.20)
                adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=det)
                p = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(p[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "seed": seed, "shift": shift, "intensity": intensity, "k": k,
                    "delta_self": delta_self, "delta_star": delta_star,
                    "triggered": bool(det.triggered), "accuracy": metrics["accuracy"], "ece": metrics["ece"],
                })
            print(f"[calib] seed={seed} {shift}/{intensity} delta_self={delta_self:.4f} done")
    OUT.joinpath("threshold_calibration.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    by_k = defaultdict(lambda: defaultdict(list))
    for r in records:
        regime = "in_envelope" if not (r["shift"] == "homophily_shift" and r["intensity"] >= 0.50) else "out_envelope"
        by_k[r["k"]][regime].append(r)
    print("\n=== Threshold auto-calibration: Delta* = k * delta_self ===")
    print(f"{'k':<6}{'in-env trig%':<14}{'in-env acc':<14}{'out-env trig%':<15}{'out-env acc':<14}")
    for k in ks:
        ie = by_k[k]["in_envelope"]; oe = by_k[k]["out_envelope"]
        print(f"{k:<6}"
              f"{mean(1.0 if r['triggered'] else 0.0 for r in ie)*100:<14.1f}"
              f"{mean(r['accuracy'] for r in ie):<14.4f}"
              f"{mean(1.0 if r['triggered'] else 0.0 for r in oe)*100:<15.1f}"
              f"{mean(r['accuracy'] for r in oe):<14.4f}")
    return records


# ----------------------------------------------------------------- Part B
def dual_checkpoint_streaming(seeds=(0, 1, 2, 3, 4), k_auto=2.0):
    """Stream-level rollback with the paper's auto-calibrated threshold
    Delta* = k * delta_self.

    The NumPy implementation used a fixed Delta* = 0.05, which was tuned to the
    NumPy stack's drift scale.  The PyG models are far better calibrated and
    drift by ~1e-3 per stream step, so the fixed threshold never fires; the
    source-side auto-calibration (Part A) is the principled, stack-independent
    way to set the tolerance and is used here.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    stream = [("homophily_shift", i) for i in (0.10, 0.20, 0.30, 0.40, 0.50)]
    records = []
    for policy in ("none", "single_checkpoint", "dual_checkpoint"):
        for seed in seeds:
            model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=240)
            delta_star = k_auto * _self_drift(model, base, seed)
            cur = model.clone()
            source_w = model.classifier_weight().detach().clone()
            consecutive = 0
            for step, (shift, intensity) in enumerate(stream):
                sb = shift_bundle(base, seed + step * 17, shift, intensity)
                # source-model confidence on this step's graph (stream-level reference)
                base_conf = group_confidence(sb.adj, _make_predict_fn(model)(sb.x_np, sb.adj))
                before_w = cur.classifier_weight().detach().clone()
                adapt_classifier(cur, sb, method="full_method", seed=seed + step, steps=25, detector=None)
                probs = _make_predict_fn(cur)(sb.x_np, sb.adj)
                cur_conf = group_confidence(sb.adj, probs)
                delta_t = float(np.mean([abs(cur_conf[k] - base_conf[k]) for k in base_conf]))
                triggered = delta_t > delta_star
                if policy == "none":
                    pass
                elif triggered:
                    consecutive += 1
                    with torch.no_grad():
                        if policy == "dual_checkpoint" and consecutive >= 2:
                            cur.classifier_weight().copy_(source_w)     # roll back to source
                        else:
                            cur.classifier_weight().copy_(before_w)     # roll back to previous step
                else:
                    consecutive = 0
                p = cur.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(p[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "policy": policy, "seed": seed, "step": step, "intensity": intensity,
                    "accuracy": metrics["accuracy"], "ece": metrics["ece"],
                    "triggered": triggered, "consecutive": consecutive, "delta_t": delta_t,
                    "delta_star": delta_star, "k_auto": k_auto,
                })
            print(f"[dual] {policy} seed={seed} done")
    OUT.joinpath("dual_checkpoint_streaming.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    print("\n=== Streaming rollback policies (homophily drift), final step (0.50) ===")
    by = defaultdict(list)
    for r in records:
        if abs(r["intensity"] - 0.50) < 1e-9:
            by[r["policy"]].append(r)
    for policy in ("none", "single_checkpoint", "dual_checkpoint"):
        rows = by[policy]
        accs = [r["accuracy"] for r in rows]
        sd = stdev(accs) if len(accs) > 1 else 0.0
        print(f"{policy:<20} acc={mean(accs):.4f}+-{sd:.4f} "
              f"trig%={mean(1.0 if r['triggered'] else 0.0 for r in rows)*100:.0f}")
    return records


if __name__ == "__main__":
    threshold_calibration()
    dual_checkpoint_streaming()
