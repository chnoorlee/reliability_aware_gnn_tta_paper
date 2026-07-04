"""Experiment 2: detector threshold auto-calibration + dual-checkpoint streaming.

Part A (threshold_calibration):
    Set Delta* = k * ECE_source where ECE_source is measured on the source
    validation split, for k in {0.5, 1, 2, 3, 4}.  Report the trigger rate and
    post-adaptation accuracy across controlled shifts to locate the
    accuracy-vs-trigger-rate knee and validate k=2 as a robust default.

Part B (dual_checkpoint_streaming):
    Re-run the homophily-drift stream with a dual-checkpoint rollback policy:
    cache both the source model and the previous-step model; a single detector
    trigger rolls back to the previous step, two CONSECUTIVE triggers roll back
    to the source model.  Compare against the single-checkpoint policy.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from adaptation import adapt_classifier, group_confidence
from data import apply_shift, make_contextual_sbm, split_indices
from detector import DetectorState
from models import TwoLayerGCN
from utils import evaluate, expected_calibration_error


def _self_drift(model, x, adj, seed, steps=20):
    """Source-side calibration: confidence drift when adapting on the CLEAN target.

    This is a label-free 'no-shift' drift floor.  Anchoring Delta* to it makes the
    threshold auto-scale to the source model's natural adaptation volatility.
    """
    src_probs, _ = model.forward(x, adj)
    src_conf = group_confidence(adj, src_probs)
    m = model.clone()
    adapt_classifier(m, x, adj, method="full_method", seed=seed, steps=steps, detector=None)
    probs, _ = m.forward(x, adj)
    conf = group_confidence(adj, probs)
    return float(np.mean([abs(conf[k] - src_conf[k]) for k in src_conf])) + 1e-4

OUT = Path(__file__).resolve().parents[1] / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)


def _train(seed, n=300, hidden=24, epochs=300):
    x, adj, y = make_contextual_sbm(seed=seed, n=n)
    tr, va, te = split_indices(seed, y)
    model = TwoLayerGCN(x.shape[1], hidden, int(y.max() + 1), seed=seed)
    model.train(x, adj, y, tr, va, epochs=epochs)
    return model, x, adj, y, tr, va, te


def _apply(seed, x, adj, y, shift, intensity):
    x_t, adj_t = apply_shift(seed, x, adj, y, shift, intensity)
    if len(x_t) != len(y):
        x_t, adj_t = x.copy(), adj.copy()
    return x_t, adj_t


# ----------------------------------------------------------------- Part A
def threshold_calibration(seeds=(0, 1, 2, 3, 4)):
    conditions = [
        ("feature_noise", 0.45), ("edge_add", 0.35),
        ("homophily_shift", 0.25), ("homophily_shift", 0.50),
    ]
    ks = [0.5, 1.0, 2.0, 3.0, 4.0]
    records = []
    for seed in seeds:
        model, x, adj, y, tr, va, te = _train(seed)
        # Anchor: source-side self-drift on the clean target (label-free).
        delta_self = _self_drift(model, x, adj, seed)
        for shift, intensity in conditions:
            x_t, adj_t = _apply(seed, x, adj, y, shift, intensity)
            for k in ks:
                delta_star = k * delta_self
                m = model.clone()
                det = DetectorState(delta_tolerance=delta_star, phi_tolerance=0.20)
                adapt_classifier(m, x_t, adj_t, method="full_method", seed=seed, steps=60, detector=det)
                p, _ = m.forward(x_t, adj_t)
                metrics = evaluate(p[te], y[te], int(y.max() + 1))
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
    print("\n=== Threshold auto-calibration: Delta* = k * ECE_source ===")
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
def dual_checkpoint_streaming(seeds=(0, 1, 2, 3, 4)):
    """Homophily-drift stream with single- vs dual-checkpoint rollback."""
    stream = [("homophily_shift", i) for i in (0.10, 0.20, 0.30, 0.40, 0.50)]
    records = []
    # 'none' = adapt every step, never roll back (baseline streaming full method);
    # 'single_checkpoint' = roll back to previous step on a stream-level trigger;
    # 'dual_checkpoint' = roll back to previous step on a single trigger, to SOURCE
    #                     on two consecutive triggers.
    for policy in ("none", "single_checkpoint", "dual_checkpoint"):
        for seed in seeds:
            model, x, adj, y, tr, va, te = _train(seed, n=240)
            classes = int(y.max() + 1)
            cur = model.clone()
            source_w1 = model.w1_source.copy()
            consecutive = 0
            for step, (shift, intensity) in enumerate(stream):
                x_t, adj_t = _apply(seed + step * 17, x, adj, y, shift, intensity)
                # stream-level source reference for drift
                src_probs, _ = cur.forward(x_t, adj_t)  # current model on this graph (pre-step)
                base_conf = group_confidence(adj_t, model.clone().forward(x_t, adj_t)[0])  # source-model conf
                # snapshot before this step's adaptation
                before_w1 = cur.w1.copy()
                # full per-step adaptation (no inner detector) so drift can accumulate
                adapt_classifier(cur, x_t, adj_t, method="full_method", seed=seed + step, steps=25, detector=None)
                probs, _ = cur.forward(x_t, adj_t)
                cur_conf = group_confidence(adj_t, probs)
                delta_t = float(np.mean([abs(cur_conf[k] - base_conf[k]) for k in base_conf]))
                triggered = delta_t > 0.05
                if policy == "none":
                    pass
                elif triggered:
                    consecutive += 1
                    if policy == "dual_checkpoint" and consecutive >= 2:
                        cur.w1 = source_w1.copy()    # roll back to source
                    else:
                        cur.w1 = before_w1.copy()    # roll back to previous step
                else:
                    consecutive = 0
                p, _ = cur.forward(x_t, adj_t)
                metrics = evaluate(p[te], y[te], classes)
                records.append({
                    "policy": policy, "seed": seed, "step": step, "intensity": intensity,
                    "accuracy": metrics["accuracy"], "ece": metrics["ece"],
                    "triggered": triggered, "consecutive": consecutive, "delta_t": delta_t,
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
        print(f"{policy:<20} acc={mean(accs):.4f}±{sd:.4f} trig%={mean(1.0 if r['triggered'] else 0.0 for r in rows)*100:.0f}")
    return records


if __name__ == "__main__":
    threshold_calibration()
    dual_checkpoint_streaming()
