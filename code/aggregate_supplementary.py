"""Aggregate supplementary experiment outputs into compact summary tables.

Reads the JSON files produced by ``supplementary_experiments.py`` and
prints LaTeX-friendly summary statistics.  This file is only an
analysis utility; it does not run any new experiments.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SUPP = Path(__file__).resolve().parents[1] / "results" / "supplementary"


def _load(name):
    with (SUPP / f"{name}.json").open(encoding="utf-8") as f:
        return json.load(f)["records"]


def aggregate_backbone():
    records = _load("backbone_study")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[(r["backbone"], r["shift"], r["intensity"])][r["method"]].append(r)
    print("\n=== Backbone study (mean ± std accuracy / ECE) ===")
    print(f"{'backbone':<10} {'shift':<22} {'intensity':<10} {'method':<20} {'acc':<14} {'ece':<14}")
    for (backbone, shift, intensity), methods in grouped.items():
        for method, rows in methods.items():
            accs = [r["accuracy"] for r in rows]
            eces = [r["ece"] for r in rows]
            acc_s = f"{mean(accs):.4f}±{stdev(accs):.4f}" if len(accs) > 1 else f"{accs[0]:.4f}"
            ece_s = f"{mean(eces):.4f}±{stdev(eces):.4f}" if len(eces) > 1 else f"{eces[0]:.4f}"
            print(f"{backbone:<10} {shift:<22} {intensity:<10} {method:<20} {acc_s:<14} {ece_s:<14}")


def aggregate_assumption():
    records = _load("assumption_verification")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["shift"], r["intensity"])].append(r)
    print("\n=== Assumption 1: Cov(r, e) ≤ 0  (averaged over 5 seeds) ===")
    print(f"{'shift':<22} {'intensity':<10} {'cov(r,e)':<14} {'error gap':<14} {'h_G hat':<10} {'holds':<6}")
    rows = []
    for (shift, intensity), recs in grouped.items():
        cov = mean(r["cov_r_e"] for r in recs)
        std_cov = stdev(r["cov_r_e"] for r in recs)
        gap = mean(r["error_rate_bottom50_reliability"] - r["error_rate_top50_reliability"] for r in recs)
        hG = mean(r["estimated_homophily"] for r in recs)
        holds = all(r["assumption_holds"] for r in recs)
        print(f"{shift:<22} {intensity:<10} {cov:+.5f}±{std_cov:.5f} {gap:+.4f}        {hG:.3f}      {str(holds):<6}")
        rows.append({"shift": shift, "intensity": intensity, "cov": cov, "gap": gap, "hG": hG, "holds": holds})
    return rows


def aggregate_detector():
    records = _load("detector_sensitivity")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["shift"], r["intensity"], r["delta_tolerance"], r["phi_tolerance"])].append(r)
    print("\n=== Detector tolerance sensitivity (mean ± std over seeds) ===")
    print(f"{'shift':<22} {'intensity':<10} {'Δ*':<6} {'Φ*':<6} {'trig %':<10} {'acc':<14}")
    for key, recs in grouped.items():
        shift, intensity, delta, phi = key
        triggered_rate = mean(1.0 if r["triggered"] else 0.0 for r in recs)
        accs = [r["final_accuracy"] for r in recs]
        acc_s = f"{mean(accs):.4f}±{stdev(accs):.4f}" if len(accs) > 1 else f"{accs[0]:.4f}"
        print(f"{shift:<22} {intensity:<10} {delta:<6} {phi:<6} {triggered_rate*100:<10.1f} {acc_s:<14}")


def aggregate_lambda():
    records = _load("lambda_sensitivity")
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["shift"], r["intensity"], r["lambda_cal"], r["lambda_af"])].append(r)
    print("\n=== Hyperparameter sensitivity (mean ± std over seeds) ===")
    print(f"{'shift':<22} {'intensity':<10} {'λ_cal':<6} {'λ_af':<6} {'acc':<14} {'ece':<14}")
    for key, recs in grouped.items():
        shift, intensity, lc, la = key
        accs = [r["accuracy"] for r in recs]
        eces = [r["ece"] for r in recs]
        acc_s = f"{mean(accs):.4f}±{stdev(accs):.4f}" if len(accs) > 1 else f"{accs[0]:.4f}"
        ece_s = f"{mean(eces):.4f}±{stdev(eces):.4f}" if len(eces) > 1 else f"{eces[0]:.4f}"
        print(f"{shift:<22} {intensity:<10} {lc:<6} {la:<6} {acc_s:<14} {ece_s:<14}")


def aggregate_correlation():
    records = _load("component_correlation")
    pair_means = defaultdict(list)
    for r in records:
        pair_means[(r["signal_a"], r["signal_b"])].append(r["correlation"])
    print("\n=== Reliability-signal correlation (mean across conditions and seeds) ===")
    print(f"{'signal_a':<14} {'signal_b':<14} {'mean corr':<14}")
    for (a, b), vals in pair_means.items():
        print(f"{a:<14} {b:<14} {mean(vals):+.4f}")


def aggregate_drift():
    records = _load("drift_trajectory")
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["shift"], r["intensity"], r["step"])
        grouped[key]["drift_low"].append(r["drift_low"])
        grouped[key]["drift_mid"].append(r["drift_mid"])
        grouped[key]["drift_high"].append(r["drift_high"])
    # Print last step
    print("\n=== Drift trajectory: final-step subgroup confidence drift ===")
    seen = set()
    for (shift, intensity, step), vals in sorted(grouped.items()):
        key = (shift, intensity)
        # Get the final step row only by tracking the max step per condition
        max_step = max(s for sh, it, s in grouped if sh == shift and it == intensity)
        if step != max_step:
            continue
        if key in seen:
            continue
        seen.add(key)
        print(
            f"{shift:<22} {intensity:<10} "
            f"low={mean(vals['drift_low']):+.4f} "
            f"mid={mean(vals['drift_mid']):+.4f} "
            f"high={mean(vals['drift_high']):+.4f}"
        )


def main():
    aggregate_assumption()
    aggregate_detector()
    aggregate_lambda()
    aggregate_correlation()
    aggregate_drift()
    aggregate_backbone()


if __name__ == "__main__":
    main()
