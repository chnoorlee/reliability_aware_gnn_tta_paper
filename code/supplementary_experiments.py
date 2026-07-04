"""Supplementary experiments for the Neurocomputing revision.

These experiments fill the empirical gaps that reviewers would flag as
"hard injuries":

1.  ``backbone_study``           -- multi-backbone evaluation (GCN, GAT, SAGE, APPNP)
2.  ``assumption_verification``  -- empirical Cov(r_i, e_i) on labeled splits
3.  ``detector_sensitivity``     -- sweep of Delta* and Phi* tolerances
4.  ``lambda_sensitivity``       -- sweep of calibration / anti-forgetting weights
5.  ``drift_trajectory``         -- per-subgroup confidence-drift trajectory
6.  ``component_correlation``    -- correlation of the five reliability signals

All runs use deterministic seeds and write JSON/CSV summaries to
``results/supplementary/``.  The intent is to keep the new evidence
self-contained, fast (CPU-only), and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from adaptation import (
    adapt_classifier,
    entropy,
    group_confidence,
    neighborhood_agreement,
    reliability_scores,
    source_consistency,
    structural_stability,
    top_margin,
)
from backbones import make_backbone
from data import apply_shift, make_contextual_sbm, split_indices
from detector import DetectorState
from utils import evaluate


def _build(seed, shift, intensity, backbone="gcn", n=300, hidden=24, train_epochs=300):
    from utils import degree_vector
    x, adj, y = make_contextual_sbm(seed=seed, n=n)
    train_idx, val_idx, test_idx = split_indices(seed, y)
    classes = int(np.max(y)) + 1
    model = make_backbone(backbone, x.shape[1], hidden, classes, seed=seed)
    train_info = model.train(x, adj, y, train_idx, val_idx, epochs=train_epochs)
    x_t, adj_t = apply_shift(seed, x, adj, y, shift, intensity)
    # degree_shift may remove nodes; replicate main.py's topology-only fallback.
    if len(x_t) != len(y):
        x_t, adj_t = x.copy(), adj.copy()
        deg = degree_vector(adj_t)
        high = np.where(deg > np.quantile(deg, 0.65))[0]
        rng = np.random.default_rng(seed + 131)
        for i in high:
            if hasattr(adj_t, "tocsr"):
                row = adj_t.tocsr()
                neigh = row.indices[row.indptr[i]: row.indptr[i + 1]]
            else:
                neigh = np.where(adj_t[i] > 0)[0]
            drop_count = int(len(neigh) * min(0.6, intensity))
            if drop_count > 0:
                drop = rng.choice(neigh, size=drop_count, replace=False)
                adj_t[i, drop] = 0.0
                adj_t[drop, i] = 0.0
    return model, x, adj, y, x_t, adj_t, classes, train_idx, val_idx, test_idx, train_info


# --------------------------------------------------------------------------- 1
def backbone_study(out_dir, seeds=(0, 1, 2), conditions=None):
    """Compare reliability-aware TTA across GCN, GAT, SAGE, APPNP backbones."""
    if conditions is None:
        conditions = [
            ("clean", 0.0),
            ("feature_noise", 0.45),
            ("edge_drop", 0.35),
            ("edge_add", 0.35),
            ("homophily_shift", 0.25),
            ("homophily_shift", 0.50),
        ]
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    backbones = ["gcn", "gat", "graphsage", "appnp"]
    records = []
    for backbone in backbones:
        for seed in seeds:
            for shift, intensity in conditions:
                model, x, adj, y, x_t, adj_t, classes, train_idx, val_idx, test_idx, _ = _build(
                    seed, shift, intensity, backbone=backbone
                )
                for method in methods:
                    m = model.clone()
                    detector = DetectorState() if method == "full_method" else None
                    t0 = time.perf_counter()
                    info = adapt_classifier(m, x_t, adj_t, method=method, seed=seed, steps=60, detector=detector)
                    runtime = time.perf_counter() - t0
                    probs, _ = m.forward(x_t, adj_t)
                    metrics = evaluate(probs[test_idx], y[test_idx], classes)
                    metrics.update({
                        "seed": seed,
                        "backbone": backbone,
                        "shift": shift,
                        "intensity": intensity,
                        "method": method,
                        "runtime_seconds": runtime,
                        "detector_triggered": bool(detector.triggered) if detector is not None else None,
                        "detector_trigger_step": detector.trigger_step if detector is not None else None,
                    })
                    records.append(metrics)
                    print(
                        f"[backbone] {backbone} seed={seed} {shift}/{intensity} {method}: "
                        f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}"
                    )
    _write(out_dir / "backbone_study.json", records)
    _write_csv(out_dir / "backbone_study.csv", records)
    return records


# --------------------------------------------------------------------------- 2
def assumption_verification(out_dir, seeds=(0, 1, 2, 3, 4), conditions=None):
    """Empirically verify Assumption 1: Cov(r_i, e_i) <= 0 inside the envelope."""
    if conditions is None:
        conditions = [
            ("clean", 0.0),
            ("feature_noise", 0.20),
            ("feature_noise", 0.45),
            ("edge_drop", 0.15),
            ("edge_drop", 0.35),
            ("edge_add", 0.15),
            ("edge_add", 0.35),
            ("degree_shift", 0.35),
            ("homophily_shift", 0.25),
            ("homophily_shift", 0.50),
        ]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, x, adj, y, x_t, adj_t, _, _, _, test_idx, _ = _build(seed, shift, intensity, n=300, hidden=24)
            probs, _ = model.forward(x_t, adj_t)
            source_probs = probs.copy()
            r, parts = reliability_scores(model, x_t, adj_t, seed=seed, reference_probs=source_probs)
            pred = np.argmax(probs, axis=1)
            e = (pred != y).astype(float)
            cov_re = float(np.cov(r, e, bias=True)[0, 1])
            corr_re = float(np.corrcoef(r, e)[0, 1]) if np.std(r) > 1e-9 and np.std(e) > 1e-9 else 0.0
            # rank correlation via Spearman (no scipy needed)
            ranks_r = np.argsort(np.argsort(r))
            ranks_e = np.argsort(np.argsort(e))
            rank_corr = float(np.corrcoef(ranks_r, ranks_e)[0, 1]) if np.std(ranks_r) > 1e-9 else 0.0
            r_top = r >= np.quantile(r, 0.5)
            error_top = float(np.mean(e[r_top])) if np.any(r_top) else 0.0
            error_bot = float(np.mean(e[~r_top])) if np.any(~r_top) else 0.0
            records.append({
                "seed": seed,
                "shift": shift,
                "intensity": intensity,
                "cov_r_e": cov_re,
                "corr_r_e": corr_re,
                "spearman_r_e": rank_corr,
                "error_rate_top50_reliability": error_top,
                "error_rate_bottom50_reliability": error_bot,
                "estimated_homophily": float(parts.get("estimated_homophily", 0.0)),
                "mean_reliability": float(np.mean(r)),
                "assumption_holds": bool(cov_re <= 0.0),
            })
            print(f"[assumption] seed={seed} {shift}/{intensity}: cov(r,e)={cov_re:.5f} holds={cov_re<=0}")
    _write(out_dir / "assumption_verification.json", records)
    _write_csv(out_dir / "assumption_verification.csv", records)
    return records


# --------------------------------------------------------------------------- 3
def detector_sensitivity(out_dir, seeds=(0, 1, 2)):
    """Sweep Delta* and Phi* on the negative-adaptation regime."""
    conditions = [
        ("homophily_shift", 0.50),   # outside envelope -- detector should fire
        ("homophily_shift", 0.25),   # boundary -- detector may or may not fire
        ("feature_noise", 0.45),     # inside envelope -- detector should NOT fire
    ]
    tolerance_grid = [
        (0.02, 0.10),
        (0.05, 0.20),
        (0.10, 0.30),
        (0.20, 0.50),
        (0.50, 1.00),  # effectively disabled
    ]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, x, adj, y, x_t, adj_t, classes, _, _, test_idx, _ = _build(seed, shift, intensity, n=300, hidden=24)
            for delta_tol, phi_tol in tolerance_grid:
                m = model.clone()
                detector = DetectorState(delta_tolerance=delta_tol, phi_tolerance=phi_tol)
                info = adapt_classifier(m, x_t, adj_t, method="full_method", seed=seed, steps=60, detector=detector)
                probs, _ = m.forward(x_t, adj_t)
                metrics = evaluate(probs[test_idx], y[test_idx], classes)
                records.append({
                    "seed": seed,
                    "shift": shift,
                    "intensity": intensity,
                    "delta_tolerance": delta_tol,
                    "phi_tolerance": phi_tol,
                    "triggered": bool(detector.triggered),
                    "trigger_step": detector.trigger_step,
                    "trigger_reason": detector.trigger_reason,
                    "final_accuracy": float(metrics["accuracy"]),
                    "final_ece": float(metrics["ece"]),
                    "max_delta_observed": float(max(detector.delta_history) if detector.delta_history else 0.0),
                    "max_phi_observed": float(max(detector.phi_history) if detector.phi_history else 0.0),
                })
                print(
                    f"[detector] seed={seed} {shift}/{intensity} tol=({delta_tol},{phi_tol}): "
                    f"triggered={detector.triggered} acc={metrics['accuracy']:.4f}"
                )
    _write(out_dir / "detector_sensitivity.json", records)
    _write_csv(out_dir / "detector_sensitivity.csv", records)
    return records


# --------------------------------------------------------------------------- 4
def lambda_sensitivity(out_dir, seeds=(0, 1, 2)):
    """Sweep lambda_cal and lambda_af to test robustness to hyperparameters."""
    conditions = [
        ("feature_noise", 0.45),
        ("edge_add", 0.35),
        ("homophily_shift", 0.25),
    ]
    lambda_grid = [
        (0.0, 0.0),
        (0.1, 0.001),
        (0.5, 0.01),    # default
        (1.0, 0.05),
        (2.0, 0.10),
    ]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, x, adj, y, x_t, adj_t, classes, _, _, test_idx, _ = _build(seed, shift, intensity, n=300, hidden=24)
            for lam_cal, lam_af in lambda_grid:
                m = model.clone()
                info = adapt_classifier(
                    m, x_t, adj_t, method="full_method", seed=seed, steps=60,
                    lambda_cal=lam_cal, lambda_af=lam_af, detector=None,
                )
                probs, _ = m.forward(x_t, adj_t)
                metrics = evaluate(probs[test_idx], y[test_idx], classes)
                records.append({
                    "seed": seed,
                    "shift": shift,
                    "intensity": intensity,
                    "lambda_cal": lam_cal,
                    "lambda_af": lam_af,
                    "accuracy": float(metrics["accuracy"]),
                    "ece": float(metrics["ece"]),
                    "nll": float(metrics["nll"]),
                    "brier": float(metrics["brier"]),
                })
                print(
                    f"[lambda] seed={seed} {shift}/{intensity} lambda=({lam_cal},{lam_af}): "
                    f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}"
                )
    _write(out_dir / "lambda_sensitivity.json", records)
    _write_csv(out_dir / "lambda_sensitivity.csv", records)
    return records


# --------------------------------------------------------------------------- 5
def drift_trajectory(out_dir, seeds=(0, 1, 2)):
    """Log per-subgroup confidence drift Delta_t^k across adaptation steps."""
    conditions = [
        ("feature_noise", 0.45),
        ("edge_add", 0.35),
        ("homophily_shift", 0.50),
    ]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, x, adj, y, x_t, adj_t, classes, _, _, test_idx, _ = _build(seed, shift, intensity, n=300, hidden=24)
            m = model.clone()
            info = adapt_classifier(m, x_t, adj_t, method="full_method", seed=seed, steps=60, detector=None)
            drift = info["drift_trace"]
            for step in range(len(drift["low"])):
                records.append({
                    "seed": seed,
                    "shift": shift,
                    "intensity": intensity,
                    "step": step,
                    "drift_low": drift["low"][step],
                    "drift_mid": drift["mid"][step],
                    "drift_high": drift["high"][step],
                    "delta_t": info["delta_trace"][step],
                    "phi_t": info["phi_trace"][step],
                })
            print(f"[drift] seed={seed} {shift}/{intensity}: T={len(drift['low'])} steps logged")
    _write(out_dir / "drift_trajectory.json", records)
    _write_csv(out_dir / "drift_trajectory.csv", records)
    return records


# --------------------------------------------------------------------------- 6
def component_correlation(out_dir, seeds=(0, 1, 2, 3, 4)):
    """Pairwise correlation of the five reliability signals across conditions."""
    conditions = [
        ("clean", 0.0),
        ("feature_noise", 0.45),
        ("edge_drop", 0.35),
        ("edge_add", 0.35),
        ("homophily_shift", 0.25),
    ]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, x, adj, y, x_t, adj_t, _, _, _, test_idx, _ = _build(seed, shift, intensity, n=300, hidden=24)
            probs, _ = model.forward(x_t, adj_t)
            source_probs = probs.copy()
            r, parts = reliability_scores(model, x_t, adj_t, seed=seed, reference_probs=source_probs)
            signals = {
                "confidence": parts["confidence"],
                "agreement": parts["agreement"],
                "stability": parts["stability"],
                "source": parts["source"],
                "degree": parts["degree"],
            }
            names = list(signals)
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    if i >= j:
                        continue
                    a = signals[ni]
                    b = signals[nj]
                    if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                        c = float(np.corrcoef(a, b)[0, 1])
                    else:
                        c = 0.0
                    records.append({
                        "seed": seed,
                        "shift": shift,
                        "intensity": intensity,
                        "signal_a": ni,
                        "signal_b": nj,
                        "correlation": c,
                    })
            print(f"[corr] seed={seed} {shift}/{intensity} done")
    _write(out_dir / "component_correlation.json", records)
    _write_csv(out_dir / "component_correlation.csv", records)
    return records


# --------------------------------------------------------------------------- io
def _write(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")


def _write_csv(path: Path, records):
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        default="all",
        help=(
            "comma-separated list among: backbone_study, assumption_verification, "
            "detector_sensitivity, lambda_sensitivity, drift_trajectory, component_correlation, all"
        ),
    )
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "supplementary"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = args.experiments.split(",")
    if "all" in experiments:
        experiments = [
            "assumption_verification",
            "detector_sensitivity",
            "lambda_sensitivity",
            "drift_trajectory",
            "component_correlation",
            "backbone_study",
        ]

    started = time.perf_counter()
    if "assumption_verification" in experiments:
        assumption_verification(out_dir)
    if "detector_sensitivity" in experiments:
        detector_sensitivity(out_dir)
    if "lambda_sensitivity" in experiments:
        lambda_sensitivity(out_dir)
    if "drift_trajectory" in experiments:
        drift_trajectory(out_dir)
    if "component_correlation" in experiments:
        component_correlation(out_dir)
    if "backbone_study" in experiments:
        backbone_study(out_dir)
    print(f"supplementary experiments completed in {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
