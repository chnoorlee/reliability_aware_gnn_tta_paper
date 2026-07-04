"""Supplementary experiments (PyG port of ``code/supplementary_experiments.py``).

Same six diagnostics, same seeds/conditions/grids and output schema, now on the
PyTorch Geometric stack:
  1. backbone_study         -- GCN / GAT / GraphSAGE / APPNP
  2. assumption_verification-- empirical Cov(r_i, e_i)
  3. detector_sensitivity   -- sweep Delta*, Phi*
  4. lambda_sensitivity     -- sweep (lambda_cal, lambda_af)
  5. drift_trajectory       -- per-subgroup confidence drift
  6. component_correlation  -- correlation of the five reliability signals

Writes JSON/CSV to ``results_torch/supplementary/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from _np_bridge import evaluate
from adaptation import _make_predict_fn, adapt_classifier
from detector import DetectorState
from exp_common import shift_bundle, train_source
from reliability import entropy, reliability_scores


def _build(seed, shift, intensity, backbone="gcn", n=300, hidden=24, train_epochs=300):
    model, base, _ = train_source("synthetic", seed, backbone=backbone, hidden=hidden,
                                  epochs=train_epochs, n=n)
    sb = shift_bundle(base, seed, shift, intensity)
    return model, base, sb


# --------------------------------------------------------------------------- 1
def backbone_study(out_dir, seeds=(0, 1, 2), conditions=None):
    if conditions is None:
        conditions = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
                      ("edge_add", 0.35), ("homophily_shift", 0.25), ("homophily_shift", 0.50)]
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    records = []
    for backbone in ["gcn", "gat", "graphsage", "appnp"]:
        for seed in seeds:
            for shift, intensity in conditions:
                model, base, sb = _build(seed, shift, intensity, backbone=backbone)
                for method in methods:
                    m = model.clone()
                    detector = DetectorState() if method == "full_method" else None
                    t0 = time.perf_counter()
                    adapt_classifier(m, sb, method=method, seed=seed, steps=60, detector=detector)
                    runtime = time.perf_counter() - t0
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    metrics.update({"seed": seed, "backbone": backbone, "shift": shift,
                                    "intensity": intensity, "method": method, "runtime_seconds": runtime,
                                    "detector_triggered": bool(detector.triggered) if detector else None,
                                    "detector_trigger_step": detector.trigger_step if detector else None})
                    records.append(metrics)
                    print(f"[backbone] {backbone} seed={seed} {shift}/{intensity} {method}: "
                          f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}")
    _write(out_dir / "backbone_study.json", records)
    _write_csv(out_dir / "backbone_study.csv", records)
    return records


# --------------------------------------------------------------------------- 2
def assumption_verification(out_dir, seeds=(0, 1, 2, 3, 4), conditions=None):
    if conditions is None:
        conditions = [("clean", 0.0), ("feature_noise", 0.20), ("feature_noise", 0.45),
                      ("edge_drop", 0.15), ("edge_drop", 0.35), ("edge_add", 0.15),
                      ("edge_add", 0.35), ("degree_shift", 0.35), ("homophily_shift", 0.25),
                      ("homophily_shift", 0.50)]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, base, sb = _build(seed, shift, intensity, n=300, hidden=24)
            predict_fn = _make_predict_fn(model)
            probs = predict_fn(sb.x_np, sb.adj)
            source_probs = probs.copy()
            r, parts = reliability_scores(predict_fn, sb.x_np, sb.adj, seed=seed, reference_probs=source_probs)
            pred = np.argmax(probs, axis=1)
            e = (pred != sb.y_np).astype(float)
            cov_re = float(np.cov(r, e, bias=True)[0, 1])
            corr_re = float(np.corrcoef(r, e)[0, 1]) if np.std(r) > 1e-9 and np.std(e) > 1e-9 else 0.0
            # gradient-weighted error mass g_i = p_iyhat * |H(p_i) + log p_iyhat|
            # (the quantity that actually enters the Theorem-1 covariance condition)
            pmax = np.max(probs, axis=1)
            ent = entropy(probs)
            g = pmax * np.abs(ent + np.log(np.clip(pmax, 1e-12, 1.0)))
            eg = e * g
            cov_reg = float(np.cov(r, eg, bias=True)[0, 1])
            corr_reg = float(np.corrcoef(r, eg)[0, 1]) if np.std(r) > 1e-9 and np.std(eg) > 1e-9 else 0.0
            ranks_r = np.argsort(np.argsort(r))
            ranks_e = np.argsort(np.argsort(e))
            rank_corr = float(np.corrcoef(ranks_r, ranks_e)[0, 1]) if np.std(ranks_r) > 1e-9 else 0.0
            r_top = r >= np.quantile(r, 0.5)
            error_top = float(np.mean(e[r_top])) if np.any(r_top) else 0.0
            error_bot = float(np.mean(e[~r_top])) if np.any(~r_top) else 0.0
            records.append({"seed": seed, "shift": shift, "intensity": intensity, "cov_r_e": cov_re,
                            "corr_r_e": corr_re, "spearman_r_e": rank_corr,
                            "cov_r_eg": cov_reg, "corr_r_eg": corr_reg,
                            "error_rate_top50_reliability": error_top,
                            "error_rate_bottom50_reliability": error_bot,
                            "estimated_homophily": float(parts.get("estimated_homophily", 0.0)),
                            "mean_reliability": float(np.mean(r)),
                            "assumption_holds": bool(cov_re <= 0.0 and cov_reg <= 0.0)})
            print(f"[assumption] seed={seed} {shift}/{intensity}: cov(r,e)={cov_re:.5f} "
                  f"cov(r,eg)={cov_reg:.5f} holds={cov_re<=0 and cov_reg<=0}")
    _write(out_dir / "assumption_verification.json", records)
    _write_csv(out_dir / "assumption_verification.csv", records)
    return records


# --------------------------------------------------------------------------- 3
def detector_sensitivity(out_dir, seeds=(0, 1, 2)):
    conditions = [("homophily_shift", 0.50), ("homophily_shift", 0.25), ("feature_noise", 0.45)]
    tolerance_grid = [(0.02, 0.10), (0.05, 0.20), (0.10, 0.30), (0.20, 0.50), (0.50, 1.00)]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, base, sb = _build(seed, shift, intensity, n=300, hidden=24)
            for delta_tol, phi_tol in tolerance_grid:
                m = model.clone()
                detector = DetectorState(delta_tolerance=delta_tol, phi_tolerance=phi_tol)
                adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=detector)
                probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({"seed": seed, "shift": shift, "intensity": intensity,
                                "delta_tolerance": delta_tol, "phi_tolerance": phi_tol,
                                "triggered": bool(detector.triggered), "trigger_step": detector.trigger_step,
                                "trigger_reason": detector.trigger_reason,
                                "final_accuracy": float(metrics["accuracy"]), "final_ece": float(metrics["ece"]),
                                "max_delta_observed": float(max(detector.delta_history) if detector.delta_history else 0.0),
                                "max_phi_observed": float(max(detector.phi_history) if detector.phi_history else 0.0)})
                print(f"[detector] seed={seed} {shift}/{intensity} tol=({delta_tol},{phi_tol}): "
                      f"triggered={detector.triggered} acc={metrics['accuracy']:.4f}")
    _write(out_dir / "detector_sensitivity.json", records)
    _write_csv(out_dir / "detector_sensitivity.csv", records)
    return records


# --------------------------------------------------------------------------- 4
def lambda_sensitivity(out_dir, seeds=(0, 1, 2)):
    conditions = [("feature_noise", 0.45), ("edge_add", 0.35), ("homophily_shift", 0.25)]
    lambda_grid = [(0.0, 0.0), (0.1, 0.001), (0.5, 0.01), (1.0, 0.05), (2.0, 0.10)]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, base, sb = _build(seed, shift, intensity, n=300, hidden=24)
            for lam_cal, lam_af in lambda_grid:
                m = model.clone()
                adapt_classifier(m, sb, method="full_method", seed=seed, steps=60,
                                 lambda_cal=lam_cal, lambda_af=lam_af, detector=None)
                probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({"seed": seed, "shift": shift, "intensity": intensity,
                                "lambda_cal": lam_cal, "lambda_af": lam_af,
                                "accuracy": float(metrics["accuracy"]), "ece": float(metrics["ece"]),
                                "nll": float(metrics["nll"]), "brier": float(metrics["brier"])})
                print(f"[lambda] seed={seed} {shift}/{intensity} lambda=({lam_cal},{lam_af}): "
                      f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}")
    _write(out_dir / "lambda_sensitivity.json", records)
    _write_csv(out_dir / "lambda_sensitivity.csv", records)
    return records


# --------------------------------------------------------------------------- 5
def drift_trajectory(out_dir, seeds=(0, 1, 2)):
    conditions = [("feature_noise", 0.45), ("edge_add", 0.35), ("homophily_shift", 0.50)]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, base, sb = _build(seed, shift, intensity, n=300, hidden=24)
            m = model.clone()
            info = adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=None)
            drift = info["drift_trace"]
            for step in range(len(drift["low"])):
                records.append({"seed": seed, "shift": shift, "intensity": intensity, "step": step,
                                "drift_low": drift["low"][step], "drift_mid": drift["mid"][step],
                                "drift_high": drift["high"][step], "delta_t": info["delta_trace"][step],
                                "phi_t": info["phi_trace"][step]})
            print(f"[drift] seed={seed} {shift}/{intensity}: T={len(drift['low'])} steps logged")
    _write(out_dir / "drift_trajectory.json", records)
    _write_csv(out_dir / "drift_trajectory.csv", records)
    return records


# --------------------------------------------------------------------------- 6
def component_correlation(out_dir, seeds=(0, 1, 2, 3, 4)):
    conditions = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
                  ("edge_add", 0.35), ("homophily_shift", 0.25)]
    records = []
    for seed in seeds:
        for shift, intensity in conditions:
            model, base, sb = _build(seed, shift, intensity, n=300, hidden=24)
            predict_fn = _make_predict_fn(model)
            source_probs = predict_fn(sb.x_np, sb.adj)
            r, parts = reliability_scores(predict_fn, sb.x_np, sb.adj, seed=seed, reference_probs=source_probs)
            signals = {k: parts[k] for k in ["confidence", "agreement", "stability", "source", "degree"]}
            names = list(signals)
            for i, ni in enumerate(names):
                for j, nj in enumerate(names):
                    if i >= j:
                        continue
                    a, b = signals[ni], signals[nj]
                    c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0
                    records.append({"seed": seed, "shift": shift, "intensity": intensity,
                                    "signal_a": ni, "signal_b": nj, "correlation": c})
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
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results_torch" / "supplementary"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = args.experiments.split(",")
    if "all" in experiments:
        experiments = ["assumption_verification", "detector_sensitivity", "lambda_sensitivity",
                       "drift_trajectory", "component_correlation", "backbone_study"]
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
    print(f"supplementary experiments completed in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
