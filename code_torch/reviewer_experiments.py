r"""Reviewer-requested experiments (PyTorch Geometric stack).

Six diagnostics added in response to reviewer questions, all on the same PyG
model + adaptation stack as ``main.py`` / ``supplementary_experiments.py``:

  A. signal_ablation      -- per-signal leave-one-out over the FIVE reliability
                             signals (confidence c_i, agreement a_i, stability
                             s_i, source-consistency b_i, degree prior d_i),
                             across three homophily regimes.
  B. homophily_robustness -- sensitivity of the reliability weighting to a
                             deliberately misestimated \hat h_G + delta.
  C. detector_operating   -- detector false-halt / false-continue operating
                             characteristic.
  D. ece_bin_sensitivity  -- ECE as a function of bin count (5,10,15,20,25)
                             with standard and debiased estimators (Q3).
  E. degree_fairness      -- per-degree-decile accuracy, ECE, and coverage
                             with vs without the degree prior (Q4).
  F. proxy_failure        -- per-step \Delta_t/\Phi_t trace analysis to
                             identify detector blind spots where accuracy
                             degrades but proxies stay below thresholds (Q2).

Writes JSON/CSV under ``results_torch/supplementary/`` (A, B, D, E, F) and
``results_torch/extended/`` (C).  ``--render`` prints LaTeX table bodies and the
prose numbers from the stored files (nothing is invented).

Usage:
    python reviewer_experiments.py --experiments all
    python reviewer_experiments.py --render
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from _np_bridge import evaluate
import _np_bridge as _br
from adaptation import _make_predict_fn, adapt_classifier
from detector import DetectorState
from exp_common import shift_bundle, train_source
from reliability import group_confidence, reliability_scores

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "results_torch" / "supplementary"
EXT = ROOT / "results_torch" / "extended"

IN_ENVELOPE = [("clean", 0.0), ("feature_noise", 0.20), ("feature_noise", 0.45),
               ("edge_drop", 0.15), ("edge_drop", 0.35), ("edge_add", 0.15),
               ("edge_add", 0.35), ("degree_shift", 0.35)]

SIGNAL_METHODS = ["full_method", "no_confidence", "no_neighborhood_agreement",
                  "no_structural_stability", "no_source_consistency", "no_degree_prior"]

# Three homophily regimes for the per-signal ablation (reviewer Q5: "across
# homophily regimes").  In-envelope reuses the eight controlled shifts; boundary
# and out-of-envelope use the homophily-shift conditions.
ABLATION_REGIMES = {
    "in_envelope": IN_ENVELOPE,
    "boundary": [("homophily_shift", 0.25)],
    "out_envelope": [("homophily_shift", 0.50)],
}


def _signal_flags(method):
    """Map an ablation method name to per-signal enable flags."""
    return dict(
        use_confidence=method != "no_confidence",
        use_agreement=method != "no_neighborhood_agreement",
        use_stability=method != "no_structural_stability",
        use_source=method != "no_source_consistency",
        use_degree=method != "no_degree_prior",
    )


def _train_per_seed(seed, n=300, hidden=24, epochs=300):
    """Train one source model per seed; conditions reuse the same base graph."""
    model, base, _ = train_source("synthetic", seed, hidden=hidden, epochs=epochs, n=n)
    return model, base


def _self_drift(model, base, seed, steps=20):
    """Label-free no-shift drift floor (anchors the auto-calibrated Delta*)."""
    predict_fn = _make_predict_fn(model)
    src_conf = group_confidence(base.adj, predict_fn(base.x_np, base.adj))
    m = model.clone()
    adapt_classifier(m, base, method="full_method", seed=seed, steps=steps, detector=None)
    conf = group_confidence(base.adj, _make_predict_fn(m)(base.x_np, base.adj))
    return float(np.mean([abs(conf[k] - src_conf[k]) for k in src_conf])) + 1e-4


# --------------------------------------------------------------------------- A
def signal_ablation(seeds=(0, 1, 2, 3, 4)):
    SUPP.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        predict_fn = _make_predict_fn(model)
        for regime, conditions in ABLATION_REGIMES.items():
            for shift, intensity in conditions:
                sb = shift_bundle(base, seed, shift, intensity)
                src_probs = predict_fn(sb.x_np, sb.adj)
                err = (np.argmax(src_probs, axis=1) != sb.y_np).astype(float)
                for method in SIGNAL_METHODS:
                    # (i) downstream adaptation metrics under this signal ablation
                    m = model.clone()
                    t0 = time.perf_counter()
                    info = adapt_classifier(m, sb, method=method, seed=seed, steps=60, detector=None)
                    runtime = time.perf_counter() - t0
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    # (ii) score-level effect: how the ablation changes the
                    # reliability score's coupling with the error indicator
                    # (the quantity Assumption 1 / Theorem 1 depend on)
                    r_abl, _ = reliability_scores(predict_fn, sb.x_np, sb.adj, seed=seed,
                                                  reference_probs=src_probs, **_signal_flags(method))
                    cov_re = float(np.cov(r_abl, err, bias=True)[0, 1])
                    records.append({
                        "seed": seed, "regime": regime, "shift": shift, "intensity": intensity,
                        "method": method, "accuracy": float(metrics["accuracy"]),
                        "ece": float(metrics["ece"]), "brier": float(metrics["brier"]),
                        "runtime_seconds": runtime, "cov_r_e": cov_re,
                        "mean_reliability": info["mean_reliability"],
                        "selected_fraction": info["selected_fraction"],
                    })
                print(f"[signal_ablation] seed={seed} {regime} {shift}/{intensity} done")
    _write(SUPP / "signal_ablation.json", records)
    _write_csv(SUPP / "signal_ablation.csv", records)
    return records


# --------------------------------------------------------------------------- B
def homophily_robustness(seeds=(0, 1, 2, 3, 4)):
    SUPP.mkdir(parents=True, exist_ok=True)
    conditions = [("feature_noise", 0.45), ("edge_add", 0.35), ("homophily_shift", 0.25)]
    deltas = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        predict_fn = _make_predict_fn(model)
        for shift, intensity in conditions:
            sb = shift_bundle(base, seed, shift, intensity)
            # measured (unperturbed) homophily estimate on this target graph
            _, parts = reliability_scores(predict_fn, sb.x_np, sb.adj, seed=seed,
                                          reference_probs=predict_fn(sb.x_np, sb.adj))
            measured_h = float(parts["estimated_homophily"])
            for delta in deltas:
                m = model.clone()
                adapt_classifier(m, sb, method="full_method", seed=seed, steps=60,
                                 detector=None, rel_kwargs={"homophily_delta": delta})
                probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "seed": seed, "shift": shift, "intensity": intensity, "delta": delta,
                    "measured_homophily": measured_h,
                    "used_homophily": float(np.clip(measured_h + delta, 0.0, 1.0)),
                    "accuracy": float(metrics["accuracy"]), "ece": float(metrics["ece"]),
                })
            print(f"[homophily_robustness] seed={seed} {shift}/{intensity} "
                  f"measured_h={measured_h:.3f} done")
    _write(SUPP / "homophily_robustness.json", records)
    _write_csv(SUPP / "homophily_robustness.csv", records)
    return records


# --------------------------------------------------------------------------- C
def detector_operating(seeds=(0, 1, 2, 3, 4)):
    EXT.mkdir(parents=True, exist_ok=True)
    in_env = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
              ("edge_add", 0.35), ("degree_shift", 0.35)]
    out_env = [("homophily_shift", 0.50)]
    # (name, delta_tol, phi_tol); "auto" is filled in per-seed from self-drift.
    settings = [("safety", 0.02, 0.10), ("balanced", 0.05, 0.20),
                ("aggressive", 0.20, 0.50), ("auto", None, 0.20)]
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        delta_self = _self_drift(model, base, seed)
        for regime, conds in (("in_envelope", in_env), ("out_envelope", out_env)):
            for shift, intensity in conds:
                sb = shift_bundle(base, seed, shift, intensity)
                # reference: source-only and adapt-without-detector accuracy
                src_probs = model.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                src_acc = float(evaluate(src_probs[sb.test_idx], sb.y_np[sb.test_idx],
                                         sb.num_classes)["accuracy"])
                m_no = model.clone()
                adapt_classifier(m_no, sb, method="full_method", seed=seed, steps=60, detector=None)
                no_det_probs = m_no.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                no_det_acc = float(evaluate(no_det_probs[sb.test_idx], sb.y_np[sb.test_idx],
                                            sb.num_classes)["accuracy"])
                for name, dtol, ptol in settings:
                    delta_tol = 2.0 * delta_self if name == "auto" else dtol
                    m = model.clone()
                    det = DetectorState(delta_tolerance=delta_tol, phi_tolerance=ptol)
                    adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=det)
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    acc = float(evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx],
                                         sb.num_classes)["accuracy"])
                    records.append({
                        "seed": seed, "regime": regime, "shift": shift, "intensity": intensity,
                        "setting": name, "delta_tolerance": float(delta_tol), "phi_tolerance": ptol,
                        "delta_self": delta_self, "triggered": bool(det.triggered),
                        "trigger_step": det.trigger_step, "source_acc": src_acc,
                        "no_detector_acc": no_det_acc, "deployed_acc": acc,
                    })
                print(f"[detector_operating] seed={seed} {regime} {shift}/{intensity} done")
    _write(EXT / "detector_operating.json", records)
    _write_csv(EXT / "detector_operating.csv", records)
    return records


# --------------------------------------------------------------------------- D
def ece_bin_sensitivity(seeds=(0, 1, 2, 3, 4)):
    """ECE as a function of bin count with standard and debiased estimators (Q3).

    Evaluates source-only and full-method across representative conditions
    spanning the envelope: clean, feature_noise 0.45, edge_drop 0.35,
    edge_add 0.35, homophily_shift 0.25, homophily_shift 0.50.
    """
    SUPP.mkdir(parents=True, exist_ok=True)
    conditions = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
                  ("edge_add", 0.35), ("homophily_shift", 0.25), ("homophily_shift", 0.50)]
    bin_counts = [5, 10, 15, 20, 25]
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        predict_fn = _make_predict_fn(model)
        for shift, intensity in conditions:
            sb = shift_bundle(base, seed, shift, intensity)
            for method in ["source_only", "full_method"]:
                if method == "source_only":
                    probs = predict_fn(sb.x_np, sb.adj)
                else:
                    m = model.clone()
                    adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=None)
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                p_test = probs[sb.test_idx]
                y_test = sb.y_np[sb.test_idx]
                for n_bins in bin_counts:
                    for debiased in [False, True]:
                        ece_val = _compute_ece(p_test, y_test, n_bins=n_bins, debiased=debiased)
                        acc_val = float(np.mean(np.argmax(p_test, axis=1) == y_test))
                        records.append({
                            "seed": seed, "shift": shift, "intensity": intensity,
                            "method": method, "n_bins": n_bins, "debiased": debiased,
                            "ece": ece_val, "accuracy": acc_val,
                        })
            print(f"[ece_bin_sensitivity] seed={seed} {shift}/{intensity} done")
    _write(SUPP / "ece_bin_sensitivity.json", records)
    _write_csv(SUPP / "ece_bin_sensitivity.csv", records)
    return records


def _compute_ece(probs, labels, n_bins=15, debiased=False):
    """ECE with configurable bin count and optional debiasing.

    Standard ECE partitions predictions into ``n_bins`` equal-width confidence
    intervals and computes the weighted absolute difference between accuracy and
    confidence per bin.  The debiased variant subtracts an expected-sampling-noise
    floor following the formulation in Nixon et al. (2019) / Roelofs et al. (2020).
    """
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        n_in = int(np.sum(in_bin))
        if n_in == 0:
            continue
        avg_conf = float(np.mean(conf[in_bin]))
        avg_acc = float(np.mean(correct[in_bin]))
        ece += (n_in / len(labels)) * abs(avg_acc - avg_conf)
    if debiased:
        ece = max(ece - 1.0 / (len(labels) * n_bins), 0.0)
    return float(ece)


# --------------------------------------------------------------------------- E
def degree_fairness(seeds=(0, 1, 2, 3, 4)):
    """Per-degree-decile accuracy, ECE, and coverage with vs without degree prior (Q4).

    Partitions test nodes into 10 degree deciles and evaluates three variants:
    source-only, full method (5 signals), and full method minus the degree prior.
    Reports accuracy, ECE, node count, mean reliability, and selected fraction per
    decile, averaged over five representative conditions.
    """
    SUPP.mkdir(parents=True, exist_ok=True)
    conditions = [("clean", 0.0), ("feature_noise", 0.45), ("edge_drop", 0.35),
                  ("edge_add", 0.35), ("homophily_shift", 0.25)]
    methods = ["source_only", "full_method", "no_degree_prior"]
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        predict_fn = _make_predict_fn(model)
        for shift, intensity in conditions:
            sb = shift_bundle(base, seed, shift, intensity)
            test_idx = sb.test_idx
            test_y = sb.y_np[test_idx]
            deg_all = _br.degree_vector(sb.adj)
            test_deg = deg_all[test_idx]
            decile_edges = np.percentile(test_deg, np.linspace(0, 100, 11))
            # decile 0: [p0, p10], ..., decile 9: (p90, p100]
            for decile in range(10):
                lo = decile_edges[decile]
                hi = decile_edges[decile + 1]
                if decile == 0:
                    mask = (test_deg >= lo) & (test_deg <= hi)
                else:
                    mask = (test_deg > lo) & (test_deg <= hi)
                n_decile = int(np.sum(mask))
                if n_decile == 0:
                    continue
                for method in methods:
                    if method == "source_only":
                        probs = predict_fn(sb.x_np, sb.adj)
                        rel_info = {"mean_reliability": 1.0, "selected_fraction": 1.0}
                    elif method == "full_method":
                        m = model.clone()
                        rel_info = adapt_classifier(m, sb, method="full_method", seed=seed,
                                                    steps=60, detector=None)
                        probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    else:  # no_degree_prior
                        m = model.clone()
                        rel_info = adapt_classifier(m, sb, method="no_degree_prior", seed=seed,
                                                    steps=60, detector=None)
                        probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    p_dec = probs[test_idx][mask]
                    y_dec = test_y[mask]
                    met = evaluate(p_dec, y_dec, sb.num_classes)
                    records.append({
                        "seed": seed, "shift": shift, "intensity": intensity,
                        "decile": decile, "deg_lo": round(float(lo), 1),
                        "deg_hi": round(float(hi), 1), "n_decile": n_decile,
                        "method": method, "accuracy": met["accuracy"],
                        "ece": met["ece"], "mean_reliability": rel_info["mean_reliability"],
                        "selected_fraction": rel_info["selected_fraction"],
                    })
            print(f"[degree_fairness] seed={seed} {shift}/{intensity} done")
    _write(SUPP / "degree_fairness.json", records)
    _write_csv(SUPP / "degree_fairness.csv", records)
    return records


# --------------------------------------------------------------------------- F
def proxy_failure(seeds=(0, 1, 2, 3, 4)):
    r"""Detector-proxy fidelity: identify conditions where accuracy degrades but
    \Delta_t / \Phi_t stay below their operator-chosen thresholds (Q2).

    Runs adaptation WITHOUT a detector (full method, up to 60 steps) and records
    the full per-step (\Delta_t, \Phi_t) trace.  After adaptation, classifies each
    (condition, seed) into one of:

      (a) proxy fires  (Delta_t > Delta* OR Phi_t > Phi*) at the step of max degradation
      (b) proxy blind  (neither fires, yet accuracy dropped > 1% vs source)
      (c) clean halt   (neither fires, accuracy within 1%)
      (d) false alarm  (proxy fires but accuracy within 1%)

    Thresholds tested: auto (2*delta_self), balanced (0.05, 0.20), safety (0.02, 0.10).
    """
    SUPP.mkdir(parents=True, exist_ok=True)
    # All 10 controlled conditions
    conditions = IN_ENVELOPE + [("homophily_shift", 0.25), ("homophily_shift", 0.50)]
    # Threshold scenarios: (name, delta_tol, phi_tol); auto filled per-seed
    scenarios = [("auto", None, 0.20), ("balanced", 0.05, 0.20), ("safety", 0.02, 0.10)]
    records = []
    for seed in seeds:
        model, base = _train_per_seed(seed)
        predict_fn = _make_predict_fn(model)
        delta_self = _self_drift(model, base, seed)
        for shift, intensity in conditions:
            sb = shift_bundle(base, seed, shift, intensity)
            src_probs = predict_fn(sb.x_np, sb.adj)
            src_acc = float(evaluate(src_probs[sb.test_idx], sb.y_np[sb.test_idx],
                                     sb.num_classes)["accuracy"])
            # adapt without detector, collect full traces
            m = model.clone()
            info = adapt_classifier(m, sb, method="full_method", seed=seed, steps=60, detector=None)
            probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
            final_acc = float(evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx],
                                       sb.num_classes)["accuracy"])
            acc_delta = final_acc - src_acc
            acc_degraded = acc_delta < -0.01  # >1% absolute drop
            d_trace = info.get("delta_trace") or []
            p_trace = info.get("phi_trace") or []
            max_delta = float(max(d_trace)) if d_trace else 0.0
            max_phi = float(max(p_trace)) if p_trace else 0.0
            # find step of max accuracy degradation (proxy: min acc along trace
            # is not tracked per-step, so use final; but we can ask: at which
            # step did delta_t/phi_t peak?)
            peak_step_delta = int(np.argmax(d_trace)) if d_trace else -1
            peak_step_phi = int(np.argmax(p_trace)) if p_trace else -1
            for name, dtol, ptol in scenarios:
                delta_tol = 2.0 * delta_self if name == "auto" else dtol
                would_halt = (max_delta > delta_tol) or (max_phi > ptol)
                outcome = "false_alarm" if would_halt and not acc_degraded else \
                          "clean_no_halt" if not would_halt and not acc_degraded else \
                          "proxy_blind" if not would_halt and acc_degraded else \
                          "proxy_fires"
                records.append({
                    "seed": seed, "shift": shift, "intensity": intensity,
                    "scenario": name, "delta_tolerance": float(delta_tol),
                    "phi_tolerance": ptol, "src_acc": src_acc, "final_acc": final_acc,
                    "acc_delta": round(acc_delta, 5), "max_delta_t": round(max_delta, 6),
                    "max_phi_t": round(max_phi, 5), "would_halt": would_halt,
                    "outcome": outcome, "delta_self": round(delta_self, 6),
                    "peak_step_delta": peak_step_delta, "peak_step_phi": peak_step_phi,
                    "steps": info["steps"],
                })
            print(f"[proxy_failure] seed={seed} {shift}/{intensity} acc_delta={acc_delta:+.4f} "
                  f"max_delta={max_delta:.6f} max_phi={max_phi:.4f} done")
    _write(SUPP / "proxy_failure.json", records)
    _write_csv(SUPP / "proxy_failure.csv", records)
    return records


# --------------------------------------------------------------------------- render
def _read(path):
    path = Path(path)
    if not path.exists():
        print(f"[render] missing {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _ms(vals, digits=4):
    vals = list(vals)
    if len(vals) > 1:
        return f"{mean(vals):.{digits}f}\\pm{stdev(vals):.{digits}f}"
    return f"{mean(vals):.{digits}f}" if vals else "n/a"


def render():
    # ---- A: signal ablation (three homophily regimes)
    recs = _read(SUPP / "signal_ablation.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:signal-ablation  (per-signal leave-one-out across homophily regimes)")
        print("=" * 90)
        label = {"full_method": "Full method (all five signals)",
                 "no_confidence": "$\\;-$ confidence $c_i$",
                 "no_neighborhood_agreement": "$\\;-$ neighborhood agreement $a_i$",
                 "no_structural_stability": "$\\;-$ perturbation stability $s_i$",
                 "no_source_consistency": "$\\;-$ source consistency $b_i$",
                 "no_degree_prior": "$\\;-$ degree prior $d_i$"}
        rlabel = {"in_envelope": "In-envelope ($\\hat h_G\\!\\ge\\!0.5$, 8 shifts)",
                  "boundary": "Boundary (homophily shift $0.25$)",
                  "out_envelope": "Out-of-envelope (homophily shift $0.50$)"}
        for regime in ["in_envelope", "boundary", "out_envelope"]:
            rr = [r for r in recs if r["regime"] == regime]
            if not rr:
                continue
            print(f"\\multicolumn{{4}}{{l}}{{\\emph{{{rlabel[regime]}}}}} \\\\")
            for m, lab in label.items():
                sub = [r for r in rr if r["method"] == m]
                if not sub:
                    continue
                print(f"{lab} & {mean(r['accuracy'] for r in sub):.4f} & "
                      f"{mean(r['ece'] for r in sub):.4f} & "
                      f"${mean(r['cov_r_e'] for r in sub):+.4f}$ \\\\")
            print("\\cmidrule(lr){1-4}")
        # score-level coupling deltas vs full (text)
        for regime in ["in_envelope", "boundary", "out_envelope"]:
            rr = [r for r in recs if r["regime"] == regime]
            full = [r for r in rr if r["method"] == "full_method"]
            if not full:
                continue
            base_cov = mean(r["cov_r_e"] for r in full)
            print(f"% [{regime}] full cov(r,e)={base_cov:+.4f}; weakening (cov->0) when a signal is dropped:")
            for m, lab in label.items():
                if m == "full_method":
                    continue
                sub = [r for r in rr if r["method"] == m]
                dcov = mean(r["cov_r_e"] for r in sub) - base_cov
                dacc = mean(r["accuracy"] for r in sub) - mean(r["accuracy"] for r in full)
                print(f"%   {m}: dcov={dcov:+.5f} dACC={dacc:+.4f}")

    # ---- B: homophily robustness
    recs = _read(SUPP / "homophily_robustness.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:homophily-robustness  (full method; \\hat h_G + delta)")
        print("=" * 90)
        by_delta = defaultdict(list)
        for r in recs:
            by_delta[r["delta"]].append(r)
        for delta in sorted(by_delta):
            sub = by_delta[delta]
            print(f"${delta:+.1f}$ & ${_ms([r['used_homophily'] for r in sub], 3)}$ & "
                  f"${_ms([r['accuracy'] for r in sub])}$ & ${_ms([r['ece'] for r in sub])}$ \\\\")
        base = by_delta[0.0]
        base_acc, base_ece = mean(r["accuracy"] for r in base), mean(r["ece"] for r in base)
        accs = [mean(r["accuracy"] for r in by_delta[d]) for d in by_delta]
        eces = [mean(r["ece"] for r in by_delta[d]) for d in by_delta]
        print(f"% measured_h band: {sorted(set(round(r['measured_homophily'],3) for r in recs))}")
        print(f"% delta=0 acc={base_acc:.4f} ece={base_ece:.4f}; "
              f"max acc swing across +-0.3 = {max(accs)-min(accs):.4f}; "
              f"max ece swing = {max(eces)-min(eces):.4f}")

    # ---- C: detector operating characteristic
    recs = _read(EXT / "detector_operating.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:detector-operating  (false-halt / false-continue + deployed acc)")
        print("=" * 90)
        slabel = {"safety": "Safety-critical ($0.02,0.10$)", "balanced": "Balanced ($0.05,0.20$)",
                  "aggressive": "Aggressive ($0.20,0.50$)", "auto": "Auto ($2\\Delta_{\\mathrm{self}},0.20$)"}
        for name, lab in slabel.items():
            ie = [r for r in recs if r["setting"] == name and r["regime"] == "in_envelope"]
            oe = [r for r in recs if r["setting"] == name and r["regime"] == "out_envelope"]
            false_halt = 100 * mean(1.0 if r["triggered"] else 0.0 for r in ie)
            detect = 100 * mean(1.0 if r["triggered"] else 0.0 for r in oe)
            false_cont = 100.0 - detect
            ie_acc = mean(r["deployed_acc"] for r in ie)
            oe_acc = mean(r["deployed_acc"] for r in oe)
            print(f"{lab} & {false_halt:.1f} & {false_cont:.1f} & {ie_acc:.4f} & {oe_acc:.4f} \\\\")
        # reference accuracies
        oe_all = [r for r in recs if r["regime"] == "out_envelope"]
        ie_all = [r for r in recs if r["regime"] == "in_envelope"]
        print(f"% out-env source-only acc={mean(r['source_acc'] for r in oe_all):.4f} "
              f"no-detector acc={mean(r['no_detector_acc'] for r in oe_all):.4f}")
        print(f"% in-env source-only acc={mean(r['source_acc'] for r in ie_all):.4f} "
              f"no-detector acc={mean(r['no_detector_acc'] for r in ie_all):.4f}")

    # ---- D: ECE bin sensitivity
    recs = _read(SUPP / "ece_bin_sensitivity.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:ece-bin-sensitivity  (ECE vs bin count, standard + debiased)")
        print("=" * 90)
        # aggregate over seeds, one row per (shift, intensity, method)
        from collections import defaultdict as dd
        groups = dd(list)
        for r in recs:
            key = (r["shift"], r["intensity"], r["method"])
            groups[key].append(r)
        for (shift, intensity, method), sub in sorted(groups.items()):
            for deb in [False, True]:
                deb_label = "debiased" if deb else "standard"
                vals = []
                for nb in [5, 10, 15, 20, 25]:
                    eces = [r["ece"] for r in sub if r["n_bins"] == nb and r["debiased"] == deb]
                    vals.append(_ms(eces) if eces else "---")
                print(f"{shift} ({intensity}) & {method} & {deb_label} & "
                      f"{' & '.join(vals)} \\\\")
        # summary: max swing across bins per condition
        print("% Summary: max ECE swing across bin counts (standard estimator):")
        for (shift, intensity, method), sub in sorted(groups.items()):
            std_eces = {nb: mean(r["ece"] for r in sub if r["n_bins"] == nb and not r["debiased"])
                        for nb in [5, 10, 15, 20, 25]}
            if std_eces:
                vals = list(std_eces.values())
                swing = max(vals) - min(vals) if len(vals) > 1 else 0.0
                print(f"%   {shift}/{intensity} {method}: ECE swing = {swing:.4f} "
                      f"(ECE range [{min(vals):.4f}, {max(vals):.4f}])")

    # ---- E: degree fairness
    recs = _read(SUPP / "degree_fairness.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:degree-fairness  (per-decile accuracy and ECE, aggregated)")
        print("=" * 90)
        from collections import defaultdict as dd
        # aggregate over seeds and conditions, one row per (decile, method)
        groups = dd(list)
        for r in recs:
            groups[(r["decile"], r["method"])].append(r)
        for decile in range(10):
            dg = [(k, v) for k, v in groups.items() if k[0] == decile]
            for (dec, method), sub in sorted(dg, key=lambda x: x[0][1]):
                acc = mean(r["accuracy"] for r in sub)
                ece = mean(r["ece"] for r in sub)
                n_nodes = mean(r["n_decile"] for r in sub)
                mrel = mean(r["mean_reliability"] for r in sub) if method != "source_only" else 1.0
                selfrac = mean(r["selected_fraction"] for r in sub) if method != "source_only" else 1.0
                deg_range = f"({sub[0]['deg_lo']:.0f}--{sub[0]['deg_hi']:.0f})"
                print(f"D{decile} & {deg_range} & {n_nodes:.1f} & {method} & "
                      f"{acc:.4f} & {ece:.4f} & {mrel:.3f} & {selfrac:.3f} \\\\")
        # summary: per-decile fairness gap (max - min accuracy across deciles)
        for method in ["source_only", "full_method", "no_degree_prior"]:
            dec_accs = {d: mean(r["accuracy"] for r in groups.get((d, method), []))
                       for d in range(10) if (d, method) in groups}
            if dec_accs:
                vals = list(dec_accs.values())
                gini = max(vals) - min(vals)
                print(f"% {method}: max-min accuracy gap across deciles = {gini:.4f}")

    # ---- F: proxy failure
    recs = _read(SUPP / "proxy_failure.json")
    if recs:
        print("\n" + "=" * 90)
        print("### tab:proxy-failure  (detector proxy fidelity)")
        print("=" * 90)
        from collections import defaultdict as dd
        # aggregate by (shift, intensity, scenario)
        groups = dd(list)
        for r in recs:
            groups[(r["shift"], r["intensity"], r["scenario"])].append(r)
        # summary table: one line per (shift, intensity) x scenario
        for (shift, intensity, scenario), sub in sorted(groups.items()):
            src_acc = mean(r["src_acc"] for r in sub)
            final_acc = mean(r["final_acc"] for r in sub)
            acc_delta = mean(r["acc_delta"] for r in sub)
            degraded = sum(1 for r in sub if r["acc_delta"] < -0.01)
            would_halt = sum(1 for r in sub if r["would_halt"])
            blind = sum(1 for r in sub if r["outcome"] == "proxy_blind")
            false_alarm = sum(1 for r in sub if r["outcome"] == "false_alarm")
            max_d = mean(r["max_delta_t"] for r in sub)
            max_p = mean(r["max_phi_t"] for r in sub)
            n = len(sub)
            print(f"{shift} ({intensity}) & {scenario} & {src_acc:.4f} & {final_acc:.4f} & "
                  f"${acc_delta:+.4f}$ & {degraded}/{n} & {would_halt}/{n} & "
                  f"{blind}/{n} & {false_alarm}/{n} & ${max_d:.5f}$ & ${max_p:.3f}$ \\\\")
        # proxy blind spot highlight
        print("% Proxy blind-spot summary (cases where acc degraded >1% but proxy would NOT halt):")
        any_blind = False
        for (shift, intensity, scenario), sub in sorted(groups.items()):
            blind_count = sum(1 for r in sub if r["outcome"] == "proxy_blind")
            if blind_count > 0:
                any_blind = True
                print(f"%   BLIND: {shift}/{intensity} [{scenario}]: {blind_count}/{len(sub)} seeds")
        if not any_blind:
            print("%   No proxy blind spots detected across all conditions and thresholds.")


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
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        render()
        return
    experiments = args.experiments.split(",")
    if "all" in experiments:
        experiments = ["signal_ablation", "homophily_robustness", "detector_operating",
                       "ece_bin_sensitivity", "degree_fairness", "proxy_failure"]
    started = time.perf_counter()
    if "signal_ablation" in experiments:
        signal_ablation()
    if "homophily_robustness" in experiments:
        homophily_robustness()
    if "detector_operating" in experiments:
        detector_operating()
    if "ece_bin_sensitivity" in experiments:
        ece_bin_sensitivity()
    if "degree_fairness" in experiments:
        degree_fairness()
    if "proxy_failure" in experiments:
        proxy_failure()
    print(f"reviewer experiments completed in {time.perf_counter() - started:.1f}s")
    render()


if __name__ == "__main__":
    main()
