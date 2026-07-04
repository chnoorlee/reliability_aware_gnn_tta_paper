"""Assumption 1 + per-signal behavior on REAL WebKB graphs
(PyG port of code/webkb_assumption.py).

For each real Texas/Cornell/Wisconsin graph: train a GCN, compute reliability
scores and the five component signals on a mildly shifted target, and report
Cov(r,e), the top/bottom-50% reliability error gap, per-signal error
correlations, and pseudo- vs feature- vs true-homophily estimates.

Writes results_torch/extended/webkb_assumption.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from _np_bridge import graph_homophily, upper_triangle_edges
from adaptation import _make_predict_fn
from exp_common import shift_bundle, train_source, webkb_bundle
from reliability import reliability_scores

OUT = Path(__file__).resolve().parents[1] / "results_torch" / "extended"


def _signal_corr(sig, err):
    if np.std(sig) < 1e-9 or np.std(err) < 1e-9:
        return 0.0
    return float(np.corrcoef(sig, err)[0, 1])


def _feature_homophily(adj, x):
    edges = upper_triangle_edges(adj)
    if len(edges) == 0:
        return 0.0
    xi = x[edges[:, 0]]
    xj = x[edges[:, 1]]
    num = np.sum(xi * xj, axis=1)
    den = np.linalg.norm(xi, axis=1) * np.linalg.norm(xj, axis=1) + 1e-12
    return float(np.mean(num / den))


def run(seeds=(0, 1, 2, 3, 4)):
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in ("texas", "cornell", "wisconsin"):
        for seed in seeds:
            base = webkb_bundle(dataset, seed)
            model, base, _ = train_source(dataset, seed, hidden=24, epochs=400, bundle=base)
            predict_fn = _make_predict_fn(model)
            clean_probs = predict_fn(base.x_np, base.adj)
            sb = shift_bundle(base, seed, "edge_drop", 0.10)
            probs = predict_fn(sb.x_np, sb.adj)
            r, parts = reliability_scores(predict_fn, sb.x_np, sb.adj, seed=seed, reference_probs=clean_probs)
            pred = np.argmax(probs, axis=1)
            e = (pred != sb.y_np).astype(float)
            adj_eval = sb.adj  # report homophily on the evaluated (shifted) graph

            cov_re = float(np.cov(r, e, bias=True)[0, 1])
            r_top = r >= np.quantile(r, 0.5)
            err_top = float(np.mean(e[r_top])) if np.any(r_top) else 0.0
            err_bot = float(np.mean(e[~r_top])) if np.any(~r_top) else 0.0

            true_h = graph_homophily(adj_eval, sb.y_np)
            edges = upper_triangle_edges(adj_eval)
            pseudo_h = float(np.mean(pred[edges[:, 0]] == pred[edges[:, 1]])) if len(edges) else 0.0
            feat_h = _feature_homophily(adj_eval, sb.x_np)

            records.append({
                "dataset": dataset, "seed": seed,
                "cov_r_e": cov_re,
                "error_gap": err_bot - err_top,
                "assumption_holds": bool(cov_re <= 0.0),
                "corr_confidence_err": _signal_corr(parts["confidence"], e),
                "corr_agreement_err": _signal_corr(parts["agreement"], e),
                "corr_stability_err": _signal_corr(parts["stability"], e),
                "corr_source_err": _signal_corr(parts["source"], e),
                "corr_degree_err": _signal_corr(parts["degree"], e),
                "true_homophily": true_h,
                "pseudo_homophily": pseudo_h,
                "feature_homophily": feat_h,
                "pseudo_h_abs_err": abs(pseudo_h - true_h),
                "feature_h_abs_err": abs(feat_h - true_h),
            })
            print(f"[webkb-assume] {dataset} seed={seed} cov(r,e)={cov_re:+.5f} "
                  f"gap={err_bot - err_top:+.4f} trueH={true_h:.3f} pseudoH={pseudo_h:.3f}")

    OUT.joinpath("webkb_assumption.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)
    print("\n=== Assumption 1 on REAL WebKB (mean over seeds) ===")
    print(f"{'dataset':<10}{'cov(r,e)':<12}{'err gap':<10}{'corr_conf':<11}{'corr_agree':<12}{'corr_src':<10}{'pseudoH_err':<12}{'holds':<6}")
    for ds, rows in by_ds.items():
        print(f"{ds:<10}"
              f"{mean(r['cov_r_e'] for r in rows):<+12.5f}"
              f"{mean(r['error_gap'] for r in rows):<+10.4f}"
              f"{mean(r['corr_confidence_err'] for r in rows):<+11.4f}"
              f"{mean(r['corr_agreement_err'] for r in rows):<+12.4f}"
              f"{mean(r['corr_source_err'] for r in rows):<+10.4f}"
              f"{mean(r['pseudo_h_abs_err'] for r in rows):<12.4f}"
              f"{str(all(r['assumption_holds'] for r in rows)):<6}")
    return records


if __name__ == "__main__":
    run()
