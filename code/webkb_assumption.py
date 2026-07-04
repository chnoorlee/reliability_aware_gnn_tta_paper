"""Experiment 3: verify Assumption 1 and per-signal behavior on REAL low-homophily graphs.

For each real WebKB graph (Texas, Cornell, Wisconsin):
  * train a GCN, compute reliability scores r_i and the five component signals,
  * compute the pseudo-label error e_i = 1[argmax p_i != y_i],
  * report Cov(r_i, e_i), the top-50%/bottom-50% reliability error gap, and the
    correlation of EACH of the five signals with the error,
  * compare the pseudo-label homophily estimate and a label-free feature-homophily
    estimate against the true homophily.

The goal is to show that (i) Assumption 1 (Cov(r,e) <= 0) holds even at homophily
~0.06-0.18, and (ii) the neighborhood-agreement signal decorrelates from the error
while confidence and source-consistency remain informative -- explaining why the
multi-signal aggregator preserves rank-calibration on heterophilous graphs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from adaptation import reliability_scores
from data import apply_shift
from models import TwoLayerGCN
from utils import upper_triangle_edges
from webkb_loader import graph_homophily, load_real_webkb

OUT = Path(__file__).resolve().parents[1] / "results" / "extended"
OUT.mkdir(parents=True, exist_ok=True)


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
    records = []
    for dataset in ("texas", "cornell", "wisconsin"):
        for seed in seeds:
            x, adj, y, tr, va, te = load_real_webkb(dataset, seed=seed)
            classes = int(np.max(y)) + 1
            model = TwoLayerGCN(x.shape[1], 24, classes, seed=seed)
            model.train(x, adj, y, tr, va, epochs=400)
            # Reference = clean-graph source prediction; signals are measured on a
            # mildly shifted target so that source-consistency (and the other
            # signals) carry variance, as they do during real adaptation.
            clean_probs, _ = model.forward(x, adj)
            x_t, adj_t = apply_shift(seed, x, adj, y, "edge_drop", 0.10)
            if len(x_t) != len(y):
                x_t, adj_t = x.copy(), adj.copy()
            probs, _ = model.forward(x_t, adj_t)
            r, parts = reliability_scores(model, x_t, adj_t, seed=seed, reference_probs=clean_probs)
            pred = np.argmax(probs, axis=1)
            e = (pred != y).astype(float)
            adj = adj_t  # report homophily on the evaluated (shifted) graph

            cov_re = float(np.cov(r, e, bias=True)[0, 1])
            r_top = r >= np.quantile(r, 0.5)
            err_top = float(np.mean(e[r_top])) if np.any(r_top) else 0.0
            err_bot = float(np.mean(e[~r_top])) if np.any(~r_top) else 0.0

            true_h = graph_homophily(adj, y)
            edges = upper_triangle_edges(adj)
            pseudo_h = float(np.mean(pred[edges[:, 0]] == pred[edges[:, 1]])) if len(edges) else 0.0
            feat_h = _feature_homophily(adj, x)

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
            print(f"[webkb-assume] {dataset} seed={seed} cov(r,e)={cov_re:+.5f} gap={err_bot-err_top:+.4f} "
                  f"trueH={true_h:.3f} pseudoH={pseudo_h:.3f}")

    OUT.joinpath("webkb_assumption.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")

    # aggregate
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)
    print("\n=== Assumption 1 on REAL WebKB (mean over 5 seeds) ===")
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
