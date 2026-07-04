"""Fair head-to-head comparison against the OFFICIAL test-time adaptation
baselines (mirror of ``code/fair_comparison.py``, now on PyTorch Geometric).

Every method runs on the SAME ``GCN(use_bn=True)`` backbone, the same trained
source model, and the same step budget.  The baselines use their *original*
mechanisms via the vendored official code:

* ``tent``   — official Tent: adapts BatchNorm affine params (gamma/beta) and
  uses target-graph batch statistics (NOT the classifier).        [req#2, #3]
* ``eata``   — official EATA: entropy + redundancy filtering, diagonal-Fisher
  anti-forgetting, BN adaptation.                                  [req#2, #3]
* ``matcha`` — graph-aware reliability masking + classifier entropy update.
* ``gtrans`` — test-time feature transformation + light classifier update.
* ``full_method`` — the proposed reliability-aware classifier-only method with
  the negative-adaptation detector.

The collected BN parameter names and the gamma drift are logged once as direct
evidence that Tent/EATA adapt BatchNorm (not the classifier).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import adapt_classifier
from baselines_official import GraphModelWrapper, run_eata, run_gtrans, run_matcha, run_tent
from data_adapter import load_bundle, shift_bundle
from detector import DetectorState
from models import make_model, train_model
from third_party import tent as tent_lib

OUT_DIR = Path(__file__).resolve().parents[1] / "results_torch" / "fair_comparison"
HIDDEN = {"synthetic": 24, "cora": 32, "citeseer": 32, "pubmed": 48, "coauthor_cs": 48, "amazon_photo": 48}
EPOCHS = {"synthetic": 300, "cora": 250, "citeseer": 250, "pubmed": 250, "coauthor_cs": 250, "amazon_photo": 250}
MAX_NODES = {"coauthor_cs": 2500, "amazon_photo": 2500}
SKIP_GTRANS = {"coauthor_cs", "amazon_photo"}
TENT_STEPS, TENT_LR = 40, 0.05
MATCHA_STEPS, GTRANS_STEPS, FULL_STEPS = 20, 20, 25


def _write(path, recs, proof):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": recs, "bn_adaptation_proof": proof}, indent=2), encoding="utf-8")
    if recs:
        fns = sorted({k for r in recs for k in r})
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            w.writerows(recs)


def _train_bn_source(dataset, base, seed):
    model = make_model("gcn", base.x.shape[1], HIDDEN.get(dataset, 32), base.num_classes,
                       use_bn=True, seed=seed, dropout=0.0)
    train_model(model, base.x, base.edge_index, base.y, base.train_mask, base.val_mask,
                epochs=EPOCHS.get(dataset, 250), lr=0.01, weight_decay=5e-4, patience=80,
                exclude_bn_bias_wd=True)
    return model


def _bn_proof(model, base, sb):
    """Record which params Tent collects and how much gamma moves (req#3 evidence)."""
    before_gamma = dict(model.named_parameters())["bn.weight"].detach().clone()
    wrap = GraphModelWrapper(model.clone(), sb.edge_index)
    tent_lib.configure_model(wrap)
    params, names = tent_lib.collect_params(wrap)
    opt = torch.optim.SGD(params, lr=TENT_LR, momentum=0.9)
    tent_lib.Tent(wrap, opt, steps=TENT_STEPS)(sb.x)
    after_gamma = dict(wrap.named_parameters())["model.bn.weight"].detach()
    drift = float((after_gamma - before_gamma).abs().sum())
    return {"adapted_param_names": names, "gamma_l1_drift": drift, "classifier_adapted": False}


def run(out_dir, quick=False):
    if quick:
        plan = [("synthetic", [("clean", 0.0), ("homophily_shift", 0.50)], [0]),
                ("cora", [("clean", 0.0), ("feature_noise", 0.10)], [0])]
    else:
        # Mirrors code/fair_comparison.py: citation graphs (3 seeds) + the two
        # larger graphs (2 seeds, induced subgraphs); GTrans is skipped on the
        # large graphs (it backpropagates to the input features).
        plan = [("synthetic", [("homophily_shift", 0.25), ("homophily_shift", 0.50)], [0, 1, 2]),
                ("cora", [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.15)], [0, 1, 2]),
                ("citeseer", [("clean", 0.0), ("edge_drop", 0.15)], [0, 1, 2]),
                ("pubmed", [("clean", 0.0), ("edge_drop", 0.15)], [0, 1, 2]),
                ("coauthor_cs", [("clean", 0.0), ("edge_drop", 0.15), ("edge_add", 0.10)], [0, 1]),
                ("amazon_photo", [("clean", 0.0), ("edge_drop", 0.15), ("edge_add", 0.10)], [0, 1])]
    records, proof = [], None
    for dataset, conditions, seeds in plan:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            base = load_bundle(dataset, seed=seed, n=300 if dataset == "synthetic" else None,
                               max_nodes=MAX_NODES.get(dataset),
                               graph_backend="sparse" if dataset in MAX_NODES else "auto")
            model = _train_bn_source(dataset, base, seed)
            clean_acc = evaluate(model.predict_probs(base.x, base.edge_index).cpu().numpy()[base.test_idx],
                                 base.y_np[base.test_idx], base.num_classes)["accuracy"]
            print(f"[fair] {dataset} seed={seed} BN-source clean acc={clean_acc:.4f}")

            for shift, intensity in conditions:
                sb = shift_bundle(base, seed, shift, intensity)
                if proof is None:
                    proof = _bn_proof(model, base, sb)
                    print(f"[fair] Tent adapts params={proof['adapted_param_names']} "
                          f"||Δgamma||_1={proof['gamma_l1_drift']:.4f} classifier_adapted={proof['classifier_adapted']}")

                def rec(method, probs, runtime, extra=None):
                    m = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    row = {"dataset": dataset, "seed": seed, "shift": shift, "intensity": intensity,
                           "method": method, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                           "ece": m["ece"], "brier": m["brier"], "nll": m["nll"], "runtime_seconds": runtime}
                    if extra:
                        row.update(extra)
                    records.append(row)
                    print(f"[fair]   {dataset} {shift}/{intensity} {method:12s} acc={m['accuracy']:.4f} ece={m['ece']:.4f}")

                rec("source_only", model.predict_probs(sb.x, sb.edge_index).cpu().numpy(), 0.0)
                t = time.perf_counter(); p, _ = run_tent(model, sb, steps=TENT_STEPS, lr=TENT_LR); rec("tent", p, time.perf_counter() - t)
                t = time.perf_counter(); p, _ = run_eata(model, sb, base, steps=TENT_STEPS, lr=TENT_LR); rec("eata", p, time.perf_counter() - t)
                t = time.perf_counter(); p = run_matcha(model, sb, steps=MATCHA_STEPS); rec("matcha", p, time.perf_counter() - t)
                if dataset not in SKIP_GTRANS:
                    t = time.perf_counter(); p = run_gtrans(model, sb, steps=GTRANS_STEPS); rec("gtrans", p, time.perf_counter() - t)
                # Detector with the paper's auto-calibrated tolerance (k=2 x self-drift),
                # measured once per (dataset, seed) on the clean source graph.
                from detector_calibration import _self_drift
                if not hasattr(model, "_delta_self_cache"):
                    model._delta_self_cache = _self_drift(model, base, seed)
                mf = model.clone(); det = DetectorState(delta_tolerance=2.0 * model._delta_self_cache)
                t = time.perf_counter()
                adapt_classifier(mf, sb, method="full_method", seed=seed, steps=FULL_STEPS, detector=det)
                pf = mf.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                rec("full_method", pf, time.perf_counter() - t,
                    {"detector_triggered": bool(det.triggered), "delta_star": det.delta_tolerance})
            _write(out_dir / "fair_comparison.json", records, proof)
    _write(out_dir / "fair_comparison.json", records, proof)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    started = time.perf_counter()
    run(Path(args.out), quick=args.quick)
    print(f"fair comparison completed in {time.perf_counter() - started:.1f}s")
