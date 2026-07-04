"""Experiment 1: fair head-to-head comparison against full original baselines.

Every method runs on the SAME GCNWithBN backbone, the same trained source model,
the same test-time learning rate, and the same step budget.  The baselines use
their *original* mechanisms (Tent/EATA adapt batch-norm parameters; Matcha uses
graph-aware masking; GTrans transforms the graph), in contrast to the
classifier-only analogues used for the controlled ablation.

Two regimes are reported:
  * in-envelope: Planetoid (Cora/Citeseer/Pubmed) + large graphs (Coauthor CS,
    Amazon Photo), high homophily -> expect the full method to match accuracy and
    achieve the lowest ECE.
  * out-of-envelope: synthetic homophily shift 0.50 -> expect every baseline to
    negative-adapt and the detector to recover the source baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from adaptation import adapt_classifier
from data import apply_shift, load_public_graph_dataset, make_contextual_sbm, split_indices
from detector import DetectorState
from full_baselines import eata_full, gtrans_full, matcha_full, tent_full
from models_bn import GCNWithBN
from utils import degree_vector, evaluate

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "fair_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": recs}, indent=2), encoding="utf-8")
    if recs:
        fns = sorted({k for r in recs for k in r})
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            w.writerows(recs)


def _degree_only_fallback(seed, x, y, intensity):
    adj = None
    return None


def _load(dataset, seed, max_nodes):
    if dataset == "synthetic":
        x, adj, y = make_contextual_sbm(seed=seed, n=300)
        tr, va, te = split_indices(seed, y)
        return x, adj, y, tr, va, te
    return load_public_graph_dataset(dataset, max_nodes=max_nodes, seed=seed, graph_backend="sparse")


def _apply(seed, x, adj, y, shift, intensity):
    x_t, adj_t = apply_shift(seed, x, adj, y, shift, intensity)
    if len(x_t) != len(y):  # degree_shift removes nodes -> topology-only fallback
        x_t, adj_t = x.copy(), adj.copy()
    return x_t, adj_t


def run(out_dir, seeds_small=(0, 1, 2), seeds_large=(0, 1)):
    plan = [
        ("synthetic", [("homophily_shift", 0.25), ("homophily_shift", 0.50)], seeds_small),
        ("cora", [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.15)], seeds_small),
        ("citeseer", [("clean", 0.0), ("edge_drop", 0.15)], seeds_small),
        ("pubmed", [("clean", 0.0), ("edge_drop", 0.15)], seeds_small),
        ("coauthor_cs", [("clean", 0.0), ("edge_drop", 0.15)], seeds_large),
        ("amazon_photo", [("clean", 0.0), ("edge_drop", 0.15)], seeds_large),
    ]
    # Resume support: load any existing records and skip completed (ds,seed,shift,method).
    existing_path = out_dir / "fair_comparison.json"
    records = []
    done = set()
    if existing_path.exists():
        try:
            records = json.loads(existing_path.read_text())["records"]
            for r in records:
                done.add((r["dataset"], r["seed"], r["shift"], float(r["intensity"]), r["method"]))
            print(f"[fair] resuming with {len(records)} existing records")
        except Exception:
            records = []
    # GTrans backprops to the input features and is prohibitively slow on the
    # high-dimensional large graphs; it is reported on the citation graphs.
    skip_gtrans = {"coauthor_cs", "amazon_photo"}
    hidden = {"synthetic": 24, "cora": 32, "citeseer": 32, "pubmed": 48, "coauthor_cs": 48, "amazon_photo": 48}
    epochs = {"synthetic": 300, "cora": 200, "citeseer": 200, "pubmed": 200, "coauthor_cs": 250, "amazon_photo": 250}
    max_nodes = {"coauthor_cs": 2500, "amazon_photo": 2500}
    steps = 40
    lr = 0.05

    for dataset, conditions, seeds in plan:
        for seed in seeds:
            first_shift, first_int = conditions[0]
            if (dataset, seed, first_shift, float(first_int), "source_only") in done:
                print(f"[fair] skip {dataset} seed={seed} (already done)")
                continue
            try:
                x, adj, y, tr, va, te = _load(dataset, seed, max_nodes.get(dataset))
            except Exception as e:
                print(f"[fair] {dataset} seed={seed} LOAD FAIL: {e}")
                continue
            classes = int(np.max(y)) + 1
            model = GCNWithBN(x.shape[1], hidden[dataset], classes, seed=seed)
            model.train(x, adj, y, tr, va, epochs=epochs[dataset])
            fisher = model.fisher_bn(x, adj, y, tr)
            clean_p, _ = model.forward(x, adj, use_running=True)
            print(f"[fair] {dataset} seed={seed} clean acc={float(np.mean(np.argmax(clean_p[te],axis=1)==y[te])):.4f}")
            for shift, intensity in conditions:
                x_t, adj_t = _apply(seed, x, adj, y, shift, intensity)

                def record(method_name, probs, runtime, extra=None):
                    m = evaluate(probs[te], y[te], classes)
                    rec = {
                        "dataset": dataset, "seed": seed, "shift": shift, "intensity": intensity,
                        "method": method_name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                        "ece": m["ece"], "brier": m["brier"], "nll": m["nll"], "runtime_seconds": runtime,
                    }
                    if extra:
                        rec.update(extra)
                    records.append(rec)
                    print(f"[fair]   {dataset} {shift}/{intensity} {method_name}: acc={m['accuracy']:.4f} ece={m['ece']:.4f}")

                # source only
                p0, _ = model.forward(x_t, adj_t, use_running=True)
                record("source_only", p0, 0.0)

                # Tent (full): BN affine + target stats
                mt = model.clone()
                t0 = time.perf_counter(); p, _ = tent_full(mt, x_t, adj_t, steps=steps, lr=lr); record("tent_full", p, time.perf_counter() - t0)

                # EATA (full): filter + Fisher + BN
                me = model.clone()
                t0 = time.perf_counter(); p, _ = eata_full(me, x_t, adj_t, fisher, steps=steps, lr=lr); record("eata_full", p, time.perf_counter() - t0)

                # Matcha (full): graph-aware mask + classifier
                mm = model.clone()
                t0 = time.perf_counter(); p, _ = matcha_full(mm, x_t, adj_t, steps=steps, lr=lr); record("matcha_full", p, time.perf_counter() - t0)

                # GTrans (full): feature transform + classifier (skipped on large graphs)
                if dataset not in skip_gtrans:
                    mg = model.clone()
                    t0 = time.perf_counter(); p, _ = gtrans_full(mg, x_t, adj_t, steps=steps, lr_feat=lr, lr_clf=0.02); record("gtrans_full", p, time.perf_counter() - t0)

                # Full method (ours) + detector
                mf = model.clone()
                det = DetectorState()
                t0 = time.perf_counter()
                adapt_classifier(mf, x_t, adj_t, method="full_method", seed=seed, steps=steps, detector=det)
                pf, _ = mf.forward(x_t, adj_t, use_running=True)
                record("full_method", pf, time.perf_counter() - t0, {"detector_triggered": bool(det.triggered)})
            _write(out_dir / "fair_comparison.json", records)
    _write(out_dir / "fair_comparison.json", records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    started = time.perf_counter()
    run(Path(args.out))
    print(f"fair comparison completed in {time.perf_counter()-started:.1f}s")
