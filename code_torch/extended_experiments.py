"""Extended external-validity experiments (PyG port of ``code/extended_experiments.py``).

1. ``large_scale_study``   -- Amazon Computers / Amazon Photo / Coauthor CS subgraphs.
   Note: the NumPy version excluded Amazon Computers because its hand-written
   full-batch optimizer converged unreliably there; with PyG + Adam this
   limitation no longer applies, so Computers is reinstated (the try/except
   keeps a failure from aborting the suite either way).
2. ``real_webkb_study``    -- REAL Texas / Cornell / Wisconsin (Geom-GCN files).
3. ``streaming_tta_study`` -- continual TTA over a stream of shifts, no reset.
4. ``adversarial_study``   -- adversarial edge insertions and feature attacks.

Same seeds, conditions, budgets, and output schema as the NumPy version; writes
to ``results_torch/extended/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from _np_bridge import (
    evaluate,
    is_sparse_matrix,
    rebuild_adjacency,
    upper_triangle_edges,
)
from adaptation import _make_predict_fn, adapt_classifier
from data_adapter import build_bundle
from detector import DetectorState
from exp_common import shift_bundle, train_source, webkb_bundle
from _np_bridge import graph_homophily

OUT_DIR = Path(__file__).resolve().parents[1] / "results_torch" / "extended"


def _write_json(path: Path, records):
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


# --------------------------------------------------------------------------- 1
def large_scale_study(out_dir, seeds=(0, 1, 2), max_nodes=2500):
    datasets = ["coauthor_cs", "amazon_photo", "amazon_computers"]
    conditions = [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.15), ("edge_add", 0.10)]
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    records = []
    for dataset in datasets:
        for seed in seeds:
            try:
                model, base, _ = train_source(dataset, seed, hidden=48, epochs=250,
                                              max_nodes=max_nodes, graph_backend="sparse")
                clean_p = model.predict_probs(base.x, base.edge_index).cpu().numpy()
                clean_acc = float(np.mean(np.argmax(clean_p[base.test_idx], axis=1) == base.y_np[base.test_idx]))
                print(f"[large] {dataset} seed={seed} clean acc={clean_acc:.4f} n={base.num_nodes}")
                for shift, intensity in conditions:
                    sb = shift_bundle(base, seed, shift, intensity)
                    for method in methods:
                        m = model.clone()
                        detector = DetectorState() if method == "full_method" else None
                        t0 = time.perf_counter()
                        adapt_classifier(m, sb, method=method, seed=seed, steps=30, detector=detector)
                        runtime = time.perf_counter() - t0
                        probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                        metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                        records.append({
                            "dataset": dataset, "seed": seed, "shift": shift, "intensity": intensity,
                            "method": method, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                            "ece": metrics["ece"], "nll": metrics["nll"], "brier": metrics["brier"],
                            "runtime_seconds": runtime, "num_nodes": int(base.num_nodes),
                            "detector_triggered": bool(detector.triggered) if detector else None,
                        })
                        print(f"[large] {dataset} seed={seed} {shift}/{intensity} {method}: "
                              f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}")
            except Exception as e:
                print(f"[large] {dataset} seed={seed} FAIL: {type(e).__name__}: {e}")
                continue
            _write_json(out_dir / "large_scale_study.json", records)
            _write_csv(out_dir / "large_scale_study.csv", records)
    _write_json(out_dir / "large_scale_study.json", records)
    _write_csv(out_dir / "large_scale_study.csv", records)
    return records


# --------------------------------------------------------------------------- 2
def real_webkb_study(out_dir, seeds=(0, 1, 2, 3, 4)):
    datasets = ["texas", "cornell", "wisconsin"]
    conditions = [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.10), ("edge_add", 0.05)]
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    records = []
    for dataset in datasets:
        for seed in seeds:
            base = webkb_bundle(dataset, seed)
            hG = graph_homophily(base.adj, base.y_np)
            model, base, _ = train_source(dataset, seed, hidden=24, epochs=400, bundle=base)
            clean_p = model.predict_probs(base.x, base.edge_index).cpu().numpy()
            clean_acc = float(np.mean(np.argmax(clean_p[base.test_idx], axis=1) == base.y_np[base.test_idx]))
            print(f"[webkb] {dataset} seed={seed} clean acc={clean_acc:.4f} hG={hG:.3f}")
            for shift, intensity in conditions:
                sb = shift_bundle(base, seed, shift, intensity)
                for method in methods:
                    m = model.clone()
                    detector = DetectorState() if method == "full_method" else None
                    adapt_classifier(m, sb, method=method, seed=seed, steps=40, detector=detector)
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    records.append({
                        "dataset": dataset, "seed": seed, "shift": shift, "intensity": intensity,
                        "method": method, "accuracy": metrics["accuracy"], "ece": metrics["ece"],
                        "nll": metrics["nll"], "homophily": hG,
                        "detector_triggered": bool(detector.triggered) if detector else None,
                    })
                    print(f"[webkb] {dataset} seed={seed} {shift}/{intensity} {method}: "
                          f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f}")
    _write_json(out_dir / "real_webkb_study.json", records)
    _write_csv(out_dir / "real_webkb_study.csv", records)
    return records


# --------------------------------------------------------------------------- 3
def streaming_tta_study(out_dir, seeds=(0, 1, 2, 3, 4)):
    streams = {
        "feature_drift_stream": [("feature_noise", i) for i in (0.05, 0.10, 0.20, 0.30, 0.40)],
        "edge_perturb_stream": [("edge_drop", 0.05), ("edge_add", 0.05), ("edge_drop", 0.10),
                                ("edge_add", 0.10), ("edge_drop", 0.15)],
        "homophily_drift_stream": [("homophily_shift", i) for i in (0.10, 0.20, 0.30, 0.40, 0.50)],
    }
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    records = []
    for stream_name, stream in streams.items():
        for seed in seeds:
            model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=240)
            running_models = {method: model.clone() for method in methods}
            running_detector = {method: DetectorState() if method == "full_method" else None for method in methods}
            for step, (shift, intensity) in enumerate(stream):
                sb = shift_bundle(base, seed + step * 17, shift, intensity)
                for method in methods:
                    m = running_models[method]
                    if method != "source_only":
                        adapt_classifier(m, sb, method=method, seed=seed + step, steps=25,
                                         detector=running_detector[method])
                    probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                    metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                    records.append({
                        "stream": stream_name, "seed": seed, "step": step, "shift": shift,
                        "intensity": intensity, "method": method, "accuracy": metrics["accuracy"],
                        "ece": metrics["ece"], "nll": metrics["nll"],
                        "detector_triggered": (bool(running_detector[method].triggered)
                                               if running_detector[method] is not None else None),
                    })
            print(f"[stream] {stream_name} seed={seed} done")
    _write_json(out_dir / "streaming_tta_study.json", records)
    _write_csv(out_dir / "streaming_tta_study.csv", records)
    return records


# --------------------------------------------------------------------------- 4
def _adversarial_edge_attack(seed, x, adj, y, intensity, predict_fn=None):
    """Adversarial edge insertions between confidently-different-class pairs.
    Verbatim NumPy attack logic from code/extended_experiments.py."""
    rng = np.random.default_rng(seed + 3331)
    n = len(y)
    use_sparse = is_sparse_matrix(adj)
    edges = upper_triangle_edges(adj)
    if predict_fn is None:
        probs = np.eye(int(y.max() + 1))[y]
    else:
        probs = predict_fn(x, adj)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    budget = max(1, int(len(edges) * intensity))
    high_conf = np.where(conf > np.quantile(conf, 0.6))[0]
    rng.shuffle(high_conf)
    pairs = []
    attempts = 0
    edge_set = {(int(i), int(j)) for i, j in edges.tolist()}
    while len(pairs) < budget and attempts < 8 * budget and len(high_conf) > 1:
        i = int(rng.choice(high_conf))
        j = int(rng.choice(high_conf))
        attempts += 1
        if i == j or pred[i] == pred[j]:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_set:
            continue
        pairs.append((a, b))
        edge_set.add((a, b))
    if pairs:
        merged = np.vstack([edges, np.asarray(pairs, dtype=int)]) if len(edges) else np.asarray(pairs, dtype=int)
    else:
        merged = edges
    return x, rebuild_adjacency(n, merged, use_sparse=use_sparse)


def _adversarial_feature_attack(seed, x, adj, y, intensity):
    """Move node features toward a different class centroid (verbatim NumPy logic)."""
    rng = np.random.default_rng(seed + 4441)
    classes = int(y.max() + 1)
    centroids = np.zeros((classes, x.shape[1]))
    for c in range(classes):
        mask = y == c
        if np.any(mask):
            centroids[c] = x[mask].mean(axis=0)
    x_adv = x.copy()
    for i in range(len(y)):
        ci = y[i]
        target = (ci + 1 + int(rng.integers(0, classes - 1))) % classes
        if target == ci:
            target = (target + 1) % classes
        delta = centroids[target] - x[i]
        x_adv[i] = x[i] + intensity * delta
    return x_adv, adj


def adversarial_study(out_dir, seeds=(0, 1, 2, 3, 4)):
    attacks = [("adv_edge_add", 0.10), ("adv_edge_add", 0.20), ("adv_feature", 0.20), ("adv_feature", 0.40)]
    methods = ["source_only", "entropy_all_nodes", "full_method"]
    records = []
    for seed in seeds:
        model, base, _ = train_source("synthetic", seed, hidden=24, epochs=300, n=300)
        predict_fn = _make_predict_fn(model)
        for attack, intensity in attacks:
            if attack == "adv_edge_add":
                x_t, adj_t = _adversarial_edge_attack(seed, base.x_np, base.adj, base.y_np,
                                                      intensity, predict_fn=predict_fn)
            else:
                x_t, adj_t = _adversarial_feature_attack(seed, base.x_np, base.adj, base.y_np, intensity)
            sb = build_bundle(x_t, adj_t, base.y_np, base.train_idx, base.val_idx, base.test_idx)
            probs0 = model.predict_probs(sb.x, sb.edge_index).cpu().numpy()
            base_acc = float(np.mean(np.argmax(probs0[sb.test_idx], axis=1) == sb.y_np[sb.test_idx]))
            for method in methods:
                m = model.clone()
                detector = DetectorState() if method == "full_method" else None
                adapt_classifier(m, sb, method=method, seed=seed, steps=40, detector=detector)
                probs = m.predict_probs(sb.x, sb.edge_index).cpu().numpy()
                metrics = evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)
                records.append({
                    "seed": seed, "attack": attack, "intensity": intensity, "method": method,
                    "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                    "ece": metrics["ece"], "nll": metrics["nll"], "base_accuracy": base_acc,
                    "detector_triggered": bool(detector.triggered) if detector else None,
                })
                print(f"[adv] seed={seed} {attack}/{intensity} {method}: "
                      f"acc={metrics['accuracy']:.4f} ece={metrics['ece']:.4f} (base={base_acc:.4f})")
    _write_json(out_dir / "adversarial_study.json", records)
    _write_csv(out_dir / "adversarial_study.csv", records)
    return records


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="all",
                        help="comma-separated: large_scale_study, real_webkb_study, "
                             "streaming_tta_study, adversarial_study, all")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments = args.experiments.split(",")
    if "all" in experiments:
        experiments = ["real_webkb_study", "adversarial_study", "streaming_tta_study", "large_scale_study"]
    started = time.perf_counter()
    if "real_webkb_study" in experiments:
        real_webkb_study(out_dir)
    if "adversarial_study" in experiments:
        adversarial_study(out_dir)
    if "streaming_tta_study" in experiments:
        streaming_tta_study(out_dir)
    if "large_scale_study" in experiments:
        large_scale_study(out_dir)
    print(f"extended experiments completed in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
