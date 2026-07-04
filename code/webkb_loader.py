"""Loader for the *real* WebKB heterophily benchmarks (Texas, Cornell, Wisconsin).

Source: Geom-GCN release (Pei et al., ICLR 2020),
https://github.com/graphdml-uiuc-jlu/geom-gcn/tree/master/new_data

Files expected under ``data/webkb/``:
    {dataset}_out1_node_feature_label.txt
    {dataset}_out1_graph_edges.txt

These are the canonical Texas/Cornell/Wisconsin splits used as low-homophily
benchmarks by the heterophily community.  The loader returns the same
``(features, adj, labels, train_idx, val_idx, test_idx)`` tuple as the
existing public-dataset loaders so it drops into the experiment runner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils import rebuild_adjacency


DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "webkb"


def load_real_webkb(name: str, seed: int = 0, train_per_class: int = 6, val_per_class: int = 6):
    name = name.lower()
    if name not in ("texas", "cornell", "wisconsin"):
        raise ValueError(f"Unsupported real WebKB dataset: {name}")

    feature_path = DATA_ROOT / f"{name}_out1_node_feature_label.txt"
    edge_path = DATA_ROOT / f"{name}_out1_graph_edges.txt"
    if not feature_path.exists() or not edge_path.exists():
        raise FileNotFoundError(
            f"Real WebKB data for {name} not found under {DATA_ROOT}. "
            f"Run the download in supplementary_experiments.py first."
        )

    # ---- Node features and labels ----------------------------------------
    features, labels = [], []
    with feature_path.open(encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            node_id, feat_str, lab_str = parts
            features.append([int(v) for v in feat_str.split(",")])
            labels.append(int(lab_str))
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = features.shape[0]

    # row-normalise features
    row_sum = np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    features = features / row_sum

    # ---- Edges ------------------------------------------------------------
    edges = []
    with edge_path.open(encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            edges.append((i, j))
    edges = np.unique(np.asarray(sorted(set(edges)), dtype=int), axis=0) if edges else np.zeros((0, 2), dtype=int)
    adj = rebuild_adjacency(n, edges, use_sparse=False)

    # ---- Splits ----------------------------------------------------------
    train_idx, val_idx, test_idx = _per_class_split(seed, labels, train_per_class, val_per_class)
    return features, adj, labels, train_idx, val_idx, test_idx


def _per_class_split(seed, labels, train_per_class=6, val_per_class=6):
    rng = np.random.default_rng(seed + 9091)
    train, val, test = [], [], []
    classes = np.unique(labels)
    for c in classes:
        nodes = np.where(labels == c)[0]
        rng.shuffle(nodes)
        take_train = min(train_per_class, max(0, len(nodes) // 3))
        take_val = min(val_per_class, max(0, (len(nodes) - take_train) // 2))
        train.extend(nodes[:take_train])
        val.extend(nodes[take_train: take_train + take_val])
        test.extend(nodes[take_train + take_val:])
    return (
        np.array(train, dtype=int),
        np.array(val, dtype=int),
        np.array(test, dtype=int),
    )


def graph_homophily(adj, labels):
    if hasattr(adj, "tocsr"):
        coo = adj.tocoo()
        i, j = coo.row, coo.col
    else:
        i, j = np.where(adj > 0)
    if len(i) == 0:
        return 0.0
    mask = i < j
    i, j = i[mask], j[mask]
    if len(i) == 0:
        return 0.0
    return float(np.mean(labels[i] == labels[j]))


if __name__ == "__main__":
    for name in ("texas", "cornell", "wisconsin"):
        try:
            x, adj, y, tr, va, te = load_real_webkb(name)
            h = graph_homophily(adj, y)
            print(f"{name}: nodes={x.shape[0]} edges={int(adj.sum() / 2)} classes={int(y.max()+1)} "
                  f"feature_dim={x.shape[1]} homophily={h:.3f} train={len(tr)} val={len(va)} test={len(te)}")
        except Exception as e:
            print(f"{name}: FAIL {e}")
