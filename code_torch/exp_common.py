"""Shared helpers for the migrated supplementary / extended / scalability /
boundary experiments: train a source model and build (real-WebKB / arxiv)
bundles, reusing the same PyG model + adaptation stack as ``main.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from _np_bridge import load_real_webkb, make_arxiv_subset
from data_adapter import build_bundle, load_bundle, shift_bundle  # noqa: F401 (re-exported)
from models import make_model, train_model


def model_kwargs(backbone, hidden):
    """GAT uses 8 heads x 8 dims (matches main.py); others use the given hidden."""
    if backbone == "gat":
        return {"hidden_dim": 8, "heads": 8}
    return {"hidden_dim": hidden}


def train_source(dataset, seed, backbone="gcn", hidden=24, epochs=300, n=None,
                 max_nodes=None, use_bn=False, graph_backend="auto",
                 train_per_class=20, val_per_class=30, bundle=None):
    """Train a source model on ``dataset`` (or a pre-built ``bundle``) and return
    ``(model, base_bundle, train_info)``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = bundle if bundle is not None else load_bundle(
        dataset, seed=seed, n=n, max_nodes=max_nodes, graph_backend=graph_backend,
        train_per_class=train_per_class, val_per_class=val_per_class)
    kw = model_kwargs(backbone, hidden)
    model = make_model(backbone, base.x.shape[1], out_dim=base.num_classes,
                       use_bn=use_bn, seed=seed, **kw)
    info = train_model(model, base.x, base.edge_index, base.y, base.train_mask, base.val_mask,
                       epochs=epochs, lr=0.01, weight_decay=5e-4, patience=max(40, epochs // 4))
    return model, base, info


def webkb_bundle(name, seed, train_per_class=6, val_per_class=6):
    """Real Texas/Cornell/Wisconsin (Geom-GCN) as a GraphBundle."""
    x, adj, y, tr, va, te = load_real_webkb(name, seed=seed,
                                            train_per_class=train_per_class, val_per_class=val_per_class)
    return build_bundle(x, adj, y, tr, va, te)


def arxiv_bundle(seed, n=2000):
    """ogbn-arxiv-like synthetic subset as a GraphBundle (scalability stress test)."""
    x, adj, y, tr, va, te = make_arxiv_subset(seed, n=n)
    return build_bundle(x, adj, y, tr, va, te)
