"""NumPy graph -> PyTorch Geometric conversion.

Loads graphs through the existing NumPy loaders (``_np_bridge``) and converts
them to torch tensors + ``edge_index``.  A :class:`GraphBundle` keeps *both* the
NumPy adjacency (used by the reliability estimator / detector, which operate in
NumPy-adj space exactly as in the original paper) and the torch tensors (used by
the PyG model forward pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from _np_bridge import (
    apply_shift,
    degree_vector,
    is_sparse_matrix,
    load_public_graph_dataset,
    make_contextual_sbm,
    make_heterophily_benchmark,
    split_indices,
    upper_triangle_edges,
)

DEVICE = torch.device("cpu")  # small full-batch graphs: CPU is fast and deterministic


def adj_to_edge_index(adj) -> torch.Tensor:
    """Symmetric 0/1 adjacency (dense ndarray or scipy sparse) -> edge_index [2, E].

    Both directions of every undirected edge are emitted (plus is left to the
    GNN layer, which adds self-loops where appropriate).
    """
    if is_sparse_matrix(adj):
        coo = adj.tocoo()
        row, col = coo.row, coo.col
    else:
        row, col = np.asarray(adj > 0).nonzero()
    if len(row) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([row, col]), dtype=torch.long, device=DEVICE)


def _mask(idx, n) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    m[torch.as_tensor(np.asarray(idx, dtype=np.int64))] = True
    return m


@dataclass
class GraphBundle:
    # NumPy-space (for reliability / detector / metrics, paper-identical)
    x_np: np.ndarray
    adj: object  # dense ndarray or scipy sparse
    y_np: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    # torch-space (for the PyG model)
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int

    @property
    def num_nodes(self) -> int:
        return int(self.x_np.shape[0])


def build_bundle(x_np, adj, y_np, train_idx, val_idx, test_idx) -> GraphBundle:
    n = x_np.shape[0]
    num_classes = int(np.max(y_np)) + 1
    x = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(np.asarray(y_np), dtype=torch.long, device=DEVICE)
    return GraphBundle(
        x_np=np.asarray(x_np), adj=adj, y_np=np.asarray(y_np),
        train_idx=np.asarray(train_idx), val_idx=np.asarray(val_idx), test_idx=np.asarray(test_idx),
        x=x, edge_index=adj_to_edge_index(adj), y=y,
        train_mask=_mask(train_idx, n), val_mask=_mask(val_idx, n), test_mask=_mask(test_idx, n),
        num_classes=num_classes,
    )


def load_bundle(dataset, seed=0, n=None, max_nodes=None, graph_backend="auto",
                train_per_class=20, val_per_class=30) -> GraphBundle:
    """Load a dataset through the NumPy loaders and wrap it as a GraphBundle."""
    heterophily = {"texas", "cornell", "wisconsin", "actor", "film"}
    if dataset == "synthetic":
        x, adj, y = make_contextual_sbm(seed=seed, n=n or 360)
        tr, va, te = split_indices(seed, y)
    elif dataset in heterophily:
        x, adj, y = make_heterophily_benchmark(seed=seed, name=dataset)
        tr, va, te = split_indices(seed, y, train_per_class=12, val_per_class=12)
    else:
        x, adj, y, tr, va, te = load_public_graph_dataset(
            dataset, max_nodes=max_nodes, seed=seed, graph_backend=graph_backend
        )
    return build_bundle(x, adj, y, tr, va, te)


def shift_bundle(base: GraphBundle, seed, shift, intensity) -> GraphBundle:
    """Apply a covariate/structure shift in NumPy-adj space, re-wrap as a bundle.

    The same ``apply_shift`` as the NumPy experiments is used, so the shifted
    graphs are identical across the two implementations.  ``degree_shift`` can
    remove nodes; to keep labels aligned we fall back to the original graph in
    that case (mirroring ``code/main.py`` / ``code/fair_comparison.py``).
    """
    x_t, adj_t = apply_shift(seed, base.x_np, base.adj, base.y_np, shift, intensity)
    if len(x_t) != len(base.y_np):
        x_t, adj_t = base.x_np.copy(), base.adj.copy() if not is_sparse_matrix(base.adj) else base.adj.copy()
    return build_bundle(x_t, adj_t, base.y_np, base.train_idx, base.val_idx, base.test_idx)
