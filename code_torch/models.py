"""PyTorch Geometric backbones for the reliability-aware graph TTA framework.

This module replaces the hand-written NumPy backbones (``code/models.py``,
``code/models_bn.py``, ``code/backbones.py``) with the *official* PyG message
passing layers (``GCNConv``, ``GATConv``, ``SAGEConv``, ``APPNP``).  Two design
points carry over from the NumPy framework so that the paper's theory and
baselines still hold:

* Every backbone exposes a **final linear classifier** (``classifier_parameters``)
  that corresponds to ``W1`` in the NumPy code.  The proposed test-time method
  adapts *only* this layer, preserving the closed-form classifier-only drift
  bound.

* Every backbone optionally carries a ``BatchNorm1d`` layer.  The official Tent
  and EATA baselines adapt the BN affine parameters and running statistics, so
  a real BN module is required for a faithful comparison (Section 3, fair
  comparison experiment).

All models run full-batch on CPU with a fixed seed for determinism; the graphs
in this project are small enough that CPU is both fast and reproducible.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPProp
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))


class _GNNBase(nn.Module):
    """Shared helpers: classifier access, snapshot/clone, probability output."""

    # Subclasses set ``self._classifier`` to the final nn.Linear-like module
    # whose weight is the adapted classifier (the NumPy ``W1``).
    _classifier: nn.Module

    def classifier_parameters(self):
        return list(self._classifier.parameters())

    def classifier_weight(self) -> torch.Tensor:
        # The trainable weight matrix adapted by the proposed method.
        return self._classifier.weight

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def enable_classifier_grad(self) -> None:
        self.freeze_all()
        for p in self.classifier_parameters():
            p.requires_grad_(True)

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state_dict())

    def load_snapshot(self, state: dict) -> None:
        self.load_state_dict(copy.deepcopy(state))

    def clone(self) -> "_GNNBase":
        other = copy.deepcopy(self)
        return other

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits = self.forward(x, edge_index)
        probs = F.softmax(logits, dim=1)
        if was_training:
            self.train()
        return probs


class GCN(_GNNBase):
    """Two-layer GCN with an optional BatchNorm1d between the convolutions.

    Mirrors ``code/models_bn.GCNWithBN``:  AX->conv1->BN->ReLU->conv2.  The
    second ``GCNConv`` is the final classifier (its ``lin`` weight is ``W1``).
    GCNConv adds self-loops and applies symmetric normalization internally, so
    the raw ``edge_index`` is passed directly.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, use_bn=True, dropout=0.5, seed=0):
        super().__init__()
        set_seed(seed)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout
        self._classifier = self.conv2.lin

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GAT(_GNNBase):
    """Two-layer GAT.  Real ``GATConv`` supplies correct attention gradients,
    fixing the NumPy ``TwoLayerGAT`` instability (synthetic clean ~0.62 -> 0.95+).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, heads=8, use_bn=True, dropout=0.6, seed=0):
        super().__init__()
        set_seed(seed)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_dim * heads) if use_bn else nn.Identity()
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        # Final classifier weight = source->target projection of the 2nd attention layer.
        self._classifier = self.conv2.lin if hasattr(self.conv2, "lin") else self.conv2.lin_src

    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.bn(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GraphSAGE(_GNNBase):
    """Two-layer mean-aggregator GraphSAGE with optional BatchNorm1d."""

    def __init__(self, in_dim, hidden_dim, out_dim, use_bn=True, dropout=0.5, seed=0):
        super().__init__()
        set_seed(seed)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout
        # SAGEConv's output projection is ``lin_l`` (root) + ``lin_r`` (neighbor).
        # The classifier weight we adapt is the root linear ``lin_l``.
        self._classifier = self.conv2.lin_l

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class APPNPNet(_GNNBase):
    """Predict-then-propagate (APPNP).  A 2-layer MLP produces logits that are
    smoothed by K personalized-PageRank iterations.  The final MLP linear
    (``lin2``) is the classifier ``W1``; propagation is parameter-free.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, K=10, alpha=0.15, use_bn=True, dropout=0.5, seed=0):
        super().__init__()
        set_seed(seed)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = APPNPProp(K=K, alpha=alpha)
        self.dropout = dropout
        self._classifier = self.lin2

    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.lin1(h)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.lin2(h)
        return self.prop(z, edge_index)


BACKBONES = {"gcn": GCN, "gat": GAT, "graphsage": GraphSAGE, "appnp": APPNPNet}


def make_model(backbone, in_dim, hidden_dim, out_dim, use_bn=True, seed=0, **kwargs):
    backbone = backbone.lower()
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    return BACKBONES[backbone](in_dim, hidden_dim, out_dim, use_bn=use_bn, seed=seed, **kwargs)


def train_model(model, x, edge_index, y, train_mask, val_mask, epochs=200, lr=0.01,
                weight_decay=5e-4, patience=50, exclude_bn_bias_wd=False):
    """Full-batch supervised training with early stopping on validation NLL.

    Returns ``{"epochs", "best_val_loss"}`` and leaves ``model`` loaded with the
    best (lowest val-loss) parameters.  After training, the current parameters
    are snapshotted as the source model for downstream adaptation.

    ``exclude_bn_bias_wd`` excludes BatchNorm affine and bias parameters from
    weight decay (standard practice for BN models; recovers source accuracy on
    the BN backbone used by the Tent/EATA fair comparison).
    """
    if exclude_bn_bias_wd:
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            (no_decay if (".bn" in name or name.endswith(".bias")) else decay).append(p)
        optimizer = torch.optim.Adam(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = model.snapshot()
    stale = 0
    last_epoch = 0
    for epoch in range(epochs):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)
            val_loss = F.cross_entropy(val_logits[val_mask], y[val_mask]).item()
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = model.snapshot()
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_snapshot(best_state)
    # Source anchor for the anti-forgetting term (the NumPy ``w1_source``).  It is
    # set once at training end and survives ``clone()``; repeated adaptation calls
    # (e.g. streaming TTA) keep pulling toward the *original* source classifier.
    model.source_classifier_weight = model.classifier_weight().detach().clone()
    return {"epochs": last_epoch + 1, "best_val_loss": best_val}
