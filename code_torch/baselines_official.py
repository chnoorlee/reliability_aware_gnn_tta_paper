"""Official / faithful test-time adaptation baselines on a shared PyG backbone.

* ``run_tent``  — the **vendored official Tent** (``third_party/tent.py``):
  adapts the BatchNorm affine params (gamma/beta) and forces batch statistics
  on the target graph.  The classifier is NOT touched.  (req#3)
* ``run_eata``  — the **vendored official EATA** (``third_party/eata.py``):
  entropy-filtered + redundancy-filtered BN adaptation with a diagonal-Fisher
  anti-forgetting regularizer computed from the labeled source graph.  (req#2/#3)
* ``run_matcha`` / ``run_gtrans`` — faithful PyTorch ports of the graph-native
  baselines (graph-aware reliability masking; test-time feature transformation),
  mirroring ``code/full_baselines.py`` on the shared backbone.

A :class:`GraphModelWrapper` lets the official ``model(x)`` API drive a PyG model
``gnn(x, edge_index)`` — the only adaptation needed to run image-CNN TTA code on
graphs (see ``third_party/ATTRIBUTION.md``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reliability import entropy as np_entropy
from reliability import neighborhood_agreement, top_margin
from third_party import eata as eata_lib
from third_party import tent as tent_lib


class GraphModelWrapper(nn.Module):
    """Expose a PyG model ``gnn(x, edge_index)`` through the single-argument
    ``model(x)`` interface the official Tent/EATA code expects.  ``edge_index``
    is held fixed for the (full-batch) target graph.
    """

    def __init__(self, gnn, edge_index):
        super().__init__()
        self.model = gnn
        self._edge_index = edge_index

    def forward(self, x):
        return self.model(x, self._edge_index)


@torch.no_grad()
def _eval_probs(wrap, x):
    """Adapted-model predictions: eval mode (dropout off); BN with
    track_running_stats=False still uses batch statistics."""
    wrap.eval()
    return F.softmax(wrap(x), dim=1).cpu().numpy()


# --------------------------------------------------------------------- Tent
def run_tent(source_model, sb, steps=10, lr=0.005, momentum=0.9):
    wrap = GraphModelWrapper(source_model.clone(), sb.edge_index)
    tent_lib.configure_model(wrap)
    params, names = tent_lib.collect_params(wrap)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    tented = tent_lib.Tent(wrap, optimizer, steps=steps)
    tented(sb.x)
    return _eval_probs(wrap, sb.x), names


# --------------------------------------------------------------------- EATA
def compute_fisher_bn(source_model, base):
    """Diagonal Fisher of the BN affine params from the labeled source graph,
    in EATA's ``{name: [fisher_diag, anchor_value]}`` format."""
    wrap = GraphModelWrapper(source_model.clone(), base.edge_index)
    eata_lib.configure_model(wrap)
    _, names = eata_lib.collect_params(wrap)
    names_set = set(names)
    wrap.zero_grad()
    logits = wrap(base.x)
    loss = F.cross_entropy(logits[base.train_mask], base.y[base.train_mask])
    loss.backward()
    fishers = {}
    for name, p in wrap.named_parameters():
        if name in names_set and p.grad is not None:
            fishers[name] = [p.grad.data.clone() ** 2, p.data.clone()]
    wrap.zero_grad()
    return fishers


def run_eata(source_model, sb, base, steps=10, lr=0.005, momentum=0.9,
             e_margin=None, d_margin=0.05, fisher_alpha=2000.0):
    num_classes = sb.num_classes
    if e_margin is None:
        e_margin = 0.4 * math.log(num_classes)  # EATA's E_0 scaled to #classes
    fishers = compute_fisher_bn(source_model, base)
    wrap = GraphModelWrapper(source_model.clone(), sb.edge_index)
    eata_lib.configure_model(wrap)
    params, names = eata_lib.collect_params(wrap)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    adapter = eata_lib.EATA(wrap, optimizer, fishers=fishers, fisher_alpha=fisher_alpha,
                            steps=steps, e_margin=e_margin, d_margin=d_margin)
    adapter(sb.x)
    return _eval_probs(wrap, sb.x), names


# ----------------------------------------------------------- shared helpers
def _graph_aware_score(adj, probs):
    """Confidence x neighborhood-agreement score (mirror of code/full_baselines)."""
    ent_conf = 1.0 - np_entropy(probs) / np.log(probs.shape[1])
    margin = top_margin(probs)
    conf = 0.65 * ent_conf + 0.35 * margin
    agree, _ = neighborhood_agreement(adj, probs)
    return np.clip(0.5 * conf + 0.5 * agree, 0.0, 1.0)


def _clip_step(weight, grad, lr, clip=2.0):
    grad_norm = float(torch.sqrt((grad ** 2).sum()))
    scale = min(1.0, clip / max(grad_norm, 1e-12))
    with torch.no_grad():
        weight.add_(grad, alpha=-lr * scale)


# ------------------------------------------------------------------- Matcha
def run_matcha(source_model, sb, steps=10, lr=0.05, mask_fraction=0.5):
    """Matcha: graph-aware reliability masking + classifier-layer entropy update."""
    gnn = source_model.clone()
    gnn.eval()
    gnn.enable_classifier_grad()  # BN frozen (running stats); adapt classifier only
    weight = gnn.classifier_weight()
    for _ in range(steps):
        logits = gnn(sb.x, sb.edge_index)
        probs_t = F.softmax(logits, dim=1)
        probs = probs_t.detach().cpu().numpy()
        score = _graph_aware_score(sb.adj, probs)
        thresh = np.quantile(score, 1.0 - mask_fraction)
        mask = (score >= thresh).astype(float)
        weights = torch.tensor(mask / max(mask.sum(), 1.0), dtype=torch.float32)
        ent_t = -(probs_t.clamp_min(1e-12).log() * probs_t).sum(dim=1)
        loss = (weights * ent_t).sum()
        if weight.grad is not None:
            weight.grad = None
        loss.backward()
        _clip_step(weight, weight.grad.detach(), lr)
    return _classifier_eval_probs(gnn, sb.x, sb.edge_index)


# ------------------------------------------------------------------- GTrans
def _feature_smoothness(x, edge_index):
    """Graph Dirichlet energy: neighboring node features should stay close."""
    src, dst = edge_index[0], edge_index[1]
    diff = x[src] - x[dst]
    return (diff * diff).sum(dim=1).mean()


def run_gtrans(source_model, sb, steps=10, lr_feat=0.05, lr_clf=0.02, smooth=0.1):
    """GTrans: test-time additive feature transformation (delta-X) minimizing
    entropy + graph smoothness, then a light classifier update."""
    gnn = source_model.clone()
    gnn.eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    delta = torch.zeros_like(sb.x, requires_grad=True)
    for _ in range(steps):
        x_t = sb.x + delta
        probs = F.softmax(gnn(x_t, sb.edge_index), dim=1)
        ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1).mean()
        loss = ent + smooth * _feature_smoothness(x_t, sb.edge_index)
        if delta.grad is not None:
            delta.grad = None
        loss.backward()
        _clip_step(delta, delta.grad.detach(), lr_feat)
    # light classifier update on the transformed graph
    gnn.enable_classifier_grad()
    weight = gnn.classifier_weight()
    x_t = (sb.x + delta).detach()
    probs = F.softmax(gnn(x_t, sb.edge_index), dim=1)
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=1).mean()
    if weight.grad is not None:
        weight.grad = None
    ent.backward()
    _clip_step(weight, weight.grad.detach(), lr_clf)
    return _classifier_eval_probs(gnn, x_t, sb.edge_index)


@torch.no_grad()
def _classifier_eval_probs(gnn, x, edge_index):
    gnn.eval()
    return F.softmax(gnn(x, edge_index), dim=1).cpu().numpy()
