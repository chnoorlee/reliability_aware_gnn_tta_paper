"""Faithful reimplementations of the *original mechanisms* of four graph-TTA
baselines, as opposed to the classifier-only analogues used in the controlled
ablation.  These are used for the fair head-to-head comparison (Experiment 1).

* ``tent_full``   -- Tent (Wang et al., ICLR 2021): use target-graph BN statistics
  and update the BN affine parameters (gamma, beta) by entropy minimization.
  The classifier is NOT updated.

* ``eata_full``   -- EATA (Niu et al., ICML 2022): entropy-based sample selection
  (adapt only low-entropy nodes) + diagonal-Fisher anti-forgetting regularizer on
  the BN affine parameters + BN updates.

* ``matcha_full`` -- Matcha (Mao et al., ICLR 2025): graph-aware reliability
  masking (select reliable nodes via a confidence x neighborhood-agreement score)
  followed by entropy minimization that updates the classifier layer.

* ``gtrans_full`` -- GTrans (Jin et al., ICLR 2023): test-time graph
  transformation -- learn an additive feature transformation delta-X (and a light
  classifier update) that minimizes an unsupervised surrogate, then run inference
  on the transformed graph.

All four operate on the shared ``GCNWithBN`` backbone so that the comparison
controls for architecture, training, test-time learning rate, and step budget.
These are faithful NumPy reimplementations of the core mechanisms; they are not
wrappers around the authors' official code (which targets PyTorch Geometric).
"""

from __future__ import annotations

import numpy as np

from adaptation import entropy, neighborhood_agreement, top_margin
from utils import normalize_adjacency


def _entropy_dlogits(probs, weights=None):
    p = np.clip(probs, 1e-12, 1.0)
    logp1 = np.log(p) + 1.0
    weighted_sum = np.sum(p * logp1, axis=1, keepdims=True)
    dlogits = p * (weighted_sum - logp1)
    if weights is not None:
        dlogits = weights[:, None] * dlogits
    else:
        dlogits = dlogits / probs.shape[0]
    return dlogits


# ----------------------------------------------------------------------- Tent
def tent_full(model, x, adj, steps=40, lr=0.05, clip=2.0):
    """Tent: update BN affine params via entropy minimization on target stats."""
    for _ in range(steps):
        grads, probs, _ = model.bn_entropy_grad(x, adj, use_running=False)
        gn = np.sqrt(np.sum(grads["gamma"] ** 2) + np.sum(grads["beta"] ** 2))
        scale = min(1.0, clip / max(gn, 1e-12))
        model.gamma -= lr * scale * grads["gamma"]
        model.beta -= lr * scale * grads["beta"]
    probs, _ = model.forward(x, adj, use_running=False)
    return probs, {"method": "tent_full"}


# ----------------------------------------------------------------------- EATA
def eata_full(model, x, adj, fisher, steps=40, lr=0.05, ent_quantile=0.5, lambda_fisher=1.0, clip=2.0):
    """EATA: entropy-filtered BN adaptation with diagonal-Fisher anti-forgetting."""
    gamma0 = model.gamma_source.copy()
    beta0 = model.beta_source.copy()
    for _ in range(steps):
        probs, _ = model.forward(x, adj, use_running=False)
        ent = entropy(probs) / np.log(probs.shape[1])
        thresh = np.quantile(ent, ent_quantile)
        mask = (ent <= thresh).astype(float)
        denom = max(float(mask.sum()), 1.0)
        weights = mask / denom
        grads, probs, _ = model.bn_entropy_grad(x, adj, use_running=False, weights=weights)
        # Fisher-weighted anti-forgetting pull toward source BN params
        grads["gamma"] += lambda_fisher * fisher["gamma"] * (model.gamma - gamma0)
        grads["beta"] += lambda_fisher * fisher["beta"] * (model.beta - beta0)
        gn = np.sqrt(np.sum(grads["gamma"] ** 2) + np.sum(grads["beta"] ** 2))
        scale = min(1.0, clip / max(gn, 1e-12))
        model.gamma -= lr * scale * grads["gamma"]
        model.beta -= lr * scale * grads["beta"]
    probs, _ = model.forward(x, adj, use_running=False)
    return probs, {"method": "eata_full"}


# --------------------------------------------------------------------- Matcha
def _graph_aware_score(adj, probs):
    """Confidence x neighborhood-agreement score used for graph-aware masking."""
    ent_conf = 1.0 - entropy(probs) / np.log(probs.shape[1])
    margin = top_margin(probs)
    conf = 0.65 * ent_conf + 0.35 * margin
    agree, _ = neighborhood_agreement(adj, probs)
    return np.clip(0.5 * conf + 0.5 * agree, 0.0, 1.0)


def matcha_full(model, x, adj, steps=40, lr=0.05, mask_fraction=0.5, clip=2.0):
    """Matcha: graph-aware reliability masking + classifier-layer entropy update."""
    for _ in range(steps):
        probs, cache = model.forward(x, adj, use_running=True)
        score = _graph_aware_score(adj, probs)
        thresh = np.quantile(score, 1.0 - mask_fraction)
        mask = (score >= thresh).astype(float)
        denom = max(float(mask.sum()), 1.0)
        weights = mask / denom
        dlogits = _entropy_dlogits(probs, weights=weights)
        dw1 = cache["ah"].T @ dlogits
        gn = np.sqrt(np.sum(dw1 ** 2))
        scale = min(1.0, clip / max(gn, 1e-12))
        model.w1 -= lr * scale * dw1
    probs, _ = model.forward(x, adj, use_running=True)
    return probs, {"method": "matcha_full"}


# --------------------------------------------------------------------- GTrans
def gtrans_full(model, x, adj, steps=40, lr_feat=0.05, lr_clf=0.02, smooth=0.1, clip=2.0):
    """GTrans: test-time feature transformation (delta-X) + light classifier update.

    Learns an additive feature transformation that minimizes prediction entropy
    plus a graph-smoothness surrogate (neighboring node features should remain
    close), then applies a small classifier update on the transformed graph.
    The structure-transformation branch of the original GTrans is approximated by
    the smoothness term; we note this scope in the paper.
    """
    a = normalize_adjacency(adj)
    delta = np.zeros_like(x)
    for _ in range(steps):
        x_t = x + delta
        probs, cache = model.forward(x_t, adj, use_running=False)
        dlogits = _entropy_dlogits(probs)
        grads = model._backward(cache, dlogits, weight_decay=0.0)
        # backprop entropy gradient to the input features: dL/dx = a^T (dz1 @ w0^T)
        dax = grads["dz1"] @ model.w0.T
        dx = a.T @ dax
        # graph-smoothness surrogate: pull each feature toward its neighbor mean
        neigh_mean = a @ x_t
        dx_smooth = smooth * (x_t - neigh_mean)
        dtotal = dx + dx_smooth
        gn = np.sqrt(np.sum(dtotal ** 2))
        scale = min(1.0, clip / max(gn, 1e-12))
        delta -= lr_feat * scale * dtotal
    # light classifier update on the transformed graph
    x_t = x + delta
    probs, cache = model.forward(x_t, adj, use_running=False)
    dlogits = _entropy_dlogits(probs)
    dw1 = cache["ah"].T @ dlogits
    gn = np.sqrt(np.sum(dw1 ** 2))
    scale = min(1.0, clip / max(gn, 1e-12))
    model.w1 -= lr_clf * scale * dw1
    probs, _ = model.forward(x_t, adj, use_running=False)
    return probs, {"method": "gtrans_full", "delta_norm": float(np.linalg.norm(delta))}
