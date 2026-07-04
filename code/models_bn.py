"""Two-layer GCN with a batch-normalization layer.

This backbone is required to reproduce the *original* mechanisms of Tent and
EATA, which adapt batch-normalization parameters (affine scale/shift and the
running statistics) rather than the classifier.  The NumPy implementation
mirrors ``TwoLayerGCN`` but inserts a BN layer between the two graph
convolutions:

    AX  = A_norm @ X
    Z1  = AX @ W0
    Zn  = BN(Z1; gamma, beta)         # normalize over nodes, per hidden dim
    H   = relu(Zn)
    AH  = A_norm @ H
    Z2  = AH @ W1                      # logits
    P   = softmax(Z2)

BN statistics are computed over the node dimension.  During training we use
batch (current-graph) statistics and maintain running estimates with momentum.
At test time the model can either use the stored source running statistics
(``use_running=True``) or recompute statistics on the target graph
(``use_running=False``) -- the latter is the covariate-shift adaptation that
Tent performs.

The class exposes helpers to (i) compute the entropy-loss gradient with respect
to the BN affine parameters (for Tent / EATA), and (ii) compute the diagonal
Fisher information of the BN parameters from the labeled source graph (for
EATA's anti-forgetting regularizer).
"""

from __future__ import annotations

import numpy as np

from utils import normalize_adjacency, one_hot, softmax


class GCNWithBN:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0, weight_scale=0.1, bn_momentum=0.1, eps=1e-5):
        rng = np.random.default_rng(seed)
        self.w0 = rng.normal(0.0, weight_scale, size=(in_dim, hidden_dim))
        self.w1 = rng.normal(0.0, weight_scale, size=(hidden_dim, out_dim))
        self.gamma = np.ones(hidden_dim, dtype=float)
        self.beta = np.zeros(hidden_dim, dtype=float)
        # running statistics (set during training)
        self.running_mean = np.zeros(hidden_dim, dtype=float)
        self.running_var = np.ones(hidden_dim, dtype=float)
        self.bn_momentum = bn_momentum
        self.eps = eps
        # source snapshots (set after training)
        self.w0_source = self.w0.copy()
        self.w1_source = self.w1.copy()
        self.gamma_source = self.gamma.copy()
        self.beta_source = self.beta.copy()

    # ------------------------------------------------------------------ forward
    def forward(self, x, adj, use_running=True, update_running=False):
        a = normalize_adjacency(adj)
        ax = a @ x
        z1 = ax @ self.w0
        if use_running:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = z1.mean(axis=0)
            var = z1.var(axis=0)
            if update_running:
                m = self.bn_momentum
                self.running_mean = (1 - m) * self.running_mean + m * mean
                self.running_var = (1 - m) * self.running_var + m * var
        std = np.sqrt(var + self.eps)
        z1_hat = (z1 - mean) / std
        zn = self.gamma * z1_hat + self.beta
        h = np.maximum(zn, 0.0)
        ah = a @ h
        logits = ah @ self.w1
        probs = softmax(logits)
        cache = {
            "a": a, "ax": ax, "z1": z1, "mean": mean, "var": var, "std": std,
            "z1_hat": z1_hat, "zn": zn, "h": h, "ah": ah, "logits": logits, "probs": probs,
        }
        return probs, cache

    # ------------------------------------------------------------------ train
    def train(self, x, adj, y, train_idx, val_idx, epochs=300, lr=None, weight_decay=5e-4, patience=40, tol=1e-8):
        if lr is None:
            lr = 5.0 if x.shape[1] >= 500 else 0.08
        classes = int(np.max(y)) + 1
        y_one = one_hot(y, classes)
        best_val = float("inf")
        best = None
        stale = 0
        for epoch in range(epochs):
            probs, cache = self.forward(x, adj, use_running=False, update_running=True)
            p_train = np.clip(probs[train_idx], 1e-12, 1.0)
            loss = -np.mean(np.sum(y_one[train_idx] * np.log(p_train), axis=1))

            dlogits = np.zeros_like(probs)
            dlogits[train_idx] = (probs[train_idx] - y_one[train_idx]) / len(train_idx)
            grads = self._backward(cache, dlogits, weight_decay)

            gn = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
            scale = min(1.0, 5.0 / max(gn, 1e-12))
            self.w0 -= lr * scale * grads["w0"]
            self.w1 -= lr * scale * grads["w1"]
            self.gamma -= lr * scale * grads["gamma"]
            self.beta -= lr * scale * grads["beta"]

            probs_val, _ = self.forward(x, adj, use_running=True)
            val_loss = -np.mean(np.log(np.clip(probs_val[val_idx, y[val_idx]], 1e-12, 1.0)))
            if val_loss + tol < best_val:
                best_val = float(val_loss)
                best = (self.w0.copy(), self.w1.copy(), self.gamma.copy(), self.beta.copy(),
                        self.running_mean.copy(), self.running_var.copy())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best is not None:
            self.w0, self.w1, self.gamma, self.beta, self.running_mean, self.running_var = best
        self.w0_source = self.w0.copy()
        self.w1_source = self.w1.copy()
        self.gamma_source = self.gamma.copy()
        self.beta_source = self.beta.copy()
        return {"epochs": epoch + 1, "best_val_loss": best_val}

    # ------------------------------------------------------------------ backward
    def _backward(self, cache, dlogits, weight_decay=0.0):
        a = cache["a"]; ah = cache["ah"]; h = cache["h"]; zn = cache["zn"]
        z1 = cache["z1"]; mean = cache["mean"]; var = cache["var"]; std = cache["std"]
        z1_hat = cache["z1_hat"]; ax = cache["ax"]
        n = z1.shape[0]

        dw1 = ah.T @ dlogits + weight_decay * self.w1
        dah = dlogits @ self.w1.T
        dh = a.T @ dah
        dzn = dh * (zn > 0.0)
        # BN affine grads
        dgamma = np.sum(dzn * z1_hat, axis=0)
        dbeta = np.sum(dzn, axis=0)
        # BN backward into z1
        dz1_hat = dzn * self.gamma
        dvar = np.sum(dz1_hat * (z1 - mean) * (-0.5) * (var + self.eps) ** (-1.5), axis=0)
        dmean = np.sum(dz1_hat * (-1.0 / std), axis=0) + dvar * np.mean(-2.0 * (z1 - mean), axis=0)
        dz1 = dz1_hat / std + dvar * 2.0 * (z1 - mean) / n + dmean / n
        dw0 = ax.T @ dz1 + weight_decay * self.w0
        return {"w0": dw0, "w1": dw1, "gamma": dgamma, "beta": dbeta, "dz1": dz1}

    # ------------------------------------------------------------------ BN-only grad
    def bn_entropy_grad(self, x, adj, use_running=False, weights=None):
        """Gradient of (weighted) entropy w.r.t. BN affine params on the target graph."""
        probs, cache = self.forward(x, adj, use_running=use_running)
        p = np.clip(probs, 1e-12, 1.0)
        logp1 = np.log(p) + 1.0
        weighted_sum = np.sum(p * logp1, axis=1, keepdims=True)
        # d entropy / d logits
        dlogits = p * (weighted_sum - logp1)
        if weights is not None:
            dlogits = weights[:, None] * dlogits
        else:
            dlogits = dlogits / probs.shape[0]
        grads = self._backward(cache, dlogits, weight_decay=0.0)
        return grads, probs, cache

    def fisher_bn(self, x, adj, y, train_idx):
        """Diagonal Fisher information of BN affine params from labeled source graph."""
        classes = int(np.max(y)) + 1
        y_one = one_hot(y, classes)
        probs, cache = self.forward(x, adj, use_running=True)
        dlogits = np.zeros_like(probs)
        dlogits[train_idx] = (probs[train_idx] - y_one[train_idx]) / len(train_idx)
        grads = self._backward(cache, dlogits, weight_decay=0.0)
        # square as a diagonal Fisher proxy
        return {"gamma": grads["gamma"] ** 2, "beta": grads["beta"] ** 2}

    def clone(self):
        other = GCNWithBN(self.w0.shape[0], self.w0.shape[1], self.w1.shape[1])
        other.w0 = self.w0.copy(); other.w1 = self.w1.copy()
        other.gamma = self.gamma.copy(); other.beta = self.beta.copy()
        other.running_mean = self.running_mean.copy(); other.running_var = self.running_var.copy()
        other.w0_source = self.w0_source.copy(); other.w1_source = self.w1_source.copy()
        other.gamma_source = self.gamma_source.copy(); other.beta_source = self.beta_source.copy()
        other.bn_momentum = self.bn_momentum; other.eps = self.eps
        return other
