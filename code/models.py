import numpy as np
from utils import normalize_adjacency, softmax, one_hot


class TwoLayerGCN:
    def __init__(self, in_dim, hidden_dim, out_dim, seed=0, weight_scale=0.1):
        rng = np.random.default_rng(seed)
        self.w0 = rng.normal(0.0, weight_scale, size=(in_dim, hidden_dim))
        self.w1 = rng.normal(0.0, weight_scale, size=(hidden_dim, out_dim))
        self.w0_source = self.w0.copy()
        self.w1_source = self.w1.copy()

    def forward(self, x, adj):
        a = normalize_adjacency(adj)
        ax = a @ x
        z1 = ax @ self.w0
        h = np.maximum(z1, 0.0)
        ah = a @ h
        logits = ah @ self.w1
        probs = softmax(logits)
        cache = {"a": a, "ax": ax, "ah": ah, "z1": z1, "h": h, "logits": logits, "probs": probs}
        return probs, cache

    def train(self, x, adj, y, train_idx, val_idx, epochs=500, lr=None, weight_decay=5e-4, patience=40, tol=1e-8):
        if lr is None:
            # Planetoid-style sparse bag-of-words features produce much smaller
            # dense full-batch gradients than the synthetic Gaussian features.
            # Use a larger default step for public graphs while preserving the
            # original conservative step for synthetic experiments.
            lr = 5.0 if x.shape[1] >= 500 else 0.08
        y_one = one_hot(y, int(np.max(y)) + 1)
        best_val = float("inf")
        best = (self.w0.copy(), self.w1.copy())
        stale = 0
        prev_loss = None
        history = []
        for epoch in range(epochs):
            probs, cache = self.forward(x, adj)
            p_train = np.clip(probs[train_idx], 1e-12, 1.0)
            loss = -np.mean(np.sum(y_one[train_idx] * np.log(p_train), axis=1))
            loss += 0.5 * weight_decay * (np.sum(self.w0 ** 2) + np.sum(self.w1 ** 2))

            dlogits = np.zeros_like(probs)
            dlogits[train_idx] = (probs[train_idx] - y_one[train_idx]) / len(train_idx)
            a = cache["a"]
            h = cache["h"]
            z1 = cache["z1"]
            ah = cache["ah"]
            ax = cache["ax"]
            dw1 = ah.T @ dlogits + weight_decay * self.w1
            dh = a.T @ dlogits @ self.w1.T
            dz1 = dh * (z1 > 0.0)
            dw0 = ax.T @ dz1 + weight_decay * self.w0

            grad_norm = np.sqrt(np.sum(dw0 ** 2) + np.sum(dw1 ** 2))
            scale = min(1.0, 5.0 / max(grad_norm, 1e-12))
            self.w0 -= lr * scale * dw0
            self.w1 -= lr * scale * dw1

            probs_val, _ = self.forward(x, adj)
            val_loss = -np.mean(np.log(np.clip(probs_val[val_idx, y[val_idx]], 1e-12, 1.0)))
            history.append(float(loss))
            if val_loss + tol < best_val:
                best_val = float(val_loss)
                best = (self.w0.copy(), self.w1.copy())
                stale = 0
            else:
                stale += 1
            if prev_loss is not None and abs(prev_loss - loss) < tol and stale >= 10:
                break
            if stale >= patience:
                break
            prev_loss = float(loss)

        self.w0, self.w1 = best
        self.w0_source = self.w0.copy()
        self.w1_source = self.w1.copy()
        return {"epochs": epoch + 1, "best_val_loss": best_val, "history": history}

    def clone(self):
        other = TwoLayerGCN(self.w0.shape[0], self.w0.shape[1], self.w1.shape[1])
        other.w0 = self.w0.copy()
        other.w1 = self.w1.copy()
        other.w0_source = self.w0_source.copy()
        other.w1_source = self.w1_source.copy()
        return other
