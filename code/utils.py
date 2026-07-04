import numpy as np


try:
    from scipy import sparse
except ImportError:
    sparse = None


def is_sparse_matrix(matrix):
    return sparse is not None and sparse.issparse(matrix)


def degree_vector(adj):
    if is_sparse_matrix(adj):
        return np.asarray(adj.sum(axis=1)).reshape(-1).astype(float)
    return np.sum(adj, axis=1).astype(float)


def upper_triangle_edges(adj):
    if is_sparse_matrix(adj):
        tri = sparse.triu(adj, k=1).tocoo()
        if tri.nnz == 0:
            return np.zeros((0, 2), dtype=int)
        return np.column_stack([tri.row, tri.col]).astype(int)
    edges = np.transpose(np.triu(np.asarray(adj), 1).nonzero())
    return edges.astype(int)


def rebuild_adjacency(num_nodes, edges, use_sparse=False):
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.size == 0:
        if use_sparse and sparse is not None:
            return sparse.csr_matrix((num_nodes, num_nodes), dtype=float)
        return np.zeros((num_nodes, num_nodes), dtype=float)
    if edge_array.ndim == 1:
        edge_array = edge_array.reshape(1, 2)
    row = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
    col = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    data = np.ones(len(row), dtype=float)
    if use_sparse and sparse is not None:
        return sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = np.zeros((num_nodes, num_nodes), dtype=float)
    adj[row, col] = 1.0
    return adj


def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def normalize_adjacency(adj):
    if is_sparse_matrix(adj):
        n = adj.shape[0]
        a = adj.tocsr() + sparse.identity(n, format="csr", dtype=float)
        deg = degree_vector(a)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        d = sparse.diags(inv_sqrt)
        return (d @ a @ d).tocsr()
    n = adj.shape[0]
    a = adj + np.eye(n)
    deg = np.sum(a, axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, num_classes):
    scores = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(float(2.0 * precision * recall / (precision + recall)))
    return float(np.mean(scores))


def nll(probs, y_true):
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs, y_true, num_classes):
    target = one_hot(y_true, num_classes)
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def expected_calibration_error(probs, y_true, bins=15):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(conf[mask]))
        avg_acc = float(np.mean(correct[mask]))
        ece += (count / len(y_true)) * abs(avg_conf - avg_acc)
    return float(ece)


def evaluate(probs, y_true, num_classes):
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred, num_classes),
        "nll": nll(probs, y_true),
        "ece": expected_calibration_error(probs, y_true),
        "brier": brier_score(probs, y_true, num_classes),
    }
