import gzip
import pickle
import ssl
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from utils import degree_vector, is_sparse_matrix, rebuild_adjacency, upper_triangle_edges


try:
    from numpy.exceptions import VisibleDeprecationWarning
except Exception:
    VisibleDeprecationWarning = DeprecationWarning

try:
    from scipy import sparse
except ImportError:
    sparse = None


def make_contextual_sbm(seed, n=360, classes=3, feature_dim=24, homophily=0.78, p_in=0.075, p_out=0.012, feature_noise=0.55):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % classes
    rng.shuffle(y)

    centers = rng.normal(0.0, 1.0, size=(classes, feature_dim))
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    x = centers[y] + feature_noise * rng.normal(0.0, 1.0, size=(n, feature_dim))

    same = y[:, None] == y[None, :]
    probs = np.where(same, p_in * homophily / 0.78, p_out * (1.0 + (0.78 - homophily)))
    probs = np.clip(probs, 0.001, 0.25)
    upper = rng.random((n, n)) < probs
    upper = np.triu(upper, 1)
    adj = upper + upper.T
    adj = adj.astype(float)
    return x.astype(float), adj, y.astype(int)



def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    context = ssl.create_default_context()
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _read_planetoid_pickle(path):
    # Older Planetoid pickles can trigger a NumPy 2.4 deprecation warning while
    # unpickling scipy sparse dtypes. The loaded arrays are still valid; suppress
    # this noisy compatibility warning so experiment logs remain readable.
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
            return pickle.load(f, encoding="latin1")


def _to_numpy_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(float)
    return np.asarray(matrix, dtype=float)


def _labels_from_onehot(labels):
    arr = np.asarray(labels)
    if arr.ndim == 1:
        return arr.astype(int)
    return np.argmax(arr, axis=1).astype(int)


def _use_sparse_backend(graph_backend, num_nodes):
    if graph_backend == "sparse":
        if sparse is None:
            raise RuntimeError("Sparse graph backend requires scipy.")
        return True
    if graph_backend == "auto":
        return sparse is not None and num_nodes >= 1800
    return False


def _sample_non_edges(rng, num_nodes, existing_edges, count):
    if count <= 0:
        return np.zeros((0, 2), dtype=int)
    sampled = set()
    budget = max(200, count * 40)
    while len(sampled) < count and budget > 0:
        i = int(rng.integers(0, num_nodes))
        j = int(rng.integers(0, num_nodes - 1))
        if j >= i:
            j += 1
        edge = (i, j) if i < j else (j, i)
        if edge not in existing_edges and edge not in sampled:
            sampled.add(edge)
        budget -= 1
    if len(sampled) < count:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = (i, j)
                if edge not in existing_edges and edge not in sampled:
                    sampled.add(edge)
                    if len(sampled) >= count:
                        break
            if len(sampled) >= count:
                break
    return np.array(sorted(sampled), dtype=int) if sampled else np.zeros((0, 2), dtype=int)


def load_planetoid_dataset(name, root=None, max_nodes=None, seed=0, graph_backend="dense"):
    """Load Cora, Citeseer, or Pubmed from the public Planetoid files.

    This loader is intentionally dependency-light: it uses urllib, pickle, gzip,
    and NumPy only. It downloads the canonical files used by many GCN baselines
    and returns dense NumPy arrays suitable for the small full-batch runner.
    When ``max_nodes`` is provided, it builds a deterministic induced subgraph
    before dense adjacency materialization, which keeps Pubmed usable in the
    NumPy-only implementation without silently substituting synthetic data.
    """
    name = name.lower()
    aliases = {"cora": "cora", "citeseer": "citeseer", "pubmed": "pubmed"}
    if name not in aliases:
        raise ValueError("Planetoid loader supports only cora, citeseer, and pubmed")
    dataset = aliases[name]
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "public"
    base_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    keys = ["x", "tx", "allx", "y", "ty", "ally", "graph"]
    objects = {}
    for key in keys:
        file_name = f"ind.{dataset}.{key}"
        path = _download(f"{base_url}/{file_name}", root / dataset / file_name)
        objects[key] = _read_planetoid_pickle(path)

    test_index_path = _download(f"{base_url}/ind.{dataset}.test.index", root / dataset / f"ind.{dataset}.test.index")
    test_idx = np.array([int(line.strip()) for line in test_index_path.read_text(encoding="utf-8").splitlines() if line.strip()], dtype=int)
    sorted_test_idx = np.sort(test_idx)

    x = _to_numpy_dense(objects["x"])
    tx = _to_numpy_dense(objects["tx"])
    allx = _to_numpy_dense(objects["allx"])
    y_mat = _to_numpy_dense(objects["y"])
    ty_mat = _to_numpy_dense(objects["ty"])
    ally_mat = _to_numpy_dense(objects["ally"])
    y = _labels_from_onehot(y_mat)
    ty = _labels_from_onehot(ty_mat)
    ally = _labels_from_onehot(ally_mat)

    if dataset == "citeseer":
        # Citeseer has isolated test nodes. The canonical Planetoid preprocessing
        # inserts zero feature/label rows so test indices remain aligned. Keep the
        # one-hot label matrix during extension; converting to class ids before
        # extension would turn missing rows into class 0 and corrupt evaluation.
        full_range = np.arange(test_idx.min(), test_idx.max() + 1)
        tx_ext = np.zeros((len(full_range), tx.shape[1]), dtype=float)
        ty_ext = np.zeros((len(full_range), ty_mat.shape[1]), dtype=float)
        pos = sorted_test_idx - test_idx.min()
        tx_ext[pos] = tx
        ty_ext[pos] = ty_mat
        tx = tx_ext
        ty_mat = ty_ext
        ty = _labels_from_onehot(ty_mat)
        sorted_test_idx = np.sort(test_idx)

    features = np.vstack([allx, tx])
    labels = np.concatenate([ally, ty])
    features[test_idx, :] = features[sorted_test_idx, :]
    labels[test_idx] = labels[sorted_test_idx]

    graph = objects["graph"]
    n = features.shape[0]
    original_train_idx = np.arange(len(y), dtype=int)

    if max_nodes is not None and n > max_nodes:
        rng = np.random.default_rng(seed + 4243)
        required = np.unique(np.concatenate([original_train_idx, test_idx]))
        budget = max(int(max_nodes), len(required) + 1)
        optional = np.setdiff1d(np.arange(n, dtype=int), required, assume_unique=False)
        optional_count = max(0, min(len(optional), budget - len(required)))
        sampled_optional = rng.choice(optional, size=optional_count, replace=False) if optional_count > 0 else np.array([], dtype=int)
        keep = np.sort(np.concatenate([required, sampled_optional])).astype(int)
        remap = {int(old): new for new, old in enumerate(keep)}
        features = features[keep]
        labels = labels[keep]
        graph = {remap[int(i)]: [remap[int(j)] for j in neigh if int(j) in remap] for i, neigh in graph.items() if int(i) in remap}
        original_train_idx = np.array([remap[int(i)] for i in original_train_idx if int(i) in remap], dtype=int)
        test_idx = np.array([remap[int(i)] for i in test_idx if int(i) in remap], dtype=int)
        n = features.shape[0]

    edge_pairs = set()
    for i, neigh in graph.items():
        if i < n:
            for j in neigh:
                if j < n and i != j:
                    edge_pairs.add((min(int(i), int(j)), max(int(i), int(j))))
    edge_array = np.array(sorted(edge_pairs), dtype=int) if edge_pairs else np.zeros((0, 2), dtype=int)
    adj = rebuild_adjacency(n, edge_array, use_sparse=_use_sparse_backend(graph_backend, n))

    train_idx = original_train_idx
    val_candidates = np.setdiff1d(np.arange(n, dtype=int), np.concatenate([train_idx, test_idx]), assume_unique=False)
    val_idx = val_candidates[:min(500, len(val_candidates))]
    if len(val_idx) == 0:
        train_idx, val_idx, test_idx = split_indices(seed, labels)

    row_sum = np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    features = features / row_sum
    return features.astype(float), adj, labels.astype(int), train_idx, val_idx, test_idx


def load_pyg_npz_dataset(name, root=None, max_nodes=None, seed=0, graph_backend="dense"):
    """Load Amazon/Coauthor datasets from PyG-style NPZ mirrors when available.

    Supported names are amazon_computers, amazon_photo, coauthor_cs, and
    coauthor_physics. These datasets are optional because mirrors can change;
    failed downloads raise a clear error rather than silently fabricating data.
    """
    name = name.lower().replace("-", "_")
    urls = {
        "amazon_computers": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz",
        "amazon_photo": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz",
        "coauthor_cs": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz",
        "coauthor_physics": "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_phy.npz",
    }
    if name not in urls:
        raise ValueError(f"Unsupported public NPZ dataset: {name}")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "public"
    path = _download(urls[name], root / f"{name}.npz")
    data = np.load(path, allow_pickle=True)
    required = {"adj_data", "adj_indices", "adj_indptr", "adj_shape", "attr_data", "attr_indices", "attr_indptr", "attr_shape", "labels"}
    if not required.issubset(set(data.files)):
        raise RuntimeError(f"Unexpected NPZ format for {name}; available keys: {sorted(data.files)}")
    if sparse is None:
        raise RuntimeError("Loading Amazon/Coauthor NPZ files requires scipy. Install scipy or use Planetoid datasets only.")
    adj = sparse.csr_matrix((data["adj_data"], data["adj_indices"], data["adj_indptr"]), shape=data["adj_shape"])
    features = sparse.csr_matrix((data["attr_data"], data["attr_indices"], data["attr_indptr"]), shape=data["attr_shape"])
    labels = np.asarray(data["labels"], dtype=int)
    if max_nodes is not None and labels.shape[0] > max_nodes:
        rng = np.random.default_rng(seed + 8081)
        train_idx, val_idx, test_idx = split_indices(seed, labels, train_per_class=20, val_per_class=30)
        required = np.unique(np.concatenate([train_idx, val_idx, test_idx[:min(len(test_idx), max(0, int(max_nodes) - len(train_idx) - len(val_idx)))]]))
        budget = max(int(max_nodes), len(required))
        optional = np.setdiff1d(np.arange(labels.shape[0], dtype=int), required, assume_unique=False)
        optional_count = max(0, min(len(optional), budget - len(required)))
        sampled_optional = rng.choice(optional, size=optional_count, replace=False) if optional_count > 0 else np.array([], dtype=int)
        keep = np.sort(np.concatenate([required, sampled_optional])).astype(int)
        adj = adj[keep][:, keep]
        features = features[keep]
        labels = labels[keep]
    if _use_sparse_backend(graph_backend, labels.shape[0]):
        adj = adj.tocsr().astype(float)
    else:
        adj = adj.toarray().astype(float)
    features = features.toarray().astype(float)
    row_sum = np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    features = features / row_sum
    train_idx, val_idx, test_idx = split_indices(seed, labels, train_per_class=20, val_per_class=30)
    return features, adj, labels, train_idx, val_idx, test_idx


def make_arxiv_subset(seed, n=2000, classes=10, feature_dim=128, homophily=0.72, avg_degree=8):
    """Generate a citation-like graph mimicking ogbn-arxiv subset properties.

    This produces a graph with ogbn-arxiv-like statistics (high homophily, many
    classes, high-dimensional features, moderate average degree) at controllable
    scale for scalability experiments. It is NOT the official ogbn-arxiv dataset;
    results are reported as scalability stress tests on arxiv-like synthetic graphs.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=n)

    centers = rng.normal(0.0, 1.0, size=(classes, feature_dim))
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    x = centers[y] + 0.6 * rng.normal(0.0, 1.0, size=(n, feature_dim))

    target_edges = int(n * avg_degree / 2)
    p_same = homophily
    p_diff = (1.0 - homophily)
    edges = set()
    while len(edges) < target_edges:
        batch = min(target_edges - len(edges), 5000)
        ii = rng.integers(0, n, size=batch)
        jj = rng.integers(0, n - 1, size=batch)
        jj[jj >= ii] += 1
        for a, b in zip(ii, jj):
            if a > b:
                a, b = b, a
            same_class = (y[a] == y[b])
            if same_class and rng.random() < p_same:
                edges.add((int(a), int(b)))
            elif not same_class and rng.random() < p_diff * 0.3:
                edges.add((int(a), int(b)))
    edges_arr = np.array(sorted(edges), dtype=int) if edges else np.zeros((0, 2), dtype=int)
    adj = np.zeros((n, n), dtype=float)
    if len(edges_arr) > 0:
        adj[edges_arr[:, 0], edges_arr[:, 1]] = 1.0
        adj[edges_arr[:, 1], edges_arr[:, 0]] = 1.0
    train_idx, val_idx, test_idx = split_indices(seed, y, train_per_class=20, val_per_class=30)
    return x.astype(float), adj, y.astype(int), train_idx, val_idx, test_idx


def load_public_graph_dataset(name, root=None, max_nodes=None, seed=0, graph_backend="dense"):
    name_norm = name.lower().replace("-", "_")
    if name_norm in {"cora", "citeseer", "pubmed"}:
        return load_planetoid_dataset(name_norm, root=root, max_nodes=max_nodes, seed=seed, graph_backend=graph_backend)
    if name_norm in {"amazon_computers", "amazon_photo", "coauthor_cs", "coauthor_physics"}:
        return load_pyg_npz_dataset(name_norm, root=root, max_nodes=max_nodes, seed=seed, graph_backend=graph_backend)
    if name_norm in {"ogbn_arxiv", "ogbn-arxiv"}:
        return load_ogbn_arxiv_placeholder(root=root)
    raise ValueError(f"Unknown public graph dataset: {name}")


def make_heterophily_benchmark(seed, name="texas", feature_dim=32):
    """Create a lightweight heterophilous graph analogue for Texas/Cornell/Wisconsin.

    The real WebKB datasets require an additional external data source. This
    dependency-light generator preserves the key experimental stressor needed in
    this project--low homophily with class-correlated features--so the runner can
    test whether a method over-relies on neighborhood label agreement. Results are
    reported as heterophily stress tests rather than as official WebKB numbers.
    """
    name = name.lower()
    configs = {
        "texas": (183, 5, 0.18, 0.018, 0.070, 0.62),
        "cornell": (183, 5, 0.20, 0.020, 0.068, 0.60),
        "wisconsin": (251, 5, 0.22, 0.018, 0.064, 0.58),
        "actor": (300, 5, 0.38, 0.022, 0.055, 0.58),
        "film": (320, 5, 0.45, 0.024, 0.050, 0.55),
    }
    if name not in configs:
        raise ValueError(f"Unknown heterophily benchmark: {name}")
    n, classes, homophily, p_in, p_out, noise = configs[name]
    return make_contextual_sbm(seed=seed, n=n, classes=classes, feature_dim=feature_dim, homophily=homophily, p_in=p_in, p_out=p_out, feature_noise=noise)


def split_indices(seed, y, train_per_class=20, val_per_class=30):
    rng = np.random.default_rng(seed + 1009)
    train, val, test = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        class_count = len(idx)
        train_count = min(train_per_class, max(3, class_count // 4))
        val_count = min(val_per_class, max(3, class_count // 4))
        if train_count + val_count >= class_count:
            train_count = max(2, class_count // 3)
            val_count = max(2, class_count // 3)
        train.extend(idx[:train_count])
        val.extend(idx[train_count:train_count + val_count])
        test.extend(idx[train_count + val_count:])
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def apply_shift(seed, x, adj, y, shift, intensity):
    rng = np.random.default_rng(seed + 7919)
    x2 = x.copy()
    adj2 = adj.copy()
    n = len(y)
    use_sparse = is_sparse_matrix(adj2)

    if shift == "clean":
        return x2, adj2

    if shift == "feature_noise":
        x2 = x2 + intensity * rng.normal(0.0, 1.0, size=x2.shape)
        return x2, adj2

    if shift == "edge_drop":
        edges = upper_triangle_edges(adj2)
        drop_count = int(len(edges) * intensity)
        if drop_count <= 0:
            return x2, adj2
        keep_mask = np.ones(len(edges), dtype=bool)
        chosen = rng.choice(len(edges), size=drop_count, replace=False)
        keep_mask[chosen] = False
        return x2, rebuild_adjacency(n, edges[keep_mask], use_sparse=use_sparse)

    if shift == "edge_add":
        edges = upper_triangle_edges(adj2)
        edge_set = {tuple(map(int, edge)) for edge in edges.tolist()} if len(edges) > 0 else set()
        max_missing = (n * (n - 1) // 2) - len(edge_set)
        add_count = min(int(len(edges) * intensity), max_missing)
        additions = _sample_non_edges(rng, n, edge_set, add_count)
        if len(additions) == 0:
            return x2, adj2
        merged = np.vstack([edges, additions]) if len(edges) > 0 else additions
        return x2, rebuild_adjacency(n, merged, use_sparse=use_sparse)

    if shift == "degree_shift":
        deg = degree_vector(adj2)
        threshold = np.quantile(deg, 0.35 + 0.3 * min(intensity, 1.0))
        mask = np.ones(n, dtype=bool)
        remove_candidates = np.where(deg > threshold)[0]
        remove_count = int(len(remove_candidates) * min(0.45, intensity))
        if remove_count > 0:
            remove = rng.choice(remove_candidates, size=remove_count, replace=False)
            mask[remove] = False
        keep = np.where(mask)[0]
        if use_sparse:
            return x2[keep], adj2[keep][:, keep]
        return x2[keep], adj2[np.ix_(keep, keep)]

    if shift == "homophily_shift":
        edges = upper_triangle_edges(adj2)
        same_edges = [(int(i), int(j)) for i, j in edges if y[int(i)] == y[int(j)]]
        rewire_count = int(len(same_edges) * min(0.7, intensity))
        if rewire_count > 0:
            edge_set = {tuple(map(int, edge)) for edge in edges.tolist()}
            chosen_ids = rng.choice(len(same_edges), size=rewire_count, replace=False)
            for idx in chosen_ids:
                i, j = same_edges[idx]
                edge_set.discard((min(i, j), max(i, j)))
                candidates = np.where(y != y[i])[0]
                for _ in range(max(20, len(candidates))):
                    k = int(rng.choice(candidates))
                    if k == i:
                        continue
                    edge = (i, k) if i < k else (k, i)
                    if edge not in edge_set:
                        edge_set.add(edge)
                        break
            rewired = np.array(sorted(edge_set), dtype=int) if edge_set else np.zeros((0, 2), dtype=int)
            return x2, rebuild_adjacency(n, rewired, use_sparse=use_sparse)
        return x2, adj2

    raise ValueError(f"Unknown shift: {shift}")
