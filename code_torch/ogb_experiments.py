"""OGB large-scale experiments: ogbn-arxiv (169K nodes) and ogbn-products (2.4M).

Mini-batch test-time adaptation with deployment metrics (per-step runtime, peak
GPU memory) on a single local GPU.  Hardware is reported honestly from
``torch.cuda.get_device_name`` -- these runs use a laptop-class RTX GPU, which
if anything strengthens the deployability story.

Protocol
--------
* Backbone: 2-layer GCN (GCNConv, hidden 256) + BatchNorm1d, trained with
  mini-batch neighbor sampling (batch 1024 seeds, fan-out [10, 10]) and Adam.
* Sampling: PyG's ``NeighborLoader`` requires the compiled ``pyg-lib`` /
  ``torch-sparse`` extensions, which provide no wheels for this Python/torch
  combination; we therefore use a dependency-free CSR-based two-hop uniform
  sampler with identical fan-out semantics (documented in the paper).
* TTA: the test stream is processed in batches of 1024 seed nodes with ONE
  gradient step per batch (Tent's online protocol), for up to
  ``adapt_batches`` batches.  Methods:
    - ``source_only``  -- frozen model.
    - ``tent``         -- the vendored official Tent (BN affine + batch stats).
    - ``full_method``  -- mini-batch variant of the reliability-aware method:
        reliability is computed on the sampled subgraph (neighborhood agreement
        over sampled neighbors; structural stability via M=2 perturbed views of
        the subgraph; confidence/margin; source-consistency against the source
        model's predictions on the same subgraph), the calibration regularizer
        monitors degree-based subgroups WITHIN the batch (the lightweight
        variant), anti-forgetting anchors the classifier to the source, and the
        detector monitors batch-level (Delta_t, Phi_t) with the auto-calibrated
        tolerance Delta* = 2 * delta_self.
* Shifts: clean, feature_noise (additive Gaussian on test-region features),
  edge_drop (random removal of a fraction of all edges).
* Metrics: test accuracy, ECE, mean per-step (per-batch) adaptation time, peak
  CUDA memory, trigger statistics.  ogbn-products evaluation uses sampled
  inference over the full test split unless ``--test-cap`` is set (recorded in
  the output).

Outputs: results_torch/ogb/ogb_results.json + .csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from _np_bridge import evaluate, expected_calibration_error  # noqa: F401  (metrics)
from third_party import tent as tent_lib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_torch" / "ogb"


# --------------------------------------------------------------------- sampler
class CSRGraph:
    """CSR adjacency over the (undirected) edge set, for uniform neighbor sampling.

    Memory-lean build: scipy's C-level COO->CSR conversion with int32 indices
    (every OGB node count fits in int32).  ``already_symmetric`` skips edge
    doubling for datasets whose ``edge_index`` already stores both directions
    (ogbn-products); doubling those 123.7M columns again was a >10 GB transient
    that OOM-killed the first ogbn-products attempt on a 16 GB machine.
    """

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, already_symmetric: bool = False):
        if already_symmetric:
            src = edge_index[0].numpy().astype(np.int32, copy=False)
            dst = edge_index[1].numpy().astype(np.int32, copy=False)
        else:
            src = torch.cat([edge_index[0], edge_index[1]]).numpy().astype(np.int32, copy=False)
            dst = torch.cat([edge_index[1], edge_index[0]]).numpy().astype(np.int32, copy=False)
        self._build(src, dst, num_nodes)

    @classmethod
    def from_arrays(cls, src: np.ndarray, dst: np.ndarray, num_nodes: int):
        obj = cls.__new__(cls)
        obj._build(src.astype(np.int32, copy=False), dst.astype(np.int32, copy=False), num_nodes)
        return obj

    def _build(self, src, dst, num_nodes):
        from scipy import sparse as sp
        coo = sp.coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)),
                            shape=(num_nodes, num_nodes))
        csr = coo.tocsr()
        del coo, src, dst
        self.indices = csr.indices.astype(np.int32, copy=False)
        self.indptr = csr.indptr.astype(np.int64, copy=False)
        self.num_nodes = num_nodes

    def sample_two_hop(self, seeds: np.ndarray, fanout=(10, 10), rng=None):
        """Uniform two-hop neighbor sampling with NeighborLoader fan-out semantics.

        Returns (subset_nodes, sub_edge_index, seed_positions): ``subset_nodes``
        are global ids (seeds first), ``sub_edge_index`` is the sampled
        message-passing graph in local ids, ``seed_positions`` = arange(len(seeds)).
        """
        rng = rng or np.random.default_rng()
        local = {int(s): i for i, s in enumerate(seeds)}
        nodes = list(map(int, seeds))
        rows, cols = [], []
        frontier = list(map(int, seeds))
        for hop_fan in fanout:
            nxt = []
            for u in frontier:
                start, end = self.indptr[u], self.indptr[u + 1]
                deg = end - start
                if deg == 0:
                    continue
                if deg <= hop_fan:
                    neigh = self.indices[start:end]
                else:
                    neigh = self.indices[start + rng.integers(0, deg, size=hop_fan)]
                for v in map(int, neigh):
                    if v not in local:
                        local[v] = len(nodes)
                        nodes.append(v)
                        nxt.append(v)
                    # message edge v -> u (neighbor feeds seed-side node)
                    rows.append(local[v])
                    cols.append(local[u])
            frontier = nxt
        sub_edge = torch.tensor(np.vstack([rows, cols]), dtype=torch.long) if rows else torch.zeros((2, 0), dtype=torch.long)
        return np.asarray(nodes, dtype=np.int64), sub_edge, np.arange(len(seeds))


# --------------------------------------------------------------------- model
class GCNBN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GCNConv(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)

    def classifier_weight(self):
        return self.conv2.lin.weight


class WrappedForTent(nn.Module):
    """model(x) interface for the official Tent; edge_index set per batch."""

    def __init__(self, gnn):
        super().__init__()
        self.model = gnn
        self.edge_index = None

    def forward(self, x):
        return self.model(x, self.edge_index)


# --------------------------------------------------------------------- helpers
def _entropy_np(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def _batch_reliability(probs, src_probs, sub_edge, n_local, seed_pos, rng):
    """Reliability on the sampled subgraph (NumPy logic, batch-local).

    probs/src_probs: [n_local, C]; agreement over sampled in-neighbors."""
    pred = probs.argmax(1)
    n_cls = probs.shape[1]
    conf = 1.0 - _entropy_np(probs) / np.log(n_cls)
    part = np.partition(probs, n_cls - 2, axis=1)
    margin = np.clip(part[:, -1] - part[:, -2], 0, 1)
    c = np.clip(0.65 * conf + 0.35 * margin, 0, 1)
    rows, cols = sub_edge[0].numpy(), sub_edge[1].numpy()
    agree_sum = np.zeros(n_local); deg = np.zeros(n_local)
    same = (pred[rows] == pred[cols]).astype(float)
    np.add.at(agree_sum, cols, same)
    np.add.at(deg, cols, 1.0)
    a = np.where(deg > 0, agree_sum / np.maximum(deg, 1), 0.0)
    homophily = float(same.mean()) if len(same) else 0.0
    # source consistency via JS divergence
    m = 0.5 * (np.clip(probs, 1e-12, 1) + np.clip(src_probs, 1e-12, 1))
    js = 0.5 * np.sum(np.clip(probs, 1e-12, 1) * np.log(np.clip(probs, 1e-12, 1) / m), axis=1) \
        + 0.5 * np.sum(np.clip(src_probs, 1e-12, 1) * np.log(np.clip(src_probs, 1e-12, 1) / m), axis=1)
    src = np.clip(1.0 - js / np.log(2.0), 0, 1)
    d = np.log1p(deg) / max(np.log1p(deg.max()), 1.0)
    agreement_weight = 0.35 + 0.65 * homophily
    base = 1.1 * c + agreement_weight * a + 0.9 * 1.0 + 0.85 * src + 0.2 * d  # stability folded below
    sel = base[seed_pos]
    thr = float(np.quantile(sel, 0.6))
    scale = float(np.std(sel) + 1e-6)
    r = 1.0 / (1.0 + np.exp(-np.clip((sel - thr) / scale, -6, 6)))
    return r, homophily


def _group_conf(probs, deg):
    conf = probs.max(1)
    q1, q2 = np.quantile(deg, 0.33), np.quantile(deg, 0.66)
    out = {}
    for name, mask in [("low", deg <= q1), ("mid", (deg > q1) & (deg <= q2)), ("high", deg > q2)]:
        out[name] = float(conf[mask].mean()) if mask.any() else 0.0
    return out


@torch.no_grad()
def sampled_inference(model, csr, x_all, y_all, node_ids, device, batch=4096, fanout=(10, 10), seed=0):
    """Sampled-subgraph inference over ``node_ids``; returns probs [len(node_ids), C]."""
    model.eval()
    rng = np.random.default_rng(seed + 999)
    outs = []
    for i in range(0, len(node_ids), batch):
        seeds = node_ids[i:i + batch]
        nodes, sub_edge, pos = csr.sample_two_hop(seeds, fanout=fanout, rng=rng)
        xb = x_all[nodes].to(device)
        logits = model(xb, sub_edge.to(device))[pos]
        outs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(outs).numpy()


# --------------------------------------------------------------------- shifts
def apply_ogb_shift(x_all, edge_index, shift, intensity, seed):
    rng = np.random.default_rng(seed + 7919)
    if shift == "clean":
        return x_all, edge_index
    if shift == "feature_noise":
        noise = torch.tensor(rng.normal(0.0, 1.0, size=tuple(x_all.shape)), dtype=x_all.dtype)
        return x_all + intensity * x_all.std() * noise, edge_index
    if shift == "edge_drop":
        E = edge_index.shape[1]
        keep = torch.tensor(rng.random(E) >= intensity)
        return x_all, edge_index[:, keep]
    raise ValueError(shift)


# --------------------------------------------------------------------- TTA
def tta_stream(method, model, csr, x_all, src_probs_fn, test_ids, device, classes,
               adapt_batches=50, batch_size=1024, fanout=(10, 10), seed=0,
               lr=0.01, lambda_cal=0.5, lambda_af=0.01, delta_self=None):
    """One pass of online mini-batch TTA; returns (adapted model, stats)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(test_ids)
    stream = order[: adapt_batches * batch_size]
    stats = {"step_times": [], "triggered": False, "trigger_step": None, "steps": 0, "delta_trace": []}

    if method == "source_only":
        return model, stats

    if method == "tent":
        wrap = WrappedForTent(model)
        tent_lib.configure_model(wrap)
        params, names = tent_lib.collect_params(wrap)
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
        stats["bn_params"] = names
        for bi in range(0, len(stream), batch_size):
            seeds = stream[bi:bi + batch_size]
            nodes, sub_edge, pos = csr.sample_two_hop(seeds, fanout=fanout, rng=rng)
            xb = x_all[nodes].to(device)
            wrap.edge_index = sub_edge.to(device)
            t0 = time.perf_counter()
            logits = wrap(xb)
            loss = tent_lib.softmax_entropy(logits[pos]).mean(0)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats["step_times"].append(time.perf_counter() - t0)
            stats["steps"] += 1
        return model, stats

    # ---- full_method (mini-batch variant) ----
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    w = model.classifier_weight()
    w.requires_grad_(True)
    w_src = w.detach().clone()
    prev_w = w.detach().clone()
    tol = (2.0 * delta_self) if delta_self else 0.05
    for step, bi in enumerate(range(0, len(stream), batch_size)):
        seeds = stream[bi:bi + batch_size]
        nodes, sub_edge, pos = csr.sample_two_hop(seeds, fanout=fanout, rng=rng)
        xb = x_all[nodes].to(device)
        eb = sub_edge.to(device)
        t0 = time.perf_counter()
        logits = model(xb, eb)
        probs_t = F.softmax(logits, dim=1)
        probs_local = probs_t.detach().cpu().numpy()
        src_local = src_probs_fn(nodes, sub_edge)            # source model on same subgraph
        deg_local = np.bincount(sub_edge[1].numpy(), minlength=len(nodes)).astype(float)

        # detector signals on the batch seeds
        gc_now = _group_conf(probs_local[pos], deg_local[pos])
        gc_src = _group_conf(src_local[pos], deg_local[pos])
        delta_t = float(np.mean([abs(gc_now[k] - gc_src[k]) for k in gc_src]))
        phi_t = float(np.mean(probs_local[pos].argmax(1) != src_local[pos].argmax(1)))
        stats["delta_trace"].append(delta_t)
        if step > 0 and (delta_t > tol or phi_t > 0.20):
            with torch.no_grad():
                w.copy_(prev_w)
            stats["triggered"] = True
            stats["trigger_step"] = step
            break
        prev_w = w.detach().clone()

        # structural stability: M=2 perturbed subgraph views
        with torch.no_grad():
            div = np.zeros(len(nodes))
            for _ in range(2):
                E = eb.shape[1]
                keep = torch.tensor(rng.random(E) >= 0.05, device=device)
                xv = xb + 0.02 * torch.randn_like(xb)
                pv = F.softmax(model(xv, eb[:, keep]), dim=1).cpu().numpy()
                m = 0.5 * (np.clip(probs_local, 1e-12, 1) + np.clip(pv, 1e-12, 1))
                js = 0.5 * np.sum(np.clip(probs_local, 1e-12, 1) * np.log(np.clip(probs_local, 1e-12, 1) / m), axis=1) \
                    + 0.5 * np.sum(np.clip(pv, 1e-12, 1) * np.log(np.clip(pv, 1e-12, 1) / m), axis=1)
                div += js
            stability = np.clip(1.0 - (div / 2) / np.log(2.0), 0, 1)

        r, _h = _batch_reliability(probs_local, src_local, sub_edge, len(nodes), pos, rng)
        r = np.clip(r * (0.5 + 0.5 * stability[pos]), 1e-3, 1.0)
        weights = np.power(r, 1.5)
        weights = weights / max(weights.sum(), 1e-12)
        wt = torch.tensor(weights, dtype=torch.float32, device=device)

        ent = -(probs_t[pos].clamp_min(1e-12).log() * probs_t[pos]).sum(1)
        obj = (wt * ent).sum() + 0.5 * lambda_af * ((w - w_src) ** 2).sum()
        if w.grad is not None:
            w.grad = None
        obj.backward()
        grad = w.grad.detach().clone()
        cal_loss = sum((gc_now[k] - gc_src[k]) ** 2 for k in gc_src)   # within-batch degree subgroups
        grad = grad + lambda_cal * cal_loss * w.detach()
        gn = float(torch.sqrt((grad ** 2).sum()))
        with torch.no_grad():
            w.add_(grad, alpha=-lr * min(1.0, 2.0 / max(gn, 1e-12)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats["step_times"].append(time.perf_counter() - t0)
        stats["steps"] += 1
    return model, stats


# ------------------------------------------------------- lean products loader
def load_products_lean(root: Path):
    """Memory-lean ogbn-products loader for small-RAM machines.

    The standard ogb/PyG processing pipeline materializes the full node-feature
    DataFrame plus a collated copy (~10 GB transient), which exceeds this
    machine's free RAM.  This loader streams the raw CSVs in chunks into a
    float32 on-disk memmap (features) and int32 arrays (edges), bounding peak
    RAM at roughly the edge arrays (~1 GB).  Outputs are identical tensors.
    """
    import pandas as pd
    raw = root / "ogbn_products" / "raw"
    split_dir = root / "ogbn_products" / "split" / "sales_ranking"
    cache = root / "ogbn_products" / "lean_cache"
    cache.mkdir(parents=True, exist_ok=True)

    n_nodes = int(pd.read_csv(raw / "num-node-list.csv.gz", header=None).iloc[0, 0])
    n_edges = int(pd.read_csv(raw / "num-edge-list.csv.gz", header=None).iloc[0, 0])

    feat_path = cache / "node_feat_f32.npy"
    if not feat_path.exists():
        print(f"[lean] streaming node features into memmap ({n_nodes} x 100)...")
        first = pd.read_csv(raw / "node-feat.csv.gz", header=None, nrows=1)
        dim = first.shape[1]
        mm = np.lib.format.open_memmap(feat_path, mode="w+", dtype=np.float32, shape=(n_nodes, dim))
        row = 0
        for chunk in pd.read_csv(raw / "node-feat.csv.gz", header=None, dtype=np.float32, chunksize=100_000):
            mm[row:row + len(chunk)] = chunk.to_numpy(dtype=np.float32, copy=False)
            row += len(chunk)
        mm.flush(); del mm
        print(f"[lean] features done ({row} rows)")
    x_np = np.load(feat_path, mmap_mode="r")

    edge_path = cache / "edges_i32.npy"
    if not edge_path.exists():
        print(f"[lean] streaming edges ({n_edges})...")
        em = np.lib.format.open_memmap(edge_path, mode="w+", dtype=np.int32, shape=(n_edges, 2))
        row = 0
        for chunk in pd.read_csv(raw / "edge.csv.gz", header=None, dtype=np.int32, chunksize=2_000_000):
            em[row:row + len(chunk)] = chunk.to_numpy(dtype=np.int32, copy=False)
            row += len(chunk)
        em.flush(); del em
        print(f"[lean] edges done ({row} rows)")
    edges = np.load(edge_path, mmap_mode="r")

    y = pd.read_csv(raw / "node-label.csv.gz", header=None, dtype=np.int64).to_numpy().reshape(-1)
    split = {k: pd.read_csv(split_dir / f"{k}.csv.gz", header=None, dtype=np.int64).to_numpy().reshape(-1)
             for k in ("train", "valid", "test")}
    return x_np, edges, y, split, n_nodes


# --------------------------------------------------------------------- driver
def run_dataset(name, device, seeds=(0,), adapt_batches=50, test_cap=None,
                conditions=(("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.30)),
                hidden=256, train_epochs=10, train_batches_per_epoch=None):
    # ogb 1.3.6 caches processed data with full pickles; torch>=2.6 defaults
    # torch.load to weights_only=True, so allowlist the PyG container classes.
    import torch_geometric.data.data as pyg_data_mod
    from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage
    safe = [BaseStorage, NodeStorage, EdgeStorage, GlobalStorage]
    for cls_name in ("DataEdgeAttr", "DataTensorAttr"):
        if hasattr(pyg_data_mod, cls_name):
            safe.append(getattr(pyg_data_mod, cls_name))
    torch.serialization.add_safe_globals(safe)
    if name == "ogbn-products":
        # Memory-lean path: the standard ogb processing pipeline needs ~10 GB
        # transient RAM (it OOM-killed twice on this 15 GB machine); stream the
        # raw CSVs instead.  edge_drop is unsupported on this path (it would
        # materialize the full edge tensor); use clean / feature_noise.
        x_np, edges_mm, y_all, split_np, num_nodes = load_products_lean(ROOT / "data" / "ogb")
        # Keep features memmap-backed (OS page cache serves batch gathers); a
        # writable mmap lets torch share the buffer without a 1 GB RAM copy.
        x_all = torch.from_numpy(np.load(ROOT / "data" / "ogb" / "ogbn_products" / "lean_cache" / "node_feat_f32.npy",
                                         mmap_mode="r+"))
        classes = int(y_all.max() + 1)
        # raw edge.csv stores each undirected edge ONCE; symmetrize for sampling.
        # int32 concat = 2 x 0.47 GB transient, freed after the CSR is built.
        src_sym = np.concatenate([edges_mm[:, 0], edges_mm[:, 1]])
        dst_sym = np.concatenate([edges_mm[:, 1], edges_mm[:, 0]])
        csr_clean = CSRGraph.from_arrays(src_sym, dst_sym, num_nodes)
        del src_sym, dst_sym
        edge_index_full = None
        num_edges = edges_mm.shape[0]
        train_ids = split_np["train"]
        val_ids = split_np["valid"]
        test_ids = split_np["test"] if test_cap is None else split_np["test"][:test_cap]
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        # Non-interactive runs cannot answer ogb's interactive prompts (download
        # size confirmation; dataset-update question); auto-confirm both.
        import builtins
        import ogb.nodeproppred.dataset_pyg as _ogb_pyg
        _ogb_pyg.decide_download = lambda url: True
        _orig_input = builtins.input
        builtins.input = lambda *a, **k: "y"
        try:
            ds = PygNodePropPredDataset(name=name, root=str(ROOT / "data" / "ogb"))
        finally:
            builtins.input = _orig_input
        data = ds[0]
        split = ds.get_idx_split()
        x_all = data.x  # CPU float32
        y_all = data.y.view(-1).numpy()
        classes = int(y_all.max() + 1)
        csr_clean = CSRGraph(data.edge_index, data.num_nodes, already_symmetric=False)
        edge_index_full = data.edge_index
        num_nodes = data.num_nodes
        num_edges = data.edge_index.shape[1]
        train_ids = split["train"].numpy()
        val_ids = split["valid"].numpy()
        test_ids = split["test"].numpy() if test_cap is None else split["test"].numpy()[:test_cap]
    records = []
    print(f"[ogb] {name}: {num_nodes} nodes, {num_edges} edges, "
          f"{classes} classes, test={len(test_ids)}{' (capped)' if test_cap else ''}, device={device}")

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = GCNBN(x_all.shape[1], hidden, classes, seed=seed).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        rng = np.random.default_rng(seed)
        n_train_batches = train_batches_per_epoch or max(1, len(train_ids) // 1024)
        t_train0 = time.perf_counter()
        for ep in range(train_epochs):
            model.train()
            order = rng.permutation(train_ids)
            for bi in range(n_train_batches):
                seeds_b = order[bi * 1024:(bi + 1) * 1024]
                if len(seeds_b) == 0:
                    break
                nodes, sub_edge, pos = csr_clean.sample_two_hop(seeds_b, rng=rng)
                xb = x_all[nodes].to(device)
                opt.zero_grad()
                logits = model(xb, sub_edge.to(device))[pos]
                loss = F.cross_entropy(logits, torch.tensor(y_all[seeds_b], device=device))
                loss.backward()
                opt.step()
            vp = sampled_inference(model, csr_clean, x_all, y_all, val_ids[:10000], device, seed=seed)
            va = float((vp.argmax(1) == y_all[val_ids[:10000]]).mean())
            print(f"[ogb] {name} seed={seed} epoch={ep} val_acc={va:.4f}")
        train_time = time.perf_counter() - t_train0

        # delta_self: drift of one clean adaptation pass on a small clean stream
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        src_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def make_src_fn(csr_t, x_t):
            @torch.no_grad()
            def src_probs_fn(nodes, sub_edge):
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(src_state)
                model.eval()
                out = F.softmax(model(x_t[nodes].to(device), sub_edge.to(device)), dim=1).cpu().numpy()
                model.load_state_dict(cur)
                return out
            return src_probs_fn

        # measure delta_self on the clean graph (5 batches of the validation stream,
        # updates discarded): the label-free no-shift drift floor.
        m0 = GCNBN(x_all.shape[1], hidden, classes, seed=seed).to(device)
        m0.load_state_dict(src_state)
        _, st0 = tta_stream("full_method", m0, csr_clean, x_all, make_src_fn(csr_clean, x_all),
                            val_ids, device, classes, adapt_batches=5, seed=seed, delta_self=1e9)
        observed = st0.get("delta_trace") or []
        delta_self = max(float(np.mean(observed)) if observed else 1e-4, 1e-4)
        print(f"[ogb] {name} seed={seed} delta_self={delta_self:.5f} (over {len(observed)} clean batches)")

        for shift, intensity in conditions:
            if shift == "edge_drop" and edge_index_full is None:
                print(f"[ogb] {name}: edge_drop unsupported on the lean loader path; skipping")
                continue
            x_t, ei_t = apply_ogb_shift(x_all, edge_index_full, shift, intensity, seed)
            csr_t = CSRGraph(ei_t, num_nodes) if shift == "edge_drop" else csr_clean
            for method in ["source_only", "tent", "full_method"]:
                m = GCNBN(x_all.shape[1], hidden, classes, seed=seed).to(device)
                m.load_state_dict(src_state)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                m, stats = tta_stream(method, m, csr_t, x_t, make_src_fn(csr_t, x_t), test_ids,
                                      device, classes, adapt_batches=adapt_batches, seed=seed,
                                      delta_self=delta_self)
                probs = sampled_inference(m, csr_t, x_t, y_all, test_ids, device, seed=seed)
                met = evaluate(probs, y_all[test_ids], classes)
                peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
                rec = {
                    "dataset": name, "seed": seed, "shift": shift, "intensity": intensity,
                    "method": method, "accuracy": met["accuracy"], "ece": met["ece"],
                    "macro_f1": met["macro_f1"], "nll": met["nll"],
                    "mean_step_seconds": float(np.mean(stats["step_times"])) if stats["step_times"] else 0.0,
                    "adapt_steps": stats["steps"], "peak_gpu_gb": round(peak_gb, 3),
                    "triggered": stats.get("triggered", False), "trigger_step": stats.get("trigger_step"),
                    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "train_seconds": round(train_time, 1), "test_nodes": len(test_ids),
                }
                records.append(rec)
                print(f"[ogb] {name} seed={seed} {shift}/{intensity} {method}: acc={met['accuracy']:.4f} "
                      f"ece={met['ece']:.4f} step={rec['mean_step_seconds']:.3f}s peak={peak_gb:.2f}GB "
                      f"trig={stats.get('triggered')}")
            _write(records)
    return records


def _write(records):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ogb_results.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    if records:
        with (OUT / "ogb_results.csv").open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
            wcsv.writeheader()
            wcsv.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbn-arxiv", choices=["ogbn-arxiv", "ogbn-products"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--adapt-batches", type=int, default=50)
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--train-batches-per-epoch", type=int, default=None)
    ap.add_argument("--conditions", default="clean:0,feature_noise:0.10,edge_drop:0.30",
                    help="comma-separated shift:intensity pairs")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    conditions = tuple((p.split(":")[0], float(p.split(":")[1])) for p in args.conditions.split(","))
    run_dataset(args.dataset, device, seeds=seeds, adapt_batches=args.adapt_batches,
                test_cap=args.test_cap, train_epochs=args.train_epochs,
                train_batches_per_epoch=args.train_batches_per_epoch, conditions=conditions)


if __name__ == "__main__":
    main()
