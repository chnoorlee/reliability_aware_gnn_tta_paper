"""Reliability estimator — verbatim port of the scoring logic from
``code/adaptation.py`` (req#6: algorithm unchanged).

These functions operate entirely in NumPy-adjacency / probability space, exactly
as in the original paper.  The *only* change versus ``code/adaptation.py`` is
that the structural-stability and reliability routines take a ``predict_fn``
closure ``(x_np, adj_np) -> probs_np`` instead of a model object, so the same
NumPy math can drive a PyTorch Geometric forward pass.  The numerical logic
(weights, thresholds, quantiles, mixing coefficients) is identical.

Source of truth: ``code/adaptation.py``.  Keep the two in sync.
"""

from __future__ import annotations

import numpy as np

from _np_bridge import degree_vector, is_sparse_matrix, rebuild_adjacency, upper_triangle_edges


def entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def js_divergence(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), axis=1) + 0.5 * np.sum(q * np.log(q / m), axis=1)


def top_margin(probs):
    if probs.shape[1] == 1:
        return np.ones(probs.shape[0], dtype=float)
    part = np.partition(probs, kth=probs.shape[1] - 2, axis=1)
    return np.clip(part[:, -1] - part[:, -2], 0.0, 1.0)


def neighborhood_agreement(adj, probs, homophily_override=None):
    """Adaptive neighborhood-agreement signal.

    ``homophily_override`` is used only by the homophily-misestimation robustness
    study: when supplied, the hard/soft mixing coefficient is driven by the given
    (deliberately perturbed) homophily estimate instead of the measured one, and
    that same value is returned so the caller can also drive the agreement weight
    with it.  ``homophily_override=None`` reproduces the original behaviour
    byte-for-byte.
    """
    pred = np.argmax(probs, axis=1)
    n = len(pred)
    out = np.zeros(n, dtype=float)
    deg = degree_vector(adj)
    if is_sparse_matrix(adj):
        csr = adj.tocsr()
        for i in range(n):
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            if start == end:
                out[i] = 0.0
                continue
            neigh = csr.indices[start:end]
            out[i] = np.mean(pred[neigh] == pred[i])
    else:
        for i in range(n):
            if deg[i] <= 0:
                out[i] = 0.0
            else:
                neigh = np.where(adj[i] > 0)[0]
                out[i] = np.mean(pred[neigh] == pred[i])
    if np.max(deg) <= 0:
        return out, 0.0
    neigh_probs = adj @ probs
    neigh_mean = np.where(deg[:, None] > 0, neigh_probs / np.maximum(deg[:, None], 1.0), probs)
    soft = np.clip(1.0 - js_divergence(probs, neigh_mean) / np.log(2.0), 0.0, 1.0)
    edges = upper_triangle_edges(adj)
    measured = float(np.mean(pred[edges[:, 0]] == pred[edges[:, 1]])) if len(edges) > 0 else 0.0
    homophily = measured if homophily_override is None else float(np.clip(homophily_override, 0.0, 1.0))
    mix = 0.25 + 0.5 * homophily
    return np.clip(mix * out + (1.0 - mix) * soft, 0.0, 1.0), homophily


def structural_stability(predict_fn, x, adj, seed=0, views=2, drop_rate=0.05, feature_noise=0.02):
    rng = np.random.default_rng(seed + 2027)
    base_probs = predict_fn(x, adj)
    divergences = []
    edges = upper_triangle_edges(adj)
    use_sparse = is_sparse_matrix(adj)
    for _ in range(views):
        keep_edges = edges
        drop_count = int(len(edges) * drop_rate)
        if drop_count > 0:
            keep_mask = np.ones(len(edges), dtype=bool)
            chosen = rng.choice(len(edges), size=drop_count, replace=False)
            keep_mask[chosen] = False
            keep_edges = edges[keep_mask]
        pert = rebuild_adjacency(adj.shape[0], keep_edges, use_sparse=use_sparse)
        x_view = x + feature_noise * rng.normal(0.0, 1.0, size=x.shape)
        probs = predict_fn(x_view, pert)
        divergences.append(js_divergence(base_probs, probs))
    if not divergences:
        return np.ones(adj.shape[0], dtype=float)
    div = np.mean(np.vstack(divergences), axis=0)
    return np.clip(1.0 - div / np.log(2.0), 0.0, 1.0)


def source_consistency(probs, reference_probs):
    if reference_probs is None:
        return np.ones(probs.shape[0], dtype=float)
    return np.clip(1.0 - js_divergence(probs, reference_probs) / np.log(2.0), 0.0, 1.0)


def reliability_scores(predict_fn, x, adj, seed=0, use_agreement=True, use_stability=True,
                       reference_probs=None, use_confidence=True, use_source=True,
                       use_degree=True, homophily_delta=0.0):
    """Structure-conditioned reliability score.

    The ``use_*`` flags drive the per-signal leave-one-out ablation (each disabled
    signal is set to a neutral constant *and* its composite weight is zeroed, the
    same convention already used for ``use_agreement`` / ``use_stability``).
    ``homophily_delta`` adds a deliberate error to the internal homophily estimate
    for the misestimation-robustness study.  All defaults reproduce the original
    estimator exactly.
    """
    probs = predict_fn(x, adj)
    entropy_conf = 1.0 - entropy(probs) / np.log(probs.shape[1])
    margin = top_margin(probs)
    c = np.clip(0.65 * entropy_conf + 0.35 * margin, 0.0, 1.0)
    if use_agreement:
        a, homophily = neighborhood_agreement(adj, probs)
        if homophily_delta != 0.0:
            # Robustness probe: drive both the hard/soft mixing and the agreement
            # weight with the misestimated homophily \hat h_G + delta.
            h_used = float(np.clip(homophily + homophily_delta, 0.0, 1.0))
            a, homophily = neighborhood_agreement(adj, probs, homophily_override=h_used)
    else:
        a, homophily = np.ones(len(c)), 0.0
    s = structural_stability(predict_fn, x, adj, seed=seed) if use_stability else np.ones(len(c))
    src = source_consistency(probs, reference_probs) if use_source else np.ones(len(c))
    deg = degree_vector(adj)
    d = np.log1p(deg) / max(np.log1p(float(np.max(deg))), 1.0)
    confidence_weight = 1.1 if use_confidence else 0.0
    agreement_weight = 0.35 + 0.65 * homophily if use_agreement else 0.0
    stability_weight = 0.9 if use_stability else 0.0
    source_weight = 0.85 if use_source else 0.0
    degree_weight = 0.2 if use_degree else 0.0
    base = (confidence_weight * c + agreement_weight * a + stability_weight * s
            + source_weight * src + degree_weight * d)
    threshold = float(np.quantile(base, 0.6 if (use_agreement or use_stability) else 0.5))
    scale = float(np.std(base) + 1e-6)
    raw = np.clip((base - threshold) / scale, -6.0, 6.0)
    r = 1.0 / (1.0 + np.exp(-raw))
    return r, {"confidence": c, "agreement": a, "stability": s, "source": src, "degree": d, "estimated_homophily": homophily}


def group_confidence(adj, probs):
    deg = degree_vector(adj)
    conf = np.max(probs, axis=1)
    low = deg <= np.quantile(deg, 0.33)
    mid = (deg > np.quantile(deg, 0.33)) & (deg <= np.quantile(deg, 0.66))
    high = deg > np.quantile(deg, 0.66)
    out = {}
    for name, mask in [("low", low), ("mid", mid), ("high", high)]:
        out[name] = float(np.mean(conf[mask])) if np.any(mask) else 0.0
    return out
