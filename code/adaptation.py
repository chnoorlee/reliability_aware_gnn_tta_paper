import numpy as np
from utils import degree_vector, is_sparse_matrix, rebuild_adjacency, upper_triangle_edges


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


def neighborhood_agreement(adj, probs):
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
    homophily = float(np.mean(pred[edges[:, 0]] == pred[edges[:, 1]])) if len(edges) > 0 else 0.0
    mix = 0.25 + 0.5 * homophily
    return np.clip(mix * out + (1.0 - mix) * soft, 0.0, 1.0), homophily


def structural_stability(model, x, adj, seed=0, views=2, drop_rate=0.05, feature_noise=0.02):
    rng = np.random.default_rng(seed + 2027)
    base_probs, _ = model.forward(x, adj)
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
        probs, _ = model.forward(x_view, pert)
        divergences.append(js_divergence(base_probs, probs))
    if not divergences:
        return np.ones(adj.shape[0], dtype=float)
    div = np.mean(np.vstack(divergences), axis=0)
    return np.clip(1.0 - div / np.log(2.0), 0.0, 1.0)


def source_consistency(probs, reference_probs):
    if reference_probs is None:
        return np.ones(probs.shape[0], dtype=float)
    return np.clip(1.0 - js_divergence(probs, reference_probs) / np.log(2.0), 0.0, 1.0)


def reliability_scores(model, x, adj, seed=0, use_agreement=True, use_stability=True, reference_probs=None):
    probs, _ = model.forward(x, adj)
    entropy_conf = 1.0 - entropy(probs) / np.log(probs.shape[1])
    margin = top_margin(probs)
    c = np.clip(0.65 * entropy_conf + 0.35 * margin, 0.0, 1.0)
    a, homophily = neighborhood_agreement(adj, probs) if use_agreement else (np.ones(len(c)), 0.0)
    s = structural_stability(model, x, adj, seed=seed) if use_stability else np.ones(len(c))
    src = source_consistency(probs, reference_probs)
    deg = degree_vector(adj)
    d = np.log1p(deg) / max(np.log1p(float(np.max(deg))), 1.0)
    agreement_weight = 0.35 + 0.65 * homophily if use_agreement else 0.0
    stability_weight = 0.9 if use_stability else 0.0
    base = 1.1 * c + agreement_weight * a + stability_weight * s + 0.85 * src + 0.2 * d
    threshold = float(np.quantile(base, 0.6 if (use_agreement or use_stability) else 0.5))
    scale = float(np.std(base) + 1e-6)
    raw = np.clip((base - threshold) / scale, -6.0, 6.0)
    r = 1.0 / (1.0 + np.exp(-raw))
    return r, {"confidence": c, "agreement": a, "stability": s, "source": src, "degree": d, "estimated_homophily": homophily}


def adapt_classifier(model, x, adj, method, seed=0, steps=80, lr=0.05, lambda_cal=0.5, lambda_af=0.01, tol=1e-8,
                     detector=None):
    """Run reliability-aware test-time adaptation.

    ``detector`` (optional) is a ``DetectorState`` from ``detector.py``.  When
    supplied, the closed-loop negative-adaptation detector monitors the
    subgroup-confidence drift Delta_t and decision-flip fraction Phi_t after
    every adaptation step and rolls back to the previous parameters as soon as
    either signal exceeds its tolerance.  The full detector trajectory is
    returned for auditing.
    """
    use_reliability = method not in ["source_only", "entropy_all_nodes", "tent_entropy"]
    use_agreement = method != "no_neighborhood_agreement"
    use_stability = method != "no_structural_stability"
    use_cal = method not in ["no_calibration_loss", "tent_entropy", "eata_filter"]
    use_af = method not in ["no_anti_forgetting", "tent_entropy"]
    use_eata_filter = method == "eata_filter"
    use_graph_consistency = method == "graph_tta_consistency"
    use_matcha_mask = method == "matcha_reliable"

    if method == "source_only":
        return {"steps": 0, "loss_history": [], "mean_reliability": 1.0, "selected_fraction": 1.0,
                "detector": detector.to_dict() if detector is not None else None}

    source_probs, _ = model.forward(x, adj)
    source_group_conf = group_confidence(adj, source_probs)
    prev = None
    stable = 0
    losses = []
    reliability_trace = []
    selected_trace = []
    delta_trace = []
    phi_trace = []
    drift_trace = {"low": [], "mid": [], "high": []}
    # Snapshot of the previous step's classifier so we can roll back on detector halt.
    prev_w1 = model.w1.copy()
    source_argmax = np.argmax(source_probs, axis=1)
    detector_halted_step = None

    for step in range(steps):
        probs, cache = model.forward(x, adj)
        # Detector signals (computed even when detector is None for auditability).
        current_group_conf = group_confidence(adj, probs)
        delta_t = float(np.mean([abs(current_group_conf[k] - source_group_conf[k]) for k in source_group_conf]))
        phi_t = float(np.mean(np.argmax(probs, axis=1) != source_argmax))
        delta_trace.append(delta_t)
        phi_trace.append(phi_t)
        for k in drift_trace:
            drift_trace[k].append(float(current_group_conf[k] - source_group_conf[k]))
        if detector is not None and step > 0:
            if delta_t > detector.delta_tolerance:
                detector.triggered = True
                detector.trigger_step = step
                detector.trigger_reason = f"delta>{detector.delta_tolerance:.3f}"
                model.w1 = prev_w1
                detector_halted_step = step
                break
            if phi_t > detector.phi_tolerance:
                detector.triggered = True
                detector.trigger_step = step
                detector.trigger_reason = f"phi>{detector.phi_tolerance:.3f}"
                model.w1 = prev_w1
                detector_halted_step = step
                break
        prev_w1 = model.w1.copy()
        if detector is not None:
            detector.delta_history.append(delta_t)
            detector.phi_history.append(phi_t)
        if use_reliability:
            r, rel_parts = reliability_scores(model, x, adj, seed=seed + step, use_agreement=use_agreement, use_stability=use_stability, reference_probs=source_probs)
        else:
            r = np.ones(x.shape[0], dtype=float)
            rel_parts = {}
        if use_eata_filter:
            ent = entropy(probs) / np.log(probs.shape[1])
            src_sim = source_consistency(probs, source_probs)
            # EATA-style filtering: adapt on confident nodes while suppressing samples
            # that drift too far from the source prediction. This is a lightweight
            # classifier-only analogue, not an exact reproduction of the original method.
            r = ((ent <= np.quantile(ent, 0.6)) & (src_sim >= np.quantile(src_sim, 0.4))).astype(float)
        if use_matcha_mask:
            # Matcha-inspired masked adaptation: retain the most reliable half of
            # nodes as hard pseudo-targets for entropy minimization.
            r = (r >= np.quantile(r, 0.5)).astype(float)
        reliability_trace.append(float(np.mean(r)))
        selected_trace.append(float(np.mean(r >= 0.5)))
        weights = np.power(np.clip(r, 1e-3, 1.0), 1.5)
        weights = weights / max(np.sum(weights), 1e-12)

        p = np.clip(probs, 1e-12, 1.0)
        logp1 = np.log(p) + 1.0
        weighted_sum = np.sum(p * logp1, axis=1, keepdims=True)
        dlogits = weights[:, None] * p * (weighted_sum - logp1)

        # Backbone-agnostic classifier-gradient: prefer the cached pre-classifier
        # feature matrix (h_concat for GraphSAGE, ah for GCN, h_agg for GAT,
        # h for APPNP).  Falls back to a@h for the original GCN convention.
        if "h_concat" in cache:
            pre_class = cache["h_concat"]
        elif "h_agg" in cache:
            pre_class = cache["h_agg"]
        elif "a" in cache and "h" in cache:
            pre_class = cache["a"] @ cache["h"]
        else:
            pre_class = cache.get("h", probs)
        dw1 = pre_class.T @ dlogits

        loss = float(np.sum(weights * entropy(probs)))

        if use_graph_consistency:
            neigh_agree, _ = neighborhood_agreement(adj, probs)
            consistency_penalty = float(np.mean((1.0 - neigh_agree) * np.max(probs, axis=1)))
            # Conservative differentiable proxy: shrink classifier confidence in
            # graph-inconsistent regions.
            dw1 += 0.05 * consistency_penalty * model.w1
            loss += 0.05 * consistency_penalty

        if use_cal:
            current_group_conf = group_confidence(adj, probs)
            cal_loss = sum((current_group_conf[k] - source_group_conf[k]) ** 2 for k in source_group_conf)
            # Conservative differentiable approximation: penalize excessive confidence by shrinking classifier.
            dw1 += lambda_cal * cal_loss * model.w1
            loss += lambda_cal * float(cal_loss)

        if use_af:
            diff = model.w1 - model.w1_source
            dw1 += lambda_af * diff
            loss += 0.5 * lambda_af * float(np.sum(diff ** 2))

        grad_norm = np.sqrt(np.sum(dw1 ** 2))
        scale = min(1.0, 2.0 / max(grad_norm, 1e-12))
        model.w1 -= lr * scale * dw1
        losses.append(loss)

        if prev is not None and abs(prev - loss) < tol:
            stable += 1
        else:
            stable = 0
        if stable >= 5:
            break
        prev = loss
    return {
        "steps": step + 1,
        "loss_history": losses,
        "mean_reliability": float(np.mean(reliability_trace)) if reliability_trace else 1.0,
        "selected_fraction": float(np.mean(selected_trace)) if selected_trace else 1.0,
        "delta_trace": delta_trace,
        "phi_trace": phi_trace,
        "drift_trace": drift_trace,
        "detector": detector.to_dict() if detector is not None else None,
        "detector_halted_step": detector_halted_step,
    }


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
