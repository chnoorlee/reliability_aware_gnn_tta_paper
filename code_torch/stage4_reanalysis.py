"""Stage-4 graph-aware reanalysis of the two frozen adaptation audits.

The module never reruns adaptation and never changes its inputs.  It validates
the frozen factorial designs, replays the endpoint guard from label-free proxy
maxima, and emits graph-sensitive descriptive analyses.  Target labels enter
only through accuracy gains that were already computed by the offline audits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

DELTA_LIMIT = 0.05
PHI_LIMIT = 0.20
PRIMARY_TAU = 0.01
TAUS = (0.0, 0.005, 0.01, 0.02, 0.05)
BOOTSTRAP_SEED = 20260819
MIN_BOOTSTRAP_VALID_FRACTION = 0.95
HELDOUT_DATASETS = {
    "amazon_photo",
    "citeseer",
    "coauthor_cs",
    "cornell",
    "pubmed",
    "wisconsin",
}
HELDOUT_SEEDS = {1, 2, 3}
HELDOUT_ADAPTERS = {"confidence_source_entropy", "uniform_entropy"}
HELDOUT_CELLS = {
    ("edge_add", 0.5),
    ("edge_add", 1.0),
    ("edge_drop", 0.3),
    ("edge_drop", 0.9),
    ("homophily_shift", 0.5),
    ("homophily_shift", 0.75),
}
OFFICIAL_METHODS = {"Matcha_T3A", "T3A", "TSA_T3A"}
OFFICIAL_SEEDS = {30, 50, 99}
OFFICIAL_TARGETS = {
    "src": {"str1", "str2"},
    "src_imb": {"css1", "css2", "nbr1", "nbr2", "str3", "str4"},
}
OFFICIAL_TARGET_CONFIGS = {
    ("src", "str1"): "CSBM7",
    ("src", "str2"): "CSBM8",
    ("src_imb", "nbr1"): "CSBM1",
    ("src_imb", "nbr2"): "CSBM2",
    ("src_imb", "css1"): "CSBM3",
    ("src_imb", "css2"): "CSBM4",
    ("src_imb", "str3"): "CSBM5",
    ("src_imb", "str4"): "CSBM6",
}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _detail(numerator: float, denominator: float) -> dict:
    return {
        "estimate": _ratio(float(numerator), float(denominator)),
        "numerator": float(numerator),
        "denominator": float(denominator),
    }


def _bootstrap_interval(values, replicates: int) -> dict:
    """Gate scalar percentile intervals on a declared valid-replicate rate."""

    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    valid = [float(value) for value in values if _finite(value)]
    valid_fraction = len(valid) / replicates
    if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION:
        interval = np.quantile(valid, [0.025, 0.975]).tolist()
        status = "percentile_95ci"
    else:
        interval = [None, None]
        status = "unstable_due_to_undefined_resamples"
    return {
        "bootstrap_valid_reps": len(valid),
        "bootstrap_invalid_reps": replicates - len(valid),
        "valid_fraction": valid_fraction,
        "ci_lower": interval[0],
        "ci_upper": interval[1],
        "ci_status": status,
        "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
    }


def binary_auc(labels, scores) -> float | None:
    """Tie-aware AUROC with high score denoting the positive class."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    u_value = float(np.sum(ranks[labels]) - n_pos * (n_pos + 1) / 2.0)
    return u_value / (n_pos * n_neg)


def _weighted_auc(labels, scores, weights) -> float | None:
    """Frequency/probability-weighted AUROC with exact tie handling."""

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if labels.shape != scores.shape or labels.shape != weights.shape:
        raise ValueError("labels, scores, and weights must have the same shape")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("AUROC weights must be finite and nonnegative")
    keep = weights > 0
    if not np.any(keep):
        return None
    labels = labels[keep]
    scores = scores[keep]
    weights = weights[keep]
    positive_weight = float(weights[labels].sum())
    negative_weight = float(weights[~labels].sum())
    if positive_weight <= 0 or negative_weight <= 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    weights = weights[order]
    concordant_weight = 0.0
    lower_negative_weight = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        tie_positive = float(weights[start:stop][labels[start:stop]].sum())
        tie_negative = float(weights[start:stop][~labels[start:stop]].sum())
        concordant_weight += tie_positive * (lower_negative_weight + 0.5 * tie_negative)
        lower_negative_weight += tie_negative
        start = stop
    return concordant_weight / (positive_weight * negative_weight)


def _normalized_score(delta: float, phi: float) -> float:
    return max(float(delta) / DELTA_LIMIT, float(phi) / PHI_LIMIT)


def _assert_close(actual, expected, label, tolerance=1e-10) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def normalize_heldout(payload: dict) -> tuple[list[dict], dict]:
    records = payload.get("records", [])
    if payload.get("status") != "complete" or len(records) != 216:
        raise ValueError("held-out input must be a complete 216-record audit")
    keys = [
        (
            row["dataset"],
            int(row["seed"]),
            row["shift"],
            float(row["intensity"]),
            row["adapter"],
        )
        for row in records
    ]
    if len(set(keys)) != 216:
        raise ValueError("held-out event keys are not unique")

    normalized = []
    for index, row in enumerate(records):
        if row.get("guard") != "unguarded":
            raise ValueError(f"held-out row {index} is not an unguarded trajectory")
        for field in (
            "accuracy",
            "source_accuracy",
            "source_relative_accuracy",
            "max_delta",
            "max_phi",
        ):
            if not _finite(row.get(field)):
                raise ValueError(f"held-out row {index} has non-finite {field}")
        gain = float(row["accuracy"]) - float(row["source_accuracy"])
        _assert_close(
            gain, row["source_relative_accuracy"], f"held-out gain row {index}"
        )
        delta_trace = row.get("delta_trace", [])
        phi_trace = row.get("phi_trace", [])
        if len(delta_trace) != 300 or len(phi_trace) != 300:
            raise ValueError(
                f"held-out row {index} does not contain two 300-step traces"
            )
        _assert_close(
            max(delta_trace), row["max_delta"], f"held-out delta maximum row {index}"
        )
        _assert_close(
            max(phi_trace), row["max_phi"], f"held-out phi maximum row {index}"
        )
        delta = float(row["max_delta"])
        phi = float(row["max_phi"])
        normalized.append(
            {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "shift": row["shift"],
                "intensity": float(row["intensity"]),
                "adapter": row["adapter"],
                "gain": gain,
                "delta": delta,
                "phi": phi,
                "score": _normalized_score(delta, phi),
                "accept": bool(delta <= DELTA_LIMIT and phi <= PHI_LIMIT),
            }
        )

    adapters = sorted({row["adapter"] for row in normalized})
    if set(adapters) != HELDOUT_ADAPTERS:
        raise ValueError(
            f"held-out adapters are {set(adapters)!r}, expected {HELDOUT_ADAPTERS!r}"
        )
    if {row["dataset"] for row in normalized} != HELDOUT_DATASETS:
        raise ValueError("held-out dataset identities do not match the frozen design")
    if {row["seed"] for row in normalized} != HELDOUT_SEEDS:
        raise ValueError("held-out seed identities do not match the frozen design")
    for adapter in adapters:
        rows = [row for row in normalized if row["adapter"] == adapter]
        if len(rows) != 108:
            raise ValueError(
                f"held-out adapter {adapter} has {len(rows)} rows, expected 108"
            )
        clusters = Counter((row["dataset"], row["seed"]) for row in rows)
        if set(clusters.values()) != {6} or len(clusters) != 18:
            raise ValueError(
                f"held-out adapter {adapter} has an invalid dataset-seed grid"
            )
        for dataset in sorted(HELDOUT_DATASETS):
            for seed in sorted(HELDOUT_SEEDS):
                cells = {
                    (row["shift"], row["intensity"])
                    for row in rows
                    if row["dataset"] == dataset and row["seed"] == seed
                }
                if cells != HELDOUT_CELLS:
                    raise ValueError(
                        "held-out fixed stress-cell matrix mismatch for "
                        f"{adapter}/{dataset}/seed={seed}: {cells!r}"
                    )
    base_event_sets = {
        adapter: {
            (row["dataset"], row["seed"], row["shift"], row["intensity"])
            for row in normalized
            if row["adapter"] == adapter
        }
        for adapter in adapters
    }
    if len({frozenset(events) for events in base_event_sets.values()}) != 1:
        raise ValueError("held-out adapters do not share identical base event keys")
    return normalized, {
        "status": "pass",
        "records": 216,
        "unique_event_keys": 216,
        "adapters": adapters,
        "trace_length": 300,
        "factorial_design": "6 datasets x 3 seeds x 6 fixed stress cells x 2 adapters",
        "fixed_cell_matrix_verified": True,
        "adapter_pairing_verified": True,
        "guard_replay": "accept iff max_delta <= 0.05 and max_phi <= 0.20",
    }


def normalize_official(payload: dict) -> tuple[list[dict], dict]:
    records = payload.get("records", [])
    if payload.get("status") != "complete" or len(records) != 72:
        raise ValueError("official input must be a complete 72-record audit")
    keys = [
        (
            row["data_config"],
            row["source_setting"],
            row["target_setting"],
            row["method"],
            int(row["seed"]),
        )
        for row in records
    ]
    if len(set(keys)) != 72:
        raise ValueError("official event keys are not unique")

    normalized = []
    for index, row in enumerate(records):
        target_key = (row["source_setting"], row["target_setting"])
        expected_config = OFFICIAL_TARGET_CONFIGS.get(target_key)
        if row.get("data_config") != expected_config:
            raise ValueError(
                f"official target/config mapping mismatch at row {index}: "
                f"{target_key!r} must map to {expected_config!r}, "
                f"got {row.get('data_config')!r}"
            )
        finite = row.get("candidate_status") == "finite"
        if finite:
            for field in (
                "source_accuracy",
                "candidate_accuracy",
                "source_relative_accuracy",
                "delta",
                "phi",
            ):
                if not _finite(row.get(field)):
                    raise ValueError(f"official finite row {index} has invalid {field}")
            gain = float(row["candidate_accuracy"]) - float(row["source_accuracy"])
            _assert_close(
                gain, row["source_relative_accuracy"], f"official gain row {index}"
            )
            delta = float(row["delta"])
            phi = float(row["phi"])
            accept = bool(delta <= DELTA_LIMIT and phi <= PHI_LIMIT)
            score = _normalized_score(delta, phi)
        else:
            if any(
                row.get(field) is not None
                for field in (
                    "candidate_accuracy",
                    "source_relative_accuracy",
                    "delta",
                    "phi",
                )
            ):
                raise ValueError(
                    f"official nonfinite row {index} contains finite-only outcomes"
                )
            gain = delta = phi = score = None
            accept = False
        if accept != bool(row.get("fixed_guard_accept")):
            raise ValueError(f"official fixed-guard replay mismatch at row {index}")
        expected_deployed = (
            float(row["candidate_accuracy"])
            if accept
            else float(row["source_accuracy"])
        )
        _assert_close(
            expected_deployed,
            row["deployed_accuracy"],
            f"official deployment row {index}",
        )
        normalized.append(
            {
                "data_config": row["data_config"],
                "source_setting": row["source_setting"],
                "target_setting": row["target_setting"],
                "method": row["method"],
                "seed": int(row["seed"]),
                "candidate_status": row["candidate_status"],
                "finite": finite,
                "gain": gain,
                "delta": delta,
                "phi": phi,
                "score": score,
                "accept": accept,
                "runtime_seconds": float(row["runtime_seconds"]),
            }
        )

    methods = sorted({row["method"] for row in normalized})
    if set(methods) != OFFICIAL_METHODS:
        raise ValueError(
            f"official methods are {set(methods)!r}, expected {OFFICIAL_METHODS!r}"
        )
    if {row["seed"] for row in normalized} != OFFICIAL_SEEDS:
        raise ValueError("official seed identities do not match the frozen design")
    if {row["source_setting"] for row in normalized} != set(OFFICIAL_TARGETS):
        raise ValueError("official source settings do not match the frozen design")
    for method in methods:
        rows = [row for row in normalized if row["method"] == method]
        if len(rows) != 24:
            raise ValueError(
                f"official method {method} has {len(rows)} rows, expected 24"
            )
        for setting, expected in (("src", 2), ("src_imb", 6)):
            clusters = Counter(
                (row["source_setting"], row["seed"])
                for row in rows
                if row["source_setting"] == setting
            )
            if len(clusters) != 3 or set(clusters.values()) != {expected}:
                raise ValueError(
                    f"official {method}/{setting} target matrix is invalid"
                )
            for seed in sorted(OFFICIAL_SEEDS):
                targets = {
                    row["target_setting"]
                    for row in rows
                    if row["source_setting"] == setting and row["seed"] == seed
                }
                if targets != OFFICIAL_TARGETS[setting]:
                    raise ValueError(
                        "official fixed target matrix mismatch for "
                        f"{method}/{setting}/seed={seed}: {targets!r}"
                    )
    method_event_sets = {
        method: {
            (
                row["data_config"],
                row["source_setting"],
                row["target_setting"],
                row["seed"],
            )
            for row in normalized
            if row["method"] == method
        }
        for method in methods
    }
    if len({frozenset(events) for events in method_event_sets.values()}) != 1:
        raise ValueError("official methods do not share identical base event keys")
    failures = Counter(
        row["candidate_status"] for row in normalized if not row["finite"]
    )
    return normalized, {
        "status": "pass",
        "records": 72,
        "unique_event_keys": 72,
        "methods": methods,
        "finite_candidates": sum(row["finite"] for row in normalized),
        "nonfinite_candidates": sum(not row["finite"] for row in normalized),
        "failure_status_counts": dict(sorted(failures.items())),
        "factorial_design": "(2 src targets + 6 src_imb targets) x 3 seeds x 3 methods",
        "fixed_target_matrix_verified": True,
        "target_config_mapping_verified": True,
        "method_pairing_verified": True,
        "guard_replay_mismatches": 0,
    }


def metric_bundle(rows: list[dict], tau: float, weights=None) -> dict:
    if not rows:
        raise ValueError("metric_bundle requires at least one finite row")
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    accepted = np.asarray([row["accept"] for row in rows], dtype=bool)
    weights = (
        np.ones(len(rows), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    harmful = gains < -float(tau)
    beneficial = gains > float(tau)
    neutral = ~(harmful | beneficial)
    rejected = ~accepted
    downside = np.where(harmful, -gains, 0.0)
    utility = np.where(beneficial, gains, 0.0)
    total_weight = float(weights.sum())
    harm_weight = float(weights[harmful].sum())
    nonharm_weight = float(weights[~harmful].sum())
    utility_total = float(np.sum(weights * utility))
    downside_total = float(np.sum(weights * downside))
    return {
        "n": int(len(rows)),
        "weight_sum": total_weight,
        "harm_n": int(harmful.sum()),
        "benefit_n": int(beneficial.sum()),
        "neutral_n": int(neutral.sum()),
        "accepted_n": int(accepted.sum()),
        "accepted_harm_n": int(np.sum(accepted & harmful)),
        "coverage": _detail(float(weights[accepted].sum()), total_weight),
        "harm_recall": _detail(float(weights[rejected & harmful].sum()), harm_weight),
        "harmful_continuation": _detail(
            float(weights[accepted & harmful].sum()), harm_weight
        ),
        "false_intervention": _detail(
            float(weights[rejected & ~harmful].sum()), nonharm_weight
        ),
        "retained_material_utility": _detail(
            float(np.sum(weights * utility * accepted)), utility_total
        ),
        "prevented_material_downside": _detail(
            float(np.sum(weights * downside * rejected)), downside_total
        ),
        "material_downside": downside_total,
        "residual_material_downside": float(np.sum(weights * downside * accepted)),
        "material_positive_utility": utility_total,
        "foregone_material_utility": float(np.sum(weights * utility * rejected)),
        "mean_candidate_gain": float(np.sum(weights * gains) / total_weight),
        "mean_deployed_gain": float(np.sum(weights * gains * accepted) / total_weight),
        "worst_deployed_gain": float(np.min(np.where(accepted, gains, 0.0))),
        "analysis_status": (
            "sparse_descriptive_only"
            if harmful.sum() < 5 or (~harmful).sum() < 5
            else "descriptive"
        ),
    }


def score_diagnostics(rows: list[dict], tau: float, weights=None) -> dict:
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    harmful = gains < -float(tau)
    values = {
        "delta": np.asarray([row["delta"] for row in rows], dtype=float),
        "phi": np.asarray([row["phi"] for row in rows], dtype=float),
        "score": np.asarray([row["score"] for row in rows], dtype=float),
    }
    if weights is None:
        aucs = {name: binary_auc(harmful, score) for name, score in values.items()}
    else:
        aucs = {
            name: _weighted_auc(harmful, score, weights)
            for name, score in values.items()
        }
    return {
        "harm_n": int(harmful.sum()),
        "nonharm_n": int((~harmful).sum()),
        "auc_delta": aucs["delta"],
        "auc_phi": aucs["phi"],
        "auc_score": aucs["score"],
        "phi_minus_score": (
            None
            if aucs["phi"] is None or aucs["score"] is None
            else float(aucs["phi"] - aucs["score"])
        ),
        "phi_minus_delta": (
            None
            if aucs["phi"] is None or aucs["delta"] is None
            else float(aucs["phi"] - aucs["delta"])
        ),
    }


def _environment_r2(rows: list[dict], keys: tuple[str, ...]) -> float | None:
    values = np.asarray([row["phi"] for row in rows], dtype=float)
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 0:
        return None
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[tuple(row[key] for key in keys)].append(index)
    residual = 0.0
    for indices in groups.values():
        group_values = values[indices]
        residual += float(np.sum((group_values - group_values.mean()) ** 2))
    return float(1.0 - residual / total)


def _conditional_concordance(
    rows: list[dict], tau: float, keys: tuple[str, ...]
) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    concordance = 0.0
    pairs = 0
    eligible = 0
    for group in groups.values():
        harmful = [row for row in group if row["gain"] < -tau]
        nonharmful = [row for row in group if row["gain"] >= -tau]
        if not harmful or not nonharmful:
            continue
        eligible += 1
        for harm in harmful:
            for safe in nonharmful:
                pairs += 1
                concordance += (
                    1.0
                    if harm["phi"] > safe["phi"]
                    else 0.5 if harm["phi"] == safe["phi"] else 0.0
                )
    return {
        "estimate": _ratio(concordance, pairs),
        "eligible_strata": eligible,
        "total_strata": len(groups),
        "eligible_pairs": pairs,
        "inferential_status": "descriptive_three_seed_strata",
    }


def phi_diagnostics(
    rows: list[dict], tau: float, environment_keys: tuple[str, ...]
) -> dict:
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    phi = np.asarray([row["phi"] for row in rows], dtype=float)
    harm = gains < -tau
    benefit = gains > tau
    exposure = np.abs(gains) > tau
    changed = harm | benefit
    directional_auc = (
        binary_auc(harm[changed], phi[changed]) if np.any(changed) else None
    )
    rho_gain = spearmanr(phi, gains).statistic
    rho_abs = spearmanr(phi, np.abs(gains)).statistic
    conditional = _conditional_concordance(rows, tau, environment_keys)
    pooled = binary_auc(harm, phi)
    return {
        "tau": tau,
        "auc_phi_harm_vs_rest": pooled,
        "auc_phi_benefit_vs_rest": binary_auc(benefit, phi),
        "auc_phi_abs_change_vs_rest": binary_auc(exposure, phi),
        "directional_auc_phi_material_change": directional_auc,
        "material_harm_n": int(harm.sum()),
        "material_benefit_n": int(benefit.sum()),
        "spearman_phi_gain": float(rho_gain) if np.isfinite(rho_gain) else None,
        "spearman_phi_abs_gain": float(rho_abs) if np.isfinite(rho_abs) else None,
        "within_environment": conditional,
        "pooled_auc_minus_conditional_concordance": (
            None
            if pooled is None or conditional["estimate"] is None
            else float(pooled - conditional["estimate"])
        ),
        "phi_node_scope": "all nodes used by stored proxy computation; exact index set not serialized",
        "gain_node_scope": "stored offline test mask; exact indices not serialized",
        "scope_match_verified": False,
        "mechanical_bound_test_applicable": False,
        "interpretation_boundary": "association diagnostic; not causal or direction-specific proof",
    }


def weighted_phi_auc_diagnostics(rows: list[dict], tau: float, weights=None) -> dict:
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    phi = np.asarray([row["phi"] for row in rows], dtype=float)
    weights = (
        np.ones(len(rows), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    harm = gains < -tau
    benefit = gains > tau
    exposure = np.abs(gains) > tau
    changed = harm | benefit
    return {
        "auc_phi_harm_vs_rest": _weighted_auc(harm, phi, weights),
        "auc_phi_benefit_vs_rest": _weighted_auc(benefit, phi, weights),
        "auc_phi_abs_change_vs_rest": _weighted_auc(exposure, phi, weights),
        "directional_auc_phi_material_change": (
            _weighted_auc(harm[changed], phi[changed], weights[changed])
            if np.any(changed)
            else None
        ),
    }


def phi_environment_summaries(
    heldout_rows: list[dict], official_rows: list[dict]
) -> list[dict]:
    output = []
    specifications = (
        ("heldout", heldout_rows, "dataset", ("dataset",)),
        ("heldout", heldout_rows, "shift_intensity", ("shift", "intensity")),
        ("heldout", heldout_rows, "adapter", ("adapter",)),
        (
            "official",
            [row for row in official_rows if row["finite"]],
            "source_setting",
            ("source_setting",),
        ),
        (
            "official",
            [row for row in official_rows if row["finite"]],
            "target_setting",
            ("target_setting",),
        ),
        (
            "official",
            [row for row in official_rows if row["finite"]],
            "method",
            ("method",),
        ),
    )
    for protocol, rows, grouping, keys in specifications:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        for group_key in sorted(groups, key=lambda value: tuple(map(str, value))):
            group = groups[group_key]
            phi = np.asarray([row["phi"] for row in group], dtype=float)
            gains = np.asarray([row["gain"] for row in group], dtype=float)
            output.append(
                {
                    "protocol": protocol,
                    "grouping": grouping,
                    "group_value": list(group_key),
                    "n": len(group),
                    "phi_mean": float(phi.mean()),
                    "phi_median": float(np.median(phi)),
                    "phi_q25": float(np.quantile(phi, 0.25)),
                    "phi_q75": float(np.quantile(phi, 0.75)),
                    "mean_gain": float(gains.mean()),
                    "mean_absolute_gain": float(np.abs(gains).mean()),
                    "inferential_status": "descriptive_environment_summary",
                }
            )
    return output


def _bootstrap_multiplicity(rows: list[dict], scheme: str, rng) -> np.ndarray:
    datasets = sorted({row["dataset"] for row in rows})
    multiplicity = Counter()
    selected_datasets = rng.choice(datasets, size=len(datasets), replace=True)
    if scheme == "dataset_block":
        dataset_counts = Counter(selected_datasets)
        return np.asarray([dataset_counts[row["dataset"]] for row in rows], dtype=int)
    if scheme != "dataset_then_seed":
        raise ValueError(f"unknown bootstrap scheme {scheme!r}")
    seeds_by_dataset = {
        dataset: sorted({row["seed"] for row in rows if row["dataset"] == dataset})
        for dataset in datasets
    }
    for dataset in selected_datasets:
        seeds = seeds_by_dataset[dataset]
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            multiplicity[(dataset, int(seed))] += 1
    return np.asarray(
        [multiplicity[(row["dataset"], row["seed"])] for row in rows], dtype=int
    )


def heldout_bootstrap(
    rows: list[dict], replicates: int
) -> tuple[list[dict], list[dict]]:
    adapters = sorted({row["adapter"] for row in rows})
    schemes = ("dataset_block", "dataset_then_seed")
    metric_names = (
        "coverage",
        "harm_recall",
        "harmful_continuation",
        "retained_material_utility",
        "prevented_material_downside",
        "mean_deployed_gain",
    )
    phi_metric_names = (
        "auc_phi_harm_vs_rest",
        "auc_phi_benefit_vs_rest",
        "auc_phi_abs_change_vs_rest",
        "directional_auc_phi_material_change",
    )
    metric_output = []
    contrast_output = []
    for scheme_index, scheme in enumerate(schemes):
        rng = np.random.default_rng(BOOTSTRAP_SEED + scheme_index)
        metric_samples = {
            (adapter, metric): []
            for adapter in adapters
            for metric in metric_names + phi_metric_names
        }
        contrast_samples = {
            (adapter, tau, contrast): []
            for adapter in adapters
            for tau in TAUS
            for contrast in ("phi_minus_score", "phi_minus_delta")
        }
        one_class = Counter()
        for _ in range(int(replicates)):
            all_weights = _bootstrap_multiplicity(rows, scheme, rng)
            for adapter in adapters:
                indices = [
                    index for index, row in enumerate(rows) if row["adapter"] == adapter
                ]
                subset = [rows[index] for index in indices]
                weights = all_weights[indices]
                primary = metric_bundle(subset, PRIMARY_TAU, weights)
                for metric in metric_names:
                    value = primary[metric]
                    if isinstance(value, dict):
                        value = value["estimate"]
                    if value is not None and np.isfinite(value):
                        metric_samples[(adapter, metric)].append(float(value))
                phi_metrics = weighted_phi_auc_diagnostics(subset, PRIMARY_TAU, weights)
                for metric in phi_metric_names:
                    value = phi_metrics[metric]
                    if value is not None and np.isfinite(value):
                        metric_samples[(adapter, metric)].append(float(value))
                for tau in TAUS:
                    diagnostics = score_diagnostics(subset, tau, weights)
                    for contrast in ("phi_minus_score", "phi_minus_delta"):
                        value = diagnostics[contrast]
                        key = (adapter, tau, contrast)
                        if value is None:
                            one_class[key] += 1
                        else:
                            contrast_samples[key].append(float(value))
        for adapter in adapters:
            subset = [row for row in rows if row["adapter"] == adapter]
            primary = metric_bundle(subset, PRIMARY_TAU)
            primary_phi = weighted_phi_auc_diagnostics(subset, PRIMARY_TAU)
            for metric in metric_names + phi_metric_names:
                values = metric_samples[(adapter, metric)]
                point = primary[metric] if metric in primary else primary_phi[metric]
                if isinstance(point, dict):
                    point = point["estimate"]
                interval = _bootstrap_interval(values, replicates)
                metric_output.append(
                    {
                        "protocol": "heldout",
                        "adapter": adapter,
                        "harm_threshold": PRIMARY_TAU,
                        "metric": metric,
                        "point_estimate": point,
                        "resampling_scheme": scheme,
                        "cluster_unit": (
                            "dataset"
                            if scheme == "dataset_block"
                            else "dataset then seed"
                        ),
                        "cluster_count": len({row["dataset"] for row in rows}),
                        "bootstrap_reps": int(replicates),
                        **interval,
                        "inferential_status": "six_graph_sensitivity_only",
                    }
                )
            full = {tau: score_diagnostics(subset, tau) for tau in TAUS}
            for tau in TAUS:
                for contrast, scores in (
                    ("phi_minus_score", ("phi", "score")),
                    ("phi_minus_delta", ("phi", "delta")),
                ):
                    key = (adapter, tau, contrast)
                    values = contrast_samples[key]
                    valid_fraction = len(values) / int(replicates)
                    interval = (
                        np.quantile(values, [0.025, 0.975]).tolist()
                        if values
                        else [None, None]
                    )
                    contrast_output.append(
                        {
                            "protocol": "heldout",
                            "adapter": adapter,
                            "harm_threshold": tau,
                            "score_a": scores[0],
                            "score_b": scores[1],
                            "auc_a": full[tau]["auc_phi"],
                            "auc_b": (
                                full[tau][f"auc_{scores[1]}"]
                                if scores[1] != "score"
                                else full[tau]["auc_score"]
                            ),
                            "paired_auc_difference": full[tau][contrast],
                            "resampling_scheme": scheme,
                            "bootstrap_reps": int(replicates),
                            "bootstrap_valid_reps": len(values),
                            "bootstrap_one_class_reps": one_class[key],
                            "valid_fraction": valid_fraction,
                            "ci_lower": (
                                interval[0]
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else None
                            ),
                            "ci_upper": (
                                interval[1]
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else None
                            ),
                            "bootstrap_distribution_min": (
                                min(values) if values else None
                            ),
                            "bootstrap_distribution_max": (
                                max(values) if values else None
                            ),
                            "ci_status": (
                                "percentile_95ci"
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else "unstable_due_to_one_class_resamples"
                            ),
                            "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
                            "inferential_status": "six_graph_sensitivity_only",
                        }
                    )
    return metric_output, contrast_output


def per_graph_and_lodo(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    adapters = sorted({row["adapter"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})
    per_graph = []
    lodo = []
    for adapter in adapters:
        adapter_rows = [row for row in rows if row["adapter"] == adapter]
        full_metrics = metric_bundle(adapter_rows, PRIMARY_TAU)
        full_auc = score_diagnostics(adapter_rows, PRIMARY_TAU)
        for dataset in datasets:
            graph_rows = [row for row in adapter_rows if row["dataset"] == dataset]
            metrics = metric_bundle(graph_rows, PRIMARY_TAU)
            diagnostics = score_diagnostics(graph_rows, PRIMARY_TAU)
            per_graph.append(
                {
                    "dataset": dataset,
                    "adapter": adapter,
                    "harm_threshold": PRIMARY_TAU,
                    **metrics,
                    **diagnostics,
                    "inferential_status": "18_event_per_graph_descriptive_only",
                }
            )
            retained = [row for row in adapter_rows if row["dataset"] != dataset]
            retained_metrics = metric_bundle(retained, PRIMARY_TAU)
            retained_auc = score_diagnostics(retained, PRIMARY_TAU)
            for metric in (
                "coverage",
                "harm_recall",
                "retained_material_utility",
                "mean_deployed_gain",
            ):
                full_value = full_metrics[metric]
                retained_value = retained_metrics[metric]
                if isinstance(full_value, dict):
                    full_value = full_value["estimate"]
                    retained_value = retained_value["estimate"]
                lodo.append(
                    {
                        "adapter": adapter,
                        "omitted_dataset": dataset,
                        "metric": metric,
                        "estimate_full": full_value,
                        "estimate_without_dataset": retained_value,
                        "influence": (
                            None
                            if full_value is None or retained_value is None
                            else retained_value - full_value
                        ),
                        "n_remaining": len(retained),
                        "inferential_status": "six_deletion_influence_diagnostic",
                    }
                )
            for metric in (
                "auc_delta",
                "auc_phi",
                "auc_score",
                "phi_minus_score",
                "phi_minus_delta",
            ):
                full_value = full_auc[metric]
                retained_value = retained_auc[metric]
                lodo.append(
                    {
                        "adapter": adapter,
                        "omitted_dataset": dataset,
                        "metric": metric,
                        "estimate_full": full_value,
                        "estimate_without_dataset": retained_value,
                        "influence": (
                            None
                            if full_value is None or retained_value is None
                            else retained_value - full_value
                        ),
                        "n_remaining": len(retained),
                        "inferential_status": "six_deletion_influence_diagnostic",
                    }
                )
    return per_graph, lodo


def _official_weights(rows: list[dict], mode: str) -> np.ndarray:
    if mode == "configuration_weighted":
        return np.full(len(rows), 1.0 / len(rows), dtype=float)
    if mode != "source_setting_balanced":
        raise ValueError(f"unknown official weighting mode {mode!r}")
    counts = Counter(row["source_setting"] for row in rows)
    weights = np.asarray(
        [0.5 / counts[row["source_setting"]] for row in rows], dtype=float
    )
    for setting in ("src", "src_imb"):
        _assert_close(
            weights[[row["source_setting"] == setting for row in rows]].sum(),
            0.5,
            f"weight sum {setting}",
        )
    return weights


def _finite_official_weights(
    rows: list[dict], weights: np.ndarray, mode: str
) -> tuple[np.ndarray, dict]:
    """Construct finite-only weights and preserve the balanced estimand."""

    finite = np.asarray([row["finite"] for row in rows], dtype=bool)
    finite_rows = [row for row in rows if row["finite"]]
    finite_weights = np.asarray(weights[finite], dtype=float)
    raw_mass = {
        setting: float(
            finite_weights[
                [row["source_setting"] == setting for row in finite_rows]
            ].sum()
        )
        for setting in OFFICIAL_TARGETS
    }
    if mode == "source_setting_balanced":
        for setting in OFFICIAL_TARGETS:
            if raw_mass[setting] <= 0:
                raise ValueError(
                    "source-setting-balanced finite estimand has no finite weight "
                    f"for {setting!r}"
                )
            mask = np.asarray(
                [row["source_setting"] == setting for row in finite_rows], dtype=bool
            )
            finite_weights[mask] *= 0.5 / raw_mass[setting]
    analysis_mass = {
        setting: float(
            finite_weights[
                [row["source_setting"] == setting for row in finite_rows]
            ].sum()
        )
        for setting in OFFICIAL_TARGETS
    }
    if mode == "source_setting_balanced":
        for setting in OFFICIAL_TARGETS:
            _assert_close(
                analysis_mass[setting], 0.5, f"finite analysis mass {setting}"
            )
    return finite_weights, {
        "raw_finite_weight_mass_by_source_setting": raw_mass,
        "finite_analysis_weight_mass_by_source_setting": analysis_mass,
        "finite_analysis_weight_sum": float(finite_weights.sum()),
        "finite_balance_renormalized": mode == "source_setting_balanced",
    }


def _official_mode_result(
    method_rows: list[dict], mode: str, multiplicity=None
) -> dict:
    weights = _official_weights(method_rows, mode)
    if multiplicity is not None:
        weights = weights * np.asarray(multiplicity, dtype=float)
    finite = np.asarray([row["finite"] for row in method_rows], dtype=bool)
    accepted = np.asarray([row["accept"] for row in method_rows], dtype=bool)
    deployed_gain = np.asarray(
        [
            row["gain"] if row["accept"] and row["finite"] else 0.0
            for row in method_rows
        ],
        dtype=float,
    )
    runtime = np.asarray([row["runtime_seconds"] for row in method_rows], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("official bootstrap produced zero total weight")
    operational = {
        "method": method_rows[0]["method"],
        "weighting": mode,
        "attempts": len(method_rows),
        "finite_candidates": int(finite.sum()),
        "nonfinite_candidates": int((~finite).sum()),
        "failure_rate": _detail(float(weights[~finite].sum()), total_weight),
        "coverage": _detail(float(weights[accepted].sum()), total_weight),
        "rejection_or_failure_rate": _detail(
            float(weights[~accepted].sum()), total_weight
        ),
        "mean_deployed_gain": float(np.sum(weights * deployed_gain) / total_weight),
        "weighted_runtime_seconds": float(np.sum(weights * runtime) / total_weight),
        "finite_weight_mass": float(weights[finite].sum()),
        "all_attempt_denominator_includes_nonfinite": True,
    }
    finite_rows = [row for row in method_rows if row["finite"]]
    finite_weights, finite_weight_metadata = _finite_official_weights(
        method_rows, weights, mode
    )
    if float(finite_weights.sum()) <= 0:
        raise ValueError("official bootstrap produced no finite candidate weight")
    operational.update(finite_weight_metadata)
    finite_metrics = metric_bundle(finite_rows, PRIMARY_TAU, finite_weights)
    finite_auc = score_diagnostics(finite_rows, PRIMARY_TAU, finite_weights)
    return {
        "operational": operational,
        "finite": finite_metrics,
        "auc": finite_auc,
        "finite_rows": finite_rows,
        "finite_weights": finite_weights,
    }


def official_analysis(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    methods = sorted({row["method"] for row in rows})
    modes = ("configuration_weighted", "source_setting_balanced")
    operational = []
    sensitivity = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        mode_results = {}
        for mode in modes:
            mode_results[mode] = _official_mode_result(method_rows, mode)
            operational.append(mode_results[mode]["operational"])
        for metric in ("coverage", "failure_rate", "mean_deployed_gain"):
            config_value = mode_results["configuration_weighted"]["operational"][metric]
            balanced_value = mode_results["source_setting_balanced"]["operational"][
                metric
            ]
            if isinstance(config_value, dict):
                config_value = config_value["estimate"]
                balanced_value = balanced_value["estimate"]
            sensitivity.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate_configuration_weighted": config_value,
                    "estimate_source_setting_balanced": balanced_value,
                    "weighting_difference": balanced_value - config_value,
                    "inferential_status": "descriptive_asymmetric_grid_sensitivity",
                }
            )
        for metric in (
            "harm_recall",
            "retained_material_utility",
            "mean_candidate_gain",
        ):
            config_value = mode_results["configuration_weighted"]["finite"][metric]
            balanced_value = mode_results["source_setting_balanced"]["finite"][metric]
            if isinstance(config_value, dict):
                config_value = config_value["estimate"]
                balanced_value = balanced_value["estimate"]
            sensitivity.append(
                {
                    "method": method,
                    "metric": f"finite_{metric}",
                    "estimate_configuration_weighted": config_value,
                    "estimate_source_setting_balanced": balanced_value,
                    "weighting_difference": (
                        None
                        if config_value is None or balanced_value is None
                        else balanced_value - config_value
                    ),
                    "inferential_status": "descriptive_asymmetric_grid_sensitivity",
                }
            )
    return operational, sensitivity


def _official_seed_multiplicity(rows: list[dict], rng) -> np.ndarray:
    multiplicity = Counter()
    settings = sorted({row["source_setting"] for row in rows})
    for setting in settings:
        seeds = sorted(
            {row["seed"] for row in rows if row["source_setting"] == setting}
        )
        if len(seeds) != 3:
            raise ValueError(f"official setting {setting!r} must have three seeds")
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            multiplicity[(setting, int(seed))] += 1
    return np.asarray(
        [multiplicity[(row["source_setting"], row["seed"])] for row in rows],
        dtype=int,
    )


def official_bootstrap(
    rows: list[dict], replicates: int
) -> tuple[list[dict], list[dict]]:
    """Source-setting-stratified, method-paired seed bootstrap."""

    methods = sorted({row["method"] for row in rows})
    modes = ("configuration_weighted", "source_setting_balanced")
    metric_specs = (
        ("operational", "failure_rate"),
        ("operational", "coverage"),
        ("operational", "rejection_or_failure_rate"),
        ("operational", "mean_deployed_gain"),
        ("finite", "harm_recall"),
        ("finite", "retained_material_utility"),
        ("finite", "mean_candidate_gain"),
    )
    metric_samples = {
        (method, mode, scope, metric): []
        for method in methods
        for mode in modes
        for scope, metric in metric_specs
    }
    contrast_samples = {
        (method, mode, tau, contrast): []
        for method in methods
        for mode in modes
        for tau in TAUS
        for contrast in ("phi_minus_score", "phi_minus_delta")
    }
    one_class = Counter()
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    method_indices = {
        method: [index for index, row in enumerate(rows) if row["method"] == method]
        for method in methods
    }
    for _ in range(int(replicates)):
        all_multiplicity = _official_seed_multiplicity(rows, rng)
        for method in methods:
            indices = method_indices[method]
            method_rows = [rows[index] for index in indices]
            multiplicity = all_multiplicity[indices]
            for mode in modes:
                result = _official_mode_result(method_rows, mode, multiplicity)
                for scope, metric in metric_specs:
                    value = result[scope][metric]
                    if isinstance(value, dict):
                        value = value["estimate"]
                    if value is not None and np.isfinite(value):
                        metric_samples[(method, mode, scope, metric)].append(
                            float(value)
                        )
                for tau in TAUS:
                    diagnostics = score_diagnostics(
                        result["finite_rows"], tau, result["finite_weights"]
                    )
                    for contrast in ("phi_minus_score", "phi_minus_delta"):
                        key = (method, mode, tau, contrast)
                        value = diagnostics[contrast]
                        if value is None:
                            one_class[key] += 1
                        else:
                            contrast_samples[key].append(float(value))

    metric_output = []
    contrast_output = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for mode in modes:
            point = _official_mode_result(method_rows, mode)
            for scope, metric in metric_specs:
                point_value = point[scope][metric]
                if isinstance(point_value, dict):
                    point_value = point_value["estimate"]
                values = metric_samples[(method, mode, scope, metric)]
                interval = _bootstrap_interval(values, replicates)
                metric_output.append(
                    {
                        "protocol": "official",
                        "method": method,
                        "weighting": mode,
                        "harm_threshold": PRIMARY_TAU,
                        "metric_scope": scope,
                        "metric": metric,
                        "point_estimate": point_value,
                        "resampling_scheme": "source_setting_stratified_seed",
                        "cluster_unit": "seed within source setting",
                        "cluster_count": 6,
                        "bootstrap_reps": int(replicates),
                        **interval,
                        "inferential_status": "descriptive_small_cluster_sensitivity",
                    }
                )
            for tau in TAUS:
                diagnostics = score_diagnostics(
                    point["finite_rows"], tau, point["finite_weights"]
                )
                for contrast, scores in (
                    ("phi_minus_score", ("phi", "score")),
                    ("phi_minus_delta", ("phi", "delta")),
                ):
                    key = (method, mode, tau, contrast)
                    values = contrast_samples[key]
                    valid_fraction = len(values) / int(replicates)
                    interval = (
                        np.quantile(values, [0.025, 0.975]).tolist()
                        if values
                        else [None, None]
                    )
                    contrast_output.append(
                        {
                            "protocol": "official",
                            "method": method,
                            "weighting": mode,
                            "harm_threshold": tau,
                            "score_a": scores[0],
                            "score_b": scores[1],
                            "auc_a": diagnostics["auc_phi"],
                            "auc_b": diagnostics[f"auc_{scores[1]}"],
                            "paired_auc_difference": diagnostics[contrast],
                            "resampling_scheme": "source_setting_stratified_seed",
                            "cluster_unit": "seed within source setting",
                            "cluster_count": 6,
                            "bootstrap_reps": int(replicates),
                            "bootstrap_valid_reps": len(values),
                            "bootstrap_one_class_reps": one_class[key],
                            "valid_fraction": valid_fraction,
                            "ci_lower": (
                                interval[0]
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else None
                            ),
                            "ci_upper": (
                                interval[1]
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else None
                            ),
                            "bootstrap_distribution_min": (
                                min(values) if values else None
                            ),
                            "bootstrap_distribution_max": (
                                max(values) if values else None
                            ),
                            "ci_status": (
                                "percentile_95ci"
                                if valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION
                                else "unstable_due_to_one_class_resamples"
                            ),
                            "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
                            "inferential_status": "descriptive_small_cluster_sensitivity",
                        }
                    )
    return metric_output, contrast_output


def _flatten_for_csv(row: dict) -> dict:
    flattened = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened[f"{key}_{nested_key}"] = nested_value
        elif isinstance(value, (list, tuple)):
            flattened[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [_flatten_for_csv(row) for row in rows]
    fields = sorted({key for row in flat for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze(
    heldout_path: Path, official_path: Path, out_dir: Path, replicates: int
) -> dict:
    heldout_payload = json.loads(heldout_path.read_text(encoding="utf-8"))
    official_payload = json.loads(official_path.read_text(encoding="utf-8"))
    heldout_rows, heldout_validation = normalize_heldout(heldout_payload)
    official_rows, official_validation = normalize_official(official_payload)

    threshold_rows = []
    phi_rows = []
    for adapter in sorted({row["adapter"] for row in heldout_rows}):
        rows = [row for row in heldout_rows if row["adapter"] == adapter]
        for tau in TAUS:
            threshold_rows.append(
                {
                    "protocol": "heldout",
                    "adapter": adapter,
                    "harm_threshold": tau,
                    **metric_bundle(rows, tau),
                    **score_diagnostics(rows, tau),
                }
            )
            diagnostics = phi_diagnostics(
                rows, tau, ("dataset", "adapter", "shift", "intensity")
            )
            diagnostics.update(
                {
                    "protocol": "heldout",
                    "adapter": adapter,
                    "r2_dataset_for_phi": _environment_r2(rows, ("dataset",)),
                    "r2_shift_intensity_for_phi": _environment_r2(
                        rows, ("shift", "intensity")
                    ),
                    "r2_environment_full_for_phi": _environment_r2(
                        rows, ("dataset", "shift", "intensity")
                    ),
                }
            )
            phi_rows.append(diagnostics)

    for method in sorted({row["method"] for row in official_rows}):
        rows = [
            row for row in official_rows if row["method"] == method and row["finite"]
        ]
        for tau in TAUS:
            diagnostics = phi_diagnostics(
                rows, tau, ("source_setting", "target_setting", "method")
            )
            diagnostics.update(
                {
                    "protocol": "official",
                    "method": method,
                    "weighting": "configuration_weighted_finite_only",
                    "r2_source_setting_for_phi": _environment_r2(
                        rows, ("source_setting",)
                    ),
                    "r2_target_setting_for_phi": _environment_r2(
                        rows, ("target_setting",)
                    ),
                    "r2_environment_full_for_phi": _environment_r2(
                        rows, ("source_setting", "target_setting")
                    ),
                }
            )
            phi_rows.append(diagnostics)

    per_graph, lodo = per_graph_and_lodo(heldout_rows)
    bootstrap_metrics, paired_contrasts = heldout_bootstrap(heldout_rows, replicates)
    official_operational, official_sensitivity = official_analysis(official_rows)
    official_bootstrap_metrics, official_paired_contrasts = official_bootstrap(
        official_rows, replicates
    )
    bootstrap_metrics.extend(official_bootstrap_metrics)
    paired_contrasts.extend(official_paired_contrasts)
    environment_summaries = phi_environment_summaries(heldout_rows, official_rows)

    heldout_sha256 = _sha256(heldout_path)
    official_sha256 = _sha256(official_path)
    analysis_sha256 = _sha256(Path(__file__))

    manifest = {
        "analysis": "stage4_fresh_full_graph_aware_reanalysis",
        "inputs": [
            {"path": str(heldout_path.resolve()), "sha256": heldout_sha256},
            {"path": str(official_path.resolve()), "sha256": official_sha256},
        ],
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": analysis_sha256,
        },
        "fixed_guard": {
            "accept_if": "delta <= 0.05 and phi <= 0.20",
            "score": "max(delta/0.05, phi/0.20)",
        },
        "harm_rule": "harm iff gain < -tau; benefit iff gain > tau",
        "primary_harm_threshold": PRIMARY_TAU,
        "sensitivity_thresholds": list(TAUS),
        "bootstrap": {
            "replicates": int(replicates),
            "scheme_rng_seeds": {
                "dataset_block": BOOTSTRAP_SEED,
                "dataset_then_seed": BOOTSTRAP_SEED + 1,
                "source_setting_stratified_seed": BOOTSTRAP_SEED + 2,
            },
            "interval": "percentile 95%",
            "schemes": [
                "dataset_block",
                "dataset_then_seed",
                "source_setting_stratified_seed",
            ],
            "inferential_status": {
                "heldout": "six_graph_sensitivity_only",
                "official": "descriptive_small_cluster_sensitivity",
            },
        },
        "official_weighting": ["configuration_weighted", "source_setting_balanced"],
        "frozen_input_policy": "read-only; no adaptation rerun and no replacement of null/nonfinite outcomes",
    }
    provenance_fields = {
        "heldout_input_sha256": heldout_sha256,
        "official_input_sha256": official_sha256,
        "analysis_code_sha256": analysis_sha256,
    }
    output_tables = (
        threshold_rows,
        phi_rows,
        per_graph,
        lodo,
        bootstrap_metrics,
        paired_contrasts,
        official_operational,
        official_sensitivity,
        environment_summaries,
    )
    for table in output_tables:
        for row in table:
            row.update(provenance_fields)
    validation = {
        "status": "pass",
        "heldout": heldout_validation,
        "official": official_validation,
        "zero_denominator_policy": "null, never zero",
        "paired_bootstrap_policy": "same multiplicity vector for all scores within each replicate",
        "byte_stability": "fixed RNG seeds; sorted-key JSON; no timestamps or random paths",
    }
    summary = {
        "manifest": manifest,
        "validation": validation,
        "heldout_threshold_sensitivity": threshold_rows,
        "heldout_per_graph": per_graph,
        "heldout_lodo": lodo,
        "bootstrap_metric_intervals": bootstrap_metrics,
        "paired_auc_contrasts": paired_contrasts,
        "phi_directionality": phi_rows,
        "phi_environment_summaries": environment_summaries,
        "official_operational": official_operational,
        "official_weighting_sensitivity": official_sensitivity,
    }
    write_json(out_dir / "reanalysis_manifest.json", manifest)
    write_json(out_dir / "validation_report.json", validation)
    write_json(out_dir / "stage4_reanalysis.json", summary)
    write_csv(out_dir / "harm_threshold_sensitivity.csv", threshold_rows)
    write_csv(out_dir / "per_graph_metrics.csv", per_graph)
    write_csv(out_dir / "lodo_influence.csv", lodo)
    write_csv(out_dir / "bootstrap_metric_intervals.csv", bootstrap_metrics)
    write_csv(out_dir / "paired_auc_contrasts.csv", paired_contrasts)
    write_csv(out_dir / "phi_directionality_diagnostics.csv", phi_rows)
    write_csv(out_dir / "official_operational.csv", official_operational)
    write_csv(
        out_dir / "source_setting_weighting_sensitivity.csv", official_sensitivity
    )
    write_csv(out_dir / "phi_environment_summaries.csv", environment_summaries)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    analyze(args.heldout, args.official, args.out_dir, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
