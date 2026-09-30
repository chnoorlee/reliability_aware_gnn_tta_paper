"""Stage-5 selective-risk diagnostics over the two frozen audit artifacts.

This module is deliberately offline.  It does not rerun adaptation or replace
the fixed controller.  Target-label gains are used only to audit matched-
coverage baselines, unattainable oracle references, and descriptive risk
curves after the frozen experiments have completed.
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

from stage4_reanalysis import (
    DELTA_LIMIT,
    PHI_LIMIT,
    PRIMARY_TAU,
    _official_weights,
    normalize_heldout,
    normalize_official,
)


BOOTSTRAP_SEED = 20260902
MIN_BOOTSTRAP_VALID_FRACTION = 0.95
COVERAGE_GRID = tuple(index / 20.0 for index in range(21))
MATCHED_POLICIES = (
    "combined_score",
    "delta_only",
    "phi_only",
    "random_expectation",
    "oracle_gain",
)
METRIC_NAMES = (
    "accepted_harm_rate",
    "retained_positive_utility",
    "mean_deployed_gain",
    "unintercepted_harm_fraction",
    "prevented_downside",
    "accepted_conditional_downside",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_tie_aware_selection(
    scores,
    target_mass: float,
    capacities=None,
    *,
    lower_is_better: bool = True,
) -> tuple[np.ndarray, dict]:
    """Select an exact effective mass with fractional boundary-tie weights.

    Every item at the boundary score receives the same selected fraction of its
    available capacity.  The result therefore does not depend on input order.
    """

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or np.any(~np.isfinite(scores)):
        raise ValueError("scores must be a finite one-dimensional array")
    capacities = (
        np.ones(scores.size, dtype=float)
        if capacities is None
        else np.asarray(capacities, dtype=float)
    )
    if capacities.shape != scores.shape:
        raise ValueError("capacities and scores must have the same shape")
    if np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError("capacities must be finite and nonnegative")
    total_mass = float(capacities.sum())
    target_mass = float(target_mass)
    tolerance = 1e-10 * max(1.0, total_mass)
    if target_mass < -tolerance or target_mass > total_mass + tolerance:
        raise ValueError("target mass is outside [0, total capacity]")
    target_mass = min(max(target_mass, 0.0), total_mass)
    selected = np.zeros_like(capacities)
    if target_mass <= tolerance:
        return selected, {
            "target_mass": target_mass,
            "selected_mass": 0.0,
            "boundary_score": None,
            "boundary_fraction": None,
            "boundary_item_count": 0,
            "lower_is_better": lower_is_better,
        }

    ordered_scores = np.unique(scores)
    if not lower_is_better:
        ordered_scores = ordered_scores[::-1]
    remaining = target_mass
    boundary_score = None
    boundary_fraction = None
    boundary_item_count = 0
    for value in ordered_scores:
        mask = scores == value
        group_mass = float(capacities[mask].sum())
        if group_mass <= 0:
            continue
        if remaining >= group_mass - tolerance:
            selected[mask] = capacities[mask]
            remaining -= group_mass
            if remaining <= tolerance:
                remaining = 0.0
                boundary_score = float(value)
                boundary_fraction = 1.0
                boundary_item_count = int(np.sum(mask & (capacities > 0)))
                break
            continue
        fraction = remaining / group_mass
        selected[mask] = capacities[mask] * fraction
        boundary_score = float(value)
        boundary_fraction = float(fraction)
        boundary_item_count = int(np.sum(mask & (capacities > 0)))
        remaining = 0.0
        break
    if remaining > tolerance:
        raise RuntimeError("exact selection did not exhaust the requested mass")
    if not math.isclose(
        float(selected.sum()), target_mass, rel_tol=0.0, abs_tol=tolerance
    ):
        raise RuntimeError("exact selection mass check failed")
    return selected, {
        "target_mass": target_mass,
        "selected_mass": float(selected.sum()),
        "boundary_score": boundary_score,
        "boundary_fraction": boundary_fraction,
        "boundary_item_count": boundary_item_count,
        "lower_is_better": lower_is_better,
    }


def policy_selection(
    rows: list[dict],
    policy: str,
    target_mass: float,
    capacities=None,
) -> tuple[np.ndarray, dict]:
    capacities = (
        np.ones(len(rows), dtype=float)
        if capacities is None
        else np.asarray(capacities, dtype=float)
    )
    total_mass = float(capacities.sum())
    if policy == "random_expectation":
        fraction = _ratio(float(target_mass), total_mass)
        if fraction is None:
            raise ValueError("random expectation requires positive total mass")
        selected = capacities * fraction
        return selected, {
            "target_mass": float(target_mass),
            "selected_mass": float(selected.sum()),
            "boundary_score": None,
            "boundary_fraction": fraction,
            "boundary_item_count": len(rows),
            "lower_is_better": None,
            "analytical_expectation": True,
        }
    specifications = {
        "combined_score": ("score", True),
        "delta_only": ("delta", True),
        "phi_only": ("phi", True),
        "oracle_gain": ("gain", False),
    }
    if policy not in specifications:
        raise ValueError(f"unknown policy {policy!r}")
    field, lower_is_better = specifications[policy]
    selected, metadata = exact_tie_aware_selection(
        [row[field] for row in rows],
        target_mass,
        capacities,
        lower_is_better=lower_is_better,
    )
    metadata["analytical_expectation"] = False
    return selected, metadata


def selective_metrics(
    rows: list[dict],
    accepted_weights,
    population_weights=None,
    *,
    tau: float = PRIMARY_TAU,
) -> dict:
    if not rows:
        raise ValueError("selective metrics require at least one row")
    gains = np.asarray([row["gain"] for row in rows], dtype=float)
    if np.any(~np.isfinite(gains)):
        raise ValueError("selective metrics require finite gains")
    accepted = np.asarray(accepted_weights, dtype=float)
    population = (
        np.ones(len(rows), dtype=float)
        if population_weights is None
        else np.asarray(population_weights, dtype=float)
    )
    if accepted.shape != gains.shape or population.shape != gains.shape:
        raise ValueError("weights must match the number of rows")
    if np.any(~np.isfinite(accepted)) or np.any(~np.isfinite(population)):
        raise ValueError("weights must be finite")
    if np.any(accepted < -1e-12) or np.any(population < 0):
        raise ValueError("weights must be nonnegative")
    if np.any(accepted - population > 1e-10):
        raise ValueError("accepted weights cannot exceed population weights")
    total_mass = float(population.sum())
    if total_mass <= 0:
        raise ValueError("population weight must be positive")
    accepted_mass = float(accepted.sum())
    harmful = gains < -float(tau)
    beneficial = gains > float(tau)
    downside = np.where(harmful, -gains, 0.0)
    positive_utility = np.where(beneficial, gains, 0.0)
    harm_mass = float(population[harmful].sum())
    accepted_harm_mass = float(accepted[harmful].sum())
    total_downside = float(np.sum(population * downside))
    accepted_downside = float(np.sum(accepted * downside))
    total_positive_utility = float(np.sum(population * positive_utility))
    retained_positive = float(np.sum(accepted * positive_utility))
    return {
        "population_mass": total_mass,
        "accepted_count": accepted_mass,
        "coverage": accepted_mass / total_mass,
        "harm_count": harm_mass,
        "accepted_harm_count": accepted_harm_mass,
        "accepted_harm_rate": _ratio(accepted_harm_mass, accepted_mass),
        "retained_positive_utility": _ratio(
            retained_positive, total_positive_utility
        ),
        "mean_deployed_gain": float(np.sum(accepted * gains) / total_mass),
        "unintercepted_harm_fraction": _ratio(accepted_harm_mass, harm_mass),
        "prevented_downside": _ratio(
            total_downside - accepted_downside, total_downside
        ),
        "accepted_conditional_downside": _ratio(
            accepted_downside, accepted_mass
        ),
        "accepted_conditional_mean_gain": _ratio(
            float(np.sum(accepted * gains)), accepted_mass
        ),
        "total_positive_utility": total_positive_utility,
        "retained_positive_utility_amount": retained_positive,
        "total_downside": total_downside,
        "accepted_downside": accepted_downside,
        "harm_threshold": float(tau),
    }


def fixed_gate_observed(rows: list[dict]) -> dict:
    accepted = np.asarray([row["accept"] for row in rows], dtype=float)
    return {
        **selective_metrics(rows, accepted),
        "literal_accepted_count": int(accepted.sum()),
        "delta_limit": DELTA_LIMIT,
        "phi_limit": PHI_LIMIT,
    }


def matched_coverage_analysis(
    rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    matched_rows = []
    observed_rows = []
    for adapter in sorted({row["adapter"] for row in rows}):
        subset = [row for row in rows if row["adapter"] == adapter]
        observed = fixed_gate_observed(subset)
        target_mass = float(observed["literal_accepted_count"])
        observed_rows.append(
            {
                "adapter": adapter,
                "policy": "fixed_rectangular_gate_observed",
                **observed,
                "estimand_status": "frozen_policy_observation",
            }
        )
        fixed_weights = np.asarray([row["accept"] for row in subset], dtype=float)
        for policy in MATCHED_POLICIES:
            selected, selection = policy_selection(subset, policy, target_mass)
            metrics = selective_metrics(subset, selected)
            matched_rows.append(
                {
                    "adapter": adapter,
                    "policy": policy,
                    "reference_fixed_gate_accepted_count": int(target_mass),
                    "reference_fixed_gate_coverage": target_mass / len(subset),
                    "disagreement_mass_with_observed_fixed_gate": float(
                        np.sum(np.abs(selected - fixed_weights)) / 2.0
                    ),
                    **selection,
                    **metrics,
                    "policy_status": (
                        "unattainable_label_using_upper_reference"
                        if policy == "oracle_gain"
                        else "analytical_random_baseline"
                        if policy == "random_expectation"
                        else "post_hoc_matched_coverage_diagnostic"
                    ),
                }
            )
    return matched_rows, observed_rows


def risk_coverage_curves(rows: list[dict]) -> list[dict]:
    output = []
    for adapter in sorted({row["adapter"] for row in rows}):
        subset = [row for row in rows if row["adapter"] == adapter]
        fixed_count = int(sum(row["accept"] for row in subset))
        fixed_coverage = fixed_count / len(subset)
        points = [("grid", coverage) for coverage in COVERAGE_GRID]
        points.append(("fixed_gate_matched", fixed_coverage))
        for point_type, coverage in points:
            target_mass = coverage * len(subset)
            for policy in MATCHED_POLICIES:
                selected, selection = policy_selection(subset, policy, target_mass)
                output.append(
                    {
                        "adapter": adapter,
                        "policy": policy,
                        "point_type": point_type,
                        "requested_coverage": coverage,
                        "reference_fixed_gate_coverage": fixed_coverage,
                        **selection,
                        **selective_metrics(subset, selected),
                        "zero_coverage_risk_policy": (
                            "null" if coverage == 0 else "not_applicable"
                        ),
                        "inferential_status": "descriptive_post_hoc_curve",
                    }
                )
    return output


def _pair_concordance(labels, scores, weights=None) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = (
        np.ones(labels.size, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if labels.shape != scores.shape or labels.shape != weights.shape:
        raise ValueError("labels, scores, and weights must have the same shape")
    numerator = 0.0
    denominator = 0.0
    positive = np.flatnonzero(labels & (weights > 0))
    negative = np.flatnonzero((~labels) & (weights > 0))
    for positive_index in positive:
        for negative_index in negative:
            pair_weight = float(
                weights[positive_index] * weights[negative_index]
            )
            denominator += pair_weight
            if scores[positive_index] > scores[negative_index]:
                numerator += pair_weight
            elif scores[positive_index] == scores[negative_index]:
                numerator += 0.5 * pair_weight
    return numerator, denominator


def pooled_within_stratum_auc(rows: list[dict]) -> list[dict]:
    output = []
    score_fields = ("delta", "phi", "score")
    stratum_keys = ("dataset", "shift", "intensity")
    for adapter in sorted({row["adapter"] for row in rows}):
        subset = [row for row in rows if row["adapter"] == adapter]
        labels = np.asarray(
            [row["gain"] < -PRIMARY_TAU for row in subset], dtype=bool
        )
        strata = defaultdict(list)
        for row in subset:
            strata[tuple(row[key] for key in stratum_keys)].append(row)
        for field in score_fields:
            pooled_num, pooled_den = _pair_concordance(
                labels, [row[field] for row in subset]
            )
            conditional_num = 0.0
            conditional_den = 0.0
            eligible_strata = 0
            for stratum in strata.values():
                stratum_labels = [
                    row["gain"] < -PRIMARY_TAU for row in stratum
                ]
                numerator, denominator = _pair_concordance(
                    stratum_labels, [row[field] for row in stratum]
                )
                conditional_num += numerator
                conditional_den += denominator
                eligible_strata += int(denominator > 0)
            pooled_auc = _ratio(pooled_num, pooled_den)
            conditional_auc = _ratio(conditional_num, conditional_den)
            output.append(
                {
                    "adapter": adapter,
                    "score": field,
                    "harm_threshold": PRIMARY_TAU,
                    "harm_count": int(labels.sum()),
                    "nonharm_count": int((~labels).sum()),
                    "pooled_auc": pooled_auc,
                    "pooled_concordant_pair_weight": pooled_num,
                    "pooled_comparable_pair_weight": pooled_den,
                    "within_stratum_pair_weighted_auc": conditional_auc,
                    "within_stratum_concordant_pair_weight": conditional_num,
                    "within_stratum_comparable_pair_weight": conditional_den,
                    "eligible_strata": eligible_strata,
                    "total_strata": len(strata),
                    "pooled_minus_within_stratum": (
                        None
                        if pooled_auc is None or conditional_auc is None
                        else pooled_auc - conditional_auc
                    ),
                    "stratum_definition": "dataset x shift x intensity; three seeds per stratum",
                    "inferential_status": "descriptive_three_seed_strata",
                }
            )
    return output


def official_operational_analysis(rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        finite = np.asarray([row["finite"] for row in subset], dtype=bool)
        accepted = np.asarray([row["accept"] for row in subset], dtype=bool)
        if np.any(accepted & ~finite):
            raise ValueError("nonfinite official candidates must fail closed")
        deployed_gain = np.asarray(
            [row["gain"] if row["finite"] and row["accept"] else 0.0 for row in subset],
            dtype=float,
        )
        for weighting in ("configuration_weighted", "source_setting_balanced"):
            weights = _official_weights(subset, weighting)
            source_mass = {}
            source_accepted_mass = {}
            source_nonfinite_mass = {}
            for setting in sorted({row["source_setting"] for row in subset}):
                mask = np.asarray(
                    [row["source_setting"] == setting for row in subset], dtype=bool
                )
                source_mass[setting] = float(weights[mask].sum())
                source_accepted_mass[setting] = float(
                    weights[mask & accepted].sum()
                )
                source_nonfinite_mass[setting] = float(
                    weights[mask & ~finite].sum()
                )
            failures = Counter(
                row["candidate_status"] for row in subset if not row["finite"]
            )
            output.append(
                {
                    "method": method,
                    "weighting": weighting,
                    "literal_attempt_count": len(subset),
                    "literal_finite_count": int(finite.sum()),
                    "literal_nonfinite_count": int((~finite).sum()),
                    "literal_accepted_count": int(accepted.sum()),
                    "literal_rejected_or_failed_count": int((~accepted).sum()),
                    "effective_total_mass": float(weights.sum()),
                    "effective_finite_mass": float(weights[finite].sum()),
                    "effective_nonfinite_mass": float(weights[~finite].sum()),
                    "effective_accepted_mass": float(weights[accepted].sum()),
                    "effective_rejected_or_failed_mass": float(
                        weights[~accepted].sum()
                    ),
                    "effective_mass_by_source_setting": source_mass,
                    "effective_accepted_mass_by_source_setting": source_accepted_mass,
                    "effective_nonfinite_mass_by_source_setting": source_nonfinite_mass,
                    "failure_status_counts": dict(sorted(failures.items())),
                    "mean_deployed_gain": float(
                        np.sum(weights * deployed_gain) / weights.sum()
                    ),
                    "nonfinite_policy": "fail_closed_to_source",
                    "denominator_policy": "all_attempts_including_nonfinite",
                    "inferential_status": "descriptive_asymmetric_grid_sensitivity",
                }
            )
    return output


def _dataset_bootstrap_capacities(rows: list[dict], rng) -> np.ndarray:
    datasets = sorted({row["dataset"] for row in rows})
    sampled = rng.choice(datasets, size=len(datasets), replace=True)
    multiplicity = Counter(sampled)
    return np.asarray([multiplicity[row["dataset"]] for row in rows], dtype=float)


def _bootstrap_interval(values: list[float], replicates: int) -> dict:
    valid = [float(value) for value in values if _finite(value)]
    valid_fraction = len(valid) / int(replicates)
    if valid and valid_fraction >= MIN_BOOTSTRAP_VALID_FRACTION:
        lower, upper = np.quantile(valid, [0.025, 0.975]).tolist()
        status = "percentile_95ci"
    else:
        lower = upper = None
        status = "unstable_due_to_undefined_resamples"
    return {
        "bootstrap_valid_reps": len(valid),
        "bootstrap_invalid_reps": int(replicates) - len(valid),
        "valid_fraction": valid_fraction,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_status": status,
        "minimum_valid_fraction": MIN_BOOTSTRAP_VALID_FRACTION,
    }


def paired_cluster_bootstrap(
    rows: list[dict], replicates: int
) -> list[dict]:
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    output = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    adapters = sorted({row["adapter"] for row in rows})
    samples = {
        (adapter, policy, metric): []
        for adapter in adapters
        for policy in MATCHED_POLICIES
        if policy != "combined_score"
        for metric in METRIC_NAMES
    }
    point_metrics = {}
    for adapter in adapters:
        subset = [row for row in rows if row["adapter"] == adapter]
        fixed_coverage = sum(row["accept"] for row in subset) / len(subset)
        target_mass = fixed_coverage * len(subset)
        for policy in MATCHED_POLICIES:
            selected, _ = policy_selection(subset, policy, target_mass)
            point_metrics[(adapter, policy)] = selective_metrics(subset, selected)

    # The same graph draw is reused across adapters and policies in each replicate.
    reference_rows = [row for row in rows if row["adapter"] == adapters[0]]
    for _ in range(int(replicates)):
        reference_capacities = _dataset_bootstrap_capacities(reference_rows, rng)
        capacity_by_key = {
            (row["dataset"], row["seed"], row["shift"], row["intensity"]): capacity
            for row, capacity in zip(reference_rows, reference_capacities)
        }
        for adapter in adapters:
            subset = [row for row in rows if row["adapter"] == adapter]
            capacities = np.asarray(
                [
                    capacity_by_key[
                        (
                            row["dataset"],
                            row["seed"],
                            row["shift"],
                            row["intensity"],
                        )
                    ]
                    for row in subset
                ],
                dtype=float,
            )
            fixed_coverage = sum(row["accept"] for row in subset) / len(subset)
            target_mass = fixed_coverage * float(capacities.sum())
            replicate_metrics = {}
            for policy in MATCHED_POLICIES:
                selected, _ = policy_selection(
                    subset, policy, target_mass, capacities
                )
                replicate_metrics[policy] = selective_metrics(
                    subset, selected, capacities
                )
            reference = replicate_metrics["combined_score"]
            for policy in MATCHED_POLICIES:
                if policy == "combined_score":
                    continue
                for metric in METRIC_NAMES:
                    left = replicate_metrics[policy][metric]
                    right = reference[metric]
                    if left is not None and right is not None:
                        samples[(adapter, policy, metric)].append(left - right)

    for adapter in adapters:
        reference = point_metrics[(adapter, "combined_score")]
        for policy in MATCHED_POLICIES:
            if policy == "combined_score":
                continue
            alternative = point_metrics[(adapter, policy)]
            for metric in METRIC_NAMES:
                point_difference = (
                    None
                    if alternative[metric] is None or reference[metric] is None
                    else alternative[metric] - reference[metric]
                )
                output.append(
                    {
                        "adapter": adapter,
                        "alternative_policy": policy,
                        "reference_policy": "combined_score",
                        "metric": metric,
                        "alternative_point_estimate": alternative[metric],
                        "reference_point_estimate": reference[metric],
                        "paired_difference": point_difference,
                        "bootstrap_reps": int(replicates),
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "resampling_scheme": "paired_dataset_block",
                        "cluster_unit": "dataset",
                        "cluster_count": len({row["dataset"] for row in rows}),
                        **_bootstrap_interval(
                            samples[(adapter, policy, metric)], replicates
                        ),
                        "inferential_status": "six_graph_sensitivity_only_not_population_inference",
                    }
                )
    return output


def lodo_sensitivity(rows: list[dict]) -> list[dict]:
    output = []
    datasets = sorted({row["dataset"] for row in rows})
    for adapter in sorted({row["adapter"] for row in rows}):
        full = [row for row in rows if row["adapter"] == adapter]
        full_target = float(sum(row["accept"] for row in full))
        full_metrics = {}
        for policy in MATCHED_POLICIES:
            selected, _ = policy_selection(full, policy, full_target)
            full_metrics[policy] = selective_metrics(full, selected)
        for omitted in datasets:
            retained = [row for row in full if row["dataset"] != omitted]
            retained_target = float(sum(row["accept"] for row in retained))
            for policy in MATCHED_POLICIES:
                selected, _ = policy_selection(retained, policy, retained_target)
                retained_metrics = selective_metrics(retained, selected)
                for metric in ("coverage",) + METRIC_NAMES:
                    full_value = full_metrics[policy][metric]
                    retained_value = retained_metrics[metric]
                    output.append(
                        {
                            "adapter": adapter,
                            "omitted_dataset": omitted,
                            "policy": policy,
                            "metric": metric,
                            "full_fixed_gate_accepted_count": int(full_target),
                            "retained_fixed_gate_accepted_count": int(retained_target),
                            "estimate_full": full_value,
                            "estimate_without_dataset": retained_value,
                            "deletion_influence": (
                                None
                                if full_value is None or retained_value is None
                                else retained_value - full_value
                            ),
                            "n_remaining": len(retained),
                            "inferential_status": "descriptive_six_graph_deletion_sensitivity_not_population_inference",
                        }
                    )
    return output


def _flatten_for_csv(row: dict) -> dict:
    flattened = {}
    for key, value in row.items():
        if isinstance(value, dict):
            flattened[key] = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            )
        elif isinstance(value, (list, tuple)):
            flattened[key] = json.dumps(value, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [_flatten_for_csv(row) for row in rows]
    fields = sorted({field for row in flattened for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze(
    heldout_path: Path,
    official_path: Path,
    out_dir: Path,
    bootstrap_replicates: int,
) -> dict:
    heldout_payload = json.loads(heldout_path.read_text(encoding="utf-8"))
    official_payload = json.loads(official_path.read_text(encoding="utf-8"))
    heldout_rows, heldout_validation = normalize_heldout(heldout_payload)
    official_rows, official_validation = normalize_official(official_payload)

    matched, fixed_observed = matched_coverage_analysis(heldout_rows)
    curves = risk_coverage_curves(heldout_rows)
    auc_rows = pooled_within_stratum_auc(heldout_rows)
    bootstrap_rows = paired_cluster_bootstrap(
        heldout_rows, int(bootstrap_replicates)
    )
    lodo_rows = lodo_sensitivity(heldout_rows)
    official_rows_output = official_operational_analysis(official_rows)

    heldout_hash = _sha256(heldout_path)
    official_hash = _sha256(official_path)
    code_hash = _sha256(Path(__file__))
    provenance = {
        "heldout_input_sha256": heldout_hash,
        "official_input_sha256": official_hash,
        "analysis_code_sha256": code_hash,
    }
    for table in (
        matched,
        fixed_observed,
        curves,
        auc_rows,
        bootstrap_rows,
        lodo_rows,
        official_rows_output,
    ):
        for row in table:
            row.update(provenance)

    validation = {
        "status": "pass",
        "heldout": heldout_validation,
        "official": official_validation,
        "exact_matched_mass_verified": all(
            math.isclose(
                row["accepted_count"],
                row["reference_fixed_gate_accepted_count"],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for row in matched
        ),
        "coverage_zero_conditional_risk_is_null": all(
            row["accepted_harm_rate"] is None
            and row["accepted_conditional_downside"] is None
            for row in curves
            if row["requested_coverage"] == 0
        ),
        "official_nonfinite_fail_closed": all(
            row["literal_accepted_count"] <= row["literal_finite_count"]
            for row in official_rows_output
        ),
        "random_policy": "closed-form analytical expectation; no Monte Carlo policy draws",
        "oracle_policy": "label-using unattainable upper reference; never a deployment claim",
        "tie_policy": "equal fractional selection weight for all items at the boundary score",
        "zero_denominator_policy": "null, never zero",
    }
    manifest = {
        "analysis": "stage5_selective_risk_and_matched_coverage_audit",
        "inputs": [
            {"path": str(heldout_path.resolve()), "sha256": heldout_hash},
            {"path": str(official_path.resolve()), "sha256": official_hash},
        ],
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": code_hash,
        },
        "fixed_gate": {
            "accept_if": "max_delta <= 0.05 and max_phi <= 0.20",
            "combined_score": "max(max_delta/0.05, max_phi/0.20)",
        },
        "matched_coverage": (
            "each adapter's literal fixed-gate accepted count; fractional "
            "weights are used only for score ties at the selection boundary"
        ),
        "harm_rule": "material harm iff gain < -0.01; material benefit iff gain > 0.01",
        "coverage_grid": list(COVERAGE_GRID),
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": BOOTSTRAP_SEED,
            "scheme": "paired dataset-block bootstrap",
            "cluster_count": 6,
            "interpretation": "six-graph sensitivity only, not population inference",
        },
        "frozen_input_policy": "read-only; no adaptation rerun",
    }
    write_json(out_dir / "validation_report.json", validation)
    output_tables = {
        "fixed_gate_observed.csv": fixed_observed,
        "matched_coverage.csv": matched,
        "risk_coverage_curves.csv": curves,
        "pooled_vs_within_stratum_auc.csv": auc_rows,
        "paired_cluster_bootstrap.csv": bootstrap_rows,
        "lodo_sensitivity.csv": lodo_rows,
        "official_operational.csv": official_rows_output,
    }
    for name, rows in output_tables.items():
        write_csv(out_dir / name, rows)

    manifest["outputs"] = [
        {"path": name, "sha256": _sha256(out_dir / name)}
        for name in output_tables
    ] + [
        {
            "path": "validation_report.json",
            "sha256": _sha256(out_dir / "validation_report.json"),
        }
    ]
    summary = {
        "manifest": manifest,
        "validation": validation,
        "fixed_gate_observed": fixed_observed,
        "matched_coverage": matched,
        "risk_coverage_curves": curves,
        "pooled_vs_within_stratum_auc": auc_rows,
        "paired_cluster_bootstrap": bootstrap_rows,
        "lodo_sensitivity": lodo_rows,
        "official_operational": official_rows_output,
    }
    write_json(out_dir / "analysis_manifest.json", manifest)
    write_json(out_dir / "stage5_selective_audit.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    analyze(
        args.heldout,
        args.official,
        args.out_dir,
        args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
