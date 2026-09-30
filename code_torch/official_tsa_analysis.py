"""Summarize risk--coverage behavior in the official TSA-repository audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from risk_coverage_analysis import (
    FIXED_DELTA,
    FIXED_PHI,
    binary_auc,
    exact_binomial_interval,
    operating_metrics,
)


def normalized_proxy_score(row: dict) -> float:
    return max(float(row["delta"]) / FIXED_DELTA, float(row["phi"]) / FIXED_PHI)


def _curve(gains, scores, harm_tolerance=0.01):
    gains = np.asarray(gains, dtype=float)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[-np.inf, np.unique(scores), np.inf]
    rows = []
    for threshold in thresholds:
        metrics = operating_metrics(gains, scores <= threshold, harm_tolerance)
        rows.append(
            {
                "threshold": None if not np.isfinite(threshold) else float(threshold),
                "threshold_kind": "negative_infinity"
                if threshold == -np.inf
                else "positive_infinity"
                if threshold == np.inf
                else "finite",
                "coverage": metrics["coverage"],
                "harmful_continuation_rate": metrics["harmful_continuation_rate"],
                "retained_positive_utility": metrics["retained_positive_utility"],
                "mean_deployed_gain": metrics["mean_deployed_gain"],
            }
        )
    return rows


def _operating_metrics_with_failures(audit_rows, harm_tolerance=0.01):
    """Count non-finite candidates as fail-closed deployment attempts.

    Harm and utility require a finite candidate outcome and therefore remain
    conditional on valid candidates.  Coverage and deployed gain are
    operational quantities and use every attempted adaptation in the
    denominator; a numerical failure deploys the source model and contributes
    zero paired gain.
    """
    valid = [
        row
        for row in audit_rows
        if row.get("candidate_status", "finite") == "finite"
    ]
    if not valid:
        return None
    gains = np.asarray(
        [row["source_relative_accuracy"] for row in valid], dtype=float
    )
    accepted = np.asarray(
        [normalized_proxy_score(row) <= 1.0 for row in valid], dtype=bool
    )
    metrics = operating_metrics(gains, accepted, harm_tolerance)
    valid_n = int(metrics["n"])
    attempt_n = len(audit_rows)
    metrics["valid_candidate_n"] = valid_n
    metrics["audit_attempts"] = attempt_n
    metrics["candidate_failures"] = attempt_n - valid_n
    metrics["valid_candidate_coverage"] = metrics["coverage"]
    metrics["coverage"] = float(np.sum(accepted) / attempt_n)
    metrics["mean_deployed_gain_valid_candidates"] = metrics[
        "mean_deployed_gain"
    ]
    metrics["mean_deployed_gain"] = float(np.sum(gains * accepted) / attempt_n)
    metrics["coverage_denominator"] = "all adaptation attempts"
    metrics["harm_and_utility_denominator"] = "finite candidate outcomes"
    return metrics


def _cluster_bootstrap(rows, metric_names, harm_tolerance=0.01, replicates=5000):
    strata = {}
    for row in rows:
        source_setting = str(row["source_setting"])
        cluster_key = (source_setting, int(row["seed"]))
        strata.setdefault(source_setting, {}).setdefault(cluster_key, []).append(row)
    cluster_count = sum(len(clusters) for clusters in strata.values())
    if cluster_count < 2:
        return {name: [None, None] for name in metric_names}
    rng = np.random.default_rng(20260819)
    samples = {name: [] for name in metric_names}
    for _ in range(int(replicates)):
        boot = []
        for clusters in strata.values():
            keys = list(clusters)
            chosen = rng.choice(len(keys), size=len(keys), replace=True)
            boot.extend(row for index in chosen for row in clusters[keys[index]])
        metrics = _operating_metrics_with_failures(boot, harm_tolerance)
        if metrics is None:
            continue
        for name in metric_names:
            value = metrics[name]
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    return {
        name: np.quantile(values, [0.025, 0.975]).astype(float).tolist()
        if values
        else [None, None]
        for name, values in samples.items()
    }


def analyze(records, harm_tolerance=0.01, bootstrap_replicates=5000):
    methods = sorted({row["method"] for row in records})
    summaries = {}
    curve_rows = []
    detail_rows = []
    for method in methods:
        audit_rows = [row for row in records if row["method"] == method]
        rows = [row for row in audit_rows if row.get("candidate_status", "finite") == "finite"]
        failure_counts = {}
        for row in audit_rows:
            status = row.get("candidate_status", "finite")
            if status != "finite":
                failure_counts[status] = failure_counts.get(status, 0) + 1
        if not rows:
            summaries[method] = {
                "audit_rows": len(audit_rows),
                "valid_candidate_rows": 0,
                "candidate_failure_counts": failure_counts,
                "fixed_guard": None,
                "score_diagnostics": {},
                "source_setting_seed_clusters": 0,
            }
            detail_rows.extend(audit_rows)
            continue
        gains = np.asarray([row["source_relative_accuracy"] for row in rows], dtype=float)
        scores = {
            "delta": np.asarray([row["delta"] for row in rows], dtype=float),
            "phi": np.asarray([row["phi"] for row in rows], dtype=float),
            "normalized_max": np.asarray(
                [normalized_proxy_score(row) for row in rows], dtype=float
            ),
        }
        harmful = gains < -float(harm_tolerance)
        accepted = scores["normalized_max"] <= 1.0
        fixed = _operating_metrics_with_failures(audit_rows, harm_tolerance)
        fixed["coverage_row_descriptive_cp_95ci"] = exact_binomial_interval(
            fixed["accepted"], fixed["audit_attempts"]
        )
        fixed["harm_recall_row_descriptive_cp_95ci"] = exact_binomial_interval(
            int(np.sum((~accepted) & harmful)), int(harmful.sum())
        )
        fixed["bootstrap_95ci"] = _cluster_bootstrap(
            audit_rows,
            (
                "coverage",
                "harm_recall",
                "retained_positive_utility",
                "prevented_downside_fraction",
                "mean_deployed_gain",
            ),
            harm_tolerance,
            bootstrap_replicates,
        )
        diagnostics = {}
        for name, values in scores.items():
            pearson = float(np.corrcoef(gains, values)[0, 1]) if len(gains) > 1 else None
            spearman = spearmanr(gains, values).statistic if len(gains) > 1 else None
            diagnostics[name] = {
                "gain_pearson": pearson,
                "gain_spearman": float(spearman)
                if spearman is not None and np.isfinite(spearman)
                else None,
                "harm_auroc": binary_auc(harmful, values),
            }
            for point in _curve(gains, values, harm_tolerance):
                curve_rows.append({"method": method, "score": name, **point})
        summaries[method] = {
            "audit_rows": len(audit_rows),
            "valid_candidate_rows": len(rows),
            "candidate_failure_counts": failure_counts,
            "fixed_guard": fixed,
            "score_diagnostics": diagnostics,
            "source_setting_seed_clusters": len(
                {(row["source_setting"], row["seed"]) for row in rows}
            ),
        }
        for row in rows:
            detail_rows.append(
                {
                    **row,
                    "normalized_max": normalized_proxy_score(row),
                    "foregone_gain": max(float(row["source_relative_accuracy"]), 0.0)
                    if not row["fixed_guard_accept"]
                    else 0.0,
                }
            )
    return summaries, curve_rows, detail_rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["method"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--curve-csv", required=True)
    parser.add_argument("--detail-csv", required=True)
    parser.add_argument("--harm-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    input_path = Path(args.input)
    input_bytes = input_path.read_bytes()
    payload = json.loads(input_bytes.decode("utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Input audit is not complete: status={payload.get('status')!r}")
    records = payload.get("records", [])
    expected = (
        len(payload.get("data_configs", []))
        * len(payload.get("methods", []))
        * len(payload.get("seeds", []))
    )
    keys = {
        (row["data_config"], row["method"], int(row["seed"])) for row in records
    }
    if len(records) != expected or len(keys) != expected:
        raise ValueError(
            f"Incomplete or duplicate factorial audit: rows={len(records)}, "
            f"unique={len(keys)}, expected={expected}"
        )
    summary, curves, details = analyze(
        records, args.harm_tolerance, args.bootstrap_replicates
    )
    output = {
        "input": str(Path(args.input).resolve()),
        "input_status": payload.get("status"),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "harm_tolerance": args.harm_tolerance,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "rng_seed": 20260819,
            "cluster_unit": "source setting and source-model seed",
            "stratification": "source setting",
            "cluster_count_per_method": 6,
        },
        "row_descriptive_interval": (
            "Clopper-Pearson intervals treat rows as independent and are retained "
            "only as descriptive summaries; cluster bootstrap is primary"
        ),
        "fixed_guard": "accept iff endpoint delta <= 0.05 and phi <= 0.20",
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_csv(Path(args.curve_csv), curves)
    _write_csv(Path(args.detail_csv), details)


if __name__ == "__main__":
    main()
