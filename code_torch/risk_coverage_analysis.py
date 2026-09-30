"""Risk--coverage analysis for label-free graph-TTA endpoint guards.

The input is an unguarded stress-surface JSON file.  A guard decision is
replayed from each candidate's stored label-free proxy trajectory; target
labels enter only through the already-computed offline accuracy gain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, rankdata, spearmanr


FIXED_DELTA = 0.05
FIXED_PHI = 0.20


def normalized_proxy_score(row: dict) -> float:
    return max(float(row["max_delta"]) / FIXED_DELTA, float(row["max_phi"]) / FIXED_PHI)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def operating_metrics(gains, accepted, harm_tolerance=0.01) -> dict:
    gains = np.asarray(gains, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    harmful = gains < -float(harm_tolerance)
    positive = gains > 0.0
    rejected = ~accepted
    downside = np.maximum(-gains, 0.0)
    positive_gain = np.maximum(gains, 0.0)
    deployed_gain = np.where(accepted, gains, 0.0)
    return {
        "n": int(len(gains)),
        "accepted": int(accepted.sum()),
        "rejected": int(rejected.sum()),
        "harmful": int(harmful.sum()),
        "positive": int(positive.sum()),
        "coverage": float(np.mean(accepted)) if len(gains) else None,
        "harm_recall": _safe_ratio(float(np.sum(rejected & harmful)), float(harmful.sum())),
        "harmful_continuation_rate": _safe_ratio(
            float(np.sum(accepted & harmful)), float(harmful.sum())
        ),
        "false_intervention_rate": _safe_ratio(
            float(np.sum(rejected & ~harmful)), float((~harmful).sum())
        ),
        "retained_positive_utility": _safe_ratio(
            float(np.sum(positive_gain * accepted)), float(np.sum(positive_gain))
        ),
        "candidate_downside_sum": float(np.sum(downside)),
        "residual_downside_sum": float(np.sum(downside * accepted)),
        "prevented_downside_fraction": _safe_ratio(
            float(np.sum(downside * rejected)), float(np.sum(downside))
        ),
        "available_positive_gain_sum": float(np.sum(positive_gain)),
        "foregone_positive_gain_sum": float(np.sum(positive_gain * rejected)),
        "mean_candidate_gain": float(np.mean(gains)) if len(gains) else None,
        "mean_deployed_gain": float(np.mean(deployed_gain)) if len(gains) else None,
        "mean_foregone_positive_gain": float(np.mean(positive_gain * rejected))
        if len(gains)
        else None,
        "worst_deployed_gain": float(np.min(deployed_gain)) if len(gains) else None,
    }


def binary_auc(labels, scores) -> float | None:
    """AUROC for high score = positive, using average ranks for ties."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    u = float(np.sum(ranks[labels]) - n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


def exact_binomial_interval(successes: int, trials: int, alpha=0.05):
    if trials == 0:
        return [None, None]
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return [lower, upper]


def _risk_coverage_curve(gains, scores, harm_tolerance=0.01, max_points=101):
    gains = np.asarray(gains, dtype=float)
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    if len(unique) > max_points - 2:
        unique = np.unique(np.quantile(unique, np.linspace(0, 1, max_points - 2)))
    thresholds = np.r_[-np.inf, unique, np.inf]
    curve = []
    for threshold in thresholds:
        accepted = scores <= threshold
        metrics = operating_metrics(gains, accepted, harm_tolerance)
        curve.append(
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
                "residual_downside_sum": metrics["residual_downside_sum"],
            }
        )
    return curve


def cluster_bootstrap(rows, metric_names, harm_tolerance=0.01, replicates=5000, seed=20260819):
    clusters = {}
    for row in rows:
        clusters.setdefault((row["dataset"], int(row["seed"])), []).append(row)
    keys = list(clusters)
    if len(keys) < 2:
        return {name: [None, None] for name in metric_names}
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in metric_names}
    for _ in range(int(replicates)):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        boot_rows = [row for index in chosen for row in clusters[keys[index]]]
        gains = np.asarray([row["source_relative_accuracy"] for row in boot_rows], dtype=float)
        accepted = np.asarray([normalized_proxy_score(row) <= 1.0 for row in boot_rows])
        metrics = operating_metrics(gains, accepted, harm_tolerance)
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
    records = [row for row in records if row.get("guard") == "unguarded"]
    adapters = sorted({row["adapter"] for row in records})
    summary = {}
    curve_rows = []
    for adapter in adapters:
        rows = [row for row in records if row["adapter"] == adapter]
        gains = np.asarray([row["source_relative_accuracy"] for row in rows], dtype=float)
        delta = np.asarray([row["max_delta"] for row in rows], dtype=float)
        phi = np.asarray([row["max_phi"] for row in rows], dtype=float)
        combined = np.asarray([normalized_proxy_score(row) for row in rows], dtype=float)
        harmful = gains < -float(harm_tolerance)
        accepted = combined <= 1.0
        fixed = operating_metrics(gains, accepted, harm_tolerance)
        fixed["coverage_row_descriptive_cp_95ci"] = exact_binomial_interval(
            fixed["accepted"], fixed["n"]
        )
        fixed["harm_recall_row_descriptive_cp_95ci"] = exact_binomial_interval(
            int(np.sum((~accepted) & harmful)), int(harmful.sum())
        )
        fixed["bootstrap_95ci"] = cluster_bootstrap(
            rows,
            (
                "coverage",
                "harm_recall",
                "harmful_continuation_rate",
                "retained_positive_utility",
                "prevented_downside_fraction",
                "mean_deployed_gain",
            ),
            harm_tolerance,
            bootstrap_replicates,
        )
        correlations = {}
        for name, values in (("max_delta", delta), ("max_phi", phi), ("normalized_max", combined)):
            pearson = float(np.corrcoef(gains, values)[0, 1]) if len(gains) > 1 else None
            spearman = spearmanr(gains, values).statistic if len(gains) > 1 else None
            correlations[name] = {
                "gain_pearson": pearson,
                "gain_spearman": float(spearman) if np.isfinite(spearman) else None,
                "harm_auroc": binary_auc(harmful, values),
            }
            for point in _risk_coverage_curve(gains, values, harm_tolerance):
                curve_rows.append({"adapter": adapter, "score": name, **point})
        summary[adapter] = {
            "fixed_guard": fixed,
            "score_diagnostics": correlations,
            "dataset_seed_clusters": len({(row["dataset"], row["seed"]) for row in rows}),
        }
    return summary, curve_rows


def write_curve_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["adapter", "score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--curve-csv", required=True)
    parser.add_argument("--harm-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    input_path = Path(args.input)
    input_bytes = input_path.read_bytes()
    payload = json.loads(input_bytes.decode("utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Input audit is not complete: status={payload.get('status')!r}")
    records = payload.get("records", [])
    if not records:
        raise ValueError("Input audit contains no records")
    summary, curves = analyze(
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
            "cluster_unit": "dataset and source seed",
        },
        "row_descriptive_interval": (
            "Clopper-Pearson intervals treat rows as independent and are retained "
            "only as descriptive summaries; cluster bootstrap is primary"
        ),
        "fixed_guard": {
            "reject_if": "max_delta > 0.05 OR max_phi > 0.20",
            "normalized_accept_threshold": 1.0,
        },
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_curve_csv(Path(args.curve_csv), curves)


if __name__ == "__main__":
    main()
