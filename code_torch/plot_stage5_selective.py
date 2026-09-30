"""Render the frozen Stage 5 selective-deployment risk--coverage audit.

This module only reads the exported audit CSV.  It does not recompute scores,
select thresholds, or access labels beyond the descriptive quantities already
stored in that frozen artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "revision_2026-09-02_stage5_redesign"
    / "selective_audit"
    / "risk_coverage_curves.csv"
)
DEFAULT_MANIFEST = DEFAULT_INPUT.with_name("analysis_manifest.json")
DEFAULT_FIXED_GATE = DEFAULT_INPUT.with_name("fixed_gate_observed.csv")
DEFAULT_OUTPUT = ROOT / "paper" / "mypaper" / "figures_revision"

ADAPTERS = (
    "confidence_source_entropy",
    "uniform_entropy",
)
POLICIES = (
    "combined_score",
    "delta_only",
    "phi_only",
    "random_expectation",
    "oracle_gain",
)
NUMERIC_COLUMNS = (
    "coverage",
    "requested_coverage",
    "accepted_harm_rate",
    "retained_positive_utility",
)
REQUIRED_COLUMNS = {
    "adapter",
    "policy",
    "point_type",
    *NUMERIC_COLUMNS,
}
PROVENANCE_COLUMNS = (
    "heldout_input_sha256",
    "official_input_sha256",
    "analysis_code_sha256",
    "inferential_status",
)
EXPECTED_INFERENTIAL_STATUS = "descriptive_post_hoc_curve"

ADAPTER_LABELS = {
    "confidence_source_entropy": "Weighted entropy",
    "uniform_entropy": "Uniform entropy",
}
POLICY_STYLE = {
    "combined_score": {
        "label": r"Combined $S$",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.9,
    },
    "delta_only": {
        "label": r"Drift only $\Delta$",
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 1.45,
    },
    "phi_only": {
        "label": r"Turnover only $\Phi$",
        "color": "#009E73",
        "linestyle": "-",
        "linewidth": 1.45,
    },
    "random_expectation": {
        "label": "Random expectation",
        "color": "#777777",
        "linestyle": "--",
        "linewidth": 1.15,
    },
    "oracle_gain": {
        "label": "Label oracle (reference)",
        "color": "#222222",
        "linestyle": ":",
        "linewidth": 1.15,
    },
}


def _optional_float(value: Any, *, column: str, row_number: int) -> float | None:
    """Parse an optional finite float while preserving undefined values."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: {column!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Row {row_number}: {column!r} must be finite or explicitly null"
        )
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_provenance_contract(
    manifest_path: Path, curve_path: Path, fixed_gate_path: Path
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = manifest["inputs"]
        expected = {
            "heldout_input_sha256": str(inputs[0]["sha256"]),
            "official_input_sha256": str(inputs[1]["sha256"]),
            "analysis_code_sha256": str(manifest["analysis_code"]["sha256"]),
            "inferential_status": EXPECTED_INFERENTIAL_STATUS,
        }
        output_hashes: dict[str, str] = {}
        for output in manifest["outputs"]:
            name = Path(str(output["path"])).name
            digest = str(output["sha256"]).lower()
            if name in output_hashes or len(digest) != 64:
                raise ValueError("invalid or duplicate output digest")
            output_hashes[name] = digest
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid selective-audit manifest: {manifest_path}") from exc

    for artifact_path in (curve_path, fixed_gate_path):
        expected_digest = output_hashes.get(artifact_path.name)
        if expected_digest is None:
            raise ValueError(
                f"Selective-audit manifest does not bind {artifact_path.name}"
            )
        try:
            actual_digest = _sha256(artifact_path)
        except OSError as exc:
            raise ValueError(f"Cannot read selective-audit output: {artifact_path}") from exc
        if actual_digest != expected_digest:
            raise ValueError(
                f"Selective-audit output digest disagrees for {artifact_path.name}"
            )

    with fixed_gate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "adapter",
            "policy",
            "coverage",
            "accepted_harm_rate",
            "retained_positive_utility",
            "heldout_input_sha256",
            "official_input_sha256",
            "analysis_code_sha256",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Fixed-gate artifact is missing columns: {sorted(missing)}")
        fixed_rows = list(reader)

    points: dict[str, dict[str, float]] = {}
    for row_number, row in enumerate(fixed_rows, start=2):
        adapter = str(row["adapter"]).strip()
        if adapter not in ADAPTERS or adapter in points:
            raise ValueError(f"Row {row_number}: invalid fixed-gate adapter {adapter!r}")
        if row["policy"] != "fixed_rectangular_gate_observed":
            raise ValueError(f"Row {row_number}: unexpected fixed-gate policy")
        for provenance in PROVENANCE_COLUMNS[:3]:
            if str(row[provenance]).strip() != expected[provenance]:
                raise ValueError(
                    f"Row {row_number}: fixed-gate {provenance} disagrees with manifest"
                )
        values: dict[str, float] = {}
        for metric in ("coverage", "accepted_harm_rate", "retained_positive_utility"):
            value = _optional_float(row[metric], column=metric, row_number=row_number)
            if value is None:
                raise ValueError(f"Row {row_number}: fixed-gate {metric} is missing")
            values[metric] = value
        points[adapter] = values
    if set(points) != set(ADAPTERS):
        raise ValueError("Fixed-gate artifact does not contain exactly one row per adapter")
    return expected, points


def load_curve_rows(
    path: Path,
    *,
    manifest_path: Path | None = None,
    fixed_gate_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and provenance-bind frozen curve rows without filling null risks."""
    manifest_path = manifest_path or path.with_name(DEFAULT_MANIFEST.name)
    fixed_gate_path = fixed_gate_path or path.with_name(DEFAULT_FIXED_GATE.name)
    expected_provenance, fixed_points = _load_provenance_contract(
        manifest_path, path, fixed_gate_path
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = (REQUIRED_COLUMNS | set(PROVENANCE_COLUMNS)) - fieldnames
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError("Frozen risk--coverage artifact is empty")
    for column in PROVENANCE_COLUMNS:
        observed = {str(row[column]).strip() for row in raw_rows}
        if "" in observed or len(observed) != 1:
            raise ValueError(f"Mixed or missing provenance in column {column!r}")
        if observed != {expected_provenance[column]}:
            raise ValueError(f"Curve {column} disagrees with the frozen manifest")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        row: dict[str, Any] = dict(raw)
        for column in NUMERIC_COLUMNS:
            row[column] = _optional_float(
                raw.get(column), column=column, row_number=row_number
            )

        adapter = row["adapter"]
        policy = row["policy"]
        point_type = row["point_type"]
        if adapter not in ADAPTERS:
            raise ValueError(f"Row {row_number}: unknown adapter {adapter!r}")
        if policy not in POLICIES:
            raise ValueError(f"Row {row_number}: unknown policy {policy!r}")
        if point_type not in {"grid", "fixed_gate_matched"}:
            raise ValueError(f"Row {row_number}: unknown point type {point_type!r}")

        coverage = row["coverage"]
        if coverage is None or not 0.0 <= coverage <= 1.0:
            raise ValueError(f"Row {row_number}: coverage must be in [0, 1]")
        for column in ("accepted_harm_rate", "retained_positive_utility"):
            value = row[column]
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"Row {row_number}: {column} must be in [0, 1]")

        harm_rate = row["accepted_harm_rate"]
        if coverage == 0.0 and harm_rate is not None:
            raise ValueError(
                f"Row {row_number}: accepted-set harm risk is undefined at zero coverage"
            )
        if coverage > 0.0 and harm_rate is None:
            raise ValueError(
                f"Row {row_number}: accepted-set harm risk is missing at positive coverage"
            )

        identity = (adapter, policy, point_type, coverage)
        if identity in seen:
            raise ValueError(f"Row {row_number}: duplicate curve identity {identity!r}")
        seen.add(identity)
        rows.append(row)

    expected_grid = {(adapter, policy) for adapter in ADAPTERS for policy in POLICIES}
    observed_grid = {
        (row["adapter"], row["policy"])
        for row in rows
        if row["point_type"] == "grid"
    }
    if observed_grid != expected_grid:
        raise ValueError("The grid does not contain every adapter-policy pair")
    expected_coverages = [index / 20.0 for index in range(21)]
    for adapter, policy in sorted(expected_grid):
        grid_coverages = sorted(
            row["coverage"]
            for row in rows
            if row["adapter"] == adapter
            and row["policy"] == policy
            and row["point_type"] == "grid"
        )
        if len(grid_coverages) != len(expected_coverages) or any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
            for observed, expected in zip(grid_coverages, expected_coverages)
        ):
            raise ValueError(
                f"Unexpected coverage grid for adapter={adapter}, policy={policy}"
            )
        fixed_count = sum(
            row["adapter"] == adapter
            and row["policy"] == policy
            and row["point_type"] == "fixed_gate_matched"
            for row in rows
        )
        if fixed_count != 1:
            raise ValueError(
                f"Expected one matched fixed-gate row for adapter={adapter}, "
                f"policy={policy}"
            )

    for adapter, expected_point in fixed_points.items():
        matches = [
            row
            for row in rows
            if row["adapter"] == adapter
            and row["policy"] == "combined_score"
            and row["point_type"] == "fixed_gate_matched"
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing combined fixed-gate point for {adapter}")
        for metric, expected_value in expected_point.items():
            observed_value = matches[0][metric]
            if observed_value is None or not math.isclose(
                observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Curve fixed-gate {metric} disagrees with frozen observation "
                    f"for {adapter}"
                )
    return rows


def curve_points(
    rows: Iterable[dict[str, Any]],
    *,
    adapter: str,
    policy: str,
    metric: str,
) -> tuple[list[float], list[float]]:
    """Return sorted grid points, omitting metrics that are explicitly null."""
    points = sorted(
        (
            (row["coverage"], row[metric])
            for row in rows
            if row["adapter"] == adapter
            and row["policy"] == policy
            and row["point_type"] == "grid"
            and row.get(metric) is not None
        ),
        key=lambda item: item[0],
    )
    return [point[0] for point in points], [point[1] for point in points]


def _fixed_gate_point(
    rows: Iterable[dict[str, Any]], *, adapter: str, metric: str
) -> tuple[float, float]:
    matches = [
        row
        for row in rows
        if row["adapter"] == adapter
        and row["policy"] == "combined_score"
        and row["point_type"] == "fixed_gate_matched"
    ]
    if len(matches) != 1 or matches[0].get(metric) is None:
        raise ValueError(f"Expected one finite fixed-gate point for {adapter}/{metric}")
    return matches[0]["coverage"], matches[0][metric]


def _style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#d9d9d9", linewidth=0.55, alpha=0.8)
    axis.tick_params(width=0.65, length=3)


def make_figure(rows: list[dict[str, Any]]) -> plt.Figure:
    """Create the two-adapter risk--utility figure."""
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharex=True)
    metrics = (
        ("accepted_harm_rate", "Accepted-set harm risk"),
        ("retained_positive_utility", "Retained positive utility"),
    )

    for row_index, adapter in enumerate(ADAPTERS):
        for column_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for policy in POLICIES:
                x_values, y_values = curve_points(
                    rows, adapter=adapter, policy=policy, metric=metric
                )
                style = POLICY_STYLE[policy]
                axis.plot(
                    x_values,
                    y_values,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=style["label"],
                    zorder=2,
                )

            fixed_x, fixed_y = _fixed_gate_point(rows, adapter=adapter, metric=metric)
            axis.scatter(
                [fixed_x],
                [fixed_y],
                marker="D",
                s=27,
                facecolor="white",
                edgecolor="#0072B2",
                linewidth=1.1,
                zorder=4,
                label="Frozen gate" if row_index == 0 and column_index == 0 else None,
            )
            axis.set_ylim(-0.025, 1.025)
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"({chr(97 + 2 * row_index + column_index)}) "
                f"{ADAPTER_LABELS[adapter]}",
                fontsize=9.1,
            )
            _style_axis(axis)

    axes[1, 0].set_xlabel("Matched deployment coverage")
    axes[1, 1].set_xlabel("Matched deployment coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0), h_pad=1.05, w_pad=1.15)
    return fig


def render(
    input_path: Path,
    output_dir: Path,
    stem: str,
    *,
    manifest_path: Path | None = None,
    fixed_gate_path: Path | None = None,
) -> tuple[Path, Path]:
    """Validate the CSV and write deterministic PDF/PNG views."""
    rows = load_curve_rows(
        input_path,
        manifest_path=manifest_path,
        fixed_gate_path=fixed_gate_path,
    )
    figure = make_figure(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=260, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixed-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default="risk_coverage_stage5")
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    pdf_path, png_path = render(
        args.input,
        args.output_dir,
        args.stem,
        manifest_path=args.manifest,
        fixed_gate_path=args.fixed_gate,
    )
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
