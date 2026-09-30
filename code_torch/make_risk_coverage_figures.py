"""Create manuscript figures from the frozen risk--coverage audit artifacts.

The script is intentionally read-only with respect to experimental results.  It
loads the completed JSON/CSV files and writes PDF/PNG figures; it does not
recompute candidates, tune thresholds, or consult target labels beyond the
offline gains already stored in the audit records.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "revision_2026-08-19" / "heldout_risk_coverage_v3"
OFFICIAL_DIR = (
    ROOT
    / "revision_2026-08-19"
    / "official_tsa_default_label_isolated_audit_v2"
)
OUTPUT_DIR = ROOT / "paper" / "mypaper" / "figures_revision"

DISPLAY_NAMES = {
    "confidence_source_entropy": "Weighted entropy",
    "uniform_entropy": "Uniform entropy",
    "T3A": "T3A",
    "Matcha_T3A": "Matcha–T3A",
    "TSA_T3A": "TSA–T3A",
}

COLORS = {
    "confidence_source_entropy": "#0072B2",
    "uniform_entropy": "#D55E00",
    "T3A": "#009E73",
    "Matcha_T3A": "#CC79A7",
    "TSA_T3A": "#E69F00",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise RuntimeError(f"Audit is not complete: {path}")
    return payload["records"]


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)


def make_gain_score_figure() -> None:
    heldout = _load_records(HELDOUT_DIR / "results.json")
    official = _load_records(OFFICIAL_DIR / "results.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=False)
    panels = [
        (
            axes[0],
            heldout,
            "adapter",
            lambda row: max(row["max_delta"] / 0.05, row["max_phi"] / 0.20),
            "(a) Held-out trajectory audit",
        ),
        (
            axes[1],
            official,
            "method",
            lambda row: max(row["delta"] / 0.05, row["phi"] / 0.20),
            "(b) Official endpoint audit",
        ),
    ]

    for ax, records, group_key, score_fn, title in panels:
        for group in sorted({row[group_key] for row in records}):
            subset = [
                row
                for row in records
                if row[group_key] == group
                and row.get("source_relative_accuracy") is not None
                and (
                    group_key != "method"
                    or (
                        row.get("delta") is not None
                        and row.get("phi") is not None
                    )
                )
            ]
            score = np.asarray([score_fn(row) for row in subset])
            gain_pp = 100 * np.asarray(
                [row["source_relative_accuracy"] for row in subset]
            )
            harmful = gain_pp < -1.0
            color = COLORS[group]
            ax.scatter(
                score[~harmful],
                gain_pp[~harmful],
                s=19,
                c=color,
                alpha=0.72,
                linewidths=0,
                label=DISPLAY_NAMES[group],
            )
            ax.scatter(
                score[harmful],
                gain_pp[harmful],
                s=23,
                facecolors="none",
                edgecolors=color,
                alpha=0.85,
                linewidths=0.8,
            )

        ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#777777", linewidth=0.7)
        ax.axhline(-1.0, color="#AA0000", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"Normalized gate score $S$ (accept if $S\leq 1$)")
        ax.set_ylabel("Candidate gain (percentage points)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.4, loc="lower left")
        _style_axes(ax)

    axes[1].text(
        0.98,
        0.98,
        "Matcha–T3A: 5/24 non-finite\nfail-closed candidates omitted",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    fig.text(
        0.5,
        -0.015,
        "Open markers denote materially harmful candidates (gain < -1 pp).",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(w_pad=1.2)
    _save(fig, "paired_gain_vs_score")


def _load_curve_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = dict(row)
        for key in (
            "threshold",
            "coverage",
            "harmful_continuation_rate",
            "retained_positive_utility",
            "mean_deployed_gain",
        ):
            if key in converted and converted[key] not in ("", None):
                converted[key] = float(converted[key])
        parsed.append(converted)
    return parsed


def _plot_curve_family(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    group_key: str,
    score_key: str,
    score_name: str,
    y_key: str,
    title: str,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[score_key] == score_name:
            grouped[row[group_key]].append(row)

    for group, group_rows in sorted(grouped.items()):
        points = sorted(
            (
                (row["coverage"], row[y_key])
                for row in group_rows
                if isinstance(row.get("coverage"), float)
                and isinstance(row.get(y_key), float)
            ),
            key=lambda item: item[0],
        )
        coverage = [point[0] for point in points]
        values = [point[1] for point in points]
        ax.plot(
            coverage,
            values,
            color=COLORS[group],
            linewidth=1.5,
            label=DISPLAY_NAMES[group],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Coverage")
    ax.set_ylabel(
        "Harmful continuation" if y_key == "harmful_continuation_rate" else "Retained positive utility"
    )
    ax.set_title(title, fontsize=9.2)
    _style_axes(ax)


def make_risk_utility_curves() -> None:
    heldout = _load_curve_rows(HELDOUT_DIR / "curves.csv")
    official = _load_curve_rows(OFFICIAL_DIR / "curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)

    _plot_curve_family(
        axes[0, 0], heldout, "adapter", "score", "normalized_max",
        "harmful_continuation_rate", "(a) Held-out residual harm"
    )
    _plot_curve_family(
        axes[0, 1], heldout, "adapter", "score", "normalized_max",
        "retained_positive_utility", "(b) Held-out retained utility"
    )
    _plot_curve_family(
        axes[1, 0], official, "method", "score", "normalized_max",
        "harmful_continuation_rate", "(c) Official residual harm"
    )
    _plot_curve_family(
        axes[1, 1], official, "method", "score", "normalized_max",
        "retained_positive_utility", "(d) Official retained utility"
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles + handles2,
        labels + labels2,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.15, w_pad=1.0)
    _save(fig, "risk_utility_curves")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    make_gain_score_figure()
    make_risk_utility_curves()


if __name__ == "__main__":
    main()
