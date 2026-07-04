import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "results" / "summary.csv"
DEFAULT_PUBLIC_SUMMARY_PATH = ROOT / "results" / "public_benchmark" / "summary.csv"
LEGACY_PUBLIC_SUMMARY_PATH = ROOT / "results" / "public_cora_quick" / "summary.csv"
FIGURE_DIR = ROOT / "paper" / "mypaper" / "figures"

METHODS = ["source_only", "entropy_all_nodes", "full_method"]
METHOD_COLORS = {
    "source_only": "#7A7A7A",
    "entropy_all_nodes": "#4E79A7",
    "full_method": "#2F6B3B",
    "no_neighborhood_agreement": "#8E6C8A",
    "no_structural_stability": "#5F8F8A",
    "no_calibration_loss": "#A9794A",
    "no_anti_forgetting": "#B85C5C",
}
METHOD_HATCHES = {
    "source_only": "",
    "entropy_all_nodes": "//",
    "full_method": "",
    "no_neighborhood_agreement": "..",
    "no_structural_stability": "xx",
    "no_calibration_loss": "\\\\",
    "no_anti_forgetting": "oo",
}
METHOD_LABELS = {
    "source_only": "Source",
    "entropy_all_nodes": "Entropy",
    "full_method": "Full",
    "no_neighborhood_agreement": "No agr.",
    "no_structural_stability": "No stab.",
    "no_calibration_loss": "No cal.",
    "no_anti_forgetting": "No AF",
}
CONDITION_ORDER = {
    "clean": 0,
    "feature_noise": 1,
    "edge_drop": 2,
    "edge_add": 3,
    "degree_shift": 4,
    "homophily_shift": 5,
}
DATASET_ORDER = [
    "cora",
    "citeseer",
    "pubmed",
    "amazon_computers",
    "amazon_photo",
    "coauthor_cs",
    "coauthor_physics",
]


def read_summary(path, dataset_fallback="synthetic"):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if not row.get("dataset"):
            row["dataset"] = dataset_fallback
    return rows


def fmean(row, metric):
    return float(row[f"{metric}_mean"])


def fstd(row, metric):
    return float(row[f"{metric}_std"])


def condition_label(shift, intensity):
    names = {
        "clean": "Clean",
        "feature_noise": "Feat",
        "edge_drop": "Drop",
        "edge_add": "Add",
        "degree_shift": "Degree",
        "homophily_shift": "Homo",
    }
    if shift == "clean":
        return names[shift]
    return f"{names.get(shift, shift)}\n{float(intensity):.2f}"


def dataset_sort_key(name):
    return DATASET_ORDER.index(name) if name in DATASET_ORDER else len(DATASET_ORDER)


def style_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.22, linestyle="--", linewidth=0.7)


def apply_bar_style(bars, methods):
    for bar, method in zip(bars, methods):
        bar.set_edgecolor("#303030")
        bar.set_linewidth(0.7)
        bar.set_hatch(METHOD_HATCHES.get(method, ""))


def add_bar_labels(ax, bars, values, digits=3, signed=False, padding=3):
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    labels = [fmt.format(float(v)) if np.isfinite(v) else "" for v in values]
    ax.bar_label(bars, labels=labels, padding=padding, fontsize=8)


def set_limits_with_padding(ax, values, errors=None, vertical=True, symmetric=False, min_pad=0.01):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    errors = np.zeros_like(values) if errors is None else np.asarray(errors, dtype=float)
    err_finite = errors[np.isfinite(values)] if errors.size == values.size else np.zeros_like(finite)
    low = float(np.min(finite - err_finite))
    high = float(np.max(finite + err_finite))
    if symmetric:
        bound = max(abs(low), abs(high), min_pad)
        pad = max(0.25 * bound, min_pad)
        if vertical:
            ax.set_ylim(-bound - pad, bound + pad)
        else:
            ax.set_xlim(-bound - pad, bound + pad)
        return
    pad = max(0.12 * max(high - low, min_pad), min_pad)
    low = max(0.0, low - 0.25 * pad)
    high = high + pad
    if vertical:
        ax.set_ylim(low, high)
    else:
        ax.set_xlim(low, high)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.7)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate(rows, methods=None, metrics=None):
    methods = methods or []
    metrics = metrics or []
    out = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        out[method] = {}
        for metric in metrics:
            vals = [fmean(row, metric) for row in subset if row.get(f"{metric}_mean") not in (None, "")]
            if vals:
                arr = np.asarray(vals, dtype=float)
                out[method][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
    return out


def plot_synthetic_overview(rows, out_dir):
    rows = [row for row in rows if row.get("dataset", "synthetic") == "synthetic" and row["method"] in METHODS]
    conds = sorted({(row["shift"], row["intensity"]) for row in rows}, key=lambda item: (CONDITION_ORDER.get(item[0], 999), float(item[1])))
    by = {(row["shift"], row["intensity"], row["method"]): row for row in rows}
    labels = [condition_label(shift, intensity) for shift, intensity in conds]
    x = np.arange(len(conds), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, metric, ylabel in [(axes[0], "accuracy", "Accuracy"), (axes[1], "ece", "ECE")]:
        for method in METHODS:
            vals = []
            stds = []
            for shift, intensity in conds:
                row = by.get((shift, intensity, method))
                vals.append(fmean(row, metric) if row is not None else np.nan)
                stds.append(fstd(row, metric) if row is not None else 0.0)
            vals = np.asarray(vals, dtype=float)
            stds = np.asarray(stds, dtype=float)
            ax.plot(x, vals, marker="o", linewidth=2.0, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(x, vals - stds, vals + stds, color=METHOD_COLORS[method], alpha=0.15)
        ax.set_ylabel(ylabel)
        style_axes(ax, grid_axis="y")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].legend(ncol=3, frameon=False, loc="lower left")
    axes[0].set_title("Controlled synthetic shifts")
    save_figure(fig, out_dir / "synthetic_overview.pdf")


def plot_synthetic_tradeoff(rows, out_dir):
    rows = [row for row in rows if row.get("dataset", "synthetic") == "synthetic"]
    methods = [
        "source_only",
        "entropy_all_nodes",
        "full_method",
        "no_neighborhood_agreement",
        "no_structural_stability",
        "no_calibration_loss",
        "no_anti_forgetting",
    ]
    agg = aggregate(rows, methods=methods, metrics=["accuracy", "ece", "runtime_seconds", "mean_reliability", "selected_fraction"])
    ordered = [method for method in methods if method in agg]
    labels = [METHOD_LABELS[method] for method in ordered]
    y = np.arange(len(ordered), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    metrics = [
        ("accuracy", "Accuracy (higher better)"),
        ("ece", "ECE (lower better)"),
        ("runtime_seconds", "Runtime (s, lower better)"),
    ]
    for ax, (metric, title) in zip(axes.flat[:3], metrics):
        vals = [agg[method][metric]["mean"] for method in ordered]
        stds = [agg[method][metric]["std"] for method in ordered]
        bars = ax.barh(
            y,
            vals,
            xerr=stds,
            color=[METHOD_COLORS[method] for method in ordered],
            alpha=0.94,
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "ecolor": "#4F4F4F"},
        )
        apply_bar_style(bars, ordered)
        ax.axvline(agg["source_only"][metric]["mean"], color=METHOD_COLORS["source_only"], linestyle=(0, (4, 2)), linewidth=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        style_axes(ax, grid_axis="x")
        set_limits_with_padding(ax, vals, errors=stds, vertical=False, min_pad=0.005 if metric != "runtime_seconds" else 0.02)
        add_bar_labels(ax, bars, vals, digits=3, signed=False, padding=3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    rel_methods = [method for method in ordered if method not in {"source_only", "entropy_all_nodes"} and "mean_reliability" in agg[method]]
    rel_y = np.arange(len(rel_methods), dtype=float)
    width = 0.38
    reliability_vals = [agg[m]["mean_reliability"]["mean"] for m in rel_methods]
    reliability_stds = [agg[m]["mean_reliability"]["std"] for m in rel_methods]
    selection_vals = [agg[m]["selected_fraction"]["mean"] for m in rel_methods]
    selection_stds = [agg[m]["selected_fraction"]["std"] for m in rel_methods]
    rel_bars = axes[1, 1].barh(
        rel_y - width / 2.0,
        reliability_vals,
        xerr=reliability_stds,
        height=width,
        color="#5A9367",
        error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "ecolor": "#4F4F4F"},
        label="Mean reliability",
    )
    sel_bars = axes[1, 1].barh(
        rel_y + width / 2.0,
        selection_vals,
        xerr=selection_stds,
        height=width,
        color="#4E79A7",
        error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8, "ecolor": "#4F4F4F"},
        label="Selected fraction",
    )
    axes[1, 1].set_yticks(rel_y)
    axes[1, 1].set_yticklabels([METHOD_LABELS[m] for m in rel_methods])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Reliability profile")
    axes[1, 1].set_xlim(0.0, 1.02)
    style_axes(axes[1, 1], grid_axis="x")
    axes[1, 1].legend(frameon=False)
    add_bar_labels(axes[1, 1], rel_bars, reliability_vals, digits=2, signed=False, padding=3)
    add_bar_labels(axes[1, 1], sel_bars, selection_vals, digits=2, signed=False, padding=3)
    save_figure(fig, out_dir / "synthetic_tradeoff.pdf")


def public_benchmark_rows(rows):
    rows = [row for row in rows if row["method"] in METHODS]
    out = []
    datasets = sorted({row["dataset"] for row in rows}, key=dataset_sort_key)
    for dataset in datasets:
        for method in METHODS:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            clean = [row for row in subset if row["shift"] == "clean"]
            shifted = [row for row in subset if row["shift"] != "clean"]
            if not clean or not shifted:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "clean_accuracy": float(sum(fmean(row, "accuracy") for row in clean) / len(clean)),
                    "shifted_accuracy": float(sum(fmean(row, "accuracy") for row in shifted) / len(shifted)),
                    "shifted_ece": float(sum(fmean(row, "ece") for row in shifted) / len(shifted)),
                }
            )
    return out


def plot_public_benchmark(rows, out_dir):
    bench_rows = public_benchmark_rows(rows)
    if not bench_rows:
        return
    datasets = sorted({row["dataset"] for row in bench_rows}, key=dataset_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex="col")
    width = 0.24
    x = np.arange(len(datasets), dtype=float)
    method_rows = {method: {row["dataset"]: row for row in bench_rows if row["method"] == method} for method in METHODS}
    for idx, method in enumerate(METHODS):
        subset = method_rows[method]
        acc_vals = [subset[dataset]["shifted_accuracy"] if dataset in subset else np.nan for dataset in datasets]
        ece_vals = [subset[dataset]["shifted_ece"] if dataset in subset else np.nan for dataset in datasets]
        offset = (idx - 1) * width
        acc_bars = axes[0, 0].bar(x + offset, acc_vals, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method], alpha=0.94)
        ece_bars = axes[0, 1].bar(x + offset, ece_vals, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method], alpha=0.94)
        apply_bar_style(acc_bars, [method] * len(datasets))
        apply_bar_style(ece_bars, [method] * len(datasets))
    axes[0, 0].set_title("Shifted accuracy")
    axes[0, 1].set_title("Shifted ECE")
    style_axes(axes[0, 0], grid_axis="y")
    style_axes(axes[0, 1], grid_axis="y")
    acc_all = [method_rows[m][dataset]["shifted_accuracy"] for m in METHODS for dataset in datasets if dataset in method_rows[m]]
    ece_all = [method_rows[m][dataset]["shifted_ece"] for m in METHODS for dataset in datasets if dataset in method_rows[m]]
    set_limits_with_padding(axes[0, 0], acc_all, vertical=True, min_pad=0.01)
    set_limits_with_padding(axes[0, 1], ece_all, vertical=True, min_pad=0.005)

    delta_methods = ["entropy_all_nodes", "full_method"]
    delta_width = 0.28
    source_rows = method_rows["source_only"]
    for idx, method in enumerate(delta_methods):
        subset = method_rows[method]
        delta_acc = [subset[dataset]["shifted_accuracy"] - source_rows[dataset]["shifted_accuracy"] if dataset in subset and dataset in source_rows else np.nan for dataset in datasets]
        delta_ece = [subset[dataset]["shifted_ece"] - source_rows[dataset]["shifted_ece"] if dataset in subset and dataset in source_rows else np.nan for dataset in datasets]
        offset = (idx - 0.5) * delta_width
        acc_bars = axes[1, 0].bar(x + offset, delta_acc, width=delta_width, color=METHOD_COLORS[method], alpha=0.94, label=METHOD_LABELS[method])
        ece_bars = axes[1, 1].bar(x + offset, delta_ece, width=delta_width, color=METHOD_COLORS[method], alpha=0.94, label=METHOD_LABELS[method])
        apply_bar_style(acc_bars, [method] * len(datasets))
        apply_bar_style(ece_bars, [method] * len(datasets))
        add_bar_labels(axes[1, 0], acc_bars, delta_acc, digits=3, signed=True, padding=2)
        add_bar_labels(axes[1, 1], ece_bars, delta_ece, digits=3, signed=True, padding=2)
    axes[1, 0].set_title("$\\Delta$ accuracy vs. source")
    axes[1, 1].set_title("$\\Delta$ ECE vs. source")
    for ax in axes[1]:
        ax.axhline(0.0, color="#505050", linewidth=1.0, linestyle=(0, (4, 2)))
        style_axes(ax, grid_axis="y")
    delta_acc_all = [
        method_rows[method][dataset]["shifted_accuracy"] - source_rows[dataset]["shifted_accuracy"]
        for method in delta_methods
        for dataset in datasets
        if dataset in method_rows[method] and dataset in source_rows
    ]
    delta_ece_all = [
        method_rows[method][dataset]["shifted_ece"] - source_rows[dataset]["shifted_ece"]
        for method in delta_methods
        for dataset in datasets
        if dataset in method_rows[method] and dataset in source_rows
    ]
    set_limits_with_padding(axes[1, 0], delta_acc_all, vertical=True, symmetric=True, min_pad=0.002)
    set_limits_with_padding(axes[1, 1], delta_ece_all, vertical=True, symmetric=True, min_pad=0.002)

    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([dataset.replace("_", "-") for dataset in datasets], rotation=20, ha="right")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    save_figure(fig, out_dir / "public_benchmark.pdf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--public-summary", default=str(DEFAULT_PUBLIC_SUMMARY_PATH))
    parser.add_argument("--out-dir", default=str(FIGURE_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    synthetic_rows = read_summary(args.summary, dataset_fallback="synthetic")
    public_path = Path(args.public_summary)
    if not public_path.exists() and public_path == DEFAULT_PUBLIC_SUMMARY_PATH and LEGACY_PUBLIC_SUMMARY_PATH.exists():
        public_path = LEGACY_PUBLIC_SUMMARY_PATH
    public_rows = read_summary(public_path, dataset_fallback="cora") if public_path.exists() else []
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_synthetic_overview(synthetic_rows, out_dir)
    plot_synthetic_tradeoff(synthetic_rows, out_dir)
    plot_public_benchmark(public_rows, out_dir)
    print(f"Updated figures in {out_dir}")


if __name__ == "__main__":
    main()
