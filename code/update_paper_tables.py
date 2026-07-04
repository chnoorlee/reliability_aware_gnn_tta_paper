import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = ROOT / "results" / "summary.csv"
DEFAULT_PUBLIC_SUMMARY_PATH = ROOT / "results" / "public_benchmark" / "summary.csv"
DEFAULT_HETEROPHILY_SUMMARY_PATH = ROOT / "results" / "heterophily_benchmark" / "summary.csv"
DEFAULT_SIGNIFICANCE_PATH = ROOT / "results" / "significance.csv"
DEFAULT_HETEROPHILY_SIGNIFICANCE_PATH = ROOT / "results" / "heterophily_benchmark" / "significance.csv"
LEGACY_PUBLIC_SUMMARY_PATH = ROOT / "results" / "public_cora_quick" / "summary.csv"
RESULTS_TEX = ROOT / "paper" / "mypaper" / "sections" / "results.tex"
FIGURE_DIR = ROOT / "paper" / "mypaper" / "figures"

METHOD_LABELS = {
    "source_only": "Source only",
    "entropy_all_nodes": "Entropy all nodes",
    "tent_entropy": "Tent entropy",
    "eata_filter": "EATA-style filter",
    "graph_tta_consistency": "Graph consistency",
    "matcha_reliable": "Matcha-style mask",
    "reliable_entropy": "Reliable entropy",
    "full_method": "Full method",
    "no_neighborhood_agreement": "No neighborhood agreement",
    "no_structural_stability": "No structural stability",
    "no_calibration_loss": "No calibration loss",
    "no_anti_forgetting": "No anti-forgetting",
}

MAIN_METHODS = ["source_only", "entropy_all_nodes", "full_method"]
STRONG_BASELINE_METHODS = ["source_only", "entropy_all_nodes", "tent_entropy", "eata_filter", "graph_tta_consistency", "matcha_reliable", "full_method"]
ABLATION_METHODS = [
    "full_method",
    "no_neighborhood_agreement",
    "no_structural_stability",
    "no_calibration_loss",
    "no_anti_forgetting",
]
AGG_METHODS = ABLATION_METHODS + ["source_only", "entropy_all_nodes"]
CONDITION_ORDER = {
    "clean": 0,
    "feature_noise": 1,
    "edge_drop": 2,
    "edge_add": 3,
    "degree_shift": 4,
    "homophily_shift": 5,
}
DATASET_ORDER = [
    "synthetic",
    "texas",
    "cornell",
    "wisconsin",
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
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    for row in rows:
        if not row.get("dataset"):
            row["dataset"] = dataset_fallback
    return rows


def has_metric(row, metric):
    return row.get(f"{metric}_mean") not in (None, "")


def fmean(row, metric):
    return float(row[f"{metric}_mean"])


def fstd(row, metric):
    return float(row[f"{metric}_std"])


def metric_mean(rows, metric):
    vals = [fmean(row, metric) for row in rows if has_metric(row, metric)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pm(row, metric):
    return f"{fmean(row, metric):.4f}$\\pm${fstd(row, metric):.4f}"


def condition_label(shift, intensity):
    intensity = float(intensity)
    names = {
        "clean": "Clean",
        "feature_noise": "Feature noise",
        "edge_drop": "Edge drop",
        "edge_add": "Edge add",
        "degree_shift": "Degree shift",
        "homophily_shift": "Homophily shift",
    }
    base = names.get(shift, shift.replace("_", " ").title())
    if shift == "clean":
        return base
    return f"{base} ({intensity:.2f})"


def dataset_label(dataset):
    return dataset.replace("_", "-")


def dataset_order(dataset):
    return DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER)


def synthetic_sort_key(row):
    return (
        CONDITION_ORDER.get(row["shift"], 999),
        float(row["intensity"]),
        MAIN_METHODS.index(row["method"]) if row["method"] in MAIN_METHODS else 999,
        row["method"],
    )


def main_table(rows):
    lines = []
    for row in sorted([row for row in rows if row["method"] in MAIN_METHODS], key=synthetic_sort_key):
        lines.append(
            f"{condition_label(row['shift'], row['intensity'])} & {METHOD_LABELS[row['method']]} & "
            f"{pm(row, 'accuracy')} & {pm(row, 'macro_f1')} & {pm(row, 'ece')} & {pm(row, 'nll')} \\\\"
        )
    return "\n".join(lines)


def aggregate(rows, methods=None, metrics=None):
    methods = methods or AGG_METHODS
    metrics = metrics or ["accuracy", "ece", "brier", "runtime_seconds", "mean_reliability", "selected_fraction"]
    out = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        out[method] = {}
        for metric in metrics:
            mean_val = metric_mean(subset, metric)
            if mean_val is not None:
                out[method][metric] = mean_val
    return out


def ablation_table(rows):
    agg = aggregate(rows)
    lines = []
    for method in ABLATION_METHODS:
        if method not in agg:
            continue
        vals = agg[method]
        lines.append(
            f"{METHOD_LABELS[method]} & {vals['accuracy']:.4f} & {vals['ece']:.4f} & "
            f"{vals['brier']:.4f} & {vals['runtime_seconds']:.4f} \\\\"
        )
    return "\n".join(lines)


def compact_findings(rows):
    by = {(row["shift"], row["intensity"], row["method"]): row for row in rows}
    conds = sorted({(row["shift"], row["intensity"]) for row in rows}, key=lambda item: (CONDITION_ORDER.get(item[0], 999), float(item[1])))
    source_ece_gains = []
    source_acc_gains = []
    entropy_ece_gains = []
    entropy_acc_gains = []
    for shift, intensity in conds:
        needed = [
            (shift, intensity, "source_only"),
            (shift, intensity, "entropy_all_nodes"),
            (shift, intensity, "full_method"),
        ]
        if any(key not in by for key in needed):
            continue
        source = by[(shift, intensity, "source_only")]
        entropy = by[(shift, intensity, "entropy_all_nodes")]
        full = by[(shift, intensity, "full_method")]
        source_ece_gains.append(fmean(source, "ece") - fmean(full, "ece"))
        source_acc_gains.append(fmean(full, "accuracy") - fmean(source, "accuracy"))
        entropy_ece_gains.append(fmean(entropy, "ece") - fmean(full, "ece"))
        entropy_acc_gains.append(fmean(full, "accuracy") - fmean(entropy, "accuracy"))
    return {
        "mean_source_ece_gain": sum(source_ece_gains) / len(source_ece_gains) if source_ece_gains else 0.0,
        "mean_source_acc_gain": sum(source_acc_gains) / len(source_acc_gains) if source_acc_gains else 0.0,
        "mean_entropy_ece_gain": sum(entropy_ece_gains) / len(entropy_ece_gains) if entropy_ece_gains else 0.0,
        "mean_entropy_acc_gain": sum(entropy_acc_gains) / len(entropy_acc_gains) if entropy_acc_gains else 0.0,
    }


def find_row(rows, shift, intensity, method, dataset="synthetic"):
    for row in rows:
        if row.get("dataset", dataset) == dataset and row["shift"] == shift and abs(float(row["intensity"]) - float(intensity)) < 1e-12 and row["method"] == method:
            return row
    return None


def benchmark_rows(rows, methods=None):
    if not rows:
        return []
    methods = methods or MAIN_METHODS
    out = []
    datasets = sorted({row["dataset"] for row in rows}, key=dataset_order)
    for dataset in datasets:
        for method in methods:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            clean = [row for row in subset if row["shift"] == "clean"]
            shifted = [row for row in subset if row["shift"] != "clean"]
            if not clean or not shifted:
                continue
            out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "clean_accuracy": metric_mean(clean, "accuracy"),
                    "shifted_accuracy": metric_mean(shifted, "accuracy"),
                    "shifted_ece": metric_mean(shifted, "ece"),
                    "runtime_seconds": metric_mean(shifted, "runtime_seconds"),
                }
            )
    return out


def public_benchmark_rows(rows):
    return benchmark_rows(rows)


def public_benchmark_table(rows):
    lines = []
    for row in public_benchmark_rows(rows):
        lines.append(
            f"{dataset_label(row['dataset'])} & {METHOD_LABELS[row['method']]} & "
            f"{row['clean_accuracy']:.4f} & {row['shifted_accuracy']:.4f} & {row['shifted_ece']:.4f} & {row['runtime_seconds']:.4f} \\\\"
        )
    return "\n".join(lines)


def heterophily_benchmark_table(rows):
    lines = []
    for row in benchmark_rows(rows, methods=STRONG_BASELINE_METHODS):
        lines.append(
            f"{dataset_label(row['dataset'])} & {METHOD_LABELS[row['method']]} & "
            f"{row['clean_accuracy']:.4f} & {row['shifted_accuracy']:.4f} & {row['shifted_ece']:.4f} & {row['runtime_seconds']:.4f} \\\\"
        )
    return "\n".join(lines)


def heterophily_section(heterophily_rows):
    if not heterophily_rows:
        return ""
    table_tex = render_table(
        "table*",
        "width=\\textwidth",
        "llrrrr",
        "Dataset & Method & Clean Acc. & Shifted Acc. & Shifted ECE & Runtime (s)",
        heterophily_benchmark_table(heterophily_rows),
        "Heterophily stress-test summary. Texas, Cornell, and Wisconsin denote low-homophily analogue graphs in the dependency-light runner rather than official WebKB benchmark numbers unless an external WebKB loader is enabled.",
        "tab:heterophily-benchmark",
    )
    return rf"""
\subsection{{Heterophily stress tests}}

Table~\ref{{tab:heterophily-benchmark}} reports the low-homophily stress-test suite. These experiments are included to evaluate whether graph TTA variants over-rely on neighborhood label agreement. The setting should be interpreted as a controlled heterophily diagnostic rather than as an official WebKB leaderboard result. Accordingly, the main conclusion is qualitative: methods that depend strongly on local agreement should be treated cautiously when the graph has weak or adversarial homophily.

{table_tex}
"""


def significance_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def significance_table(rows, dataset_filter=None, max_rows=12):
    if dataset_filter is not None:
        rows = [row for row in rows if row.get("dataset") in dataset_filter]
    picked = []
    for row in rows:
        if row.get("method_a") != "full_method":
            continue
        if row.get("method_b") not in {"source_only", "entropy_all_nodes", "tent_entropy", "eata_filter", "graph_tta_consistency", "matcha_reliable"}:
            continue
        picked.append(row)
    picked = sorted(picked, key=lambda row: (dataset_order(row.get("dataset", "")), row.get("shift", ""), float(row.get("intensity", 0.0)), row.get("method_b", "")))[:max_rows]
    lines = []
    for row in picked:
        p = float(row.get("accuracy_p") or 1.0)
        marker = "$^*$" if p < 0.05 else ""
        lines.append(
            f"{dataset_label(row['dataset'])} & {condition_label(row['shift'], row['intensity'])} & Full vs. {METHOD_LABELS.get(row['method_b'], row['method_b'])} & "
            f"{float(row['accuracy_diff']):.4f}{marker} & {p:.4f} & {float(row['ece_diff']):.4f} \\\\"
        )
    return "\n".join(lines)


def significance_section(sig_rows, hetero_sig_rows):
    rows = (sig_rows or []) + (hetero_sig_rows or [])
    if not rows:
        return ""
    table_body = significance_table(rows)
    if not table_body:
        return ""
    table_tex = render_table(
        "table*",
        "width=\\textwidth",
        "lllrrr",
        "Dataset & Condition & Comparison & Acc. diff. & $p$ & ECE diff.",
        table_body,
        "Seed-matched paired significance summary. A positive accuracy difference favors the full method; $^*$ marks $p<0.05$ under the stored paired test output.",
        "tab:significance",
    )
    return rf"""
\subsection{{Seed-matched significance analysis}}

Table~\ref{{tab:significance}} summarizes representative paired tests generated from the stored per-seed result files. The table is intentionally interpreted together with effect sizes: statistically marked differences are not used to claim broad dominance unless the corresponding accuracy, calibration, and runtime trade-offs are also favorable.

{table_tex}
"""

def public_findings(rows):
    benchmark_rows = public_benchmark_rows(rows)
    by = {(row["dataset"], row["method"]): row for row in benchmark_rows}
    datasets = sorted({row["dataset"] for row in benchmark_rows}, key=dataset_order)
    source_ece_gains = []
    source_acc_gains = []
    entropy_ece_gains = []
    entropy_acc_gains = []
    for dataset in datasets:
        needed = [(dataset, "source_only"), (dataset, "entropy_all_nodes"), (dataset, "full_method")]
        if any(key not in by for key in needed):
            continue
        source = by[(dataset, "source_only")]
        entropy = by[(dataset, "entropy_all_nodes")]
        full = by[(dataset, "full_method")]
        source_ece_gains.append(source["shifted_ece"] - full["shifted_ece"])
        source_acc_gains.append(full["shifted_accuracy"] - source["shifted_accuracy"])
        entropy_ece_gains.append(entropy["shifted_ece"] - full["shifted_ece"])
        entropy_acc_gains.append(full["shifted_accuracy"] - entropy["shifted_accuracy"])
    return {
        "dataset_count": len(datasets),
        "mean_source_ece_gain": sum(source_ece_gains) / len(source_ece_gains) if source_ece_gains else 0.0,
        "mean_source_acc_gain": sum(source_acc_gains) / len(source_acc_gains) if source_acc_gains else 0.0,
        "mean_entropy_ece_gain": sum(entropy_ece_gains) / len(entropy_ece_gains) if entropy_ece_gains else 0.0,
        "mean_entropy_acc_gain": sum(entropy_acc_gains) / len(entropy_acc_gains) if entropy_acc_gains else 0.0,
    }


def render_table(env, width_spec, columns, header, body, caption, label):
    return rf"""
\begin{{{env}}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{{caption}}}
\label{{{label}}}
\begin{{adjustbox}}{{{width_spec}}}
\begin{{tabular}}{{{columns}}}
\toprule
{header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{{env}}}
"""


def figure_block(file_name, caption, label, width="\\textwidth"):
    if not (FIGURE_DIR / file_name).exists():
        return ""
    return rf"""
\begin{{figure}}[t]
\centering
\includegraphics[width={width}]{{figures/{file_name}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def public_benchmark_section(public_rows):
    if not public_rows:
        return ""
    findings = public_findings(public_rows)
    table_tex = render_table(
        "table*",
        "width=\\textwidth",
        "llrrrr",
        "Dataset & Method & Clean Acc. & Shifted Acc. & Shifted ECE & Runtime (s)",
        public_benchmark_table(public_rows),
        "Public-data benchmark summary. Each row reports clean accuracy together with averages over the completed shifted public-graph conditions for that dataset-method pair.",
        "tab:public-benchmark",
    )
    figure_tex = figure_block(
        "public_benchmark.pdf",
        "Public benchmark summary across the completed public datasets. The top row reports shifted accuracy and shifted ECE, while the bottom row highlights method deltas relative to the source baseline so that small public-data differences remain visually interpretable.",
        "fig:public-benchmark",
    )
    return rf"""
\subsection{{Public-data benchmark under sparse graph propagation}}

The revised public-data pipeline evaluates {findings['dataset_count']} citation-network datasets using sparse adjacency propagation and a reduced set of main adaptation variants. Table~\ref{{tab:public-benchmark}} summarizes clean accuracy together with averages over the completed shifted public conditions in the stored summary file. The clean source-only accuracies confirm that the earlier public-data loading and training failure has been resolved. Averaged over the available public datasets, the full method changes shifted accuracy by {findings['mean_source_acc_gain']:.4f} and shifted ECE by {-findings['mean_source_ece_gain']:.4f} relative to source-only. Relative to entropy adaptation, the full method changes shifted accuracy by {findings['mean_entropy_acc_gain']:.4f} and shifted ECE by {-findings['mean_entropy_ece_gain']:.4f}. The public benchmark therefore supports a measured conclusion: under the current mild public-graph shifts, reliability weighting preserves source-level predictive performance and produces small calibration changes, while its advantages over entropy adaptation remain limited and must be strengthened before making broad superiority claims.

{table_tex}
{figure_tex}
"""


def render_results(synthetic_rows, public_rows=None, heterophily_rows=None, sig_rows=None, hetero_sig_rows=None):
    public_rows = public_rows or []
    heterophily_rows = heterophily_rows or []
    sig_rows = sig_rows or []
    hetero_sig_rows = hetero_sig_rows or []
    synthetic_rows = [row for row in synthetic_rows if row.get("dataset", "synthetic") == "synthetic"]
    agg = aggregate(synthetic_rows)
    findings = compact_findings(synthetic_rows)
    severe_source = find_row(synthetic_rows, "homophily_shift", 0.50, "source_only")
    severe_entropy = find_row(synthetic_rows, "homophily_shift", 0.50, "entropy_all_nodes")
    severe_full = find_row(synthetic_rows, "homophily_shift", 0.50, "full_method")
    reliability_mean = agg.get("full_method", {}).get("mean_reliability")
    selected_fraction = agg.get("full_method", {}).get("selected_fraction")
    severe_text = ""
    if severe_source and severe_entropy and severe_full:
        severe_text = (
            f"The strongest failure mode occurs under severe homophily shift. When the homophily-shift intensity is 0.50, "
            f"the source-only model obtains {fmean(severe_source, 'accuracy'):.4f} accuracy and {fmean(severe_source, 'ece'):.4f} ECE, "
            f"whereas entropy minimization decreases accuracy to {fmean(severe_entropy, 'accuracy'):.4f} and the full method further decreases it to {fmean(severe_full, 'accuracy'):.4f}. "
            "This result indicates negative adaptation: even the strengthened reliability estimator is not sufficiently robust when the graph homophily assumption is strongly violated. "
            "The paper therefore should not claim broad superiority under arbitrary structural shift; instead, it should emphasize measured calibration gains under moderate perturbations and explicitly discuss severe homophily shift as a limitation."
        )

    reliability_text = ""
    if reliability_mean is not None and selected_fraction is not None:
        reliability_text = (
            f"The strengthened estimator assigns an average reliability score of {reliability_mean:.4f} and keeps an average selected fraction of {100.0 * selected_fraction:.1f}\\% across the completed synthetic conditions. "
            "This indicates that the updated score behaves as a soft gate rather than collapsing to either full selection or near-zero coverage."
        )

    main_table_tex = render_table(
        "table*",
        "width=\\textwidth",
        "llrrrr",
        "Shift & Method & Accuracy & Macro-F1 & ECE & NLL",
        main_table(synthetic_rows),
        "Main results under controlled structural distribution shifts. Values are mean$\\pm$standard deviation over five seeds. Lower ECE and NLL are better.",
        "tab:main",
    )
    ablation_table_tex = render_table(
        "table",
        "max width=\\linewidth",
        "lrrrr",
        "Method & Accuracy & ECE & Brier & Runtime (s)",
        ablation_table(synthetic_rows),
        "Ablation results averaged over all controlled synthetic shift settings. Lower ECE, Brier score, and runtime are better.",
        "tab:ablation",
    )
    synthetic_overview_fig = figure_block(
        "synthetic_overview.pdf",
        "Synthetic benchmark overview. The top panel summarizes accuracy across the controlled shifts and the bottom panel summarizes ECE for the main methods.",
        "fig:synthetic-overview",
    )
    synthetic_tradeoff_fig = figure_block(
        "synthetic_tradeoff.pdf",
        "Aggregate synthetic trade-offs. Horizontal bars report mean accuracy, ECE, and runtime with source-only reference lines, while the final panel summarizes the reliability score and selected fraction of the reliability-aware variants.",
        "fig:synthetic-tradeoff",
    )

    return rf"""\section{{Results}}
\label{{sec:results}}

\subsection{{Status of reported numbers}}

All numbers in this section are extracted from completed result files. The main quantitative evidence comes from the controlled synthetic run stored in \texttt{{results/results.json}} and \texttt{{results/summary.csv}}. That run uses five random seeds, ten controlled shift settings, and eight adaptation variants. Values are reported as mean$\pm$standard deviation across seeds. The public-data numbers, when shown, are drawn only from completed public benchmark result files and are interpreted strictly at the level supported by those stored outputs.

\subsection{{Main results under controlled distribution shifts}}

Table~\ref{{tab:main}} reports accuracy, macro-F1, expected calibration error (ECE), and negative log-likelihood (NLL). On clean, feature-noise, edge-add, and degree-shift settings, test-time adaptation often improves calibration relative to the source-only model. Averaged over all ten synthetic conditions, the full method changes accuracy by {findings['mean_source_acc_gain']:.4f} and ECE by {-findings['mean_source_ece_gain']:.4f} relative to source-only, where a negative ECE change means improved calibration. However, the full method is not uniformly better than entropy minimization in this implementation: its average accuracy change relative to entropy adaptation is {findings['mean_entropy_acc_gain']:.4f}, and its average ECE change is {-findings['mean_entropy_ece_gain']:.4f}. These measured values require a conservative interpretation of the proposed reliability mechanism.

{main_table_tex}
{synthetic_overview_fig}

{severe_text}

\subsection{{Ablation analysis}}

Table~\ref{{tab:ablation}} averages each method over all ten synthetic shift settings. The ablations show that, in the current classifier-only implementation, the proposed components now produce a more interpretable trade-off than in the earlier draft: the reliability-weighted variants expose explicit coverage and runtime behavior in addition to predictive metrics. {reliability_text}

{ablation_table_tex}
{synthetic_tradeoff_fig}

    For reference, source-only averages accuracy {agg['source_only']['accuracy']:.4f} and ECE {agg['source_only']['ece']:.4f}, with Brier score {agg['source_only']['brier']:.4f} and runtime {agg['source_only']['runtime_seconds']:.4f}. Entropy-all-nodes averages accuracy {agg['entropy_all_nodes']['accuracy']:.4f} and ECE {agg['entropy_all_nodes']['ece']:.4f}, with Brier score {agg['entropy_all_nodes']['brier']:.4f} and runtime {agg['entropy_all_nodes']['runtime_seconds']:.4f}. The dominant empirical effect in this run therefore remains the calibration gain from entropy-based adaptation on several moderate shifts. At the same time, the updated reliability estimator exposes a measurable selection profile that can be audited directly.

\subsection{{Calibration behavior}}

The calibration metrics show a mixed but informative pattern. Under feature noise, edge drop, edge addition, and degree shift, adaptation typically lowers ECE and NLL relative to the source-only baseline. For example, under edge addition at intensity 0.35, ECE decreases from {fmean(find_row(synthetic_rows, 'edge_add', 0.35, 'source_only'), 'ece'):.4f} for source-only to {fmean(find_row(synthetic_rows, 'edge_add', 0.35, 'full_method'), 'ece'):.4f} for the full method. Under homophily shift at intensity 0.25, ECE also decreases from {fmean(find_row(synthetic_rows, 'homophily_shift', 0.25, 'source_only'), 'ece'):.4f} to {fmean(find_row(synthetic_rows, 'homophily_shift', 0.25, 'full_method'), 'ece'):.4f}, but accuracy drops from {fmean(find_row(synthetic_rows, 'homophily_shift', 0.25, 'source_only'), 'accuracy'):.4f} to {fmean(find_row(synthetic_rows, 'homophily_shift', 0.25, 'full_method'), 'accuracy'):.4f}. Under homophily shift at intensity 0.50, both accuracy and calibration deteriorate after adaptation. These results support the need for reliability-aware safeguards, but they also show that the present safeguard remains insufficient under strong violations of local-label consistency.

\subsection{{Runtime}}

Runtime confirms the expected computational trade-off. Source-only inference has negligible adaptation cost. Entropy-all-nodes adaptation averages {agg['entropy_all_nodes']['runtime_seconds']:.4f} seconds per graph, whereas the full method averages {agg['full_method']['runtime_seconds']:.4f} seconds because structural-stability estimation requires additional perturbed graph views. Removing structural stability reduces average runtime to {agg['no_structural_stability']['runtime_seconds']:.4f} seconds while leaving aggregate predictive metrics close to the full method. This finding motivates either a cheaper stability approximation or an adaptive trigger that computes stability only for uncertain regions.
{public_benchmark_section(public_rows)}
{heterophily_section(heterophily_rows)}
{significance_section(sig_rows, hetero_sig_rows)}
\subsection{{Implications for the next experimental iteration}}

The completed runs are useful because they prevent unsupported claims. The controlled synthetic run shows that the implementation performs real graph training, adaptation, and metric computation, but it also reveals that the proposed full method still does not dominate simpler entropy adaptation under every shift. The new public-data benchmark is substantially stronger than the earlier single-dataset pipeline check because it uses sparse/scalable public-graph training and multiple public datasets, yet its interpretation must still remain tied to the exact datasets and shift settings present in the stored summaries. The next iteration should further improve the reliability estimator, continue tuning public-data protocols carefully, and evaluate whether reliability weighting provides clearer benefits on realistic structural shifts where unreliable target nodes can be identified more sharply.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="main synthetic summary CSV")
    parser.add_argument("--public-summary", default=str(DEFAULT_PUBLIC_SUMMARY_PATH), help="optional public benchmark summary CSV")
    parser.add_argument("--heterophily-summary", default=str(DEFAULT_HETEROPHILY_SUMMARY_PATH), help="optional heterophily benchmark summary CSV")
    parser.add_argument("--significance", default=str(DEFAULT_SIGNIFICANCE_PATH), help="optional synthetic significance CSV")
    parser.add_argument("--heterophily-significance", default=str(DEFAULT_HETEROPHILY_SIGNIFICANCE_PATH), help="optional heterophily significance CSV")
    parser.add_argument("--out", default=str(RESULTS_TEX), help="output results.tex path")
    return parser.parse_args()


def main():
    args = parse_args()
    synthetic_rows = read_summary(args.summary, dataset_fallback="synthetic")
    public_path = Path(args.public_summary)
    if not public_path.exists() and public_path == DEFAULT_PUBLIC_SUMMARY_PATH and LEGACY_PUBLIC_SUMMARY_PATH.exists():
        public_path = LEGACY_PUBLIC_SUMMARY_PATH
    public_rows = read_summary(public_path, dataset_fallback="cora") if public_path.exists() else []
    heterophily_path = Path(args.heterophily_summary)
    heterophily_rows = read_summary(heterophily_path, dataset_fallback="texas") if heterophily_path.exists() else []
    sig_rows = significance_rows(args.significance)
    hetero_sig_rows = significance_rows(args.heterophily_significance)
    out_path = Path(args.out)
    out_path.write_text(render_results(synthetic_rows, public_rows, heterophily_rows, sig_rows, hetero_sig_rows), encoding="utf-8")
    print(f"Updated {out_path}")


if __name__ == "__main__":
    main()
