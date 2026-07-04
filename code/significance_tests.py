import argparse
import csv
import json
import math
from pathlib import Path

METRICS = ["accuracy", "macro_f1", "nll", "ece", "brier"]
COMPARISONS = [
    ("full_method", "source_only"),
    ("full_method", "entropy_all_nodes"),
    ("full_method", "tent_entropy"),
    ("full_method", "eata_filter"),
    ("full_method", "graph_tta_consistency"),
    ("full_method", "matcha_reliable"),
]


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_significance(values_a, values_b):
    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    if n < 2:
        return None, None, n
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    std = math.sqrt(max(var, 0.0))
    if std <= 1e-12:
        p_value = 1.0 if abs(mean) <= 1e-12 else 0.0
    else:
        t_stat = mean / (std / math.sqrt(n))
        # Normal approximation to a two-sided paired t-test. With five seeds this is
        # intentionally conservative in interpretation and reported as approximate.
        p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return mean, max(0.0, min(1.0, p_value)), n


def load_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("records", [])


def grouped_records(records):
    grouped = {}
    for record in records:
        key = (record.get("dataset", "synthetic"), record["shift"], float(record["intensity"]), record["method"])
        grouped.setdefault(key, {})[int(record["seed"])] = record
    return grouped


def compute_tests(records):
    grouped = grouped_records(records)
    rows = []
    conditions = sorted({(d, s, i) for d, s, i, _ in grouped}, key=lambda item: (item[0], item[1], item[2]))
    for dataset, shift, intensity in conditions:
        for method_a, method_b in COMPARISONS:
            rows_a = grouped.get((dataset, shift, intensity, method_a))
            rows_b = grouped.get((dataset, shift, intensity, method_b))
            if not rows_a or not rows_b:
                continue
            seeds = sorted(set(rows_a).intersection(rows_b))
            if len(seeds) < 2:
                continue
            out = {
                "dataset": dataset,
                "shift": shift,
                "intensity": f"{intensity:.12g}",
                "method_a": method_a,
                "method_b": method_b,
                "n": len(seeds),
            }
            for metric in METRICS:
                vals_a = [float(rows_a[seed][metric]) for seed in seeds]
                vals_b = [float(rows_b[seed][metric]) for seed in seeds]
                mean_diff, p_value, _ = paired_significance(vals_a, vals_b)
                out[f"{metric}_diff"] = f"{mean_diff:.8f}" if mean_diff is not None else ""
                out[f"{metric}_p"] = f"{p_value:.8f}" if p_value is not None else ""
                out[f"{metric}_significant"] = "1" if p_value is not None and p_value < 0.05 else "0"
            rows.append(out)
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "shift", "intensity", "method_a", "method_b", "n"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_diff", f"{metric}_p", f"{metric}_significant"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="path to results.json")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()
    rows = compute_tests(load_records(args.results))
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} significance rows to {args.out}")


if __name__ == "__main__":
    main()
