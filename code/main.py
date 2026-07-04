import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from adaptation import adapt_classifier
from data import apply_shift, load_public_graph_dataset, make_contextual_sbm, make_heterophily_benchmark, split_indices
from models import TwoLayerGCN
from utils import degree_vector, evaluate, upper_triangle_edges


METHODS = [
    "source_only",
    "entropy_all_nodes",
    "tent_entropy",
    "eata_filter",
    "graph_tta_consistency",
    "matcha_reliable",
    "reliable_entropy",
    "no_neighborhood_agreement",
    "no_structural_stability",
    "no_calibration_loss",
    "no_anti_forgetting",
    "full_method",
]

PUBLIC_BENCHMARK_METHODS = [
    "source_only",
    "entropy_all_nodes",
    "tent_entropy",
    "eata_filter",
    "graph_tta_consistency",
    "matcha_reliable",
    "full_method",
]

HETEROPHILY_DATASETS = {"texas", "cornell", "wisconsin", "actor", "film"}


def methods_for_dataset(dataset):
    return METHODS if dataset == "synthetic" or dataset in HETEROPHILY_DATASETS else PUBLIC_BENCHMARK_METHODS

BENCHMARK_DATASETS = {
    "public_core": ["cora", "citeseer", "pubmed"],
    "public_extended": ["cora", "citeseer", "pubmed", "amazon_computers", "amazon_photo", "coauthor_cs"],
    "heterophily_core": ["texas", "cornell", "wisconsin"],
    "heterophily_extended": ["texas", "cornell", "wisconsin", "actor", "film"],
}

DEFAULT_MAX_NODES = {
    "amazon_computers": 5000,
    "amazon_photo": 5000,
    "coauthor_cs": 6000,
    "coauthor_physics": 7000,
}

DEFAULT_HIDDEN = {
    "synthetic": 24,
    "cora": 32,
    "citeseer": 32,
    "pubmed": 48,
    "amazon_computers": 48,
    "amazon_photo": 48,
    "coauthor_cs": 64,
    "coauthor_physics": 64,
    "texas": 24,
    "cornell": 24,
    "wisconsin": 24,
    "actor": 24,
    "film": 24,
}


def resolve_dataset_list(dataset_arg, datasets_arg):
    raw = []
    if datasets_arg:
        raw.extend([item.strip().lower().replace("-", "_") for item in datasets_arg.split(",") if item.strip()])
    else:
        raw.append(dataset_arg.lower().replace("-", "_"))
    out = []
    for item in raw:
        out.extend(BENCHMARK_DATASETS.get(item, [item]))
    ordered = []
    seen = set()
    for item in out:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def resolve_setup(dataset, quick, train_epochs=None, adapt_steps=None):
    if dataset == "synthetic":
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.35), ("edge_drop", 0.25)]
            n = 240
            hidden = 16
            default_train_epochs = 260
            default_adapt_steps = 35
        else:
            seeds = [0, 1, 2, 3, 4]
            conditions = [
                ("clean", 0.0),
                ("feature_noise", 0.20),
                ("feature_noise", 0.45),
                ("edge_drop", 0.15),
                ("edge_drop", 0.35),
                ("edge_add", 0.15),
                ("edge_add", 0.35),
                ("degree_shift", 0.35),
                ("homophily_shift", 0.25),
                ("homophily_shift", 0.50),
            ]
            n = 360
            hidden = 24
            default_train_epochs = 520
            default_adapt_steps = 90
    elif dataset in HETEROPHILY_DATASETS:
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.10), ("edge_drop", 0.10), ("edge_add", 0.10)]
            default_train_epochs = 260
            default_adapt_steps = 20
        else:
            seeds = [0, 1, 2, 3, 4]
            conditions = [
                ("clean", 0.0),
                ("feature_noise", 0.10),
                ("edge_drop", 0.10),
                ("edge_add", 0.10),
                ("homophily_shift", 0.25),
            ]
            default_train_epochs = 360
            default_adapt_steps = 35
        n = 180
        hidden = DEFAULT_HIDDEN.get(dataset, 24)
    else:
        if quick:
            seeds = [0, 1]
            conditions = [("clean", 0.0), ("feature_noise", 0.08), ("edge_drop", 0.10), ("edge_add", 0.05)]
            default_train_epochs = 320
            default_adapt_steps = 20
        else:
            seeds = [0, 1, 2]
            conditions = [
                ("clean", 0.0),
                ("feature_noise", 0.08),
                ("edge_drop", 0.10),
                ("edge_add", 0.05),
                ("degree_shift", 0.15),
            ]
            default_train_epochs = 320
            default_adapt_steps = 25
        n = 180
        hidden = DEFAULT_HIDDEN.get(dataset, 32)
    return {
        "seeds": seeds,
        "conditions": conditions,
        "n": n,
        "hidden": hidden,
        "train_epochs": train_epochs if train_epochs is not None else default_train_epochs,
        "adapt_steps": adapt_steps if adapt_steps is not None else default_adapt_steps,
    }


def record_identity(record):
    return (
        record.get("dataset", "synthetic"),
        int(record["seed"]),
        record["shift"],
        float(record["intensity"]),
        record["method"],
    )


def condition_identity(dataset, seed, shift, intensity):
    return (dataset, int(seed), shift, float(intensity))


def load_existing_records(out_dir, resume=False):
    if not resume:
        return {}
    results_path = out_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    records = payload.get("records", [])
    out = {}
    for record in records:
        required = {"seed", "shift", "intensity", "method"}
        if required.issubset(record):
            out[record_identity(record)] = record
    return out


def completed_condition_keys(records_by_key):
    grouped = {}
    for dataset, seed, shift, intensity, method in records_by_key:
        grouped.setdefault((dataset, seed, shift, intensity), set()).add(method)
    completed = set()
    for key, methods in grouped.items():
        dataset = key[0]
        expected_methods = set(methods_for_dataset(dataset))
        if set(methods).issuperset(expected_methods):
            completed.add(key)
    return completed


def run_one(seed, shift, intensity, n, hidden, quick=False, dataset="synthetic", max_nodes=None, graph_backend="auto", train_epochs=None, adapt_steps=None):
    if dataset == "synthetic":
        x, adj, y = make_contextual_sbm(seed=seed, n=n)
        train_idx, val_idx, test_idx = split_indices(seed, y)
    elif dataset in HETEROPHILY_DATASETS:
        x, adj, y = make_heterophily_benchmark(seed=seed, name=dataset)
        train_idx, val_idx, test_idx = split_indices(seed, y, train_per_class=12, val_per_class=12)
    else:
        x, adj, y, train_idx, val_idx, test_idx = load_public_graph_dataset(dataset, max_nodes=max_nodes, seed=seed, graph_backend=graph_backend)
    classes = int(np.max(y)) + 1
    model = TwoLayerGCN(x.shape[1], hidden, classes, seed=seed)
    train_info = model.train(x, adj, y, train_idx, val_idx, epochs=train_epochs if train_epochs is not None else (260 if quick else 520))

    x_t, adj_t = apply_shift(seed, x, adj, y, shift, intensity)
    # degree_shift removes nodes, so regenerate aligned labels by applying same shift logic through index tracking is avoided in full mode.
    # For scientific consistency, degree_shift is implemented as topology weakening rather than node removal below.
    if len(x_t) != len(y):
        x_t, adj_t = x.copy(), adj.copy()
        deg = degree_vector(adj_t)
        high = np.where(deg > np.quantile(deg, 0.65))[0]
        rng = np.random.default_rng(seed + 131)
        for i in high:
            if hasattr(adj_t, "tocsr"):
                row = adj_t.tocsr()
                neigh = row.indices[row.indptr[i]:row.indptr[i + 1]]
            else:
                neigh = np.where(adj_t[i] > 0)[0]
            drop_count = int(len(neigh) * min(0.6, intensity))
            if drop_count > 0:
                drop = rng.choice(neigh, size=drop_count, replace=False)
                adj_t[i, drop] = 0.0
                adj_t[drop, i] = 0.0
    y_t = y
    num_edges = float(len(upper_triangle_edges(adj_t)))

    records = []
    method_list = methods_for_dataset(dataset)
    for method in method_list:
        m = model.clone()
        start = time.perf_counter()
        adapt_info = adapt_classifier(m, x_t, adj_t, method=method, seed=seed, steps=adapt_steps if adapt_steps is not None else (35 if quick else 90))
        runtime = time.perf_counter() - start
        probs, _ = m.forward(x_t, adj_t)
        metrics = evaluate(probs[test_idx], y_t[test_idx], classes)
        metrics.update({
            "seed": seed,
            "dataset": dataset,
            "shift": shift,
            "intensity": intensity,
            "method": method,
            "runtime_seconds": runtime,
            "adapted_steps": adapt_info["steps"],
            "mean_reliability": adapt_info["mean_reliability"],
            "selected_fraction": adapt_info["selected_fraction"],
            "train_epochs": train_info["epochs"],
            "best_val_loss": train_info["best_val_loss"],
            "num_nodes": float(x_t.shape[0]),
            "num_edges": num_edges,
        })
        records.append(metrics)
    return records


def summarize(records):
    grouped = {}
    for r in records:
        key = (r.get("dataset", "synthetic"), r["shift"], r["intensity"], r["method"])
        grouped.setdefault(key, []).append(r)
    summary = []
    metric_keys = ["accuracy", "macro_f1", "nll", "ece", "brier", "runtime_seconds", "adapted_steps", "mean_reliability", "selected_fraction", "num_nodes", "num_edges"]
    for key, rows in grouped.items():
        out = {"dataset": key[0], "shift": key[1], "intensity": key[2], "method": key[3], "count": len(rows)}
        for mk in metric_keys:
            vals = np.array([row[mk] for row in rows], dtype=float)
            out[mk + "_mean"] = float(np.mean(vals))
            out[mk + "_std"] = float(np.std(vals))
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dataset", default="synthetic", help="synthetic, cora, citeseer, pubmed, amazon_computers, amazon_photo, coauthor_cs, coauthor_physics, texas, cornell, wisconsin")
    parser.add_argument("--datasets", default="", help="comma-separated datasets or benchmark presets such as public_core, public_extended, or heterophily_core")
    parser.add_argument("--resume", action="store_true", help="resume from an existing results.json in the output directory")
    parser.add_argument("--max-nodes", type=int, default=None, help="optional deterministic induced-subgraph size for dense public-data runs")
    parser.add_argument("--graph-backend", default="auto", choices=["dense", "sparse", "auto"])
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=1800.0)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = resolve_dataset_list(args.dataset, args.datasets)
    setups = {dataset: resolve_setup(dataset, args.quick, train_epochs=args.train_epochs, adapt_steps=args.adapt_steps) for dataset in dataset_list}
    record_map = load_existing_records(out_dir, resume=args.resume)
    completed_keys = completed_condition_keys(record_map)
    if args.resume and record_map:
        print(f"RESUME_FOUND: {len(record_map)} records and {len(completed_keys)} completed conditions")

    estimated = 0.0
    for dataset in dataset_list:
        setup = setups[dataset]
        pending = [
            (seed, shift, intensity)
            for seed in setup["seeds"]
            for shift, intensity in setup["conditions"]
            if condition_identity(dataset, seed, shift, intensity) not in completed_keys
        ]
        if not pending:
            continue
        max_nodes = args.max_nodes if args.max_nodes is not None else DEFAULT_MAX_NODES.get(dataset)
        pilot_start = time.perf_counter()
        pilot_seed, pilot_shift, pilot_intensity = pending[0]
        _ = run_one(
            pilot_seed,
            pilot_shift,
            pilot_intensity,
            n=120 if args.quick else 180,
            hidden=setup["hidden"],
            quick=True,
            dataset=dataset,
            max_nodes=max_nodes,
            graph_backend=args.graph_backend,
            train_epochs=min(setup["train_epochs"], 80),
            adapt_steps=min(setup["adapt_steps"], 10),
        )
        pilot_time = time.perf_counter() - pilot_start
        estimated += pilot_time * len(pending)
    print(f"TIME_ESTIMATE: {estimated:.2f}s")

    start_all = time.perf_counter()
    stop_early = False
    for dataset in dataset_list:
        setup = setups[dataset]
        max_nodes = args.max_nodes if args.max_nodes is not None else DEFAULT_MAX_NODES.get(dataset)
        for seed in setup["seeds"]:
            for shift, intensity in setup["conditions"]:
                if condition_identity(dataset, seed, shift, intensity) in completed_keys:
                    continue
                if time.perf_counter() - start_all > 0.8 * args.budget_seconds:
                    print("TIME_GUARD: stopping early and saving partial results")
                    stop_early = True
                    break
                print(f"running dataset={dataset} seed={seed} shift={shift} intensity={intensity}")
                new_records = run_one(
                    seed,
                    shift,
                    intensity,
                    n=setup["n"],
                    hidden=setup["hidden"],
                    quick=args.quick,
                    dataset=dataset,
                    max_nodes=max_nodes,
                    graph_backend=args.graph_backend,
                    train_epochs=setup["train_epochs"],
                    adapt_steps=setup["adapt_steps"],
                )
                for record in new_records:
                    record_map[record_identity(record)] = record
            if stop_early:
                break
        if stop_early:
            break

    all_records = list(record_map.values())
    summary = summarize(all_records)
    payload = {
        "records": all_records,
        "summary": summary,
        "quick": args.quick,
        "datasets": dataset_list,
        "graph_backend": args.graph_backend,
        "max_nodes": args.max_nodes,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    for row in summary[:20]:
        print(f"{row['dataset']} {row['shift']} {row['intensity']} {row['method']} accuracy: {row['accuracy_mean']:.4f} ece: {row['ece_mean']:.4f}")
    print(f"results: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
