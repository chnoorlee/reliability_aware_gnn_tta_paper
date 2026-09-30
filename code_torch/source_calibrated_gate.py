"""Development audit for a source-prepared paired-gain rollback gate.

This is a development runner.  Calibration uses source validation labels on a
separate corruption bank; evaluation labels are read only after the accept or
rollback decision has been fixed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adaptation import adapt_classifier
from exp_common import train_source, webkb_bundle
from risk_estimator import (
    FEATURE_NAMES,
    SourceRiskReference,
    fit_paired_gain_estimator,
    paired_risk_features,
)
from safety_holdout_audit import HIDDEN, MAX_NODES, WEBKB, make_target


CALIBRATION_GRID = (
    ("feature_noise_rms", 0.50),
    ("feature_noise_rms", 2.00),
    ("edge_drop", 0.15),
    ("edge_drop", 0.60),
    ("edge_add", 0.20),
    ("edge_add", 0.75),
    ("homophily_shift", 0.25),
    ("homophily_shift", 0.60),
)
EVALUATION_GRID = (
    ("edge_add", 0.50),
    ("edge_add", 1.00),
    ("edge_drop", 0.30),
    ("edge_drop", 0.90),
    ("homophily_shift", 0.50),
    ("homophily_shift", 0.75),
)
ADAPTERS = (
    ("confidence_source_entropy", True),
    ("tent_entropy", False),
)


def _accuracy(probs, labels, indices):
    idx = np.asarray(indices, dtype=int)
    return float(np.mean(np.argmax(probs[idx], axis=1) == np.asarray(labels)[idx]))


def _candidate(model, target, method, seed, steps):
    candidate = model.clone()
    info = adapt_classifier(
        candidate, target.unlabeled(), method=method, seed=seed, steps=steps
    )
    probs = candidate.predict_probs(
        target.x, target.edge_index
    ).cpu().numpy()
    return probs, info


def run(out_path, datasets, seeds, train_epochs=300, adapt_steps=300, harm_tolerance=0.01):
    out_path = Path(out_path)
    started = time.perf_counter()
    calibration_records = []
    evaluation_records = []
    artifacts = []
    metadata = {
        "evidence_status": "exploratory_development_calibration",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "train_epochs": train_epochs,
        "adapt_steps": adapt_steps,
        "harm_tolerance": harm_tolerance,
        "calibration_grid": [list(x) for x in CALIBRATION_GRID],
        "evaluation_grid": [list(x) for x in EVALUATION_GRID],
        "feature_names": list(FEATURE_NAMES),
    }

    for dataset in datasets:
        for seed in seeds:
            bundle = webkb_bundle(dataset, seed) if dataset in WEBKB else None
            model, base, train_info = train_source(
                dataset,
                seed,
                hidden=HIDDEN[dataset],
                epochs=train_epochs,
                max_nodes=MAX_NODES.get(dataset),
                bundle=bundle,
            )
            base_source_probs = model.predict_probs(base.x, base.edge_index).cpu().numpy()
            reference = SourceRiskReference.from_predictions(base, base_source_probs)

            local_calibration = []
            for condition_index, (shift, intensity) in enumerate(CALIBRATION_GRID):
                calibration_seed = seed + 10007 + 101 * condition_index
                target = make_target(base, calibration_seed, shift, intensity)
                source_probs = model.predict_probs(target.x, target.edge_index).cpu().numpy()
                source_val_acc = _accuracy(source_probs, target.y_np, target.val_idx)
                for method, confidence_weighted in ADAPTERS:
                    candidate_probs, info = _candidate(
                        model, target, method, calibration_seed, adapt_steps
                    )
                    candidate_val_acc = _accuracy(candidate_probs, target.y_np, target.val_idx)
                    features = paired_risk_features(
                        reference,
                        target,
                        source_probs,
                        candidate_probs,
                        confidence_weighted=confidence_weighted,
                    )
                    record = {
                        "dataset": dataset,
                        "seed": seed,
                        "shift": shift,
                        "intensity": intensity,
                        "corruption_seed": calibration_seed,
                        "adapter": method,
                        "condition_group": f"{shift}:{intensity:.4f}",
                        "source_val_accuracy": source_val_acc,
                        "candidate_val_accuracy": candidate_val_acc,
                        "paired_val_gain": candidate_val_acc - source_val_acc,
                        "features": features.tolist(),
                        "adaptation_attempts": info["steps"],
                    }
                    local_calibration.append(record)
                    calibration_records.append(record)

            estimator = fit_paired_gain_estimator(
                [row["features"] for row in local_calibration],
                [row["paired_val_gain"] for row in local_calibration],
                [row["condition_group"] for row in local_calibration],
            )
            artifacts.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "reference": reference.__dict__,
                    "estimator": estimator.to_dict(),
                    "train_epochs_completed": train_info["epochs"],
                }
            )

            for condition_index, (shift, intensity) in enumerate(EVALUATION_GRID):
                evaluation_seed = seed + 7919 + 131 * condition_index
                target = make_target(base, evaluation_seed, shift, intensity)
                source_probs = model.predict_probs(target.x, target.edge_index).cpu().numpy()
                for method, confidence_weighted in ADAPTERS:
                    candidate_probs, info = _candidate(
                        model, target, method, evaluation_seed, adapt_steps
                    )
                    features = paired_risk_features(
                        reference,
                        target,
                        source_probs,
                        candidate_probs,
                        confidence_weighted=confidence_weighted,
                    )
                    predicted_gain = estimator.predict(features)
                    lower_bound = estimator.lower_bound(features)
                    accepted = lower_bound >= -float(harm_tolerance)
                    # Evaluation labels are first read after the gate decision
                    # has been fixed from unlabeled features.
                    source_test_acc = _accuracy(
                        source_probs, target.y_np, target.test_idx
                    )
                    candidate_test_acc = _accuracy(candidate_probs, target.y_np, target.test_idx)
                    deployed_test_acc = candidate_test_acc if accepted else source_test_acc
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "shift": shift,
                        "intensity": intensity,
                        "corruption_seed": evaluation_seed,
                        "adapter": method,
                        "source_test_accuracy": source_test_acc,
                        "candidate_test_accuracy": candidate_test_acc,
                        "candidate_test_gain": candidate_test_acc - source_test_acc,
                        "predicted_gain": predicted_gain,
                        "lower_gain_bound": lower_bound,
                        "accepted": bool(accepted),
                        "deployed_test_accuracy": deployed_test_acc,
                        "deployed_test_gain": deployed_test_acc - source_test_acc,
                        "foregone_gain": max(0.0, candidate_test_acc - source_test_acc)
                        if not accepted else 0.0,
                        "residual_harm": max(0.0, source_test_acc - deployed_test_acc),
                        "features": features.tolist(),
                        "adaptation_attempts": info["steps"],
                    }
                    evaluation_records.append(row)
                    print(
                        f"[paired-gate] {dataset} seed={seed} {shift}/{intensity} {method}: "
                        f"true={row['candidate_test_gain']:+.4f} pred={predicted_gain:+.4f} "
                        f"lcb={lower_bound:+.4f} accept={accepted} "
                        f"deployed={row['deployed_test_gain']:+.4f}"
                    )

            payload = dict(metadata)
            payload.update(
                {
                    "status": "running",
                    "elapsed_seconds": time.perf_counter() - started,
                    "calibration_records": calibration_records,
                    "evaluation_records": evaluation_records,
                    "artifacts": artifacts,
                }
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.perf_counter() - started
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--datasets", default="cora,texas")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--adapt-steps", type=int, default=300)
    parser.add_argument("--harm-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    run(
        args.out,
        datasets,
        seeds,
        args.train_epochs,
        args.adapt_steps,
        args.harm_tolerance,
    )


if __name__ == "__main__":
    main()
