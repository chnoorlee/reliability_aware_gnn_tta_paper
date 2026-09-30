"""Paired pilot audit of the candidate-update proxy guard.

This runner deliberately compares the source model, the canonical adapter
without a guard, and the *same* adapter with fixed or source-side calibrated
proxy tolerances. It is an exploratory synthetic pilot, not confirmatory
evidence and not a safety guarantee.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from _np_bridge import evaluate
from adaptation import adapt_classifier
from detector import DetectorState
from exp_common import shift_bundle, train_source


DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results_torch"
DEFAULT_CONDITIONS = (("homophily_shift", 0.50), ("homophily_shift", 0.25))


def _evaluate_model(model, sb):
    probs = model.predict_probs(sb.x, sb.edge_index).cpu().numpy()
    return evaluate(probs[sb.test_idx], sb.y_np[sb.test_idx], sb.num_classes)


def run(out_dir=DEFAULT_OUT, seeds=(0, 1, 2, 3, 4), k_auto=2.0,
        train_epochs=300, adapt_steps=90, n=360, adapter_method="full_method"):
    """Run a paired source/unguarded/guarded synthetic audit."""
    from detector_calibration import _self_drift

    out_dir = Path(out_dir)
    records = []
    start_all = time.perf_counter()
    for seed in seeds:
        model, base, train_info = train_source(
            "synthetic", seed, hidden=24, epochs=train_epochs, n=n
        )
        delta_self = _self_drift(model, base, seed)
        for shift, intensity in DEFAULT_CONDITIONS:
            sb = shift_bundle(base, seed, shift, intensity)
            adapter_label = adapter_method.replace("full_method", "reliability_entropy")
            variants = (
                ("source", None, "source_only"),
                (f"{adapter_label}_unguarded", None, adapter_method),
                (f"{adapter_label}_guard_fixed", DetectorState(), adapter_method),
                (
                    f"{adapter_label}_guard_auto",
                    DetectorState(delta_tolerance=k_auto * delta_self),
                    adapter_method,
                ),
            )
            condition_rows = []
            for method_name, detector, adapter_method in variants:
                adapted = model.clone()
                start = time.perf_counter()
                info = adapt_classifier(
                    adapted,
                    sb.unlabeled(),
                    method=adapter_method,
                    seed=seed,
                    steps=adapt_steps,
                    detector=detector,
                )
                runtime = time.perf_counter() - start
                metrics = _evaluate_model(adapted, sb)
                record = {
                    "seed": seed,
                    "shift": shift,
                    "intensity": intensity,
                    "method": method_name,
                    "delta_tolerance": detector.delta_tolerance if detector else None,
                    "phi_tolerance": detector.phi_tolerance if detector else None,
                    "delta_self": delta_self,
                    "k_auto": k_auto if method_name.endswith("guard_auto") else None,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ece": metrics["ece"],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "runtime_seconds": runtime,
                    "triggered": bool(detector.triggered) if detector else False,
                    "trigger_step": detector.trigger_step if detector else None,
                    "trigger_reason": detector.trigger_reason if detector else None,
                    "detector_state": detector.state if detector else None,
                    "accepted_candidates": detector.accepted_candidates if detector else None,
                    "rejected_candidates": detector.rejected_candidates if detector else None,
                    "steps": info["steps"],
                    "train_epochs_completed": train_info["epochs"],
                }
                records.append(record)
                condition_rows.append(record)

            source_accuracy = next(
                row["accuracy"] for row in condition_rows if row["method"] == "source"
            )
            for row in condition_rows:
                row["source_relative_accuracy"] = row["accuracy"] - source_accuracy
                row["source_relative_downside"] = max(0.0, source_accuracy - row["accuracy"])
                print(
                    f"[guard-pilot] seed={seed} {shift}/{intensity} {row['method']}: "
                    f"acc={row['accuracy']:.4f} source_diff={row['source_relative_accuracy']:+.4f} "
                    f"trig={row['triggered']}@{row['trigger_step']}"
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "exploratory_pilot",
        "design": "paired source vs identical unguarded/guarded adapter",
        "records": records,
        "seeds": list(seeds),
        "conditions": [list(item) for item in DEFAULT_CONDITIONS],
        "k_auto": k_auto,
        "train_epochs": train_epochs,
        "adapt_steps": adapt_steps,
        "adapter_method": adapter_method,
        "n": n,
        "elapsed_seconds": time.perf_counter() - start_all,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
        },
    }
    (out_dir / "detector_paired_pilot.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (out_dir / "detector_paired_pilot.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)

    for shift, intensity in DEFAULT_CONDITIONS:
        adapter_label = adapter_method.replace("full_method", "reliability_entropy")
        for method_name in (
            "source",
            f"{adapter_label}_unguarded",
            f"{adapter_label}_guard_fixed",
            f"{adapter_label}_guard_auto",
        ):
            rows = [
                row for row in records
                if row["shift"] == shift
                and row["intensity"] == intensity
                and row["method"] == method_name
            ]
            acc = np.asarray([row["accuracy"] for row in rows], dtype=float)
            diff = np.asarray([row["source_relative_accuracy"] for row in rows], dtype=float)
            print(
                f"{shift}/{intensity} {method_name}: "
                f"acc={acc.mean():.4f}+-{acc.std(ddof=1) if len(acc) > 1 else 0.0:.4f} "
                f"source_diff={diff.mean():+.4f} "
                f"trig={sum(row['triggered'] for row in rows)}/{len(rows)}"
            )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--k-auto", type=float, default=2.0)
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--adapt-steps", type=int, default=90)
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument(
        "--adapter-method",
        default="full_method",
        choices=(
            "full_method",
            "no_structural_stability",
            "confidence_source_entropy",
            "tent_entropy",
        ),
    )
    args = parser.parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    run(
        out_dir=args.out,
        seeds=seeds,
        k_auto=args.k_auto,
        train_epochs=args.train_epochs,
        adapt_steps=args.adapt_steps,
        n=args.n,
        adapter_method=args.adapter_method,
    )


if __name__ == "__main__":
    main()
