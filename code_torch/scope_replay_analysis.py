"""Validate and summarize the frozen diagnostic-scope replay.

The replay adds observational proxy traces on the full target graph, the
benchmark evaluation nodes, and their complement.  This module never reruns
adaptation and never changes the frozen gate.  It fails closed unless the new
records reproduce every original result field other than wall-clock runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


DELTA_LIMIT = 0.05
PHI_LIMIT = 0.20
IDENTITY_FIELDS = ("dataset", "seed", "shift", "intensity", "adapter", "guard")
SCOPES = ("target", "evaluation", "non_evaluation")
RUNTIME_FIELD = "runtime_seconds"
TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict) -> tuple:
    try:
        return tuple(row[field] for field in IDENTITY_FIELDS)
    except KeyError as error:
        raise ValueError(f"record lacks identity field {error.args[0]!r}") from error


def _index_unique(rows: list[dict], label: str) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    for row in rows:
        key = _identity(row)
        if key in indexed:
            raise ValueError(f"duplicate {label} record identity: {key!r}")
        indexed[key] = row
    return indexed


def _finite_number(value, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _close(left: float, right: float, label: str, tolerance=TOLERANCE) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _accept(delta: float, phi: float) -> bool:
    return float(delta) <= DELTA_LIMIT and float(phi) <= PHI_LIMIT


def _max_or_zero(values: list[float]) -> float:
    return max(values, default=0.0)


def _validate_scope_trace(
    scope_name: str,
    trace: dict,
    endpoint: dict,
    adaptation_attempts: int,
) -> dict:
    required_trace = {
        "num_nodes",
        "fraction_of_target_nodes",
        "scope_index_sha256",
        "source_prediction_sha256",
        "nonempty_degree_groups",
        "step_trace",
        "delta_trace",
        "phi_trace",
        "flip_count_trace",
        "max_delta",
        "max_delta_step",
        "max_delta_trace_index",
        "max_phi",
        "max_phi_step",
        "max_phi_trace_index",
    }
    missing = sorted(required_trace - set(trace))
    if missing:
        raise ValueError(f"{scope_name} trace lacks fields: {missing}")
    for field in (
        "num_nodes",
        "fraction_of_target_nodes",
        "scope_index_sha256",
        "source_prediction_sha256",
        "nonempty_degree_groups",
    ):
        if trace[field] != endpoint[field]:
            raise ValueError(f"{scope_name} trace/endpoint invariant mismatch: {field}")

    steps = trace["step_trace"]
    deltas = trace["delta_trace"]
    phis = trace["phi_trace"]
    flips = trace["flip_count_trace"]
    lengths = {len(steps), len(deltas), len(phis), len(flips)}
    if lengths != {int(adaptation_attempts)}:
        raise ValueError(
            f"{scope_name} trace length does not equal adaptation attempts"
        )
    if steps != list(range(int(adaptation_attempts))):
        raise ValueError(f"{scope_name} observer steps are not contiguous from zero")
    if any(not math.isfinite(float(value)) for value in deltas + phis):
        raise ValueError(f"{scope_name} trace contains a non-finite proxy")
    if any(value < 0 or value > 1 for value in phis):
        raise ValueError(f"{scope_name} turnover lies outside [0, 1]")
    if any(int(value) < 0 or int(value) > int(trace["num_nodes"]) for value in flips):
        raise ValueError(f"{scope_name} flip count lies outside its scope")
    for phi, flip_count in zip(phis, flips):
        _close(
            phi,
            int(flip_count) / int(trace["num_nodes"]),
            f"{scope_name} phi/count identity",
        )

    max_delta = _max_or_zero(deltas)
    max_phi = _max_or_zero(phis)
    if adaptation_attempts:
        _close(trace["max_delta"], max_delta, f"{scope_name} maximum delta")
        _close(trace["max_phi"], max_phi, f"{scope_name} maximum phi")
        max_delta_index = deltas.index(max_delta)
        max_phi_index = phis.index(max_phi)
        if trace["max_delta_trace_index"] != max_delta_index:
            raise ValueError(f"{scope_name} maximum-delta index mismatch")
        if trace["max_phi_trace_index"] != max_phi_index:
            raise ValueError(f"{scope_name} maximum-phi index mismatch")
        if trace["max_delta_step"] != steps[max_delta_index]:
            raise ValueError(f"{scope_name} maximum-delta step mismatch")
        if trace["max_phi_step"] != steps[max_phi_index]:
            raise ValueError(f"{scope_name} maximum-phi step mismatch")
        _close(endpoint["delta"], deltas[-1], f"{scope_name} endpoint delta")
        _close(endpoint["phi"], phis[-1], f"{scope_name} endpoint phi")
    else:
        _close(endpoint["delta"], 0.0, f"{scope_name} empty endpoint delta")
        _close(endpoint["phi"], 0.0, f"{scope_name} empty endpoint phi")

    if endpoint["flip_count"] < 0 or endpoint["flip_count"] > endpoint["num_nodes"]:
        raise ValueError(f"{scope_name} endpoint flip count is invalid")
    _close(
        endpoint["phi"],
        endpoint["flip_count"] / endpoint["num_nodes"],
        f"{scope_name} endpoint phi/count identity",
    )
    return {
        "trajectory_max_delta": max_delta,
        "trajectory_max_phi": max_phi,
        "trajectory_accept": _accept(max_delta, max_phi),
        "endpoint_delta": float(endpoint["delta"]),
        "endpoint_phi": float(endpoint["phi"]),
        "endpoint_accept": _accept(endpoint["delta"], endpoint["phi"]),
    }


def validate_and_summarize(original_payload: dict, replay_payload: dict) -> tuple[list[dict], dict]:
    """Return row diagnostics only after strict replay validation."""

    if original_payload.get("status") != "complete":
        raise ValueError("original held-out artifact is not complete")
    if replay_payload.get("status") != "complete":
        raise ValueError("diagnostic-scope replay is not complete")
    original_rows = original_payload.get("records")
    replay_rows = replay_payload.get("records")
    if not isinstance(original_rows, list) or not original_rows:
        raise ValueError("original artifact has no records")
    if not isinstance(replay_rows, list) or not replay_rows:
        raise ValueError("scope replay has no records")

    original_index = _index_unique(original_rows, "original")
    replay_index = _index_unique(replay_rows, "replay")
    if set(original_index) != set(replay_index):
        missing = sorted(set(original_index) - set(replay_index))
        extra = sorted(set(replay_index) - set(original_index))
        raise ValueError(f"replay identity set mismatch; missing={missing}, extra={extra}")

    original_schema = set(original_rows[0])
    if any(set(row) != original_schema for row in original_rows):
        raise ValueError("original record schema is inconsistent")
    compared_fields = sorted(original_schema - {RUNTIME_FIELD})
    metadata_fields = (
        "datasets",
        "seeds",
        "train_epochs",
        "adapt_steps",
        "include_guard",
        "guard_mode",
        "stress_grid",
    )
    for field in metadata_fields:
        if original_payload.get(field) != replay_payload.get(field):
            raise ValueError(f"replay metadata mismatch: {field}")

    output_rows = []
    for key in sorted(original_index):
        original = original_index[key]
        replay = replay_index[key]
        for field in compared_fields:
            if field not in replay or original[field] != replay[field]:
                raise ValueError(f"noninterference mismatch for {key!r}/{field}")
        _finite_number(original[RUNTIME_FIELD], "original runtime")
        _finite_number(replay[RUNTIME_FIELD], "replay runtime")

        traces = replay.get("diagnostic_scope_traces")
        endpoints = replay.get("endpoint_proxy_scopes")
        if not isinstance(traces, dict) or set(traces) != set(SCOPES):
            raise ValueError(f"scope trace names mismatch for {key!r}")
        if not isinstance(endpoints, dict) or set(endpoints) != set(SCOPES):
            raise ValueError(f"endpoint scope names mismatch for {key!r}")
        scope_values = {
            name: _validate_scope_trace(
                name,
                traces[name],
                endpoints[name],
                replay["adaptation_attempts"],
            )
            for name in SCOPES
        }

        target = scope_values["target"]
        if traces["target"]["delta_trace"] != replay["delta_trace"]:
            raise ValueError(f"target delta trace does not reproduce primary trace: {key!r}")
        if traces["target"]["phi_trace"] != replay["phi_trace"]:
            raise ValueError(f"target phi trace does not reproduce primary trace: {key!r}")
        _close(target["trajectory_max_delta"], replay["max_delta"], "primary max delta")
        _close(target["trajectory_max_phi"], replay["max_phi"], "primary max phi")

        eval_endpoint = endpoints["evaluation"]
        non_endpoint = endpoints["non_evaluation"]
        target_endpoint = endpoints["target"]
        if eval_endpoint["num_nodes"] + non_endpoint["num_nodes"] != target_endpoint["num_nodes"]:
            raise ValueError(f"evaluation scopes do not partition the target: {key!r}")
        if eval_endpoint["flip_count"] + non_endpoint["flip_count"] != target_endpoint["flip_count"]:
            raise ValueError(f"scope flip counts do not partition the target: {key!r}")
        weighted_phi = (
            eval_endpoint["flip_count"] + non_endpoint["flip_count"]
        ) / target_endpoint["num_nodes"]
        _close(target_endpoint["phi"], weighted_phi, "partitioned target endpoint phi")

        gain = float(replay["source_relative_accuracy"])
        slack = float(eval_endpoint["phi"]) - abs(gain)
        _close(
            replay["evaluation_turnover_bound_slack"],
            slack,
            "stored evaluation turnover-bound slack",
        )
        if slack < -1e-6:
            raise ValueError(f"same-scope turnover bound failed for {key!r}")

        row = dict(zip(IDENTITY_FIELDS, key))
        row.update(
            {
                "source_relative_accuracy": gain,
                "evaluation_turnover_bound_slack": slack,
                "target_vs_evaluation_trajectory_agree": (
                    target["trajectory_accept"]
                    == scope_values["evaluation"]["trajectory_accept"]
                ),
                "target_vs_non_evaluation_trajectory_agree": (
                    target["trajectory_accept"]
                    == scope_values["non_evaluation"]["trajectory_accept"]
                ),
                "target_trajectory_vs_endpoint_agree": (
                    target["trajectory_accept"] == target["endpoint_accept"]
                ),
                "evaluation_trajectory_vs_endpoint_agree": (
                    scope_values["evaluation"]["trajectory_accept"]
                    == scope_values["evaluation"]["endpoint_accept"]
                ),
            }
        )
        for name in SCOPES:
            for field, value in scope_values[name].items():
                row[f"{name}_{field}"] = value
        output_rows.append(row)

    summaries = []
    adapters = ["all"] + sorted({row["adapter"] for row in output_rows})
    for adapter in adapters:
        subset = (
            output_rows
            if adapter == "all"
            else [row for row in output_rows if row["adapter"] == adapter]
        )
        summary = {"adapter": adapter, "records": len(subset)}
        for scope in SCOPES:
            for stage in ("trajectory", "endpoint"):
                field = f"{scope}_{stage}_accept"
                accepted = sum(bool(row[field]) for row in subset)
                summary[f"{scope}_{stage}_accepted"] = accepted
                summary[f"{scope}_{stage}_coverage"] = accepted / len(subset)
        for field in (
            "target_vs_evaluation_trajectory_agree",
            "target_vs_non_evaluation_trajectory_agree",
            "target_trajectory_vs_endpoint_agree",
            "evaluation_trajectory_vs_endpoint_agree",
        ):
            agreements = sum(bool(row[field]) for row in subset)
            summary[field + "_count"] = agreements
            summary[field + "_fraction"] = agreements / len(subset)
        summary["target_reject_evaluation_accept_trajectory"] = sum(
            (not row["target_trajectory_accept"])
            and row["evaluation_trajectory_accept"]
            for row in subset
        )
        summary["target_accept_evaluation_reject_trajectory"] = sum(
            row["target_trajectory_accept"]
            and (not row["evaluation_trajectory_accept"])
            for row in subset
        )
        slacks = [row["evaluation_turnover_bound_slack"] for row in subset]
        summary["evaluation_turnover_bound_min_slack"] = min(slacks)
        summary["evaluation_turnover_bound_max_slack"] = max(slacks)
        summaries.append(summary)

    validation = {
        "status": "pass",
        "record_count": len(output_rows),
        "unique_identity_count": len(replay_index),
        "original_fields_compared": compared_fields,
        "runtime_exclusion": "wall-clock runtime is expected to differ and is not a deterministic result field",
        "original_fields_exact_except_runtime": True,
        "target_scope_reproduces_operational_trajectory": True,
        "evaluation_and_non_evaluation_partition_target": True,
        "same_scope_turnover_bounds_absolute_accuracy_change": True,
        "same_scope_bound_interpretation": "magnitude only; it does not identify gain sign or certify safety",
        "scope_decisions_status": "diagnostic counterfactuals; neither scope-specific gate was deployed or tuned",
        "summaries": summaries,
    }
    return output_rows, validation


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(original_path: Path, replay_path: Path, out_dir: Path) -> dict:
    original_payload = json.loads(original_path.read_text(encoding="utf-8"))
    replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
    rows, validation = validate_and_summarize(original_payload, replay_payload)
    manifest = {
        "analysis": "frozen_diagnostic_scope_replay",
        "inputs": [
            {"role": "original", "path": str(original_path.resolve()), "sha256": _sha256(original_path)},
            {"role": "scope_replay", "path": str(replay_path.resolve()), "sha256": _sha256(replay_path)},
        ],
        "analysis_code": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "fixed_gate": {"delta_limit": DELTA_LIMIT, "phi_limit": PHI_LIMIT, "uses": "trajectory maxima"},
        "label_policy": "target labels are used only for the already-recorded endpoint accuracy change",
        "inferential_status": "descriptive matched-event scope diagnostic, not an independent efficacy test",
    }
    output = {"manifest": manifest, "validation": validation, "rows": rows}
    _write_json(out_dir / "scope_replay_analysis.json", output)
    _write_json(out_dir / "validation_report.json", validation)
    _write_csv(out_dir / "scope_replay_rows.csv", rows)
    _write_csv(out_dir / "scope_replay_summary.csv", validation["summaries"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.original, args.replay, args.out_dir)


if __name__ == "__main__":
    main()
