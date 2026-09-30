"""One-shot evaluator for sealed prospective Twitch candidate artifacts."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

try:
    from .natural_shift_io import (
        TARGET_DOMAINS,
        ArtifactValidationError,
        array_sha256,
        assert_regular_file,
        atomic_write_json,
        canonical_json_bytes,
        load_json_strict_with_sha256,
        load_protocol_with_sha256,
        load_public_graph_from_manifest,
        load_public_manifest_with_sha256,
        load_source_targets_pack,
        read_npz_strict_with_sha256,
        read_regular_file_bytes,
        resolve_manifest_path,
        sha256_bytes,
        sha256_file,
        validate_relative_path,
    )
except ImportError:
    from natural_shift_io import (
        TARGET_DOMAINS,
        ArtifactValidationError,
        array_sha256,
        assert_regular_file,
        atomic_write_json,
        canonical_json_bytes,
        load_json_strict_with_sha256,
        load_protocol_with_sha256,
        load_public_graph_from_manifest,
        load_public_manifest_with_sha256,
        load_source_targets_pack,
        read_npz_strict_with_sha256,
        read_regular_file_bytes,
        resolve_manifest_path,
        sha256_bytes,
        sha256_file,
        validate_relative_path,
    )


VAULT_PACK_SCHEMA = "twitch_target_vault.v1"
VAULT_MANIFEST_SCHEMA = "twitch_vault_manifest.v1"
CANDIDATE_MANIFEST_SCHEMA = "twitch_natural_shift_candidate.v1"
EVALUATION_SCHEMA = "twitch_natural_shift_evaluation.v1"
CONSUMPTION_RECEIPT_SCHEMA = "twitch_candidate_consumption_receipt.v1"
CONSUMPTION_MARKER_SCHEMA = "twitch_sealed_vault_consumption.v1"
_STRATEGY_IDS = (
    "reject_all",
    "always",
    "fixed_combined",
    "delta_only",
    "phi_only",
    "source_risk",
    "online_rollback",
)


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ArtifactValidationError(
            f"{context} keys mismatch: expected {sorted(expected)}, got {sorted(value)}"
        )


def _sha(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ArtifactValidationError(f"{context} is not lowercase SHA-256")
    return value


def _integer(value: Any, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ArtifactValidationError(f"{context} must be an integer >= {minimum}")
    return value


def _finite(value: Any, context: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ArtifactValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{context} must be finite")
    return result


def _text(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise ArtifactValidationError(f"{context} must be non-empty text")
    return value


def _finite_trace(value: Any, context: str) -> list[float]:
    if type(value) is not list:
        raise ArtifactValidationError(f"{context} must be a list")
    return [_finite(item, f"{context}[{index}]") for index, item in enumerate(value)]


def _scalar_text(array: np.ndarray, context: str) -> str:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(f"{context} must be scalar text")
    value = array.item()
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if type(value) is not str or not value:
        raise ArtifactValidationError(f"{context} must be non-empty text")
    return value


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _load_vault_manifest(
    path: str | Path,
    expected_public_manifest_sha256: str,
    expected_vault_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, actual_digest = load_json_strict_with_sha256(path)
    if actual_digest != _sha(
        expected_vault_manifest_sha256, "expected vault manifest digest"
    ):
        raise ArtifactValidationError("vault manifest SHA-256 mismatch")
    _exact_keys(
        value,
        {
            "hash_algorithm",
            "public_manifest_sha256",
            "schema_version",
            "source_targets",
            "targets",
        },
        "vault manifest",
    )
    if value["schema_version"] != VAULT_MANIFEST_SCHEMA:
        raise ArtifactValidationError("unsupported vault manifest schema")
    if value["hash_algorithm"] != "sha256":
        raise ArtifactValidationError("vault manifest must use SHA-256")
    if _sha(value["public_manifest_sha256"], "vault public digest") != _sha(
        expected_public_manifest_sha256, "expected public digest"
    ):
        raise ArtifactValidationError("vault/public manifest SHA-256 mismatch")
    source_entry = value["source_targets"]
    if type(source_entry) is not dict:
        raise ArtifactValidationError("source_targets entry must be an object")
    _exact_keys(
        source_entry,
        {
            "node_ids_sha256",
            "num_nodes",
            "path",
            "public_pack_sha256",
            "sha256",
        },
        "vault source entry",
    )
    if validate_relative_path(source_entry["path"]) != "source/DE.npz":
        raise ArtifactValidationError("unexpected source-target path")
    for name in ("node_ids_sha256", "public_pack_sha256", "sha256"):
        _sha(source_entry[name], f"source_targets.{name}")
    _integer(source_entry["num_nodes"], "source_targets.num_nodes", 2)

    targets = value["targets"]
    if type(targets) is not dict or list(targets) != list(TARGET_DOMAINS):
        raise ArtifactValidationError(
            "vault targets do not match the frozen domain order"
        )
    for domain, entry in targets.items():
        if type(entry) is not dict:
            raise ArtifactValidationError(f"vault entry {domain} must be an object")
        _exact_keys(
            entry,
            {
                "node_ids_sha256",
                "num_nodes",
                "path",
                "public_pack_sha256",
                "sha256",
            },
            f"vault entry {domain}",
        )
        if validate_relative_path(entry["path"]) != f"vault/{domain}.npz":
            raise ArtifactValidationError(f"unexpected vault path for {domain}")
        for name in ("node_ids_sha256", "public_pack_sha256", "sha256"):
            _sha(entry[name], f"targets.{domain}.{name}")
        _integer(entry["num_nodes"], f"targets.{domain}.num_nodes", 2)
    return value, actual_digest


def _load_target_vault(
    vault_manifest_path: Path,
    entry: Mapping[str, Any],
    domain: str,
    public_graph: Any,
) -> np.ndarray:
    path = resolve_manifest_path(vault_manifest_path.parent, entry["path"])
    arrays, actual_digest = read_npz_strict_with_sha256(
        path,
        expected_keys=(
            "domain",
            "node_ids",
            "public_pack_sha256",
            "schema_version",
            "y",
        ),
    )
    if actual_digest != entry["sha256"]:
        raise ArtifactValidationError(f"vault pack SHA-256 mismatch for {domain}")
    if _scalar_text(arrays["schema_version"], "vault schema") != VAULT_PACK_SCHEMA:
        raise ArtifactValidationError("unsupported target-vault schema")
    if _scalar_text(arrays["domain"], "vault domain") != domain:
        raise ArtifactValidationError(f"vault domain mismatch for {domain}")
    if (
        _scalar_text(arrays["public_pack_sha256"], "vault public pack digest")
        != public_graph.sha256
    ):
        raise ArtifactValidationError(f"vault/public pack mismatch for {domain}")
    if entry["public_pack_sha256"] != public_graph.sha256:
        raise ArtifactValidationError(
            f"vault-manifest/public pack mismatch for {domain}"
        )
    node_ids = arrays["node_ids"]
    if (
        node_ids.dtype != np.dtype("int64")
        or node_ids.shape != public_graph.node_ids.shape
    ):
        raise ArtifactValidationError(f"invalid vault node IDs for {domain}")
    if array_sha256(node_ids) != entry["node_ids_sha256"]:
        raise ArtifactValidationError(f"vault node-ID digest mismatch for {domain}")
    if not np.array_equal(node_ids, public_graph.node_ids):
        raise ArtifactValidationError(
            f"vault/public node IDs are misaligned for {domain}"
        )
    targets = arrays["y"]
    if targets.dtype != np.dtype("int64") or targets.shape != (public_graph.num_nodes,):
        raise ArtifactValidationError(f"invalid target array for {domain}")
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ArtifactValidationError(
            f"target array must contain both classes for {domain}"
        )
    if entry["num_nodes"] != public_graph.num_nodes:
        raise ArtifactValidationError(f"vault node count mismatch for {domain}")
    targets.setflags(write=False)
    return targets


def _load_candidate_manifest(
    candidate_dir: Path,
    expected_manifest_sha256: str,
    protocol: Mapping[str, Any],
    protocol_file_sha256: str,
    public_manifest_sha256: str,
) -> dict[str, Any]:
    manifest_path = resolve_manifest_path(candidate_dir, "candidate_manifest.json")
    value, actual_digest = load_json_strict_with_sha256(manifest_path)
    if actual_digest != _sha(
        expected_manifest_sha256, "expected candidate manifest digest"
    ):
        raise ArtifactValidationError("candidate manifest SHA-256 mismatch")
    sidecar_path = resolve_manifest_path(candidate_dir, "candidate_manifest.sha256")
    expected_sidecar = f"{actual_digest}  candidate_manifest.json\n"
    try:
        sidecar = read_regular_file_bytes(sidecar_path, max_bytes=256).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError("candidate manifest sidecar is not ASCII") from exc
    if sidecar != expected_sidecar:
        raise ArtifactValidationError("candidate manifest sidecar mismatch")
    _exact_keys(
        value,
        {
            "artifact_count",
            "execution_policy",
            "hash_algorithm",
            "manifest_payload_sha256",
            "protocol_file_sha256",
            "protocol_id",
            "protocol_payload_sha256",
            "public_manifest_sha256",
            "runs",
            "schema_version",
            "source_models",
            "source_targets_sha256",
            "status",
            "total_runtime_seconds",
        },
        "candidate manifest",
    )
    if value["schema_version"] != CANDIDATE_MANIFEST_SCHEMA:
        raise ArtifactValidationError("unsupported candidate manifest schema")
    if value["status"] != "finalized" or value["hash_algorithm"] != "sha256":
        raise ArtifactValidationError(
            "candidate manifest is not finalized with SHA-256"
        )
    if _manifest_payload_sha256(value) != _sha(
        value["manifest_payload_sha256"], "candidate payload digest"
    ):
        raise ArtifactValidationError("candidate payload SHA-256 mismatch")
    if value["protocol_id"] != protocol["protocol_id"]:
        raise ArtifactValidationError("candidate/protocol ID mismatch")
    if value["protocol_payload_sha256"] != protocol["protocol_payload_sha256"]:
        raise ArtifactValidationError("candidate/protocol payload mismatch")
    if value["protocol_file_sha256"] != protocol_file_sha256:
        raise ArtifactValidationError("candidate/protocol file mismatch")
    if value["public_manifest_sha256"] != public_manifest_sha256:
        raise ArtifactValidationError("candidate/public manifest mismatch")
    _sha(value["source_targets_sha256"], "source_targets_sha256")
    _finite(value["total_runtime_seconds"], "total_runtime_seconds")

    expected_count = (
        len(protocol["seeds"])
        * len(protocol["models"]["backbones"])
        * len(protocol["adaptation"]["adapters"])
        * len(TARGET_DOMAINS)
    )
    if _integer(value["artifact_count"], "artifact_count") != expected_count:
        raise ArtifactValidationError("candidate artifact count is incomplete")
    runs = value["runs"]
    if type(runs) is not list or len(runs) != expected_count:
        raise ArtifactValidationError("candidate run ledger is incomplete")
    expected_combinations = {
        (domain, seed, backbone, adapter)
        for seed in protocol["seeds"]
        for backbone in protocol["models"]["backbones"]
        for domain in TARGET_DOMAINS
        for adapter in protocol["adaptation"]["adapters"]
    }
    seen: set[tuple[str, int, str, str]] = set()
    for index, run in enumerate(runs):
        if type(run) is not dict:
            raise ArtifactValidationError(f"candidate run {index} must be an object")
        _exact_keys(
            run,
            {
                "adapter",
                "artifact_path",
                "artifact_sha256",
                "backbone",
                "decisions",
                "domain",
                "endpoint_delta",
                "endpoint_phi",
                "failure",
                "node_ids_sha256",
                "online",
                "online_trajectory_sha256",
                "proxy_trace",
                "public_pack_sha256",
                "run_id",
                "runtime_seconds",
                "seed",
                "source_checkpoint_sha256",
                "source_risk_estimated_gain",
                "status",
                "trajectory_sha256",
            },
            f"candidate run {index}",
        )
        domain = _text(run["domain"], f"candidate run {index}.domain")
        seed = _integer(run["seed"], f"candidate run {index}.seed")
        backbone = _text(run["backbone"], f"candidate run {index}.backbone")
        adapter = _text(run["adapter"], f"candidate run {index}.adapter")
        combination = (domain, seed, backbone, adapter)
        if combination not in expected_combinations or combination in seen:
            raise ArtifactValidationError("unexpected or duplicate candidate run")
        seen.add(combination)
        expected_id = f"{domain}__seed-{seed}__{backbone}__{adapter}"
        if _text(run["run_id"], f"candidate run {index}.run_id") != expected_id:
            raise ArtifactValidationError("candidate run ID mismatch")
        if (
            validate_relative_path(run["artifact_path"])
            != f"artifacts/{expected_id}.npz"
        ):
            raise ArtifactValidationError("candidate artifact path mismatch")
        for name in (
            "artifact_sha256",
            "node_ids_sha256",
            "public_pack_sha256",
            "source_checkpoint_sha256",
            "trajectory_sha256",
            "online_trajectory_sha256",
        ):
            _sha(run[name], f"candidate run {index}.{name}")
        endpoint_delta = _finite(
            run["endpoint_delta"], f"candidate run {index}.endpoint_delta"
        )
        endpoint_phi = _finite(
            run["endpoint_phi"], f"candidate run {index}.endpoint_phi"
        )
        estimated_gain = _finite(
            run["source_risk_estimated_gain"],
            f"candidate run {index}.source_risk_estimated_gain",
        )
        runtime = _finite(
            run["runtime_seconds"], f"candidate run {index}.runtime_seconds"
        )
        if not 0.0 <= endpoint_delta <= 1.0 or not 0.0 <= endpoint_phi <= 1.0:
            raise ArtifactValidationError("candidate endpoint proxy is outside [0, 1]")
        if not -1.0 <= estimated_gain <= 1.0 or runtime < 0.0:
            raise ArtifactValidationError("candidate gain/runtime value is invalid")
        if run["status"] not in {"complete", "failed_closed"}:
            raise ArtifactValidationError("candidate run status is invalid")
        if run["failure"] is not None and type(run["failure"]) is not str:
            raise ArtifactValidationError("candidate failure state is invalid")
        if (run["failure"] is None) != (run["status"] == "complete"):
            raise ArtifactValidationError("candidate failure/status values disagree")

        proxy_trace = run["proxy_trace"]
        if type(proxy_trace) is not dict:
            raise ArtifactValidationError("candidate proxy_trace must be an object")
        _exact_keys(proxy_trace, {"delta", "phi"}, "candidate proxy_trace")
        delta_trace = _finite_trace(proxy_trace["delta"], "candidate delta trace")
        phi_trace = _finite_trace(proxy_trace["phi"], "candidate phi trace")
        if (
            len(delta_trace) != len(phi_trace)
            or len(delta_trace) > protocol["adaptation"]["steps"]
        ):
            raise ArtifactValidationError("candidate proxy trace length is invalid")
        expected_delta = delta_trace[-1] if delta_trace else 0.0
        expected_phi = phi_trace[-1] if phi_trace else 0.0
        if endpoint_delta != expected_delta or endpoint_phi != expected_phi:
            raise ArtifactValidationError(
                "candidate endpoint does not match proxy trace"
            )

        online = run["online"]
        if type(online) is not dict:
            raise ArtifactValidationError("candidate online state must be an object")
        _exact_keys(online, {"failure", "state", "steps"}, "candidate online state")
        if online["failure"] is not None and type(online["failure"]) is not str:
            raise ArtifactValidationError("candidate online failure is invalid")
        online_steps = _integer(online["steps"], "candidate online steps")
        if online_steps > protocol["adaptation"]["steps"]:
            raise ArtifactValidationError("candidate online step count is invalid")
        detector_state = online["state"]
        if type(detector_state) is not dict:
            raise ArtifactValidationError("candidate detector state must be an object")
        _validate_detector_state(
            detector_state, protocol["thresholds"], online_steps
        )

        decisions = run["decisions"]
        if type(decisions) is not dict or set(decisions) != set(_STRATEGY_IDS):
            raise ArtifactValidationError("candidate strategy decisions are incomplete")
        for strategy, decision in decisions.items():
            if type(decision) is not dict:
                raise ArtifactValidationError(f"decision {strategy} must be an object")
            _exact_keys(decision, {"accepted", "reason"}, f"decision {strategy}")
            if type(decision["accepted"]) is not bool:
                raise ArtifactValidationError(f"decision {strategy} has invalid types")
            _text(decision["reason"], f"decision {strategy}.reason")
        candidate_available = run["failure"] is None
        expected_static = {
            "reject_all": False,
            "always": candidate_available,
            "fixed_combined": bool(
                candidate_available
                and endpoint_delta <= protocol["thresholds"]["delta"]
                and endpoint_phi <= protocol["thresholds"]["phi"]
            ),
            "delta_only": bool(
                candidate_available
                and endpoint_delta <= protocol["thresholds"]["delta"]
            ),
            "phi_only": bool(
                candidate_available and endpoint_phi <= protocol["thresholds"]["phi"]
            ),
            "source_risk": bool(
                candidate_available
                and estimated_gain >= -protocol["thresholds"]["harm"]
            ),
        }
        for strategy, accepted in expected_static.items():
            if decisions[strategy]["accepted"] is not accepted:
                raise ArtifactValidationError(
                    f"decision {strategy} violates its protocol"
                )
        if online["failure"] is not None and decisions["online_rollback"]["accepted"]:
            raise ArtifactValidationError("failed online trajectory cannot be accepted")
    if seen != expected_combinations:
        raise ArtifactValidationError("candidate run Cartesian product is incomplete")
    source_models = value["source_models"]
    expected_source_models = {
        f"seed-{seed}__{backbone}"
        for seed in protocol["seeds"]
        for backbone in protocol["models"]["backbones"]
    }
    if type(source_models) is not dict or set(source_models) != expected_source_models:
        raise ArtifactValidationError("source_models ledger is incomplete")
    for model_id, model in source_models.items():
        if type(model) is not dict:
            raise ArtifactValidationError(f"source model {model_id} must be an object")
        _exact_keys(
            model,
            {
                "backbone",
                "checkpoint_sha256",
                "seed",
                "source_atc_threshold",
                "split_counts",
                "training_epochs_completed",
            },
            f"source model {model_id}",
        )
        expected_model_id = f"seed-{model['seed']}__{model['backbone']}"
        if expected_model_id != model_id:
            raise ArtifactValidationError(
                f"source model identity mismatch for {model_id}"
            )
        _sha(model["checkpoint_sha256"], f"source model {model_id}.checkpoint")
        threshold = _finite(
            model["source_atc_threshold"], f"source model {model_id}.atc_threshold"
        )
        if not 0.0 <= threshold <= 1.0:
            raise ArtifactValidationError(
                f"source model {model_id} ATC threshold is invalid"
            )
        epochs = _integer(
            model["training_epochs_completed"], f"source model {model_id}.epochs", 1
        )
        if epochs > protocol["models"]["train_epochs"]:
            raise ArtifactValidationError(
                f"source model {model_id} epoch count is invalid"
            )
        split_counts = model["split_counts"]
        if type(split_counts) is not dict:
            raise ArtifactValidationError(
                f"source model {model_id} split_counts is invalid"
            )
        _exact_keys(
            split_counts,
            {"test", "train", "validation"},
            f"source model {model_id}.split_counts",
        )
        for split_name, count in split_counts.items():
            _integer(count, f"source model {model_id}.{split_name}", 1)
    for run in runs:
        model_id = f"seed-{run['seed']}__{run['backbone']}"
        if (
            run["source_checkpoint_sha256"]
            != source_models[model_id]["checkpoint_sha256"]
        ):
            raise ArtifactValidationError(
                f"candidate/source checkpoint mismatch for {run['run_id']}"
            )
    return value


def _validate_detector_state(
    state: Mapping[str, Any], thresholds: Mapping[str, Any], online_steps: int
) -> None:
    _exact_keys(
        state,
        {
            "accepted_candidates",
            "decision_history",
            "delta_history",
            "delta_tolerance",
            "phi_history",
            "phi_tolerance",
            "rejected_candidates",
            "state",
            "strikes",
            "trigger_reason",
            "trigger_step",
            "triggered",
        },
        "candidate detector state",
    )
    if _finite(state["delta_tolerance"], "detector delta tolerance") != float(
        thresholds["delta"]
    ):
        raise ArtifactValidationError("detector delta tolerance violates protocol")
    if _finite(state["phi_tolerance"], "detector phi tolerance") != float(
        thresholds["phi"]
    ):
        raise ArtifactValidationError("detector phi tolerance violates protocol")
    delta = _finite_trace(state["delta_history"], "detector delta history")
    phi = _finite_trace(state["phi_history"], "detector phi history")
    if len(delta) != len(phi):
        raise ArtifactValidationError("detector history lengths disagree")
    if any(not 0.0 <= value <= 1.0 for value in (*delta, *phi)):
        raise ArtifactValidationError("detector proxy history is outside [0, 1]")
    accepted = _integer(state["accepted_candidates"], "detector accepted_candidates")
    rejected = _integer(state["rejected_candidates"], "detector rejected_candidates")
    strikes = _integer(state["strikes"], "detector strikes")
    if strikes > 2:
        raise ArtifactValidationError("detector strikes exceed the two-strike rule")
    if accepted + rejected > online_steps or len(delta) > online_steps:
        raise ArtifactValidationError("detector counts exceed online step count")
    if type(state["triggered"]) is not bool:
        raise ArtifactValidationError("detector triggered must be boolean")
    if state["trigger_step"] is not None:
        trigger_step = _integer(state["trigger_step"], "detector trigger_step")
        if trigger_step >= online_steps:
            raise ArtifactValidationError("detector trigger_step exceeds online steps")
    if state["trigger_reason"] is not None:
        _text(state["trigger_reason"], "detector trigger_reason")
    if state["state"] not in {"ADAPTING", "FINISHED", "SOURCE_HALTED"}:
        raise ArtifactValidationError("detector state name is invalid")
    if state["state"] == "SOURCE_HALTED" and not state["triggered"]:
        raise ArtifactValidationError("halted detector must be triggered")
    if state["triggered"]:
        if state["trigger_step"] is None or state["trigger_reason"] is None:
            raise ArtifactValidationError("triggered detector lacks trigger metadata")
    elif any(
        value is not None
        for value in (state["trigger_step"], state["trigger_reason"])
    ) or rejected:
        raise ArtifactValidationError("untriggered detector has rejection metadata")
    history = state["decision_history"]
    if type(history) is not list or any(type(item) is not str for item in history):
        raise ArtifactValidationError("detector decision history is invalid")
    allowed_history = {
        "ACCEPT",
        "EXCEPTION_SOURCE_RESTORE",
        "REJECT_NONFINITE_SOURCE_HALT",
        "REJECT_RETAIN_CHECKPOINT",
        "REJECT_SOURCE_HALT",
        "SKIP_EMPTY_SELECTION",
    }
    if any(item not in allowed_history for item in history) or len(history) > online_steps:
        raise ArtifactValidationError("detector decision history violates protocol")


def _validate_probabilities(
    value: np.ndarray, num_nodes: int, context: str
) -> np.ndarray:
    if value.dtype != np.dtype("float32") or value.shape != (num_nodes, 2):
        raise ArtifactValidationError(f"{context} must be float32 with shape (N, 2)")
    if not np.isfinite(value).all():
        raise ArtifactValidationError(f"{context} contains non-finite values")
    if np.any(value < -1e-7) or np.any(value > 1.0 + 1e-7):
        raise ArtifactValidationError(f"{context} lies outside [0, 1]")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise ArtifactValidationError(f"{context} rows do not sum to one")
    return value.astype(np.float64)


def _endpoint_proxies(
    graph: Any, source_probabilities: np.ndarray, candidate_probabilities: np.ndarray
) -> tuple[float, float]:
    """Recompute the exact local degree-group delta and prediction-flip phi."""

    degree = np.bincount(
        graph.edge_index[0], minlength=graph.num_nodes
    ).astype(np.float64)
    order = np.lexsort((np.arange(graph.num_nodes, dtype=np.int64), degree))
    source_confidence = np.max(source_probabilities, axis=1)
    candidate_confidence = np.max(candidate_probabilities, axis=1)
    differences: list[float] = []
    for indices in np.array_split(order, 3):
        source_group = float(np.mean(source_confidence[indices]))
        candidate_group = float(np.mean(candidate_confidence[indices]))
        differences.append(abs(candidate_group - source_group))
    delta = float(np.mean(differences))
    phi = float(
        np.mean(
            np.argmax(candidate_probabilities, axis=1)
            != np.argmax(source_probabilities, axis=1)
        )
    )
    return delta, phi


def _load_validated_candidate_artifact(
    run: Mapping[str, Any],
    candidate_root: Path,
    graph: Any,
    source_models: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate one label-free run artifact and its decision semantics."""

    if run["public_pack_sha256"] != graph.sha256:
        raise ArtifactValidationError(
            f"candidate/public graph mismatch for {run['run_id']}"
        )
    expected_arrays = (
        "candidate_probabilities",
        "node_ids",
        "source_probabilities",
        *(f"deployed__{strategy}" for strategy in _STRATEGY_IDS),
    )
    artifact_path = resolve_manifest_path(candidate_root, run["artifact_path"])
    arrays, artifact_digest = read_npz_strict_with_sha256(
        artifact_path, expected_keys=expected_arrays
    )
    if artifact_digest != run["artifact_sha256"]:
        raise ArtifactValidationError(
            f"candidate artifact SHA-256 mismatch for {run['run_id']}"
        )
    node_ids = arrays["node_ids"]
    if (
        node_ids.dtype != np.dtype("int64")
        or not np.array_equal(node_ids, graph.node_ids)
        or array_sha256(node_ids) != run["node_ids_sha256"]
    ):
        raise ArtifactValidationError(
            f"candidate/public node IDs mismatch for {run['run_id']}"
        )
    source_raw = arrays["source_probabilities"]
    candidate_raw = arrays["candidate_probabilities"]
    _validate_probabilities(source_raw, graph.num_nodes, "source probabilities")
    _validate_probabilities(candidate_raw, graph.num_nodes, "candidate probabilities")

    trace_payload = {
        "adapter": run["adapter"],
        "candidate_probabilities_sha256": array_sha256(candidate_raw),
        "delta": run["proxy_trace"]["delta"],
        "phi": run["proxy_trace"]["phi"],
        "seed": run["seed"],
        "source_probabilities_sha256": array_sha256(source_raw),
    }
    if sha256_bytes(canonical_json_bytes(trace_payload)) != run["trajectory_sha256"]:
        raise ArtifactValidationError(
            f"candidate trajectory digest mismatch for {run['run_id']}"
        )

    recomputed_delta, recomputed_phi = _endpoint_proxies(
        graph, source_raw, candidate_raw
    )
    if not math.isclose(
        recomputed_delta, run["endpoint_delta"], rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(
        recomputed_phi, run["endpoint_phi"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ArtifactValidationError(
            f"candidate endpoint proxy recomputation mismatch for {run['run_id']}"
        )

    model_id = f"seed-{run['seed']}__{run['backbone']}"
    source_atc_threshold = float(source_models[model_id]["source_atc_threshold"])
    source_atc = float(
        np.mean(np.max(source_raw, axis=1) >= source_atc_threshold)
    )
    candidate_atc = float(
        np.mean(np.max(candidate_raw, axis=1) >= source_atc_threshold)
    )
    recomputed_gain = candidate_atc - source_atc
    if not math.isclose(
        recomputed_gain,
        run["source_risk_estimated_gain"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ArtifactValidationError(
            f"source-risk proxy recomputation mismatch for {run['run_id']}"
        )

    online_raw = arrays["deployed__online_rollback"]
    _validate_probabilities(online_raw, graph.num_nodes, "deployed online_rollback")
    online_payload = {
        "adapter": run["adapter"],
        "deployed_probabilities_sha256": array_sha256(online_raw),
        "failure": run["online"]["failure"],
        "seed": run["seed"],
        "state": run["online"]["state"],
        "steps": run["online"]["steps"],
    }
    if (
        sha256_bytes(canonical_json_bytes(online_payload))
        != run["online_trajectory_sha256"]
    ):
        raise ArtifactValidationError(
            f"online trajectory digest mismatch for {run['run_id']}"
        )
    expected_online_accepted = bool(
        run["online"]["failure"] is None
        and not np.array_equal(online_raw, source_raw)
    )
    if (
        run["decisions"]["online_rollback"]["accepted"]
        is not expected_online_accepted
    ):
        raise ArtifactValidationError(
            f"online rollback decision violates deployment for {run['run_id']}"
        )

    for strategy in _STRATEGY_IDS:
        deployed = arrays[f"deployed__{strategy}"]
        _validate_probabilities(deployed, graph.num_nodes, f"deployed {strategy}")
        if strategy == "online_rollback":
            expected = online_raw if expected_online_accepted else source_raw
        else:
            expected = (
                candidate_raw
                if run["decisions"][strategy]["accepted"]
                else source_raw
            )
        if not np.array_equal(deployed, expected):
            raise ArtifactValidationError(
                f"deployed {strategy} artifact violates its decision for "
                f"{run['run_id']}"
            )
    return arrays


def _claim_sealed_vault_once(vault_root: Path, claim: Mapping[str, Any]) -> tuple[Path, str]:
    """Atomically consume a sealed root before any target outcome is opened."""

    marker = vault_root / "consumed_once.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"sealed vault was already consumed: {marker}") from exc
    payload = canonical_json_bytes(dict(claim))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Never remove a partial claim: fail closed because another process
        # cannot know whether label unsealing had already begun.
        raise
    return marker, sha256_file(marker)


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = (np.argmax(probabilities, axis=1) == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == 0:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if np.any(selected):
            result += float(np.mean(selected)) * abs(
                float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
            )
    return float(result)


def _predictive_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    auroc: float | None
    if len(np.unique(targets)) == 2:
        auroc = float(roc_auc_score(targets, probabilities[:, 1]))
    else:
        auroc = None
    return {
        "accuracy": float(accuracy_score(targets, prediction)),
        "auroc": auroc,
        "balanced_accuracy": float(balanced_accuracy_score(targets, prediction)),
        "ece_15": _ece(probabilities, targets),
        "macro_f1": float(
            f1_score(targets, prediction, average="macro", zero_division=0)
        ),
        "nll": float(-np.mean(np.log(clipped[np.arange(len(targets)), targets]))),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in _STRATEGY_IDS:
        selected = [row for row in rows if row["strategy"] == strategy]
        domain_gain = {
            domain: _mean(
                [row["deployed_gain"] for row in selected if row["domain"] == domain]
            )
            for domain in TARGET_DOMAINS
        }
        result[strategy] = {
            "accuracy": _mean([row["accuracy"] for row in selected]),
            "auroc": _mean(
                [row["auroc"] for row in selected if row["auroc"] is not None]
            ),
            "balanced_accuracy": _mean([row["balanced_accuracy"] for row in selected]),
            "coverage": _mean([row["coverage"] for row in selected]),
            "deployed_gain": _mean([row["deployed_gain"] for row in selected]),
            "ece_15": _mean([row["ece_15"] for row in selected]),
            "foregone_gain": _mean([row["foregone_gain"] for row in selected]),
            "harm_over_1pp": _mean(
                [float(row["harm_over_1pp"]) for row in selected]
            ),
            "macro_f1": _mean([row["macro_f1"] for row in selected]),
            "nll": _mean([row["nll"] for row in selected]),
            "per_domain_deployed_gain": domain_gain,
            "residual_harm": _mean([row["residual_harm"] for row in selected]),
            "worst_domain_deployed_gain": min(domain_gain.values()),
        }
    return result


def _cleanup_staging(staging: Path, parent: Path) -> None:
    if not staging.exists():
        return
    resolved_parent = parent.resolve(strict=True)
    resolved_staging = staging.resolve(strict=True)
    try:
        resolved_staging.relative_to(resolved_parent)
    except ValueError as exc:
        raise RuntimeError(
            "refusing to clean staging outside evaluation parent"
        ) from exc
    if not staging.name.startswith(".natural-evaluation-"):
        raise RuntimeError("refusing to clean an unexpected staging directory")
    shutil.rmtree(staging)


def evaluate_candidate_once(
    *,
    protocol_path: str | Path,
    public_manifest_path: str | Path,
    vault_manifest_path: str | Path,
    vault_manifest_sha256: str,
    candidate_dir: str | Path,
    candidate_manifest_sha256: str,
    output_dir: str | Path,
) -> dict[str, str]:
    """Unseal target outcomes once and atomically publish metrics plus receipt."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    protocol_path_checked = assert_regular_file(protocol_path).resolve(strict=True)
    public_manifest_path_checked = assert_regular_file(public_manifest_path).resolve(
        strict=True
    )
    vault_manifest_path_checked = assert_regular_file(vault_manifest_path).resolve(
        strict=True
    )
    candidate_root = Path(candidate_dir).resolve(strict=True)
    protocol, protocol_file_digest = load_protocol_with_sha256(protocol_path_checked)
    public_manifest, public_digest = load_public_manifest_with_sha256(
        public_manifest_path_checked
    )
    vault_manifest, actual_vault_digest = _load_vault_manifest(
        vault_manifest_path_checked,
        public_digest,
        vault_manifest_sha256,
    )
    candidate_manifest = _load_candidate_manifest(
        candidate_root,
        candidate_manifest_sha256,
        protocol,
        protocol_file_digest,
        public_digest,
    )

    source_entry = vault_manifest["source_targets"]
    source_targets_file = resolve_manifest_path(
        vault_manifest_path_checked.parent, source_entry["path"]
    )
    if candidate_manifest["source_targets_sha256"] != source_entry["sha256"]:
        raise ArtifactValidationError("candidate/source-supervision SHA-256 mismatch")
    source_public_entry = public_manifest["graphs"]["DE"]
    if (
        source_entry["public_pack_sha256"] != source_public_entry["sha256"]
        or source_entry["node_ids_sha256"] != source_public_entry["node_ids_sha256"]
        or source_entry["num_nodes"] != source_public_entry["num_nodes"]
    ):
        raise ArtifactValidationError("source-supervision/public DE metadata mismatch")
    source_graph = load_public_graph_from_manifest(
        public_manifest_path_checked, public_manifest, "DE"
    )
    source_supervision = load_source_targets_pack(
        source_targets_file,
        expected_public_sha256=source_public_entry["sha256"],
        expected_node_ids=source_graph.node_ids,
    )
    if source_supervision.sha256 != source_entry["sha256"]:
        raise ArtifactValidationError("source-target pack SHA-256 mismatch")
    for model in candidate_manifest["source_models"].values():
        if sum(model["split_counts"].values()) != source_entry["num_nodes"]:
            raise ArtifactValidationError("source split counts do not cover DE exactly")

    public_graphs = {
        domain: load_public_graph_from_manifest(
            public_manifest_path_checked, public_manifest, domain
        )
        for domain in TARGET_DOMAINS
    }

    # Validate the entire candidate package and every label-free decision before
    # atomically claiming the private root. No target outcome has been opened yet.
    validated_artifacts: dict[str, dict[str, np.ndarray]] = {}
    for run in candidate_manifest["runs"]:
        validated_artifacts[run["run_id"]] = _load_validated_candidate_artifact(
            run,
            candidate_root,
            public_graphs[run["domain"]],
            candidate_manifest["source_models"],
        )

    claim_path, claim_digest = _claim_sealed_vault_once(
        vault_manifest_path_checked.parent,
        {
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "protocol_file_sha256": protocol_file_digest,
            "protocol_payload_sha256": protocol["protocol_payload_sha256"],
            "public_manifest_sha256": public_digest,
            "schema_version": CONSUMPTION_MARKER_SCHEMA,
            "status": "claimed_before_target_unseal",
            "vault_manifest_sha256": actual_vault_digest,
        },
    )
    target_values = {
        domain: _load_target_vault(
            vault_manifest_path_checked,
            vault_manifest["targets"][domain],
            domain,
            public_graphs[domain],
        )
        for domain in TARGET_DOMAINS
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".natural-evaluation-", dir=destination.parent)
    )
    try:
        rows: list[dict[str, Any]] = []
        consumed_artifacts: list[dict[str, str]] = []
        harm_threshold = float(protocol["thresholds"]["harm"])
        for run in candidate_manifest["runs"]:
            domain = run["domain"]
            graph = public_graphs[domain]
            arrays = validated_artifacts[run["run_id"]]
            source_probabilities = _validate_probabilities(
                arrays["source_probabilities"], graph.num_nodes, "source probabilities"
            )
            candidate_probabilities = _validate_probabilities(
                arrays["candidate_probabilities"],
                graph.num_nodes,
                "candidate probabilities",
            )
            targets = target_values[domain]
            source_accuracy = float(
                accuracy_score(targets, np.argmax(source_probabilities, axis=1))
            )
            candidate_accuracy = float(
                accuracy_score(targets, np.argmax(candidate_probabilities, axis=1))
            )
            candidate_gain = candidate_accuracy - source_accuracy
            for strategy in _STRATEGY_IDS:
                deployed = _validate_probabilities(
                    arrays[f"deployed__{strategy}"],
                    graph.num_nodes,
                    f"deployed {strategy}",
                )
                accepted = bool(run["decisions"][strategy]["accepted"])
                metrics = _predictive_metrics(deployed, targets)
                deployed_gain = metrics["accuracy"] - source_accuracy
                rows.append(
                    {
                        "accepted": accepted,
                        "accuracy": metrics["accuracy"],
                        "adapter": run["adapter"],
                        "auroc": metrics["auroc"],
                        "backbone": run["backbone"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "candidate_gain": float(candidate_gain),
                        "coverage": float(accepted),
                        "deployed_gain": float(deployed_gain),
                        "domain": domain,
                        "ece_15": metrics["ece_15"],
                        "foregone_gain": float(
                            max(0.0, candidate_gain) if not accepted else 0.0
                        ),
                        "harm_over_1pp": bool(deployed_gain < -harm_threshold),
                        "macro_f1": metrics["macro_f1"],
                        "nll": metrics["nll"],
                        "residual_harm": float(max(0.0, -deployed_gain)),
                        "run_id": run["run_id"],
                        "seed": run["seed"],
                        "source_accuracy": source_accuracy,
                        "strategy": strategy,
                    }
                )
            consumed_artifacts.append(
                {
                    "path": run["artifact_path"],
                    "sha256": run["artifact_sha256"],
                }
            )

        evaluation = {
            "aggregate": _aggregate(rows),
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "consumption_claim_path": claim_path.name,
            "consumption_claim_sha256": claim_digest,
            "evidence_status": "prospective_one_shot_evaluation",
            "protocol_payload_sha256": protocol["protocol_payload_sha256"],
            "rows": rows,
            "schema_version": EVALUATION_SCHEMA,
            "vault_manifest_sha256": actual_vault_digest,
        }
        evaluation_path = staging / "metrics.json"
        atomic_write_json(evaluation_path, evaluation)
        evaluation_digest = sha256_file(evaluation_path)
        receipt = {
            "candidate_artifacts": consumed_artifacts,
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "consumption_claim_path": claim_path.name,
            "consumption_claim_sha256": claim_digest,
            "evaluation_sha256": evaluation_digest,
            "protocol_file_sha256": protocol_file_digest,
            "public_manifest_sha256": public_digest,
            "schema_version": CONSUMPTION_RECEIPT_SCHEMA,
            "vault_manifest_sha256": actual_vault_digest,
        }
        receipt_path = staging / "consumed_manifest_receipt.json"
        atomic_write_json(receipt_path, receipt)
        receipt_digest = sha256_file(receipt_path)
        os.replace(staging, destination)
        return {
            "evaluation_sha256": evaluation_digest,
            "receipt_sha256": receipt_digest,
        }
    except BaseException:
        _cleanup_staging(staging, destination.parent)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot evaluation of a finalized Twitch candidate manifest"
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--public-manifest", required=True)
    parser.add_argument("--vault-manifest", required=True)
    parser.add_argument("--vault-manifest-sha256", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = evaluate_candidate_once(
        protocol_path=args.protocol,
        public_manifest_path=args.public_manifest,
        vault_manifest_path=args.vault_manifest,
        vault_manifest_sha256=args.vault_manifest_sha256,
        candidate_dir=args.candidate_dir,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        output_dir=args.output_dir,
    )
    for name, digest in sorted(result.items()):
        print(f"{name}={digest}")


if __name__ == "__main__":
    main()
