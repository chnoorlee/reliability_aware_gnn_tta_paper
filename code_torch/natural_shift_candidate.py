"""Label-isolated candidate runner for the prospective Twitch domain shift.

This stage trains on separately supplied DE supervision, then consumes only
public graph packs for the six target domains.  It writes predictions, proxy
traces, decisions, provenance hashes, and execution state; target outcomes and
target performance metrics are outside this module's interface.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse
import torch

try:
    from .natural_shift_io import (
        TARGET_DOMAINS,
        ArtifactValidationError,
        PublicGraphPack,
        array_sha256,
        atomic_write_bytes,
        atomic_write_json,
        canonical_json_bytes,
        load_protocol_with_sha256,
        load_public_graph_from_manifest,
        load_public_manifest_with_sha256,
        load_source_targets_pack,
        resolve_manifest_path,
        sha256_bytes,
        sha256_file,
        validate_candidate_release_root,
        write_deterministic_npz,
    )
except ImportError:
    from natural_shift_io import (
        TARGET_DOMAINS,
        ArtifactValidationError,
        PublicGraphPack,
        array_sha256,
        atomic_write_bytes,
        atomic_write_json,
        canonical_json_bytes,
        load_protocol_with_sha256,
        load_public_graph_from_manifest,
        load_public_manifest_with_sha256,
        load_source_targets_pack,
        resolve_manifest_path,
        sha256_bytes,
        sha256_file,
        validate_candidate_release_root,
        write_deterministic_npz,
    )


CANDIDATE_MANIFEST_SCHEMA = "twitch_natural_shift_candidate.v1"
_STRATEGY_IDS = (
    "reject_all",
    "always",
    "fixed_combined",
    "delta_only",
    "phi_only",
    "source_risk",
    "online_rollback",
)


def _load_local_stack() -> tuple[Any, Any, Any, Any]:
    """Import the repository's local model/adaptation stack in either run mode."""

    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from adaptation import adapt_classifier
    from data_adapter import UnlabeledGraphView
    from detector import DetectorState
    from models import make_model, train_model

    return (
        adapt_classifier,
        UnlabeledGraphView,
        DetectorState,
        (make_model, train_model),
    )


def _source_path(export_root: Path, supplied: str | Path) -> Path:
    candidate = Path(supplied)
    if not candidate.is_absolute():
        candidate = export_root / candidate
    resolved = candidate.resolve(strict=True)
    root = export_root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ArtifactValidationError(
            "source supervision pack must remain inside the public export root"
        ) from exc
    if relative != "source/DE.npz":
        raise ArtifactValidationError(
            "source supervision must be the canonical source/DE.npz artifact"
        )
    return resolve_manifest_path(root, relative)


def _adjacency(graph: PublicGraphPack) -> scipy.sparse.csr_matrix:
    edges = graph.edge_index
    values = np.ones(edges.shape[1], dtype=np.float32)
    return scipy.sparse.csr_matrix(
        (values, (edges[0], edges[1])),
        shape=(graph.num_nodes, graph.num_nodes),
        dtype=np.float32,
    )


def _stratified_split(
    targets: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for class_value in sorted(np.unique(targets).tolist()):
        indices = np.flatnonzero(targets == class_value).astype(np.int64)
        if len(indices) < 3:
            raise ArtifactValidationError(
                "each source class needs at least three nodes for the frozen split"
            )
        rng = np.random.default_rng(int(seed) + 1009 * int(class_value))
        indices = indices[rng.permutation(len(indices))]
        train_count = max(1, int(math.floor(len(indices) * train_fraction)))
        validation_count = max(1, int(math.floor(len(indices) * validation_fraction)))
        if train_count + validation_count >= len(indices):
            validation_count = 1
            train_count = len(indices) - 2
        train_parts.append(indices[:train_count])
        validation_parts.append(indices[train_count : train_count + validation_count])
        test_parts.append(indices[train_count + validation_count :])
    return tuple(
        np.sort(np.concatenate(parts)).astype(np.int64)
        for parts in (train_parts, validation_parts, test_parts)
    )


def _mask(indices: np.ndarray, num_nodes: int) -> torch.Tensor:
    result = torch.zeros(num_nodes, dtype=torch.bool)
    result[torch.as_tensor(indices, dtype=torch.long)] = True
    return result


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / math.log(clipped.shape[1])


def _validate_probabilities(
    probabilities: np.ndarray, num_nodes: int, context: str
) -> np.ndarray:
    value = np.asarray(probabilities, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != num_nodes or value.shape[1] != 2:
        raise FloatingPointError(f"{context} has an invalid probability shape")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{context} contains non-finite probabilities")
    if np.any(value < -1e-7) or np.any(value > 1.0 + 1e-7):
        raise FloatingPointError(f"{context} is outside [0, 1]")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise FloatingPointError(f"{context} rows do not sum to one")
    return value.astype(np.float32)


def _source_atc_threshold(
    probabilities: np.ndarray, targets: np.ndarray, validation_indices: np.ndarray
) -> float:
    local = probabilities[validation_indices]
    confidence = np.max(local, axis=1)
    correctness = np.argmax(local, axis=1) == targets[validation_indices]
    source_fraction_correct = float(np.mean(correctness))
    return float(
        np.quantile(
            confidence,
            np.clip(1.0 - source_fraction_correct, 0.0, 1.0),
            method="linear",
        )
    )


def _native_float_list(values: Any, context: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise FloatingPointError(f"{context} contains a non-finite value")
    return result


def _run_trajectory(
    model: Any,
    graph: PublicGraphPack,
    adapter: str,
    seed: int,
    adaptation_config: Mapping[str, Any],
    source_atc_threshold: float,
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    adapt_classifier, unlabeled_type, detector_type, _ = _load_local_stack()
    # Public packs are immutable NumPy views; Torch must own writable copies.
    x_tensor = torch.from_numpy(np.array(graph.x, copy=True)).to(dtype=torch.float32)
    edge_tensor = torch.from_numpy(np.array(graph.edge_index, copy=True)).to(
        dtype=torch.long
    )
    graph_view = unlabeled_type(
        x_np=graph.x,
        adj=_adjacency(graph),
        x=x_tensor,
        edge_index=edge_tensor,
    )
    source_probabilities = _validate_probabilities(
        model.predict_probs(x_tensor, edge_tensor).detach().cpu().numpy(),
        graph.num_nodes,
        "source model",
    )
    run_kwargs = {
        "method": adapter,
        "seed": int(seed),
        "steps": int(adaptation_config["steps"]),
        "lr": float(adaptation_config["learning_rate"]),
        "lambda_cal": float(adaptation_config["lambda_cal"]),
        "lambda_af": float(adaptation_config["lambda_af"]),
    }

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    unguarded_model = model.clone()
    trajectory_failure: str | None = None
    try:
        unguarded_info = adapt_classifier(
            unguarded_model, graph_view, detector=None, **run_kwargs
        )
        candidate_probabilities = _validate_probabilities(
            unguarded_model.predict_probs(x_tensor, edge_tensor).detach().cpu().numpy(),
            graph.num_nodes,
            "unguarded candidate",
        )
        delta_trace = _native_float_list(unguarded_info["delta_trace"], "delta trace")
        phi_trace = _native_float_list(unguarded_info["phi_trace"], "phi trace")
        if unguarded_info.get("numerical_failure") is not None:
            raise FloatingPointError(str(unguarded_info["numerical_failure"]))
    except Exception as exc:
        trajectory_failure = f"{type(exc).__name__}:{exc}"
        candidate_probabilities = source_probabilities.copy()
        delta_trace = []
        phi_trace = []

    endpoint_delta = delta_trace[-1] if delta_trace else 0.0
    endpoint_phi = phi_trace[-1] if phi_trace else 0.0
    source_atc = float(
        np.mean(np.max(source_probabilities, axis=1) >= source_atc_threshold)
    )
    candidate_atc = float(
        np.mean(np.max(candidate_probabilities, axis=1) >= source_atc_threshold)
    )
    estimated_gain = candidate_atc - source_atc
    candidate_available = trajectory_failure is None

    fixed_pass = bool(
        candidate_available
        and endpoint_delta <= float(thresholds["delta"])
        and endpoint_phi <= float(thresholds["phi"])
    )
    decisions: dict[str, dict[str, Any]] = {
        "reject_all": {"accepted": False, "reason": "precommitted_reject_all"},
        "always": {
            "accepted": bool(candidate_available),
            "reason": "candidate_available" if candidate_available else "failed_closed",
        },
        "fixed_combined": {
            "accepted": fixed_pass,
            "reason": (
                "endpoint_delta_and_phi" if fixed_pass else "threshold_or_failure"
            ),
        },
        "delta_only": {
            "accepted": bool(
                candidate_available and endpoint_delta <= float(thresholds["delta"])
            ),
            "reason": "endpoint_delta",
        },
        "phi_only": {
            "accepted": bool(
                candidate_available and endpoint_phi <= float(thresholds["phi"])
            ),
            "reason": "endpoint_phi",
        },
        "source_risk": {
            "accepted": bool(
                candidate_available and estimated_gain >= -float(thresholds["harm"])
            ),
            "reason": (
                "source_calibrated_atc_gain_lower_than_negative_harm"
                if estimated_gain < -float(thresholds["harm"])
                else "source_calibrated_atc_gain"
            ),
        },
    }

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    online_model = model.clone()
    online_detector = detector_type(
        delta_tolerance=float(thresholds["delta"]),
        phi_tolerance=float(thresholds["phi"]),
    )
    online_failure: str | None = None
    try:
        online_info = adapt_classifier(
            online_model, graph_view, detector=online_detector, **run_kwargs
        )
        online_probabilities = _validate_probabilities(
            online_model.predict_probs(x_tensor, edge_tensor).detach().cpu().numpy(),
            graph.num_nodes,
            "online rollback",
        )
        if online_info.get("numerical_failure") is not None:
            raise FloatingPointError(str(online_info["numerical_failure"]))
    except Exception as exc:
        online_failure = f"{type(exc).__name__}:{exc}"
        online_probabilities = source_probabilities.copy()
        detector_snapshot = online_detector.to_dict()
        online_info = {
            "detector": detector_snapshot,
            "steps": int(
                detector_snapshot["accepted_candidates"]
                + detector_snapshot["rejected_candidates"]
            ),
            "delta_trace": [],
            "phi_trace": [],
        }
    online_deployed = bool(
        online_failure is None
        and not np.array_equal(online_probabilities, source_probabilities)
    )
    decisions["online_rollback"] = {
        "accepted": online_deployed,
        "reason": (
            "online_checkpoint_deployed"
            if online_deployed
            else "source_checkpoint_deployed"
        ),
    }

    deployed: dict[str, np.ndarray] = {}
    for strategy in _STRATEGY_IDS:
        if strategy == "online_rollback":
            deployed[strategy] = online_probabilities
        elif decisions[strategy]["accepted"]:
            deployed[strategy] = candidate_probabilities
        else:
            deployed[strategy] = source_probabilities
    arrays = {
        "candidate_probabilities": candidate_probabilities,
        "node_ids": graph.node_ids,
        "source_probabilities": source_probabilities,
        **{f"deployed__{strategy}": deployed[strategy] for strategy in _STRATEGY_IDS},
    }
    trace_payload = {
        "adapter": adapter,
        "candidate_probabilities_sha256": array_sha256(candidate_probabilities),
        "delta": delta_trace,
        "phi": phi_trace,
        "seed": int(seed),
        "source_probabilities_sha256": array_sha256(source_probabilities),
    }
    online_record = {
        "failure": online_failure,
        "state": online_info.get("detector"),
        "steps": int(online_info.get("steps", 0)),
    }
    online_trace_payload = {
        "adapter": adapter,
        "deployed_probabilities_sha256": array_sha256(online_probabilities),
        "failure": online_record["failure"],
        "seed": int(seed),
        "state": online_record["state"],
        "steps": online_record["steps"],
    }
    metadata = {
        "decisions": decisions,
        "endpoint_delta": float(endpoint_delta),
        "endpoint_phi": float(endpoint_phi),
        "failure": trajectory_failure,
        "online": online_record,
        "online_trajectory_sha256": sha256_bytes(
            canonical_json_bytes(online_trace_payload)
        ),
        "proxy_trace": {"delta": delta_trace, "phi": phi_trace},
        "source_risk_estimated_gain": float(estimated_gain),
        "status": "complete" if trajectory_failure is None else "failed_closed",
        "trajectory_sha256": sha256_bytes(canonical_json_bytes(trace_payload)),
    }
    return arrays, metadata


def _cleanup_staging(staging: Path, parent: Path) -> None:
    if not staging.exists():
        return
    resolved_parent = parent.resolve(strict=True)
    resolved_staging = staging.resolve(strict=True)
    try:
        resolved_staging.relative_to(resolved_parent)
    except ValueError as exc:
        raise RuntimeError(
            "refusing to clean staging outside candidate parent"
        ) from exc
    if not staging.name.startswith(".natural-candidate-"):
        raise RuntimeError("refusing to clean an unexpected staging directory")
    shutil.rmtree(staging)


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def run_candidate_pipeline(
    *,
    protocol_path: str | Path,
    public_manifest_path: str | Path,
    source_targets_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """Execute the complete frozen candidate Cartesian product atomically."""

    protocol, protocol_file_digest = load_protocol_with_sha256(protocol_path)
    manifest_path = Path(public_manifest_path).resolve(strict=True)
    release_root = validate_candidate_release_root(manifest_path.parent)
    if manifest_path != release_root / "public_manifest.json":
        raise ArtifactValidationError(
            "public manifest must be the canonical candidate release manifest"
        )
    public_manifest, public_manifest_digest = load_public_manifest_with_sha256(
        manifest_path
    )
    if protocol["data"]["source_domain"] != public_manifest["source_domain"]:
        raise ArtifactValidationError("protocol/public source domain mismatch")
    if protocol["data"]["target_domains"] != public_manifest["target_domains"]:
        raise ArtifactValidationError("protocol/public target domains mismatch")
    if protocol["data"]["feature_dim"] != public_manifest["feature_dim"]:
        raise ArtifactValidationError("protocol/public feature dimension mismatch")
    if protocol["data"]["node_id_semantics"] != public_manifest["node_id_semantics"]:
        raise ArtifactValidationError("protocol/public node-ID semantics mismatch")

    export_root = release_root
    source_graph = load_public_graph_from_manifest(manifest_path, public_manifest, "DE")
    source_supervision_path = _source_path(export_root, source_targets_path)
    source_supervision = load_source_targets_pack(
        source_supervision_path,
        expected_public_sha256=public_manifest["graphs"]["DE"]["sha256"],
        expected_node_ids=source_graph.node_ids,
    )
    target_graphs = {
        domain: load_public_graph_from_manifest(manifest_path, public_manifest, domain)
        for domain in TARGET_DOMAINS
    }

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".natural-candidate-", dir=destination.parent)
    )
    started = time.perf_counter()
    try:
        (staging / "artifacts").mkdir()
        adapt_classifier, unlabeled_type, detector_type, model_stack = (
            _load_local_stack()
        )
        del adapt_classifier, unlabeled_type, detector_type
        make_model, train_model = model_stack
        source_models: dict[str, dict[str, Any]] = {}
        runs: list[dict[str, Any]] = []
        split_config = protocol["source_split"]
        model_config = protocol["models"]
        adaptation_config = protocol["adaptation"]

        for seed in protocol["seeds"]:
            train_indices, validation_indices, test_indices = _stratified_split(
                source_supervision.targets,
                int(seed),
                float(split_config["train_fraction"]),
                float(split_config["validation_fraction"]),
            )
            source_x = torch.from_numpy(np.array(source_graph.x, copy=True)).to(
                dtype=torch.float32
            )
            source_edges = torch.from_numpy(
                np.array(source_graph.edge_index, copy=True)
            ).to(dtype=torch.long)
            source_values = torch.from_numpy(
                np.array(source_supervision.targets, copy=True)
            ).to(dtype=torch.long)
            for backbone in model_config["backbones"]:
                torch.manual_seed(int(seed))
                np.random.seed(int(seed))
                model = make_model(
                    backbone,
                    source_graph.x.shape[1],
                    int(model_config["hidden_dim"]),
                    2,
                    use_bn=bool(model_config["use_batch_norm"]),
                    seed=int(seed),
                )
                train_info = train_model(
                    model,
                    source_x,
                    source_edges,
                    source_values,
                    _mask(train_indices, source_graph.num_nodes),
                    _mask(validation_indices, source_graph.num_nodes),
                    epochs=int(model_config["train_epochs"]),
                    lr=float(model_config["learning_rate"]),
                    weight_decay=float(model_config["weight_decay"]),
                    patience=int(model_config["patience"]),
                )
                source_probabilities = _validate_probabilities(
                    model.predict_probs(source_x, source_edges).detach().cpu().numpy(),
                    source_graph.num_nodes,
                    "trained DE source model",
                )
                atc_threshold = _source_atc_threshold(
                    source_probabilities,
                    source_supervision.targets,
                    validation_indices,
                )
                model_key = f"seed-{seed}__{backbone}"
                model_digest = _model_state_sha256(model)
                source_models[model_key] = {
                    "backbone": backbone,
                    "checkpoint_sha256": model_digest,
                    "seed": int(seed),
                    "source_atc_threshold": float(atc_threshold),
                    "split_counts": {
                        "test": int(len(test_indices)),
                        "train": int(len(train_indices)),
                        "validation": int(len(validation_indices)),
                    },
                    "training_epochs_completed": int(train_info["epochs"]),
                }

                for domain in TARGET_DOMAINS:
                    graph = target_graphs[domain]
                    for adapter in adaptation_config["adapters"]:
                        run_started = time.perf_counter()
                        run_seed = int(seed)
                        arrays, metadata = _run_trajectory(
                            model,
                            graph,
                            adapter,
                            run_seed,
                            adaptation_config,
                            atc_threshold,
                            protocol["thresholds"],
                        )
                        run_id = f"{domain}__seed-{seed}__{backbone}__{adapter}"
                        artifact_relative = f"artifacts/{run_id}.npz"
                        artifact_path = staging / Path(artifact_relative)
                        write_deterministic_npz(artifact_path, arrays)
                        run_record = {
                            "adapter": adapter,
                            "artifact_path": artifact_relative,
                            "artifact_sha256": sha256_file(artifact_path),
                            "backbone": backbone,
                            "decisions": metadata["decisions"],
                            "domain": domain,
                            "endpoint_delta": metadata["endpoint_delta"],
                            "endpoint_phi": metadata["endpoint_phi"],
                            "failure": metadata["failure"],
                            "node_ids_sha256": array_sha256(graph.node_ids),
                            "online": metadata["online"],
                            "online_trajectory_sha256": metadata[
                                "online_trajectory_sha256"
                            ],
                            "proxy_trace": metadata["proxy_trace"],
                            "public_pack_sha256": graph.sha256,
                            "run_id": run_id,
                            "runtime_seconds": float(time.perf_counter() - run_started),
                            "seed": int(seed),
                            "source_checkpoint_sha256": model_digest,
                            "source_risk_estimated_gain": metadata[
                                "source_risk_estimated_gain"
                            ],
                            "status": metadata["status"],
                            "trajectory_sha256": metadata["trajectory_sha256"],
                        }
                        runs.append(run_record)

        candidate_manifest: dict[str, Any] = {
            "artifact_count": int(len(runs)),
            "execution_policy": "complete frozen Cartesian product; no target outcomes",
            "hash_algorithm": "sha256",
            "manifest_payload_sha256": "0" * 64,
            "protocol_file_sha256": protocol_file_digest,
            "protocol_id": protocol["protocol_id"],
            "protocol_payload_sha256": protocol["protocol_payload_sha256"],
            "public_manifest_sha256": public_manifest_digest,
            "runs": runs,
            "schema_version": CANDIDATE_MANIFEST_SCHEMA,
            "source_models": source_models,
            "source_targets_sha256": source_supervision.sha256,
            "status": "finalized",
            "total_runtime_seconds": float(time.perf_counter() - started),
        }
        candidate_manifest["manifest_payload_sha256"] = _manifest_payload_sha256(
            candidate_manifest
        )
        candidate_manifest_path = staging / "candidate_manifest.json"
        atomic_write_json(candidate_manifest_path, candidate_manifest)
        full_manifest_digest = sha256_file(candidate_manifest_path)
        atomic_write_bytes(
            staging / "candidate_manifest.sha256",
            f"{full_manifest_digest}  candidate_manifest.json\n".encode("ascii"),
        )
        os.replace(staging, destination)
        return {
            "candidate_manifest_sha256": full_manifest_digest,
            "candidate_payload_sha256": candidate_manifest["manifest_payload_sha256"],
        }
    except BaseException:
        _cleanup_staging(staging, destination.parent)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the label-isolated prospective Twitch candidates"
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--public-manifest", required=True)
    parser.add_argument("--source-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_candidate_pipeline(
        protocol_path=args.protocol,
        public_manifest_path=args.public_manifest,
        source_targets_path=args.source_targets,
        output_dir=args.output_dir,
    )
    for name, digest in sorted(result.items()):
        print(f"{name}={digest}")


if __name__ == "__main__":
    main()
