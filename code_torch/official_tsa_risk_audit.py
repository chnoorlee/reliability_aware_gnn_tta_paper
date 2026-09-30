"""Audit official TSA-repository adapters with a label-free endpoint guard.

This script orchestrates the MIT-licensed official TSA repository without
reimplementing its models or adapters.  The adapter receives a newly constructed
graph containing only model inputs and zero labels; true target labels and all
evaluation masks remain outside the adapter and determine offline accuracy gain
only after prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.utils import degree

from proxy_scope_audit import audit_proxy_scopes, scope_index_sha256
from run_official_tsa_compat import install_compat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSA_ROOT = PROJECT_ROOT / "external_official" / "TSA"
FIXED_DELTA = 0.05
FIXED_PHI = 0.20
FINGERPRINT_SCHEMA = "official-tsa-row-v2"
ADAPTER_DATA_FIELDS = frozenset({"x", "edge_index", "edge_weight", "y"})
LOCAL_CODE_FILES = (
    "official_tsa_risk_audit.py",
    "proxy_scope_audit.py",
    "run_official_tsa_compat.py",
)

METHOD_OVERRIDES = {
    "T3A": ["adapter=T3A"],
    "Matcha_T3A": ["adapter=Matcha_T3A"],
    "TSA_T3A": ["adapter=TSA_T3A"],
}


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status(path: Path) -> dict:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked = [line for line in status if not line.startswith("??")]
    untracked = [line[3:] for line in status if line.startswith("??")]
    return {"tracked_changes": tracked, "untracked_paths": untracked}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _local_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha256_file(root / name) for name in LOCAL_CODE_FILES}


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _tensor_fingerprint(value: torch.Tensor, *, field: str, kind: str) -> dict:
    """Return a device- and layout-independent semantic tensor fingerprint."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{field} must be a dense strided tensor")
    if value.is_quantized:
        raise TypeError(f"{field} must not be quantized")

    detached = value.detach().cpu()
    observed_dtype = str(detached.dtype)
    nonfinite_counts = None
    if kind == "float":
        if detached.is_complex():
            raise TypeError(f"{field} must be real-valued")
        canonical = (
            detached.to(torch.float64).contiguous().numpy().astype("<f8", copy=False)
        )
        canonical = np.array(canonical, dtype="<f8", order="C", copy=True)
        nan_mask = np.isnan(canonical)
        positive_inf = int(np.count_nonzero(np.isposinf(canonical)))
        negative_inf = int(np.count_nonzero(np.isneginf(canonical)))
        nan_count = int(np.count_nonzero(nan_mask))
        canonical[canonical == 0.0] = 0.0
        canonical[nan_mask] = np.nan
        canonical_dtype = "<f8"
        nonfinite_counts = {
            "nan": nan_count,
            "positive_infinity": positive_inf,
            "negative_infinity": negative_inf,
        }
    elif kind == "integer":
        if (
            detached.dtype == torch.bool
            or detached.is_floating_point()
            or detached.is_complex()
        ):
            raise TypeError(f"{field} must contain integers")
        canonical = (
            detached.to(torch.int64).contiguous().numpy().astype("<i8", copy=False)
        )
        canonical = np.ascontiguousarray(canonical)
        canonical_dtype = "<i8"
    elif kind == "mask":
        if detached.is_floating_point() or detached.is_complex():
            raise TypeError(f"{field} must be a boolean or integer mask")
        if not bool(torch.all((detached == 0) | (detached == 1))):
            raise ValueError(f"{field} must contain only zero/one mask values")
        canonical = detached.to(torch.uint8).contiguous().numpy()
        canonical = np.ascontiguousarray(canonical)
        canonical_dtype = "|u1"
    else:
        raise ValueError(f"Unsupported tensor fingerprint kind: {kind!r}")

    metadata = {
        "schema": "official-tsa-tensor-v1",
        "field": field,
        "shape": list(detached.shape),
        "observed_dtype": observed_dtype,
        "canonical_dtype": canonical_dtype,
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    result: dict[str, object] = {**metadata, "sha256": digest.hexdigest()}
    if nonfinite_counts is not None:
        result["nonfinite_counts"] = nonfinite_counts
    return result


def _data_keys(data) -> tuple[str, ...]:
    keys = getattr(data, "keys", None)
    if callable(keys):
        return tuple(sorted(str(key) for key in keys()))
    return tuple(sorted(str(key) for key in vars(data)))


def _integer_scalar_fingerprint(value, *, field: str) -> dict:
    """Return a canonical fingerprint for an integer graph scalar."""

    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        if detached.numel() != 1:
            raise TypeError(f"{field} must be an integer scalar")
        if (
            detached.dtype == torch.bool
            or detached.is_floating_point()
            or detached.is_complex()
        ):
            raise TypeError(f"{field} must be an integer scalar")
        observed_type = str(detached.dtype)
        canonical_value = int(detached.item())
    elif isinstance(value, np.integer) and not isinstance(value, np.bool_):
        observed_type = str(value.dtype)
        canonical_value = int(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        observed_type = type(value).__name__
        canonical_value = value
    else:
        raise TypeError(f"{field} must be an integer scalar")

    result = {
        "schema": "official-tsa-scalar-v1",
        "field": field,
        "observed_type": observed_type,
        "canonical_type": "integer",
        "canonical_value": canonical_value,
    }
    result["sha256"] = _json_sha256(result)
    return result


def _adapter_data_from_target(target_data) -> Data:
    """Build the complete, label-free graph boundary seen by an adapter."""

    required_tensors = ("x", "edge_index", "y")
    for field in required_tensors:
        if not isinstance(getattr(target_data, field, None), torch.Tensor):
            raise TypeError(f"official target data is missing tensor field {field!r}")

    fields = {
        "x": target_data.x.detach().clone(),
        "edge_index": target_data.edge_index.detach().clone(),
        "y": torch.zeros_like(target_data.y),
    }
    edge_weight = getattr(target_data, "edge_weight", None)
    if edge_weight is not None:
        if not isinstance(edge_weight, torch.Tensor):
            raise TypeError("official target edge_weight must be a torch.Tensor")
        fields["edge_weight"] = edge_weight.detach().clone()
    return Data(**fields)


def _source_fingerprint(source_data) -> dict:
    """Bind every source-graph field consumed by training and source statistics."""

    required_tensors = (
        "x",
        "edge_index",
        "edge_weight",
        "y",
        "src_train_mask",
        "src_val_mask",
        "src_test_mask",
    )
    for field in required_tensors:
        if not isinstance(getattr(source_data, field, None), torch.Tensor):
            raise TypeError(f"official source data is missing tensor field {field!r}")

    if source_data.edge_index.ndim != 2 or source_data.edge_index.shape[0] != 2:
        raise ValueError("official source edge_index must have shape [2, num_edges]")
    num_nodes = int(source_data.x.shape[0])
    num_edges = int(source_data.edge_index.shape[1])
    if source_data.y.ndim != 1 or len(source_data.y) != num_nodes:
        raise ValueError("official source labels must have shape [num_nodes]")

    mask_names = tuple(
        name
        for name in _data_keys(source_data)
        if name.endswith("_mask")
        and isinstance(getattr(source_data, name, None), torch.Tensor)
    )
    for required_mask in ("src_train_mask", "src_val_mask", "src_test_mask"):
        if required_mask not in mask_names:
            raise ValueError(f"official source data must expose {required_mask}")
    for name in mask_names:
        mask = getattr(source_data, name)
        if mask.ndim != 1 or len(mask) != num_nodes:
            raise ValueError(
                f"official source mask {name!r} must have shape [num_nodes]"
            )

    components = {
        "x": _tensor_fingerprint(source_data.x, field="source.x", kind="float"),
        "edge_index_ordered": _tensor_fingerprint(
            source_data.edge_index,
            field="source.edge_index_ordered",
            kind="integer",
        ),
        "edge_weight": _tensor_fingerprint(
            source_data.edge_weight,
            field="source.edge_weight",
            kind="float",
        ),
        "y": _tensor_fingerprint(source_data.y, field="source.y", kind="integer"),
        "masks": {
            name: _tensor_fingerprint(
                getattr(source_data, name), field=f"source.{name}", kind="mask"
            )
            for name in mask_names
        },
    }
    edge_attr = getattr(source_data, "edge_attr", None)
    if isinstance(edge_attr, torch.Tensor):
        components["edge_attr"] = _tensor_fingerprint(
            edge_attr, field="source.edge_attr", kind="float"
        )

    scalars = {
        name: _integer_scalar_fingerprint(
            getattr(source_data, name), field=f"source.{name}"
        )
        for name in ("num_nodes", "num_edges", "num_classes")
    }
    if scalars["num_nodes"]["canonical_value"] != num_nodes:
        raise ValueError("official source num_nodes disagrees with source.x")
    if scalars["num_edges"]["canonical_value"] != num_edges:
        raise ValueError("official source num_edges disagrees with source.edge_index")
    if scalars["num_classes"]["canonical_value"] <= 0:
        raise ValueError("official source num_classes must be positive")

    fingerprint = {
        "schema": FINGERPRINT_SCHEMA,
        "components": components,
        "scalars": scalars,
    }
    fingerprint["sha256"] = _json_sha256(fingerprint)
    return fingerprint


def _target_fingerprints(target_data, adapt_data, scopes) -> dict:
    """Bind actual adapter inputs separately from offline evaluation labels."""

    required = ("x", "edge_index", "y", "tgt_test_mask")
    for field in required:
        if not isinstance(getattr(target_data, field, None), torch.Tensor):
            raise TypeError(f"official target data is missing tensor field {field!r}")
    num_nodes = int(target_data.x.shape[0])
    if target_data.edge_index.ndim != 2 or target_data.edge_index.shape[0] != 2:
        raise ValueError("official target edge_index must have shape [2, num_edges]")
    if target_data.y.ndim != 1 or len(target_data.y) != num_nodes:
        raise ValueError("official target labels must have shape [num_nodes]")

    adapter_keys = set(_data_keys(adapt_data))
    required_adapter_keys = {"x", "edge_index", "y"}
    missing_adapter_keys = sorted(required_adapter_keys - adapter_keys)
    if missing_adapter_keys:
        raise ValueError(
            f"official adapter data is missing required fields: {missing_adapter_keys}"
        )
    unexpected_adapter_keys = sorted(adapter_keys - ADAPTER_DATA_FIELDS)
    if unexpected_adapter_keys:
        raise ValueError(
            "official adapter data crossed the structural input boundary with "
            f"unexpected fields: {unexpected_adapter_keys}"
        )
    for field in ("x", "edge_index", "y"):
        if not isinstance(getattr(adapt_data, field, None), torch.Tensor):
            raise TypeError(f"official adapter data field {field!r} must be a tensor")
    if adapt_data.x.shape[0] != num_nodes:
        raise ValueError("official adapter x disagrees with target node count")
    if adapt_data.edge_index.ndim != 2 or adapt_data.edge_index.shape[0] != 2:
        raise ValueError("official adapter edge_index must have shape [2, num_edges]")
    if adapt_data.y.ndim != 1 or len(adapt_data.y) != num_nodes:
        raise ValueError("official adapter labels must have shape [num_nodes]")
    if bool(torch.any(adapt_data.y != 0)):
        raise ValueError("official adapter labels must be identically zero")

    adapter_components = {
        "x": _tensor_fingerprint(adapt_data.x, field="adapter.x", kind="float"),
        "edge_index_ordered": _tensor_fingerprint(
            adapt_data.edge_index,
            field="adapter.edge_index_ordered",
            kind="integer",
        ),
        "y_zeroed": _tensor_fingerprint(
            adapt_data.y, field="adapter.y_zeroed", kind="integer"
        ),
    }
    for optional_name in ("edge_weight",):
        optional_value = getattr(adapt_data, optional_name, None)
        if isinstance(optional_value, torch.Tensor):
            adapter_components[optional_name] = _tensor_fingerprint(
                optional_value,
                field=f"adapter.{optional_name}",
                kind="float",
            )
        elif optional_name in adapter_keys:
            raise TypeError(f"official adapter {optional_name} must be a tensor")
    adapter_input = {
        "schema": FINGERPRINT_SCHEMA,
        "components": adapter_components,
    }
    adapter_input["sha256"] = _json_sha256(adapter_input)

    offline_evaluation = {
        "schema": FINGERPRINT_SCHEMA,
        "components": {
            "y": _tensor_fingerprint(target_data.y, field="offline.y", kind="integer"),
            "tgt_test_mask": _tensor_fingerprint(
                target_data.tgt_test_mask,
                field="offline.tgt_test_mask",
                kind="mask",
            ),
        },
    }
    offline_evaluation["sha256"] = _json_sha256(offline_evaluation)

    scope_indices = {
        name: scope_index_sha256(mask, num_nodes) for name, mask in scopes.items()
    }
    return {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "adapter_input_fingerprint": adapter_input,
        "offline_evaluation_fingerprint": offline_evaluation,
        "proxy_scope_index_sha256": scope_indices,
    }


def _write_json_atomic(path: Path, value) -> None:
    """Replace a JSON artifact atomically so interrupted runs cannot truncate it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_audit_lock():
    """Prevent concurrent audits from sharing official checkpoints and outputs."""
    lock_path = TSA_ROOT / ".official_tsa_risk_audit.lock"
    run_id = str(uuid.uuid4())
    lock_payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "created_unix_seconds": time.time(),
    }
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            "Another official TSA audit owns the shared checkpoint directory. "
            f"Lock contents: {owner}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(lock_payload, stream, indent=2, allow_nan=False)
        yield lock_payload
    finally:
        lock_path.unlink(missing_ok=True)


def _degree_groups(edge_index: torch.Tensor, num_nodes: int):
    deg = degree(edge_index[0], num_nodes=num_nodes).detach().cpu().numpy()
    node_id: np.ndarray = np.arange(num_nodes, dtype=int)
    order = np.lexsort((node_id, deg))
    return tuple(np.asarray(group, dtype=int) for group in np.array_split(order, 3))


def _guard_signals(edge_index, source_probs, candidate_probs):
    source = source_probs.detach().cpu().numpy()
    candidate = candidate_probs.detach().cpu().numpy()
    groups = _degree_groups(edge_index, len(source))
    source_conf = np.max(source, axis=1)
    candidate_conf = np.max(candidate, axis=1)
    delta = float(
        np.mean(
            [
                abs(float(np.mean(candidate_conf[g])) - float(np.mean(source_conf[g])))
                for g in groups
            ]
        )
    )
    phi = float(np.mean(np.argmax(candidate, axis=1) != np.argmax(source, axis=1)))
    return delta, phi


def _scope_masks(target_data, num_nodes: int) -> dict[str, np.ndarray]:
    evaluation = (
        target_data.tgt_test_mask.detach().cpu().numpy().astype(bool, copy=True)
    )
    if evaluation.shape != (num_nodes,):
        raise ValueError("official evaluation mask must have shape [num_nodes]")
    return {
        "target": np.ones(num_nodes, dtype=bool),
        "evaluation": evaluation,
        "non_evaluation": ~evaluation,
    }


def _scope_manifest(scopes, num_nodes: int) -> dict[str, dict]:
    return {
        name: {
            "num_nodes": int(np.count_nonzero(mask)),
            "scope_index_sha256": scope_index_sha256(mask, num_nodes),
        }
        for name, mask in scopes.items()
    }


def _endpoint_proxy_scopes(target_data, source_probs, candidate_probs) -> dict:
    source = source_probs.detach().cpu().numpy()
    candidate = candidate_probs.detach().cpu().numpy()
    scopes = _scope_masks(target_data, len(source))
    groups = _degree_groups(target_data.edge_index, len(source))
    return audit_proxy_scopes(groups, source, candidate, scopes)


def _accuracy(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    predicted = torch.argmax(probs, dim=1)
    labels = labels.to(predicted.device)
    mask = mask.to(predicted.device)
    return float((predicted[mask] == labels[mask]).float().mean().item())


def _finalize_row_hashes(
    identity: dict,
    result_fields: dict,
    provenance: dict,
    row_protocol_sha256: str,
) -> dict:
    """Bind row identity, inputs, outputs, protocol, and derived decisions."""

    row_identity_sha256 = _json_sha256(
        {"schema": FINGERPRINT_SCHEMA, "identity": identity}
    )
    row_inputs_sha256 = _json_sha256(
        {
            "schema": FINGERPRINT_SCHEMA,
            "resolved_config_sha256": provenance["resolved_config_sha256"],
            "source_training_config_sha256": provenance[
                "source_training_config_sha256"
            ],
            "source_checkpoint_sha256": provenance["source_checkpoint_sha256"],
            "source_data_sha256": provenance["source_data_fingerprint"]["sha256"],
            "adapter_input_sha256": provenance["adapter_input_fingerprint"]["sha256"],
            "offline_evaluation_sha256": provenance["offline_evaluation_fingerprint"][
                "sha256"
            ],
            "proxy_scope_index_sha256": provenance["proxy_scope_index_sha256"],
        }
    )
    row_outputs_sha256 = _json_sha256(
        {
            "schema": FINGERPRINT_SCHEMA,
            "source_probabilities_sha256": provenance[
                "source_probabilities_fingerprint"
            ]["sha256"],
            "candidate_probabilities_sha256": provenance[
                "candidate_probabilities_fingerprint"
            ]["sha256"],
        }
    )
    row_evidence_sha256 = _json_sha256(
        {
            "schema": FINGERPRINT_SCHEMA,
            "row_identity_sha256": row_identity_sha256,
            "row_protocol_sha256": row_protocol_sha256,
            "row_inputs_sha256": row_inputs_sha256,
            "row_outputs_sha256": row_outputs_sha256,
            "derived_results": result_fields,
        }
    )
    row = {
        **identity,
        **result_fields,
        **provenance,
        "row_identity_sha256": row_identity_sha256,
        "row_protocol_sha256": row_protocol_sha256,
        "row_inputs_sha256": row_inputs_sha256,
        "row_outputs_sha256": row_outputs_sha256,
        "row_evidence_sha256": row_evidence_sha256,
    }
    _refresh_row_content_sha256(row)
    return row


def _refresh_row_content_sha256(row: dict) -> None:
    """Bind the hash to the complete emitted row, excluding the hash itself."""

    content = {key: value for key, value in row.items() if key != "row_content_sha256"}
    row["row_content_sha256"] = _json_sha256(content)


def _compose_config(data_name: str, method: str, model_name: str):
    compute_device = "cpu" if method == "Matcha_T3A" else "cuda:0"
    overrides = [
        f"data={data_name}",
        *METHOD_OVERRIDES[method],
        f"model={model_name}",
        "+model_config.train_type=supervised",
        f"general_config.device={compute_device}",
    ]
    with initialize_config_dir(
        version_base=None, config_dir=str((TSA_ROOT / "configs").resolve())
    ):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg, overrides


def _forward_probs(model, data):
    model.eval()
    model.to(model.device)
    data = data.to(model.device)
    with torch.no_grad():
        _, logits = model(data)
        return torch.softmax(logits, dim=-1)


def _quarantine_checkpoint(checkpoint: Path, quarantine_root: Path):
    relative = checkpoint.resolve().relative_to(TSA_ROOT.resolve())
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        suffix = 1
        while destination.with_suffix(destination.suffix + f".bak{suffix}").exists():
            suffix += 1
        destination = destination.with_suffix(destination.suffix + f".bak{suffix}")
    checkpoint.replace(destination)
    return destination


def _run_one(
    data_name: str,
    method: str,
    seed: int,
    model_name: str,
    prepared_checkpoints: dict[tuple[str, int, str], dict],
    quarantine_root: Path,
    snapshot_root: Path,
    force_retrain: bool,
    row_protocol_sha256: str,
):
    from src.adaptation import adapter_manager
    from src.data import dataset_manager
    from src.model import model_manager
    from src.utils import Metrics, set_config_seed, set_seed

    cfg, overrides = _compose_config(data_name, method, model_name)
    set_seed(seed)
    set_config_seed(seed, cfg.data_config, cfg.model_config, cfg.adapter_config)
    metrics = Metrics(str(PROJECT_ROOT / "revision_2026-08-19"), cfg.data_config.name)

    source_data = dataset_manager(src_tgt="source", data_config=cfg.data_config)
    source_data_fingerprint = _source_fingerprint(source_data)
    model = model_manager(metrics=metrics, model_config=cfg.model_config)
    checkpoint = model.model_path.resolve()
    source_key = (str(cfg.data_config.source), int(seed), model_name)
    quarantined = None
    source_training_config = {
        "data_name": cfg.data_config.name,
        "source_setting": cfg.data_config.source,
        "seed": seed,
        "model_config": OmegaConf.to_container(cfg.model_config, resolve=True),
    }
    if source_key not in prepared_checkpoints:
        if force_retrain and checkpoint.exists():
            quarantined = _quarantine_checkpoint(checkpoint, quarantine_root)
        model.get_pretrain(source_data)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Expected source checkpoint was not created: {checkpoint}"
            )
        checkpoint_sha256 = _sha256_file(checkpoint)
        relative = checkpoint.relative_to(TSA_ROOT.resolve())
        snapshot = (
            snapshot_root
            / relative.parent
            / f"{checkpoint.stem}.{checkpoint_sha256}{checkpoint.suffix}"
        )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, snapshot)
        if _sha256_file(snapshot) != checkpoint_sha256:
            raise RuntimeError("Content-addressed source-checkpoint snapshot mismatch")
        prepared_checkpoints[source_key] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "snapshot": snapshot.resolve(),
            "source_training_config": source_training_config,
            "source_training_config_sha256": _json_sha256(source_training_config),
            "source_data_sha256": source_data_fingerprint["sha256"],
        }
    else:
        checkpoint_provenance = prepared_checkpoints[source_key]
        if (
            source_data_fingerprint["sha256"]
            != checkpoint_provenance["source_data_sha256"]
        ):
            raise RuntimeError(
                "Source data changed before checkpoint reuse; "
                "refusing contaminated audit"
            )
        if checkpoint != checkpoint_provenance["checkpoint"]:
            raise RuntimeError(
                "One source-setting/seed key resolved to multiple checkpoint paths: "
                f"{checkpoint_provenance['checkpoint']} versus {checkpoint}"
            )
        if (
            not checkpoint.exists()
            or _sha256_file(checkpoint) != checkpoint_provenance["checkpoint_sha256"]
        ):
            raise RuntimeError(
                "Source checkpoint changed before reuse; refusing contaminated audit"
            )
        model.get_pretrain(source_data)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Expected source checkpoint was not created: {checkpoint}"
        )
    checkpoint_provenance = prepared_checkpoints[source_key]
    if source_data_fingerprint["sha256"] != checkpoint_provenance["source_data_sha256"]:
        raise RuntimeError(
            "Source data changed before checkpoint reuse; refusing contaminated audit"
        )
    if _sha256_file(checkpoint) != checkpoint_provenance["checkpoint_sha256"]:
        raise RuntimeError(
            "Source checkpoint changed while loading source statistics; "
            "refusing contaminated audit"
        )
    if _source_fingerprint(source_data)["sha256"] != source_data_fingerprint["sha256"]:
        raise RuntimeError(
            "Source data changed while loading the checkpoint; refusing contaminated audit"
        )
    source_stats = model.get_src_stats(source_data)
    if _source_fingerprint(source_data)["sha256"] != source_data_fingerprint["sha256"]:
        raise RuntimeError(
            "Source data changed while computing source statistics; "
            "refusing contaminated audit"
        )

    target_data = dataset_manager(src_tgt="target", data_config=cfg.data_config)
    true_labels = target_data.y.detach().clone()
    adapt_data = _adapter_data_from_target(target_data)
    adapt_data = adapt_data.to(model.device)
    scopes = _scope_masks(target_data, int(target_data.x.shape[0]))
    target_fingerprints = _target_fingerprints(target_data, adapt_data, scopes)
    source_probs = _forward_probs(model, adapt_data)
    adapter = adapter_manager(model, source_stats, adapter_config=cfg.adapter_config)
    candidate_probs = adapter.adapt(adapt_data)

    if _sha256_file(checkpoint) != checkpoint_provenance["checkpoint_sha256"]:
        raise RuntimeError(
            "Source checkpoint changed during adaptation; refusing contaminated audit"
        )

    if not torch.isfinite(source_probs).all():
        raise FloatingPointError("Non-finite official source probabilities")

    source_accuracy = _accuracy(source_probs, true_labels, target_data.tgt_test_mask)
    scope_manifest = _scope_manifest(scopes, len(source_probs))
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    source_probabilities_fingerprint = _tensor_fingerprint(
        source_probs, field="output.source_probabilities", kind="float"
    )
    candidate_probabilities_fingerprint = _tensor_fingerprint(
        candidate_probs, field="output.candidate_probabilities", kind="float"
    )
    provenance = {
        "hydra_overrides": overrides,
        "resolved_config": resolved_config,
        "resolved_config_sha256": _json_sha256(resolved_config),
        "source_training_config": checkpoint_provenance["source_training_config"],
        "source_training_config_sha256": checkpoint_provenance[
            "source_training_config_sha256"
        ],
        "source_checkpoint_path": str(checkpoint.relative_to(TSA_ROOT.resolve())),
        "source_checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
        "source_checkpoint_snapshot_path": str(
            checkpoint_provenance["snapshot"].relative_to(PROJECT_ROOT.resolve())
        ),
        "source_checkpoint_snapshot_sha256": _sha256_file(
            checkpoint_provenance["snapshot"]
        ),
        "source_data_fingerprint": source_data_fingerprint,
        "preexisting_checkpoint_quarantined_to": (
            str(quarantined) if quarantined is not None else None
        ),
        **target_fingerprints,
        "source_probabilities_fingerprint": source_probabilities_fingerprint,
        "candidate_probabilities_fingerprint": candidate_probabilities_fingerprint,
    }
    identity = {
        "data_config": data_name,
        "source_setting": cfg.data_config.source,
        "target_setting": cfg.data_config.target,
        "method": method,
        "model": model_name,
        "seed": seed,
    }
    if not torch.isfinite(candidate_probs).all():
        result_fields = {
            "candidate_status": "nonfinite_probability",
            "source_accuracy": source_accuracy,
            "candidate_accuracy": None,
            "source_relative_accuracy": None,
            "delta": None,
            "phi": None,
            "proxy_scope_manifest": scope_manifest,
            "endpoint_proxy_scopes": None,
            "evaluation_turnover_bound_slack": None,
            "fixed_guard_accept": False,
            "deployed_accuracy": source_accuracy,
            "harmful_over_1pp": None,
        }
        return _finalize_row_hashes(
            identity, result_fields, provenance, row_protocol_sha256
        )

    delta, phi = _guard_signals(target_data.edge_index, source_probs, candidate_probs)
    endpoint_proxy_scopes = _endpoint_proxy_scopes(
        target_data, source_probs, candidate_probs
    )
    if not np.isclose(
        delta, endpoint_proxy_scopes["target"]["delta"], rtol=0.0, atol=1e-12
    ) or not np.isclose(
        phi, endpoint_proxy_scopes["target"]["phi"], rtol=0.0, atol=1e-12
    ):
        raise AssertionError(
            "target-scope endpoint diagnostics do not reproduce guard signals"
        )
    candidate_accuracy = _accuracy(
        candidate_probs, true_labels, target_data.tgt_test_mask
    )
    gain = candidate_accuracy - source_accuracy
    evaluation_phi = endpoint_proxy_scopes["evaluation"]["phi"]
    if abs(gain) > evaluation_phi + 1e-6:
        raise AssertionError(
            "same-scope prediction turnover failed to bound the absolute accuracy change"
        )
    accepted = delta <= FIXED_DELTA and phi <= FIXED_PHI
    result_fields = {
        "candidate_status": "finite",
        "source_accuracy": source_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "source_relative_accuracy": gain,
        "delta": delta,
        "phi": phi,
        "proxy_scope_manifest": scope_manifest,
        "endpoint_proxy_scopes": endpoint_proxy_scopes,
        "evaluation_turnover_bound_slack": evaluation_phi - abs(gain),
        "fixed_guard_accept": accepted,
        "deployed_accuracy": candidate_accuracy if accepted else source_accuracy,
        "harmful_over_1pp": gain < -0.01,
    }
    return _finalize_row_hashes(
        identity, result_fields, provenance, row_protocol_sha256
    )


def _run_locked(
    out_path: Path,
    data_configs,
    methods,
    seeds,
    model_name: str,
    lock_payload: dict,
    force_retrain: bool = True,
):
    out_path = out_path.resolve()
    if not data_configs:
        raise ValueError("At least one data configuration is required")
    if not methods:
        raise ValueError("At least one adaptation method is required")
    if not seeds:
        raise ValueError("At least one seed is required")
    install_compat(TSA_ROOT)
    started = time.perf_counter()
    records: list[dict] = []
    repository_status_before = _git_status(TSA_ROOT)
    if repository_status_before["tracked_changes"]:
        raise RuntimeError(
            "Official TSA repository has tracked changes; refusing provenance audit: "
            f"{repository_status_before['tracked_changes']}"
        )
    quarantine_root = out_path.parent / "preexisting_checkpoint_quarantine"
    snapshot_root = out_path.parent / "source_checkpoint_snapshots"
    prepared_checkpoints: dict[tuple[str, int, str], dict] = {}
    expected_target_fingerprints: dict[tuple[str, str, int], dict] = {}
    official_commit = _git_commit(TSA_ROOT)
    local_code_sha256 = _local_code_sha256()
    row_protocol = {
        "schema": FINGERPRINT_SCHEMA,
        "official_commit": official_commit,
        "local_code_sha256": local_code_sha256,
        "guard": {
            "delta_max": FIXED_DELTA,
            "phi_max": FIXED_PHI,
            "comparison": "inclusive",
        },
        "fresh_source_checkpoint_training": bool(force_retrain),
    }
    row_protocol_sha256 = _json_sha256(row_protocol)
    metadata = {
        "evidence_status": "official_repository_default_config_provenance_audit",
        "official_repository": "https://github.com/Graph-COM/TSA",
        "official_commit": official_commit,
        "license": "MIT",
        "repository_status_before": repository_status_before,
        "compatibility_scope": "import and Python-3.14 CLI shims only",
        "tracked_official_code_required_clean": True,
        "exclusive_audit_lock": lock_payload,
        "atomic_result_writes": True,
        "fresh_source_checkpoint_training": bool(force_retrain),
        "preexisting_checkpoint_quarantine": str(quarantine_root.resolve()),
        "content_addressed_source_checkpoint_snapshots": str(snapshot_root.resolve()),
        "configuration_scope": (
            "official repository defaults for GPRGNN and each adapter; only data, "
            "adapter, model, supervised source-training selectors, and the compute-only "
            "device selector are overridden. CUDA is used for T3A/TSA-T3A and source "
            "training; Matcha-T3A uses CPU to avoid a label-diagnostic device mismatch"
        ),
        "implementation_scope": (
            "T3A, Matcha_T3A, and TSA_T3A implementations bundled in the "
            "official TSA repository; Matcha_T3A is not claimed as a run of "
            "the separate Matcha-authors repository"
        ),
        "label_isolation": (
            "source and candidate probabilities are computed from a newly constructed "
            "Data object containing only x, ordered edge_index, optional edge_weight, "
            "and zero y; true target labels, masks, evaluation indices, and target-derived "
            "metadata remain outside the adapter and are read only by the outer offline "
            "evaluator after both predictions"
        ),
        "adapter_input_fields": sorted(ADAPTER_DATA_FIELDS),
        "data_configs": list(data_configs),
        "methods": list(methods),
        "model": model_name,
        "seeds": list(seeds),
        "guard": "accept iff endpoint delta <= 0.05 and phi <= 0.20",
        "proxy_scope_audit": {
            "status": "diagnostic_only_not_used_by_adapter_or_guard",
            "timing": "final candidate endpoint",
            "scopes": ["target", "evaluation", "non_evaluation"],
            "degree_groups": (
                "fixed on the full target graph, then intersected with each scope"
            ),
            "evaluation_scope": "official tgt_test_mask without labels",
        },
        "local_code_sha256": local_code_sha256,
        "row_protocol": row_protocol,
        "row_protocol_sha256": row_protocol_sha256,
    }
    payload = {
        **metadata,
        "status": "running",
        "elapsed_seconds": 0.0,
        "records": records,
    }
    _write_json_atomic(out_path, payload)
    original_cwd = Path.cwd()
    try:
        # The official configuration uses ./data and ./model.  Running from the
        # repository root is therefore part of the native protocol, not merely
        # a convenience for imports.
        os.chdir(TSA_ROOT)
        for data_name in data_configs:
            for seed in seeds:
                for method in methods:
                    row_started = time.perf_counter()
                    row = _run_one(
                        data_name,
                        method,
                        seed,
                        model_name,
                        prepared_checkpoints,
                        quarantine_root,
                        snapshot_root,
                        force_retrain,
                        row_protocol_sha256,
                    )
                    row["runtime_seconds"] = time.perf_counter() - row_started
                    _refresh_row_content_sha256(row)
                    target_key = (
                        row["data_config"],
                        row["target_setting"],
                        int(row["seed"]),
                    )
                    target_signature = {
                        "adapter_input_sha256": row["adapter_input_fingerprint"][
                            "sha256"
                        ],
                        "offline_evaluation_sha256": row[
                            "offline_evaluation_fingerprint"
                        ]["sha256"],
                        "proxy_scope_index_sha256": row["proxy_scope_index_sha256"],
                    }
                    expected_signature = expected_target_fingerprints.setdefault(
                        target_key, target_signature
                    )
                    if target_signature != expected_signature:
                        raise RuntimeError(
                            "Target data changed across methods for one audit condition; "
                            "refusing incomparable rows"
                        )
                    records.append(row)
                    payload["elapsed_seconds"] = time.perf_counter() - started
                    _write_json_atomic(out_path, payload)
                    print(
                        f"[official-tsa] {data_name} seed={seed} {method}: "
                        f"source={row['source_accuracy']:.4f} "
                        f"candidate={row['candidate_accuracy']} "
                        f"gain={row['source_relative_accuracy']} "
                        f"status={row['candidate_status']} "
                        f"accept={row['fixed_guard_accept']}"
                    )
    finally:
        os.chdir(original_cwd)
    repository_status_after = _git_status(TSA_ROOT)
    if repository_status_after["tracked_changes"]:
        payload["status"] = "failed"
        payload["repository_status_after"] = repository_status_after
        _write_json_atomic(out_path, payload)
        raise RuntimeError("Official TSA tracked files changed during the audit")
    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.perf_counter() - started
    payload["repository_status_after"] = repository_status_after
    payload["unique_source_checkpoints"] = sorted(
        {
            (row["source_checkpoint_path"], row["source_checkpoint_sha256"])
            for row in records
        }
    )
    _write_json_atomic(out_path, payload)
    return records


def run(
    out_path: Path,
    data_configs,
    methods,
    seeds,
    model_name: str,
    force_retrain: bool = True,
):
    """Persist an unambiguous lifecycle record around the official replay."""

    out_path = Path(out_path).resolve()
    data_configs = tuple(data_configs)
    methods = tuple(methods)
    seeds = tuple(seeds)
    started = time.perf_counter()
    initial = {
        "evidence_status": "official_repository_default_config_provenance_audit",
        "status": "running",
        "elapsed_seconds": 0.0,
        "data_configs": list(data_configs),
        "methods": list(methods),
        "model": model_name,
        "seeds": list(seeds),
        "fresh_source_checkpoint_training": bool(force_retrain),
        "records": [],
    }
    _write_json_atomic(out_path, initial)
    try:
        if not data_configs:
            raise ValueError("At least one data configuration is required")
        if not methods:
            raise ValueError("At least one adaptation method is required")
        if not seeds:
            raise ValueError("At least one seed is required")
        unknown = sorted(set(methods) - set(METHOD_OVERRIDES))
        if unknown:
            raise ValueError(f"Unsupported methods: {unknown}")
        with _exclusive_audit_lock() as lock_payload:
            return _run_locked(
                out_path,
                data_configs,
                methods,
                seeds,
                model_name,
                lock_payload,
                force_retrain=force_retrain,
            )
    except BaseException as error:
        try:
            failed = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            failed = initial
        if not isinstance(failed, dict):
            failed = initial
        failed.update(
            {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _write_json_atomic(out_path, failed)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--data-configs", default="CSBM1,CSBM2,CSBM3,CSBM4,CSBM5,CSBM6,CSBM7,CSBM8"
    )
    parser.add_argument("--methods", default="T3A,Matcha_T3A,TSA_T3A")
    parser.add_argument("--seeds", default="99,30,50")
    parser.add_argument("--model", choices=("GPRGNN", "GSN"), default="GPRGNN")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse existing source checkpoints instead of quarantining and retraining them.",
    )
    args = parser.parse_args()
    data_configs = tuple(
        item.strip() for item in args.data_configs.split(",") if item.strip()
    )
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHOD_OVERRIDES))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if args.model != "GPRGNN" and any(name.startswith("Matcha") for name in methods):
        raise ValueError(
            "The TSA repository's Matcha implementation requires the GPRGNN "
            "prop1 module; use --model GPRGNN or exclude Matcha methods."
        )
    run(
        Path(args.out),
        data_configs,
        methods,
        seeds,
        args.model,
        force_retrain=not args.reuse_checkpoints,
    )


if __name__ == "__main__":
    main()
