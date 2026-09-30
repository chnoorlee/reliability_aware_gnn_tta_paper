"""Adversarial tests for the prospective Twitch label-vault workflow."""

from __future__ import annotations

import ast
import copy
import inspect
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import natural_shift_candidate as candidate
import natural_shift_evaluate as evaluator
import twitch_vault_export as exporter
from natural_shift_io import (
    TARGET_DOMAINS,
    TWITCH_DOMAINS,
    TWITCH_FEATURE_DIM,
    ArtifactValidationError,
    array_sha256,
    atomic_write_json,
    load_json_strict,
    load_protocol,
    load_public_graph_pack,
    load_public_manifest,
    read_npz_strict,
    sha256_file,
    with_protocol_hash,
    write_deterministic_npz,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PROTOCOL = PROJECT_ROOT / "protocols" / "twitch_de_multidomain_v1.json"
VAULT_KEYS = ("domain", "node_ids", "public_pack_sha256", "schema_version", "y")


def _copy_fresh_vault(source: Path, destination: Path) -> Path:
    assert not (source / "consumed_once.json").exists()
    shutil.copytree(source, destination)
    return destination


def _toy_dataset(domain: str, *, permute_target_labels: bool) -> Any:
    domain_index = TWITCH_DOMAINS.index(domain)
    num_nodes = 8
    x = np.zeros((num_nodes, TWITCH_FEATURE_DIM), dtype=np.float32)
    rows = np.arange(num_nodes)
    x[rows, rows + 3 * domain_index] = 1.0
    x[:, 200 + domain_index] = np.linspace(-0.5, 0.5, num_nodes, dtype=np.float32)
    forward = np.vstack((rows, np.roll(rows, -1)))
    backward = np.vstack((np.roll(rows, -1), rows))
    edge_index = np.hstack((forward, backward)).astype(np.int64)
    targets = (rows % 2).astype(np.int64)
    if domain != "DE" and permute_target_labels:
        targets = targets[::-1].copy()
    graph = {
        "edge_index": torch.from_numpy(edge_index),
        "edge_feat": None,
        "node_feat": torch.from_numpy(x),
        "num_nodes": num_nodes,
    }
    return SimpleNamespace(graph=graph, label=torch.from_numpy(targets))


def _write_toy_protocol(path: Path) -> Path:
    value = copy.deepcopy(load_json_strict(PRODUCTION_PROTOCOL))
    value["protocol_id"] = "twitch-de-multidomain-toy-v1"
    value["seeds"] = [0]
    value["models"].update(
        {
            "backbones": ["gcn"],
            "hidden_dim": 4,
            "patience": 2,
            "train_epochs": 2,
        }
    )
    value["adaptation"].update(
        {
            "adapter_provenance": {
                "tent_entropy": value["adaptation"]["adapter_provenance"][
                    "tent_entropy"
                ]
            },
            "adapters": ["tent_entropy"],
            "steps": 1,
        }
    )
    atomic_write_json(path, with_protocol_hash(value))
    return path


@pytest.fixture(scope="module")
def toy_pipeline(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("natural-shift")
    raw = root / "raw"
    raw.mkdir()
    protocol = _write_toy_protocol(root / "toy_protocol.json")
    release_a = root / "candidate-release-a"
    release_b = root / "candidate-release-b"
    vault_a = root / "sealed-vault-a"
    vault_b = root / "sealed-vault-b"

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            exporter,
            "_load_matcha_domain",
            lambda _raw, domain: _toy_dataset(domain, permute_target_labels=False),
        )
        exporter.export_twitch_vault(raw, release_a, vault_a)
        monkeypatch.setattr(
            exporter,
            "_load_matcha_domain",
            lambda _raw, domain: _toy_dataset(domain, permute_target_labels=True),
        )
        exporter.export_twitch_vault(raw, release_b, vault_b)
    finally:
        monkeypatch.undo()

    candidate_a = root / "candidate-a"
    candidate_b = root / "candidate-b"
    result_a = candidate.run_candidate_pipeline(
        protocol_path=protocol,
        public_manifest_path=release_a / "public_manifest.json",
        source_targets_path=release_a / "source" / "DE.npz",
        output_dir=candidate_a,
    )
    result_b = candidate.run_candidate_pipeline(
        protocol_path=protocol,
        public_manifest_path=release_b / "public_manifest.json",
        source_targets_path=release_b / "source" / "DE.npz",
        output_dir=candidate_b,
    )
    return SimpleNamespace(
        root=root,
        protocol=protocol,
        export_a=release_a,
        export_b=release_b,
        release_a=release_a,
        release_b=release_b,
        vault_a=vault_a,
        vault_b=vault_b,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        result_a=result_a,
        result_b=result_b,
    )


def test_production_protocol_freezes_the_prospective_design() -> None:
    protocol = load_protocol(PRODUCTION_PROTOCOL)
    assert protocol["adaptation"]["steps"] == 100
    assert protocol["thresholds"] == {"delta": 0.05, "harm": 0.01, "phi": 0.2}
    assert protocol["data"]["source_domain"] == "DE"
    assert protocol["data"]["target_domains"] == list(TARGET_DOMAINS)
    assert protocol["execution"]["no_target_derived_tuning"] is True
    assert (
        "not the official Tent"
        in protocol["adaptation"]["adapter_provenance"]["tent_entropy"]
    )


def test_candidate_has_no_target_vault_interface_or_import(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    parameters = inspect.signature(candidate.run_candidate_pipeline).parameters
    assert "vault_manifest_path" not in parameters
    assert "target_labels" not in parameters

    tree = ast.parse(Path(candidate.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any("twitch_vault_export" in name for name in imported_modules)
    assert not any("natural_shift_evaluate" in name for name in imported_modules)
    assert not (toy_pipeline.release_a / "vault").exists()
    assert not (toy_pipeline.release_a / "vault_manifest.json").exists()

    with pytest.raises(ArtifactValidationError, match="inside the public export root"):
        candidate.run_candidate_pipeline(
            protocol_path=toy_pipeline.protocol,
            public_manifest_path=toy_pipeline.export_a / "public_manifest.json",
            source_targets_path=toy_pipeline.vault_a / "vault" / "ENGB.npz",
            output_dir=tmp_path / "must-not-exist",
        )


def test_public_graph_view_is_constructible_without_targets(
    toy_pipeline: SimpleNamespace,
) -> None:
    manifest_path = toy_pipeline.export_a / "public_manifest.json"
    manifest = load_public_manifest(manifest_path)
    entry = manifest["graphs"]["ENGB"]
    graph = load_public_graph_pack(
        toy_pipeline.export_a / entry["path"],
        expected_domain="ENGB",
        expected_sha256=entry["sha256"],
    )
    _, unlabeled_type, _, _ = candidate._load_local_stack()
    view = unlabeled_type(
        x_np=graph.x,
        adj=candidate._adjacency(graph),
        x=torch.from_numpy(np.array(graph.x, copy=True)),
        edge_index=torch.from_numpy(np.array(graph.edge_index, copy=True)),
    )
    assert not hasattr(view, "y")
    assert not hasattr(view, "label")
    assert set(graph.__dataclass_fields__) == {
        "domain",
        "edge_index",
        "node_ids",
        "path",
        "sha256",
        "x",
    }


def test_public_pack_rejects_target_or_mask_fields(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    original = toy_pipeline.export_a / "public" / "ENGB.npz"
    arrays = read_npz_strict(
        original,
        expected_keys=(
            "domain",
            "edge_index",
            "feature_dim",
            "node_ids",
            "schema_version",
            "x",
        ),
    )
    arrays["y"] = np.zeros(len(arrays["node_ids"]), dtype=np.int64)
    malicious = tmp_path / "public-with-y.npz"
    write_deterministic_npz(malicious, arrays)
    with pytest.raises(ArtifactValidationError, match="NPZ schema mismatch"):
        load_public_graph_pack(malicious, expected_domain="ENGB")

    public_manifest = load_json_strict(toy_pipeline.export_a / "public_manifest.json")
    public_manifest["label_mask"] = "forbidden"
    malicious_manifest = tmp_path / "public-with-mask.json"
    atomic_write_json(malicious_manifest, public_manifest)
    with pytest.raises(ArtifactValidationError, match="forbidden public field"):
        load_public_manifest(malicious_manifest)


def test_target_label_permutation_cannot_change_candidate_artifacts(
    toy_pipeline: SimpleNamespace,
) -> None:
    assert sha256_file(toy_pipeline.export_a / "public_manifest.json") == sha256_file(
        toy_pipeline.export_b / "public_manifest.json"
    )
    assert sha256_file(toy_pipeline.export_a / "source" / "DE.npz") == sha256_file(
        toy_pipeline.export_b / "source" / "DE.npz"
    )
    assert sha256_file(toy_pipeline.vault_a / "vault_manifest.json") != sha256_file(
        toy_pipeline.vault_b / "vault_manifest.json"
    )

    manifest_a = load_json_strict(toy_pipeline.candidate_a / "candidate_manifest.json")
    manifest_b = load_json_strict(toy_pipeline.candidate_b / "candidate_manifest.json")
    assert manifest_a["source_targets_sha256"] == manifest_b["source_targets_sha256"]
    comparable_a = {
        run["run_id"]: (
            run["artifact_sha256"],
            run["trajectory_sha256"],
            run["proxy_trace"],
            run["decisions"],
            run["status"],
            run["failure"],
        )
        for run in manifest_a["runs"]
    }
    comparable_b = {
        run["run_id"]: (
            run["artifact_sha256"],
            run["trajectory_sha256"],
            run["proxy_trace"],
            run["decisions"],
            run["status"],
            run["failure"],
        )
        for run in manifest_b["runs"]
    }
    assert comparable_a == comparable_b


def test_public_artifact_tampering_fails_before_candidate_execution(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    damaged_export = tmp_path / "damaged-export"
    shutil.copytree(toy_pipeline.export_a, damaged_export)
    damaged_pack = damaged_export / "public" / "ENGB.npz"
    content = bytearray(damaged_pack.read_bytes())
    content[-1] ^= 0x01
    damaged_pack.write_bytes(content)

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        candidate.run_candidate_pipeline(
            protocol_path=toy_pipeline.protocol,
            public_manifest_path=damaged_export / "public_manifest.json",
            source_targets_path=damaged_export / "source" / "DE.npz",
            output_dir=tmp_path / "damaged-candidate",
        )


def test_evaluator_rejects_misaligned_vault_node_ids(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    damaged_export = tmp_path / "misaligned-export"
    _copy_fresh_vault(toy_pipeline.vault_a, damaged_export)
    vault_path = damaged_export / "vault" / "ENGB.npz"
    arrays = read_npz_strict(vault_path, expected_keys=VAULT_KEYS)
    arrays["node_ids"] = arrays["node_ids"][::-1].copy()
    write_deterministic_npz(vault_path, arrays, overwrite=True)

    vault_manifest_path = damaged_export / "vault_manifest.json"
    vault_manifest = load_json_strict(vault_manifest_path)
    vault_manifest["targets"]["ENGB"]["node_ids_sha256"] = array_sha256(
        arrays["node_ids"]
    )
    vault_manifest["targets"]["ENGB"]["sha256"] = sha256_file(vault_path)
    atomic_write_json(vault_manifest_path, vault_manifest, overwrite=True)

    with pytest.raises(ArtifactValidationError, match="misaligned"):
        evaluator.evaluate_candidate_once(
            protocol_path=toy_pipeline.protocol,
            public_manifest_path=toy_pipeline.export_a / "public_manifest.json",
            vault_manifest_path=vault_manifest_path,
            vault_manifest_sha256=sha256_file(vault_manifest_path),
            candidate_dir=toy_pipeline.candidate_a,
            candidate_manifest_sha256=toy_pipeline.result_a[
                "candidate_manifest_sha256"
            ],
            output_dir=tmp_path / "misaligned-evaluation",
        )


def test_evaluator_rejects_tampered_finalized_candidate_manifest(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    vault = _copy_fresh_vault(toy_pipeline.vault_a, tmp_path / "tampered-vault")
    damaged_candidate = tmp_path / "damaged-candidate"
    shutil.copytree(toy_pipeline.candidate_a, damaged_candidate)
    manifest_path = damaged_candidate / "candidate_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" \n")

    with pytest.raises(ArtifactValidationError, match="manifest SHA-256 mismatch"):
        evaluator.evaluate_candidate_once(
            protocol_path=toy_pipeline.protocol,
            public_manifest_path=toy_pipeline.export_a / "public_manifest.json",
            vault_manifest_path=vault / "vault_manifest.json",
            vault_manifest_sha256=sha256_file(vault / "vault_manifest.json"),
            candidate_dir=damaged_candidate,
            candidate_manifest_sha256=toy_pipeline.result_a[
                "candidate_manifest_sha256"
            ],
            output_dir=tmp_path / "tampered-evaluation",
        )
    assert not (vault / "consumed_once.json").exists()


def test_toy_pipeline_evaluates_once_and_refuses_overwrite(
    toy_pipeline: SimpleNamespace, tmp_path: Path
) -> None:
    vault = _copy_fresh_vault(toy_pipeline.vault_a, tmp_path / "sealed-vault")
    output = tmp_path / "evaluation-a"
    result = evaluator.evaluate_candidate_once(
        protocol_path=toy_pipeline.protocol,
        public_manifest_path=toy_pipeline.export_a / "public_manifest.json",
        vault_manifest_path=vault / "vault_manifest.json",
        vault_manifest_sha256=sha256_file(vault / "vault_manifest.json"),
        candidate_dir=toy_pipeline.candidate_a,
        candidate_manifest_sha256=toy_pipeline.result_a["candidate_manifest_sha256"],
        output_dir=output,
    )
    metrics = load_json_strict(output / "metrics.json")
    receipt = load_json_strict(output / "consumed_manifest_receipt.json")
    assert len(metrics["rows"]) == len(TARGET_DOMAINS) * 7
    assert set(metrics["aggregate"]) == {
        "always",
        "delta_only",
        "fixed_combined",
        "online_rollback",
        "phi_only",
        "reject_all",
        "source_risk",
    }
    assert sha256_file(output / "metrics.json") == result["evaluation_sha256"]
    assert (
        receipt["candidate_manifest_sha256"]
        == toy_pipeline.result_a["candidate_manifest_sha256"]
    )
    assert (vault / "consumed_once.json").exists()
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluator.evaluate_candidate_once(
            protocol_path=toy_pipeline.protocol,
            public_manifest_path=toy_pipeline.export_a / "public_manifest.json",
            vault_manifest_path=vault / "vault_manifest.json",
            vault_manifest_sha256=sha256_file(vault / "vault_manifest.json"),
            candidate_dir=toy_pipeline.candidate_a,
            candidate_manifest_sha256=toy_pipeline.result_a[
                "candidate_manifest_sha256"
            ],
            output_dir=tmp_path / "evaluation-b",
        )


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":2}\n', encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="duplicate JSON key"):
        load_json_strict(duplicate)
