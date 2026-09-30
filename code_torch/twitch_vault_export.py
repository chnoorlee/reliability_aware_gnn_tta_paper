"""One-time label-vault export for the prospective Twitch experiment.

Run this module inside the data-owner/evaluator environment. It publishes two
explicit, non-nested roots: the candidate release contains only public graph
packs plus DE supervision, while the sealed root contains the evaluator's
private target outcomes and its matching DE supervision copy. The two roots
must be handed to processes with different OS-level visibility.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

try:
    from .natural_shift_io import (
        PUBLIC_MANIFEST_SCHEMA,
        PUBLIC_PACK_SCHEMA,
        SOURCE_TARGETS_SCHEMA,
        TARGET_DOMAINS,
        TWITCH_DOMAINS,
        TWITCH_FEATURE_DIM,
        ArtifactValidationError,
        array_sha256,
        atomic_write_json,
        sha256_file,
        write_deterministic_npz,
    )
except ImportError:
    from natural_shift_io import (
        PUBLIC_MANIFEST_SCHEMA,
        PUBLIC_PACK_SCHEMA,
        SOURCE_TARGETS_SCHEMA,
        TARGET_DOMAINS,
        TWITCH_DOMAINS,
        TWITCH_FEATURE_DIM,
        ArtifactValidationError,
        array_sha256,
        atomic_write_json,
        sha256_file,
        write_deterministic_npz,
    )


VAULT_PACK_SCHEMA = "twitch_target_vault.v1"
VAULT_MANIFEST_SCHEMA = "twitch_vault_manifest.v1"
EXPORT_RECEIPT_SCHEMA = "twitch_vault_export_receipt.v1"
_MATCHA_LOADER_RELATIVE = Path("external_official/Matcha/src/dataset/NCDataset.py")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_matcha_module() -> tuple[ModuleType, Path]:
    """Load the pinned Matcha loader by exact file, avoiding package shadowing."""

    loader_path = (_project_root() / _MATCHA_LOADER_RELATIVE).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        "_prospective_matcha_nc_dataset", loader_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Matcha dataset module: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_path = Path(module.__file__).resolve(strict=True)
    if loaded_path != loader_path:
        raise ArtifactValidationError("Matcha loader resolved to an unexpected file")
    return module, loader_path


def _load_matcha_domain(data_dir: Path, domain: str) -> Any:
    """Evaluator-only seam; tests monkeypatch this without downloading data."""

    module, _ = _load_matcha_module()
    return module.load_twitch_dataset(str(data_dir), domain)


def _extract_domain(
    dataset: Any, domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one official Matcha graph without changing its row semantics."""

    try:
        graph = dataset.graph
        x = dataset.graph["node_feat"].detach().cpu().numpy()
        edge_index = dataset.graph["edge_index"].detach().cpu().numpy()
        targets = dataset.label.detach().cpu().numpy()
    except (AttributeError, KeyError, TypeError) as exc:
        raise ArtifactValidationError(
            f"Matcha loader returned an unsupported object for {domain}"
        ) from exc
    if set(graph) != {"edge_index", "edge_feat", "node_feat", "num_nodes"}:
        raise ArtifactValidationError(f"unexpected Matcha graph fields for {domain}")
    x = np.asarray(x, dtype=np.float32)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != TWITCH_FEATURE_DIM or x.shape[0] < 2:
        raise ArtifactValidationError(
            f"invalid Twitch feature shape for {domain}: {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ArtifactValidationError(f"non-finite Twitch features for {domain}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ArtifactValidationError(f"invalid Twitch edge_index for {domain}")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]
    ):
        raise ArtifactValidationError(f"out-of-range Twitch edge for {domain}")
    if targets.shape != (x.shape[0],):
        raise ArtifactValidationError(f"Twitch target length mismatch for {domain}")
    if set(np.unique(targets).tolist()) != {0, 1}:
        raise ArtifactValidationError(f"Twitch targets are not binary for {domain}")
    declared_nodes = graph["num_nodes"]
    if type(declared_nodes) is not int or declared_nodes != x.shape[0]:
        raise ArtifactValidationError(f"Matcha num_nodes mismatch for {domain}")
    # The official loader reindexes all graph rows and edge endpoints to this
    # canonical 0..N-1 space; these are the only IDs exposed by that loader.
    node_ids = np.arange(x.shape[0], dtype=np.int64)
    return x, edge_index, targets, node_ids


def _pack_text(value: str) -> np.ndarray:
    return np.asarray(value, dtype=f"<U{max(len(value), 1)}")


def _cleanup_staging(staging: Path, parent: Path) -> None:
    if not staging.exists():
        return
    resolved_parent = parent.resolve(strict=True)
    resolved_staging = staging.resolve(strict=True)
    try:
        resolved_staging.relative_to(resolved_parent)
    except ValueError as exc:
        raise RuntimeError("refusing to clean staging outside export parent") from exc
    if not staging.name.startswith(".twitch-export-"):
        raise RuntimeError("refusing to clean an unexpected staging directory")
    shutil.rmtree(staging)


def export_twitch_vault(
    data_dir: str | Path,
    candidate_release_dir: str | Path,
    sealed_vault_dir: str | Path,
) -> dict[str, str]:
    """Atomically publish each of two non-nested candidate/evaluator roots."""

    raw_root = Path(data_dir)
    if not raw_root.exists() or not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    candidate_destination = Path(candidate_release_dir)
    sealed_destination = Path(sealed_vault_dir)
    for destination in (candidate_destination, sealed_destination):
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    candidate_destination = candidate_destination.resolve(strict=False)
    sealed_destination = sealed_destination.resolve(strict=False)
    if (
        candidate_destination == sealed_destination
        or candidate_destination in sealed_destination.parents
        or sealed_destination in candidate_destination.parents
    ):
        raise ArtifactValidationError(
            "candidate release and sealed vault roots must be non-nested"
        )
    candidate_staging = Path(
        tempfile.mkdtemp(
            prefix=".twitch-export-", dir=candidate_destination.parent
        )
    )
    sealed_staging = Path(
        tempfile.mkdtemp(prefix=".twitch-export-", dir=sealed_destination.parent)
    )
    try:
        candidate_root = candidate_staging
        sealed_root = sealed_staging
        (candidate_root / "public").mkdir(parents=True)
        (candidate_root / "source").mkdir()
        (sealed_root / "source").mkdir(parents=True)
        (sealed_root / "vault").mkdir()
        graph_entries: dict[str, dict[str, Any]] = {}
        vault_entries: dict[str, dict[str, Any]] = {}
        source_entry: dict[str, Any] | None = None

        for domain in TWITCH_DOMAINS:
            dataset = _load_matcha_domain(raw_root, domain)
            x, edge_index, targets, node_ids = _extract_domain(dataset, domain)
            public_path = candidate_root / "public" / f"{domain}.npz"
            write_deterministic_npz(
                public_path,
                {
                    "domain": _pack_text(domain),
                    "edge_index": edge_index,
                    "feature_dim": np.asarray(TWITCH_FEATURE_DIM, dtype=np.int64),
                    "node_ids": node_ids,
                    "schema_version": _pack_text(PUBLIC_PACK_SCHEMA),
                    "x": x,
                },
            )
            public_digest = sha256_file(public_path)
            node_digest = array_sha256(node_ids)
            graph_entries[domain] = {
                "node_ids_sha256": node_digest,
                "num_edges": int(edge_index.shape[1]),
                "num_nodes": int(x.shape[0]),
                "path": f"public/{domain}.npz",
                "sha256": public_digest,
            }

            if domain == "DE":
                source_arrays = {
                    "domain": _pack_text(domain),
                    "node_ids": node_ids,
                    "public_pack_sha256": _pack_text(public_digest),
                    "schema_version": _pack_text(SOURCE_TARGETS_SCHEMA),
                    "source_targets": targets,
                }
                source_path = candidate_root / "source" / "DE.npz"
                write_deterministic_npz(source_path, source_arrays)
                sealed_source_path = sealed_root / "source" / "DE.npz"
                write_deterministic_npz(sealed_source_path, source_arrays)
                source_digest = sha256_file(source_path)
                if sha256_file(sealed_source_path) != source_digest:
                    raise RuntimeError("DE supervision copies are not byte-identical")
                source_entry = {
                    "node_ids_sha256": node_digest,
                    "num_nodes": int(len(node_ids)),
                    "path": "source/DE.npz",
                    "public_pack_sha256": public_digest,
                    "sha256": source_digest,
                }
                continue

            vault_path = sealed_root / "vault" / f"{domain}.npz"
            write_deterministic_npz(
                vault_path,
                {
                    "domain": _pack_text(domain),
                    "node_ids": node_ids,
                    "public_pack_sha256": _pack_text(public_digest),
                    "schema_version": _pack_text(VAULT_PACK_SCHEMA),
                    "y": targets,
                },
            )
            vault_entries[domain] = {
                "node_ids_sha256": node_digest,
                "num_nodes": int(len(node_ids)),
                "path": f"vault/{domain}.npz",
                "public_pack_sha256": public_digest,
                "sha256": sha256_file(vault_path),
            }

        if source_entry is None:
            raise RuntimeError("DE source pack was not created")
        public_manifest = {
            "feature_dim": TWITCH_FEATURE_DIM,
            "graphs": graph_entries,
            "hash_algorithm": "sha256",
            "node_id_semantics": "Matcha canonical graph row index (0..N-1)",
            "schema_version": PUBLIC_MANIFEST_SCHEMA,
            "source_domain": "DE",
            "target_domains": list(TARGET_DOMAINS),
        }
        public_manifest_path = candidate_root / "public_manifest.json"
        atomic_write_json(public_manifest_path, public_manifest)
        public_manifest_digest = sha256_file(public_manifest_path)

        vault_manifest = {
            "hash_algorithm": "sha256",
            "public_manifest_sha256": public_manifest_digest,
            "schema_version": VAULT_MANIFEST_SCHEMA,
            "source_targets": source_entry,
            "targets": vault_entries,
        }
        vault_manifest_path = sealed_root / "vault_manifest.json"
        atomic_write_json(vault_manifest_path, vault_manifest)
        vault_manifest_digest = sha256_file(vault_manifest_path)

        # Bind provenance to the exact vendored source file.  Loading the module
        # here would needlessly couple the integrity receipt to the test seam
        # used above and could execute loader import side effects twice.
        loader_path = (_project_root() / _MATCHA_LOADER_RELATIVE).resolve(strict=True)
        receipt = {
            "export_policy": "target labels sealed; DE supervision separated",
            "hash_algorithm": "sha256",
            "isolation_policy": (
                "Run the candidate under an OS account/container/mount that can "
                "read only the candidate release root; do not expose this sealed root."
            ),
            "loader_path": _MATCHA_LOADER_RELATIVE.as_posix(),
            "loader_sha256": sha256_file(loader_path),
            "public_manifest_sha256": public_manifest_digest,
            "schema_version": EXPORT_RECEIPT_SCHEMA,
            "vault_manifest_sha256": vault_manifest_digest,
        }
        atomic_write_json(sealed_root / "export_receipt.json", receipt)
        # Publish the private root first. If the public publish fails, no
        # candidate can run against a release whose sealed counterpart is absent.
        os.replace(sealed_staging, sealed_destination)
        os.replace(candidate_staging, candidate_destination)
        return {
            "public_manifest_sha256": public_manifest_digest,
            "vault_manifest_sha256": vault_manifest_digest,
            "export_receipt_sha256": sha256_file(
                sealed_destination / "export_receipt.json"
            ),
        }
    except BaseException:
        _cleanup_staging(candidate_staging, candidate_destination.parent)
        _cleanup_staging(sealed_staging, sealed_destination.parent)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create separate candidate-release and sealed-vault roots"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--candidate-release-dir", required=True)
    parser.add_argument("--sealed-vault-dir", required=True)
    args = parser.parse_args()
    digests = export_twitch_vault(
        args.data_dir, args.candidate_release_dir, args.sealed_vault_dir
    )
    for name, digest in sorted(digests.items()):
        print(f"{name}={digest}")


if __name__ == "__main__":
    main()
