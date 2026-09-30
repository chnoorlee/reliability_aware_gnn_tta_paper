"""Strict, deterministic I/O for the prospective Twitch shift audit.

The public graph format is intentionally incapable of carrying target labels.
Source supervision uses a distinct, DE-only pack with a different schema.  All
untrusted JSON and NPZ inputs are checked before an experiment module receives
their contents.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

PUBLIC_PACK_SCHEMA = "twitch_public_graph.v1"
SOURCE_TARGETS_SCHEMA = "twitch_de_source_targets.v1"
PUBLIC_MANIFEST_SCHEMA = "twitch_public_manifest.v1"
PROTOCOL_SCHEMA = "twitch_natural_shift_protocol.v1"
TWITCH_DOMAINS = ("DE", "ENGB", "ES", "FR", "PTBR", "RU", "TW")
TARGET_DOMAINS = TWITCH_DOMAINS[1:]
TWITCH_FEATURE_DIM = 3170
_SHA256_HEX_LENGTH = 64
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
_FORBIDDEN_PUBLIC_TOKENS = frozenset(
    {
        "y",
        "label",
        "labels",
        "mask",
        "masks",
        "train_mask",
        "val_mask",
        "test_mask",
        "target_mask",
        "vault",
    }
)


class ArtifactValidationError(ValueError):
    """Raised when an artifact violates a frozen schema or integrity check."""


@dataclass(frozen=True, slots=True)
class PublicGraphPack:
    domain: str
    x: np.ndarray
    edge_index: np.ndarray
    node_ids: np.ndarray
    path: Path
    sha256: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True, slots=True)
class SourceTargetsPack:
    domain: str
    node_ids: np.ndarray
    targets: np.ndarray
    public_pack_sha256: str
    path: Path
    sha256: str


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular, non-link file."""

    return sha256_bytes(read_regular_file_bytes(path))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash an array with its NumPy header, shape, dtype, and bytes bound."""

    return sha256_bytes(_npy_bytes(np.asarray(array)))


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_link(path: Path) -> None:
    if path.is_symlink() or _is_reparse_point(path):
        raise ArtifactValidationError(f"symlink/reparse point is forbidden: {path}")


def assert_regular_file(path: str | Path) -> Path:
    checked = Path(path)
    if not checked.exists():
        raise FileNotFoundError(checked)
    _reject_link(checked)
    if not checked.is_file():
        raise ArtifactValidationError(f"expected regular file: {checked}")
    return checked


def read_regular_file_bytes(
    path: str | Path, *, max_bytes: int = _MAX_INPUT_BYTES
) -> bytes:
    """Read one immutable file-descriptor snapshot with a strict size bound."""

    checked = assert_regular_file(path)
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(checked, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactValidationError(f"expected regular file: {checked}")
        if metadata.st_size > max_bytes:
            raise ArtifactValidationError(f"input file exceeds size limit: {checked}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                raise ArtifactValidationError(
                    f"input file exceeds size limit: {checked}"
                )
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def validate_relative_path(value: Any) -> str:
    """Require one normalized, traversal-free POSIX relative path."""

    if type(value) is not str or not value:
        raise ArtifactValidationError("manifest path must be a non-empty string")
    if "\\" in value:
        raise ArtifactValidationError(f"manifest path must use '/': {value!r}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ArtifactValidationError(f"unsafe relative manifest path: {value!r}")
    if candidate.as_posix() != value:
        raise ArtifactValidationError(f"non-canonical manifest path: {value!r}")
    return value


def resolve_manifest_path(root: str | Path, relative: Any) -> Path:
    """Resolve a manifest member while rejecting containment and link escapes."""

    relative_value = validate_relative_path(relative)
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise ArtifactValidationError(f"manifest root is not a directory: {root_path}")
    _reject_link(root_path)
    root_resolved = root_path.resolve(strict=True)
    current = root_resolved
    for part in PurePosixPath(relative_value).parts:
        current = current / part
        if current.exists():
            _reject_link(current)
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"manifest member escapes root: {relative_value!r}"
        ) from exc
    return assert_regular_file(resolved)


def validate_candidate_release_root(path: str | Path) -> Path:
    """Require the exact label-free directory tree exposed to a candidate.

    This is a filesystem boundary, not merely a manifest convention: the
    candidate-facing root may contain only the seven public graphs, the DE
    supervision pack, and the public manifest. Extra entries and link-like
    objects fail closed before any model code runs.
    """

    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise ArtifactValidationError(f"candidate release root is invalid: {root}")
    _reject_link(root)
    resolved_root = root.resolve(strict=True)
    expected_top_level = {"public", "source", "public_manifest.json"}
    actual_top_level = {entry.name for entry in resolved_root.iterdir()}
    if actual_top_level != expected_top_level:
        raise ArtifactValidationError(
            "candidate release root must contain exactly public/, source/, "
            "and public_manifest.json"
        )

    expected_by_directory = {
        "public": {f"{domain}.npz" for domain in TWITCH_DOMAINS},
        "source": {"DE.npz"},
    }
    for directory_name, expected_names in expected_by_directory.items():
        directory = resolved_root / directory_name
        _reject_link(directory)
        if not directory.is_dir():
            raise ArtifactValidationError(
                f"candidate release {directory_name} entry is not a directory"
            )
        actual_names = {entry.name for entry in directory.iterdir()}
        if actual_names != expected_names:
            raise ArtifactValidationError(
                f"candidate release {directory_name}/ contents are not exact"
            )
        for name in expected_names:
            member = assert_regular_file(directory / name).resolve(strict=True)
            try:
                member.relative_to(resolved_root)
            except ValueError as exc:
                raise ArtifactValidationError(
                    f"candidate release member escapes root: {directory_name}/{name}"
                ) from exc

    manifest = assert_regular_file(resolved_root / "public_manifest.json").resolve(
        strict=True
    )
    try:
        manifest.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactValidationError("candidate release manifest escapes root") from exc
    return resolved_root


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON number is forbidden: {value}")


def _ensure_json_native(value: Any, path: str = "$") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactValidationError(f"non-finite number at {path}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _ensure_json_native(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ArtifactValidationError(f"non-string key at {path}")
            _ensure_json_native(item, f"{path}.{key}")
        return
    raise ArtifactValidationError(
        f"non-native JSON value at {path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    _ensure_json_native(value)
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_json_bytes_strict(data: bytes, context: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except UnicodeDecodeError as exc:
        raise ArtifactValidationError(f"JSON is not UTF-8: {context}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"invalid JSON: {context}: {exc}") from exc
    if type(value) is not dict:
        raise ArtifactValidationError(f"top-level JSON must be an object: {context}")
    _ensure_json_native(value)
    return value


def load_json_strict_with_sha256(path: str | Path) -> tuple[dict[str, Any], str]:
    checked = assert_regular_file(path)
    data = read_regular_file_bytes(checked)
    return _load_json_bytes_strict(data, str(checked)), sha256_bytes(data)


def load_json_strict(path: str | Path) -> dict[str, Any]:
    value, _ = load_json_strict_with_sha256(path)
    return value


def atomic_write_bytes(
    path: str | Path, data: bytes, *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    _reject_link(destination.parent)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any] | list[Any],
    *,
    overwrite: bool = False,
) -> Path:
    return atomic_write_bytes(path, canonical_json_bytes(value), overwrite=overwrite)


def _npy_bytes(array: np.ndarray) -> bytes:
    normalized = np.asarray(array)
    if normalized.dtype.hasobject:
        raise ArtifactValidationError("object arrays are forbidden")
    buffer = io.BytesIO()
    np.save(buffer, normalized, allow_pickle=False)
    return buffer.getvalue()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise ArtifactValidationError("NPZ must contain at least one array")
    if any(type(name) is not str or not name for name in arrays):
        raise ArtifactValidationError("NPZ array names must be non-empty strings")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            if "/" in name or "\\" in name or name in {".", ".."}:
                raise ArtifactValidationError(f"unsafe NPZ array name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    return output.getvalue()


def write_deterministic_npz(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> Path:
    return atomic_write_bytes(
        path, deterministic_npz_bytes(arrays), overwrite=overwrite
    )


def _read_npz_bytes_strict(
    data: bytes, *, expected_keys: Sequence[str], context: str
) -> dict[str, np.ndarray]:
    expected = set(expected_keys)
    if len(expected) != len(tuple(expected_keys)):
        raise ArtifactValidationError("expected NPZ schema contains duplicate keys")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                raise ArtifactValidationError("duplicate NPZ member name")
            if archive.testzip() is not None:
                raise ArtifactValidationError("NPZ CRC check failed")
            for item in members:
                member = PurePosixPath(item.filename)
                if (
                    member.is_absolute()
                    or len(member.parts) != 1
                    or not item.filename.endswith(".npy")
                    or stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF)
                ):
                    raise ArtifactValidationError(
                        f"unsafe NPZ member: {item.filename!r}"
                    )
            actual = {name.removesuffix(".npy") for name in names}
            if actual != expected:
                raise ArtifactValidationError(
                    f"NPZ schema mismatch: expected {sorted(expected)}, got {sorted(actual)}"
                )
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in expected}
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        if isinstance(exc, ArtifactValidationError):
            raise
        raise ArtifactValidationError(f"invalid NPZ file: {context}: {exc}") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ArtifactValidationError(f"object array forbidden: {name}")
    return arrays


def read_npz_strict_with_sha256(
    path: str | Path, *, expected_keys: Sequence[str]
) -> tuple[dict[str, np.ndarray], str]:
    checked = assert_regular_file(path)
    data = read_regular_file_bytes(checked)
    arrays = _read_npz_bytes_strict(
        data, expected_keys=expected_keys, context=str(checked)
    )
    return arrays, sha256_bytes(data)


def read_npz_strict(
    path: str | Path, *, expected_keys: Sequence[str]
) -> dict[str, np.ndarray]:
    arrays, _ = read_npz_strict_with_sha256(path, expected_keys=expected_keys)
    return arrays


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ArtifactValidationError(
            f"{context} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _expect_str(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise ArtifactValidationError(f"{context} must be a non-empty string")
    return value


def _expect_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ArtifactValidationError(f"{context} must be a boolean")
    return value


def _expect_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ArtifactValidationError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ArtifactValidationError(f"{context} must be >= {minimum}")
    return value


def _expect_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ArtifactValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactValidationError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ArtifactValidationError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ArtifactValidationError(f"{context} must be <= {maximum}")
    return result


def _expect_sha256(value: Any, context: str) -> str:
    digest = _expect_str(value, context)
    if len(digest) != _SHA256_HEX_LENGTH or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ArtifactValidationError(f"{context} is not lowercase SHA-256")
    return digest


def _scalar_text(array: np.ndarray, context: str) -> str:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ArtifactValidationError(f"{context} must be a scalar text array")
    value = array.item()
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return _expect_str(value, context)


def _validate_node_ids(
    node_ids: np.ndarray, num_nodes: int, context: str
) -> np.ndarray:
    if node_ids.dtype != np.dtype("int64") or node_ids.shape != (num_nodes,):
        raise ArtifactValidationError(
            f"{context} must be int64 with shape ({num_nodes},)"
        )
    if len(np.unique(node_ids)) != num_nodes:
        raise ArtifactValidationError(f"{context} must be unique")
    return node_ids


def _public_key_is_forbidden(key: str) -> bool:
    tokens = key.lower().replace("-", "_").split("_")
    return any(token in _FORBIDDEN_PUBLIC_TOKENS for token in tokens)


def reject_forbidden_public_fields(value: Any, context: str = "$") -> None:
    """Reject label-, split-, or vault-like fields at any manifest depth."""

    if type(value) is dict:
        for key, item in value.items():
            if _public_key_is_forbidden(key):
                raise ArtifactValidationError(
                    f"forbidden public field {key!r} at {context}"
                )
            reject_forbidden_public_fields(item, f"{context}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            reject_forbidden_public_fields(item, f"{context}[{index}]")


def load_public_graph_pack(
    path: str | Path,
    *,
    expected_domain: str,
    expected_sha256: str | None = None,
    expected_feature_dim: int = TWITCH_FEATURE_DIM,
) -> PublicGraphPack:
    checked = assert_regular_file(path)
    arrays, digest = read_npz_strict_with_sha256(
        checked,
        expected_keys=(
            "domain",
            "edge_index",
            "feature_dim",
            "node_ids",
            "schema_version",
            "x",
        ),
    )
    if expected_sha256 is not None and digest != _expect_sha256(
        expected_sha256, "expected public pack digest"
    ):
        raise ArtifactValidationError("public graph pack SHA-256 mismatch")
    reject_forbidden_public_fields({name: None for name in arrays})
    if _scalar_text(arrays["schema_version"], "public schema") != PUBLIC_PACK_SCHEMA:
        raise ArtifactValidationError("unsupported public graph schema")
    domain = _scalar_text(arrays["domain"], "public domain")
    if domain != expected_domain:
        raise ArtifactValidationError(
            f"public graph domain mismatch: expected {expected_domain}, got {domain}"
        )
    feature_dim_array = arrays["feature_dim"]
    if feature_dim_array.shape != () or feature_dim_array.dtype != np.dtype("int64"):
        raise ArtifactValidationError("feature_dim must be an int64 scalar")
    if int(feature_dim_array.item()) != expected_feature_dim:
        raise ArtifactValidationError("public graph feature_dim mismatch")
    x = arrays["x"]
    if x.dtype != np.dtype("float32") or x.ndim != 2:
        raise ArtifactValidationError("x must be a rank-2 float32 array")
    if x.shape[1] != expected_feature_dim or x.shape[0] < 2:
        raise ArtifactValidationError("x shape violates the Twitch graph contract")
    if not np.isfinite(x).all():
        raise ArtifactValidationError("x contains non-finite values")
    node_ids = _validate_node_ids(arrays["node_ids"], x.shape[0], "node_ids")
    edge_index = arrays["edge_index"]
    if (
        edge_index.dtype != np.dtype("int64")
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ArtifactValidationError("edge_index must be int64 with shape (2, E)")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]
    ):
        raise ArtifactValidationError("edge_index contains an out-of-range node")
    for array in (x, node_ids, edge_index):
        array.setflags(write=False)
    return PublicGraphPack(domain, x, edge_index, node_ids, checked, digest)


def load_source_targets_pack(
    path: str | Path,
    *,
    expected_public_sha256: str,
    expected_node_ids: np.ndarray,
) -> SourceTargetsPack:
    """Read the explicitly source-only DE supervision pack.

    A path component named ``vault`` is rejected so this function cannot be
    repurposed to cross the sealed target boundary.
    """

    checked = assert_regular_file(path)
    if any(part.lower() == "vault" for part in checked.parts):
        raise ArtifactValidationError("source supervision cannot be read from vault")
    arrays, digest = read_npz_strict_with_sha256(
        checked,
        expected_keys=(
            "domain",
            "node_ids",
            "public_pack_sha256",
            "schema_version",
            "source_targets",
        ),
    )
    if _scalar_text(arrays["schema_version"], "source schema") != SOURCE_TARGETS_SCHEMA:
        raise ArtifactValidationError("unsupported source-target schema")
    domain = _scalar_text(arrays["domain"], "source domain")
    if domain != "DE":
        raise ArtifactValidationError("only DE may supply source supervision")
    public_digest = _scalar_text(
        arrays["public_pack_sha256"], "source public pack digest"
    )
    if public_digest != _expect_sha256(
        expected_public_sha256, "expected source public pack digest"
    ):
        raise ArtifactValidationError("source targets/public graph SHA-256 mismatch")
    node_ids = _validate_node_ids(
        arrays["node_ids"], len(expected_node_ids), "source node_ids"
    )
    if not np.array_equal(node_ids, expected_node_ids):
        raise ArtifactValidationError(
            "source target node IDs do not align with DE graph"
        )
    targets = arrays["source_targets"]
    if targets.dtype != np.dtype("int64") or targets.shape != (len(node_ids),):
        raise ArtifactValidationError(
            "source_targets must be int64 with one value per node"
        )
    if (
        not set(np.unique(targets).tolist()).issubset({0, 1})
        or len(np.unique(targets)) != 2
    ):
        raise ArtifactValidationError("source_targets must contain both binary classes")
    node_ids.setflags(write=False)
    targets.setflags(write=False)
    return SourceTargetsPack(
        domain,
        node_ids,
        targets,
        public_digest,
        checked,
        digest,
    )


def _validate_public_manifest(value: dict[str, Any]) -> dict[str, Any]:
    reject_forbidden_public_fields(value)
    _expect_exact_keys(
        value,
        {
            "feature_dim",
            "graphs",
            "hash_algorithm",
            "node_id_semantics",
            "schema_version",
            "source_domain",
            "target_domains",
        },
        "public manifest",
    )
    if value["schema_version"] != PUBLIC_MANIFEST_SCHEMA:
        raise ArtifactValidationError("unsupported public manifest schema")
    if value["hash_algorithm"] != "sha256":
        raise ArtifactValidationError("public manifest must use SHA-256")
    if _expect_int(value["feature_dim"], "feature_dim") != TWITCH_FEATURE_DIM:
        raise ArtifactValidationError("Twitch public feature_dim must be 3170")
    if value["source_domain"] != "DE":
        raise ArtifactValidationError("Twitch source domain must be DE")
    if value["target_domains"] != list(TARGET_DOMAINS):
        raise ArtifactValidationError("Twitch target-domain order is not frozen")
    _expect_str(value["node_id_semantics"], "node_id_semantics")
    graphs = value["graphs"]
    if type(graphs) is not dict or list(graphs) != list(TWITCH_DOMAINS):
        raise ArtifactValidationError(
            "public graphs must enumerate the frozen domain order"
        )
    for domain, entry in graphs.items():
        if type(entry) is not dict:
            raise ArtifactValidationError(f"graph entry {domain} must be an object")
        _expect_exact_keys(
            entry,
            {"node_ids_sha256", "num_edges", "num_nodes", "path", "sha256"},
            f"graph entry {domain}",
        )
        if validate_relative_path(entry["path"]) != f"public/{domain}.npz":
            raise ArtifactValidationError(f"unexpected public path for {domain}")
        _expect_sha256(entry["sha256"], f"graphs.{domain}.sha256")
        _expect_sha256(entry["node_ids_sha256"], f"graphs.{domain}.node_ids_sha256")
        _expect_int(entry["num_nodes"], f"graphs.{domain}.num_nodes", minimum=2)
        _expect_int(entry["num_edges"], f"graphs.{domain}.num_edges", minimum=0)
    return value


def load_public_manifest_with_sha256(
    path: str | Path,
) -> tuple[dict[str, Any], str]:
    value, digest = load_json_strict_with_sha256(path)
    return _validate_public_manifest(value), digest


def load_public_manifest(path: str | Path) -> dict[str, Any]:
    value, _ = load_public_manifest_with_sha256(path)
    return value


def load_public_graph_from_manifest(
    manifest_path: str | Path, manifest: Mapping[str, Any], domain: str
) -> PublicGraphPack:
    if domain not in TWITCH_DOMAINS:
        raise ArtifactValidationError(f"unsupported Twitch domain: {domain}")
    entry = manifest["graphs"][domain]
    path = resolve_manifest_path(Path(manifest_path).parent, entry["path"])
    graph = load_public_graph_pack(
        path,
        expected_domain=domain,
        expected_sha256=entry["sha256"],
        expected_feature_dim=manifest["feature_dim"],
    )
    if graph.num_nodes != entry["num_nodes"]:
        raise ArtifactValidationError(f"public node count mismatch for {domain}")
    if graph.edge_index.shape[1] != entry["num_edges"]:
        raise ArtifactValidationError(f"public edge count mismatch for {domain}")
    if array_sha256(graph.node_ids) != entry["node_ids_sha256"]:
        raise ArtifactValidationError(f"public node-ID digest mismatch for {domain}")
    return graph


def protocol_payload_sha256(protocol: Mapping[str, Any]) -> str:
    payload = dict(protocol)
    payload.pop("protocol_payload_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def with_protocol_hash(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(protocol)
    result["protocol_payload_sha256"] = protocol_payload_sha256(result)
    return result


def _validate_protocol(value: dict[str, Any]) -> dict[str, Any]:
    _expect_exact_keys(
        value,
        {
            "adaptation",
            "created_date",
            "data",
            "execution",
            "hash_algorithm",
            "metrics",
            "models",
            "protocol_id",
            "protocol_payload_sha256",
            "schema_version",
            "seeds",
            "source_split",
            "status",
            "strategies",
            "thresholds",
        },
        "protocol",
    )
    if value["schema_version"] != PROTOCOL_SCHEMA:
        raise ArtifactValidationError("unsupported natural-shift protocol schema")
    if value["hash_algorithm"] != "sha256":
        raise ArtifactValidationError("protocol hash algorithm must be SHA-256")
    expected_digest = _expect_sha256(
        value["protocol_payload_sha256"], "protocol_payload_sha256"
    )
    if protocol_payload_sha256(value) != expected_digest:
        raise ArtifactValidationError("protocol payload SHA-256 mismatch")
    _expect_str(value["protocol_id"], "protocol_id")
    _expect_str(value["created_date"], "created_date")
    if value["status"] != "precommitted_unexecuted":
        raise ArtifactValidationError(
            "protocol status must remain precommitted_unexecuted"
        )

    data = value["data"]
    if type(data) is not dict:
        raise ArtifactValidationError("data must be an object")
    _expect_exact_keys(
        data,
        {
            "feature_dim",
            "loader",
            "node_id_semantics",
            "source_domain",
            "target_domains",
            "target_tuning",
        },
        "protocol.data",
    )
    if _expect_int(data["feature_dim"], "data.feature_dim") != TWITCH_FEATURE_DIM:
        raise ArtifactValidationError("protocol feature_dim must be 3170")
    if data["source_domain"] != "DE" or data["target_domains"] != list(TARGET_DOMAINS):
        raise ArtifactValidationError(
            "protocol domains do not match the frozen Twitch design"
        )
    if data["target_tuning"] != "forbidden":
        raise ArtifactValidationError("target-derived tuning must be forbidden")
    _expect_str(data["loader"], "data.loader")
    _expect_str(data["node_id_semantics"], "data.node_id_semantics")

    seeds = value["seeds"]
    if (
        type(seeds) is not list
        or not seeds
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ArtifactValidationError("seeds must be a non-empty integer list")
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ArtifactValidationError("seeds must be unique non-negative integers")

    split = value["source_split"]
    if type(split) is not dict:
        raise ArtifactValidationError("source_split must be an object")
    _expect_exact_keys(
        split,
        {"method", "test_fraction", "train_fraction", "validation_fraction"},
        "source_split",
    )
    if split["method"] != "stratified_seeded_node_split":
        raise ArtifactValidationError("unsupported source split method")
    fractions = [
        _expect_float(split[name], f"source_split.{name}", minimum=0.0, maximum=1.0)
        for name in ("train_fraction", "validation_fraction", "test_fraction")
    ]
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ArtifactValidationError("source split fractions must sum to one")

    models = value["models"]
    if type(models) is not dict:
        raise ArtifactValidationError("models must be an object")
    _expect_exact_keys(
        models,
        {
            "backbones",
            "hidden_dim",
            "learning_rate",
            "patience",
            "train_epochs",
            "use_batch_norm",
            "weight_decay",
        },
        "models",
    )
    backbones = models["backbones"]
    if type(backbones) is not list or not backbones:
        raise ArtifactValidationError("models.backbones must be non-empty")
    allowed_backbones = {"gcn", "graphsage", "appnp", "gat"}
    if any(
        type(item) is not str or item not in allowed_backbones for item in backbones
    ):
        raise ArtifactValidationError("models.backbones contains an unsupported model")
    if len(backbones) != len(set(backbones)):
        raise ArtifactValidationError("models.backbones must be unique")
    _expect_int(models["hidden_dim"], "models.hidden_dim", minimum=1)
    _expect_int(models["train_epochs"], "models.train_epochs", minimum=1)
    _expect_int(models["patience"], "models.patience", minimum=1)
    _expect_float(models["learning_rate"], "models.learning_rate", minimum=0.0)
    _expect_float(models["weight_decay"], "models.weight_decay", minimum=0.0)
    _expect_bool(models["use_batch_norm"], "models.use_batch_norm")

    adaptation = value["adaptation"]
    if type(adaptation) is not dict:
        raise ArtifactValidationError("adaptation must be an object")
    _expect_exact_keys(
        adaptation,
        {
            "adapter_provenance",
            "adapters",
            "lambda_af",
            "lambda_cal",
            "learning_rate",
            "steps",
        },
        "adaptation",
    )
    adapters = adaptation["adapters"]
    allowed_adapters = {"full_method", "tent_entropy", "confidence_source_entropy"}
    if type(adapters) is not list or not adapters:
        raise ArtifactValidationError("adaptation.adapters must be non-empty")
    if any(type(item) is not str or item not in allowed_adapters for item in adapters):
        raise ArtifactValidationError(
            "adaptation.adapters contains an unsupported adapter"
        )
    if len(adapters) != len(set(adapters)):
        raise ArtifactValidationError("adaptation.adapters must be unique")
    provenance = adaptation["adapter_provenance"]
    if type(provenance) is not dict or set(provenance) != set(adapters):
        raise ArtifactValidationError(
            "adaptation.adapter_provenance must describe every frozen adapter"
        )
    for adapter, description in provenance.items():
        _expect_str(description, f"adaptation.adapter_provenance.{adapter}")
    _expect_int(adaptation["steps"], "adaptation.steps", minimum=1)
    for name in ("learning_rate", "lambda_af", "lambda_cal"):
        _expect_float(adaptation[name], f"adaptation.{name}", minimum=0.0)

    thresholds = value["thresholds"]
    if type(thresholds) is not dict:
        raise ArtifactValidationError("thresholds must be an object")
    _expect_exact_keys(thresholds, {"delta", "harm", "phi"}, "thresholds")
    for name in ("delta", "phi", "harm"):
        _expect_float(thresholds[name], f"thresholds.{name}", minimum=0.0, maximum=1.0)

    strategies = value["strategies"]
    if type(strategies) is not list:
        raise ArtifactValidationError("strategies must be a list")
    expected_ids = [
        "reject_all",
        "always",
        "fixed_combined",
        "delta_only",
        "phi_only",
        "source_risk",
        "online_rollback",
    ]
    actual_ids: list[str] = []
    for index, strategy in enumerate(strategies):
        if type(strategy) is not dict:
            raise ArtifactValidationError(f"strategy {index} must be an object")
        _expect_exact_keys(strategy, {"definition", "id"}, f"strategy {index}")
        actual_ids.append(_expect_str(strategy["id"], f"strategy {index}.id"))
        _expect_str(strategy["definition"], f"strategy {index}.definition")
    if actual_ids != expected_ids:
        raise ArtifactValidationError("strategy identities/order are not frozen")

    metrics = value["metrics"]
    expected_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "nll",
        "ece_15",
        "deployed_gain",
        "coverage",
        "harm_over_1pp",
        "residual_harm",
        "foregone_gain",
        "worst_domain_deployed_gain",
    ]
    if metrics != expected_metrics:
        raise ArtifactValidationError("metric identities/order are not frozen")

    execution = value["execution"]
    if type(execution) is not dict:
        raise ArtifactValidationError("execution must be an object")
    _expect_exact_keys(
        execution,
        {
            "candidate_output_policy",
            "evaluation_policy",
            "no_target_derived_tuning",
            "trajectory_policy",
        },
        "execution",
    )
    if not _expect_bool(
        execution["no_target_derived_tuning"], "execution.no_target_derived_tuning"
    ):
        raise ArtifactValidationError("no_target_derived_tuning must be true")
    for name in ("candidate_output_policy", "evaluation_policy", "trajectory_policy"):
        _expect_str(execution[name], f"execution.{name}")
    return value


def load_protocol_with_sha256(path: str | Path) -> tuple[dict[str, Any], str]:
    value, digest = load_json_strict_with_sha256(path)
    return _validate_protocol(value), digest


def load_protocol(path: str | Path) -> dict[str, Any]:
    value, _ = load_protocol_with_sha256(path)
    return value


__all__ = [
    "ArtifactValidationError",
    "PUBLIC_MANIFEST_SCHEMA",
    "PUBLIC_PACK_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SOURCE_TARGETS_SCHEMA",
    "TARGET_DOMAINS",
    "TWITCH_DOMAINS",
    "TWITCH_FEATURE_DIM",
    "PublicGraphPack",
    "SourceTargetsPack",
    "array_sha256",
    "assert_regular_file",
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_json_bytes",
    "deterministic_npz_bytes",
    "load_json_strict",
    "load_json_strict_with_sha256",
    "load_protocol",
    "load_protocol_with_sha256",
    "load_public_graph_from_manifest",
    "load_public_graph_pack",
    "load_public_manifest",
    "load_public_manifest_with_sha256",
    "load_source_targets_pack",
    "protocol_payload_sha256",
    "read_regular_file_bytes",
    "read_npz_strict",
    "read_npz_strict_with_sha256",
    "reject_forbidden_public_fields",
    "resolve_manifest_path",
    "sha256_bytes",
    "sha256_file",
    "validate_candidate_release_root",
    "validate_relative_path",
    "with_protocol_hash",
    "write_deterministic_npz",
]
