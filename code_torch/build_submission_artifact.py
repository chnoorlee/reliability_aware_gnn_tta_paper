"""Build and strictly verify the deterministic manuscript artifact."""

from __future__ import annotations

import argparse
import errno
import hashlib
import io
import json
import os
import stat
import tempfile
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, TypedDict, cast

ROOT = Path(__file__).resolve(strict=True).parents[1]
MANIFEST_REL = "submission_artifact_manifest.json"
ARCHIVE_REL = "reliability_audit_artifact_v1.zip"
MANIFEST_PATH = ROOT / MANIFEST_REL
ARCHIVE_PATH = ROOT / ARCHIVE_REL
PUBLISH_LOCK_NAME = ".reliability_audit_artifact_v1.publish.lock"
LOCK_ACQUISITION_TIMEOUT_SECONDS = 30.0
LOCK_ACQUISITION_POLL_SECONDS = 0.05
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_MODE = 0o100644 << 16
REPARSE_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class FileEntry(TypedDict):
    path: str
    role: str
    bytes: int
    sha256: str


class FrozenCheck(TypedDict):
    path: str
    records: int
    sha256: str
    status: str


class HashCheck(TypedDict):
    path: str
    sha256: str
    status: str


class Manifest(TypedDict):
    artifact: str
    manifest_schema: int
    hash_algorithm: str
    path_policy: str
    evidence_boundary: str
    frozen_result_checks: list[FrozenCheck]
    fixed_hash_checks: list[HashCheck]
    recovered_source_checks: list[HashCheck]
    exclusions: list[str]
    files: list[FileEntry]


@dataclass(frozen=True)
class FrozenResult:
    path: str
    records: int
    sha256: str


@dataclass(frozen=True)
class FileSnapshot:
    data: bytes
    bytes: int
    sha256: str


@dataclass(frozen=True)
class StagedFile:
    path: Path
    snapshot: FileSnapshot
    fingerprint: tuple[int, ...]


FROZEN_RESULTS = (
    FrozenResult(
        "revision_2026-08-19/stage4_fresh_full/heldout_endpoint_final_current/results.json",
        216,
        "aa2a4cf290b03d279f8fb081ba268dff67bd490fb1943d8f8546dcbbe11b9ca4",
    ),
    FrozenResult(
        "revision_2026-08-19/stage4_fresh_full/controller_audit_final_current_v3/results.json",
        216,
        "ee68c88348446c6b544681847752fbf7d28839c92df08751009ba46b6b29cfd6",
    ),
    FrozenResult(
        "revision_2026-08-19/official_tsa_default_label_isolated_audit_v2/results.json",
        72,
        "dbda336d23d379cb00cb6fee877cc8aea1bc94a5eda4dff2a53176c48f20409f",
    ),
    FrozenResult(
        "revision_2026-09-02_stage5_redesign/scope_matched_heldout_v2/results.json",
        216,
        "6561cfd7427b7a7ce9a8822df1e5af5e1cd8399f5e8e318fd185845f03f28942",
    ),
)


EXPECTED_HASHES = {
    "revision_2026-08-19/heldout_risk_coverage_v3/results.json": "2efb1e340d351e4aec00b838ffed926d05f987107b8b70d2f2e01afcb90ffcd9",
    "code_torch/scope_replay_analysis.py": "66c281809a7fa32108bde3cd7836d5be6820d828ee9cca779ec0b205966846c5",
    "code_torch/stage4_reanalysis.py": "2780c55e694e5b6ebae4e3e3b5de8149909b2a75f5073430720d755730082905",
    "code_torch/stage5_selective_audit.py": "88e01e56688e9b04cb664558c983f0ab711c44ef6609f172a0c4e6fef32ec725",
    "revision_2026-08-19/stage4_fresh_full/reanalysis_current_final_10k_1/reanalysis_manifest.json": "7bbc1577e6c8b4bce18e67067051e73ee8cc3a7e3997e0adf9a57899e11a692e",
    "revision_2026-08-19/stage4_fresh_full/reanalysis_current_final_10k_1/stage4_reanalysis.json": "591f6eb2b47391435c4676afd6d6627a20d75a586c6bb2da6d23b785611e79a5",
    "revision_2026-09-02_stage5_redesign/scope_replay_analysis/scope_replay_analysis.json": "2e7aed2b88e75c76084c52a08d26c1f35b671022026357c21b55c361c769c679",
    "revision_2026-09-02_stage5_redesign/scope_replay_analysis/validation_report.json": "eaa9991949c00b1dc6e4d87ec00e920054452f59d1ac30465786e2a12df831bb",
}


RECOVERED_ROOT = "revision_2026-09-02_stage5_redesign/recovered_source_snapshots"
RECOVERED_HASHES = {
    f"{RECOVERED_ROOT}/heldout_controller/adaptation.py": (
        "bc904bd20bf91a3c79957db14a8a785a2695090cb73a72c2f9de59c0af79b37f"
    ),
    f"{RECOVERED_ROOT}/heldout_controller/stress_surface.py": (
        "050c4183b723141755872dc96c690d6893e9ff93bad2b886e5e2b48f26bd96bf"
    ),
    f"{RECOVERED_ROOT}/scope_replay/proxy_scope_audit.py": (
        "997d3db595b1be13f38413c1994dde7097a59ca39ce7f653ba96ba4696aea948"
    ),
    f"{RECOVERED_ROOT}/scope_replay/stress_surface.py": (
        "b196ea35fc2a14ea861e61ae13d8a7b69161ed72ddfa1f1770cb27221f96fc70"
    ),
}
RECOVERED_PROVENANCE = f"{RECOVERED_ROOT}/RECOVERY_PROVENANCE.md"
RECOVERED_FILES = frozenset({RECOVERED_PROVENANCE, *RECOVERED_HASHES})
RECOVERED_DIRECTORIES = frozenset(
    {
        RECOVERED_ROOT,
        f"{RECOVERED_ROOT}/heldout_controller",
        f"{RECOVERED_ROOT}/scope_replay",
    }
)


FIXED_FILES = {
    "ARTIFACT_README.md": "artifact_documentation",
    "environment_snapshot.json": "environment_provenance",
    "requirements-dev.txt": "environment_specification",
    "requirements.txt": "environment_specification",
    "RESTRICTS.yaml": "claim_and_project_constraints",
    "paper/mypaper/highlights.txt": "submission_material",
    "paper/mypaper/main_submission.pdf": "built_manuscript",
    "paper/mypaper/main_submission.tex": "manuscript_source",
    "paper/mypaper/references_revision.bib": "manuscript_source",
    "paper/mypaper/supplementary.pdf": "built_supplement",
    "paper/mypaper/supplementary.tex": "manuscript_source",
    "revision_2026-08-19/heldout_risk_coverage_v3/results.json": (
        "test_fixture_not_evidence"
    ),
    "code/data.py": "shared_result_generation_source",
    "code/utils.py": "shared_result_generation_source",
    "code/webkb_loader.py": "shared_result_generation_source",
    "code_torch/__init__.py": "analysis_source",
    "code_torch/_np_bridge.py": "result_generation_source",
    "code_torch/adaptation.py": "current_result_generation_source",
    "code_torch/build_submission_artifact.py": "artifact_builder",
    "code_torch/data_adapter.py": "result_generation_source",
    "code_torch/detector.py": "result_generation_source",
    "code_torch/exp_common.py": "result_generation_source",
    "code_torch/models.py": "result_generation_source",
    "code_torch/official_tsa_analysis.py": "analysis_source",
    "code_torch/official_tsa_risk_audit.py": "official_audit_wrapper",
    "code_torch/plot_stage5_selective.py": "figure_source",
    "code_torch/proxy_scope_audit.py": "current_result_generation_source",
    "code_torch/reliability.py": "result_generation_source",
    "code_torch/risk_coverage_analysis.py": "analysis_source",
    "code_torch/run_official_tsa_compat.py": "official_audit_wrapper",
    "code_torch/safety_holdout_audit.py": "result_generation_source",
    "code_torch/scope_replay_analysis.py": "analysis_source",
    "code_torch/stage4_reanalysis.py": "analysis_source",
    "code_torch/stage5_selective_audit.py": "analysis_source",
    "code_torch/stress_surface.py": "current_result_generation_source",
    "code_torch/third_party/ATTRIBUTION.md": "third_party_notice",
    "code_torch/third_party/__init__.py": "third_party_source",
    "code_torch/third_party/eata.py": "third_party_source",
    "code_torch/third_party/tent.py": "third_party_source",
    RECOVERED_PROVENANCE: "recovery_provenance",
    **{path: "recovered_result_generation_source" for path in RECOVERED_HASHES},
}


GLOB_GROUPS = (
    ("paper/mypaper/revision_sections/*.tex", "manuscript_source"),
    ("paper/mypaper/figures_revision/*.pdf", "built_figure"),
    ("paper/mypaper/figures_revision/*.png", "figure_preview"),
    ("code_torch/tests/*.py", "test_source"),
    (
        "revision_2026-08-19/stage4_fresh_full/reanalysis_current_final_10k_1/*",
        "final_derived_analysis",
    ),
    (
        "revision_2026-09-02_stage5_redesign/scope_replay_analysis/*",
        "final_scope_analysis",
    ),
    (
        "revision_2026-09-02_stage5_redesign/selective_audit/*",
        "final_selective_analysis",
    ),
)


EXCLUDED_TESTS = frozenset(
    {
        "code_torch/tests/test_build_submission_artifact.py",
        "code_torch/tests/test_natural_shift_pipeline.py",
        "code_torch/tests/test_risk_estimator.py",
        "code_torch/tests/test_topology_anchor.py",
    }
)


def canonical_relative(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError(f"Path is not canonical POSIX-relative: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Path is not canonical POSIX-relative: {value!r}")
    canonical = pure.as_posix()
    if canonical != value:
        raise ValueError(f"Path is not canonical POSIX-relative: {value!r}")
    return canonical


def has_reparse_flag(file_stat: os.stat_result) -> bool:
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return bool(attributes & REPARSE_FLAG)


def stat_fingerprint(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        getattr(file_stat, "st_file_attributes", 0),
    )


def lock_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    """Return identity fields that do not change when the lock marker is written."""

    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        getattr(file_stat, "st_file_attributes", 0),
    )


def checked_path(relative: str, *, directory: bool = False) -> Path:
    relative = canonical_relative(relative)
    path = ROOT.joinpath(*PurePosixPath(relative).parts)
    root_stat = ROOT.lstat()
    if has_reparse_flag(root_stat) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"Repository root is a symlink or reparse point: {ROOT}")

    current = ROOT
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Required artifact path is missing: {relative}"
            ) from exc
        if has_reparse_flag(item_stat) or stat.S_ISLNK(item_stat.st_mode):
            raise ValueError(f"Symlink or reparse point is forbidden: {relative}")

    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or resolved != path:
        raise ValueError(f"Selected path escapes the repository root: {relative}")
    final_stat = path.lstat()
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(final_stat.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"Selected path is not a {kind}: {relative}")
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_checked(relative: str) -> FileSnapshot:
    path = checked_path(relative)
    before = path.lstat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if has_reparse_flag(opened) or not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Opened artifact path is not a regular file: {relative}")
        if stat_fingerprint(before) != stat_fingerprint(opened):
            raise RuntimeError(f"Artifact path changed before reading: {relative}")
        data = handle.read()
        after_handle = os.fstat(handle.fileno())
    after_path = path.lstat()
    checked_again = checked_path(relative)
    fingerprints = {
        stat_fingerprint(before),
        stat_fingerprint(opened),
        stat_fingerprint(after_handle),
        stat_fingerprint(after_path),
    }
    if len(fingerprints) != 1 or checked_again != path:
        raise RuntimeError(f"Artifact path changed while reading: {relative}")
    if len(data) != after_path.st_size:
        raise RuntimeError(f"Artifact file size changed while reading: {relative}")
    return FileSnapshot(data=data, bytes=len(data), sha256=sha256_bytes(data))


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def parse_json_object(data: bytes, context: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        parsed = json.loads(text, object_pairs_hook=unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid JSON in {context}: {exc}") from exc
    if type(parsed) is not dict:
        raise ValueError(f"Expected a JSON object in {context}")
    return cast(dict[str, Any], parsed)


def require_exact_keys(
    value: dict[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"Invalid keys in {context}; missing={missing}, unknown={unknown}"
        )


def require_string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"Expected a non-empty string at {context}")
    return value


def require_integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected an integer >= {minimum} at {context}")
    return value


def require_sha256(value: Any, context: str) -> str:
    digest = require_string(value, context)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"Expected a lowercase SHA-256 digest at {context}")
    return digest


def require_list(value: Any, context: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"Expected a JSON array at {context}")
    return value


def require_object(value: Any, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"Expected a JSON object at {context}")
    return cast(dict[str, Any], value)


def parse_file_entry(value: Any, index: int) -> FileEntry:
    context = f"manifest.files[{index}]"
    item = require_object(value, context)
    require_exact_keys(item, frozenset({"path", "role", "bytes", "sha256"}), context)
    path = canonical_relative(require_string(item["path"], f"{context}.path"))
    return {
        "path": path,
        "role": require_string(item["role"], f"{context}.role"),
        "bytes": require_integer(item["bytes"], f"{context}.bytes"),
        "sha256": require_sha256(item["sha256"], f"{context}.sha256"),
    }


def parse_frozen_check(value: Any, index: int) -> FrozenCheck:
    context = f"manifest.frozen_result_checks[{index}]"
    item = require_object(value, context)
    require_exact_keys(
        item, frozenset({"path", "records", "sha256", "status"}), context
    )
    status = require_string(item["status"], f"{context}.status")
    if status != "verified":
        raise ValueError(f"Unexpected status at {context}.status: {status!r}")
    return {
        "path": canonical_relative(require_string(item["path"], f"{context}.path")),
        "records": require_integer(item["records"], f"{context}.records", minimum=1),
        "sha256": require_sha256(item["sha256"], f"{context}.sha256"),
        "status": status,
    }


def parse_hash_check(value: Any, index: int, field: str) -> HashCheck:
    context = f"manifest.{field}[{index}]"
    item = require_object(value, context)
    require_exact_keys(item, frozenset({"path", "sha256", "status"}), context)
    status = require_string(item["status"], f"{context}.status")
    if status != "verified":
        raise ValueError(f"Unexpected status at {context}.status: {status!r}")
    return {
        "path": canonical_relative(require_string(item["path"], f"{context}.path")),
        "sha256": require_sha256(item["sha256"], f"{context}.sha256"),
        "status": status,
    }


def parse_manifest(data: bytes) -> Manifest:
    raw = parse_json_object(data, MANIFEST_REL)
    top_keys = frozenset(
        {
            "artifact",
            "manifest_schema",
            "hash_algorithm",
            "path_policy",
            "evidence_boundary",
            "frozen_result_checks",
            "fixed_hash_checks",
            "recovered_source_checks",
            "exclusions",
            "files",
        }
    )
    require_exact_keys(raw, top_keys, "manifest")

    file_values = require_list(raw["files"], "manifest.files")
    files = [parse_file_entry(value, index) for index, value in enumerate(file_values)]
    paths = [entry["path"] for entry in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate file path is forbidden in manifest.files")

    frozen_values = require_list(
        raw["frozen_result_checks"], "manifest.frozen_result_checks"
    )
    fixed_values = require_list(raw["fixed_hash_checks"], "manifest.fixed_hash_checks")
    recovered_values = require_list(
        raw["recovered_source_checks"], "manifest.recovered_source_checks"
    )
    exclusions_values = require_list(raw["exclusions"], "manifest.exclusions")
    exclusions = [
        require_string(value, f"manifest.exclusions[{index}]")
        for index, value in enumerate(exclusions_values)
    ]
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("Duplicate manifest exclusion is forbidden")

    return {
        "artifact": require_string(raw["artifact"], "manifest.artifact"),
        "manifest_schema": require_integer(
            raw["manifest_schema"], "manifest.manifest_schema", minimum=1
        ),
        "hash_algorithm": require_string(
            raw["hash_algorithm"], "manifest.hash_algorithm"
        ),
        "path_policy": require_string(raw["path_policy"], "manifest.path_policy"),
        "evidence_boundary": require_string(
            raw["evidence_boundary"], "manifest.evidence_boundary"
        ),
        "frozen_result_checks": [
            parse_frozen_check(value, index)
            for index, value in enumerate(frozen_values)
        ],
        "fixed_hash_checks": [
            parse_hash_check(value, index, "fixed_hash_checks")
            for index, value in enumerate(fixed_values)
        ],
        "recovered_source_checks": [
            parse_hash_check(value, index, "recovered_source_checks")
            for index, value in enumerate(recovered_values)
        ],
        "exclusions": exclusions,
        "files": files,
    }


def validate_recovered_tree() -> None:
    root = checked_path(RECOVERED_ROOT, directory=True)
    found_files: set[str] = set()
    found_directories: set[str] = {RECOVERED_ROOT}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_rel = directory_path.relative_to(ROOT).as_posix()
        checked_path(directory_rel, directory=True)
        for dirname in dirnames:
            child = directory_path / dirname
            child_rel = child.relative_to(ROOT).as_posix()
            checked_path(child_rel, directory=True)
            found_directories.add(child_rel)
        for filename in filenames:
            child = directory_path / filename
            child_rel = child.relative_to(ROOT).as_posix()
            checked_path(child_rel)
            found_files.add(child_rel)
    if found_files != set(RECOVERED_FILES):
        raise ValueError(
            "Recovered-source file allowlist mismatch; "
            f"missing={sorted(RECOVERED_FILES - found_files)}, "
            f"unknown={sorted(found_files - RECOVERED_FILES)}"
        )
    if found_directories != set(RECOVERED_DIRECTORIES):
        raise ValueError(
            "Recovered-source directory allowlist mismatch; "
            f"missing={sorted(RECOVERED_DIRECTORIES - found_directories)}, "
            f"unknown={sorted(found_directories - RECOVERED_DIRECTORIES)}"
        )


def collect_files() -> dict[str, str]:
    validate_recovered_tree()
    files = dict(FIXED_FILES)
    for result in FROZEN_RESULTS:
        files[result.path] = "frozen_raw_result"
    for pattern, role in GLOB_GROUPS:
        selected: list[str] = []
        for path in sorted(
            ROOT.glob(pattern), key=lambda candidate: candidate.as_posix()
        ):
            rel = canonical_relative(path.relative_to(ROOT).as_posix())
            if rel in EXCLUDED_TESTS:
                continue
            checked_path(rel)
            selected.append(rel)
            previous = files.setdefault(rel, role)
            if previous != role:
                raise ValueError(
                    f"Conflicting artifact roles for {rel}: {previous}, {role}"
                )
        if not selected:
            raise FileNotFoundError(f"Required artifact group is empty: {pattern}")
    if EXCLUDED_TESTS & set(files):
        raise ValueError("An excluded exploratory test entered the artifact inventory")
    for rel in sorted(files):
        checked_path(rel)
    return files


def snapshot_files(files: dict[str, str]) -> dict[str, FileSnapshot]:
    return {rel: read_checked(rel) for rel in sorted(files)}


def validate_frozen_results(
    snapshots: dict[str, FileSnapshot],
) -> list[FrozenCheck]:
    checks: list[FrozenCheck] = []
    for expected in FROZEN_RESULTS:
        snapshot = snapshots[expected.path]
        if snapshot.sha256 != expected.sha256:
            raise ValueError(
                f"Frozen result hash mismatch for {expected.path}: {snapshot.sha256}"
            )
        payload = parse_json_object(snapshot.data, expected.path)
        records = payload.get("records")
        if type(records) is not list or len(records) != expected.records:
            raise ValueError(
                f"Frozen result record-count mismatch for {expected.path}: "
                f"expected {expected.records}"
            )
        if payload.get("status") != "complete":
            raise ValueError(f"Frozen result is not complete: {expected.path}")
        checks.append(
            {
                "path": expected.path,
                "records": expected.records,
                "sha256": snapshot.sha256,
                "status": "verified",
            }
        )
    return checks


def validate_expected_hashes(
    snapshots: dict[str, FileSnapshot],
) -> list[HashCheck]:
    checks: list[HashCheck] = []
    for rel, expected in sorted(EXPECTED_HASHES.items()):
        actual = snapshots[rel].sha256
        if actual != expected:
            raise ValueError(f"Expected hash mismatch for {rel}: {actual}")
        checks.append({"path": rel, "sha256": actual, "status": "verified"})
    return checks


def validate_recovered_sources(
    snapshots: dict[str, FileSnapshot],
) -> list[HashCheck]:
    checks: list[HashCheck] = []
    for rel, expected in sorted(RECOVERED_HASHES.items()):
        actual = snapshots[rel].sha256
        if actual != expected:
            raise ValueError(f"Recovered source hash mismatch for {rel}: {actual}")
        checks.append({"path": rel, "sha256": actual, "status": "verified"})
    return checks


def make_manifest(
    files: dict[str, str], snapshots: dict[str, FileSnapshot]
) -> Manifest:
    entries: list[FileEntry] = []
    for rel, role in sorted(files.items()):
        snapshot = snapshots[rel]
        entries.append(
            {
                "path": rel,
                "role": role,
                "bytes": snapshot.bytes,
                "sha256": snapshot.sha256,
            }
        )
    return {
        "artifact": "reliability_audit_artifact_v1",
        "manifest_schema": 1,
        "hash_algorithm": "SHA-256",
        "path_policy": "repository-relative canonical POSIX paths; no links",
        "evidence_boundary": (
            "Audit evidence only; no distribution-free safety certificate and no "
            "population inference from the six-graph sensitivity intervals."
        ),
        "frozen_result_checks": validate_frozen_results(snapshots),
        "fixed_hash_checks": validate_expected_hashes(snapshots),
        "recovered_source_checks": validate_recovered_sources(snapshots),
        "exclusions": [
            "development, smoke, aborted, and corrupted runs",
            "legacy incomplete source snapshots",
            "downloaded data and model checkpoints",
            "LaTeX auxiliaries and rendered QA pages",
            "external official TSA checkout",
            "prospective natural-shift pipeline and its tests, which have no reported results",
            "exploratory risk-estimator and topology-anchor diagnostics not used in the manuscript",
        ],
        "files": entries,
    }


def encode_manifest(manifest: Manifest) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def prepare_output_target(relative: str) -> Path:
    relative = canonical_relative(relative)
    path = ROOT.joinpath(*PurePosixPath(relative).parts)
    parent_rel = path.parent.relative_to(ROOT).as_posix()
    if parent_rel == ".":
        root_stat = ROOT.lstat()
        if has_reparse_flag(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"Invalid repository root: {ROOT}")
    else:
        checked_path(parent_rel, directory=True)
    if path.exists() or path.is_symlink():
        checked_path(relative)
    return path


def stage_bytes(destination: Path, data: bytes) -> StagedFile:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=ROOT,
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        file_stat = path.lstat()
        if (
            has_reparse_flag(file_stat)
            or not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_nlink != 1
        ):
            raise RuntimeError(f"Unsafe staged output path: {path.name}")
        snapshot = FileSnapshot(
            data=data,
            bytes=len(data),
            sha256=sha256_bytes(data),
        )
        return StagedFile(
            path=path,
            snapshot=snapshot,
            fingerprint=stat_fingerprint(file_stat),
        )
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink == 1:
                path.unlink(missing_ok=True)
        raise


def validate_staged(staged: StagedFile) -> None:
    try:
        before = staged.path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Staged output disappeared: {staged.path.name}") from exc
    if (
        has_reparse_flag(before)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat_fingerprint(before) != staged.fingerprint
    ):
        raise RuntimeError(f"Staged output identity changed: {staged.path.name}")
    with staged.path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if opened.st_nlink != 1 or stat_fingerprint(opened) != staged.fingerprint:
            raise RuntimeError(
                f"Staged output changed before reading: {staged.path.name}"
            )
        data = handle.read()
        after_handle = os.fstat(handle.fileno())
    after_path = staged.path.lstat()
    if (
        stat_fingerprint(after_handle) != staged.fingerprint
        or stat_fingerprint(after_path) != staged.fingerprint
        or data != staged.snapshot.data
    ):
        raise RuntimeError(f"Staged output changed while reading: {staged.path.name}")


def cleanup_staged(staged: StagedFile | None) -> None:
    if staged is None:
        return
    try:
        file_stat = staged.path.lstat()
    except FileNotFoundError:
        return
    if (
        not has_reparse_flag(file_stat)
        and stat.S_ISREG(file_stat.st_mode)
        and file_stat.st_nlink == 1
        and stat_fingerprint(file_stat) == staged.fingerprint
    ):
        staged.path.unlink()


def replace_staged(staged: StagedFile, destination: Path) -> None:
    validate_staged(staged)
    os.replace(staged.path, destination)


def acquire_file_lock(handle: BinaryIO) -> None:
    deadline = time.monotonic() + LOCK_ACQUISITION_TIMEOUT_SECONDS
    if os.name == "nt":
        import msvcrt

        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for the artifact publication lock"
                    ) from exc
                time.sleep(LOCK_ACQUISITION_POLL_SECONDS)

    handle.seek(0)
    fcntl_module: Any = __import__("fcntl")
    while True:
        try:
            fcntl_module.flock(
                handle.fileno(),
                fcntl_module.LOCK_EX | fcntl_module.LOCK_NB,
            )
            return
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the artifact publication lock"
                ) from exc
            time.sleep(LOCK_ACQUISITION_POLL_SECONDS)


def release_file_lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl_module: Any = __import__("fcntl")
    fcntl_module.flock(handle.fileno(), fcntl_module.LOCK_UN)


def validate_lock_handle(handle: BinaryIO, lock_path: Path) -> os.stat_result:
    opened = os.fstat(handle.fileno())
    path_stat = lock_path.lstat()
    if (
        has_reparse_flag(opened)
        or has_reparse_flag(path_stat)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or lock_identity(opened) != lock_identity(path_stat)
    ):
        raise RuntimeError(f"Unsafe publication lock path: {lock_path.name}")
    return opened


def initialize_and_validate_lock_marker(handle: BinaryIO, lock_path: Path) -> None:
    opened = validate_lock_handle(handle, lock_path)
    if os.name == "nt":
        if opened.st_size == 0:
            # The byte-range lock is already held, so recovery of an abandoned
            # empty lock file cannot race another initializer.
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
            opened = validate_lock_handle(handle, lock_path)
        handle.seek(0)
        if opened.st_size != 1 or handle.read(2) != b"\0":
            raise RuntimeError(f"Invalid publication lock marker: {lock_path.name}")


def open_publication_lock(lock_path: Path) -> BinaryIO:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        validate_lock_handle(handle, lock_path)
        return handle
    except BaseException:
        handle.close()
        raise


@contextmanager
def publication_lock() -> Iterator[None]:
    lock_path = ROOT / PUBLISH_LOCK_NAME
    handle: BinaryIO | None = None
    acquired = False
    try:
        handle = open_publication_lock(lock_path)
        acquire_file_lock(handle)
        acquired = True
        initialize_and_validate_lock_marker(handle, lock_path)
        yield
    finally:
        if handle is not None:
            try:
                if acquired:
                    release_file_lock(handle)
            finally:
                handle.close()


def write_temp_manifest(destination: Path, data: bytes) -> StagedFile:
    return stage_bytes(destination, data)


def configured_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = ZIP_MODE
    info.internal_attr = 0
    return info


def encode_archive(
    files: dict[str, str],
    snapshots: dict[str, FileSnapshot],
    manifest_bytes: bytes,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for rel in sorted(files):
            archive.writestr(
                configured_zip_info(rel),
                snapshots[rel].data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        archive.writestr(
            configured_zip_info(MANIFEST_REL),
            manifest_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    return output.getvalue()


def write_temp_archive(
    destination: Path,
    files: dict[str, str],
    snapshots: dict[str, FileSnapshot],
    manifest_bytes: bytes,
) -> StagedFile:
    return stage_bytes(
        destination,
        encode_archive(files, snapshots, manifest_bytes),
    )


def publish_pair_locked(
    archive: StagedFile,
    manifest: StagedFile,
    archive_target: Path,
    manifest_target: Path,
) -> tuple[StagedFile | None, StagedFile | None]:
    archive_exists = archive_target.exists() or archive_target.is_symlink()
    manifest_exists = manifest_target.exists() or manifest_target.is_symlink()
    if archive_exists != manifest_exists:
        raise RuntimeError("Existing archive/manifest pair is incomplete")

    backup_archive: StagedFile | None = None
    backup_manifest: StagedFile | None = None
    archive_published = False
    manifest_published = False
    try:
        if archive_exists:
            previous_archive = read_checked(ARCHIVE_REL)
            previous_manifest = read_checked(MANIFEST_REL)
            backup_archive = stage_bytes(archive_target, previous_archive.data)
            backup_manifest = stage_bytes(manifest_target, previous_manifest.data)
        replace_staged(archive, archive_target)
        archive_published = True
        replace_staged(manifest, manifest_target)
        manifest_published = True
        if read_checked(ARCHIVE_REL) != archive.snapshot:
            raise RuntimeError("Published archive differs from staged bytes")
        if read_checked(MANIFEST_REL) != manifest.snapshot:
            raise RuntimeError("Published manifest differs from staged bytes")
        return backup_archive, backup_manifest
    except Exception as publication_error:
        retained = [
            str(staged.path)
            for staged in (backup_archive, backup_manifest)
            if staged is not None and staged.path.exists()
        ]
        retained_text = ", ".join(retained) if retained else "none"
        state = (
            f"archive_published={archive_published}, "
            f"manifest_published={manifest_published}"
        )
        raise RuntimeError(
            "Artifact publication failed; automatic rollback is unsafe because "
            "a non-cooperating writer may have changed an output. Current outputs "
            "may be a mixed archive/manifest pair; no recovery write was attempted. "
            "prior copies retained at: "
            f"{retained_text}; {state}; publication error: {publication_error!r}"
        ) from publication_error


def publish_pair(
    archive: StagedFile,
    manifest: StagedFile,
    archive_target: Path,
    manifest_target: Path,
) -> None:
    with publication_lock():
        backup_archive, backup_manifest = publish_pair_locked(
            archive, manifest, archive_target, manifest_target
        )
        cleanup_staged(backup_archive)
        cleanup_staged(backup_manifest)


def validate_source_snapshot(
    expected_files: dict[str, str],
    expected_snapshots: dict[str, FileSnapshot],
) -> None:
    current_files = collect_files()
    if current_files != expected_files:
        raise RuntimeError("Artifact source inventory changed during the build")
    current_snapshots = snapshot_files(current_files)
    if current_snapshots != expected_snapshots:
        raise RuntimeError("Artifact source files changed during the build")


def publish_pair_for_source_snapshot(
    archive: StagedFile,
    manifest: StagedFile,
    archive_target: Path,
    manifest_target: Path,
    files: dict[str, str],
    snapshots: dict[str, FileSnapshot],
) -> None:
    """Publish only if the staged source snapshot is still current."""

    with publication_lock():
        validate_source_snapshot(files, snapshots)
        backup_archive, backup_manifest = publish_pair_locked(
            archive, manifest, archive_target, manifest_target
        )
        try:
            validate_source_snapshot(files, snapshots)
            archive_snapshot = read_checked(ARCHIVE_REL)
            manifest_snapshot = read_checked(MANIFEST_REL)
            verify_pair(
                archive_snapshot.data,
                manifest_snapshot.data,
                files,
                snapshots,
            )
        except Exception as validation_error:
            retained = [
                str(staged.path)
                for staged in (backup_archive, backup_manifest)
                if staged is not None and staged.path.exists()
            ]
            retained_text = ", ".join(retained) if retained else "none"
            raise RuntimeError(
                "Artifact publication completed, but post-publication validation "
                "failed. Current outputs are not certified and may have been "
                "changed by a non-cooperating writer; no recovery write was "
                "attempted. prior copies retained at: "
                f"{retained_text}; validation error: {validation_error!r}"
            ) from validation_error
        cleanup_staged(backup_archive)
        cleanup_staged(backup_manifest)


def verify_zip_metadata(info: zipfile.ZipInfo, name: str) -> None:
    if info.filename != canonical_relative(name):
        raise ValueError(f"Non-canonical ZIP member path: {info.filename!r}")
    if info.is_dir() or info.filename.endswith("/"):
        raise ValueError(f"Directory member is forbidden in ZIP: {name}")
    if info.date_time != ZIP_TIMESTAMP:
        raise ValueError(f"Non-deterministic timestamp in ZIP member: {name}")
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f"Unexpected compression method in ZIP member: {name}")
    if info.create_system != 3 or info.external_attr != ZIP_MODE:
        raise ValueError(f"Unexpected platform metadata in ZIP member: {name}")
    if info.extra or info.comment:
        raise ValueError(f"Unexpected extra metadata in ZIP member: {name}")
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted ZIP member is forbidden: {name}")


def verify_archive_bytes(
    archive_bytes: bytes,
    files: dict[str, str],
    snapshots: dict[str, FileSnapshot],
    manifest: Manifest,
    manifest_bytes: bytes,
) -> None:
    expected_names = [*sorted(files), MANIFEST_REL]
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
            if archive.comment:
                raise ValueError("ZIP archive comment must be empty")
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP member name is forbidden")
            if names != expected_names:
                raise ValueError(
                    "ZIP member inventory/order mismatch; "
                    f"expected={expected_names}, actual={names}"
                )
            recorded = {entry["path"]: entry for entry in manifest["files"]}
            for info in infos:
                verify_zip_metadata(info, info.filename)
                expected_size = (
                    len(manifest_bytes)
                    if info.filename == MANIFEST_REL
                    else recorded[info.filename]["bytes"]
                )
                if info.file_size != expected_size:
                    raise ValueError(f"ZIP member size mismatch: {info.filename}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"ZIP CRC failure: {bad_member}")
            for info in infos:
                data = archive.read(info)
                if info.filename == MANIFEST_REL:
                    if data != manifest_bytes:
                        raise ValueError(
                            "Embedded manifest differs byte-for-byte from external manifest"
                        )
                    continue
                snapshot = snapshots[info.filename]
                entry = recorded[info.filename]
                if (
                    len(data) != entry["bytes"]
                    or sha256_bytes(data) != entry["sha256"]
                    or data != snapshot.data
                ):
                    raise ValueError(
                        f"ZIP member differs from manifest/current file: {info.filename}"
                    )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {exc}") from exc
    canonical_archive = encode_archive(files, snapshots, manifest_bytes)
    if archive_bytes != canonical_archive:
        raise ValueError("ZIP bytes differ from the canonical deterministic encoding")


def verify_pair(
    archive_bytes: bytes,
    manifest_bytes: bytes,
    files: dict[str, str],
    snapshots: dict[str, FileSnapshot],
) -> Manifest:
    manifest = parse_manifest(manifest_bytes)
    expected = make_manifest(files, snapshots)
    if manifest != expected:
        raise ValueError(
            "Manifest structure/content differs from the recomputed manifest"
        )
    canonical_bytes = encode_manifest(expected)
    if manifest_bytes != canonical_bytes:
        raise ValueError("External manifest is not in canonical deterministic encoding")
    verify_archive_bytes(archive_bytes, files, snapshots, manifest, manifest_bytes)
    return manifest


def verify_published() -> tuple[int, str]:
    with publication_lock():
        files = collect_files()
        snapshots = snapshot_files(files)
        manifest_snapshot = read_checked(MANIFEST_REL)
        archive_snapshot = read_checked(ARCHIVE_REL)
        verify_pair(
            archive_snapshot.data,
            manifest_snapshot.data,
            files,
            snapshots,
        )
        return len(files), archive_snapshot.sha256


def build() -> tuple[int, int, str]:
    files = collect_files()
    snapshots = snapshot_files(files)
    manifest = make_manifest(files, snapshots)
    manifest_bytes = encode_manifest(manifest)
    archive_target = prepare_output_target(ARCHIVE_REL)
    manifest_target = prepare_output_target(MANIFEST_REL)
    archive_temp: StagedFile | None = None
    manifest_temp: StagedFile | None = None
    try:
        archive_temp = write_temp_archive(
            archive_target, files, snapshots, manifest_bytes
        )
        manifest_temp = write_temp_manifest(manifest_target, manifest_bytes)
        verify_pair(
            archive_temp.snapshot.data,
            manifest_temp.snapshot.data,
            files,
            snapshots,
        )
        prepare_output_target(ARCHIVE_REL)
        prepare_output_target(MANIFEST_REL)
        publish_pair_for_source_snapshot(
            archive_temp,
            manifest_temp,
            archive_target,
            manifest_target,
            files,
            snapshots,
        )
        archive_size = archive_temp.snapshot.bytes
        archive_hash = archive_temp.snapshot.sha256
        archive_temp = None
        manifest_temp = None
        return len(files), archive_size, archive_hash
    finally:
        cleanup_staged(archive_temp)
        cleanup_staged(manifest_temp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Strictly validate the existing manifest, archive, and current files.",
    )
    args = parser.parse_args()
    if args.verify_only:
        count, archive_hash = verify_published()
        print(
            f"PASS: {count} files and ZIP members match; "
            f"archive_sha256={archive_hash}"
        )
        return
    count, archive_size, archive_hash = build()
    print(
        f"WROTE+VERIFIED: {MANIFEST_PATH.name} ({count} files); "
        f"{ARCHIVE_PATH.name} ({archive_size} bytes, "
        f"sha256={archive_hash})"
    )


if __name__ == "__main__":
    main()
