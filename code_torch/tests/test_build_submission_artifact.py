"""Adversarial regression tests for the release artifact builder."""

from __future__ import annotations

import io
import errno
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import pytest

import build_submission_artifact as builder


def archive_fixture() -> tuple[
    dict[str, str],
    dict[str, builder.FileSnapshot],
    builder.Manifest,
    bytes,
]:
    relative = "payload.bin"
    data = bytes(range(256)) * 128
    snapshot = builder.FileSnapshot(
        data=data,
        bytes=len(data),
        sha256=builder.sha256_bytes(data),
    )
    manifest_bytes = b'{"fixture":true}\n'
    manifest = cast(
        builder.Manifest,
        cast(
            Any,
            {
                "files": [
                    {
                        "path": relative,
                        "role": "fixture",
                        "bytes": len(data),
                        "sha256": snapshot.sha256,
                    }
                ]
            },
        ),
    )
    return {relative: "fixture"}, {relative: snapshot}, manifest, manifest_bytes


def assert_archive_rejected(
    archive_bytes: bytes,
    files: dict[str, str],
    snapshots: dict[str, builder.FileSnapshot],
    manifest: builder.Manifest,
    manifest_bytes: bytes,
) -> None:
    with pytest.raises(ValueError):
        builder.verify_archive_bytes(
            archive_bytes,
            files,
            snapshots,
            manifest,
            manifest_bytes,
        )


def test_verifier_requires_complete_canonical_zip_bytes() -> None:
    files, snapshots, manifest, manifest_bytes = archive_fixture()
    canonical = builder.encode_archive(files, snapshots, manifest_bytes)
    builder.verify_archive_bytes(
        canonical,
        files,
        snapshots,
        manifest,
        manifest_bytes,
    )

    assert_archive_rejected(
        b"unexpected-prefix" + canonical,
        files,
        snapshots,
        manifest,
        manifest_bytes,
    )

    altered_metadata = bytearray(canonical)
    central_offset = altered_metadata.index(b"PK\x01\x02")
    altered_metadata[central_offset + 4] = 21
    assert_archive_rejected(
        bytes(altered_metadata),
        files,
        snapshots,
        manifest,
        manifest_bytes,
    )


def test_verifier_rejects_alternate_deflate_encoding() -> None:
    files, snapshots, manifest, manifest_bytes = archive_fixture()
    canonical = builder.encode_archive(files, snapshots, manifest_bytes)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for relative in sorted(files):
            archive.writestr(
                builder.configured_zip_info(relative),
                snapshots[relative].data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=1,
            )
        archive.writestr(
            builder.configured_zip_info(builder.MANIFEST_REL),
            manifest_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        )
    alternate = output.getvalue()
    assert alternate != canonical
    assert_archive_rejected(
        alternate,
        files,
        snapshots,
        manifest,
        manifest_bytes,
    )


def test_staged_identity_check_rejects_hard_link_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    destination = tmp_path / "output.bin"
    staged = builder.stage_bytes(destination, b"release-bytes")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"must-not-change")

    staged.path.unlink()
    os.link(victim, staged.path)
    with pytest.raises(RuntimeError, match="identity changed"):
        builder.validate_staged(staged)
    assert victim.read_bytes() == b"must-not-change"


def test_second_replace_failure_retains_backups_without_unsafe_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    archive = builder.stage_bytes(archive_target, b"new-archive")
    manifest = builder.stage_bytes(manifest_target, b"new-manifest")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated locked manifest")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_second_replace)
    try:
        with pytest.raises(RuntimeError, match="automatic rollback is unsafe"):
            builder.publish_pair(
                archive,
                manifest,
                archive_target,
                manifest_target,
            )
        assert archive_target.read_bytes() == b"new-archive"
        assert manifest_target.read_bytes() == b"old-manifest"
        retained = [path.read_bytes() for path in tmp_path.glob(".*.tmp")]
        assert b"old-archive" in retained
        assert b"old-manifest" in retained
    finally:
        builder.cleanup_staged(archive)
        builder.cleanup_staged(manifest)


def test_publication_lock_serializes_concurrent_publishers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    first_archive = builder.stage_bytes(archive_target, b"first-archive")
    first_manifest = builder.stage_bytes(manifest_target, b"first-manifest")
    second_archive = builder.stage_bytes(archive_target, b"second-archive")
    second_manifest = builder.stage_bytes(manifest_target, b"second-manifest")
    real_replace = os.replace
    first_archive_published = threading.Event()
    release_first = threading.Event()
    second_archive_published = threading.Event()
    failures: list[BaseException] = []

    def pause_first_replace(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        if Path(source) == first_archive.path:
            first_archive_published.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first publisher")
        elif Path(source) == second_archive.path:
            second_archive_published.set()

    def publish(archive: builder.StagedFile, manifest: builder.StagedFile) -> None:
        try:
            builder.publish_pair(
                archive,
                manifest,
                archive_target,
                manifest_target,
            )
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(builder.os, "replace", pause_first_replace)
    first = threading.Thread(target=publish, args=(first_archive, first_manifest))
    second = threading.Thread(target=publish, args=(second_archive, second_manifest))
    first.start()
    assert first_archive_published.wait(timeout=5)
    second.start()
    assert not second_archive_published.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert archive_target.read_bytes() == b"second-archive"
    assert manifest_target.read_bytes() == b"second-manifest"


def test_concurrent_change_is_not_overwritten_and_backups_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    archive = builder.stage_bytes(archive_target, b"new-archive")
    manifest = builder.stage_bytes(manifest_target, b"new-manifest")
    real_replace = os.replace
    calls = 0

    def change_archive_before_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            archive_target.write_bytes(b"concurrent-archive")
            raise PermissionError("simulated locked manifest")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", change_archive_before_second_replace)
    try:
        with pytest.raises(RuntimeError, match="prior copies retained"):
            builder.publish_pair(
                archive,
                manifest,
                archive_target,
                manifest_target,
            )
        assert archive_target.read_bytes() == b"concurrent-archive"
        assert manifest_target.read_bytes() == b"old-manifest"
        retained = [path.read_bytes() for path in tmp_path.glob(".*.tmp")]
        assert b"old-archive" in retained
        assert b"old-manifest" in retained
    finally:
        builder.cleanup_staged(archive)
        builder.cleanup_staged(manifest)


def test_publication_failure_never_attempts_an_automatic_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    archive = builder.stage_bytes(archive_target, b"new-archive")
    manifest = builder.stage_bytes(manifest_target, b"new-manifest")
    real_replace = os.replace
    calls = 0

    def fail_manifest_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("simulated locked manifest")
        if calls > 2:
            raise AssertionError("publication attempted an unsafe restore")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_manifest_replace)
    try:
        with pytest.raises(RuntimeError, match="automatic rollback is unsafe"):
            builder.publish_pair(
                archive,
                manifest,
                archive_target,
                manifest_target,
            )
        assert calls == 2
        assert archive_target.read_bytes() == b"new-archive"
        assert manifest_target.read_bytes() == b"old-manifest"
        retained = [path.read_bytes() for path in tmp_path.glob(".*.tmp")]
        assert b"old-archive" in retained
        assert b"old-manifest" in retained
    finally:
        builder.cleanup_staged(archive)
        builder.cleanup_staged(manifest)


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-lock initialization")
def test_windows_first_use_lock_initialization_is_serialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    start = threading.Barrier(3)
    active = 0
    max_active = 0
    failures: list[BaseException] = []
    state_lock = threading.Lock()

    def enter_lock() -> None:
        nonlocal active, max_active
        try:
            start.wait(timeout=5)
            with builder.publication_lock():
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)
                with state_lock:
                    active -= 1
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=enter_lock)
    second = threading.Thread(target=enter_lock)
    first.start()
    second.start()
    start.wait(timeout=5)
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert max_active == 1
    assert (tmp_path / builder.PUBLISH_LOCK_NAME).read_bytes() == b"\0"


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-lock initialization")
def test_windows_empty_marker_initialization_occurs_after_lock_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    second_observed_empty = threading.Event()
    first_acquired = threading.Event()
    failures: list[BaseException] = []
    real_validate = builder.validate_lock_handle
    real_acquire = builder.acquire_file_lock

    def coordinated_validate(handle: BinaryIO, lock_path: Path) -> os.stat_result:
        opened = real_validate(handle, lock_path)
        if (
            threading.current_thread().name == "initializer-b"
            and not first_acquired.is_set()
        ):
            assert opened.st_size == 0
            second_observed_empty.set()
            assert first_acquired.wait(timeout=5)
        return opened

    def tracked_acquire(handle: BinaryIO) -> None:
        real_acquire(handle)
        if threading.current_thread().name == "initializer-a":
            first_acquired.set()

    monkeypatch.setattr(builder, "validate_lock_handle", coordinated_validate)
    monkeypatch.setattr(builder, "acquire_file_lock", tracked_acquire)

    def enter_lock() -> None:
        try:
            with builder.publication_lock():
                time.sleep(0.1)
        except BaseException as exc:
            failures.append(exc)

    second = threading.Thread(target=enter_lock, name="initializer-b")
    second.start()
    assert second_observed_empty.wait(timeout=5)
    first = threading.Thread(target=enter_lock, name="initializer-a")
    first.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert (tmp_path / builder.PUBLISH_LOCK_NAME).read_bytes() == b"\0"


@pytest.mark.skipif(os.name != "nt", reason="Windows lock error classification")
def test_windows_lock_propagates_non_contention_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import msvcrt

    lock_path = tmp_path / builder.PUBLISH_LOCK_NAME
    lock_path.write_bytes(b"\0")
    handle = lock_path.open("r+b", buffering=0)

    def fail_lock(_descriptor: int, _mode: int, _bytes: int) -> None:
        raise OSError(errno.EBADF, "invalid lock handle")

    monkeypatch.setattr(msvcrt, "locking", fail_lock)
    try:
        with pytest.raises(OSError, match="invalid lock handle"):
            builder.acquire_file_lock(handle)
    finally:
        handle.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows byte-lock initialization")
def test_windows_abandoned_empty_lock_is_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    lock_path = tmp_path / builder.PUBLISH_LOCK_NAME
    lock_path.write_bytes(b"")

    with builder.publication_lock():
        pass
    assert lock_path.read_bytes() == b"\0"


@pytest.mark.skipif(os.name != "nt", reason="Windows cross-process byte locking")
def test_windows_waits_for_healthy_holder_longer_than_native_retry_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    ready = tmp_path / "holder-ready"
    module_root = Path(builder.__file__).resolve().parent
    child_code = "\n".join(
        (
            "import sys, time",
            "from pathlib import Path",
            "sys.path.insert(0, sys.argv[1])",
            "import build_submission_artifact as builder",
            "builder.ROOT = Path(sys.argv[2])",
            "with builder.publication_lock():",
            "    Path(sys.argv[3]).write_text('ready', encoding='ascii')",
            "    time.sleep(10.5)",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(module_root), str(tmp_path), str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("lock holder did not become ready")
            time.sleep(0.02)
        assert process.poll() is None

        started = time.monotonic()
        with builder.publication_lock():
            elapsed = time.monotonic() - started
        assert elapsed >= 10.0
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock timeout")
def test_posix_lock_acquisition_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / builder.PUBLISH_LOCK_NAME
    lock_path.write_bytes(b"")
    holder = lock_path.open("r+b", buffering=0)
    waiter = lock_path.open("r+b", buffering=0)
    monkeypatch.setattr(builder, "LOCK_ACQUISITION_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(builder, "LOCK_ACQUISITION_POLL_SECONDS", 0.005)
    try:
        builder.acquire_file_lock(holder)
        with pytest.raises(TimeoutError, match="publication lock"):
            builder.acquire_file_lock(waiter)
    finally:
        builder.release_file_lock(holder)
        holder.close()
        waiter.close()


def test_source_change_before_locked_publication_preserves_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"staged-source")
    files = {"payload.bin": "fixture"}
    snapshots = builder.snapshot_files(files)
    monkeypatch.setattr(builder, "collect_files", lambda: dict(files))

    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    archive = builder.stage_bytes(archive_target, b"new-archive")
    manifest = builder.stage_bytes(manifest_target, b"new-manifest")
    source.write_bytes(b"changed-source")

    try:
        with pytest.raises(RuntimeError, match="source files changed"):
            builder.publish_pair_for_source_snapshot(
                archive,
                manifest,
                archive_target,
                manifest_target,
                files,
                snapshots,
            )
        assert archive_target.read_bytes() == b"old-archive"
        assert manifest_target.read_bytes() == b"old-manifest"
    finally:
        builder.cleanup_staged(archive)
        builder.cleanup_staged(manifest)


def test_source_change_after_publication_retains_previous_pair_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    source = tmp_path / "payload.bin"
    source.write_bytes(b"staged-source")
    files = {"payload.bin": "fixture"}
    snapshots = builder.snapshot_files(files)
    monkeypatch.setattr(builder, "collect_files", lambda: dict(files))

    archive_target = tmp_path / builder.ARCHIVE_REL
    manifest_target = tmp_path / builder.MANIFEST_REL
    archive_target.write_bytes(b"old-archive")
    manifest_target.write_bytes(b"old-manifest")
    archive = builder.stage_bytes(archive_target, b"new-archive")
    manifest = builder.stage_bytes(manifest_target, b"new-manifest")
    real_publish = builder.publish_pair_locked

    def publish_then_change_source(
        *args: Any, **kwargs: Any
    ) -> tuple[builder.StagedFile | None, builder.StagedFile | None]:
        backups = real_publish(*args, **kwargs)
        source.write_bytes(b"changed-after-publication")
        return backups

    monkeypatch.setattr(builder, "publish_pair_locked", publish_then_change_source)
    try:
        with pytest.raises(RuntimeError, match="post-publication validation failed"):
            builder.publish_pair_for_source_snapshot(
                archive,
                manifest,
                archive_target,
                manifest_target,
                files,
                snapshots,
            )
        assert archive_target.read_bytes() == b"new-archive"
        assert manifest_target.read_bytes() == b"new-manifest"
        retained = [path.read_bytes() for path in tmp_path.glob(".*.tmp")]
        assert b"old-archive" in retained
        assert b"old-manifest" in retained
    finally:
        builder.cleanup_staged(archive)
        builder.cleanup_staged(manifest)


def test_main_reports_size_and_digest_from_the_build_snapshot(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_submission_artifact.py"])
    monkeypatch.setattr(builder, "build", lambda: (99, 1234, "abc123"))

    builder.main()

    output = capsys.readouterr().out
    assert "(99 files)" in output
    assert "(1234 bytes, sha256=abc123)" in output
