from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pytest

from plot_stage5_selective import (
    DEFAULT_FIXED_GATE,
    DEFAULT_INPUT,
    DEFAULT_MANIFEST,
    curve_points,
    load_curve_rows,
    make_figure,
)


def _refresh_output_digest(manifest_path: Path, output_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for output in manifest["outputs"]:
        if Path(output["path"]).name == output_path.name:
            output["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
            break
    else:
        raise AssertionError(f"manifest does not bind {output_path.name}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_variant(tmp_path: Path, mutate, *, refresh_digest: bool = True) -> Path:
    with DEFAULT_INPUT.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    mutate(rows)
    target_path = tmp_path / DEFAULT_INPUT.name
    with target_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(DEFAULT_MANIFEST, tmp_path / DEFAULT_MANIFEST.name)
    shutil.copy2(DEFAULT_FIXED_GATE, tmp_path / DEFAULT_FIXED_GATE.name)
    if refresh_digest:
        _refresh_output_digest(tmp_path / DEFAULT_MANIFEST.name, target_path)
    return target_path


def test_frozen_curve_shape_and_null_zero_coverage_risk() -> None:
    rows = load_curve_rows(DEFAULT_INPUT)

    assert len(rows) == 220
    zero_grid = [
        row
        for row in rows
        if row["point_type"] == "grid" and row["coverage"] == 0.0
    ]
    assert len(zero_grid) == 10
    assert all(row["accepted_harm_rate"] is None for row in zero_grid)
    assert all(row["retained_positive_utility"] == 0.0 for row in zero_grid)


def test_risk_curve_omits_undefined_origin_but_utility_keeps_it() -> None:
    rows = load_curve_rows(DEFAULT_INPUT)

    risk_x, risk_y = curve_points(
        rows,
        adapter="confidence_source_entropy",
        policy="combined_score",
        metric="accepted_harm_rate",
    )
    utility_x, utility_y = curve_points(
        rows,
        adapter="confidence_source_entropy",
        policy="combined_score",
        metric="retained_positive_utility",
    )

    assert len(risk_x) == len(risk_y) == 20
    assert risk_x[0] == pytest.approx(0.05)
    assert len(utility_x) == len(utility_y) == 21
    assert utility_x[0] == 0.0
    assert utility_y[0] == 0.0


def test_loader_rejects_numeric_zero_coverage_risk(tmp_path: Path) -> None:
    invalid = _write_variant(
        tmp_path, lambda rows: rows[0].__setitem__("accepted_harm_rate", "0")
    )

    with pytest.raises(ValueError, match="undefined at zero coverage"):
        load_curve_rows(invalid)


def test_loader_rejects_nan_as_data_corruption(tmp_path: Path) -> None:
    invalid = _write_variant(
        tmp_path, lambda rows: rows[1].__setitem__("retained_positive_utility", "nan")
    )

    with pytest.raises(ValueError, match="finite or explicitly null"):
        load_curve_rows(invalid)


def test_loader_rejects_mixed_input_hashes(tmp_path: Path) -> None:
    invalid = _write_variant(
        tmp_path, lambda rows: rows[1].__setitem__("heldout_input_sha256", "0" * 64)
    )

    with pytest.raises(ValueError, match="Mixed or missing provenance"):
        load_curve_rows(invalid)


def test_loader_rejects_curve_output_digest_tampering(tmp_path: Path) -> None:
    invalid = _write_variant(
        tmp_path,
        lambda rows: rows[1].__setitem__("retained_positive_utility", "0.123"),
        refresh_digest=False,
    )

    with pytest.raises(ValueError, match="output digest disagrees"):
        load_curve_rows(invalid)


def test_loader_rejects_fixed_point_not_bound_to_observation(tmp_path: Path) -> None:
    invalid = _write_variant(tmp_path, lambda rows: None)
    fixed_path = tmp_path / DEFAULT_FIXED_GATE.name
    with fixed_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["accepted_harm_rate"] = "0.99"
    with fixed_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_output_digest(tmp_path / DEFAULT_MANIFEST.name, fixed_path)

    with pytest.raises(ValueError, match="disagrees with frozen observation"):
        load_curve_rows(invalid)


def test_figure_has_four_populated_axes() -> None:
    figure = make_figure(load_curve_rows(DEFAULT_INPUT))
    try:
        assert len(figure.axes) == 4
        assert all(len(axis.lines) == 5 for axis in figure.axes)
        assert all(len(axis.collections) == 1 for axis in figure.axes)
    finally:
        plt.close(figure)
