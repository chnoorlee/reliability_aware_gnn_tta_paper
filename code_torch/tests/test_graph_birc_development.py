"""Contract tests for Graph-BiRC development labels and threshold freezing."""

from __future__ import annotations

import copy
import json
from fractions import Fraction

import numpy as np
import pytest

import graph_birc as graph_birc_module
from graph_birc import (
    GRAPH_BIRC_FEATURE_SCHEMA,
    GraphBiRCHarmScorer,
    GraphBiRCValidationError,
)
from graph_birc_development import (
    GraphBiRCDevelopmentEvent,
    GraphBiRCJointMetricSufficientStatistics,
    GraphBiRCMetricSufficientStatistics,
    GraphBiRCThresholdArtifact,
    build_graph_birc_development_event,
    classify_graph_birc_harm,
    fit_graph_birc_harm_scorer_from_events,
    select_graph_birc_threshold as _select_graph_birc_threshold,
    verify_graph_birc_harm_scorer_against_events,
)

_TEST_PROTOCOL_ID = "test-graph-birc-protocol"
_TEST_PROTOCOL_SHA256 = "a" * 64
_TEST_L2_STRENGTH = 1.0
_TEST_MINIMUM_DEVELOPMENT_GROUPS = 3
_TEST_MAX_ITERATIONS = 100
_TEST_OPTIMIZER_TOLERANCE = 1e-10


def _scorer_deployment_kwargs(**overrides):
    values = {
        "expected_protocol_id": _TEST_PROTOCOL_ID,
        "expected_protocol_sha256": _TEST_PROTOCOL_SHA256,
        "expected_l2_strength": _TEST_L2_STRENGTH,
        "expected_minimum_development_groups": (_TEST_MINIMUM_DEVELOPMENT_GROUPS),
        "expected_max_iterations": _TEST_MAX_ITERATIONS,
        "expected_optimizer_tolerance": _TEST_OPTIMIZER_TOLERANCE,
    }
    values.update(overrides)
    return values


def _threshold_accept_kwargs(artifact: GraphBiRCThresholdArtifact, **overrides):
    values = {
        **_scorer_deployment_kwargs(),
        "expected_threshold_artifact_sha256": artifact.artifact_sha256,
        "expected_maximum_accepted_harm_risk": (artifact.maximum_accepted_harm_risk),
        "expected_minimum_coverage": artifact.minimum_coverage,
    }
    values.update(overrides)
    return values


def select_graph_birc_threshold(*args, **kwargs):
    kwargs.setdefault("protocol_id", _TEST_PROTOCOL_ID)
    kwargs.setdefault("protocol_sha256", _TEST_PROTOCOL_SHA256)
    kwargs.setdefault("expected_l2_strength", _TEST_L2_STRENGTH)
    kwargs.setdefault(
        "expected_minimum_development_groups",
        _TEST_MINIMUM_DEVELOPMENT_GROUPS,
    )
    kwargs.setdefault("expected_max_iterations", _TEST_MAX_ITERATIONS)
    kwargs.setdefault("expected_optimizer_tolerance", _TEST_OPTIMIZER_TOLERANCE)
    return _select_graph_birc_threshold(*args, **kwargs)


def _toy_event(
    event_id: str,
    group_id: str,
    *,
    low_group_confidence: float = 0.8,
    harmful: bool = False,
    node_id_offset: int = 0,
) -> GraphBiRCDevelopmentEvent:
    targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    source = np.asarray(
        [
            [low_group_confidence, 1.0 - low_group_confidence],
            [low_group_confidence, 1.0 - low_group_confidence],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    candidate = source.copy()
    if harmful:
        candidate[0] = [0.1, 0.9]
    return build_graph_birc_development_event(
        event_id=event_id,
        group_id=group_id,
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=candidate,
        targets=targets,
        node_ids=np.arange(targets.size, dtype=np.int64) + node_id_offset,
    )


def _fixed_scorer(
    *, intercept: float = 0.0, evidence_bound: bool = True
) -> GraphBiRCHarmScorer:
    development_groups = ["dev-a", "dev-b", "dev-c"]
    heldout_groups = ["cal-a", "cal-b"]
    split_payload = {
        "scope_claim_status": "caller_declared_not_independently_verified",
        "development_group_ids": development_groups,
        "heldout_group_ids": heldout_groups,
    }
    coefficients = np.zeros(len(GRAPH_BIRC_FEATURE_SCHEMA), dtype=np.float64)
    coefficients[0] = 1.0
    evidence = {}
    if evidence_bound:
        evidence = {
            "protocol_id": _TEST_PROTOCOL_ID,
            "protocol_sha256": _TEST_PROTOCOL_SHA256,
            "development_event_ids": tuple(
                f"synthetic-dev-{index:02d}" for index in range(6)
            ),
            "development_event_sha256s": tuple(
                f"{index + 1:064x}" for index in range(6)
            ),
            "development_input_bundle_sha256s": tuple(
                f"{index + 101:064x}" for index in range(6)
            ),
        }
    return GraphBiRCHarmScorer(
        feature_mean=tuple(np.zeros_like(coefficients)),
        feature_scale=tuple(np.ones_like(coefficients)),
        coefficients=tuple(coefficients),
        intercept=intercept,
        l2_strength=1.0,
        development_event_count=6,
        development_group_count=3,
        heldout_group_count=2,
        development_data_sha256="0" * 64,
        group_split_sha256=graph_birc_module._sha256_canonical(split_payload),
        optimizer_iterations=1,
        minimum_development_groups=_TEST_MINIMUM_DEVELOPMENT_GROUPS,
        max_iterations=_TEST_MAX_ITERATIONS,
        optimizer_tolerance=_TEST_OPTIMIZER_TOLERANCE,
        **evidence,
    )


def _event_fitted_scorer(
    *, l2_strength: float = _TEST_L2_STRENGTH
) -> tuple[GraphBiRCHarmScorer, list[GraphBiRCDevelopmentEvent]]:
    groups = ("dev-a", "dev-b", "dev-c")
    events = [
        _toy_event(
            f"fit-{index:02d}",
            groups[index % len(groups)],
            low_group_confidence=0.55 + 0.05 * index,
            harmful=index % 2 == 1,
            node_id_offset=10 * index,
        )
        for index in range(6)
    ]
    scorer = fit_graph_birc_harm_scorer_from_events(
        reversed(events),
        heldout_group_ids=["cal-a", "cal-b"],
        protocol_id=_TEST_PROTOCOL_ID,
        protocol_sha256=_TEST_PROTOCOL_SHA256,
        l2_strength=l2_strength,
        minimum_development_groups=_TEST_MINIMUM_DEVELOPMENT_GROUPS,
    )
    return scorer, events


def _rehash(artifact: dict) -> None:
    payload = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    artifact["artifact_sha256"] = graph_birc_module._sha256_canonical(payload)


def test_harm_definition_uses_strict_frozen_boundaries() -> None:
    boundary = classify_graph_birc_harm(
        accuracy_gain=Fraction(-1, 100),
        balanced_accuracy_gain=Fraction(-1, 100),
        macro_f1_gain=Fraction(-1, 100),
        minimum_degree_tercile_accuracy_gain=Fraction(-1, 50),
    )
    assert boundary.is_harm is False
    assert boundary.triggered_components == ()

    just_above = classify_graph_birc_harm(
        accuracy_gain=Fraction(-99, 10_000),
        balanced_accuracy_gain=Fraction(-99, 10_000),
        macro_f1_gain=Fraction(-99, 10_000),
        minimum_degree_tercile_accuracy_gain=Fraction(-199, 10_000),
    )
    assert just_above.is_harm is False

    accuracy_harm = classify_graph_birc_harm(
        accuracy_gain=Fraction(-101, 10_000),
        balanced_accuracy_gain=Fraction(),
        macro_f1_gain=Fraction(),
        minimum_degree_tercile_accuracy_gain=Fraction(),
    )
    assert accuracy_harm.is_harm is True
    assert accuracy_harm.triggered_components == ("accuracy_gain",)

    degree_harm = classify_graph_birc_harm(
        accuracy_gain=Fraction(),
        balanced_accuracy_gain=Fraction(),
        macro_f1_gain=Fraction(),
        minimum_degree_tercile_accuracy_gain=Fraction(-201, 10_000),
    )
    assert degree_harm.triggered_components == ("minimum_degree_tercile_accuracy_gain",)

    balanced_harm = classify_graph_birc_harm(
        accuracy_gain=Fraction(),
        balanced_accuracy_gain=Fraction(-101, 10_000),
        macro_f1_gain=Fraction(),
        minimum_degree_tercile_accuracy_gain=Fraction(),
    )
    assert balanced_harm.triggered_components == ("balanced_accuracy_gain",)

    f1_harm = classify_graph_birc_harm(
        accuracy_gain=Fraction(),
        balanced_accuracy_gain=Fraction(),
        macro_f1_gain=Fraction(-101, 10_000),
        minimum_degree_tercile_accuracy_gain=Fraction(),
    )
    assert f1_harm.triggered_components == ("macro_f1_gain",)

    with pytest.raises(GraphBiRCValidationError, match="exact Fraction"):
        classify_graph_birc_harm(
            accuracy_gain=0.99 - 1.0,  # type: ignore[arg-type]
            balanced_accuracy_gain=Fraction(),
            macro_f1_gain=Fraction(),
            minimum_degree_tercile_accuracy_gain=Fraction(),
        )


def test_event_builder_preserves_exact_one_percentage_point_boundary() -> None:
    num_nodes = 400
    targets = np.arange(num_nodes, dtype=np.int64) % 2
    source = np.full((num_nodes, 2), 0.1, dtype=np.float64)
    source[np.arange(num_nodes), targets] = 0.9
    candidate = source.copy()

    # Two symmetric mistakes in the low tercile and one in each other tercile
    # yield exact global gains of -1/100 without crossing the -2/100 group floor.
    for node_index in (0, 1, 134, 267):
        candidate[node_index] = candidate[node_index, ::-1]

    event = build_graph_birc_development_event(
        event_id="exact-one-percent-boundary",
        group_id="dev-a",
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=candidate,
        targets=targets,
    )

    assert event.source_metrics.accuracy == 1.0
    assert event.candidate_metrics.accuracy == 0.99
    assert event.outcome.accuracy_gain == -0.01
    assert event.outcome.balanced_accuracy_gain == -0.01
    assert event.outcome.macro_f1_gain == -0.01
    assert event.outcome.minimum_degree_tercile_accuracy_gain > -0.02
    assert event.outcome.triggered_components == ()
    assert event.outcome.is_harm is False


def test_event_builder_preserves_exact_two_percentage_point_degree_boundary() -> None:
    num_nodes = 150
    targets = np.arange(num_nodes, dtype=np.int64) % 2
    source = np.full((num_nodes, 2), 0.1, dtype=np.float64)
    source[np.arange(num_nodes), targets] = 0.9
    candidate = source.copy()
    candidate[0] = candidate[0, ::-1]

    event = build_graph_birc_development_event(
        event_id="exact-two-percent-degree-boundary",
        group_id="dev-a",
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=candidate,
        targets=targets,
    )

    assert event.outcome.minimum_degree_tercile_accuracy_gain == -0.02
    assert event.outcome.triggered_components == ()
    assert event.outcome.is_harm is False


def test_event_builder_normalizes_valid_near_unit_probability_rows() -> None:
    targets = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    events = []
    for suffix, row_total in (("below", 0.9999992), ("above", 1.0000008)):
        probabilities = np.tile(
            np.asarray([[0.7, 0.3]], dtype=np.float64) * row_total,
            (targets.size, 1),
        )
        events.append(
            build_graph_birc_development_event(
                event_id=f"near-unit-{suffix}",
                group_id="dev-a",
                edge_index=np.empty((2, 0), dtype=np.int64),
                source_probabilities=probabilities,
                candidate_probabilities=probabilities,
                targets=targets,
            )
        )

    assert events[0].features.values == events[1].features.values
    assert (
        max(
            item.source_mean_normalized_entropy
            for event in events
            for item in event.features.diagnostics
        )
        <= 1.0
    )
    assert (
        events[0].input_sha256["source_probabilities"]
        != events[1].input_sha256["source_probabilities"]
    )


def test_event_rejects_rehashed_exact_ratio_not_supported_by_integer_counts() -> None:
    num_nodes = 400
    targets = np.arange(num_nodes, dtype=np.int64) % 2
    source = np.full((num_nodes, 2), 0.1, dtype=np.float64)
    source[np.arange(num_nodes), targets] = 0.9
    candidate = source.copy()
    for node_index in (0, 1, 134, 267):
        candidate[node_index] = candidate[node_index, ::-1]
    artifact = build_graph_birc_development_event(
        event_id="exact-ratio-forgery",
        group_id="dev-a",
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=candidate,
        targets=targets,
    ).to_dict()

    forged_candidate = Fraction.from_float(0.99)
    forged_gain = forged_candidate - 1
    artifact["candidate_metrics"]["exact_ratios"]["accuracy"] = {
        "numerator": forged_candidate.numerator,
        "denominator": forged_candidate.denominator,
    }
    artifact["outcome"]["exact_gain_ratios"]["accuracy_gain"] = {
        "numerator": forged_gain.numerator,
        "denominator": forged_gain.denominator,
    }
    artifact["outcome"]["accuracy_gain"] = float(forged_gain)
    artifact["outcome"]["is_harm"] = True
    artifact["outcome"]["triggered_components"] = ["accuracy_gain"]
    _rehash(artifact)

    with pytest.raises(GraphBiRCValidationError, match="sufficient statistics"):
        GraphBiRCDevelopmentEvent.from_dict(artifact)


def test_event_verify_against_binds_the_original_arrays() -> None:
    targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    source = np.asarray(
        [
            [0.8, 0.2],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )
    event = build_graph_birc_development_event(
        event_id="verify-original-arrays",
        group_id="dev-a",
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=source,
        targets=targets,
    )

    event.verify_against(
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=source,
        candidate_probabilities=source,
        targets=targets,
    )
    changed = source.copy()
    changed[0] = changed[0, ::-1]
    with pytest.raises(GraphBiRCValidationError, match="original arrays"):
        event.verify_against(
            edge_index=np.empty((2, 0), dtype=np.int64),
            source_probabilities=source,
            candidate_probabilities=changed,
            targets=targets,
        )


def test_event_builder_computes_exact_metrics_degree_groups_and_hash() -> None:
    event = _toy_event("event-001", "dev-a", harmful=True)

    assert event.class_labels == (0, 1)
    assert event.outcome.is_harm is True
    assert event.source_metrics.accuracy == pytest.approx(1.0)
    assert event.candidate_metrics.accuracy == pytest.approx(5.0 / 6.0)
    assert event.outcome.accuracy_gain == pytest.approx(-1.0 / 6.0)
    assert event.source_metrics.degree_tercile_accuracy == (1.0, 1.0, 1.0)
    assert event.candidate_metrics.degree_tercile_accuracy == (0.5, 1.0, 1.0)
    assert event.outcome.minimum_degree_tercile_accuracy_gain == pytest.approx(-0.5)
    assert event.outcome.triggered_components == (
        "accuracy_gain",
        "balanced_accuracy_gain",
        "macro_f1_gain",
        "minimum_degree_tercile_accuracy_gain",
    )
    assert set(event.input_sha256) == {
        "candidate_probabilities",
        "edge_index",
        "node_ids",
        "source_probabilities",
        "targets",
    }

    restored = GraphBiRCDevelopmentEvent.from_dict(event.to_dict())
    assert restored.to_dict() == event.to_dict()
    assert restored.artifact_sha256 == event.artifact_sha256

    canonical_restored = GraphBiRCDevelopmentEvent.from_dict(
        json.loads(event.canonical_json_bytes())
    )
    assert canonical_restored == event

    tampered = copy.deepcopy(event.to_dict())
    tampered["outcome"]["accuracy_gain"] += 0.1
    with pytest.raises(GraphBiRCValidationError, match="SHA-256 mismatch"):
        GraphBiRCDevelopmentEvent.from_dict(tampered)


def test_event_deserialization_rejects_rehashed_impossible_diagnostics() -> None:
    artifact = _toy_event("event-001", "dev-a").to_dict()
    artifact["features"]["diagnostics"][0]["group_mass"] = 0.9
    artifact["features"]["values"][8] = 0.9
    _rehash(artifact)

    with pytest.raises(GraphBiRCValidationError, match="group_mass"):
        GraphBiRCDevelopmentEvent.from_dict(artifact)


def test_event_deserialization_rejects_inconsistent_feature_diagnostics() -> None:
    baseline = _toy_event("diagnostic-contract", "dev-a").to_dict()

    def change_support(artifact: dict) -> None:
        diagnostic = artifact["features"]["diagnostics"][0]
        for key in ("node_indices", "node_ids", "out_degrees"):
            diagnostic[key].pop()

    def duplicate_index(artifact: dict) -> None:
        diagnostics = artifact["features"]["diagnostics"]
        diagnostics[1]["node_indices"][0] = diagnostics[0]["node_indices"][0]

    def duplicate_node_id(artifact: dict) -> None:
        diagnostics = artifact["features"]["diagnostics"]
        diagnostics[1]["node_ids"][0] = diagnostics[0]["node_ids"][0]

    def change_order(artifact: dict) -> None:
        diagnostic = artifact["features"]["diagnostics"][0]
        for key in ("node_indices", "node_ids", "out_degrees"):
            diagnostic[key][0], diagnostic[key][1] = (
                diagnostic[key][1],
                diagnostic[key][0],
            )

    def negative_degree(artifact: dict) -> None:
        artifact["features"]["diagnostics"][0]["out_degrees"][0] = -1

    def out_of_range_feature(artifact: dict) -> None:
        artifact["features"]["diagnostics"][0]["source_mean_confidence"] = 1.1
        artifact["features"]["values"][0] = 1.1

    def wrong_isolate_fraction(artifact: dict) -> None:
        artifact["features"]["diagnostics"][0]["isolate_fraction"] = 0.5
        artifact["features"]["values"][9] = 0.5

    cases = (
        (change_support, "supports"),
        (duplicate_index, "partition"),
        (duplicate_node_id, "globally unique"),
        (change_order, "ordering"),
        (negative_degree, "JSON integer"),
        (out_of_range_feature, "valid range"),
        (wrong_isolate_fraction, "isolate_fraction"),
    )
    for mutate, match in cases:
        artifact = copy.deepcopy(baseline)
        mutate(artifact)
        _rehash(artifact)
        with pytest.raises(GraphBiRCValidationError, match=match):
            GraphBiRCDevelopmentEvent.from_dict(artifact)


def test_metric_sufficient_statistics_reject_inconsistent_counts() -> None:
    valid = {
        "confusion_matrix": ((3, 0), (0, 3)),
        "degree_tercile_correct_counts": (2, 2, 2),
        "degree_tercile_supports": (2, 2, 2),
    }
    cases = (
        ({**valid, "confusion_matrix": ((3, 0, 0), (0, 3, 0))}, "square"),
        ({**valid, "confusion_matrix": ((3, 0), (0, -1))}, "non-negative"),
        (
            {**valid, "degree_tercile_correct_counts": (2, 2)},
            "exactly three",
        ),
        ({**valid, "degree_tercile_supports": (0, 3, 3)}, "positive"),
        (
            {**valid, "degree_tercile_correct_counts": (3, 2, 2)},
            "cannot exceed",
        ),
        (
            {**valid, "degree_tercile_supports": (3, 2, 2)},
            "total disagrees",
        ),
        (
            {**valid, "degree_tercile_correct_counts": (1, 2, 2)},
            "diagonal disagrees",
        ),
        (
            {**valid, "confusion_matrix": ((6, 0), (0, 0))},
            "two observed",
        ),
    )
    for arguments, match in cases:
        with pytest.raises(GraphBiRCValidationError, match=match):
            GraphBiRCMetricSufficientStatistics(**arguments)


def test_event_rejects_candidate_statistics_with_different_target_counts() -> None:
    artifact = _toy_event("target-count-contract", "dev-a").to_dict()
    candidate = artifact["metric_sufficient_statistics"]["candidate"]
    candidate["confusion_matrix"] = [[2, 0], [0, 4]]
    _rehash(artifact)

    with pytest.raises(GraphBiRCValidationError, match="joint sufficient statistics"):
        GraphBiRCDevelopmentEvent.from_dict(artifact)


def test_joint_statistics_reject_individually_valid_incompatible_marginals() -> None:
    artifact = _toy_event("joint-contract", "dev-a").to_dict()
    source = artifact["metric_sufficient_statistics"]["source"]
    candidate = artifact["metric_sufficient_statistics"]["candidate"]
    source["confusion_matrix"] = [[3, 0], [3, 0]]
    source["degree_tercile_correct_counts"] = {"low": 2, "mid": 1, "high": 0}
    candidate["confusion_matrix"] = [[0, 3], [0, 3]]
    candidate["degree_tercile_correct_counts"] = {
        "low": 2,
        "mid": 1,
        "high": 0,
    }
    _rehash(artifact)

    with pytest.raises(GraphBiRCValidationError, match="joint sufficient statistics"):
        GraphBiRCDevelopmentEvent.from_dict(artifact)


def test_joint_statistics_round_trip_and_reject_malformed_counts() -> None:
    event = _toy_event("joint-round-trip", "dev-a")
    restored = GraphBiRCJointMetricSufficientStatistics.from_dict(
        event.joint_statistics.to_dict()
    )
    assert restored == event.joint_statistics
    assert restored.source_statistics == event.source_statistics
    assert restored.candidate_statistics == event.candidate_statistics

    malformed = event.joint_statistics.to_dict()
    malformed["degree_tercile_counts"]["low"][0][0][0] = -1
    with pytest.raises(GraphBiRCValidationError, match="JSON integer"):
        GraphBiRCJointMetricSufficientStatistics.from_dict(malformed)


def test_event_builder_and_direct_extractor_share_probability_normalization() -> None:
    source = np.asarray(
        [
            [0.6000003, 0.4],
            [0.7000002, 0.3],
            [0.2, 0.8000001],
            [0.1, 0.9000002],
            [0.8000002, 0.2],
            [0.3, 0.7000003],
        ],
        dtype=np.float64,
    )
    candidate = source[:, ::-1].copy()
    edge_index = np.empty((2, 0), dtype=np.int64)
    node_ids = np.arange(6, dtype=np.int64)
    event = build_graph_birc_development_event(
        event_id="normalization-contract",
        group_id="dev-a",
        edge_index=edge_index,
        source_probabilities=source,
        candidate_probabilities=candidate,
        targets=np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64),
        node_ids=node_ids,
    )
    direct = graph_birc_module.graph_birc_features(
        edge_index,
        source,
        candidate,
        node_ids=node_ids,
    )

    assert event.features == direct


def test_scorer_fit_and_verification_use_event_derived_labels() -> None:
    scorer, events = _event_fitted_scorer()
    verify_graph_birc_harm_scorer_against_events(
        scorer,
        events,
        heldout_group_ids=["cal-a", "cal-b"],
        expected_protocol_id=_TEST_PROTOCOL_ID,
        expected_protocol_sha256=_TEST_PROTOCOL_SHA256,
        expected_l2_strength=_TEST_L2_STRENGTH,
        expected_minimum_development_groups=(_TEST_MINIMUM_DEVELOPMENT_GROUPS),
    )

    changed = list(events)
    changed[0] = _toy_event(
        "fit-00",
        "dev-a",
        low_group_confidence=0.56,
        node_id_offset=0,
    )
    with pytest.raises(GraphBiRCValidationError, match="pinned fit contract"):
        verify_graph_birc_harm_scorer_against_events(
            scorer,
            changed,
            heldout_group_ids=["cal-a", "cal-b"],
            expected_protocol_id=_TEST_PROTOCOL_ID,
            expected_protocol_sha256=_TEST_PROTOCOL_SHA256,
            expected_l2_strength=_TEST_L2_STRENGTH,
            expected_minimum_development_groups=(_TEST_MINIMUM_DEVELOPMENT_GROUPS),
        )


def test_threshold_selector_rejects_renamed_development_input_bundle() -> None:
    scorer, development_events = _event_fitted_scorer()
    calibration_events = [
        _toy_event(
            "renamed-calibration-event",
            "cal-a",
            low_group_confidence=0.55,
            harmful=False,
            node_id_offset=0,
        ),
        _toy_event(
            "unique-calibration-event",
            "cal-b",
            low_group_confidence=0.77,
            harmful=False,
            node_id_offset=100,
        ),
    ]

    assert calibration_events[0].input_sha256 == development_events[0].input_sha256
    assert calibration_events[0].event_id != development_events[0].event_id
    with pytest.raises(GraphBiRCValidationError, match="input bundles overlap"):
        select_graph_birc_threshold(
            scorer,
            calibration_events,
            scorer_fit_group_ids=[item.group_id for item in development_events],
        )


def test_threshold_selector_rejects_wrong_scorer_fit_contract() -> None:
    scorer, development_events = _event_fitted_scorer(l2_strength=2.0)
    calibration_events = [
        _toy_event("cal-00", "cal-a", node_id_offset=100),
        _toy_event("cal-01", "cal-b", node_id_offset=110),
    ]

    with pytest.raises(GraphBiRCValidationError, match="fit contract"):
        select_graph_birc_threshold(
            scorer,
            calibration_events,
            scorer_fit_group_ids=[item.group_id for item in development_events],
        )


def test_threshold_selector_rejects_unbound_generic_scorer() -> None:
    scorer = _fixed_scorer(evidence_bound=False)
    events = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b", node_id_offset=10),
    ]

    with pytest.raises(GraphBiRCValidationError, match="lacks protocol-bound"):
        select_graph_birc_threshold(
            scorer,
            events,
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        )


def test_macro_f1_uses_fixed_probability_column_class_space() -> None:
    targets = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    event = build_graph_birc_development_event(
        event_id="event-three-class",
        group_id="dev-a",
        edge_index=np.empty((2, 0), dtype=np.int64),
        source_probabilities=probabilities,
        candidate_probabilities=probabilities,
        targets=targets,
    )

    assert event.source_metrics.accuracy == 1.0
    assert event.class_labels == (0, 1, 2)
    assert event.source_metrics.macro_f1 == pytest.approx(2.0 / 3.0)
    assert event.candidate_metrics.macro_f1 == event.source_metrics.macro_f1


@pytest.mark.parametrize(
    ("targets", "match"),
    [
        (np.asarray([[0, 1, 0, 1, 0, 1]], dtype=np.int64), "one-dimensional"),
        (np.asarray([0, 1, 0, 1, 0, 2], dtype=np.int64), "class range"),
        (np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64), "two observed classes"),
    ],
)
def test_event_builder_rejects_invalid_targets(targets: np.ndarray, match: str) -> None:
    source = np.tile(np.asarray([[0.8, 0.2]]), (6, 1))
    with pytest.raises(GraphBiRCValidationError, match=match):
        build_graph_birc_development_event(
            event_id="event-001",
            group_id="dev-a",
            edge_index=np.empty((2, 0), dtype=np.int64),
            source_probabilities=source,
            candidate_probabilities=source,
            targets=targets,
        )


def test_event_builder_rejects_untrimmed_identifiers_and_noninteger_targets() -> None:
    source = np.tile(np.asarray([[0.8, 0.2]]), (6, 1))
    with pytest.raises(GraphBiRCValidationError, match="trimmed non-empty"):
        build_graph_birc_development_event(
            event_id=" event-001",
            group_id="dev-a",
            edge_index=np.empty((2, 0), dtype=np.int64),
            source_probabilities=source,
            candidate_probabilities=source,
            targets=np.asarray([0, 1, 0, 1, 0, 1]),
        )
    with pytest.raises(GraphBiRCValidationError, match="integer class labels"):
        build_graph_birc_development_event(
            event_id="event-001",
            group_id="dev-a",
            edge_index=np.empty((2, 0), dtype=np.int64),
            source_probabilities=source,
            candidate_probabilities=source,
            targets=np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )


def test_threshold_selector_maximizes_eligible_coverage_deterministically() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.51 + 0.03 * index,
            harmful=index >= 4,
        )
        for index in range(10)
    ]
    development_group_ids = ["dev-a", "dev-a", "dev-b", "dev-b", "dev-c", "dev-c"]

    first = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=development_group_ids,
        maximum_accepted_harm_risk=0.2,
        minimum_coverage=0.2,
    )
    second = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=development_group_ids,
        maximum_accepted_harm_risk=0.2,
        minimum_coverage=0.2,
    )

    expected_threshold = scorer.score(events[4].features)
    assert first == second
    assert first.decision_mode == "score_threshold"
    assert first.threshold == pytest.approx(expected_threshold)
    assert first.coverage == pytest.approx(0.5)
    assert first.accepted_count == 5
    assert first.accepted_harm_count == 1
    assert first.accepted_harm_risk == pytest.approx(0.2)
    assert first.candidate_threshold_count == 10
    assert first.constraints_satisfied is True
    assert (
        first.accept(
            scorer,
            events[4].features,
            **_threshold_accept_kwargs(first),
        )
        is True
    )
    assert (
        first.accept(
            scorer,
            events[5].features,
            **_threshold_accept_kwargs(first),
        )
        is False
    )

    restored = GraphBiRCThresholdArtifact.from_dict(first.to_dict())
    assert restored.to_dict() == first.to_dict()

    permuted = select_graph_birc_threshold(
        scorer,
        reversed(events),
        scorer_fit_group_ids=reversed(development_group_ids),
        maximum_accepted_harm_risk=0.2,
        minimum_coverage=0.2,
    )
    assert permuted.to_dict() == first.to_dict()


def test_threshold_selector_falls_back_to_reject_all() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
            harmful=True,
        )
        for index in range(6)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
    )

    assert artifact.decision_mode == "reject_all"
    assert artifact.threshold is None
    assert artifact.coverage == 0.0
    assert artifact.accepted_count == 0
    assert artifact.accepted_harm_count == 0
    assert artifact.accepted_harm_risk is None
    assert artifact.constraints_satisfied is False
    assert (
        artifact.accept(
            scorer,
            events[0].features,
            **_threshold_accept_kwargs(artifact),
        )
        is False
    )


def test_reject_all_still_validates_deployment_features() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
            harmful=True,
        )
        for index in range(6)
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
    )

    assert artifact.decision_mode == "reject_all"
    with pytest.raises(
        GraphBiRCValidationError, match="feature vector length mismatch"
    ):
        artifact.accept(
            scorer,
            [0.0],
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
            **_threshold_accept_kwargs(artifact),
        )


def test_threshold_selector_accepts_exact_coverage_boundary() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
            harmful=index > 0,
        )
        for index in range(5)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=0.0,
        minimum_coverage=0.2,
    )

    assert artifact.decision_mode == "score_threshold"
    assert artifact.accepted_count == 1
    assert artifact.coverage == 0.2


def test_threshold_selector_rejects_one_count_below_coverage() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.54 + 0.05 * index,
            harmful=index > 0,
        )
        for index in range(6)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=0.0,
        minimum_coverage=0.2,
    )

    assert artifact.decision_mode == "reject_all"


def test_threshold_selector_rejects_one_harm_over_risk_limit() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
            harmful=index >= 3,
        )
        for index in range(5)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=0.2,
        minimum_coverage=1.0,
    )

    assert artifact.decision_mode == "reject_all"


def test_threshold_selector_enumerates_nonmonotonic_prefix_risk() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
            harmful=index == 0,
        )
        for index in range(5)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=0.2,
        minimum_coverage=1.0,
    )

    assert artifact.decision_mode == "score_threshold"
    assert artifact.accepted_count == 5
    assert artifact.accepted_harm_count == 1
    assert artifact.accepted_harm_risk == 0.2


def test_threshold_selector_selects_full_coverage_when_all_events_are_safe() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
        )
        for index in range(5)
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
    )

    assert artifact.decision_mode == "score_threshold"
    assert artifact.coverage == 1.0
    assert artifact.accepted_count == len(events)
    assert artifact.accepted_harm_count == 0


def test_threshold_selector_treats_tied_scores_as_one_atomic_threshold() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a", low_group_confidence=0.6),
        _toy_event(
            "cal-01",
            "cal-b",
            low_group_confidence=0.6,
            harmful=True,
            node_id_offset=10,
        ),
        _toy_event(
            "cal-02",
            "cal-a",
            low_group_confidence=0.8,
            harmful=True,
            node_id_offset=20,
        ),
        _toy_event(
            "cal-03",
            "cal-b",
            low_group_confidence=0.8,
            harmful=True,
            node_id_offset=30,
        ),
    ]

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
    )

    assert artifact.candidate_threshold_count == 2
    assert artifact.decision_mode == "reject_all"


def test_threshold_selector_binds_split_event_ids_and_scorer() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b", harmful=True),
    ]

    with pytest.raises(GraphBiRCValidationError, match="split digest"):
        select_graph_birc_threshold(
            scorer,
            events,
            scorer_fit_group_ids=["dev-a", "dev-b", "wrong-dev"],
        )

    with pytest.raises(GraphBiRCValidationError, match="event IDs must be unique"):
        select_graph_birc_threshold(
            scorer,
            [events[0], events[0]],
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        )

    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    )
    with pytest.raises(GraphBiRCValidationError, match="scorer SHA-256 mismatch"):
        artifact.accept(
            _fixed_scorer(intercept=0.1),
            events[0].features,
            **_threshold_accept_kwargs(artifact),
        )


def test_threshold_artifact_rejects_tampering_even_after_rehash() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b", harmful=True),
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    ).to_dict()

    tampered = copy.deepcopy(artifact)
    tampered["selection"]["coverage"] = 0.25
    with pytest.raises(GraphBiRCValidationError, match="SHA-256 mismatch"):
        GraphBiRCThresholdArtifact.from_dict(tampered)

    impossible = copy.deepcopy(artifact)
    impossible["selection"]["accepted_count"] = 99
    _rehash(impossible)
    with pytest.raises(GraphBiRCValidationError, match="accepted_count"):
        GraphBiRCThresholdArtifact.from_dict(impossible)


def test_threshold_accept_rejects_synchronized_rehash_against_pinned_digest() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b", harmful=True),
    ]
    original = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    ).to_dict()
    externally_pinned_digest = original["artifact_sha256"]

    forged = copy.deepcopy(original)
    forged["calibration"]["group_split_sha256"] = "f" * 64
    _rehash(forged)
    restored = GraphBiRCThresholdArtifact.from_dict(forged)

    # A valid self-hash proves internal integrity, not external authentication.
    assert restored.artifact_sha256 == forged["artifact_sha256"]
    assert restored.artifact_sha256 != externally_pinned_digest
    with pytest.raises(GraphBiRCValidationError, match="externally pinned digest"):
        restored.accept(
            scorer,
            events[0].features,
            **_threshold_accept_kwargs(
                restored,
                expected_threshold_artifact_sha256=externally_pinned_digest,
            ),
        )


@pytest.mark.parametrize(
    "constraint_override",
    [
        {"expected_maximum_accepted_harm_risk": 0.2},
        {"expected_minimum_coverage": 0.3},
    ],
)
def test_threshold_accept_rejects_external_constraint_mismatch(
    constraint_override: dict[str, float],
) -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b", harmful=True),
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
        minimum_coverage=0.2,
    )

    with pytest.raises(GraphBiRCValidationError, match="threshold constraints"):
        artifact.accept(
            scorer,
            events[0].features,
            **_threshold_accept_kwargs(artifact, **constraint_override),
        )


def test_threshold_selector_rejects_duplicate_input_bundles() -> None:
    scorer = _fixed_scorer()
    duplicate_inputs = [
        _toy_event("cal-00", "cal-a"),
        _toy_event("cal-01", "cal-b"),
    ]

    with pytest.raises(GraphBiRCValidationError, match="unique input bundles"):
        select_graph_birc_threshold(
            scorer,
            duplicate_inputs,
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        )


def test_threshold_artifact_rejects_rehashed_suboptimal_selection() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event(
            f"cal-{index:02d}",
            "cal-a" if index % 2 == 0 else "cal-b",
            low_group_confidence=0.55 + 0.05 * index,
        )
        for index in range(5)
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
    ).to_dict()
    artifact["selection"]["threshold"] = artifact["calibration"]["records"][0]["score"]
    artifact["selection"]["coverage"] = 0.2
    artifact["selection"]["accepted_count"] = 1
    _rehash(artifact)

    with pytest.raises(GraphBiRCValidationError, match="reproducible optimum"):
        GraphBiRCThresholdArtifact.from_dict(artifact)


def test_threshold_verify_against_rebuilds_records_and_selection() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a", low_group_confidence=0.7),
        _toy_event(
            "cal-01",
            "cal-b",
            low_group_confidence=0.8,
            harmful=True,
        ),
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    )
    artifact.verify_against(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        expected_maximum_accepted_harm_risk=1.0,
        expected_minimum_coverage=0.2,
        **_scorer_deployment_kwargs(),
    )

    changed_events = [
        events[0],
        _toy_event(
            "cal-01",
            "cal-b",
            low_group_confidence=0.81,
            harmful=True,
        ),
    ]
    with pytest.raises(GraphBiRCValidationError, match="supplied scorer or events"):
        artifact.verify_against(
            scorer,
            changed_events,
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
            expected_maximum_accepted_harm_risk=1.0,
            expected_minimum_coverage=0.2,
            **_scorer_deployment_kwargs(),
        )


def test_threshold_accept_rejects_forged_group_split_digest() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a", low_group_confidence=0.7),
        _toy_event(
            "cal-01",
            "cal-b",
            low_group_confidence=0.8,
            harmful=True,
        ),
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    ).to_dict()
    artifact["calibration"]["group_split_sha256"] = "f" * 64
    _rehash(artifact)
    restored = GraphBiRCThresholdArtifact.from_dict(artifact)

    with pytest.raises(GraphBiRCValidationError, match="group-split"):
        restored.accept(
            scorer,
            events[0].features,
            **_threshold_accept_kwargs(restored),
        )


def test_threshold_verification_rejects_external_protocol_mismatch() -> None:
    scorer = _fixed_scorer()
    events = [
        _toy_event("cal-00", "cal-a", low_group_confidence=0.7),
        _toy_event("cal-01", "cal-b", low_group_confidence=0.8, harmful=True),
    ]
    artifact = select_graph_birc_threshold(
        scorer,
        events,
        scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
        maximum_accepted_harm_risk=1.0,
    )

    with pytest.raises(GraphBiRCValidationError, match="protocol binding"):
        artifact.verify_against(
            scorer,
            events,
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
            expected_maximum_accepted_harm_risk=1.0,
            expected_minimum_coverage=0.2,
            **_scorer_deployment_kwargs(expected_protocol_sha256="b" * 64),
        )
    with pytest.raises(GraphBiRCValidationError, match="pinned protocol"):
        artifact.verify_against(
            scorer,
            events,
            scorer_fit_group_ids=["dev-a", "dev-b", "dev-c"],
            expected_maximum_accepted_harm_risk=0.2,
            expected_minimum_coverage=0.2,
            **_scorer_deployment_kwargs(),
        )
