import copy

import numpy as np
import pytest

import graph_birc as graph_birc_module
from graph_birc import (
    GRAPH_BIRC_FEATURE_SCHEMA,
    GRAPH_BIRC_GROUP_ORDER,
    GraphBiRCFeatures,
    GraphBiRCHarmScorer,
    GraphBiRCValidationError,
    fit_graph_birc_harm_scorer,
    graph_birc_features,
)


def _binary_probabilities(confidences):
    confidences = np.asarray(confidences, dtype=float)
    return np.column_stack((confidences, 1.0 - confidences))


def test_degree_terciles_break_ties_by_node_id_deterministically():
    edge_index = np.asarray([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=np.int64)
    node_ids = np.asarray([30, 10, 20, 60, 40, 50], dtype=np.int64)
    source = _binary_probabilities([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    first = graph_birc_features(edge_index, source, source, node_ids=node_ids)
    second = graph_birc_features(edge_index, source, source, node_ids=node_ids)

    assert first == second
    assert tuple(item.group for item in first.diagnostics) == GRAPH_BIRC_GROUP_ORDER
    assert tuple(item.node_ids for item in first.diagnostics) == (
        (10, 20),
        (30, 40),
        (50, 60),
    )
    assert tuple(item.node_indices for item in first.diagnostics) == (
        (1, 2),
        (0, 4),
        (5, 3),
    )
    assert first.feature_schema == GRAPH_BIRC_FEATURE_SCHEMA


def test_directional_confidence_changes_do_not_cancel():
    edge_index = np.empty((2, 0), dtype=np.int64)
    source = _binary_probabilities([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    candidate = _binary_probabilities([0.9, 0.5, 0.7, 0.7, 0.7, 0.7])

    result = graph_birc_features(edge_index, source, candidate)
    low = result.diagnostics[0]

    assert np.mean(
        np.max(candidate[:2], axis=1) - np.max(source[:2], axis=1)
    ) == pytest.approx(0.0)
    assert low.positive_confidence_change_mean == pytest.approx(0.1)
    assert low.negative_confidence_change_mean == pytest.approx(-0.1)
    mapping = result.as_mapping()
    assert mapping["low.positive_confidence_change_mean"] == pytest.approx(0.1)
    assert mapping["low.negative_confidence_change_mean"] == pytest.approx(-0.1)


def test_directional_neighborhood_changes_do_not_cancel():
    edge_index = np.asarray(
        [
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [2, 3, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    source = _binary_probabilities([0.8, 0.8, 0.2, 0.8, 0.8, 0.8])
    candidate = source.copy()
    candidate[2] = [0.8, 0.2]
    candidate[3] = [0.2, 0.8]

    result = graph_birc_features(edge_index, source, candidate)
    low = result.diagnostics[0]

    assert low.node_indices == (0, 1)
    assert low.positive_neighborhood_agreement_change_mean == pytest.approx(0.5)
    assert low.negative_neighborhood_agreement_change_mean == pytest.approx(-0.5)


def test_isolates_have_zero_agreement_change_and_are_reported():
    edge_index = np.asarray([[2, 3, 4], [3, 2, 5]], dtype=np.int64)
    source = _binary_probabilities([0.8, 0.7, 0.9, 0.9, 0.8, 0.8])
    candidate = source.copy()
    candidate[0] = [0.2, 0.8]
    candidate[1] = [0.3, 0.7]

    assert not np.array_equal(source, candidate)

    result = graph_birc_features(edge_index, source, candidate)
    low = result.diagnostics[0]

    assert low.node_indices == (0, 1)
    assert low.isolate_fraction == pytest.approx(1.0)
    assert low.positive_neighborhood_agreement_change_mean == pytest.approx(0.0)
    assert low.negative_neighborhood_agreement_change_mean == pytest.approx(0.0)
    assert sum(item.group_mass for item in result.diagnostics) == pytest.approx(1.0)


def test_probability_and_graph_validation_fail_closed():
    edge_index = np.empty((2, 0), dtype=np.int64)
    valid = _binary_probabilities([0.8, 0.7, 0.6])

    nonfinite = valid.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(GraphBiRCValidationError, match="non-finite"):
        graph_birc_features(edge_index, nonfinite, valid)

    not_normalized = valid.copy()
    not_normalized[0] = [0.4, 0.4]
    with pytest.raises(GraphBiRCValidationError, match="sum to one"):
        graph_birc_features(edge_index, valid, not_normalized)

    with pytest.raises(GraphBiRCValidationError, match="out-of-range"):
        graph_birc_features(np.asarray([[0], [3]]), valid, valid)

    with pytest.raises(GraphBiRCValidationError, match="unique"):
        graph_birc_features(edge_index, valid, valid, node_ids=[1, 1, 2])

    too_large = np.asarray([0, 1, 2**63], dtype=np.uint64)
    with pytest.raises(GraphBiRCValidationError, match="signed 64-bit"):
        graph_birc_features(edge_index, valid, valid, node_ids=too_large)


def _development_matrix():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(8, len(GRAPH_BIRC_FEATURE_SCHEMA)))
    matrix[:, 0] = np.linspace(-2.0, 2.0, len(matrix))
    labels = (matrix[:, 0] > 0.0).astype(int)
    groups = [f"condition-{index // 2}" for index in range(len(matrix))]
    return matrix, labels, groups


def test_fit_rejects_nonfinite_schema_and_insufficient_groups():
    matrix, labels, groups = _development_matrix()
    heldout_groups = ["heldout-a", "heldout-b"]

    with pytest.raises(GraphBiRCValidationError, match="explicit feature_schema"):
        fit_graph_birc_harm_scorer(
            matrix, labels, groups, heldout_group_ids=heldout_groups
        )

    wrong_schema = list(GRAPH_BIRC_FEATURE_SCHEMA)
    wrong_schema[-1] = "high.wrong_field"
    with pytest.raises(GraphBiRCValidationError, match="schema mismatch"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            groups,
            heldout_group_ids=heldout_groups,
            feature_schema=wrong_schema,
        )

    nonfinite = matrix.copy()
    nonfinite[0, 3] = np.inf
    with pytest.raises(GraphBiRCValidationError, match="finite"):
        fit_graph_birc_harm_scorer(
            nonfinite,
            labels,
            groups,
            heldout_group_ids=heldout_groups,
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    with pytest.raises(GraphBiRCValidationError, match="at least 3 development groups"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            ["a"] * 4 + ["b"] * 4,
            heldout_group_ids=heldout_groups,
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    with pytest.raises(GraphBiRCValidationError, match="trimmed non-empty"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            groups[:-1] + ["   "],
            heldout_group_ids=heldout_groups,
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    with pytest.raises(GraphBiRCValidationError, match="sequences of strings"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            "abcdefgh",
            heldout_group_ids=heldout_groups,
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    with pytest.raises(GraphBiRCValidationError, match="sequences of strings"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            groups,
            heldout_group_ids="WXYZ",
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    with pytest.raises(GraphBiRCValidationError, match="overlap"):
        fit_graph_birc_harm_scorer(
            matrix,
            labels,
            groups,
            heldout_group_ids=[groups[0], "heldout-b"],
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )


def test_fit_score_and_serialization_are_deterministic_and_tamper_evident():
    matrix, labels, groups = _development_matrix()
    keyword_args = {
        "heldout_group_ids": ["heldout-a", "heldout-b"],
        "feature_schema": GRAPH_BIRC_FEATURE_SCHEMA,
        "l2_strength": 2.0,
    }

    first = fit_graph_birc_harm_scorer(matrix, labels, groups, **keyword_args)
    second = fit_graph_birc_harm_scorer(matrix, labels, groups, **keyword_args)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.artifact_sha256 == first.to_dict()["artifact_sha256"]

    restored = GraphBiRCHarmScorer.from_dict(first.to_dict())
    assert restored.to_dict() == first.to_dict()
    assert (
        restored.to_dict()["training"]["data_scope"]
        == "caller_declared_labeled_development_events"
    )
    low_risk_score = restored.score(matrix[0], feature_schema=GRAPH_BIRC_FEATURE_SCHEMA)
    high_risk_score = restored.score(
        matrix[-1], feature_schema=GRAPH_BIRC_FEATURE_SCHEMA
    )
    assert low_risk_score < high_risk_score
    threshold = 0.5 * (low_risk_score + high_risk_score)
    assert restored.accept(
        matrix[0], threshold, feature_schema=GRAPH_BIRC_FEATURE_SCHEMA
    )
    assert not restored.accept(
        matrix[-1], threshold, feature_schema=GRAPH_BIRC_FEATURE_SCHEMA
    )
    with pytest.raises(GraphBiRCValidationError, match="feature schema"):
        restored.accept(matrix[0], threshold, feature_schema=("wrong",))
    invalid = matrix[0].copy()
    invalid[0] = np.nan
    with pytest.raises(GraphBiRCValidationError, match="finite"):
        restored.accept(invalid, threshold, feature_schema=GRAPH_BIRC_FEATURE_SCHEMA)
    with pytest.raises(GraphBiRCValidationError, match="must be numeric"):
        restored.accept(
            [[1.0], [2.0, 3.0]],
            threshold,
            feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        )

    tampered = copy.deepcopy(first.to_dict())
    tampered["model"]["intercept"] += 0.1
    with pytest.raises(GraphBiRCValidationError, match="SHA-256 mismatch"):
        GraphBiRCHarmScorer.from_dict(tampered)


def _rehash_artifact(artifact):
    payload = {
        key: value for key, value in artifact.items() if key != "artifact_sha256"
    }
    artifact["artifact_sha256"] = graph_birc_module._sha256_canonical(payload)


def test_deserialization_rejects_type_drift_and_impossible_metadata():
    matrix, labels, groups = _development_matrix()
    scorer = fit_graph_birc_harm_scorer(
        matrix,
        labels,
        groups,
        heldout_group_ids=["heldout-a"],
        feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
        l2_strength=2.0,
    )

    integer_l2 = copy.deepcopy(scorer.to_dict())
    integer_l2["model"]["l2_strength"] = 2
    _rehash_artifact(integer_l2)
    with pytest.raises(GraphBiRCValidationError, match="JSON float"):
        GraphBiRCHarmScorer.from_dict(integer_l2)

    boolean_ddof = copy.deepcopy(scorer.to_dict())
    boolean_ddof["standardization"]["ddof"] = False
    _rehash_artifact(boolean_ddof)
    with pytest.raises(GraphBiRCValidationError, match="standardization convention"):
        GraphBiRCHarmScorer.from_dict(boolean_ddof)

    impossible_counts = copy.deepcopy(scorer.to_dict())
    impossible_counts["training"]["development_event_count"] = 1
    _rehash_artifact(impossible_counts)
    with pytest.raises(
        GraphBiRCValidationError, match="at least development_group_count"
    ):
        GraphBiRCHarmScorer.from_dict(impossible_counts)


def test_public_artifacts_copy_mutable_sequences():
    matrix, labels, groups = _development_matrix()
    scorer = fit_graph_birc_harm_scorer(
        matrix,
        labels,
        groups,
        heldout_group_ids=["heldout-a"],
        feature_schema=GRAPH_BIRC_FEATURE_SCHEMA,
    )
    mutable_mean = list(scorer.feature_mean)
    mutable_coefficients = list(scorer.coefficients)
    copied = GraphBiRCHarmScorer(
        feature_mean=mutable_mean,
        feature_scale=list(scorer.feature_scale),
        coefficients=mutable_coefficients,
        intercept=scorer.intercept,
        l2_strength=scorer.l2_strength,
        development_event_count=scorer.development_event_count,
        development_group_count=scorer.development_group_count,
        heldout_group_count=scorer.heldout_group_count,
        development_data_sha256=scorer.development_data_sha256,
        group_split_sha256=scorer.group_split_sha256,
        optimizer_iterations=scorer.optimizer_iterations,
    )
    original_digest = copied.artifact_sha256
    mutable_mean[0] += 10.0
    mutable_coefficients[0] += 10.0
    assert copied.artifact_sha256 == original_digest

    extracted = graph_birc_features(
        np.empty((2, 0), dtype=np.int64),
        _binary_probabilities([0.8, 0.7, 0.6]),
        _binary_probabilities([0.8, 0.7, 0.6]),
    )
    mutable_values = list(extracted.values)
    mutable_diagnostics = list(extracted.diagnostics)
    copied_features = GraphBiRCFeatures(mutable_values, mutable_diagnostics)
    mutable_values[0] += 1.0
    mutable_diagnostics.clear()
    assert copied_features == extracted
