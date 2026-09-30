"""Auditable development labels and threshold selection for Graph-BiRC.

This module is deliberately separate from target-domain execution.  Labels are
accepted only for caller-declared development events, where they define the
frozen multi-metric harm outcome and a held-out development threshold.  The
code verifies the scorer's declared group split but cannot authenticate the
real-world provenance or chronology of the arrays supplied by the caller.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__:
    from .graph_birc import (
        GRAPH_BIRC_FEATURE_SCHEMA,
        GRAPH_BIRC_FEATURE_SCHEMA_VERSION,
        GRAPH_BIRC_GROUP_FIELDS,
        GRAPH_BIRC_GROUP_ORDER,
        GraphBiRCFeatures,
        GraphBiRCGroupDiagnostics,
        GraphBiRCHarmScorer,
        GraphBiRCValidationError,
        _canonical_json_bytes,
        _finite_float,
        _sha256_canonical,
        _validate_digest,
        _validate_edge_index,
        _validate_node_ids,
        _validate_probabilities,
        fit_graph_birc_harm_scorer,
        graph_birc_features,
    )
else:
    from graph_birc import (  # type: ignore[import-not-found, no-redef]
        GRAPH_BIRC_FEATURE_SCHEMA,
        GRAPH_BIRC_FEATURE_SCHEMA_VERSION,
        GRAPH_BIRC_GROUP_FIELDS,
        GRAPH_BIRC_GROUP_ORDER,
        GraphBiRCFeatures,
        GraphBiRCGroupDiagnostics,
        GraphBiRCHarmScorer,
        GraphBiRCValidationError,
        _canonical_json_bytes,
        _finite_float,
        _sha256_canonical,
        _validate_digest,
        _validate_edge_index,
        _validate_node_ids,
        _validate_probabilities,
        fit_graph_birc_harm_scorer,
        graph_birc_features,
    )


GRAPH_BIRC_DEVELOPMENT_EVENT_SCHEMA_VERSION = "graph_birc_development_event.v4"
GRAPH_BIRC_THRESHOLD_SCHEMA_VERSION = "graph_birc_threshold.v4"

ACCURACY_HARM_FLOOR = -0.01
BALANCED_ACCURACY_HARM_FLOOR = -0.01
MACRO_F1_HARM_FLOOR = -0.01
DEGREE_TERCILE_ACCURACY_HARM_FLOOR = -0.02

_INPUT_NAMES = (
    "candidate_probabilities",
    "edge_index",
    "node_ids",
    "source_probabilities",
    "targets",
)
_HARM_COMPONENT_ORDER = (
    "accuracy_gain",
    "balanced_accuracy_gain",
    "macro_f1_gain",
    "minimum_degree_tercile_accuracy_gain",
)
_HARM_FLOORS = {
    "accuracy_gain": ACCURACY_HARM_FLOOR,
    "balanced_accuracy_gain": BALANCED_ACCURACY_HARM_FLOOR,
    "macro_f1_gain": MACRO_F1_HARM_FLOOR,
    "minimum_degree_tercile_accuracy_gain": DEGREE_TERCILE_ACCURACY_HARM_FLOOR,
}
_HARM_FLOOR_FRACTIONS = {
    "accuracy_gain": Fraction(-1, 100),
    "balanced_accuracy_gain": Fraction(-1, 100),
    "macro_f1_gain": Fraction(-1, 100),
    "minimum_degree_tercile_accuracy_gain": Fraction(-1, 50),
}
_DEVELOPMENT_SCOPE = "caller_declared_labeled_development_event"
_SCOPE_CLAIM_STATUS = "caller_declared_not_independently_verified"
_METRIC_CONVENTION = {
    "probability_preprocessing": (
        "after row-sum validation, source and candidate rows are independently "
        "renormalized to sum exactly to one in float64; input hashes bind the "
        "validated pre-renormalization float64 arrays"
    ),
    "accuracy": "fraction of all nodes classified correctly",
    "balanced_accuracy": "unweighted mean recall over classes observed in targets",
    "macro_f1": (
        "unweighted mean one-vs-rest F1 over the fixed probability-column class "
        "space; zero denominator maps to zero"
    ),
    "degree_terciles": (
        "same deterministic outgoing-degree and node-ID groups carried by "
        "graph_birc_features.v1"
    ),
}
_THRESHOLD_RULE = "accept iff scorer score <= threshold"
_THRESHOLD_SELECTION_ORDER = (
    "maximize_coverage",
    "minimize_accepted_harm_risk",
    "choose_smaller_threshold",
)


def _identifier(value: Any, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GraphBiRCValidationError(f"{context} must be a trimmed non-empty string")
    return value


def _json_float(value: Any, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GraphBiRCValidationError(f"{context} must be a finite JSON float")
    return value


def _json_int(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        range_description = f"at least {minimum}"
        if maximum is not None:
            range_description = f"between {minimum} and {maximum}"
        raise GraphBiRCValidationError(
            f"{context} must be a JSON integer {range_description}"
        )
    return value


def _unit_interval(value: Any, context: str) -> float:
    checked = _finite_float(value, context)
    if checked < 0.0 or checked > 1.0:
        raise GraphBiRCValidationError(f"{context} must lie in [0, 1]")
    return checked


def _unit_fraction(value: float) -> Fraction:
    """Interpret the serialized decimal constraint exactly for count comparisons."""

    return Fraction(str(value))


def _ratio_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _ratio_from_dict(value: Any, context: str) -> Fraction:
    if type(value) is not dict or set(value) != {"numerator", "denominator"}:
        raise GraphBiRCValidationError(f"{context} must be an exact-ratio object")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int:
        raise GraphBiRCValidationError(f"{context}.numerator must be a JSON integer")
    denominator = _json_int(denominator, f"{context}.denominator", minimum=1)
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise GraphBiRCValidationError(f"{context} must be in lowest terms")
    return result


def _array_sha256(value: np.ndarray) -> str:
    """Hash normalized shape, dtype, and C-order bytes without serializing values."""

    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_targets(value: Any, *, num_nodes: int, num_classes: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError(
            "targets must contain integer class labels"
        ) from exc
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise GraphBiRCValidationError("targets must contain integer class labels")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise GraphBiRCValidationError("targets exceed the signed 64-bit range")
    targets = np.asarray(raw, dtype=np.int64)
    if targets.ndim != 1:
        raise GraphBiRCValidationError("targets must be one-dimensional")
    if len(targets) != num_nodes:
        raise GraphBiRCValidationError("targets must contain one label per node")
    if int(targets.min()) < 0 or int(targets.max()) >= num_classes:
        raise GraphBiRCValidationError(
            "targets contain a label outside the class range"
        )
    if len(np.unique(targets)) < 2:
        raise GraphBiRCValidationError(
            "targets must contain at least two observed classes"
        )
    return targets


def _metric_fractions_from_counts(
    confusion_matrix: tuple[tuple[int, ...], ...],
    degree_correct_counts: tuple[int, ...],
    degree_supports: tuple[int, ...],
) -> tuple[Fraction, ...]:
    total = sum(sum(row) for row in confusion_matrix)
    accuracy = Fraction(
        sum(confusion_matrix[index][index] for index in range(len(confusion_matrix))),
        total,
    )

    recalls = [
        Fraction(row[class_index], sum(row))
        for class_index, row in enumerate(confusion_matrix)
        if sum(row) > 0
    ]
    balanced_accuracy = sum(recalls, start=Fraction()) / len(recalls)

    f1_values: list[Fraction] = []
    for class_index, row in enumerate(confusion_matrix):
        true_positive = row[class_index]
        false_positive = sum(
            confusion_matrix[other][class_index]
            for other in range(len(confusion_matrix))
            if other != class_index
        )
        false_negative = sum(
            count for predicted, count in enumerate(row) if predicted != class_index
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(
            Fraction() if denominator == 0 else Fraction(2 * true_positive, denominator)
        )
    macro_f1 = sum(f1_values, start=Fraction()) / len(f1_values)
    degree_accuracy = tuple(
        Fraction(correct, support)
        for correct, support in zip(degree_correct_counts, degree_supports, strict=True)
    )
    return accuracy, balanced_accuracy, macro_f1, *degree_accuracy


@dataclass(frozen=True, slots=True)
class GraphBiRCPredictiveMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    degree_tercile_accuracy: tuple[float, ...]
    _exact_values: tuple[Fraction, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        try:
            degree_values = tuple(
                _unit_interval(value, "degree_tercile_accuracy")
                for value in self.degree_tercile_accuracy
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "degree_tercile_accuracy must be iterable"
            ) from exc
        if len(degree_values) != len(GRAPH_BIRC_GROUP_ORDER):
            raise GraphBiRCValidationError(
                "degree_tercile_accuracy must contain exactly three values"
            )
        object.__setattr__(self, "degree_tercile_accuracy", degree_values)
        rendered_values = (
            self.accuracy,
            self.balanced_accuracy,
            self.macro_f1,
            *degree_values,
        )
        if self._exact_values is None:
            exact_values = tuple(Fraction(str(value)) for value in rendered_values)
        else:
            try:
                exact_values = tuple(self._exact_values)
            except TypeError as exc:
                raise GraphBiRCValidationError(
                    "predictive metric exact values must be iterable"
                ) from exc
            if len(exact_values) != len(rendered_values) or not all(
                isinstance(value, Fraction) for value in exact_values
            ):
                raise GraphBiRCValidationError(
                    "predictive metric exact values do not match schema"
                )
        if any(value < 0 or value > 1 for value in exact_values):
            raise GraphBiRCValidationError(
                "predictive metric exact values must lie in [0, 1]"
            )
        if tuple(float(value) for value in exact_values) != rendered_values:
            raise GraphBiRCValidationError(
                "predictive metric floats disagree with exact ratios"
            )
        object.__setattr__(self, "_exact_values", exact_values)

    @property
    def exact_values(self) -> tuple[Fraction, ...]:
        if self._exact_values is None:  # pragma: no cover - guarded in __post_init__
            raise GraphBiRCValidationError("predictive metric ratios are unavailable")
        return self._exact_values

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "macro_f1": float(self.macro_f1),
            "degree_tercile_accuracy": {
                group: float(value)
                for group, value in zip(
                    GRAPH_BIRC_GROUP_ORDER,
                    self.degree_tercile_accuracy,
                    strict=True,
                )
            },
            "exact_ratios": {
                "accuracy": _ratio_payload(self.exact_values[0]),
                "balanced_accuracy": _ratio_payload(self.exact_values[1]),
                "macro_f1": _ratio_payload(self.exact_values[2]),
                "degree_tercile_accuracy": {
                    group: _ratio_payload(value)
                    for group, value in zip(
                        GRAPH_BIRC_GROUP_ORDER,
                        self.exact_values[3:],
                        strict=True,
                    )
                },
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphBiRCPredictiveMetrics":
        if type(value) is not dict or set(value) != {
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "degree_tercile_accuracy",
            "exact_ratios",
        }:
            raise GraphBiRCValidationError("predictive metric keys do not match schema")
        degree = value["degree_tercile_accuracy"]
        if type(degree) is not dict or set(degree) != set(GRAPH_BIRC_GROUP_ORDER):
            raise GraphBiRCValidationError(
                "degree-tercile metric keys do not match schema"
            )
        exact = value["exact_ratios"]
        if type(exact) is not dict or set(exact) != {
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "degree_tercile_accuracy",
        }:
            raise GraphBiRCValidationError(
                "predictive metric ratio keys do not match schema"
            )
        exact_degree = exact["degree_tercile_accuracy"]
        if type(exact_degree) is not dict or set(exact_degree) != set(
            GRAPH_BIRC_GROUP_ORDER
        ):
            raise GraphBiRCValidationError(
                "degree-tercile metric ratio keys do not match schema"
            )
        return cls(
            accuracy=_json_float(value["accuracy"], "accuracy"),
            balanced_accuracy=_json_float(
                value["balanced_accuracy"], "balanced_accuracy"
            ),
            macro_f1=_json_float(value["macro_f1"], "macro_f1"),
            degree_tercile_accuracy=tuple(
                _json_float(degree[group], f"degree_tercile_accuracy.{group}")
                for group in GRAPH_BIRC_GROUP_ORDER
            ),
            _exact_values=(
                _ratio_from_dict(exact["accuracy"], "exact_ratios.accuracy"),
                _ratio_from_dict(
                    exact["balanced_accuracy"], "exact_ratios.balanced_accuracy"
                ),
                _ratio_from_dict(exact["macro_f1"], "exact_ratios.macro_f1"),
                *(
                    _ratio_from_dict(
                        exact_degree[group],
                        f"exact_ratios.degree_tercile_accuracy.{group}",
                    )
                    for group in GRAPH_BIRC_GROUP_ORDER
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphBiRCMetricSufficientStatistics:
    """Integer counts that uniquely determine every serialized event metric."""

    confusion_matrix: tuple[tuple[int, ...], ...]
    degree_tercile_correct_counts: tuple[int, ...]
    degree_tercile_supports: tuple[int, ...]

    def __post_init__(self) -> None:
        def checked_count(value: Any, context: str) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise GraphBiRCValidationError(f"{context} must be an integer")
            result = int(value)
            if result < 0:
                raise GraphBiRCValidationError(f"{context} must be non-negative")
            return result

        try:
            matrix = tuple(
                tuple(checked_count(count, "confusion_matrix count") for count in row)
                for row in self.confusion_matrix
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "confusion_matrix must be a square iterable"
            ) from exc
        if len(matrix) < 2 or any(len(row) != len(matrix) for row in matrix):
            raise GraphBiRCValidationError(
                "confusion_matrix must be square with at least two classes"
            )
        object.__setattr__(self, "confusion_matrix", matrix)

        try:
            correct_counts = tuple(
                checked_count(count, "degree-tercile correct count")
                for count in self.degree_tercile_correct_counts
            )
            supports = tuple(
                checked_count(count, "degree-tercile support")
                for count in self.degree_tercile_supports
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "degree-tercile counts and supports must be iterable"
            ) from exc
        if len(correct_counts) != len(GRAPH_BIRC_GROUP_ORDER) or len(supports) != len(
            GRAPH_BIRC_GROUP_ORDER
        ):
            raise GraphBiRCValidationError(
                "degree-tercile statistics must contain exactly three values"
            )
        if any(support < 1 for support in supports):
            raise GraphBiRCValidationError(
                "degree-tercile supports must all be positive"
            )
        if any(
            correct > support
            for correct, support in zip(correct_counts, supports, strict=True)
        ):
            raise GraphBiRCValidationError(
                "degree-tercile correct counts cannot exceed their supports"
            )
        total = sum(sum(row) for row in matrix)
        if total != sum(supports):
            raise GraphBiRCValidationError(
                "confusion-matrix total disagrees with degree-tercile supports"
            )
        global_correct = sum(matrix[index][index] for index in range(len(matrix)))
        if global_correct != sum(correct_counts):
            raise GraphBiRCValidationError(
                "confusion-matrix diagonal disagrees with degree-tercile correct counts"
            )
        if sum(sum(row) > 0 for row in matrix) < 2:
            raise GraphBiRCValidationError(
                "confusion_matrix must contain at least two observed target classes"
            )
        object.__setattr__(self, "degree_tercile_correct_counts", correct_counts)
        object.__setattr__(self, "degree_tercile_supports", supports)

    @property
    def num_classes(self) -> int:
        return len(self.confusion_matrix)

    @property
    def target_class_counts(self) -> tuple[int, ...]:
        return tuple(sum(row) for row in self.confusion_matrix)

    @property
    def exact_metric_values(self) -> tuple[Fraction, ...]:
        return _metric_fractions_from_counts(
            self.confusion_matrix,
            self.degree_tercile_correct_counts,
            self.degree_tercile_supports,
        )

    def predictive_metrics(self) -> GraphBiRCPredictiveMetrics:
        exact = self.exact_metric_values
        return GraphBiRCPredictiveMetrics(
            accuracy=float(exact[0]),
            balanced_accuracy=float(exact[1]),
            macro_f1=float(exact[2]),
            degree_tercile_accuracy=tuple(float(value) for value in exact[3:]),
            _exact_values=exact,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "degree_tercile_correct_counts": {
                group: count
                for group, count in zip(
                    GRAPH_BIRC_GROUP_ORDER,
                    self.degree_tercile_correct_counts,
                    strict=True,
                )
            },
            "degree_tercile_supports": {
                group: support
                for group, support in zip(
                    GRAPH_BIRC_GROUP_ORDER,
                    self.degree_tercile_supports,
                    strict=True,
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphBiRCMetricSufficientStatistics":
        if type(value) is not dict or set(value) != {
            "confusion_matrix",
            "degree_tercile_correct_counts",
            "degree_tercile_supports",
        }:
            raise GraphBiRCValidationError(
                "metric sufficient-statistic keys do not match schema"
            )
        raw_matrix = value["confusion_matrix"]
        if type(raw_matrix) is not list or not all(
            type(row) is list for row in raw_matrix
        ):
            raise GraphBiRCValidationError("confusion_matrix must be a JSON matrix")
        raw_correct = value["degree_tercile_correct_counts"]
        raw_supports = value["degree_tercile_supports"]
        if (
            type(raw_correct) is not dict
            or set(raw_correct) != set(GRAPH_BIRC_GROUP_ORDER)
            or type(raw_supports) is not dict
            or set(raw_supports) != set(GRAPH_BIRC_GROUP_ORDER)
        ):
            raise GraphBiRCValidationError(
                "degree-tercile sufficient-statistic keys do not match schema"
            )
        return cls(
            confusion_matrix=tuple(
                tuple(
                    _json_int(count, "confusion_matrix count", minimum=0)
                    for count in row
                )
                for row in raw_matrix
            ),
            degree_tercile_correct_counts=tuple(
                _json_int(
                    raw_correct[group],
                    f"degree_tercile_correct_counts.{group}",
                    minimum=0,
                )
                for group in GRAPH_BIRC_GROUP_ORDER
            ),
            degree_tercile_supports=tuple(
                _json_int(
                    raw_supports[group],
                    f"degree_tercile_supports.{group}",
                    minimum=1,
                )
                for group in GRAPH_BIRC_GROUP_ORDER
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphBiRCJointMetricSufficientStatistics:
    """Joint counts proving source/candidate metrics describe the same nodes."""

    degree_tercile_counts: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]

    def __post_init__(self) -> None:
        def checked_count(value: Any) -> int:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise GraphBiRCValidationError(
                    "joint sufficient-statistic counts must be integers"
                )
            result = int(value)
            if result < 0:
                raise GraphBiRCValidationError(
                    "joint sufficient-statistic counts must be non-negative"
                )
            return result

        try:
            cubes = tuple(
                tuple(
                    tuple(
                        tuple(checked_count(count) for count in candidate_axis)
                        for candidate_axis in source_axis
                    )
                    for source_axis in target_axis
                )
                for target_axis in self.degree_tercile_counts
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "joint sufficient statistics must be a four-dimensional iterable"
            ) from exc
        if len(cubes) != len(GRAPH_BIRC_GROUP_ORDER):
            raise GraphBiRCValidationError(
                "joint sufficient statistics must contain exactly three degree groups"
            )
        dimension = len(cubes[0])
        if dimension < 2 or any(
            len(cube) != dimension
            or any(
                len(source_axis) != dimension
                or any(
                    len(candidate_axis) != dimension for candidate_axis in source_axis
                )
                for source_axis in cube
            )
            for cube in cubes
        ):
            raise GraphBiRCValidationError(
                "joint sufficient statistics must have equal class dimensions of at least two"
            )
        supports = tuple(
            sum(
                count
                for target_axis in cube
                for source_axis in target_axis
                for count in source_axis
            )
            for cube in cubes
        )
        if any(support < 1 for support in supports):
            raise GraphBiRCValidationError(
                "joint sufficient-statistic degree groups must all be nonempty"
            )
        observed_targets = sum(
            any(
                cubes[group][target][source][candidate] > 0
                for group in range(len(cubes))
                for source in range(dimension)
                for candidate in range(dimension)
            )
            for target in range(dimension)
        )
        if observed_targets < 2:
            raise GraphBiRCValidationError(
                "joint sufficient statistics require at least two observed target classes"
            )
        object.__setattr__(self, "degree_tercile_counts", cubes)

    @property
    def num_classes(self) -> int:
        return len(self.degree_tercile_counts[0])

    @property
    def degree_tercile_supports(self) -> tuple[int, ...]:
        return tuple(
            sum(
                count
                for target_axis in cube
                for source_axis in target_axis
                for count in source_axis
            )
            for cube in self.degree_tercile_counts
        )

    @property
    def source_statistics(self) -> GraphBiRCMetricSufficientStatistics:
        dimension = self.num_classes
        matrix = tuple(
            tuple(
                sum(
                    self.degree_tercile_counts[group][target][source][candidate]
                    for group in range(len(GRAPH_BIRC_GROUP_ORDER))
                    for candidate in range(dimension)
                )
                for source in range(dimension)
            )
            for target in range(dimension)
        )
        correct = tuple(
            sum(
                cube[target][target][candidate]
                for target in range(dimension)
                for candidate in range(dimension)
            )
            for cube in self.degree_tercile_counts
        )
        return GraphBiRCMetricSufficientStatistics(
            confusion_matrix=matrix,
            degree_tercile_correct_counts=correct,
            degree_tercile_supports=self.degree_tercile_supports,
        )

    @property
    def candidate_statistics(self) -> GraphBiRCMetricSufficientStatistics:
        dimension = self.num_classes
        matrix = tuple(
            tuple(
                sum(
                    self.degree_tercile_counts[group][target][source][candidate]
                    for group in range(len(GRAPH_BIRC_GROUP_ORDER))
                    for source in range(dimension)
                )
                for candidate in range(dimension)
            )
            for target in range(dimension)
        )
        correct = tuple(
            sum(
                cube[target][source][target]
                for target in range(dimension)
                for source in range(dimension)
            )
            for cube in self.degree_tercile_counts
        )
        return GraphBiRCMetricSufficientStatistics(
            confusion_matrix=matrix,
            degree_tercile_correct_counts=correct,
            degree_tercile_supports=self.degree_tercile_supports,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_order": [
                "degree_tercile",
                "target_class",
                "source_prediction",
                "candidate_prediction",
            ],
            "degree_tercile_counts": {
                group: [
                    [list(candidate_axis) for candidate_axis in source_axis]
                    for source_axis in cube
                ]
                for group, cube in zip(
                    GRAPH_BIRC_GROUP_ORDER,
                    self.degree_tercile_counts,
                    strict=True,
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphBiRCJointMetricSufficientStatistics":
        if type(value) is not dict or set(value) != {
            "axis_order",
            "degree_tercile_counts",
        }:
            raise GraphBiRCValidationError(
                "joint sufficient-statistic keys do not match schema"
            )
        if value["axis_order"] != [
            "degree_tercile",
            "target_class",
            "source_prediction",
            "candidate_prediction",
        ]:
            raise GraphBiRCValidationError(
                "joint sufficient-statistic axis order does not match schema"
            )
        raw_counts = value["degree_tercile_counts"]
        if type(raw_counts) is not dict or set(raw_counts) != set(
            GRAPH_BIRC_GROUP_ORDER
        ):
            raise GraphBiRCValidationError(
                "joint sufficient-statistic degree-group keys do not match schema"
            )
        cubes: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
        for group in GRAPH_BIRC_GROUP_ORDER:
            raw_cube = raw_counts[group]
            if type(raw_cube) is not list or not all(
                type(source_axis) is list
                and all(type(candidate_axis) is list for candidate_axis in source_axis)
                for source_axis in raw_cube
            ):
                raise GraphBiRCValidationError(
                    "joint sufficient-statistic counts must be JSON arrays"
                )
            cubes.append(
                tuple(
                    tuple(
                        tuple(
                            _json_int(
                                count,
                                "joint sufficient-statistic count",
                                minimum=0,
                            )
                            for count in candidate_axis
                        )
                        for candidate_axis in source_axis
                    )
                    for source_axis in raw_cube
                )
            )
        return cls(degree_tercile_counts=tuple(cubes))


def _joint_classification_statistics(
    source_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    targets: np.ndarray,
    degree_groups: Sequence[np.ndarray],
    num_classes: int,
) -> GraphBiRCJointMetricSufficientStatistics:
    cubes: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    for group_indices in degree_groups:
        cube = np.zeros((num_classes, num_classes, num_classes), dtype=np.int64)
        np.add.at(
            cube,
            (
                targets[group_indices],
                source_predictions[group_indices],
                candidate_predictions[group_indices],
            ),
            1,
        )
        cubes.append(
            tuple(
                tuple(
                    tuple(int(count) for count in candidate_axis)
                    for candidate_axis in source_axis
                )
                for source_axis in cube
            )
        )
    return GraphBiRCJointMetricSufficientStatistics(degree_tercile_counts=tuple(cubes))


@dataclass(frozen=True, slots=True)
class GraphBiRCHarmOutcome:
    accuracy_gain: float
    balanced_accuracy_gain: float
    macro_f1_gain: float
    minimum_degree_tercile_accuracy_gain: float
    _exact_gains: tuple[Fraction, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in _HARM_COMPONENT_ORDER:
            checked = _finite_float(getattr(self, name), name)
            if checked < -1.0 or checked > 1.0:
                raise GraphBiRCValidationError(f"{name} must lie in [-1, 1]")
            object.__setattr__(self, name, checked)
        rendered_gains = tuple(
            float(getattr(self, name)) for name in _HARM_COMPONENT_ORDER
        )
        if self._exact_gains is None:
            exact_gains = tuple(Fraction(str(value)) for value in rendered_gains)
        else:
            try:
                exact_gains = tuple(self._exact_gains)
            except TypeError as exc:
                raise GraphBiRCValidationError(
                    "harm exact gains must be iterable"
                ) from exc
            if len(exact_gains) != len(_HARM_COMPONENT_ORDER) or not all(
                isinstance(value, Fraction) for value in exact_gains
            ):
                raise GraphBiRCValidationError("harm exact gains do not match schema")
        if any(value < -1 or value > 1 for value in exact_gains):
            raise GraphBiRCValidationError("harm exact gains must lie in [-1, 1]")
        if tuple(float(value) for value in exact_gains) != rendered_gains:
            raise GraphBiRCValidationError(
                "harm gain floats disagree with exact ratios"
            )
        object.__setattr__(self, "_exact_gains", exact_gains)

    @property
    def exact_gains(self) -> tuple[Fraction, ...]:
        if self._exact_gains is None:  # pragma: no cover - guarded in __post_init__
            raise GraphBiRCValidationError("harm gain ratios are unavailable")
        return self._exact_gains

    @property
    def triggered_components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, gain in zip(_HARM_COMPONENT_ORDER, self.exact_gains, strict=True)
            if gain < _HARM_FLOOR_FRACTIONS[name]
        )

    @property
    def is_harm(self) -> bool:
        return bool(self.triggered_components)

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: float(getattr(self, name)) for name in _HARM_COMPONENT_ORDER},
            "is_harm": self.is_harm,
            "triggered_components": list(self.triggered_components),
            "strict_lower_floors": {
                name: float(_HARM_FLOORS[name]) for name in _HARM_COMPONENT_ORDER
            },
            "exact_gain_ratios": {
                name: _ratio_payload(gain)
                for name, gain in zip(
                    _HARM_COMPONENT_ORDER, self.exact_gains, strict=True
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphBiRCHarmOutcome":
        expected = {
            *_HARM_COMPONENT_ORDER,
            "is_harm",
            "triggered_components",
            "strict_lower_floors",
            "exact_gain_ratios",
        }
        if type(value) is not dict or set(value) != expected:
            raise GraphBiRCValidationError("harm-outcome keys do not match schema")
        floors = value["strict_lower_floors"]
        if type(floors) is not dict or set(floors) != set(_HARM_COMPONENT_ORDER):
            raise GraphBiRCValidationError("harm floors do not match schema")
        if any(
            _json_float(floors[name], f"harm floor {name}") != _HARM_FLOORS[name]
            for name in _HARM_COMPONENT_ORDER
        ):
            raise GraphBiRCValidationError(
                "harm floors do not match the frozen definition"
            )
        exact_gains = value["exact_gain_ratios"]
        if type(exact_gains) is not dict or set(exact_gains) != set(
            _HARM_COMPONENT_ORDER
        ):
            raise GraphBiRCValidationError("harm exact-gain keys do not match schema")
        result = cls(
            **{name: _json_float(value[name], name) for name in _HARM_COMPONENT_ORDER},
            _exact_gains=tuple(
                _ratio_from_dict(exact_gains[name], f"exact_gain_ratios.{name}")
                for name in _HARM_COMPONENT_ORDER
            ),
        )
        if type(value["is_harm"]) is not bool or value["is_harm"] != result.is_harm:
            raise GraphBiRCValidationError("serialized harm label is inconsistent")
        components = value["triggered_components"]
        if (
            type(components) is not list
            or tuple(components) != result.triggered_components
        ):
            raise GraphBiRCValidationError(
                "serialized harm components are inconsistent"
            )
        return result


def classify_graph_birc_harm(
    *,
    accuracy_gain: Fraction,
    balanced_accuracy_gain: Fraction,
    macro_f1_gain: Fraction,
    minimum_degree_tercile_accuracy_gain: Fraction,
) -> GraphBiRCHarmOutcome:
    """Apply the exact disjunctive harm definition with strict lower boundaries."""

    exact_gains = (
        accuracy_gain,
        balanced_accuracy_gain,
        macro_f1_gain,
        minimum_degree_tercile_accuracy_gain,
    )
    if not all(isinstance(value, Fraction) for value in exact_gains):
        raise GraphBiRCValidationError(
            "classify_graph_birc_harm requires exact Fraction inputs"
        )

    return GraphBiRCHarmOutcome(
        accuracy_gain=float(accuracy_gain),
        balanced_accuracy_gain=float(balanced_accuracy_gain),
        macro_f1_gain=float(macro_f1_gain),
        minimum_degree_tercile_accuracy_gain=float(
            minimum_degree_tercile_accuracy_gain
        ),
        _exact_gains=exact_gains,
    )


def _features_from_dict(value: Any) -> GraphBiRCFeatures:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "feature_schema",
        "values",
        "diagnostics",
    }:
        raise GraphBiRCValidationError("serialized Graph-BiRC feature keys mismatch")
    if value["schema_version"] != GRAPH_BIRC_FEATURE_SCHEMA_VERSION:
        raise GraphBiRCValidationError("serialized feature version mismatch")
    if value["feature_schema"] != list(GRAPH_BIRC_FEATURE_SCHEMA):
        raise GraphBiRCValidationError("serialized feature schema mismatch")
    raw_values = value["values"]
    raw_diagnostics = value["diagnostics"]
    if type(raw_values) is not list or type(raw_diagnostics) is not list:
        raise GraphBiRCValidationError("serialized features must contain JSON lists")
    diagnostic_keys = {
        "group",
        "node_indices",
        "node_ids",
        "out_degrees",
        *GRAPH_BIRC_GROUP_FIELDS,
    }
    diagnostics: list[GraphBiRCGroupDiagnostics] = []
    for index, item in enumerate(raw_diagnostics):
        if type(item) is not dict or set(item) != diagnostic_keys:
            raise GraphBiRCValidationError("serialized diagnostic keys mismatch")
        raw_node_indices = item["node_indices"]
        raw_node_ids = item["node_ids"]
        raw_out_degrees = item["out_degrees"]
        if not all(
            type(sequence) is list
            for sequence in (raw_node_indices, raw_node_ids, raw_out_degrees)
        ):
            raise GraphBiRCValidationError(
                "diagnostic node metadata must contain JSON lists"
            )
        node_indices = tuple(
            _json_int(node_index, "diagnostic node_indices", minimum=0)
            for node_index in raw_node_indices
        )
        node_ids = tuple(
            _json_int(
                node_id,
                "diagnostic node_ids",
                minimum=-(2**63),
                maximum=2**63 - 1,
            )
            for node_id in raw_node_ids
        )
        out_degrees = tuple(
            _json_int(
                degree,
                "diagnostic out_degrees",
                minimum=0,
                maximum=2**63 - 1,
            )
            for degree in raw_out_degrees
        )
        diagnostics.append(
            GraphBiRCGroupDiagnostics(
                group=_identifier(item["group"], f"diagnostic group {index}"),
                node_indices=node_indices,
                node_ids=node_ids,
                out_degrees=out_degrees,
                source_mean_confidence=_json_float(
                    item["source_mean_confidence"], "source_mean_confidence"
                ),
                source_mean_normalized_entropy=_json_float(
                    item["source_mean_normalized_entropy"],
                    "source_mean_normalized_entropy",
                ),
                positive_confidence_change_mean=_json_float(
                    item["positive_confidence_change_mean"],
                    "positive_confidence_change_mean",
                ),
                negative_confidence_change_mean=_json_float(
                    item["negative_confidence_change_mean"],
                    "negative_confidence_change_mean",
                ),
                prediction_flip_fraction=_json_float(
                    item["prediction_flip_fraction"], "prediction_flip_fraction"
                ),
                mean_js_divergence=_json_float(
                    item["mean_js_divergence"], "mean_js_divergence"
                ),
                positive_neighborhood_agreement_change_mean=_json_float(
                    item["positive_neighborhood_agreement_change_mean"],
                    "positive_neighborhood_agreement_change_mean",
                ),
                negative_neighborhood_agreement_change_mean=_json_float(
                    item["negative_neighborhood_agreement_change_mean"],
                    "negative_neighborhood_agreement_change_mean",
                ),
                group_mass=_json_float(item["group_mass"], "group_mass"),
                isolate_fraction=_json_float(
                    item["isolate_fraction"], "isolate_fraction"
                ),
            )
        )
    return GraphBiRCFeatures(
        values=tuple(
            _json_float(item, "Graph-BiRC feature value") for item in raw_values
        ),
        diagnostics=tuple(diagnostics),
    )


def _validate_feature_diagnostics(features: GraphBiRCFeatures) -> None:
    diagnostics = features.diagnostics
    support = tuple(len(item.node_indices) for item in diagnostics)
    num_nodes = sum(support)
    base_size, remainder = divmod(num_nodes, len(GRAPH_BIRC_GROUP_ORDER))
    expected_support = tuple(
        base_size + (1 if index < remainder else 0)
        for index in range(len(GRAPH_BIRC_GROUP_ORDER))
    )
    if support != expected_support:
        raise GraphBiRCValidationError(
            "diagnostic supports disagree with deterministic degree terciles"
        )

    flattened_indices = tuple(
        node_index for item in diagnostics for node_index in item.node_indices
    )
    if sorted(flattened_indices) != list(range(num_nodes)):
        raise GraphBiRCValidationError(
            "diagnostic node indices must partition all probability rows"
        )
    flattened_node_ids = tuple(
        node_id for item in diagnostics for node_id in item.node_ids
    )
    if len(set(flattened_node_ids)) != num_nodes:
        raise GraphBiRCValidationError("diagnostic node IDs must be globally unique")

    node_ids_by_index = [0] * num_nodes
    degrees_by_index = [0] * num_nodes
    for item in diagnostics:
        for node_index, node_id, degree in zip(
            item.node_indices, item.node_ids, item.out_degrees, strict=True
        ):
            node_ids_by_index[node_index] = node_id
            degrees_by_index[node_index] = degree
    expected_order = tuple(
        sorted(
            range(num_nodes),
            key=lambda node_index: (
                degrees_by_index[node_index],
                node_ids_by_index[node_index],
            ),
        )
    )
    if flattened_indices != expected_order:
        raise GraphBiRCValidationError(
            "diagnostic groups disagree with degree and node-ID ordering"
        )

    bounded_fields = {
        "source_mean_confidence": (0.0, 1.0),
        "source_mean_normalized_entropy": (-1e-12, 1.0 + 1e-12),
        "positive_confidence_change_mean": (0.0, 1.0),
        "negative_confidence_change_mean": (-1.0, 0.0),
        "prediction_flip_fraction": (0.0, 1.0),
        "mean_js_divergence": (0.0, math.log(2.0) + 1e-12),
        "positive_neighborhood_agreement_change_mean": (0.0, 1.0),
        "negative_neighborhood_agreement_change_mean": (-1.0, 0.0),
        "group_mass": (0.0, 1.0),
        "isolate_fraction": (0.0, 1.0),
    }
    for item in diagnostics:
        for name, (lower, upper) in bounded_fields.items():
            feature_value = float(getattr(item, name))
            if feature_value < lower or feature_value > upper:
                raise GraphBiRCValidationError(
                    f"diagnostic {name} lies outside its valid range"
                )
        expected_mass = len(item.node_indices) / num_nodes
        if item.group_mass != expected_mass:
            raise GraphBiRCValidationError(
                "diagnostic group_mass disagrees with group support"
            )
        expected_isolate_fraction = sum(
            degree == 0 for degree in item.out_degrees
        ) / len(item.out_degrees)
        if item.isolate_fraction != expected_isolate_fraction:
            raise GraphBiRCValidationError(
                "diagnostic isolate_fraction disagrees with out_degrees"
            )


@dataclass(frozen=True, slots=True)
class GraphBiRCDevelopmentEvent:
    event_id: str
    group_id: str
    class_labels: tuple[int, ...]
    features: GraphBiRCFeatures
    joint_statistics: GraphBiRCJointMetricSufficientStatistics
    source_statistics: GraphBiRCMetricSufficientStatistics
    candidate_statistics: GraphBiRCMetricSufficientStatistics
    source_metrics: GraphBiRCPredictiveMetrics
    candidate_metrics: GraphBiRCPredictiveMetrics
    outcome: GraphBiRCHarmOutcome
    input_sha256: Mapping[str, str]
    schema_version: str = field(
        default=GRAPH_BIRC_DEVELOPMENT_EVENT_SCHEMA_VERSION, init=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        try:
            class_labels = tuple(
                _json_int(label, "class_labels", minimum=0)
                for label in self.class_labels
            )
        except TypeError as exc:
            raise GraphBiRCValidationError("class_labels must be iterable") from exc
        if class_labels != tuple(range(len(class_labels))) or len(class_labels) < 2:
            raise GraphBiRCValidationError(
                "class_labels must be the contiguous probability-column class space"
            )
        object.__setattr__(self, "class_labels", class_labels)
        if not isinstance(self.features, GraphBiRCFeatures):
            raise GraphBiRCValidationError("features must be GraphBiRCFeatures")
        _validate_feature_diagnostics(self.features)
        if not isinstance(
            self.joint_statistics, GraphBiRCJointMetricSufficientStatistics
        ):
            raise GraphBiRCValidationError(
                "event joint sufficient statistics use an unsupported type"
            )
        if not isinstance(
            self.source_statistics, GraphBiRCMetricSufficientStatistics
        ) or not isinstance(
            self.candidate_statistics, GraphBiRCMetricSufficientStatistics
        ):
            raise GraphBiRCValidationError(
                "event sufficient statistics use an unsupported type"
            )
        if (
            self.joint_statistics.num_classes != len(class_labels)
            or self.source_statistics.num_classes != len(class_labels)
            or self.candidate_statistics.num_classes != len(class_labels)
        ):
            raise GraphBiRCValidationError(
                "sufficient-statistic class dimensions disagree with class_labels"
            )
        if (
            self.joint_statistics.source_statistics != self.source_statistics
            or self.joint_statistics.candidate_statistics != self.candidate_statistics
        ):
            raise GraphBiRCValidationError(
                "source or candidate statistics disagree with joint sufficient statistics"
            )
        if (
            self.source_statistics.target_class_counts
            != self.candidate_statistics.target_class_counts
        ):
            raise GraphBiRCValidationError(
                "source and candidate target-class counts disagree"
            )
        expected_supports = tuple(
            len(item.node_indices) for item in self.features.diagnostics
        )
        if (
            self.joint_statistics.degree_tercile_supports != expected_supports
            or self.source_statistics.degree_tercile_supports != expected_supports
            or self.candidate_statistics.degree_tercile_supports != expected_supports
        ):
            raise GraphBiRCValidationError(
                "degree-tercile statistics disagree with feature diagnostics"
            )
        if not isinstance(
            self.source_metrics, GraphBiRCPredictiveMetrics
        ) or not isinstance(self.candidate_metrics, GraphBiRCPredictiveMetrics):
            raise GraphBiRCValidationError("event metrics use an unsupported type")
        if (
            self.source_metrics.exact_values
            != self.source_statistics.exact_metric_values
            or self.candidate_metrics.exact_values
            != self.candidate_statistics.exact_metric_values
        ):
            raise GraphBiRCValidationError(
                "event metrics disagree with integer sufficient statistics"
            )
        if not isinstance(self.outcome, GraphBiRCHarmOutcome):
            raise GraphBiRCValidationError("outcome must be GraphBiRCHarmOutcome")
        if type(self.input_sha256) is not dict and not isinstance(
            self.input_sha256, MappingProxyType
        ):
            raise GraphBiRCValidationError("input_sha256 must be a mapping")
        if set(self.input_sha256) != set(_INPUT_NAMES):
            raise GraphBiRCValidationError("input_sha256 keys do not match schema")
        digests = {
            name: _validate_digest(self.input_sha256[name], f"input_sha256.{name}")
            for name in _INPUT_NAMES
        }
        object.__setattr__(self, "input_sha256", MappingProxyType(digests))

        source_exact = self.source_metrics.exact_values
        candidate_exact = self.candidate_metrics.exact_values
        expected_gains = (
            candidate_exact[0] - source_exact[0],
            candidate_exact[1] - source_exact[1],
            candidate_exact[2] - source_exact[2],
            min(
                candidate - source
                for candidate, source in zip(
                    candidate_exact[3:],
                    source_exact[3:],
                    strict=True,
                )
            ),
        )
        if self.outcome.exact_gains != expected_gains:
            raise GraphBiRCValidationError(
                "harm outcome disagrees with source and candidate metrics"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_scope": _DEVELOPMENT_SCOPE,
            "scope_claim_status": _SCOPE_CLAIM_STATUS,
            "event_id": self.event_id,
            "group_id": self.group_id,
            "class_labels": list(self.class_labels),
            "features": self.features.to_dict(),
            "metric_convention": dict(_METRIC_CONVENTION),
            "metric_sufficient_statistics": {
                "joint": self.joint_statistics.to_dict(),
                "source": self.source_statistics.to_dict(),
                "candidate": self.candidate_statistics.to_dict(),
            },
            "source_metrics": self.source_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "outcome": self.outcome.to_dict(),
            "input_sha256": {name: self.input_sha256[name] for name in _INPUT_NAMES},
        }

    @property
    def artifact_sha256(self) -> str:
        return _sha256_canonical(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "artifact_sha256": _sha256_canonical(payload)}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphBiRCDevelopmentEvent":
        expected = {
            "schema_version",
            "data_scope",
            "scope_claim_status",
            "event_id",
            "group_id",
            "class_labels",
            "features",
            "metric_convention",
            "metric_sufficient_statistics",
            "source_metrics",
            "candidate_metrics",
            "outcome",
            "input_sha256",
            "artifact_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise GraphBiRCValidationError("development-event keys do not match schema")
        supplied_digest = _validate_digest(value["artifact_sha256"], "artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if not hmac.compare_digest(supplied_digest, _sha256_canonical(payload)):
            raise GraphBiRCValidationError("development-event SHA-256 mismatch")
        if value["schema_version"] != GRAPH_BIRC_DEVELOPMENT_EVENT_SCHEMA_VERSION:
            raise GraphBiRCValidationError("unsupported development-event schema")
        if value["data_scope"] != _DEVELOPMENT_SCOPE:
            raise GraphBiRCValidationError("invalid development-event data scope")
        if value["scope_claim_status"] != _SCOPE_CLAIM_STATUS:
            raise GraphBiRCValidationError("invalid development-event scope status")
        if value["metric_convention"] != _METRIC_CONVENTION:
            raise GraphBiRCValidationError("metric convention does not match schema")
        input_sha256 = value["input_sha256"]
        if type(input_sha256) is not dict:
            raise GraphBiRCValidationError("input_sha256 must be a JSON object")
        raw_class_labels = value["class_labels"]
        if type(raw_class_labels) is not list:
            raise GraphBiRCValidationError("class_labels must be a JSON list")
        statistics = value["metric_sufficient_statistics"]
        if type(statistics) is not dict or set(statistics) != {
            "joint",
            "source",
            "candidate",
        }:
            raise GraphBiRCValidationError(
                "metric sufficient-statistics container does not match schema"
            )
        result = cls(
            event_id=_identifier(value["event_id"], "event_id"),
            group_id=_identifier(value["group_id"], "group_id"),
            class_labels=tuple(
                _json_int(label, "class_labels", minimum=0)
                for label in raw_class_labels
            ),
            features=_features_from_dict(value["features"]),
            joint_statistics=GraphBiRCJointMetricSufficientStatistics.from_dict(
                statistics["joint"]
            ),
            source_statistics=GraphBiRCMetricSufficientStatistics.from_dict(
                statistics["source"]
            ),
            candidate_statistics=GraphBiRCMetricSufficientStatistics.from_dict(
                statistics["candidate"]
            ),
            source_metrics=GraphBiRCPredictiveMetrics.from_dict(
                value["source_metrics"]
            ),
            candidate_metrics=GraphBiRCPredictiveMetrics.from_dict(
                value["candidate_metrics"]
            ),
            outcome=GraphBiRCHarmOutcome.from_dict(value["outcome"]),
            input_sha256=input_sha256,
        )
        if not hmac.compare_digest(result.artifact_sha256, supplied_digest):
            raise GraphBiRCValidationError(
                "development event changes under strict deserialization"
            )
        return result

    def verify_against(
        self,
        *,
        edge_index: Any,
        source_probabilities: Any,
        candidate_probabilities: Any,
        targets: Any,
        node_ids: Any = None,
    ) -> None:
        """Rebuild from the original arrays and reject any content mismatch."""

        rebuilt = build_graph_birc_development_event(
            event_id=self.event_id,
            group_id=self.group_id,
            edge_index=edge_index,
            source_probabilities=source_probabilities,
            candidate_probabilities=candidate_probabilities,
            targets=targets,
            node_ids=node_ids,
        )
        if not hmac.compare_digest(rebuilt.artifact_sha256, self.artifact_sha256):
            raise GraphBiRCValidationError(
                "development event disagrees with the supplied original arrays"
            )


def build_graph_birc_development_event(
    *,
    event_id: str,
    group_id: str,
    edge_index: Any,
    source_probabilities: Any,
    candidate_probabilities: Any,
    targets: Any,
    node_ids: Any = None,
) -> GraphBiRCDevelopmentEvent:
    """Build one hash-bound labeled development event under the v4 schema."""

    checked_event_id = _identifier(event_id, "event_id")
    checked_group_id = _identifier(group_id, "group_id")
    raw_source = _validate_probabilities(source_probabilities, "source_probabilities")
    raw_candidate = _validate_probabilities(
        candidate_probabilities, "candidate_probabilities"
    )
    if raw_source.shape != raw_candidate.shape:
        raise GraphBiRCValidationError(
            "source and candidate probability matrices must have identical shapes"
        )
    source = raw_source / raw_source.sum(axis=1, keepdims=True)
    candidate = raw_candidate / raw_candidate.sum(axis=1, keepdims=True)
    num_nodes, num_classes = source.shape
    checked_edges = _validate_edge_index(edge_index, num_nodes)
    checked_node_ids = _validate_node_ids(node_ids, num_nodes)
    checked_targets = _validate_targets(
        targets, num_nodes=num_nodes, num_classes=num_classes
    )
    features = graph_birc_features(
        checked_edges,
        raw_source,
        raw_candidate,
        node_ids=checked_node_ids,
    )
    degree_groups = tuple(
        np.asarray(item.node_indices, dtype=np.int64) for item in features.diagnostics
    )
    joint_statistics = _joint_classification_statistics(
        np.argmax(source, axis=1),
        np.argmax(candidate, axis=1),
        checked_targets,
        degree_groups,
        num_classes,
    )
    source_statistics = joint_statistics.source_statistics
    candidate_statistics = joint_statistics.candidate_statistics
    source_metrics = source_statistics.predictive_metrics()
    candidate_metrics = candidate_statistics.predictive_metrics()
    source_exact = source_metrics.exact_values
    candidate_exact = candidate_metrics.exact_values
    exact_gains = (
        candidate_exact[0] - source_exact[0],
        candidate_exact[1] - source_exact[1],
        candidate_exact[2] - source_exact[2],
        min(
            candidate_value - source_value
            for candidate_value, source_value in zip(
                candidate_exact[3:],
                source_exact[3:],
                strict=True,
            )
        ),
    )
    outcome = GraphBiRCHarmOutcome(
        accuracy_gain=float(exact_gains[0]),
        balanced_accuracy_gain=float(exact_gains[1]),
        macro_f1_gain=float(exact_gains[2]),
        minimum_degree_tercile_accuracy_gain=float(exact_gains[3]),
        _exact_gains=exact_gains,
    )
    normalized_edges = np.asarray(checked_edges, dtype="<i8")
    normalized_source = np.asarray(raw_source, dtype="<f8")
    normalized_candidate = np.asarray(raw_candidate, dtype="<f8")
    normalized_targets = np.asarray(checked_targets, dtype="<i8")
    normalized_node_ids = np.asarray(checked_node_ids, dtype="<i8")
    return GraphBiRCDevelopmentEvent(
        event_id=checked_event_id,
        group_id=checked_group_id,
        class_labels=tuple(range(num_classes)),
        features=features,
        joint_statistics=joint_statistics,
        source_statistics=source_statistics,
        candidate_statistics=candidate_statistics,
        source_metrics=source_metrics,
        candidate_metrics=candidate_metrics,
        outcome=outcome,
        input_sha256={
            "candidate_probabilities": _array_sha256(normalized_candidate),
            "edge_index": _array_sha256(normalized_edges),
            "node_ids": _array_sha256(normalized_node_ids),
            "source_probabilities": _array_sha256(normalized_source),
            "targets": _array_sha256(normalized_targets),
        },
    )


@dataclass(frozen=True, slots=True)
class GraphBiRCCalibrationRecord:
    event_id: str
    group_id: str
    event_artifact_sha256: str
    input_bundle_sha256: str
    score: float
    harm_label: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "event_id"))
        object.__setattr__(self, "group_id", _identifier(self.group_id, "group_id"))
        object.__setattr__(
            self,
            "event_artifact_sha256",
            _validate_digest(self.event_artifact_sha256, "event_artifact_sha256"),
        )
        object.__setattr__(
            self,
            "input_bundle_sha256",
            _validate_digest(self.input_bundle_sha256, "input_bundle_sha256"),
        )
        object.__setattr__(self, "score", _unit_interval(self.score, "score"))
        if type(self.harm_label) is not bool:
            raise GraphBiRCValidationError("harm_label must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "group_id": self.group_id,
            "event_artifact_sha256": self.event_artifact_sha256,
            "input_bundle_sha256": self.input_bundle_sha256,
            "score": float(self.score),
            "harm_label": self.harm_label,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GraphBiRCCalibrationRecord":
        if type(value) is not dict or set(value) != {
            "event_id",
            "group_id",
            "event_artifact_sha256",
            "input_bundle_sha256",
            "score",
            "harm_label",
        }:
            raise GraphBiRCValidationError(
                "calibration-record keys do not match schema"
            )
        return cls(
            event_id=_identifier(value["event_id"], "event_id"),
            group_id=_identifier(value["group_id"], "group_id"),
            event_artifact_sha256=_validate_digest(
                value["event_artifact_sha256"], "event_artifact_sha256"
            ),
            input_bundle_sha256=_validate_digest(
                value["input_bundle_sha256"], "input_bundle_sha256"
            ),
            score=_json_float(value["score"], "score"),
            harm_label=value["harm_label"],
        )


@dataclass(frozen=True, slots=True)
class _ThresholdSelection:
    decision_mode: str
    threshold: float | None
    coverage: float
    accepted_count: int
    accepted_harm_count: int
    accepted_harm_risk: float | None
    candidate_threshold_count: int
    selection_reason: str
    constraints_satisfied: bool


def _select_operating_point(
    records: Sequence[GraphBiRCCalibrationRecord],
    *,
    maximum_accepted_harm_risk: float,
    minimum_coverage: float,
) -> _ThresholdSelection:
    maximum_risk_fraction = _unit_fraction(maximum_accepted_harm_risk)
    required_coverage_fraction = _unit_fraction(minimum_coverage)
    ranked = sorted(records, key=lambda item: (item.score, item.event_id))
    eligible: list[tuple[int, Fraction, float, int]] = []
    cumulative_harm = 0
    candidate_threshold_count = 0
    for index, record in enumerate(ranked):
        cumulative_harm += int(record.harm_label)
        if index + 1 < len(ranked) and ranked[index + 1].score == record.score:
            continue
        candidate_threshold_count += 1
        accepted_count = index + 1
        coverage_fraction = Fraction(accepted_count, len(ranked))
        risk_fraction = Fraction(cumulative_harm, accepted_count)
        if (
            coverage_fraction >= required_coverage_fraction
            and risk_fraction <= maximum_risk_fraction
        ):
            eligible.append(
                (accepted_count, risk_fraction, record.score, cumulative_harm)
            )

    if not eligible:
        return _ThresholdSelection(
            decision_mode="reject_all",
            threshold=None,
            coverage=0.0,
            accepted_count=0,
            accepted_harm_count=0,
            accepted_harm_risk=None,
            candidate_threshold_count=candidate_threshold_count,
            selection_reason="no_eligible_threshold_reject_all",
            constraints_satisfied=False,
        )

    accepted_count, risk_fraction, threshold, accepted_harm_count = min(
        eligible, key=lambda item: (-item[0], item[1], item[2])
    )
    return _ThresholdSelection(
        decision_mode="score_threshold",
        threshold=threshold,
        coverage=accepted_count / len(ranked),
        accepted_count=accepted_count,
        accepted_harm_count=accepted_harm_count,
        accepted_harm_risk=float(risk_fraction),
        candidate_threshold_count=candidate_threshold_count,
        selection_reason="eligible_threshold_selected",
        constraints_satisfied=True,
    )


def _calibration_payload(
    scorer_artifact_sha256: str,
    records: Sequence[GraphBiRCCalibrationRecord],
) -> dict[str, Any]:
    return {
        "data_scope": _DEVELOPMENT_SCOPE,
        "scope_claim_status": _SCOPE_CLAIM_STATUS,
        "feature_schema": list(GRAPH_BIRC_FEATURE_SCHEMA),
        "scorer_artifact_sha256": scorer_artifact_sha256,
        "events": [record.to_dict() for record in records],
    }


@dataclass(frozen=True, slots=True)
class GraphBiRCThresholdArtifact:
    protocol_id: str
    protocol_sha256: str
    scorer_artifact_sha256: str
    calibration_data_sha256: str
    group_split_sha256: str
    calibration_records: tuple[GraphBiRCCalibrationRecord, ...]
    calibration_event_count: int
    calibration_group_count: int
    calibration_harm_count: int
    maximum_accepted_harm_risk: float
    minimum_coverage: float
    decision_mode: str
    threshold: float | None
    coverage: float
    accepted_count: int
    accepted_harm_count: int
    accepted_harm_risk: float | None
    candidate_threshold_count: int
    selection_reason: str
    constraints_satisfied: bool
    schema_version: str = field(default=GRAPH_BIRC_THRESHOLD_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", _identifier(self.protocol_id, "protocol_id")
        )
        object.__setattr__(
            self,
            "protocol_sha256",
            _validate_digest(self.protocol_sha256, "protocol_sha256"),
        )
        for name in (
            "scorer_artifact_sha256",
            "calibration_data_sha256",
            "group_split_sha256",
        ):
            object.__setattr__(self, name, _validate_digest(getattr(self, name), name))
        try:
            records = tuple(self.calibration_records)
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "calibration_records must be iterable"
            ) from exc
        if not records or not all(
            isinstance(item, GraphBiRCCalibrationRecord) for item in records
        ):
            raise GraphBiRCValidationError(
                "calibration_records must contain calibration records"
            )
        event_ids = [item.event_id for item in records]
        if event_ids != sorted(event_ids) or len(set(event_ids)) != len(event_ids):
            raise GraphBiRCValidationError(
                "calibration record event IDs must be unique and sorted"
            )
        input_bundles = [item.input_bundle_sha256 for item in records]
        if len(set(input_bundles)) != len(input_bundles):
            raise GraphBiRCValidationError(
                "calibration records must have unique input bundles"
            )
        object.__setattr__(self, "calibration_records", records)
        for name, minimum in (
            ("calibration_event_count", 1),
            ("calibration_group_count", 1),
            ("calibration_harm_count", 0),
            ("accepted_count", 0),
            ("accepted_harm_count", 0),
            ("candidate_threshold_count", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise GraphBiRCValidationError(f"{name} must be an integer")
            value = int(value)
            if value < minimum:
                raise GraphBiRCValidationError(f"{name} must be at least {minimum}")
            object.__setattr__(self, name, value)
        if self.calibration_group_count > self.calibration_event_count:
            raise GraphBiRCValidationError(
                "calibration_group_count exceeds calibration_event_count"
            )
        if self.calibration_harm_count > self.calibration_event_count:
            raise GraphBiRCValidationError(
                "calibration_harm_count exceeds calibration_event_count"
            )
        if self.accepted_harm_count > self.calibration_harm_count:
            raise GraphBiRCValidationError(
                "accepted_harm_count exceeds calibration_harm_count"
            )
        if self.candidate_threshold_count > self.calibration_event_count:
            raise GraphBiRCValidationError(
                "candidate_threshold_count exceeds calibration_event_count"
            )
        object.__setattr__(
            self,
            "maximum_accepted_harm_risk",
            _unit_interval(
                self.maximum_accepted_harm_risk,
                "maximum_accepted_harm_risk",
            ),
        )
        object.__setattr__(
            self,
            "minimum_coverage",
            _unit_interval(self.minimum_coverage, "minimum_coverage"),
        )
        object.__setattr__(self, "coverage", _unit_interval(self.coverage, "coverage"))
        object.__setattr__(
            self, "decision_mode", _identifier(self.decision_mode, "decision_mode")
        )
        object.__setattr__(
            self,
            "selection_reason",
            _identifier(self.selection_reason, "selection_reason"),
        )
        if type(self.constraints_satisfied) is not bool:
            raise GraphBiRCValidationError("constraints_satisfied must be boolean")
        if self.decision_mode not in {"score_threshold", "reject_all"}:
            raise GraphBiRCValidationError("unsupported threshold decision_mode")
        if self.decision_mode == "score_threshold":
            if not self.constraints_satisfied:
                raise GraphBiRCValidationError(
                    "score_threshold must satisfy the declared constraints"
                )
            if self.threshold is None:
                raise GraphBiRCValidationError("score_threshold requires a threshold")
            object.__setattr__(
                self, "threshold", _unit_interval(self.threshold, "threshold")
            )
            if (
                self.accepted_count < 1
                or self.accepted_count > self.calibration_event_count
            ):
                raise GraphBiRCValidationError("accepted_count is impossible")
            if self.accepted_harm_count > self.accepted_count:
                raise GraphBiRCValidationError(
                    "accepted_harm_count exceeds accepted_count"
                )
            if self.accepted_harm_risk is None:
                raise GraphBiRCValidationError(
                    "score_threshold requires accepted_harm_risk"
                )
            object.__setattr__(
                self,
                "accepted_harm_risk",
                _unit_interval(self.accepted_harm_risk, "accepted_harm_risk"),
            )
            expected_coverage = self.accepted_count / self.calibration_event_count
            expected_risk = self.accepted_harm_count / self.accepted_count
            if self.coverage != expected_coverage:
                raise GraphBiRCValidationError("coverage disagrees with accepted_count")
            if self.accepted_harm_risk != expected_risk:
                raise GraphBiRCValidationError(
                    "accepted_harm_risk disagrees with accepted counts"
                )
            if Fraction(
                self.accepted_count, self.calibration_event_count
            ) < _unit_fraction(self.minimum_coverage):
                raise GraphBiRCValidationError("selected threshold violates coverage")
            if Fraction(self.accepted_harm_count, self.accepted_count) > _unit_fraction(
                self.maximum_accepted_harm_risk
            ):
                raise GraphBiRCValidationError("selected threshold violates harm risk")
            if self.selection_reason != "eligible_threshold_selected":
                raise GraphBiRCValidationError("invalid threshold selection reason")
        else:
            if self.constraints_satisfied:
                raise GraphBiRCValidationError(
                    "reject_all cannot claim that threshold constraints were satisfied"
                )
            if self.threshold is not None:
                raise GraphBiRCValidationError("reject_all threshold must be null")
            if (
                self.coverage != 0.0
                or self.accepted_count != 0
                or self.accepted_harm_count != 0
                or self.accepted_harm_risk is not None
            ):
                raise GraphBiRCValidationError(
                    "reject_all selection must have zero accepted coverage"
                )
            if self.selection_reason != "no_eligible_threshold_reject_all":
                raise GraphBiRCValidationError("invalid reject-all selection reason")

        if (
            self.calibration_harm_count - self.accepted_harm_count
            > self.calibration_event_count - self.accepted_count
        ):
            raise GraphBiRCValidationError(
                "rejected events cannot contain the declared remaining harms"
            )

        if self.calibration_event_count != len(records):
            raise GraphBiRCValidationError(
                "calibration_event_count disagrees with calibration records"
            )
        if self.calibration_group_count != len({item.group_id for item in records}):
            raise GraphBiRCValidationError(
                "calibration_group_count disagrees with calibration records"
            )
        if self.calibration_harm_count != sum(int(item.harm_label) for item in records):
            raise GraphBiRCValidationError(
                "calibration_harm_count disagrees with calibration records"
            )
        expected_data_sha256 = _sha256_canonical(
            _calibration_payload(self.scorer_artifact_sha256, records)
        )
        if not hmac.compare_digest(self.calibration_data_sha256, expected_data_sha256):
            raise GraphBiRCValidationError(
                "calibration_data_sha256 disagrees with calibration records"
            )
        expected_selection = _select_operating_point(
            records,
            maximum_accepted_harm_risk=self.maximum_accepted_harm_risk,
            minimum_coverage=self.minimum_coverage,
        )
        selection_fields = (
            "decision_mode",
            "threshold",
            "coverage",
            "accepted_count",
            "accepted_harm_count",
            "accepted_harm_risk",
            "candidate_threshold_count",
            "selection_reason",
            "constraints_satisfied",
        )
        if any(
            getattr(self, name) != getattr(expected_selection, name)
            for name in selection_fields
        ):
            raise GraphBiRCValidationError(
                "declared threshold selection is not the reproducible optimum"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_binding": {
                "protocol_id": self.protocol_id,
                "protocol_sha256": self.protocol_sha256,
            },
            "scorer_artifact_sha256": self.scorer_artifact_sha256,
            "calibration": {
                "data_scope": _DEVELOPMENT_SCOPE,
                "scope_claim_status": _SCOPE_CLAIM_STATUS,
                "data_sha256": self.calibration_data_sha256,
                "group_split_sha256": self.group_split_sha256,
                "event_count": int(self.calibration_event_count),
                "group_count": int(self.calibration_group_count),
                "harm_count": int(self.calibration_harm_count),
                "records": [record.to_dict() for record in self.calibration_records],
            },
            "constraints": {
                "maximum_accepted_harm_risk": float(self.maximum_accepted_harm_risk),
                "minimum_coverage": float(self.minimum_coverage),
            },
            "selection": {
                "decision_mode": self.decision_mode,
                "threshold": (
                    None if self.threshold is None else float(self.threshold)
                ),
                "coverage": float(self.coverage),
                "accepted_count": int(self.accepted_count),
                "accepted_harm_count": int(self.accepted_harm_count),
                "accepted_harm_risk": (
                    None
                    if self.accepted_harm_risk is None
                    else float(self.accepted_harm_risk)
                ),
                "candidate_threshold_count": int(self.candidate_threshold_count),
                "selection_reason": self.selection_reason,
                "constraints_satisfied": self.constraints_satisfied,
            },
            "decision_semantics": {
                "score_direction": "lower_is_safer",
                "rule": _THRESHOLD_RULE,
                "fallback": "reject_all_when_no_threshold_satisfies_both_constraints",
                "selection_order": list(_THRESHOLD_SELECTION_ORDER),
                "guarantee_status": (
                    "held-out development operating point; no population, "
                    "calibration, or conformal guarantee"
                ),
            },
        }

    @property
    def artifact_sha256(self) -> str:
        return _sha256_canonical(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {**payload, "artifact_sha256": _sha256_canonical(payload)}

    def canonical_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    def accept(
        self,
        scorer: GraphBiRCHarmScorer,
        features: GraphBiRCFeatures | Sequence[float],
        *,
        expected_protocol_id: str,
        expected_protocol_sha256: str,
        expected_l2_strength: float,
        expected_minimum_development_groups: int,
        expected_max_iterations: int,
        expected_optimizer_tolerance: float,
        expected_threshold_artifact_sha256: str,
        expected_maximum_accepted_harm_risk: float,
        expected_minimum_coverage: float,
        feature_schema: Sequence[str] | None = None,
    ) -> bool:
        """Decide using a threshold pinned after independent evidence checks.

        Expected values must come from a separately trusted freeze manifest,
        not from the artifact being checked. The freeze procedure must rebuild
        events from original arrays, refit the scorer, and rebuild the threshold.
        Invalid inputs raise even when the frozen policy rejects every event.
        """

        self.verify_protocol_binding(
            expected_protocol_id=expected_protocol_id,
            expected_protocol_sha256=expected_protocol_sha256,
        )
        checked_threshold_sha256 = _validate_digest(
            expected_threshold_artifact_sha256,
            "expected_threshold_artifact_sha256",
        )
        if not hmac.compare_digest(self.artifact_sha256, checked_threshold_sha256):
            raise GraphBiRCValidationError(
                "threshold artifact SHA-256 disagrees with externally pinned digest"
            )
        expected_maximum_risk = _unit_interval(
            expected_maximum_accepted_harm_risk,
            "expected_maximum_accepted_harm_risk",
        )
        expected_coverage = _unit_interval(
            expected_minimum_coverage, "expected_minimum_coverage"
        )
        if (
            self.maximum_accepted_harm_risk != expected_maximum_risk
            or self.minimum_coverage != expected_coverage
        ):
            raise GraphBiRCValidationError(
                "threshold constraints disagree with the externally pinned protocol"
            )
        if not isinstance(scorer, GraphBiRCHarmScorer):
            raise GraphBiRCValidationError("scorer has an unsupported type")
        scorer.verify_deployment_contract(
            expected_protocol_id=expected_protocol_id,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_l2_strength=expected_l2_strength,
            expected_minimum_development_groups=(expected_minimum_development_groups),
            expected_max_iterations=expected_max_iterations,
            expected_optimizer_tolerance=expected_optimizer_tolerance,
        )
        if not hmac.compare_digest(scorer.artifact_sha256, self.scorer_artifact_sha256):
            raise GraphBiRCValidationError("scorer SHA-256 mismatch")
        if not hmac.compare_digest(scorer.group_split_sha256, self.group_split_sha256):
            raise GraphBiRCValidationError("scorer group-split SHA-256 mismatch")
        score = scorer.score(features, feature_schema=feature_schema)
        if self.decision_mode == "reject_all":
            return False
        if self.threshold is None:
            raise GraphBiRCValidationError("threshold artifact is internally invalid")
        return score <= self.threshold

    def verify_protocol_binding(
        self,
        *,
        expected_protocol_id: str,
        expected_protocol_sha256: str,
    ) -> None:
        """Reject use under any protocol other than the externally pinned one."""

        checked_id = _identifier(expected_protocol_id, "expected_protocol_id")
        checked_sha256 = _validate_digest(
            expected_protocol_sha256, "expected_protocol_sha256"
        )
        if self.protocol_id != checked_id or not hmac.compare_digest(
            self.protocol_sha256, checked_sha256
        ):
            raise GraphBiRCValidationError("threshold protocol binding mismatch")

    def verify_against(
        self,
        scorer: GraphBiRCHarmScorer,
        threshold_calibration_events: Iterable[GraphBiRCDevelopmentEvent],
        *,
        scorer_fit_group_ids: Iterable[str],
        expected_protocol_id: str,
        expected_protocol_sha256: str,
        expected_l2_strength: float,
        expected_minimum_development_groups: int,
        expected_max_iterations: int,
        expected_optimizer_tolerance: float,
        expected_maximum_accepted_harm_risk: float,
        expected_minimum_coverage: float,
    ) -> None:
        """Rebuild under externally pinned protocol values and reject mismatch."""

        self.verify_protocol_binding(
            expected_protocol_id=expected_protocol_id,
            expected_protocol_sha256=expected_protocol_sha256,
        )
        scorer.verify_deployment_contract(
            expected_protocol_id=expected_protocol_id,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_l2_strength=expected_l2_strength,
            expected_minimum_development_groups=(expected_minimum_development_groups),
            expected_max_iterations=expected_max_iterations,
            expected_optimizer_tolerance=expected_optimizer_tolerance,
        )
        expected_maximum_risk = _unit_interval(
            expected_maximum_accepted_harm_risk,
            "expected_maximum_accepted_harm_risk",
        )
        expected_coverage = _unit_interval(
            expected_minimum_coverage, "expected_minimum_coverage"
        )
        if (
            self.maximum_accepted_harm_risk != expected_maximum_risk
            or self.minimum_coverage != expected_coverage
        ):
            raise GraphBiRCValidationError(
                "threshold constraints disagree with the externally pinned protocol"
            )

        rebuilt = select_graph_birc_threshold(
            scorer,
            threshold_calibration_events,
            scorer_fit_group_ids=scorer_fit_group_ids,
            protocol_id=expected_protocol_id,
            protocol_sha256=expected_protocol_sha256,
            expected_l2_strength=expected_l2_strength,
            expected_minimum_development_groups=(expected_minimum_development_groups),
            expected_max_iterations=expected_max_iterations,
            expected_optimizer_tolerance=expected_optimizer_tolerance,
            maximum_accepted_harm_risk=expected_maximum_risk,
            minimum_coverage=expected_coverage,
        )
        if not hmac.compare_digest(rebuilt.artifact_sha256, self.artifact_sha256):
            raise GraphBiRCValidationError(
                "threshold artifact disagrees with supplied scorer or events"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphBiRCThresholdArtifact":
        expected = {
            "schema_version",
            "protocol_binding",
            "scorer_artifact_sha256",
            "calibration",
            "constraints",
            "selection",
            "decision_semantics",
            "artifact_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise GraphBiRCValidationError(
                "threshold-artifact keys do not match schema"
            )
        supplied_digest = _validate_digest(value["artifact_sha256"], "artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if not hmac.compare_digest(supplied_digest, _sha256_canonical(payload)):
            raise GraphBiRCValidationError("threshold-artifact SHA-256 mismatch")
        if value["schema_version"] != GRAPH_BIRC_THRESHOLD_SCHEMA_VERSION:
            raise GraphBiRCValidationError("unsupported threshold-artifact schema")
        protocol_binding = value["protocol_binding"]
        calibration = value["calibration"]
        constraints = value["constraints"]
        selection = value["selection"]
        semantics = value["decision_semantics"]
        if type(protocol_binding) is not dict or set(protocol_binding) != {
            "protocol_id",
            "protocol_sha256",
        }:
            raise GraphBiRCValidationError("threshold protocol binding keys mismatch")
        if type(calibration) is not dict or set(calibration) != {
            "data_scope",
            "scope_claim_status",
            "data_sha256",
            "group_split_sha256",
            "event_count",
            "group_count",
            "harm_count",
            "records",
        }:
            raise GraphBiRCValidationError("threshold calibration keys mismatch")
        if (
            calibration["data_scope"] != _DEVELOPMENT_SCOPE
            or calibration["scope_claim_status"] != _SCOPE_CLAIM_STATUS
        ):
            raise GraphBiRCValidationError("invalid threshold calibration scope")
        raw_records = calibration["records"]
        if type(raw_records) is not list:
            raise GraphBiRCValidationError("calibration records must be a JSON list")
        if type(constraints) is not dict or set(constraints) != {
            "maximum_accepted_harm_risk",
            "minimum_coverage",
        }:
            raise GraphBiRCValidationError("threshold constraint keys mismatch")
        if type(selection) is not dict or set(selection) != {
            "decision_mode",
            "threshold",
            "coverage",
            "accepted_count",
            "accepted_harm_count",
            "accepted_harm_risk",
            "candidate_threshold_count",
            "selection_reason",
            "constraints_satisfied",
        }:
            raise GraphBiRCValidationError("threshold selection keys mismatch")
        expected_semantics = {
            "score_direction": "lower_is_safer",
            "rule": _THRESHOLD_RULE,
            "fallback": "reject_all_when_no_threshold_satisfies_both_constraints",
            "selection_order": list(_THRESHOLD_SELECTION_ORDER),
            "guarantee_status": (
                "held-out development operating point; no population, "
                "calibration, or conformal guarantee"
            ),
        }
        if semantics != expected_semantics:
            raise GraphBiRCValidationError("threshold decision semantics mismatch")

        raw_threshold = selection["threshold"]
        raw_risk = selection["accepted_harm_risk"]
        result = cls(
            protocol_id=_identifier(protocol_binding["protocol_id"], "protocol_id"),
            protocol_sha256=_validate_digest(
                protocol_binding["protocol_sha256"], "protocol_sha256"
            ),
            scorer_artifact_sha256=_validate_digest(
                value["scorer_artifact_sha256"], "scorer_artifact_sha256"
            ),
            calibration_data_sha256=_validate_digest(
                calibration["data_sha256"], "calibration.data_sha256"
            ),
            group_split_sha256=_validate_digest(
                calibration["group_split_sha256"],
                "calibration.group_split_sha256",
            ),
            calibration_records=tuple(
                GraphBiRCCalibrationRecord.from_dict(item) for item in raw_records
            ),
            calibration_event_count=_json_int(
                calibration["event_count"], "calibration.event_count", minimum=1
            ),
            calibration_group_count=_json_int(
                calibration["group_count"], "calibration.group_count", minimum=1
            ),
            calibration_harm_count=_json_int(
                calibration["harm_count"], "calibration.harm_count"
            ),
            maximum_accepted_harm_risk=_json_float(
                constraints["maximum_accepted_harm_risk"],
                "maximum_accepted_harm_risk",
            ),
            minimum_coverage=_json_float(
                constraints["minimum_coverage"], "minimum_coverage"
            ),
            decision_mode=_identifier(selection["decision_mode"], "decision_mode"),
            threshold=(
                None
                if raw_threshold is None
                else _json_float(raw_threshold, "threshold")
            ),
            coverage=_json_float(selection["coverage"], "coverage"),
            accepted_count=_json_int(selection["accepted_count"], "accepted_count"),
            accepted_harm_count=_json_int(
                selection["accepted_harm_count"], "accepted_harm_count"
            ),
            accepted_harm_risk=(
                None
                if raw_risk is None
                else _json_float(raw_risk, "accepted_harm_risk")
            ),
            candidate_threshold_count=_json_int(
                selection["candidate_threshold_count"],
                "candidate_threshold_count",
                minimum=1,
            ),
            selection_reason=_identifier(
                selection["selection_reason"], "selection_reason"
            ),
            constraints_satisfied=selection["constraints_satisfied"],
        )
        if not hmac.compare_digest(result.artifact_sha256, supplied_digest):
            raise GraphBiRCValidationError(
                "threshold artifact changes under strict deserialization"
            )
        return result


def _group_ids(value: Iterable[str], context: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        raise GraphBiRCValidationError(f"{context} must be an iterable of strings")
    try:
        values = list(value)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            f"{context} must be an iterable of strings"
        ) from exc
    if not values:
        raise GraphBiRCValidationError(f"{context} must not be empty")
    return [_identifier(item, context) for item in values]


def _unique_development_events(
    value: Iterable[GraphBiRCDevelopmentEvent], context: str
) -> list[GraphBiRCDevelopmentEvent]:
    try:
        events = list(value)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            f"{context} must be an iterable of development events"
        ) from exc
    if not events or not all(
        isinstance(item, GraphBiRCDevelopmentEvent) for item in events
    ):
        raise GraphBiRCValidationError(f"{context} must contain development events")
    event_ids = [item.event_id for item in events]
    if len(set(event_ids)) != len(event_ids):
        raise GraphBiRCValidationError(f"{context} event IDs must be unique")
    input_bundles = [_sha256_canonical(dict(item.input_sha256)) for item in events]
    if len(set(input_bundles)) != len(input_bundles):
        raise GraphBiRCValidationError(f"{context} input bundles must be unique")
    return sorted(events, key=lambda item: item.event_id)


def fit_graph_birc_harm_scorer_from_events(
    development_events: Iterable[GraphBiRCDevelopmentEvent],
    *,
    heldout_group_ids: Sequence[str],
    protocol_id: str,
    protocol_sha256: str,
    l2_strength: float = 1.0,
    minimum_development_groups: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> GraphBiRCHarmScorer:
    """Fit the scorer from event-derived features and exact harm labels only.

    Every event must first be checked with ``event.verify_against`` using the
    externally pinned original arrays. The scorer binds the ordered event and
    input-bundle digests, but this function cannot authenticate that the caller
    actually performed the prior raw-array verification.
    """

    events = _unique_development_events(development_events, "development_events")
    return fit_graph_birc_harm_scorer(
        [item.features for item in events],
        [item.outcome.is_harm for item in events],
        [item.group_id for item in events],
        heldout_group_ids=heldout_group_ids,
        l2_strength=l2_strength,
        minimum_development_groups=minimum_development_groups,
        max_iterations=max_iterations,
        tolerance=tolerance,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha256,
        development_event_ids=[item.event_id for item in events],
        development_event_sha256s=[item.artifact_sha256 for item in events],
        development_input_bundle_sha256s=[
            _sha256_canonical(dict(item.input_sha256)) for item in events
        ],
    )


def verify_graph_birc_harm_scorer_against_events(
    scorer: GraphBiRCHarmScorer,
    development_events: Iterable[GraphBiRCDevelopmentEvent],
    *,
    heldout_group_ids: Sequence[str],
    expected_protocol_id: str,
    expected_protocol_sha256: str,
    expected_l2_strength: float,
    expected_minimum_development_groups: int,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> None:
    """Refit under externally pinned hyperparameters and compare exact bytes."""

    if not isinstance(scorer, GraphBiRCHarmScorer):
        raise GraphBiRCValidationError("scorer has an unsupported type")
    rebuilt = fit_graph_birc_harm_scorer_from_events(
        development_events,
        heldout_group_ids=heldout_group_ids,
        protocol_id=expected_protocol_id,
        protocol_sha256=expected_protocol_sha256,
        l2_strength=expected_l2_strength,
        minimum_development_groups=expected_minimum_development_groups,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if not hmac.compare_digest(rebuilt.artifact_sha256, scorer.artifact_sha256):
        raise GraphBiRCValidationError(
            "scorer artifact disagrees with supplied events or pinned fit contract"
        )


def select_graph_birc_threshold(
    scorer: GraphBiRCHarmScorer,
    threshold_calibration_events: Iterable[GraphBiRCDevelopmentEvent],
    *,
    scorer_fit_group_ids: Iterable[str],
    protocol_id: str,
    protocol_sha256: str,
    expected_l2_strength: float,
    expected_minimum_development_groups: int,
    expected_max_iterations: int,
    expected_optimizer_tolerance: float,
    maximum_accepted_harm_risk: float = 0.2,
    minimum_coverage: float = 0.2,
) -> GraphBiRCThresholdArtifact:
    """Select the best feasible held-out-development operating threshold."""

    if not isinstance(scorer, GraphBiRCHarmScorer):
        raise GraphBiRCValidationError("scorer has an unsupported type")
    checked_protocol_id = _identifier(protocol_id, "protocol_id")
    checked_protocol_sha256 = _validate_digest(protocol_sha256, "protocol_sha256")
    scorer.verify_deployment_contract(
        expected_protocol_id=checked_protocol_id,
        expected_protocol_sha256=checked_protocol_sha256,
        expected_l2_strength=expected_l2_strength,
        expected_minimum_development_groups=expected_minimum_development_groups,
        expected_max_iterations=expected_max_iterations,
        expected_optimizer_tolerance=expected_optimizer_tolerance,
    )
    try:
        events = list(threshold_calibration_events)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            "threshold_calibration_events must be an iterable of development events"
        ) from exc
    if not events or not all(
        isinstance(item, GraphBiRCDevelopmentEvent) for item in events
    ):
        raise GraphBiRCValidationError(
            "calibration_events must contain development events"
        )
    event_ids = [item.event_id for item in events]
    if len(set(event_ids)) != len(event_ids):
        raise GraphBiRCValidationError("calibration event IDs must be unique")
    input_bundle_digests = [
        _sha256_canonical(dict(item.input_sha256)) for item in events
    ]
    if len(set(input_bundle_digests)) != len(input_bundle_digests):
        raise GraphBiRCValidationError(
            "calibration events must have unique input bundles"
        )
    events.sort(key=lambda item: item.event_id)

    calibration_event_ids = {item.event_id for item in events}
    calibration_event_sha256s = {item.artifact_sha256 for item in events}
    calibration_input_sha256s = set(input_bundle_digests)
    if calibration_event_ids & set(scorer.development_event_ids):
        raise GraphBiRCValidationError("development and calibration event IDs overlap")
    if calibration_event_sha256s & set(scorer.development_event_sha256s):
        raise GraphBiRCValidationError(
            "development and calibration event artifacts overlap"
        )
    if calibration_input_sha256s & set(scorer.development_input_bundle_sha256s):
        raise GraphBiRCValidationError(
            "development and calibration input bundles overlap"
        )

    development_ids = _group_ids(scorer_fit_group_ids, "scorer_fit_group_ids")
    unique_development = sorted(set(development_ids))
    unique_calibration = sorted({item.group_id for item in events})
    if len(unique_development) != scorer.development_group_count:
        raise GraphBiRCValidationError("development group count disagrees with scorer")
    if len(unique_calibration) != scorer.heldout_group_count:
        raise GraphBiRCValidationError("calibration group count disagrees with scorer")
    if set(unique_development) & set(unique_calibration):
        raise GraphBiRCValidationError("development and calibration group IDs overlap")
    split_payload = {
        "scope_claim_status": _SCOPE_CLAIM_STATUS,
        "development_group_ids": unique_development,
        "heldout_group_ids": unique_calibration,
    }
    split_digest = _sha256_canonical(split_payload)
    if not hmac.compare_digest(split_digest, scorer.group_split_sha256):
        raise GraphBiRCValidationError(
            "calibration split digest disagrees with scorer declaration"
        )

    maximum_risk = _unit_interval(
        maximum_accepted_harm_risk, "maximum_accepted_harm_risk"
    )
    required_coverage = _unit_interval(minimum_coverage, "minimum_coverage")
    records = tuple(
        GraphBiRCCalibrationRecord(
            event_id=item.event_id,
            group_id=item.group_id,
            event_artifact_sha256=item.artifact_sha256,
            input_bundle_sha256=_sha256_canonical(dict(item.input_sha256)),
            score=scorer.score(item.features),
            harm_label=item.outcome.is_harm,
        )
        for item in events
    )
    calibration_payload = _calibration_payload(scorer.artifact_sha256, records)
    calibration_data_sha256 = _sha256_canonical(calibration_payload)
    selection = _select_operating_point(
        records,
        maximum_accepted_harm_risk=maximum_risk,
        minimum_coverage=required_coverage,
    )
    return GraphBiRCThresholdArtifact(
        protocol_id=checked_protocol_id,
        protocol_sha256=checked_protocol_sha256,
        scorer_artifact_sha256=scorer.artifact_sha256,
        calibration_data_sha256=calibration_data_sha256,
        group_split_sha256=split_digest,
        calibration_records=records,
        calibration_event_count=len(events),
        calibration_group_count=len(unique_calibration),
        calibration_harm_count=sum(int(record.harm_label) for record in records),
        maximum_accepted_harm_risk=maximum_risk,
        minimum_coverage=required_coverage,
        decision_mode=selection.decision_mode,
        threshold=selection.threshold,
        coverage=selection.coverage,
        accepted_count=selection.accepted_count,
        accepted_harm_count=selection.accepted_harm_count,
        accepted_harm_risk=selection.accepted_harm_risk,
        candidate_threshold_count=selection.candidate_threshold_count,
        selection_reason=selection.selection_reason,
        constraints_satisfied=selection.constraints_satisfied,
    )


__all__ = [
    "GRAPH_BIRC_DEVELOPMENT_EVENT_SCHEMA_VERSION",
    "GRAPH_BIRC_THRESHOLD_SCHEMA_VERSION",
    "ACCURACY_HARM_FLOOR",
    "BALANCED_ACCURACY_HARM_FLOOR",
    "MACRO_F1_HARM_FLOOR",
    "DEGREE_TERCILE_ACCURACY_HARM_FLOOR",
    "GraphBiRCPredictiveMetrics",
    "GraphBiRCMetricSufficientStatistics",
    "GraphBiRCJointMetricSufficientStatistics",
    "GraphBiRCHarmOutcome",
    "GraphBiRCDevelopmentEvent",
    "GraphBiRCCalibrationRecord",
    "GraphBiRCThresholdArtifact",
    "classify_graph_birc_harm",
    "build_graph_birc_development_event",
    "fit_graph_birc_harm_scorer_from_events",
    "verify_graph_birc_harm_scorer_against_events",
    "select_graph_birc_threshold",
]
