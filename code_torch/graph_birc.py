"""Auditable graph-bidirectional reliability change (Graph-BiRC) scores.

Graph-BiRC keeps degree-tercile and change-direction information separate.  It
does not pool groups or cancel positive and negative changes.  The fitted harm
score is a regularized development-set ranking model; it has no calibration or
conformal coverage guarantee.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

GRAPH_BIRC_FEATURE_SCHEMA_VERSION = "graph_birc_features.v1"
GRAPH_BIRC_SCORER_SCHEMA_VERSION = "graph_birc_harm_scorer.v3"
GRAPH_BIRC_GROUP_ORDER = ("low", "mid", "high")
GRAPH_BIRC_GROUP_FIELDS = (
    "source_mean_confidence",
    "source_mean_normalized_entropy",
    "positive_confidence_change_mean",
    "negative_confidence_change_mean",
    "prediction_flip_fraction",
    "mean_js_divergence",
    "positive_neighborhood_agreement_change_mean",
    "negative_neighborhood_agreement_change_mean",
    "group_mass",
    "isolate_fraction",
)
GRAPH_BIRC_FEATURE_SCHEMA = tuple(
    f"{group}.{name}"
    for group in GRAPH_BIRC_GROUP_ORDER
    for name in GRAPH_BIRC_GROUP_FIELDS
)

_PROBABILITY_SUM_ATOL = 1e-6
_SCORER_TYPE = "l2_regularized_logistic_harm_score"
_DECISION_RULE = "accept iff score <= threshold"
_DATA_SCOPE = "caller_declared_labeled_development_events"
_SCOPE_CLAIM_STATUS = "caller_declared_not_independently_verified"
_SPLIT_DISJOINTNESS = "verified_from_caller_supplied_group_ids"
_GUARANTEE_STATUS = (
    "development-fit ranking score; no calibration or conformal coverage guarantee"
)


class GraphBiRCValidationError(ValueError):
    """Raised when an input or serialized artifact violates the Graph-BiRC contract."""


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise GraphBiRCValidationError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise GraphBiRCValidationError(f"{context} must be finite")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GraphBiRCValidationError(f"{context} must be an integer")
    return int(value)


def _positive_int(value: Any, context: str, *, minimum: int = 1) -> int:
    result = _integer(value, context)
    if result < minimum:
        raise GraphBiRCValidationError(f"{context} must be at least {minimum}")
    return result


def _serialized_float(value: Any, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise GraphBiRCValidationError(f"{context} must be a finite JSON float")
    return value


def _serialized_int(value: Any, context: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise GraphBiRCValidationError(
            f"{context} must be a JSON integer of at least {minimum}"
        )
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise GraphBiRCValidationError(
            "artifact is not canonical-JSON serializable"
        ) from exc


def _sha256_canonical(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GraphBiRCValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, context: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GraphBiRCValidationError(f"{context} must be a trimmed non-empty string")
    return value


def _validate_probabilities(value: Any, context: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError(
            f"{context} must be a numeric probability matrix"
        ) from exc
    if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
        raise GraphBiRCValidationError(
            f"{context} must be a numeric probability matrix"
        )
    probabilities = np.asarray(raw, dtype=np.float64)
    if probabilities.ndim != 2:
        raise GraphBiRCValidationError(f"{context} must have shape [nodes, classes]")
    if probabilities.shape[0] < len(GRAPH_BIRC_GROUP_ORDER):
        raise GraphBiRCValidationError(
            f"{context} requires at least {len(GRAPH_BIRC_GROUP_ORDER)} nodes"
        )
    if probabilities.shape[1] < 2:
        raise GraphBiRCValidationError(f"{context} requires at least two classes")
    if not np.all(np.isfinite(probabilities)):
        raise GraphBiRCValidationError(f"{context} contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise GraphBiRCValidationError(f"{context} contains values outside [0, 1]")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=_PROBABILITY_SUM_ATOL):
        raise GraphBiRCValidationError(f"{context} rows must sum to one")
    return probabilities


def _validate_edge_index(value: Any, num_nodes: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError(
            "edge_index must contain integer node indices"
        ) from exc
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise GraphBiRCValidationError("edge_index must contain integer node indices")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise GraphBiRCValidationError("edge_index exceeds the signed 64-bit range")
    edge_index = np.asarray(raw, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphBiRCValidationError("edge_index must have shape [2, edges]")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise GraphBiRCValidationError("edge_index contains an out-of-range node index")
    return edge_index


def _validate_node_ids(value: Any, num_nodes: int) -> np.ndarray:
    if value is None:
        return np.arange(num_nodes, dtype=np.int64)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError("node_ids must contain integers") from exc
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise GraphBiRCValidationError("node_ids must contain integers")
    if raw.dtype.kind == "u" and raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise GraphBiRCValidationError("node_ids exceed the signed 64-bit range")
    node_ids = np.asarray(raw, dtype=np.int64)
    if node_ids.ndim != 1 or len(node_ids) != num_nodes:
        raise GraphBiRCValidationError(
            "node_ids must contain one ID per probability row"
        )
    if len(np.unique(node_ids)) != num_nodes:
        raise GraphBiRCValidationError("node_ids must be unique")
    return node_ids


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    return -np.sum(probabilities * np.log(clipped), axis=1) / math.log(
        probabilities.shape[1]
    )


def _js_divergence(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (source + candidate)
    tiny = np.finfo(np.float64).tiny
    source_term = np.where(
        source > 0.0,
        source * np.log(np.clip(source, tiny, 1.0) / np.clip(midpoint, tiny, 1.0)),
        0.0,
    )
    candidate_term = np.where(
        candidate > 0.0,
        candidate
        * np.log(np.clip(candidate, tiny, 1.0) / np.clip(midpoint, tiny, 1.0)),
        0.0,
    )
    return 0.5 * np.sum(source_term + candidate_term, axis=1)


def _outgoing_agreement(
    edge_index: np.ndarray, predictions: np.ndarray, out_degree: np.ndarray
) -> np.ndarray:
    agreement = np.zeros(len(predictions), dtype=np.float64)
    if edge_index.shape[1] == 0:
        return agreement
    sources, targets = edge_index
    matching = (predictions[sources] == predictions[targets]).astype(np.float64)
    matching_count = np.bincount(
        sources, weights=matching, minlength=len(predictions)
    ).astype(np.float64)
    np.divide(matching_count, out_degree, out=agreement, where=out_degree > 0)
    return agreement


@dataclass(frozen=True, slots=True)
class GraphBiRCGroupDiagnostics:
    """Immutable audit trace for one deterministic degree tercile."""

    group: str
    node_indices: tuple[int, ...]
    node_ids: tuple[int, ...]
    out_degrees: tuple[int, ...]
    source_mean_confidence: float
    source_mean_normalized_entropy: float
    positive_confidence_change_mean: float
    negative_confidence_change_mean: float
    prediction_flip_fraction: float
    mean_js_divergence: float
    positive_neighborhood_agreement_change_mean: float
    negative_neighborhood_agreement_change_mean: float
    group_mass: float
    isolate_fraction: float

    def __post_init__(self) -> None:
        if self.group not in GRAPH_BIRC_GROUP_ORDER:
            raise GraphBiRCValidationError("unknown Graph-BiRC diagnostic group")
        try:
            node_indices = tuple(
                _positive_int(value, "node_indices", minimum=0)
                for value in self.node_indices
            )
            node_ids = tuple(_integer(value, "node_ids") for value in self.node_ids)
            out_degrees = tuple(
                _positive_int(value, "out_degrees", minimum=0)
                for value in self.out_degrees
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "diagnostic node metadata must be iterable"
            ) from exc
        if not node_indices or not (
            len(node_indices) == len(node_ids) == len(out_degrees)
        ):
            raise GraphBiRCValidationError(
                "diagnostic node metadata must have equal non-zero lengths"
            )
        if len(set(node_indices)) != len(node_indices):
            raise GraphBiRCValidationError("diagnostic node indices must be unique")
        if len(set(node_ids)) != len(node_ids):
            raise GraphBiRCValidationError("diagnostic node IDs must be unique")
        object.__setattr__(self, "node_indices", node_indices)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "out_degrees", out_degrees)
        for name in GRAPH_BIRC_GROUP_FIELDS:
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    @property
    def feature_values(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in GRAPH_BIRC_GROUP_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "node_indices": list(self.node_indices),
            "node_ids": list(self.node_ids),
            "out_degrees": list(self.out_degrees),
            **{name: float(getattr(self, name)) for name in GRAPH_BIRC_GROUP_FIELDS},
        }


@dataclass(frozen=True, slots=True)
class GraphBiRCFeatures:
    """Named immutable feature vector plus its ordered group diagnostics."""

    values: tuple[float, ...]
    diagnostics: tuple[GraphBiRCGroupDiagnostics, ...]
    schema_version: str = field(default=GRAPH_BIRC_FEATURE_SCHEMA_VERSION, init=False)
    feature_schema: tuple[str, ...] = field(
        default=GRAPH_BIRC_FEATURE_SCHEMA, init=False
    )

    def __post_init__(self) -> None:
        try:
            values = tuple(
                _finite_float(value, "Graph-BiRC feature value")
                for value in self.values
            )
            diagnostics = tuple(self.diagnostics)
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "Graph-BiRC values and diagnostics must be iterable"
            ) from exc
        if not all(isinstance(item, GraphBiRCGroupDiagnostics) for item in diagnostics):
            raise GraphBiRCValidationError(
                "Graph-BiRC diagnostics must contain diagnostic objects"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "diagnostics", diagnostics)
        if len(values) != len(GRAPH_BIRC_FEATURE_SCHEMA):
            raise GraphBiRCValidationError("Graph-BiRC feature vector length mismatch")
        if tuple(item.group for item in diagnostics) != GRAPH_BIRC_GROUP_ORDER:
            raise GraphBiRCValidationError("Graph-BiRC diagnostic group order mismatch")
        flattened = tuple(
            value for diagnostic in diagnostics for value in diagnostic.feature_values
        )
        if flattened != values:
            raise GraphBiRCValidationError(
                "Graph-BiRC values disagree with ordered group diagnostics"
            )

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(self.feature_schema, self.values, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_schema": list(self.feature_schema),
            "values": list(self.values),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def graph_birc_features(
    edge_index: Any,
    source_probabilities: Any,
    candidate_probabilities: Any,
    *,
    node_ids: Any = None,
) -> GraphBiRCFeatures:
    """Extract ordered, label-free Graph-BiRC features.

    Terciles are formed by stable sorting on ``(out_degree, node_id)`` and
    splitting the resulting order into three near-equal, non-empty groups.
    For directed inputs, degree and agreement are outgoing-only; ``isolate``
    therefore means zero out-degree rather than zero total degree.
    Positive parts are ``mean(max(change, 0))`` and negative parts are
    ``mean(min(change, 0))`` within each group, so opposing changes cannot
    cancel before they reach the scorer.
    """

    source = _validate_probabilities(source_probabilities, "source_probabilities")
    candidate = _validate_probabilities(
        candidate_probabilities, "candidate_probabilities"
    )
    if source.shape != candidate.shape:
        raise GraphBiRCValidationError(
            "source and candidate probability matrices must have identical shapes"
        )
    source = source / source.sum(axis=1, keepdims=True)
    candidate = candidate / candidate.sum(axis=1, keepdims=True)
    num_nodes = source.shape[0]
    checked_edges = _validate_edge_index(edge_index, num_nodes)
    checked_node_ids = _validate_node_ids(node_ids, num_nodes)

    out_degree = np.bincount(checked_edges[0], minlength=num_nodes).astype(np.int64)
    sorted_indices = np.lexsort((checked_node_ids, out_degree))
    group_indices = np.array_split(sorted_indices, len(GRAPH_BIRC_GROUP_ORDER))

    source_confidence = np.max(source, axis=1)
    candidate_confidence = np.max(candidate, axis=1)
    confidence_change = candidate_confidence - source_confidence
    source_entropy = _normalized_entropy(source)
    source_prediction = np.argmax(source, axis=1)
    candidate_prediction = np.argmax(candidate, axis=1)
    flips = candidate_prediction != source_prediction
    divergence = _js_divergence(source, candidate)
    source_agreement = _outgoing_agreement(checked_edges, source_prediction, out_degree)
    candidate_agreement = _outgoing_agreement(
        checked_edges, candidate_prediction, out_degree
    )
    agreement_change = candidate_agreement - source_agreement
    agreement_change[out_degree == 0] = 0.0

    diagnostics: list[GraphBiRCGroupDiagnostics] = []
    for group, indices in zip(GRAPH_BIRC_GROUP_ORDER, group_indices, strict=True):
        group_confidence_change = confidence_change[indices]
        group_agreement_change = agreement_change[indices]
        diagnostics.append(
            GraphBiRCGroupDiagnostics(
                group=group,
                node_indices=tuple(int(index) for index in indices),
                node_ids=tuple(int(checked_node_ids[index]) for index in indices),
                out_degrees=tuple(int(out_degree[index]) for index in indices),
                source_mean_confidence=float(np.mean(source_confidence[indices])),
                source_mean_normalized_entropy=float(np.mean(source_entropy[indices])),
                positive_confidence_change_mean=float(
                    np.mean(np.maximum(group_confidence_change, 0.0))
                ),
                negative_confidence_change_mean=float(
                    np.mean(np.minimum(group_confidence_change, 0.0))
                ),
                prediction_flip_fraction=float(np.mean(flips[indices])),
                mean_js_divergence=float(np.mean(divergence[indices])),
                positive_neighborhood_agreement_change_mean=float(
                    np.mean(np.maximum(group_agreement_change, 0.0))
                ),
                negative_neighborhood_agreement_change_mean=float(
                    np.mean(np.minimum(group_agreement_change, 0.0))
                ),
                group_mass=float(len(indices) / num_nodes),
                isolate_fraction=float(np.mean(out_degree[indices] == 0)),
            )
        )
    values = tuple(value for item in diagnostics for value in item.feature_values)
    return GraphBiRCFeatures(values=values, diagnostics=tuple(diagnostics))


def _coerce_schema(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        raise GraphBiRCValidationError(
            "raw feature values require an explicit feature_schema"
        )
    if isinstance(value, (str, bytes)):
        raise GraphBiRCValidationError("feature_schema must be a sequence of names")
    try:
        schema = tuple(value)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            "feature_schema must be a sequence of names"
        ) from exc
    if schema != GRAPH_BIRC_FEATURE_SCHEMA:
        raise GraphBiRCValidationError("Graph-BiRC feature schema mismatch")
    return schema


def _coerce_feature_row(
    value: GraphBiRCFeatures | Sequence[float],
    feature_schema: Sequence[str] | None,
) -> np.ndarray:
    if isinstance(value, GraphBiRCFeatures):
        if feature_schema is not None:
            _coerce_schema(feature_schema)
        row = value.as_array()
    else:
        _coerce_schema(feature_schema)
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GraphBiRCValidationError(
                "Graph-BiRC features must be numeric"
            ) from exc
        if raw.dtype.kind not in "fiu" or raw.dtype.kind == "b":
            raise GraphBiRCValidationError("Graph-BiRC features must be numeric")
        row = np.asarray(raw, dtype=np.float64)
    if row.ndim != 1 or len(row) != len(GRAPH_BIRC_FEATURE_SCHEMA):
        raise GraphBiRCValidationError("Graph-BiRC feature vector length mismatch")
    if not np.all(np.isfinite(row)):
        raise GraphBiRCValidationError("Graph-BiRC features must be finite")
    return row


def _coerce_development_matrix(
    values: Iterable[GraphBiRCFeatures | Sequence[float]],
    feature_schema: Sequence[str] | None,
) -> np.ndarray:
    try:
        rows = list(values)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            "development features must be an iterable of feature rows"
        ) from exc
    if not rows:
        raise GraphBiRCValidationError("at least one development event is required")
    object_rows = [isinstance(row, GraphBiRCFeatures) for row in rows]
    if any(object_rows) and not all(object_rows):
        raise GraphBiRCValidationError(
            "development features cannot mix named and raw feature rows"
        )
    return np.vstack([_coerce_feature_row(row, feature_schema) for row in rows]).astype(
        np.float64, copy=False
    )


def _sigmoid(logits: np.ndarray | float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def _logistic_objective(
    design: np.ndarray, labels: np.ndarray, parameters: np.ndarray, l2_strength: float
) -> float:
    logits = design @ parameters
    return float(
        np.sum(np.logaddexp(0.0, logits) - labels * logits)
        + 0.5 * l2_strength * np.dot(parameters[1:], parameters[1:])
    )


def _fit_logistic_newton(
    standardized: np.ndarray,
    labels: np.ndarray,
    *,
    l2_strength: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    design = np.column_stack((np.ones(len(standardized)), standardized))
    prevalence = float(np.mean(labels))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    parameters[0] = math.log(prevalence / (1.0 - prevalence))

    for iteration in range(1, max_iterations + 1):
        logits = design @ parameters
        probabilities = _sigmoid(logits)
        residual = probabilities - labels
        gradient = design.T @ residual
        gradient[1:] += l2_strength * parameters[1:]
        if float(np.max(np.abs(gradient))) <= tolerance:
            return parameters, iteration

        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        hessian = design.T @ (design * weights[:, None])
        hessian[1:, 1:] += l2_strength * np.eye(standardized.shape[1])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise GraphBiRCValidationError(
                "regularized logistic optimization failed"
            ) from exc

        objective = _logistic_objective(design, labels, parameters, l2_strength)
        step_scale = 1.0
        accepted_step = False
        for _ in range(50):
            candidate = parameters - step_scale * step
            candidate_objective = _logistic_objective(
                design, labels, candidate, l2_strength
            )
            if math.isfinite(candidate_objective) and candidate_objective <= objective:
                parameters = candidate
                accepted_step = True
                break
            step_scale *= 0.5
        if not accepted_step:
            raise GraphBiRCValidationError(
                "regularized logistic optimization did not find a finite descent step"
            )
        if float(np.max(np.abs(step_scale * step))) <= tolerance:
            return parameters, iteration

    raise GraphBiRCValidationError("regularized logistic optimization did not converge")


@dataclass(frozen=True, slots=True)
class GraphBiRCHarmScorer:
    """Immutable L2-logistic harm score fitted on caller-declared development data."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2_strength: float
    development_event_count: int
    development_group_count: int
    heldout_group_count: int
    development_data_sha256: str
    group_split_sha256: str
    optimizer_iterations: int
    minimum_development_groups: int = 3
    max_iterations: int = 100
    optimizer_tolerance: float = 1e-10
    protocol_id: str | None = None
    protocol_sha256: str | None = None
    development_event_ids: tuple[str, ...] = ()
    development_event_sha256s: tuple[str, ...] = ()
    development_input_bundle_sha256s: tuple[str, ...] = ()
    feature_schema: tuple[str, ...] = field(
        default=GRAPH_BIRC_FEATURE_SCHEMA, init=False
    )
    schema_version: str = field(default=GRAPH_BIRC_SCORER_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        expected_length = len(GRAPH_BIRC_FEATURE_SCHEMA)
        for name, values in (
            ("feature_mean", self.feature_mean),
            ("feature_scale", self.feature_scale),
            ("coefficients", self.coefficients),
        ):
            try:
                normalized = tuple(
                    _finite_float(value, f"{name} value") for value in values
                )
            except TypeError as exc:
                raise GraphBiRCValidationError(f"{name} must be iterable") from exc
            object.__setattr__(self, name, normalized)
            if len(normalized) != expected_length:
                raise GraphBiRCValidationError(f"{name} length disagrees with schema")
        if any(float(value) <= 0.0 for value in self.feature_scale):
            raise GraphBiRCValidationError("feature_scale must be strictly positive")
        object.__setattr__(
            self, "intercept", _finite_float(self.intercept, "intercept")
        )
        checked_l2 = _finite_float(self.l2_strength, "l2_strength")
        object.__setattr__(self, "l2_strength", checked_l2)
        if checked_l2 <= 0.0:
            raise GraphBiRCValidationError("l2_strength must be positive")
        event_count = _positive_int(
            self.development_event_count, "development_event_count"
        )
        group_count = _positive_int(
            self.development_group_count,
            "development_group_count",
            minimum=3,
        )
        heldout_count = _positive_int(self.heldout_group_count, "heldout_group_count")
        if event_count < group_count:
            raise GraphBiRCValidationError(
                "development_event_count must be at least development_group_count"
            )
        object.__setattr__(self, "development_event_count", event_count)
        object.__setattr__(self, "development_group_count", group_count)
        object.__setattr__(self, "heldout_group_count", heldout_count)
        object.__setattr__(
            self,
            "optimizer_iterations",
            _positive_int(self.optimizer_iterations, "optimizer_iterations"),
        )
        checked_minimum_groups = _positive_int(
            self.minimum_development_groups,
            "minimum_development_groups",
            minimum=3,
        )
        checked_max_iterations = _positive_int(self.max_iterations, "max_iterations")
        checked_tolerance = _finite_float(
            self.optimizer_tolerance, "optimizer_tolerance"
        )
        if checked_tolerance <= 0.0:
            raise GraphBiRCValidationError("optimizer_tolerance must be positive")
        if checked_minimum_groups > group_count:
            raise GraphBiRCValidationError(
                "minimum_development_groups exceeds development_group_count"
            )
        if self.optimizer_iterations > checked_max_iterations:
            raise GraphBiRCValidationError(
                "optimizer_iterations exceeds max_iterations"
            )
        object.__setattr__(self, "minimum_development_groups", checked_minimum_groups)
        object.__setattr__(self, "max_iterations", checked_max_iterations)
        object.__setattr__(self, "optimizer_tolerance", checked_tolerance)
        object.__setattr__(
            self,
            "development_data_sha256",
            _validate_digest(self.development_data_sha256, "development_data_sha256"),
        )
        object.__setattr__(
            self,
            "group_split_sha256",
            _validate_digest(self.group_split_sha256, "group_split_sha256"),
        )

        try:
            event_ids = tuple(self.development_event_ids)
            event_sha256s = tuple(self.development_event_sha256s)
            input_sha256s = tuple(self.development_input_bundle_sha256s)
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "development event evidence must be iterable"
            ) from exc
        evidence_values_present = bool(
            event_ids
            or event_sha256s
            or input_sha256s
            or self.protocol_id is not None
            or self.protocol_sha256 is not None
        )
        if evidence_values_present:
            if self.protocol_id is None or self.protocol_sha256 is None:
                raise GraphBiRCValidationError(
                    "bound scorer evidence requires a protocol ID and SHA-256"
                )
            checked_protocol_id = _identifier(self.protocol_id, "protocol_id")
            checked_protocol_sha256 = _validate_digest(
                self.protocol_sha256, "protocol_sha256"
            )
            if not (
                len(event_ids)
                == len(event_sha256s)
                == len(input_sha256s)
                == event_count
            ):
                raise GraphBiRCValidationError(
                    "bound development evidence must contain one record per event"
                )
            checked_event_ids = tuple(
                _identifier(value, "development_event_id") for value in event_ids
            )
            if checked_event_ids != tuple(sorted(checked_event_ids)) or len(
                set(checked_event_ids)
            ) != len(checked_event_ids):
                raise GraphBiRCValidationError(
                    "development event IDs must be unique and sorted"
                )
            checked_event_sha256s = tuple(
                _validate_digest(value, "development_event_sha256")
                for value in event_sha256s
            )
            checked_input_sha256s = tuple(
                _validate_digest(value, "development_input_bundle_sha256")
                for value in input_sha256s
            )
            if len(set(checked_event_sha256s)) != len(checked_event_sha256s):
                raise GraphBiRCValidationError(
                    "development event artifact SHA-256 values must be unique"
                )
            if len(set(checked_input_sha256s)) != len(checked_input_sha256s):
                raise GraphBiRCValidationError(
                    "development input bundles must be unique"
                )
        else:
            checked_protocol_id = None
            checked_protocol_sha256 = None
            checked_event_ids = ()
            checked_event_sha256s = ()
            checked_input_sha256s = ()
        object.__setattr__(self, "protocol_id", checked_protocol_id)
        object.__setattr__(self, "protocol_sha256", checked_protocol_sha256)
        object.__setattr__(self, "development_event_ids", checked_event_ids)
        object.__setattr__(self, "development_event_sha256s", checked_event_sha256s)
        object.__setattr__(
            self, "development_input_bundle_sha256s", checked_input_sha256s
        )

    @property
    def event_evidence_bound(self) -> bool:
        return self.protocol_id is not None

    def verify_deployment_contract(
        self,
        *,
        expected_protocol_id: str,
        expected_protocol_sha256: str,
        expected_l2_strength: float,
        expected_minimum_development_groups: int,
        expected_max_iterations: int,
        expected_optimizer_tolerance: float,
    ) -> None:
        """Check declared event bindings and externally pinned hyperparameters.

        Binding a digest does not authenticate dataset identity or chronology.
        The event evidence must be verified against the original arrays first.
        """

        if not self.event_evidence_bound:
            raise GraphBiRCValidationError(
                "scorer lacks protocol-bound development event evidence"
            )
        checked_protocol_id = _identifier(expected_protocol_id, "expected_protocol_id")
        checked_protocol_sha256 = _validate_digest(
            expected_protocol_sha256, "expected_protocol_sha256"
        )
        if self.protocol_id != checked_protocol_id or not hmac.compare_digest(
            self.protocol_sha256 or "", checked_protocol_sha256
        ):
            raise GraphBiRCValidationError("scorer protocol binding mismatch")
        checked_l2 = _finite_float(expected_l2_strength, "expected_l2_strength")
        checked_minimum_groups = _positive_int(
            expected_minimum_development_groups,
            "expected_minimum_development_groups",
            minimum=3,
        )
        checked_max_iterations = _positive_int(
            expected_max_iterations, "expected_max_iterations"
        )
        checked_tolerance = _finite_float(
            expected_optimizer_tolerance, "expected_optimizer_tolerance"
        )
        if checked_l2 <= 0.0 or checked_tolerance <= 0.0:
            raise GraphBiRCValidationError(
                "expected scorer hyperparameters must be positive"
            )
        if (
            self.l2_strength != checked_l2
            or self.minimum_development_groups != checked_minimum_groups
            or self.max_iterations != checked_max_iterations
            or self.optimizer_tolerance != checked_tolerance
        ):
            raise GraphBiRCValidationError(
                "scorer fit contract disagrees with the externally pinned protocol"
            )

    def score(
        self,
        features: GraphBiRCFeatures | Sequence[float],
        *,
        feature_schema: Sequence[str] | None = None,
    ) -> float:
        """Return a logistic harm score in [0, 1]; lower is safer."""

        row = _coerce_feature_row(features, feature_schema)
        standardized = (
            row - np.asarray(self.feature_mean, dtype=np.float64)
        ) / np.asarray(self.feature_scale, dtype=np.float64)
        logit = self.intercept + standardized @ np.asarray(
            self.coefficients, dtype=np.float64
        )
        score = float(_sigmoid(float(logit)))
        if not math.isfinite(score):
            raise GraphBiRCValidationError("Graph-BiRC harm score is non-finite")
        return score

    def accept(
        self,
        features: GraphBiRCFeatures | Sequence[float],
        threshold: float,
        *,
        feature_schema: Sequence[str] | None = None,
    ) -> bool:
        """Accept exactly when a valid score is at most the threshold.

        Invalid thresholds and feature inputs are caller errors. Deployment
        code that needs fail-closed behavior must catch the validation error
        explicitly and record that fallback decision.
        """

        checked_threshold = _finite_float(threshold, "threshold")
        if checked_threshold < 0.0 or checked_threshold > 1.0:
            raise GraphBiRCValidationError("threshold must lie in [0, 1]")
        return self.score(features, feature_schema=feature_schema) <= checked_threshold

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_binding": {
                "status": (
                    "event_evidence_bound"
                    if self.event_evidence_bound
                    else "unbound_generic_fit"
                ),
                "protocol_id": self.protocol_id,
                "protocol_sha256": self.protocol_sha256,
            },
            "feature_schema": list(self.feature_schema),
            "group_order": list(GRAPH_BIRC_GROUP_ORDER),
            "group_fields": list(GRAPH_BIRC_GROUP_FIELDS),
            "standardization": {
                "mean": list(self.feature_mean),
                "scale": list(self.feature_scale),
                "ddof": 0,
                "constant_feature_scale": 1.0,
            },
            "model": {
                "type": _SCORER_TYPE,
                "coefficients": list(self.coefficients),
                "intercept": float(self.intercept),
                "l2_strength": float(self.l2_strength),
            },
            "training": {
                "data_scope": _DATA_SCOPE,
                "scope_claim_status": _SCOPE_CLAIM_STATUS,
                "development_event_count": int(self.development_event_count),
                "development_group_count": int(self.development_group_count),
                "heldout_group_count": int(self.heldout_group_count),
                "development_data_sha256": self.development_data_sha256,
                "group_split_sha256": self.group_split_sha256,
                "split_disjointness": _SPLIT_DISJOINTNESS,
                "development_event_evidence": [
                    {
                        "event_id": event_id,
                        "event_artifact_sha256": event_sha256,
                        "input_bundle_sha256": input_sha256,
                    }
                    for event_id, event_sha256, input_sha256 in zip(
                        self.development_event_ids,
                        self.development_event_sha256s,
                        self.development_input_bundle_sha256s,
                    )
                ],
                "fit_contract": {
                    "minimum_development_groups": int(self.minimum_development_groups),
                    "max_iterations": int(self.max_iterations),
                    "optimizer_tolerance": float(self.optimizer_tolerance),
                },
                "optimizer": "deterministic_newton_with_backtracking",
                "optimizer_iterations": int(self.optimizer_iterations),
            },
            "decision_semantics": {
                "direction": "lower_is_safer",
                "rule": _DECISION_RULE,
                "guarantee_status": _GUARANTEE_STATUS,
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraphBiRCHarmScorer":
        if type(value) is not dict:
            raise GraphBiRCValidationError("scorer artifact must be a JSON object")
        expected_keys = {
            "schema_version",
            "protocol_binding",
            "feature_schema",
            "group_order",
            "group_fields",
            "standardization",
            "model",
            "training",
            "decision_semantics",
            "artifact_sha256",
        }
        if set(value) != expected_keys:
            raise GraphBiRCValidationError("scorer artifact keys do not match schema")
        supplied_digest = _validate_digest(value["artifact_sha256"], "artifact_sha256")
        payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
        computed_digest = _sha256_canonical(payload)
        if not hmac.compare_digest(supplied_digest, computed_digest):
            raise GraphBiRCValidationError("scorer artifact SHA-256 mismatch")
        if value["schema_version"] != GRAPH_BIRC_SCORER_SCHEMA_VERSION:
            raise GraphBiRCValidationError("unsupported scorer schema version")
        if value["feature_schema"] != list(GRAPH_BIRC_FEATURE_SCHEMA):
            raise GraphBiRCValidationError("serialized feature schema mismatch")
        if value["group_order"] != list(GRAPH_BIRC_GROUP_ORDER):
            raise GraphBiRCValidationError("serialized group order mismatch")
        if value["group_fields"] != list(GRAPH_BIRC_GROUP_FIELDS):
            raise GraphBiRCValidationError("serialized group fields mismatch")

        protocol_binding = value["protocol_binding"]
        if type(protocol_binding) is not dict or set(protocol_binding) != {
            "status",
            "protocol_id",
            "protocol_sha256",
        }:
            raise GraphBiRCValidationError("invalid scorer protocol binding")
        binding_status = protocol_binding["status"]
        if binding_status not in {"event_evidence_bound", "unbound_generic_fit"}:
            raise GraphBiRCValidationError("unsupported scorer protocol binding status")
        if binding_status == "event_evidence_bound":
            parsed_protocol_id = _identifier(
                protocol_binding["protocol_id"], "protocol_id"
            )
            parsed_protocol_sha256 = _validate_digest(
                protocol_binding["protocol_sha256"], "protocol_sha256"
            )
        else:
            if (
                protocol_binding["protocol_id"] is not None
                or protocol_binding["protocol_sha256"] is not None
            ):
                raise GraphBiRCValidationError(
                    "unbound scorer cannot declare a protocol binding"
                )
            parsed_protocol_id = None
            parsed_protocol_sha256 = None

        standardization = value["standardization"]
        if type(standardization) is not dict or set(standardization) != {
            "mean",
            "scale",
            "ddof",
            "constant_feature_scale",
        }:
            raise GraphBiRCValidationError("invalid standardization artifact")
        if (
            type(standardization["ddof"]) is not int
            or standardization["ddof"] != 0
            or type(standardization["constant_feature_scale"]) is not float
            or standardization["constant_feature_scale"] != 1.0
        ):
            raise GraphBiRCValidationError("unsupported standardization convention")

        model = value["model"]
        if type(model) is not dict or set(model) != {
            "type",
            "coefficients",
            "intercept",
            "l2_strength",
        }:
            raise GraphBiRCValidationError("invalid scorer model artifact")
        if model["type"] != _SCORER_TYPE:
            raise GraphBiRCValidationError("unsupported scorer model type")

        training = value["training"]
        if type(training) is not dict or set(training) != {
            "data_scope",
            "scope_claim_status",
            "development_event_count",
            "development_group_count",
            "heldout_group_count",
            "development_data_sha256",
            "group_split_sha256",
            "split_disjointness",
            "development_event_evidence",
            "fit_contract",
            "optimizer",
            "optimizer_iterations",
        }:
            raise GraphBiRCValidationError("invalid scorer training artifact")
        if training["data_scope"] != _DATA_SCOPE:
            raise GraphBiRCValidationError("invalid scorer data-scope declaration")
        if training["scope_claim_status"] != _SCOPE_CLAIM_STATUS:
            raise GraphBiRCValidationError("invalid scorer scope-claim status")
        if training["split_disjointness"] != _SPLIT_DISJOINTNESS:
            raise GraphBiRCValidationError("invalid scorer split-disjointness status")
        if training["optimizer"] != "deterministic_newton_with_backtracking":
            raise GraphBiRCValidationError("unsupported scorer optimizer")
        raw_evidence = training["development_event_evidence"]
        if type(raw_evidence) is not list:
            raise GraphBiRCValidationError(
                "development_event_evidence must be a JSON list"
            )
        if binding_status == "event_evidence_bound" and not raw_evidence:
            raise GraphBiRCValidationError(
                "bound scorer must contain development event evidence"
            )
        if binding_status == "unbound_generic_fit" and raw_evidence:
            raise GraphBiRCValidationError(
                "unbound scorer cannot contain development event evidence"
            )
        parsed_event_ids: list[str] = []
        parsed_event_sha256s: list[str] = []
        parsed_input_sha256s: list[str] = []
        for record in raw_evidence:
            if type(record) is not dict or set(record) != {
                "event_id",
                "event_artifact_sha256",
                "input_bundle_sha256",
            }:
                raise GraphBiRCValidationError(
                    "development event evidence keys do not match schema"
                )
            parsed_event_ids.append(_identifier(record["event_id"], "event_id"))
            parsed_event_sha256s.append(
                _validate_digest(
                    record["event_artifact_sha256"], "event_artifact_sha256"
                )
            )
            parsed_input_sha256s.append(
                _validate_digest(record["input_bundle_sha256"], "input_bundle_sha256")
            )
        fit_contract = training["fit_contract"]
        if type(fit_contract) is not dict or set(fit_contract) != {
            "minimum_development_groups",
            "max_iterations",
            "optimizer_tolerance",
        }:
            raise GraphBiRCValidationError("invalid scorer fit contract")

        semantics = value["decision_semantics"]
        if type(semantics) is not dict or semantics != {
            "direction": "lower_is_safer",
            "rule": _DECISION_RULE,
            "guarantee_status": _GUARANTEE_STATUS,
        }:
            raise GraphBiRCValidationError("invalid scorer decision semantics")

        def tuple_of_serialized_floats(items: Any, context: str) -> tuple[float, ...]:
            if type(items) is not list:
                raise GraphBiRCValidationError(f"{context} must be a list")
            return tuple(_serialized_float(item, context) for item in items)

        result = cls(
            feature_mean=tuple_of_serialized_floats(
                standardization["mean"], "standardization.mean"
            ),
            feature_scale=tuple_of_serialized_floats(
                standardization["scale"], "standardization.scale"
            ),
            coefficients=tuple_of_serialized_floats(
                model["coefficients"], "model.coefficients"
            ),
            intercept=_serialized_float(model["intercept"], "model.intercept"),
            l2_strength=_serialized_float(model["l2_strength"], "model.l2_strength"),
            development_event_count=_serialized_int(
                training["development_event_count"], "development_event_count"
            ),
            development_group_count=_serialized_int(
                training["development_group_count"],
                "development_group_count",
                minimum=3,
            ),
            heldout_group_count=_serialized_int(
                training["heldout_group_count"], "heldout_group_count"
            ),
            development_data_sha256=_validate_digest(
                training["development_data_sha256"], "development_data_sha256"
            ),
            group_split_sha256=_validate_digest(
                training["group_split_sha256"], "group_split_sha256"
            ),
            optimizer_iterations=_serialized_int(
                training["optimizer_iterations"], "optimizer_iterations"
            ),
            minimum_development_groups=_serialized_int(
                fit_contract["minimum_development_groups"],
                "minimum_development_groups",
                minimum=3,
            ),
            max_iterations=_serialized_int(
                fit_contract["max_iterations"], "max_iterations"
            ),
            optimizer_tolerance=_serialized_float(
                fit_contract["optimizer_tolerance"], "optimizer_tolerance"
            ),
            protocol_id=parsed_protocol_id,
            protocol_sha256=parsed_protocol_sha256,
            development_event_ids=tuple(parsed_event_ids),
            development_event_sha256s=tuple(parsed_event_sha256s),
            development_input_bundle_sha256s=tuple(parsed_input_sha256s),
        )
        if not hmac.compare_digest(result.artifact_sha256, supplied_digest):
            raise GraphBiRCValidationError(
                "scorer artifact changes under strict deserialization"
            )
        return result


def fit_graph_birc_harm_scorer(
    development_features: Iterable[GraphBiRCFeatures | Sequence[float]],
    development_harm_labels: Sequence[bool | int | float],
    development_group_ids: Sequence[str],
    *,
    heldout_group_ids: Sequence[str],
    feature_schema: Sequence[str] | None = None,
    l2_strength: float = 1.0,
    minimum_development_groups: int = 3,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
    protocol_id: str | None = None,
    protocol_sha256: str | None = None,
    development_event_ids: Sequence[str] | None = None,
    development_event_sha256s: Sequence[str] | None = None,
    development_input_bundle_sha256s: Sequence[str] | None = None,
) -> GraphBiRCHarmScorer:
    """Fit a deterministic harm scorer on caller-declared development events.

    The caller must supply development and held-out group identities. Their
    disjointness is checked and hashed, but their real-world provenance cannot
    be independently verified here. Raw numeric rows require the exact feature
    schema explicitly, while ``GraphBiRCFeatures`` objects carry it intrinsically.
    """

    matrix = _coerce_development_matrix(development_features, feature_schema)
    try:
        raw_labels = np.asarray(development_harm_labels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GraphBiRCValidationError(
            "development harm labels must be a one-dimensional numeric sequence"
        ) from exc
    if raw_labels.ndim != 1 or len(raw_labels) != len(matrix):
        raise GraphBiRCValidationError(
            "development_harm_labels must contain one label per event"
        )
    if raw_labels.dtype.kind not in "biuf":
        raise GraphBiRCValidationError("development harm labels must be binary")
    labels = np.asarray(raw_labels, dtype=np.float64)
    if not np.all(np.isfinite(labels)) or not np.all((labels == 0.0) | (labels == 1.0)):
        raise GraphBiRCValidationError(
            "development harm labels must be finite 0/1 values"
        )
    if len(np.unique(labels)) != 2:
        raise GraphBiRCValidationError(
            "development harm labels must contain both classes"
        )

    if isinstance(development_group_ids, (str, bytes)) or isinstance(
        heldout_group_ids, (str, bytes)
    ):
        raise GraphBiRCValidationError("group IDs must be sequences of strings")
    try:
        group_ids = list(development_group_ids)
        heldout_ids = list(heldout_group_ids)
    except TypeError as exc:
        raise GraphBiRCValidationError(
            "group IDs must be sequences of strings"
        ) from exc
    if len(group_ids) != len(matrix) or any(
        type(group) is not str or not group or group != group.strip()
        for group in group_ids
    ):
        raise GraphBiRCValidationError(
            "development_group_ids must contain one trimmed non-empty string per event"
        )
    if not heldout_ids or any(
        type(group) is not str or not group or group != group.strip()
        for group in heldout_ids
    ):
        raise GraphBiRCValidationError(
            "heldout_group_ids must contain trimmed non-empty strings"
        )
    checked_minimum_groups = _positive_int(
        minimum_development_groups, "minimum_development_groups", minimum=3
    )
    unique_groups = sorted(set(group_ids))
    unique_heldout_groups = sorted(set(heldout_ids))
    if len(unique_groups) < checked_minimum_groups:
        raise GraphBiRCValidationError(
            f"at least {checked_minimum_groups} development groups are required"
        )
    overlap = sorted(set(unique_groups) & set(unique_heldout_groups))
    if overlap:
        raise GraphBiRCValidationError(
            f"development and held-out group IDs overlap: {overlap}"
        )

    evidence_arguments = (
        protocol_id,
        protocol_sha256,
        development_event_ids,
        development_event_sha256s,
        development_input_bundle_sha256s,
    )
    if any(value is not None for value in evidence_arguments) and not all(
        value is not None for value in evidence_arguments
    ):
        raise GraphBiRCValidationError(
            "protocol-bound fitting requires all development event evidence fields"
        )
    if protocol_id is None:
        checked_protocol_id = None
        checked_protocol_sha256 = None
        checked_event_ids: tuple[str, ...] = ()
        checked_event_sha256s: tuple[str, ...] = ()
        checked_input_sha256s: tuple[str, ...] = ()
    else:
        if (
            isinstance(development_event_ids, (str, bytes))
            or isinstance(development_event_sha256s, (str, bytes))
            or isinstance(development_input_bundle_sha256s, (str, bytes))
        ):
            raise GraphBiRCValidationError(
                "development event evidence fields must be sequences"
            )
        try:
            raw_event_ids = list(
                development_event_ids if development_event_ids is not None else ()
            )
            raw_event_sha256s = list(
                development_event_sha256s
                if development_event_sha256s is not None
                else ()
            )
            raw_input_sha256s = list(
                development_input_bundle_sha256s
                if development_input_bundle_sha256s is not None
                else ()
            )
        except TypeError as exc:
            raise GraphBiRCValidationError(
                "development event evidence fields must be sequences"
            ) from exc
        if not (
            len(raw_event_ids)
            == len(raw_event_sha256s)
            == len(raw_input_sha256s)
            == len(matrix)
        ):
            raise GraphBiRCValidationError(
                "protocol-bound fitting requires one evidence record per event"
            )
        checked_protocol_id = _identifier(protocol_id, "protocol_id")
        checked_protocol_sha256 = _validate_digest(protocol_sha256, "protocol_sha256")
        checked_event_ids = tuple(
            _identifier(value, "development_event_id") for value in raw_event_ids
        )
        checked_event_sha256s = tuple(
            _validate_digest(value, "development_event_sha256")
            for value in raw_event_sha256s
        )
        checked_input_sha256s = tuple(
            _validate_digest(value, "development_input_bundle_sha256")
            for value in raw_input_sha256s
        )

    checked_l2 = _finite_float(l2_strength, "l2_strength")
    if checked_l2 <= 0.0:
        raise GraphBiRCValidationError("l2_strength must be positive")
    checked_iterations = _positive_int(max_iterations, "max_iterations")
    checked_tolerance = _finite_float(tolerance, "tolerance")
    if checked_tolerance <= 0.0:
        raise GraphBiRCValidationError("tolerance must be positive")

    feature_mean_array = matrix.mean(axis=0)
    feature_scale_array = matrix.std(axis=0, ddof=0)
    feature_scale_array = np.where(
        feature_scale_array > 1e-12, feature_scale_array, 1.0
    )
    standardized = (matrix - feature_mean_array) / feature_scale_array
    parameters, optimizer_iterations = _fit_logistic_newton(
        standardized,
        labels,
        l2_strength=checked_l2,
        max_iterations=checked_iterations,
        tolerance=checked_tolerance,
    )

    development_payload = {
        "data_scope": _DATA_SCOPE,
        "feature_schema": list(GRAPH_BIRC_FEATURE_SCHEMA),
        "features": matrix.tolist(),
        "harm_labels": [int(value) for value in labels],
        "development_group_ids": group_ids,
        "development_event_evidence": [
            {
                "event_id": event_id,
                "event_artifact_sha256": event_sha256,
                "input_bundle_sha256": input_sha256,
            }
            for event_id, event_sha256, input_sha256 in zip(
                checked_event_ids,
                checked_event_sha256s,
                checked_input_sha256s,
            )
        ],
    }
    split_payload = {
        "scope_claim_status": _SCOPE_CLAIM_STATUS,
        "development_group_ids": unique_groups,
        "heldout_group_ids": unique_heldout_groups,
    }
    return GraphBiRCHarmScorer(
        feature_mean=tuple(float(value) for value in feature_mean_array),
        feature_scale=tuple(float(value) for value in feature_scale_array),
        coefficients=tuple(float(value) for value in parameters[1:]),
        intercept=float(parameters[0]),
        l2_strength=checked_l2,
        development_event_count=len(matrix),
        development_group_count=len(unique_groups),
        heldout_group_count=len(unique_heldout_groups),
        development_data_sha256=_sha256_canonical(development_payload),
        group_split_sha256=_sha256_canonical(split_payload),
        optimizer_iterations=optimizer_iterations,
        minimum_development_groups=checked_minimum_groups,
        max_iterations=checked_iterations,
        optimizer_tolerance=checked_tolerance,
        protocol_id=checked_protocol_id,
        protocol_sha256=checked_protocol_sha256,
        development_event_ids=checked_event_ids,
        development_event_sha256s=checked_event_sha256s,
        development_input_bundle_sha256s=checked_input_sha256s,
    )


__all__ = [
    "GRAPH_BIRC_FEATURE_SCHEMA_VERSION",
    "GRAPH_BIRC_SCORER_SCHEMA_VERSION",
    "GRAPH_BIRC_GROUP_ORDER",
    "GRAPH_BIRC_GROUP_FIELDS",
    "GRAPH_BIRC_FEATURE_SCHEMA",
    "GraphBiRCValidationError",
    "GraphBiRCGroupDiagnostics",
    "GraphBiRCFeatures",
    "GraphBiRCHarmScorer",
    "graph_birc_features",
    "fit_graph_birc_harm_scorer",
]
