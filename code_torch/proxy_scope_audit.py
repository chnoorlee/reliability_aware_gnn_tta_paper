"""Label-free proxy diagnostics on explicitly named node scopes.

The helpers in this module are observational only.  They receive probabilities
after a candidate has been produced and never expose labels to adaptation.
Degree groups are defined once on the full target graph and then intersected
with each requested scope so that a scope comparison changes only the nodes
being aggregated, not the structural partition.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable, Mapping
from typing import TypeVar

import numpy as np

_ScopeName = TypeVar("_ScopeName")


def canonical_scope_mask(scope, num_nodes: int) -> np.ndarray:
    """Return a non-empty boolean mask after strict scope validation."""

    if int(num_nodes) <= 0:
        raise ValueError("num_nodes must be positive")
    array = np.asarray(scope)
    if array.ndim == 1 and array.size == 0:
        raise ValueError("scope must contain at least one node")
    if array.dtype == np.bool_:
        if array.ndim != 1 or len(array) != num_nodes:
            raise ValueError("boolean scope must have shape [num_nodes]")
        mask = array.astype(bool, copy=True)
    else:
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise TypeError(
                "scope must be a one-dimensional bool mask or integer indices"
            )
        indices: np.ndarray = array.astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("scope indices must be unique")
        if np.any(indices < 0) or np.any(indices >= num_nodes):
            raise ValueError("scope index is outside [0, num_nodes)")
        mask = np.zeros(num_nodes, dtype=bool)
        mask[indices] = True
    if not np.any(mask):
        raise ValueError("scope must contain at least one node")
    return mask


def scope_index_sha256(scope, num_nodes: int) -> str:
    """Hash the canonical little-endian index list and its dimensions."""

    mask = canonical_scope_mask(scope, num_nodes)
    indices: np.ndarray = np.flatnonzero(mask).astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray([num_nodes, len(indices)], dtype="<i8").tobytes())
    digest.update(indices.tobytes())
    return digest.hexdigest()


def _prediction_sha256(predictions: np.ndarray, mask: np.ndarray) -> str:
    selected = np.asarray(predictions, dtype="<i8")[mask]
    digest = hashlib.sha256()
    digest.update(np.asarray([len(predictions), len(selected)], dtype="<i8").tobytes())
    digest.update(selected.tobytes())
    return digest.hexdigest()


def _probability_sha256(probabilities: np.ndarray, mask: np.ndarray) -> str:
    """Hash scoped probability values in a platform-stable representation."""

    selected = np.asarray(probabilities, dtype="<f8")[mask]
    digest = hashlib.sha256()
    digest.update(
        np.asarray(
            [probabilities.shape[0], probabilities.shape[1], selected.shape[0]],
            dtype="<i8",
        ).tobytes()
    )
    digest.update(np.ascontiguousarray(selected).tobytes())
    return digest.hexdigest()


def _canonical_groups(degree_groups, num_nodes: int) -> tuple[np.ndarray, ...]:
    values: Iterable = (
        degree_groups.values() if isinstance(degree_groups, Mapping) else degree_groups
    )
    group_list: list[np.ndarray] = []
    for value in values:
        array = np.asarray(value)
        if array.dtype == np.bool_:
            if array.ndim != 1 or len(array) != num_nodes:
                raise ValueError("boolean degree group must have shape [num_nodes]")
            indices: np.ndarray = np.flatnonzero(array).astype(np.int64, copy=False)
        else:
            if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
                raise TypeError("degree groups must be bool masks or integer indices")
            indices = array.astype(np.int64, copy=False)
            if len(np.unique(indices)) != len(indices):
                raise ValueError(
                    "degree-group indices must be unique within each group"
                )
            if np.any(indices < 0) or np.any(indices >= num_nodes):
                raise ValueError("degree-group index is outside [0, num_nodes)")
        if len(indices) == 0:
            raise ValueError("degree groups must be non-empty")
        group_list.append(indices)
    groups = tuple(group_list)
    if not groups:
        raise ValueError("degree_groups must contain at least one group")
    membership: np.ndarray = np.zeros(num_nodes, dtype=np.int64)
    for indices in groups:
        membership[indices] += 1
    if not np.all(membership == 1):
        raise ValueError("degree_groups must be a disjoint partition of all nodes")
    return groups


def scoped_proxy_signals(
    degree_groups,
    source_probs,
    candidate_probs,
    scope,
) -> dict:
    """Compute Delta and Phi on one scope with fixed full-graph groups."""

    source = np.asarray(source_probs)
    candidate = np.asarray(candidate_probs)
    if source.ndim != 2 or source.shape != candidate.shape:
        raise ValueError("source_probs and candidate_probs must share shape [N, C]")
    if source.shape[0] == 0 or source.shape[1] < 2:
        raise ValueError("probabilities must contain nodes and at least two classes")
    if not np.isfinite(source).all() or not np.isfinite(candidate).all():
        raise ValueError("probabilities must be finite")

    num_nodes = int(source.shape[0])
    mask = canonical_scope_mask(scope, num_nodes)
    groups = _canonical_groups(degree_groups, num_nodes)
    source_confidence = np.max(source, axis=1)
    candidate_confidence = np.max(candidate, axis=1)
    group_differences = []
    for group in groups:
        scoped_group = group[mask[group]]
        if len(scoped_group):
            group_differences.append(
                abs(
                    float(np.mean(candidate_confidence[scoped_group]))
                    - float(np.mean(source_confidence[scoped_group]))
                )
            )
    source_prediction = np.argmax(source, axis=1)
    candidate_prediction = np.argmax(candidate, axis=1)
    flip_count = int(
        np.count_nonzero(candidate_prediction[mask] != source_prediction[mask])
    )
    selected_count = int(np.count_nonzero(mask))
    return {
        "num_nodes": selected_count,
        "fraction_of_target_nodes": selected_count / num_nodes,
        "scope_index_sha256": scope_index_sha256(mask, num_nodes),
        "delta": float(np.mean(group_differences)),
        "phi": flip_count / selected_count,
        "flip_count": flip_count,
        "nonempty_degree_groups": len(group_differences),
        "source_probability_sha256": _probability_sha256(source, mask),
        "source_prediction_sha256": _prediction_sha256(source_prediction, mask),
        "candidate_prediction_sha256": _prediction_sha256(candidate_prediction, mask),
    }


def audit_proxy_scopes(
    degree_groups,
    source_probs,
    candidate_probs,
    scopes: Mapping[_ScopeName, object],
) -> dict[str, dict]:
    """Return proxy diagnostics for a non-empty mapping of named scopes."""

    normalized_scopes = _stringify_scope_names(scopes)
    return {
        name: scoped_proxy_signals(degree_groups, source_probs, candidate_probs, scope)
        for name, scope in normalized_scopes.items()
    }


def _stringify_scope_names(
    scopes: Mapping[_ScopeName, object],
) -> dict[str, object]:
    if not scopes:
        raise ValueError("scopes must contain at least one named scope")
    normalized = {}
    for raw_name, scope in scopes.items():
        name = str(raw_name)
        if name in normalized:
            raise ValueError("scope names must remain unique after string conversion")
        normalized[name] = scope
    return normalized


class ScopeTraceRecorder:
    """Record scope diagnostics outside the adaptation boundary.

    The recorder owns copies of the structural groups and scopes.  It is passed
    to adaptation only as an opaque callable and receives read-only probability
    snapshots after the operational proxy decision has been computed.
    """

    def __init__(self, degree_groups, scopes: Mapping[_ScopeName, object]):
        values: Iterable = (
            degree_groups.values()
            if isinstance(degree_groups, Mapping)
            else degree_groups
        )
        self._degree_groups = tuple(np.array(value, copy=True) for value in values)
        if not self._degree_groups:
            raise ValueError("degree_groups must contain at least one group")
        self._scopes = {
            name: np.array(scope, copy=True)
            for name, scope in _stringify_scope_names(scopes).items()
        }
        self._traces: dict[str, dict] = {}
        self._steps: list[int] = []

    def __call__(
        self,
        *,
        step,
        source_probs,
        candidate_probs,
        delta,
        phi,
    ) -> None:
        step = int(step)
        if step < 0:
            raise ValueError("observer step must be non-negative")
        if self._steps and step <= self._steps[-1]:
            raise ValueError("observer steps must be strictly increasing")

        scoped = audit_proxy_scopes(
            self._degree_groups,
            source_probs,
            candidate_probs,
            self._scopes,
        )
        if "target" in scoped and (
            scoped["target"]["delta"] != float(delta)
            or scoped["target"]["phi"] != float(phi)
        ):
            raise AssertionError(
                "target-scope diagnostics do not exactly reproduce operational proxies"
            )

        pending_new = {}
        pending_updates = []
        invariant_keys = (
            "num_nodes",
            "fraction_of_target_nodes",
            "scope_index_sha256",
            "source_prediction_sha256",
            "source_probability_sha256",
            "nonempty_degree_groups",
        )
        for name, signals in scoped.items():
            if name not in self._traces:
                pending_new[name] = {
                    "num_nodes": signals["num_nodes"],
                    "fraction_of_target_nodes": signals["fraction_of_target_nodes"],
                    "scope_index_sha256": signals["scope_index_sha256"],
                    "source_prediction_sha256": signals["source_prediction_sha256"],
                    "source_probability_sha256": signals["source_probability_sha256"],
                    "nonempty_degree_groups": signals["nonempty_degree_groups"],
                    "step_trace": [],
                    "delta_trace": [],
                    "phi_trace": [],
                    "flip_count_trace": [],
                    "max_delta": None,
                    "max_delta_step": None,
                    "max_delta_trace_index": None,
                    "max_delta_candidate_prediction_sha256": None,
                    "max_phi": None,
                    "max_phi_step": None,
                    "max_phi_trace_index": None,
                    "max_phi_candidate_prediction_sha256": None,
                }
            trace = self._traces[name] if name in self._traces else pending_new[name]
            for key in invariant_keys:
                if trace[key] != signals[key]:
                    raise AssertionError(
                        f"scope invariant changed during replay: {name}/{key}"
                    )

            pending_updates.append(
                {
                    "name": name,
                    "step": step,
                    "delta": signals["delta"],
                    "phi": signals["phi"],
                    "flip_count": signals["flip_count"],
                    "candidate_prediction_sha256": signals[
                        "candidate_prediction_sha256"
                    ],
                }
            )

        # Commit only after every scope has passed validation.  This prevents a
        # later-scope invariant failure from exposing a partial observer step.
        self._traces.update(pending_new)
        for update in pending_updates:
            trace = self._traces[update["name"]]

            trace_index = len(trace["delta_trace"])
            trace["step_trace"].append(update["step"])
            trace["delta_trace"].append(update["delta"])
            trace["phi_trace"].append(update["phi"])
            trace["flip_count_trace"].append(update["flip_count"])
            if trace["max_delta"] is None or update["delta"] > trace["max_delta"]:
                trace["max_delta"] = update["delta"]
                trace["max_delta_step"] = update["step"]
                trace["max_delta_trace_index"] = trace_index
                trace["max_delta_candidate_prediction_sha256"] = update[
                    "candidate_prediction_sha256"
                ]
            if trace["max_phi"] is None or update["phi"] > trace["max_phi"]:
                trace["max_phi"] = update["phi"]
                trace["max_phi_step"] = update["step"]
                trace["max_phi_trace_index"] = trace_index
                trace["max_phi_candidate_prediction_sha256"] = update[
                    "candidate_prediction_sha256"
                ]
        self._steps.append(step)
        return None

    def to_dict(self) -> dict[str, dict]:
        """Return a detached snapshot; incomplete failed calls are not exposed."""

        return copy.deepcopy(self._traces)
