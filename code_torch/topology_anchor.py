"""Topology-anchored, label-free conditional diagnostic for paired GNN outputs.

The target-side API in :func:`topology_anchor_certificate` never accepts target
labels. Its finite-sample interpretation is only model-conditional: disjoint
graph supports induce a proposed dependency graph, but topology alone does not
prove conditional independence. On one fixed target graph, the
anchor observation ``z=s`` is deterministic; Hoeffding coverage therefore
requires an explicitly declared repeated-measurement or target-graph sampling
law and separate audit-anchor measurements. Callers must acknowledge every
assumption, and an unacknowledged assumption makes the diagnostic fail closed.
Nothing in this module is distribution-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Iterable, Self

import numpy as np
from scipy import sparse

_GROUPING_GLOBAL = "global"
_GROUPING_DEGREE_TERCILES = "degree_terciles"
_GROUP_ORDER = {"all": 0, "low": 1, "mid": 2, "high": 3}
_RESULT_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class AnchorGroupCalibration:
    """A source-calibrated interval for the anchor channel in one group.

    ``lambda`` is the attenuation factor induced by a symmetric K-class anchor
    noise model.  Eligibility requires a strictly positive lower endpoint.
    """

    name: str
    lambda_lower: float
    lambda_upper: float
    sample_size: int
    color_count: int
    rho_hat: float | None
    rho_lower: float | None
    rho_upper: float | None
    eligible: bool
    reasons: tuple[str, ...] = ()
    eta_upper: float = 0.0

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ValueError("group name must be a non-empty string")
        lower = _native_float(self.lambda_lower, "lambda_lower")
        upper = _native_float(self.lambda_upper, "lambda_upper")
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError("lambda interval must satisfy 0 <= lower <= upper <= 1")
        sample_size = _native_int(self.sample_size, "sample_size", minimum=0)
        color_count = _native_int(self.color_count, "color_count", minimum=0)
        eligible = _native_bool(self.eligible, "eligible")
        reasons = _normalize_reasons(self.reasons)
        eta_upper = _native_float(self.eta_upper, "eta_upper")
        if not 0.0 <= eta_upper <= 2.0:
            raise ValueError("eta_upper must lie in [0, 2]")

        rho_values = (self.rho_lower, self.rho_hat, self.rho_upper)
        if sample_size == 0:
            if color_count != 0:
                raise ValueError("empty groups require color_count == 0")
            if any(value is not None for value in rho_values):
                raise ValueError("empty groups require all rho fields to be None")
            if lower != 0.0 or upper != 1.0:
                raise ValueError(
                    "empty groups require the vacuous lambda interval [0, 1]"
                )
        else:
            if not 1 <= color_count <= sample_size:
                raise ValueError(
                    "non-empty groups require 1 <= color_count <= sample_size"
                )
            if any(value is None for value in rho_values):
                raise ValueError("non-empty groups require all rho fields")
            rho_lower = _native_float(self.rho_lower, "rho_lower")
            rho_hat = _native_float(self.rho_hat, "rho_hat")
            rho_upper = _native_float(self.rho_upper, "rho_upper")
            if not 0.0 <= rho_lower <= rho_hat <= rho_upper <= 1.0:
                raise ValueError(
                    "rho fields must satisfy 0 <= lower <= hat <= upper <= 1"
                )

        if eligible:
            if sample_size == 0 or lower <= 0.0:
                raise ValueError(
                    "an eligible group requires samples and lambda_lower > 0"
                )
            if reasons:
                raise ValueError("eligible groups cannot contain failure reasons")
        elif not reasons:
            raise ValueError("ineligible groups require at least one failure reason")

        object.__setattr__(self, "lambda_lower", lower)
        object.__setattr__(self, "lambda_upper", upper)
        object.__setattr__(self, "sample_size", sample_size)
        object.__setattr__(self, "color_count", color_count)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "eta_upper", eta_upper)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class AnchorCalibration:
    """Frozen source-side calibration consumed by the target diagnostic."""

    num_classes: int
    confidence_threshold: float
    stability_threshold: float | None
    radius: int
    alpha: float
    audit_protocol_id: str
    grouping: str
    groups: tuple[AnchorGroupCalibration, ...]
    eligible: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        num_classes = _native_int(self.num_classes, "num_classes", minimum=2)
        threshold = _native_float(self.confidence_threshold, "confidence_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        stability = self.stability_threshold
        if stability is not None:
            stability = _native_float(stability, "stability_threshold")
            if not 0.0 <= stability <= 1.0:
                raise ValueError("stability_threshold must lie in [0, 1]")
        radius = _native_int(self.radius, "radius", minimum=0)
        alpha = _native_float(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if (
            type(self.audit_protocol_id) is not str
            or not self.audit_protocol_id.strip()
        ):
            raise ValueError("audit_protocol_id must be a non-empty string")
        if self.grouping not in {_GROUPING_GLOBAL, _GROUPING_DEGREE_TERCILES}:
            raise ValueError("grouping must be 'global' or 'degree_terciles'")
        eligible = _native_bool(self.eligible, "eligible")
        reasons = _normalize_reasons(self.reasons)

        if not isinstance(self.groups, (tuple, list)):
            raise TypeError("groups must be a tuple or list")
        groups = tuple(self.groups)
        if not all(isinstance(group, AnchorGroupCalibration) for group in groups):
            raise TypeError("groups must contain AnchorGroupCalibration values")
        if not groups:
            raise ValueError("at least one calibration group is required")
        names = [group.name for group in groups]
        if len(names) != len(set(names)):
            raise ValueError("calibration group names must be unique")
        expected = (
            {"all"} if self.grouping == _GROUPING_GLOBAL else {"low", "mid", "high"}
        )
        if set(names) != expected:
            raise ValueError(
                f"{self.grouping} calibration requires groups {sorted(expected)}"
            )
        canonical_groups: list[AnchorGroupCalibration] = []
        for group in groups:
            if group.sample_size == 0:
                canonical_groups.append(group)
                continue
            assert group.rho_lower is not None
            assert group.rho_hat is not None
            assert group.rho_upper is not None
            expected_lower = _lambda_from_rho(group.rho_lower, num_classes)
            expected_upper = _lambda_from_rho(group.rho_upper, num_classes)
            canonical_groups.append(
                AnchorGroupCalibration(
                    name=group.name,
                    lambda_lower=expected_lower,
                    lambda_upper=expected_upper,
                    sample_size=group.sample_size,
                    color_count=group.color_count,
                    rho_hat=group.rho_hat,
                    rho_lower=group.rho_lower,
                    rho_upper=group.rho_upper,
                    eligible=group.eligible,
                    reasons=group.reasons,
                    eta_upper=group.eta_upper,
                )
            )
        groups = tuple(canonical_groups)

        expected_eligible = any(group.eligible for group in groups)
        if eligible != expected_eligible:
            raise ValueError("calibration eligibility must match group eligibility")
        if eligible and reasons:
            raise ValueError("eligible calibration cannot contain failure reasons")
        if not eligible and not reasons:
            raise ValueError(
                "ineligible calibration requires at least one failure reason"
            )

        object.__setattr__(self, "num_classes", num_classes)
        object.__setattr__(self, "confidence_threshold", threshold)
        object.__setattr__(self, "stability_threshold", stability)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "reasons", reasons)

    def group_map(self) -> dict[str, AnchorGroupCalibration]:
        return {group.name: group for group in self.groups}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CertificateAssumptions:
    """Assumptions that cannot be established from unlabeled target data.

    A ``True`` value is an explicit caller acknowledgement, not evidence that
    the assumption holds.
    """

    frozen_post_selection_design: bool = False
    post_selection_symmetric_channel: bool = False
    source_to_target_transport: bool = False
    independent_audit_anchor_measurements: bool = False
    repeated_sampling_model: bool = False
    true_conditional_dependency_graph: bool = False

    def __post_init__(self) -> None:
        for name in (
            "frozen_post_selection_design",
            "post_selection_symmetric_channel",
            "source_to_target_transport",
            "independent_audit_anchor_measurements",
            "repeated_sampling_model",
            "true_conditional_dependency_graph",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a native Python boolean")

    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.frozen_post_selection_design:
            reasons.append("frozen_post_selection_design_not_acknowledged")
        if not self.post_selection_symmetric_channel:
            reasons.append("post_selection_symmetric_channel_not_acknowledged")
        if not self.source_to_target_transport:
            reasons.append("source_to_target_transport_not_acknowledged")
        if not self.independent_audit_anchor_measurements:
            reasons.append("independent_audit_anchor_measurements_not_acknowledged")
        if not self.repeated_sampling_model:
            reasons.append("repeated_sampling_model_not_acknowledged")
        if not self.true_conditional_dependency_graph:
            reasons.append("true_conditional_dependency_graph_not_acknowledged")
        return tuple(reasons)


@dataclass(frozen=True, init=False)
class GroupDiagnostic:
    group: str
    color: int
    dependency_degree: int
    node_indices: tuple[int, ...]
    size: int
    mean_score: float
    hoeffding_radius: float
    score_lower: float
    eta_upper: float
    adjusted_score_lower: float
    lambda_lower: float
    lambda_upper: float
    gain_lower: float
    contribution: float

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "GroupDiagnostic instances are created only by topology_anchor_certificate"
        )

    @classmethod
    def _from_factory(cls, *, _token: object, **values: object) -> Self:
        if _token is not _RESULT_FACTORY_TOKEN:
            raise TypeError("invalid GroupDiagnostic construction provenance")
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise TypeError("internal GroupDiagnostic fields are incomplete")
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        if type(self.group) is not str or self.group not in _GROUP_ORDER:
            raise ValueError("group must be one of all, low, mid, or high")
        color = _native_int(self.color, "color", minimum=-1)
        if color != -1:
            raise ValueError("group diagnostics must use aggregate color -1")
        dependency_degree = _native_int(
            self.dependency_degree, "dependency_degree", minimum=0
        )
        if type(self.node_indices) is not tuple:
            raise TypeError("node_indices must be a tuple of native Python integers")
        if any(type(node) is not int or node < 0 for node in self.node_indices):
            raise ValueError("node_indices must contain non-negative native integers")
        if tuple(sorted(self.node_indices)) != self.node_indices:
            raise ValueError("node_indices must be strictly increasing")
        if len(set(self.node_indices)) != len(self.node_indices):
            raise ValueError("node_indices must be unique")
        size = _native_int(self.size, "size", minimum=1)
        if size != len(self.node_indices):
            raise ValueError("size must equal len(node_indices)")
        if dependency_degree >= size:
            raise ValueError("dependency_degree must be smaller than size")

        mean_score = _native_float(self.mean_score, "mean_score")
        if not -1.0 <= mean_score <= 1.0:
            raise ValueError("mean_score must lie in [-1, 1]")
        hoeffding_radius = _native_float(self.hoeffding_radius, "hoeffding_radius")
        if hoeffding_radius < 0.0:
            raise ValueError("hoeffding_radius must be non-negative")
        score_lower = _native_float(self.score_lower, "score_lower")
        if score_lower != mean_score - hoeffding_radius:
            raise ValueError("score_lower must equal mean_score - hoeffding_radius")
        eta_upper = _native_float(self.eta_upper, "eta_upper")
        if not 0.0 <= eta_upper <= 2.0:
            raise ValueError("eta_upper must lie in [0, 2]")
        adjusted_score_lower = _native_float(
            self.adjusted_score_lower, "adjusted_score_lower"
        )
        if adjusted_score_lower != score_lower - eta_upper:
            raise ValueError("adjusted_score_lower must equal score_lower - eta_upper")
        lambda_lower = _native_float(self.lambda_lower, "lambda_lower")
        lambda_upper = _native_float(self.lambda_upper, "lambda_upper")
        if not 0.0 < lambda_lower <= lambda_upper <= 1.0:
            raise ValueError(
                "diagnostic lambda interval must satisfy 0 < lower <= upper <= 1"
            )
        transformed = adjusted_score_lower / (
            lambda_upper if adjusted_score_lower >= 0.0 else lambda_lower
        )
        expected_gain = min(1.0, max(-1.0, transformed))
        gain_lower = _native_float(self.gain_lower, "gain_lower")
        if gain_lower != expected_gain:
            raise ValueError("gain_lower is inconsistent with the adjusted score")
        contribution = _native_float(self.contribution, "contribution")
        if contribution != size * gain_lower:
            raise ValueError("contribution must equal size * gain_lower")

        object.__setattr__(self, "color", color)
        object.__setattr__(self, "dependency_degree", dependency_degree)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "mean_score", mean_score)
        object.__setattr__(self, "hoeffding_radius", hoeffding_radius)
        object.__setattr__(self, "score_lower", score_lower)
        object.__setattr__(self, "eta_upper", eta_upper)
        object.__setattr__(self, "adjusted_score_lower", adjusted_score_lower)
        object.__setattr__(self, "lambda_lower", lambda_lower)
        object.__setattr__(self, "lambda_upper", lambda_upper)
        object.__setattr__(self, "gain_lower", gain_lower)
        object.__setattr__(self, "contribution", contribution)


@dataclass(frozen=True, init=False)
class TopologyAnchorCertificate:
    """Structured diagnostic result, not evidence that assumptions are true.

    ``accepted`` is only the configured policy decision under the acknowledged
    model. ``mode`` and ``statistical_scope`` distinguish that conditional path
    from the fail-closed monitor path.
    """

    mode: str
    lower_bound: float
    accepted: bool
    eligible: bool
    coverage: float
    coverage_definition: str
    colors: tuple[int, ...]
    group_diagnostics: tuple[GroupDiagnostic, ...]
    num_nodes: int
    num_changed: int
    num_covered: int
    num_uncovered: int
    num_colors: int
    delta: float
    conditional_failure_budget: float | None
    min_gain: float
    reasons: tuple[str, ...]
    assumptions: CertificateAssumptions
    statistical_scope: str
    distribution_free: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "TopologyAnchorCertificate instances are created only by "
            "topology_anchor_certificate"
        )

    @classmethod
    def _from_factory(cls, *, _token: object, **values: object) -> Self:
        if _token is not _RESULT_FACTORY_TOKEN:
            raise TypeError("invalid certificate construction provenance")
        expected = {field.name for field in fields(cls)}
        if set(values) != expected:
            raise TypeError("internal certificate fields are incomplete")
        instance = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(instance, name, value)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        conditional_mode = "conditional_model_diagnostic"
        monitor_mode = "heuristic_monitor_only"
        if type(self.mode) is not str or self.mode not in {
            conditional_mode,
            monitor_mode,
        }:
            raise ValueError("mode is not a supported diagnostic mode")
        lower_bound = _native_float(self.lower_bound, "lower_bound")
        if not -1.0 <= lower_bound <= 1.0:
            raise ValueError("lower_bound must lie in [-1, 1]")
        accepted = _native_bool(self.accepted, "accepted")
        eligible = _native_bool(self.eligible, "eligible")
        coverage = _native_float(self.coverage, "coverage")
        if not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage must lie in [0, 1]")
        expected_coverage_definition = "covered_changed_nodes / changed_nodes"
        if (
            type(self.coverage_definition) is not str
            or self.coverage_definition != expected_coverage_definition
        ):
            raise ValueError("coverage_definition is not recognized")

        num_nodes = _native_int(self.num_nodes, "num_nodes", minimum=1)
        num_changed = _native_int(self.num_changed, "num_changed", minimum=0)
        num_covered = _native_int(self.num_covered, "num_covered", minimum=0)
        num_uncovered = _native_int(self.num_uncovered, "num_uncovered", minimum=0)
        num_colors = _native_int(self.num_colors, "num_colors", minimum=0)
        if num_changed > num_nodes:
            raise ValueError("num_changed cannot exceed num_nodes")
        if num_covered + num_uncovered != num_changed:
            raise ValueError("covered and uncovered counts must sum to num_changed")
        expected_coverage = 1.0 if num_changed == 0 else num_covered / num_changed
        if coverage != expected_coverage:
            raise ValueError("coverage is inconsistent with the reported counts")

        if type(self.colors) is not tuple:
            raise TypeError("colors must be a tuple of native Python integers")
        colors = self.colors
        if len(colors) != num_nodes:
            raise ValueError("colors must have one entry per node")
        if any(type(color) is not int or color < -1 for color in colors):
            raise ValueError("colors must contain native integers no smaller than -1")
        used_colors = {color for color in colors if color >= 0}
        if used_colors != set(range(num_colors)):
            raise ValueError(
                "non-negative colors must be contiguous and match num_colors"
            )

        if type(self.group_diagnostics) is not tuple:
            raise TypeError("group_diagnostics must be a tuple")
        diagnostics = self.group_diagnostics
        if any(type(item) is not GroupDiagnostic for item in diagnostics):
            raise TypeError("group_diagnostics must contain GroupDiagnostic values")
        group_names = [item.group for item in diagnostics]
        if len(group_names) != len(set(group_names)):
            raise ValueError("diagnostic group names must be unique")
        diagnostic_nodes = tuple(
            node for diagnostic in diagnostics for node in diagnostic.node_indices
        )
        if len(diagnostic_nodes) != len(set(diagnostic_nodes)):
            raise ValueError("diagnostic node sets must be disjoint")
        if any(node >= num_nodes for node in diagnostic_nodes):
            raise ValueError("diagnostic node index is outside the graph")
        colored_nodes = tuple(index for index, color in enumerate(colors) if color >= 0)
        if set(diagnostic_nodes) != set(colored_nodes):
            raise ValueError(
                "diagnostics and non-negative color entries must cover the same nodes"
            )
        if len(diagnostic_nodes) != num_covered:
            raise ValueError("diagnostic node count must equal num_covered")

        delta = _native_float(self.delta, "delta")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between zero and one")
        min_gain = _native_float(self.min_gain, "min_gain")
        if not 0.0 <= min_gain <= 1.0:
            raise ValueError("min_gain must lie in [0, 1]")
        if type(self.reasons) is not tuple:
            raise TypeError("reasons must be a tuple of native Python strings")
        reasons = _normalize_reasons(self.reasons)
        if len(reasons) != len(set(reasons)):
            raise ValueError("reasons must be unique")
        if type(self.assumptions) is not CertificateAssumptions:
            raise TypeError("assumptions must be a CertificateAssumptions")
        if type(self.statistical_scope) is not str:
            raise TypeError("statistical_scope must be a native Python string")
        distribution_free = _native_bool(self.distribution_free, "distribution_free")
        if distribution_free:
            raise ValueError("topology-anchor diagnostics are never distribution-free")

        is_conditional = self.mode == conditional_mode
        if eligible != is_conditional:
            raise ValueError(
                "eligible must be true exactly in conditional diagnostic mode"
            )
        expected_accepted = (
            eligible and is_conditional and num_covered > 0 and lower_bound > min_gain
        )
        if accepted != expected_accepted:
            raise ValueError(
                "accepted is inconsistent with mode, coverage, and lower bound"
            )

        if is_conditional:
            self._validate_conditional_state(
                lower_bound=lower_bound,
                accepted=accepted,
                num_nodes=num_nodes,
                num_changed=num_changed,
                num_covered=num_covered,
                num_uncovered=num_uncovered,
                num_colors=num_colors,
                diagnostics=diagnostics,
                reasons=reasons,
            )
        else:
            self._validate_monitor_state(
                lower_bound=lower_bound,
                num_nodes=num_nodes,
                num_changed=num_changed,
                num_covered=num_covered,
                num_uncovered=num_uncovered,
                num_colors=num_colors,
                diagnostics=diagnostics,
                colors=colors,
                reasons=reasons,
            )

        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "num_changed", num_changed)
        object.__setattr__(self, "num_covered", num_covered)
        object.__setattr__(self, "num_uncovered", num_uncovered)
        object.__setattr__(self, "num_colors", num_colors)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "min_gain", min_gain)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "distribution_free", distribution_free)

    def _validate_conditional_state(
        self,
        *,
        lower_bound: float,
        accepted: bool,
        num_nodes: int,
        num_changed: int,
        num_covered: int,
        num_uncovered: int,
        num_colors: int,
        diagnostics: tuple[GroupDiagnostic, ...],
        reasons: tuple[str, ...],
    ) -> None:
        expected_scope = (
            "conditional_on_acknowledged_post_selection_measurement_assumptions"
        )
        if self.statistical_scope != expected_scope:
            raise ValueError("statistical_scope does not match conditional mode")
        budget = self.conditional_failure_budget
        if budget is None:
            raise ValueError("conditional mode requires a failure budget")
        budget = _native_float(budget, "conditional_failure_budget")
        if not 0.0 < budget < 1.0:
            raise ValueError("conditional_failure_budget must lie strictly in (0, 1)")
        if budget <= self.delta:
            raise ValueError("conditional_failure_budget must be greater than delta")
        if self.assumptions.failure_reasons():
            raise ValueError("conditional mode requires every assumption acknowledged")
        if (num_covered > 0) != bool(diagnostics):
            raise ValueError(
                "conditional diagnostics must exist exactly when nodes are covered"
            )
        if (num_covered > 0) != (num_colors > 0):
            raise ValueError(
                "num_colors must be positive exactly when nodes are covered"
            )
        expected_bound = (
            sum(diagnostic.contribution for diagnostic in diagnostics) - num_uncovered
        ) / num_nodes
        expected_bound = min(1.0, max(-1.0, expected_bound))
        if lower_bound != expected_bound:
            raise ValueError(
                "lower_bound is inconsistent with diagnostic contributions"
            )

        required_flags = {
            "no_changed_nodes": num_changed == 0,
            "uncovered_changed_nodes": num_uncovered > 0,
            "no_covered_changed_nodes": num_changed > 0 and num_covered == 0,
            "conditional_model_diagnostic_passed": accepted,
            "lower_bound_not_above_min_gain": not accepted,
        }
        for reason, required in required_flags.items():
            if (reason in reasons) != required:
                raise ValueError(
                    f"reason {reason} is inconsistent with certificate state"
                )
        allowed = set(required_flags)
        if any(
            reason not in allowed
            and not reason.startswith("calibration_group_ineligible:")
            for reason in reasons
        ):
            raise ValueError("conditional mode contains an unrecognized reason")
        if (
            any(
                reason.startswith("calibration_group_ineligible:") for reason in reasons
            )
            and num_uncovered == 0
        ):
            raise ValueError("ineligible calibration groups require uncovered nodes")
        object.__setattr__(self, "conditional_failure_budget", budget)

    def _validate_monitor_state(
        self,
        *,
        lower_bound: float,
        num_nodes: int,
        num_changed: int,
        num_covered: int,
        num_uncovered: int,
        num_colors: int,
        diagnostics: tuple[GroupDiagnostic, ...],
        colors: tuple[int, ...],
        reasons: tuple[str, ...],
    ) -> None:
        expected_scope = "heuristic_monitor_only_no_finite_sample_claim"
        if self.statistical_scope != expected_scope:
            raise ValueError("statistical_scope does not match monitor mode")
        if self.conditional_failure_budget is not None:
            raise ValueError("monitor mode cannot report a conditional failure budget")
        if diagnostics or num_covered != 0 or num_uncovered != num_changed:
            raise ValueError("monitor mode cannot report covered diagnostics")
        if num_colors != 0 or any(color != -1 for color in colors):
            raise ValueError("monitor mode cannot report diagnostic colors")
        expected_bound = -float(num_changed) / num_nodes
        if lower_bound != expected_bound:
            raise ValueError("monitor lower_bound must be the fail-closed worst case")

        assumption_reasons = set(self.assumptions.failure_reasons())
        allowed_blockers = assumption_reasons | {
            "calibration_ineligible",
            "independent_audit_anchor_probabilities_missing",
            "audit_protocol_id_missing",
            "audit_protocol_id_mismatch",
            "noninformative_total_failure_budget",
        }
        if not allowed_blockers.intersection(reasons):
            raise ValueError("monitor mode requires at least one fail-closed blocker")
        if not assumption_reasons.issubset(reasons):
            raise ValueError("monitor reasons omit an unacknowledged assumption")
        allowed_reasons = allowed_blockers | {"no_changed_nodes"}
        if any(reason not in allowed_reasons for reason in reasons):
            raise ValueError("monitor mode contains an unrecognized reason")
        if ("no_changed_nodes" in reasons) != (num_changed == 0):
            raise ValueError("no_changed_nodes reason is inconsistent with counts")

    def to_dict(self) -> dict:
        return asdict(self)


def _make_group_diagnostic(
    *,
    group: str,
    dependency_degree: int,
    node_indices: tuple[int, ...],
    mean_score: float,
    hoeffding_radius: float,
    eta_upper: float,
    lambda_lower: float,
    lambda_upper: float,
) -> GroupDiagnostic:
    score_lower = mean_score - hoeffding_radius
    adjusted_score_lower = score_lower - eta_upper
    denominator = lambda_upper if adjusted_score_lower >= 0.0 else lambda_lower
    gain_lower = min(1.0, max(-1.0, adjusted_score_lower / denominator))
    size = len(node_indices)
    return GroupDiagnostic._from_factory(
        _token=_RESULT_FACTORY_TOKEN,
        group=group,
        color=-1,
        dependency_degree=dependency_degree,
        node_indices=node_indices,
        size=size,
        mean_score=mean_score,
        hoeffding_radius=hoeffding_radius,
        score_lower=score_lower,
        eta_upper=eta_upper,
        adjusted_score_lower=adjusted_score_lower,
        lambda_lower=lambda_lower,
        lambda_upper=lambda_upper,
        gain_lower=gain_lower,
        contribution=size * gain_lower,
    )


def _make_certificate(
    *,
    conditional: bool,
    colors: tuple[int, ...],
    diagnostics: tuple[GroupDiagnostic, ...],
    num_changed: int,
    delta: float,
    min_gain: float,
    assumptions: CertificateAssumptions,
    reasons: Iterable[str],
    calibration_alpha: float | None = None,
) -> TopologyAnchorCertificate:
    num_nodes = len(colors)
    num_covered = sum(diagnostic.size for diagnostic in diagnostics)
    num_uncovered = num_changed - num_covered
    coverage = 1.0 if num_changed == 0 else num_covered / num_changed
    result_reasons = list(reasons)
    if conditional:
        lower_bound = (
            sum(diagnostic.contribution for diagnostic in diagnostics) - num_uncovered
        ) / num_nodes
        lower_bound = min(1.0, max(-1.0, lower_bound))
        accepted = num_covered > 0 and lower_bound > min_gain
        result_reasons.append(
            "conditional_model_diagnostic_passed"
            if accepted
            else "lower_bound_not_above_min_gain"
        )
        if calibration_alpha is None:
            raise TypeError("conditional certificates require calibration_alpha")
        failure_budget = calibration_alpha + delta
        mode = "conditional_model_diagnostic"
        scope = "conditional_on_acknowledged_post_selection_measurement_assumptions"
        num_colors = max(colors) + 1 if num_covered else 0
    else:
        lower_bound = -float(num_changed) / num_nodes
        accepted = False
        failure_budget = None
        mode = "heuristic_monitor_only"
        scope = "heuristic_monitor_only_no_finite_sample_claim"
        num_colors = 0

    return TopologyAnchorCertificate._from_factory(
        _token=_RESULT_FACTORY_TOKEN,
        mode=mode,
        lower_bound=lower_bound,
        accepted=accepted,
        eligible=conditional,
        coverage=coverage,
        coverage_definition="covered_changed_nodes / changed_nodes",
        colors=colors,
        group_diagnostics=diagnostics,
        num_nodes=num_nodes,
        num_changed=num_changed,
        num_covered=num_covered,
        num_uncovered=num_uncovered,
        num_colors=num_colors,
        delta=delta,
        conditional_failure_budget=failure_budget,
        min_gain=min_gain,
        reasons=_unique_tuple(result_reasons),
        assumptions=assumptions,
        statistical_scope=scope,
        distribution_free=False,
    )


def calibrate_anchor_channel(
    adjacency,
    frozen_source_probabilities,
    source_validation_labels,
    source_validation_indices,
    *,
    source_audit_anchor_probabilities,
    audit_protocol_id: str,
    confidence_threshold: float = 0.8,
    precomputed_source_stability=None,
    stability_threshold: float | None = None,
    radius: int = 0,
    alpha: float = 0.05,
    grouping: str = _GROUPING_DEGREE_TERCILES,
    min_group_size: int = 1,
    eta_upper: float = 0.0,
) -> AnchorCalibration:
    """Estimate source anchor-channel intervals from labeled source validation data.

    Frozen source probabilities determine confidence/stability selection only.
    Channel outcomes come from the separately supplied source-audit stream.
    The interval uses a dependency-graph Hoeffding bound with the graph's
    maximum degree. It is valid only under the declared repeated source sampling
    law and a true conditional dependency graph. ``audit_protocol_id`` records
    provenance but neither that string nor later boolean acknowledgements prove
    independence, cross-fitting, or source-to-target transport. ``eta_upper`` is
    caller supplied; this routine does not estimate or validate it.
    """

    probabilities = _validate_probabilities(
        frozen_source_probabilities, "frozen_source_probabilities"
    )
    node_count, num_classes = probabilities.shape
    audit_probabilities = _validate_probabilities(
        source_audit_anchor_probabilities, "source_audit_anchor_probabilities"
    )
    if audit_probabilities.shape != probabilities.shape:
        raise ValueError(
            "source audit and frozen source probabilities must have identical shapes"
        )
    audit_protocol_id = _validate_protocol_id(audit_protocol_id)
    neighbors, degrees = _validate_adjacency(adjacency, node_count)
    confidence_threshold = _unit_interval(confidence_threshold, "confidence_threshold")
    alpha = _open_unit_interval(alpha, "alpha")
    radius = _non_negative_integer(radius, "radius")
    if grouping not in {_GROUPING_GLOBAL, _GROUPING_DEGREE_TERCILES}:
        raise ValueError("grouping must be 'global' or 'degree_terciles'")
    if (
        not isinstance(min_group_size, (int, np.integer))
        or isinstance(min_group_size, (bool, np.bool_))
        or min_group_size < 1
    ):
        raise ValueError("min_group_size must be a positive integer")
    min_group_size = int(min_group_size)
    eta_upper = _finite_scalar(eta_upper, "eta_upper")
    if not 0.0 <= eta_upper <= 2.0:
        raise ValueError("eta_upper must lie in [0, 2]")
    if stability_threshold is not None:
        stability_threshold = _unit_interval(stability_threshold, "stability_threshold")

    validation_indices = _normalize_indices(source_validation_indices, node_count)
    validation_labels = _normalize_source_labels(
        source_validation_labels, validation_indices, node_count, num_classes
    )
    audit_predictions = np.argmax(audit_probabilities, axis=1)
    anchor_mask = np.max(probabilities, axis=1) >= confidence_threshold
    stability = _validate_stability(
        precomputed_source_stability,
        node_count,
        required=stability_threshold is not None,
    )
    if stability_threshold is not None:
        assert stability is not None
        anchor_mask &= stability >= stability_threshold

    group_names = _assign_groups(degrees, grouping)
    endpoint_scores: dict[int, float] = {}
    endpoint_anchors: dict[int, np.ndarray] = {}
    endpoint_label_map = {
        int(index): int(label)
        for index, label in zip(validation_indices, validation_labels)
    }
    for endpoint in validation_indices:
        endpoint = int(endpoint)
        anchors = neighbors[endpoint][anchor_mask[neighbors[endpoint]]]
        if anchors.size == 0:
            continue
        endpoint_anchors[endpoint] = anchors
        endpoint_scores[endpoint] = float(
            np.mean(audit_predictions[anchors] == endpoint_label_map[endpoint])
        )

    supports = {
        endpoint: _expanded_support(
            (endpoint, *endpoint_anchors[endpoint].tolist()), neighbors, radius
        )
        for endpoint in endpoint_scores
    }
    dependency_graph = _build_dependency_graph(sorted(endpoint_scores), supports)
    colors = _greedy_color(sorted(endpoint_scores), dependency_graph)
    expected_groups = (
        ("all",) if grouping == _GROUPING_GLOBAL else ("low", "mid", "high")
    )
    grouped_nodes = {
        group_name: [
            endpoint
            for endpoint in sorted(endpoint_scores)
            if str(group_names[endpoint]) == group_name
        ]
        for group_name in expected_groups
    }
    simultaneous_group_count = sum(bool(nodes) for nodes in grouped_nodes.values())

    calibrated_groups: list[AnchorGroupCalibration] = []
    for group_name in expected_groups:
        nodes = grouped_nodes[group_name]
        sample_size = len(nodes)
        if sample_size == 0:
            calibrated_groups.append(
                AnchorGroupCalibration(
                    name=group_name,
                    lambda_lower=0.0,
                    lambda_upper=1.0,
                    sample_size=0,
                    color_count=0,
                    rho_hat=None,
                    rho_lower=None,
                    rho_upper=None,
                    eligible=False,
                    reasons=("no_covered_source_validation_nodes",),
                    eta_upper=eta_upper,
                )
            )
            continue

        dependency_degree = _group_dependency_degree(nodes, dependency_graph)
        rho_hat = float(np.mean([endpoint_scores[node] for node in nodes]))
        radius_bound = math.sqrt(
            (dependency_degree + 1)
            * math.log(2.0 * simultaneous_group_count / alpha)
            / (2.0 * sample_size)
        )
        rho_lower = max(0.0, rho_hat - radius_bound)
        rho_upper = min(1.0, rho_hat + radius_bound)
        lambda_lower = max(0.0, (num_classes * rho_lower - 1.0) / (num_classes - 1.0))
        lambda_upper = max(0.0, (num_classes * rho_upper - 1.0) / (num_classes - 1.0))
        reasons: list[str] = []
        if sample_size < min_group_size:
            reasons.append("source_calibration_group_too_small")
        if lambda_lower <= 0.0:
            reasons.append("anchor_signal_not_above_chance")
        group_eligible = not reasons
        calibrated_groups.append(
            AnchorGroupCalibration(
                name=group_name,
                lambda_lower=lambda_lower,
                lambda_upper=lambda_upper,
                sample_size=sample_size,
                color_count=len({colors[node] for node in nodes}),
                rho_hat=rho_hat,
                rho_lower=rho_lower,
                rho_upper=rho_upper,
                eligible=group_eligible,
                reasons=tuple(reasons),
                eta_upper=eta_upper,
            )
        )

    calibration_eligible = any(group.eligible for group in calibrated_groups)
    calibration_reasons: list[str] = []
    if not calibration_eligible:
        calibration_reasons.append("no_eligible_calibration_group")
    return AnchorCalibration(
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        stability_threshold=stability_threshold,
        radius=radius,
        alpha=alpha,
        audit_protocol_id=audit_protocol_id,
        grouping=grouping,
        groups=tuple(calibrated_groups),
        eligible=calibration_eligible,
        reasons=tuple(calibration_reasons),
    )


def topology_anchor_certificate(
    adjacency,
    frozen_source_probabilities,
    candidate_probabilities,
    calibration: AnchorCalibration,
    *,
    assumptions: CertificateAssumptions | None = None,
    source_stability=None,
    audit_anchor_probabilities=None,
    audit_protocol_id: str | None = None,
    delta: float = 0.05,
    min_gain: float = 0.0,
) -> TopologyAnchorCertificate:
    """Compute a model-conditional paired-gain diagnostic without target labels.

    For every changed node ``i``, anchors are its confidence-qualified graph
    neighbors. With frozen source prediction ``s``, candidate prediction ``c``,
    and anchor set ``A_i``, the observed topology score is

    ``X_i = mean_{j in A_i}[1{c_i=z_j} - 1{s_i=z_j}]``,

    where ``z`` must come from the separately supplied audit-anchor
    probabilities. The original frozen source output is not reused as ``z``.

    Changed nodes conflict when their expanded supports overlap. The bound uses
    the maximum degree of that dependency graph, which is invariant to a
    synchronous node permutation. Deterministic greedy colors are returned only
    as diagnostics and are not used to assert independence or set the bound.
    A simultaneous one-sided dependency-graph Hoeffding bound is transformed
    through the calibrated lambda interval. Every uncovered changed node
    contributes the worst possible paired gain, -1.

    The channel identity, with misspecification allowance ``eta_b``, is a
    post-selection assumption. Let ``F_sel`` contain
    graph topology, confidence/stability eligibility, changed/anchor membership,
    grouping, and coloring, but not the realized anchor categorical measurements
    used in ``X``. Conditional on ``F_sel`` and endpoint truth, each independent
    audit-anchor measurement must follow the stated symmetric K-class channel,
    up to ``eta_b`` in score expectation. On a single observed graph the frozen
    ``z=s`` is already deterministic, so it is never substituted for a missing
    audit measurement. The caller must also posit a repeated-measurement or
    target-graph sampling law. Global GNN
    normalization or globally coupled adaptation can invalidate finite-radius
    dependency supports; overlap coloring does not repair that violation.

    The returned bound is conditional on all acknowledged assumptions; the
    acknowledgements and matching ``audit_protocol_id`` are not verified
    evidence of independence, randomness, or transport. ``accepted`` is a policy
    gate under that model, not proof that adaptation improves accuracy. A
    rejection only means that this diagnostic did not establish improvement;
    it is not evidence that adaptation is harmful. If the frozen source
    calibration interval has failure probability at most ``alpha``, a union
    bound gives the reported conditional failure budget ``alpha + delta`` when
    that sum is strictly below one;
    that number has no validity when the stated assumptions do not hold.
    """

    if not isinstance(calibration, AnchorCalibration):
        raise TypeError("calibration must be an AnchorCalibration")
    assumptions = assumptions or CertificateAssumptions()
    if not isinstance(assumptions, CertificateAssumptions):
        raise TypeError("assumptions must be a CertificateAssumptions")
    source = _validate_probabilities(
        frozen_source_probabilities, "frozen_source_probabilities"
    )
    candidate = _validate_probabilities(
        candidate_probabilities, "candidate_probabilities"
    )
    if source.shape != candidate.shape:
        raise ValueError(
            "source and candidate probabilities must have identical shapes"
        )
    node_count, num_classes = source.shape
    if num_classes != calibration.num_classes:
        raise ValueError("probability class count does not match calibration")
    neighbors, degrees = _validate_adjacency(adjacency, node_count)
    audit = None
    if audit_anchor_probabilities is not None:
        audit = _validate_probabilities(
            audit_anchor_probabilities, "audit_anchor_probabilities"
        )
        if audit.shape != source.shape:
            raise ValueError(
                "audit anchor probabilities must match source probability shape"
            )
    delta = _open_unit_interval(delta, "delta")
    min_gain = _finite_scalar(min_gain, "min_gain")
    if not 0.0 <= min_gain <= 1.0:
        raise ValueError("min_gain must lie in [0, 1]")

    stability = _validate_stability(
        source_stability,
        node_count,
        required=calibration.stability_threshold is not None,
    )
    source_predictions = np.argmax(source, axis=1)
    candidate_predictions = np.argmax(candidate, axis=1)
    changed_nodes = np.flatnonzero(source_predictions != candidate_predictions)
    num_changed = int(changed_nodes.size)
    colors_array = np.full(node_count, -1, dtype=int)

    failure_reasons = list(assumptions.failure_reasons())
    if not calibration.eligible:
        failure_reasons.append("calibration_ineligible")
    if audit is None:
        failure_reasons.append("independent_audit_anchor_probabilities_missing")
    if audit_protocol_id is None or type(audit_protocol_id) is not str:
        failure_reasons.append("audit_protocol_id_missing")
    elif not audit_protocol_id.strip():
        failure_reasons.append("audit_protocol_id_missing")
    elif audit_protocol_id != calibration.audit_protocol_id:
        failure_reasons.append("audit_protocol_id_mismatch")
    if calibration.alpha + delta >= 1.0:
        failure_reasons.append("noninformative_total_failure_budget")
    if failure_reasons:
        if num_changed == 0:
            failure_reasons.append("no_changed_nodes")
        return _make_certificate(
            conditional=False,
            colors=tuple(int(color) for color in colors_array),
            diagnostics=(),
            num_changed=num_changed,
            delta=delta,
            min_gain=min_gain,
            reasons=_unique_tuple(failure_reasons),
            assumptions=assumptions,
        )

    assert audit is not None
    audit_predictions = np.argmax(audit, axis=1)
    anchor_mask = np.max(source, axis=1) >= calibration.confidence_threshold
    if calibration.stability_threshold is not None:
        assert stability is not None
        anchor_mask &= stability >= calibration.stability_threshold
    group_names = _assign_groups(degrees, calibration.grouping)
    calibration_groups = calibration.group_map()

    scores: dict[int, float] = {}
    supports: dict[int, frozenset[int]] = {}
    uncovered: list[int] = []
    uncovered_group_reasons: set[str] = set()
    for node_value in changed_nodes:
        node = int(node_value)
        group = calibration_groups[str(group_names[node])]
        if not group.eligible:
            uncovered.append(node)
            uncovered_group_reasons.add(f"calibration_group_ineligible:{group.name}")
            continue
        anchors = neighbors[node][anchor_mask[neighbors[node]]]
        if anchors.size == 0:
            uncovered.append(node)
            continue
        candidate_matches = candidate_predictions[node] == audit_predictions[anchors]
        source_matches = source_predictions[node] == audit_predictions[anchors]
        scores[node] = float(
            np.mean(candidate_matches.astype(float) - source_matches.astype(float))
        )
        supports[node] = _expanded_support(
            (node, *anchors.tolist()), neighbors, calibration.radius
        )

    dependency_graph = _build_dependency_graph(sorted(scores), supports)
    color_map = _greedy_color(sorted(scores), dependency_graph)
    for node, color in color_map.items():
        colors_array[node] = color
    grouped_nodes: dict[str, list[int]] = {}
    for node in sorted(scores):
        grouped_nodes.setdefault(str(group_names[node]), []).append(node)

    simultaneous_group_count = len(grouped_nodes)
    diagnostics: list[GroupDiagnostic] = []
    sorted_groups = sorted(
        grouped_nodes.items(), key=lambda item: _GROUP_ORDER.get(item[0], 99)
    )
    for group_name, nodes in sorted_groups:
        group = calibration_groups[group_name]
        count = len(nodes)
        dependency_degree = _group_dependency_degree(nodes, dependency_graph)
        mean_score = float(np.mean([scores[node] for node in nodes]))
        hoeffding_radius = math.sqrt(
            2.0
            * (dependency_degree + 1)
            * math.log(simultaneous_group_count / delta)
            / count
        )
        diagnostic = _make_group_diagnostic(
            group=group_name,
            dependency_degree=dependency_degree,
            node_indices=tuple(nodes),
            mean_score=mean_score,
            hoeffding_radius=hoeffding_radius,
            eta_upper=group.eta_upper,
            lambda_lower=group.lambda_lower,
            lambda_upper=group.lambda_upper,
        )
        diagnostics.append(diagnostic)

    num_covered = len(scores)
    num_uncovered = num_changed - num_covered
    result_reasons: list[str] = []
    if num_changed == 0:
        result_reasons.append("no_changed_nodes")
    if num_uncovered:
        result_reasons.append("uncovered_changed_nodes")
    if num_covered == 0 and num_changed:
        result_reasons.append("no_covered_changed_nodes")
    result_reasons.extend(sorted(uncovered_group_reasons))
    return _make_certificate(
        conditional=True,
        colors=tuple(int(color) for color in colors_array),
        diagnostics=tuple(diagnostics),
        num_changed=num_changed,
        delta=delta,
        min_gain=min_gain,
        reasons=_unique_tuple(result_reasons),
        assumptions=assumptions,
        calibration_alpha=calibration.alpha,
    )


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar, not a boolean")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _native_float(value, name: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise TypeError(f"{name} must be a native Python scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _native_int(value, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a native Python integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _native_bool(value, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a native Python boolean")
    return value


def _normalize_reasons(values) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or isinstance(values, str):
        raise TypeError("reasons must be a tuple or list of strings")
    reasons = tuple(values)
    if any(type(reason) is not str or not reason.strip() for reason in reasons):
        raise ValueError("reasons must contain non-empty native strings")
    return reasons


def _lambda_from_rho(rho: float, num_classes: int) -> float:
    return max(0.0, (num_classes * rho - 1.0) / (num_classes - 1.0))


def _validate_protocol_id(value) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("audit_protocol_id must be a non-empty string")
    return value


def _unit_interval(value, name: str) -> float:
    value = _finite_scalar(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _open_unit_interval(value, name: str) -> float:
    value = _finite_scalar(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return value


def _non_negative_integer(value, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_probabilities(values, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] == 0
        or probabilities.shape[1] < 2
    ):
        raise ValueError(f"{name} must have shape [num_nodes, num_classes>=2]")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{name} must contain values in [0, 1]")
    if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError(f"rows of {name} must sum to one")
    return probabilities


def _validate_adjacency(
    adjacency, node_count: int
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    if sparse.issparse(adjacency):
        matrix = adjacency.tocsr(copy=True).astype(float)
        if matrix.shape != (node_count, node_count):
            raise ValueError(
                "adjacency must be square and match the probability node count"
            )
        if matrix.data.size and not np.all(np.isfinite(matrix.data)):
            raise ValueError("adjacency contains non-finite values")
        if matrix.data.size and np.any(matrix.data < 0.0):
            raise ValueError("adjacency cannot contain negative weights")
        matrix.sum_duplicates()
        matrix.setdiag(0.0)
        matrix.eliminate_zeros()
        topology = matrix.copy()
        topology.data = np.ones_like(topology.data)
        topology.eliminate_zeros()
        if (topology != topology.T).nnz:
            raise ValueError("adjacency topology must be symmetric")
        topology.sort_indices()
        neighbors = tuple(
            topology.indices[topology.indptr[node] : topology.indptr[node + 1]].copy()
            for node in range(node_count)
        )
    else:
        matrix = np.asarray(adjacency, dtype=float)
        if matrix.shape != (node_count, node_count):
            raise ValueError(
                "adjacency must be square and match the probability node count"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("adjacency contains non-finite values")
        if np.any(matrix < 0.0):
            raise ValueError("adjacency cannot contain negative weights")
        topology = matrix > 0.0
        np.fill_diagonal(topology, False)
        if not np.array_equal(topology, topology.T):
            raise ValueError("adjacency topology must be symmetric")
        neighbors = tuple(np.flatnonzero(topology[node]) for node in range(node_count))
    degrees = np.asarray([len(row) for row in neighbors], dtype=float)
    return neighbors, degrees


def _validate_stability(
    values, node_count: int, *, required: bool
) -> np.ndarray | None:
    if values is None:
        if required:
            raise ValueError(
                "source_stability is required by the calibration threshold"
            )
        return None
    stability = np.asarray(values, dtype=float)
    if stability.shape != (node_count,):
        raise ValueError("source_stability must have shape [num_nodes]")
    if (
        not np.all(np.isfinite(stability))
        or np.any(stability < 0.0)
        or np.any(stability > 1.0)
    ):
        raise ValueError("source_stability must contain finite values in [0, 1]")
    return stability


def _normalize_indices(values, node_count: int) -> np.ndarray:
    indices = np.asarray(values)
    if indices.dtype == bool:
        if indices.shape != (node_count,):
            raise ValueError(
                "boolean source_validation_indices must have shape [num_nodes]"
            )
        indices = np.flatnonzero(indices)
    else:
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                "source_validation_indices must be a non-empty one-dimensional array"
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("source_validation_indices must contain integers")
        indices = indices.astype(int, copy=False)
    if indices.size == 0:
        raise ValueError("source_validation_indices cannot be empty")
    if np.any(indices < 0) or np.any(indices >= node_count):
        raise ValueError("source_validation_indices are out of range")
    if np.unique(indices).size != indices.size:
        raise ValueError("source_validation_indices must be unique")
    return indices


def _normalize_source_labels(
    values, indices: np.ndarray, node_count: int, num_classes: int
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError("source_validation_labels must be one-dimensional")
    if labels.shape[0] == node_count:
        labels = labels[indices]
    elif labels.shape[0] != indices.size:
        raise ValueError(
            "source_validation_labels must align with source_validation_indices"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("source_validation_labels must contain integers")
    labels = labels.astype(int, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(
            "source_validation_labels are outside the probability class range"
        )
    return labels


def _assign_groups(degrees: np.ndarray, grouping: str) -> np.ndarray:
    node_count = degrees.size
    if grouping == _GROUPING_GLOBAL:
        return np.full(node_count, "all", dtype=object)
    lower, upper = (float(value) for value in np.quantile(degrees, (1 / 3, 2 / 3)))
    names = np.full(node_count, "high", dtype=object)
    names[degrees <= upper] = "mid"
    names[degrees <= lower] = "low"
    return names


def _expanded_support(
    seeds: Iterable[int], neighbors: tuple[np.ndarray, ...], radius: int
) -> frozenset[int]:
    support = {int(node) for node in seeds}
    frontier = set(support)
    for _ in range(radius):
        next_frontier: set[int] = set()
        for node in sorted(frontier):
            next_frontier.update(int(neighbor) for neighbor in neighbors[node])
        next_frontier.difference_update(support)
        if not next_frontier:
            break
        support.update(next_frontier)
        frontier = next_frontier
    return frozenset(support)


def _build_dependency_graph(
    nodes: Iterable[int], supports: dict[int, frozenset[int]]
) -> dict[int, set[int]]:
    """Build support-overlap conflicts via an inverted support-element index."""

    ordered_nodes = sorted(int(node) for node in nodes)
    graph: dict[int, set[int]] = {node: set() for node in ordered_nodes}
    postings: dict[int, list[int]] = {}
    for node in ordered_nodes:
        conflicts: set[int] = set()
        for element in supports[node]:
            conflicts.update(postings.get(element, ()))
        for other in conflicts:
            graph[node].add(other)
            graph[other].add(node)
        for element in supports[node]:
            postings.setdefault(element, []).append(node)
    return graph


def _greedy_color(
    nodes: Iterable[int], dependency_graph: dict[int, set[int]]
) -> dict[int, int]:
    colors: dict[int, int] = {}
    ordered_nodes = sorted(int(node) for node in nodes)
    for node in ordered_nodes:
        unavailable: set[int] = set()
        for previous in dependency_graph[node]:
            if previous in colors:
                unavailable.add(colors[previous])
        color = 0
        while color in unavailable:
            color += 1
        colors[node] = color
    return colors


def _group_dependency_degree(
    nodes: Iterable[int], dependency_graph: dict[int, set[int]]
) -> int:
    node_set = {int(node) for node in nodes}
    return max(
        (
            sum(neighbor in node_set for neighbor in dependency_graph[node])
            for node in node_set
        ),
        default=0,
    )


def _unique_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "AnchorCalibration",
    "AnchorGroupCalibration",
    "CertificateAssumptions",
    "GroupDiagnostic",
    "TopologyAnchorCertificate",
    "calibrate_anchor_channel",
    "topology_anchor_certificate",
]
