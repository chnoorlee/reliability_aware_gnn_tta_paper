import inspect
import math
import unittest

import numpy as np
from scipy import sparse

try:
    from topology_anchor import (
        AnchorCalibration,
        AnchorGroupCalibration,
        CertificateAssumptions,
        GroupDiagnostic,
        TopologyAnchorCertificate,
        calibrate_anchor_channel,
        topology_anchor_certificate,
    )
except ModuleNotFoundError as exc:
    if exc.name != "topology_anchor":
        raise
    from code_torch.topology_anchor import (
        AnchorCalibration,
        AnchorGroupCalibration,
        CertificateAssumptions,
        GroupDiagnostic,
        TopologyAnchorCertificate,
        calibrate_anchor_channel,
        topology_anchor_certificate,
    )


def _probabilities(classes, confidence=0.95, num_classes=2):
    classes = np.asarray(classes, dtype=int)
    off_class = (1.0 - confidence) / (num_classes - 1)
    probabilities = np.full((classes.size, num_classes), off_class, dtype=float)
    probabilities[np.arange(classes.size), classes] = confidence
    return probabilities


AUDIT_PROTOCOL_ID = "independent-audit-v1"


def _audit_probabilities(classes, num_classes=2):
    """Synthetic realization from the test's separately declared audit stream."""

    return _probabilities(classes, confidence=0.85, num_classes=num_classes)


def _global_calibration(
    *,
    radius=0,
    eligible=True,
    stability_threshold=None,
    eta_upper=0.0,
    alpha=0.05,
    lambda_lower=None,
    lambda_upper=1.0,
):
    if lambda_lower is None:
        lambda_lower = 1.0 if eligible else 0.0
    rho_lower = (lambda_lower + 1.0) / 2.0
    rho_upper = (lambda_upper + 1.0) / 2.0
    group = AnchorGroupCalibration(
        name="all",
        lambda_lower=lambda_lower,
        lambda_upper=lambda_upper,
        sample_size=100,
        color_count=1,
        rho_hat=(rho_lower + rho_upper) / 2.0,
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        eligible=eligible,
        reasons=() if eligible else ("anchor_signal_not_above_chance",),
        eta_upper=eta_upper,
    )
    return AnchorCalibration(
        num_classes=2,
        confidence_threshold=0.8,
        stability_threshold=stability_threshold,
        radius=radius,
        alpha=alpha,
        audit_protocol_id=AUDIT_PROTOCOL_ID,
        grouping="global",
        groups=(group,),
        eligible=eligible,
        reasons=() if eligible else ("no_eligible_calibration_group",),
    )


def _paired_target_fixture(scores):
    scores = np.asarray(scores, dtype=int)
    changed_count = scores.size
    node_count = 2 * changed_count
    source_classes = np.zeros(node_count, dtype=int)
    candidate_classes = source_classes.copy()
    candidate_classes[:changed_count] = 1
    audit_classes = np.zeros(node_count, dtype=int)
    audit_classes[changed_count:] = (scores > 0).astype(int)
    adjacency = np.zeros((node_count, node_count), dtype=float)
    endpoints = np.arange(changed_count)
    anchors = endpoints + changed_count
    adjacency[endpoints, anchors] = 1.0
    adjacency[anchors, endpoints] = 1.0
    return (
        adjacency,
        _probabilities(source_classes),
        _probabilities(candidate_classes),
        _audit_probabilities(audit_classes),
    )


ACKNOWLEDGED_ASSUMPTIONS = CertificateAssumptions(
    frozen_post_selection_design=True,
    post_selection_symmetric_channel=True,
    source_to_target_transport=True,
    independent_audit_anchor_measurements=True,
    repeated_sampling_model=True,
    true_conditional_dependency_graph=True,
)


class TopologyAnchorCertificateTests(unittest.TestCase):
    def test_certificate_api_cannot_accept_target_labels(self):
        parameter_names = inspect.signature(topology_anchor_certificate).parameters
        self.assertFalse(any("label" in name.lower() for name in parameter_names))

        adjacency = np.zeros((2, 2), dtype=float)
        source = _probabilities([0, 0])
        candidate = _probabilities([1, 0])
        with self.assertRaises(TypeError):
            topology_anchor_certificate(
                adjacency,
                source,
                candidate,
                _global_calibration(),
                assumptions=ACKNOWLEDGED_ASSUMPTIONS,
                target_labels=np.array([1, 0]),
            )

    def test_source_calibration_uses_only_source_validation_labels(self):
        node_count = 12
        adjacency = np.zeros((node_count, node_count), dtype=float)
        endpoint_classes = np.array([0, 1, 0, 1, 0, 1])
        source_classes = np.r_[endpoint_classes, 1 - endpoint_classes]
        audit_classes = np.r_[1 - endpoint_classes, endpoint_classes]
        for endpoint, anchor in zip(range(6), range(6, 12)):
            adjacency[endpoint, anchor] = 1.0
            adjacency[anchor, endpoint] = 1.0

        calibration = calibrate_anchor_channel(
            adjacency,
            _probabilities(source_classes),
            source_validation_labels=endpoint_classes,
            source_validation_indices=np.arange(6),
            source_audit_anchor_probabilities=_audit_probabilities(audit_classes),
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            confidence_threshold=0.8,
            radius=0,
            alpha=0.5,
            grouping="global",
        )

        self.assertTrue(calibration.eligible)
        self.assertEqual(calibration.num_classes, 2)
        self.assertEqual(calibration.audit_protocol_id, AUDIT_PROTOCOL_ID)
        self.assertEqual(calibration.groups[0].sample_size, 6)
        self.assertAlmostEqual(calibration.groups[0].rho_hat, 1.0)

    def test_edge_rewiring_changes_score_and_acceptance(self):
        changed_count = 30
        node_count = 2 * changed_count
        source_classes = np.r_[
            np.zeros(changed_count, dtype=int), np.ones(changed_count, dtype=int)
        ]
        candidate_classes = np.ones(node_count, dtype=int)
        source = _probabilities(source_classes)
        candidate = _probabilities(candidate_classes)
        audit = _audit_probabilities(source_classes)

        favorable = np.zeros((node_count, node_count), dtype=float)
        for changed, anchor in zip(
            range(changed_count), range(changed_count, node_count)
        ):
            favorable[changed, anchor] = favorable[anchor, changed] = 1.0

        unfavorable = np.zeros_like(favorable)
        for left in range(0, changed_count, 2):
            right = left + 1
            unfavorable[left, right] = unfavorable[right, left] = 1.0

        favorable_result = topology_anchor_certificate(
            favorable,
            source,
            candidate,
            _global_calibration(radius=0),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
            min_gain=0.0,
        )
        unfavorable_result = topology_anchor_certificate(
            unfavorable,
            source,
            candidate,
            _global_calibration(radius=0),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
            min_gain=0.0,
        )

        self.assertTrue(favorable_result.accepted)
        self.assertFalse(unfavorable_result.accepted)
        self.assertGreater(favorable_result.lower_bound, 0.0)
        self.assertLess(unfavorable_result.lower_bound, 0.0)
        self.assertNotEqual(favorable_result.colors, unfavorable_result.colors)

    def test_uncovered_changed_node_receives_worst_case_minus_one(self):
        adjacency = np.zeros((2, 2), dtype=float)
        source = _probabilities([0, 0])
        candidate = _probabilities([1, 0])
        audit = _audit_probabilities([0, 0])

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertEqual(result.num_changed, 1)
        self.assertEqual(result.num_uncovered, 1)
        self.assertEqual(result.coverage, 0.0)
        self.assertEqual(
            result.coverage_definition, "covered_changed_nodes / changed_nodes"
        )
        self.assertFalse(result.distribution_free)
        self.assertAlmostEqual(result.lower_bound, -0.5)
        self.assertFalse(result.accepted)
        self.assertIn("uncovered_changed_nodes", result.reasons)

    def test_simultaneous_hoeffding_bound_arithmetic(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(radius=0),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        expected_radius = math.sqrt(2.0 * math.log(2.0))
        expected_bound = (1.0 - expected_radius) / 2.0
        self.assertEqual(len(result.group_diagnostics), 1)
        diagnostic = result.group_diagnostics[0]
        self.assertAlmostEqual(diagnostic.mean_score, 1.0)
        self.assertAlmostEqual(diagnostic.hoeffding_radius, expected_radius)
        self.assertAlmostEqual(result.lower_bound, expected_bound)
        self.assertAlmostEqual(result.conditional_failure_budget, 0.55)
        self.assertEqual(result.mode, "conditional_model_diagnostic")
        self.assertEqual(
            result.statistical_scope,
            "conditional_on_acknowledged_post_selection_measurement_assumptions",
        )
        self.assertFalse(result.distribution_free)

    def test_unacknowledged_assumption_fails_closed(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])
        assumption_fields = (
            "frozen_post_selection_design",
            "post_selection_symmetric_channel",
            "source_to_target_transport",
            "independent_audit_anchor_measurements",
            "repeated_sampling_model",
            "true_conditional_dependency_graph",
        )
        for missing_field in assumption_fields:
            values = {field: True for field in assumption_fields}
            values[missing_field] = False
            with self.subTest(missing_field=missing_field):
                result = topology_anchor_certificate(
                    adjacency,
                    source,
                    candidate,
                    _global_calibration(),
                    assumptions=CertificateAssumptions(**values),
                    audit_anchor_probabilities=audit,
                    audit_protocol_id=AUDIT_PROTOCOL_ID,
                    delta=0.5,
                )

                self.assertFalse(result.eligible)
                self.assertFalse(result.accepted)
                self.assertAlmostEqual(result.lower_bound, -0.5)
                self.assertIn(f"{missing_field}_not_acknowledged", result.reasons)
                self.assertFalse(result.distribution_free)

    def test_optional_source_stability_can_make_node_uncovered(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(stability_threshold=0.9),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            source_stability=np.array([1.0, 0.2]),
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertEqual(result.num_uncovered, 1)
        self.assertAlmostEqual(result.lower_bound, -0.5)

    def test_sparse_and_dense_adjacency_are_equivalent(self):
        adjacency = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        source = _probabilities([0, 0, 1, 1])
        candidate = _probabilities([1, 0, 1, 1])
        audit = _audit_probabilities([0, 0, 1, 1])
        kwargs = dict(
            calibration=_global_calibration(radius=1),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        dense_result = topology_anchor_certificate(
            adjacency, source, candidate, **kwargs
        )
        sparse_result = topology_anchor_certificate(
            sparse.csr_matrix(adjacency), source, candidate, **kwargs
        )

        self.assertEqual(dense_result.to_dict(), sparse_result.to_dict())

    def test_invalid_graph_and_calibration_are_rejected_or_fail_closed(self):
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            topology_anchor_certificate(
                np.array([[0.0, 1.0], [0.0, 0.0]]),
                source,
                candidate,
                _global_calibration(),
                assumptions=ACKNOWLEDGED_ASSUMPTIONS,
                audit_anchor_probabilities=audit,
                audit_protocol_id=AUDIT_PROTOCOL_ID,
            )

        ineligible = topology_anchor_certificate(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            source,
            candidate,
            _global_calibration(eligible=False),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )
        self.assertFalse(ineligible.eligible)
        self.assertFalse(ineligible.accepted)
        self.assertAlmostEqual(ineligible.lower_bound, -0.5)
        self.assertIn("calibration_ineligible", ineligible.reasons)

    def test_missing_independent_audit_measurement_demotes_to_monitor(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertEqual(result.mode, "heuristic_monitor_only")
        self.assertFalse(result.eligible)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.conditional_failure_budget)
        self.assertAlmostEqual(result.lower_bound, -0.5)
        self.assertIn("independent_audit_anchor_probabilities_missing", result.reasons)

    def test_missing_or_mismatched_audit_protocol_fails_closed(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])

        for protocol_id, expected_reason in (
            (None, "audit_protocol_id_missing"),
            ("other-audit-v2", "audit_protocol_id_mismatch"),
        ):
            with self.subTest(protocol_id=protocol_id):
                result = topology_anchor_certificate(
                    adjacency,
                    source,
                    candidate,
                    _global_calibration(),
                    assumptions=ACKNOWLEDGED_ASSUMPTIONS,
                    audit_anchor_probabilities=audit,
                    audit_protocol_id=protocol_id,
                    delta=0.5,
                )

                self.assertEqual(result.mode, "heuristic_monitor_only")
                self.assertFalse(result.eligible)
                self.assertFalse(result.accepted)
                self.assertIsNone(result.conditional_failure_budget)
                self.assertIn(expected_reason, result.reasons)

    def test_positive_bound_uses_lambda_upper(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(20))
        calibration = _global_calibration(lambda_lower=0.4, lambda_upper=0.8)

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        radius = math.sqrt(2.0 * math.log(2.0) / 20.0)
        expected = 0.5 * (1.0 - radius) / 0.8
        self.assertAlmostEqual(result.lower_bound, expected)
        self.assertGreater(result.lower_bound, 0.0)
        self.assertTrue(result.accepted)

    def test_negative_bound_uses_lambda_lower(self):
        scores = np.r_[np.ones(10), -np.ones(10)]
        adjacency, source, candidate, audit = _paired_target_fixture(scores)
        calibration = _global_calibration(lambda_lower=0.4, lambda_upper=0.8)

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        radius = math.sqrt(2.0 * math.log(2.0) / 20.0)
        expected = 0.5 * (-radius) / 0.4
        self.assertAlmostEqual(result.lower_bound, expected)
        self.assertLess(result.lower_bound, 0.0)

    def test_eta_can_move_adjusted_bound_across_zero(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(20))
        calibration = _global_calibration(
            lambda_lower=0.4, lambda_upper=0.8, eta_upper=0.8
        )

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        radius = math.sqrt(2.0 * math.log(2.0) / 20.0)
        expected = 0.5 * (1.0 - radius - 0.8) / 0.4
        diagnostic = result.group_diagnostics[0]
        self.assertGreater(diagnostic.score_lower, 0.0)
        self.assertLess(diagnostic.adjusted_score_lower, 0.0)
        self.assertAlmostEqual(result.lower_bound, expected)

    def test_tied_degree_bound_is_invariant_to_node_permutation(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(20))
        eligible = AnchorGroupCalibration(
            name="low",
            lambda_lower=0.4,
            lambda_upper=0.8,
            sample_size=20,
            color_count=1,
            rho_hat=0.8,
            rho_lower=0.7,
            rho_upper=0.9,
            eligible=True,
        )
        empty_groups = tuple(
            AnchorGroupCalibration(
                name=name,
                lambda_lower=0.0,
                lambda_upper=1.0,
                sample_size=0,
                color_count=0,
                rho_hat=None,
                rho_lower=None,
                rho_upper=None,
                eligible=False,
                reasons=("no_covered_source_validation_nodes",),
            )
            for name in ("mid", "high")
        )
        calibration = AnchorCalibration(
            num_classes=2,
            confidence_threshold=0.8,
            stability_threshold=None,
            radius=0,
            alpha=0.05,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            grouping="degree_terciles",
            groups=(eligible, *empty_groups),
            eligible=True,
        )
        original = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )
        permutation = np.random.default_rng(7).permutation(adjacency.shape[0])
        permuted = topology_anchor_certificate(
            adjacency[np.ix_(permutation, permutation)],
            source[permutation],
            candidate[permutation],
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit[permutation],
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertAlmostEqual(original.lower_bound, permuted.lower_bound)
        self.assertEqual(original.accepted, permuted.accepted)
        self.assertAlmostEqual(original.coverage, permuted.coverage)
        self.assertEqual(original.num_covered, permuted.num_covered)
        self.assertEqual(original.num_uncovered, permuted.num_uncovered)

    def test_overlapping_support_bound_is_invariant_to_node_permutation(self):
        changed_count = 6
        node_count = 2 * changed_count
        adjacency = np.zeros((node_count, node_count), dtype=float)
        for node in range(changed_count):
            for anchor in (
                changed_count + node,
                changed_count + (node + 1) % changed_count,
            ):
                adjacency[node, anchor] = 1.0
                adjacency[anchor, node] = 1.0
        source_classes = np.zeros(node_count, dtype=int)
        candidate_classes = source_classes.copy()
        candidate_classes[:changed_count] = 1
        audit_classes = source_classes.copy()
        audit_classes[changed_count:] = 1
        source = _probabilities(source_classes)
        candidate = _probabilities(candidate_classes)
        audit = _audit_probabilities(audit_classes)
        calibration = _global_calibration(lambda_lower=0.4, lambda_upper=0.8)

        kwargs = dict(
            calibration=calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )
        original = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            audit_anchor_probabilities=audit,
            **kwargs,
        )
        permutation = np.array([5, 0, 3, 8, 11, 2, 7, 1, 10, 4, 9, 6])
        permuted = topology_anchor_certificate(
            adjacency[np.ix_(permutation, permutation)],
            source[permutation],
            candidate[permutation],
            audit_anchor_probabilities=audit[permutation],
            **kwargs,
        )

        self.assertAlmostEqual(original.lower_bound, permuted.lower_bound)
        self.assertEqual(original.accepted, permuted.accepted)
        self.assertAlmostEqual(original.coverage, permuted.coverage)
        self.assertEqual(original.num_covered, permuted.num_covered)
        self.assertEqual(original.num_uncovered, permuted.num_uncovered)
        self.assertEqual(
            original.group_diagnostics[0].dependency_degree,
            permuted.group_diagnostics[0].dependency_degree,
        )
        self.assertEqual(original.group_diagnostics[0].dependency_degree, 2)

    def test_dataclass_cross_field_invariants_reject_malformed_values(self):
        valid_group = dict(
            name="all",
            lambda_lower=0.4,
            lambda_upper=0.8,
            sample_size=10,
            color_count=1,
            rho_hat=0.8,
            rho_lower=0.7,
            rho_upper=0.9,
            eligible=True,
        )
        malformed_groups = (
            {**valid_group, "sample_size": 0, "color_count": 0},
            {**valid_group, "color_count": 0},
            {**valid_group, "rho_hat": None},
            {**valid_group, "rho_lower": 0.85},
            {**valid_group, "reasons": ("unexpected",)},
            {**valid_group, "lambda_lower": np.float64(0.4)},
            {**valid_group, "reasons": "not-a-sequence", "eligible": False},
        )
        for kwargs in malformed_groups:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    AnchorGroupCalibration(**kwargs)

        group = AnchorGroupCalibration(**valid_group)
        canonical = AnchorCalibration(
            num_classes=3,
            confidence_threshold=0.8,
            stability_threshold=None,
            radius=0,
            alpha=0.05,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            grouping="global",
            groups=(group,),
            eligible=True,
        )
        self.assertEqual(canonical.groups[0].lambda_lower, (3 * 0.7 - 1) / 2)
        self.assertEqual(canonical.groups[0].lambda_upper, (3 * 0.9 - 1) / 2)
        with self.assertRaisesRegex(ValueError, "audit_protocol_id"):
            AnchorCalibration(
                num_classes=2,
                confidence_threshold=0.8,
                stability_threshold=None,
                radius=0,
                alpha=0.05,
                audit_protocol_id="",
                grouping="global",
                groups=(group,),
                eligible=True,
            )

    def test_multiclass_nonunit_lambda_calibration_is_supported(self):
        group = AnchorGroupCalibration(
            name="all",
            lambda_lower=0.4,
            lambda_upper=0.7,
            sample_size=10,
            color_count=1,
            rho_hat=0.7,
            rho_lower=0.6,
            rho_upper=0.8,
            eligible=True,
        )
        calibration = AnchorCalibration(
            num_classes=3,
            confidence_threshold=0.8,
            stability_threshold=None,
            radius=0,
            alpha=0.05,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            grouping="global",
            groups=(group,),
            eligible=True,
        )
        result = topology_anchor_certificate(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            _probabilities([0, 2], num_classes=3),
            _probabilities([1, 2], num_classes=3),
            calibration,
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=_audit_probabilities([0, 1], num_classes=3),
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertEqual(result.group_diagnostics[0].lambda_lower, (3 * 0.6 - 1) / 2)
        self.assertEqual(result.group_diagnostics[0].lambda_upper, (3 * 0.8 - 1) / 2)

    def test_negative_min_gain_is_rejected(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(2))
        with self.assertRaisesRegex(ValueError, "min_gain"):
            topology_anchor_certificate(
                adjacency,
                source,
                candidate,
                _global_calibration(),
                assumptions=ACKNOWLEDGED_ASSUMPTIONS,
                audit_anchor_probabilities=audit,
                audit_protocol_id=AUDIT_PROTOCOL_ID,
                delta=0.5,
                min_gain=-0.1,
            )

    def test_boolean_numeric_parameters_are_rejected(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        calibration_args = dict(
            adjacency=adjacency,
            frozen_source_probabilities=_probabilities([0, 1]),
            source_validation_labels=np.array([0]),
            source_validation_indices=np.array([0]),
            source_audit_anchor_probabilities=_audit_probabilities([0, 1]),
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            grouping="global",
        )
        for value in (True, np.bool_(True)):
            for parameter in (
                "confidence_threshold",
                "stability_threshold",
                "radius",
                "alpha",
                "min_group_size",
                "eta_upper",
            ):
                with self.subTest(value=type(value).__name__, parameter=parameter):
                    with self.assertRaises((TypeError, ValueError)):
                        calibrate_anchor_channel(
                            **calibration_args, **{parameter: value}
                        )

            target = _paired_target_fixture(np.ones(2))
            for parameter in ("delta", "min_gain"):
                with self.subTest(value=type(value).__name__, parameter=parameter):
                    with self.assertRaises((TypeError, ValueError)):
                        topology_anchor_certificate(
                            target[0],
                            target[1],
                            target[2],
                            _global_calibration(),
                            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
                            audit_anchor_probabilities=target[3],
                            audit_protocol_id=AUDIT_PROTOCOL_ID,
                            **{parameter: value},
                        )

    def test_exact_acceptance_boundary_and_failure_budget(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(20))
        calibration = _global_calibration(alpha=0.05)
        kwargs = dict(
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )
        baseline = topology_anchor_certificate(
            adjacency, source, candidate, calibration, **kwargs
        )
        at_boundary = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            min_gain=baseline.lower_bound,
            **kwargs,
        )
        below_boundary = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            calibration,
            min_gain=np.nextafter(baseline.lower_bound, -np.inf),
            **kwargs,
        )

        self.assertFalse(at_boundary.accepted)
        self.assertTrue(below_boundary.accepted)
        self.assertEqual(
            baseline.conditional_failure_budget, calibration.alpha + kwargs["delta"]
        )
        self.assertGreater(baseline.conditional_failure_budget, baseline.delta)

    def test_sparse_disjoint_supports_scale_without_pairwise_intersections(self):
        changed_count = 600
        rows = np.r_[np.arange(changed_count), np.arange(changed_count, 1200)]
        columns = np.r_[np.arange(changed_count, 1200), np.arange(changed_count)]
        adjacency = sparse.csr_matrix(
            (np.ones(rows.size), (rows, columns)), shape=(1200, 1200)
        )
        source_classes = np.zeros(1200, dtype=int)
        candidate_classes = source_classes.copy()
        candidate_classes[:changed_count] = 1
        audit_classes = source_classes.copy()
        audit_classes[changed_count:] = 1
        result = topology_anchor_certificate(
            adjacency,
            _probabilities(source_classes),
            _probabilities(candidate_classes),
            _global_calibration(lambda_lower=0.4, lambda_upper=0.8),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=_audit_probabilities(audit_classes),
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertEqual(result.num_covered, changed_count)
        self.assertEqual(result.num_colors, 1)
        self.assertEqual(result.group_diagnostics[0].dependency_degree, 0)

    def test_public_result_constructors_reject_fabricated_results(self):
        for result_type in (GroupDiagnostic, TopologyAnchorCertificate):
            with self.subTest(result_type=result_type.__name__):
                with self.assertRaisesRegex(TypeError, "created only"):
                    result_type()

    def test_factory_returns_serializable_accepted_and_fail_closed_results(self):
        adjacency, source, candidate, audit = _paired_target_fixture(np.ones(20))
        accepted = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )
        fail_closed = topology_anchor_certificate(
            adjacency, source, candidate, _global_calibration(), delta=0.5
        )

        self.assertTrue(accepted.accepted)
        self.assertTrue(accepted.to_dict()["accepted"])
        self.assertFalse(fail_closed.accepted)
        self.assertFalse(fail_closed.eligible)
        self.assertEqual(fail_closed.to_dict()["mode"], "heuristic_monitor_only")

    def test_noninformative_total_failure_budget_fails_closed(self):
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        source = _probabilities([0, 1])
        candidate = _probabilities([1, 1])
        audit = _audit_probabilities([0, 1])

        result = topology_anchor_certificate(
            adjacency,
            source,
            candidate,
            _global_calibration(alpha=0.6),
            assumptions=ACKNOWLEDGED_ASSUMPTIONS,
            audit_anchor_probabilities=audit,
            audit_protocol_id=AUDIT_PROTOCOL_ID,
            delta=0.5,
        )

        self.assertFalse(result.eligible)
        self.assertFalse(result.accepted)
        self.assertIsNone(result.conditional_failure_budget)
        self.assertIn("noninformative_total_failure_budget", result.reasons)


if __name__ == "__main__":
    unittest.main()
