import unittest

import numpy as np

from proxy_scope_audit import (
    ScopeTraceRecorder,
    audit_proxy_scopes,
    canonical_scope_mask,
    scope_index_sha256,
    scoped_proxy_signals,
)


class ProxyScopeAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
        self.candidate = np.asarray([[0.6, 0.4], [0.2, 0.8], [0.45, 0.55], [0.8, 0.2]])
        self.groups = {
            "low": np.asarray([True, True, False, False]),
            "high": np.asarray([False, False, True, True]),
        }

    def test_indices_and_boolean_mask_have_same_hash(self):
        mask = np.asarray([False, True, False, True])
        self.assertEqual(scope_index_sha256(mask, 4), scope_index_sha256([1, 3], 4))
        np.testing.assert_array_equal(canonical_scope_mask([1, 3], 4), mask)

    def test_scoped_signals_report_integer_turnover(self):
        result = scoped_proxy_signals(
            self.groups, self.source, self.candidate, np.asarray([1, 2])
        )
        self.assertEqual(result["num_nodes"], 2)
        self.assertEqual(result["flip_count"], 1)
        self.assertAlmostEqual(result["phi"], 0.5)
        self.assertAlmostEqual(result["delta"], 0.025)
        self.assertEqual(result["nonempty_degree_groups"], 2)

    def test_scope_bundle_separates_target_and_evaluation(self):
        result = audit_proxy_scopes(
            self.groups,
            self.source,
            self.candidate,
            {
                "target": np.ones(4, dtype=bool),
                "evaluation": np.asarray([1, 2]),
            },
        )
        self.assertAlmostEqual(result["target"]["phi"], 0.5)
        self.assertAlmostEqual(result["evaluation"]["phi"], 0.5)
        self.assertNotEqual(
            result["target"]["scope_index_sha256"],
            result["evaluation"]["scope_index_sha256"],
        )

    def test_scope_name_string_collision_fails_closed(self):
        colliding = {1: np.ones(4, dtype=bool), "1": np.ones(4, dtype=bool)}
        with self.assertRaisesRegex(ValueError, "unique after string conversion"):
            audit_proxy_scopes(
                self.groups,
                self.source,
                self.candidate,
                colliding,
            )
        with self.assertRaisesRegex(ValueError, "unique after string conversion"):
            ScopeTraceRecorder(self.groups, colliding)

    def test_empty_scope_and_overlapping_groups_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            scoped_proxy_signals(self.groups, self.source, self.candidate, [])
        overlapping = {
            "a": np.asarray([True, True, False, False]),
            "b": np.asarray([False, True, True, True]),
        }
        with self.assertRaisesRegex(ValueError, "disjoint partition"):
            scoped_proxy_signals(overlapping, self.source, self.candidate, [0, 1])

    def test_trace_recorder_preserves_steps_and_maximum_hashes(self):
        recorder = ScopeTraceRecorder(
            self.groups,
            {"target": np.ones(4, dtype=bool), "evaluation": [1, 2]},
        )
        primary = scoped_proxy_signals(
            self.groups, self.source, self.candidate, np.ones(4, dtype=bool)
        )
        self.assertIsNone(
            recorder(
                step=7,
                source_probs=self.source,
                candidate_probs=self.candidate,
                delta=primary["delta"],
                phi=primary["phi"],
            )
        )
        traces = recorder.to_dict()
        self.assertEqual(traces["target"]["step_trace"], [7])
        self.assertEqual(traces["target"]["max_delta_step"], 7)
        self.assertEqual(traces["target"]["max_delta_trace_index"], 0)
        self.assertEqual(traces["target"]["max_phi_step"], 7)
        self.assertEqual(traces["target"]["max_phi_trace_index"], 0)
        self.assertEqual(
            traces["target"]["max_phi_candidate_prediction_sha256"],
            primary["candidate_prediction_sha256"],
        )

    def test_trace_recorder_invariant_failure_is_atomic_across_scopes(self):
        recorder = ScopeTraceRecorder(
            self.groups,
            {"first": [0, 1], "second": [2, 3]},
        )
        recorder(
            step=0,
            source_probs=self.source,
            candidate_probs=self.candidate,
            delta=0.0,
            phi=0.0,
        )
        before = recorder.to_dict()
        changed_source = self.source.copy()
        changed_source[2] = [0.7, 0.3]

        with self.assertRaisesRegex(AssertionError, "second/source_prediction_sha256"):
            recorder(
                step=1,
                source_probs=changed_source,
                candidate_probs=self.candidate,
                delta=0.0,
                phi=0.0,
            )

        self.assertEqual(recorder.to_dict(), before)

    def test_same_argmax_source_confidence_change_fails_atomically(self):
        recorder = ScopeTraceRecorder(
            self.groups,
            {"target": np.ones(4, dtype=bool), "evaluation": [1, 2]},
        )
        primary = scoped_proxy_signals(
            self.groups, self.source, self.candidate, np.ones(4, dtype=bool)
        )
        recorder(
            step=0,
            source_probs=self.source,
            candidate_probs=self.candidate,
            delta=primary["delta"],
            phi=primary["phi"],
        )
        before = recorder.to_dict()
        changed_source = self.source.copy()
        changed_source[0] = [0.55, 0.45]
        changed_primary = scoped_proxy_signals(
            self.groups,
            changed_source,
            self.candidate,
            np.ones(4, dtype=bool),
        )
        self.assertEqual(
            changed_primary["source_prediction_sha256"],
            primary["source_prediction_sha256"],
        )
        self.assertNotEqual(
            changed_primary["source_probability_sha256"],
            primary["source_probability_sha256"],
        )

        with self.assertRaisesRegex(AssertionError, "target/source_probability_sha256"):
            recorder(
                step=1,
                source_probs=changed_source,
                candidate_probs=self.candidate,
                delta=changed_primary["delta"],
                phi=changed_primary["phi"],
            )

        self.assertEqual(recorder.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
