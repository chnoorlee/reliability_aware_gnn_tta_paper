import unittest

import numpy as np

from stage5_selective_audit import (
    exact_tie_aware_selection,
    official_operational_analysis,
    policy_selection,
    selective_metrics,
)


class Stage5SelectiveAuditTests(unittest.TestCase):
    @staticmethod
    def _rows(gains):
        return [
            {
                "gain": gain,
                "delta": delta,
                "phi": phi,
                "score": max(delta / 0.05, phi / 0.20),
                "accept": delta <= 0.05 and phi <= 0.20,
            }
            for gain, delta, phi in gains
        ]

    def test_exact_coverage_fractionates_boundary_ties(self):
        selected, metadata = exact_tie_aware_selection([0, 0, 1, 1], 3)
        np.testing.assert_allclose(selected, [1, 1, 0.5, 0.5])
        self.assertAlmostEqual(float(selected.sum()), 3.0)
        self.assertEqual(metadata["boundary_item_count"], 2)
        self.assertAlmostEqual(metadata["boundary_fraction"], 0.5)

    def test_tie_selection_is_independent_of_row_order(self):
        scores = np.asarray([2.0, 1.0, 2.0, 0.0])
        identifiers = np.asarray(["a", "b", "c", "d"])
        first, _ = exact_tie_aware_selection(scores, 2.0)
        permutation = np.asarray([2, 0, 3, 1])
        second, _ = exact_tie_aware_selection(scores[permutation], 2.0)
        first_by_id = dict(zip(identifiers, first))
        second_by_id = dict(zip(identifiers[permutation], second))
        self.assertEqual(first_by_id, second_by_id)

    def test_exact_selection_respects_bootstrap_capacities(self):
        selected, metadata = exact_tie_aware_selection(
            [0.0, 1.0, 1.0], 3.0, capacities=[2.0, 2.0, 1.0]
        )
        np.testing.assert_allclose(selected, [2.0, 2.0 / 3.0, 1.0 / 3.0])
        self.assertAlmostEqual(metadata["selected_mass"], 3.0)
        self.assertAlmostEqual(metadata["boundary_fraction"], 1.0 / 3.0)

    def test_random_policy_is_analytical_expectation(self):
        rows = self._rows(
            [(-0.03, 0.01, 0.01), (0.04, 0.02, 0.02), (0.00, 0.03, 0.03)]
        )
        selected, metadata = policy_selection(rows, "random_expectation", 1.5)
        np.testing.assert_allclose(selected, [0.5, 0.5, 0.5])
        self.assertTrue(metadata["analytical_expectation"])
        metrics = selective_metrics(rows, selected)
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["retained_positive_utility"], 0.5)
        self.assertAlmostEqual(metrics["unintercepted_harm_fraction"], 0.5)
        self.assertAlmostEqual(metrics["prevented_downside"], 0.5)

    def test_oracle_selects_highest_gain_at_exact_mass(self):
        rows = self._rows(
            [(-0.05, 0.01, 0.01), (0.01, 0.02, 0.02), (0.08, 0.03, 0.03)]
        )
        selected, _ = policy_selection(rows, "oracle_gain", 1.0)
        np.testing.assert_allclose(selected, [0.0, 0.0, 1.0])
        metrics = selective_metrics(rows, selected)
        self.assertAlmostEqual(metrics["mean_deployed_gain"], 0.08 / 3.0)

    def test_zero_coverage_has_null_conditional_risk(self):
        rows = self._rows([(-0.03, 0.01, 0.01), (0.04, 0.02, 0.02)])
        selected, _ = policy_selection(rows, "combined_score", 0.0)
        metrics = selective_metrics(rows, selected)
        self.assertEqual(metrics["accepted_count"], 0.0)
        self.assertIsNone(metrics["accepted_harm_rate"])
        self.assertIsNone(metrics["accepted_conditional_downside"])
        self.assertIsNone(metrics["accepted_conditional_mean_gain"])

    def test_official_nonfinite_candidates_fail_closed(self):
        rows = [
            {
                "method": "method",
                "source_setting": "src",
                "candidate_status": "finite",
                "finite": True,
                "accept": True,
                "gain": 0.02,
            },
            {
                "method": "method",
                "source_setting": "src",
                "candidate_status": "nonfinite_probability",
                "finite": False,
                "accept": False,
                "gain": None,
            },
            {
                "method": "method",
                "source_setting": "src_imb",
                "candidate_status": "finite",
                "finite": True,
                "accept": False,
                "gain": -0.02,
            },
            {
                "method": "method",
                "source_setting": "src_imb",
                "candidate_status": "finite",
                "finite": True,
                "accept": False,
                "gain": 0.01,
            },
        ]
        output = official_operational_analysis(rows)
        self.assertEqual(len(output), 2)
        for row in output:
            self.assertEqual(row["literal_attempt_count"], 4)
            self.assertEqual(row["literal_nonfinite_count"], 1)
            self.assertEqual(row["literal_accepted_count"], 1)
            self.assertEqual(row["nonfinite_policy"], "fail_closed_to_source")
            self.assertLessEqual(
                row["effective_accepted_mass"], row["effective_finite_mass"]
            )

    def test_official_nonfinite_acceptance_is_rejected(self):
        rows = [
            {
                "method": "method",
                "source_setting": "src",
                "candidate_status": "nonfinite_probability",
                "finite": False,
                "accept": True,
                "gain": None,
            },
            {
                "method": "method",
                "source_setting": "src_imb",
                "candidate_status": "finite",
                "finite": True,
                "accept": False,
                "gain": 0.0,
            },
        ]
        with self.assertRaisesRegex(ValueError, "must fail closed"):
            official_operational_analysis(rows)


if __name__ == "__main__":
    unittest.main()
