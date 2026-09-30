import unittest

import numpy as np

from risk_coverage_analysis import binary_auc, operating_metrics


class RiskCoverageAnalysisTests(unittest.TestCase):
    def test_operating_metrics_jointly_count_risk_and_utility(self):
        gains = np.asarray([-0.20, -0.02, 0.00, 0.10, 0.30])
        accepted = np.asarray([False, True, True, False, True])
        result = operating_metrics(gains, accepted, harm_tolerance=0.01)
        self.assertAlmostEqual(result["coverage"], 0.6)
        self.assertAlmostEqual(result["harm_recall"], 0.5)
        self.assertAlmostEqual(result["retained_positive_utility"], 0.75)
        self.assertAlmostEqual(result["residual_downside_sum"], 0.02)
        self.assertAlmostEqual(result["foregone_positive_gain_sum"], 0.10)

    def test_binary_auc_uses_high_score_as_harm(self):
        self.assertAlmostEqual(binary_auc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(binary_auc([False, False, True, True], [0.9, 0.8, 0.2, 0.1]), 0.0)


if __name__ == "__main__":
    unittest.main()
