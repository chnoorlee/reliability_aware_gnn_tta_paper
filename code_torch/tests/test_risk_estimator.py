import unittest

import numpy as np

from risk_estimator import FEATURE_NAMES, fit_paired_gain_estimator


class PairedGainEstimatorTests(unittest.TestCase):
    def test_group_heldout_fit_returns_finite_lower_bound(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, len(FEATURE_NAMES)))
        y = 0.03 * x[:, 0] - 0.02 * x[:, 3]
        groups = [f"g{i // 2}" for i in range(12)]
        estimator = fit_paired_gain_estimator(x, y, groups)
        prediction = estimator.predict(x[0])
        lower = estimator.lower_bound(x[0])
        self.assertTrue(np.isfinite(prediction))
        self.assertTrue(np.isfinite(lower))
        self.assertLessEqual(lower, prediction)

    def test_requires_multiple_condition_groups(self):
        x = np.zeros((4, len(FEATURE_NAMES)))
        with self.assertRaises(ValueError):
            fit_paired_gain_estimator(x, np.zeros(4), ["a", "a", "b", "b"])


if __name__ == "__main__":
    unittest.main()
