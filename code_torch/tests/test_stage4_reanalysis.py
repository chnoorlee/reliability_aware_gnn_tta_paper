import copy
import json
import unittest
from pathlib import Path

import numpy as np

from stage4_reanalysis import (
    _bootstrap_interval,
    _official_mode_result,
    _official_weights,
    _weighted_auc,
    binary_auc,
    heldout_bootstrap,
    metric_bundle,
    normalize_heldout,
    normalize_official,
)


class Stage4ReanalysisTests(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    @staticmethod
    def _row(dataset, seed, adapter, gain, delta, phi, accept):
        return {
            "dataset": dataset,
            "seed": seed,
            "shift": "edge_drop",
            "intensity": 0.3,
            "adapter": adapter,
            "gain": gain,
            "delta": delta,
            "phi": phi,
            "score": max(delta / 0.05, phi / 0.20),
            "accept": accept,
        }

    def test_binary_auc_is_tie_aware(self):
        self.assertAlmostEqual(binary_auc([True, False], [0.5, 0.5]), 0.5)
        self.assertIsNone(binary_auc([True, True], [0.1, 0.2]))

    def test_weighted_auc_accepts_fractional_weights_and_ties(self):
        labels = [True, True, False, False]
        scores = [0.8, 0.4, 0.4, 0.1]
        weights = [0.25, 0.25, 0.25, 0.25]
        self.assertAlmostEqual(_weighted_auc(labels, scores, weights), 0.875)
        self.assertAlmostEqual(
            _weighted_auc(labels, scores, [1.0, 2.0, 3.0, 4.0]),
            binary_auc(
                np.repeat(labels, [1, 2, 3, 4]),
                np.repeat(scores, [1, 2, 3, 4]),
            ),
        )

    def test_zero_harm_denominator_is_null(self):
        rows = [self._row("a", 1, "x", 0.1, 0.01, 0.01, True)]
        metrics = metric_bundle(rows, tau=0.01)
        self.assertIsNone(metrics["harm_recall"]["estimate"])
        self.assertEqual(metrics["harm_recall"]["denominator"], 0.0)

    def test_source_setting_balanced_weights_sum_to_half(self):
        rows = [
            *[{"source_setting": "src"} for _ in range(2)],
            *[{"source_setting": "src_imb"} for _ in range(6)],
        ]
        weights = _official_weights(rows, "source_setting_balanced")
        self.assertAlmostEqual(float(weights[:2].sum()), 0.5)
        self.assertAlmostEqual(float(weights[2:].sum()), 0.5)

    def test_source_setting_balanced_finite_metrics_rebalance_each_stratum(self):
        rows = []
        for setting, count in (("src", 2), ("src_imb", 6)):
            for index in range(count):
                finite = index != 0
                rows.append(
                    {
                        "method": "x",
                        "source_setting": setting,
                        "finite": finite,
                        "accept": finite,
                        "gain": 0.02 if finite else None,
                        "delta": 0.01 if finite else None,
                        "phi": 0.01 if finite else None,
                        "score": 0.2 if finite else None,
                        "runtime_seconds": 1.0,
                    }
                )
        result = _official_mode_result(rows, "source_setting_balanced")
        masses = result["operational"]["finite_analysis_weight_mass_by_source_setting"]
        self.assertEqual(masses, {"src": 0.5, "src_imb": 0.5})
        raw = result["operational"]["raw_finite_weight_mass_by_source_setting"]
        self.assertNotAlmostEqual(raw["src"], raw["src_imb"])

    def test_scalar_bootstrap_interval_suppresses_sparse_valid_distribution(self):
        interval = _bootstrap_interval([0.1] * 94, replicates=100)
        self.assertEqual(interval["valid_fraction"], 0.94)
        self.assertIsNone(interval["ci_lower"])
        self.assertIsNone(interval["ci_upper"])
        self.assertEqual(interval["ci_status"], "unstable_due_to_undefined_resamples")

    def test_heldout_validator_rejects_mutated_fixed_cell(self):
        path = (
            self.PROJECT_ROOT
            / "revision_2026-08-19"
            / "heldout_risk_coverage_v3"
            / "results.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(payload)
        mutated["records"][0]["intensity"] = 0.51
        with self.assertRaisesRegex(ValueError, "fixed stress-cell matrix"):
            normalize_heldout(mutated)

    def test_official_validator_rejects_mutated_fixed_target(self):
        path = (
            self.PROJECT_ROOT
            / "revision_2026-08-19"
            / "official_tsa_default_label_isolated_audit_v2"
            / "results.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(payload)
        mutated["records"][0]["target_setting"] = "not_a_frozen_target"
        with self.assertRaisesRegex(ValueError, "target/config mapping mismatch"):
            normalize_official(mutated)

    def test_official_validator_rejects_duplicate_event_key(self):
        path = (
            self.PROJECT_ROOT
            / "revision_2026-08-19"
            / "official_tsa_default_label_isolated_audit_v2"
            / "results.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(payload)
        for field in (
            "data_config",
            "source_setting",
            "target_setting",
            "method",
            "seed",
        ):
            mutated["records"][0][field] = mutated["records"][1][field]
        with self.assertRaisesRegex(ValueError, "event keys are not unique"):
            normalize_official(mutated)

    def test_official_validator_rejects_forged_data_config(self):
        path = (
            self.PROJECT_ROOT
            / "revision_2026-08-19"
            / "official_tsa_default_label_isolated_audit_v2"
            / "results.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(payload)
        mutated["records"][0]["data_config"] = "forged_config"
        with self.assertRaisesRegex(ValueError, "target/config mapping mismatch"):
            normalize_official(mutated)

    def test_bootstrap_is_reproducible_and_paired(self):
        rows = []
        for dataset in ("a", "b"):
            for seed in (1, 2):
                for adapter in ("x", "y"):
                    gain = -0.02 if seed == 1 else 0.03
                    rows.append(
                        self._row(
                            dataset,
                            seed,
                            adapter,
                            gain,
                            0.06 if seed == 1 else 0.01,
                            0.30 if seed == 1 else 0.02,
                            seed != 1,
                        )
                    )
        first = heldout_bootstrap(rows, replicates=20)
        second = heldout_bootstrap(rows, replicates=20)
        self.assertEqual(first, second)
        for table in first:
            self.assertTrue(table)
        np.testing.assert_equal(first, second)


if __name__ == "__main__":
    unittest.main()
