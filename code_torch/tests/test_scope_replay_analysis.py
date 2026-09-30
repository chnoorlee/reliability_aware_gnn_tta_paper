import copy
import unittest

from scope_replay_analysis import validate_and_summarize


def _payloads():
    original_row = {
        "dataset": "graph",
        "seed": 1,
        "shift": "edge_add",
        "intensity": 0.5,
        "adapter": "weighted",
        "guard": "unguarded",
        "runtime_seconds": 1.0,
        "adaptation_attempts": 2,
        "source_relative_accuracy": 0.0,
        "max_delta": 0.06,
        "max_phi": 0.25,
        "delta_trace": [0.02, 0.06],
        "phi_trace": [0.10, 0.25],
    }
    original = {
        "status": "complete",
        "datasets": ["graph"],
        "seeds": [1],
        "train_epochs": 1,
        "adapt_steps": 2,
        "include_guard": False,
        "guard_mode": "unguarded",
        "stress_grid": [["edge_add", 0.5]],
        "records": [original_row],
    }
    replay_row = copy.deepcopy(original_row)
    replay_row["runtime_seconds"] = 2.0

    def trace(name, nodes, deltas, phis, flips):
        return {
            "num_nodes": nodes,
            "fraction_of_target_nodes": nodes / 10,
            "scope_index_sha256": name + "-scope",
            "source_prediction_sha256": name + "-source",
            "nonempty_degree_groups": 2,
            "step_trace": [0, 1],
            "delta_trace": deltas,
            "phi_trace": phis,
            "flip_count_trace": flips,
            "max_delta": max(deltas),
            "max_delta_step": deltas.index(max(deltas)),
            "max_delta_trace_index": deltas.index(max(deltas)),
            "max_phi": max(phis),
            "max_phi_step": phis.index(max(phis)),
            "max_phi_trace_index": phis.index(max(phis)),
        }

    def endpoint(name, nodes, delta, phi, flips):
        return {
            "num_nodes": nodes,
            "fraction_of_target_nodes": nodes / 10,
            "scope_index_sha256": name + "-scope",
            "source_prediction_sha256": name + "-source",
            "candidate_prediction_sha256": name + "-candidate",
            "nonempty_degree_groups": 2,
            "delta": delta,
            "phi": phi,
            "flip_count": flips,
        }

    replay_row["diagnostic_scope_traces"] = {
        "target": trace("target", 10, [0.02, 0.06], [0.10, 0.20], [1, 2]),
        "evaluation": trace("evaluation", 4, [0.01, 0.03], [0.0, 0.0], [0, 0]),
        "non_evaluation": trace(
            "non_evaluation", 6, [0.03, 0.08], [1 / 6, 1 / 3], [1, 2]
        ),
    }
    replay_row["phi_trace"][-1] = 0.2
    replay_row["max_phi"] = 0.2
    original_row["phi_trace"][-1] = 0.2
    original_row["max_phi"] = 0.2
    replay_row["endpoint_proxy_scopes"] = {
        "target": endpoint("target", 10, 0.06, 0.2, 2),
        "evaluation": endpoint("evaluation", 4, 0.03, 0.0, 0),
        "non_evaluation": endpoint("non_evaluation", 6, 0.08, 1 / 3, 2),
    }
    replay_row["evaluation_turnover_bound_slack"] = 0.0
    replay = {**copy.deepcopy(original), "records": [replay_row]}
    return original, replay


class ScopeReplayAnalysisTests(unittest.TestCase):
    def test_valid_replay_allows_runtime_change_and_summarizes_transport(self):
        original, replay = _payloads()
        rows, validation = validate_and_summarize(original, replay)
        self.assertEqual(validation["status"], "pass")
        self.assertTrue(validation["original_fields_exact_except_runtime"])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["target_trajectory_accept"])
        self.assertTrue(rows[0]["evaluation_trajectory_accept"])
        self.assertEqual(
            validation["summaries"][0][
                "target_reject_evaluation_accept_trajectory"
            ],
            1,
        )

    def test_primary_field_change_fails_noninterference(self):
        original, replay = _payloads()
        replay["records"][0]["source_relative_accuracy"] = -0.09
        with self.assertRaisesRegex(ValueError, "noninterference mismatch"):
            validate_and_summarize(original, replay)

    def test_target_trace_mismatch_fails(self):
        original, replay = _payloads()
        replay["records"][0]["diagnostic_scope_traces"]["target"][
            "delta_trace"
        ][0] = 0.01
        with self.assertRaisesRegex(ValueError, "target delta trace"):
            validate_and_summarize(original, replay)

    def test_incomplete_replay_fails_closed(self):
        original, replay = _payloads()
        replay["status"] = "running"
        with self.assertRaisesRegex(ValueError, "not complete"):
            validate_and_summarize(original, replay)

    def test_negative_same_scope_bound_slack_fails(self):
        original, replay = _payloads()
        original["records"][0]["source_relative_accuracy"] = -0.3
        replay["records"][0]["source_relative_accuracy"] = -0.3
        replay["records"][0]["evaluation_turnover_bound_slack"] = -0.3
        with self.assertRaisesRegex(ValueError, "same-scope turnover bound"):
            validate_and_summarize(original, replay)


if __name__ == "__main__":
    unittest.main()
