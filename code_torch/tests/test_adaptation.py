from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import adaptation as adaptation_module
from adaptation import (
    adapt_classifier,
    differentiable_objective,
    normalized_reliability_weights,
)
from data_adapter import UnlabeledGraphView, load_bundle
from detector import DetectorState, detector_should_halt
from models import make_model
from proxy_scope_audit import ScopeTraceRecorder
from reliability import degree_group_masks


class ReliabilityFormulaTests(unittest.TestCase):
    def test_linear_weight_normalization(self):
        r = np.array([0.1, 0.4, 0.9], dtype=float)
        q = normalized_reliability_weights(r)
        np.testing.assert_allclose(q, r / r.sum(), rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(q.sum()), 1.0)

    def test_reliability_quantile_is_fixed_at_point_six(self):
        source = Path(__import__("reliability").__file__)
        text = source.read_text(encoding="utf-8")
        self.assertIn("np.quantile(base, 0.6)", text)
        self.assertNotIn("0.6 if (use_agreement or use_stability) else 0.5", text)

    def test_degree_groups_are_fixed_near_equal_partitions(self):
        adj = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        first = degree_group_masks(adj)
        second = degree_group_masks(adj)
        covered = np.zeros(6, dtype=int)
        for name in ("low", "mid", "high"):
            np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(int(first[name].sum()), 2)
            covered += first[name].astype(int)
        np.testing.assert_array_equal(covered, np.ones(6, dtype=int))


class ObjectiveGradientTests(unittest.TestCase):
    def _objective(self, weight):
        x = torch.tensor(
            [[1.0, 0.2], [0.4, 1.1], [1.2, -0.3], [-0.2, 0.8], [0.7, 0.5], [0.1, 1.4]],
            dtype=torch.float64,
        )
        probs = F.softmax(x @ weight.T, dim=1)
        q = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        masks = {
            "low": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
            "mid": torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool),
            "high": torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
        }
        source_conf = {
            "low": torch.tensor(0.51, dtype=torch.float64),
            "mid": torch.tensor(0.54, dtype=torch.float64),
            "high": torch.tensor(0.57, dtype=torch.float64),
        }
        source_weight = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64)
        total, _ = differentiable_objective(
            probs,
            q,
            masks,
            source_conf,
            weight,
            source_weight,
            lambda_cal=0.5,
            lambda_af=0.01,
        )
        return total

    def test_full_objective_autograd_matches_finite_difference(self):
        weight = torch.tensor(
            [[0.8, -0.2], [-0.4, 0.6]], dtype=torch.float64, requires_grad=True
        )
        loss = self._objective(weight)
        loss.backward()
        autograd = weight.grad.detach().clone()

        eps = 1e-6
        numeric = torch.zeros_like(weight)
        for row in range(weight.shape[0]):
            for col in range(weight.shape[1]):
                plus = weight.detach().clone()
                minus = weight.detach().clone()
                plus[row, col] += eps
                minus[row, col] -= eps
                numeric[row, col] = (self._objective(plus) - self._objective(minus)) / (
                    2 * eps
                )
        torch.testing.assert_close(autograd, numeric, rtol=2e-5, atol=2e-6)


class AdaptationStateMachineTests(unittest.TestCase):
    @staticmethod
    def _fixture(seed=3, use_bn=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        bundle = load_bundle("synthetic", seed=seed, n=60)
        model = make_model(
            "gcn",
            bundle.x.shape[1],
            hidden_dim=8,
            out_dim=bundle.num_classes,
            use_bn=use_bn,
            dropout=0.0,
            seed=seed,
        )
        model.set_source_classifier_anchor()
        return model, bundle

    def test_classifier_only_update(self):
        model, bundle = self._fixture()
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        adapt_classifier(
            model,
            bundle.unlabeled(),
            method="full_method",
            seed=3,
            steps=1,
            detector=None,
        )
        changed = []
        for name, value in model.named_parameters():
            if not torch.equal(before[name], value.detach()):
                changed.append(name)
        self.assertEqual(changed, ["conv2.lin.weight"])

    def test_scope_diagnostics_do_not_change_candidate_or_primary_traces(self):
        model, bundle = self._fixture()
        baseline = model.clone()
        audited = model.clone()
        baseline_detector = DetectorState()
        audited_detector = DetectorState()
        baseline_info = adapt_classifier(
            baseline,
            bundle.unlabeled(),
            method="confidence_source_entropy",
            seed=3,
            steps=3,
            detector=baseline_detector,
        )
        evaluation_mask = np.zeros(bundle.num_nodes, dtype=bool)
        evaluation_mask[bundle.test_idx] = True
        recorder = ScopeTraceRecorder(
            degree_group_masks(bundle.adj),
            {
                "target": np.ones(bundle.num_nodes, dtype=bool),
                "evaluation": evaluation_mask,
            },
        )
        audited_info = adapt_classifier(
            audited,
            bundle.unlabeled(),
            method="confidence_source_entropy",
            seed=3,
            steps=3,
            detector=audited_detector,
            candidate_observer=recorder,
        )
        torch.testing.assert_close(
            baseline.classifier_weight(), audited.classifier_weight(), rtol=0, atol=0
        )
        self.assertEqual(baseline_info["delta_trace"], audited_info["delta_trace"])
        self.assertEqual(baseline_info["phi_trace"], audited_info["phi_trace"])
        self.assertEqual(baseline_detector.to_dict(), audited_detector.to_dict())
        trace = recorder.to_dict()["evaluation"]
        self.assertEqual(len(trace["delta_trace"]), len(audited_info["delta_trace"]))
        self.assertEqual(len(trace["phi_trace"]), len(audited_info["phi_trace"]))
        self.assertTrue(
            all(
                count / trace["num_nodes"] == phi
                for count, phi in zip(trace["flip_count_trace"], trace["phi_trace"])
            )
        )

    def test_observer_receives_read_only_copies(self):
        model, bundle = self._fixture()
        baseline = model.clone()
        audited = model.clone()
        observed = []

        def observer(**payload):
            self.assertFalse(payload["source_probs"].flags.writeable)
            self.assertFalse(payload["candidate_probs"].flags.writeable)
            with self.assertRaises(ValueError):
                payload["candidate_probs"][0, 0] = -1.0
            payload["candidate_probs"].setflags(write=True)
            payload["candidate_probs"][0, 0] = -1.0
            observed.append(payload["step"])

        baseline_info = adapt_classifier(
            baseline,
            bundle.unlabeled(),
            method="confidence_source_entropy",
            seed=3,
            steps=3,
        )
        audited_info = adapt_classifier(
            audited,
            bundle.unlabeled(),
            method="confidence_source_entropy",
            seed=3,
            steps=3,
            candidate_observer=observer,
        )
        self.assertEqual(observed, list(range(len(audited_info["delta_trace"]))))
        torch.testing.assert_close(
            baseline.classifier_weight(), audited.classifier_weight(), rtol=0, atol=0
        )
        self.assertEqual(baseline_info["delta_trace"], audited_info["delta_trace"])
        self.assertEqual(baseline_info["phi_trace"], audited_info["phi_trace"])

    def test_observer_exception_restores_source_and_halts_detector(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState(delta_tolerance=1.0, phi_tolerance=1.0)

        def failing_observer(**_payload):
            raise RuntimeError("audit failed")

        with self.assertRaisesRegex(RuntimeError, "audit failed"):
            adapt_classifier(
                model,
                bundle.unlabeled(),
                method="confidence_source_entropy",
                seed=3,
                steps=3,
                detector=detector,
                candidate_observer=failing_observer,
            )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.state, "SOURCE_HALTED")
        self.assertEqual(detector.trigger_reason, "exception:RuntimeError")
        self.assertEqual(detector.decision_history, ["EXCEPTION_SOURCE_RESTORE"])

    def test_observer_return_value_is_rejected(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()

        with self.assertRaisesRegex(TypeError, "must return None"):
            adapt_classifier(
                model,
                bundle.unlabeled(),
                method="confidence_source_entropy",
                seed=3,
                steps=1,
                candidate_observer=lambda **_payload: "feedback",
            )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)

    def test_label_bearing_bundle_is_rejected_at_adaptation_boundary(self):
        model, bundle = self._fixture()
        view = bundle.unlabeled()
        self.assertFalse(hasattr(view, "y"))
        self.assertFalse(hasattr(view, "y_np"))
        self.assertFalse(hasattr(view, "test_idx"))
        with self.assertRaisesRegex(TypeError, "UnlabeledGraphView"):
            adapt_classifier(model, bundle, "full_method", steps=1)

        with self.assertRaises((AttributeError, TypeError)):
            view.y = bundle.y

        class LabeledSubclass(UnlabeledGraphView):
            pass

        subclass_view = LabeledSubclass(
            x_np=view.x_np,
            adj=view.adj,
            x=view.x,
            edge_index=view.edge_index,
        )
        object.__setattr__(subclass_view, "y", bundle.y)
        with self.assertRaisesRegex(TypeError, "exact UnlabeledGraphView"):
            adapt_classifier(model, subclass_view, "full_method", steps=1)

    def test_first_reject_retains_last_accepted_checkpoint(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState(delta_tolerance=-1.0, phi_tolerance=-1.0)
        info = adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=1, detector=detector
        )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.strikes, 1)
        self.assertEqual(detector.state, "FINISHED")
        self.assertEqual(detector.decision_history, ["REJECT_RETAIN_CHECKPOINT"])
        self.assertIsNone(info["detector_halted_step"])

    def test_two_consecutive_rejects_restore_source_and_halt(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState(delta_tolerance=-1.0, phi_tolerance=-1.0)
        info = adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=4, detector=detector
        )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.strikes, 2)
        self.assertEqual(detector.state, "SOURCE_HALTED")
        self.assertEqual(
            detector.decision_history,
            ["REJECT_RETAIN_CHECKPOINT", "REJECT_SOURCE_HALT"],
        )
        self.assertEqual(info["detector_halted_step"], 1)

    def test_accepted_candidate_resets_existing_strike(self):
        model, bundle = self._fixture()
        detector = DetectorState(delta_tolerance=1.0, phi_tolerance=1.0, strikes=1)
        adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=1, detector=detector
        )
        self.assertEqual(detector.strikes, 0)
        self.assertEqual(detector.accepted_candidates, 1)
        self.assertEqual(detector.decision_history, ["ACCEPT"])

    def test_new_stream_call_preserves_last_checkpoint(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=1, detector=None
        )
        checkpoint = model.classifier_weight().detach().clone()
        self.assertFalse(torch.equal(checkpoint, source))

        adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=0, detector=None
        )
        torch.testing.assert_close(
            model.classifier_weight(), checkpoint, rtol=0, atol=0
        )

    def test_nonfinite_detector_signal_rejects_fail_closed(self):
        detector = DetectorState()
        rejected, reason = detector_should_halt(detector, float("nan"), 0.0)
        self.assertTrue(rejected)
        self.assertEqual(reason, "nonfinite_proxy")

    def test_nonfinite_objective_restores_source_and_halts(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState()

        def poisoned_objective(*args, **kwargs):
            total, components = differentiable_objective(*args, **kwargs)
            return total * torch.tensor(float("nan"), device=total.device), components

        with patch(
            "adaptation.differentiable_objective", side_effect=poisoned_objective
        ):
            info = adapt_classifier(
                model, bundle.unlabeled(), "full_method", steps=1, detector=detector
            )

        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.state, "SOURCE_HALTED")
        self.assertEqual(detector.decision_history, ["REJECT_NONFINITE_SOURCE_HALT"])
        self.assertEqual(
            info["numerical_failure"], "nonfinite_objective_or_probability"
        )

    def test_exception_after_candidate_mutation_restores_source_and_reraises(self):
        model, bundle = self._fixture(use_bn=True)
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState()
        model.train()
        model.bn.eval()
        entry_module_training = [module.training for module in model.modules()]
        entry_flags = [parameter.requires_grad for parameter in model.parameters()]
        model.classifier_weight().grad = torch.ones_like(model.classifier_weight())
        entry_gradient = model.classifier_weight().grad.detach().clone()
        real_predict = adaptation_module._make_predict_fn(model)
        candidate_mutated = False

        def fail_on_candidate(x_np, adj_np):
            nonlocal candidate_mutated
            if not torch.equal(model.classifier_weight().detach(), source):
                candidate_mutated = True
                raise RuntimeError("injected post-update evaluation failure")
            return real_predict(x_np, adj_np)

        with patch("adaptation._make_predict_fn", return_value=fail_on_candidate):
            with self.assertRaisesRegex(RuntimeError, "injected post-update"):
                adapt_classifier(
                    model,
                    bundle.unlabeled(),
                    "full_method",
                    steps=1,
                    detector=detector,
                )

        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertTrue(candidate_mutated)
        self.assertEqual(detector.state, "SOURCE_HALTED")
        self.assertEqual(detector.strikes, 2)
        self.assertEqual(detector.rejected_candidates, 1)
        self.assertEqual(detector.trigger_step, 0)
        self.assertEqual(detector.decision_history, ["EXCEPTION_SOURCE_RESTORE"])
        self.assertEqual(detector.trigger_reason, "exception:RuntimeError")
        self.assertTrue(model.training)
        self.assertEqual(
            [module.training for module in model.modules()], entry_module_training
        )
        self.assertEqual(
            [parameter.requires_grad for parameter in model.parameters()], entry_flags
        )
        torch.testing.assert_close(model.classifier_weight().grad, entry_gradient)

    def test_invalid_source_anchors_never_mutate_call_start_weight(self):
        for invalid_kind in ("missing", "nonfinite", "shape"):
            with self.subTest(invalid_kind=invalid_kind):
                model, bundle = self._fixture()
                call_start = model.classifier_weight().detach().clone()
                if invalid_kind == "missing":
                    model.source_classifier_anchor_valid.fill_(False)
                    pattern = "not initialized"
                elif invalid_kind == "nonfinite":
                    model.source_classifier_weight.fill_(float("nan"))
                    pattern = "non-finite"
                else:
                    delattr(model, "source_classifier_weight")
                    model.source_classifier_weight = torch.zeros(1)
                    pattern = "not registered persistently"
                with self.assertRaisesRegex(RuntimeError, pattern):
                    adapt_classifier(model, bundle.unlabeled(), "full_method", steps=1)
                torch.testing.assert_close(
                    model.classifier_weight(), call_start, rtol=0, atol=0
                )

    def test_forged_anchor_without_validity_buffer_is_rejected(self):
        model, bundle = self._fixture()
        call_start = model.classifier_weight().detach().clone()
        delattr(model, "source_classifier_anchor_valid")
        model.source_classifier_anchor_valid = torch.tensor(True)
        model.source_classifier_weight.zero_()
        with self.assertRaisesRegex(RuntimeError, "not registered persistently"):
            adapt_classifier(model, bundle.unlabeled(), "source_only")
        torch.testing.assert_close(
            model.classifier_weight(), call_start, rtol=0, atol=0
        )

    def test_nonpersistent_anchor_buffers_are_rejected(self):
        for buffer_name in (
            "source_classifier_weight",
            "source_classifier_anchor_valid",
        ):
            with self.subTest(buffer_name=buffer_name):
                model, bundle = self._fixture()
                call_start = model.classifier_weight().detach().clone()
                model._non_persistent_buffers_set.add(buffer_name)
                with self.assertRaisesRegex(
                    RuntimeError, "not registered persistently"
                ):
                    adapt_classifier(model, bundle.unlabeled(), "source_only")
                torch.testing.assert_close(
                    model.classifier_weight(), call_start, rtol=0, atol=0
                )

    def test_detector_exception_uses_local_step_not_history_length(self):
        model, bundle = self._fixture()
        detector = DetectorState()
        with patch(
            "adaptation.detector_should_halt",
            side_effect=RuntimeError("injected detector failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected detector"):
                adapt_classifier(
                    model,
                    bundle.unlabeled(),
                    "full_method",
                    steps=1,
                    detector=detector,
                )
        self.assertEqual(len(detector.delta_history), 0)
        self.assertEqual(detector.trigger_step, 0)
        self.assertEqual(detector.rejected_candidates, 1)
        self.assertEqual(detector.accepted_candidates, 0)
        self.assertEqual(detector.decision_history, ["EXCEPTION_SOURCE_RESTORE"])

    def test_exception_after_recorded_accept_rolls_back_detector_transaction(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState(delta_tolerance=1.0, phi_tolerance=1.0)
        with patch.object(
            DetectorState,
            "to_dict",
            side_effect=RuntimeError("injected serialization failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "serialization failure"):
                adapt_classifier(
                    model,
                    bundle.unlabeled(),
                    "full_method",
                    steps=1,
                    detector=detector,
                )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.accepted_candidates, 0)
        self.assertEqual(detector.rejected_candidates, 1)
        self.assertEqual(detector.delta_history, [])
        self.assertEqual(detector.phi_history, [])
        self.assertEqual(detector.decision_history, ["EXCEPTION_SOURCE_RESTORE"])

    def test_exception_after_recorded_reject_rolls_back_detector_transaction(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState(delta_tolerance=-1.0, phi_tolerance=-1.0)
        with patch.object(
            DetectorState,
            "to_dict",
            side_effect=RuntimeError("injected serialization failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "serialization failure"):
                adapt_classifier(
                    model,
                    bundle.unlabeled(),
                    "full_method",
                    steps=1,
                    detector=detector,
                )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(detector.accepted_candidates, 0)
        self.assertEqual(detector.rejected_candidates, 1)
        self.assertEqual(detector.delta_history, [])
        self.assertEqual(detector.phi_history, [])
        self.assertEqual(detector.decision_history, ["EXCEPTION_SOURCE_RESTORE"])

    def test_success_restores_mixed_module_training_flags(self):
        model, bundle = self._fixture(use_bn=True)
        model.train()
        model.bn.eval()
        entry = [module.training for module in model.modules()]
        adapt_classifier(model, bundle.unlabeled(), "full_method", steps=1)
        self.assertEqual([module.training for module in model.modules()], entry)

    def test_source_anchor_survives_state_dict_round_trip(self):
        model, bundle = self._fixture()
        state = model.state_dict()
        restored, _ = self._fixture(seed=4)
        restored.load_state_dict(state)
        self.assertTrue(bool(restored.source_classifier_anchor_valid))
        torch.testing.assert_close(
            restored.source_classifier_weight,
            model.source_classifier_weight,
            rtol=0,
            atol=0,
        )

    def test_reject_retains_non_source_accepted_checkpoint(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        adapt_classifier(model, bundle.unlabeled(), "full_method", steps=1)
        accepted = model.classifier_weight().detach().clone()
        self.assertFalse(torch.equal(accepted, source))

        detector = DetectorState(delta_tolerance=-1.0, phi_tolerance=-1.0)
        adapt_classifier(
            model, bundle.unlabeled(), "full_method", steps=1, detector=detector
        )
        torch.testing.assert_close(model.classifier_weight(), accepted, rtol=0, atol=0)
        self.assertEqual(detector.decision_history, ["REJECT_RETAIN_CHECKPOINT"])

    def test_accept_then_two_rejects_restore_source(self):
        model, bundle = self._fixture()
        source = model.source_classifier_weight.detach().clone()
        detector = DetectorState()
        decisions = [(False, None), (True, "injected"), (True, "injected")]
        with patch("adaptation.detector_should_halt", side_effect=decisions):
            adapt_classifier(
                model,
                bundle.unlabeled(),
                "full_method",
                steps=3,
                detector=detector,
                tol=0.0,
            )
        torch.testing.assert_close(model.classifier_weight(), source, rtol=0, atol=0)
        self.assertEqual(
            detector.decision_history,
            ["ACCEPT", "REJECT_RETAIN_CHECKPOINT", "REJECT_SOURCE_HALT"],
        )

    def test_unknown_method_is_rejected(self):
        model, bundle = self._fixture()
        with self.assertRaisesRegex(ValueError, "Unsupported adaptation method"):
            adapt_classifier(model, bundle.unlabeled(), "full_methd", steps=1)


if __name__ == "__main__":
    unittest.main()
