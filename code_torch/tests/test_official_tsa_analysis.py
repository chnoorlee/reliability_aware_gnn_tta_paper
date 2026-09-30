import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import official_tsa_risk_audit as official_audit
from official_tsa_analysis import _operating_metrics_with_failures
from official_tsa_risk_audit import (
    _adapter_data_from_target,
    _data_keys,
    _endpoint_proxy_scopes,
    _finalize_row_hashes,
    _guard_signals,
    _json_sha256,
    _local_code_sha256,
    _refresh_row_content_sha256,
    _scope_masks,
    _source_fingerprint,
    _target_fingerprints,
    _tensor_fingerprint,
)


class OfficialAuditFailureAccountingTests(unittest.TestCase):
    def test_scope_target_reproduces_official_guard_reduction_order(self):
        generator = torch.Generator().manual_seed(11)
        source = torch.softmax(torch.randn(1000, 4, generator=generator), dim=1)
        candidate = torch.softmax(torch.randn(1000, 4, generator=generator), dim=1)
        edge_index = torch.randint(0, 1000, (2, 5000), generator=generator)
        target = SimpleNamespace(
            edge_index=edge_index,
            tgt_test_mask=torch.arange(1000) % 3 == 0,
        )
        delta, phi = _guard_signals(edge_index, source, candidate)
        scoped = _endpoint_proxy_scopes(target, source, candidate)["target"]
        self.assertEqual(scoped["delta"], delta)
        self.assertEqual(scoped["phi"], phi)

    def test_tensor_fingerprint_is_device_and_layout_independent(self):
        contiguous = torch.tensor([[1.0, -0.0], [float("nan"), float("inf")]])
        noncontiguous = contiguous.t().contiguous().t()

        first = _tensor_fingerprint(contiguous, field="x", kind="float")
        second = _tensor_fingerprint(noncontiguous, field="x", kind="float")

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["canonical_dtype"], "<f8")
        self.assertEqual(
            first["nonfinite_counts"],
            {"nan": 1, "positive_infinity": 1, "negative_infinity": 0},
        )
        self.assertNotEqual(
            first["sha256"],
            _tensor_fingerprint(contiguous, field="other_x", kind="float")["sha256"],
        )
        if torch.cuda.is_available():
            cuda = _tensor_fingerprint(contiguous.cuda(), field="x", kind="float")
            self.assertEqual(first["sha256"], cuda["sha256"])

    def test_target_fingerprint_binds_inputs_labels_masks_and_edge_order(self):
        target = SimpleNamespace(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_weight=torch.tensor([1.0, 0.5, 1.5]),
            y=torch.tensor([0, 1, 0]),
            tgt_val_mask=torch.tensor([True, False, False]),
            tgt_test_mask=torch.tensor([False, True, True]),
            tgt_test_idx=torch.tensor([1, 2]),
            num_classes=2,
        )
        adapt = _adapter_data_from_target(target)
        scopes = _scope_masks(target, 3)

        baseline = _target_fingerprints(target, adapt, scopes)
        self.assertEqual(baseline, _target_fingerprints(target, adapt, scopes))
        self.assertEqual(len(baseline["adapter_input_fingerprint"]["sha256"]), 64)
        self.assertNotIn("masks", baseline["adapter_input_fingerprint"]["components"])
        self.assertNotIn("scalars", baseline["adapter_input_fingerprint"])
        json.dumps(baseline, sort_keys=True)

        changed_x = adapt.clone()
        changed_x.x = adapt.x.clone()
        changed_x.x[0, 0] += 1.0
        self.assertNotEqual(
            baseline["adapter_input_fingerprint"]["sha256"],
            _target_fingerprints(target, changed_x, scopes)[
                "adapter_input_fingerprint"
            ]["sha256"],
        )

        changed_edges = adapt.clone()
        changed_edges.edge_index = adapt.edge_index[:, [1, 0, 2]]
        self.assertNotEqual(
            baseline["adapter_input_fingerprint"]["sha256"],
            _target_fingerprints(target, changed_edges, scopes)[
                "adapter_input_fingerprint"
            ]["sha256"],
        )

        changed_labels = SimpleNamespace(**vars(target))
        changed_labels.y = target.y.clone()
        changed_labels.y[0] = 1
        changed_label_fingerprints = _target_fingerprints(
            changed_labels, adapt, _scope_masks(changed_labels, 3)
        )
        self.assertEqual(
            baseline["adapter_input_fingerprint"]["sha256"],
            changed_label_fingerprints["adapter_input_fingerprint"]["sha256"],
        )
        self.assertNotEqual(
            baseline["offline_evaluation_fingerprint"]["sha256"],
            changed_label_fingerprints["offline_evaluation_fingerprint"]["sha256"],
        )

        changed_metadata = SimpleNamespace(**vars(target))
        changed_metadata.num_classes = torch.tensor(999)
        changed_metadata_adapt = _adapter_data_from_target(changed_metadata)
        self.assertEqual(
            baseline["adapter_input_fingerprint"]["sha256"],
            _target_fingerprints(changed_metadata, changed_metadata_adapt, scopes)[
                "adapter_input_fingerprint"
            ]["sha256"],
        )

        changed_mask_target = SimpleNamespace(**vars(target))
        changed_mask_target.tgt_test_mask = torch.tensor([True, False, True])
        changed_mask_fingerprints = _target_fingerprints(
            changed_mask_target,
            adapt,
            _scope_masks(changed_mask_target, 3),
        )
        self.assertEqual(
            baseline["adapter_input_fingerprint"]["sha256"],
            changed_mask_fingerprints["adapter_input_fingerprint"]["sha256"],
        )
        self.assertNotEqual(
            baseline["offline_evaluation_fingerprint"]["sha256"],
            changed_mask_fingerprints["offline_evaluation_fingerprint"]["sha256"],
        )
        self.assertNotEqual(
            baseline["proxy_scope_index_sha256"],
            changed_mask_fingerprints["proxy_scope_index_sha256"],
        )

    def test_adapter_boundary_excludes_evaluation_fields_and_clones_storage(self):
        target = SimpleNamespace(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_weight=torch.tensor([1.0, 0.5]),
            y=torch.tensor([0, 1]),
            tgt_train_mask=torch.tensor([True, False]),
            tgt_val_mask=torch.tensor([False, True]),
            tgt_test_mask=torch.tensor([True, True]),
            tgt_test_idx=torch.tensor([0, 1]),
            evaluation_indices=torch.tensor([0]),
            num_classes=torch.tensor(2),
        )
        originals = {
            name: getattr(target, name).clone()
            for name in ("x", "edge_index", "edge_weight", "y", "num_classes")
        }

        adapt = _adapter_data_from_target(target)

        self.assertEqual(
            set(_data_keys(adapt)),
            {"x", "edge_index", "edge_weight", "y"},
        )
        self.assertFalse(any(name.endswith("_mask") for name in _data_keys(adapt)))
        non_graph_index_fields = {
            name
            for name in _data_keys(adapt)
            if name != "edge_index" and ("index" in name or name.endswith("_idx"))
        }
        self.assertEqual(non_graph_index_fields, set())
        self.assertTrue(torch.equal(adapt.y, torch.zeros_like(target.y)))
        self.assertNotEqual(adapt.x.data_ptr(), target.x.data_ptr())
        self.assertNotEqual(adapt.edge_index.data_ptr(), target.edge_index.data_ptr())
        self.assertNotEqual(adapt.edge_weight.data_ptr(), target.edge_weight.data_ptr())

        adapt.x[0, 0] += 10
        adapt.edge_index[0, 0] = 1
        adapt.edge_weight[0] += 10
        adapt.y[0] = 1
        for name, expected in originals.items():
            self.assertTrue(torch.equal(getattr(target, name), expected), name)

        for leaked_field, value in (
            ("tgt_test_mask", target.tgt_test_mask.clone()),
            ("num_classes", target.y.max() + 1),
        ):
            with self.subTest(leaked_field=leaked_field):
                leaked = _adapter_data_from_target(target)
                setattr(leaked, leaked_field, value)
                with self.assertRaisesRegex(ValueError, "structural input boundary"):
                    _target_fingerprints(target, leaked, _scope_masks(target, 2))

    def test_row_hashes_bind_protocol_inputs_outputs_and_derived_results(self):
        identity = {
            "data_config": "CSBM1",
            "source_setting": "source",
            "target_setting": "target",
            "method": "T3A",
            "model": "GPRGNN",
            "seed": 99,
        }
        result_fields = {
            "candidate_status": "finite",
            "source_accuracy": 0.7,
            "candidate_accuracy": 0.6,
            "source_relative_accuracy": -0.1,
            "delta": 0.02,
            "phi": 0.1,
            "fixed_guard_accept": True,
        }
        provenance = {
            "resolved_config_sha256": "a" * 64,
            "source_training_config_sha256": "b" * 64,
            "source_checkpoint_sha256": "c" * 64,
            "source_data_fingerprint": {"sha256": "d" * 64},
            "adapter_input_fingerprint": {"sha256": "e" * 64},
            "offline_evaluation_fingerprint": {"sha256": "f" * 64},
            "proxy_scope_index_sha256": {"target": "1" * 64},
            "source_probabilities_fingerprint": {"sha256": "2" * 64},
            "candidate_probabilities_fingerprint": {"sha256": "3" * 64},
        }
        protocol = _json_sha256({"guard": [0.05, 0.20], "commit": "upstream"})

        baseline = _finalize_row_hashes(identity, result_fields, provenance, protocol)
        self.assertEqual(
            baseline,
            _finalize_row_hashes(identity, result_fields, provenance, protocol),
        )
        for name in (
            "row_identity_sha256",
            "row_protocol_sha256",
            "row_inputs_sha256",
            "row_outputs_sha256",
            "row_evidence_sha256",
            "row_content_sha256",
        ):
            self.assertEqual(len(baseline[name]), 64, name)

        deterministic_content_hash = baseline["row_content_sha256"]
        baseline["runtime_seconds"] = 1.25
        _refresh_row_content_sha256(baseline)
        self.assertNotEqual(deterministic_content_hash, baseline["row_content_sha256"])
        self.assertEqual(
            baseline["row_content_sha256"],
            _json_sha256(
                {
                    key: value
                    for key, value in baseline.items()
                    if key != "row_content_sha256"
                }
            ),
        )
        first_runtime_hash = baseline["row_content_sha256"]
        baseline["runtime_seconds"] = 2.5
        _refresh_row_content_sha256(baseline)
        self.assertNotEqual(first_runtime_hash, baseline["row_content_sha256"])

        changed_result = {**result_fields, "fixed_guard_accept": False}
        changed_derived = _finalize_row_hashes(
            identity, changed_result, provenance, protocol
        )
        self.assertNotEqual(
            baseline["row_evidence_sha256"], changed_derived["row_evidence_sha256"]
        )
        self.assertNotEqual(
            baseline["row_content_sha256"], changed_derived["row_content_sha256"]
        )

        changed_protocol = _finalize_row_hashes(
            identity, result_fields, provenance, "9" * 64
        )
        self.assertNotEqual(
            baseline["row_evidence_sha256"], changed_protocol["row_evidence_sha256"]
        )

        changed_inputs = {
            **provenance,
            "source_data_fingerprint": {"sha256": "8" * 64},
        }
        self.assertNotEqual(
            baseline["row_inputs_sha256"],
            _finalize_row_hashes(identity, result_fields, changed_inputs, protocol)[
                "row_inputs_sha256"
            ],
        )

        changed_outputs = {
            **provenance,
            "candidate_probabilities_fingerprint": {"sha256": "7" * 64},
        }
        self.assertNotEqual(
            baseline["row_outputs_sha256"],
            _finalize_row_hashes(identity, result_fields, changed_outputs, protocol)[
                "row_outputs_sha256"
            ],
        )

    def test_local_code_manifest_includes_compatibility_layer(self):
        manifest = _local_code_sha256()
        self.assertEqual(
            set(manifest),
            {
                "official_tsa_risk_audit.py",
                "proxy_scope_audit.py",
                "run_official_tsa_compat.py",
            },
        )
        self.assertTrue(all(len(value) == 64 for value in manifest.values()))

    def test_source_fingerprint_binds_consumed_graph_fields(self):
        source = SimpleNamespace(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            edge_weight=torch.tensor([1.0, 0.5, 1.5]),
            y=torch.tensor([0, 1, 0]),
            src_train_mask=torch.tensor([True, False, False]),
            src_val_mask=torch.tensor([False, True, False]),
            src_test_mask=torch.tensor([False, False, True]),
            src_mask=torch.tensor([True, True, True]),
            num_nodes=3,
            num_edges=3,
            num_classes=2,
        )
        baseline = _source_fingerprint(source)
        self.assertEqual(baseline, _source_fingerprint(source))
        self.assertEqual(len(baseline["sha256"]), 64)
        self.assertEqual(
            set(baseline["components"]["masks"]),
            {"src_mask", "src_test_mask", "src_train_mask", "src_val_mask"},
        )
        json.dumps(baseline, sort_keys=True)

        variants = []
        changed_x = SimpleNamespace(**vars(source))
        changed_x.x = source.x.clone()
        changed_x.x[0, 0] += 1.0
        variants.append(("x", changed_x))

        changed_edges = SimpleNamespace(**vars(source))
        changed_edges.edge_index = source.edge_index[:, [1, 0, 2]]
        variants.append(("edge order", changed_edges))

        changed_weight = SimpleNamespace(**vars(source))
        changed_weight.edge_weight = source.edge_weight.clone()
        changed_weight.edge_weight[0] += 0.25
        variants.append(("edge weight", changed_weight))

        changed_y = SimpleNamespace(**vars(source))
        changed_y.y = source.y.clone()
        changed_y.y[0] = 1
        variants.append(("labels", changed_y))

        changed_mask = SimpleNamespace(**vars(source))
        changed_mask.src_train_mask = torch.tensor([False, True, False])
        variants.append(("mask", changed_mask))

        changed_scalar = SimpleNamespace(**vars(source))
        changed_scalar.num_classes = 3
        variants.append(("scalar", changed_scalar))

        for label, changed in variants:
            with self.subTest(field=label):
                self.assertNotEqual(
                    baseline["sha256"], _source_fingerprint(changed)["sha256"]
                )

    def test_run_replaces_stale_success_on_pre_first_row_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            output.write_text('{"status":"complete"}', encoding="utf-8")
            with (
                patch.object(
                    official_audit,
                    "_exclusive_audit_lock",
                    return_value=nullcontext({"run_id": "test"}),
                ),
                patch.object(
                    official_audit,
                    "_run_locked",
                    side_effect=RuntimeError("pre-first-row failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "pre-first-row"),
            ):
                official_audit.run(output, ("CSBM1",), ("T3A",), (99,), "GPRGNN")

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error_type"], "RuntimeError")
            self.assertEqual(payload["records"], [])

    def test_run_persists_base_exception_failures(self):
        for error in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(
                error=type(error).__name__
            ), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "audit.json"
                with (
                    patch.object(
                        official_audit,
                        "_exclusive_audit_lock",
                        return_value=nullcontext({"run_id": "test"}),
                    ),
                    patch.object(official_audit, "_run_locked", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    official_audit.run(output, ("CSBM1",), ("T3A",), (99,), "GPRGNN")

                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error_type"], type(error).__name__)
                self.assertEqual(payload["records"], [])

    def test_post_repository_status_failure_preserves_completed_rows(self):
        row = {
            "data_config": "CSBM1",
            "target_setting": "target",
            "seed": 99,
            "source_accuracy": 0.7,
            "candidate_accuracy": 0.6,
            "source_relative_accuracy": -0.1,
            "candidate_status": "finite",
            "fixed_guard_accept": False,
            "source_checkpoint_path": "model/source.pt",
            "source_checkpoint_sha256": "d" * 64,
            "adapter_input_fingerprint": {"sha256": "a" * 64},
            "offline_evaluation_fingerprint": {"sha256": "b" * 64},
            "proxy_scope_index_sha256": {"target": "c" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tsa_root = root / "tsa"
            tsa_root.mkdir()
            output = root / "audit.json"
            clean = {"tracked_changes": [], "untracked_paths": []}
            with (
                patch.object(official_audit, "TSA_ROOT", tsa_root),
                patch.object(official_audit, "install_compat"),
                patch.object(official_audit, "_git_commit", return_value="commit"),
                patch.object(
                    official_audit,
                    "_git_status",
                    side_effect=[clean, RuntimeError("post-status failure")],
                ),
                patch.object(official_audit, "_run_one", return_value=row),
                patch.object(
                    official_audit,
                    "_exclusive_audit_lock",
                    return_value=nullcontext({"run_id": "test"}),
                ),
                self.assertRaisesRegex(RuntimeError, "post-status"),
            ):
                official_audit.run(output, ("CSBM1",), ("T3A",), (99,), "GPRGNN")

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error_type"], "RuntimeError")
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["data_config"], "CSBM1")

    def test_target_fingerprint_drift_across_methods_fails_closed(self):
        first = {
            "data_config": "CSBM1",
            "target_setting": "target",
            "seed": 99,
            "source_accuracy": 0.7,
            "candidate_accuracy": 0.6,
            "source_relative_accuracy": -0.1,
            "candidate_status": "finite",
            "fixed_guard_accept": False,
            "source_checkpoint_path": "model/source.pt",
            "source_checkpoint_sha256": "d" * 64,
            "adapter_input_fingerprint": {"sha256": "a" * 64},
            "offline_evaluation_fingerprint": {"sha256": "b" * 64},
            "proxy_scope_index_sha256": {"target": "c" * 64},
        }
        second = {
            **first,
            "adapter_input_fingerprint": {"sha256": "e" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tsa_root = root / "tsa"
            tsa_root.mkdir()
            output = root / "audit.json"
            clean = {"tracked_changes": [], "untracked_paths": []}
            with (
                patch.object(official_audit, "TSA_ROOT", tsa_root),
                patch.object(official_audit, "install_compat"),
                patch.object(official_audit, "_git_commit", return_value="commit"),
                patch.object(official_audit, "_git_status", return_value=clean),
                patch.object(official_audit, "_run_one", side_effect=[first, second]),
                patch.object(
                    official_audit,
                    "_exclusive_audit_lock",
                    return_value=nullcontext({"run_id": "test"}),
                ),
                self.assertRaisesRegex(RuntimeError, "Target data changed"),
            ):
                official_audit.run(
                    output,
                    ("CSBM1",),
                    ("T3A", "TSA_T3A"),
                    (99,),
                    "GPRGNN",
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(len(payload["records"]), 1)
            self.assertIn("Target data changed", payload["error_message"])

    def test_empty_run_selection_fails_closed(self):
        cases = (
            ((), ("T3A",), (99,)),
            (("CSBM1",), (), (99,)),
            (("CSBM1",), ("T3A",), ()),
        )
        for data_configs, methods, seeds in cases:
            with self.subTest(
                data_configs=data_configs, methods=methods, seeds=seeds
            ), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "audit.json"
                with self.assertRaises(ValueError):
                    official_audit.run(output, data_configs, methods, seeds, "GPRGNN")
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error_type"], "ValueError")

    def test_atomic_write_cleans_temporary_file_after_replace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            with (
                patch.object(
                    official_audit.os, "replace", side_effect=OSError("replace")
                ),
                self.assertRaisesRegex(OSError, "replace"),
            ):
                official_audit._write_json_atomic(output, {"status": "running"})
            self.assertEqual(list(Path(temporary).glob(".audit.json.*.tmp")), [])

    def test_nonfinite_candidate_counts_as_fail_closed_attempt(self):
        rows = [
            {
                "candidate_status": "finite",
                "source_relative_accuracy": 0.12,
                "delta": 0.01,
                "phi": 0.02,
            },
            {
                "candidate_status": "finite",
                "source_relative_accuracy": -0.20,
                "delta": 0.10,
                "phi": 0.40,
            },
            {"candidate_status": "nonfinite_probability"},
        ]

        metrics = _operating_metrics_with_failures(rows)

        self.assertEqual(metrics["valid_candidate_n"], 2)
        self.assertEqual(metrics["audit_attempts"], 3)
        self.assertEqual(metrics["candidate_failures"], 1)
        self.assertAlmostEqual(metrics["valid_candidate_coverage"], 0.5)
        self.assertAlmostEqual(metrics["coverage"], 1 / 3)
        self.assertAlmostEqual(metrics["mean_deployed_gain"], 0.04)
        self.assertAlmostEqual(metrics["mean_deployed_gain_valid_candidates"], 0.06)


if __name__ == "__main__":
    unittest.main()
