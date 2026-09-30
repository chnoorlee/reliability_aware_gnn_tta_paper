import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import scipy.sparse
import torch

import stress_surface


class StressSurfacePersistenceTests(unittest.TestCase):
    @staticmethod
    def _bundle():
        x_np = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        return SimpleNamespace(
            x_np=x_np,
            adj=scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=np.float64),
            y_np=np.asarray([0, 1], dtype=np.int64),
            train_idx=np.asarray([0], dtype=np.int64),
            val_idx=np.asarray([], dtype=np.int64),
            test_idx=np.asarray([1], dtype=np.int64),
            x=torch.as_tensor(x_np, dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.tensor([0, 1], dtype=torch.long),
            train_mask=torch.tensor([True, False]),
            val_mask=torch.tensor([False, False]),
            test_mask=torch.tensor([False, True]),
            num_classes=2,
        )

    def test_bundle_hash_binds_consumed_tensor_and_scalar_representations(self):
        baseline = self._bundle()
        baseline_hash = stress_surface._bundle_sha256(baseline)
        mutations = {
            "x": lambda bundle: bundle.x.__setitem__((0, 0), 2.0),
            "edge_index": lambda bundle: bundle.edge_index.__setitem__((0, 0), 1),
            "test_mask": lambda bundle: bundle.test_mask.__setitem__(0, True),
            "num_classes": lambda bundle: setattr(bundle, "num_classes", 3),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                self.assertNotEqual(
                    stress_surface._bundle_sha256(changed), baseline_hash
                )

    def test_code_provenance_binds_all_local_execution_dependencies(self):
        manifest = stress_surface._code_provenance()
        expected = {
            "models.py",
            "_np_bridge.py",
            "../code/data.py",
            "../code/utils.py",
            "../code/webkb_loader.py",
        }
        self.assertTrue(expected <= set(manifest))
        root = Path(stress_surface.__file__).resolve().parent
        for name in expected:
            with self.subTest(name=name):
                self.assertEqual(
                    manifest[name], stress_surface._sha256_file(root / name)
                )

    def test_run_materializes_one_shot_iterables_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            with mock.patch.object(
                stress_surface, "_run_impl", return_value=[]
            ) as implementation:
                stress_surface.run(
                    output,
                    datasets=(item for item in ["citeseer"]),
                    seeds=(item for item in [1]),
                    stress_grid=(item for item in [("edge_add", 0.5)]),
                )

            call = implementation.call_args.kwargs
            self.assertEqual(call["datasets"], ("citeseer",))
            self.assertEqual(call["seeds"], (1,))
            self.assertEqual(call["stress_grid"], (("edge_add", 0.5),))

    def test_run_records_empty_dimensions_as_failed(self):
        cases = (
            ("datasets", [], [1], [("edge_add", 0.5)]),
            ("seeds", ["citeseer"], [], [("edge_add", 0.5)]),
            ("stress_grid", ["citeseer"], [1], []),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (name, datasets, seeds, grid) in enumerate(cases):
                output = Path(directory) / f"results-{index}.json"
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        ValueError, f"{name} must not be empty"
                    ):
                        stress_surface.run(
                            output,
                            datasets=datasets,
                            seeds=seeds,
                            stress_grid=grid,
                        )
                    failed = json.loads(output.read_text(encoding="utf-8"))
                    self.assertEqual(failed["status"], "failed")
                    self.assertEqual(failed["records"], [])
                    self.assertEqual(failed["failure_type"], "ValueError")
                    self.assertEqual(
                        failed["failure_message"], f"{name} must not be empty"
                    )

    def test_run_replaces_stale_success_on_empty_input_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text(
                json.dumps({"status": "complete", "records": ["stale"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "stress_grid must not be empty"):
                stress_surface.run(
                    output,
                    datasets=["citeseer"],
                    seeds=[1],
                    stress_grid=[],
                )

            failed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["records"], [])
            self.assertEqual(failed["failure_type"], "ValueError")
            self.assertEqual(failed["failure_message"], "stress_grid must not be empty")

    def test_run_replaces_stale_success_on_generator_failure(self):
        def failing_datasets():
            yield "citeseer"
            raise RuntimeError("dataset generator failed")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text(
                json.dumps({"status": "complete", "records": ["stale"]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "dataset generator failed"):
                stress_surface.run(
                    output,
                    datasets=failing_datasets(),
                    seeds=[1],
                    stress_grid=[("edge_add", 0.5)],
                )

            failed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["records"], [])
            self.assertEqual(failed["failure_type"], "RuntimeError")
            self.assertEqual(failed["failure_message"], "dataset generator failed")

    def test_atomic_json_replace_writes_complete_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            stress_surface._write_json_atomic(output, {"status": "complete"})

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "complete"},
            )
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_atomic_json_replace_preserves_previous_file_on_replace_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            original = b'{"status":"running","records":[1]}\n'
            output.write_bytes(original)

            with mock.patch.object(
                stress_surface.os, "replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    stress_surface._write_json_atomic(
                        output, {"status": "complete", "records": [1, 2]}
                    )

            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])

    def test_run_replaces_stale_success_before_training_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            output.write_text(
                json.dumps({"status": "complete", "records": ["stale"]}),
                encoding="utf-8",
            )

            def fail_after_observing_running(*args, **kwargs):
                current = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(current["status"], "running")
                self.assertEqual(current["records"], [])
                raise RuntimeError("simulated first-training failure")

            with mock.patch.object(
                stress_surface, "train_source", side_effect=fail_after_observing_running
            ):
                with self.assertRaisesRegex(RuntimeError, "first-training failure"):
                    stress_surface.run(
                        output,
                        datasets=["citeseer"],
                        seeds=[1],
                        train_epochs=1,
                        adapt_steps=1,
                        stress_grid=(("edge_add", 0.5),),
                    )

            failed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["records"], [])
            self.assertEqual(failed["failure_type"], "RuntimeError")
            self.assertEqual(
                failed["failure_message"], "simulated first-training failure"
            )
            self.assertEqual(list(output.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
