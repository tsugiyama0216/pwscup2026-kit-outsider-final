"""Same-world baseline, strict alignment and pure numeric series boundaries."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

from rank_validation import series_worker as worker
from rank_validation.tests.test_adapters import public_world


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.root = worker.MOCK / "local-output" / ("series-baseline-test-" + uuid4().hex)
        self.root.mkdir(parents=True)
        self.public = public_world()
        self.hashes = {"B.csv": "a" * 64}
        self.mia = pd.DataFrame({"record_id": self.public["candidates"].record_id,
            **{str(k): (np.arange(1049) + k) % 1049 / 1049 for k in worker.TARGETS}})
        self.aia = pd.concat([pd.DataFrame({"target_k": k, "challenge_row_id": np.arange(326),
            "time_hat": np.arange(326)/100 + k/10000}) for k in worker.TARGETS], ignore_index=True)
        self.diagnostic = {"model_id": "Rv2", "adapter_sha256": "1"*64, "worker_sha256": "2"*64,
            "input_files_unchanged": True, "public_input_sha256": self.hashes,
            "diagnostics": {"model_id": "Rv2", "frozen_gates": {"test_fixture_only": 1}, "blocked_reads": []}}
        self.save()

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to((worker.MOCK / "local-output").resolve()))
        for path in self.root.iterdir():
            path.unlink()
        self.root.rmdir()

    def save(self):
        for name, frame in (("F_mia.csv", self.mia), ("F_aia.csv", self.aia)):
            (self.root/name).write_bytes(frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode())
        self.diagnostic["output_sha256"] = {name: worker.sha(self.root/name) for name in worker.BASELINE_FILES[:2]}
        self.write_diagnostic()

    def write_diagnostic(self):
        (self.root/"diagnostics.json").write_text(json.dumps(self.diagnostic), encoding="utf-8")

    def test_rows_and_columns_are_aligned_by_ids(self):
        self.mia = self.mia.iloc[::-1][["record_id", *map(str, reversed(worker.TARGETS))]]
        self.aia = self.aia.iloc[::-1]
        self.save()
        m, a, evidence = worker.load_baseline(self.root, self.public, self.hashes)
        np.testing.assert_array_equal(m[0], np.arange(1049)/1049)
        np.testing.assert_array_equal(a[0], np.arange(326)/100)
        self.assertEqual(evidence["model_signature"]["model_id"], "Rv2")
        worker.verify_baseline_unchanged(self.root, evidence)

    def test_other_world_or_model_is_rejected_before_numeric_use(self):
        for key, value in (("public_input_sha256", {"B.csv": "b"*64}), ("model_id", "Rv1"), ("input_files_unchanged", False)):
            original = copy.deepcopy(self.diagnostic)
            self.diagnostic[key] = value
            self.write_diagnostic()
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.load_baseline(self.root, self.public, self.hashes)
            self.diagnostic = original

    def test_saved_bytes_must_match_recorded_output_hash(self):
        with (self.root/"F_mia.csv").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "output bytes changed"):
            worker.load_baseline(self.root, self.public, self.hashes)

    def test_frozen_model_signature_is_required_and_cannot_be_forged_plugin(self):
        for change in ({"worker_sha256": "invalid"}, {"diagnostics": {**self.diagnostic["diagnostics"], "trusted_plugin": True}}):
            original = copy.deepcopy(self.diagnostic)
            self.diagnostic.update(change)
            self.write_diagnostic()
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.load_baseline(self.root, self.public, self.hashes)
            self.diagnostic = original

    def test_preexisting_full_signature_must_match_recomputed_identity(self):
        self.diagnostic["model_signature"] = {"signature_sha256": "0"*64}
        self.write_diagnostic()
        with self.assertRaisesRegex(ValueError, "Recorded baseline model signature"):
            worker.load_baseline(self.root, self.public, self.hashes)

    def test_hash_relabeling_does_not_allow_duplicate_or_foreign_ids(self):
        for frame, key, value in (("mia", "record_id", 999999), ("aia", "challenge_row_id", 999999), ("aia", "challenge_row_id", 1)):
            original_mia, original_aia = self.mia.copy(), self.aia.copy()
            getattr(self, frame).loc[0, key] = value
            self.save()
            with self.subTest(frame=frame, value=value), self.assertRaises(ValueError):
                worker.load_baseline(self.root, self.public, self.hashes)
            self.mia, self.aia = original_mia, original_aia

    def test_nonfinite_baseline_is_rejected(self):
        self.mia.loc[0, "0"] = np.nan
        self.save()
        with self.assertRaisesRegex(ValueError, "finite"):
            worker.load_baseline(self.root, self.public, self.hashes)

    def test_extra_files_and_post_load_changes_are_rejected(self):
        _, _, evidence = worker.load_baseline(self.root, self.public, self.hashes)
        extra = self.root/"fake_private.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "only two predictions"):
            worker.load_baseline(self.root, self.public, self.hashes)
        with self.assertRaisesRegex(ValueError, "Baseline files changed"):
            worker.verify_baseline_unchanged(self.root, evidence)
        extra.unlink()
        self.diagnostic["pid"] = 999999
        self.write_diagnostic()
        with self.assertRaisesRegex(ValueError, "Baseline files changed"):
            worker.verify_baseline_unchanged(self.root, evidence)


class SeriesBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker._bootstrap()

    def outputs(self):
        return {name: {"mia": {k: np.full(1049, .5) for k in worker.TARGETS},
                       "aia": {k: np.full(326, 11.) for k in worker.TARGETS}} for name in worker.MODELS}

    def test_all_six_output_frames_preserve_public_ids_and_schema(self):
        public = public_world()
        frames = worker.frames_from_outputs(self.outputs(), public)
        self.assertEqual(set(frames), set(worker.MODELS))
        self.assertEqual(frames["Rv6"]["validation"], {"mia_rows": 1049, "aia_rows": 7172, "target_count": 22})
        np.testing.assert_array_equal(frames["Rv11"]["F_mia"].record_id, public["candidates"].record_id)

    def test_incomplete_models_targets_wrong_shapes_and_nonfinite_rejected(self):
        for kind in ("model", "target", "shape", "nan"):
            outputs = self.outputs()
            if kind == "model": del outputs["Rv11"]
            if kind == "target": del outputs["Rv6"]["mia"][0]
            if kind == "shape": outputs["Rv6"]["aia"][0] = np.zeros(325)
            if kind == "nan": outputs["Rv6"]["mia"][0][0] = np.nan
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                worker.frames_from_outputs(outputs, public_world())

    def test_one_numeric_call_public_arguments_and_private_mask_from_public_api(self):
        public = public_world()
        before = public["candidates"].copy()
        calls = []
        def numeric(**kwargs):
            self.assertEqual(set(kwargs), {"candidates", "background", "targets", "challenges", "baseline_mia", "baseline_aia", "known_negative", "settings", "progress"})
            self.assertEqual(set(kwargs["targets"]), set(worker.TARGETS))
            self.assertEqual(kwargs["known_negative"].dtype, np.dtype(bool))
            self.assertFalse(kwargs["known_negative"].any())
            kwargs["candidates"].loc[0, "SBP"] = -9999
            calls.append(1)
            return self.outputs(), {}
        fake_modules = [SimpleNamespace(), SimpleNamespace(predict_all=numeric),
                        SimpleNamespace(derive_private_mask=lambda *a: (np.zeros(1049, dtype=bool), {"test_public_mask": True}))]
        gate = {"series_verified_files": 126, "manifest_sha256": "1"*64, "protocol_sha256": "2"*64}
        @contextmanager
        def audited(_source):
            yield {"blocked": [], "code_changes": [], "code_hashes": {}}
        with patch.object(worker, "_load", side_effect=fake_modules), patch.object(worker, "_sources", return_value={}), patch.object(worker, "series_gate", return_value=({"series": list(worker.MODELS)}, gate)), patch.object(worker, "read_guard", side_effect=audited):
            frames, record, _ = worker.predict_series(public, {}, {})
        self.assertEqual(calls, [1])
        pd.testing.assert_frame_equal(public["candidates"], before)
        self.assertEqual(record["frozen_gates"], gate)


if __name__ == "__main__":
    unittest.main()