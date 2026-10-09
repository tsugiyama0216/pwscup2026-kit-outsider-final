"""Public input boundaries and frozen dependency gates for model adapters."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

from rank_validation import adapters


def public_world():
    def people(n):
        frame = pd.DataFrame({name: np.zeros(n) for name in adapters.FEATURES})
        frame["age"] = np.full(n, 50, dtype=int)
        frame["prefecture"] = ["pref"] * n
        frame["SBP"] = np.arange(n, dtype=float) / 100 + 100
        frame["time"] = 10.
        for name in ("sex", "smoking", "onset", "death"):
            frame[name] = frame[name].astype(int)
        return frame
    b = people(1049)
    b.insert(0, "record_id", np.arange(1049))
    labels = b.copy()
    labels["is_rare"] = 0
    labels.loc[:23, "is_rare"] = 1
    challenge = people(326)[adapters.QI]
    challenge.insert(0, "challenge_row_id", np.arange(326))
    return dict(candidates=b, self_labels=labels, background=people(5143),
                self_challenge=challenge.copy(),
                targets={k: {"release": people(1049), "challenge": challenge.copy()} for k in adapters.TARGETS},
                schema={"submitted_c_columns": adapters.FEATURES, "aia_qi_columns": adapters.QI}, cfg={})


class AdapterBoundaryTests(unittest.TestCase):
    def test_complete_joint_world_passes_without_reading_private_truth(self):
        public = public_world()
        adapters.validate_public(**public)

    def test_original_background_record_ids_are_public_and_preserved(self):
        public = public_world()
        public["background"].insert(0, "record_id", np.arange(5143))
        before = public["background"].copy(deep=True)
        adapters.validate_public(**public)
        pd.testing.assert_frame_equal(public["background"], before)

    def test_single_target_and_preliminary_protocol_are_not_silently_substituted(self):
        public = public_world()
        public["targets"] = {0: public["targets"][0]}
        with self.assertRaisesRegex(ValueError, "all 22"):
            adapters.validate_public(**public)
        with self.assertRaisesRegex(ValueError, "separately verified"):
            adapters.validate_public(**public_world(), attacker_id=24)

    def test_hidden_labels_and_generation_metadata_are_rejected(self):
        for where in ("target", "candidates"):
            public = public_world()
            if where == "target":
                public["targets"][0]["style"] = "known generator"
            else:
                public["candidates"]["true_member"] = 0
            with self.assertRaises(ValueError):
                adapters.validate_public(**public)
        with self.assertRaises(TypeError):
            adapters.predict_world("Rv1", **public_world(), private_path="never_pass_truth")

    def test_bad_ids_and_self_labels_rejected(self):
        public = public_world()
        public["candidates"].loc[1, "record_id"] = 0
        with self.assertRaisesRegex(ValueError, "IDs"):
            adapters.validate_public(**public)
        public = public_world()
        public["self_labels"].loc[0, "SBP"] = -999
        with self.assertRaisesRegex(ValueError, "values differ"):
            adapters.validate_public(**public)

    def test_nonfinite_and_missing_target_rows_rejected(self):
        public = public_world()
        public["targets"][0]["release"].loc[0, "SBP"] = np.nan
        with self.assertRaisesRegex(ValueError, "values"):
            adapters.validate_public(**public)
        public = public_world()
        public["targets"][0]["challenge"] = public["targets"][0]["challenge"].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "frame"):
            adapters.validate_public(**public)

    def test_predict_rejects_unknown_model_before_import_or_inference(self):
        with patch.object(adapters, "_prepare", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "Unknown frozen model"):
                adapters.predict_world("unknown", **public_world())

    def test_guard_blocks_private_truth_and_cached_predictions(self):
        for path in (adapters.MOCK / "local-output/private/aia_truth.csv",
                     adapters.ATTACK / "Attack_R/v1/submission/F_aia.csv",
                     adapters.ROOT / "main_round/anonymization/team10/participant_data/B.csv"):
            with self.subTest(path=path):
                with adapters._inference_guard({}) as state:
                    with self.assertRaisesRegex(RuntimeError, "non-public"):
                        path.read_bytes()
                self.assertEqual(len(state["blocked_reads"]), 1)
                self.assertFalse(state["active"])

    def test_csv_cannot_be_whitelisted_as_a_fitted_model_asset(self):
        relative = "main_round/attack/Attack_R/v1/submission/F_aia.csv"
        with adapters._inference_guard({relative: "ignored"}):
            with self.assertRaisesRegex(RuntimeError, "non-public"):
                (adapters.ROOT / relative).read_bytes()

    def test_schema_exception_requires_identical_supplied_public_schema(self):
        with self.assertRaisesRegex(ValueError, "schema differs"):
            with adapters._inference_guard({}, {"wrong": True}):
                self.fail("mismatched schema entered inference")

    def test_gate_detects_changed_dependency_without_touching_frozen_files(self):
        scratch = adapters.MOCK / "local-output" / ("adapter-test-" + uuid4().hex)
        scratch.mkdir(parents=True)
        try:
            asset = scratch / "asset.json"
            manifest = scratch / "manifest.json"
            asset.write_text("{}", encoding="utf-8")
            manifest.write_text(json.dumps({"files_sha256": {"asset.json": adapters.sha(asset)}}), encoding="utf-8")
            with patch.object(adapters, "ROOT", scratch):
                self.assertEqual(len(adapters._manifest(manifest)), 1)
                asset.write_text('{"changed":true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen dependency mismatch"):
                    adapters._manifest(manifest)
        finally:
            self.assertTrue(scratch.resolve().is_relative_to((adapters.MOCK / "local-output").resolve()))
            for path in scratch.iterdir():
                path.unlink()
            scratch.rmdir()


if __name__ == "__main__":
    unittest.main()