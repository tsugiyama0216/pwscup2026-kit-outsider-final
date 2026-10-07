"""Exercise real child processes and reject cross-case or changed artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import textwrap
import unittest
from unittest.mock import patch
from uuid import uuid4

from mock_TS.common import OUTPUT_ROOT, read_csv, read_json, sha256
from mock_TS.generation import generate
from mock_TS.runner import attack_bundle, export_bundle, load_suite, score_submission


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner_id = uuid4().hex
        cls.temporary = (OUTPUT_ROOT / ("test_runner_" + cls.owner_id)).resolve()
        cls.temporary.mkdir(parents=True, exist_ok=False)
        cls.addClassCleanup(cls.cleanup_owned_directory)
        cls.suite = cls.temporary / "suite"
        cls.manifest = generate(cls.suite, count=1, r_count=1)
        cls.case_id = cls.manifest["cases"][0]["id"]
        cls.model = "M1_v5"
        cls.bundle = export_bundle(cls.suite, cls.case_id, cls.model,
                                   cls.temporary / "shared_bundle")
        cls.predictions = attack_bundle(cls.bundle, cls.temporary / "shared_predictions",
                                        method="constant", timeout=60)

    @classmethod
    def cleanup_owned_directory(cls):
        # Resolve and check the exact owned UUID subtree before any recursive removal.
        resolved = cls.temporary.resolve()
        if (resolved.parent != OUTPUT_ROOT.resolve()
                or resolved.name != "test_runner_" + cls.owner_id):
            raise AssertionError("Refusing to remove a directory not owned by this test")
        if resolved.exists():
            shutil.rmtree(resolved)

    def setUp(self):
        self.output = self.temporary / self._testMethodName
        self.output.mkdir(exist_ok=False)

    def plugin(self, source):
        path = self.output / "candidate.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return path

    def test_dataclass_plugin_runs_in_real_child(self):
        plugin = self.plugin('''
            from __future__ import annotations
            from dataclasses import dataclass

            @dataclass
            class Settings:
                confidence: float = 0.375
                time: float = 9.5

            def predict(*, c, background, candidates, challenge, cfg):
                settings = Settings()
                return {"mia": [settings.confidence] * len(candidates),
                        "aia": [settings.time] * len(challenge)}
        ''')
        predictions = attack_bundle(self.bundle, self.output / "predictions",
                                     plugin=plugin, timeout=60)
        self.assertTrue(read_csv(predictions / "F_mia.csv")["0"].eq(.375).all())
        self.assertTrue(read_csv(predictions / "F_aia.csv").time_hat.eq(9.5).all())
        self.assertNotEqual(read_json(predictions / "worker.json")["pid"], os.getpid())
        self.assertEqual(read_json(predictions / "prediction_manifest.json")["plugin_sha256"],
                         sha256(plugin))

    def test_consecutive_releases_have_fresh_process_and_working_state(self):
        plugin = self.plugin('''
            import builtins
            import json
            from pathlib import Path

            def predict(*, c, background, candidates, challenge, cfg):
                count = getattr(builtins, "_mock_ts_test_calls", 0) + 1
                builtins._mock_ts_test_calls = count
                state = Path("previous_attack.txt")
                disk_count = int(state.read_text()) + 1 if state.exists() else 1
                state.write_text(str(disk_count))
                print("STATE " + json.dumps({"cwd": str(Path.cwd()),
                      "memory_calls": count, "disk_calls": disk_count}), flush=True)
                return {"mia": [.25 if count == disk_count == 1 else .75] * len(candidates),
                        "aia": [float(count + disk_count)] * len(challenge)}
        ''')
        second_bundle = export_bundle(self.suite, self.case_id, "S1", self.output / "S1_bundle")
        results, states, receipts = [], [], []
        for index, bundle in enumerate((self.bundle, second_bundle)):
            result = attack_bundle(bundle, self.output / f"prediction_{index}",
                                   plugin=plugin, timeout=60)
            results.append(read_json(result / "worker.json"))
            lines = (result / "attack.log").read_text(encoding="utf-8").splitlines()
            states.append(json.loads(next(line[6:] for line in lines if line.startswith("STATE "))))
            receipts.append(read_json(result / "prediction_manifest.json"))
            self.assertTrue(read_csv(result / "F_mia.csv")["0"].eq(.25).all())
            self.assertTrue(read_csv(result / "F_aia.csv").time_hat.eq(2.).all())
        self.assertNotEqual(results[0]["pid"], results[1]["pid"])
        self.assertNotEqual(states[0]["cwd"], states[1]["cwd"])
        self.assertNotEqual(receipts[0]["input_fingerprint"], receipts[1]["input_fingerprint"])
        for state, result in zip(states, results):
            self.assertEqual((state["memory_calls"], state["disk_calls"]), (1, 1))
            self.assertFalse(Path(state["cwd"]).exists(), "Worker scratch must be removed")
            self.assertEqual(result["public_arguments"],
                             ["c", "background", "candidates", "challenge", "cfg"])
            self.assertTrue(result["fresh_process"])
        load_suite(self.suite)  # Original suite and release bytes still match their manifest.

    def test_plugin_mutating_input_is_rejected_without_changing_bundle(self):
        plugin = self.plugin('''
            from pathlib import Path

            def predict(*, c, background, candidates, challenge, cfg):
                Path("B.csv").write_text("mutated input\\n", encoding="utf-8")
                return {"mia": [.5] * len(candidates), "aia": [10.] * len(challenge)}
        ''')
        before = sha256(self.bundle / "B.csv")
        output = self.output / "predictions"
        with self.assertRaisesRegex(ValueError, "Input changed: B.csv"):
            attack_bundle(self.bundle, output, plugin=plugin, timeout=60)
        self.assertEqual(sha256(self.bundle / "B.csv"), before)
        self.assertFalse((output / "prediction_manifest.json").exists())
        self.assertFalse((output / "F_mia.csv").exists())
        load_suite(self.suite)

    def test_receipt_for_another_C_is_rejected_before_floor(self):
        with patch("mock_TS.scoring.estimate_floor") as estimate:
            with self.assertRaisesRegex(ValueError, "different case or C"):
                score_submission(self.suite, self.case_id, "S1", self.predictions,
                                 self.output / "evaluation")
            estimate.assert_not_called()
        self.assertFalse((self.output / "evaluation").exists())

    def test_receipt_for_another_case_is_rejected_before_floor(self):
        other_suite = self.output / "other_suite"
        generate(other_suite, seed=int(self.manifest["seed"]) + 1, count=1, r_count=1)
        with patch("mock_TS.scoring.estimate_floor") as estimate:
            with self.assertRaisesRegex(ValueError, "different case or C"):
                score_submission(other_suite, self.case_id, self.model, self.predictions,
                                 self.output / "evaluation")
            estimate.assert_not_called()
        self.assertFalse((self.output / "evaluation").exists())

    def test_changed_prediction_cannot_keep_verified_receipt(self):
        copied = self.output / "changed_predictions"
        shutil.copytree(self.predictions, copied)
        frame = read_csv(copied / "F_mia.csv")
        frame.loc[0, "0"] = .75
        frame.to_csv(copied / "F_mia.csv", index=False)
        with patch("mock_TS.scoring.estimate_floor") as estimate:
            with self.assertRaisesRegex(ValueError, "Input changed: F_mia.csv"):
                score_submission(self.suite, self.case_id, self.model, copied,
                                 self.output / "evaluation")
            estimate.assert_not_called()

    def test_invalid_external_ID_is_rejected_before_floor(self):
        external = self.output / "external"
        external.mkdir()
        frame = read_csv(self.predictions / "F_mia.csv")
        frame.loc[0, "record_id"] = frame.loc[1, "record_id"]
        frame.to_csv(external / "F_mia.csv", index=False)
        shutil.copyfile(self.predictions / "F_aia.csv", external / "F_aia.csv")
        with patch("mock_TS.scoring.estimate_floor") as estimate:
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                score_submission(self.suite, self.case_id, self.model, external,
                                 self.output / "evaluation")
            estimate.assert_not_called()

    def test_export_does_not_overwrite_existing_output(self):
        output = self.output / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"previous experiment\n")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            export_bundle(self.suite, self.case_id, self.model, output)
        self.assertEqual(sentinel.read_bytes(), b"previous experiment\n")
        self.assertEqual(list(output.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
