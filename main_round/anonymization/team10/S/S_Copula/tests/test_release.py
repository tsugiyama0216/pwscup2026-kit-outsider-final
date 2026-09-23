"""Release-ledger tests: fabricated CSVs, mocked fits, no contest B access."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("release", ROOT / "release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.output, self.ledger = [self.root / name for name in ["data", "out", "ledger"]]
        self.data.mkdir()
        self.config_path = self.root / "frozen.json"
        self.config = {"version": 1, "postprocess_seed": 20260918,
            "releases": {str(e): {"copula_fraction": "1", "bins": 8, "sample": {},
                                    "cox_calibration": {"strata": "none"}} for e in [1, 3, 10]}}
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.schema = {"submitted_c_columns": ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"],
            "ranges": {"BMI": [10, 55], "SBP": [40, 250], "TG": [1, 3000], "HDL": [10, 160], "ALT": [.1, 700], "FPG": [40, 400]},
            "age_range": [40, 74], "rare_detector": {"columns": release.dp_copula.CONT,
                "mu": [55, 25, 125, 100, 50, 20, 95], "sigma": np.eye(7).tolist(), "band": {"lo": 4}}}
        (self.data / "schema.json").write_text(json.dumps(self.schema), encoding="utf-8")
        self.frame = pd.DataFrame({"age": [55]*1049, "sex": [0]*1049, "prefecture": ["Public A"]*1049,
            "BMI": [25.0]*1049, "SBP": [125.0]*1049, "TG": [100.0]*1049, "HDL": [50.0]*1049,
            "ALT": [20.0]*1049, "smoking": [0]*1049, "FPG": [95.0]*1049,
            "time": [9.99]*1049, "onset": [0]*1049, "death": [0]*1049})
        self.frame.to_csv(self.data / "A_bg.csv", index=False)
        self.frame.to_csv(self.data / "B.csv", index=False)

    def run_script(self, selected=(1, 3, 10)):
        return release.run(self.config_path, self.data, self.output,
                           selected=selected, ledger_dir=self.ledger)

    def fake_model(self, *args, **kwargs):
        return {"categories": ["Public A"], "public_defaults": {c: float(self.frame[c].iloc[0]) for c in release.dp_copula.NUMERIC},
                "schema": self.schema, "test_only": True}

    def fake_generate(self, model, item):
        return self.frame.copy(), {"selection_input": "fabricated_fixture_only"}

    def test_exact_allocation_before_execution(self):
        self.config["releases"]["3"]["copula_fraction"] = "4/5"
        plans = release.validate_config(self.config)
        self.assertEqual(plans[3]["copula_epsilon_exact"], "12/5")
        self.assertEqual(plans[3]["cox_epsilon_exact"], "3/5")
        self.assertEqual(sum(Fraction(p["epsilon_exact"]) for p in plans.values()), 14)
        self.config["releases"]["3"]["copula_fraction"] = .8
        with self.assertRaises(ValueError):
            release.validate_config(self.config)

    def test_invalid_late_config_fails_before_any_reservation(self):
        self.config["releases"]["10"]["sample"]["public_radial_strength"] = 2
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_script()
        self.assertFalse(self.ledger.exists())

    def test_three_releases_read_private_once_and_resume_without_fit(self):
        read_csv = pd.read_csv
        reads = []
        def track(path, *args, **kwargs):
            reads.append(Path(path).name)
            return read_csv(path, *args, **kwargs)
        with patch.object(release.dp_copula, "fit", side_effect=self.fake_model) as fit, \
             patch.object(release, "_generate", side_effect=self.fake_generate), \
             patch.object(release.pd, "read_csv", side_effect=track):
            first = self.run_script()
            second = self.run_script()
        self.assertEqual(reads.count("B.csv"), 1)
        self.assertEqual(fit.call_count, 3)
        self.assertEqual([r["status"] for r in first], ["complete"]*3)
        self.assertEqual([r["status"] for r in second], ["already_complete"]*3)
        for eps in [1, 3, 10]:
            self.assertTrue((self.ledger/f"epsilon_{eps}.reservation.json").exists())
            manifest = json.loads((self.output/f"eps{eps}/manifest.json").read_text())
            self.assertFalse(manifest["private_input_hash_included"])
            self.assertFalse(manifest["private_evaluation_used_for_selection"])

    def test_failure_is_reserved_and_never_retried(self):
        with patch.object(release.dp_copula, "fit", side_effect=RuntimeError("PRIVATE_SENTINEL_DO_NOT_LOG")):
            with self.assertRaises(RuntimeError):
                self.run_script(selected=(1,))
        self.assertTrue((self.ledger/"epsilon_1.reservation.json").exists())
        self.assertNotIn("PRIVATE_SENTINEL", (self.ledger/"events.jsonl").read_text())
        with patch.object(release.dp_copula, "fit") as fit:
            with self.assertRaises(release.ReleaseBlocked):
                self.run_script(selected=(1,))
            fit.assert_not_called()

    def test_resume_after_model_save_is_postprocess_only(self):
        with patch.object(release.dp_copula, "fit", side_effect=self.fake_model), \
             patch.object(release, "_generate", side_effect=RuntimeError("fake postprocess interruption")):
            with self.assertRaises(RuntimeError):
                self.run_script(selected=(1,))
        self.assertTrue((self.output/"eps1/model.json").exists())
        with patch.object(release.dp_copula, "fit") as fit, \
             patch.object(release, "_generate", side_effect=self.fake_generate):
            self.run_script(selected=(1,))
            fit.assert_not_called()
        self.assertTrue((self.output/"eps1/C.csv").exists())

    def test_config_drift_blocks_remaining_budget(self):
        with patch.object(release.dp_copula, "fit", side_effect=self.fake_model), \
             patch.object(release, "_generate", side_effect=self.fake_generate):
            self.run_script(selected=(1,))
        self.config["releases"]["3"]["bins"] = 7
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        with patch.object(release.dp_copula, "fit") as fit:
            with self.assertRaises(release.ReleaseBlocked):
                self.run_script(selected=(3,))
            fit.assert_not_called()

    def test_private_file_never_hashed(self):
        hashed = []
        sha = release._sha
        def track(path):
            hashed.append(Path(path).name)
            return sha(path)
        with patch.object(release.dp_copula, "fit", side_effect=self.fake_model), \
             patch.object(release, "_generate", side_effect=self.fake_generate), \
             patch.object(release, "_sha", side_effect=track):
            self.run_script(selected=(1,))
        self.assertNotIn("B.csv", hashed)
        self.assertNotIn("B_self.csv", hashed)
        self.assertNotIn("utility_ref.json", hashed)
        self.assertIn("A_bg.csv", hashed)

    def test_active_lock_blocks_another_run(self):
        self.ledger.mkdir()
        (self.ledger/"active.lock").write_text("existing process or crash")
        with patch.object(release.dp_copula, "fit") as fit:
            with self.assertRaises(release.ReleaseBlocked):
                self.run_script(selected=(1,))
            fit.assert_not_called()

    def test_missing_reservation_cannot_erase_append_history(self):
        with patch.object(release.dp_copula, "fit", side_effect=RuntimeError("fake failure")):
            with self.assertRaises(RuntimeError):
                self.run_script(selected=(1,))
        (self.ledger/"epsilon_1.reservation.json").unlink()
        with self.assertRaises(release.ReleaseBlocked):
            self.run_script(selected=(1,))

    def test_cox_uses_same_public_clipping_and_exact_fraction(self):
        self.config["releases"]["1"]["copula_fraction"] = "4/5"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        changed = self.frame.copy()
        changed.loc[0, "FPG"] = np.inf
        changed.loc[1, "BMI"] = 100000
        changed.to_csv(self.data/"B.csv", index=False)
        def measure(frame, model, **kwargs):
            self.assertEqual(frame.loc[0, "FPG"], 400)
            self.assertEqual(frame.loc[1, "BMI"], 55)
            self.assertEqual(kwargs["epsilon"], Fraction(1, 5))
            return {"noisy_score_integer": [0]*10}
        with patch.object(release.dp_copula, "fit", side_effect=self.fake_model), \
             patch.object(release.dp_cox, "build_public_cox_model", return_value={"public_only": True}), \
             patch.object(release.dp_cox, "measure_cox_update", side_effect=measure) as fit_cox, \
             patch.object(release, "_generate", side_effect=self.fake_generate):
            self.run_script(selected=(1,))
        fit_cox.assert_called_once()


if __name__ == "__main__":
    unittest.main()
