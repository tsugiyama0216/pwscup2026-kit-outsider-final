"""Acceptance checks against the actual released preliminary-round answers."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mock_TS_using_prelim import scoring


class PreliminaryScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (scoring.DATA_ROOT / "reference" / "attack_submissions" /
                        "extracted" / "cohort24")
        cls.mia = pd.read_csv(cls.original / "F_mia.csv")
        cls.aia = pd.read_csv(cls.original / "F_aia.csv")
        cls.baseline = scoring.score_predictions(cls.original, historical=True)

    def score_frames(self, mia, aia, *, historical=False):
        with tempfile.TemporaryDirectory(prefix="prelim-scoring-test-") as folder:
            root = Path(folder)
            mia.to_csv(root / "F_mia.csv", index=False)
            aia.to_csv(root / "F_aia.csv", index=False)
            return scoring.score_predictions(root, historical=historical)

    def test_actual_team24_reproduces_official_result(self):
        summary = self.baseline["summary"]
        for key, expected in (("mia_power", .15792307692307692),
                              ("aia_power", .18274506769014637),
                              ("attack_power", .3406681446132233)):
            self.assertAlmostEqual(summary[key], expected, places=12)
        self.assertEqual(summary["mia_target_count"], 20)
        self.assertEqual(summary["aia_target_count"], 20)
        self.assertFalse(self.baseline["aia_floor_applied"])
        self.assertEqual(summary["score_1"], summary["attack_power"])
        json.dumps(self.baseline, allow_nan=False)

    def test_permutations_preserve_scores_and_per_target_alignment(self):
        mia = self.mia.sample(frac=1, random_state=47)
        mia = mia[["record_id", *list(reversed(mia.columns[1:]))]]
        result = self.score_frames(mia, self.aia.sample(frac=1, random_state=82))
        self.assertEqual(result["summary"], self.baseline["summary"])
        self.assertEqual(result["by_target"], self.baseline["by_target"])

    def test_self_attack_is_excluded_and_reported(self):
        mia = self.mia.copy()
        mia["24"] = 1.0
        own_truth = pd.read_csv(scoring.DATA_ROOT / "private" / "ground_truth" / "aia_truth_24.csv")
        own = pd.DataFrame({"target_k": 24, "challenge_row_id": own_truth["challenge_row_id"], "time_hat": 0.0})
        result = self.score_frames(mia, pd.concat([self.aia, own], ignore_index=True))
        self.assertEqual(result["summary"], self.baseline["summary"])
        self.assertEqual(result["excluded_self"], {"mia_column": True, "aia_rows": 326})

    def test_tied_confidence_is_chance_and_order_invariant(self):
        _, attack_score, cfg = scoring._load_official()
        confidence = np.full(100, .5)
        truth = np.array([1] * 10 + [0] * 90)
        for perm in (np.arange(100), np.arange(99, -1, -1)):
            result = attack_score.score_mia(confidence[perm], truth[perm], cfg)
            self.assertAlmostEqual(result.score, .01, places=15)
            self.assertAlmostEqual(result.breakdown["auc"], .5, places=15)

    def test_invalid_ids_values_and_incomplete_targets_are_rejected(self):
        variants = []
        mia = self.mia.copy()
        mia.loc[0, "record_id"] = -123456
        variants.append(("unknown_mia_id", mia, self.aia))
        mia = self.mia.copy()
        mia.loc[0, "record_id"] = mia.loc[1, "record_id"]
        variants.append(("duplicate_mia_id", mia, self.aia))
        mia = self.mia.astype({"record_id": float})
        mia.loc[0, "record_id"] += .5
        variants.append(("fractional_mia_id", mia, self.aia))
        mia = self.mia.copy()
        mia.loc[0, mia.columns[1]] = math.inf
        variants.append(("infinite_confidence", mia, self.aia))
        mia = self.mia.copy()
        mia.loc[0, mia.columns[1]] = 1.1
        variants.append(("out_of_range_confidence", mia, self.aia))
        variants.append(("missing_mia_target", self.mia.drop(columns=[self.mia.columns[1]]), self.aia))
        variants.append(("missing_aia_row", self.mia, self.aia.iloc[1:]))
        aia = self.aia.copy()
        aia.loc[0, "challenge_row_id"] = -123456
        variants.append(("unknown_aia_id", self.mia, aia))
        aia = pd.concat([self.aia, self.aia.iloc[[0]]], ignore_index=True)
        variants.append(("duplicate_aia_id", self.mia, aia))
        aia = self.aia.astype({"target_k": float})
        aia.loc[0, "target_k"] += .25
        variants.append(("fractional_target", self.mia, aia))
        aia = self.aia.copy()
        aia.loc[0, "time_hat"] = math.inf
        variants.append(("infinite_time", self.mia, aia))
        aia = self.aia.copy()
        aia.loc[0, "time_hat"] = -.1
        variants.append(("negative_time", self.mia, aia))
        for name, mia, aia in variants:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.score_frames(mia, aia)

    def test_historical_missing_pair_matches_published_cohort04(self):
        root = scoring.DATA_ROOT / "reference" / "attack_submissions" / "extracted" / "cohort04"
        result = scoring.score_predictions(root, attacker_id=4, historical=True)
        self.assertEqual(result["summary"]["aia_target_count"], 19)
        self.assertEqual(result["missing_aia_targets"], [23])
        self.assertAlmostEqual(result["summary"]["attack_power"], .1822037946511402, places=12)
        missing = next(row for row in result["by_target"] if row["target_k"] == 23)
        self.assertIsNone(missing["aia_score"])
        with self.assertRaisesRegex(ValueError, "target set mismatch"):
            scoring.score_predictions(root, attacker_id=4)

    def test_historical_mode_rejects_modified_predictions(self):
        mia = self.mia.copy()
        mia.loc[0, mia.columns[1]] = .321
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            self.score_frames(mia, self.aia, historical=True)


if __name__ == "__main__":
    unittest.main()
