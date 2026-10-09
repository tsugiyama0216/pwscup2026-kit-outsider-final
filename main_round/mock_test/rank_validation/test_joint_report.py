import unittest
import pandas as pd

from .joint_report import compare_scores, validate_target_frame, TARGET_IDS


class RankEvidenceTests(unittest.TestCase):
    def test_all_pair_order(self):
        official = {"A": (.195, .205), "B": (.145, .155), "C": (.095, .105)}
        report = compare_scores({"A": .4, "B": .3, "C": .1}, official)
        self.assertTrue(report["all_correct"])
        self.assertEqual(report["correct"], 3)

    def test_tie_is_not_a_correct_strict_rank(self):
        report = compare_scores({"A": .3, "B": .3}, {"A": (.195, .205), "B": (.145, .155)})
        self.assertFalse(report["all_correct"])
        self.assertEqual(report["pairs"][0]["local_direction"], 0)

    def test_overlapping_rounded_scores_are_not_labelled(self):
        report = compare_scores({"A": .3, "B": .2}, {"A": (.195, .205), "B": (.2, .21)})
        self.assertIsNone(report["pairs"][0]["agrees"])
        self.assertFalse(report["all_correct"])

    def test_mismatched_model_sets_fail(self):
        with self.assertRaises(ValueError):
            compare_scores({"A": .3}, {"A": (.195, .205), "B": (.2, .21)})

    def test_nonfinite_fails(self):
        with self.assertRaises(ValueError):
            compare_scores({"A": float("nan"), "B": .2}, {"A": (.195, .205), "B": (.1, .11)})

    def frame(self):
        return pd.DataFrame({"attack_power": [.3] * 22, "mia_power": [.2] * 22,
                             "aia_power": [.1] * 22}, index=TARGET_IDS)

    def test_twenty_two_rows_with_self_substitution_are_rejected(self):
        frame = self.frame()
        frame.index = [10, *TARGET_IDS[1:]]
        with self.assertRaises(ValueError):
            validate_target_frame(frame, .3)

    def test_missing_component_cannot_be_silently_dropped(self):
        frame = self.frame()
        frame.loc[TARGET_IDS[0], "aia_power"] = float("nan")
        with self.assertRaises(ValueError):
            validate_target_frame(frame, .3)

    def test_matching_mean_does_not_hide_wrong_component_sum(self):
        frame = self.frame()
        frame.loc[TARGET_IDS[0], "aia_power"] = .2
        frame.loc[TARGET_IDS[1], "aia_power"] = 0
        with self.assertRaises(ValueError):
            validate_target_frame(frame, .3)

    def test_complete_valid_official_components_pass(self):
        validate_target_frame(self.frame(), .3)


if __name__ == "__main__":
    unittest.main()
