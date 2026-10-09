"""Roster and ID regressions: agreeing incomplete CSVs must still fail."""
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from rank_validation.replay import MAIN_TARGETS, expected_targets, read_csv_strict, validate_frames


def fixtures(attacker=None):
    targets = expected_targets(attacker)
    ids = np.arange(1049) + 10000
    pool = pd.DataFrame({"record_id": ids, "pool_id": np.arange(1049), "is_rare": [1] * 24 + [0] * 1025})
    target_pools = {k: pool.iloc[:26].copy() if attacker is None else pool.copy() for k in targets}
    query = np.arange(326) + 2000
    truths = {k: pd.DataFrame({"challenge_row_id": query, "label": ["member"] * 163 + ["control"] * 163,
                              "onset": 0, "time": 10., "is_rare": ([1] * 13 + [0] * 150) * 2}) for k in targets}
    mia = pd.DataFrame({"record_id": ids, **{str(k): np.full(1049, .5) for k in sorted(targets)}})
    aia = pd.concat([pd.DataFrame({"target_k": k, "challenge_row_id": query, "time_hat": 10.}) for k in sorted(targets)], ignore_index=True)
    floor = pd.DataFrame({"team_id": sorted(targets), "aia_floor": .03})
    return mia, aia, pool, target_pools, truths, floor, attacker


class ReplayValidationTests(unittest.TestCase):
    def test_local_complete_ids_allow_permutations_and_nonconsecutive_ids(self):
        args = list(fixtures())
        args[0] = args[0].iloc[::-1].copy()
        args[1] = args[1].iloc[::-1].copy()
        result = validate_frames(*args)
        self.assertEqual(result["candidate_rows"], 1049)
        self.assertEqual(result["challenge_rows"], 326)
        self.assertEqual(result["target_count"], 1)

    def test_joint_complete_22_target_roster(self):
        result = validate_frames(*fixtures(10))
        self.assertEqual(result["target_count"], 22)
        self.assertEqual(result["challenge_rows"], 7172)
        self.assertNotIn(10, MAIN_TARGETS)

    def test_matching_missing_target_in_both_predictions_rejected(self):
        args = list(fixtures(10))
        args[0] = args[0].drop(columns="28")
        args[1] = args[1][args[1].target_k != 28]
        with self.assertRaisesRegex(ValueError, "F_mia"):
            validate_frames(*args)

    def test_missing_target_in_truth_and_predictions_is_still_rejected(self):
        args = list(fixtures(10))
        args[0] = args[0].drop(columns="28")
        args[1] = args[1][args[1].target_k != 28]
        del args[3][28]
        del args[4][28]
        args[5] = args[5][args[5].team_id != 28]
        with self.assertRaisesRegex(ValueError, "Truth target roster"):
            validate_frames(*args)

    def test_extra_self_target_is_rejected(self):
        args = list(fixtures(10))
        args[0]["10"] = .5
        with self.assertRaisesRegex(ValueError, "F_mia"):
            validate_frames(*args)

    def test_target_column_alias_is_not_another_valid_id(self):
        args = list(fixtures())
        args[0] = args[0].rename(columns={"0": "00"})
        with self.assertRaisesRegex(ValueError, "F_mia"):
            validate_frames(*args)

    def test_fractional_nonfinite_or_inexact_ids_rejected(self):
        for value in (10000.5, np.nan, np.inf, 2**53):
            for index, column in ((0, "record_id"), (1, "challenge_row_id"), (1, "target_k"), (2, "pool_id")):
                with self.subTest(value=value, table=index, column=column):
                    args = list(fixtures())
                    args[index][column] = args[index][column].astype(float)
                    args[index].loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, "finite exact integers"):
                        validate_frames(*args)

    def test_duplicate_mia_and_aia_ids_rejected(self):
        for index, column in ((0, "record_id"), (1, "challenge_row_id")):
            with self.subTest(index=index):
                args = list(fixtures())
                args[index].loc[1, column] = args[index].loc[0, column]
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    validate_frames(*args)

    def test_truth_ids_are_checked_even_when_bad_values_match_predictions(self):
        args = list(fixtures())
        args[2].loc[1, "record_id"] = args[2].loc[0, "record_id"]
        args[0].loc[1, "record_id"] = args[0].loc[0, "record_id"]
        with self.assertRaisesRegex(ValueError, "attacker pool.*duplicate"):
            validate_frames(*args)
        args = list(fixtures())
        args[4][0].loc[1, "challenge_row_id"] = args[4][0].loc[0, "challenge_row_id"]
        with self.assertRaisesRegex(ValueError, "truth.*duplicate"):
            validate_frames(*args)

    def test_extra_or_missing_challenge_ids_rejected(self):
        for change in ("remove", "replace"):
            args = list(fixtures())
            if change == "remove":
                args[1] = args[1].iloc[:-1]
            else:
                args[1].loc[0, "challenge_row_id"] = 99999
            with self.assertRaisesRegex(ValueError, "challenge IDs differ"):
                validate_frames(*args)

    def test_truth_label_and_numeric_fields_rejected(self):
        for column, value in (("label", "wrong"), ("is_rare", .5), ("onset", .5), ("time", np.nan)):
            with self.subTest(column=column):
                args = list(fixtures())
                args[4][0][column] = args[4][0][column].astype(object)
                args[4][0].loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_frames(*args)

    def test_floor_missing_duplicate_or_wrong_target_rejected(self):
        for change in ("missing", "duplicate", "wrong", "nonfinite"):
            with self.subTest(change=change):
                args = list(fixtures())
                if change == "missing":
                    args[5] = args[5].iloc[:0]
                elif change == "duplicate":
                    args[5] = pd.concat([args[5], args[5]], ignore_index=True)
                elif change == "wrong":
                    args[5].loc[0, "team_id"] = 1
                else:
                    args[5].loc[0, "aia_floor"] = np.nan
                with self.assertRaises(ValueError):
                    validate_frames(*args)

    def test_bad_prediction_values_and_columns_rejected(self):
        for index, column, value in ((0, "0", 1.1), (1, "time_hat", -1), (1, "time_hat", np.inf)):
            args = list(fixtures())
            args[index].loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_frames(*args)
        args = list(fixtures())
        args[1]["extra"] = 0
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_frames(*args)

    def test_unknown_attacker_protocol_rejected(self):
        for value in (24, "10", 10.0, True):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                expected_targets(value)

    def test_duplicate_csv_header_rejected_before_pandas_can_mangle_it(self):
        with patch("pathlib.Path.open", return_value=StringIO("record_id,0,0\n1,.2,.3\n")):
            with self.assertRaisesRegex(ValueError, "duplicate CSV columns"):
                read_csv_strict("unused.csv")


if __name__ == "__main__":
    unittest.main()