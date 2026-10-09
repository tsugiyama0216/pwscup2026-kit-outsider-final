"""Tests of the fixed target-level stable-index arithmetic and boundaries."""
import copy
import math
import unittest

from rank_validation.ranking_metric import compute_metrics, METRIC_VERSION, TARGET_IDS


def rows():
    return [{"target_k": 0, "mia_all": .1, "mia_power": .7, "aia_power": .4,
             "mia_rare": None, "mia_positive": 1},
            {"target_k": 1, "mia_all": .3, "mia_power": .3, "aia_power": .2,
             "mia_rare": None, "mia_positive": 1000}]


class RankingMetricTests(unittest.TestCase):
    def test_fixed_equal_target_formula_and_separate_rare_lift(self):
        answer = compute_metrics(rows(), expected_target_ids=(0, 1))
        self.assertEqual(answer["stable_index"], .5)
        self.assertAlmostEqual(answer["rare_lift"], .3)
        self.assertAlmostEqual(answer["official_attack_power"], .8)
        self.assertAlmostEqual(answer["stable_index"] + answer["rare_lift"], answer["official_attack_power"])
        self.assertEqual(answer["metric_version"], METRIC_VERSION)
        self.assertFalse(answer["official_scorer_or_scores_changed"])
        # Different positive counts never introduce person weights.
        self.assertEqual(answer["target_count"], 2)

    def test_existing_corrected_aia_is_used_without_another_floor_or_max(self):
        data = rows()
        data[0].update(aia_before_floor=.95, aia_all=.9, aia_rare=.99, aia_floor=.55)
        self.assertEqual(compute_metrics(data, expected_target_ids=(0, 1))["stable_index"], .5)

    def test_inputs_are_unchanged_and_optional_rare_null_is_preserved(self):
        data = rows()
        before = copy.deepcopy(data)
        compute_metrics(data, expected_target_ids=(0, 1))
        self.assertEqual(data, before)
        self.assertIsNone(data[0]["mia_rare"])

    def test_default_requires_all_twenty_two_targets(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compute_metrics(rows())
        data = [{**rows()[0], "target_k": target} for target in TARGET_IDS]
        self.assertEqual(compute_metrics(data)["target_count"], 22)

    def test_leave_one_target_out_requires_explicit_remaining_roster(self):
        data = [{**rows()[0], "target_k": target} for target in TARGET_IDS]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compute_metrics(data[1:])
        answer = compute_metrics(data[1:], expected_target_ids=TARGET_IDS[1:])
        self.assertEqual(answer["target_count"], 21)

    def test_target_order_and_generator_input_do_not_change_result(self):
        expected = compute_metrics(rows(), expected_target_ids=(0, 1))
        actual = compute_metrics((row for row in reversed(rows())), expected_target_ids=(1, 0))
        self.assertEqual(actual, expected)

    def test_required_columns_cannot_be_missing(self):
        for key in ("target_k", "mia_all", "mia_power", "aia_power"):
            data = rows()
            del data[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Missing metric columns"):
                compute_metrics(data, expected_target_ids=(0, 1))

    def test_nonfinite_and_missing_required_numbers_rejected(self):
        for key in ("target_k", "mia_all", "mia_power", "aia_power"):
            for value in (math.nan, math.inf, -math.inf, None, "", True):
                data = rows()
                data[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    compute_metrics(data, expected_target_ids=(0, 1))

    def test_component_range_cannot_be_invalid(self):
        for key in ("mia_all", "mia_power", "aia_power"):
            for value in (-.01, 1.01):
                data = rows()
                data[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    compute_metrics(data, expected_target_ids=(0, 1))

    def test_negative_rare_lift_rejected_per_target_even_if_mean_positive(self):
        data = rows()
        data[1]["mia_power"] = .1
        with self.assertRaisesRegex(ValueError, "negative rare lift"):
            compute_metrics(data, expected_target_ids=(0, 1))

    def test_roundoff_tolerance_preserves_formula_and_reports_count(self):
        data = [{"target_k": 0, "mia_all": 5e-13, "mia_power": 0., "aia_power": .5}]
        answer = compute_metrics(data, expected_target_ids=(0,))
        self.assertEqual(answer["rare_lift"], -5e-13)
        self.assertEqual(answer["tolerated_negative_lift_count"], 1)
        data[0]["mia_all"] = 2e-12
        with self.assertRaisesRegex(ValueError, "negative rare lift"):
            compute_metrics(data, expected_target_ids=(0,))

    def test_duplicate_unexpected_and_malformed_ids_rejected(self):
        for target in (0, 2, -1, .5, 2**53):
            data = rows()
            data[1]["target_k"] = target
            with self.subTest(target=target), self.assertRaises(ValueError):
                compute_metrics(data, expected_target_ids=(0, 1))

    def test_empty_malformed_rows_and_expected_rosters_rejected(self):
        for data in ([], None, {}, "bad", [3]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                compute_metrics(data, expected_target_ids=(0, 1))
        for expected in ([], None, (0, 0), (0, True)):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                compute_metrics(rows(), expected_target_ids=expected)

    def test_csv_string_values_are_accepted_without_silent_missing_values(self):
        data = [{"target_k": "0", "mia_all": "0.1", "mia_power": "0.2", "aia_power": "0.3"}]
        answer = compute_metrics(data, expected_target_ids=(0,))
        self.assertAlmostEqual(answer["stable_index"], .4)
        self.assertAlmostEqual(answer["rare_lift"], .1)


if __name__ == "__main__":
    unittest.main()