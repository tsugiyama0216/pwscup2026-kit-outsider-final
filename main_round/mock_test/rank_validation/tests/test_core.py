"""Meaningful rejection and rank-sensitivity examples for paired evaluation."""
import copy
import math
import unittest

from rank_validation.core import compare_paired


def row(case, release, score, model, **changes):
    result = {"case_key": case, "release_id": release, "model_id": model,
              "experiment_id": "matched_experiment", "evaluation_sha256": case + "_truth_cfg",
              "scorer_id": "pinned_official", "input_attribution": "archived_fixture_mapping_public_inputs_not_in_bundle",
              **{f"score_{i}": score if i == 1 else score / 2 for i in range(1, 6)}}
    result.update(changes)
    return result


class PairedComparisonTests(unittest.TestCase):
    def compare(self, a, b, **kwargs):
        return compare_paired(a, b, model_a="A", model_b="B", **kwargs)

    def test_matched_row_order_and_all_five_metrics(self):
        a = [row("c1", "r1", .2, "A"), row("c2", "r1", .4, "A")]
        b = [row("c2", "r1", .5, "B"), row("c1", "r1", .6, "B")]
        before = copy.deepcopy((a, b))
        result = self.compare(a, b)
        self.assertAlmostEqual(result["metrics"]["score_1"]["mean_delta_b_minus_a"], .25)
        self.assertAlmostEqual(result["metrics"]["score_5"]["mean_delta_b_minus_a"], .125)
        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["release_count"], 1)
        self.assertEqual(result["direction"], "b_higher")
        self.assertEqual(result["cell_counts"], {"a_higher": 0, "b_higher": 2, "tie": 0})
        self.assertTrue(result["complete_case_release_grid"])
        self.assertEqual((a, b), before)
        self.assertFalse(result["statistical_significance_established"])
        self.assertFalse(result["official_ranking_validated"])
        self.assertIn("archived_fixture_mapping_public_inputs_not_in_bundle", result["input_attributions"])

    def test_single_case_cannot_be_called_leave_one_case_out_validation(self):
        result = self.compare([row("c1", "r1", .4, "A")], [row("c1", "r1", .5, "B")])
        self.assertFalse(result["leave_one_case_out"]["available"])
        self.assertIsNone(result["leave_one_case_out"]["rank_reversal"])
        self.assertFalse(result["leave_one_release_out"]["available"])

    def test_winner_reverses_when_one_case_is_removed(self):
        a = [row("c1", "r1", .3, "A"), row("c2", "r1", .3, "A")]
        b = [row("c1", "r1", .7, "B"), row("c2", "r1", .1, "B")]
        result = self.compare(a, b)
        self.assertEqual(result["direction"], "b_higher")
        self.assertTrue(result["leave_one_case_out"]["rank_reversal"])
        self.assertAlmostEqual(result["case_mean_delta_range"][0], -.2)
        self.assertAlmostEqual(result["case_mean_delta_range"][1], .4)

    def test_winner_reverses_when_one_defense_is_removed(self):
        a = [row("c1", "r1", .3, "A"), row("c1", "r2", .3, "A")]
        b = [row("c1", "r1", .7, "B"), row("c1", "r2", .1, "B")]
        result = self.compare(a, b)
        self.assertTrue(result["leave_one_release_out"]["rank_reversal"])
        self.assertFalse(result["leave_one_case_out"]["available"])

    def test_unbalanced_grid_discloses_weights(self):
        a = [row("c1", "r1", .4, "A"), row("c1", "r2", .4, "A"), row("c2", "r1", .4, "A")]
        b = [row("c1", "r1", .8, "B"), row("c1", "r2", .8, "B"), row("c2", "r1", 0., "B")]
        result = self.compare(a, b)
        self.assertFalse(result["complete_case_release_grid"])
        self.assertGreater(result["metrics"]["score_1"]["mean_delta_b_minus_a"], 0)
        self.assertAlmostEqual(result["equal_case_mean_delta"], 0)
        self.assertTrue(any("Incomplete" in s for s in result["limitations"]))

    def test_input_missing_on_one_side_is_not_silently_intersected(self):
        a = [row("c1", "r1", .4, "A"), row("c2", "r1", .4, "A")]
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            self.compare(a, [row("c1", "r1", .5, "B")])

    def test_same_truth_is_not_enough_when_defense_differs(self):
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            self.compare([row("c1", "r1", .4, "A")], [row("c1", "r2", .5, "B")])

    def test_duplicate_pair_rejected(self):
        x = row("c1", "r1", .4, "A")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.compare([x, dict(x)], [row("c1", "r1", .5, "B")])

    def test_fingerprint_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Condition mismatch"):
            self.compare([row("c1", "r1", .4, "A")],
                         [row("c1", "r1", .5, "B", evaluation_sha256="different")])

    def test_experiments_and_scorers_cannot_be_mixed_even_if_each_pair_matches(self):
        for field in ("experiment_id", "scorer_id"):
            with self.subTest(field=field):
                a = [row("c1", "r1", .4, "A"), row("c2", "r1", .4, "A", **{field: "other"})]
                b = [row("c1", "r1", .5, "B"), row("c2", "r1", .5, "B", **{field: "other"})]
                with self.assertRaisesRegex(ValueError, "Cannot mix"):
                    self.compare(a, b)

    def test_missing_component_and_nonfinite_scores_rejected(self):
        for invalid in (None, math.nan, math.inf, -math.inf, True, "0.3"):
            with self.subTest(invalid=invalid):
                a = row("c1", "r1", .4, "A", score_5=invalid)
                with self.assertRaisesRegex(ValueError, "finite number"):
                    self.compare([a], [row("c1", "r1", .5, "B")])
        a = row("c1", "r1", .4, "A")
        del a["score_4"]
        with self.assertRaisesRegex(ValueError, "finite number"):
            self.compare([a], [row("c1", "r1", .5, "B")])

    def test_scores_outside_official_ranges_are_rejected(self):
        for field, value in (("score_1", 2.1), ("score_2", 1.1), ("score_5", -0.1), ("score_1", 1e308)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, "official range"):
                    self.compare([row("c1", "r1", .4, "A", **{field: value})], [row("c1", "r1", .5, "B")])
    def test_missing_provenance_and_wrong_model_rejected(self):
        for changes in ({"evaluation_sha256": ""}, {"case_key": None}, {"model_id": "C"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.compare([row("c1", "r1", .4, "A", **changes)], [row("c1", "r1", .5, "B")])

    def test_ties_are_not_wins_and_other_metric_can_be_selected(self):
        a = [row("c1", "r1", .4, "A")]
        b = [row("c1", "r1", .4 + 1e-14, "B", score_3=.6)]
        self.assertEqual(self.compare(a, b)["direction"], "tie")
        self.assertEqual(self.compare(a, b, score_key="score_3")["direction"], "b_higher")

    def test_invalid_options_and_empty_inputs(self):
        a, b = [row("c1", "r1", .4, "A")], [row("c1", "r1", .5, "B")]
        for options in ({"score_key": "auc"}, {"tie_tolerance": -1}, {"tie_tolerance": math.nan},
                        {"condition_fields": ()}, {"condition_fields": ("evaluation_sha256",)}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    self.compare(a, b, **options)
        with self.assertRaises(ValueError):
            self.compare([], b)


if __name__ == "__main__":
    unittest.main()