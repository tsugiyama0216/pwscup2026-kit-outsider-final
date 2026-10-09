import copy
import unittest

from rank_validation.__main__ import official_check
from rank_validation.core import compare_paired
from rank_validation.sensitivity import mock_sensitivity, DEFENSE_FAMILIES


class ReportTests(unittest.TestCase):
    def test_official_missing_total_remains_unknown(self):
        comparison = {"model_a": "Rv1", "model_b": "pl1", "experiment_id": "test", "metrics": {f"score_{i}": {"mean_delta_b_minus_a": .03} for i in range(1, 6)}}
        official = {"Rv1": {"score_1": .169, "score_2": .0472}, "pl1": {"score_1": None, "score_2": .0399}}
        rows = official_check(comparison, official)
        self.assertEqual(rows[0]["status"], "missing_official")
        self.assertEqual(rows[1]["status"], "opposite_direction")

    def test_family_and_zero_rare_sensitivity(self):
        a, b = [], []
        for name, family in DEFENSE_FAMILIES.items():
            for n in (0, 1, 2):
                row = {"case_key": str(n), "release_id": name, "experiment_id": "test", "evaluation_sha256": str(n), "scorer_id": "fixed", "metadata": {"experiment": {"method_family": "mock_TS"}, "mia_rare_positive_count": n}, **{f"score_{i}": .5 for i in range(1, 6)}}
                other = copy.deepcopy(row)
                other["score_1"] += .1 if family == "R" else -.2
                a.append(row)
                b.append(other)
        result = compare_paired(a, b, model_a="a", model_b="b")
        scenario = mock_sensitivity(result, a)
        self.assertGreater(result["metrics"]["score_1"]["mean_delta_b_minus_a"], 0)
        self.assertLess(scenario["equal_family_mean_delta"]["score_1"], 0)
        self.assertFalse(scenario["rare_positive_strata"]["0"]["mia_rare_defined"])
        self.assertFalse(scenario["weights_are_official_population_estimates"])
        result["paired_rows"] = result["paired_rows"][:-3]
        self.assertFalse(mock_sensitivity(result, a)["available"])


if __name__ == "__main__":
    unittest.main()
