import unittest

from rank_validation.feedback import add_feedback, validate_data
from rank_validation.__main__ import official_check


class FeedbackTests(unittest.TestCase):
    def data(self):
        return {"records": [], "official": [{"model_id": "Rv2", "score_1": .1966605682146923}], "unresolved": [{"model_id": None, "score_1": None}], "provenance": {}}

    def feedback(self):
        return {"rows": [{"model_id": "Tv105", "score_1_display": .0981, "decimal_places": 4}, {"model_id": "Rv2", "score_1_display": .1967, "decimal_places": 4}]}

    def test_supplement_preserves_exact_and_unidentified(self):
        source = self.data()
        result = add_feedback(source, self.feedback())
        self.assertEqual(len(source["official"]), 1)
        self.assertEqual(result["official"][0]["score_1"], .1966605682146923)
        self.assertEqual(result["official"][1]["score_1"], .0981)
        self.assertIsNone(result["official"][1]["score_2"])
        self.assertIsNone(result["unresolved"][0]["model_id"])

    def test_duplicate_invalid_and_conflicting_scores_rejected(self):
        data = self.data()
        data["official"].append(dict(data["official"][0]))
        with self.assertRaises(ValueError):
            validate_data(data)
        for value in ("0.1", float("nan"), float("inf"), True, -1):
            data = self.data()
            data["official"][0]["score_1"] = value
            with self.assertRaises(ValueError):
                validate_data(data)
        feedback = self.feedback()
        feedback["rows"][1]["score_1_display"] = .2
        with self.assertRaises(ValueError):
            add_feedback(self.data(), feedback)

    def test_overlapping_rounded_scores_do_not_establish_a_rank(self):
        comparison = {"model_a": "a", "model_b": "b", "experiment_id": "test", "metrics": {f"score_{i}": {"mean_delta_b_minus_a": .1} for i in range(1, 6)}}
        official = {"a": {"score_1": .1, "score_intervals": {"score_1": [.09995, .10005]}}, "b": {"score_1": .1, "score_intervals": {"score_1": [.09995, .10005]}}}
        self.assertEqual(official_check(comparison, official)[0]["status"], "display_precision_inconclusive")


if __name__ == "__main__":
    unittest.main()
