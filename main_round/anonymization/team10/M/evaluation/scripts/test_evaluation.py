from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("evaluate_resistance.py")
SPEC = importlib.util.spec_from_file_location("evaluate_resistance", MODULE_PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_constant_confidence_is_chance_at_one_percent_fpr(self) -> None:
        confidence = np.ones(100)
        truth = np.r_[np.ones(20, dtype=int), np.zeros(80, dtype=int)]
        self.assertAlmostEqual(evaluation.ordered_tpr_at_fpr(confidence, truth, 0.01), 0.01)

    def test_perfect_order_reaches_full_tpr(self) -> None:
        confidence = np.r_[np.ones(10), np.zeros(90)]
        truth = np.r_[np.ones(10, dtype=int), np.zeros(90, dtype=int)]
        self.assertAlmostEqual(evaluation.ordered_tpr_at_fpr(confidence, truth, 0.01), 1.0)

    def test_rare_worst_case_is_selected(self) -> None:
        confidence = np.asarray([0.9, 0.8, 0.7, 0.1, 0.0])
        truth = np.asarray([1, 0, 0, 1, 0])
        rare = np.asarray([1, 1, 1, 0, 0], dtype=bool)
        worst, overall, rare_value = evaluation.mia_raw(confidence, truth, rare, 0.01)
        self.assertGreaterEqual(worst, overall)
        self.assertEqual(worst, rare_value)

    def test_exact_overlap_counts_rows(self) -> None:
        left = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        right = pd.DataFrame({"a": [2, 8], "b": [5, 9]})
        self.assertEqual(evaluation.exact_overlap_count(left, right, ["a", "b"]), 1)

    def test_load_utility_accepts_standalone_and_nested_scores(self) -> None:
        score = {"utility": 0.7, "facets": {"general": 0.8}}
        with tempfile.TemporaryDirectory() as directory:
            standalone = Path(directory) / "score.json"
            nested = Path(directory) / "report.json"
            standalone.write_text(json.dumps(score), encoding="utf-8")
            nested.write_text(json.dumps({"score": score}), encoding="utf-8")
            self.assertEqual(evaluation.load_utility(standalone), score)
            self.assertEqual(evaluation.load_utility(nested), score)


if __name__ == "__main__":
    unittest.main()
