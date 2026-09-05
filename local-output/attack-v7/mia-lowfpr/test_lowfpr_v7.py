"""Focused QA for the research-only v7 MIA candidate."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER_SRC = ROOT / "starter" / "src"
if str(STARTER_SRC) not in sys.path:
    sys.path.insert(0, str(STARTER_SRC))

from pwscup2026.scoring.attack_score import _tpr_at_fpr  # noqa: E402


def load_benchmark():
    path = HERE / "benchmark_lowfpr_v7.py"
    spec = importlib.util.spec_from_file_location("benchmark_lowfpr_v7_tested", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bm = load_benchmark()


class ExactMetricTests(unittest.TestCase):
    def test_tie_free_low_fpr_is_positives_before_eleventh_negative(self) -> None:
        # 25 positives + 1024 negatives => 1% FPR lies between the 10th and
        # 11th negatives.  With no score ties interpolation keeps k/25.
        for k in (0, 3, 6, 19, 25):
            truth = np.r_[
                np.ones(k),
                np.zeros(10),
                np.zeros(1),  # the 11th negative fixes the operating prefix
                np.ones(25 - k),
                np.zeros(1013),
            ]
            confidence = np.linspace(1.0, 0.0, len(truth), endpoint=False)
            observed, _ = _tpr_at_fpr(confidence, truth, 0.01)
            self.assertAlmostEqual(observed, k / 25.0, places=15)


class ProductionInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        team = ROOT / "submissions" / "prelim" / "team24"
        cls.v3 = pd.read_csv(team / "attack-candidate-v3" / "F_mia.csv")
        cls.v4 = pd.read_csv(team / "attack-candidate-v4" / "F_mia.csv")
        cls.day = pd.read_csv(team / "attack-candidate-daybreak-independent" / "F_mia.csv")
        diagnostics = json.loads(
            (ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8")
        )
        cls.rare = bm.mask_from_ids(cls.v4, diagnostics["mia"]["eligible_rare_record_ids"])
        cls.private = bm.mask_from_ids(cls.v4, diagnostics["mia"]["private_record_ids"])
        cls.variants = bm.recipes(cls.v4, cls.v3, cls.day, cls.rare, cls.private)
        cls.candidate = cls.variants["daybreak_v4_rarelock"]

    def test_candidate_is_normal_f_mia(self) -> None:
        result = bm.legacy.validate_mia(self.candidate, self.v4)
        self.assertEqual(result["rows"], 1049)
        self.assertTrue(all(value == 0 for value in result["ties"].values()))

    def test_eligible_rare_order_is_exact_v4(self) -> None:
        rare_rows = np.flatnonzero(self.rare)
        for target in self.v4.columns[1:]:
            expected = rare_rows[
                np.argsort(-self.v4[target].to_numpy(float)[rare_rows], kind="mergesort")
            ]
            observed = rare_rows[
                np.argsort(-self.candidate[target].to_numpy(float)[rare_rows], kind="mergesort")
            ]
            np.testing.assert_array_equal(observed, expected)

    def test_private_bottom_set_is_daybreaks(self) -> None:
        private_rows = set(np.flatnonzero(self.private).tolist())
        for target in self.day.columns[1:]:
            order = np.argsort(-self.candidate[target].to_numpy(float), kind="mergesort")
            self.assertEqual(set(order[-len(private_rows) :]), private_rows)

    def test_official_band_structural_bound(self) -> None:
        bound = bm.official_structural_bound(self.day, self.candidate, self.rare)
        self.assertEqual(bound["candidate_changed_slots_that_can_cross_an_official_operating_prefix"], {"19": [12]})
        self.assertEqual(bound["changed_slots_always_inside_band4_prefix"], {"1": [7, 25]})
        self.assertGreater(bound["lower_bound_delta_vs_v4_all"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
