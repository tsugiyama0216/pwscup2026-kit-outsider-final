from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v5.aia_structured_ensemble import (
    build_aia_v5_variants,
    filter_linkable_releases,
    stratified_half_selection,
)


class StructuredAiaTests(unittest.TestCase):
    def test_even_strata_select_exactly_half(self) -> None:
        challenge = pd.DataFrame(
            {
                "challenge_row_id": np.arange(8),
                "age": [41, 42, 43, 44, 51, 52, 53, 54],
                "sex": [0] * 8,
                "prefecture": ["A"] * 8,
            }
        )
        selected, detail = stratified_half_selection(
            challenge, np.arange(8, dtype=float), member_count=4
        )
        self.assertEqual(int(selected[:4].sum()), 2)
        self.assertEqual(int(selected[4:].sum()), 2)
        self.assertEqual(detail.odd_strata, 0)

    def test_odd_strata_assign_ceil_by_marginal_score(self) -> None:
        challenge = pd.DataFrame(
            {
                "challenge_row_id": np.arange(6),
                "age": [41, 42, 43, 51, 52, 53],
                "sex": [0] * 6,
                "prefecture": ["A"] * 6,
            }
        )
        score = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1])
        selected, detail = stratified_half_selection(
            challenge, score, member_count=3
        )
        self.assertEqual(int(selected[:3].sum()), 2)
        self.assertEqual(int(selected[3:].sum()), 1)
        self.assertEqual(detail.extra_ceil_strata, 1)

    def test_builder_is_aligned_and_keeps_official_shape(self) -> None:
        challenge = pd.DataFrame(
            {
                "challenge_row_id": np.arange(4),
                "age": [41, 42, 43, 44],
                "SBP": [110.0, 111.0, 130.0, 131.0],
                "TG": [80.0, 82.0, 150.0, 152.0],
                "HDL": [60.0, 61.0, 40.0, 41.0],
                "ALT": [20.0, 21.0, 40.0, 41.0],
                "sex": [0, 0, 0, 0],
                "prefecture": ["A"] * 4,
            }
        )
        release = challenge.drop(columns="challenge_row_id").copy()
        release["BMI"] = [20.0, 21.0, 22.0, 23.0]
        release["smoking"] = [0, 0, 1, 1]
        release["FPG"] = [90.0, 91.0, 100.0, 101.0]
        release["time"] = [10.0, 10.0, 5.0, 10.0]
        release["onset"] = [0, 0, 1, 0]
        release["death"] = [0, 0, 0, 0]
        expert = pd.DataFrame(
            {
                "target_k": [1] * 4,
                "challenge_row_id": np.arange(4),
                "time_hat": [10.0, 10.0, 11.0, 11.0],
            }
        )
        variants, diagnostics = build_aia_v5_variants(
            expert,
            expert,
            {"1": release},
            {"1": challenge},
            member_count=2,
        )
        self.assertEqual(len(variants), 28)
        for frame in variants.values():
            self.assertEqual(frame.shape, (4, 3))
            self.assertEqual(int((frame["time_hat"] != 11.0).sum()), 2)
        self.assertEqual(diagnostics["targets"]["1"]["active_overlap"], 2)

    def test_cross_target_exact_rows_are_removed_with_guard(self) -> None:
        base = pd.DataFrame(
            {
                "age": np.arange(40, 70),
                "sex": [0] * 30,
                "prefecture": ["A"] * 30,
                "BMI": np.arange(30, dtype=float),
                "SBP": np.arange(30, dtype=float) + 100,
                "TG": np.arange(30, dtype=float) + 50,
                "HDL": np.arange(30, dtype=float) + 30,
                "ALT": np.arange(30, dtype=float) + 10,
                "smoking": [0] * 30,
                "FPG": np.arange(30, dtype=float) + 80,
                "time": [10.0] * 30,
                "onset": [0] * 30,
                "death": [0] * 30,
            }
        )
        other = base.copy()
        other.loc[1:, "BMI"] += 100.0
        filtered, detail = filter_linkable_releases({"1": base, "2": other})
        self.assertEqual(detail["1"]["cross_target_exact_rows"], 1)
        self.assertTrue(detail["1"]["cross_target_filter_applied"])
        self.assertEqual(len(filtered["1"]), 29)


if __name__ == "__main__":
    unittest.main()
