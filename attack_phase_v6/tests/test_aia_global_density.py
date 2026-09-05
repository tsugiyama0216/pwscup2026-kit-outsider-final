from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v6.aia_global_density import (
    build_global_density_horizon,
    build_sex_consensus_horizon,
    global_density_ratio_score,
    sex_copula_proximity_score,
)


def _reference(offset: float) -> pd.DataFrame:
    values = np.arange(8, dtype=float) + offset
    return pd.DataFrame(
        {
            "age": np.full(8, 42.0),
            "sex": np.zeros(8, dtype=int),
            "prefecture": ["A"] * 8,
            "BMI": 20.0 + values,
            "SBP": 100.0 + values,
            "TG": 50.0 + values,
            "HDL": 40.0 + values,
            "ALT": 10.0 + values,
            "smoking": np.zeros(8, dtype=int),
            "FPG": 80.0 + values,
            "time": np.full(8, 10.0),
            "onset": np.zeros(8, dtype=int),
            "death": np.zeros(8, dtype=int),
        }
    )


class GlobalDensityAiaTests(unittest.TestCase):
    def test_density_score_prefers_release_like_query(self) -> None:
        release = _reference(0.0)
        background = _reference(100.0)
        challenge = pd.DataFrame(
            {
                "challenge_row_id": [0, 1],
                "age": [42.0, 42.0],
                "sex": [0, 0],
                "prefecture": ["A", "A"],
                "SBP": [103.0, 203.0],
                "TG": [53.0, 153.0],
                "HDL": [43.0, 143.0],
                "ALT": [13.0, 113.0],
            }
        )

        score = global_density_ratio_score(release, challenge, background)

        self.assertGreater(score[0], score[1])

    def test_builder_emits_exact_requested_member_count(self) -> None:
        release = _reference(0.0)
        background = _reference(100.0)
        challenge = pd.DataFrame(
            {
                "challenge_row_id": [0, 1],
                "age": [42.0, 42.0],
                "sex": [0, 0],
                "prefecture": ["A", "A"],
                "SBP": [103.0, 203.0],
                "TG": [53.0, 153.0],
                "HDL": [43.0, 143.0],
                "ALT": [13.0, 113.0],
            }
        )

        output, detail = build_global_density_horizon(
            {"1": release}, {"1": challenge}, background, member_count=1
        )

        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(int((output["time_hat"] != 11.0).sum()), 1)
        self.assertEqual(detail["targets"]["1"]["selected_rows"], 1)
        self.assertEqual(
            detail["targets"]["1"]["raw_release_background_exact_rows"], 0
        )

    def test_inverted_builder_selects_the_opposite_density_tail(self) -> None:
        release = _reference(0.0)
        background = _reference(100.0)
        challenge = pd.DataFrame(
            {
                "challenge_row_id": [0, 1],
                "age": [42.0, 42.0],
                "sex": [0, 0],
                "prefecture": ["A", "A"],
                "SBP": [103.0, 203.0],
                "TG": [53.0, 153.0],
                "HDL": [43.0, 143.0],
                "ALT": [13.0, 113.0],
            }
        )

        ordinary, _ = build_global_density_horizon(
            {"1": release}, {"1": challenge}, background, member_count=1
        )
        inverted, detail = build_global_density_horizon(
            {"1": release},
            {"1": challenge},
            background,
            member_count=1,
            invert=True,
        )

        ordinary_selected = ordinary.loc[ordinary["time_hat"].ne(11.0), "challenge_row_id"]
        inverted_selected = inverted.loc[inverted["time_hat"].ne(11.0), "challenge_row_id"]
        self.assertNotEqual(ordinary_selected.item(), inverted_selected.item())
        self.assertTrue(detail["inverted"])

    def test_sex_consensus_builder_ignores_prefecture_for_proximity(self) -> None:
        release = _reference(0.0)
        release["prefecture"] = [f"P{index}" for index in range(len(release))]
        challenge = pd.DataFrame(
            {
                "challenge_row_id": [0, 1],
                "age": [42.0, 42.0],
                "sex": [0, 0],
                "prefecture": ["UNSEEN", "UNSEEN"],
                "SBP": [103.0, 106.0],
                "TG": [53.0, 56.0],
                "HDL": [43.0, 46.0],
                "ALT": [13.0, 16.0],
            }
        )
        expert = pd.DataFrame(
            {
                "target_k": [1, 1],
                "challenge_row_id": [0, 1],
                "time_hat": [10.0, 11.0],
            }
        )

        score = sex_copula_proximity_score(release, challenge)
        output, detail = build_sex_consensus_horizon(
            expert,
            expert,
            {"1": release},
            {"1": challenge},
            member_count=1,
        )

        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(int((output["time_hat"] != 11.0).sum()), 1)
        self.assertEqual(detail["targets"]["1"]["selected_rows"], 1)


if __name__ == "__main__":
    unittest.main()
