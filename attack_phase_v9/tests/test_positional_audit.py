from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase_v9.positional_audit import (
    FEATURE_SETS,
    exact_qi_membership,
    permute_within_public_strata,
    positional_feature_frame,
    prediction_metrics,
)


class PositionalAuditTests(unittest.TestCase):
    def test_exact_qi_membership_does_not_use_record_id(self) -> None:
        source = pd.DataFrame(
            {
                "record_id": [91, 92],
                "age": [50, 60],
                "SBP": [120.0, 130.0],
                "TG": [100.0, 110.0],
                "HDL": [50.0, 55.0],
                "ALT": [20.0, 25.0],
                "sex": [0, 1],
                "prefecture": ["A", "B"],
            }
        )
        challenge = pd.DataFrame(
            {
                "challenge_row_id": [0, 1, 2],
                "age": [60, 50, 55],
                "SBP": [130.0, 120.0, 125.0],
                "TG": [110.0, 100.0, 105.0],
                "HDL": [55.0, 50.0, 52.0],
                "ALT": [25.0, 20.0, 22.0],
                "sex": [1, 0, 0],
                "prefecture": ["B", "A", "A"],
            }
        )
        member, index, diagnostic = exact_qi_membership(source, challenge)
        np.testing.assert_array_equal(member, [True, True, False])
        np.testing.assert_array_equal(index, [1, 0, -1])
        self.assertEqual(diagnostic["matched_challenge_rows"], 2)

    def test_stratified_permutation_preserves_every_cell_count(self) -> None:
        challenge = pd.DataFrame(
            {
                "age": [40, 41, 42, 43, 50, 51, 52, 53],
                "sex": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        labels = np.asarray([1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
        permuted = permute_within_public_strata(
            labels, challenge, np.random.default_rng(7)
        )
        band = ((challenge["age"] - 40) // 5).astype(int)
        before = pd.DataFrame(
            {"sex": challenge["sex"], "band": band, "label": labels}
        ).groupby(["sex", "band"])["label"].sum()
        after = pd.DataFrame(
            {"sex": challenge["sex"], "band": band, "label": permuted}
        ).groupby(["sex", "band"])["label"].sum()
        pd.testing.assert_series_equal(before, after)

    def test_prediction_metrics_selects_exactly_half(self) -> None:
        truth = np.asarray([True, True, False, False] * 81 + [True, False])
        score = np.arange(326, dtype=float)
        metrics = prediction_metrics(score, truth)
        self.assertTrue(0.0 <= metrics["auc"] <= 1.0)
        self.assertTrue(0 <= metrics["selected_members"] <= 163)

    def test_feature_contract_contains_no_outcome_or_distance(self) -> None:
        forbidden = {
            "time",
            "onset",
            "death",
            "rare",
            "distance",
            "label",
            "multiplicity",
        }
        all_names = {name for columns in FEATURE_SETS.values() for name in columns}
        for token in forbidden:
            self.assertFalse(any(token in name for name in all_names))

    def test_positional_features_are_finite(self) -> None:
        challenge = pd.DataFrame(
            {
                "age": np.resize(np.arange(40, 75), 326),
                "sex": np.arange(326) % 2,
            }
        )
        source_index = np.arange(326) % 101
        features = positional_feature_frame(challenge, source_index, 101)
        self.assertEqual(len(features), 326)
        self.assertTrue(np.isfinite(features.to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
