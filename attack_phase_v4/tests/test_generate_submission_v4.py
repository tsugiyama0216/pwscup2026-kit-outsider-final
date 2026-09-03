from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from attack_phase_v4.generate_submission_v4 import (
    build_mia_v4,
    exact_background_donor_mask,
    filter_background_donor_targets,
    write_outputs,
)
from attack_phase_v4.mia_private_conformal import (
    demote_known_private,
    empirical_lower_tail,
    empirical_upper_tail,
    preserve_subset_order,
    reciprocal_rank_fusion,
    strict_lexicographic_rank,
)


class AttackV4Tests(unittest.TestCase):
    @staticmethod
    def _rows(values: list[float]) -> pd.DataFrame:
        n = len(values)
        return pd.DataFrame(
            {
                "age": np.arange(40, 40 + n),
                "sex": np.arange(n) % 2,
                "prefecture": [f"P{i}" for i in range(n)],
                "BMI": np.asarray(values),
                "SBP": np.asarray(values) + 100,
                "TG": np.asarray(values) + 50,
                "HDL": np.asarray(values) + 40,
                "ALT": np.asarray(values) + 10,
                "smoking": np.arange(n) % 2,
                "FPG": np.asarray(values) + 80,
                "time": np.asarray(values),
                "onset": np.arange(n) % 2,
                "death": np.zeros(n, dtype=int),
            }
        )

    def test_exact_background_donor_mask_uses_full_rows(self) -> None:
        background = self._rows([1.0, 2.0, 3.0])
        released = pd.concat(
            [background.iloc[[1]], self._rows([9.0])], ignore_index=True
        )
        mask = exact_background_donor_mask(released, background)
        self.assertEqual(mask.tolist(), [True, False])

    def test_empirical_tail_probabilities_are_smoothed(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 3.0])
        null = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            empirical_lower_tail(values, null),
            [1 / 3, 2 / 3, 1.0, 1.0],
        )
        np.testing.assert_allclose(
            empirical_upper_tail(values, null),
            [1.0, 1.0, 2 / 3, 1 / 3],
        )

    def test_lexicographic_rank_uses_continuous_tie_break(self) -> None:
        ranked = strict_lexicographic_rank(
            np.array([1.0, 1.0, 2.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([10, 20, 30]),
        )
        self.assertGreater(ranked[0], ranked[1])
        self.assertGreater(ranked[2], ranked[0])

    def test_preserve_subset_order_keeps_non_subset_scores(self) -> None:
        candidate = pd.DataFrame(
            {
                "record_id": [10, 20, 30, 40],
                "1": [0.1, 0.8, 0.4, 0.6],
                "2": [0.9, 0.2, 0.5, 0.3],
            }
        )
        reference = pd.DataFrame(
            {
                "record_id": [10, 20, 30, 40],
                "1": [0.1, 0.2, 0.3, 0.4],
                "2": [0.4, 0.3, 0.2, 0.1],
            }
        )
        subset = np.array([False, True, False, True])
        output = preserve_subset_order(candidate, reference, subset)
        np.testing.assert_allclose(output.loc[~subset, "1"], [0.1, 0.4])
        np.testing.assert_allclose(output.loc[~subset, "2"], [0.9, 0.5])
        self.assertLess(output.loc[1, "1"], output.loc[3, "1"])
        self.assertGreater(output.loc[1, "2"], output.loc[3, "2"])

    def test_rrf_rewards_consensus_and_demotes_known_private(self) -> None:
        reference = pd.DataFrame(
            {"record_id": [0, 1, 2, 3], "1": [0.2, 0.6, 0.8, 0.4]}
        )
        candidate = pd.DataFrame(
            {"record_id": [0, 1, 2, 3], "1": [0.9, 0.7, 0.8, 0.1]}
        )
        output = reciprocal_rank_fusion(
            reference,
            candidate,
            np.array([True, False, False, False]),
            rank_constant=10.0,
        )
        self.assertEqual(output.loc[0, "1"], output["1"].min())
        self.assertEqual(output.loc[2, "1"], output["1"].max())

    def test_mia_v4_requires_public_rare_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "rare mask"):
            build_mia_v4(
                pd.DataFrame(),
                pd.DataFrame(),
                {},
                np.array([], dtype=bool),
                minimum=0,
                maximum=1,
                total=0,
                transport_epsilon=0.35,
                include_domias=False,
                density_weight=0.35,
                rare_mask=None,
                rare_weight=0.65,
            )

    @patch("attack_phase_v4.generate_submission_v4.build_private_conformal_mia")
    @patch("attack_phase_v4.generate_submission_v4.v3.build_mia_v3")
    @patch("attack_phase_v4.generate_submission_v4.filter_background_donor_targets")
    def test_donor_filter_remains_experimental(
        self,
        filtered_mock,
        v3_mock,
        conformal_mock,
    ) -> None:
        reference = pd.DataFrame(
            {
                "record_id": [0, 1, 2, 3, 4],
                "1": [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6],
            }
        )
        unfiltered = pd.DataFrame(
            {
                "record_id": [0, 1, 2, 3, 4],
                "1": [1 / 6, 3 / 6, 5 / 6, 2 / 6, 4 / 6],
            }
        )
        filtered = pd.DataFrame(
            {
                "record_id": [0, 1, 2, 3, 4],
                "1": [1 / 6, 3 / 6, 2 / 6, 5 / 6, 4 / 6],
            }
        )
        filtered_mock.return_value = (
            {"1": pd.DataFrame()},
            {"affected_targets": ["1"], "targets": {}},
        )
        v3_mock.return_value = (reference, {}, {})
        conformal_mock.side_effect = [
            (
                {
                    "conformal_margin_0p50": unfiltered,
                    "conformal_margin_0p25": unfiltered,
                    "conformal_plain": unfiltered,
                },
                {},
            ),
            (
                {
                    "conformal_margin_0p50": filtered,
                    "conformal_margin_0p25": filtered,
                    "conformal_plain": filtered,
                },
                {},
            ),
        ]
        known_private = np.array([True, False, False, False, False])
        rare = np.array([False, True, False, False, True])
        main, variants, diagnostics = build_mia_v4(
            pd.DataFrame({"record_id": [0, 1, 2, 3, 4]}),
            pd.DataFrame(),
            {"1": pd.DataFrame()},
            known_private,
            minimum=0,
            maximum=1,
            total=1,
            transport_epsilon=0.35,
            include_domias=False,
            density_weight=0.35,
            rare_mask=rare,
            rare_weight=0.65,
        )
        expected = reciprocal_rank_fusion(
            reference, unfiltered, known_private, rank_constant=10.0
        )
        expected = preserve_subset_order(expected, reference, rare)
        expected = demote_known_private(expected, known_private)
        self.assertTrue(main.equals(expected))
        self.assertFalse(
            main.equals(
                variants[
                    "v4_rrf_k10_donor_filtered_rare_preserved_experimental"
                ]
            )
        )
        self.assertFalse(
            diagnostics["background_donor_filter_experiment"][
                "selected_for_main"
            ]
        )

    def test_write_outputs_removes_stale_generated_variants(self) -> None:
        mia = pd.DataFrame({"record_id": [0], "1": [0.5]})
        aia = pd.DataFrame(
            {"target_k": [1], "challenge_row_id": [0], "time_hat": [10.0]}
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            variants_dir = destination / "variants"
            variants_dir.mkdir()
            stale = variants_dir / "F_mia_stale.csv"
            stale.write_text("stale\n", encoding="utf-8")
            unrelated = variants_dir / "keep.txt"
            unrelated.write_text("keep\n", encoding="utf-8")
            write_outputs(
                destination,
                mia,
                aia,
                diagnostics=None,
                variants={"mia": {"current": mia}, "aia": {"current": aia}},
                candidate=False,
            )
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue((variants_dir / "F_mia_current.csv").exists())
            self.assertTrue((variants_dir / "F_aia_current.csv").exists())

    def test_filter_applies_only_when_rate_and_signal_floor_hold(self) -> None:
        background = self._rows([1.0, 2.0, 3.0, 4.0])
        signal = self._rows([8.0, 9.0])
        dominated = pd.concat([background, signal], ignore_index=True)
        filtered, diagnostics = filter_background_donor_targets(
            {"1": dominated},
            background,
            trigger_rate=0.5,
            minimum_signal_rows=2,
        )
        self.assertEqual(len(filtered["1"]), 2)
        self.assertEqual(diagnostics["affected_targets"], ["1"])

        unchanged, diagnostics = filter_background_donor_targets(
            {"1": dominated},
            background,
            trigger_rate=0.5,
            minimum_signal_rows=3,
        )
        self.assertIs(unchanged["1"], dominated)
        self.assertEqual(diagnostics["affected_targets"], [])

    @patch("attack_phase_v4.generate_submission_v4.build_private_conformal_mia")
    @patch("attack_phase_v4.generate_submission_v4.v3.build_mia_v3")
    @patch("attack_phase_v4.generate_submission_v4.filter_background_donor_targets")
    def test_mia_v4_selects_rrf_and_preserves_rare_order(
        self,
        filtered_mock,
        v3_mock,
        conformal_mock,
    ) -> None:
        reference = pd.DataFrame(
            {"record_id": [0, 1, 2, 3], "1": [0.2, 0.4, 0.6, 0.8]}
        )
        conformal = pd.DataFrame(
            {"record_id": [0, 1, 2, 3], "1": [0.1, 0.9, 0.8, 0.2]}
        )
        filtered_mock.return_value = (
            {"1": pd.DataFrame()},
            {"affected_targets": [], "targets": {}},
        )
        v3_mock.return_value = (reference, {}, {})
        conformal_mock.return_value = (
            {
                "conformal_margin_0p50": conformal,
                "conformal_margin_0p25": conformal,
                "conformal_plain": conformal,
            },
            {},
        )
        main, variants, diagnostics = build_mia_v4(
            pd.DataFrame({"record_id": [0, 1, 2, 3]}),
            pd.DataFrame(),
            {"1": pd.DataFrame()},
            np.array([True, False, False, False]),
            minimum=0,
            maximum=1,
            total=1,
            transport_epsilon=0.35,
            include_domias=False,
            density_weight=0.35,
            rare_mask=np.array([False, True, False, True]),
            rare_weight=0.65,
        )
        self.assertLess(main.loc[1, "1"], main.loc[3, "1"])
        self.assertTrue(
            variants["v4_rrf_k10_rare_preserved"].equals(main)
        )
        self.assertEqual(
            diagnostics["main_variant"],
            "v4_rrf_k10_rare_preserved",
        )


if __name__ == "__main__":
    unittest.main()
