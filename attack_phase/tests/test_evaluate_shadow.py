from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from attack_phase.evaluate_shadow import (
    ShadowSpec,
    allocate_shared_counts,
    coerce_release,
    match_unique_controls,
    sample_aia_members,
    scenario_release,
    split_public_and_hidden_background,
)
from attack_phase.generate_submission import MIA_REQUIRED_COLUMNS


class ShadowEvaluationTests(unittest.TestCase):
    @staticmethod
    def _population(seed: int, n: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        onset = rng.integers(0, 2, size=n)
        death = np.where(onset == 0, rng.integers(0, 2, size=n), 0)
        return pd.DataFrame(
            {
                "record_id": np.arange(n),
                "age": rng.integers(40, 75, size=n),
                "sex": rng.integers(0, 2, size=n),
                "prefecture": rng.choice(["A", "B", "C"], size=n),
                "BMI": rng.normal(23.0, 3.0, size=n),
                "SBP": rng.normal(125.0, 15.0, size=n),
                "TG": rng.lognormal(4.5, 0.4, size=n),
                "HDL": rng.normal(50.0, 8.0, size=n),
                "ALT": rng.lognormal(2.8, 0.4, size=n),
                "smoking": rng.integers(0, 2, size=n),
                "FPG": rng.normal(95.0, 12.0, size=n),
                "time": rng.uniform(0.1, 10.0, size=n),
                "onset": onset,
                "death": death,
            }
        )

    @staticmethod
    def _cfg() -> dict:
        return {
            "population": {"age_range": [40, 74]},
            "onset": {"horizon_years": 10.0},
            "schema": {
                "ranges": {
                    "BMI": [10.0, 55.0],
                    "SBP": [40.0, 250.0],
                    "TG": [1.0, 3000.0],
                    "HDL": [10.0, 160.0],
                    "ALT": [0.1, 700.0],
                    "FPG": [40.0, 400.0],
                }
            },
        }

    def test_published_membership_cardinalities(self) -> None:
        spec = ShadowSpec()
        counts = allocate_shared_counts(spec, np.random.default_rng(7))
        self.assertEqual(len(counts), 29)
        self.assertEqual(int(counts.sum()), 736)
        self.assertGreaterEqual(int(counts.min()), 25)
        self.assertLessEqual(int(counts.max()), 27)
        self.assertIn(27, counts)

    def test_control_matching_is_unique_and_excludes_target_rows(self) -> None:
        background = self._population(8, 220)
        excluded = np.arange(80)
        members = background.iloc[excluded[:25]].reset_index(drop=True)
        chosen, metadata = match_unique_controls(members, background, excluded)
        self.assertEqual(len(chosen), len(members))
        self.assertEqual(len(np.unique(chosen)), len(members))
        self.assertFalse(bool(np.isin(chosen, excluded).any()))
        self.assertEqual(metadata["unique"], len(members))

    def test_aia_member_sampling_matches_published_strata(self) -> None:
        population = self._population(81, 2_000)
        # Give the test a deterministic rare label with enough rows in both onset strata.
        rare = np.zeros(len(population), dtype=bool)
        rare[:300] = True
        population.loc[:49, "onset"] = 1
        population.loc[50:299, "onset"] = 0
        indices = sample_aia_members(
            population, rare, ShadowSpec(), np.random.default_rng(82)
        )
        selected = population.iloc[indices]
        selected_rare = rare[indices]
        selected_onset = selected["onset"].astype(bool).to_numpy()
        self.assertEqual(len(indices), 163)
        self.assertEqual(int(selected_rare.sum()), 13)
        self.assertEqual(int(selected_onset.sum()), 16)
        self.assertEqual(int((selected_rare & selected_onset).sum()), 4)

    def test_background_split_is_stable_and_disjoint(self) -> None:
        population = self._population(83, 500)
        visible_a, hidden_a = split_public_and_hidden_background(population)
        visible_b, hidden_b = split_public_and_hidden_background(population)
        self.assertTrue(visible_a.equals(visible_b))
        self.assertTrue(hidden_a.equals(hidden_b))
        self.assertFalse(
            bool(
                set(visible_a["record_id"].tolist())
                & set(hidden_a["record_id"].tolist())
            )
        )
        self.assertEqual(len(visible_a) + len(hidden_a), len(population))

    def test_raw_and_time_shuffle_scenarios_have_expected_invariants(self) -> None:
        population = self._population(9, 120)
        source = population.iloc[:60].copy()
        raw = scenario_release(
            source,
            population,
            "raw",
            self._cfg(),
            np.random.default_rng(10),
        )
        shuffled = scenario_release(
            source,
            population,
            "time_shuffled",
            self._cfg(),
            np.random.default_rng(11),
        )
        submitted = list(MIA_REQUIRED_COLUMNS)
        self.assertEqual(sorted(map(tuple, raw[submitted].to_numpy())), sorted(map(tuple, source[submitted].to_numpy())))
        static = [column for column in submitted if column not in ("time", "onset", "death")]
        self.assertEqual(
            sorted(map(tuple, shuffled[static].to_numpy())),
            sorted(map(tuple, source[static].to_numpy())),
        )
        self.assertEqual(
            sorted(map(tuple, shuffled[["time", "onset", "death"]].to_numpy())),
            sorted(map(tuple, source[["time", "onset", "death"]].to_numpy())),
        )

    def test_coercion_enforces_ranges_and_outcome_exclusivity(self) -> None:
        frame = self._population(12, 5)
        frame.loc[:, "age"] = [1, 40, 60, 74, 100]
        frame.loc[:, "BMI"] = [-100, 10, 23, 55, 100]
        frame.loc[:, "time"] = [-1, 0, 5, 10, 20]
        frame.loc[:, "onset"] = 1
        frame.loc[:, "death"] = 1
        out = coerce_release(frame, self._cfg())
        self.assertTrue(out["age"].between(40, 74).all())
        self.assertTrue(out["BMI"].between(10, 55).all())
        self.assertTrue(out["time"].between(1.0 / 365.25, 10).all())
        self.assertTrue(((out["onset"] + out["death"]) <= 1).all())


if __name__ == "__main__":
    unittest.main()
