"""Behavior checks for the new release families; no attack model is imported."""
import copy
import unittest

import numpy as np
import pandas as pd

from sugiyama_v2.releases import FEATURES, PROFILES, STYLES, describe_styles, make_release


SCHEMA = {
    "ranges": {"BMI": [10, 55], "SBP": [40, 250], "TG": [1, 3000],
               "HDL": [10, 160], "ALT": [.1, 700], "FPG": [40, 400]},
    "age_range": [40, 74],
    "allowed_onset_death": [[0, 0], [0, 1], [1, 0]],
}


def sample(seed, n):
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 75, size=n)
    event = rng.choice([0, 1, 2], size=n, p=[.80, .10, .10])
    frame = pd.DataFrame({
        "age": age, "sex": rng.integers(2, size=n),
        "prefecture": rng.choice(["P01", "P02", "P03"], size=n),
        "BMI": rng.uniform(16, 35, size=n), "SBP": 60 + age + rng.normal(0, 12, size=n),
        "TG": rng.uniform(20, 350, size=n), "HDL": rng.uniform(25, 100, size=n),
        "ALT": rng.uniform(2, 150, size=n), "smoking": rng.integers(2, size=n),
        "FPG": rng.uniform(60, 220, size=n), "time": rng.uniform(.1, 10, size=n),
        "onset": (event == 2).astype(int), "death": (event == 1).astype(int),
    })
    return frame[FEATURES]


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target = sample(41, 1049)
        cls.background = sample(73, 1457)

    def test_every_style_is_reproducible_valid_and_does_not_mutate_inputs(self):
        target_before = self.target.copy(deep=True)
        background_before = self.background.copy(deep=True)
        schema = copy.deepcopy(SCHEMA)
        for style in STYLES:
            with self.subTest(style=style):
                first = make_release(self.target, self.background, style, 101, schema)
                second = make_release(self.target, self.background, style, 101, schema)
                pd.testing.assert_frame_equal(first, second)
                self.assertEqual(list(first.columns), FEATURES)
                self.assertEqual(first.shape, (1049, 13))
                self.assertTrue(np.isfinite(first.drop(columns="prefecture").to_numpy(float)).all())
                self.assertTrue(first.age.between(40, 74).all())
                self.assertTrue(np.equal(first.age, np.rint(first.age)).all())
                self.assertTrue(first.time.between(1 / 365.25, 10).all())
                for column, bounds in SCHEMA["ranges"].items():
                    self.assertTrue(first[column].between(*bounds).all())
                for column in ("sex", "smoking", "onset", "death"):
                    self.assertTrue(first[column].isin([0, 1]).all())
                self.assertTrue(first.prefecture.isin(self.background.prefecture.unique()).all())
                self.assertFalse(((first.onset == 1) & (first.death == 1)).any())
        pd.testing.assert_frame_equal(self.target, target_before)
        pd.testing.assert_frame_equal(self.background, background_before)
        self.assertEqual(schema, SCHEMA)

    def test_independent_release_ignores_all_target_content_and_size(self):
        first = make_release(self.target, self.background, "independent", 19, SCHEMA)
        for replacement in (None, pd.DataFrame(), sample(97, 1), sample(13, 1049)):
            with self.subTest(replacement_type=type(replacement).__name__):
                result = make_release(replacement, self.background, "independent", 19, SCHEMA)
                pd.testing.assert_frame_equal(first, result)

    def test_seed_changes_stochastic_releases(self):
        for style in ("jitter", "strong_jitter", "copula", "conditional_resample", "background_mix", "independent"):
            with self.subTest(style=style):
                first = make_release(self.target, self.background, style, 101, SCHEMA)
                second = make_release(self.target, self.background, style, 102, SCHEMA)
                self.assertFalse(first.equals(second))

    def test_copy_preserves_public_values_as_a_multiset(self):
        result = make_release(self.target, self.background, "copy", 43, SCHEMA)
        pd.testing.assert_frame_equal(result.sort_values("time").reset_index(drop=True),
                                      self.target.sort_values("time").reset_index(drop=True))

    def test_outcome_shuffle_preserves_joint_outcomes_within_public_strata(self):
        result = make_release(self.target, self.background, "outcome_shuffle", 17, SCHEMA)
        for sex in (0, 1):
            for decade in (4, 5, 6, 7):
                left = self.target[(self.target.sex == sex) & (self.target.age // 10 == decade)]
                right = result[(result.sex == sex) & (result.age // 10 == decade)]
                columns = ["time", "onset", "death"]
                pd.testing.assert_frame_equal(left[columns].sort_values("time").reset_index(drop=True),
                                              right[columns].sort_values("time").reset_index(drop=True))
        self.assertFalse(result.sort_values("SBP").reset_index(drop=True).equals(
            self.target.sort_values("SBP").reset_index(drop=True)))

    def test_copula_is_new_data_and_handles_constant_columns(self):
        changed = self.target.copy(deep=True)
        changed["sex"] = 0
        changed["smoking"] = 0
        changed["onset"] = 0
        changed["death"] = 0
        result = make_release(changed, self.background, "copula", 52, SCHEMA)
        self.assertEqual(len(result.merge(changed, on=FEATURES)), 0)
        for column in ("sex", "smoking", "onset", "death"):
            self.assertTrue(result[column].eq(0).all())

    def test_profiles_include_nulls_without_easy_scenario_dominance(self):
        metadata = describe_styles()
        self.assertFalse(metadata["fit_to_official_feedback"])
        for name, profile in PROFILES.items():
            self.assertEqual(len(profile), 22)
            self.assertTrue(set(profile).issubset(STYLES))
            self.assertIn("independent", profile)
            self.assertLessEqual(profile.count("copy") + profile.count("rounded"), 1)
            self.assertEqual(metadata["profiles"][name]["styles"], list(profile))

    def test_bad_input_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            make_release(self.target, self.background, "typo", 11, SCHEMA)
        with self.assertRaisesRegex(ValueError, "seed"):
            make_release(self.target, self.background, "copy", -1, SCHEMA)
        with self.assertRaisesRegex(ValueError, "1049"):
            make_release(self.target.iloc[:10], self.background, "copy", 11, SCHEMA)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            make_release(None, self.background.iloc[:0], "independent", 11, SCHEMA)
        broken = self.target.copy(deep=True)
        broken.loc[0, "time"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            make_release(broken, self.background, "conditional_resample", 11, SCHEMA)


if __name__ == "__main__":
    unittest.main()
