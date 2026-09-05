import unittest

import numpy as np
import pandas as pd

from attack_phase_v9.generate_guarded_submission_v9 import (
    EXPECTED_TARGET_IDS, SOURCE_ORDER, ZERO_TARGETS, canonical_aia,
    compose_guarded_aia, conditional_floor,
)


class GuardedSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.sources = {label: pd.DataFrame([
            (int(t), i, 8.0 + n) for t in EXPECTED_TARGET_IDS for i in range(326)],
            columns=["target_k", "challenge_row_id", "time_hat"])
            for n, label in enumerate(SOURCE_ORDER)}
        self.scores = {label: {t: 0.0 if t in ZERO_TARGETS else (n + 1) / 10
                               for t in EXPECTED_TARGET_IDS}
                       for n, label in enumerate(SOURCE_ORDER)}
        self.v7 = self.sources["v4"].copy()
        self.v7["time_hat"] = 9.5

    def test_whole_positive_blocks_and_v7_zeros_preserved(self):
        out, info = compose_guarded_aia(self.sources, self.scores, self.v7)
        zero = out.target_k.isin([int(t) for t in ZERO_TARGETS])
        self.assertTrue((out.loc[zero, "time_hat"] == 9.5).all())
        self.assertTrue((out.loc[~zero, "time_hat"] == 10).all())
        self.assertAlmostEqual(info["measured_aia_floor"], 13 * .3 / 20)
        self.assertEqual(len(out), 6520)

    def test_input_order_does_not_change_output(self):
        first, _ = compose_guarded_aia(self.sources, self.scores, self.v7)
        shuffled = {k: v.sample(frac=1, random_state=19) for k, v in self.sources.items()}
        second, _ = compose_guarded_aia(shuffled, self.scores, self.v7.iloc[::-1])
        pd.testing.assert_frame_equal(first, second)

    def test_fractional_duplicate_missing_extra_keys_fail(self):
        variants = []
        fractional = self.v7.astype({"challenge_row_id": float})
        fractional.loc[0, "challenge_row_id"] = .5
        variants.append(fractional)
        variants.append(pd.concat([self.v7, self.v7.iloc[:1]]))
        variants.append(self.v7.iloc[1:])
        extra = self.v7.copy()
        extra.loc[0, "target_k"] = 24
        variants.append(extra)
        for frame in variants:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                canonical_aia(frame)

    def test_nonfinite_times_and_feedback_fail(self):
        invalid = self.v7.copy()
        invalid.loc[0, "time_hat"] = np.inf
        with self.assertRaises(ValueError):
            compose_guarded_aia(self.sources, self.scores, invalid)
        self.scores["v4"]["1"] = np.nan
        with self.assertRaises(ValueError):
            compose_guarded_aia(self.sources, self.scores, self.v7)

    def test_positive_feedback_cannot_be_overwritten_as_zero(self):
        self.scores["routed"]["6"] = .001
        with self.assertRaises(ValueError):
            compose_guarded_aia(self.sources, self.scores, self.v7)

    def test_floor_does_not_count_unknown_experiments(self):
        feedback = {"score_1_displayed": .3,
                    "aia_per_target": {t: .1 for t in EXPECTED_TARGET_IDS}}
        floor = conditional_floor(feedback, .15)
        self.assertAlmostEqual(floor["conditional_lower_bound"], .34995)
        self.assertTrue(floor["not_a_new_official_score"])
        self.assertTrue(floor["not_proven_superior_to_v7"])


if __name__ == "__main__":
    unittest.main()
