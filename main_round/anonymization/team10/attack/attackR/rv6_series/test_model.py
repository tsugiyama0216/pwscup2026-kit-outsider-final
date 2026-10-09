"""Mechanism/invariance tests, not a mock scoring or selection benchmark."""
import copy
import unittest
import numpy as np
import pandas as pd
import runtime as rt
from model import (Metric, AttributeEvidence, blend_membership, choose_window,
                   predict_all, specificity, window_mass, window_options)


def fixture(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'age': rng.integers(40, 75, n), 'BMI': rng.uniform(18, 35, n),
        'SBP': rng.uniform(90, 160, n), 'TG': rng.uniform(30, 220, n), 'HDL': rng.uniform(25, 80, n),
        'ALT': rng.uniform(5, 90, n), 'FPG': rng.uniform(60, 160, n), 'time': rng.integers(1, 11, n).astype(float),
        'sex': rng.integers(0, 2, n), 'prefecture': rng.choice(['A', 'B'], n),
        'smoking': rng.integers(0, 2, n), 'onset': 0, 'death': 0})


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.settings = rt.read(rt.HERE / 'protocol.json')

    def test_closed_half_year_interval_matches_direct_count(self):
        times = np.array([1., 1.5, 2., 8., 9.])
        weights = np.array([.1, .2, .3, .15, .25])
        options = np.array([1.5, 2.5, 8.5, 0., 20.])
        expected = [weights[np.abs(times - t) <= .5].sum() for t in options]
        np.testing.assert_allclose(window_mass(times, weights, options), expected, atol=1e-14)

    def test_multimodal_time_chooses_supported_mode_not_mean(self):
        value, _ = choose_window(np.array([2., 2.2, 8., 8.2]), np.array([.35, .35, .15, .15]),
            np.array([5., 6.]), np.array([.5, .5]), 5., .1, .35, .12, 1., .5)
        self.assertLessEqual(abs(value - 2.1), .5)
        self.assertGreater(abs(value - 5.), 1.)

    def test_background_evidence_prevents_generic_success(self):
        times = np.array([1., 2., 3., 8.])
        weights = np.array([.1, .6, .2, .1])
        value, _ = choose_window(times, weights, times, weights, 7., .1, .35, .12, 1., .5)
        self.assertEqual(value, 7.)

    def test_window_candidates_include_joint_support_center(self):
        self.assertIn(1.5, window_options(np.array([1., 2., 8.]), 5.))

    def test_private_negatives_and_invalid_baselines(self):
        private = np.array([True, False, False, False])
        out = blend_membership(np.array([.8, 0., .4, 1.]), np.array([.99, .1, .3, .8]), private, self.settings)
        self.assertEqual(out[0], 0.)
        self.assertTrue(((out[1:] > 0) & (out[1:] < 1)).all())
        for old in [[.1, np.nan, .2, .3], [.1, -.1, .2, .3], [.1, 1.1, .2, .3]]:
            with self.assertRaises(ValueError):
                blend_membership(old, np.ones(4) / 2, private, self.settings)

    def test_specificity_has_no_exclusive_target_constraint(self):
        old = np.array([[.99, .98, .97], [.8, .2, .3], [.1, .3, .2], [.4, .5, .1]])
        mask = np.zeros(4, bool)
        out = specificity(old, mask, self.settings)
        self.assertTrue((out[0] > .5).all())
        np.testing.assert_allclose(out[:, [2, 0, 1]], specificity(old[:, [2, 0, 1]], mask, self.settings))

    def test_duplicate_reference_and_singular_covariance(self):
        bg = pd.concat([fixture(1, 4)] * 20, ignore_index=True)
        metric = Metric(bg, ['age', 'BMI'], True)
        values = metric.radii(bg, bg, [1, 3, 10], True)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values == 0).all())

    def test_distribution_forest_weights_are_probability_mass(self):
        settings = copy.deepcopy(self.settings); settings['forest_trees'] = 8
        bg, query = fixture(90, 1), fixture(6, 2)
        a = AttributeEvidence(bg, settings)
        weights = a.forest_weights(a.fit_forest(a.bg), a.bg, query)
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)

    def test_full_suite_permutation_invariance_and_no_input_mutation(self):
        settings = copy.deepcopy(self.settings); settings['forest_trees'] = 8
        b, bg = fixture(24, 2), fixture(100, 3)
        cs = {k: fixture(44, k + 4) for k in [0, 1]}
        qs = {k: fixture(5, k + 8).drop(columns=['time', 'BMI', 'FPG', 'smoking', 'onset', 'death']) for k in cs}
        m = {k: np.linspace(.01, .99, len(b)) for k in cs}
        a = {k: np.arange(5.) + 1 for k in cs}
        mask = np.zeros(len(b), bool); mask[3] = True
        copies = [f.copy(deep=True) for f in [b, bg, *cs.values(), *qs.values()]]
        result, _ = predict_all(b, bg, cs, qs, m, a, mask, settings)
        rng = np.random.default_rng(19)
        p, q = rng.permutation(len(b)), rng.permutation(5)
        flipped = {k: f.iloc[::-1].reset_index(drop=True) for k, f in cs.items()}
        # Invented IDs and hidden-looking extra columns are not predictors.
        query = {k: f.iloc[q].assign(time=-9999, truth=999, challenge_row_id=777).reset_index(drop=True) for k, f in qs.items()}
        other, _ = predict_all(b.iloc[p], bg.iloc[::-1], flipped, query,
                              {k: v[p] for k, v in m.items()}, {k: v[q] for k, v in a.items()}, mask[p], settings)
        for name in result:
            for k in cs:
                np.testing.assert_allclose(other[name]['mia'][k], result[name]['mia'][k][p], atol=1e-12, rtol=0)
                np.testing.assert_allclose(other[name]['aia'][k], result[name]['aia'][k][q], atol=1e-12, rtol=0)
                self.assertEqual(result[name]['mia'][k][3], 0.)
        for before, after in zip(copies, [b, bg, *cs.values(), *qs.values()]):
            pd.testing.assert_frame_equal(before, after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
