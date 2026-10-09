"""Behavioral checks for Rv12's public-only score and conservative AIA veto."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[4]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

model = load('rv12_numeric_test', Path(__file__).with_name('model.py'))
frozen = load('rv8_numeric_reference', ROOT / 'main_round/anonymization/team10/attack/attackR/rv6_series/model.py')


def frame(seed, n=80):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'age': rng.uniform(40, 80, n), 'sex': rng.integers(0, 2, n),
        'prefecture': rng.integers(1, 4, n), 'BMI': rng.uniform(18, 33, n),
        'SBP': rng.uniform(90, 160, n), 'TG': rng.uniform(50, 240, n),
        'HDL': rng.uniform(35, 90, n), 'ALT': rng.uniform(5, 80, n),
        'smoking': rng.integers(0, 3, n), 'FPG': rng.uniform(70, 150, n),
        'time': rng.uniform(0, 10, n), 'onset': rng.integers(0, 2, n),
        'death': rng.integers(0, 2, n),
    })


def problem():
    background, c0, c2 = frame(17), frame(19), frame(23)
    candidates = frame(29, 9)
    candidates.insert(0, 'record_id', np.arange(901, 910))
    targets, am = {}, {}
    for k, c in [(0, c0), (2, c2)]:
        q = c.iloc[:7][model.QI + ['sex', 'prefecture']].copy().reset_index(drop=True)
        q.insert(0, 'challenge_row_id', np.arange(300 + k * 10, 307 + k * 10))
        targets[k] = {'release': c, 'challenge': q}
        am[k] = np.array([10., 11., 5., 1., 9.5, 3., 0.])
    return dict(candidates=candidates, background=background, targets=targets,
                baseline_mia={0: np.array([0., .2, .2, .4, .9, .7, .6, .1, .5]),
                              2: np.array([0., .4, .6, .8, .3, .3, .5, .9, .1])},
                baseline_aia=am, known_negative=np.array([True] + [False] * 8))


class MembershipTests(unittest.TestCase):
    def test_matches_frozen_rv8_on_ties_extremes_and_known_negatives(self):
        rng = np.random.default_rng(61)
        old = np.round(rng.uniform(size=(73, 22)), 1)
        old[0] = 0.
        old[1] = 1.
        mask = np.arange(len(old)) % 9 == 0
        before = old.copy()
        expected = frozen.specificity(old, mask,
            {'mia_logit_cap': 7., 'specificity_strength': .35})
        actual = model.membership(old, mask)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(old, before)
        np.testing.assert_array_equal(actual[mask], 0.)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_candidate_and_target_order_equivariance(self):
        rng = np.random.default_rng(101)
        old = rng.uniform(size=(37, 5))
        mask = np.arange(37) % 7 == 0
        rows, columns = rng.permutation(37), rng.permutation(5)
        expected = model.membership(old, mask)[rows][:, columns]
        actual = model.membership(old[rows][:, columns], mask[rows])
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_mia_rejected(self):
        for old, mask in [(np.ones(5), np.zeros(5, bool)),
                          (np.ones((5, 1)), np.zeros(5, bool)),
                          (np.ones((5, 2)), np.zeros(4, bool)),
                          (np.ones((5, 2)), np.ones(5, bool)),
                          (np.full((5, 2), np.nan), np.zeros(5, bool)),
                          (np.full((5, 2), 1.01), np.zeros(5, bool))]:
            with self.subTest(shape=old.shape), self.assertRaises(ValueError):
                model.membership(old, mask)


class VetoTests(unittest.TestCase):
    def test_all_conditions_and_threshold_boundaries(self):
        old = np.array([10., 10., 10., 10., 10., 10., 5., 11., 9.5])
        ev = {'pc': np.full((9, 2), .8), 'pb': np.full((9, 2), .8),
              'ratio': np.full(9, 2.), 'strong_link': np.zeros(9, bool)}
        ev['ratio'][1] = 1.5           # Strictly greater is necessary.
        ev['pb'][2, 1] = .79          # Both background scales must pass.
        ev['pc'][3, 0] = .81          # Both excess scales must pass.
        ev['strong_link'][4] = True   # Strong links are always retained.
        ev['ratio'][5] = 1.49
        before = old.copy()
        actual, removed = model.veto(old, ev, model.VARIANTS['censor_strict'])
        np.testing.assert_array_equal(removed, [True, False, False, False, False, False, False, False, True])
        np.testing.assert_array_equal(actual, [11., 10., 10., 10., 10., 10., 5., 11., 11.])
        np.testing.assert_array_equal(old, before)
        alltime, mask = model.veto(old, ev, model.VARIANTS['contrast_strict'])
        self.assertTrue(mask[6])
        self.assertEqual(alltime[6], 11.)
        self.assertFalse(mask[7])

    def test_retain_is_unchanged_independent_copy(self):
        old = np.array([0., 4., 9.5, 10., 11.])
        answer, removed = model.veto(old, {}, None)
        np.testing.assert_array_equal(answer, old)
        self.assertFalse(removed.any())
        answer[0] = 1.
        self.assertEqual(old[0], 0.)

    def test_window_mass_includes_endpoints_and_respects_weight(self):
        masses, _ = model.AttributeVeto.masses(
            np.zeros((1, 4)), np.array([4.5, 5.5, 5.500001, 2.]), np.array([5.]), [2, 4])
        np.testing.assert_array_equal(masses, [[1., .5]])

    def test_exact_link_exemption_and_reference_permutation(self):
        bg, release = frame(211), frame(307)
        query = release.iloc[[7, 11, 3]][model.QI + ['sex', 'prefecture']].copy()
        old = np.array([10., 10., 10.])
        before_b, before_c, before_q = bg.copy(deep=True), release.copy(deep=True), query.copy(deep=True)
        engine = model.AttributeVeto(bg)
        ev = engine.evidence(release, query, old)
        self.assertTrue(ev['strong_link'].all())
        ev2 = model.AttributeVeto(bg.sample(frac=1, random_state=4)).evidence(
            release.sample(frac=1, random_state=5), query, old)
        for key in ev:
            np.testing.assert_array_equal(ev[key], ev2[key])
        for name, settings in model.VARIANTS.items():
            with self.subTest(variant=name):
                answer, removed = model.veto(old, ev, settings)
                np.testing.assert_array_equal(answer, old)
                self.assertFalse(removed.any())
        assert_frame_equal(bg, before_b)
        assert_frame_equal(release, before_c)
        assert_frame_equal(query, before_q)

    def test_sentinel_validity_requires_public_time_range(self):
        bg, release = frame(401), frame(409)
        q = release.iloc[:2][model.QI + ['sex', 'prefecture']]
        for name, bad in [('negative', -.01), ('beyond_horizon', 10.01), ('nonfinite', np.nan)]:
            for which in ('release', 'background'):
                with self.subTest(value=name, reference=which):
                    b, c = bg.copy(), release.copy()
                    (b if which == 'background' else c).loc[0, 'time'] = bad
                    with self.assertRaises(ValueError):
                        model.AttributeVeto(b).evidence(c, q, np.array([5., 10.]))
        # For every valid truth in the contest horizon, 11 cannot be a hit.
        self.assertTrue((np.abs(np.linspace(0, 10, 101) - model.ABSTAIN) > .5).all())

    def test_bad_baseline_qi_and_short_reference_rejected(self):
        bg, release = frame(421), frame(431)
        q = release.iloc[:2][model.QI + ['sex', 'prefecture']].copy()
        engine = model.AttributeVeto(bg)
        for old in [np.array([1.]), np.array([1., np.nan]), np.array([-1., 2.])]:
            with self.assertRaises(ValueError):
                engine.evidence(release, q, old)
        for name, value in [('TG', -1.), ('ALT', -1.), ('age', np.inf)]:
            bad = q.copy(); bad.iloc[0, bad.columns.get_loc(name)] = value
            with self.assertRaises(ValueError):
                engine.evidence(release, bad, np.array([1., 2.]))
        with self.assertRaises(ValueError):
            model.AttributeVeto(bg.iloc[:63])
        with self.assertRaises(ValueError):
            engine.evidence(release.iloc[:63], q, np.array([1., 2.]))


class PublicWorldTests(unittest.TestCase):
    def test_complete_outputs_same_mia_and_no_input_mutation(self):
        p = problem()
        bcopy, ccopy = p['background'].copy(deep=True), p['candidates'].copy(deep=True)
        mcopy = {k: v.copy() for k, v in p['baseline_mia'].items()}
        acopy = {k: v.copy() for k, v in p['baseline_aia'].items()}
        tcopies = {k: {name: value.copy(deep=True) for name, value in target.items()}
                   for k, target in p['targets'].items()}
        outputs = model.predict_all(**p)
        self.assertEqual(set(outputs), set(model.VARIANTS))
        for name, out in outputs.items():
            with self.subTest(variant=name):
                self.assertEqual(out['F_mia'].shape, (9, 3))
                self.assertEqual(out['F_aia'].shape, (14, 3))
                self.assertTrue(np.isfinite(out['F_mia'].to_numpy()).all())
                self.assertTrue(np.isfinite(out['F_aia'].to_numpy()).all())
                assert_frame_equal(out['F_mia'], outputs['retain']['F_mia'])
                for k in p['targets']:
                    actual = out['F_aia'].loc[out['F_aia'].target_k == k, 'time_hat'].to_numpy()
                    self.assertTrue(((actual == acopy[k]) | (actual == 11.)).all())
                    self.assertEqual(out['diagnostics'][str(k)]['suppressed'], int((actual != acopy[k]).sum()))
        assert_frame_equal(p['background'], bcopy)
        assert_frame_equal(p['candidates'], ccopy)
        for k in mcopy:
            np.testing.assert_array_equal(p['baseline_mia'][k], mcopy[k])
            np.testing.assert_array_equal(p['baseline_aia'][k], acopy[k])
            for name in tcopies[k]:
                assert_frame_equal(p['targets'][k][name], tcopies[k][name])

    def test_public_id_alignment_survives_all_row_permutations(self):
        p = problem()
        before = model.predict_all(**p)
        cand_order = np.array([4, 2, 7, 0, 8, 1, 5, 6, 3])
        q_order = np.array([4, 0, 6, 1, 5, 3, 2])
        p['candidates'] = p['candidates'].iloc[cand_order].reset_index(drop=True)
        p['known_negative'] = p['known_negative'][cand_order]
        p['background'] = p['background'].sample(frac=1, random_state=501)
        for k in p['targets']:
            p['baseline_mia'][k] = p['baseline_mia'][k][cand_order]
            p['baseline_aia'][k] = p['baseline_aia'][k][q_order]
            p['targets'][k]['release'] = p['targets'][k]['release'].sample(frac=1, random_state=503)
            p['targets'][k]['challenge'] = p['targets'][k]['challenge'].iloc[q_order].reset_index(drop=True)
        p['targets'] = dict(reversed(list(p['targets'].items())))
        after = model.predict_all(**p)
        for name in before:
            for kind, ids in [('F_mia', ['record_id']), ('F_aia', ['target_k', 'challenge_row_id'])]:
                left = before[name][kind].sort_values(ids).reset_index(drop=True)
                right = after[name][kind].sort_values(ids).reset_index(drop=True)
                assert_frame_equal(left, right, check_exact=True)

    def test_mismatched_targets_or_nonpublic_arguments_rejected(self):
        p = problem(); p['baseline_aia'].pop(2)
        with self.assertRaises(ValueError):
            model.predict_all(**p)
        p = problem(); p['targets'][0]['hidden_truth'] = object()
        with self.assertRaises(ValueError):
            model.predict_all(**p)


if __name__ == '__main__':
    unittest.main()
