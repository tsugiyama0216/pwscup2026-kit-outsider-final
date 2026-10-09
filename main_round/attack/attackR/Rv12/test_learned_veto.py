"""Learned veto checks independent of any scoring panel or hidden truth."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rv12_learned_test', HERE / 'learned_veto.py')
veto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(veto)


class LearnedDecisionTests(unittest.TestCase):
    def evidence(self):
        old = np.array([10., 5., 4., 11., 3., 2.])
        # Row0 threshold .6; row1 threshold1.2; row2 protected exact link;
        # row3 already abstains; row4 tie at .75; row5 always supported.
        ms = np.array([[.3, .1], [.6, .1], [.3, .1], [.1, .05], [.375, .1], [.9, .8]])
        cs = np.full((6, 2), .5)
        exact = np.array([False, False, True, False, False, False])
        return old, ms, cs, exact

    def test_suppression_monotonic_penalty_exact_and_old_times(self):
        old, ms, cs, exact = self.evidence()
        before = [x.copy() for x in (old, ms, cs, exact)]
        out = veto.decisions(old, old.copy(), ms, cs, exact, 'independent')
        self.assertEqual(tuple(out), ('learned_075', 'learned_100', 'learned_150'))
        np.testing.assert_array_equal(out['learned_075'][0], [11., 5., 4., 11., 11., 2.])
        np.testing.assert_array_equal(out['learned_100'][0], [11., 5., 4., 11., 11., 2.])
        np.testing.assert_array_equal(out['learned_150'][0], [11., 11., 4., 11., 11., 2.])
        for name, (answer, diagnostic) in out.items():
            self.assertEqual(diagnostic['suppressed'], int((answer != old).sum()))
            self.assertTrue(diagnostic['eligible'])
            self.assertTrue(diagnostic['baseline_equal'])
            self.assertTrue(((answer == old) | (answer == 11.)).all())
        for old_array, new_array in zip(before, (old, ms, cs, exact)):
            np.testing.assert_array_equal(old_array, new_array)

    def test_entire_target_baseline_parity_is_required(self):
        old, ms, cs, exact = self.evidence()
        changed = old.copy(); changed[5] += 1e-10
        for answer, diagnostic in veto.decisions(old, changed, ms, cs, exact, 'copy').values():
            np.testing.assert_array_equal(answer, old)
            self.assertFalse(diagnostic['eligible'])
            self.assertFalse(diagnostic['baseline_equal'])

    def test_specialized_and_unknown_routes_are_untouched(self):
        old, ms, cs, exact = self.evidence()
        for route in ('copula', 'bootstrap', 'jitter', 'strong_jitter', 'rounded', 'unknown'):
            for answer, diagnostic in veto.decisions(old, old, ms, cs, exact, route).values():
                np.testing.assert_array_equal(answer, old)
                self.assertFalse(diagnostic['eligible'])

    def test_rows_equivariant_under_permutation(self):
        old, ms, cs, exact = self.evidence()
        ix = np.array([4, 1, 0, 5, 2, 3])
        first = veto.decisions(old, old, ms, cs, exact, 'copy')
        second = veto.decisions(old[ix], old[ix], ms[ix], cs[ix], exact[ix], 'copy')
        for name in first:
            np.testing.assert_array_equal(first[name][0][ix], second[name][0])

    def test_malformed_or_nonfinite_probabilities_rejected(self):
        old, ms, cs, exact = self.evidence()
        cases = [dict(ms=np.full_like(ms, np.nan)), dict(ms=np.full_like(ms, 1.1)),
                 dict(cs=np.full_like(cs, -.1)), dict(cs=np.zeros((6, 1))),
                 dict(exact=np.zeros(2, bool)), dict(old=np.ones((6, 1)))]
        for bad in cases:
            args = dict(old=old, baseline=old, member_success=ms, control_success=cs, exact=exact, route='copy')
            args.update({{'ms': 'member_success', 'cs': 'control_success'}.get(k, k): v for k, v in bad.items()})
            with self.subTest(bad=list(bad)), self.assertRaises(ValueError):
                veto.decisions(**args)

    def test_frozen_dependency_gate_rejects_hash_mismatch(self):
        with patch.object(veto, 'digest', return_value='0' * 64):
            with self.assertRaisesRegex(ValueError, 'dependency differs'):
                veto.verify_dependencies()


class PublicInferenceTests(unittest.TestCase):
    @staticmethod
    def tables():
        base = {name: [1., 2., 3.] for name in veto.QI}
        release = pd.DataFrame({**base, 'time': [5., 10., 2.]})
        background = release.copy()
        challenge = pd.DataFrame({name: [1., 9.] for name in veto.QI})
        return release, background, challenge

    def test_unsupported_route_skips_feature_extraction(self):
        release, background, challenge = self.tables()
        engine = veto.LearnedVeto.__new__(veto.LearnedVeto)
        class Router:
            def predict(self, x):
                return ['copula']
        engine.router = Router()
        engine.profile = lambda *args: np.ones(3)
        engine.extract = lambda *args: self.fail('specialized route should not extract')
        outputs = engine.predict(release, background, challenge, np.array([10., 5.]))
        for answer, diagnostic in outputs.values():
            np.testing.assert_array_equal(answer, [10., 5.])
            self.assertEqual(diagnostic['exact_links'], 1)
            self.assertFalse(diagnostic['eligible'])

    def test_original_half_penalty_reconstructed_and_exact_match_kept(self):
        release, background, challenge = self.tables()
        engine = veto.LearnedVeto.__new__(veto.LearnedVeto)
        class Router:
            def predict(self, x):
                return ['copy']
        class Member:
            def predict_proba(self, x):
                return np.tile([.5, .5], (len(x), 1))
        class Hit:
            classes_ = np.array([0, 1, 2, 3])
            def predict_proba(self, x):
                # Member hit .6, control hit1.0: selected at .5, rejected at .75.
                return np.tile([0., .5, .2, .3], (len(x), 1))
        engine.router, engine.model = Router(), {'aia': Member(), 'hit': Hit()}
        engine.profile = lambda *args: np.ones(3)
        engine.extract = lambda *args: (np.ones((2, 1)), np.array([[5., 10.], [5., 10.]]), None, None)
        engine.proposal_features = lambda x, proposals: np.ones((len(x) * proposals.shape[1], 1))
        engine.condition = lambda q, p: p
        outputs = engine.predict(release, background, challenge, np.array([5., 5.]))
        for answer, diagnostic in outputs.values():
            np.testing.assert_array_equal(answer, [5., 11.])
            self.assertTrue(diagnostic['baseline_equal'])
            self.assertEqual(diagnostic['suppressed'], 1)
            self.assertEqual(diagnostic['exact_links'], 1)

    def test_invalid_horizon_rejected_before_router(self):
        release, background, challenge = self.tables()
        release.loc[0, 'time'] = 10.1
        engine = veto.LearnedVeto.__new__(veto.LearnedVeto)
        with self.assertRaisesRegex(ValueError, 'time range'):
            engine.predict(release, background, challenge, np.array([5., 5.]))


if __name__ == '__main__':
    unittest.main()
