"""Composition regressions: joint MIA scoring, target order, and unchanged AIA."""
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

import common
from model import predict_aia_one, predict_mia_all


class CompositionTests(unittest.TestCase):
    def test_mia_preserves_joint_target_order_and_mask(self):
        upstream = ModuleType("M.v2.mia")
        upstream.raw_scores = Mock(side_effect=lambda **kw: kw["c"])
        upstream.adjust_scores = Mock(return_value=np.array([[.2, .8], [0., 0.]]))
        mask = np.array([False, True])
        with patch.dict(sys.modules, {"M.v2.mia": upstream}):
            keys, scores = predict_mia_all(targets={7: [3., 4.], 2: [1., 2.]},
                background="bg", candidates="b", private_mask=mask, strength=1.)
        self.assertEqual(keys, [2, 7])
        self.assertEqual(upstream.adjust_scores.call_count, 1)
        args, kwargs = upstream.adjust_scores.call_args
        np.testing.assert_array_equal(args[0], [[1., 3.], [2., 4.]])
        self.assertIs(kwargs["private_mask"], mask)
        self.assertEqual(kwargs["strength"], 1.)
        np.testing.assert_array_equal(scores, [[.2, .8], [0., 0.]])

    def test_aia_uses_combined_aia_only_and_preserves_abstention(self):
        attack = Mock()
        attack.predict.return_value = ({"combined": {"mia": object(), "aia": [11., 2., 0.]},
                                        "other": {"aia": [8., 8., 8.]}}, "bootstrap")
        values, route = predict_aia_one(attack, c="c", background="bg", candidates="b",
                                       challenge=[3, 1, 2], cfg={})
        np.testing.assert_array_equal(values, [11., 2., 0.])
        self.assertEqual(route, "bootstrap")
        attack.predict.assert_called_once_with(c="c", background="bg", candidates="b",
                                               challenge=[3, 1, 2], cfg={})

    def test_aia_rejects_bad_shape_or_invalid_time(self):
        for values in ([1.], [[1., 2.]], [1., np.nan], [1., np.inf], [1., -1.]):
            with self.subTest(values=values):
                attack = Mock()
                attack.predict.return_value = ({"combined": {"aia": values}}, "jitter")
                with self.assertRaises(ValueError):
                    predict_aia_one(attack, c=None, background=None, candidates=None,
                                    challenge=[0, 1], cfg={})

    def test_live_token_digest_is_prohibited(self):
        with self.assertRaises(ValueError):
            common.sha("token.txt")


if __name__ == "__main__":
    unittest.main()
