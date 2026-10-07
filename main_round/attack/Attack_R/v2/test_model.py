"""Behavioral tests for rank ties, known negatives, and public geometry."""
import unittest
import numpy as np
import pandas as pd
from model import finish_mia, geometry_mia, geometry_aia, FEATURES


def data(n=40):
    rng = np.random.default_rng(9321)
    frame = pd.DataFrame({name: rng.uniform(1, 80, n) for name in FEATURES})
    frame["sex"] = np.arange(n) % 2
    frame["prefecture"] = np.where(np.arange(n) % 3, "A", "B")
    frame["time"] = rng.uniform(.1, 10., n)
    frame["onset"], frame["death"], frame["smoking"] = 0, 0, 0
    return frame


class ModelTests(unittest.TestCase):
    def test_private_mask_and_ties(self):
        raw = np.array([[1., 2.], [1., 2.], [3., 4.], [4., 1.]])
        mask = np.array([False, False, True, False])
        for strength in (0., .5, 1.):
            answer = finish_mia(raw, mask, strength)
            np.testing.assert_array_equal(answer[0], answer[1])
            self.assertTrue((answer[2] == 0).all())
            self.assertTrue(((answer >= 0) & (answer <= 1)).all())

    def test_candidate_and_target_permutation_equivariance(self):
        raw = np.random.default_rng(71).normal(size=(10, 4))
        mask = np.arange(10) % 3 == 0
        base = finish_mia(raw, mask, .5)
        np.testing.assert_array_equal(finish_mia(raw[::-1, ::-1], mask[::-1], .5), base[::-1, ::-1])

    def test_bad_scores_rejected(self):
        with self.assertRaises(ValueError):
            finish_mia(np.array([[np.nan]]), np.array([False]))
        with self.assertRaises(ValueError):
            finish_mia(np.ones((3, 2)), np.array([0, 1, 0]))

    def test_mia_does_not_depend_on_outcomes(self):
        bg = data()
        c, q = bg.iloc[:20].copy(), bg.iloc[20:25].copy()
        original = geometry_mia(c, bg, q)
        c["time"], c["onset"], c["death"] = 99., 1, 1
        np.testing.assert_array_equal(geometry_mia(c, bg, q), original)

    def test_geometry_is_invariant_to_release_order(self):
        bg = data()
        c, q = bg.iloc[:20].copy(), bg.iloc[20:25].copy()
        np.testing.assert_allclose(geometry_mia(c, bg, q), geometry_mia(c.iloc[::-1], bg.iloc[::-1], q), rtol=0, atol=1e-12)
        a = geometry_aia(c, bg, q, np.full(5, 11.))
        b = geometry_aia(c.iloc[::-1], bg.iloc[::-1], q, np.full(5, 11.))
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])

    def test_aia_query_ids_do_not_affect_predictions(self):
        bg = data()
        c, q = bg.iloc[:20].copy(), bg.iloc[20:25].copy()
        a = geometry_aia(c, bg, q, np.full(5, 11.))
        q["challenge_row_id"] = np.arange(5)*7381
        b = geometry_aia(c, bg, q, np.full(5, 11.))
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])

    def test_exact_qi_nearest_recovers_time(self):
        bg = data()
        c = bg.iloc[:20].copy()
        q = c.iloc[:3].copy()
        answer = geometry_aia(c, bg.iloc[20:], q, np.full(3, 11.))
        np.testing.assert_array_equal(answer["nearest"], q.time.to_numpy())
        for value in answer.values():
            self.assertTrue(np.isfinite(value).all())
            self.assertTrue((value >= 0).all())


if __name__ == "__main__":
    unittest.main()
