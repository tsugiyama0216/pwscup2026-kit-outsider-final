import unittest
import runtime
import numpy as np
import pandas as pd
from model import rank, mia_variants, apply_private_mask, time_evidence, aia_variants, QI


class ModelTests(unittest.TestCase):
    def test_ties_and_monotone_mia(self):
        x = np.array([.8, .2, .8, .5])
        y = mia_variants(x, x[::-1])["T"]
        self.assertEqual(y[0], y[2])
        self.assertTrue(y[0] > y[3] > y[1])

    def test_private_mask_cannot_change_positive_order(self):
        x = np.array([.1, .9, .4])
        y = apply_private_mask(x, np.array([False, True, False]))
        np.testing.assert_array_equal(y, [.1, 0., .4])
        np.testing.assert_array_equal(x, [.1, .9, .4])
        with self.assertRaises(ValueError):
            apply_private_mask(x, [0, 1, 0])

    def test_aia_gate_only_activates_strong_excess_and_nearby_queries(self):
        t = np.array([11., 11., 11., 2.])
        ev = (np.array([3., 3., 3., 3.]), np.array([.5, .1, .5, .5]),
              np.array([.1, .1, .9, .1]), np.array([3., 3., 3., 3.]), np.zeros(4, bool))
        np.testing.assert_array_equal(aia_variants(t, ev)["activate_040"], [3., 11., 11., 2.])
        np.testing.assert_array_equal(aia_variants(t, ev)["T"], t)

    def test_evidence_is_permutation_equivariant_and_ignores_non_qi_query_fields(self):
        rng = np.random.default_rng(34)
        b = pd.DataFrame({name: rng.normal(size=24) for name in QI})
        b["age"] += 50
        b["sex"] = np.arange(24)%2
        b["prefecture"] = ["A", "B"]*12
        b["time"] = np.linspace(.2, 10, 24)
        c = b.iloc[:20].copy()
        q = b.iloc[:4][QI+["sex", "prefecture"]].copy()
        original = time_evidence(c, b, q)
        q["record_id"] = [-999, 10000, 77, 88]
        q["time"] = [123, 456, 789, 101]
        q["label"] = [0, 1, 1, 0]
        perm = np.array([2, 0, 3, 1])
        changed = time_evidence(c.iloc[::-1], b.iloc[::-1], q.iloc[perm])
        for first, second in zip(original, changed):
            np.testing.assert_allclose(first[perm], second, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
