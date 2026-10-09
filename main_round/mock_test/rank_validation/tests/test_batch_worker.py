"""Shared baseline reuse must preserve frozen ranking including tied scores."""
import unittest
import numpy as np

from rank_validation import adapters
from rank_validation.batch_worker import finish_shared_rv2


class SharedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.numeric = adapters._load("_rank_test_rv2_numeric", adapters.ATTACK / "Attack_R/v2/model.py")

    def test_frozen_double_rank_and_private_mask_with_ties(self):
        raw = np.array([0., 0., .2, .2, .2, .3, .9])
        mask = np.array([False, True, False, False, True, False, False])
        baseline = self.numeric.ranks(self.numeric.ranks(raw))
        got = finish_shared_rv2(baseline, mask, self.numeric)
        expected = self.numeric.finish_mia(self.numeric.ranks(raw)[:, None], mask)[:, 0]
        np.testing.assert_array_equal(got, expected)
        self.assertTrue(np.all(got[mask] == 0))
        self.assertTrue(np.all(baseline > 0))

    def test_nonbaseline_values_are_not_silently_used(self):
        with self.assertRaisesRegex(ValueError, "Shared Rv2 baseline differs"):
            finish_shared_rv2(np.array([.1, .2, .3]), np.zeros(3, dtype=bool), self.numeric)


if __name__ == "__main__":
    unittest.main()