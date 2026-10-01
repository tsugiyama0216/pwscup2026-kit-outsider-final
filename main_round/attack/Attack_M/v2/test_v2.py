"""Run v2 component tests and composition checks using the pinned runtime."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import unittest

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class CompositionTests(unittest.TestCase):
    def test_target_alignment_and_mask_are_passed_without_ids(self):
        from unittest.mock import patch
        import numpy as np
        import pandas as pd
        from M.v2.model import predict_all
        public = pd.DataFrame({"time": [2., 4.]})
        query = pd.DataFrame({"age": [60.]})
        calls = []
        def raw_scores(**kwargs):
            calls.append(kwargs)
            return np.array([1., 2.])
        def adjust(raw, private_mask, strength):
            self.assertEqual(strength, .5)
            self.assertEqual(raw.shape, (2, 2))
            self.assertEqual(list(private_mask), [True, False])
            return np.array([[0., 0.], [.75, .75]])
        with patch("M.v2.model.mia.raw_scores", side_effect=raw_scores), \
             patch("M.v2.model.mia.adjust_scores", side_effect=adjust), \
             patch("M.v2.model.aia.predict", return_value=np.array([2.])):
            p = predict_all(targets={9:public, 3:public}, background=public,
                candidates=public, challenges={3:query, 9:query}, cfg={},
                mia_strength=.5, aia_variant="v1_reference", private_mask=[True, False])
        self.assertEqual(p["target_order"], [3, 9])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(set(call) == {"c", "background", "candidates"} for call in calls))

    def test_mismatched_target_sets_rejected(self):
        from M.v2.model import predict_all
        with self.assertRaises(ValueError):
            predict_all(targets={1:None}, background=None, candidates=None,
                challenges={2:None}, cfg={}, mia_strength=0., aia_variant="v1_reference")


if __name__ == "__main__":
    from attack_workspace import load_runtime, verify_hashes
    from M.v2.artifacts import check_v1_preserved
    verify_hashes()
    load_runtime()
    components = []
    for pattern in ("test_aia.py", "test_mia.py", "test_v2.py"):
        cases = unittest.defaultTestLoader.discover(
            str(Path(__file__).parent), pattern=pattern, top_level_dir=str(ROOT))
        if cases.countTestCases() == 0:
            raise ValueError(f"Missing required test module: {pattern}")
        components.append(cases)
    suite = unittest.TestSuite(components)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    verify_hashes()
    check_v1_preserved()
    sys.exit(0 if result.wasSuccessful() else 1)
