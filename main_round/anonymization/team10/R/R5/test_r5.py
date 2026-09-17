import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sequential import synthesize, ORDER

class SequentialTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(19)
        n = 400
        sex = np.repeat([0, 1], n//2)
        b = pd.DataFrame({'sex': sex, 'age': rng.integers(40, 75, n),
                          'smoking': rng.integers(0, 2, n)})
        for j, name in enumerate(['BMI', 'FPG', 'TG', 'HDL', 'SBP', 'ALT']):
            b[name] = 20 + j*10 + 50*sex + rng.uniform(0, 1, n)
        b['prefecture'] = np.where(sex == 0, '東京都', '岐阜県')
        b['record_id'] = np.arange(n)
        return b

    def run_synth(self, b):
        outcomes = pd.DataFrame({'time': np.full(len(b), 10.), 'onset': 0, 'death': 0})
        with patch('sequential.synthesize_outcomes', return_value=(outcomes, {})), patch('sequential.calibrate', side_effect=lambda b,c,cfg,strength: (c,{})):
            return synthesize(b, {'onset': {'horizon_years': 10}}, leaf_size=20)

    def test_conditioning_and_leaf_support(self):
        c, audit = self.run_synth(self.fixture())
        self.assertTrue((c.loc[c.sex == 0, 'BMI'] < 22).all())
        self.assertTrue((c.loc[c.sex == 1, 'BMI'] > 69).all())
        for i, step in enumerate(audit['sequence']):
            self.assertEqual(step['conditions'], ORDER[:i])
            self.assertGreaterEqual(step['minimum_leaf_support'], 20)

    def test_reproduction_schema_and_new_continuous_values(self):
        b = self.fixture()
        c, _ = self.run_synth(b)
        d, _ = self.run_synth(b.assign(record_id=-999))
        pd.testing.assert_frame_equal(c, d)
        self.assertEqual(c.shape, (len(b), 13))
        self.assertFalse(c.isna().any().any())
        self.assertNotIn('record_id', c)
        self.assertTrue(set(c.prefecture) <= set(b.prefecture))
        self.assertTrue((~c.BMI.isin(b.BMI)).all())
        self.assertTrue(np.issubdtype(c.age.dtype, np.integer))

    def test_reject_small_leaf(self):
        with self.assertRaises(ValueError):
            synthesize(self.fixture(), {}, leaf_size=1)

if __name__ == '__main__':
    unittest.main()
