"""Analytical interpolation, boundary protection, and reproduction tests."""
import copy
import json
import unittest
import numpy as np
import pandas as pd
from model import fit, generate, rare_mask, LABS, CONT


class CopulaTests(unittest.TestCase):
    def setUp(self):
        # Radius depends only on age, isolating known lab interpolation results.
        self.frame = pd.DataFrame({name:[0.,1.,4.,9.,20.,30.] for name in LABS})
        self.frame['age'] = [0.,0.,0.,0.,10.,10.]
        self.frame['onset'] = [0,1,0,1,1,0]
        self.detector = dict(mu=[0.]*7, sigma=np.diag([1.,0,0,0,0,0,0]).tolist(), band={'lo':5.})

    def test_two_knots_have_known_linear_values(self):
        model = fit(self.frame,self.detector,2)
        out = generate(model)
        np.testing.assert_array_equal(out['BMI'],[0.,3.,6.,9.,20.,30.])
        pd.testing.assert_frame_equal(out.drop(columns=LABS),self.frame.drop(columns=LABS))

    def test_full_knots_and_json_roundtrip_are_lossless(self):
        model = fit(self.frame,self.detector,4)
        out = generate(json.loads(json.dumps(model)))
        pd.testing.assert_frame_equal(out,self.frame,check_exact=True)

    def test_boundary_exception_prevents_false_rare_row(self):
        # Majorities have BMI [0,0,0,9] with age [4,4,4,0]. Interpolation
        # changes the third radius from 4 to sqrt(4^2+6^2)>5.
        frame = self.frame.copy()
        frame['age'] = [4.,4.,4.,0.,10.,10.]
        frame['BMI'] = [0.,0.,0.,9.,20.,30.]
        # The row at BMI=9 must still be majority: use a high threshold and
        # age near that threshold on the interior rank instead.
        frame['age'] = [9.5,9.5,9.5,0.,20.,20.]
        detector = dict(mu=[0.]*7,sigma=np.diag([1.,1.,0,0,0,0,0]).tolist(),band={'lo':10.})
        plain = generate(fit(frame,detector,2,preserve_boundary=False))
        self.assertFalse(np.array_equal(rare_mask(frame,detector),rare_mask(plain,detector)))
        protected = fit(frame,detector,2)
        out = generate(protected)
        np.testing.assert_array_equal(rare_mask(frame,detector),rare_mask(out,detector))
        ids = protected['boundary_exceptions']['indices']
        self.assertGreater(len(ids),0)
        pd.testing.assert_frame_equal(out.loc[ids],frame.loc[ids],check_exact=True)

    def test_bad_ranks_are_rejected(self):
        model = fit(self.frame,self.detector,2)
        model['groups']['majority']['labs']['BMI']['ranks'] = [0,0,2,3]
        with self.assertRaises(ValueError):
            generate(model)


if __name__ == '__main__':
    unittest.main()
