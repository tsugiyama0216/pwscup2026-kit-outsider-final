"""Numerical edge cases and invariants, without loading fitted T assets."""
import unittest
import numpy as np
import pandas as pd
from model import ranks,fallback,conditional_evidence

class ModelTests(unittest.TestCase):
    def test_tied_scores_are_not_arbitrary_members(self):
        np.testing.assert_array_equal(ranks(np.zeros(9)),np.full(9,.5))
    def test_unknown_time_uses_expert_not_mean(self):
        a={'mia':np.array([.1,.9]),'aia':np.array([2.,9.])}
        b={'mia':np.array([.9,.1]),'aia':np.array([8.,3.])}
        np.testing.assert_array_equal(fallback(a,b,True)['aia'],b['aia'])
        np.testing.assert_array_equal(fallback(a,b,False)['aia'],a['aia'])
        np.testing.assert_array_equal(a['aia'],[2.,9.])
    def test_conditional_evidence_permutation(self):
        rng=np.random.default_rng(51)
        def frame(n):
            f=pd.DataFrame(rng.uniform(1,10,(n,7)),columns=['age','BMI','SBP','TG','HDL','ALT','FPG'])
            f['sex']=rng.integers(0,2,n);f['prefecture']=rng.choice(['a','b'],n)
            return f
        c,b,q=frame(40),frame(80),frame(10)
        a,_=conditional_evidence(c,b,q)
        d,_=conditional_evidence(c.iloc[::-1],b.iloc[::-1],q)
        np.testing.assert_array_equal(a,d)
        self.assertTrue(np.isfinite(a).all());self.assertTrue(((a>=0)&(a<=1)).all())
if __name__=='__main__':unittest.main()
