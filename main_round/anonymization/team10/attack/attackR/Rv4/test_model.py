import unittest
import numpy as np
import pandas as pd
from model import evidence,ranks
class NumericTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(174)
        self.bg=pd.DataFrame({c:rng.uniform(20,100,60) for c in ['age','BMI','SBP','TG','HDL','ALT','FPG']})
        self.bg['sex']=rng.integers(0,2,60);self.bg['prefecture']='P';self.bg['time']=rng.uniform(0,10,60)
        self.c=self.bg.iloc[:20].copy();self.c['age']+=.123
        self.b=self.c.iloc[:8].copy();self.q=self.c.iloc[:5].copy()
    def test_exact_signal_and_valid_window(self):
        m,t,ratio,support,full=evidence(self.c,self.bg,self.b,self.q)
        self.assertTrue(np.all(ratio==0));self.assertTrue(np.all(full>0))
        self.assertTrue(np.isfinite(m).all());self.assertTrue(np.all((support>0)&(support<=1)))
        self.assertTrue(np.isin(t,self.c.time).all())
    def test_release_and_background_order_invariant(self):
        a=evidence(self.c,self.bg,self.b,self.q)
        b=evidence(self.c.iloc[::-1],self.bg.iloc[::-1],self.b,self.q)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=0,atol=0)
    def test_monotone_calibration_preserves_ties_and_order(self):
        x=np.array([.2,.8,.8,.1]);y=ranks(x)
        np.testing.assert_array_equal(np.argsort(x,kind='stable'),np.argsort(y,kind='stable'))
        self.assertEqual(y[1],y[2])
if __name__=='__main__':unittest.main()
