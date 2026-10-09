import unittest
import numpy as np
import pandas as pd
import model
import runtime as rt

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(19);self.settings=rt.read(rt.HERE/'protocol.json')
        self.bg=pd.DataFrame({c:rng.uniform(20,80,100) for c in model.FULL})
        self.bg['sex']=rng.integers(0,2,100);self.bg['prefecture']='P'
        self.bg['time']=rng.uniform(0,10,100)
        self.c=self.bg.iloc[:30].copy();self.b=self.bg.iloc[35:45].copy();self.b['record_id']=np.arange(10)
    def test_window_boundary(self):
        np.testing.assert_allclose(model.window_mass(np.array([2.,3.]),np.array([.5,.5]),np.array([2.5,4.])),[1,0])
    def test_invalid_baseline_rejected(self):
        cache=model.prepare(self.b,self.bg)
        with self.assertRaises(ValueError):model.membership(self.c,self.b,cache,np.full(10,np.nan),np.zeros(10,bool),self.settings)
    def test_known_negative_and_bounds(self):
        cache=model.prepare(self.b,self.bg);private=np.arange(10)<3
        ans,_=model.membership(self.c,self.b,cache,np.linspace(.1,.9,10),private,self.settings)
        self.assertTrue((ans[private]==0).all());self.assertTrue(((ans>=0)&(ans<=1)).all())
    def test_reference_permutation_invariance(self):
        old=np.linspace(.1,.9,10);mask=np.zeros(10,bool)
        a,_=model.membership(self.c,self.b,model.prepare(self.b,self.bg),old,mask,self.settings)
        b,_=model.membership(self.c.iloc[::-1],self.b,model.prepare(self.b,self.bg.iloc[::-1]),old,mask,self.settings)
        np.testing.assert_array_equal(a,b)
    def test_no_contrast_keeps_aia(self):
        q=self.bg.iloc[:8][model.QI+['sex','prefecture']];old=np.full(8,11.)
        a,_=model.attribute(self.bg,self.bg,q,old,self.settings)
        np.testing.assert_array_equal(a,old)
    def test_unique_release_link_recovers_time(self):
        c=self.c.copy();c.loc[c.index[0],model.FULL]=1.;c.loc[c.index[0],'time']=2.75
        q=c.iloc[:1][model.QI+['sex','prefecture']]
        a,d=model.attribute(c,self.bg,q,np.array([11.]),self.settings)
        self.assertEqual(a[0],2.75);self.assertEqual(d['unique_links'],1)
    def test_supported_time_preserved(self):
        q=self.c.iloc[:3][model.QI+['sex','prefecture']];old=self.c.time.iloc[:3].to_numpy()
        a,_=model.attribute(self.c,self.bg,q,old,self.settings)
        np.testing.assert_array_equal(a,old)
if __name__=='__main__':unittest.main()
