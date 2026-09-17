"""Artificial-data copula invariants and release safeguards."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import copula
import generate

def artificial(seed):
    rng=np.random.default_rng(seed)
    n=160
    z=rng.normal(size=(n,7))
    x=z*np.array([8,3,15,25,8,7,12])+np.array([55,24,125,100,50,25,95])
    c=pd.DataFrame(x,columns=copula.CONT)
    c['age']=np.clip(np.rint(c.age),40,74).astype(int)
    c['sex']=rng.integers(0,2,n)
    c['smoking']=rng.integers(0,2,n)
    c['prefecture']='東京都'
    c['time']=10.
    c['onset']=0
    c['death']=0
    return c[copula.COLUMNS]

class CopulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(1),artificial(2)
        cls.detector={'mu':cls.b[copula.CONT].mean().tolist(),
                      'sigma':np.cov(cls.b[copula.CONT].to_numpy(),rowvar=False).tolist(),
                      'band':{'lo':2.}}
        cls.cfg={}

    def test_zero_strength_identity(self):
        c,_=copula.transform(self.b,self.c,self.detector,self.cfg,0,0)
        pd.testing.assert_frame_equal(c,self.c,check_exact=True)

    def test_non_tail_and_fixed_columns_unchanged(self):
        c,_=copula.transform(self.b,self.c,self.detector,self.cfg,.5,0)
        mask=copula.detect(self.c,self.detector)
        pd.testing.assert_frame_equal(c.loc[~mask],self.c.loc[~mask],check_exact=True)
        cols=['sex','prefecture','smoking','time','onset','death']
        pd.testing.assert_frame_equal(c[cols],self.c[cols],check_exact=True)
        self.assertTrue(np.issubdtype(c.age.dtype,np.integer))

    def test_determinism_input_immutability_and_no_labels(self):
        b,c=self.b.copy(deep=True),self.c.copy(deep=True)
        out,_=copula.transform(b,c,self.detector,self.cfg,.5,0)
        out2,_=copula.transform(b.assign(record_id=99,is_rare=True),c,self.detector,self.cfg,.5,0)
        pd.testing.assert_frame_equal(out,out2,check_exact=True)
        pd.testing.assert_frame_equal(b,self.b,check_exact=True)
        pd.testing.assert_frame_equal(c,self.c,check_exact=True)

    def test_convex_support_bounds(self):
        out,_=copula.transform(self.b,self.c,self.detector,self.cfg,1,0)
        mb,mc=copula.detect(self.b,self.detector),copula.detect(self.c,self.detector)
        for name in copula.CONT:
            self.assertTrue(out.loc[mc,name].between(self.b.loc[mb,name].min(),self.b.loc[mb,name].max()).all())

    def test_transport_changes_joint_dependence(self):
        c,audit=copula.transform(self.b,self.c,self.detector,self.cfg,.5,0)
        self.assertFalse(np.allclose(np.asarray(audit['copula_mapping']),np.eye(7)))
        self.assertFalse(c[copula.CONT].equals(self.c[copula.CONT]))

    def test_insufficient_support_rejected(self):
        detector=copy.deepcopy(self.detector)
        detector['band']['lo']=100
        with self.assertRaises(ValueError):
            copula.transform(self.b,self.c,detector,self.cfg,.5,0)

    def test_invalid_strengths_rejected(self):
        for strength in (-1,2,np.nan,np.inf,True):
            with self.subTest(value=strength),self.assertRaises(ValueError):
                copula.transform(self.b,self.c,self.detector,self.cfg,strength,0)

    def test_degenerate_covariance_rejected(self):
        with self.assertRaises(ValueError):
            copula.matrix_power(np.zeros((3,3)),.5)

    def test_frozen_base_hash_guard(self):
        with patch.object(copula,'digest',return_value='modified'),self.assertRaises(RuntimeError):
            copula.load_base()

    def test_release_gate(self):
        base={'U_gen':.93,'U_spec':.91,'U_rare':.86,'U_valid':1.}
        generate.release_gate({'utility':.92,'facets':{'U_valid':1.}},base)
        for score in ({'utility':.85,'facets':{'U_valid':1.}}, {'utility':.92,'facets':{'U_valid':.99}}):
            with self.assertRaises(RuntimeError):
                generate.release_gate(score,base)

if __name__=='__main__':
    unittest.main()
