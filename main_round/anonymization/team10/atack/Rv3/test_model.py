import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT']:os.environ[key]='1'
import unittest
import numpy as np
import pandas as pd
from model import rank,distance,local,candidates_all

class PublicModelTests(unittest.TestCase):
    def test_rank_preserves_ties_and_rejects_invalid(self):
        a=rank([2,1,2]);self.assertEqual(a[0],a[2]);self.assertGreater(a[0],a[1])
        with self.assertRaises(ValueError):rank([1,np.nan])
    def test_mode_uses_half_year_window(self):
        p=local(np.array([[0.,1.,2.]]),np.array([2.,2.1,9.]),3)
        from model import GRID
        self.assertTrue(abs(GRID[p.argmax()]-2.)<=.5)
    def test_permutation_equivariance_and_training_order(self):
        rng=np.random.default_rng(311)
        def sample(n):
            return pd.DataFrame(dict(age=rng.integers(20,80,n),BMI=rng.normal(24,3,n),SBP=rng.normal(120,15,n),TG=rng.uniform(20,150,n),HDL=rng.uniform(25,80,n),ALT=rng.uniform(3,60,n),FPG=rng.normal(100,15,n),time=rng.uniform(0,10,n),sex=rng.integers(0,2,n),prefecture=rng.choice(['A','B','C'],n),smoking=rng.integers(0,2,n),onset=rng.integers(0,2,n),death=rng.integers(0,2,n)))
        c,b,q,z=sample(65),sample(130),sample(10),sample(12)
        a=candidates_all(c=c,background=b,candidates=q,challenge=z,cfg={})
        d=candidates_all(c=c.iloc[::-1],background=b.iloc[::-1],candidates=q.iloc[::-1],challenge=z.iloc[::-1],cfg={})
        for part,options in a.items():
            for name,v in options.items():
                np.testing.assert_array_equal(v,d[part][name][::-1])
                self.assertTrue(np.isfinite(v).all());self.assertTrue((v>=0).all())
                if part=='mia':self.assertTrue((v<=1).all())
        from selected import predict_selected
        chosen={'mia':'full5','aia':'forest_contrast5_1.0'}
        first=predict_selected(chosen,c=c,background=b,candidates=q,challenge=z,cfg={})
        reused=predict_selected(chosen,c=c,background=b,candidates=q.iloc[::-1],challenge=z.iloc[::-1],cfg={})
        changed=b.copy();changed['time']=np.clip(changed.time+.3,0,10)
        predict_selected(chosen,c=c,background=changed,candidates=q,challenge=z,cfg={})
        restored=predict_selected(chosen,c=c,background=b,candidates=q,challenge=z,cfg={})
        for part in ['mia','aia']:
            np.testing.assert_array_equal(first[part],a[part][chosen[part]])
            np.testing.assert_array_equal(reused[part][::-1],first[part])
            np.testing.assert_array_equal(restored[part],first[part])
if __name__=='__main__':unittest.main()
