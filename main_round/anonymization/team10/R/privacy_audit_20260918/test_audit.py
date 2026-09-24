import unittest
import numpy as np
import pandas as pd
from audit import calibration_threshold, wilson, mia_metrics, aia_metrics, rare_mask, row_hashes, COLS


class AuditCalculations(unittest.TestCase):
    def test_low_fpr_order_statistic_and_strict_ties(self):
        x=np.arange(1000.)
        t=calibration_threshold(x)
        self.assertEqual(t,990.)
        self.assertLessEqual((x>t).mean(),.01)
        self.assertTrue(np.isinf(calibration_threshold(np.arange(50.))))
        self.assertEqual((np.ones(1000)>calibration_threshold(np.ones(1000))).sum(),0)

    def test_perfect_attack_and_heldout_fpr(self):
        r=mia_metrics(np.ones(10)*2,np.zeros(100),np.zeros(100))
        self.assertEqual(r['auc'],1.)
        self.assertEqual(r['tpr'],1.)
        self.assertEqual(r['fpr'],0.)
        self.assertGreater(r['fpr_wilson_95'][1],0.)

    def test_half_year_boundary_and_paired_baseline(self):
        r=aia_metrics([1,2,10],[1.5,2.6,10],[1,2,9])
        self.assertAlmostEqual(r['accuracy_half_year'],2/3)
        self.assertAlmostEqual(r['public_baseline_accuracy'],2/3)
        self.assertAlmostEqual(r['accuracy_gain_over_public'],0.)
        self.assertAlmostEqual(r['always_ten_accuracy'],1/3)

    def test_rare_mahalanobis_radius(self):
        schema={'rare_detector':{'columns':['a','b'],'mu':[0,0],'sigma':[[1,0],[0,4]],'band':{'lo':4}}}
        frame=pd.DataFrame({'a':[0,4,0,3],'b':[0,0,8,0]})
        self.assertEqual(rare_mask(frame,schema).tolist(),[False,True,True,False])

    def test_hash_does_not_depend_on_dataframe_index(self):
        a=pd.DataFrame({c:[1.,2.] for c in COLS})
        b=a.copy();b.index=[20,21]
        np.testing.assert_array_equal(row_hashes(a),row_hashes(b))
        b.loc[20,'time']=9
        self.assertNotEqual(row_hashes(a)[0],row_hashes(b)[0])

    def test_zero_observations_not_zero_upper_bound(self):
        lo,hi=wilson(0,71)
        self.assertAlmostEqual(lo,0.)
        self.assertGreater(hi,.05)

    def test_interval_mode_optimizes_mass_not_conditional_mean(self):
        from aia_mode import modal_time
        result=modal_time(np.array([[.2,.8],[1.,0.]]),np.array([0,40]))
        self.assertEqual(result[0],10.)
        self.assertLessEqual(abs(result[1]-.125),.5)


if __name__=='__main__':
    unittest.main()
