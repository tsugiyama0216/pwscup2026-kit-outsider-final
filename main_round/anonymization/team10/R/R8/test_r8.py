import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from test_fixture import artificial
from model import uncertainty_scale,fit_information,features,FEATURES,load_base,majority_copula,radii,tail_correction
from outcomes import calibrate_outcomes,cox_moments

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(710),artificial(711)
        for c in (cls.b,cls.c): c['age']=c.age.clip(40,74)
        cls.cfg={'schema':{'ranges':{n:[.00001,1e6] for n in ['BMI','SBP','TG','HDL','ALT','FPG']}},'population':{'age_range':[40,74]}}
        cols=['age','BMI','SBP','TG','HDL','ALT','FPG']
        cls.detector={'mu':cls.b[cols].mean().tolist(),'sigma':np.cov(cls.b[cols].to_numpy(),rowvar=False).tolist(),'band':{'lo':2.5}}

    def test_zero_scale_identity(self):
        out,_=uncertainty_scale(self.b,self.c,self.cfg,0)
        pd.testing.assert_frame_equal(out,self.c,check_exact=True)

    def test_scale_preserves_all_predictor_ranks(self):
        out,_=uncertainty_scale(self.b,self.c,self.cfg,.5)
        np.testing.assert_allclose(spearmanr(features(out)).statistic,spearmanr(features(self.c)).statistic,atol=1e-14)
        cols=['age','sex','smoking','prefecture','time','onset','death']
        pd.testing.assert_frame_equal(out[cols],self.c[cols],check_exact=True)

    def test_scaling_matches_analytical_standard_errors(self):
        out,audit=uncertainty_scale(self.b,self.c,self.cfg,.5)
        _,before=fit_information(self.c); _,after=fit_information(out)
        for name,data in audit['factors'].items():
            feature={'TG':'log_TG','ALT':'log_ALT'}.get(name,name); j=FEATURES.index(feature)
            self.assertAlmostEqual(np.sqrt(after[j,j]),np.sqrt(before[j,j])/data['applied'],places=6)

    def test_input_immutability(self):
        b,c=self.b.copy(deep=True),self.c.copy(deep=True)
        uncertainty_scale(b,c,self.cfg,.5)
        pd.testing.assert_frame_equal(b,self.b); pd.testing.assert_frame_equal(c,self.c)

    def test_frozen_base_guard(self):
        with patch('model.digest',return_value='modified'),self.assertRaises(RuntimeError): load_base()

    def test_efron_matches_lifelines(self):
        beta,var=fit_information(self.b); x=features(self.b).to_numpy(); x-=x.mean(axis=0)
        score,info=cox_moments(x,np.exp(x@beta),self.b.time.to_numpy(),self.b.onset.to_numpy())
        self.assertLess(float(score@var@score),1e-5)
        np.testing.assert_allclose(np.linalg.inv(info),var,rtol=1e-5,atol=1e-8)

    def test_outcome_constraints_and_determinism(self):
        mask=radii(self.c,self.detector)>=2.5
        a,aa=calibrate_outcomes(self.b,self.c,mask,2,True)
        d,da=calibrate_outcomes(self.b,self.c,mask,2,True)
        pd.testing.assert_frame_equal(a,d,check_exact=True)
        self.assertEqual(aa,da)
        pd.testing.assert_series_equal(a.groupby(['time','onset','death']).size(),self.c.groupby(['time','onset','death']).size())
        self.assertTrue(aa['rare_onset_count_preserved'])
        pd.testing.assert_frame_equal(a.drop(columns=['time','onset']),self.c.drop(columns=['time','onset']))

    def test_majority_keeps_tail(self):
        mask=radii(self.c,self.detector)>=2.5
        out,_=majority_copula(self.b,self.c,self.detector,.5)
        pd.testing.assert_frame_equal(out.loc[mask],self.c.loc[mask],check_exact=True)

    def test_tail_keeps_majority(self):
        mask=radii(self.c,self.detector)<2.5
        out,_=tail_correction(self.b,self.c,self.cfg,self.detector,.25)
        pd.testing.assert_frame_equal(out.loc[mask],self.c.loc[mask],check_exact=True)

    def test_bad_strength_rejected(self):
        with self.assertRaises(ValueError): uncertainty_scale(self.b,self.c,self.cfg,2)

    def test_selective_scale_preserves_other_features(self):
        out,_=uncertainty_scale(self.b,self.c,self.cfg,.5,'weak')
        pd.testing.assert_frame_equal(out.drop(columns=['HDL','ALT']),self.c.drop(columns=['HDL','ALT']),check_exact=True)

    def test_rare_event_target_and_global_event_distribution(self):
        mask=radii(self.c,self.detector)>=2.5
        target=int(self.c.loc[mask,'onset'].sum())+1
        out,audit=calibrate_outcomes(self.b,self.c,mask,5,True,rare_target=target)
        self.assertEqual(int(out.loc[mask,'onset'].sum()),target)
        pd.testing.assert_series_equal(out.groupby(['time','onset','death']).size(),self.c.groupby(['time','onset','death']).size())
        death=self.c.death==1
        pd.testing.assert_frame_equal(out.loc[death],self.c.loc[death])

    def test_tail_radius_correction_keeps_majority_and_ranges(self):
        mask=radii(self.c,self.detector)<2.5
        out,_=tail_correction(self.b,self.c,self.cfg,self.detector,.25,.5)
        pd.testing.assert_frame_equal(out.loc[mask],self.c.loc[mask],check_exact=True)
        for name,(lo,hi) in self.cfg['schema']['ranges'].items():
            self.assertTrue(out[name].between(lo,hi).all())

    def test_empirical_tail_marginals(self):
        from tail_copula import transform,CONT,detect
        mb,mc=detect(self.b,self.detector),detect(self.c,self.detector)
        out,_=transform(self.b,self.c,self.detector,self.cfg,1.,0.,True)
        for name in CONT:
            expected=np.quantile(self.b.loc[mb,name],(np.arange(mc.sum())+.5)/mc.sum())
            if name=='age': expected=np.rint(expected)
            np.testing.assert_allclose(np.sort(out.loc[mc,name]),expected,rtol=1e-12,atol=1e-12)

    def test_active_scaling_does_not_touch_inactive_rows(self):
        mask=radii(self.c,self.detector)<2.5
        out,_=uncertainty_scale(self.b,self.c,self.cfg,.5,active_mask=mask,range_strategy='clip')
        pd.testing.assert_frame_equal(out.loc[~mask],self.c.loc[~mask],check_exact=True)

    def test_cubic_scaling_is_monotone_bounded_and_deterministic(self):
        out,a=uncertainty_scale(self.b,self.c,self.cfg,1.,range_strategy='clip',shape='cubic')
        again,aa=uncertainty_scale(self.b,self.c,self.cfg,1.,range_strategy='clip',shape='cubic')
        pd.testing.assert_frame_equal(out,again,check_exact=True)
        self.assertEqual(a,aa)
        for name,(lo,hi) in self.cfg['schema']['ranges'].items():
            self.assertTrue(out[name].between(lo,hi).all())
            order=np.argsort(self.c[name].to_numpy(),kind='stable')
            self.assertTrue((np.diff(out[name].to_numpy()[order])>=0).all())

    def test_release_gate_rejects_roundoff_and_regression(self):
        from generate import release_gate
        baseline={'U_gen':.97,'U_spec':.96,'U_rare':.97,'U_valid':1.}
        with self.assertRaises(RuntimeError):
            release_gate({'utility':.96+1e-16,'facets':baseline},baseline)
        with self.assertRaises(RuntimeError):
            release_gate({'utility':.965,'facets':{**baseline,'U_gen':.965,'U_spec':.98}},baseline)

if __name__=='__main__': unittest.main()
