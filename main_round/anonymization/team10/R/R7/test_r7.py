import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_fixture import artificial
import model
from outcomes import cox_moments,calibrate_outcomes

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(710),artificial(711)
        for f in (cls.b,cls.c):
            f['age']=f.age.clip(40,74)
        cls.cfg={'schema':{'ranges':{n:[.00001,1e6] for n in ['BMI','SBP','TG','HDL','ALT','FPG']}},
                 'population':{'age_range':[40,74]}}
        cls.detector={'mu':cls.b[model.CONT].mean().tolist(),
                      'sigma':np.cov(cls.b[model.CONT].to_numpy(),rowvar=False).tolist()}

    def test_radial_zero_identity(self):
        out,_=model.radial(self.b,self.c,self.detector,0,self.cfg)
        pd.testing.assert_frame_equal(out,self.c,check_exact=True)

    def test_radial_bounds_and_fixed_columns(self):
        out,_=model.radial(self.b,self.c,self.detector,.5,self.cfg)
        self.assertTrue(out.age.between(40,74).all())
        fixed=['sex','smoking','prefecture','time','onset','death']
        pd.testing.assert_frame_equal(out[fixed],self.c[fixed],check_exact=True)

    def test_radial_input_immutability_and_determinism(self):
        b,c=self.b.copy(deep=True),self.c.copy(deep=True)
        a,_=model.radial(b,c,self.detector,.5,self.cfg)
        d,_=model.radial(b,c,self.detector,.5,self.cfg)
        pd.testing.assert_frame_equal(a,d,check_exact=True)
        pd.testing.assert_frame_equal(b,self.b,check_exact=True)
        pd.testing.assert_frame_equal(c,self.c,check_exact=True)

    def test_information_calibration_improves_effect_and_uncertainty(self):
        out,audit=model.information_calibration(self.b,self.c,1.,self.cfg)
        bb,vb=model.fit_information(self.b)
        bc,vc=model.fit_information(self.c)
        bn,vn=model.fit_information(out)
        se=np.sqrt(np.diag(vb))
        def error(beta,var):
            return np.linalg.norm((beta-bb)/se)+np.linalg.norm(np.log(np.sqrt(np.diag(var))/se))
        self.assertLess(error(bn,vn),error(bc,vc))
        self.assertTrue(audit['optimizer_success'])
        pd.testing.assert_frame_equal(out[['age','sex','smoking','time','onset','death']],self.c[['age','sex','smoking','time','onset','death']])

    def test_identifiers_not_used(self):
        out,_=model.radial(self.b,self.c,self.detector,.5,self.cfg)
        other,_=model.radial(self.b.assign(record_id=99,is_rare=True),self.c,self.detector,.5,self.cfg)
        pd.testing.assert_frame_equal(out,other,check_exact=True)

    def test_frozen_base_guard(self):
        with patch.object(model,'digest',return_value='changed'),self.assertRaises(RuntimeError):
            model.load_base()

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            model.radial(self.b,self.c,self.detector,2,self.cfg)
        with self.assertRaises(ValueError):
            model.information_calibration(self.b,self.c,0,self.cfg)

    def test_majority_copula_keeps_tail(self):
        detector=dict(self.detector,band={'lo':3.})
        out,_=model.majority_copula(self.b,self.c,detector,.5)
        delta=self.c[model.CONT].to_numpy()-np.asarray(detector['mu'])
        mask=np.sqrt(np.einsum('ij,jk,ik->i',delta,np.linalg.pinv(detector['sigma']),delta))>=3.
        pd.testing.assert_frame_equal(out.loc[mask],self.c.loc[mask],check_exact=True)

    def test_efron_score_at_fitted_coefficient(self):
        beta,variance=model.fit_information(self.b)
        x=model.features(self.b).to_numpy()
        x=x-x.mean(axis=0)
        score,info=cox_moments(x,np.exp(x@beta),self.b.time.to_numpy(),self.b.onset.to_numpy())
        self.assertLess(float(score@variance@score),1e-5)
        np.testing.assert_allclose(np.linalg.inv(info),variance,rtol=1e-5,atol=1e-8)

    def test_outcome_swaps_preserve_constraints_and_reduce_loss(self):
        mask=np.arange(len(self.c))%3==0
        out,audit=calibrate_outcomes(self.b,self.c,mask,max_steps=3)
        pd.testing.assert_frame_equal(out.drop(columns=['time','onset']),self.c.drop(columns=['time','onset']))
        before=self.c.groupby(['time','onset','death']).size().sort_index()
        after=out.groupby(['time','onset','death']).size().sort_index()
        pd.testing.assert_series_equal(before,after)
        self.assertEqual(int(out.loc[mask,'onset'].sum()),int(self.c.loc[mask,'onset'].sum()))
        self.assertLessEqual(audit['final_score_loss'],audit['initial_score_loss'])
        self.assertTrue(all(a>b for a,b in zip([audit['initial_score_loss']]+audit['loss_history'],audit['loss_history'])))

    def test_guarded_outcome_reproducibility(self):
        mask=np.arange(len(self.c))%3==0
        a,aa=calibrate_outcomes(self.b,self.c,mask,max_steps=2,significance_guard=True)
        d,da=calibrate_outcomes(self.b,self.c,mask,max_steps=2,significance_guard=True)
        pd.testing.assert_frame_equal(a,d,check_exact=True)
        self.assertEqual(aa,da)
        self.assertTrue(aa['rare_onset_count_preserved'])

    def test_no_outcome_steps_is_identity(self):
        out,_=calibrate_outcomes(self.b,self.c,np.zeros(len(self.c),dtype=bool),max_steps=0)
        pd.testing.assert_frame_equal(out,self.c,check_exact=True)

if __name__=='__main__':
    unittest.main()
