import unittest
from unittest.mock import patch
import numpy as np
from autograd import grad
from aggregate_optimizer import cdf_interp,efron_design,cox_moments,make_problem,refine
from model import load_base,features,fit_information,CONT
from outcomes import cox_moments as reference_moments
from test_fixture import artificial

class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(810),artificial(811)
        for c in (cls.b,cls.c): c['age']=c.age.clip(40,74)
        cls.cfg={'schema':{'ranges':{'BMI':[10,55],'SBP':[40,250],'TG':[1,3000],'HDL':[10,160],'ALT':[.1,700],'FPG':[40,400]}}}
        cls.detector={'mu':cls.b[CONT].mean().tolist(),'sigma':np.cov(cls.b[CONT].to_numpy(),rowvar=False).tolist(),'band':{'lo':2.5}}

    def test_interpolated_cdf_gradient(self):
        import autograd.numpy as anp
        knots=np.array([-2.,-1.,.5,3.]); levels=np.array([0.,.3,.7,1.])
        x=np.array([-3.,-1.5,0.,2.,4.])
        actual=grad(lambda v:anp.sum(cdf_interp(v,knots,levels)))(x)
        np.testing.assert_allclose(actual,[0.,.3,.4/1.5,.3/2.5,0.],atol=1e-12)

    def test_cox_moments_match_lifelines(self):
        beta,cov=fit_information(self.b); x=features(self.b).to_numpy(); x-=x.mean(axis=0)
        e=self.b.onset.to_numpy(); t=self.b.time.to_numpy()
        score,info=cox_moments(x,beta,e,efron_design(t,e))
        self.assertLess(float(score@cov@score),1e-5)
        np.testing.assert_allclose(np.linalg.inv(info),cov,rtol=1e-5,atol=1e-8)

    def test_tied_events_match_reference(self):
        x=features(self.b).to_numpy(); x=(x-x.mean(axis=0))/x.std(axis=0)
        beta=np.linspace(-.05,.05,9); t=np.round(self.b.time.to_numpy(),0); e=self.b.onset.to_numpy()
        actual=cox_moments(x,beta,e,efron_design(t,e))
        expected=reference_moments(x,np.exp(x@beta),t,e)
        for a,v in zip(actual,expected): np.testing.assert_allclose(a,v,rtol=1e-11,atol=1e-10)

    def test_joint_objective_gradient(self):
        obj,x,_,_,_,_,_=make_problem(self.b,self.c,self.cfg,self.detector,4.,4.,100.,100.,.02,1.)
        rng=np.random.default_rng(85); direction=rng.normal(size=len(x)); direction/=np.linalg.norm(direction)
        actual=float(grad(obj)(x)@direction); eps=1e-6
        expected=(obj(x+eps*direction)-obj(x-eps*direction))/(2*eps)
        self.assertAlmostEqual(actual,float(expected),places=5)

    def test_refinement_preserves_fixed_data_bounds_and_inputs(self):
        import pandas as pd
        b,c=self.b.copy(deep=True),self.c.copy(deep=True)
        out,a=refine(b,c,self.cfg,self.detector,2,4.,4.,100.,100.,.02,1.)
        pd.testing.assert_frame_equal(b,self.b);pd.testing.assert_frame_equal(c,self.c)
        fixed=['age','sex','prefecture','smoking','time','onset','death']
        pd.testing.assert_frame_equal(out[fixed],c[fixed],check_exact=True)
        for name,(lo,hi) in self.cfg['schema']['ranges'].items(): self.assertTrue(out[name].between(lo,hi).all())
        self.assertLess(sum(a['final_components']),sum(a['initial_components']))

    def test_frozen_base_guard(self):
        with patch('model.digest',return_value='changed'),self.assertRaises(RuntimeError): load_base()

    def test_category_counts_match_without_changing_other_columns(self):
        import pandas as pd
        from categorical import calibrate_prefecture
        out,a=calibrate_prefecture(self.b,self.c)
        pd.testing.assert_series_equal(out.prefecture.value_counts().sort_index(),self.b.prefecture.value_counts().sort_index())
        pd.testing.assert_frame_equal(out.drop(columns='prefecture'),self.c.drop(columns='prefecture'),check_exact=True)
        again,_=calibrate_prefecture(self.b,self.c)
        pd.testing.assert_frame_equal(out,again,check_exact=True)

    def test_tail_class_projection_and_unchanged_ages(self):
        import pandas as pd
        from aggregate_optimizer import restore_membership
        from model import radii
        modified=self.c.copy(deep=True)
        for name in CONT[1:]: modified[name]=(modified[name]+self.b[name].mean())/2
        out,count=restore_membership(self.c,modified,self.detector)
        self.assertGreater(count,0)
        np.testing.assert_array_equal(radii(out,self.detector)>=2.5,radii(self.c,self.detector)>=2.5)
        pd.testing.assert_series_equal(out.age,self.c.age,check_exact=True)

    def test_age_calibration_preserves_radius_and_outcomes(self):
        import pandas as pd
        from categorical import calibrate_age
        from model import radii
        out,_=calibrate_age(self.b,self.c,self.cfg,self.detector)
        np.testing.assert_allclose(radii(out,self.detector),radii(self.c,self.detector),rtol=1e-10,atol=1e-10)
        fixed=['sex','prefecture','smoking','time','onset','death']
        pd.testing.assert_frame_equal(out[fixed],self.c[fixed],check_exact=True)
        self.assertTrue((out.age==np.rint(out.age)).all())
        self.assertTrue(out.age.between(40,74).all())

    def test_release_rejects_roundoff_and_regression(self):
        from generate import release_gate
        baseline={'U_gen':.98,'U_spec':.97,'U_rare':.98,'U_valid':1.}
        with self.assertRaises(RuntimeError): release_gate({'utility':.97+1e-16,'facets':baseline},baseline)
        with self.assertRaises(RuntimeError): release_gate({'utility':.975,'facets':{**baseline,'U_gen':.975,'U_spec':.99}},baseline)

if __name__=='__main__': unittest.main()
