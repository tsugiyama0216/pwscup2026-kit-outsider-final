import unittest
import numpy as np
import pandas as pd
from autograd import grad
from refinement import LABS
from optimizer import make_problem,refine
from aggregate_optimizer import make_problem as original_problem
from test_fixture import artificial
from model import CONT,radii

class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b,cls.c=artificial(810),artificial(811)
        for c in (cls.b,cls.c): c['age']=c.age.clip(40,74)
        cls.cfg={'schema':{'ranges':{'BMI':[10,55],'SBP':[40,250],'TG':[1,3000],'HDL':[10,160],'ALT':[.1,700],'FPG':[40,400]}}}
        cls.detector={'mu':cls.b[CONT].mean().tolist(),'sigma':np.cov(cls.b[CONT].to_numpy(),rowvar=False).tolist(),'band':{'lo':2.5}}

    def arguments(self):
        return (self.b,self.c,self.cfg,self.detector,10.,4.,300.,100.,.02,.5,10000.)

    def test_half_phase_reproduces_original_objective_and_gradient(self):
        original,x,*_=original_problem(*self.arguments())
        updated,y,*_=make_problem(*self.arguments(),tail_phase=.5)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(original(x,True),updated(y,True),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(grad(original)(x),grad(updated)(y),atol=1e-12,rtol=1e-12)

    def test_shifted_phase_gradient_matches_finite_difference(self):
        objective,x,*_=make_problem(*self.arguments(),tail_phase=.75)
        direction=np.random.default_rng(56).normal(size=len(x)); direction/=np.linalg.norm(direction)
        eps=1e-6
        numeric=(objective(x+eps*direction)-objective(x-eps*direction))/(2*eps)
        self.assertAlmostEqual(float(grad(objective)(x)@direction),float(numeric),places=5)

    def test_constraints_and_input_preservation(self):
        b,c=self.b.copy(deep=True),self.c.copy(deep=True)
        out,_=refine(b,c,self.cfg,self.detector,2,10.,4.,300.,100.,.02,.5,10000.,True,tail_phase=.75)
        pd.testing.assert_frame_equal(b,self.b); pd.testing.assert_frame_equal(c,self.c)
        pd.testing.assert_frame_equal(out.drop(columns=LABS),c.drop(columns=LABS),check_exact=True)
        np.testing.assert_array_equal(radii(out,self.detector)>=2.5,radii(c,self.detector)>=2.5)
        for name,(lo,hi) in self.cfg['schema']['ranges'].items(): self.assertTrue(out[name].between(lo,hi).all())

    def test_invalid_phase_rejected(self):
        for phase in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError): make_problem(*self.arguments(),tail_phase=phase)

    def test_locked_tail_remains_exact_while_majority_can_move(self):
        out,audit=refine(self.b,self.c,self.cfg,self.detector,3,10.,4.,300.,100.,.02,.5,10000.,True,tail_phase=.75,locked_tail=True)
        rare=radii(self.c,self.detector)>=2.5
        pd.testing.assert_frame_equal(out.loc[rare],self.c.loc[rare],check_exact=True)
        self.assertGreater(float(np.abs(out.loc[~rare,LABS].to_numpy()-self.c.loc[~rare,LABS].to_numpy()).sum()),0.)
        self.assertEqual(audit['free_parameters'],int((~rare).sum()*6))

if __name__=='__main__': unittest.main()
