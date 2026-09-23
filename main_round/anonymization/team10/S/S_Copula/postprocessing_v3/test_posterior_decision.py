import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from posterior_decision import expected_contribution, decide, Z95


class DecisionTests(unittest.TestCase):
    def test_matches_piecewise_numerical_integral(self):
        for mean,sd,se_b,se_c in [(0,.2,.1,.15),(.4,.1,.2,.1),(-.2,.4,.2,.22)]:
            for candidate in [-.7,0,.15,.8]:
                hb,hc=Z95*se_b,Z95*se_c
                sig_c=abs(candidate)>hc
                def integrand(value):
                    if (abs(value)>hb)!=sig_c or (sig_c and np.sign(value)!=np.sign(candidate)):
                        return 0.
                    overlap=max(0,min(value+hb,candidate+hc)-max(value-hb,candidate-hc))
                    return .25*(1/hb+1/hc)*overlap*norm.pdf(value,loc=mean,scale=sd)
                limits=sorted(set([mean-12*sd,mean+12*sd,-hb,hb,
                    candidate-hb-hc,candidate-abs(hb-hc),candidate+abs(hb-hc),candidate+hb+hc]))
                actual=sum(quad(integrand,a,b,epsabs=1e-12)[0] for a,b in zip(limits[:-1],limits[1:]))
                self.assertAlmostEqual(float(expected_contribution(candidate,mean,sd,se_b,se_c)),actual,places=9)

    def test_decision_weakly_improves_declared_objective_and_is_symmetric(self):
        mean=np.array([0,.19,.6]); covariance=np.diag([.04,.02,.05]); se=np.array([.1,.12,.15])
        answer,details=decide(mean,covariance,se,se)
        reverse,_=decide(-mean,covariance,se,se)
        np.testing.assert_allclose(answer,-reverse,atol=1e-7)
        self.assertTrue(np.all(np.asarray(details['expected_contributions'])>=np.asarray(details['posterior_mean_contributions'])))
        np.testing.assert_allclose(mean,[0,.19,.6])

    def test_rejects_invalid_uncertainty(self):
        with self.assertRaises(ValueError):
            decide(np.array([.2]),np.array([[-1.]]),np.array([.1]),np.array([.1]))
        with self.assertRaises(ValueError):
            expected_contribution(.2,.2,0.,.1,.1)


if __name__=='__main__': unittest.main()
