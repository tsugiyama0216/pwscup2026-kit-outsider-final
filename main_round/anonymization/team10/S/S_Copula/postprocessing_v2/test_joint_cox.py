import copy
import unittest
import numpy as np
from joint_cox import combine_cox

def fixture(epsilon):
    public={'quantization':128,'clip_norm':1.,'public_integer_mean':[0]*10,
      'public_score_covariance':np.eye(10).tolist(),'information_inverse':np.eye(10).tolist(),
      'beta_standardized':[0]*9,'scale':[1]*9}
    return {'public_cox':public,'accounting':{'epsilon_exact':str(epsilon),'cox_epsilon_exact':str(epsilon)},
      'cox_mechanism':{'l1_sensitivity':256,'epsilon_allocated_exact':str(epsilon)},
      'cox':{'public_n':1049,'noisy_score_integer':[epsilon]*10}}

class TestJoint(unittest.TestCase):
    def test_composition_and_precision(self):
        models=[fixture(e) for e in (1,3,10)]
        r=combine_cox(models)
        self.assertEqual(r['parent_epsilon_exact'],'14')
        self.assertEqual(r['additional_epsilon_exact'],'0')
        self.assertAlmostEqual(sum(r['independent_parent_measurement_weights']),1)
        single=combine_cox(models[-1:])
        self.assertLess(r['combined_DP_noise_variance_of_score_mean'],single['combined_DP_noise_variance_of_score_mean'])
    def test_repeated_measurement_rejected(self):
        m=fixture(1)
        with self.assertRaises(ValueError):combine_cox([m,copy.deepcopy(m)])
    def test_different_public_map_rejected(self):
        a,b=fixture(1),fixture(3);b['public_cox']['clip_norm']=2
        with self.assertRaises(ValueError):combine_cox([a,b])

if __name__=='__main__':unittest.main()
