"""Check the optimized permutation statistic against official scalar scoring."""
import unittest
import numpy as np
from evaluate import batch_tpr, permutation_mia, score_mia
from pwscup2026.scoring.attack_score import _tpr_at_fpr

class ScoringTests(unittest.TestCase):
    def test_batch_matches_official_with_ties_and_rare_subsets(self):
        rng=np.random.default_rng(9821)
        for n in (24,73,1049):
            conf=rng.integers(0,12,n)/12
            order=np.argsort(-conf,kind='stable');sorted_conf=conf[order]
            ends=np.r_[np.flatnonzero(np.diff(sorted_conf)!=0),n-1]
            base=np.zeros(n,np.int8);base[:max(2,n//40)]=1
            batches=np.array([rng.permutation(base) for _ in range(80)])
            actual=batch_tpr(batches[:,order],ends,n-base.sum(),base.sum())
            expected=[_tpr_at_fpr(conf,y,.01)[0] for y in batches]
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
            rarebase=np.full(n,-1,np.int8);rarebase[:2]=1;rarebase[2:min(n,24)]=0
            batches=np.array([rng.permutation(rarebase) for _ in range(80)])
            actual=batch_tpr(batches[:,order],ends,min(n,24)-2,2)
            expected=[_tpr_at_fpr(conf[y>=0],y[y>=0],.01)[0] for y in batches]
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)

    def test_no_information_constant_scores(self):
        y=np.r_[np.ones(26,int),np.zeros(1023,int)];rare=np.zeros(1049,bool);rare[[0,1,*range(30,52)]]=True
        result=permutation_mia([np.full(1049,.5)]*3,[y]*3,[rare]*3,reps=80,low_fpr=.01,fold_k=12,batch_size=17)
        self.assertAlmostEqual(result['beta_raw'],.01)
        self.assertAlmostEqual(result['chance95'],.01)
        self.assertEqual(result['a_MIA'],0)

    def test_attacker_mean_is_not_max_or_pooled_roc(self):
        y=np.array([1,0,0,0]);rare=np.zeros(4,bool)
        confidence=[np.array([1,.4,.3,.2]),np.array([0,.4,.3,.2])]
        result=permutation_mia(confidence,[y,y],[rare,rare],reps=50,low_fpr=.01,fold_k=12)
        expected=np.mean([score_mia(c,y,{'scoring':{'low_fpr':.01}},rare_mask=rare).score for c in confidence])
        self.assertAlmostEqual(result['beta_raw'],expected)
        self.assertAlmostEqual(result['beta_raw'],.5)

if __name__=='__main__':unittest.main()
