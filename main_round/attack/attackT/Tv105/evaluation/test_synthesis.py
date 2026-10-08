"""Meaningful consensus and mixture invariants, independent of numeric scores."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from synthesis import TV,aia_consensus,mia_blend

class SynthesisTests(unittest.TestCase):
    def test_distant_times_never_interpolated(self):
        a={n:{'mia':np.array([.1,.8]),'aia':np.array([v])} for n,v in zip(TV,[1.,8.,8.2,8.1])}
        result=aia_consensus(a,'Tv101')[0]
        self.assertIn(result,[8.,8.2,8.1]);self.assertNotAlmostEqual(result,6.325)
    def test_anchor_only_breaks_supported_ties(self):
        a={n:{'aia':np.array([v])} for n,v in zip(TV,[1.,1.1,8.,8.1])}
        self.assertEqual(aia_consensus(a,'Tv103')[0],8.)
    def test_mia_monotone_scaling_invariance(self):
        a={n:{'mia':np.array([.1,.2,.7])} for n in TV}
        x=mia_blend(a,('Tv101','Tv103',.25))
        a['Tv101']['mia']=a['Tv101']['mia']**3
        np.testing.assert_array_equal(x,mia_blend(a,('Tv101','Tv103',.25)))
if __name__=='__main__':unittest.main()
