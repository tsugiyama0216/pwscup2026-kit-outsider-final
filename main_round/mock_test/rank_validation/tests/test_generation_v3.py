"""Tests for score-blind v3 world-generation boundaries and protocol."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from rank_validation import generation_v3 as v3


class GenerationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=v3.load_protocol()
        root=v3.ROOT
        cls.c=pd.read_csv(root/'main_round/attack/targets/C_0.csv',float_precision='round_trip')
        cls.bg=pd.read_csv(root/'main_round/anonymization/team10/participant_data/A_bg.csv',float_precision='round_trip')
        import json
        cls.schema=json.loads((root/'main_round/anonymization/team10/participant_data/schema.json').read_text(encoding='utf-8-sig'))

    def test_protocol_fixed_and_reserves_all_three_worlds(self):
        self.assertEqual(v3._sha(v3.PROTOCOL_PATH),v3.PROTOCOL_SHA256)
        self.assertEqual(v3._reserved_split(2026100901,self.protocol),('dev','development'))
        for seed in (2026100911,2026100912):
            self.assertEqual(v3._reserved_split(seed,self.protocol),('holdout','confirmation'))
        with self.assertRaises(ValueError):
            v3._reserved_split(7,self.protocol)

    def test_invalid_seed_split_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError,'fixed protocol'):
            v3.generate_world(2026100911,'dev')

    def test_all_grid_strengths_positive_and_no_copy_family(self):
        self.assertEqual(len(self.protocol['grid']),26)
        self.assertTrue(all(x['strength']>0 for x in self.protocol['grid']))
        self.assertNotIn('copy',{x['family'] for x in self.protocol['grid']})

    def test_independent_release_does_not_read_target(self):
        spec={'family':'independent','strength':.2}
        a=v3.make_candidate(None,self.bg,spec,11,self.schema)
        b=v3.make_candidate(object(),self.bg,spec,11,self.schema)
        pd.testing.assert_frame_equal(a,b)

    def test_weak_release_changes_qi_and_preserves_legal_event_states(self):
        spec={'family':'jitter','strength':.02}
        before=self.c.copy(deep=True)
        a=v3.make_candidate(self.c,self.bg,spec,17,self.schema)
        b=v3.make_candidate(self.c,self.bg,spec,17,self.schema)
        pd.testing.assert_frame_equal(a,b)
        pd.testing.assert_frame_equal(self.c,before)
        self.assertEqual(len(a),1049)
        self.assertTrue(np.isfinite(a.drop(columns='prefecture').to_numpy(float)).all())
        self.assertFalse(((a.onset==1)&(a.death==1)).any())
        self.assertTrue((a.loc[(a.onset==0)&(a.death==0),'time']==10).all())
        qi=['age','SBP','TG','HDL','ALT','sex','prefecture']
        original=set(map(tuple,self.c[qi].to_numpy()))
        self.assertFalse(original & set(map(tuple,a[qi].to_numpy())))

    def test_profile_selection_ignores_extra_truth_or_score_fields(self):
        metrics=self.protocol['profile_metrics']
        row={k:1. for k in metrics}; desired={k:.8 for k in metrics}; scales={k:.1 for k in metrics}
        original=v3.public_fit_loss(row,desired,scales,self.protocol)
        augmented=v3.public_fit_loss(row|{'official_score':0,'is_member':1},desired|{'official_score':2},scales,self.protocol)
        self.assertEqual(original,augmented)

    def test_nonfinite_profile_disqualified(self):
        row={k:1. for k in self.protocol['profile_metrics']}
        row[self.protocol['profile_metrics'][0]]=None
        loss,residuals=v3.public_fit_loss(row,row,{k:1 for k in row},self.protocol)
        self.assertTrue(np.isinf(loss))
        self.assertEqual(residuals,{})


if __name__=='__main__':
    unittest.main()
