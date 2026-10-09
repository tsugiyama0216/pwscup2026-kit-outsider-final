"""Checks for v4 demographic transport without attack predictions or scores."""
import copy
from collections import Counter
import json
import unittest
import numpy as np
import pandas as pd
from rank_validation import generation_v4 as v4
from rank_validation import generation_v3 as v3
from sugiyama_v2 import generate as gen


class GenerationV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=v4.ROOT
        source=root/'main_round/anonymization/team10/participant_data'
        cls.actual={int(p.stem.split('_')[-1]):pd.read_csv(p,float_precision='round_trip')
            for p in (root/'main_round/attack/targets').glob('aia_challenge_*.csv')}
        cls.world=gen.build_world(2026100941,profile='mixed',split='dev',
            donors=pd.read_csv(source/'B_self.csv',float_precision='round_trip'),
            background_source=pd.read_csv(source/'A_bg.csv',float_precision='round_trip'),
            schema=json.loads((source/'schema.json').read_text(encoding='utf-8-sig')),
            release_factory=lambda target,*args:target[gen.FEATURES].copy())
        v4._synchronize_censoring(cls.world)
        cls.before=copy.deepcopy(cls.world)
        cls.diagnostic=v4.transport_public_demographics(cls.world,cls.actual,2026100941)

    def test_protocol_preserves_grid_loss_and_reserved_seeds(self):
        protocol=v4.load_protocol(); old=v3.load_protocol()
        for key in ('grid','profile_metrics','scale_floors','metric_weights','loss','fit_policy'):
            self.assertEqual(protocol[key],old[key])
        self.assertEqual(protocol['development_seeds'],[2026100941])
        self.assertEqual(protocol['confirmation_seeds'],[2026100951,2026100952])
        with self.assertRaises(ValueError):
            v4.generate_world(2026100951,'dev')

    def test_public_projection_minimal_and_symmetric(self):
        odd_targets={}
        for team,observed in self.actual.items():
            slots,d=v4.paired_demographic_slots(observed,np.random.default_rng(team))
            self.assertEqual(len(slots),163)
            self.assertEqual(sum(c['projected'] for c in d['cells']),326)
            self.assertEqual(d['pooled_count_l1_change'],d['odd_cells'])
            for s in slots:
                self.assertEqual((s['age_member']-40)//5,s['age5'])
                self.assertEqual((s['age_control']-40)//5,s['age5'])
            for cell in d['cells']:
                self.assertLessEqual(abs(cell['difference']),1)
                if cell['observed']%2==0:
                    self.assertEqual(cell['difference'],0)
            if d['odd_cells']:
                odd_targets[team]=d['odd_cells']
        self.assertEqual(odd_targets,{0:4,6:6,22:2,27:8})

    def test_pair_assignment_is_invariant_to_role_exchange(self):
        a=pd.DataFrame({'sex':[0,1,0],'age':[41,54,65]})
        b=pd.DataFrame({'sex':[0,1,0],'age':[44,50,69]})
        pairs=a.copy(); pairs.age=(a.age.to_numpy()+b.age.to_numpy())/2
        swapped=b.copy(); swapped.age=(b.age.to_numpy()+a.age.to_numpy())/2
        forward=v4._assignment(pairs,[0,0,1],[63,42,52],np.random.default_rng(11))
        reverse=v4._assignment(swapped,[0,0,1],[63,42,52],np.random.default_rng(11))
        np.testing.assert_array_equal(forward,reverse)

    def test_transport_invariant_to_exchanging_original_pair_ages(self):
        swapped=copy.deepcopy(self.before)
        hidden=swapped['private']; pool=hidden['pool']
        for team in self.actual:
            truth=hidden['aia_truth'][team]
            ids=np.asarray(hidden['challenge_pool_ids'][team])
            members=np.sort(ids[truth.label.to_numpy()=='member'])
            controls=np.sort(ids[truth.label.to_numpy()=='control'])
            old_member=pool.loc[members,'age'].to_numpy().copy()
            pool.loc[members,'age']=pool.loc[controls,'age'].to_numpy()
            pool.loc[controls,'age']=old_member
        v4.transport_public_demographics(swapped,self.actual,2026100941)
        columns=['prefecture','sex','age']
        pd.testing.assert_frame_equal(swapped['private']['pool'][columns],self.world['private']['pool'][columns])

    def test_shared_ids_features_truth_and_clinical_values_unchanged(self):
        old=self.before['private']; new=self.world['private']
        for key in ('cohorts','candidate_pool_ids','challenge_pool_ids'):
            self.assertEqual(old[key],new[key])
        counts=Counter(pid for ids in old['cohorts'].values() for pid in ids)
        shared=[pid for pid,n in counts.items() if n>1]
        pd.testing.assert_frame_equal(old['pool'].loc[shared],new['pool'].loc[shared])
        unchanged=[c for c in old['pool'].columns if c not in ('prefecture','sex','age')]
        pd.testing.assert_frame_equal(old['pool'][unchanged],new['pool'][unchanged])
        for team in range(30):
            pd.testing.assert_frame_equal(old['aia_truth'][team],new['aia_truth'][team])

    def test_both_roles_match_projected_public_cells(self):
        hidden=self.world['private']; pool=hidden['pool']
        for d in self.diagnostic['targets']:
            team=d['target']; truth=hidden['aia_truth'][team]
            ids=np.asarray(hidden['challenge_pool_ids'][team])
            expected={(c['prefecture'],c['sex'],c['age5']):c['per_side'] for c in d['cells'] if c['per_side']}
            for role in ('member','control'):
                frame=pool.iloc[ids[truth.label.to_numpy()==role]].copy()
                frame['age5']=((frame.age-40)//5).astype(int)
                counts=frame.groupby(['prefecture','sex','age5']).size().to_dict()
                self.assertEqual(counts,expected)
                self.assertEqual(len(frame),163)
                self.assertEqual(int(frame.is_rare.sum()),13)
                self.assertEqual(int(frame.onset.sum()),16)
                self.assertEqual(int(((frame.is_rare==1)&(frame.onset==1)).sum()),4)

    def test_public_tables_reflect_transported_pool_and_self_overlap(self):
        public=self.world['public']; hidden=self.world['private']; pool=hidden['pool']
        pd.testing.assert_frame_equal(public['candidates'][gen.FEATURES],pool.iloc[hidden['candidate_pool_ids']][gen.FEATURES].reset_index(drop=True))
        for team in self.actual:
            challenge=public['self_challenge'] if team==10 else public['targets'][team]['challenge']
            pd.testing.assert_frame_equal(challenge[gen.QI],pool.iloc[hidden['challenge_pool_ids'][team]][gen.QI].reset_index(drop=True))
        qi=gen.QI
        known=set(map(tuple,public['candidates'][qi].to_numpy())) & set(map(tuple,public['self_challenge'][qi].to_numpy()))
        self.assertEqual(len(known),163)
        self.assertFalse(pool[qi].duplicated().any())

    def test_background_audit_is_diagnostic_and_non_mutating(self):
        bg=self.before['public']['background']; before=bg.copy(deep=True)
        d=v4.background_demographic_diagnostics(bg,bg,self.actual[10])
        self.assertTrue(d['diagnostic_only'])
        self.assertEqual(d['distributions']['prefecture']['total_variation'],0)
        pd.testing.assert_frame_equal(bg,before)


if __name__=='__main__':
    unittest.main()
