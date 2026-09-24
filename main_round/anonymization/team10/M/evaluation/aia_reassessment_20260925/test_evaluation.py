"""Focused checks for corrected query disclosure, challenge design, and aggregation."""
import unittest
import numpy as np
import pandas as pd
import evaluate as e

class ReassessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,_=e.ev.utility.environment()
        cls.cfg['aia'].update({'qi_continuous':e.QI[:5],'block_keys':['sex','prefecture'],'k_nn_onset':5})
        cls.b=pd.read_csv(e.TEAM/'participant_data/B_self.csv')
        cls.a=pd.read_csv(e.TEAM/'participant_data/A_bg.csv')
        cls.br=cls.b.is_rare.to_numpy(bool)
        cls.ar=e.ev.local.rare_mask(cls.a,e.read(e.TEAM/'participant_data/schema.json'))
        cls.c=pd.read_csv(e.TEAM/'M/M1_v5/C.csv')

    def test_pairing_all_seeds_and_replay(self):
        for seed in e.SEEDS:
            plan=e.paired_plan(self.b,self.a,self.br,self.ar,seed)
            e.validate_plan(self.b,self.a,self.br,self.ar,plan['member_indices'],plan['control_indices'])
            self.assertEqual(plan,e.read(e.HERE/f'private/plans_{seed}.json'))
            self.assertEqual(plan['excluded_B_count'],50)
        plan=e.paired_plan(self.b,self.a,self.br,self.ar,e.SEEDS[0])
        bad=plan['control_indices'].copy();bad[1]=bad[0]
        with self.assertRaises(AssertionError):
            e.validate_plan(self.b,self.a,self.br,self.ar,plan['member_indices'],bad)

    def test_query_hidden_attributes_cannot_change_predictions(self):
        q=self.b.head(60).copy()
        original=e.predict(self.c,q,self.a,self.cfg)
        for col in ['BMI','FPG','time','onset','is_rare','record_id']:
            q[col]=np.nan
        poisoned=e.predict(self.c,q,self.a,self.cfg)
        only_qi=e.predict(self.c,q[e.QI],self.a[e.QI],self.cfg)
        for attack in original:
            np.testing.assert_array_equal(original[attack],poisoned[attack])
            np.testing.assert_array_equal(original[attack],only_qi[attack])

    def test_success_threshold_and_rare_deadzone_against_hand_calculation(self):
        # Inclusive .5-year boundary, 10/20 versus 4/20 all successes;
        # rare successes 3/4 versus 0/4, independently calculate SE adjustment.
        pm=np.full(20,.6);pm[:10]=.5
        pc=np.full(20,.6);pc[:4]=.5
        mr=np.zeros(20,bool);mr[[0,1,2,10]]=True
        cr=np.zeros(20,bool);cr[[4,5,6,7]]=True
        result=e.ev.score_aia(None,pm,np.zeros(20),np.zeros(20),None,pc,np.zeros(20),np.zeros(20),
                               self.cfg,member_rare_mask=mr,control_rare_mask=cr)
        overall=(.5-.2)/(1-.2)
        rare=max(0,.75-1.64*np.sqrt(.75*.25/4))
        self.assertAlmostEqual(result.breakdown['r_time_all'],overall)
        self.assertAlmostEqual(result.breakdown['r_time_rare'],rare)
        self.assertAlmostEqual(result.score,max(overall,rare))

    def test_worst_attack_then_single_phi_and_objective(self):
        aia,anon,obj=e.combine([.1,.3],.2,.9,.05)
        self.assertAlmostEqual(aia,.25)
        self.assertAlmostEqual(anon,.75)
        self.assertAlmostEqual(obj,.75)
        self.assertEqual(e.combine([.01,.02],.2,.7,.05),(0.,.8,.7))
        self.assertEqual(e.combine([.2,.1],.4,.9),(0.2,.6,.6))

    def test_saved_comparisons_arithmetic_and_input_integrity(self):
        table=pd.read_csv(e.HERE/'comparison_all_seeds.csv')
        self.assertEqual(len(table),27)
        self.assertFalse(table.duplicated(['model','seed']).any())
        self.assertFalse(table.isna().any().any())
        for r in table.itertuples():
            self.assertAlmostEqual(r.Anon_no_phi_proxy,1-max(r.a_MIA,r.AIA_before_phi))
            self.assertAlmostEqual(r.Objective_no_phi_proxy,min(r.U,r.Anon_no_phi_proxy))
            self.assertAlmostEqual(r.AIA_phi_scenario,max(0,r.AIA_before_phi-e.PHI))
        for path,h in e.read(e.HERE/'protocol.json')['input_sha256'].items():
            self.assertEqual(e.sha(e.TEAM/path),h)

if __name__=='__main__': unittest.main(verbosity=2)
