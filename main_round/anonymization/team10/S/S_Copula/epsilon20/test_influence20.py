from pathlib import Path
import sys,unittest
from unittest.mock import patch
from fractions import Fraction
import json,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;CORE=HERE.parent;TEAM=CORE.parents[1]
sys.path[:0]=[str(CORE),str(CORE/'postprocessing_v2')]
import dp_copula as dc
import dp_cox
import influence_cox as ic
from cox_optimizer import EfronRiskSets

class InfluenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema=json.loads((TEAM/'participant_data/schema.json').read_text())
        raw=pd.read_csv(TEAM/'participant_data/A_bg.csv')
        defaults={c:float(raw[c].median()) for c in dc.NUMERIC}
        cls.public=dc._prepare(raw,cls.schema,sorted(raw.prefecture.unique()),defaults)
        cls.base=dp_cox.build_public_cox_model(cls.public)
        cls.model=ic.build_public_model(cls.public,base_model=cls.base)
    def test_row_map_matches_public_efron_score_away_from_optimum(self):
        beta=np.array(self.base['beta_standardized'])+np.linspace(-.08,.1,9)
        model=ic.build_public_model(self.public,beta=beta,base_model=self.base)
        order=np.argsort(self.public.time,kind='stable')
        x=(dp_cox._features(self.public)-self.base['center'])/self.base['scale']
        risk=EfronRiskSets(self.public.time.to_numpy()[order],self.public.onset.to_numpy()[order])
        exact,_,_=risk.evaluate(x[order],beta)
        np.testing.assert_allclose(ic.score_rows(self.public,model).sum(axis=0),exact,atol=1e-9)
    def test_integer_bound_and_replace_one_extreme_sensitivity(self):
        frame=self.public.iloc[:40].copy()
        for i,col in enumerate(dc.NUMERIC):
            lo,hi=dc._bounds(self.schema)[col]
            frame.loc[frame.index[::2],col]=lo
            frame.loc[frame.index[1::2],col]=hi
        for bound in (1,200,1000):
            rows=ic.bounded_integer_rows(frame,self.model,l1_bound=bound)
            self.assertTrue((np.abs(rows).sum(axis=1)<=4096).all())
            self.assertTrue((np.abs(rows[:,None,:]-rows[None,:,:]).sum(axis=2)<=8192).all())
    def test_mapping_is_record_local_and_permutation_equivariant(self):
        subset=self.public.iloc[:41]
        rows=ic.bounded_integer_rows(subset,self.model,l1_bound=600)
        for i in (0,7,40):
            np.testing.assert_array_equal(rows[i],ic.bounded_integer_rows(subset.iloc[[i]],self.model,l1_bound=600)[0])
        np.testing.assert_array_equal(rows[::-1],ic.bounded_integer_rows(subset.iloc[::-1],self.model,l1_bound=600))
    def test_adaptive_budget_and_final_target_finite(self):
        calls=[]
        def identity(values,epsilon,sensitivity):
            calls.append((Fraction(epsilon),sensitivity))
            return values,{'epsilon_allocated_exact':str(epsilon),'l1_sensitivity':sensitivity}
        plan={'cox_round_epsilons':['16/5','24/5','8'],'clip_quantile':.97,'quantization':4096,'damping':1}
        with patch.object(dc,'release_integer_vector',identity):
            result=ic.measure(self.public.iloc[:1049],self.public.iloc[1049:],plan)
        self.assertEqual(sum(e for e,_ in calls),16)
        self.assertEqual([s for _,s in calls],[8192]*3)
        self.assertTrue(np.isfinite(result['beta_standardized']).all())
        self.assertTrue((np.asarray(result['standard_errors_standardized'])>0).all())

if __name__=='__main__':unittest.main()
