"""Checks replay, immutable fields, source algorithm and public boundary guard."""
import copy,csv,io,json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from generate import generate
from experiment import public_target,smooth,encode,evaluation

HERE=Path(__file__).resolve().parent

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=(HERE.parent/'M1_v4/C.csv').read_bytes();cls.model=json.loads((HERE/'model.json').read_text(encoding='utf-8'))
        cls.a=pd.read_csv(HERE.parents[1]/'participant_data/A_bg.csv')
        cls.schema=json.loads((HERE.parents[1]/'participant_data/schema.json').read_text(encoding='utf-8'))
        cls.c=pd.read_csv(io.BytesIO(cls.base));cls.target,cls.rare,_=public_target(cls.c,cls.a,cls.schema)

    def test_exact_replay_and_original_field_strings(self):
        result=generate(self.base,self.model);self.assertEqual(result,(HERE/'C.csv').read_bytes())
        before=list(csv.reader(io.StringIO(self.base.decode('utf-8'))));after=list(csv.reader(io.StringIO(result.decode('utf-8'))))
        immutable=[i for i,n in enumerate(before[0]) if n not in ('BMI','TG','ALT')]
        for a,b in zip(before,after):self.assertEqual([a[i] for i in immutable],[b[i] for i in immutable])
        for i in np.flatnonzero(~self.rare):self.assertEqual(before[i+1],after[i+1])

    def test_public_source_algorithm_reproduces_selected_csv(self):
        c,audit=smooth(self.c,self.target,self.rare,self.schema,['BMI','TG','ALT'],.35)
        self.assertEqual(encode(self.base,c,['BMI','TG','ALT'],self.rare),(HERE/'C.csv').read_bytes())
        self.assertEqual(audit['boundary_limited_rows'],11)

    def test_identity_boundaries_and_invalid_strength(self):
        for strength in (0.,.05,.1,.2,.35,1.):
            for fields in (['BMI','TG','ALT'],['BMI','SBP','TG','HDL','ALT','FPG']):
                c,_=smooth(self.c,self.target,self.rare,self.schema,fields,strength)
                np.testing.assert_array_equal(evaluation.local.rare_mask(c,self.schema),self.rare)
                if strength==0:pd.testing.assert_frame_equal(c,self.c)
        with self.assertRaises(ValueError):smooth(self.c,self.target,self.rare,self.schema,['BMI'],1.1)

    def test_rejects_changed_base_and_invalid_patches(self):
        with self.assertRaises(ValueError):generate(self.base+b'\n',self.model)
        for kind in ('forbidden','duplicate','outside'):
            bad=copy.deepcopy(self.model)
            if kind=='forbidden':bad['patches'][0]['onset']='1'
            elif kind=='duplicate':bad['patches'].append(bad['patches'][0])
            else:bad['patches'][0]['row_index']=len(self.c)
            with self.assertRaises(ValueError):generate(self.base,bad)

if __name__=='__main__':unittest.main()
