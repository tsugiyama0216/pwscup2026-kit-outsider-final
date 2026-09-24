"""Frozen-artifact safety and model invariants, without rerunning selection."""
import copy, csv, io, json, unittest
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from generate import generate

HERE=Path(__file__).resolve().parent

class FrozenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=(HERE.parent/'M1_v3/C.csv').read_bytes()
        cls.model=json.loads((HERE/'model.json').read_text(encoding='utf-8'))

    def test_reproduces_exact_bytes_and_preserves_static_strings(self):
        result=generate(self.base,self.model)
        self.assertEqual(result,(HERE/'C.csv').read_bytes())
        before=list(csv.reader(io.StringIO(self.base.decode('utf-8'))))
        after=list(csv.reader(io.StringIO(result.decode('utf-8'))))
        static=[i for i,n in enumerate(before[0]) if n not in ('time','onset')]
        for a,b in zip(before,after): self.assertEqual([a[i] for i in static],[b[i] for i in static])

    def test_stratum_outcome_distributions_are_preserved(self):
        schema=json.loads((HERE.parents[1]/'participant_data/schema.json').read_text(encoding='utf-8'))
        a=pd.read_csv(io.BytesIO(self.base)); b=pd.read_csv(HERE/'C.csv')
        detector=schema['rare_detector']; x=a[detector['columns']].to_numpy()-np.asarray(detector['mu'])
        rare=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',x,np.linalg.pinv(detector['sigma']),x)))>=detector['band']['lo']
        for mask in (rare,~rare):
            self.assertEqual(Counter(map(tuple,a.loc[mask,['time','onset','death']].to_numpy())),
                             Counter(map(tuple,b.loc[mask,['time','onset','death']].to_numpy())))

    def test_rejects_wrong_base_and_forbidden_or_duplicate_patches(self):
        with self.assertRaises(ValueError): generate(self.base+b'\n',self.model)
        wrong=copy.deepcopy(self.model); wrong['patches'][0]['BMI']='0'
        with self.assertRaises(ValueError): generate(self.base,wrong)
        wrong=copy.deepcopy(self.model); wrong['patches'].append(wrong['patches'][0])
        with self.assertRaises(ValueError): generate(self.base,wrong)

if __name__=='__main__': unittest.main()
