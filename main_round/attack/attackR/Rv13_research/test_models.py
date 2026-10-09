import unittest,importlib.util,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'main_round/mock_test/local-output/rv13-research-20261009'
def load(name):
 p=ROOT/'main_round/attack/attackR'/name/'model.py';s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
MODELS={n:load(n) for n in ['Rv13','Rv14','Rv15']}
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  originals=ROOT/'main_round/anonymization/team10/participant_data';targets=ROOT/'main_round/attack/targets'
  def read(f):
   source=targets/f.removeprefix('targets/') if f.startswith('targets/') or f=='aia_challenge_10.csv' else originals/f
   return pd.read_csv(source,float_precision='round_trip')
  cls.public={'candidates':read('B.csv'),'self_labels':read('B_self.csv'),'background':read('A_bg.csv').iloc[:250].copy(),'self_challenge':read('aia_challenge_10.csv'),'targets':{0:{'release':read('targets/C_0.csv').iloc[:120].copy(),'challenge':read('targets/aia_challenge_0.csv').iloc[:50].copy()}}}
 def test_frozen_sources(self):
  for n in MODELS:
   d=ROOT/'main_round/attack/attackR'/n;s=json.loads((d/'selection.json').read_text());self.assertEqual(hashlib.sha256((d/'model.py').read_bytes()).hexdigest(),s['source_sha256'])
 def test_private_proof(self):
  for m in MODELS.values():self.assertEqual(m.private_mask(self.public['candidates'],self.public['self_challenge']).sum(),163)
 def test_insufficient_private_proof_rejected(self):
  q=self.public['self_challenge'];b=self.public['candidates'];m=MODELS['Rv13'];matched=set(map(tuple,b[m.QI+m.CATS].to_numpy()));remove=next(i for i,r in enumerate(q[m.QI+m.CATS].to_numpy()) if tuple(r) in matched)
  with self.assertRaises(ValueError):m.private_mask(b,q.drop(q.index[remove]))
 def test_tie_rule(self):
  m=MODELS['Rv13'];d=np.array([[1.,1.,2.],[2.,0.,0.]]);c=pd.DataFrame({'time':[3.,5.,9.]})
  np.testing.assert_array_equal(m.nearest_times(d,c),[4.,7.])
  np.testing.assert_array_equal(m.nearest_times(d[:,[2,0,1]],c.iloc[[2,0,1]]),[4.,7.])
 def test_copy_signal(self):
  m=MODELS['Rv14'];bg=pd.DataFrame(np.arange(80).reshape(10,8)+10,columns=m.NUM);b=pd.DataFrame([[1.]*8,[100.]*8],columns=m.NUM);c=b.iloc[:1].copy()
  s=m.numerical_membership(b,c,bg);self.assertGreater(s[0],s[1])
 def test_nonfinite_rejected(self):
  m=MODELS['Rv14'];b=self.public['candidates'].copy();b.loc[0,'BMI']=np.nan
  with self.assertRaises(ValueError):m.numerical_membership(b,self.public['targets'][0]['release'],self.public['background'])
 def test_invariance_and_no_mutation(self):
  import copy
  before=copy.deepcopy(self.public)
  for name,m in MODELS.items():
   p=copy.deepcopy(self.public);first=m.predict_world(**p)
   p['candidates']=p['candidates'].sample(frac=1,random_state=4);p['targets'][0]['release']=p['targets'][0]['release'].sample(frac=1,random_state=5);p['targets'][0]['challenge']=p['targets'][0]['challenge'].sample(frac=1,random_state=6)
   second=m.predict_world(**p)
   for key,cols in [('F_mia',['record_id']),('F_aia',['target_k','challenge_row_id'])]:
    pd.testing.assert_frame_equal(first[key].sort_values(cols).reset_index(drop=True),second[key].sort_values(cols).reset_index(drop=True),check_exact=True)
   self.assertTrue(np.isfinite(first['F_mia'].to_numpy()).all());self.assertTrue(first['F_mia'].iloc[:,1:].ge(0).all().all());self.assertTrue(first['F_mia'].iloc[:,1:].le(1).all().all())
   mask=m.private_mask(before['candidates'],before['self_challenge']);self.assertTrue(first['F_mia'].loc[mask,'0'].eq(0).all())
  for key in ['candidates','self_labels','background','self_challenge']:pd.testing.assert_frame_equal(self.public[key],before[key],check_exact=True)
  for key in ['release','challenge']:pd.testing.assert_frame_equal(self.public['targets'][0][key],before['targets'][0][key],check_exact=True)
if __name__=='__main__':unittest.main(verbosity=2)
