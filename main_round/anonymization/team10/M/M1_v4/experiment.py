"""Bounded local M1_v3 improvement study. No formal DP or server submission."""
from pathlib import Path
import csv, hashlib, importlib.util, io, json, sys, time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[1]
BASE = HERE.parent / 'M1_v3/C.csv'
OLD = HERE.parent / 'evaluation/domias_main_round_20260924'
EXPECTED = 'd425f6d612f0d3f0d8af3ee0a7f042d846c25ad9555966b437f3623773c175a1'
STATIC = ['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG']

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

evaluation=module('m1v4_existing_evaluation', OLD/'evaluate.py')
sys.path.insert(0, str(TEAM/'R/R8'))
from outcomes import calibrate_outcomes

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,(np.bool_,bool)): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    return x
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def csv_from_original(c0,c):
    """Preserve every original static-field string and copy existing outcome tuples."""
    rows=list(csv.reader(io.StringIO(BASE.read_text(encoding='utf-8'))))
    header,body=rows[0],rows[1:]; cols=['time','onset','death']; ix=[header.index(n) for n in cols]
    sources={tuple(v):[body[i][j] for j in ix] for i,v in enumerate(c0[cols].to_numpy())}
    for i,value in enumerate(c[cols].to_numpy()):
        for j,text in zip(ix,sources[tuple(value)]): body[i][j]=text
    stream=io.StringIO(newline=''); writer=csv.writer(stream,lineterminator='\n'); writer.writerow(header); writer.writerows(body)
    return stream.getvalue().encode('utf-8')

def check_invariants(c0,c,rare):
    if not np.array_equal(c0[STATIC].to_numpy(),c[STATIC].to_numpy()): raise ValueError('Static covariates changed')
    if not np.array_equal(c0.death.to_numpy(),c.death.to_numpy()): raise ValueError('Death changed')
    for mask in (rare,~rare):
        left=sorted(map(tuple,c0.loc[mask,['time','onset','death']].to_numpy()))
        right=sorted(map(tuple,c.loc[mask,['time','onset','death']].to_numpy()))
        if left!=right: raise ValueError('Within-stratum outcome multiset changed')

def main():
    if sha(BASE)!=EXPECTED: raise RuntimeError('Frozen M1_v3 changed')
    if (HERE/'selection.json').exists(): raise FileExistsError('Completed experiment is immutable')
    cfg,provenance=evaluation.utility.environment()
    cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    schema=read(TEAM/'participant_data/schema.json')
    b=pd.read_csv(TEAM/'participant_data/B.csv')
    bs=pd.read_csv(TEAM/'participant_data/B_self.csv'); a=pd.read_csv(TEAM/'participant_data/A_bg.csv')
    c0=pd.read_csv(BASE); rare=evaluation.local.rare_mask(c0,schema)
    options=[('plain',False,2.1),('guard200',True,2.0),('guard210',True,2.1),('guard225',True,2.25)]
    protocol={'base_sha256':EXPECTED,'method':'R8 aggregate Cox outcome calibration applied to frozen M1_v3; outcome-only swaps',
              'candidates':[{'name':n,'guard':g,'margin':m,'max_steps':24} for n,g,m in options],
              'selection':'Maximum local U among candidates with U above M1_v3 and AIA worst before phi <= baseline; static ten columns and death exactly fixed',
              'exploratory_mia_seeds':[20260924,20260925,20260926],
              'confirmation_mia_seeds':[20261001,20261002,20261003],
              'confirmation_policy':'Freeze selected CSV before confirmation; no retuning against confirmation results',
              'code_sha256':sha(Path(__file__)), 'official_kit':provenance,
              'dependencies_sha256':{str(p.relative_to(TEAM)):sha(p) for p in [TEAM/'R/R8/outcomes.py',TEAM/'R/R8/model.py',TEAM/'R/R8/cox_calibration.py',TEAM/'R/R8/copula_helpers.py',OLD/'evaluate.py',OLD.parent/'scripts/evaluate_resistance.py']},
              'inputs_sha256':{n:sha(TEAM/'participant_data'/n) for n in ['B.csv','B_self.csv','A_bg.csv','schema.json','utility_ref.json']},
              'privacy':'Private aggregate fitting and score-based selection; no formal DP guarantee; local proxy only'}
    if (HERE/'protocol.json').exists() and read(HERE/'protocol.json')!=protocol: raise RuntimeError('Protocol changed')
    save(HERE/'protocol.json',protocol)
    base_u=read(OLD/'results/M1_v3/utility.json')['score']['utility']
    base_aia=read(OLD/'results/M1_v3/aia.json')['worst_before_phi']
    results=[]
    for name,guard,margin in options:
        out=HERE/'candidates'/name; out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            result=read(out/'result.json'); assert result['C_sha256']==sha(out/'C.csv')
        else:
            start=time.monotonic()
            c,audit=calibrate_outcomes(b,c0,rare,max_steps=24,significance_guard=guard,significance_margin=margin)
            check_invariants(c0,c,rare)
            (out/'C.csv').write_bytes(csv_from_original(c0,c))
            parsed=pd.read_csv(out/'C.csv'); check_invariants(c0,parsed,rare)
            utility=evaluation.utility.evaluate_csv(out/'C.csv'); save(out/'utility.json',utility)
            aia=evaluation.evaluate_aia(parsed,bs,a,schema,cfg); save(out/'aia.json',aia)
            result={'name':name,'C_sha256':sha(out/'C.csv'),'U':utility['score']['utility'],'facets':utility['score']['facets'],
                    'AIA':aia['worst_before_phi'],'changed_rows':int((parsed!=c0).any(axis=1).sum()),'audit':audit,
                    'elapsed_seconds':time.monotonic()-start}
            save(out/'result.json',result)
        result['eligible']=result['U']>base_u+1e-6 and result['AIA']<=base_aia+1e-12
        results.append(result)
        print(json.dumps({k:result[k] for k in ['name','U','AIA','changed_rows','eligible']},ensure_ascii=False),flush=True)
    eligible=[r for r in results if r['eligible']]
    selected=max(eligible,key=lambda r:(r['U'],-r['changed_rows'],r['name'])) if eligible else None
    selection={'baseline':{'U':base_u,'AIA':base_aia},'results':results,'selected':selected['name'] if selected else None,
               'selected_csv_sha256':selected['C_sha256'] if selected else None,'confirmation_results_used_for_selection':False}
    if selected:
        (HERE/'C.csv').write_bytes((HERE/'candidates'/selected['name']/'C.csv').read_bytes())
    save(HERE/'selection.json',selection)
    print(json.dumps({'selected':selection['selected']},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
