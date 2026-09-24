"""Bounded M1_v5 public-neighborhood smoothing study; local, non-DP selection."""
from pathlib import Path
import csv, io, hashlib, importlib.util, json, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
TEAM=HERE.parents[1]
BASE=HERE.parent/'M1_v4/C.csv'
V4=BASE.parent
OLD=HERE.parent/'evaluation/domias_main_round_20260924'
EXPECTED='d32640fd8a85866121235053dc7c02c29ab87979c8fd5e0013aae9ca49db7201'
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']
FIELDS={'three':['BMI','TG','ALT'],'six':['BMI','SBP','TG','HDL','ALT','FPG']}
TRAIN=[20260924,20260925,20260926,20261001,20261002,20261003]
CONFIRM=[20261011,20261012,20261013]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
evaluation=module('v5_evaluation',OLD/'evaluate.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(np.bool_,bool)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    return x
def save(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def public_target(c,a,schema,k=50):
    """All detector-rare synthetic rows, selected without B or attack labels."""
    rare=evaluation.local.rare_mask(c,schema); ids=np.flatnonzero(rare)
    mean=a[CONT].mean().to_numpy(); scale=a[CONT].std(ddof=0).to_numpy()
    scale=np.where(scale>0,scale,1.)
    za=(a[CONT].to_numpy()-mean)/scale; zc=(c[CONT].to_numpy()-mean)/scale
    distances=cdist(zc[ids],za,metric='sqeuclidean')
    targets=c[CONT].to_numpy().copy()
    neighbors=[]
    for i,d in zip(ids,distances):
        nearest=np.argsort(d,kind='stable')[:k]
        w=np.exp(-(d[nearest]-d[nearest].min())/2);w/=w.sum()
        targets[i]=(za[nearest]*w[:,None]).sum(axis=0)*scale+mean
        neighbors.append(nearest.tolist())
    return targets,rare,{'rare_rows':ids.tolist(),'neighbor_indices':neighbors,'k':k}

def smooth(c,targets,rare,schema,fields,strength):
    if not 0<=strength<=1:raise ValueError('Strength must be in [0,1]')
    out=c.copy(deep=True); positions=[CONT.index(x) for x in fields]
    old=c[CONT].to_numpy(); delta=np.zeros_like(old)
    delta[np.ix_(rare,positions)]=targets[np.ix_(rare,positions)]-old[np.ix_(rare,positions)]
    # Use a public detector only. Keep each original rare classification by
    # reducing its step if the interpolated row would cross the boundary.
    detector=schema['rare_detector']; index=[CONT.index(n) for n in detector['columns']]
    mu=np.asarray(detector['mu']); inverse=np.linalg.pinv(detector['sigma']); lo=detector['band']['lo']
    def radius(alpha):
        d=(old+alpha[:,None]*delta)[:,index]-mu
        return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',d,inverse,d)))
    alpha=np.full(len(c),strength); crossing=rare&(radius(alpha)<lo)
    lower=np.zeros(len(c));upper=alpha.copy()
    for _ in range(45):
        middle=(lower+upper)/2;valid=radius(middle)>=lo
        lower=np.where(valid,middle,lower);upper=np.where(valid,upper,middle)
    alpha[crossing]=lower[crossing]*(1-1e-10)
    values=old+alpha[:,None]*delta
    for field in fields:out.loc[rare,field]=values[rare,CONT.index(field)]
    if not np.array_equal(evaluation.local.rare_mask(out,schema),rare):raise RuntimeError('Rare labels changed')
    return out,{'strength':strength,'fields':fields,'boundary_limited_rows':int(crossing.sum()),
                'applied_strength_min_rare':float(alpha[rare].min()),'applied_strength_mean_rare':float(alpha[rare].mean())}

def encode(base_bytes,c,fields,rare):
    rows=list(csv.reader(io.StringIO(base_bytes.decode('utf-8'))));header=rows[0]
    for i in np.flatnonzero(rare):
        for name in fields:rows[i+1][header.index(name)]=format(c.loc[i,name],'.17g')
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerows(rows)
    return stream.getvalue().encode('utf-8')

def raw(conf,plan,cfg):
    per=[evaluation.score_mia(conf[p['pool_indices']],np.array(p['truth']),cfg,rare_mask=np.array(p['rare'],bool)) for p in plan['attackers']]
    return float(np.mean([v.score for v in per]))

def main():
    if sha(BASE)!=EXPECTED:raise RuntimeError('Frozen v4 changed')
    if (HERE/'screening.json').exists():raise FileExistsError('Completed screening is immutable')
    cfg,env=evaluation.utility.environment();cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    options=[{'name':f'{group}_{int(strength*100):02d}','fields':fields,'strength':strength} for group,fields in FIELDS.items() for strength in [.05,.1,.2,.35]]
    protocol={'base_sha256':EXPECTED,'candidates':options,'training_seeds':TRAIN,'confirmation_seeds':CONFIRM,
        'utility_floor':.86,'aia_risk_ceiling':.10,'public_neighbor_k':50,
        'screening':'Same six known cohort sets; rank candidates by minimum objective with v4 chance95 fixed for all candidates, then mean objective, then U. Do not exploit candidate-specific permutation randomness during selection.',
        'selection':'U>=0.86, AIA<=0.10, minimum screening objective strictly above v4 baseline minimum; select maximum(minimum, mean, U), freeze before final permutation/confirmation.',
        'fallback':'If no eligible improvement, preserve all results and report failure; any subsequent study must use a separate protocol.',
        'confirmation':'Selected model and v4 on known six + new three seeds with exact per-model 20000 permutations. No further tuning from confirmation.',
        'private_data_use':'No B in smoothing; B used for score-based selection. Inherits non-DP v4/v3. No formal DP guarantee.',
        'code_sha256':sha(Path(__file__)),'environment':env,
        'inputs_sha256':{n:sha(TEAM/'participant_data'/n) for n in ['B.csv','B_self.csv','A_bg.csv','schema.json','utility_ref.json']},
        'dependency_sha256':{str(p.relative_to(TEAM)):sha(p) for p in [OLD/'evaluate.py',OLD.parent/'scripts/evaluate_resistance.py',TEAM/'S/S_Copula/evaluate.py']}}
    if (HERE/'protocol.json').exists() and read(HERE/'protocol.json')!=protocol:raise RuntimeError('Protocol changed')
    save(HERE/'protocol.json',protocol)
    c0=pd.read_csv(BASE);a=pd.read_csv(TEAM/'participant_data/A_bg.csv');b=pd.read_csv(TEAM/'participant_data/B_self.csv')
    schema=read(TEAM/'participant_data/schema.json');pool=pd.concat([b.drop(columns='is_rare'),a],ignore_index=True)
    plans={s:read(V4/f'confirmation/cohorts_{s}.json') for s in TRAIN}
    previous=pd.read_csv(V4/'confirmation/comparison.csv').query('model=="M1_v4"').set_index('seed')
    target,rare,donors=public_target(c0,a,schema);save(HERE/'public_neighbors.json',donors)
    results=[]
    for opt in options:
        out=HERE/'candidates'/opt['name'];out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            result=read(out/'result.json');assert sha(out/'C.csv')==result['C_sha256']
        else:
            start=time.monotonic();c,audit=smooth(c0,target,rare,schema,opt['fields'],opt['strength'])
            (out/'C.csv').write_bytes(encode(BASE.read_bytes(),c,opt['fields'],rare));c=pd.read_csv(out/'C.csv')
            utility=evaluation.utility.evaluate_csv(out/'C.csv');save(out/'utility.json',utility)
            aia=evaluation.evaluate_aia(c,b,a,schema,cfg);save(out/'aia.json',aia)
            conf=evaluation.overlap_mia(c,a,pool,cfg);np.save(out/'domias_confidence.npy',conf)
            u=utility['score']['utility'];risk=aia['worst_before_phi'];seeds=[]
            for seed,plan in plans.items():
                beta_raw=raw(conf,plan,cfg);chance=previous.loc[seed,'chance95']
                beta=max(beta_raw-chance+.01,.01);a_mia=float(np.clip((beta/.01-1)/11,0,1));anon=1-max(a_mia,risk)
                seeds.append({'seed':seed,'raw':beta_raw,'fixed_v4_chance95':chance,'screening_anon':anon,'screening_objective':min(u,anon)})
            result={'name':opt['name'],'U':u,'facets':utility['score']['facets'],'AIA':risk,'C_sha256':sha(out/'C.csv'),
                    'seeds':seeds,'min_screening_objective':min(r['screening_objective'] for r in seeds),
                    'mean_screening_objective':float(np.mean([r['screening_objective'] for r in seeds])),
                    'min_screening_anon':min(r['screening_anon'] for r in seeds),'audit':audit,'elapsed_seconds':time.monotonic()-start}
            save(out/'result.json',result)
        result['eligible']=bool(result['U']>=.86 and result['AIA']<=.10 and result['min_screening_objective']>previous.Objective_proxy.min()+1e-9)
        results.append(result); print(json.dumps({k:result[k] for k in ['name','U','AIA','min_screening_anon','min_screening_objective','eligible']}),flush=True)
    eligible=[r for r in results if r['eligible']]
    selected=max(eligible,key=lambda r:(r['min_screening_objective'],r['mean_screening_objective'],r['U'])) if eligible else None
    save(HERE/'screening.json',{'results':results,'selected':selected['name'] if selected else None,'selected_sha256':selected['C_sha256'] if selected else None,
         'baseline_U':float(previous.U.iloc[0]),'baseline_min_anon':float(previous.Anon_proxy.min()),'baseline_min_objective':float(previous.Objective_proxy.min()),
         'confirmation_used_for_selection':False})
    if selected:(HERE/'C.csv').write_bytes((HERE/'candidates'/selected['name']/'C.csv').read_bytes())
    print(json.dumps({'selected':selected['name'] if selected else None}),flush=True)

if __name__=='__main__':main()
