"""Reproducible main-round-style local DOMIAS comparison; not official Anon.

No generator, private fitting, candidate tuning, or submission is performed.
All models share frozen pseudo-cohorts and three preselected seeds.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
TEAM=HERE.parents[2]
REPO=TEAM.parents[2]
KIT=REPO.parent/'official-kit'
sys.path.insert(0,str(KIT/'starter/src'))
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict
from pwscup2026.scoring.attack_score import score_mia, score_aia

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

local=module('existing_resistance',HERE.parent/'scripts/evaluate_resistance.py')
utility=module('existing_utility',TEAM/'S/S_Copula/evaluate.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(local.json_safe(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def batch_tpr(labels, endpoints, negatives, positives, fpr=.01):
    """Exact grouped-score ROC interpolation for a batch, preserving ties."""
    if positives==0 or negatives==0:return np.zeros(len(labels))
    tp=np.cumsum(labels==1,axis=1)[:,endpoints]
    fp=np.cumsum(labels==0,axis=1)[:,endpoints]
    target=negatives*fpr
    left=(fp<=target).sum(axis=1)-1
    right=left+1;rows=np.arange(len(labels))
    lo_tp=np.where(left>=0,tp[rows,np.maximum(left,0)],0)
    lo_fp=np.where(left>=0,fp[rows,np.maximum(left,0)],0)
    hi_tp=tp[rows,np.minimum(right,len(endpoints)-1)]
    hi_fp=fp[rows,np.minimum(right,len(endpoints)-1)]
    weight=np.divide(target-lo_fp,hi_fp-lo_fp,out=np.zeros(len(labels),float),where=hi_fp>lo_fp)
    return (lo_tp+weight*(hi_tp-lo_tp))/positives

def permutation_mia(confidences, truths, rares, *, reps, low_fpr, fold_k, batch_size=256):
    per=[];null=np.zeros(reps,float)
    digest=hashlib.sha256()
    for conf,y,r in zip(confidences,truths,rares):
        digest.update(np.ascontiguousarray(np.column_stack([conf,y,r]),dtype='<f8').tobytes())
    # Explicitly frozen hash convention; official server serialization is unknown.
    seed=int.from_bytes(digest.digest()[:8],'little')
    rng=np.random.default_rng(seed)
    cfg={'scoring':{'low_fpr':low_fpr}}
    for conf,y,rare in zip(confidences,truths,rares):
        result=score_mia(conf,y,cfg,rare_mask=rare)
        per.append({'raw':result.score,**result.breakdown,'members':int(y.sum()),'rare_members':int((y.astype(bool)&rare).sum()),'rare_candidates':int(rare.sum()),'unique_scores':int(np.unique(conf).size)})
        order=np.argsort(-conf,kind='stable');ranked=conf[order]
        endpoints=np.r_[np.flatnonzero(np.diff(ranked)!=0),len(conf)-1]
        # Codes keep truth and rare labels together while permuting each attacker.
        codes=(y.astype(np.int8)+2*rare.astype(np.int8))[order]
        p=int(y.sum());n=len(y)-p;rp=int((y.astype(bool)&rare).sum());rn=int(rare.sum())-rp
        for start in range(0,reps,batch_size):
            stop=min(reps,start+batch_size)
            shuffled=rng.permuted(np.broadcast_to(codes,(stop-start,len(codes))),axis=1)
            all_labels=shuffled&1
            val=batch_tpr(all_labels,endpoints,n,p,low_fpr)
            if rp>0:
                rare_labels=np.where(shuffled>=2,all_labels,-1)
                val=np.maximum(val,batch_tpr(rare_labels,endpoints,rn,rp,low_fpr))
            null[start:stop]+=val/len(confidences)
    raw=float(np.mean([p['raw'] for p in per]));chance=float(np.quantile(null,.95))
    beta=max(raw-chance+low_fpr,low_fpr)
    a_mia=float(np.clip((beta/low_fpr-1)/(fold_k-1),0,1))
    return {'beta_raw':raw,'chance95':chance,'beta':beta,'a_MIA':a_mia,
            'permutations':reps,'permutation_seed':seed,'hash_serialization':'attacker ordered little-endian float64 columns(confidence,truth,rare), sha256 first8 little',
            'null_mean':float(null.mean()),'null_median':float(np.median(null)),
            'per_attackers':per,'null_values':null}

def make_cohorts(b,a,schema,seed,config):
    rng=np.random.default_rng(seed);J=config['pseudo_attackers'];n=config['members_per_attacker']
    b_rare=b.is_rare.to_numpy(bool);a_rare=local.rare_mask(a,schema)
    # Real shared blocks are disjoint; their identity is unknown, so sample once.
    # Keep official 0..2 rare members per block by rejection of the whole draw.
    for attempt in range(10000):
        member=rng.permutation(len(b))[:J*n].reshape(J,n)
        if np.max(b_rare[member].sum(axis=1))<=2:break
    else:raise RuntimeError('Could not make bounded-rare pseudo-cohorts')
    plans=[]
    for idx in member:
        r=int(b_rare[idx].sum());rare_n=config['rare_candidates_per_attacker']-r
        normal_n=config['candidates_per_attacker']-n-rare_n
        controls=np.r_[rng.choice(np.flatnonzero(a_rare),rare_n,replace=False),rng.choice(np.flatnonzero(~a_rare),normal_n,replace=False)]
        pool_idx=np.r_[idx,len(b)+controls]
        truth=np.r_[np.ones(n,int),np.zeros(len(controls),int)]
        rare=np.r_[b_rare[idx],a_rare[controls]]
        perm=rng.permutation(len(pool_idx))
        plans.append({'pool_indices':pool_idx[perm].tolist(),'truth':truth[perm].tolist(),'rare':rare[perm].tolist()})
    return {'seed':seed,'construction_attempts':attempt+1,'unique_members':int(np.unique(member).size),
            'rare_member_count':int(b_rare[member].sum()),'rare_members_by_attacker':b_rare[member].sum(axis=1).tolist(),
            'control_rare_label':'public detector surrogate, not known true type','attackers':plans}

def evaluate_aia(c,b,a,schema,cfg):
    query_b=b.drop(columns='is_rare');rare_b=b.is_rare.to_numpy(bool);rare_a=local.rare_mask(a,schema)
    attacks={}
    for label in ('official_qi_nearest','team_multiview_linkage'):
        if label=='official_qi_nearest':
            _,pm=aia_predict(c,query_b,a,cfg);_,pc=aia_predict(c,a,a,cfg)
        else:
            pm=local.team_multiview_aia(c,query_b,a);pc=local.team_multiview_aia(c,a,a)
        result=score_aia(None,pm,query_b.onset.to_numpy(),query_b.time.to_numpy(),None,pc,a.onset.to_numpy(),a.time.to_numpy(),cfg,member_rare_mask=rare_b,control_rare_mask=rare_a)
        attacks[label]={'risk_before_phi':result.score,'breakdown':result.breakdown}
    return {'attacks':attacks,'worst_before_phi':max(x['risk_before_phi'] for x in attacks.values()),'phi':None,
            'scope':'all B as targets, public A as controls; true B rare flags, detector proxy for public rare; not official AIA challenges'}

def main():
    config=read(HERE/'protocol.json');frozen=read(HERE/'inputs/frozen_inputs.json')
    if sha(HERE/'protocol.json')!=frozen['protocol_sha256'] or sha(Path(__file__))!=frozen['evaluator_sha256']:
        raise RuntimeError('Evaluation protocol/code differs from frozen input')
    for rel,h in frozen['team_input_sha256'].items():
        if sha(TEAM/rel)!=h:raise RuntimeError('Input changed: '+rel)
    cfg,environment=utility.environment(KIT)
    cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    b=pd.read_csv(TEAM/'participant_data/B_self.csv');a=pd.read_csv(TEAM/'participant_data/A_bg.csv')
    schema=read(TEAM/'participant_data/schema.json');pool=pd.concat([b.drop(columns='is_rare'),a],ignore_index=True)
    plans={}
    for seed in config['seeds']:
        plan=make_cohorts(b,a,schema,seed,config);plans[seed]=plan
        save(HERE/f'inputs/cohorts_{seed}.json',plan)
    rows=[]
    for name,relative in config['models'].items():
        start=time.monotonic();out=HERE/'results'/name;out.mkdir(parents=True,exist_ok=True)
        cpath=TEAM/relative;c=pd.read_csv(cpath)
        score_path=out/'utility.json'
        if score_path.exists():
            score=read(score_path)
            if score['provenance']['C_sha256']!=sha(cpath):raise RuntimeError('Cached utility input mismatch')
        else:
            score=utility.evaluate_csv(cpath);save(score_path,score)
        conf_path=out/'domias_confidence.npy'
        if conf_path.exists():conf=np.load(conf_path,allow_pickle=False)
        else:
            conf=overlap_mia(c,a,pool,cfg);np.save(conf_path,conf,allow_pickle=False)
        if not np.isfinite(conf).all() or conf.shape!=(len(pool),):raise RuntimeError('Bad confidences')
        aia_path=out/'aia.json'
        if aia_path.exists():aia=read(aia_path)
        else:aia=evaluate_aia(c,b,a,schema,cfg);save(aia_path,aia)
        u=float(score['score']['utility'])
        for seed,plan in plans.items():
            result_path=out/f'mia_{seed}.json'
            if result_path.exists():result=read(result_path)
            else:
                cs=[conf[x['pool_indices']] for x in plan['attackers']]
                ys=[np.array(x['truth'],int) for x in plan['attackers']]
                rs=[np.array(x['rare'],bool) for x in plan['attackers']]
                result=permutation_mia(cs,ys,rs,reps=config['permutations'],low_fpr=float(cfg['scoring']['low_fpr']),fold_k=float(cfg['scoring']['mia_fold_K']))
                np.save(out/f'null_{seed}.npy',result.pop('null_values'),allow_pickle=False)
                save(result_path,result)
            anon=1-max(result['a_MIA'],aia['worst_before_phi'])
            row={'model':name,'seed':seed,'U':u,**{k:result[k] for k in ('beta_raw','chance95','beta','a_MIA')},
                 'AIA_worst_before_phi':aia['worst_before_phi'],'Anon_proxy':anon,'Objective_proxy':min(u,anon)}
            rows.append(row);pd.DataFrame(rows).to_csv(HERE/'comparison_all_seeds.csv',index=False)
            print(json.dumps({**row,'elapsed_model_seconds':round(time.monotonic()-start,1)}),flush=True)
        save(out/'provenance.json',{'C_sha256':sha(cpath),'environment':environment,'frozen_inputs_sha256':sha(HERE/'inputs/frozen_inputs.json'),
                                  'confidence_sha256':sha(conf_path),'generator_run':False})
    table=pd.DataFrame(rows)
    primary=table.loc[table.seed.eq(config['seeds'][0])].sort_values('Objective_proxy',ascending=False,kind='stable')
    primary.to_csv(HERE/'comparison_primary.csv',index=False)
    sensitivity=table.groupby('model',sort=False).agg(U=('U','first'),Anon_min=('Anon_proxy','min'),Anon_max=('Anon_proxy','max'),Anon_mean=('Anon_proxy','mean'),Objective_min=('Objective_proxy','min'),Objective_max=('Objective_proxy','max'))
    sensitivity.to_csv(HERE/'sensitivity.csv')
    save(HERE/'completion.json',{'complete':True,'models':len(config['models']),'seeds':len(config['seeds']),
                                'permutations_per_model_seed':config['permutations'],'frozen_inputs_sha256':sha(HERE/'inputs/frozen_inputs.json'),
                                'comparison_sha256':sha(HERE/'comparison_all_seeds.csv')})

if __name__=='__main__':main()
