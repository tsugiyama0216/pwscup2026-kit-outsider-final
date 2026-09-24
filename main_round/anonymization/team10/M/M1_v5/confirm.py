"""Exact per-model permutation evaluation after freezing the v5 selection."""
from pathlib import Path
import time,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment import HERE,TEAM,BASE,V4,OLD,TRAIN,CONFIRM,sha,read,save,evaluation

def multiview_raw(conf,plan,cfg):
    per=[]
    for p in plan['attackers']:
        ids=np.asarray(p['pool_indices']);y=np.asarray(p['truth']);r=np.asarray(p['rare'],bool)
        score=evaluation.score_mia(conf[ids],y,cfg,rare_mask=r)
        per.append({'raw':score.score,'auc':roc_auc_score(y,conf[ids]),**score.breakdown})
    return {'raw':float(np.mean([p['raw'] for p in per])), 'auc':float(np.mean([p['auc'] for p in per])),'per_attackers':per}

def main():
    protocol=read(HERE/'protocol.json');selection=read(HERE/'screening.json');cfg,env=evaluation.utility.environment()
    if sha(BASE)!=protocol['base_sha256'] or sha(HERE/'C.csv')!=selection['selected_sha256']:raise RuntimeError('Frozen CSV changed')
    if sha(HERE/'experiment.py')!=protocol['code_sha256']:raise RuntimeError('Experiment code changed')
    for name,h in protocol['inputs_sha256'].items():
        if sha(TEAM/'participant_data'/name)!=h:raise RuntimeError('Input changed')
    for name,h in protocol['dependency_sha256'].items():
        if sha(TEAM/name)!=h:raise RuntimeError('Dependency changed')
    cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    schema=read(TEAM/'participant_data/schema.json');b=pd.read_csv(TEAM/'participant_data/B_self.csv');a=pd.read_csv(TEAM/'participant_data/A_bg.csv')
    c0=pd.read_csv(BASE);c1=pd.read_csv(HERE/'C.csv');pool=pd.concat([b.drop(columns='is_rare'),a],ignore_index=True)
    out=HERE/'confirmation';out.mkdir(parents=True,exist_ok=True)
    frozen={'v4_sha256':sha(BASE),'v5_sha256':sha(HERE/'C.csv'),'screening_sha256':sha(HERE/'screening.json'),
            'protocol_sha256':sha(HERE/'protocol.json'),'confirmation_code_sha256':sha(Path(__file__)),
            'known_seeds':TRAIN,'new_seeds':CONFIRM,'environment':env,
            'v3_csv_sha256':sha(V4.parent/'M1_v3/C.csv'),'v3_zip_sha256':sha(V4.parent/'M1_v3/submission/team10_M1_v3_defense.zip')}
    if (out/'frozen.json').exists() and read(out/'frozen.json')!=frozen:raise RuntimeError('Confirmation inputs changed')
    save(out/'frozen.json',frozen)
    conf={};multi={};scores={};aias={};control=np.arange(len(pool))>=len(b)
    start=time.monotonic()
    for label,frame,path in [('M1_v4',c0,BASE),('M1_v5',c1,HERE/'C.csv')]:
        conf[label]=evaluation.overlap_mia(frame,a,pool,cfg)
        np.save(out/f'{label}_domias_confidence.npy',conf[label])
        expected=(V4/'confirmation/domias_confidence.npy' if label=='M1_v4' else HERE/'candidates'/selection['selected']/'domias_confidence.npy')
        if not np.array_equal(conf[label],np.load(expected)):raise RuntimeError('Stored confidences do not reproduce')
        scores[label]=evaluation.utility.evaluate_csv(path);save(out/f'{label}_utility.json',scores[label])
        aias[label]=evaluation.evaluate_aia(frame,b,a,schema,cfg);save(out/f'{label}_aia.json',aias[label])
        multi[label],details=evaluation.local.team_multiview_mia(frame,pool,a,control)
        np.save(out/f'{label}_multiview_confidence.npy',multi[label]);save(out/f'{label}_multiview_details.json',details)
    rows=[];setting=read(OLD/'protocol.json')
    for seed in TRAIN+CONFIRM:
        plan=read(V4/f'confirmation/cohorts_{seed}.json') if seed in TRAIN else evaluation.make_cohorts(b,a,schema,seed,setting)
        save(out/f'cohorts_{seed}.json',plan)
        for label in ('M1_v4','M1_v5'):
            folder=out/label; folder.mkdir(parents=True,exist_ok=True);result_file=folder/f'domias_{seed}.json'
            if result_file.exists():result=read(result_file)
            elif label=='M1_v4' and seed in TRAIN:
                result=read(V4/f'confirmation/domias_{seed}.json');save(result_file,result)
            else:
                cs=[conf[label][p['pool_indices']] for p in plan['attackers']]
                ys=[np.asarray(p['truth']) for p in plan['attackers']];rs=[np.asarray(p['rare'],bool) for p in plan['attackers']]
                result=evaluation.permutation_mia(cs,ys,rs,reps=20000,low_fpr=.01,fold_k=12)
                np.save(folder/f'null_{seed}.npy',result.pop('null_values'));save(result_file,result)
            aux=multiview_raw(multi[label],plan,cfg);save(folder/f'multiview_{seed}.json',aux)
            u=scores[label]['score']['utility'];aia=aias[label]['worst_before_phi'];anon=1-max(result['a_MIA'],aia)
            row={'model':label,'seed':seed,'new_confirmation_seed':seed in CONFIRM,'U':u,
                 **{k:result[k] for k in ['beta_raw','chance95','beta','a_MIA']},'AIA_before_phi':aia,
                 'Anon_proxy':anon,'Objective_proxy':min(u,anon),'additional_multiview_raw':aux['raw'],'additional_multiview_auc':aux['auc']}
            rows.append(row);pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
            print(json.dumps({'model':label,'seed':seed,'Anon_proxy':anon,'Objective_proxy':min(u,anon),'additional_MIA_raw':aux['raw'],'elapsed':round(time.monotonic()-start,1)}),flush=True)
    save(HERE/'utility.json',scores['M1_v5']);save(HERE/'aia.json',aias['M1_v5'])
    save(out/'completion.json',{'complete':True,'rows':len(rows),'selected_sha256':sha(HERE/'C.csv'),
        'permutations_per_model_per_seed':20000,'original_v4_unchanged':sha(BASE)==frozen['v4_sha256'],
        'v3_csv_unchanged':sha(V4.parent/'M1_v3/C.csv')==frozen['v3_csv_sha256'],
        'v3_submission_zip_unchanged':sha(V4.parent/'M1_v3/submission/team10_M1_v3_defense.zip')==frozen['v3_zip_sha256'],
        'new_seeds_used_for_tuning':False,'scope':'Local B/A proxy; new candidate subsets, not independent data holdout or official score',
        'additional_mia_scope':'Raw TPR and AUC, no chance correction, not included in Anon proxy','elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':main()
