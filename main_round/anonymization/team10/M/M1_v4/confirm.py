"""Frozen-candidate comparison; confirmation results never retune the model."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiment import HERE, TEAM, BASE, OLD, STATIC, sha, read, save, evaluation, check_invariants

def raw_summary(conf,plans,cfg):
    values=[]
    for plan in plans['attackers']:
        ids=np.asarray(plan['pool_indices']); y=np.asarray(plan['truth']); rare=np.asarray(plan['rare'],bool)
        score=evaluation.score_mia(conf[ids],y,cfg,rare_mask=rare)
        values.append({'raw':score.score,'auc':roc_auc_score(y,conf[ids]),**score.breakdown})
    return {'beta_raw':float(np.mean([v['raw'] for v in values])),
            'mean_auc':float(np.mean([v['auc'] for v in values])),'per_attackers':values}

def main():
    protocol=read(HERE/'protocol.json'); selection=read(HERE/'selection.json')
    if sha(BASE)!=protocol['base_sha256'] or sha(HERE/'C.csv')!=selection['selected_csv_sha256']:
        raise RuntimeError('Selected or base CSV changed')
    for name,h in protocol['inputs_sha256'].items():
        if sha(TEAM/'participant_data'/name)!=h: raise RuntimeError('Input changed: '+name)
    for name,h in protocol['dependencies_sha256'].items():
        if sha(TEAM/name)!=h: raise RuntimeError('Dependency changed: '+name)
    cfg,environment=evaluation.utility.environment()
    schema=read(TEAM/'participant_data/schema.json')
    b=pd.read_csv(TEAM/'participant_data/B_self.csv'); a=pd.read_csv(TEAM/'participant_data/A_bg.csv')
    pool=pd.concat([b.drop(columns='is_rare'),a],ignore_index=True)
    c0=pd.read_csv(BASE); c=pd.read_csv(HERE/'C.csv')
    rare=evaluation.local.rare_mask(c0,schema); check_invariants(c0,c,rare)
    out=HERE/'confirmation'; out.mkdir(parents=True,exist_ok=True)
    frozen={'selected_sha256':sha(HERE/'C.csv'),'selection_sha256':sha(HERE/'selection.json'),
            'confirmation_code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'protocol.json'),
            'environment':environment,'base_submission_zip_sha256':sha(BASE.parent/'submission/team10_M1_v3_defense.zip')}
    if (out/'frozen.json').exists() and read(out/'frozen.json')!=frozen: raise RuntimeError('Confirmation changed')
    save(out/'frozen.json',frozen)
    start=time.monotonic()
    # Recompute the entire candidate-pool confidence vector for both models.
    d0=evaluation.overlap_mia(c0,a,pool,cfg); d1=evaluation.overlap_mia(c,a,pool,cfg)
    if not np.array_equal(d0,d1): raise RuntimeError('DOMIAS confidence changed despite fixed inputs')
    np.save(out/'domias_confidence.npy',d1)
    previous=np.load(OLD/'results/M1_v3/domias_confidence.npy')
    if not np.array_equal(d0,previous): raise RuntimeError('Baseline differs from previous evaluation')
    control=np.arange(len(pool))>=len(b)
    multi={}
    for name,frame in [('M1_v3',c0),('M1_v4',c)]:
        conf,detail=evaluation.local.team_multiview_mia(frame,pool,a,control)
        np.save(out/f'{name}_multiview_confidence.npy',conf); multi[name]=conf
        save(out/f'{name}_multiview_details.json',detail)
    rows=[]; original_config=read(OLD/'protocol.json')
    base_aia=read(OLD/'results/M1_v3/aia.json')['worst_before_phi']
    new_aia=read(HERE/'candidates'/selection['selected']/'aia.json')['worst_before_phi']
    u0=selection['baseline']['U']; u1=next(r['U'] for r in selection['results'] if r['name']==selection['selected'])
    for seed in protocol['exploratory_mia_seeds']+protocol['confirmation_mia_seeds']:
        is_new=seed in protocol['confirmation_mia_seeds']
        plan=(evaluation.make_cohorts(b,a,schema,seed,original_config) if is_new else read(OLD/f'inputs/cohorts_{seed}.json'))
        save(out/f'cohorts_{seed}.json',plan)
        if is_new:
            result_file=out/f'domias_{seed}.json'
            if result_file.exists(): domias=read(result_file)
            else:
                conf=[d1[np.asarray(p['pool_indices'])] for p in plan['attackers']]
                truth=[np.asarray(p['truth']) for p in plan['attackers']]
                rares=[np.asarray(p['rare'],bool) for p in plan['attackers']]
                domias=evaluation.permutation_mia(conf,truth,rares,reps=20000,low_fpr=.01,fold_k=12)
                np.save(out/f'null_{seed}.npy',domias.pop('null_values')); save(result_file,domias)
        else:
            domias=read(OLD/f'results/M1_v3/mia_{seed}.json')
            save(out/f'domias_{seed}.json',domias)
        for name,u,aia in [('M1_v3',u0,base_aia),('M1_v4',u1,new_aia)]:
            aux=raw_summary(multi[name],plan,cfg); save(out/f'{name}_multiview_{seed}.json',aux)
            anon=1-max(domias['a_MIA'],aia)
            rows.append({'model':name,'seed':seed,'new_confirmation_seed':is_new,'U':u,'DOMIAS_beta_raw':domias['beta_raw'],
                         'chance95':domias['chance95'],'DOMIAS_beta':domias['beta'],'a_MIA':domias['a_MIA'],
                         'AIA_before_phi':aia,'Anon_proxy':anon,'Objective_proxy':min(u,anon),
                         'additional_multiview_MIA_raw':aux['beta_raw'],'additional_multiview_MIA_mean_AUC':aux['mean_auc']})
        print(json.dumps({'seed':seed,'new_confirmation':is_new,'DOMIAS_a_MIA':domias['a_MIA'],
                          'multiview_raw_v3':rows[-2]['additional_multiview_MIA_raw'],
                          'multiview_raw_v4':rows[-1]['additional_multiview_MIA_raw']}),flush=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    # Independently rerun selected candidate scoring and AIA after frozen selection.
    score=evaluation.utility.evaluate_csv(HERE/'C.csv'); save(HERE/'utility.json',score)
    if not np.isclose(score['score']['utility'],u1,rtol=0,atol=1e-12): raise RuntimeError('Utility changed')
    cfg.setdefault('aia',{}).update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
    aia=evaluation.evaluate_aia(c,b,a,schema,cfg); save(HERE/'aia.json',aia)
    if aia['worst_before_phi']!=new_aia: raise RuntimeError('AIA changed')
    save(out/'completion.json',{'complete':True,'rows':len(rows),'DOMIAS_all_pool_confidences_identical':True,
         'previous_M1_v3_DOMIAS_confidences_reproduced':True,'new_seeds':protocol['confirmation_mia_seeds'],
         'frozen_selected_sha256':sha(HERE/'C.csv'),'base_csv_unchanged':sha(BASE)==protocol['base_sha256'],
         'base_submission_zip_unchanged':sha(BASE.parent/'submission/team10_M1_v3_defense.zip')==frozen['base_submission_zip_sha256'],
         'elapsed_seconds':time.monotonic()-start,'scope':'Local proxy, new candidate subsets of the same B/A data; not independent data holdout or official scores',
         'additional_mia_scope':'Multiview MIA includes time/onset/death; raw TPR and AUC diagnostics only, no permutation normalization and not included in Anon proxy'})

if __name__=='__main__': main()
