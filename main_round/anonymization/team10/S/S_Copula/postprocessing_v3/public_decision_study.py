"""Four cached PUBLIC-cohort tests of loss-aligned DP postprocessing.

No B, B_self, utility_ref, R output, or actual saved DP release is read.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE=Path(__file__).resolve().parent; CORE=HERE.parent; V2=CORE/'postprocessing_v2'
TEAM=CORE.parents[1]; WORK=CORE.parents[5]; KIT=WORK/'official-kit'
sys.path[:0]=[str(CORE),str(V2),str(KIT/'starter/src')]
import dp_cox as cx
import dp_copula as dc
from pipeline import construct
from cox_optimizer import calibrate
from pwscup2026.scoring.utility import _score_u_spec
from posterior_decision import decide,expected_contribution


def fit(table,public_model):
    x=(cx._features(table)-np.asarray(public_model['center']))/np.asarray(public_model['scale'])
    frame=pd.DataFrame(x,columns=cx.COX_COLUMNS)
    frame['time']=table.time.to_numpy(); frame['onset']=table.onset.to_numpy()
    model=CoxPHFitter().fit(frame,duration_col='time',event_col='onset')
    return model.params_.to_numpy(),model.standard_errors_.to_numpy()


def posterior_value(table,public_model,mean,covariance,true_se):
    beta,se=fit(table,public_model)
    sd=np.sqrt(np.maximum(np.diag(covariance),1e-16))
    contributions=[float(expected_contribution(b,m,s,t,c)) for b,m,s,t,c in zip(beta,mean,sd,true_se,se)]
    return float(np.mean(contributions)),beta,se


def refine_decision(start,posterior,public_model,detector,*,passes=3,settings=None):
    """Only public / saved-DP posterior and synthetic data are accepted."""
    settings=settings or {}
    mean=np.asarray(posterior['mean']); covariance=np.asarray(posterior['covariance'])
    true_se=np.asarray(posterior['true_se_public'])
    best=start.copy(); best_value,best_beta,best_se=posterior_value(best,public_model,mean,covariance,true_se)
    history=[{'pass':0,'expected_IO':best_value,'beta':best_beta.tolist(),'SE':best_se.tolist()}]
    for k in range(passes):
        wanted,decision=decide(mean,covariance,true_se,best_se,boundary_margin=.1)
        strata=(dc._radii(best,detector)>=float(detector['band']['lo'])).astype(int)
        candidate,trace=calibrate(best,wanted,public_model,strata=strata,
            iterations=settings.get('iterations',180),candidates=settings.get('swap_candidates',4096),
            trials=settings.get('swap_trials',12),seed=settings.get('postprocess_seed',20260918)+k,
            coefficient_tolerance=.002)
        value,beta,se=posterior_value(candidate,public_model,mean,covariance,true_se)
        accepted=value>best_value+1e-12
        history.append({'pass':k+1,'expected_IO':value,'accepted':accepted,
            'beta':beta.tolist(),'SE':se.tolist(),'decision':decision,'optimizer':trace})
        if accepted: best,best_value,best_beta,best_se=candidate,value,beta,se
    return best,{'passes':passes,'selected_expected_IO':best_value,'history':history,
        'selection_inputs':'Synthetic fit and public/DP posterior only; not actual cohort score'}


def main():
    destination=HERE/'public_decision_study.json'
    if destination.exists(): raise FileExistsError('Preserve completed public study')
    public_path=TEAM/'participant_data/A_bg.csv'; schema_path=TEAM/'participant_data/schema.json'
    all_public=pd.read_csv(public_path); schema=json.loads(schema_path.read_text(encoding='utf-8'))
    settings=json.loads((V2/'joint14_config.json').read_text(encoding='utf-8'))['settings']
    rows=[]
    for cohort_seed in (2026091811,2026091827):
        order=np.random.default_rng(cohort_seed).permutation(len(all_public))
        truth=all_public.iloc[order[:1049]].sort_values('record_id').reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        for replicate in (0,1):
            cache=WORK/f'public_postprocessing_v2/cohort{cohort_seed}_noise{replicate}'
            models=[json.loads((cache/f'epsilon{e}.json').read_text(encoding='utf-8')) for e in (1,3,10)]
            baseline,model,_=construct(models,settings)
            target=model['cox_target']; public_model=models[0]['public_cox']
            _,public_se=fit(public,public_model)
            posterior={'mean':target['beta_standardized'],
                'covariance':np.asarray(target['approximate_posterior_covariance'])[1:,1:],
                'true_se_public':public_se*np.sqrt(len(public)/1049)}
            result,trace=refine_decision(baseline,posterior,public_model,schema['rare_detector'],settings=settings)
            before,before_details=_score_u_spec(baseline,truth)
            after,after_details=_score_u_spec(result,truth)
            row={'cohort_seed':cohort_seed,'noise_replicate':replicate,
                'baseline_U_spec_public':before,'decision_U_spec_public':after,
                'baseline_details_public':before_details,'decision_details_public':after_details,
                'trace':trace}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('trace','baseline_details_public','decision_details_public')}),flush=True)
    result={'source':'PUBLIC A_bg only; four cached proxy draws', 'complete':True,'rows':rows,
        'baseline_mean_U_spec':float(np.mean([r['baseline_U_spec_public'] for r in rows])),
        'decision_mean_U_spec':float(np.mean([r['decision_U_spec_public'] for r in rows])),
        'source_sha256':{str(p.relative_to(CORE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [HERE/'posterior_decision.py',HERE/'public_decision_study.py']},
        'limitations':['Four reused public validation draws, not independent confirmation.',
                      'Gaussian coefficient posterior and public SE are approximations.',
                      'No inference about actual B utility without frozen separate evaluation.']}
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}),flush=True)


if __name__=='__main__': main()
