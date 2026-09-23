"""Select additional epsilon using PUBLIC-cohort proxies only.

Study outputs are resumable public caches. Never point this script at B.
The production runner is separate and uses an immutable private reservation.
"""
from pathlib import Path
import sys,json,hashlib
from fractions import Fraction
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; CORE=HERE.parent; V2=CORE/'postprocessing_v2'
TEAM=CORE.parents[1]; WORK=CORE.parents[5]; KIT=WORK/'official-kit'
sys.path[:0]=[str(CORE),str(V2),str(KIT/'starter/src')]
import dp_copula as dc
from pipeline import construct
from pwscup2026.scoring.utility import _score_u_spec
from dp_cox_target import build_public_target,measure_target
from target_calibration import calibrate_target

EPSILONS=(50,100,200,400,800)
COHORTS=(2026091811,2026091827)
PLAN={'radius':16,'quantization':65536,'se_weight':2,
      'postprocess_seed':20260918,'iterations':300,'candidates':4096,'trials':12}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def assert_preservation(before,after,detector):
    outcomes=['time','onset','death']; predictors=[c for c in before.columns if c not in outcomes]
    pd.testing.assert_frame_equal(before[predictors],after[predictors])
    strata=dc._radii(before,detector)>=float(detector['band']['lo'])
    for group in (False,True):
        b=before.loc[strata==group,outcomes].sort_values(outcomes).reset_index(drop=True)
        a=after.loc[strata==group,outcomes].sort_values(outcomes).reset_index(drop=True)
        pd.testing.assert_frame_equal(b,a)


def main():
    sources=[HERE/'dp_cox_target.py',HERE/'target_calibration.py',HERE/'public_budget_study.py',
             CORE/'dp_copula.py',CORE/'dp_cox.py',V2/'pipeline.py',V2/'histogram_refine.py',
             V2/'joint_cox.py',V2/'cox_optimizer.py']
    hashes={str(p.relative_to(CORE)):sha(p) for p in sources}
    public_path=TEAM/'participant_data/A_bg.csv'; schema_path=TEAM/'participant_data/schema.json'
    all_public=pd.read_csv(public_path); schema=read(schema_path); detector=schema['rare_detector']
    old_settings=read(V2/'joint14_config.json')['settings']
    cache=WORK/'expanded_budget_public'; cache.mkdir(parents=True,exist_ok=True)
    identity={'code_sha256':hashes,'plan':PLAN,'public_input_sha256':{
        public_path.name:sha(public_path),schema_path.name:sha(schema_path)}}
    frozen_path=cache/'study_identity.json'
    if frozen_path.exists() and read(frozen_path)!=identity:
        raise RuntimeError('Public study code changed; preserve existing study cache before rerunning')
    if not frozen_path.exists(): save(frozen_path,identity)
    cohorts={}; baselines={}
    for cohort_seed in COHORTS:
        order=np.random.default_rng(cohort_seed).permutation(len(all_public))
        cohort=all_public.iloc[order[:1049]].sort_values('record_id').reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        defaults={c:float(public[c].median()) for c in dc.NUMERIC}
        categories=sorted(public.prefecture.unique())
        clean_public=dc._prepare(public,schema,categories,defaults)
        clean_cohort=dc._prepare(cohort,schema,categories,defaults)
        public_target=build_public_target(clean_public,public_n=1049)
        cohorts[cohort_seed]=(cohort,clean_cohort,public_target)
        for base_noise in (0,1):
            folder=WORK/f'public_postprocessing_v2/cohort{cohort_seed}_noise{base_noise}'
            models=[read(folder/f'epsilon{e}.json') for e in (1,3,10)]
            baseline,_,_=construct(models,old_settings)
            baselines[(cohort_seed,base_noise)]=baseline
    rows=[]; summaries=[]; selected=None
    for epsilon in EPSILONS:
        for cohort_seed in COHORTS:
            cohort,clean_cohort,public_target=cohorts[cohort_seed]
            for replicate in (0,1,2):
                entry=cache/f'epsilon{epsilon}_cohort{cohort_seed}_noise{replicate}'
                target_path=entry/'target.json'; result_path=entry/'result.json'
                baseline=baselines[(cohort_seed,replicate%2)]
                if target_path.exists(): target=read(target_path)
                else:
                    target=measure_target(clean_cohort,public_target,epsilon=Fraction(epsilon),
                        radius=PLAN['radius'],quantization=PLAN['quantization'],se_weight=PLAN['se_weight'])
                    save(target_path,target)
                if result_path.exists(): row=read(result_path)
                else:
                    candidate,trace=calibrate_target(baseline,target,public_target,detector,
                        seed=PLAN['postprocess_seed'],iterations=PLAN['iterations'],
                        candidates=PLAN['candidates'],trials=PLAN['trials'])
                    assert_preservation(baseline,candidate,detector)
                    score,details=_score_u_spec(candidate,cohort)
                    row={'additional_epsilon':epsilon,'cohort_seed':cohort_seed,'noise_replicate':replicate,
                         'base_noise_replicate':replicate%2,'U_spec_public':score,
                         'details_public':details,'trace':trace,'target_sha256':sha(target_path)}
                    save(result_path,row)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('details_public','trace')}),flush=True)
        values=[r['U_spec_public'] for r in rows if r['additional_epsilon']==epsilon]
        summary={'additional_epsilon':epsilon,'composed_epsilon':14+epsilon,'n':len(values),
                 'mean_U_spec_public':float(np.mean(values)),'minimum_U_spec_public':float(min(values)),
                 'attained_0_9':sum(x>=.9 for x in values)}
        summaries.append(summary)
        print('BUDGET SUMMARY '+json.dumps(summary),flush=True)
        if summary['mean_U_spec_public']>=.95 and summary['minimum_U_spec_public']>=.92:
            selected=epsilon; break
    if any(sha(CORE/name)!=value for name,value in hashes.items()):
        raise RuntimeError('Source changed during public study')
    result={'complete':True,'source':'PUBLIC team10 A_bg only, no B read',**identity,
        'selection_rule':'First epsilon in 50/100/200/400/800 with mean Cox>=.95 and min Cox>=.92 over six public trials',
        'rows':rows,'summaries':summaries,'selected_additional_epsilon':selected,
        'limitations':['Two public cohorts and three target-noise realizations; not a success guarantee.',
            'Base histograms use two existing public-noise replicates, reused cyclically.',
            'Public Cox score only; U_gen and U_rare inputs are preserved, but U_valid can change with onset associations and novelty. Full private validation remains separate.',
            'Selection uses public proxy scores only. Actual private evaluation stays separate.']}
    save(HERE/'public_budget_study.json',result)
    if selected is not None:
        save(HERE/'expanded_config.json',{'version':1,'additional_epsilon_exact':str(selected),
             'previous_epsilon_exact':'14','composed_epsilon_exact':str(14+selected),'delta':0,
             'public_n':1049,'plan':PLAN,'public_selection_sha256':sha(HERE/'public_budget_study.json'),
             'selection_rule':result['selection_rule']})


if __name__=='__main__': main()
