"""Public-only model/optimizer selection, never reads private contest inputs."""
from pathlib import Path
import sys,json,hashlib
from fractions import Fraction
import numpy as np
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent; CORE=HERE.parent; TEAM=HERE.parents[2]
WORK=HERE.parents[6]; KIT=WORK/'official-kit'; CACHE=WORK/'public_postprocessing_v2'
sys.path[:0]=[str(CORE),str(KIT/'starter/src')]
import dp_copula as dc
import dp_cox as cx
from pipeline import construct
from pwscup2026.kit.selfscore import self_score
from pwscup2026.scoring.utility import UtilityReference

BASE={'histogram_prior_multiplier':1.0,'cox_prior_multiplier':1.0,
    'histogram_beta_weight':0.0,'large_sample_n':10490,'radial_strength':0.0,
    'postprocess_seed':20260918,'iterations':180,'swap_candidates':4096,
    'swap_trials':12,'coefficient_tolerance':.005}


def proxy_models(protected,public,schema,folder):
    folder.mkdir(parents=True,exist_ok=True)
    paths=[folder/f'epsilon{e}.json' for e in (1,3,10)]
    if all(p.exists() for p in paths):
        return [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    if any(p.exists() for p in paths):
        raise RuntimeError('Partial public-only model cache; inspect before rebuilding')
    categories=sorted(public.prefecture.unique())
    defaults={k:float(public[k].median()) for k in dc.NUMERIC}
    clean_public=dc._prepare(public,schema,categories,defaults)
    clean_proxy=dc._prepare(protected,schema,categories,defaults)
    pub=cx.build_public_cox_model(clean_public)
    models=[]
    for epsilon,path in zip((1,3,10),paths):
        copula=dc.fit(protected,public,schema,Fraction(epsilon)*Fraction(4,5))
        mechanism=None
        def noise(values,sensitivity,budget):
            nonlocal mechanism
            result,mechanism=dc.release_integer_vector(values,budget,sensitivity)
            return result
        target=cx.measure_cox_update(clean_proxy,pub,epsilon=Fraction(epsilon)/5,
            public_n=1049,noise_integer_vector=noise)
        model={'copula':copula,'public_cox':pub,'cox':target,'cox_mechanism':mechanism,
            'accounting':{'epsilon_exact':str(epsilon),'cox_epsilon_exact':str(Fraction(epsilon)/5)},
            'source':'PUBLIC A_bg proxy only; does not consume protected B budget'}
        path.write_text(json.dumps(model,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        models.append(model)
    return models


def main():
    sources=[HERE/name for name in ['histogram_refine.py','cox_optimizer.py','joint_cox.py','pipeline.py','public_pipeline_study.py']]+[CORE/name for name in ['dp_copula.py','dp_cox.py']]
    source_hashes={str(p.relative_to(CORE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    public_path=TEAM/'participant_data/A_bg.csv'; schema_path=TEAM/'participant_data/schema.json'
    all_public=pd.read_csv(public_path)
    schema=json.loads(schema_path.read_text(encoding='utf-8'))
    cfg=yaml.safe_load((KIT/'starter/src/pwscup2026/kit/kit_config.yaml').read_text(encoding='utf-8'))
    detector=schema['rare_detector']; rows=[]
    for cohort_seed in (2026091811,2026091827):
        order=np.random.default_rng(cohort_seed).permutation(len(all_public))
        protected=all_public.iloc[order[:1049]].sort_values('record_id').reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        truth=dc._radii(protected,detector)>=float(detector['band']['lo'])
        reference=UtilityReference(mu=np.asarray(detector['mu']),sigma=np.asarray(detector['sigma']),
            band=detector['band'],b_is_rare=truth,a_bg=public)
        for replicate in range(2):
            models=proxy_models(protected,public,schema,CACHE/f'cohort{cohort_seed}_noise{replicate}')
            for radial in (0.0,1.0):
                for blend in (0.0,.5,1.0):
                    settings={**BASE,'radial_strength':radial,'histogram_beta_weight':blend}
                    generated,model,original=construct(models,settings)
                    score=self_score(generated,protected,reference,cfg)
                    before=self_score(original,protected,reference,cfg)
                    row={'cohort_seed':cohort_seed,'noise_replicate':replicate,
                         'radial_strength':radial,'histogram_beta_weight':blend,
                         'utility_proxy':score['utility'],'facets_proxy':score['facets'],
                         'before_optimization_utility_proxy':before['utility'],
                         'before_optimization_facets_proxy':before['facets'],
                         'optimizer_trace':model['optimizer_trace']}
                    rows.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k!='optimizer_trace'}),flush=True)
                    (HERE/'public_pipeline_study.json').write_text(json.dumps({'complete':False,'rows':rows}),encoding='utf-8')
    summary=[]
    for radial in (0.0,1.0):
        for blend in (0.0,.5,1.0):
            selected=[r for r in rows if r['radial_strength']==radial and r['histogram_beta_weight']==blend]
            summary.append({'radial_strength':radial,'histogram_beta_weight':blend,
                'mean_utility_proxy':float(np.mean([r['utility_proxy'] for r in selected])),
                'minimum_utility_proxy':min(r['utility_proxy'] for r in selected),
                'mean_Cox_proxy':float(np.mean([r['facets_proxy']['U_spec'] for r in selected]))})
    # Frozen public-only deterministic selection: mean whole-utility, then worst-case,
    # then lower blend/radial complexity on exact ties. No actual private score.
    chosen=max(summary,key=lambda r:(r['mean_utility_proxy'],r['minimum_utility_proxy'],
                                    -r['histogram_beta_weight'],-r['radial_strength']))
    settings={**BASE,'radial_strength':chosen['radial_strength'],'histogram_beta_weight':chosen['histogram_beta_weight']}
    result={'complete':True,'source':'PUBLIC A_bg held-out proxies only',
            'selection_rule':'Max mean whole-utility across 2 cohorts x 2 noise draws, then minimum, then simpler blend/radial',
            'limitations':['Rare truth replaced by public detector; not actual main-round U.',
                          'KM floor zero, public split sample finite; no private-cohort guarantee.',
                          'Two cohorts x two noise realizations are limited public evidence.'],
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [public_path,schema_path]},
            'rows':rows,'summaries':summary,'chosen_settings':settings,'code_sha256':source_hashes}
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=source_hashes[str(p.relative_to(CORE))] for p in sources):
        raise RuntimeError('Source changed during public selection; configuration may not be frozen')
    (HERE/'public_pipeline_study.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'joint14_config.json').write_text(json.dumps({'version':1,'parent_epsilons':[1,3,10],
            'parent_epsilon_exact':'14','additional_epsilon_exact':'0','settings':settings,
            'selection_rule':result['selection_rule']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('PUBLIC CHOSEN',json.dumps(chosen),flush=True)

if __name__=='__main__':main()
