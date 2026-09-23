"""Compare public-only histogram simulations independently of protected B."""
from pathlib import Path
import sys,json,itertools
from unittest.mock import patch
from fractions import Fraction
import numpy as np
import pandas as pd
import yaml
HERE=Path(__file__).resolve().parent;CORE=HERE.parent;TEAM=CORE.parents[1];WORK=CORE.parents[5]
KIT=WORK/'official-kit';sys.path[:0]=[str(CORE),str(CORE/'postprocessing_v2'),str(KIT/'starter/src')]
import dp_copula as dc
from histogram_refine import refine_models,sample_refined
from pwscup2026.scoring.utility import UtilityReference,_score_u_time,_score_u_gen,_score_u_rare

def main():
    data=TEAM/'participant_data';all_public=pd.read_csv(data/'A_bg.csv')
    schema=json.loads((data/'schema.json').read_text());detector=schema['rare_detector']
    cfg=yaml.safe_load((KIT/'starter/src/pwscup2026/kit/kit_config.yaml').read_text(encoding='utf-8'));rows=[]
    for cohort_seed in (2026091811,2026091827):
        order=np.random.default_rng(cohort_seed).permutation(len(all_public))
        proxy=all_public.iloc[order[:1049]].sort_values('record_id').reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        ref=UtilityReference(mu=np.asarray(detector['mu']),sigma=np.asarray(detector['sigma']),band=detector['band'],
            b_is_rare=dc._radii(proxy,detector)>=float(detector['band']['lo']),a_bg=public)
        for hist,rep in itertools.product((4,8),(0,1)):
            rng=np.random.default_rng(cohort_seed+10000*rep)
            def simulated(counts,query):
                a=np.asarray(counts);q=Fraction(query['geometric_q_numerator'],query['geometric_q_denominator'])
                return (a+rng.geometric(float(1-q),a.shape)-rng.geometric(float(1-q),a.shape)).tolist()
            with patch.object(dc,'_noise_counts',simulated):
                original=dc.fit(proxy,public,schema,hist)
            model=refine_models([original])
            for radial in (0,1):
                c=sample_refined(model,seed=20260918,public_radial_strength=radial)
                c=c.sort_values(list(c.columns),kind='stable').reset_index(drop=True)
                c.insert(0,'record_id',np.arange(1,1050))
                t,td=_score_u_time(c,proxy,ref,cfg)
                gen,detail=_score_u_gen(c,proxy,t,td,ref,cfg)
                rare,_=_score_u_rare(c,proxy,c,proxy,ref,cfg)
                row={'histogram_epsilon':hist,'radial':radial,'cohort_seed':cohort_seed,'replicate':rep,
                     'U_gen_public_KM_floor0':gen,'U_rare_public_detector_proxy':rare,
                     'gen_without_KM':min(detail[k] for k in ('marginal_mean','corr_score','pmse_score','tail_score'))}
                rows.append(row)
            print(json.dumps({'public_cohort':cohort_seed,'hist_epsilon':hist,'replicate':rep}),flush=True)
    summaries=[]
    for hist,radial in itertools.product((4,8),(0,1)):
        rs=[r for r in rows if r['histogram_epsilon']==hist and r['radial']==radial]
        summaries.append({'histogram_epsilon':hist,'radial':radial,'mean_general_without_KM':float(np.mean([r['gen_without_KM'] for r in rs])),
                          'mean_rare_proxy':float(np.mean([r['U_rare_public_detector_proxy'] for r in rs])),
                          'mean_gen_floor0':float(np.mean([r['U_gen_public_KM_floor0'] for r in rs]))})
    result={'source':'PUBLIC A_bg only, seeded simulator','rows':rows,'summaries':summaries,
            'limitations':['Rare truth is detector proxy; KM floor zero; not actual utility; 2 public cohorts x 2 noise draws.']}
    (HERE/'public_histogram_study.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
