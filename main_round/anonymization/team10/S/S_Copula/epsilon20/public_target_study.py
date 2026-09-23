"""Seeded mechanism SIMULATION on public A_bg only. Never accepts a private path."""
from pathlib import Path
import sys,json,itertools
from unittest.mock import patch
from fractions import Fraction
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;CORE=HERE.parent;TEAM=CORE.parents[1]
sys.path[:0]=[str(CORE),str(CORE/'postprocessing_v2')]
import dp_copula as dc
import dp_cox
import influence_cox as ic
from cox_optimizer import _fit

def io_score(beta,se,true_beta,true_se):
    lo,hi=beta-1.959963984540054*se,beta+1.959963984540054*se
    tl,th=true_beta-1.959963984540054*true_se,true_beta+1.959963984540054*true_se
    overlap=np.maximum(0,np.minimum(hi,th)-np.maximum(lo,tl))
    io=.5*overlap*(1/(hi-lo)+1/(th-tl))
    sig=(lo>0)|(hi<0);ts=(tl>0)|(th<0)
    return float(np.mean(np.where((sig!=ts)|(sig & ts & (np.sign(beta)!=np.sign(true_beta))),0,io)))

def main():
    data=TEAM/'participant_data';all_public=pd.read_csv(data/'A_bg.csv')
    schema=json.loads((data/'schema.json').read_text());rows=[]
    for cohort_seed in (2026091811,2026091827):
        order=np.random.default_rng(cohort_seed).permutation(len(all_public))
        proxy=all_public.iloc[order[:1049]].reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        defaults={c:float(public[c].median()) for c in dc.NUMERIC};cats=sorted(public.prefecture.unique())
        public=dc._prepare(public,schema,cats,defaults);proxy=dc._prepare(proxy,schema,cats,defaults)
        base=dp_cox.build_public_cox_model(public)
        x=(dp_cox._features(proxy)-base['center'])/base['scale']
        truth,tse=_fit(x,proxy.time,proxy.onset)
        for hist,rounds,quantile,damp in itertools.product((4,8),(1,2,3),(.90,.97),(.7,1.0)):
            available=20-hist
            fractions={1:[1],2:[.3,.7],3:[.2,.3,.5]}[rounds]
            budgets=[str(Fraction(available)*Fraction(str(f))) for f in fractions]
            plan={'histogram_epsilon':str(hist),'cox_round_epsilons':budgets,'clip_quantile':quantile,'damping':damp,'quantization':4096}
            for replicate in (0,1,2):
                rng=np.random.default_rng(cohort_seed+replicate*10000)
                def public_noise(values,epsilon,sensitivity):
                    p=float(Fraction(epsilon)/(sensitivity+Fraction(epsilon)))
                    v=np.asarray(values,dtype=object)+rng.geometric(p,9)-rng.geometric(p,9)
                    return [int(x) for x in v],{'PUBLIC_SIMULATION_ONLY':True}
                with patch.object(dc,'release_integer_vector',public_noise):
                    target=ic.measure(proxy,public,plan)
                row={'plan':plan,'public_cohort_seed':cohort_seed,'replicate':replicate,
                     'ideal_target_IO_public':io_score(np.asarray(target['beta_standardized']),np.asarray(target['standard_errors_standardized']),truth,tse)}
                rows.append(row)
        print(json.dumps({'public_cohort_done':cohort_seed,'trials':len(rows)}),flush=True)
    summaries=[]
    for identity in sorted({json.dumps(r['plan'],sort_keys=True) for r in rows}):
        vals=[r['ideal_target_IO_public'] for r in rows if json.dumps(r['plan'],sort_keys=True)==identity]
        summaries.append({'plan':json.loads(identity),'mean':float(np.mean(vals)),'minimum':min(vals),'n':len(vals)})
    summaries.sort(key=lambda x:(x['mean'],x['minimum']),reverse=True)
    result={'source':'PUBLIC A_bg only; target IO, not actual full utility','rows':rows,'summaries':summaries,
            'limitations':['2 public cohorts x 3 simulated noise draws; selection bias; no utility guarantee']}
    (HERE/'public_target_study.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries[:8],indent=2),flush=True)

if __name__=='__main__':main()
