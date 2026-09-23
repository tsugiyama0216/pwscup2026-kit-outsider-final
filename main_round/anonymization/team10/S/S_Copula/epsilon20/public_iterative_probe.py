"""Numerical accuracy probe using PUBLIC A_bg splits only; no protected input."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; CORE=HERE.parent; TEAM=CORE.parents[1]
sys.path[:0]=[str(CORE),str(CORE/'postprocessing_v2')]
import dp_copula as dc
import dp_cox
import influence_cox as ic
from cox_optimizer import _fit

def main():
    data=TEAM/'participant_data'
    public_all=pd.read_csv(data/'A_bg.csv')
    schema=json.loads((data/'schema.json').read_text())
    for seed in (2026091811,2026091827):
        order=np.random.default_rng(seed).permutation(len(public_all))
        cohort=public_all.iloc[order[:1049]].reset_index(drop=True)
        public=public_all.iloc[order[1049:]].reset_index(drop=True)
        defaults={c:float(public[c].median()) for c in dc.NUMERIC}
        public=dc._prepare(public,schema,sorted(public.prefecture.unique()),defaults)
        cohort=dc._prepare(cohort,schema,sorted(public.prefecture.unique()),defaults)
        base=dp_cox.build_public_cox_model(public)
        from lifelines import CoxPHFitter
        x=(dp_cox._features(cohort)-base['center'])/base['scale']
        f=pd.DataFrame(x,columns=dp_cox.COX_COLUMNS);f['time']=cohort.time;f['onset']=cohort.onset
        true=CoxPHFitter().fit(f,'time','onset')
        beta=np.array(base['beta_standardized'])
        for step in range(5):
            model=ic.build_public_model(public,beta=beta,base_model=base)
            update=ic.influence_rows(cohort,model).mean(axis=0)
            beta=beta+update
            print(json.dumps({'public_seed':seed,'round':step+1,'beta_error':float(np.linalg.norm(beta-true.params_)),
                  'mean_abs_SE_error':float(np.mean(np.abs((beta-true.params_)/true.standard_errors_))),
                  'step_norm':float(np.linalg.norm(update))}),flush=True)

if __name__=='__main__':main()
