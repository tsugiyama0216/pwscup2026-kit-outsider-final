"""Candidate competition around released rows, from public inputs only."""
import numpy as np
from scipy.stats import rankdata

from ..features import CLINICAL,FEATURES,canonical,distance
from ..v2.features import extract_mia


def extract(c,background,candidates,baseline):
    x,rule=extract_mia(c,background,candidates)
    p=np.clip(baseline.predict_proba(x)[:,1],1e-7,1-1e-7)
    release=canonical(c,FEATURES)
    query=candidates[FEATURES].reset_index(drop=True)
    features=[p,np.log(p/(1-p)),rule]
    for logs,blocks in ((False,('smoking','onset','death')),(True,('smoking','onset','death')),
                        (True,())):
        d=distance(query,release,background,CLINICAL+['time'],logs,blocks)
        rows=np.arange(len(query))
        closest=d.argmin(axis=1)
        best=d[rows,closest]
        second=np.partition(d,1,axis=1)[:,1]
        column_best=d.min(axis=0)
        column_second=np.partition(d,1,axis=0)[1]
        column_choice=d.argmin(axis=0)
        reciprocal=column_choice[closest]==rows
        competitor=np.where(reciprocal,column_second[closest],column_best[closest])
        count=np.bincount(closest,minlength=len(release))[closest]
        features.extend([np.log1p(best),np.log1p(second),
            np.log1p(second)-np.log1p(best),
            np.log1p(competitor)-np.log1p(best),
            np.log(np.maximum(competitor,1e-9)/np.maximum(best,1e-9)).clip(-20,20),
            reciprocal.astype(float),np.log1p(count),
            rankdata(-best,method='average')/len(query)])
    result=np.column_stack(features).astype(np.float32)
    if not np.isfinite(result).all():raise ArithmeticError('Nonfinite MIA competition features')
    return result
