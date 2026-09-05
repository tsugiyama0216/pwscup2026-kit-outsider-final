"""Leakage-controlled own-challenge membership classifier feasibility test.

Ground truth is only the exact match to the user's actual B in self challenge24.
No other target labels or leaderboard labels are constructed or used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from audit_actual_inputs import ACTUAL,TARGETS

OUT=Path(__file__).resolve().parent
QI=['age','SBP','TG','HDL','ALT','sex','prefecture']

def features(frame, background):
    columns=['age','SBP','TG','HDL','ALT']
    reference=background[columns].to_numpy(dtype=float)
    values=frame[columns].to_numpy(dtype=float)
    for c in (2,4):
        reference[:,c]=np.log1p(reference[:,c])
        values[:,c]=np.log1p(values[:,c])
    center=np.median(reference,axis=0)
    scale=np.maximum(np.std(reference,axis=0),1e-9)
    z=(values-center)/scale
    precision=np.linalg.pinv(np.cov(((reference-center)/scale).T))
    mahal=np.einsum('ij,jk,ik->i',z,precision,z)
    ranks=[]
    for j in range(len(columns)):
        r=np.sort(reference[:,j]); v=values[:,j]
        ranks.append((np.searchsorted(r,v,'left')+np.searchsorted(r,v,'right'))/(2*len(r)))
    return np.column_stack([z,frame.sex.to_numpy(dtype=float),mahal,np.column_stack(ranks)])

def model(name):
    if name=='linear':
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000))
    if name=='quadratic':
        return make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),LogisticRegression(C=.01,max_iter=1000))
    if name=='trees':
        return ExtraTreesClassifier(n_estimators=300,min_samples_leaf=12,max_features=.8,random_state=2801,n_jobs=1)
    if name=='boost':
        return HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=10,learning_rate=.04,random_state=2801)
    raise ValueError(name)

def main():
    b=pd.read_csv(ACTUAL/'B.csv')
    bg=pd.read_csv(ACTUAL/'A_bg.csv')
    challenge=pd.read_csv(TARGETS/'aia_challenge_24.csv')
    y=pd.MultiIndex.from_frame(challenge[QI]).isin(pd.MultiIndex.from_frame(b[QI])).astype(int)
    assert y.sum()==163 and len(y)==326
    x=features(challenge,bg)
    pref=challenge.prefecture.astype(str).to_numpy()
    # A real team cohort can contain only one prefecture. Keep entire visible
    # matching blocks together so paired controls cannot leak across folds.
    bands=np.floor((challenge.age.to_numpy()-40)/5).astype(int)
    groups=np.array([f'{p}:{s}:{a}' for p,s,a in zip(pref,challenge.sex,bands)])
    folds=list(GroupKFold(5).split(x,y,groups))
    rows=[]; oof={}; fold_results=[]
    for name in ('linear','quadratic','trees','boost'):
        pred=np.empty(len(y))
        for fold,(train,test) in enumerate(folds):
            estimator=model(name).fit(x[train],y[train])
            pred[test]=estimator.predict_proba(x[test])[:,1]
            fold_results.append({'model':name,'fold':fold,'rows':len(test),
                'auc':float(roc_auc_score(y[test],pred[test]))})
        oof[name]=pred
        rows.append({'model':name,'auc':float(roc_auc_score(y,pred)),
            'logloss':float(log_loss(y,pred)),'fold_auc_min':min(r['auc'] for r in fold_results if r['model']==name)})
    best=max(rows,key=lambda r:r['auc'])
    # Stratified label permutations respect the exact visible matching blocks.
    # Rank/model selection is included in the null by taking the best AUC.
    strata=pd.DataFrame({'pref':pref,'sex':challenge.sex,'band':bands}).groupby(['pref','sex','band']).indices
    rng=np.random.default_rng(280109)
    null=[]
    for repetition in range(99):
        perm=y.copy()
        for indices in strata.values():
            perm[indices]=rng.permutation(perm[indices])
        # Hold predictions fixed: a fast association negative control, not a
        # full refitted randomization test. This limitation is explicit below.
        null.append(max(float(roc_auc_score(perm,p)) for p in oof.values()))
    passed=best['auc']>=.60 and best['fold_auc_min']>.50 and best['auc']>float(np.quantile(null,.95))
    report={'scope':'actual selfchallenge24 known labels only','source':str(ACTUAL),
        'member_rows':int(y.sum()),'control_rows':int((1-y).sum()),'models':rows,
        'folds':fold_results,'best':best,'grouping':'5-fold held-out prefecture/sex/age-band matching blocks',
        'gate':'AUC>=.60, everyfold>.50, bestobserved above p95 best-fixed-score permutation',
        'association_null_p95':float(np.quantile(null,.95)),
        'permutation_caveat':'fixed OOF predictions, not refitted; not a calibrated hypothesis-test p-value',
        'gate_passed':passed,'production_candidate_written':False,
        'validation_status':'OOF feasibility/model comparison only; no separate untouched final test',
        'limitation':'single326row ownchallenge, best-of-four selection is optimistic; cross-target transfer unproven'}
    (OUT/'control_classifier_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame({'challenge_row_id':challenge.challenge_row_id,**oof}).to_csv(OUT/'control_classifier_oof.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
