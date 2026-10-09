"""Standalone independent attacks. No old model imports, assets, or truth reads."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

NUM=['age','BMI','SBP','TG','HDL','ALT','FPG','time']
QI=['age','SBP','TG','HDL','ALT']
CATS=['sex','prefecture']


def private_mask(b,challenge):
    cols=QI+CATS
    bkeys=list(map(tuple,b[cols].to_numpy()));qkeys=list(map(tuple,challenge[cols].to_numpy()))
    if len(set(bkeys))!=len(bkeys) or len(set(qkeys))!=len(qkeys):raise ValueError('Ambiguous public QI')
    qset=set(qkeys);mask=np.array([key in qset for key in bkeys])
    if int(mask.sum())!=163:raise ValueError('Public intersection must prove precisely 163 self-private people')
    return mask


def numeric_distance(x,y,background,columns,white=False):
    ref=background[columns].to_numpy(float);scale=np.maximum(ref.std(axis=0),1e-4)
    xx=x[columns].to_numpy(float)/scale;yy=y[columns].to_numpy(float)/scale
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():raise ValueError('Nonfinite feature values')
    if white:
        values,vectors=np.linalg.eigh(np.cov(ref/scale,rowvar=False))
        transform=(vectors/np.sqrt(np.maximum(values,.25)))@vectors.T
        xx=xx@transform;yy=yy@transform
    return np.maximum(cdist(xx,yy,'sqeuclidean'),0)


def qi_distance(x,y,background,penalty=0,white=False):
    d=numeric_distance(x,y,background,QI,white)
    for col in CATS:d+=penalty*(x[col].astype(str).to_numpy()[:,None]!=y[col].astype(str).to_numpy()[None,:])
    return d


def empirical_membership(b,c,bg):
    ordered=bg.sort_values(NUM,kind='stable').reset_index(drop=True)
    reference=ordered.iloc[::2];null=ordered.iloc[1::2]
    bd=qi_distance(b,c,ordered,1.5).min(axis=1);nd=qi_distance(null,c,ordered,1.5).min(axis=1)
    br=np.partition(qi_distance(b,reference,ordered,1.5),2,axis=1)[:,2]
    nr=np.partition(qi_distance(null,reference,ordered,1.5),2,axis=1)[:,2]
    score=np.log((br+1e-10)/(bd+1e-10));null_score=np.sort(np.log((nr+1e-10)/(nd+1e-10)))
    return -np.log((len(null_score)-np.searchsorted(null_score,score,side='left')+.5)/(len(null_score)+1))


def numerical_membership(b,c,bg):
    dc=numeric_distance(b,c,bg,NUM).min(axis=1)
    da=np.partition(numeric_distance(b,bg,bg,NUM),4,axis=1)[:,4]
    return np.log((da+1e-10)/(dc+1e-10))


def nearest_times(d,c):
    # Average exact ties in canonical value order; release row order is irrelevant.
    minimum=d.min(axis=1);times=c.time.to_numpy(float)
    return np.array([np.sort(times[row==low]).mean() for row,low in zip(d,minimum)])


def predict(model,*,candidates,self_labels,background,self_challenge,targets,**unused):
    if model not in ['Rv13','Rv14','Rv15']:raise ValueError('Unknown independent model')
    mask=private_mask(candidates,self_challenge)
    mia=pd.DataFrame({'record_id':candidates.record_id.to_numpy()});aia=[];diagnostics={}
    for k,item in sorted(targets.items()):
        c=item['release'].sort_values(NUM+CATS+['smoking','onset','death'],kind='stable').reset_index(drop=True)
        q=item['challenge'];raw=empirical_membership(candidates,c,background) if model=='Rv15' else numerical_membership(candidates,c,background)
        score=(rankdata(raw,method='average')-.5)/len(raw);score[mask]=0;mia[str(k)]=score
        d=qi_distance(q,c,background,penalty=.15 if model=='Rv13' else (1.5 if model=='Rv15' else 0),white=model=='Rv13')
        t=nearest_times(d,c);suppressed=0
        if model=='Rv15':
            count=(qi_distance(q,background,background,1.5)<=d.min(axis=1)[:,None]).sum(axis=1)
            t[count>16]=11.;suppressed=int((count>16).sum())
        aia.append(pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id.to_numpy(),'time_hat':t}))
        diagnostics[str(k)]={'suppressed':suppressed}
    return {'F_mia':mia,'F_aia':pd.concat(aia,ignore_index=True),'diagnostics':{'model':model,'known_self_private':int(mask.sum()),'targets':diagnostics,'old_models_used':False,'fitted_assets_used':False,'source_of_learning':'provided public tables only'}}


def predict_world(**public_tables):
    return predict('Rv13',**public_tables)
