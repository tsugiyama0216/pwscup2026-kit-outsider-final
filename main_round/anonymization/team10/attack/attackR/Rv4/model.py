"""Rv2 plus public, release-adaptive nearest-neighbor evidence. No truth access."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
MIA_OPTIONS=('Rv2','blend15','blend35','confident')
AIA_OPTIONS=('Rv2','rescue10','rescue25','contrast_gate')
NUM=('age','BMI','SBP','TG','HDL','ALT','FPG')
QI=('age','SBP','TG','HDL','ALT')
def ranks(x):return (rankdata(x,method='average')-.5)/len(x)
def distances(q,c,bg,cols):
    def matrix(f):
        a=f[list(cols)].to_numpy(float).copy()
        for name in ('TG','ALT'):
            if name in cols:a[:,cols.index(name)]=np.log1p(np.maximum(a[:,cols.index(name)],0))
        return a
    b=matrix(bg);scale=np.maximum(np.std(b,axis=0),1e-5)
    d=cdist(matrix(q)/scale,matrix(c)/scale,'sqeuclidean')/len(cols)
    for name,w in [('sex',1.),('prefecture',.15)]:
        d+=w*(q[name].to_numpy()[:,None]!=c[name].to_numpy()[None,:])
    return d
def evidence(c,bg,b,q):
    # Stable sorting makes ties independent of file row order.
    c=c.sort_values(list(c.columns),kind='stable').reset_index(drop=True)
    bg=bg.sort_values(list(bg.columns),kind='stable').reset_index(drop=True)
    ratios=[]
    for cols in (NUM,QI,('BMI','FPG','age','SBP')):
        dc=distances(b,c,bg,cols);db=distances(b,bg,bg,cols)
        ratios.append(np.log(np.maximum(db.min(axis=1),1e-12))-np.log(np.maximum(dc.min(axis=1),1e-12)))
    m=np.mean([ranks(x) for x in ratios],axis=0)
    dc=distances(q,c,bg,QI);db=distances(q,bg,bg,QI)
    near=np.argsort(dc,axis=1,kind='stable')[:,:5]
    d=np.take_along_axis(dc,near,axis=1)
    times=c.time.to_numpy(float)[near]
    # Use the densest +/-0.5-year local window, rather than averaging times.
    weights=np.exp(-(d-d[:,:1])/np.maximum(d[:,-1:],1e-9))
    windows=(np.abs(times[:,:,None]-times[:,None,:])<=.5)
    masses=(windows*weights[:,None,:]).sum(axis=2)
    best=masses.argmax(axis=1);t=times[np.arange(len(q)),best]
    support=masses[np.arange(len(q)),best]/weights.sum(axis=1)
    ratio=d[:,0]/np.maximum(db.min(axis=1),1e-12)
    return m,t,ratio,support,ratios[0]
def variants(*,candidates,self_labels,background,self_challenge,targets,schema,cfg,baseline=None):
    import runtime as rt
    maskmod=rt.load('_rv4_mask',rt.ATTACK/'Attack_M/v1/actual.py')
    mask,proof=maskmod.derive_private_mask(candidates.astype(str),self_labels.astype(str),self_challenge.astype(str),schema['aia_qi_columns'],schema['submitted_c_columns'])
    attack=None
    if baseline is None:
        from attack_research.v88.predict import CombinedAttack
        attack=CombinedAttack()
    mia={n:pd.DataFrame({'record_id':candidates.record_id.to_numpy()}) for n in MIA_OPTIONS}
    aia={n:[] for n in AIA_OPTIONS};diag={}
    for k,item in sorted(targets.items()):
        c,q=item['release'],item['challenge']
        if baseline is None:
            all_old,route=attack.predict(c=c,background=background,candidates=candidates,challenge=q,cfg=cfg)
            old=all_old['combined'];om=ranks(old['mia']);oa=np.asarray(old['aia'])
        else:
            route='cached frozen Rv2'
            om=baseline['F_mia'].set_index('record_id').loc[candidates.record_id,str(k)].to_numpy()
            oa=baseline['F_aia'].query('target_k == @k').set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy()
        # Re-rank only the unknown rows; the proven negatives always remain 0.
        om=om.copy();om[~mask]=ranks(om[~mask]);om[mask]=0
        m,t,ratio,support,full=evidence(c,background,candidates,q)
        raw={'Rv2':om,'blend15':.85*om+.15*m,'blend35':.65*om+.35*m,
             'confident':om+.35*np.clip((full-3.)/5.,0,1)}
        for name,v in raw.items():
            s=np.zeros(len(v));s[~mask]=ranks(v[~mask]);mia[name][str(k)]=s
        times={'Rv2':oa,'rescue10':np.where((ratio<.10)&(support>=.6),t,oa),
               'rescue25':np.where((ratio<.25)&(support>=.6),t,oa),
               'contrast_gate':np.where((ratio>1.5)&(oa<=10),11.,oa)}
        for name,values in times.items():
            aia[name].append(pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id.to_numpy(),'time_hat':values}))
        diag[str(k)]={'route':str(route),'rescue10_rows':int(((ratio<.1)&(support>=.6)).sum()),'rescue25_rows':int(((ratio<.25)&(support>=.6)).sum())}
        print('Rv4 public target',k,'complete',flush=True)
    return {'mia':mia,'aia':{n:pd.concat(p,ignore_index=True) for n,p in aia.items()},'diagnostics':{'private_proof':proof,'targets':diag}}
