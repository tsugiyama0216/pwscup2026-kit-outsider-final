import numpy as np
from scipy.special import ndtr

from ..features import canonical, QI_CONT
from ..v11.features import extract as previous_extract, neighbor_view
from local_eval.data import FEATURES, QI


def extract(c, background, challenge):
    c,background=canonical(c,FEATURES),canonical(background,FEATURES)
    query=challenge[QI].reset_index(drop=True)
    x,previous,_,_=previous_extract(c,background,query)
    original=previous[:,:8]
    # A 0.5-year grid covers every permitted truth within the 0.5-year tolerance.
    times=np.tile(np.linspace(0.,10.,21),(len(query),1))
    views=[neighbor_view(query,c,background,QI_CONT,False,.5),
        neighbor_view(query,c,background,QI_CONT,True,.5),
        neighbor_view(query,c,background,['age','SBP','HDL'],False,.5),
        neighbor_view(query,background,background,QI_CONT,True,.5)]
    expanded=[np.repeat(x,times.shape[1],axis=0),times.reshape(-1,1),
        np.abs(times[:,:,None]-original[:,None,:]).reshape(-1,original.shape[1]),
        (times>=9.5).astype(float).reshape(-1,1)]
    for _,weights,donor_time,events in views:
        delta=times[:,:,None]-donor_time[:,None,:]
        exact=(np.abs(delta)<=.5).astype(float)
        event=events[:,None,:]
        window_features=[(weights[:,None,:]*exact).sum(axis=2)]
        for sd in (.5,1.):
            mass=ndtr((delta+.5)/sd)-ndtr((delta-.5)/sd)
            mass=np.where(event,mass,exact)
            window_features.append((weights[:,None,:]*mass).sum(axis=2))
        window_features += [(weights[:,None,:]*exact*event).sum(axis=2),
            np.min(np.abs(delta)+np.where(event,0.,100.),axis=2)]
        expanded.append(np.stack(window_features,axis=2).reshape(len(query)*times.shape[1],-1))
    hx=np.column_stack(expanded).astype(np.float32)
    return x,times.astype(np.float32),hx,np.ones(len(hx))


def predict(models,c,background,challenge):
    from ..v5.conditioned import visible_conditioning
    x,times,hx,_=extract(c,background,challenge)
    member=visible_conditioning(challenge,models['membership'].predict_proba(x)[:,1])
    out={}
    for name,model in models.items():
        if name=='membership':continue
        probability=model.predict_proba(hx).reshape(len(x),times.shape[1],-1)
        index={int(k):i for i,k in enumerate(model.classes_)}
        mh=probability[:,:,index[3]]/np.maximum(probability[:,:,index[2]]+probability[:,:,index[3]],1e-8)
        ch=probability[:,:,index[1]]/np.maximum(probability[:,:,index[0]]+probability[:,:,index[1]],1e-8)
        for penalty in (.1,.25,.5,.75,1.):
            gain=member[:,None]*mh-penalty*(1-member[:,None])*ch
            chosen=gain.argmax(axis=1)
            score=gain[np.arange(len(x)),chosen]
            selected=times[np.arange(len(x)),chosen].astype(float)
            out[f'{name}_{penalty:g}']=np.where(score>0,selected,11.)
    return out
