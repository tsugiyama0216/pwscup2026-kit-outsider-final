"""Public-input AIA decisions from frozen copula models."""
import numpy as np
from scipy.special import expit,logit

from ..v5.conditioned import visible_conditioning
from ..v12.features import extract as grid_extract
from ..v13.decision import ratio_optimal
from ..v27.features import extract as membership_extract


def variants(membership_model,grid_model,c,background,challenge,temperatures,penalties):
    x,times,hx,_=grid_extract(c,background,challenge)
    mx=membership_extract(c,background,challenge,neighbor_x=x)
    p=np.clip(membership_model.predict_proba(mx)[:,1],1e-6,1-1e-6)
    model=grid_model['grid']
    probability=model.predict_proba(hx).reshape(len(x),times.shape[1],-1)
    index={int(k):i for i,k in enumerate(model.classes_)}
    mh=probability[:,:,index[3]]/np.maximum(probability[:,:,index[2]]+probability[:,:,index[3]],1e-8)
    ch=probability[:,:,index[1]]/np.maximum(probability[:,:,index[0]]+probability[:,:,index[1]],1e-8)
    output={}
    for temperature in temperatures:
        member=visible_conditioning(challenge,expit(float(temperature)*logit(p)))
        for penalty in penalties:
            gain=member[:,None]*mh-float(penalty)*(1-member[:,None])*ch
            chosen=gain.argmax(axis=1)
            score=gain[np.arange(len(x)),chosen]
            selected=times[np.arange(len(x)),chosen].astype(float)
            output[f't{temperature:g}_p{penalty:g}']=np.where(score>0,selected,11.)
        output[f't{temperature:g}_ratio']=ratio_optimal(member[:,None]*mh,
            (1-member[:,None])*ch,times,len(challenge)//2)
    return output
