"""Public-input copula AIA using a new membership model and frozen time grid."""
import numpy as np

from ..v5.conditioned import visible_conditioning
from ..v12.features import extract as grid_extract
from .features import extract as membership_extract


def predict(membership_model,grid_model,c,background,challenge,penalty=.25):
    x,times,hx,_=grid_extract(c,background,challenge)
    features=membership_extract(c,background,challenge,neighbor_x=x)
    member=visible_conditioning(challenge,membership_model.predict_proba(features)[:,1])
    model=grid_model['grid']
    probability=model.predict_proba(hx).reshape(len(x),times.shape[1],-1)
    index={int(k):i for i,k in enumerate(model.classes_)}
    mh=probability[:,:,index[3]]/np.maximum(probability[:,:,index[2]]+probability[:,:,index[3]],1e-8)
    ch=probability[:,:,index[1]]/np.maximum(probability[:,:,index[0]]+probability[:,:,index[1]],1e-8)
    gain=member[:,None]*mh-penalty*(1-member[:,None])*ch
    chosen=gain.argmax(axis=1)
    score=gain[np.arange(len(x)),chosen]
    selected=times[np.arange(len(x)),chosen].astype(float)
    return np.where(score>0,selected,11.)
