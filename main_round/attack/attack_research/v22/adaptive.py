"""Choose AIA grid answers by their predicted official all-person ratio."""
import numpy as np

from ..v5.conditioned import visible_conditioning
from ..v12.features import extract
from ..v13.decision import ratio_optimal


def predict(models,c,background,challenge):
    x,times,hx,_=extract(c,background,challenge)
    member=visible_conditioning(challenge,models['membership'].predict_proba(x)[:,1])
    model=models['grid']
    probability=model.predict_proba(hx).reshape(len(x),times.shape[1],-1)
    index={int(k):i for i,k in enumerate(model.classes_)}
    mh=probability[:,:,index[3]]/np.maximum(probability[:,:,index[2]]+probability[:,:,index[3]],1e-8)
    ch=probability[:,:,index[1]]/np.maximum(probability[:,:,index[0]]+probability[:,:,index[1]],1e-8)
    return ratio_optimal(member[:,None]*mh,(1-member[:,None])*ch,times,len(challenge)//2)
