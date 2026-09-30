import numpy as np

from ..v5.conditioned import visible_conditioning
from ..features import proposal_features


def ratio_optimal(member_success, control_success, proposals, expected_members):
    """Maximize the official all-person excess-success ratio under model probabilities."""
    n,k = proposals.shape
    if member_success.shape != (n,k) or control_success.shape != (n,k):
        raise ValueError('Probability and proposal dimensions differ')
    denominator=float(expected_members)
    if not 0<denominator<=n:raise ValueError('Invalid member count')
    ratio=0.
    choice=np.full(n,-1,dtype=int)
    for _ in range(100):
        utility=member_success-(1-ratio)*control_success
        best=utility.argmax(axis=1)
        gain=utility[np.arange(n),best]
        choice=np.where(gain>0,best,-1)
        active=choice>=0
        sm=float(member_success[np.arange(n)[active],choice[active]].sum())
        sc=float(control_success[np.arange(n)[active],choice[active]].sum())
        new_ratio=float(np.clip((sm-sc)/max(denominator-sc,1e-9),0,1))
        if abs(new_ratio-ratio)<1e-12:break
        ratio=new_ratio
    return np.where(choice>=0,proposals[np.arange(n),np.maximum(choice,0)],11.).astype(float)


def predict(model, extractor, c, background, challenge):
    x,proposals,_,_=extractor(c,background,challenge)
    p=model['aia'].predict_proba(x)[:,1]
    conditioned=visible_conditioning(challenge,p)
    hp=model['hit'].predict_proba(proposal_features(x,proposals)).reshape(len(x),proposals.shape[1],-1)
    index={int(k):i for i,k in enumerate(model['hit'].classes_)}
    mh=hp[:,:,index[3]]/np.maximum(hp[:,:,index[2]]+hp[:,:,index[3]],1e-8)
    ch=hp[:,:,index[1]]/np.maximum(hp[:,:,index[0]]+hp[:,:,index[1]],1e-8)
    return ratio_optimal(conditioned[:,None]*mh,(1-conditioned[:,None])*ch,proposals,len(challenge)//2)
