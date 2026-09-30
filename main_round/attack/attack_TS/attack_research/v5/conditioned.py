import numpy as np
from scipy.special import logsumexp


def count_conditioned(probability,count):
    """Independent Bernoulli posteriors conditional on a fixed number of members."""
    p=np.clip(np.asarray(probability,dtype=float),1e-6,1-1e-6)
    n=len(p)
    if not 0<=count<=n:raise ValueError('Invalid count')
    if count==0:return np.zeros(n)
    if count==n:return np.ones(n)
    log_odds=np.log(p)-np.log1p(-p)
    forward=np.full((n+1,count+1),-np.inf);forward[0,0]=0.
    backward=np.full((n+1,count+1),-np.inf);backward[n,0]=0.
    for i in range(n):
        forward[i+1]=forward[i]
        forward[i+1,1:]=np.logaddexp(forward[i,1:],forward[i,:-1]+log_odds[i])
    for i in range(n-1,-1,-1):
        backward[i]=backward[i+1]
        backward[i,1:]=np.logaddexp(backward[i+1,1:],backward[i+1,:-1]+log_odds[i])
    normalizer=forward[n,count]
    result=np.empty(n)
    for i in range(n):
        result[i]=np.exp(log_odds[i]+logsumexp(forward[i,:count]+backward[i+1,count-1::-1])-normalizer)
    if not np.isclose(result.sum(),count,atol=1e-7):raise ArithmeticError('Conditional count does not match')
    return np.clip(result,0,1)


def visible_conditioning(challenge,probability):
    # These keys are all public. No hidden onset/rare label is used here.
    work=challenge[['sex','prefecture','age']].copy().reset_index(drop=True)
    work['age_band']=((work.age-40)//5).astype(int)
    result=np.asarray(probability,dtype=float).copy()
    for rows in work.groupby(['sex','prefecture','age_band'],sort=True).indices.values():
        # The shadow protocol has even groups. For odd real-data groups, keep the prior.
        if len(rows)%2==0:
            result[rows]=count_conditioned(result[rows],len(rows)//2)
    return result


def predictions(model,extractor,c,background,challenge,selector):
    from ..features import proposal_features
    x,proposals,_,simple_time=extractor(c,background,challenge)
    p=model['aia'].predict_proba(x)[:,1]
    pc=visible_conditioning(challenge,p)
    hp=model['hit'].predict_proba(proposal_features(x,proposals)).reshape(len(x),proposals.shape[1],-1)
    classes={int(k):i for i,k in enumerate(model['hit'].classes_)}
    mh=hp[:,:,classes[3]]/np.maximum(hp[:,:,classes[2]]+hp[:,:,classes[3]],1e-8)
    ch=hp[:,:,classes[1]]/np.maximum(hp[:,:,classes[0]]+hp[:,:,classes[1]],1e-8)
    output={}
    chosen,_=selector(challenge,p,member_count=len(challenge)//2)
    output['strata_simple']=np.where(chosen,simple_time,11.)
    for conditioned,membership in ((False,p),(True,pc)):
        for penalty in (.1,.25,.5,.75,1.,1.5,2.):
            gain=membership[:,None]*mh-penalty*(1-membership[:,None])*ch
            best=gain.argmax(axis=1)
            time=proposals[np.arange(len(x)),best].astype(float)
            strength=gain[np.arange(len(x)),best]
            prefix=('conditioned' if conditioned else 'separate')+f'_{penalty:g}'
            output[prefix]=np.where(strength>0,time,11.)
            selected,_=selector(challenge,strength,member_count=len(challenge)//2)
            output[prefix+'_strata']=np.where(selected,time,11.)
    return output
