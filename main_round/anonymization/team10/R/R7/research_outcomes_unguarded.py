"""Deterministic aggregate Cox-score calibration by synthetic outcome swaps."""
import numpy as np
from cox_calibration import features

def cox_moments(x,w,time,event):
    """Efron partial-likelihood score and observed information at fixed beta."""
    wx=w[:,None]*x
    wxx=w[:,None,None]*x[:,:,None]*x[:,None,:]
    score=np.zeros(x.shape[1]); information=np.zeros((x.shape[1],x.shape[1]))
    for t in np.unique(time[event==1]):
        risk=time>=t; died=(time==t)&(event==1); n=int(died.sum())
        r0,r1,r2=w[risk].sum(),wx[risk].sum(axis=0),wxx[risk].sum(axis=0)
        e0,e1,e2=w[died].sum(),wx[died].sum(axis=0),wxx[died].sum(axis=0)
        score+=x[died].sum(axis=0)
        for k in range(n):
            frac=k/n; den=r0-frac*e0
            mean=(r1-frac*e1)/den
            score-=mean
            information+=(r2-frac*e2)/den-np.outer(mean,mean)
    return score,information

def calibrate_outcomes(b,c,rare_mask,max_steps=60):
    from model import fit_information
    xb,xc=features(b),features(c)
    scale=xb.std(ddof=1).to_numpy()
    x=(xc.to_numpy()-xc.mean().to_numpy())/scale
    beta,variance=fit_information(b)
    target_variance=variance*scale[:,None]*scale[None,:]
    w=np.exp(np.clip(x@(beta*scale),-30,30))
    time=c.time.to_numpy().copy(); event=c.onset.to_numpy().copy(); death=c.death.to_numpy()
    def evaluate(time,event):
        score,info=cox_moments(x,w,time,event)
        return float(score@target_variance@score),score,info
    objective,score,info=evaluate(time,event)
    initial=objective; history=[]
    for step in range(max_steps):
        positives=np.flatnonzero(event==1)
        negatives=np.flatnonzero((event==0)&(death==0))
        xp,xn=x[positives],x[negatives]
        gradient=target_variance@score
        approx=(2*(xn@gradient)[None,:]-2*(xp@gradient)[:,None]
                +np.sum((xn@target_variance)*xn,axis=1)[None,:]
                +np.sum((xp@target_variance)*xp,axis=1)[:,None]
                -2*(xp@target_variance)@xn.T)
        approx[rare_mask[positives,None]!=rare_mask[negatives][None,:]]=np.inf
        count=min(12,np.isfinite(approx).sum())
        if not count or objective<.01:
            break
        choices=np.argpartition(approx.ravel(),count-1)[:count]
        choices=sorted(choices,key=lambda k:(approx.ravel()[k],int(k)))
        best=None
        for k in choices:
            a,d=np.unravel_index(k,approx.shape); i,j=positives[a],negatives[d]
            tt,ee=time.copy(),event.copy()
            tt[i],tt[j]=tt[j],tt[i]; ee[i],ee[j]=ee[j],ee[i]
            value,newscore,newinfo=evaluate(tt,ee)
            if value<objective-1e-10 and (best is None or value<best[0]):
                best=(value,newscore,newinfo,tt,ee)
        if best is None:
            break
        objective,score,info,time,event=best
        history.append(objective)
    out=c.copy(deep=True); out['time']=time; out['onset']=event
    return out,{'method':'Efron score moment calibration by synthetic onset/censor swaps',
                'initial_score_loss':initial,'final_score_loss':objective,'swaps':len(history),
                'loss_history':history,'max_steps':max_steps,
                'rare_onset_count_preserved':int(c.loc[rare_mask,'onset'].sum())==int(out.loc[rare_mask,'onset'].sum()),
                'source_record_matching':False}
