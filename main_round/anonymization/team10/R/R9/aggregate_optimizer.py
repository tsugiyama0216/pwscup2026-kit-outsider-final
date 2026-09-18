"""Autodifferentiable aggregate Cox, distribution and copula moment matching.

Targets are aggregate statistics. No source-to-synthetic row correspondence
is constructed. Discrete attributes and synthetic event/time tuples are fixed.
"""
import numpy as np
import autograd.numpy as anp
from autograd import value_and_grad
from autograd.extend import primitive,defvjp
from scipy.optimize import minimize
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from cox_calibration import features,FEATURES,MUTABLE
from copula_helpers import fit_information,CONT

@primitive
def cdf_interp(x,knots,levels):
    return np.interp(x,knots,levels)

def _cdf_vjp(ans,x,knots,levels):
    idx=np.clip(np.searchsorted(knots,x,side='right')-1,0,len(knots)-2)
    slope=(levels[idx+1]-levels[idx])/(knots[idx+1]-knots[idx])
    slope=np.where((x<=knots[0])|(x>=knots[-1]),0.,slope)
    return lambda g:g*slope

defvjp(cdf_interp,_cdf_vjp,None,None)

def efron_design(time,event):
    """Risk sets and Efron tie fractions; arrays depend only on fixed outcomes."""
    order=np.argsort(time,kind='stable')
    sorted_time=time[order]
    starts=[]; fractions=[]; groups=[]; event_groups=[]
    for t in np.unique(time[event==1]):
        group=np.flatnonzero((time==t)&(event==1)); n=len(group)
        event_groups.append(group)
        for k in range(n):
            starts.append(np.searchsorted(sorted_time,t,side='left'))
            fractions.append(k/n); groups.append(len(event_groups)-1)
    return order,np.asarray(starts),np.asarray(fractions),np.asarray(groups),event_groups

def cox_moments(x,beta,event,design):
    order,starts,frac,groups,event_groups=design
    w=anp.exp(anp.dot(x,beta))
    wx=w[:,None]*x
    wxx=wx[:,:,None]*x[:,None,:]
    r0=anp.cumsum(w[order][::-1],axis=0)[::-1][starts]
    r1=anp.cumsum(wx[order][::-1],axis=0)[::-1][starts]
    r2=anp.cumsum(wxx[order][::-1],axis=0)[::-1][starts]
    e0=anp.stack([anp.sum(w[g]) for g in event_groups])[groups]
    e1=anp.stack([anp.sum(wx[g],axis=0) for g in event_groups])[groups]
    e2=anp.stack([anp.sum(wxx[g],axis=0) for g in event_groups])[groups]
    den=r0-frac*e0
    mean=(r1-frac[:,None]*e1)/den[:,None]
    score=anp.sum(x[event==1],axis=0)-anp.sum(mean,axis=0)
    info=anp.sum((r2-frac[:,None,None]*e2)/den[:,None,None]-mean[:,:,None]*mean[:,None,:],axis=0)
    return score,info

def make_problem(b,c,cfg,detector,marginal_weight,rare_weight,correlation_weight,information_weight,movement_weight,trust_radius,membership_weight=100.):
    xb,xc=features(b).to_numpy(),features(c).to_numpy()
    center=xb.mean(axis=0); scale=xb.std(axis=0,ddof=1)
    z0=(xc-center)/scale; zb=(xb-center)/scale
    beta,cov=fit_information(b); beta=beta*scale
    cov=cov*scale[:,None]*scale[None,:]
    pos=[FEATURES.index(n) for n in MUTABLE]
    fixed={j:z0[:,j] for j in range(9) if j not in pos}
    assignment={j:k for k,j in enumerate(pos)}
    n=len(c); event=c.onset.to_numpy()
    design=efron_design(c.time.to_numpy(),event)
    from model import radii
    rare=radii(c,detector)>=detector['band']['lo']
    b_rare=radii(b,detector)>=detector['band']['lo']
    if min(rare.sum(),b_rare.sum())<20: raise ValueError('Insufficient tail support')
    target=np.quantile(zb[:,pos],(np.arange(n)+.5)/n,axis=0)
    tail_target=np.quantile(zb[b_rare][:,pos],(np.arange(rare.sum())+.5)/rare.sum(),axis=0)
    tail_scale=np.std(zb[b_rare][:,pos],axis=0)
    target_corr=np.corrcoef(rankdata(xb,axis=0),rowvar=False)
    levels=np.linspace(0,1,101)
    knots={j:np.quantile(zb[:,j],levels) for j in pos}
    for j in pos:
        if not (np.diff(knots[j])>0).all(): raise ValueError('Repeated CDF knots')
    fixed_ranks={j:rankdata(xc[:,j])/n for j in fixed}
    mu=np.asarray(detector['mu']); inv=np.linalg.pinv(detector['sigma'])
    threshold=float(detector['band']['lo'])**2
    groups=[]
    for name in ['sex','smoking','prefecture']:
        groups.append(np.column_stack([(c[name]==v).to_numpy(float) for v in sorted(c[name].unique())]))
    raw_b=b[CONT[1:]].to_numpy(float)
    raw_bounds=np.column_stack([raw_b.min(axis=0),raw_b.max(axis=0)])
    raw_sd=raw_b.std(axis=0)
    group_assoc=[]
    for name in ['sex','smoking','prefecture']:
        gb=np.column_stack([(b[name]==v).to_numpy(float) for v in sorted(c[name].unique())])
        sizes=gb.sum(axis=0); means=gb.T@raw_b/np.maximum(sizes[:,None],1)
        group_assoc.append(np.sqrt(np.sum(sizes[:,None]*(means-raw_b.mean(axis=0))**2,axis=0)/len(b))/raw_sd)
    upper=np.triu_indices(9,1)
    def assemble(p):
        v=anp.reshape(p,(n,6))
        return anp.stack([v[:,assignment[j]] if j in assignment else fixed[j] for j in range(9)],axis=1)
    def objective(p,parts=False):
        z=assemble(p)
        score,info=cox_moments(z,beta,event,design)
        variance=anp.linalg.inv(info)
        cox_loss=anp.dot(score,anp.dot(cov,score))/9
        se_loss=anp.mean(anp.log(anp.diag(variance)/np.diag(cov))**2)/4
        r=anp.stack([cdf_interp(z[:,j],knots[j],levels) if j in knots else fixed_ranks[j] for j in range(9)],axis=1)
        r=r-anp.mean(r,axis=0); r=r/anp.sqrt(anp.mean(r*r,axis=0))
        corr=anp.dot(r.T,r)/n
        corr_loss=anp.mean((corr[upper]-target_corr[upper])**2)
        v=z[:,pos]
        marginal=anp.mean((anp.stack([anp.sort(v[:,j]) for j in range(6)],axis=1)-target)**2)
        tail=anp.mean(((anp.stack([anp.sort(v[rare,j]) for j in range(6)],axis=1)-tail_target)/tail_scale)**2)
        raw=z*scale+center
        raw_cont=anp.stack([raw[:,0],raw[:,2],raw[:,3],anp.exp(raw[:,4]),raw[:,5],anp.exp(raw[:,6]),raw[:,7]],axis=1)
        d=raw_cont-mu
        radius2=anp.sum(anp.dot(d,inv)*d,axis=1)
        radial=anp.mean(anp.maximum(anp.where(rare,threshold+.04-radius2,radius2-threshold+.04),0)**2)/threshold**2
        raw_mutable=anp.clip(raw_cont[:,1:],raw_bounds[:,0],raw_bounds[:,1])
        nofab=0.
        for g,ab in zip(groups,group_assoc):
            sizes=g.sum(axis=0); means=anp.dot(g.T,raw_mutable)/sizes[:,None]
            assoc=anp.sqrt(anp.sum(sizes[:,None]*(means-anp.mean(raw_mutable,axis=0))**2,axis=0)/n+1e-20)/raw_sd
            nofab+=anp.mean(anp.maximum(assoc-ab-.06,0)**2)
        movement=anp.mean((v-z0[:,pos])**2)
        components=anp.stack([cox_loss,information_weight*se_loss,correlation_weight*corr_loss,marginal_weight*marginal,rare_weight*tail,membership_weight*radial,100*nofab,movement_weight*movement])
        return components if parts else anp.sum(components)
    bounds=[]
    for i in range(n):
        for j in pos:
            name={'log_TG':'TG','log_ALT':'ALT'}.get(FEATURES[j],FEATURES[j])
            lo,hi=cfg['schema']['ranges'][name]
            if FEATURES[j].startswith('log_'): lo,hi=np.log(lo),np.log(hi)
            lo,hi=(lo-center[j])/scale[j],(hi-center[j])/scale[j]
            bounds.append((max(lo,z0[i,j]-trust_radius),min(hi,z0[i,j]+trust_radius)))
    return objective,z0[:,pos].ravel(),bounds,assemble,center,scale,rare

def refine(b,c,cfg,detector,steps,marginal_weight,rare_weight,correlation_weight,information_weight,movement_weight,trust_radius,membership_weight=100.,preserve_membership=False):
    problem=make_problem(b,c,cfg,detector,marginal_weight,rare_weight,correlation_weight,information_weight,movement_weight,trust_radius,membership_weight)
    objective,start,bounds,assemble,center,scale,rare=problem
    history=[]
    with threadpool_limits(limits=1):
        initial=objective(start,True).tolist()
        result=minimize(value_and_grad(objective),start,jac=True,bounds=bounds,method='L-BFGS-B',
                        options={'maxiter':steps,'ftol':1e-12,'gtol':1e-7,'maxls':30,'maxcor':20},
                        callback=lambda p:history.append(float(objective(p))))
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun) or result.fun>sum(initial)+1e-10:
        raise RuntimeError('Aggregate optimization failed to produce a finite improving iterate')
    raw=np.asarray(assemble(result.x))*scale+center
    out=c.copy(deep=True)
    for name in MUTABLE:
        j=FEATURES.index(name); column={'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
        out[column]=np.exp(raw[:,j]) if name.startswith('log_') else raw[:,j]
    restored=0
    if preserve_membership:
        out,restored=restore_membership(c,out,detector)
    return out,{'method':'aggregate constrained refinement of copula synthetic data','optimizer_success':bool(result.success),
               'message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),'loss_history':history,
               'initial_components':initial,'final_components':objective(result.x,True).tolist(),'loss_components_before_membership_restoration':True,
               'component_order':['cox_score','cox_uncertainty','rank_correlation','marginals','rare_marginals','rare_membership','group_associations','movement'],
               'training_targets':'B aggregate Cox, marginal quantiles, rank correlation and schema-detected tail; no row matching',
               'fixed_event_tuples':True,'frozen_tail_count':int(rare.sum()),'parameter_count':len(start),'membership_restored_rows':restored}

def restore_membership(base,c,detector):
    from model import radii
    threshold=detector['band']['lo']
    original=radii(base,detector)>=threshold
    changed=(radii(c,detector)>=threshold)!=original
    out=c.copy(deep=True)
    mu=np.asarray(detector['mu']); inv=np.linalg.pinv(detector['sigma'])
    for i in np.flatnonzero(changed):
        start=base.loc[base.index[i],CONT].to_numpy(float)
        end=c.loc[c.index[i],CONT].to_numpy(float)
        lo,hi=0.,1.
        for _ in range(50):
            mid=(lo+hi)/2; d=start+mid*(end-start)-mu
            if (d@inv@d>=threshold**2)==original[i]: lo=mid
            else: hi=mid
        value=start+max(0.,lo-1e-6)*(end-start)
        out.loc[out.index[i],CONT[1:]]=value[1:]
    if not np.array_equal(radii(out,detector)>=threshold,original): raise RuntimeError('Tail restoration failed')
    return out,int(changed.sum())
