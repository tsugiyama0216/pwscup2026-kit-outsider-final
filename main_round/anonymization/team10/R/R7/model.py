"""Aggregate radial, dependence and Cox-information calibration of copula R6."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import least_squares
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from evaluation import HERE, DATA, COLUMNS, digest
from cox_calibration import features, FEATURES, MUTABLE
from joint_transport import transport

BASE_SHA256='addbfafedc6b60b6e1d3ba27a4bacc7edb5af594be455ae6d3601819c484e893'
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']

def load_base():
    path=HERE.parent/'R6/C.csv'
    if digest(path)!=BASE_SHA256:
        raise RuntimeError('Frozen R6 changed')
    return pd.read_csv(path,float_precision='round_trip')[COLUMNS]

def majority_copula(b,c,detector,strength):
    if not strength:
        return c.copy(deep=True),None
    mu=np.asarray(detector['mu']); inv=np.linalg.pinv(np.asarray(detector['sigma']))
    def majority(f):
        d=f[CONT].to_numpy(float)-mu
        return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',d,inv,d)))<detector['band']['lo']
    mb,mc=majority(b),majority(c)
    xb,xc=b.loc[mb,CONT].to_numpy(float),c.loc[mc,CONT].to_numpy(float)
    zb=norm.ppf((rankdata(xb,axis=0)-.5)/len(xb))
    zc=norm.ppf((rankdata(xc,axis=0)-.5)/len(xc))
    db=np.column_stack([np.ones(mb.sum()),b.loc[mb,['sex','smoking']].to_numpy(float)])
    dc=np.column_stack([np.ones(mc.sum()),c.loc[mc,['sex','smoking']].to_numpy(float)])
    cb,cc=np.linalg.lstsq(db,zb,rcond=None)[0],np.linalg.lstsq(dc,zc,rcond=None)[0]
    rb,rc=zb-db@cb,zc-dc@cc
    sb,sc=np.cov(rb,rowvar=False),np.cov(rc,rowvar=False)
    sb=.97*sb+.03*np.diag(np.diag(sb)); sc=.97*sc+.03*np.diag(np.diag(sc))
    from joint_transport import matrix_power
    half,inverse=matrix_power(sc,.5),matrix_power(sc,-.5)
    mapping=inverse@matrix_power(half@sb@half,.5)@inverse
    targetz=dc@cb+rc@mapping
    quantile=norm.cdf(targetz)
    target=np.column_stack([np.quantile(xb[:,j],quantile[:,j]) for j in range(7)])
    result=(1-strength)*xc+strength*target
    result[:,0]=np.rint(result[:,0])
    out=c.copy(deep=True)
    for j,n in enumerate(CONT):
        out.loc[mc,n]=result[:,j].astype(int) if n=='age' else result[:,j]
    return out,{'strength':strength,'B_majority':int(mb.sum()),'C_majority':int(mc.sum()),'mapping':mapping.tolist()}

def radial(b,c,detector,strength,cfg):
    if not 0<=strength<=1:
        raise ValueError('Invalid radial strength')
    mu=np.asarray(detector['mu'])
    inverse=np.linalg.pinv(np.asarray(detector['sigma']))
    def radius(frame):
        d=frame[CONT].to_numpy(float)-mu
        return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',d,inverse,d)))
    rb,rc=radius(b),radius(c)
    target=np.quantile(rb,(rankdata(rc)-.5)/len(rc))
    ratio=np.divide((1-strength)*rc+strength*target,rc,out=np.ones(len(rc)),where=rc>1e-12)
    direction=c[CONT].to_numpy(float)-mu
    limits=np.array([cfg['population']['age_range'] if name=='age' else cfg['schema']['ranges'][name] for name in CONT])
    raymax=np.min(np.divide(np.where(direction>=0,limits[:,1]-mu,limits[:,0]-mu),direction,
                           out=np.full_like(direction,np.inf),where=np.abs(direction)>1e-12),axis=1)
    capped=int((ratio>raymax).sum())
    ratio=np.minimum(ratio,raymax)
    out=c.copy(deep=True)
    if strength:
        values=mu+(c[CONT].to_numpy(float)-mu)*ratio[:,None]
        values[:,0]=np.rint(values[:,0])
        for j,name in enumerate(CONT):
            out[name]=values[:,j].astype(int) if name=='age' else values[:,j]
    return out,{'strength':strength,'radial_scale_min':float(ratio.min()),'radial_scale_max':float(ratio.max()),'range_limited_rays':capped}

def fit_information(frame):
    x=features(frame)
    train=x.assign(time=frame.time.to_numpy(),event=frame.onset.to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=CoxPHFitter(penalizer=0).fit(train,'time','event')
    return model.params_.reindex(FEATURES).to_numpy(),model.variance_matrix_.loc[FEATURES,FEATURES].to_numpy()

def information_calibration(b,c,regularization,cfg,covariance_weight=0.):
    if regularization<=0:
        raise ValueError('Positive regularization required')
    xb,xc=features(b),features(c)
    scale=xb.std(ddof=1).to_numpy()
    beta_b,var_b=fit_information(b)
    beta_c,var_c=fit_information(c)
    bb,bc=beta_b*scale,beta_c*scale
    vb,vc=var_b*scale[:,None]*scale[None,:],var_c*scale[:,None]*scale[None,:]
    se=np.sqrt(np.diag(vb))
    pos=[FEATURES.index(n) for n in MUTABLE]
    center=xc.mean().to_numpy()/scale
    standardized=xc.to_numpy()/scale-center
    cov_b=np.cov(xb.to_numpy()/scale,rowvar=False)
    cov_c=np.cov(xc.to_numpy()/scale,rowvar=False)
    def correlation(v):
        sd=np.sqrt(np.diag(v))
        return v/np.outer(sd,sd)
    corr_b=correlation(cov_b)
    upper=np.triu_indices(9,1)
    limits=np.array([cfg['schema']['ranges'][{'log_TG':'TG','log_ALT':'ALT'}.get(n,n)] for n in MUTABLE],dtype=float)
    for j,n in enumerate(MUTABLE):
        if n.startswith('log_'):
            limits[j]=np.log(limits[j])
    limits/=scale[pos,None]
    def matrix(p):
        m=np.eye(9)
        m[:,pos]+=p.reshape(9,6)
        return m
    def residual(p):
        m=matrix(p)
        inv=np.linalg.inv(m)
        beta=inv@bc
        cov=inv@vc@inv.T
        # Match both effects and uncertainty with a penalty for moving the data.
        y=(standardized@m+center)[:,pos]
        violation=np.maximum(limits[:,0]-y,0)+np.maximum(y-limits[:,1],0)
        cov_data=m.T@cov_c@m
        return np.concatenate([(beta-bb)/se,2*np.log(np.sqrt(np.diag(cov))/se),
                               regularization*p,100*violation.ravel(),
                               covariance_weight*(correlation(cov_data)-corr_b)[upper],
                               covariance_weight*.5*(np.diag(cov_data)-np.diag(cov_b))])
    solved=least_squares(residual,np.zeros(54),bounds=(-.3,.3),max_nfev=150,
                         ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not solved.success:
        raise RuntimeError('Cox information calibration did not converge')
    mapping=matrix(solved.x)
    centered=xc.mean().to_numpy()/scale
    values=((xc.to_numpy()/scale-centered)@mapping+centered)*scale
    out=c.copy(deep=True)
    clipped={}
    for name in MUTABLE:
        raw={'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
        rawvalues=np.exp(values[:,FEATURES.index(name)]) if name.startswith('log_') else values[:,FEATURES.index(name)]
        lo,hi=cfg['schema']['ranges'][raw]
        clipped[raw]=int(((rawvalues<lo)|(rawvalues>hi)).sum())
        out[raw]=np.clip(rawvalues,lo,hi)
    return out,{'regularization':regularization,'covariance_weight':covariance_weight,'optimizer_success':bool(solved.success),
                'nfev':solved.nfev,'matrix':mapping.tolist(),'matrix_condition':float(np.linalg.cond(mapping)),'clipped_counts':clipped,
                'beta_B':beta_b.tolist(),'beta_initial':beta_c.tolist(),
                'se_B':np.sqrt(np.diag(var_b)).tolist(),'se_initial':np.sqrt(np.diag(var_c)).tolist()}

def synthesize(b,cfg,radial_strength,joint_strength,regularization,covariance_weight=0.,information_strength=1.,preserve_strength=0.,majority_strength=0.,outcome_steps=0,significance_guard=False,tail_strength=0.,preserve_tail_radius=False):
    base=load_base()
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    c,ma=majority_copula(b,base,detector,majority_strength)
    c,ra=radial(b,c,detector,radial_strength,cfg)
    c,ta=transport(b,c,cfg,strength=joint_strength,preserve_risk=False)
    if information_strength:
        before=c
        candidate,ca=information_calibration(b,c,regularization,cfg,covariance_weight)
        c=before.copy(deep=True)
        for name in MUTABLE:
            raw={'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
            if name.startswith('log_'):
                c[raw]=np.exp((1-information_strength)*np.log(before[raw])+information_strength*np.log(candidate[raw]))
            else:
                c[raw]=(1-information_strength)*before[raw]+information_strength*candidate[raw]
        ca['information_strength']=information_strength
    else:
        ca=None
    pa=None
    if preserve_strength:
        c,pa=transport(b,c,cfg,strength=preserve_strength,preserve_risk=True)
    tail_audit=None
    if tail_strength:
        from tail_copula import transform
        before_tail=c.copy(deep=True)
        c,tail_audit=transform(b,c,detector,cfg,tail_strength,0.)
        if preserve_tail_radius:
            mu=np.asarray(detector['mu']); inverse=np.linalg.pinv(detector['sigma'])
            delta0=before_tail[CONT].to_numpy()-mu; delta=c[CONT].to_numpy()-mu
            r0=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',delta0,inverse,delta0)))
            r=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',delta,inverse,delta)))
            mask=r0>=detector['band']['lo']
            ratio=np.divide(r0,r,out=np.ones(len(r)),where=r>1e-12)
            limits=np.array([cfg['population']['age_range'] if n=='age' else cfg['schema']['ranges'][n] for n in CONT])
            maxratio=np.min(np.divide(np.where(delta>=0,limits[:,1]-mu,limits[:,0]-mu),delta,
                                     out=np.full_like(delta,np.inf),where=np.abs(delta)>1e-12),axis=1)
            ratio=np.minimum(ratio,maxratio)
            values=mu+delta*ratio[:,None]; values[:,0]=np.rint(values[:,0])
            for j,n in enumerate(CONT):
                c.loc[mask,n]=values[mask,j].astype(int) if n=='age' else values[mask,j]
            tail_audit['radius_preservation']=True
    for name,(lo,hi) in cfg['schema']['ranges'].items():
        if not c[name].between(lo,hi).all():
            raise ValueError('Candidate outside physiological ranges')
    lo,hi=cfg['population']['age_range']
    if not c.age.between(lo,hi).all():
        raise ValueError('Candidate age outside allowed range')
    fixed=['sex','smoking','prefecture','time','onset','death']
    pd.testing.assert_frame_equal(c[fixed],base[fixed],check_exact=True)
    oa=None
    if outcome_steps:
        from outcomes import calibrate_outcomes
        delta=c[CONT].to_numpy()-np.asarray(detector['mu'])
        mask=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',delta,np.linalg.pinv(detector['sigma']),delta)))>=detector['band']['lo']
        c,oa=calibrate_outcomes(b,c,mask,outcome_steps,significance_guard)
    return c,{'majority_copula':ma,'radial':ra,'joint':ta,'cox_information':ca,'risk_preserving_transport':pa,'tail_copula':tail_audit,'outcome_calibration':oa}
