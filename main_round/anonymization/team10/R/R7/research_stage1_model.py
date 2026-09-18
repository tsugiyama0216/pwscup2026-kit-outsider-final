"""Aggregate radial, dependence and Cox-information calibration of copula R6."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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

def information_calibration(b,c,regularization,cfg):
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
        return np.concatenate([(beta-bb)/se,2*np.log(np.sqrt(np.diag(cov))/se),
                               regularization*p,100*violation.ravel()])
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
    return out,{'regularization':regularization,'optimizer_success':bool(solved.success),
                'nfev':solved.nfev,'matrix':mapping.tolist(),'matrix_condition':float(np.linalg.cond(mapping)),'clipped_counts':clipped,
                'beta_B':beta_b.tolist(),'beta_initial':beta_c.tolist(),
                'se_B':np.sqrt(np.diag(var_b)).tolist(),'se_initial':np.sqrt(np.diag(var_c)).tolist()}

def synthesize(b,cfg,radial_strength,joint_strength,regularization):
    base=load_base()
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    c,ra=radial(b,base,detector,radial_strength,cfg)
    c,ta=transport(b,c,cfg,strength=joint_strength,preserve_risk=False)
    c,ca=information_calibration(b,c,regularization,cfg)
    for name,(lo,hi) in cfg['schema']['ranges'].items():
        if not c[name].between(lo,hi).all():
            raise ValueError('Candidate outside physiological ranges')
    lo,hi=cfg['population']['age_range']
    if not c.age.between(lo,hi).all():
        raise ValueError('Candidate age outside allowed range')
    fixed=['sex','smoking','prefecture','time','onset','death']
    pd.testing.assert_frame_equal(c[fixed],base[fixed],check_exact=True)
    return c,{'radial':ra,'joint':ta,'cox_information':ca}
