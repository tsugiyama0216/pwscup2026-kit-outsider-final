import warnings
import numpy as np
from scipy.stats import rankdata,norm
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from cox_calibration import features,FEATURES
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']

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
