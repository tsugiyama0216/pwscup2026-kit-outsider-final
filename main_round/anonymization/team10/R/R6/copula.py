"""Tail-conditional Gaussian copula transport of frozen R4 synthetic records."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from evaluation import HERE, DATA, COLUMNS, digest
from cox_calibration import calibrate

CONT = ['age','BMI','SBP','TG','HDL','ALT','FPG']
BASE_SHA256 = 'afe9d9251ffb2bdb439902e63826f4d3ff57869aeac6260baf19459f77a06807'

def load_base():
    path = HERE.parent/'R4/C.csv'
    if digest(path) != BASE_SHA256:
        raise RuntimeError('Frozen R4 changed')
    return pd.read_csv(path, float_precision='round_trip')[COLUMNS]

def detect(frame, detector):
    x = frame[CONT].to_numpy(dtype=float)-np.asarray(detector['mu'])
    inv = np.linalg.pinv(np.asarray(detector['sigma']))
    return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',x,inv,x))) >= detector['band']['lo']

def matrix_power(a, power):
    eig, vectors = np.linalg.eigh((a+a.T)/2)
    if not np.isfinite(eig).all() or eig.min() <= 1e-10:
        raise ValueError('Degenerate copula covariance')
    return (vectors*eig**power)@vectors.T

def latent(x):
    return norm.ppf((rankdata(x,axis=0,method='average')-.5)/len(x))

def transform(b, c, detector, cfg, strength, cox_strength):
    for v in (strength,cox_strength):
        if isinstance(v,(bool,np.bool_)) or not np.isfinite(v) or not 0<=v<=1:
            raise ValueError('Strength must lie in [0,1]')
    mb,mc = detect(b,detector),detect(c,detector)
    if min(mb.sum(),mc.sum()) < 20:
        raise ValueError('Insufficient tail support')
    xb,xc = b.loc[mb,CONT].to_numpy(float),c.loc[mc,CONT].to_numpy(float)
    zb,zc = latent(xb),latent(xc)
    sb,sc = np.cov(zb,rowvar=False),np.cov(zc,rowvar=False)
    # Fixed shrinkage prevents narrow small-tail dependence estimates.
    sb=.9*sb+.1*np.diag(np.diag(sb))
    sc=.9*sc+.1*np.diag(np.diag(sc))
    half,inv=matrix_power(sc,.5),matrix_power(sc,-.5)
    mapping=inv@matrix_power(half@sb@half,.5)@inv
    mapped=zc@mapping
    uniforms=norm.cdf(mapped)
    target=np.column_stack([np.quantile(xb[:,j],uniforms[:,j],method='linear') for j in range(len(CONT))])
    out=c.copy(deep=True)
    result=xc+strength*(target-xc)
    result[:,0]=np.rint(result[:,0])
    for j,column in enumerate(CONT):
        if strength:
            out.loc[mc,column]=result[:,j].astype(int) if column=='age' else result[:,j]
    audit={'method':'tail_conditional_gaussian_copula_transport','strength':float(strength),
           'cox_strength':float(cox_strength),'covariance_shrinkage':.1,
           'B_tail_n':int(mb.sum()),'R4_tail_n':int(mc.sum()),
           'tail_after_copula':int(detect(out,detector).sum()),
           'copula_mapping':mapping.tolist(),'source_row_matching':False}
    if cox_strength:
        out,ca=calibrate(b,out,cfg,strength=cox_strength)
        if any(ca['clipped_counts'].values()):
            raise ValueError('Cox correction would clip values')
        audit['cox_calibration']=ca
    fixed=['sex','prefecture','smoking','time','onset','death']
    pd.testing.assert_frame_equal(c[fixed],out[fixed],check_exact=True)
    return out,audit

def synthesize(b,cfg,strength,cox_strength):
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    return transform(b,load_base(),detector,cfg,strength,cox_strength)
