"""Conditional mean/covariance transport, fitted only to aggregate B statistics."""
import numpy as np
from cox_calibration import features, MUTABLE, FEATURES, calibrate, fit


def matrix_power(matrix,power):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    if not np.isfinite(values).all() or values.min()<=1e-12:
        raise ValueError('Degenerate conditional covariance')
    return (vectors*(values**power))@vectors.T


def transport(b,c,cfg,strength=0.5,cox_strength=0.0,preserve_risk=False):
    for value in (strength,cox_strength):
        if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('Strengths must lie in [0,1]')
    if len(b)!=len(c) or len(c)<20:
        raise ValueError('B and C must have equal row counts of at least 20')
    xb,xc=features(b),features(c)
    # Standardization gives different measurement units equal treatment.
    scale=xb[MUTABLE].std(ddof=1).to_numpy()
    if not np.isfinite(scale).all() or (scale<=1e-12).any():
        raise ValueError('Degenerate laboratory scale')
    yb,yc=xb[MUTABLE].to_numpy()/scale,xc[MUTABLE].to_numpy()/scale
    center=xb[['age','sex','smoking']].mean().to_numpy()
    def design(x):
        return np.column_stack([np.ones(len(x)),x[['age','sex','smoking']].to_numpy()-center])
    db,dc=design(xb),design(xc)
    coef_b=np.linalg.lstsq(db,yb,rcond=None)[0]
    coef_c=np.linalg.lstsq(dc,yc,rcond=None)[0]
    rb,rc=yb-db@coef_b,yc-dc@coef_c
    sb,sc=np.cov(rb,rowvar=False),np.cov(rc,rowvar=False)
    half=matrix_power(sc,0.5)
    inv=matrix_power(sc,-0.5)
    mapping=inv@matrix_power(half@sb@half,0.5)@inv
    target=dc@coef_b+rc@mapping
    delta=target-yc
    risk_audit=None
    if preserve_risk:
        if cox_strength:
            raise ValueError('Risk preservation cannot be combined with Cox recalibration')
        beta,_=fit(c,'onset')
        gamma=beta[[FEATURES.index(name) for name in MUTABLE]]*scale
        variance=float(gamma@sc@gamma)
        if not np.isfinite(variance) or variance<=1e-12:
            raise ValueError('Degenerate synthetic Cox direction')
        direction=sc@gamma/variance
        delta=delta-np.outer(delta@gamma,direction)
        risk_audit={'method':'covariance_metric_projection_onto_cox_nullspace',
                    'initial_coefficients':beta.tolist(),
                    'maximum_log_risk_change_before_clipping':float(np.max(np.abs(strength*delta@gamma)))}
    transformed=(yc+strength*delta)*scale
    out=c.copy(deep=True)
    clipped={}
    for j,name in enumerate(MUTABLE):
        raw={'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
        values=transformed[:,j]
        if name.startswith('log_'):
            values=np.exp(values)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite transport')
        lo,hi=cfg['schema']['ranges'][raw]
        clipped[raw]=int(((values<lo)|(values>hi)).sum())
        if strength!=0:
            out[raw]=np.clip(values,lo,hi)
    calibration_audit=None
    if cox_strength:
        out,calibration_audit=calibrate(b,out,cfg,cox_strength,('onset',))
    audit={'method':'conditional_gaussian_optimal_transport','strength':float(strength),
           'cox_strength':float(cox_strength),'conditional_features':['age','sex','smoking'],
           'laboratory_features':MUTABLE,'scale':scale.tolist(),
           'source_regression':coef_b.tolist(),'synthetic_regression':coef_c.tolist(),
           'residual_mapping':mapping.tolist(),'clipped_counts':clipped,
           'preserve_risk':bool(preserve_risk),'risk_projection':risk_audit,
           'cox_calibration':calibration_audit,'source_row_matching':False,
           'training_inputs':'B and current R6-derived synthetic frame','formal_differential_privacy':False}
    return out,audit
