"""Copula dependence and rank-preserving uncertainty correction of frozen R7."""
import json
import numpy as np
import pandas as pd
from evaluation import HERE,DATA,COLUMNS,digest
from cox_calibration import features,FEATURES,MUTABLE
from copula_helpers import majority_copula,radial,fit_information,CONT
from tail_copula import transform
from outcomes import calibrate_outcomes

BASE_SHA256='a07312b4a5ef6121a4c0bc660b2a0d3f81299ad56921664b8612f562727b6836'

def load_base():
    path=HERE.parent/'R7/C.csv'
    if digest(path)!=BASE_SHA256:
        raise RuntimeError('Frozen R7 changed')
    return pd.read_csv(path,float_precision='round_trip')[COLUMNS]

def radii(c,detector):
    d=c[CONT].to_numpy(float)-np.asarray(detector['mu'])
    return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',d,np.linalg.pinv(detector['sigma']),d)))

def uncertainty_scale(b,c,cfg,strength,scope='all',active_mask=None,range_strategy='global',shape='linear'):
    if scope not in ('all','weak'):
        raise ValueError('Invalid uncertainty scope')
    if range_strategy not in ('global','clip'):
        raise ValueError('Invalid range strategy')
    if shape not in ('linear','cubic'):
        raise ValueError('Invalid uncertainty shape')
    if not 0<=strength<=1:
        raise ValueError('Invalid uncertainty strength')
    if strength==0:
        return c.copy(deep=True),{'strength':0.,'factors':{}}
    _,vb=fit_information(b); _,vc=fit_information(c)
    factor=np.exp(strength*np.log(np.sqrt(np.diag(vc)/np.diag(vb))))
    x=features(c); out=c.copy(deep=True); audit={}
    for name in MUTABLE:
        if scope=='weak' and name not in ('HDL','log_ALT'):
            continue
        raw={'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
        values=x[name].to_numpy(); center=float(values.mean()); delta=values-center
        lo,hi=cfg['schema']['ranges'][raw]
        if name.startswith('log_'):
            lo,hi=np.log(lo),np.log(hi)
        limit_delta=delta if active_mask is None else delta[active_mask]
        maximum=np.min(np.divide(np.where(limit_delta>=0,hi-center,lo-center),limit_delta,
                                 out=np.full(len(limit_delta),np.inf),where=np.abs(limit_delta)>1e-12))
        applied=factor[FEATURES.index(name)] if range_strategy=='clip' else min(factor[FEATURES.index(name)],maximum*(1-1e-12))
        target=center+applied*delta
        if shape=='cubic' and applied>1:
            target=center+delta+(applied-1)*delta**3/(3*np.mean(delta**2))
        clipped=((target<lo)|(target>hi))
        if active_mask is not None: clipped=clipped & active_mask
        target=np.clip(target,lo,hi)
        values=np.exp(target) if name.startswith('log_') else target
        if active_mask is None:
            out[raw]=values
        else:
            out.loc[active_mask,raw]=values[active_mask]
        audit[raw]={'requested':float(factor[FEATURES.index(name)]),'applied':float(applied),'clipped':int(clipped.sum())}
    return out,{'strength':strength,'factors':audit,'range_strategy':range_strategy,'shape':shape}

def tail_correction(b,c,cfg,detector,strength,radius_strength=0.,empirical_quantiles=False):
    if not 0<=strength<=1 or not 0<=radius_strength<=1:
        raise ValueError('Invalid tail strength')
    if not strength and not radius_strength:
        return c.copy(deep=True),None
    before=radii(c,detector); mask=before>=detector['band']['lo']
    out,audit=transform(b,c,detector,cfg,strength,0.,empirical_quantiles)
    after=radii(out,detector)
    target=before.copy()
    if radius_strength:
        from scipy.stats import rankdata
        rb=radii(b,detector); rb=rb[rb>=detector['band']['lo']]
        target[mask]=(1-radius_strength)*before[mask]+radius_strength*np.quantile(rb,(rankdata(before[mask])-.5)/mask.sum())
    ratio=np.divide(target,after,out=np.ones(len(before)),where=after>1e-12)
    mu=np.asarray(detector['mu']); d=out[CONT].to_numpy()-mu
    limits=np.array([cfg['population']['age_range'] if n=='age' else cfg['schema']['ranges'][n] for n in CONT])
    maximum=np.min(np.divide(np.where(d>=0,limits[:,1]-mu,limits[:,0]-mu),d,
                            out=np.full_like(d,np.inf),where=np.abs(d)>1e-12),axis=1)
    ratio=np.minimum(ratio,maximum); values=mu+d*ratio[:,None]; values[:,0]=np.rint(values[:,0])
    for j,name in enumerate(CONT):
        out.loc[mask,name]=values[mask,j].astype(int) if name=='age' else values[mask,j]
    audit['radius_preservation']=radius_strength==0
    audit['radius_quantile_strength']=radius_strength
    return out,audit

def synthesize(b,cfg,majority_strength,uncertainty_strength,tail_strength=0.,radial_strength=0.,outcome_steps=80,uncertainty_scope='all',calibrate_rare_rate=False,uncertainty_group='all',tail_radius_strength=0.,empirical_tail=False,active_range_limits=False,significance_margin=2.1,range_strategy='global',uncertainty_shape='linear'):
    if uncertainty_group not in ('all','majority','stable_majority') or not 0<=majority_strength<=1:
        raise ValueError('Invalid majority settings')
    c=load_base(); detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    c,ma=majority_copula(b,c,detector,majority_strength)
    c,ra=radial(b,c,detector,radial_strength,cfg)
    before_scale=c.copy(deep=True)
    active=(radii(c,detector)<detector['band']['lo']) if active_range_limits and uncertainty_group!='all' else None
    c,ua=uncertainty_scale(b,c,cfg,uncertainty_strength,uncertainty_scope,active,range_strategy,uncertainty_shape)
    if uncertainty_group in ('majority','stable_majority'):
        rare=radii(before_scale,detector)>=detector['band']['lo']
        if uncertainty_group=='stable_majority':
            rare=rare | (radii(c,detector)>=detector['band']['lo'])
        c.loc[rare]=before_scale.loc[rare]
        ua['protected_rows']=int(rare.sum())
    ua['group']=uncertainty_group
    c,ta=tail_correction(b,c,cfg,detector,tail_strength,tail_radius_strength,empirical_tail)
    mask=radii(c,detector)>=detector['band']['lo']
    rare_target=int(np.rint(b.loc[radii(b,detector)>=detector['band']['lo'],'onset'].mean()*mask.sum())) if calibrate_rare_rate else None
    c,oa=calibrate_outcomes(b,c,mask,max_steps=outcome_steps,significance_guard=True,rare_target=rare_target,significance_margin=significance_margin)
    for name,(lo,hi) in cfg['schema']['ranges'].items():
        if not c[name].between(lo,hi).all():
            raise ValueError('Range violation: '+name)
    return c,{'majority_copula':ma,'radial':ra,'uncertainty':ua,'tail_copula':ta,'outcomes':oa}
