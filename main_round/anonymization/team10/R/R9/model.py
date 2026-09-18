"""Aggregate-constrained refinement of frozen R8 synthetic records."""
import json
import numpy as np
import pandas as pd
from evaluation import HERE,DATA,COLUMNS,digest
from cox_calibration import features,FEATURES,MUTABLE
from copula_helpers import fit_information,CONT

BASE_SHA256='d9892abc49f506f57bb890da6e5f8ae606f11af592606ffc015fd85913c5ed6b'

def load_base():
    path=HERE.parent/'R8/C.csv'
    if digest(path)!=BASE_SHA256:
        raise RuntimeError('Frozen R8 changed')
    return pd.read_csv(path,float_precision='round_trip')[COLUMNS]

def radii(c,detector):
    d=c[CONT].to_numpy(float)-np.asarray(detector['mu'])
    return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',d,np.linalg.pinv(detector['sigma']),d)))

def synthesize(b,cfg,steps=200,marginal_weight=4.,rare_weight=4.,correlation_weight=100.,information_weight=100.,movement_weight=.02,trust_radius=1.,membership_weight=100.,preserve_membership=False,balance_prefecture=False,balance_age=False):
    from aggregate_optimizer import refine
    c=load_base()
    categorical=None
    if balance_prefecture:
        from categorical import calibrate_prefecture
        c,categorical=calibrate_prefecture(b,c)
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    age_audit=None
    if balance_age:
        from categorical import calibrate_age
        c,age_audit=calibrate_age(b,c,cfg,detector)
    out,audit=refine(b,c,cfg,detector,steps,marginal_weight,rare_weight,correlation_weight,information_weight,movement_weight,trust_radius,membership_weight,preserve_membership)
    audit['categorical']=categorical
    audit['age']=age_audit
    return out,audit
