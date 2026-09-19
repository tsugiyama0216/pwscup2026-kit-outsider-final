"""R9-based clinical tail quantile refinement without source-row linkage."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
R9=HERE.parent/'R9'
sys.path.insert(0,str(R9))
from evaluation import check_environment,read_training_data,evaluate_frame,DATA,digest
from model import radii
from aggregate_optimizer import restore_membership,refine

BASE_SHA='ae3d363321b58743e5f7ddf9e27b97e5525f1403c587d12ae6b5368311501136'
LABS=['BMI','SBP','TG','HDL','ALT','FPG']

def load_base():
    if digest(R9/'C.csv')!=BASE_SHA: raise RuntimeError('Frozen R9 changed')
    return pd.read_csv(R9/'C.csv',float_precision='round_trip')

def synthesize(b,cfg,phase=.5,strength=.5,steps=0,joint=False,rare_weight=4.,lock_tail=False):
    c=load_base()
    audit={}
    detector=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    if joint:
        from optimizer import refine as joint_refine
        out,audit=joint_refine(b,c,cfg,detector,steps,10.,rare_weight,300.,100.,.02,.5,10000.,True,tail_phase=phase)
        audit.update(tail_phase=phase,training_inputs='own B, frozen R9, schema; no true rare labels')
        return out,audit
    if steps and not lock_tail:
        c,audit['aggregate']=refine(b,c,cfg,detector,steps,10.,4.,300.,100.,.02,.25,10000.,True)
    rare=radii(c,detector)>=detector['band']['lo']
    rb=radii(b,detector)>=detector['band']['lo']
    n=int(rare.sum())
    if not 0<=phase<=1 or not 0<=strength<=1: raise ValueError('Invalid quantile parameters')
    out=c.copy(deep=True)
    levels=(np.arange(n)+phase)/n
    for name in LABS:
        values=c.loc[rare,name].to_numpy()
        order=np.argsort(values,kind='stable')
        target=np.quantile(b.loc[rb,name].to_numpy(),levels)
        shifted=values.copy()
        shifted[order]=(1-strength)*values[order]+strength*target
        out.loc[rare,name]=shifted
    out,restored=restore_membership(c,out,detector)
    if lock_tail:
        from optimizer import refine as joint_refine
        out,audit['aggregate']=joint_refine(b,out,cfg,detector,steps,10.,rare_weight,300.,100.,.02,.5,10000.,True,tail_phase=phase,locked_tail=True)
    audit.update(method='Clinical quantiles of schema-detected own B tail using synthetic ranks, followed by detector-class restoration',
                 phase=phase,strength=strength,rare_count=n,restored_rows=restored,source_row_matching=False,
                 training_inputs='own B, frozen R9 and schema; no true rare labels')
    return out,audit
