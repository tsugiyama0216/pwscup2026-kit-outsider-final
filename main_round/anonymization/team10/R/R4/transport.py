"""B-only partial marginal quantile transport of synthetic laboratory values."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

LABS = ['BMI','SBP','TG','HDL','ALT','FPG']


def transport(b, c, cfg, strength=0.5, scope='global'):
    """Blend synthetic values toward B marginal quantiles without row matching.

    Mid-rank probabilities avoid matching source extrema. For strength < 1,
    strict order and ties within each group are preserved. No rare label or
    evaluation reference enters this function; all six labs use one strength.
    """
    if isinstance(strength,(bool,np.bool_)) or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('strength must lie in [0,1]')
    if scope not in ('global','sex'):
        raise ValueError('scope must be global or sex')
    if not len(b) or len(b)!=len(c):
        raise ValueError('B and C must have equal nonzero row counts')
    for frame in (b,c):
        if not np.isfinite(frame[LABS].to_numpy(dtype=float)).all():
            raise ValueError('Laboratory values must be finite')
    out=c.copy(deep=True)
    groups=[None] if scope=='global' else sorted(c.sex.unique().tolist())
    diagnostics=[]
    for group in groups:
        mask=np.ones(len(c),dtype=bool) if group is None else c.sex.to_numpy()==group
        pool=b if group is None else b.loc[b.sex==group]
        if len(pool)<2:
            raise ValueError('Target group needs at least two rows')
        record={'sex':group,'source_rows':len(pool),'synthetic_rows':int(mask.sum()),'labs':{}}
        for name in LABS:
            original=c.loc[mask,name].to_numpy(dtype=float)
            p=(rankdata(original,method='average')-0.5)/len(original)
            target=np.quantile(pool[name].to_numpy(dtype=float),p,method='linear')
            values=original+strength*(target-original)
            lo,hi=cfg['schema']['ranges'][name]
            if not np.isfinite(values).all() or ((values<lo)|(values>hi)).any():
                raise ValueError('Transported value outside official range: '+name)
            if strength!=0:
                out.loc[mask,name]=values
            record['labs'][name]={'rms_change':float(np.sqrt(np.mean((values-original)**2))),
                                 'maximum_change':float(np.max(np.abs(values-original)))}
        diagnostics.append(record)
    return out,{'method':'partial_midrank_quantile_transport','strength':float(strength),
                'scope':scope,'columns':LABS,'groups':diagnostics,'source_row_matching':False,
                'rare_labels_used':False,'clipped_counts':{name:0 for name in LABS},
                'formal_differential_privacy':False}
