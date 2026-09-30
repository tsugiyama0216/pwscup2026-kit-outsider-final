"""Copula AIA membership features from the seven public challenge attributes."""
import numpy as np

from ..features import QI_CONT
from ..v9.features import local_marginals
from ..v11.features import extract as neighbor_extract
from local_eval.data import QI


def extract(c,background,challenge,neighbor_x=None):
    query=challenge[QI].reset_index(drop=True)
    if neighbor_x is None:
        neighbor_x,_,_,_=neighbor_extract(c,background,query)
    features=[neighbor_x]
    for col in QI_CONT:
        release=local_marginals(query,c,col)
        own=local_marginals(query,query,col,self_reference=True)
        independent=local_marginals(query,background,col)
        features.extend([release,own,independent,release-own,release-independent])
    for col in ('sex','prefecture'):
        values=query[col]
        release=values.map(c[col].value_counts(normalize=True)).fillna(0).to_numpy(float)
        own=values.map(query[col].value_counts(normalize=True)).fillna(0).to_numpy(float)
        independent=values.map(background[col].value_counts(normalize=True)).fillna(0).to_numpy(float)
        features.append(np.column_stack([release,own,independent,release-own,release-independent]))
    result=np.column_stack(features).astype(np.float32)
    if not np.isfinite(result).all():raise ArithmeticError('Nonfinite copula AIA membership features')
    return result
