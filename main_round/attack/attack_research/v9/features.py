import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from ..features import CLINICAL


def local_marginals(query, reference, column, self_reference=False):
    q = query[column].to_numpy(float)
    r = np.sort(reference[column].to_numpy(float))
    scale = max(float(np.std(r)), 1e-6)
    index = np.searchsorted(r, q, side='left')
    right = np.searchsorted(r, q, side='right')
    # Remove one known self observation for the mostly-negative candidate reference.
    equality = np.maximum(right-index-int(self_reference), 0)
    features = [(index+right)/(2*len(r)), equality / len(r),
                np.maximum(r[0]-q, 0)/scale, np.maximum(q-r[-1], 0)/scale]
    left_distance = np.abs(q-r[np.clip(index-1, 0, len(r)-1)]) / scale
    right_distance = np.abs(q-r[np.clip(right, 0, len(r)-1)]) / scale
    features += [np.log(left_distance+1e-7), np.log(right_distance+1e-7),
        np.log((left_distance+1e-7)/(right_distance+1e-7))]
    for k in (1, 2, 4, 8, 16, 32):
        lo = r[np.clip(index-k, 0, len(r)-1)]
        hi = r[np.clip(right+k-1, 0, len(r)-1)]
        features.append(np.log(np.maximum(hi-lo, 0)/scale+1e-7))
    return np.column_stack(features)


def gaussian_profile(query, reference, columns, ridge):
    qr, rr = [], []
    for col in columns:
        q = query[col].to_numpy(float)
        r = reference[col].to_numpy(float)
        sr = np.sort(r)
        qr.append(ndtri(np.clip((np.searchsorted(sr,q,'left')+np.searchsorted(sr,q,'right')+1)/(2*(len(sr)+1)), 1e-4, 1-1e-4)))
        rr.append(ndtri((rankdata(r, method='average')-.5)/len(r)))
    qz, rz = np.column_stack(qr), np.column_stack(rr)
    cov = np.cov(rz, rowvar=False) + np.eye(len(columns))*ridge
    centered = qz-rz.mean(axis=0)
    quad = np.einsum('ij,jk,ik->i', centered, np.linalg.pinv(cov), centered)
    return np.column_stack([-.5*(quad+np.linalg.slogdet(cov)[1]), quad,
        np.max(np.abs(centered), axis=1)])


def extract(c, background, candidates):
    query = candidates.reset_index(drop=True)
    features = []
    for col in CLINICAL + ['time']:
        released = local_marginals(query, c, col)
        reference = local_marginals(query, query, col, self_reference=True)
        independent = local_marginals(query, background, col)
        features.extend([released, reference, independent,
            released-reference, released-independent])
    for cols in [CLINICAL, CLINICAL+['time'], CLINICAL+['sex', 'smoking', 'onset', 'death']]:
        for ridge in (.01, .1):
            released = gaussian_profile(query, c, cols, ridge)
            reference = gaussian_profile(query, query, cols, ridge)
            independent = gaussian_profile(query, background, cols, ridge)
            features.extend([released, reference, independent, released-reference, released-independent])
    for col in ['sex', 'prefecture', 'smoking', 'onset', 'death']:
        values = query[col]
        release_frequency = values.map(c[col].value_counts(normalize=True)).fillna(0).to_numpy()
        own_counts = query[col].value_counts()
        reference_frequency = ((values.map(own_counts).to_numpy()-1)/(len(query)-1))
        features.append(np.column_stack([release_frequency, reference_frequency, release_frequency-reference_frequency]))
    out = np.column_stack(features).astype(np.float32)
    if not np.isfinite(out).all():
        raise ArithmeticError('Nonfinite marginal features')
    return out
