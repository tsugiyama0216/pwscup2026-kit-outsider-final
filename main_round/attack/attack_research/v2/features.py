"""Add smooth density evidence for releases that do not retain individual rows."""
import numpy as np
from scipy.stats import gaussian_kde, norm, rankdata

from .. import features as base
from ..features import proposal_features


def density_features(query, c, background, columns):
    features = []
    for logs in (False, True):
        q = base.transformed(query, background, columns, logs)
        r = base.transformed(c, background, columns, logs)
        b = base.transformed(background, background, columns, logs)
        try:
            kr, kb = gaussian_kde(r.T), gaussian_kde(b.T)
            lr, lb = kr.logpdf(q.T), kb.logpdf(q.T)
            ratio = np.clip(lr-lb, -100, 100)
        except np.linalg.LinAlgError:
            lr = lb = ratio = np.zeros(len(query))
        features += [np.clip(lr,-1000,100), np.clip(lb,-1000,100), ratio,
                     rankdata(ratio,method="average")/len(query)]
    for col in columns:
        q = query[col].to_numpy(float)
        r, b = np.sort(c[col].to_numpy(float)), np.sort(background[col].to_numpy(float))
        zr = norm.ppf(np.clip(np.searchsorted(r,q,side="right")/len(r),1e-4,1-1e-4))
        zb = norm.ppf(np.clip(np.searchsorted(b,q,side="right")/len(b),1e-4,1-1e-4))
        scale = max(float(np.std(b)),1e-6)
        ri = np.clip(np.searchsorted(r,q),1,len(r)-1)
        bi = np.clip(np.searchsorted(b,q),1,len(b)-1)
        dr = np.minimum(abs(q-r[ri]),abs(q-r[ri-1]))/scale
        db = np.minimum(abs(q-b[bi]),abs(q-b[bi-1]))/scale
        features += [zr, zb, zr-zb, dr, db, np.log((db+1e-9)/(dr+1e-9))]
    # A public multivariate outlier diagnostic, not the hidden evaluation mask.
    q = base.transformed(query,background,columns,True)
    for reference in (c,background):
        values = base.transformed(reference,background,columns,True)
        centered = q-values.mean(axis=0)
        cov = np.cov(values,rowvar=False)
        inv = np.linalg.pinv(cov + np.eye(len(columns))*.01)
        features.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i",centered,inv,centered),0)))
    return np.column_stack(features)


def extract_mia(c, background, candidates):
    x, rule = base.extract_mia(c,background,candidates)
    extra = density_features(candidates,c,background,base.CLINICAL)
    return np.column_stack([x,extra]).astype(np.float32), rule


def extract_aia(c, background, challenge):
    x, proposals, rule, times = base.extract_aia(c,background,challenge)
    query = challenge[base.QI].copy()
    extra = density_features(query,c,background,base.QI_CONT)
    return np.column_stack([x,extra]).astype(np.float32), proposals, rule, times
