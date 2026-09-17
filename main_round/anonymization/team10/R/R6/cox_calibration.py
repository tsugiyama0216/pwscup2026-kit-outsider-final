"""Aggregate Cox calibration of synthetic data; no source-row matching."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

FEATURES = ['age', 'sex', 'BMI', 'SBP', 'log_TG', 'HDL', 'log_ALT', 'FPG', 'smoking']
MUTABLE = ['BMI', 'SBP', 'log_TG', 'HDL', 'log_ALT', 'FPG']


def features(frame):
    x = frame[['age','sex','BMI','SBP','TG','HDL','ALT','FPG','smoking']].astype(float).copy()
    if not np.isfinite(x.to_numpy()).all() or (x[['TG','ALT']] <= 0).any().any():
        raise ValueError('Predictors must be finite; TG and ALT must be positive')
    x['log_TG'] = np.log(x.pop('TG'))
    x['log_ALT'] = np.log(x.pop('ALT'))
    return x[FEATURES]


def fit(frame, cause):
    train = features(frame).assign(time=frame.time.to_numpy(), event=frame[cause].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = CoxPHFitter(penalizer=0.0).fit(train, 'time', 'event')
    beta = model.params_.reindex(FEATURES).to_numpy()
    se = model.standard_errors_.reindex(FEATURES).to_numpy()
    if not np.isfinite(beta).all() or not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError('Invalid Cox fit')
    return beta, se


def calibrate(b, c, cfg, strength=1.0, causes=('onset',)):
    """Minimum covariance-metric displacement satisfying Cox coefficient targets.

    For transformed X'=X+(X-center)(beta_C-beta_B)V.T, choose V.T beta_B=I.
    At strength one, X' beta_B equals X beta_C up to a constant, before clipping.
    All nine coefficients enter symmetrically; discrete predictors stay unchanged.
    Public scoring references and private rare labels are not inputs.
    """
    if isinstance(strength, (bool, np.bool_)) or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must lie in [0,1]')
    if tuple(causes) not in [('onset',), ('onset','death')]:
        raise ValueError('Unsupported causes')
    xb, xc = features(b), features(c)
    if len(b) != len(c) or not len(c):
        raise ValueError('B and C must have equal nonzero row counts')
    targets, initial, fits = [], [], {}
    for cause in causes:
        beta_b, se_b = fit(b, cause)
        beta_c, se_c = fit(c, cause)
        targets.append(beta_b)
        initial.append(beta_c)
        fits[cause] = {'beta_B': beta_b.tolist(), 'beta_initial': beta_c.tolist(),
                       'se_B': se_b.tolist(), 'se_initial': se_c.tolist()}
    target, initial = np.column_stack(targets), np.column_stack(initial)
    positions = [FEATURES.index(name) for name in MUTABLE]
    covariance = np.cov(xb[MUTABLE].to_numpy(), rowvar=False, ddof=1)
    # Small fixed covariance shrinkage stabilizes correlated laboratory variables.
    covariance = 0.97 * covariance + 0.03 * np.diag(np.diag(covariance))
    beta = target[positions]
    gram = beta.T @ covariance @ beta
    if not np.isfinite(gram).all() or np.linalg.cond(gram) > 1e8 or np.linalg.eigvalsh(gram).min() <= 1e-12:
        raise ValueError('Degenerate target Cox directions')
    directions = np.linalg.solve(gram, (covariance @ beta).T).T
    displacement = strength * ((xc-xc.mean()).to_numpy() @ (initial-target)) @ directions.T
    transformed = xc[MUTABLE].to_numpy() + displacement
    out = c.copy(deep=True)
    clipped = {}
    for j, name in enumerate(MUTABLE):
        raw_name = {'log_TG':'TG','log_ALT':'ALT'}.get(name,name)
        values = transformed[:,j]
        if name.startswith('log_'):
            values = np.exp(values)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite transformed predictor')
        lo, hi = cfg['schema']['ranges'][raw_name]
        clipped[raw_name] = int(((values<lo)|(values>hi)).sum())
        if strength != 0:
            out[raw_name] = np.clip(values,lo,hi)
    audit = {'method':'covariance_metric_affine_cox_calibration', 'strength':float(strength),
             'causes':list(causes), 'features':FEATURES, 'mutable_features':MUTABLE,
             'covariance_shrinkage':0.03, 'target_fits':fits, 'clipped_counts':clipped,
             'directions':directions.tolist(), 'gram_condition':float(np.linalg.cond(gram)),
             'standardized_rms_change':dict(zip(MUTABLE,np.sqrt(np.mean(displacement**2,axis=0))/xb[MUTABLE].std(ddof=1).to_numpy())),
             'row_matching':False, 'formal_differential_privacy':False}
    return out,audit
