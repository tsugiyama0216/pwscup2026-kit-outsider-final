"""Distil frozen R11 into a deterministic empirical-copula interpolation model.

Training reads synthetic R11 and the public detector only. Generation reads
the fitted model only; it needs no source B, Cox fit, or numerical optimizer.
"""
import numpy as np
import pandas as pd

LABS = ['BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
CONT = ['age', *LABS]


def rare_mask(frame, detector):
    delta = frame[CONT].to_numpy(float) - np.asarray(detector['mu'])
    radius = np.sqrt(np.maximum(0, np.einsum(
        'ij,jk,ik->i', delta, np.linalg.pinv(detector['sigma']), delta)))
    return radius >= detector['band']['lo']


def fit(frame, detector, knots, preserve_boundary=True):
    """Keep joint synthetic ranks; replace majority marginals by few knots."""
    if not isinstance(knots, int) or knots < 2:
        raise ValueError('At least two knots are required')
    rare = rare_mask(frame, detector)
    fixed = frame.drop(columns=LABS)
    model = dict(version=1, columns=list(frame.columns),
                 fixed={name: fixed[name].tolist() for name in fixed},
                 majority_knots=knots, groups={})
    for label, mask in [('majority', ~rare), ('rare', rare)]:
        indices = np.flatnonzero(mask)
        n = len(indices)
        if n < 2:
            raise ValueError('Both groups need at least two synthetic rows')
        # Rare-group values are retained exactly, to protect its utility.
        positions = np.arange(n) if label == 'rare' else np.linspace(0, n-1, min(knots, n))
        group = dict(indices=indices.tolist(), positions=positions.tolist(), labs={})
        for name in LABS:
            values = frame.loc[mask, name].to_numpy(float)
            order = np.argsort(values, kind='stable')
            ranks = np.empty(n, dtype=int)
            ranks[order] = np.arange(n)
            support = np.interp(positions, np.arange(n), values[order])
            group['labs'][name] = dict(ranks=ranks.tolist(), values=support.tolist())
        model['groups'][label] = group
    if preserve_boundary:
        interpolated = generate(model)
        exceptions = np.flatnonzero(rare_mask(interpolated, detector) != rare)
        model['boundary_exceptions'] = dict(
            indices=exceptions.tolist(),
            labs={name:frame.loc[exceptions,name].tolist() for name in LABS})
    return model


def generate(model):
    """Read each synthetic rank and interpolate its marginal value."""
    if model['version'] != 1:
        raise ValueError('Unsupported model version')
    out = pd.DataFrame(model['fixed'])
    for name in LABS:
        out[name] = np.nan
    covered = []
    for group in model['groups'].values():
        indices = group['indices']
        covered.extend(indices)
        positions = np.asarray(group['positions'])
        if not (np.diff(positions) > 0).all():
            raise ValueError('Knot positions must increase strictly')
        for name, marginal in group['labs'].items():
            ranks = marginal['ranks']
            if sorted(ranks) != list(range(len(indices))):
                raise ValueError('Each group must contain every rank exactly once')
            out.loc[indices, name] = np.interp(ranks, positions, marginal['values'])
    if sorted(covered) != list(range(len(out))):
        raise ValueError('Groups must partition the synthetic rows')
    if 'boundary_exceptions' in model:
        saved = model['boundary_exceptions']
        for name, values in saved['labs'].items():
            out.loc[saved['indices'], name] = values
    return out[model['columns']]
