"""Likelihood of public noisy QI observations given a challenge QI donor."""
import numpy as np
from scipy.special import log_ndtr, logsumexp

from ..v8.likelihood import log_interval


NUMERIC = ('age', 'SBP', 'TG', 'HDL', 'ALT')


def observation_likelihood(challenge, released, background, schema, strength):
    result = np.zeros((len(challenge), len(released)), dtype=float)
    for col in NUMERIC:
        x = challenge[col].to_numpy(float)[:, None]
        y = released[col].to_numpy(float)[None, :]
        sd = max(strength*float(background[col].std(ddof=0)), 1e-6)
        if col == 'age':
            lower, upper = schema['age_range']
            ll = log_interval((y-.5-x)/sd, (y+.5-x)/sd)
            ll = np.where(y <= lower, log_ndtr((lower+.5-x)/sd), ll)
            ll = np.where(y >= upper, log_ndtr((x-upper+.5)/sd), ll)
        else:
            lower, upper = schema['ranges'][col]
            ll = -.5*((y-x)/sd)**2 - np.log(sd*np.sqrt(2*np.pi))
            ll = np.where(y <= lower, log_ndtr((lower-x)/sd), ll)
            ll = np.where(y >= upper, log_ndtr((x-upper)/sd), ll)
        result += ll
    for col in ('sex', 'prefecture'):
        match = challenge[col].to_numpy()[:, None] == released[col].to_numpy()[None, :]
        result = np.where(match, result, -np.inf)
    return result


def membership_features(challenge, c, background, schema, strength):
    ll = observation_likelihood(challenge, c, background, schema, strength)
    best = ll.max(axis=1)
    summed = logsumexp(ll, axis=1) - np.log(len(c))
    # For an otherwise unseen category, retain a finite low score.
    floor = np.finfo(float).min / 8
    best = np.where(np.isfinite(best), best, floor)
    summed = np.where(np.isfinite(summed), summed, floor)
    return {'best': best, 'summed': summed}
