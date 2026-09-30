import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import rankdata

from ..features import CLINICAL, canonical
from local_eval.data import FEATURES


def log_interval(lo, hi):
    """Stable log probability of a standard normal in [lo,hi]."""
    # Reflect positive-tail intervals before subtracting their CDFs.
    a = np.where(lo > 0, -hi, lo)
    b = np.where(lo > 0, -lo, hi)
    lb, la = log_ndtr(b), log_ndtr(a)
    return lb + np.log(-np.expm1(np.minimum(la - lb, -1e-15)))


def observation_loglikelihood(donors, released, background, schema, strength):
    """Rows are potential donors; columns are observations; no hidden labels."""
    result = np.zeros((len(donors), len(released)))
    for col in CLINICAL + ['time']:
        x = donors[col].to_numpy(float)[:, None]
        y = released[col].to_numpy(float)[None, :]
        sd = strength * float(background[col].std(ddof=0)) if col != 'time' else strength * 2
        sd = max(sd, 1e-6)
        if col == 'age':
            lower, upper = schema['age_range']
            ll = log_interval((y - .5 - x) / sd, (y + .5 - x) / sd)
            ll = np.where(y <= lower, log_ndtr((lower + .5 - x) / sd), ll)
            ll = np.where(y >= upper, log_ndtr((x - upper + .5) / sd), ll)
        else:
            lower, upper = [1 / 365.25, 10.] if col == 'time' else schema['ranges'][col]
            ll = -.5 * ((y-x) / sd) ** 2 - np.log(sd * np.sqrt(2*np.pi))
            ll = np.where(y <= lower, log_ndtr((lower-x) / sd), ll)
            ll = np.where(y >= upper, log_ndtr((x-upper) / sd), ll)
        result += ll
    for col in ['sex', 'prefecture', 'smoking', 'onset', 'death']:
        match = donors[col].to_numpy()[:, None] == released[col].to_numpy()[None, :]
        result = np.where(match, result, -np.inf)
    return result


def predictions(c, background, candidates, schema):
    c, background = canonical(c, FEATURES), canonical(background, FEATURES)
    query = candidates[FEATURES].reset_index(drop=True)
    output = {}
    for strength in (.05, .1, .2, .35, .5, .75):
        ll = observation_loglikelihood(query, c, background, schema, strength)
        background_ll = observation_loglikelihood(background, c, background, schema, strength)
        # A public candidate cohort is mostly negative, but contamination is acknowledged.
        bg = logsumexp(background_ll, axis=0) - np.log(len(background))
        mixed = np.logaddexp(bg, logsumexp(ll, axis=0)-np.log(len(query))) - np.log(2)
        for name, null in [('background', bg), ('mixed', mixed)]:
            # Missing categorical strata cannot support a calibrated density ratio.
            supported = np.isfinite(null)
            ratio = ll[:, supported] - null[supported]
            for aggregation, raw in [('sum', logsumexp(ratio, axis=1)), ('max', ratio.max(axis=1))]:
                output[f'{name}_{aggregation}_{strength:g}'] = rankdata(raw, method='average') / len(query)
        output[f'nearest_{strength:g}'] = rankdata(ll.max(axis=1), method='average') / len(query)
    return output
