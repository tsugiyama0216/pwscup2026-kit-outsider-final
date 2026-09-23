"""Public/DP-only Bayes decisions for the official coefficient-overlap loss.

This is postprocessing, not a private measurement. The Gaussian coefficient
posterior and the target standard errors are public-model approximations.
Neither the decision nor its expected score is an actual-cohort accuracy bound.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.optimize import minimize_scalar

Z95 = 1.959963984540054


def _normal_integral(lo, hi, mean, sd):
    """Probability and unnormalized first moment over [lo, hi]."""
    lo, hi = np.broadcast_arrays(lo, hi)
    a, b = (lo-mean)/sd, (hi-mean)/sd
    probability = np.maximum(ndtr(b)-ndtr(a), 0)
    moment = mean*probability + sd*(np.exp(-a*a/2)-np.exp(-b*b/2))/np.sqrt(2*np.pi)
    valid = hi > lo
    return np.where(valid, probability, 0), np.where(valid, moment, 0)


def expected_contribution(candidate, mean, sd, true_se, synthetic_se):
    """Exact 1-D Gaussian expectation of significance-gated interval overlap.

    Uses fixed positive SEs. Integration, rather than a smooth replacement,
    preserves the official discontinuity when significance labels differ.
    """
    if not np.isfinite([mean, sd, true_se, synthetic_se]).all() or min(sd,true_se,synthetic_se)<=0:
        raise ValueError('Finite mean and strictly positive SD/SE are required')
    c = np.asarray(candidate, float)
    if not np.isfinite(c).all():
        raise ValueError('Candidate must be finite')
    hb, hc = Z95*true_se, Z95*synthetic_se
    lower = np.where(c>hc, hb, np.where(c < -hc, -np.inf, -hb))
    upper = np.where(c < -hc, -hb, np.where(c>hc, np.inf, hb))
    plateau, reach, height = abs(hb-hc), hb+hc, 2*min(hb,hc)
    total = np.zeros_like(c)
    for lo,hi,intercept,slope in (
        (c-reach,c-plateau,reach-c,1),
        (c-plateau,c+plateau,np.full_like(c,height),0),
        (c+plateau,c+reach,c+reach,-1),
    ):
        probability, moment = _normal_integral(np.maximum(lo,lower),np.minimum(hi,upper),mean,sd)
        total += intercept*probability+slope*moment
    return np.clip(.25*(1/hb+1/hc)*total,0,1)


def choose_coefficient(mean, sd, true_se, synthetic_se, *, boundary_margin=.1):
    """Maximize expected contribution separately in the three label regions."""
    if not 0 < boundary_margin < 1:
        raise ValueError('Boundary margin must be between zero and one SE')
    h = Z95*synthetic_se
    pad = boundary_margin*synthetic_se
    span = max(abs(mean)+8*sd, h+4*(true_se+synthetic_se))
    intervals = [(-span,-h-pad),(-h+pad,h-pad),(h+pad,span)]
    candidates=[float(mean)]
    for lo,hi in intervals:
        result=minimize_scalar(lambda c:-float(expected_contribution(c,mean,sd,true_se,synthetic_se)),
                               bounds=(lo,hi),method='bounded',options={'xatol':1e-10})
        candidates.extend([lo,hi,float(result.x)])
    values=expected_contribution(candidates,mean,sd,true_se,synthetic_se)
    # Preserve the posterior mean on exact ties. All inputs are DP/public.
    k=int(np.argmax(values))
    return candidates[k], float(values[k]), float(values[0])


def decide(mean, covariance, true_se, synthetic_se, *, boundary_margin=.1):
    mean,covariance,true_se,synthetic_se=map(np.asarray,(mean,covariance,true_se,synthetic_se))
    n=len(mean)
    if mean.shape!=(n,) or covariance.shape!=(n,n) or true_se.shape!=(n,) or synthetic_se.shape!=(n,):
        raise ValueError('Incompatible posterior shapes')
    if not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T,atol=1e-10):
        raise ValueError('Covariance must be finite and symmetric')
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-10:
        raise ValueError('Covariance must be positive semidefinite')
    sd=np.sqrt(np.maximum(np.diag(covariance),1e-16))
    rows=[choose_coefficient(m,s,b,c,boundary_margin=boundary_margin)
          for m,s,b,c in zip(mean,sd,true_se,synthetic_se)]
    return np.asarray([r[0] for r in rows]), {
        'kind':'Gaussian_public_prior_expected_official_IO_decision',
        'expected_contributions':[r[1] for r in rows],
        'posterior_mean_contributions':[r[2] for r in rows],
        'boundary_margin_SE':boundary_margin,
        'scope':'Approximate posterior expected IO, not actual U or an accuracy guarantee',
        'selection_inputs':'Saved DP posterior, public target SE, synthetic SE only',
    }
