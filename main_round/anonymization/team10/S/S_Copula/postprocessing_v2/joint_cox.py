"""Combine already released independent DP Cox measurements; no private API."""
from fractions import Fraction
import hashlib
import json
import numpy as np


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def combine_cox(models, *, prior_multiplier=1.0, max_step=1.0):
    if not models or prior_multiplier <= 0 or max_step <= 0:
        raise ValueError('A nonempty set and positive public settings are required')
    public=models[0]['public_cox']; key=identity(public)
    if any(identity(m['public_cox']) != key for m in models):
        raise ValueError('Cox measurements must use identical public row maps')
    if len({identity(m) for m in models}) != len(models):
        raise ValueError('Repeated model is not an independent measurement')
    size=models[0]['cox']['public_n']; q=public['quantization']; sensitivity=2*q
    vectors=[]; variances=[]; budgets=[]
    for m in models:
        if m['cox']['public_n'] != size or m['cox_mechanism']['l1_sensitivity'] != sensitivity:
            raise ValueError('Incompatible sensitivity or fixed row count')
        epsilon=Fraction(m['accounting']['cox_epsilon_exact'])
        if Fraction(m['cox_mechanism']['epsilon_allocated_exact']) != epsilon:
            raise ValueError('Cox mechanism accounting mismatch')
        e=float(epsilon)
        variances.append(2*sensitivity*(sensitivity+e)/e**2)
        vectors.append([float(max(-10**12,min(10**12,int(v)))) for v in m['cox']['noisy_score_integer']])
        budgets.append(Fraction(m['accounting']['epsilon_exact']))
    if np.asarray(vectors).shape != (len(models),10):
        raise ValueError('Expected ten dimensional DP score vectors')
    precision=1/np.asarray(variances)
    weights=precision/precision.sum()
    combined=weights@np.asarray(vectors)
    variance=float(1/precision.sum())*(public['clip_norm']/q/size)**2
    mean=(combined/size-np.asarray(public['public_integer_mean']))*public['clip_norm']/q
    prior=np.asarray(public['public_score_covariance'])*prior_multiplier/size
    shrink=prior@np.linalg.inv(prior+variance*np.eye(10))
    information=np.asarray(public['information_inverse'])
    step=np.clip(information@shrink@mean,-max_step,max_step)
    beta=np.asarray(public['beta_standardized'])+step[1:]
    posterior=information@(prior-shrink@prior)@information.T
    return {'kind':'joint_previously_released_DP_Cox_target',
        'parent_epsilon_exact':str(sum(budgets,Fraction())), 'additional_epsilon_exact':'0','delta':0,
        'independent_parent_measurement_weights':weights.tolist(),
        'public_map_sha256':key,'beta_standardized':beta.tolist(),
        'beta_original_units':(beta/np.asarray(public['scale'])).tolist(),
        'log_baseline_multiplier':float(step[0]),'combined_DP_noise_variance_of_score_mean':variance,
        'prior_variance_multiplier':prior_multiplier,'shrinkage_trace':float(np.trace(shrink)),
        'approximate_posterior_covariance':posterior.tolist(),
        'uncertainty_scope':'Public-prior linear approximation, not a private coefficient accuracy guarantee',
        'selection_inputs':'Saved DP measurements and public constants only'}
