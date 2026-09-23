"""Public influence-regression target estimation; no private fit/read API."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "postprocessing_v2")]
import dp_cox
import dp_copula
from cox_optimizer import EfronRiskSets


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def public_influence(public: pd.DataFrame, public_model: dict):
    """Return influence of public case weights on unpenalized Efron Cox beta.

    Differentiate the weighted Efron score at unit weights, including the
    average event-weight factor in tied event sets. The result is n times the
    derivative of fitted beta with respect to each row weight. Only the
    expressly public background is permitted here.
    """
    x = (dp_cox._features(public)-np.asarray(public_model["center"]))/np.asarray(public_model["scale"])
    order = np.argsort(public.time.to_numpy(float), kind="stable")
    x = x[order]
    time, event = public.time.to_numpy(float)[order], public.onset.to_numpy(float)[order]
    beta = np.asarray(public_model["beta_standardized"])
    risk = EfronRiskSets(time, event)
    score, information, _ = risk.evaluate(x, beta, information=True)
    if np.linalg.norm(score) > .1:
        raise ValueError("Public data must match the saved public Cox nuisance model")
    weight = np.exp(x @ beta - np.max(x @ beta))
    weighted = weight[:, None] * x
    r0 = np.cumsum(weight[::-1])[::-1][risk.first]
    r1 = np.cumsum(weighted[::-1], axis=0)[::-1][risk.first]
    d0 = np.bincount(risk.inverse, weights=weight*event)
    d1 = np.zeros_like(r1)
    np.add.at(d1, risk.inverse, weighted*event[:, None])
    g, f = risk.tie_group, risk.fraction
    denominator = r0[g] - f*d0[g]
    mean = (r1[g]-f[:, None]*d1[g])/denominator[:, None]
    group_count = len(risk.first)
    a0 = np.bincount(g, weights=1/denominator, minlength=group_count)
    a1 = np.bincount(g, weights=f/denominator, minlength=group_count)
    v0, v1, event_mean = np.zeros_like(r1), np.zeros_like(r1), np.zeros_like(r1)
    np.add.at(v0, g, mean/denominator[:, None])
    np.add.at(v1, g, mean*(f/denominator)[:, None])
    np.add.at(event_mean, g, mean)
    counts = np.bincount(risk.inverse, weights=event)
    event_mean /= np.maximum(counts, 1)[:, None]
    a = np.cumsum(a0)[risk.inverse]-event*a1[risk.inverse]
    v = np.cumsum(v0, axis=0)[risk.inverse]-event[:, None]*v1[risk.inverse]
    score_influence = event[:, None]*(x-event_mean[risk.inverse])-weight[:, None]*(a[:, None]*x-v)
    influence_sorted = len(x)*score_influence @ np.linalg.inv(information)
    influence = np.empty_like(influence_sorted)
    influence[order] = influence_sorted
    return influence, information


def fit_public_regression(public: pd.DataFrame, public_model: dict, *, schema=None) -> dict:
    """Fit row-level covariance regression exclusively on public background."""
    influence, information = public_influence(public, public_model)
    observation = dp_cox._integer_rows(public, public_model).astype(float)
    observation *= public_model["clip_norm"]/public_model["quantization"]
    if schema is not None:
        groups = dp_copula._group_ids(public, schema["rare_detector"])
        group_rows = np.eye(24)[groups]
        observation = np.column_stack([observation, group_rows])
    both = np.cov(np.column_stack([influence, observation]), rowvar=False)
    return {
        "kind": "public_Efron_influence_clipped_score_regression",
        "public_map_sha256": identity(public_model), "public_rows": len(public),
        "influence_covariance": both[:9, :9].tolist(),
        "cross_covariance": both[:9, 9:].tolist(),
        "score_covariance": both[9:, 9:].tolist(),
        "information_per_public_row": (information/len(public)).tolist(),
        "public_influence_mean": influence.mean(axis=0).tolist(),
        "includes_group_counts": schema is not None,
        "public_group_probabilities": None if schema is None else group_rows.mean(axis=0).tolist(),
        "public_schema_sha256": None if schema is None else identity(schema),
        "fit_input": "public background only",
    }


def estimate_target(models: list[dict], regression: dict, *, prior_multiplier=1.0) -> dict:
    """Condition public beta influence on the saved noisy clipped score vector.

    Assumes independent parent measurements of the same public bounded row
    map. The reported posterior covariance is a public Gaussian approximation,
    not a privacy or frequentist confidence bound.
    """
    if not models or prior_multiplier <= 0:
        raise ValueError("Models and a positive prior multiplier are required")
    public = models[0]["public_cox"]
    key = identity(public)
    if key != regression["public_map_sha256"] or any(identity(m["public_cox"]) != key for m in models):
        raise ValueError("Public model mismatch")
    if len({identity(m) for m in models}) != len(models):
        raise ValueError("Duplicate parent model")
    n = models[0]["cox"]["public_n"]
    q, clip = public["quantization"], public["clip_norm"]
    sensitivity = 2*q
    vectors, variances, budgets = [], [], []
    for model in models:
        epsilon = Fraction(model["accounting"]["cox_epsilon_exact"])
        mechanism = model["cox_mechanism"]
        if (model["cox"]["public_n"] != n or mechanism["l1_sensitivity"] != sensitivity or
                Fraction(mechanism["epsilon_allocated_exact"]) != epsilon):
            raise ValueError("Incompatible Cox measurement")
        vectors.append([float(max(-10**12, min(10**12, int(v)))) for v in model["cox"]["noisy_score_integer"]])
        e = float(epsilon)
        variances.append(2*sensitivity*(sensitivity+e)/e**2)
        budgets.append(Fraction(model["accounting"]["epsilon_exact"]))
    if np.asarray(vectors).shape != (len(models), 10):
        raise ValueError("Wrong score dimension")
    precision = 1/np.asarray(variances)
    weights = precision/precision.sum()
    observed = (weights @ np.asarray(vectors)/n-np.asarray(public["public_integer_mean"]))*clip/q
    noise = (1/precision.sum())*(clip/q/n)**2
    noise_diagonal = np.full(10, noise)
    if regression.get("includes_group_counts", False):
        group_vectors, group_variances = [], []
        for model in models:
            copula = model["copula"]
            if identity(copula["schema"]) != regression["public_schema_sha256"]:
                raise ValueError("Public group definition mismatch")
            query = next(q for q in copula["privacy"]["queries"] if q["name"] == "group_counts")
            group_epsilon = Fraction(query["epsilon_allocated_exact"])
            group_sensitivity = query["l1_sensitivity"]
            expected_q = Fraction(group_sensitivity)/(group_sensitivity+group_epsilon)
            if Fraction(query["geometric_q_numerator"], query["geometric_q_denominator"]) != expected_q:
                raise ValueError("Invalid group noise accounting")
            group_variances.append(2*float(expected_q)/(1-float(expected_q))**2)
            group_vectors.append([float(max(-10**12, min(10**12, int(v)))) for v in copula["dp_group_counts"]])
        group_precision = 1/np.asarray(group_variances)
        group_mean = (group_precision/group_precision.sum()) @ np.asarray(group_vectors)/n
        group_noise = (1/group_precision.sum())/n**2
        observed = np.r_[observed, group_mean-np.asarray(regression["public_group_probabilities"])]
        noise_diagonal = np.r_[noise_diagonal, np.full(24, group_noise)]
    covariance = np.asarray(regression["score_covariance"])*prior_multiplier/n
    cross = np.asarray(regression["cross_covariance"])*prior_multiplier/n
    prior = np.asarray(regression["influence_covariance"])*prior_multiplier/n
    mapping = np.linalg.solve(covariance+np.diag(noise_diagonal), cross.T).T
    step = mapping @ observed
    posterior = prior-mapping @ cross.T
    posterior = (posterior+posterior.T)/2
    beta = np.asarray(public["beta_standardized"])+step
    return {
        "kind": "public_influence_regression_DP_Cox_target",
        "parent_epsilon_exact": str(sum(budgets, Fraction())), "additional_epsilon_exact": "0", "delta": 0,
        "public_map_sha256": key, "beta_standardized": beta.tolist(),
        "beta_original_units": (beta/np.asarray(public["scale"])).tolist(),
        "posterior_beta_covariance": posterior.tolist(),
        "prior_variance_multiplier": prior_multiplier,
        "independent_parent_measurement_weights": weights.tolist(),
        "combined_DP_noise_variance_of_score_mean": float(noise),
        "includes_group_counts": bool(regression.get("includes_group_counts", False)),
        "target_standard_error_public": np.sqrt(np.diag(np.linalg.inv(np.asarray(regression["information_per_public_row"])*n))).tolist(),
        "selection_inputs": "saved DP score vectors, public covariance regression only",
        "uncertainty_scope": "Public Gaussian sampling approximation, not a private accuracy guarantee",
    }
