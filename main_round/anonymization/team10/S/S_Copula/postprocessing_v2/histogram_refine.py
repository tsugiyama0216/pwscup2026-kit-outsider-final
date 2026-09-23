"""Deterministic postprocessing of saved, independent DP histogram releases.

The approximation is Gaussian Bayes with the EXACT variance of the released
difference of geometric random variables. It does not claim an exact geometric
posterior. No private row, score, or data-dependent prior enters this module.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

try:
    from .. import dp_copula as dc
except ImportError:
    import dp_copula as dc


DEFAULT_PRIOR_MULTIPLIER = 1.0
GROUP_PUBLIC_PSEUDOCOUNT = 0.5


def geometric_variance(query: dict) -> float:
    """Var(G1-G2) = 2 q / (1-q)^2; q is recorded exactly."""
    q = Fraction(int(query["geometric_q_numerator"]),
                 int(query["geometric_q_denominator"]))
    if not 0 < q < 1:
        raise ValueError("Invalid geometric noise parameter")
    return float(2*q/(1-q)**2)


def _public_identity(model: dict) -> str:
    value = copy.deepcopy(model)
    for key in ("privacy", "default_sample_options", "dp_group_counts",
                "dp_prefecture_counts"):
        value.pop(key, None)
    for spec in value["groups"]:
        for margin in spec["margins"].values():
            margin.pop("dp_counts", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _prob(values):
    p = np.maximum(np.asarray(values, dtype=float), 0)
    if not p.sum() > 0:
        raise ValueError("Empty public prior")
    return p/p.sum()


def _apportion(values, total):
    mass = dc._simplex(np.asarray(values, float), float(total))
    result = np.floor(mass).astype(int)
    left = int(total)-int(result.sum())
    order = np.argsort(-(mass-result), kind="stable")
    result[order[:left]] += 1
    return result.tolist()


def _posterior(y, p, n, variance, multiplier, reference_n):
    """Public multinomial covariance plus uncertainty of public proportions."""
    p = _prob(p)
    covariance = (multiplier*n*(1+n/max(float(reference_n), 1.0)) *
                  (np.diag(p)-np.outer(p, p)))
    noise = np.diag(np.broadcast_to(variance, (len(p),)))
    mean = n*p + covariance @ np.linalg.solve(covariance+noise, np.asarray(y)-n*p)
    return dc._simplex(mean, n)


def refine_models(models: Sequence[dict], *,
                  prior_multiplier: float = DEFAULT_PRIOR_MULTIPLIER,
                  pool_group_totals: bool = True) -> dict:
    """Return a sample-compatible copula model with posterior count fields.

    Accepts full release wrappers or their ``copula`` dictionaries. Inputs must
    be independent releases of IDENTICAL queries/public specifications. The
    caller is responsible for establishing independence/provenance; passing
    an identical saved noisy release twice is rejected here. A wrapper's full
    epsilon is conservatively composed, including its existing Cox release.

    Consumers wanting continuous probabilities should read
    ``posterior_group_counts``, ``posterior_prefecture_counts``, and each
    margin's ``posterior_counts`` / ``posterior_probabilities``. ``dp_counts``
    are deterministically apportioned for unchanged ``dp_copula.sample``.
    """
    if not models or not np.isfinite(prior_multiplier) or prior_multiplier <= 0:
        raise ValueError("Require saved releases and a positive public prior multiplier")
    cores = [m.get("copula", m) for m in models]
    if any("histogram_postprocessing" in m for m in cores):
        raise ValueError("Supply original releases, not previously refined counts")
    identities = [_public_identity(m) for m in cores]
    if len(set(identities)) != 1:
        raise ValueError("Saved releases have different public/query specifications")
    digests = [hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
               for m in cores]
    if len(set(digests)) != len(digests):
        raise ValueError("Repeated release: independent noise is required")
    query_maps = [{q["name"]: q for q in m["privacy"]["queries"]} for m in cores]
    for model, qm in zip(cores, query_maps):
        if model["privacy"]["public_n"] != dc.PRIVATE_N:
            raise ValueError("Public row count must be 1049")
        if model["privacy"]["adjacency"] != "replace_one_fixed_public_size":
            raise ValueError("Unsupported adjacency")
        if set(qm) != set(query_maps[0]):
            raise ValueError("Query names differ")
        for name, query in qm.items():
            if query["l1_sensitivity"] != query_maps[0][name]["l1_sensitivity"]:
                raise ValueError("Query sensitivities differ")
    weights, variances = {}, {}
    for name in query_maps[0]:
        v = np.array([geometric_variance(q[name]) for q in query_maps])
        precisions = 1/v
        weights[name] = precisions/precisions.sum()
        variances[name] = float(1/precisions.sum())

    def combine(name, values):
        arrays = [dc._released_float(v) for v in values]
        if len({a.shape for a in arrays}) != 1:
            raise ValueError("Query vector dimensions differ")
        return sum(w*a for w, a in zip(weights[name], arrays))

    result = copy.deepcopy(cores[-1])
    n = dc.PRIVATE_N
    group_y = combine("group_counts", [m["dp_group_counts"] for m in cores])
    group_precision = np.full(len(group_y), 1/variances["group_counts"])
    group_numerator = group_y*group_precision
    combined_margins = {}
    for col in dc.NUMERIC:
        name = f"marginal_{col}"
        for gid in range(len(dc.GROUPS)):
            y = combine(name, [m["groups"][gid]["margins"][col]["dp_counts"] for m in cores])
            combined_margins[gid, col] = y
            if pool_group_totals:
                precision = 1/(len(y)*variances[name])
                group_precision[gid] += precision
                group_numerator[gid] += precision*y.sum()
    group_y = group_numerator/group_precision
    # Jeffreys half-count avoids declaring a publicly unobserved group
    # impossible; the smoothing is public and fixed before refinement.
    gp = _prob(np.asarray(result["public_group_counts"], float)+GROUP_PUBLIC_PSEUDOCOUNT)
    group_counts = _posterior(group_y, gp, n, 1/group_precision,
                              prior_multiplier, result["public_rows"])
    result["posterior_group_counts"] = group_counts.tolist()
    result["dp_group_counts"] = _apportion(group_counts, n)
    for gid, spec in enumerate(result["groups"]):
        for col in dc.NUMERIC:
            margin = spec["margins"][col]
            mass = _posterior(combined_margins[gid, col], margin["public_probabilities"],
                              group_counts[gid], variances[f"marginal_{col}"],
                              prior_multiplier, spec["public_fit_n"])
            p = mass/mass.sum() if mass.sum() > 0 else _prob(margin["public_probabilities"])
            margin["posterior_counts"] = mass.tolist()
            margin["posterior_probabilities"] = p.tolist()
            margin["dp_counts"] = _apportion(p*result["dp_group_counts"][gid],
                                              result["dp_group_counts"][gid])
    pref_y = combine("prefecture_counts", [m["dp_prefecture_counts"] for m in cores])
    pref = _posterior(pref_y, result["public_prefecture_probabilities"], n,
                      variances["prefecture_counts"], prior_multiplier, result["public_rows"])
    result["posterior_prefecture_counts"] = pref.tolist()
    result["dp_prefecture_counts"] = _apportion(pref, n)
    epsilon = sum((Fraction(str(m.get("accounting", {}).get("epsilon_exact",
                                      c["privacy"]["epsilon_exact"])))
                   for m, c in zip(models, cores)), Fraction())
    result["default_sample_options"].update({"group_prior": 0,
        "prefecture_prior": 0, "marginal_prior_noise_multiplier": 0})
    result["histogram_postprocessing"] = {
        "kind": "inverse_exact_geometric_variance_public_multinomial_bayes_v1",
        "prior_multiplier": float(prior_multiplier), "pool_group_totals": bool(pool_group_totals),
        "group_public_pseudocount": GROUP_PUBLIC_PSEUDOCOUNT,
        "public_spec_sha256": identities[0], "source_model_sha256": digests,
        "parent_epsilon_exact": str(epsilon), "delta": 0,
        "additional_epsilon": 0, "combined_noise_variance": variances,
        "source_weights": {k:v.tolist() for k,v in weights.items()},
        "private_evaluation_used_for_selection": False,
        "noise_likelihood": "Gaussian moment approximation using exact geometric variances",
        "sampling_dp_counts": "deterministic largest-remainder rounded posterior counts",
        "privacy_note": "Pure postprocessing; original query metadata retained for sampler compatibility; use parent_epsilon_exact for composed output.",
    }
    return result


def sample_refined(model: dict, n: int = dc.PRIVATE_N, seed: int = 20260918, **options):
    """Sample continuous posterior masses, preserving the public copula logic."""
    if n <= 0 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    unknown = set(options)-set(dc.DEFAULT_SAMPLE)
    if unknown:
        raise ValueError(f"Unknown sample options: {sorted(unknown)}")
    opt = dict(dc.DEFAULT_SAMPLE)
    opt.update(model.get("default_sample_options", {}))
    opt.update(options)
    if any(opt[k] != 0 for k in ("group_prior", "prefecture_prior", "marginal_prior_noise_multiplier")):
        raise ValueError("Posterior probabilities already include the public prior")
    strength, radial = float(opt["public_correlation_strength"]), float(opt["public_radial_strength"])
    if not 0 <= strength <= 1 or not 0 <= radial <= 1:
        raise ValueError("Correlation and radial strengths must lie in [0,1]")
    rng = np.random.default_rng(seed)
    bounds = dc._bounds(model["schema"])
    if opt["enforce_public_support"]:
        for col, limits in model["public_support"].items():
            bounds[col] = [max(bounds[col][0], limits[0]), min(bounds[col][1], limits[1])]
    gp = _prob(model["posterior_group_counts"])
    counts = dc._allocate(gp, n, rng) if opt["stratified"] else rng.multinomial(n, gp)
    chunks, target_rare = [], []
    for spec, count in zip(model["groups"], counts):
        if count == 0:
            continue
        corr = strength*np.asarray(spec["public_correlation"]) + (1-strength)*np.eye(len(dc.NUMERIC))
        z = rng.multivariate_normal(np.zeros(len(dc.NUMERIC)), corr, size=int(count), check_valid="raise")
        u = (rankdata(z, axis=0)-0.5)/count if opt["stratified"] else norm.cdf(z)
        data = {col: dc._inverse_margin(u[:,j],
                  np.asarray(spec["margins"][col]["posterior_probabilities"]), spec["margins"][col])
                for j, col in enumerate(dc.NUMERIC)}
        rare, sex, smoking, event = spec["group"]
        data.update({"sex": np.full(count, sex), "smoking": np.full(count, smoking),
                     "onset": np.full(count, int(event == 2)), "death": np.full(count, int(event == 1))})
        chunk = dc._public_radial_adjust(pd.DataFrame(data), spec, radial,
                                         model["schema"]["rare_detector"], bounds)
        chunks.append(chunk)
        target_rare.extend([rare]*count)
    frame = pd.concat(chunks, ignore_index=True)
    if opt["enforce_rare_partition"]:
        frame = dc._project_rare(frame, np.asarray(target_rare), model["schema"]["rare_detector"], bounds)
    for col, limits in bounds.items():
        frame[col] = np.clip(frame[col], *limits)
    pp = _prob(model["posterior_prefecture_counts"])
    pc = dc._allocate(pp, n, rng) if opt["stratified"] else rng.multinomial(n, pp)
    prefs = np.repeat(np.asarray(model["categories"], dtype=object), pc)
    rng.shuffle(prefs)
    frame["prefecture"] = prefs
    for col in ["age", "sex", "smoking", "onset", "death"]:
        frame[col] = np.rint(frame[col]).astype(int)
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)[model["schema"]["submitted_c_columns"]]
