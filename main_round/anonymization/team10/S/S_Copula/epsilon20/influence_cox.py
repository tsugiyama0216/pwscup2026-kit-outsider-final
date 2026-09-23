"""PUBLIC-designed bounded efficient Cox influence measurements.

Adaptive queries use public nuisance curves and previous DP beta only.
Every row map depends only on one record and explicit public nuisance curves;
the integer L1 ball gives replace-one sensitivity 2Q for their sum.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from fractions import Fraction

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent/"postprocessing_v2")]
import dp_cox
import dp_copula as dc
from cox_optimizer import EfronRiskSets


def build_public_model(public, *, beta=None, base_model=None, public_n=1049):
    """Fit nuisance curves and information solely on public background."""
    base = dp_cox.build_public_cox_model(public) if base_model is None else base_model
    beta = np.asarray(base["beta_standardized"] if beta is None else beta, float)
    x = (dp_cox._features(public)-base["center"])/base["scale"]
    order = np.argsort(public.time.to_numpy(float), kind="stable")
    x = x[order]
    time, event = public.time.to_numpy(float)[order], public.onset.to_numpy(float)[order]
    risk = EfronRiskSets(time, event)
    _, information, _ = risk.evaluate(x, beta, information=True)
    eta_shift = float(np.max(x@beta))
    weight = np.exp(x@beta-eta_shift)
    weighted = weight[:, None]*x
    r0 = np.cumsum(weight[::-1])[::-1][risk.first]
    r1 = np.cumsum(weighted[::-1], axis=0)[::-1][risk.first]
    d0 = np.bincount(risk.inverse, weights=weight*event)
    d1 = np.zeros_like(r1)
    np.add.at(d1, risk.inverse, weighted*event[:, None])
    g, f = risk.tie_group, risk.fraction
    den = r0[g]-f*d0[g]
    mu = (r1[g]-f[:, None]*d1[g])/den[:, None]
    a0 = np.bincount(g, weights=1/den, minlength=len(risk.first))
    a1 = np.bincount(g, weights=f/den, minlength=len(risk.first))
    v0, v1, event_mean = np.zeros_like(r1), np.zeros_like(r1), np.zeros_like(r1)
    np.add.at(v0, g, mu/den[:, None])
    np.add.at(v1, g, mu*(f/den)[:, None])
    np.add.at(event_mean, g, mu)
    counts = np.bincount(risk.inverse, weights=event)
    event_mean = np.where((counts>0)[:, None], event_mean/np.maximum(counts, 1)[:, None], r1/r0[:, None])
    return {"kind":"public_efficient_Efron_influence_row_map", "center":base["center"], "scale":base["scale"],
        "beta_standardized":beta.tolist(), "public_rows":len(public), "public_n":public_n,
        "times":time[risk.first].tolist(), "risk_mean":(r1/r0[:, None]).tolist(), "event_mean":event_mean.tolist(),
        "cumulative_a":np.cumsum(a0).tolist(), "cumulative_v":np.cumsum(v0, axis=0).tolist(),
        "event_a_adjust":a1.tolist(), "event_v_adjust":v1.tolist(), "eta_shift":eta_shift,
        "information_per_row":(information/len(public)).tolist(),
        "expected_se_standardized":np.sqrt(np.diag(np.linalg.inv(information/len(public)*public_n))).tolist()}


def score_rows(frame, model):
    """Public-risk-set efficient score contribution of each separate record."""
    x = (dp_cox._features(frame)-model["center"])/model["scale"]
    time = frame.time.to_numpy(float)
    event = frame.onset.to_numpy(float)
    grid = np.asarray(model["times"])
    before = np.searchsorted(grid, time, side="right")-1
    after = np.minimum(np.searchsorted(grid, time, side="left"), len(grid)-1)
    has_before = before >= 0
    safe = np.maximum(before, 0)
    a = np.where(has_before, np.asarray(model["cumulative_a"])[safe], 0.0)
    v = np.where(has_before[:, None], np.asarray(model["cumulative_v"])[safe], 0.0)
    matched = (grid[after] == time) & (event == 1)
    a -= matched*np.asarray(model["event_a_adjust"])[after]
    v -= matched[:, None]*np.asarray(model["event_v_adjust"])[after]
    event_mean = np.asarray(model["risk_mean"])[after].copy()
    event_mean[matched] = np.asarray(model["event_mean"])[after[matched]]
    weight = np.exp(np.clip(x@np.asarray(model["beta_standardized"])-model["eta_shift"], -100, 100))
    return event[:, None]*(x-event_mean)-weight[:, None]*(a[:, None]*x-v)


def influence_rows(frame, model):
    return score_rows(frame, model) @ np.linalg.inv(np.asarray(model["information_per_row"]))


def bounded_integer_rows(frame, model, *, l1_bound, quantization=4096):
    """Clip influence in public cohort-SE units and exactly bound integer L1."""
    if not np.isfinite(l1_bound) or l1_bound <= 0 or not isinstance(quantization,int) or not 16 <= quantization <= 65536:
        raise ValueError("Positive public L1 bound and integer quantization >=16 required")
    z = influence_rows(frame, model)/np.asarray(model["expected_se_standardized"])
    # A deterministic per-record numerical fallback preserves the row bound.
    z = np.nan_to_num(z, nan=0.0, posinf=1e100, neginf=-1e100)
    z = np.clip(z, -1e100, 1e100)
    z *= np.minimum(1.0, l1_bound/np.maximum(np.abs(z).sum(axis=1), 1e-300))[:, None]
    integer = np.rint(z*quantization/l1_bound).astype(np.int64)
    for row in integer:
        excess = int(np.abs(row).sum())-quantization
        if excess > 0:
            for j in np.argsort(-np.abs(row), kind="stable"):
                step = min(excess, abs(int(row[j])))
                row[j] -= int(np.sign(row[j]))*step
                excess -= step
                if excess == 0:
                    break
    return integer


def decode_mean(noisy_integer_sum, model, public_integer_mean, *, l1_bound, quantization=4096):
    """Postprocess an already-noisy sum; this function takes no raw cohort."""
    released = np.asarray([float(max(-10**12, min(10**12, int(v)))) for v in noisy_integer_sum])
    if released.shape != (9,):
        raise ValueError("Nine DP coordinates required")
    step = (released/model["public_n"]-np.asarray(public_integer_mean))*l1_bound/quantization
    beta = np.asarray(model["beta_standardized"])+step*np.asarray(model["expected_se_standardized"])
    return {"beta_standardized":beta.tolist(), "standard_errors_standardized":model["expected_se_standardized"],
            "additional_private_queries":0, "postprocessing_only":True}


def measure(private, public, plan):
    """Pure-DP composition of bounded row-vector measurements, no retries.

    Private rows must already have fixed public row-local preprocessing.
    Public calibration bias uses unbounded PUBLIC scores minus clipped PUBLIC
    scores, so it does not subtract the public estimating-equation signal at
    a non-optimal adaptive reference. No private Hessian is computed.
    """
    if len(private) != 1049:
        raise ValueError('Requires fixed public size 1049')
    budgets=[Fraction(str(e)) for e in plan['cox_round_epsilons']]
    if not budgets or any(e <= 0 for e in budgets):
        raise ValueError('Positive per-round allocations required')
    base=dp_cox.build_public_cox_model(public)
    beta=np.asarray(base['beta_standardized'])
    initial=build_public_model(public,base_model=base)
    origin=beta.copy(); initial_se=np.asarray(initial['expected_se_standardized'])
    releases=[]; q=int(plan['quantization'])
    for epsilon in budgets:
        nuisance=build_public_model(public,beta=beta,base_model=base)
        se=np.asarray(nuisance['expected_se_standardized'])
        public_influence=influence_rows(public,nuisance)
        bound=max(1.0,float(np.quantile(np.abs(public_influence/se).sum(axis=1),plan['clip_quantile'])))
        public_mean=bounded_integer_rows(public,nuisance,l1_bound=bound,quantization=q).mean(axis=0)
        raw=bounded_integer_rows(private,nuisance,l1_bound=bound,quantization=q)
        # Each row integer L1 <= Q, hence the sum has replacement sensitivity 2Q.
        integer_sum=[sum(int(v) for v in raw[:,j]) for j in range(9)]
        noisy,mechanism=dc.release_integer_vector(integer_sum,epsilon,2*q)
        decoded=decode_mean(noisy,nuisance,public_mean,l1_bound=bound,quantization=q)
        corrected_step=np.asarray(decoded['beta_standardized'])-beta+public_influence.mean(axis=0)
        beta=np.clip(beta+plan['damping']*corrected_step,origin-6*initial_se,origin+6*initial_se)
        releases.append({'mechanism':mechanism,'l1_bound_public':bound,
                         'noisy_integer_sum':noisy,'reference_beta_DP_or_public':nuisance['beta_standardized'],
                         'beta_after_DP_postprocessing':beta.tolist()})
    final_public=build_public_model(public,beta=beta,base_model=base)
    return {'beta_standardized':beta.tolist(),
            'standard_errors_standardized':final_public['expected_se_standardized'],
            'public_center':base['center'],'public_scale':base['scale'],
            'epsilon_exact':str(sum(budgets,Fraction())), 'delta':0,
            'rounds':releases,'method':'adaptive_bounded_public_efficient_Cox_score',
            'standard_error_source':'public_information_at_DP_beta'}
