"""Bounded whole-table Cox coefficient/SE target with pure integer DP noise.

The whole estimator, rather than each patient contribution, is placed inside
a fixed PUBLIC integer L1 ball. Its global sensitivity is therefore 2Q even
when the Cox estimator is unstable. No private convergence or clipping flag
is released. Callers must freeze all parameters publicly and sanitize rows by
public bounds before invoking measure_target on a protected cohort.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dp_copula as dc
import dp_cox as cx

VERSION = "whole_table_bounded_integer_cox_beta_se_v1"
DIMENSION = 18
DEFAULT_RADIUS = 16.0
DEFAULT_QUANTIZATION = 65_536
DEFAULT_SE_WEIGHT = 2.0
SE_RATIO_BOUNDS = (0.1, 10.0)


def _fit_summary(frame: pd.DataFrame, public_target: dict) -> tuple[np.ndarray, np.ndarray]:
    """Internal fit; exceptions are handled at the protected estimator boundary."""
    x = (cx._features(frame)-np.asarray(public_target["center"]))/np.asarray(public_target["scale"])
    table = pd.DataFrame(x, columns=cx.COX_COLUMNS)
    table["time"] = frame["time"].to_numpy(float)
    table["onset"] = frame["onset"].to_numpy(int)
    if not np.isfinite(table.to_numpy()).all() or (table["time"] <= 0).any():
        raise ValueError("Finite rows with positive follow-up required")
    if table["onset"].sum() < 2:
        raise ValueError("Insufficient events for a numerical fit")
    with warnings.catch_warnings():
        # Numerical/convergence warnings also take the same public fallback.
        # In particular, no private warning text reaches the caller or logs.
        warnings.simplefilter("error")
        fit = CoxPHFitter(penalizer=0.0)
        fit.fit(table, duration_col="time", event_col="onset", show_progress=False)
    beta = fit.params_.reindex(cx.COX_COLUMNS).to_numpy(float)
    se = fit.standard_errors_.reindex(cx.COX_COLUMNS).to_numpy(float)
    if not np.isfinite(beta).all() or not np.isfinite(se).all() or np.any(se <= 0):
        raise ValueError("Numerical Cox estimates must be finite with positive SE")
    return beta, se


def build_public_target(public: pd.DataFrame, public_n: int = dc.PRIVATE_N) -> dict:
    """Fit target center and expected cohort SE using only public rows.

    SE(public fit)*sqrt(len(public)/public_n) is a public approximation to the
    uncertainty of a cohort of the fixed target size. Its accuracy affects
    utility and clipping, but not the global-sensitivity proof.
    """
    if isinstance(public_n, bool) or not isinstance(public_n, int) or public_n <= 0:
        raise ValueError("public_n must be a fixed positive integer")
    if len(public) < 20:
        raise ValueError("Require at least twenty public rows")
    values = cx._features(public)
    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    model = {"kind": VERSION+"_public_reference", "columns": list(cx.COX_COLUMNS),
             "public_n": public_n, "public_rows": len(public),
             "center": center.tolist(), "scale": scale.tolist()}
    beta, se = _fit_summary(public, model)
    expected_se = se*np.sqrt(len(public)/public_n)
    model.update({"beta_standardized": beta.tolist(),
                  "public_fit_se_standardized": se.tolist(),
                  "expected_se_standardized": expected_se.tolist(),
                  "beta_original_units": (beta/scale).tolist(),
                  "expected_se_original_units": (expected_se/scale).tolist(),
                  "expected_se_rule": "public_fit_se*sqrt(public_rows/public_n)"})
    return model


def _validate(public_target, radius, quantization, se_weight):
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(se_weight) or se_weight <= 0:
        raise ValueError("radius and se_weight must be positive public constants")
    if isinstance(quantization, bool) or not isinstance(quantization, int) or not 16 <= quantization <= 2**40:
        raise ValueError("quantization must be a public integer from 16 through 2**40")
    if public_target.get("columns") != cx.COX_COLUMNS:
        raise ValueError("Unexpected public feature order")
    for key in ("center", "scale", "beta_standardized", "expected_se_standardized"):
        values = np.asarray(public_target[key], float)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("Invalid public target")
        if key in ("scale", "expected_se_standardized") and np.any(values <= 0):
            raise ValueError("Public scales and expected SE must be positive")
    n = public_target["public_n"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("Invalid fixed public size")


def _integer_l1_projection(coordinates, radius, quantization):
    """Quantize then exactly repair the L1 bound with Python integers."""
    values = np.asarray(coordinates, float)
    if values.shape != (DIMENSION,) or not np.isfinite(values).all():
        return [0]*DIMENSION
    # Scale by the largest magnitude first to avoid overflow in the L1 sum.
    largest = float(np.max(np.abs(values)))
    if largest:
        normalized = values/largest
        normalized_l1 = float(np.abs(normalized).sum())
        if largest > radius/normalized_l1:
            values = normalized*(radius/normalized_l1)
    integer = [int(round(float(value/radius*quantization))) for value in values]
    excess = sum(abs(value) for value in integer)-quantization
    if excess > 0:
        for j in sorted(range(DIMENSION), key=lambda j: (-abs(integer[j]), j)):
            step = min(excess, abs(integer[j]))
            integer[j] -= (1 if integer[j] >= 0 else -1)*step
            excess -= step
            if excess == 0:
                break
    assert sum(abs(value) for value in integer) <= quantization
    return integer


def _bounded_integer_target(private, public_target, radius, quantization, se_weight):
    """Sensitive intermediate: never publish it or its norm/failure status."""
    try:
        # All data-dependent fitting and feature preparation is inside the
        # same fallback boundary. Neither warning text nor exception escapes.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(all="raise"):
                beta, se = _fit_summary(private, public_target)
                expected = np.asarray(public_target["expected_se_standardized"])
                beta_delta = (beta-np.asarray(public_target["beta_standardized"]))/expected
                se_delta = se_weight*(np.log(se)-np.log(expected))
                coordinates = np.r_[beta_delta, se_delta]
                return _integer_l1_projection(coordinates, radius, quantization)
    except Exception:
        # A public constant in the SAME range, followed by the SAME noise.
        # Do not return the exception, a failure flag, or a clipping statistic.
        return [0]*DIMENSION


def postprocess_target(noisy_integer, public_target, *, epsilon, radius=DEFAULT_RADIUS,
                       quantization=DEFAULT_QUANTIZATION, se_weight=DEFAULT_SE_WEIGHT,
                       mechanism=None):
    """Reconstruct finite beta/SE targets using only the already-DP vector."""
    _validate(public_target, radius, quantization, se_weight)
    eps = Fraction(str(epsilon))
    if eps <= 0 or len(noisy_integer) != DIMENSION:
        raise ValueError("Positive epsilon and eighteen DP coordinates required")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in noisy_integer):
        raise ValueError("DP coordinates must be integers")
    released = [int(v) for v in noisy_integer]
    finite = np.asarray([float(max(-10**12, min(10**12, v))) for v in released])
    decoded = finite*(float(radius)/quantization)
    expected = np.asarray(public_target["expected_se_standardized"])
    beta = np.asarray(public_target["beta_standardized"])+expected*np.clip(decoded[:9], -radius, radius)
    log_ratio = np.clip(decoded[9:]/se_weight, *np.log(SE_RATIO_BOUNDS))
    se = expected*np.exp(log_ratio)
    scale = np.asarray(public_target["scale"])
    identity = hashlib.sha256(json.dumps(public_target, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {"kind": VERSION, "public_n": public_target["public_n"],
            "epsilon_exact": str(eps), "epsilon": float(eps), "delta": 0,
            "adjacency": "replace_one_fixed_public_size", "integer_l1_sensitivity": 2*quantization,
            "radius": float(radius), "quantization": quantization, "se_weight": float(se_weight),
            "public_target_sha256": identity, "noisy_target_integer": released,
            "beta_standardized": beta.tolist(), "standard_errors_standardized": se.tolist(),
            "beta_original_units": (beta/scale).tolist(),
            "standard_errors_original_units": (se/scale).tolist(),
            "se_ratio_bounds": list(SE_RATIO_BOUNDS),
            "mechanism": mechanism,
            "privacy_proof": "Whole estimator has integer L1 norm at most Q; any replacement changes it by at most 2Q.",
            "private_diagnostics_released": False}


def measure_target(private: pd.DataFrame, public_target: dict, epsilon, *,
                   radius=DEFAULT_RADIUS, quantization=DEFAULT_QUANTIZATION,
                   se_weight=DEFAULT_SE_WEIGHT, noise_callable=None):
    """Spend epsilon ONCE to release the bounded whole-table beta/SE target.

    The runner must reserve/compose this budget before calling, sanitize every
    row with the frozen public schema, and never publish fit success, clipping,
    raw estimates, or timing diagnostics. This module never logs those values.
    ``noise_callable`` is a public-test injection point with the same signature
    and return pair as ``dp_copula.release_integer_vector``. Production must use
    the default exact noise mechanism. A retry is another budget expenditure.
    """
    _validate(public_target, radius, quantization, se_weight)
    eps = Fraction(str(epsilon))
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if len(private) != public_target["public_n"]:
        raise ValueError("Dataset must have the fixed public cohort size")
    integer = _bounded_integer_target(private, public_target, radius, quantization, se_weight)
    noise = dc.release_integer_vector if noise_callable is None else noise_callable
    released, mechanism = noise(integer, eps, 2*quantization)
    return postprocess_target(released, public_target, epsilon=eps, radius=radius,
                              quantization=quantization, se_weight=se_weight, mechanism=mechanism)
