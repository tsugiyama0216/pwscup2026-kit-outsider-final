"""Public-reference, bounded-score Cox correction.

The only private release is a bounded integer-vector sum passed to a caller-
supplied DP mechanism. Public parameters and a fixed output row count must be
chosen before inspecting the private table. This is a one-step approximation,
not an exact private Cox fit, and no utility guarantee is made.
"""
from __future__ import annotations

from typing import Callable
from fractions import Fraction

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError


COX_COLUMNS = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]


def _features(df: pd.DataFrame) -> np.ndarray:
    frame = df.copy()
    frame["log_TG"] = np.log(np.maximum(frame["TG"].to_numpy(float), 1e-6))
    frame["log_ALT"] = np.log(np.maximum(frame["ALT"].to_numpy(float), 1e-6))
    values = frame[COX_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Cox features must be finite")
    return values


def _score_rows(df: pd.DataFrame, model: dict) -> np.ndarray:
    x = (_features(df) - np.asarray(model["center"])) / np.asarray(model["scale"])
    z = np.column_stack([np.ones(len(x)), x])
    time = np.clip(df["time"].to_numpy(float), 0.0, 10.0)
    grid = np.asarray(model["hazard_times"])
    cumulative = np.asarray(model["cumulative_hazard"])
    index = np.searchsorted(grid, time, side="right") - 1
    hazard = np.where(index >= 0, cumulative[np.maximum(index, 0)], 0.0)
    expected = hazard * np.exp(np.clip(x @ np.asarray(model["beta_standardized"]), -30.0, 30.0))
    return z * (df["onset"].to_numpy(float) - expected)[:, None]


def build_public_cox_model(public: pd.DataFrame, *, quantization: int = 128,
                           ridge: float = 0.01, clip_quantile: float = 0.99) -> dict:
    """Fit all nuisance parameters using the explicitly public background only."""
    if quantization < 16 or not 0.0 < clip_quantile <= 1.0 or ridge <= 0.0:
        raise ValueError("Invalid public Cox settings")
    features = _features(public)
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    x = (features - center) / scale
    frame = pd.DataFrame(x, columns=COX_COLUMNS)
    frame["time"] = public["time"].to_numpy(float)
    frame["onset"] = public["onset"].to_numpy(int)
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(frame, duration_col="time", event_col="onset")
    beta = cph.params_.to_numpy(float)
    baseline = cph.baseline_cumulative_hazard_
    # lifelines centers the partial hazard at its training mean.
    baseline_values = baseline.iloc[:, 0].to_numpy(float) * np.exp(-float(cph._norm_mean.to_numpy(float) @ beta))
    model = {
        "kind": "public_reference_bounded_cox_score_v1",
        "columns": COX_COLUMNS,
        "center": center.tolist(), "scale": scale.tolist(),
        "beta_standardized": beta.tolist(),
        "hazard_times": baseline.index.to_numpy(float).tolist(),
        "cumulative_hazard": baseline_values.tolist(),
        "quantization": int(quantization), "ridge": float(ridge),
        "clip_quantile": float(clip_quantile),
    }
    scores = _score_rows(public, model)
    # Fit the fixed-hazard augmented information with a public ridge. Intercept
    # absorbs onset-rate deviations so slopes need not absorb the whole shift.
    z = np.column_stack([np.ones(len(x)), x])
    expected = public["onset"].to_numpy(float) - scores[:, 0]
    info = (z.T * np.maximum(expected, 0.0)) @ z / len(z)
    info += ridge * np.eye(info.shape[0])
    model["information_inverse"] = np.linalg.inv(info).tolist()
    model["score_center"] = scores.mean(axis=0).tolist()
    centered = scores - np.asarray(model["score_center"])
    model["clip_norm"] = max(1.0, float(np.quantile(np.abs(centered).sum(axis=1), clip_quantile)))
    # Public expected clipping bias is removed after privatization.
    integer_rows = _integer_rows(public, model)
    model["public_integer_mean"] = integer_rows.mean(axis=0).tolist()
    rescaled = integer_rows * (model["clip_norm"] / quantization)
    model["public_score_covariance"] = np.cov(rescaled, rowvar=False).tolist()
    return model


def _integer_rows(df: pd.DataFrame, model: dict) -> np.ndarray:
    """Deterministic row map with exact integer L1 norm <= quantization."""
    scores = _score_rows(df, model) - np.asarray(model["score_center"])
    bound = float(model["clip_norm"])
    quantization = int(model["quantization"])
    scores *= np.minimum(1.0, bound / np.maximum(np.abs(scores).sum(axis=1), 1e-300))[:, None]
    rows = np.rint(scores * (quantization / bound)).astype(np.int64)
    # Rounding can exceed the ball by at most dimension/2. Remove excess from
    # the largest coordinate, with deterministic tie-breaking.
    for row in rows:
        excess = int(np.abs(row).sum()) - quantization
        while excess > 0:
            j = int(np.argmax(np.abs(row)))
            step = min(excess, abs(int(row[j])))
            row[j] -= int(np.sign(row[j])) * step
            excess -= step
    return rows


def measure_cox_update(private: pd.DataFrame, model: dict, *, epsilon: float,
                       public_n: int,
                       noise_integer_vector: Callable[[list[int], int, float], list[int]]) -> dict:
    """Release an approximate Cox target through one DP measurement.

    ``noise_integer_vector(values, sensitivity, epsilon)`` must implement a
    vetted or exact epsilon-DP integer-vector L1 mechanism. It is called once.
    The neighboring relation is replacement of one row, at public fixed size.
    No private Hessian, private p-value, raw coefficient, or seed is released.
    """
    if epsilon <= 0.0 or public_n <= 0 or len(private) != public_n:
        raise ValueError("Positive budget and the fixed public row count are required")
    rows = _integer_rows(private, model)
    counts = rows.sum(axis=0).tolist()
    sensitivity = 2 * int(model["quantization"])
    noisy = noise_integer_vector(counts, sensitivity, epsilon)
    return postprocess_cox_update(noisy, model, epsilon=epsilon, public_n=public_n)


def postprocess_cox_update(noisy: list[int], model: dict, *, epsilon: float,
                           public_n: int, damping: float = 1.0,
                           max_standardized_step: float = 1.0,
                           prior_variance_multiplier: float = 1.0) -> dict:
    """Postprocess the same noisy vector without further private access."""
    if len(noisy) != len(COX_COLUMNS) + 1:
        raise ValueError("Wrong noisy score dimension")
    # The DP mechanism has unbounded integer support. Saturation is public
    # postprocessing; retain original integers in the released model.
    finite_noisy = np.asarray([float(max(-10**12, min(10**12, int(v)))) for v in noisy])
    mean = (finite_noisy / public_n - np.asarray(model["public_integer_mean"]))
    mean *= float(model["clip_norm"]) / int(model["quantization"])
    if prior_variance_multiplier < 0:
        raise ValueError("Prior variance multiplier must be nonnegative")
    # Public covariance describes sampling variation of a cohort score mean.
    # Account analytically for exact rational geometric noise. The shrinkage
    # matrix is wholly public; it is not fitted on private validation scores.
    sensitivity = 2 * int(model["quantization"])
    noise_variance = 2.0 * sensitivity * (sensitivity + float(epsilon)) / float(epsilon)**2
    noise_variance *= (float(model["clip_norm"]) / int(model["quantization"]) / public_n)**2
    prior_covariance = np.asarray(model["public_score_covariance"]) * prior_variance_multiplier / public_n
    shrinkage = prior_covariance @ np.linalg.inv(prior_covariance + noise_variance*np.eye(len(mean)))
    mean = shrinkage @ mean
    delta = np.asarray(model["information_inverse"]) @ mean
    delta = np.clip(float(damping) * delta, -max_standardized_step, max_standardized_step)
    beta = np.asarray(model["beta_standardized"]) + delta[1:]
    return {
        "kind": model["kind"], "epsilon": float(epsilon), "delta": 0.0,
        "epsilon_exact": str(epsilon if isinstance(epsilon, Fraction) else Fraction(str(epsilon))),
        "adjacency": "fixed-size replace-one",
        "integer_l1_sensitivity": 2 * int(model["quantization"]),
        "public_n": int(public_n), "noisy_score_integer": [int(v) for v in noisy],
        "beta_standardized": beta.tolist(),
        "beta_original_units": (beta / np.asarray(model["scale"])).tolist(),
        "log_baseline_multiplier": float(delta[0]),
        "postprocess_damping": float(damping),
        "postprocess_max_standardized_step": float(max_standardized_step),
        "postprocess_prior_variance_multiplier": float(prior_variance_multiplier),
        "postprocess_shrinkage_trace": float(np.trace(shrinkage)),
    }


def calibrate_synthetic_cox(synthetic: pd.DataFrame, target: dict, public_model: dict,
                            *, iterations: int = 16, learning_rate: float = 0.7,
                            strata: np.ndarray | None = None,
                            max_corr_mean_shift: float = 0.02,
                            max_corr_tail_shift: float = 0.04) -> tuple[pd.DataFrame, dict]:
    """Match a DP coefficient target using synthetic-only rank rearrangements.

    Numeric univariate distributions, all event/time values, and all categorical
    values stay fixed. Six mutable continuous predictors can be rearranged;
    integer age, sex and smoking stay fixed. Optional strata must derive from
    public data or the already-DP synthetic table. This is DP postprocessing.
    No private data or unprotected evaluation result belongs in this function.
    """
    if iterations < 0 or not 0.0 < learning_rate <= 1.0:
        raise ValueError("Invalid synthetic calibration settings")
    original = synthetic.copy().reset_index(drop=True)
    result = original.copy()
    groups = np.zeros(len(result), dtype=int) if strata is None else np.asarray(strata)
    if len(groups) != len(result):
        raise ValueError("Strata length mismatch")
    mutable = [2, 3, 4, 5, 6, 7]  # BMI, SBP, log TG, HDL, log ALT, FPG
    wanted = np.asarray(target["beta_standardized"], float)
    center = np.asarray(public_model["center"])
    scale = np.asarray(public_model["scale"])
    original_x = (_features(original) - center) / scale
    corr_features = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]
    original_corr = original[corr_features].corr(method="spearman").to_numpy()
    upper = np.triu_indices(len(corr_features), 1)
    basis = wanted[mutable].copy()
    basis /= max(float(basis @ basis), 0.1)
    rows = {key: np.flatnonzero(groups == key) for key in np.unique(groups)}
    best, best_error = result.copy(), float("inf")
    errors = []
    # Every candidate selection below uses only the synthetic C and DP target.
    for step in range(iterations + 1):
        x = (_features(result) - center) / scale
        frame = pd.DataFrame(x, columns=COX_COLUMNS)
        frame["time"] = result["time"].to_numpy(float)
        frame["onset"] = result["onset"].to_numpy(int)
        try:
            fitted = CoxPHFitter(penalizer=0.0).fit(frame, duration_col="time", event_col="onset")
            delta = fitted.params_.to_numpy(float) - wanted
        except (ValueError, np.linalg.LinAlgError, ConvergenceError):
            break
        error = float(np.linalg.norm(delta))
        if not np.isfinite(error):
            break
        errors.append(error)
        current_corr = result[corr_features].corr(method="spearman").to_numpy()
        corr_shift = np.nan_to_num(np.abs(current_corr-original_corr)[upper], nan=1.0)
        allowed = (float(corr_shift.mean()) <= max_corr_mean_shift and
                   float(np.quantile(corr_shift, 0.9)) <= max_corr_tail_shift)
        if error < best_error and allowed:
            best, best_error = result.copy(), error
        if step == iterations:
            break
        risk_shift = np.clip(x @ delta, -2.0, 2.0)
        changed = x[:, mutable] + learning_rate * risk_shift[:, None] * basis[None, :]
        for j, feature_id in enumerate(mutable):
            for ix in rows.values():
                order = ix[np.argsort(changed[ix, j], kind="stable")]
                x[order, feature_id] = np.sort(original_x[ix, feature_id])
                feature = COX_COLUMNS[feature_id]
                column = feature[4:] if feature.startswith("log_") else feature
                result.loc[order, column] = np.sort(original.loc[ix, column].to_numpy())
    return best, {"method": "synthetic_rank_cox_calibration", "iterations_requested": iterations,
                  "iterations_completed": len(errors), "coefficient_l2_errors": errors,
                  "selected_coefficient_l2_error": best_error if np.isfinite(best_error) else None,
                  "max_corr_mean_shift": max_corr_mean_shift,
                  "max_corr_tail_shift": max_corr_tail_shift,
                  "selection_input": "synthetic_C_and_DP_target_only"}
