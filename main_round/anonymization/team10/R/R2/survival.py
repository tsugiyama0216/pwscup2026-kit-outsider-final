"""B-only conditional cause-specific Cox generation for R2.

Aggregate event counts in fixed time bins define a new event schedule. At each
event the corresponding Cox risk weights select one still-at-risk synthetic
row. No source row is matched to a synthetic row and no source event time is
copied. This preserves observed event counts, not a formal privacy guarantee.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning


FEATURES = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]
RAW_FEATURES = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]
CAUSES = ("onset", "death")
LOG_FLOOR = 1e-6
TIME_FLOOR = 1.0 / 365.25
# Distributed B may encode a rounded number of days (e.g. a half-day below
# ten years). One day is deliberately tighter than the public utility check's
# 0.01-year tolerance; output times still obey the strict horizon upper bound.
ADMIN_CENSOR_ATOL = 1.0 / 365.25
TIME_BINS = 40
RIDGE_SEQUENCE = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
MAX_CALIBRATION_STEPS = 8
CALIBRATION_STEP = 0.5
CALIBRATION_UPDATE_SE_LIMIT = 1.0
CALIBRATION_TOTAL_SE_LIMIT = 3.0
CALIBRATION_INFORMATION_WEIGHT = 0.25


def cox_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The same nine predictors and log floor as the public Cox analysis."""
    out = frame[RAW_FEATURES].astype(float).reset_index(drop=True).copy()
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Cox predictors must be finite")
    if (out[["TG", "ALT"]] < 0).any().any():
        raise ValueError("TG and ALT must be nonnegative")
    out["log_TG"] = np.log(np.maximum(out.pop("TG"), LOG_FLOOR))
    out["log_ALT"] = np.log(np.maximum(out.pop("ALT"), LOG_FLOOR))
    return out[FEATURES]


def _fit_cause(x: pd.DataFrame, b: pd.DataFrame, cause: str) -> tuple[pd.Series, dict]:
    active = x.columns[x.std(ddof=0) > 1e-12].tolist()
    omitted = [name for name in FEATURES if name not in active]
    zeros = pd.Series(0.0, index=FEATURES)
    count = int(b[cause].sum())
    audit = {"cause": cause, "event_count": count, "features": active,
             "constant_predictors_excluded": omitted, "attempts": []}
    if count == 0 or not active:
        audit.update({"status": "no_events" if count == 0 else "no_variable_predictors",
                      "penalizer": None, "coefficients": zeros.to_dict(),
                      "standard_errors": {}, "ridge_fallback": False})
        return zeros, audit
    train = x[active].assign(time=b["time"].to_numpy(), event=b[cause].to_numpy())
    for penalizer in RIDGE_SEQUENCE:
        try:
            model = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train, duration_col="time", event_col="event", show_progress=False)
            beta = model.params_.to_numpy(dtype=float)
            se = model.standard_errors_.to_numpy(dtype=float)
            if not (np.isfinite(beta).all() and np.isfinite(se).all() and (se > 0).all()):
                raise ConvergenceError("Non-finite coefficient/standard error or non-positive standard error")
        except (ConvergenceError, ConvergenceWarning, np.linalg.LinAlgError,
                ValueError, ZeroDivisionError, FloatingPointError) as exc:
            audit["attempts"].append({"penalizer": penalizer, "status": "failed",
                                      "reason": type(exc).__name__})
            continue
        audit["attempts"].append({"penalizer": penalizer, "status": "converged"})
        zeros.loc[active] = beta
        audit.update({"status": "converged", "penalizer": penalizer,
                      "coefficients": zeros.to_dict(),
                      "standard_errors": model.standard_errors_.to_dict(),
                      "ridge_fallback": penalizer != 0.0})
        return zeros, audit
    raise RuntimeError(f"{cause} Cox failed at every predeclared penalizer: {audit['attempts']}")


def _event_schedule(b: pd.DataFrame, horizon: float, rng: np.random.Generator
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    # The partition is fixed by horizon, not tuned using event times or scores.
    edges = np.linspace(TIME_FLOOR, horizon, TIME_BINS + 1)
    bins = np.clip(np.searchsorted(edges, b["time"].to_numpy(), side="right") - 1,
                   0, TIME_BINS - 1)
    counts = {cause: np.bincount(bins[b[cause].to_numpy() == 1], minlength=TIME_BINS)
              for cause in CAUSES}
    times, causes = [], []
    for k in range(TIME_BINS):
        onset_n, death_n = int(counts["onset"][k]), int(counts["death"][k])
        count = onset_n + death_n
        if not count:
            continue
        # A new pooled midpoint grid makes all event times distinct; randomly
        # interleave the two causes within each bin while retaining both counts.
        labels = np.repeat(np.arange(2), [onset_n, death_n])
        labels = rng.permutation(labels)
        grid = edges[k] + (np.arange(count) + 0.5) * (edges[k + 1] - edges[k]) / count
        times.extend(grid.tolist())
        causes.extend(labels.tolist())
    return np.asarray(times), np.asarray(causes, dtype=int), {
        "method": "fixed_bin_counts_pooled_midpoints", "n_bins": TIME_BINS,
        "bin_edges": edges.tolist(),
        "cause_bin_counts": {cause: value.tolist() for cause, value in counts.items()},
        "source_individual_times_copied": False}


def _assign_events(x: pd.DataFrame, center: pd.Series, coefficients: dict,
                   times: np.ndarray, causes: np.ndarray, horizon: float,
                   uniforms: np.ndarray) -> pd.DataFrame:
    n = len(x)
    out_time = np.full(n, horizon, dtype=float)
    out_onset, out_death = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    available = np.ones(n, dtype=bool)
    logits = {cause: (x - center).to_numpy() @ coefficients[cause].to_numpy()
              for cause in CAUSES}
    if any(not np.isfinite(value).all() for value in logits.values()):
        raise ValueError("Non-finite Cox linear predictor")
    for time, label, uniform in zip(times, causes, uniforms):
        remaining = np.flatnonzero(available)
        eta = logits[CAUSES[int(label)]][remaining]
        weights = np.exp(eta - np.max(eta))
        cumulative = np.cumsum(weights)
        position = min(int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right")),
                       len(remaining) - 1)
        selected = remaining[position]
        out_time[selected] = time
        (out_onset if label == 0 else out_death)[selected] = 1
        available[selected] = False
    return pd.DataFrame({"time": out_time, "onset": out_onset, "death": out_death})


def _calibration_loss(target_fits: dict, generated_fits: dict) -> tuple[float, dict]:
    """Symmetric all-coefficient loss, without significance gates or score APIs.

    Inverse marginal coefficient variance (1 / SE**2) is an information proxy,
    not the diagonal of the full information matrix. Its log ratio treats an
    equal multiplicative information gain and loss symmetrically.
    """
    coefficient_errors, information_errors, details = [], [], {}
    for cause in CAUSES:
        target, generated = target_fits[cause], generated_fits[cause]
        details[cause] = {}
        for feature, target_se in target["standard_errors"].items():
            generated_se = generated["standard_errors"].get(feature)
            if generated_se is None or not np.isfinite(generated_se) or generated_se <= 0:
                raise ValueError("Generated Cox lacks a target coefficient standard error")
            target_beta = target["coefficients"][feature]
            generated_beta = generated["coefficients"][feature]
            z_error = (generated_beta - target_beta) / target_se
            log_information_ratio = 2.0 * np.log(target_se / generated_se)
            coefficient_errors.append(z_error ** 2)
            information_errors.append(log_information_ratio ** 2)
            details[cause][feature] = {
                "target_coefficient": float(target_beta), "generated_fit_coefficient": float(generated_beta),
                "target_standard_error": float(target_se), "generated_standard_error": float(generated_se),
                "standardized_coefficient_error": float(z_error),
                "log_inverse_variance_ratio": float(log_information_ratio)}
    coefficient_loss = float(np.mean(coefficient_errors)) if coefficient_errors else 0.0
    information_loss = float(np.mean(information_errors)) if information_errors else 0.0
    total = coefficient_loss + CALIBRATION_INFORMATION_WEIGHT * information_loss
    if not np.isfinite(total):
        raise ValueError("Non-finite calibration loss")
    return total, {"coefficient_mse": coefficient_loss, "information_log_ratio_mse": information_loss,
                   "coefficient_count": len(coefficient_errors), "per_coefficient": details}


def _calibrate(x: pd.DataFrame, center: pd.Series, initial_coefficients: dict, target_fits: dict,
               times: np.ndarray, causes: np.ndarray, horizon: float, uniforms: np.ndarray,
               initial_outcomes: pd.DataFrame, steps: int) -> tuple[pd.DataFrame, dict]:
    """Bounded indirect inference with common random numbers, including step 0."""
    current = {cause: values.copy() for cause, values in initial_coefficients.items()}
    history, best_outcomes = [], None
    best_loss, best_iteration = np.inf, None
    stopped_reason = None
    for iteration in range(steps + 1):
        outcomes = initial_outcomes if iteration == 0 else _assign_events(
            x, center, current, times, causes, horizon, uniforms)
        generated_fits = {}
        record = {"iteration": iteration,
                  "generation_coefficients": {cause: values.to_dict() for cause, values in current.items()}}
        try:
            for cause in CAUSES:
                _, generated_fits[cause] = _fit_cause(x, outcomes, cause)
            loss, details = _calibration_loss(target_fits, generated_fits)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            record.update({"status": "failed", "loss": None, "reason": type(exc).__name__,
                           "generated_fits": generated_fits})
            history.append(record)
            if best_outcomes is None:
                raise RuntimeError("Initial synthetic Cox analysis failed during calibration") from exc
            stopped_reason = "synthetic_cox_fit_or_loss_failed"
            break
        record.update({"status": "evaluated", "loss": loss, "loss_detail": details,
                       "generated_fits": generated_fits})
        history.append(record)
        # The uncalibrated candidate is eligible. Exact ties prefer the earlier
        # iterate. All candidates share one schedule and one fixed uniform array.
        if loss < best_loss:
            best_loss, best_iteration, best_outcomes = loss, iteration, outcomes.copy()
        if iteration == steps:
            continue
        for cause in CAUSES:
            target = target_fits[cause]
            for feature, se in target["standard_errors"].items():
                desired = target["coefficients"][feature]
                fitted = generated_fits[cause]["coefficients"][feature]
                delta = np.clip(CALIBRATION_STEP * (desired - fitted),
                                -CALIBRATION_UPDATE_SE_LIMIT * se, CALIBRATION_UPDATE_SE_LIMIT * se)
                current[cause].loc[feature] = np.clip(
                    current[cause][feature] + delta,
                    desired - CALIBRATION_TOTAL_SE_LIMIT * se, desired + CALIBRATION_TOTAL_SE_LIMIT * se)
    audit = {"method": "bounded_indirect_inference_common_random_numbers",
             "requested_steps": int(steps), "evaluated_candidates": sum(r["status"] == "evaluated" for r in history),
             "step_size": CALIBRATION_STEP, "update_se_limit": CALIBRATION_UPDATE_SE_LIMIT,
             "total_displacement_se_limit": CALIBRATION_TOTAL_SE_LIMIT,
             "information_weight": CALIBRATION_INFORMATION_WEIGHT,
             "loss": "mean(((beta_C-beta_B)/se_B)^2) + information_weight * mean(log((se_B/se_C)^2)^2)",
             "selection": "minimum training-statistic loss; earliest iteration on ties",
             "same_timeline_and_uniforms_every_iteration": True, "official_score_used": False,
             "significance_threshold_used": False, "external_evaluation_data_used": False,
             "selected_iteration": int(best_iteration), "selected_loss": float(best_loss),
             "initial_loss": float(history[0]["loss"]), "stopped_reason": stopped_reason,
             "history": history}
    return best_outcomes, audit


def synthesize_outcomes(b: pd.DataFrame, c: pd.DataFrame, horizon: float,
                        seed: int = 20260917, calibration_steps: int = 0
                        ) -> tuple[pd.DataFrame, dict]:
    """Generate exclusive competing events from B-only fitted statistics.

    B and C must have the same number of rows because event counts are preserved
    exactly. Extra source columns (including identifiers and private labels) are
    ignored. Optionally perform up to eight bounded coefficient updates using
    B-only aggregate targets, a fixed schedule and common random uniforms. Zero
    steps preserves the original uncalibrated generation and audit path.
    """
    if (isinstance(calibration_steps, (bool, np.bool_))
            or not isinstance(calibration_steps, (int, np.integer))
            or not 0 <= calibration_steps <= MAX_CALIBRATION_STEPS):
        raise ValueError("calibration_steps must be an integer from 0 to 8")
    if not np.isfinite(horizon) or horizon <= TIME_FLOOR:
        raise ValueError("horizon must be finite and greater than TIME_FLOOR")
    if len(b) == 0 or len(b) != len(c):
        raise ValueError("B and C must have the same nonzero row count")
    b = b[RAW_FEATURES + ["time", "onset", "death"]].reset_index(drop=True).copy()
    events = b[list(CAUSES)].to_numpy()
    source_times = b["time"].to_numpy(dtype=float)
    if not np.isin(events, [0, 1]).all() or (events.sum(axis=1) > 1).any():
        raise ValueError("Source events must be exclusive binary indicators")
    if not np.isfinite(source_times).all() or (source_times <= 0).any() or (source_times > horizon).any():
        raise ValueError("Source times must be finite and lie in (0, horizon]")
    admin = events.sum(axis=1) == 0
    admin_times = source_times[admin]
    admin_delta = np.abs(admin_times - horizon)
    if (admin_delta > ADMIN_CENSOR_ATOL + 1e-12).any():
        raise ValueError("Only administrative censoring within one day of horizon is supported")
    # Retain original B times for Cox estimation, matching analysis of the
    # distributed B. The generated administrative censoring is exactly horizon.
    censor_audit = {"input_tolerance_years": ADMIN_CENSOR_ATOL,
                    "administrative_count": int(admin.sum()),
                    "output_times_canonicalized_count": int((admin_times != horizon).sum()),
                    "source_time_min": float(admin_times.min()) if len(admin_times) else None,
                    "source_time_max": float(admin_times.max()) if len(admin_times) else None,
                    "maximum_adjustment_years": float(admin_delta.max()) if len(admin_delta) else 0.0,
                    "training_times_modified": False, "output_censor_time": float(horizon)}
    x_b, x_c = cox_features(b), cox_features(c)
    coefficients, fits = {}, {}
    for cause in CAUSES:
        coefficients[cause], fits[cause] = _fit_cause(x_b, b, cause)
    timeline_seed, assignment_seed = np.random.SeedSequence(seed).spawn(2)
    times, causes, schedule_audit = _event_schedule(b, horizon, np.random.default_rng(timeline_seed))
    uniforms = np.random.default_rng(assignment_seed).random(len(times))
    outcomes = _assign_events(x_c, x_b.mean(), coefficients, times, causes, horizon, uniforms)
    calibration_audit = None
    if calibration_steps:
        outcomes, calibration_audit = _calibrate(
            x_c, x_b.mean(), coefficients, fits, times, causes, horizon, uniforms, outcomes, int(calibration_steps))
    audit = {"method": "conditional_cause_specific_cox_risk_set", "seed": int(seed),
             "source_rows": len(b), "generated_rows": len(c), "training_input": "B only",
             "calibration_steps": int(calibration_steps), "score_based_seed_search": False,
             "horizon": float(horizon), "time_floor": TIME_FLOOR,
             "administrative_censoring": censor_audit,
             "source_row_matching": False, "fits": fits, "schedule": schedule_audit,
             "event_counts": {cause: int(outcomes[cause].sum()) for cause in CAUSES}}
    if calibration_audit is not None:
        audit["calibration"] = calibration_audit
    return outcomes, audit
