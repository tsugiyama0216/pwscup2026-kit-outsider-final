"""Robust, fail-closed registration features for the PWSCUP2026 attack.

The release tables do not carry stable record identifiers.  This module treats
putative row correspondences as *unlabelled* anchors and fits a positive affine
map per numeric column with iterative closest points and RANSAC.  Two disjoint
numeric views nominate separate tentative-anchor sets.  The two resulting maps
are then compared using the same full numeric and categorical distance, so the
final agreement is deliberately treated as correlated corroboration rather
than as two independent linkage tests.  Guaranteed non-members are split into
calibration and untouched audit folds; they never fit an affine map.

The routines only produce research scores / normal-format-shaped overlays.
Every complete output still needs the separate official logical validator
before production use.  They do not read a token, create an archive, submit,
or infer hidden truth from public leaderboard feedback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, RANSACRegressor


DEFAULT_NUMERIC_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
DEFAULT_CATEGORICAL_COLUMNS = ("sex", "prefecture", "smoking")
DEFAULT_VIEW_A = ("age", "SBP", "TG", "ALT")
DEFAULT_VIEW_B = ("BMI", "HDL", "FPG")


@dataclass(frozen=True)
class RegistrationConfig:
    """Conservative defaults for sparse-anchor release registration."""

    iterations: int = 3
    max_tentative_anchors: int = 160
    anchor_quantile: float = 0.35
    categorical_penalty: float = 2.0
    ransac_residual_fraction: float = 0.10
    ransac_trials: int = 300
    calibration_fraction: float = 0.50
    false_positive_rate: float = 0.01
    random_state: int = 7319


@dataclass(frozen=True)
class AffineRegistration:
    """One fitted numeric map and its fit diagnostics."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    slopes: np.ndarray
    intercepts: np.ndarray
    scales: np.ndarray
    categorical_penalty: float
    fit_view: tuple[str, ...]
    tentative_pairs: np.ndarray
    inlier_fractions: np.ndarray

    def transform_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_matrix(frame, self.numeric_columns)
        return values * self.slopes + self.intercepts

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "fit_view": list(self.fit_view),
            "slopes": dict(zip(self.numeric_columns, self.slopes.astype(float))),
            "intercepts": dict(zip(self.numeric_columns, self.intercepts.astype(float))),
            "scales": dict(zip(self.numeric_columns, self.scales.astype(float))),
            "categorical_penalty": float(self.categorical_penalty),
            "tentative_pairs": int(len(self.tentative_pairs)),
            "ransac_inlier_fraction": dict(
                zip(self.numeric_columns, self.inlier_fractions.astype(float))
            ),
        }


@dataclass(frozen=True)
class RegisteredLinkageResult:
    """Correlated full-distance scores from two anchor-view fits and their audit."""

    model_a: AffineRegistration
    model_b: AffineRegistration
    score: np.ndarray
    nearest_release_row: np.ndarray
    view_agreement: np.ndarray
    mutual_agreement: np.ndarray
    accepted: np.ndarray
    threshold: float
    calibration_indices: np.ndarray
    audit_indices: np.ndarray
    diagnostics: Mapping[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")
    if len(frame) < 4:
        raise ValueError(f"{label} has too few rows for registration")


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("registration numeric columns must be finite")
    return values


def _categorical_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    values = frame.loc[:, list(columns)].astype("string").fillna("<NA>").to_numpy(object)
    return values


def _robust_scale(values: np.ndarray) -> np.ndarray:
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    iqr = q75 - q25
    mad = 1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0)
    std = np.std(values, axis=0)
    return np.maximum.reduce([iqr, mad, std, np.full(values.shape[1], 1e-8)])


def _quantile_affine(source: np.ndarray, release: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_q = np.quantile(source, [0.25, 0.50, 0.75], axis=0)
    release_q = np.quantile(release, [0.25, 0.50, 0.75], axis=0)
    source_iqr = source_q[2] - source_q[0]
    release_iqr = release_q[2] - release_q[0]
    slopes = np.divide(
        release_iqr,
        source_iqr,
        out=np.ones_like(release_iqr, dtype=float),
        where=np.abs(source_iqr) > 1e-12,
    )
    # Numeric QIs should retain orientation.  Very broad bounds allow unit
    # changes while refusing an implausible sign reversal or collapse.
    slopes = np.clip(slopes, 1e-3, 1e3)
    intercepts = release_q[1] - slopes * source_q[1]
    return slopes, intercepts


def _distance_matrix(
    mapped_source: np.ndarray,
    release_numeric: np.ndarray,
    scales: np.ndarray,
    source_categorical: np.ndarray,
    release_categorical: np.ndarray,
    numeric_indices: Sequence[int],
    categorical_penalty: float,
) -> np.ndarray:
    if numeric_indices:
        left = mapped_source[:, numeric_indices] / scales[list(numeric_indices)]
        right = release_numeric[:, numeric_indices] / scales[list(numeric_indices)]
        distances = cdist(left, right, metric="sqeuclidean") / len(numeric_indices)
    else:
        distances = np.zeros((len(mapped_source), len(release_numeric)), dtype=float)
    if source_categorical.shape[1]:
        mismatches = np.zeros_like(distances)
        for column in range(source_categorical.shape[1]):
            mismatches += (
                source_categorical[:, column, None]
                != release_categorical[None, :, column]
            )
        distances += categorical_penalty * mismatches / source_categorical.shape[1]
    return distances


def _nearest_two(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if distances.shape[1] < 2:
        raise ValueError("release needs at least two rows")
    partition = np.argpartition(distances, kth=1, axis=1)[:, :2]
    pair_distances = np.take_along_axis(distances, partition, axis=1)
    order = np.argsort(pair_distances, axis=1, kind="mergesort")
    pair_indices = np.take_along_axis(partition, order, axis=1)
    pair_distances = np.take_along_axis(pair_distances, order, axis=1)
    nearest = pair_indices[:, 0]
    best = pair_distances[:, 0]
    margin = (pair_distances[:, 1] - best) / np.maximum(pair_distances[:, 1], 1e-12)
    return nearest.astype(int), best, margin


def _tentative_pairs(
    distances: np.ndarray,
    source_fit_indices: np.ndarray,
    *,
    anchor_quantile: float,
    max_anchors: int,
) -> np.ndarray:
    local = distances[source_fit_indices]
    nearest, best, margin = _nearest_two(local)
    reverse = np.argmin(local, axis=0)
    mutual = reverse[nearest] == np.arange(len(source_fit_indices))
    candidates = np.flatnonzero(mutual)
    if not len(candidates):
        candidates = np.arange(len(source_fit_indices))
    cutoff = float(np.quantile(best[candidates], anchor_quantile))
    candidates = candidates[best[candidates] <= cutoff + 1e-12]
    # Prefer a clear nearest-neighbour gap, then low absolute distance.
    ordering = np.lexsort((best[candidates], -margin[candidates]))
    candidates = candidates[ordering[:max_anchors]]
    return np.column_stack([source_fit_indices[candidates], nearest[candidates]]).astype(int)


def _fit_ransac_column(
    source: np.ndarray,
    release: np.ndarray,
    pairs: np.ndarray,
    initial_slope: float,
    initial_intercept: float,
    scale: float,
    config: RegistrationConfig,
    seed: int,
) -> tuple[float, float, float]:
    if len(pairs) < 4 or np.ptp(source[pairs[:, 0]]) <= 1e-12:
        return initial_slope, initial_intercept, 0.0
    x = source[pairs[:, 0]].reshape(-1, 1)
    y = release[pairs[:, 1]]
    threshold = max(config.ransac_residual_fraction * scale, 1e-8)
    estimator = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=max(3, int(np.ceil(0.20 * len(pairs)))),
        residual_threshold=threshold,
        max_trials=config.ransac_trials,
        random_state=seed,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            estimator.fit(x, y)
        slope = float(estimator.estimator_.coef_[0])
        intercept = float(estimator.estimator_.intercept_)
        inliers = float(np.mean(estimator.inlier_mask_))
    except (ValueError, ArithmeticError):
        return initial_slope, initial_intercept, 0.0
    ratio = slope / max(initial_slope, 1e-12)
    if not np.isfinite([slope, intercept, inliers]).all() or slope <= 0 or not 0.1 <= ratio <= 10:
        return initial_slope, initial_intercept, 0.0
    return slope, intercept, inliers


def fit_affine_icp(
    source: pd.DataFrame,
    release: pd.DataFrame,
    source_fit_mask: np.ndarray,
    *,
    fit_view: Sequence[str],
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    config: RegistrationConfig = RegistrationConfig(),
) -> AffineRegistration:
    """Fit a positive per-column affine map from tentative mutual anchors."""
    numeric_columns = tuple(numeric_columns)
    categorical_columns = tuple(categorical_columns)
    fit_view = tuple(fit_view)
    _require_columns(source, numeric_columns + categorical_columns, "source")
    _require_columns(release, numeric_columns + categorical_columns, "release")
    if not set(fit_view) <= set(numeric_columns) or not fit_view:
        raise ValueError("fit_view must be a nonempty numeric-column subset")
    source_fit_mask = np.asarray(source_fit_mask, dtype=bool)
    if source_fit_mask.shape != (len(source),) or source_fit_mask.sum() < 8:
        raise ValueError("source_fit_mask must select at least eight rows")
    if config.iterations < 1 or config.max_tentative_anchors < 4:
        raise ValueError("invalid registration iteration/anchor configuration")
    if not 0 < config.anchor_quantile <= 1 or config.categorical_penalty < 0:
        raise ValueError("invalid registration distance configuration")

    source_numeric = _numeric_matrix(source, numeric_columns)
    release_numeric = _numeric_matrix(release, numeric_columns)
    source_categorical = _categorical_matrix(source, categorical_columns)
    release_categorical = _categorical_matrix(release, categorical_columns)
    source_fit_indices = np.flatnonzero(source_fit_mask)
    slopes, intercepts = _quantile_affine(source_numeric[source_fit_indices], release_numeric)
    scales = _robust_scale(release_numeric)
    view_indices = [numeric_columns.index(column) for column in fit_view]
    inlier_fractions = np.zeros(len(numeric_columns), dtype=float)
    pairs = np.empty((0, 2), dtype=int)

    for iteration in range(config.iterations):
        mapped = source_numeric * slopes + intercepts
        distances = _distance_matrix(
            mapped,
            release_numeric,
            scales,
            source_categorical,
            release_categorical,
            view_indices,
            config.categorical_penalty,
        )
        pairs = _tentative_pairs(
            distances,
            source_fit_indices,
            anchor_quantile=config.anchor_quantile,
            max_anchors=config.max_tentative_anchors,
        )
        next_slopes, next_intercepts, next_inliers = [], [], []
        for column in range(len(numeric_columns)):
            slope, intercept, fraction = _fit_ransac_column(
                source_numeric[:, column],
                release_numeric[:, column],
                pairs,
                float(slopes[column]),
                float(intercepts[column]),
                float(scales[column]),
                config,
                config.random_state + 101 * iteration + column,
            )
            next_slopes.append(slope)
            next_intercepts.append(intercept)
            next_inliers.append(fraction)
        slopes = np.asarray(next_slopes)
        intercepts = np.asarray(next_intercepts)
        inlier_fractions = np.asarray(next_inliers)

    return AffineRegistration(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        slopes=slopes,
        intercepts=intercepts,
        scales=scales,
        categorical_penalty=float(config.categorical_penalty),
        fit_view=fit_view,
        tentative_pairs=pairs,
        inlier_fractions=inlier_fractions,
    )


def _model_link_metrics(
    model: AffineRegistration,
    source: pd.DataFrame,
    release: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mapped = model.transform_numeric(source)
    release_numeric = _numeric_matrix(release, model.numeric_columns)
    distances = _distance_matrix(
        mapped,
        release_numeric,
        model.scales,
        _categorical_matrix(source, model.categorical_columns),
        _categorical_matrix(release, model.categorical_columns),
        range(len(model.numeric_columns)),
        categorical_penalty=model.categorical_penalty,
    )
    nearest, best, margin = _nearest_two(distances)
    reverse = np.argmin(distances, axis=0)
    mutual = reverse[nearest] == np.arange(len(source))
    return nearest, best, margin, mutual


def registered_linkage_scores(
    source: pd.DataFrame,
    release: pd.DataFrame,
    known_negative_mask: np.ndarray,
    *,
    protected_mask: np.ndarray | None = None,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    view_a: Sequence[str] = DEFAULT_VIEW_A,
    view_b: Sequence[str] = DEFAULT_VIEW_B,
    config: RegistrationConfig = RegistrationConfig(),
) -> RegisteredLinkageResult:
    """Score links from two anchor-fit views and split-negative calibration.

    The anchor sets come from disjoint numeric views, but each fitted model's
    final nearest-neighbour score uses all numeric columns and the same
    categories.  Thus ``view_agreement`` is correlated model agreement, not an
    independent replication.  The returned ``accepted`` mask is research
    output only: protected and known-negative rows are always false.  A caller
    still needs target-level source-swap validation and the separate official
    logical validator before changing a submission.
    """
    known_negative_mask = np.asarray(known_negative_mask, dtype=bool)
    if known_negative_mask.shape != (len(source),) or known_negative_mask.sum() < 20:
        raise ValueError("known_negative_mask must select at least twenty rows")
    if protected_mask is None:
        protected_mask = np.zeros(len(source), dtype=bool)
    protected_mask = np.asarray(protected_mask, dtype=bool)
    if protected_mask.shape != (len(source),):
        raise ValueError("protected_mask has wrong shape")
    if set(view_a) & set(view_b) or not set(tuple(view_a) + tuple(view_b)) <= set(numeric_columns):
        raise ValueError("registration views must be disjoint numeric subsets")
    if not 0 < config.calibration_fraction < 1 or not 0 < config.false_positive_rate < 0.5:
        raise ValueError("invalid negative calibration configuration")

    rng = np.random.default_rng(config.random_state)
    negatives = np.flatnonzero(known_negative_mask)
    negatives = negatives[rng.permutation(len(negatives))]
    split = int(np.floor(config.calibration_fraction * len(negatives)))
    split = min(max(split, 10), len(negatives) - 10)
    calibration_indices = np.sort(negatives[:split])
    audit_indices = np.sort(negatives[split:])
    fit_mask = ~(known_negative_mask | protected_mask)
    model_a = fit_affine_icp(
        source,
        release,
        fit_mask,
        fit_view=view_a,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=config,
    )
    model_b = fit_affine_icp(
        source,
        release,
        fit_mask,
        fit_view=view_b,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config=RegistrationConfig(**{**config.__dict__, "random_state": config.random_state + 10000}),
    )
    nearest_a, distance_a, margin_a, mutual_a = _model_link_metrics(model_a, source, release)
    nearest_b, distance_b, margin_b, mutual_b = _model_link_metrics(model_b, source, release)
    view_agreement = nearest_a == nearest_b
    mutual_agreement = mutual_a & mutual_b
    # Agreement does not enter as an arbitrary large bonus: non-agreeing rows
    # are made ineligible, and the continuous score remains calibratable.
    score = -0.5 * (distance_a + distance_b) + 0.25 * np.minimum(margin_a, margin_b)
    calibration_scores = score[calibration_indices]
    threshold = float(
        np.quantile(calibration_scores, 1.0 - config.false_positive_rate, method="higher")
    )
    eligible = ~(known_negative_mask | protected_mask)
    accepted = eligible & view_agreement & mutual_agreement & (score > threshold)
    calibration_hits = view_agreement[calibration_indices] & mutual_agreement[calibration_indices] & (
        score[calibration_indices] > threshold
    )
    audit_hits = view_agreement[audit_indices] & mutual_agreement[audit_indices] & (
        score[audit_indices] > threshold
    )
    diagnostics = {
        "method": "two_anchor_view_affine_icp_ransac_split_negative_conformal",
        "rows": int(len(source)),
        "release_rows": int(len(release)),
        "fit_rows": int(fit_mask.sum()),
        "known_negative_rows": int(known_negative_mask.sum()),
        "protected_rows": int(protected_mask.sum()),
        "calibration_rows": int(len(calibration_indices)),
        "audit_rows": int(len(audit_indices)),
        "threshold": threshold,
        "calibration_false_links": int(calibration_hits.sum()),
        "audit_false_links": int(audit_hits.sum()),
        "audit_false_link_rate": float(audit_hits.mean()),
        "view_agreement_rows": int(view_agreement.sum()),
        "mutual_agreement_rows": int(mutual_agreement.sum()),
        "accepted_rows": int(accepted.sum()),
        "model_a": model_a.as_dict(),
        "model_b": model_b.as_dict(),
    }
    return RegisteredLinkageResult(
        model_a=model_a,
        model_b=model_b,
        score=score,
        nearest_release_row=nearest_a,
        view_agreement=view_agreement,
        mutual_agreement=mutual_agreement,
        accepted=accepted,
        threshold=threshold,
        calibration_indices=calibration_indices,
        audit_indices=audit_indices,
        diagnostics=diagnostics,
    )


def cross_view_anchor_audit(
    model: AffineRegistration,
    source: pd.DataFrame,
    release: pd.DataFrame,
    *,
    validation_columns: Sequence[str],
    train_fraction: float = 0.50,
    permutations: int = 128,
    random_state: int = 991,
) -> dict[str, Any]:
    """Test tentative anchors on columns not used to nominate those anchors.

    Validation-column affine maps are fitted on one anchor half and scored on
    the other.  Jointly permuting the held-out release rows supplies a null
    residual without breaking within-row correlations.  This is a mechanism
    diagnostic, not hidden membership ground truth.
    """
    validation_columns = tuple(validation_columns)
    if not validation_columns or set(validation_columns) & set(model.fit_view):
        raise ValueError("validation columns must be nonempty and disjoint from fit_view")
    if not set(validation_columns) <= set(model.numeric_columns):
        raise ValueError("unknown validation column")
    if not 0.25 <= train_fraction <= 0.75 or permutations < 16:
        raise ValueError("invalid holdout audit configuration")
    pairs = np.asarray(model.tentative_pairs, dtype=int)
    if len(pairs) < 12:
        return {
            "status": "insufficient_tentative_pairs",
            "pairs": int(len(pairs)),
            "validation_columns": list(validation_columns),
        }
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(pairs))
    split = int(np.floor(train_fraction * len(pairs)))
    train_pairs = pairs[order[:split]]
    audit_pairs = pairs[order[split:]]
    source_numeric = _numeric_matrix(source, model.numeric_columns)
    release_numeric = _numeric_matrix(release, model.numeric_columns)
    initial_slopes, initial_intercepts = _quantile_affine(source_numeric, release_numeric)
    config = RegistrationConfig(random_state=random_state)
    predicted, observed = [], []
    for offset, column_name in enumerate(validation_columns):
        column = model.numeric_columns.index(column_name)
        slope, intercept, _ = _fit_ransac_column(
            source_numeric[:, column],
            release_numeric[:, column],
            train_pairs,
            float(initial_slopes[column]),
            float(initial_intercepts[column]),
            float(model.scales[column]),
            config,
            random_state + offset,
        )
        predicted.append(source_numeric[audit_pairs[:, 0], column] * slope + intercept)
        observed.append(release_numeric[audit_pairs[:, 1], column])
    predicted_matrix = np.column_stack(predicted)
    observed_matrix = np.column_stack(observed)
    scale = np.array(
        [model.scales[model.numeric_columns.index(column)] for column in validation_columns]
    )
    residual = np.sqrt(np.mean(((predicted_matrix - observed_matrix) / scale) ** 2, axis=1))
    observed_median = float(np.median(residual))
    null_medians = []
    for _ in range(permutations):
        permuted = observed_matrix[rng.permutation(len(observed_matrix))]
        null_residual = np.sqrt(
            np.mean(((predicted_matrix - permuted) / scale) ** 2, axis=1)
        )
        null_medians.append(float(np.median(null_residual)))
    null_medians_array = np.asarray(null_medians)
    null_median = float(np.median(null_medians_array))
    return {
        "status": "ok",
        "selection_view": list(model.fit_view),
        "validation_columns": list(validation_columns),
        "training_pairs": int(len(train_pairs)),
        "audit_pairs": int(len(audit_pairs)),
        "observed_median_scaled_residual": observed_median,
        "permuted_median_scaled_residual": null_median,
        "residual_ratio": float(observed_median / max(null_median, 1e-12)),
        "permutation_p_lower": float(
            (1 + np.count_nonzero(null_medians_array <= observed_median))
            / (1 + len(null_medians_array))
        ),
    }


def registered_event_times(
    challenge: pd.DataFrame,
    release: pd.DataFrame,
    registration: AffineRegistration,
    *,
    horizon: float = 10.0,
    abstain: float = 11.0,
    maximum_distance: float | None = None,
    link_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a fitted raw-to-release map to challenge QIs and copy trusted events.

    This is a decoder, not an activation policy or complete format validator.
    Rows without a close released neighbour or without a released event remain
    at the normal-format abstain value.  A complete AIA frame must separately
    pass the official logical validator and target-transfer validation before
    production use.
    """
    if not np.isfinite([horizon, abstain]).all() or horizon <= 0 or abstain <= horizon:
        raise ValueError("invalid AIA horizon/abstention values")
    if link_columns is None:
        link_columns = registration.fit_view
    link_columns = tuple(link_columns)
    if not link_columns or not set(link_columns) <= set(registration.numeric_columns):
        raise ValueError("link_columns must be a nonempty fitted numeric subset")
    required_release = registration.numeric_columns + registration.categorical_columns + (
        "time",
        "onset",
        "death",
    )
    _require_columns(challenge, registration.numeric_columns + registration.categorical_columns, "challenge")
    _require_columns(release, required_release, "release")
    mapped = registration.transform_numeric(challenge)
    distances = _distance_matrix(
        mapped,
        _numeric_matrix(release, registration.numeric_columns),
        registration.scales,
        _categorical_matrix(challenge, registration.categorical_columns),
        _categorical_matrix(release, registration.categorical_columns),
        [registration.numeric_columns.index(column) for column in link_columns],
        categorical_penalty=registration.categorical_penalty,
    )
    nearest, best, margin = _nearest_two(distances)
    reverse = np.argmin(distances, axis=0)
    mutual = reverse[nearest] == np.arange(len(challenge))
    if maximum_distance is None:
        maximum_distance = float(np.quantile(best[mutual], 0.25)) if mutual.any() else -np.inf
    event = (
        pd.to_numeric(release["onset"], errors="coerce").fillna(0).to_numpy(float)
        + pd.to_numeric(release["death"], errors="coerce").fillna(0).to_numpy(float)
    ) > 0
    release_time = pd.to_numeric(release["time"], errors="coerce").to_numpy(float)
    if not np.isfinite(release_time).all() or (release_time < 0).any():
        raise ValueError("release time must be finite and nonnegative")
    gate = mutual & (best <= maximum_distance) & event[nearest] & (release_time[nearest] <= horizon)
    output = np.full(len(challenge), abstain, dtype=float)
    output[gate] = release_time[nearest[gate]]
    return output, {
        "method": "registered_affine_nearest_event_decoder",
        "link_columns": list(link_columns),
        "rows": int(len(challenge)),
        "mutual_rows": int(mutual.sum()),
        "decoded_event_rows": int(gate.sum()),
        "maximum_distance": float(maximum_distance),
        "median_best_distance": float(np.median(best)),
        "median_margin": float(np.median(margin)),
    }


def overlay_registered_mia(
    base: pd.DataFrame,
    target: str,
    accepted: np.ndarray,
    confidence: np.ndarray,
    *,
    known_negative_mask: np.ndarray,
    rare_mask: np.ndarray,
    protected_targets: Iterable[str] = ("1", "2"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Promote research rows while preserving guards; validate full output separately."""
    target = str(target)
    if target in set(map(str, protected_targets)):
        raise ValueError("ceiling target is protected")
    if "record_id" not in base or target not in base:
        raise ValueError("MIA frame lacks record_id or target column")
    n = len(base)
    accepted = np.asarray(accepted, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    known_negative_mask = np.asarray(known_negative_mask, dtype=bool)
    rare_mask = np.asarray(rare_mask, dtype=bool)
    if any(values.shape != (n,) for values in (accepted, confidence, known_negative_mask, rare_mask)):
        raise ValueError("MIA overlay masks/scores have wrong shape")
    if not np.isfinite(confidence).all():
        raise ValueError("MIA overlay confidence must be finite")
    mutable = accepted & ~known_negative_mask & ~rare_mask
    output = base.copy()
    original = pd.to_numeric(output[target], errors="coerce").to_numpy(float)
    if not np.isfinite(original).all():
        raise ValueError("base MIA scores must be finite")
    nominated = mutable & (original < 1.0)
    changed = np.zeros(n, dtype=bool)
    if nominated.any():
        order = np.argsort(confidence[nominated], kind="mergesort")
        promoted = np.linspace(0.995, 0.999, nominated.sum(), endpoint=True)
        positions = np.flatnonzero(nominated)[order]
        values = original.copy()
        values[positions] = np.maximum(values[positions], promoted)
        output[target] = values
        changed = ~np.isclose(values, original, rtol=0.0, atol=0.0)
    if not np.array_equal(
        output.loc[known_negative_mask | rare_mask, target].to_numpy(),
        base.loc[known_negative_mask | rare_mask, target].to_numpy(),
    ):
        raise RuntimeError("MIA guard rows changed")
    return output, {
        "target": target,
        "accepted_rows": int(accepted.sum()),
        "changed_rows": int(changed.sum()),
        "known_private_preserved": True,
        "rare_preserved": True,
    }
