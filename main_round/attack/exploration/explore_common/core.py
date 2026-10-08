"""Shared geometry; no filesystem access and no evaluation labels."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

CONTINUOUS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
QI_CONTINUOUS = ("age", "SBP", "TG", "HDL", "ALT")
QI_CATEGORICAL = ("sex", "prefecture")


def score_to_confidence(raw):
    """Monotone midranks, not calibrated membership probabilities; ties stay tied."""
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite one-dimensional score array")
    return (rankdata(values, method="average") - 0.5) / len(values)


def _numeric(frame, columns):
    result = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Attack input contains nonfinite numeric values")
    return result


def _blocked_distances(release, queries, background):
    """QI RMS distance, (sex,prefecture) -> sex -> all release rows fallback."""
    if not len(release) or len(background) < 2:
        raise ValueError("A nonempty release and at least two background rows are required")
    ref = _numeric(background, QI_CONTINUOUS)
    scale = ref.std(axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    c = _numeric(release, QI_CONTINUOUS) / scale
    q = _numeric(queries, QI_CONTINUOUS) / scale
    release_keys = release.loc[:, list(QI_CATEGORICAL)].to_numpy()
    query_keys = queries.loc[:, list(QI_CATEGORICAL)].to_numpy()
    # Positional masks make behaviour independent of DataFrame index labels.
    for i, key in enumerate(query_keys):
        keep = np.flatnonzero(np.all(release_keys == key, axis=1))
        if not len(keep):
            keep = np.flatnonzero(release_keys[:, 0] == key[0])
        if not len(keep):
            keep = np.arange(len(release))
        # Match the official AIA floating-point operation order, including ties.
        distances = np.sqrt(np.mean((c[keep] - q[i]) ** 2, axis=1))
        yield keep, distances


def nearest_distance_scores(release, queries, background):
    """ClosestDistanceMIA on the same seven public QI attributes as AIA."""
    return np.asarray([-float(dist.min()) for _, dist in _blocked_distances(release, queries, background)])


def densest_time_center(values, tau=0.5):
    """Center of a maximum-count closed window of width 2*tau.

    Sort times, choose the earliest maximal window on count ties, then return
    its occupied-endpoint midpoint. This is deterministic and order invariant.
    """
    times = np.sort(np.asarray(values, dtype=float))
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or tau < 0:
        raise ValueError("Expected finite nonempty times and nonnegative tolerance")
    right = 0
    best_left, best_right, best_count = 0, 0, 0
    for left in range(len(times)):
        right = max(right, left)
        while right < len(times) and times[right] - times[left] <= 2 * tau:
            right += 1
        count = right - left
        if count > best_count:
            best_left, best_right, best_count = left, right - 1, count
    return float((times[best_left] + times[best_right]) / 2)


def local_time_predictions(release, queries, background, k=10, mode="window"):
    """AIA: exact 1-NN with tie mean, or unweighted k-NN interval mode."""
    if not isinstance(k, int) or k < 1 or mode not in {"nearest", "window"}:
        raise ValueError("k must be positive and mode must be nearest or window")
    times = _numeric(release, ("time",))[:, 0]
    output = []
    for indices, distances in _blocked_distances(release, queries, background):
        if mode == "nearest":
            selected = times[indices[distances <= distances.min()]]
            output.append(float(np.mean(np.sort(selected))))
        else:
            kth = np.partition(distances, min(k, len(distances)) - 1)[min(k, len(distances)) - 1]
            selected = times[indices[distances <= kth]]
            output.append(densest_time_center(selected))
    return np.asarray(output)
