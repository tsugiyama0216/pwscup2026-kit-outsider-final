"""Deterministic v1 attack models, using only the supplied public feature tables.

This module is pure: no file reads/writes, labels, record IDs, random seeds, or
previous attack candidates. The caller applies known-private exclusions and
cross-target constraints separately. C and background are canonically sorted
by their 13 public values; candidate and challenge output order is preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
PUBLIC_CONTINUOUS = ["age", "SBP", "TG", "HDL", "ALT"]
PUBLIC_COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL",
                  "ALT", "smoking", "FPG", "time", "onset", "death"]
QI_COLUMNS = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
VARIANTS = ("official", "clinical_nearest", "full_nearest",
            "ensemble_nearest", "ensemble_window", "full_reference_aia")


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, PUBLIC_COLUMNS].copy(deep=True).sort_values(
        PUBLIC_COLUMNS, kind="mergesort").reset_index(drop=True)


def _validate(frame: pd.DataFrame, columns: list[str], label: str,
              *, allow_empty=False) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing public columns: {sorted(missing)}")
    if not allow_empty and len(frame) == 0:
        raise ValueError(f"{label}: empty reference table")
    numeric = [column for column in columns if column != "prefecture"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite public numeric value")
    if "prefecture" in columns and frame["prefecture"].isna().any():
        raise ValueError(f"{label}: missing prefecture")


def _continuous(frame: pd.DataFrame, columns: list[str], log_labs=True) -> np.ndarray:
    result = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    if log_labs:
        for column in ("TG", "ALT"):
            if column in columns:
                position = columns.index(column)
                if np.any(result[:, position] <= -1):
                    raise ValueError(f"{column}: log1p requires values greater than -1")
                result[:, position] = np.log1p(result[:, position])
    return result


def _scaled(frames: tuple[pd.DataFrame, ...], background: pd.DataFrame,
            columns: list[str], log_labs=True) -> tuple[np.ndarray, ...]:
    ref = _continuous(background, columns, log_labs)
    mean = ref.mean(axis=0)
    scale = ref.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return tuple((_continuous(frame, columns, log_labs) - mean) / scale for frame in frames)


def _full_features(frames: tuple[pd.DataFrame, ...], background: pd.DataFrame
                   ) -> tuple[np.ndarray, ...]:
    clinical = _scaled(frames, background, CONTINUOUS, True)
    # Outcomes supplement the clinical signal. They cannot dominate through
    # the very frequent time=10 value or an isolated exact event-time match.
    time_scale = float(background["time"].std(ddof=0))
    if time_scale < 1e-12:
        time_scale = 1.0
    pref = sorted(set().union(*(set(frame["prefecture"].astype(str)) for frame in frames)))
    output = []
    for frame, base in zip(frames, clinical):
        pieces = [base]
        for column, weight in (("sex", .35), ("smoking", .25),
                               ("onset", .30), ("death", .30)):
            pieces.append(frame[column].to_numpy(dtype=float)[:, None] * weight)
        pieces.append(frame["time"].to_numpy(dtype=float)[:, None] * (.30 / time_scale))
        labels = frame["prefecture"].astype(str).to_numpy()
        pieces.append(np.column_stack([labels == value for value in pref]).astype(float) * .35)
        output.append(np.column_stack(pieces))
    return tuple(output)


def _groups(frame: pd.DataFrame, columns: list[str]) -> dict[tuple, np.ndarray]:
    groups: dict[tuple, list[int]] = {}
    for position, row in enumerate(frame[columns].itertuples(index=False, name=None)):
        groups.setdefault(tuple(row), []).append(position)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def _radii(reference: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.empty(len(query), dtype=float)
    kth = np.empty(len(query), dtype=float)
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        distances = cdist(query[start:stop], reference, metric="euclidean")
        first[start:stop] = distances.min(axis=1)
        kth[start:stop] = np.partition(distances, k - 1, axis=1)[:, k - 1]
    return first, kth


def _local_density_score(c: pd.DataFrame, background: pd.DataFrame,
                         candidates: pd.DataFrame, arrays: tuple[np.ndarray, ...],
                         dimension: float) -> np.ndarray:
    """Mixture of nearest and fifth-neighbor density ratios.

    Same-sex/prefecture groups are used only when BOTH references have enough
    observations. Otherwise both return to the global population. The reference
    sample-size term avoids treating a larger reference as inherently denser.
    This is a feature for ranking, not a calibrated membership probability.
    """
    c_values, bg_values, queries = arrays
    c_groups = _groups(c, ["sex", "prefecture"])
    bg_groups = _groups(background, ["sex", "prefecture"])
    query_groups = _groups(candidates, ["sex", "prefecture"])
    c_all, bg_all = np.arange(len(c)), np.arange(len(background))
    scores = np.empty(len(candidates), dtype=float)
    for key, query_indices in query_groups.items():
        ci, bi = c_groups.get(key), bg_groups.get(key)
        if ci is None or bi is None or len(ci) < 8 or len(bi) < 32:
            ci, bi = c_all, bg_all
        k = min(5, len(ci), len(bi))
        c_first, c_k = _radii(c_values[ci], queries[query_indices], k)
        bg_first, bg_k = _radii(bg_values[bi], queries[query_indices], k)
        correction = np.log(len(bi) / len(ci))
        nearest = dimension * (np.log(np.maximum(bg_first, 1e-10)) -
                               np.log(np.maximum(c_first, 1e-10))) + correction
        local = dimension * (np.log(np.maximum(bg_k, 1e-10)) -
                             np.log(np.maximum(c_k, 1e-10))) + correction
        scores[query_indices] = .65 * nearest + .35 * local
    return scores


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite membership score")
    # Average tied ranks preserve ties. Output range is open (0, 1), leaving
    # exact zero available to the caller's independently known-private mask.
    return (rankdata(values, method="average") - .5) / len(values)


def _max_window(times: np.ndarray, weights: np.ndarray, horizon: float,
                 tau: float=.5) -> float:
    """Choose a point whose +/-tau interval covers maximum neighbor weight.

    Ties prefer the interval containing the nearest-supported time, then its
    smaller center. Midpoints of the selected times avoid averaging separated
    outcome modes. Only public C times are used.
    """
    anchor = float(times[int(np.argmax(weights))])
    order = np.argsort(times, kind="mergesort")
    times, weights = times[order], weights[order]
    sums = np.concatenate(([0.0], np.cumsum(weights)))
    best_key = None
    best_center = anchor
    right = 0
    for left in range(len(times)):
        right = max(right, left)
        while right + 1 < len(times) and times[right + 1] - times[left] <= 2 * tau:
            right += 1
        mass = float(sums[right + 1] - sums[left])
        low = max(float(times[right]) - tau, 1e-12)
        high = min(float(times[left]) + tau, horizon)
        center = float(np.clip(anchor, low, high))
        key = (mass, -abs(center - anchor), -center)
        if best_key is None or key > best_key:
            best_key, best_center = key, center
    return float(np.clip(best_center, 1e-12, horizon))


def _aia_predictions(c: pd.DataFrame, background: pd.DataFrame,
                      challenge: pd.DataFrame, horizon: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    raw_c, raw_q = _scaled((c, challenge), background, PUBLIC_CONTINUOUS, False)
    log_c, log_q = _scaled((c, challenge), background, PUBLIC_CONTINUOUS, True)
    c_time = c["time"].to_numpy(dtype=float)
    blocks = _groups(c, ["sex", "prefecture"])
    sex_blocks = _groups(c, ["sex"])
    all_indices = np.arange(len(c))
    nearest = np.empty(len(challenge), dtype=float)
    window = np.empty(len(challenge), dtype=float)
    keys = challenge[["sex", "prefecture"]].itertuples(index=False, name=None)
    for i, key in enumerate(keys):
        indices = blocks.get(tuple(key))
        if indices is None or len(indices) == 0:
            indices = sex_blocks.get((key[0],), all_indices)
        d_log = np.sqrt(np.mean((log_c[indices] - log_q[i]) ** 2, axis=1))
        nearest_times = np.sort(c_time[indices[d_log <= d_log.min()]])
        nearest[i] = float(nearest_times.mean())
        d_raw = np.sqrt(np.mean((raw_c[indices] - raw_q[i]) ** 2, axis=1))
        view_times, view_weights = [], []
        for distances in (d_log, d_raw):
            k = min(7, len(indices))
            cutoff = float(np.partition(distances, k - 1)[k - 1])
            keep = distances <= cutoff
            selected = distances[keep]
            # Each view has total weight 1/2. Include all kth-distance ties,
            # making the result insensitive to row order in a rounded C.
            scale = max(float(np.median(selected)), 1e-8)
            weights = np.exp(-np.minimum((selected / scale) ** 2, 700))
            weights *= .5 / weights.sum()
            view_times.append(c_time[indices[keep]])
            view_weights.append(weights)
        window[i] = _max_window(np.concatenate(view_times), np.concatenate(view_weights), horizon)
    return np.clip(nearest, 1e-12, horizon), np.clip(window, 1e-12, horizon)


def predict_variants(*, c: pd.DataFrame, background: pd.DataFrame,
                     candidates: pd.DataFrame, challenge: pd.DataFrame,
                     cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """Return deterministic variants, each with 1-D ``mia`` and ``aia``.

    ``mia`` corresponds to candidates in their original order, and ``aia`` to
    challenge in its original order. Extra columns (including IDs and labels)
    are neither read nor retained. C/background contain the 13 public columns;
    challenge requires only the seven published QI columns.
    """
    _validate(c, PUBLIC_COLUMNS, "C")
    _validate(background, PUBLIC_COLUMNS, "background")
    _validate(candidates, PUBLIC_COLUMNS, "candidates", allow_empty=True)
    _validate(challenge, QI_COLUMNS, "challenge", allow_empty=True)
    c = _canonical(c)
    background = _canonical(background)
    candidates = candidates.loc[:, PUBLIC_COLUMNS].copy(deep=True).reset_index(drop=True)
    challenge = challenge.loc[:, QI_COLUMNS].copy(deep=True).reset_index(drop=True)
    horizon = float(cfg.get("onset", {}).get("horizon_years", 10.0))
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Invalid time horizon")

    from pwscup2026.attack.mia import overlap_mia
    from pwscup2026.attack.aia import aia_predict

    if len(candidates):
        official_mia = np.asarray(overlap_mia(c, background, candidates, cfg), dtype=float)
        # Reference KDE can theoretically underflow or overflow in extreme
        # synthetic data. Do not propagate NaN as a legal submission value.
        official_mia = np.nan_to_num(official_mia, nan=.5, posinf=1., neginf=0.)
        official_mia = np.clip(official_mia, 0., 1.)
        frames = (c, background, candidates)
        clinical_score = _local_density_score(c, background, candidates,
            _scaled(frames, background, CONTINUOUS, True), 7.)
        full_score = _local_density_score(c, background, candidates,
            _full_features(frames, background), 7.)
        clinical_mia = _rank01(clinical_score)
        full_mia = _rank01(full_score)
        ensemble_mia = _rank01(.40 * clinical_mia + .25 * full_mia +
                               .35 * _rank01(official_mia))
    else:
        official_mia = clinical_mia = full_mia = ensemble_mia = np.empty(0, dtype=float)
    if len(challenge):
        _, official_aia = aia_predict(c, challenge, background, cfg)
        nearest_aia, window_aia = _aia_predictions(c, background, challenge, horizon)
        official_aia = np.clip(np.asarray(official_aia, dtype=float), 1e-12, horizon)
    else:
        official_aia = nearest_aia = window_aia = np.empty(0, dtype=float)
    pairs = {
        "official": (official_mia, official_aia),
        "clinical_nearest": (clinical_mia, nearest_aia),
        "full_nearest": (full_mia, nearest_aia),
        "ensemble_nearest": (ensemble_mia, nearest_aia),
        "ensemble_window": (ensemble_mia, window_aia),
        "full_reference_aia": (full_mia, official_aia),
    }
    return {name: {"mia": mia.copy(), "aia": aia.copy()}
            for name, (mia, aia) in pairs.items()}


def predict(*, c: pd.DataFrame, background: pd.DataFrame,
            candidates: pd.DataFrame, challenge: pd.DataFrame, cfg: dict,
            variant: str="full_reference_aia") -> dict[str, np.ndarray]:
    """Return one variant. ``ensemble`` aliases ``ensemble_nearest``."""
    variant = "ensemble_nearest" if variant == "ensemble" else variant
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}; expected one of {VARIANTS}")
    return predict_variants(c=c, background=background, candidates=candidates,
                            challenge=challenge, cfg=cfg)[variant]
