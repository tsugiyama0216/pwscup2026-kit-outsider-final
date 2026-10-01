"""Pure, deterministic AIA with observable regional-support routing.

Queries use only the seven official QI attributes. Reference C time is the
prediction target; IDs, labels, team numbers and truth are never used. I/O and
selection loading belong to the caller, not this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
CONTINUOUS = ["age", "SBP", "TG", "HDL", "ALT"]
PUBLIC = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
          "smoking", "FPG", "time", "onset", "death"]
VARIANTS = ("v1_reference", "support8", "support20", "support40", "guarded40", "soft40")
SETTINGS = {"support8": 8, "support20": 20, "support40": 40,
            "guarded40": 40, "soft40": 40}
DISTANCE_RATIO = 1.5
SOFT_PREFECTURE_PENALTY = .15


def _frame(frame, required, label):
    if not set(required) <= set(frame.columns):
        raise ValueError(f"{label}: missing public feature columns")
    chosen = [name for name in PUBLIC if name in frame.columns]
    result = frame[chosen].copy(deep=True)
    numeric = [name for name in required if name != "prefecture"]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: non-finite numeric feature")
    if result["prefecture"].isna().any():
        raise ValueError(f"{label}: missing prefecture")
    return result


def _canonical(frame):
    return frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)


def _mean_nearest(times, squared_distance):
    selected = times[squared_distance <= squared_distance.min()]
    return float(np.mean(np.sort(selected)))


def _run(c, background, challenge, cfg, requested=None):
    from pwscup2026.attack.aia import aia_predict
    requested = VARIANTS if requested is None else tuple(requested)
    c = _canonical(_frame(c, QI + ["time", "onset"], "C"))
    background = _canonical(_frame(background, QI, "background"))
    q = _frame(challenge, QI, "challenge")[QI].reset_index(drop=True)
    if not len(c) or not len(background):
        raise ValueError("C and background must be nonempty")
    if list(cfg["aia"]["qi_continuous"]) != CONTINUOUS or list(cfg["aia"]["block_keys"]) != ["sex", "prefecture"]:
        raise ValueError("This component requires the official five continuous QI and sex/prefecture blocks")
    if not len(q):
        return {name: np.empty(0, dtype=float) for name in requested}, {
            "queries": 0, "routes": {name: {"reference": 0, "sex_only": 0, "soft_prefecture": 0} for name in requested}}
    # Same pandas sample SD as the fixed official/v1 AIA.
    scale = background[CONTINUOUS].std().to_numpy(dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    cv = c[CONTINUOUS].to_numpy(dtype=float) / scale
    qv = q[CONTINUOUS].to_numpy(dtype=float) / scale
    times = c["time"].to_numpy(dtype=float)
    sexes, prefs = c["sex"].to_numpy(), c["prefecture"].to_numpy()
    _, reference = aia_predict(c, q, background, cfg)
    results = {name: np.asarray(reference, dtype=float).copy() for name in requested}
    route_counts = {name: {"reference": len(q), "sex_only": 0, "soft_prefecture": 0} for name in requested}
    supports, sex_supports, ratios = [], [], []
    empty_groups = 0
    for i, query in enumerate(q.itertuples(index=False)):
        sex_indices = np.flatnonzero(sexes == query.sex)
        if not len(sex_indices):
            sex_indices = np.arange(len(c))
        block = sex_indices[prefs[sex_indices] == query.prefecture]
        support = len(block)
        supports.append(support)
        sex_supports.append(len(sex_indices))
        squared = np.mean((cv[sex_indices] - qv[i]) ** 2, axis=1)
        best_sex = float(squared.min())
        sex_prediction = _mean_nearest(times[sex_indices], squared)
        if support:
            block_squared = np.mean((cv[block] - qv[i]) ** 2, axis=1)
            best_block = float(block_squared.min())
            ratio = float(np.sqrt(best_block / max(best_sex, 1e-24))) if best_block else 1.
        else:
            empty_groups += 1
            ratio = 1.
        ratios.append(ratio)
        for name, minimum in SETTINGS.items():
            if name not in requested:
                continue
            route = support < minimum
            if name == "guarded40":
                route = route and (support == 0 or ratio > DISTANCE_RATIO)
            if not route:
                continue
            route_counts[name]["reference"] -= 1
            if name == "soft40" and support:
                penalty = (prefs[sex_indices] != query.prefecture) * SOFT_PREFECTURE_PENALTY ** 2
                results[name][i] = _mean_nearest(times[sex_indices], squared + penalty)
                route_counts[name]["soft_prefecture"] += 1
            else:
                results[name][i] = sex_prediction
                route_counts[name]["sex_only"] += 1
    if not all(np.isfinite(values).all() for values in results.values()):
        raise ValueError("AIA output must be finite")
    return results, {
        "queries": len(q), "empty_reference_blocks": empty_groups,
        "sex_prefecture_support_min": int(np.min(supports)),
        "sex_prefecture_support_median": float(np.median(supports)),
        "sex_prefecture_support_max": int(np.max(supports)),
        "sex_only_support_min": int(np.min(sex_supports)),
        "sex_only_support_median": float(np.median(sex_supports)),
        "distance_penalty_ratio_median": float(np.median(ratios)),
        "routes": route_counts,
    }


def predict_variants(c, background, challenge, cfg):
    """Return name -> time array in input query order for all six fixed options."""
    return _run(c, background, challenge, cfg)[0]


def predict(c, background, challenge, cfg, variant="v1_reference"):
    """Return one explicitly selected option. Selection is loaded by the caller."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown AIA variant: {variant}")
    return _run(c, background, challenge, cfg, requested=(variant,))[0][variant]


def diagnose(c, background, challenge, cfg):
    """Return aggregate support/routing counts, with no IDs or patient values."""
    return _run(c, background, challenge, cfg)[1]
