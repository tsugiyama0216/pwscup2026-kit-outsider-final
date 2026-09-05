"""Mechanism-aware, normal-format AIA time decoders.

The routines intentionally operate only on public release/challenge/background
tables.  They return complete numeric prediction vectors to be overlaid on an
already chosen membership mask by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors


QI = ("age", "SBP", "TG", "HDL", "ALT")
BLOCK = ("sex", "prefecture")


def _midrank(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rank(method="average")
        .to_numpy(dtype=float)
        / (len(values) + 1.0)
    )


def _reference_quantile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source = np.sort(np.asarray(reference, dtype=float))
    query = np.asarray(values, dtype=float)
    left = np.searchsorted(source, query, side="left")
    right = np.searchsorted(source, query, side="right")
    return (0.5 * (left + right) + 0.5) / (len(source) + 1.0)


def _rank_intervals(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return empirical rank lower/mid/upper points, retaining tie widths."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    unique, first, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    del unique
    lower_sorted = np.empty(len(array), dtype=float)
    upper_sorted = np.empty(len(array), dtype=float)
    for start, count in zip(first, counts, strict=True):
        stop = int(start + count)
        lower_sorted[start:stop] = (float(start) + 0.5) / (len(array) + 1.0)
        upper_sorted[start:stop] = (float(stop) - 0.5) / (len(array) + 1.0)
    lower = np.empty(len(array), dtype=float)
    upper = np.empty(len(array), dtype=float)
    lower[order] = lower_sorted
    upper[order] = upper_sorted
    return lower, 0.5 * (lower + upper), upper


def _interval_distance(
    query: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    # Shape: query rows x release rows x dimensions.  With 326x1049x5 this is
    # small enough for deterministic exact distances.
    q = query[:, None, :]
    low = lower[None, :, :]
    high = upper[None, :, :]
    below = np.maximum(low - q, 0.0)
    above = np.maximum(q - high, 0.0)
    return np.sqrt(np.mean((below + above) ** 2, axis=2))


def _block_pools(released: pd.DataFrame, challenge: pd.DataFrame) -> list[np.ndarray]:
    rsex = released["sex"].to_numpy()
    rpref = released["prefecture"].astype(str).to_numpy()
    pools: list[np.ndarray] = []
    for sex, pref in zip(
        challenge["sex"].to_numpy(), challenge["prefecture"].astype(str).to_numpy(), strict=True
    ):
        pool = np.flatnonzero((rsex == sex) & (rpref == pref))
        if not len(pool):
            pool = np.flatnonzero(rsex == sex)
        if not len(pool):
            pool = np.arange(len(released), dtype=int)
        pools.append(pool)
    return pools


@dataclass(frozen=True)
class LinkGeometry:
    nearest: np.ndarray
    distance: np.ndarray
    second_distance: np.ndarray
    margin: np.ndarray
    relative_margin: np.ndarray
    mutual: np.ndarray
    donor_event: np.ndarray
    donor_time: np.ndarray
    likelihood: np.ndarray


def interval_link_geometry(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    query_cdf: str = "challenge",
) -> LinkGeometry:
    """Estimate piecewise-monotone links while respecting rounded intervals.

    `query_cdf='challenge'` is invariant to target-specific monotone margins.
    `query_cdf='background'` anchors original-scale query ranks to public A_bg.
    Release ties become rank intervals, rather than arbitrary midpoint points.
    """
    for label, frame, required in (
        ("released", released, (*QI, *BLOCK, "time", "onset", "death")),
        ("challenge", challenge, (*QI, *BLOCK)),
        ("background", background, QI),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
    if query_cdf not in {"challenge", "background"}:
        raise ValueError("query_cdf must be challenge or background")

    lows: list[np.ndarray] = []
    mids: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    bg_points: list[np.ndarray] = []
    for column in QI:
        low, mid, high = _rank_intervals(released[column].to_numpy(dtype=float))
        lows.append(low)
        mids.append(mid)
        highs.append(high)
        if query_cdf == "challenge":
            queries.append(_midrank(challenge[column].to_numpy(dtype=float)))
        else:
            queries.append(
                _reference_quantile(
                    challenge[column].to_numpy(dtype=float),
                    background[column].to_numpy(dtype=float),
                )
            )
        bg_points.append(
            _reference_quantile(
                background[column].to_numpy(dtype=float),
                background[column].to_numpy(dtype=float),
            )
        )
    lower = np.column_stack(lows)
    midpoint = np.column_stack(mids)
    upper = np.column_stack(highs)
    query = np.column_stack(queries)
    bg = np.column_stack(bg_points)

    pair = _interval_distance(query, lower, upper)
    pools = _block_pools(released, challenge)
    nearest = np.empty(len(challenge), dtype=int)
    first = np.empty(len(challenge), dtype=float)
    second = np.empty(len(challenge), dtype=float)
    mutual = np.zeros(len(challenge), dtype=bool)
    for row, pool in enumerate(pools):
        local = pair[row, pool]
        order = np.argsort(local, kind="mergesort")
        nearest[row] = int(pool[order[0]])
        first[row] = float(local[order[0]])
        second[row] = float(local[order[1]]) if len(order) > 1 else np.inf

    # Reverse nearest is evaluated within each exact public block.  Ties are
    # deterministic; a tiny tolerance admits interval ties created by rounding.
    for row, donor in enumerate(nearest):
        candidate_rows = np.array(
            [i for i, pool in enumerate(pools) if donor in pool], dtype=int
        )
        if not len(candidate_rows):
            continue
        distances = pair[candidate_rows, donor]
        best = float(np.min(distances))
        mutual[row] = first[row] <= best + 1e-12

    margin = second - first
    relative = np.ones(len(first), dtype=float)
    finite_second = np.isfinite(second)
    relative[finite_second] = margin[finite_second] / (
        second[finite_second] + first[finite_second] + 1e-9
    )
    event = (
        released["onset"].to_numpy(dtype=int)[nearest]
        + released["death"].to_numpy(dtype=int)[nearest]
    ) > 0
    donor_time = released["time"].to_numpy(dtype=float)[nearest]

    # Density ratio supplies an independent target-vs-background gate.  It is
    # rank-based and therefore compatible with monotone marginal transforms.
    k = min(8, len(released), len(background))
    release_nn = NearestNeighbors(n_neighbors=k).fit(midpoint)
    background_nn = NearestNeighbors(n_neighbors=k).fit(bg)
    rd = release_nn.kneighbors(query, return_distance=True)[0][:, -1]
    bd = background_nn.kneighbors(query, return_distance=True)[0][:, -1]
    evidence = len(QI) * np.log((bd + 1e-6) / (rd + 1e-6))
    likelihood = pd.Series(evidence).rank(method="average", pct=True).to_numpy(dtype=float)
    return LinkGeometry(
        nearest=nearest,
        distance=first,
        second_distance=second,
        margin=margin,
        relative_margin=relative,
        mutual=mutual,
        donor_event=event,
        donor_time=donor_time,
        likelihood=likelihood,
    )


def interval_event_links(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    query_cdf: str = "challenge",
    margin_quantile: float = 0.50,
    likelihood_quantile: float = 0.0,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    geometry = interval_link_geometry(
        released, challenge, background, query_cdf=query_cdf
    )
    if not 0.0 <= margin_quantile <= 1.0:
        raise ValueError("margin_quantile outside [0,1]")
    if not 0.0 <= likelihood_quantile <= 1.0:
        raise ValueError("likelihood_quantile outside [0,1]")
    margin_cut = float(np.quantile(geometry.relative_margin, margin_quantile))
    gate = (
        geometry.mutual
        & geometry.donor_event
        & (geometry.relative_margin >= margin_cut)
        & (geometry.likelihood >= likelihood_quantile)
    )
    linked = np.full(len(challenge), horizon_time, dtype=float)
    linked[gate] = geometry.donor_time[gate]
    details = {
        "query_cdf": query_cdf,
        "margin_quantile": margin_quantile,
        "likelihood_quantile": likelihood_quantile,
        "mutual": int(geometry.mutual.sum()),
        "nearest_event": int(geometry.donor_event.sum()),
        "trusted_event_links": int(gate.sum()),
        "median_distance_trusted": float(np.median(geometry.distance[gate]))
        if gate.any()
        else None,
    }
    return linked, gate, details


def interval_links_from_geometry(
    geometry: LinkGeometry,
    *,
    query_cdf: str,
    margin_quantile: float,
    likelihood_quantile: float,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply gates to an already computed geometry for efficient screening."""
    if not 0.0 <= margin_quantile <= 1.0:
        raise ValueError("margin_quantile outside [0,1]")
    if not 0.0 <= likelihood_quantile <= 1.0:
        raise ValueError("likelihood_quantile outside [0,1]")
    margin_cut = float(np.quantile(geometry.relative_margin, margin_quantile))
    gate = (
        geometry.mutual
        & geometry.donor_event
        & (geometry.relative_margin >= margin_cut)
        & (geometry.likelihood >= likelihood_quantile)
    )
    linked = np.full(len(gate), horizon_time, dtype=float)
    linked[gate] = geometry.donor_time[gate]
    details = {
        "query_cdf": query_cdf,
        "margin_quantile": margin_quantile,
        "likelihood_quantile": likelihood_quantile,
        "mutual": int(geometry.mutual.sum()),
        "nearest_event": int(geometry.donor_event.sum()),
        "trusted_event_links": int(gate.sum()),
        "median_distance_trusted": float(np.median(geometry.distance[gate]))
        if gate.any()
        else None,
    }
    return linked, gate, details


def conditional_hazard_prediction(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    seed: int,
    event_probability_quantile: float = 0.90,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Predict high-confidence event windows using a discrete event hazard proxy.

    This is deliberately a smooth conditional model, not a record linker.  It
    is useful as a negative/control comparison for direct mechanism links.
    """
    geometry = interval_link_geometry(
        released, challenge, background, query_cdf="challenge"
    )
    release_x = np.column_stack([_rank_intervals(released[c].to_numpy(dtype=float))[1] for c in QI])
    challenge_x = np.column_stack([_midrank(challenge[c].to_numpy(dtype=float)) for c in QI])
    event = (
        released["onset"].to_numpy(dtype=int)
        + released["death"].to_numpy(dtype=int)
    ) > 0
    classes = np.unique(event.astype(int))
    if len(classes) == 1:
        probability = np.full(len(challenge), float(classes[0]), dtype=float)
    else:
        classifier = ExtraTreesClassifier(
            n_estimators=192,
            min_samples_leaf=12,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ).fit(release_x, event.astype(int))
        probability = classifier.predict_proba(challenge_x)[:, 1]
    cutoff = float(np.quantile(probability, event_probability_quantile))
    # Quantile comparisons can activate every row when probabilities tie.
    # Use a deterministic exact top-k gate instead.
    gate_count = int(np.ceil((1.0 - event_probability_quantile) * len(challenge)))
    gate_count = int(np.clip(gate_count, 0, len(challenge)))
    gate = np.zeros(len(challenge), dtype=bool)
    if gate_count and event.any():
        if "challenge_row_id" in challenge:
            tie_key = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        else:
            tie_key = np.arange(len(challenge), dtype=np.int64)
        order = np.lexsort((tie_key, -probability))
        gate[order[:gate_count]] = True
    predicted = np.full(len(challenge), horizon_time, dtype=float)
    if event.any():
        regressor = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=seed + 1,
            n_jobs=1,
        ).fit(release_x[event], released.loc[event, "time"].to_numpy(dtype=float))
        predicted[gate] = np.clip(regressor.predict(challenge_x[gate]), 0.0, horizon_time)
    return predicted, gate, {
        "event_probability_quantile": event_probability_quantile,
        "gate_rows": int(gate.sum()),
        "release_event_rows": int(event.sum()),
        "median_gate_probability": float(np.median(probability[gate])) if gate.any() else None,
        "probability_unique": int(np.unique(probability).size),
        "one_class_release": bool(len(classes) == 1),
        "nearest_link_event_in_gate": int(np.count_nonzero(gate & geometry.donor_event)),
    }
