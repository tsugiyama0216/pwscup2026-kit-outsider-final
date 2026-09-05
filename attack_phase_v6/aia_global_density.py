"""Global density-ratio AIA selector for transformed public releases.

The public challenges match controls to members on coarse demographic blocks,
but several production releases visibly destroy the corresponding row links.
This selector therefore compares laboratory-value density globally, then still
enforces the published half-per-visible-stratum member construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from attack_phase_v5.aia_structured_ensemble import (
    BLOCK_COLUMNS,
    LOG_COLUMNS,
    OFFICIAL_COLUMNS,
    QI_COLUMNS,
    _aligned_expert,
    _marginal_midrank_matrix,
    _require_columns,
    filter_linkable_releases,
    strict_rank,
    stratified_half_selection,
)


DEFAULT_NEIGHBORS = (1, 3, 8)
NO_HDL_COLUMNS = ("age", "SBP", "TG", "ALT")


def _background_scaled_matrix(
    frame: pd.DataFrame,
    background: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    """Embed values using robust scales learned only from public background."""
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    reference = background.loc[:, columns].to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("global density columns must be finite")
    for index, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
            reference[:, index] = np.log1p(
                np.clip(reference[:, index], 0.0, None)
            )
    center = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(reference, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - center) / scale


def global_density_ratio_score(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    columns: tuple[str, ...] = QI_COLUMNS,
    neighbors: tuple[int, ...] = DEFAULT_NEIGHBORS,
) -> np.ndarray:
    """Rank challenge rows by released-versus-background kNN density ratio."""
    _require_columns(released, columns, "released table")
    _require_columns(challenge, ("challenge_row_id", *columns), "challenge")
    _require_columns(background, columns, "background")
    if not len(released) or not len(challenge) or not len(background):
        raise ValueError("released table, challenge, and background must be non-empty")
    if not neighbors or min(neighbors) < 1:
        raise ValueError("density neighbors must be positive")
    maximum = max(neighbors)
    if len(released) < maximum or len(background) < maximum:
        raise ValueError("density reference tables have too few rows")

    release_x = _background_scaled_matrix(released, background, columns)
    challenge_x = _background_scaled_matrix(challenge, background, columns)
    background_x = _background_scaled_matrix(background, background, columns)
    release_distance = NearestNeighbors(n_neighbors=maximum, n_jobs=1).fit(
        release_x
    ).kneighbors(challenge_x, return_distance=True)[0]
    background_distance = NearestNeighbors(n_neighbors=maximum, n_jobs=1).fit(
        background_x
    ).kneighbors(challenge_x, return_distance=True)[0]
    components = [
        len(columns)
        * np.log(
            (background_distance[:, value - 1] + 1e-6)
            / (release_distance[:, value - 1] + 1e-6)
        )
        for value in neighbors
    ]
    raw = np.mean(np.column_stack(components), axis=1)
    return strict_rank(raw, challenge["challenge_row_id"].to_numpy(dtype=np.int64))


def sex_copula_proximity_score(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    columns: tuple[str, ...] = NO_HDL_COLUMNS,
) -> np.ndarray:
    """Rank separate-margin proximity within sex while ignoring prefecture."""
    _require_columns(released, (*columns, "sex"), "released table")
    _require_columns(challenge, ("challenge_row_id", *columns, "sex"), "challenge")
    if not len(released) or not len(challenge):
        raise ValueError("released table and challenge must be non-empty")
    release_x = _marginal_midrank_matrix(released, columns)
    challenge_x = _marginal_midrank_matrix(challenge, columns)
    release_sex = released["sex"].to_numpy()
    nearest = np.empty(len(challenge), dtype=float)
    for sex, indices in challenge.groupby("sex", sort=False).indices.items():
        rows = np.asarray(indices, dtype=int)
        pool = np.flatnonzero(release_sex == sex)
        if not len(pool):
            pool = np.arange(len(released), dtype=int)
        nearest[rows] = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(
            release_x[pool]
        ).kneighbors(challenge_x[rows], return_distance=True)[0][:, 0]
    return strict_rank(
        -nearest, challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    )


def build_global_density_horizon(
    releases: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    background: pd.DataFrame,
    *,
    columns: tuple[str, ...] = QI_COLUMNS,
    member_count: int = 163,
    abstain_time: float = 11.0,
    horizon_time: float = 10.0,
    invert: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a complete AIA frame using global density and horizon time.

    ``invert`` is reserved for a release that is demonstrably background-like:
    there the ordinary release-density direction identifies controls, so the
    complementary rank is the member hypothesis.
    """
    if set(releases) != set(challenges) or not releases:
        raise ValueError("release and challenge target sets must be equal and non-empty")
    _require_columns(background, columns, "background")
    linkable, filter_diagnostics = filter_linkable_releases(releases)
    parts: list[pd.DataFrame] = []
    target_diagnostics: dict[str, Any] = {}
    background_keys = pd.MultiIndex.from_frame(background.loc[:, columns])
    for target_id in sorted(releases, key=int):
        challenge = challenges[target_id].reset_index(drop=True)
        raw_background_match = pd.MultiIndex.from_frame(
            releases[target_id].loc[:, columns]
        ).isin(background_keys)
        linkable_background_match = pd.MultiIndex.from_frame(
            linkable[target_id].loc[:, columns]
        ).isin(background_keys)
        score = global_density_ratio_score(
            linkable[target_id], challenge, background, columns=columns
        )
        if invert:
            score = -score
        selected, detail = stratified_half_selection(
            challenge, score, member_count=member_count
        )
        time_hat = np.full(len(challenge), abstain_time, dtype=float)
        time_hat[selected] = horizon_time
        parts.append(
            pd.DataFrame(
                {
                    "target_k": int(target_id),
                    "challenge_row_id": challenge["challenge_row_id"].to_numpy(
                        dtype=np.int64
                    ),
                    "time_hat": time_hat,
                }
            )
        )
        target_diagnostics[target_id] = {
            **detail.__dict__,
            "score_min": float(np.min(score)),
            "score_median": float(np.median(score)),
            "score_max": float(np.max(score)),
            "raw_release_background_exact_rows": int(
                np.count_nonzero(raw_background_match)
            ),
            "raw_release_background_exact_fraction": float(
                np.mean(raw_background_match)
            ),
            "linkable_release_background_exact_rows": int(
                np.count_nonzero(linkable_background_match)
            ),
            "linkable_release_background_exact_fraction": float(
                np.mean(linkable_background_match)
            ),
        }
    output = (
        pd.concat(parts, ignore_index=True)
        .loc[:, OFFICIAL_COLUMNS]
        .sort_values(["target_k", "challenge_row_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return output, {
        "method": (
            "inverted_global_knn_density_ratio_horizon"
            if invert
            else "global_knn_density_ratio_horizon"
        ),
        "columns": list(columns),
        "neighbors": list(DEFAULT_NEIGHBORS),
        "inverted": invert,
        "release_filter": filter_diagnostics,
        "targets": target_diagnostics,
    }


def build_sex_consensus_horizon(
    v4: pd.DataFrame,
    independent: pd.DataFrame,
    releases: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    member_count: int = 163,
    abstain_time: float = 11.0,
    horizon_time: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a prefecture-relaxed selector anchored to both known experts."""
    if set(releases) != set(challenges) or not releases:
        raise ValueError("release and challenge target sets must be equal and non-empty")
    linkable, filter_diagnostics = filter_linkable_releases(releases)
    parts: list[pd.DataFrame] = []
    target_diagnostics: dict[str, Any] = {}
    for target_id in sorted(releases, key=int):
        challenge = challenges[target_id].reset_index(drop=True)
        challenge_ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        v4_time = _aligned_expert(v4, target_id, challenge_ids, "v4")
        independent_time = _aligned_expert(
            independent, target_id, challenge_ids, "independent"
        )
        active_v4 = ~np.isclose(v4_time, abstain_time)
        active_independent = ~np.isclose(independent_time, abstain_time)
        proximity = sex_copula_proximity_score(linkable[target_id], challenge)
        score = 1.10 * active_v4 + 0.90 * active_independent + 0.55 * proximity
        selected, detail = stratified_half_selection(
            challenge, score, member_count=member_count
        )
        time_hat = np.full(len(challenge), abstain_time, dtype=float)
        time_hat[selected] = horizon_time
        parts.append(
            pd.DataFrame(
                {
                    "target_k": int(target_id),
                    "challenge_row_id": challenge_ids,
                    "time_hat": time_hat,
                }
            )
        )
        target_diagnostics[target_id] = {
            **detail.__dict__,
            "selection_changes_from_v4": int(
                np.count_nonzero(selected != active_v4)
            ),
            "selection_changes_from_independent": int(
                np.count_nonzero(selected != active_independent)
            ),
        }
    output = (
        pd.concat(parts, ignore_index=True)
        .loc[:, OFFICIAL_COLUMNS]
        .sort_values(["target_k", "challenge_row_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return output, {
        "method": "sex_only_nohdl_copula_consensus_horizon",
        "columns": list(NO_HDL_COLUMNS),
        "weights": {"v4": 1.10, "independent": 0.90, "proximity": 0.55},
        "release_filter": filter_diagnostics,
        "targets": target_diagnostics,
    }
