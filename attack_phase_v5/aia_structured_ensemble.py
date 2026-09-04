"""Public-structure AIA ensemble used by the version-5 attack.

The preliminary AIA challenge contains 163 members and 163 matched controls.
Controls are matched to members on sex, prefecture, and five-year age band.
Consequently, almost every visible matching stratum contains exactly half
members.  This module applies that public construction constraint to two
independent attack candidates and never reads organizer-only truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


QI_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT")
BLOCK_COLUMNS = ("sex", "prefecture")
OFFICIAL_COLUMNS = ("target_k", "challenge_row_id", "time_hat")
PROXIMITY_VIEWS = (
    ("age", "SBP", "TG", "HDL", "ALT"),
    ("age", "SBP", "HDL"),
    ("SBP", "TG", "HDL", "ALT"),
)
LOG_COLUMNS = frozenset(("TG", "ALT"))
RELEASE_COLUMNS = (
    "age",
    "sex",
    "prefecture",
    "BMI",
    "SBP",
    "TG",
    "HDL",
    "ALT",
    "smoking",
    "FPG",
    "time",
    "onset",
    "death",
)


@dataclass(frozen=True)
class SelectionDiagnostics:
    rows: int
    strata: int
    odd_strata: int
    extra_ceil_strata: int
    selected_rows: int
    exact_half_strata: int


def strict_rank(values: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    """Return deterministic ascending ranks in (0, 1), with larger being better."""
    values = np.asarray(values, dtype=float)
    tie_key = np.asarray(tie_key)
    if values.ndim != 1 or values.shape != tie_key.shape:
        raise ValueError("rank inputs must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("rank values must be finite")
    order = np.lexsort((tie_key, values))
    ranked = np.empty(len(values), dtype=float)
    ranked[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranked / (len(values) + 1.0)


def age_band(values: Iterable[float]) -> np.ndarray:
    """Apply the published five-year band used to construct AIA controls."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("age contains non-finite values")
    return np.floor((array - 40.0) / 5.0).astype(np.int64)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _aligned_expert(
    frame: pd.DataFrame,
    target_id: str,
    challenge_ids: np.ndarray,
    label: str,
) -> np.ndarray:
    if list(frame.columns) != list(OFFICIAL_COLUMNS):
        raise ValueError(f"{label} must have the official AIA columns")
    if frame.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError(f"{label} has duplicate AIA keys")
    block = frame.loc[
        frame["target_k"].astype(int).eq(int(target_id)),
        ["challenge_row_id", "time_hat"],
    ].copy()
    if len(block) != len(challenge_ids):
        raise ValueError(f"{label} target {target_id} row count differs from challenge")
    block = block.set_index(block["challenge_row_id"].astype(np.int64))
    if set(block.index) != set(np.asarray(challenge_ids, dtype=np.int64)):
        raise ValueError(f"{label} target {target_id} challenge ids differ")
    values = block.loc[np.asarray(challenge_ids, dtype=np.int64), "time_hat"].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"{label} target {target_id} has invalid times")
    return values


def _scaled_matrix(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    release_values = released.loc[:, columns].to_numpy(dtype=float, copy=True)
    challenge_values = challenge.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in LOG_COLUMNS:
            release_values[:, index] = np.log1p(np.clip(release_values[:, index], 0.0, None))
            challenge_values[:, index] = np.log1p(
                np.clip(challenge_values[:, index], 0.0, None)
            )
    combined = np.vstack((release_values, challenge_values))
    center = np.median(combined, axis=0)
    q25, q75 = np.quantile(combined, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(combined, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    dimension = np.sqrt(len(columns))
    return (release_values - center) / scale / dimension, (
        challenge_values - center
    ) / scale / dimension


def proximity_score(released: pd.DataFrame, challenge: pd.DataFrame) -> np.ndarray:
    """Rank multiview challenge-to-release proximity within public blocks."""
    _require_columns(released, (*QI_COLUMNS, *BLOCK_COLUMNS), "released table")
    _require_columns(
        challenge,
        ("challenge_row_id", *QI_COLUMNS, *BLOCK_COLUMNS),
        "challenge",
    )
    if not len(released) or not len(challenge):
        raise ValueError("released table and challenge must be non-empty")
    challenge_ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    view_ranks: list[np.ndarray] = []
    release_sex = released["sex"].to_numpy()
    release_pref = released["prefecture"].astype(str).to_numpy()
    challenge_blocks = pd.DataFrame(
        {
            "sex": challenge["sex"].to_numpy(),
            "prefecture": challenge["prefecture"].astype(str).to_numpy(),
        }
    )

    for columns in PROXIMITY_VIEWS:
        release_x, challenge_x = _scaled_matrix(released, challenge, columns)
        nearest = np.empty(len(challenge), dtype=float)
        for raw_key, indices in challenge_blocks.groupby(
            ["sex", "prefecture"], sort=False
        ).indices.items():
            sex, prefecture = raw_key if isinstance(raw_key, tuple) else (raw_key, "")
            rows = np.asarray(indices, dtype=int)
            pool = np.flatnonzero(
                (release_sex == sex) & (release_pref == str(prefecture))
            )
            if not len(pool):
                pool = np.flatnonzero(release_sex == sex)
            if not len(pool):
                pool = np.arange(len(released), dtype=int)
            model = NearestNeighbors(n_neighbors=1, metric="euclidean")
            model.fit(release_x[pool])
            distance, _ = model.kneighbors(challenge_x[rows])
            nearest[rows] = distance[:, 0]
        view_ranks.append(strict_rank(-nearest, challenge_ids))
    return np.mean(np.column_stack(view_ranks), axis=1)


def filter_linkable_releases(
    releases: dict[str, pd.DataFrame],
    *,
    maximum_cross_target_fraction: float = 0.50,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Remove exact rows that occur in another target when enough rows remain.

    Preliminary AIA members come from a target's non-shared block.  A complete
    row copied verbatim into another target therefore cannot be direct evidence
    for such a member.  A high-fraction guard makes the operation inert in raw
    shadow worlds, where every target intentionally receives the same source.
    Internal duplicate rows are collapsed because repeated synthetic rows do
    not provide additional nearest-neighbour evidence.
    """
    if not 0.0 <= maximum_cross_target_fraction <= 1.0:
        raise ValueError("maximum cross-target fraction must be in [0, 1]")
    if not releases:
        raise ValueError("releases are empty")
    for target_id, frame in releases.items():
        _require_columns(frame, RELEASE_COLUMNS, f"release {target_id}")
    hashes = {
        target_id: pd.util.hash_pandas_object(
            frame.loc[:, RELEASE_COLUMNS], index=False
        ).to_numpy(dtype=np.uint64)
        for target_id, frame in releases.items()
    }
    unique_hashes = {
        target_id: set(values.tolist()) for target_id, values in hashes.items()
    }
    filtered: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, Any] = {}
    for target_id, frame in releases.items():
        other_hashes: set[int] = set()
        for other_id, values in unique_hashes.items():
            if other_id != target_id:
                other_hashes.update(values)
        cross = np.fromiter(
            (int(value) in other_hashes for value in hashes[target_id]),
            dtype=bool,
            count=len(frame),
        )
        cross_count = int(cross.sum())
        cross_fraction = cross_count / len(frame)
        apply_cross_filter = (
            cross_count > 0
            and cross_fraction <= maximum_cross_target_fraction
            and int((~cross).sum()) >= 25
        )
        selected = frame.loc[~cross] if apply_cross_filter else frame
        deduplicated = selected.drop_duplicates(list(RELEASE_COLUMNS), keep="first")
        if len(deduplicated) < 25:
            deduplicated = selected
        filtered[target_id] = deduplicated.reset_index(drop=True)
        diagnostics[target_id] = {
            "input_rows": len(frame),
            "cross_target_exact_rows": cross_count,
            "cross_target_exact_fraction": cross_fraction,
            "cross_target_filter_applied": apply_cross_filter,
            "internal_duplicate_occurrences": int(
                frame.duplicated(list(RELEASE_COLUMNS)).sum()
            ),
            "linkable_rows": len(filtered[target_id]),
        }
    return filtered, diagnostics


def stratified_half_selection(
    challenge: pd.DataFrame,
    score: np.ndarray,
    *,
    member_count: int = 163,
) -> tuple[np.ndarray, SelectionDiagnostics]:
    """Select floor/ceil half per visible stratum and exactly member_count rows."""
    _require_columns(
        challenge,
        ("challenge_row_id", "age", *BLOCK_COLUMNS),
        "challenge",
    )
    score = np.asarray(score, dtype=float)
    if score.shape != (len(challenge),) or not np.isfinite(score).all():
        raise ValueError("selection scores must be finite and aligned")
    if not 0 <= member_count <= len(challenge):
        raise ValueError("member_count is outside the challenge")

    work = pd.DataFrame(
        {
            "row": np.arange(len(challenge), dtype=int),
            "sex": challenge["sex"].to_numpy(),
            "prefecture": challenge["prefecture"].astype(str).to_numpy(),
            "age_band": age_band(challenge["age"]),
        }
    )
    challenge_ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    floor_total = 0
    odd_margins: list[tuple[float, int, int]] = []
    for group_index, indices in enumerate(
        work.groupby(["sex", "prefecture", "age_band"], sort=True).indices.values()
    ):
        rows = np.asarray(indices, dtype=int)
        order = rows[np.lexsort((challenge_ids[rows], -score[rows]))]
        groups.append((rows, order))
        quota = len(rows) // 2
        floor_total += quota
        if len(rows) % 2:
            marginal_row = int(order[quota])
            odd_margins.append((float(score[marginal_row]), -int(challenge_ids[marginal_row]), group_index))

    extras = member_count - floor_total
    if extras < 0 or extras > len(odd_margins):
        raise ValueError(
            "visible strata cannot satisfy the requested member count with half quotas"
        )
    ceil_groups = {
        item[2]
        for item in sorted(odd_margins, reverse=True)[:extras]
    }
    selected = np.zeros(len(challenge), dtype=bool)
    exact_half = 0
    for group_index, (rows, order) in enumerate(groups):
        quota = len(rows) // 2 + int(group_index in ceil_groups)
        selected[order[:quota]] = True
        if len(rows) % 2 == 0 and quota * 2 == len(rows):
            exact_half += 1
    if int(selected.sum()) != member_count:
        raise RuntimeError("stratified selection produced the wrong cardinality")
    return selected, SelectionDiagnostics(
        rows=len(challenge),
        strata=len(groups),
        odd_strata=len(odd_margins),
        extra_ceil_strata=extras,
        selected_rows=int(selected.sum()),
        exact_half_strata=exact_half,
    )


def _decode_times(
    selected: np.ndarray,
    v4_time: np.ndarray,
    independent_time: np.ndarray,
    *,
    decoder: str,
    abstain_time: float,
    horizon_time: float,
) -> np.ndarray:
    active_v4 = ~np.isclose(v4_time, abstain_time)
    active_independent = ~np.isclose(independent_time, abstain_time)
    output = np.full(len(selected), abstain_time, dtype=float)
    if decoder == "horizon":
        output[selected] = horizon_time
        return output

    if decoder == "v4_fallback":
        chosen = np.where(
            active_v4,
            v4_time,
            np.where(active_independent, independent_time, horizon_time),
        )
    elif decoder == "independent_fallback":
        chosen = np.where(
            active_independent,
            independent_time,
            np.where(active_v4, v4_time, horizon_time),
        )
    elif decoder == "consensus_event":
        v4_event = active_v4 & (v4_time < horizon_time - 0.5)
        independent_event = active_independent & (
            independent_time < horizon_time - 0.5
        )
        agreed_event = (
            v4_event
            & independent_event
            & (np.abs(v4_time - independent_time) <= 1.0)
        )
        chosen = np.full(len(selected), horizon_time, dtype=float)
        chosen[agreed_event] = 0.5 * (
            v4_time[agreed_event] + independent_time[agreed_event]
        )
    else:
        raise ValueError(f"unknown AIA decoder: {decoder}")
    output[selected] = chosen[selected]
    return output


def build_aia_v5_variants(
    v4: pd.DataFrame,
    independent: pd.DataFrame,
    releases: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    member_count: int = 163,
    abstain_time: float = 11.0,
    horizon_time: float = 10.0,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build public-structure variants from the v4 and independent experts."""
    if set(releases) != set(challenges) or not releases:
        raise ValueError("release and challenge target sets must be equal and non-empty")
    score_specs = {
        "stratum_v4": (1.00, 0.00, 0.30),
        "stratum_independent": (0.00, 1.00, 0.30),
        "stratum_proximity": (0.00, 0.00, 1.00),
        "consensus_equal": (1.00, 1.00, 0.35),
        "consensus_v4_lean": (1.10, 0.90, 0.35),
        "consensus_independent_lean": (0.90, 1.10, 0.35),
        "consensus_soft_proximity": (0.60, 0.60, 1.00),
    }
    decoders = (
        "horizon",
        "v4_fallback",
        "independent_fallback",
        "consensus_event",
    )
    parts: dict[str, list[pd.DataFrame]] = {
        f"{score_name}__{decoder}": []
        for score_name in score_specs
        for decoder in decoders
    }
    linkable_releases, release_filter_diagnostics = filter_linkable_releases(releases)
    diagnostics: dict[str, Any] = {
        "targets": {},
        "score_specs": score_specs,
        "release_filter": release_filter_diagnostics,
    }

    for target_id in sorted(releases, key=int):
        challenge = challenges[target_id].reset_index(drop=True)
        challenge_ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        v4_time = _aligned_expert(v4, target_id, challenge_ids, "v4")
        independent_time = _aligned_expert(
            independent, target_id, challenge_ids, "independent"
        )
        active_v4 = ~np.isclose(v4_time, abstain_time)
        active_independent = ~np.isclose(independent_time, abstain_time)
        proximity = proximity_score(linkable_releases[target_id], challenge)
        target_details: dict[str, Any] = {
            "v4_active": int(np.count_nonzero(active_v4)),
            "independent_active": int(np.count_nonzero(active_independent)),
            "active_overlap": int(np.count_nonzero(active_v4 * active_independent)),
            "variants": {},
        }

        for score_name, (v4_weight, independent_weight, proximity_weight) in score_specs.items():
            score = (
                v4_weight * active_v4
                + independent_weight * active_independent
                + proximity_weight * proximity
            )
            selected, detail = stratified_half_selection(
                challenge, score, member_count=member_count
            )
            target_details["variants"][score_name] = {
                **detail.__dict__,
                "selected_consensus": int(
                    np.count_nonzero(selected & active_v4 & active_independent)
                ),
                "selected_v4_only": int(
                    np.count_nonzero(selected & active_v4 & ~active_independent)
                ),
                "selected_independent_only": int(
                    np.count_nonzero(selected & ~active_v4 & active_independent)
                ),
                "selected_neither": int(
                    np.count_nonzero(selected & ~active_v4 & ~active_independent)
                ),
                "selection_changes_from_v4": int(
                    np.count_nonzero(selected != active_v4)
                ),
                "selection_changes_from_independent": int(
                    np.count_nonzero(selected != active_independent)
                ),
            }
            for decoder in decoders:
                name = f"{score_name}__{decoder}"
                time_hat = _decode_times(
                    selected,
                    v4_time,
                    independent_time,
                    decoder=decoder,
                    abstain_time=abstain_time,
                    horizon_time=horizon_time,
                )
                parts[name].append(
                    pd.DataFrame(
                        {
                            "target_k": int(target_id),
                            "challenge_row_id": challenge_ids,
                            "time_hat": time_hat,
                        }
                    )
                )
        diagnostics["targets"][target_id] = target_details

    variants = {
        name: pd.concat(target_parts, ignore_index=True)
        .sort_values(["target_k", "challenge_row_id"], kind="mergesort")
        .reset_index(drop=True)
        for name, target_parts in parts.items()
    }
    return variants, diagnostics
