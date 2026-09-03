"""Private-negative conformal membership ranking for attack version 4.

The preliminary design exposes a guaranteed-negative subset through the
attacker's own AIA challenge.  This module uses those rows as an empirical
null for each target rather than only demoting them after a v1/v2 score has
already been built.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from attack_phase import generate_submission as v1


CONFORMAL_VIEW_SPECS = (
    v1.ViewSpec("qi", ("age", "SBP", "TG", "HDL", "ALT"), ("TG", "ALT")),
    v1.ViewSpec("stable", ("age", "SBP", "HDL")),
    v1.ViewSpec(
        "clinical",
        ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"),
        ("TG", "ALT", "FPG"),
    ),
    v1.ViewSpec(
        "outcome",
        (
            "BMI",
            "SBP",
            "TG",
            "HDL",
            "ALT",
            "FPG",
            "time",
            "smoking",
            "onset",
            "death",
        ),
        ("TG", "ALT", "FPG"),
    ),
)


def _normalize_boolean_mask(mask: np.ndarray, length: int, label: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 1 or len(array) != length or array.dtype.kind != "b":
        raise ValueError(f"{label} must be a one-dimensional boolean mask")
    return array.astype(bool, copy=False)


def empirical_lower_tail(values: np.ndarray, null_values: np.ndarray) -> np.ndarray:
    """Return smoothed empirical p-values when smaller values are stronger."""
    values = np.asarray(values, dtype=float)
    null = np.sort(np.asarray(null_values, dtype=float))
    if not len(null) or not np.isfinite(values).all() or not np.isfinite(null).all():
        raise ValueError("conformal lower-tail inputs must be finite and non-empty")
    counts = np.searchsorted(null, values, side="right")
    return (counts + 1.0) / (len(null) + 1.0)


def empirical_upper_tail(values: np.ndarray, null_values: np.ndarray) -> np.ndarray:
    """Return smoothed empirical p-values when larger values are stronger."""
    values = np.asarray(values, dtype=float)
    null = np.sort(np.asarray(null_values, dtype=float))
    if not len(null) or not np.isfinite(values).all() or not np.isfinite(null).all():
        raise ValueError("conformal upper-tail inputs must be finite and non-empty")
    counts_below = np.searchsorted(null, values, side="left")
    return (len(null) - counts_below + 1.0) / (len(null) + 1.0)


def strict_lexicographic_rank(
    primary: np.ndarray,
    secondary: np.ndarray,
    tie_key: np.ndarray,
) -> np.ndarray:
    """Rank by primary evidence, then continuous evidence, then record id."""
    primary = np.asarray(primary, dtype=float)
    secondary = np.asarray(secondary, dtype=float)
    tie_key = np.asarray(tie_key)
    if primary.ndim != 1 or secondary.shape != primary.shape or tie_key.shape != primary.shape:
        raise ValueError("rank evidence arrays must have equal one-dimensional shapes")
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError("rank evidence must be finite")
    order = np.lexsort((tie_key, secondary, primary))
    ranks = np.empty(len(primary), dtype=float)
    ranks[order] = np.arange(1, len(primary) + 1, dtype=float)
    return ranks / (len(primary) + 1.0)


def preserve_subset_order(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    subset_mask: np.ndarray,
) -> pd.DataFrame:
    """Preserve reference ordering within a subset using candidate score slots.

    Reassigning only the subset's existing score values leaves the full score
    distribution and all non-subset scores unchanged.  It makes evaluation on
    the subset depend on the validated reference ordering exactly.
    """
    if list(candidate.columns) != list(reference.columns):
        raise ValueError("MIA candidate and reference columns differ")
    if not candidate["record_id"].equals(reference["record_id"]):
        raise ValueError("MIA candidate and reference record order differs")
    subset = _normalize_boolean_mask(subset_mask, len(candidate), "subset_mask")
    if not subset.any():
        raise ValueError("subset_mask is empty")
    output = candidate.copy()
    positions = np.flatnonzero(subset)
    record_ids = output["record_id"].to_numpy(dtype=np.int64)
    for target_id in [column for column in output.columns if column != "record_id"]:
        candidate_values = output[target_id].to_numpy(dtype=float, copy=True)
        reference_values = reference[target_id].to_numpy(dtype=float)
        destination_order = positions[
            np.lexsort((record_ids[positions], reference_values[positions]))
        ]
        candidate_values[destination_order] = np.sort(candidate_values[positions])
        output[target_id] = candidate_values
    return output


def demote_known_private(
    frame: pd.DataFrame,
    known_private: np.ndarray,
) -> pd.DataFrame:
    """Move guaranteed negatives below every eligible row and restore ranks."""
    known_private = _normalize_boolean_mask(
        known_private, len(frame), "known_private"
    )
    if not known_private.any():
        raise ValueError("known_private is empty")
    output = frame.copy()
    record_ids = output["record_id"].to_numpy(dtype=np.int64)
    for target_id in [column for column in output.columns if column != "record_id"]:
        values = output[target_id].to_numpy(dtype=float, copy=True)
        values[known_private] = np.min(values) - max(float(np.ptp(values)), 1.0)
        output[target_id] = v1.strict_rank(values, record_ids)
    return output


def reciprocal_rank_fusion(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    known_private: np.ndarray,
    *,
    rank_constant: float = 10.0,
) -> pd.DataFrame:
    """Fuse two strict rank matrices with a precision-oriented RRF score."""
    if rank_constant < 0.0:
        raise ValueError("RRF rank_constant must be non-negative")
    if list(reference.columns) != list(candidate.columns):
        raise ValueError("MIA reference and candidate columns differ")
    if not reference["record_id"].equals(candidate["record_id"]):
        raise ValueError("MIA reference and candidate record order differs")
    known_private = _normalize_boolean_mask(
        known_private, len(reference), "known_private"
    )
    output = reference[["record_id"]].copy()
    n_rows = len(reference)
    for target_id in [column for column in reference.columns if column != "record_id"]:
        reference_position = np.clip(
            (n_rows + 1.0)
            * (1.0 - reference[target_id].to_numpy(dtype=float)),
            1.0,
            n_rows,
        )
        candidate_position = np.clip(
            (n_rows + 1.0)
            * (1.0 - candidate[target_id].to_numpy(dtype=float)),
            1.0,
            n_rows,
        )
        output[target_id] = (
            1.0 / (rank_constant + reference_position)
            + 1.0 / (rank_constant + candidate_position)
        )
    return demote_known_private(output, known_private)


def build_private_conformal_mia(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    known_private: np.ndarray,
    *,
    margin_weights: tuple[float, ...] = (0.25, 0.50),
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build target-wise MIA ranks calibrated by guaranteed non-members."""
    if not targets:
        raise ValueError("MIA targets are empty")
    v1.require_columns(candidates, ("record_id", *v1.MIA_REQUIRED_COLUMNS), "candidates")
    v1.require_columns(background, v1.MIA_REQUIRED_COLUMNS, "background")
    known_private = _normalize_boolean_mask(
        known_private, len(candidates), "known_private"
    )
    if not known_private.any() or known_private.all():
        raise ValueError("known_private must contain both null and eligible rows")
    record_ids = candidates["record_id"].to_numpy(dtype=np.int64)
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("candidate record_id must be unique")

    target_ids = list(targets)
    fitted_views = [v1.fit_view(background, spec) for spec in CONFORMAL_VIEW_SPECS]
    candidate_views = [view.transform(candidates) for view in fitted_views]
    shape = (len(candidates), len(target_ids), len(fitted_views))
    distance_evidence = np.empty(shape, dtype=float)
    reciprocal_evidence = np.empty(shape, dtype=float)
    nearest_rows = np.empty(shape, dtype=int)
    target_details: dict[str, Any] = {}

    for target_index, target_id in enumerate(target_ids):
        released = targets[target_id]
        v1.require_columns(released, v1.MIA_REQUIRED_COLUMNS, f"target {target_id}")
        if len(released) < 1:
            raise ValueError(f"target {target_id} is empty")
        view_details: dict[str, Any] = {}
        for view_index, (fitted_view, candidate_x) in enumerate(
            zip(fitted_views, candidate_views, strict=True)
        ):
            released_x = fitted_view.transform(released)
            candidate_distance, _, candidate_nearest, candidate_fallback = (
                v1.blockwise_nearest(
                    released,
                    released_x,
                    candidates,
                    candidate_x,
                    n_neighbors=2,
                )
            )
            release_distance, _, release_nearest, release_fallback = (
                v1.blockwise_nearest(
                    candidates,
                    candidate_x,
                    released,
                    released_x,
                    n_neighbors=2,
                )
            )
            reverse_distance = release_distance[candidate_nearest]
            reciprocal_ratio = candidate_distance / np.clip(
                reverse_distance, 1e-12, None
            )
            reciprocal_match = (
                release_nearest[candidate_nearest]
                == np.arange(len(candidates), dtype=int)
            )
            reciprocal_ratio *= np.where(reciprocal_match, 0.5, 1.0)
            distance_p = empirical_lower_tail(
                candidate_distance, candidate_distance[known_private]
            )
            reciprocal_p = empirical_lower_tail(
                reciprocal_ratio, reciprocal_ratio[known_private]
            )
            distance_evidence[:, target_index, view_index] = -np.log(distance_p)
            reciprocal_evidence[:, target_index, view_index] = -np.log(reciprocal_p)
            nearest_rows[:, target_index, view_index] = candidate_nearest
            view_details[fitted_view.spec.name] = {
                "candidate_fallback_rows": int(np.count_nonzero(candidate_fallback)),
                "release_fallback_rows": int(np.count_nonzero(release_fallback)),
                "reciprocal_candidate_rows": int(np.count_nonzero(reciprocal_match)),
            }
        target_details[str(target_id)] = {
            "released_rows": len(released),
            "views": view_details,
        }

    sorted_distance = np.sort(distance_evidence, axis=2)
    robust_distance = (
        0.55 * sorted_distance[:, :, -2:].mean(axis=2)
        + 0.45 * np.median(distance_evidence, axis=2)
    )
    robust_reciprocal = np.sort(reciprocal_evidence, axis=2)[:, :, -2:].mean(
        axis=2
    )
    coherence = np.empty((len(candidates), len(target_ids)), dtype=float)
    for target_index in range(len(target_ids)):
        for candidate_index in range(len(candidates)):
            _, counts = np.unique(
                nearest_rows[candidate_index, target_index], return_counts=True
            )
            coherence[candidate_index, target_index] = float(counts.max()) / len(
                fitted_views
            )
    combined = robust_distance + 0.20 * robust_reciprocal + 0.25 * coherence

    log_e = np.empty_like(combined)
    for target_index in range(len(target_ids)):
        p_value = empirical_upper_tail(
            combined[:, target_index], combined[known_private, target_index]
        )
        log_e[:, target_index] = -np.log(p_value)
    other_max = np.empty_like(log_e)
    for target_index in range(len(target_ids)):
        other_max[:, target_index] = np.max(
            np.delete(log_e, target_index, axis=1), axis=1
        )
    margin = log_e - other_max

    raw_variants: dict[str, np.ndarray] = {"conformal_plain": log_e}
    for weight in margin_weights:
        if weight < 0.0:
            raise ValueError("conformal margin weights must be non-negative")
        label = f"{weight:.2f}".replace(".", "p")
        raw_variants[f"conformal_margin_{label}"] = log_e + weight * margin

    frames: dict[str, pd.DataFrame] = {}
    primary_ties: dict[str, dict[str, int]] = {}
    for name, raw in raw_variants.items():
        ranked = np.empty_like(raw)
        per_target_ties: dict[str, int] = {}
        for target_index, target_id in enumerate(target_ids):
            primary = raw[:, target_index].copy()
            primary[known_private] = np.min(primary) - max(float(np.ptp(primary)), 1.0)
            per_target_ties[str(target_id)] = int(
                len(primary) - len(np.unique(primary))
            )
            ranked[:, target_index] = strict_lexicographic_rank(
                primary,
                combined[:, target_index],
                record_ids,
            )
        frame = pd.DataFrame(ranked, columns=target_ids)
        frame.insert(0, "record_id", record_ids)
        frames[name] = frame
        primary_ties[name] = per_target_ties

    winner = np.argmax(log_e, axis=1)
    diagnostics = {
        "known_private_rows": int(known_private.sum()),
        "eligible_rows": int((~known_private).sum()),
        "views": [spec.name for spec in CONFORMAL_VIEW_SPECS],
        "distance_aggregation": "0.55*mean(top2)+0.45*median(all4)",
        "reciprocal_weight": 0.20,
        "coherence_weight": 0.25,
        "winner_counts": {
            str(target_id): int(np.count_nonzero(winner == index))
            for index, target_id in enumerate(target_ids)
        },
        "primary_ties_resolved_by_continuous_evidence": primary_ties,
        "targets": target_details,
    }
    return frames, diagnostics
