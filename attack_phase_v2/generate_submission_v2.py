"""Generate the independent PWS Cup 2026 attack candidate version 2.

Version 2 deliberately lives outside ``attack_phase`` so that the previously
validated v1 candidate remains reproducible.  It reuses v1's input validation,
KDE evidence and capacity solvers, then evaluates two bounded changes:

* MIA experiments blend candidate-local k-nearest-neighbour density ratios with
  the v1 multi-view evidence before the cross-target transport step.  The v2
  main output keeps v1 transport because it won the shadow regression suite.
* AIA experiments separate membership selection from row linkage.  The v2 main
  output keeps v1's stronger selection and adaptively shrinks uncertain event
  times to the public administrative-censoring horizon.  This avoids treating a
  weak nearest-neighbour link as an individual time disclosure.

Only locally distributed competition files are read.  No submission, token or
network operation is performed by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack_phase import generate_submission as v1  # noqa: E402


MIA_DENSITY_NEIGHBOURS = (1, 3, 8)
AIA_DENSITY_NEIGHBOURS = (1, 3, 8)
DEFAULT_MIA_DENSITY_WEIGHT = 0.35
DEFAULT_MIA_RARE_WEIGHT = 0.65
DEFAULT_AIA_LINK_TRUST_DISTANCE = 0.20
DEFAULT_AIA_ANOMALOUS_EVENT_RATE = 0.50
DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE = 0.08


@dataclass(frozen=True)
class NeighbourDistances:
    distances: np.ndarray
    pool_size: np.ndarray
    fallback: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PWS Cup 2026の独立した攻撃候補v2をローカル生成します。"
    )
    parser.add_argument(
        "--dist",
        type=Path,
        required=True,
        help="B.csv（またはB_practice.csv）とA_bg.csvを含む参加者データ",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="C_<k>.csv、aia_challenge_<k>.csv、mia_columns.jsonのディレクトリ",
    )
    parser.add_argument(
        "--exclude-targets",
        default="",
        help="処理対象から除外するtarget_kのカンマ区切り（例: 24）",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="比較出力とは別に本命CSV 2枚だけを複製する提出候補ディレクトリ",
    )
    parser.add_argument("--aia-active-fraction", type=float, default=0.5)
    parser.add_argument(
        "--aia-variant-fractions",
        default="0.25,0.5,0.75,1.0",
        help="比較用AIA候補の選択割合",
    )
    parser.add_argument(
        "--aia-abstain-time",
        type=float,
        default=None,
        help="非選択行のtime_hat。省略時は真値域の外にある有限sentinel",
    )
    parser.add_argument(
        "--aia-link-trust-distance",
        type=float,
        default=DEFAULT_AIA_LINK_TRUST_DISTANCE,
        help="イベント時刻の個票リンクを信頼する最大ロバストRMS距離",
    )
    parser.add_argument(
        "--aia-anomalous-event-rate",
        type=float,
        default=DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        help="このイベント率を超える公開Cでは信頼距離を保守化",
    )
    parser.add_argument(
        "--aia-anomalous-link-trust-distance",
        type=float,
        default=DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        help="イベント率が極端な公開Cで使う最大ロバストRMS距離",
    )
    parser.add_argument("--mia-min-per-target", type=int, default=25)
    parser.add_argument("--mia-max-per-target", type=int, default=27)
    parser.add_argument("--mia-total-shared", type=int, default=736)
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument(
        "--mia-density-weight",
        type=float,
        default=DEFAULT_MIA_DENSITY_WEIGHT,
        help="v1証拠に混ぜる候補局所kNN密度比の重み",
    )
    parser.add_argument(
        "--mia-rare-weight",
        type=float,
        default=DEFAULT_MIA_RARE_WEIGHT,
        help="B_self既知希少行で専用transport順位を混ぜる重み",
    )
    parser.add_argument("--skip-domias", action="store_true")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="F_mia.csvとF_aia.csvだけを書き出す（提出候補ディレクトリ向け）",
    )
    args = parser.parse_args()
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        parser.error("--aia-active-fraction は0〜1です")
    if args.aia_link_trust_distance < 0.0:
        parser.error("--aia-link-trust-distance は0以上です")
    if not 0.0 <= args.aia_anomalous_event_rate <= 1.0:
        parser.error("--aia-anomalous-event-rate は0〜1です")
    if args.aia_anomalous_link_trust_distance < 0.0:
        parser.error("--aia-anomalous-link-trust-distance は0以上です")
    if not 0.0 <= args.mia_density_weight <= 1.0:
        parser.error("--mia-density-weight は0〜1です")
    if not 0.0 <= args.mia_rare_weight <= 1.0:
        parser.error("--mia-rare-weight は0〜1です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    args.exclude_targets = tuple(
        value.strip() for value in args.exclude_targets.split(",") if value.strip()
    )
    if len(args.exclude_targets) != len(set(args.exclude_targets)):
        parser.error("--exclude-targets に重複があります")
    return args


def _reference_groups(
    frame: pd.DataFrame,
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[tuple[Any, ...], np.ndarray]]:
    return (
        v1._group_indices(frame, v1.BLOCK_COLUMNS),
        v1._group_indices(frame, (v1.BLOCK_COLUMNS[0],)),
    )


def blockwise_neighbour_distances(
    reference: pd.DataFrame,
    reference_x: np.ndarray,
    query: pd.DataFrame,
    query_x: np.ndarray,
    neighbours: Sequence[int],
) -> NeighbourDistances:
    """Return requested kNN distances with v1-compatible block fallbacks."""
    requested = tuple(sorted({int(value) for value in neighbours}))
    if not requested or requested[0] < 1:
        raise ValueError("neighbours must contain positive integers")
    exact_reference, sex_reference = _reference_groups(reference)
    query_groups = v1._group_indices(query, v1.BLOCK_COLUMNS)
    all_reference = np.arange(len(reference), dtype=int)
    distances = np.empty((len(query), len(requested)), dtype=float)
    pool_size = np.empty(len(query), dtype=float)
    fallback = np.empty(len(query), dtype=int)
    dimension_scale = math.sqrt(reference_x.shape[1])
    maximum_k = requested[-1]

    for key, query_indices in query_groups.items():
        candidates = exact_reference.get(key)
        level = 0
        if candidates is None or len(candidates) < maximum_k:
            candidates = sex_reference.get((key[0],))
            level = 1
        if candidates is None or len(candidates) < maximum_k:
            candidates = all_reference
            level = 2
        candidates = np.asarray(candidates, dtype=int)
        fitted_k = min(maximum_k, len(candidates))
        model = NearestNeighbors(n_neighbors=fitted_k, metric="euclidean")
        model.fit(reference_x[candidates])
        raw, _ = model.kneighbors(query_x[query_indices])
        raw = raw / dimension_scale
        for output_index, requested_k in enumerate(requested):
            distances[query_indices, output_index] = raw[:, min(requested_k, fitted_k) - 1]
        pool_size[query_indices] = len(candidates)
        fallback[query_indices] = level

    return NeighbourDistances(
        distances=distances,
        pool_size=pool_size,
        fallback=fallback,
    )


class LocalDensityCalibrator:
    """Candidate-local release/background kNN density-ratio evidence."""

    def __init__(
        self,
        background: pd.DataFrame,
        query: pd.DataFrame,
        fitted_views: Sequence[v1.FittedView],
        neighbours: Sequence[int],
        tie_key: np.ndarray,
    ) -> None:
        self.background = background
        self.query = query
        self.fitted_views = tuple(fitted_views)
        self.neighbours = tuple(sorted({int(value) for value in neighbours}))
        self.tie_key = np.asarray(tie_key)
        self.query_x = [view.transform(query) for view in self.fitted_views]
        self.background_distances: list[NeighbourDistances] = []
        for view, query_x in zip(self.fitted_views, self.query_x, strict=True):
            background_x = view.transform(background)
            self.background_distances.append(
                blockwise_neighbour_distances(
                    background,
                    background_x,
                    query,
                    query_x,
                    self.neighbours,
                )
            )

    def score(self, released: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        ranked_components: list[np.ndarray] = []
        diagnostics: dict[str, Any] = {}
        floor = 1e-8
        for fitted_view, query_x, background_knn in zip(
            self.fitted_views,
            self.query_x,
            self.background_distances,
            strict=True,
        ):
            released_x = fitted_view.transform(released)
            released_knn = blockwise_neighbour_distances(
                released,
                released_x,
                self.query,
                query_x,
                self.neighbours,
            )
            dimension = len(fitted_view.spec.columns)
            view_diagnostics: dict[str, Any] = {
                "release_fallback_rows": int(np.count_nonzero(released_knn.fallback)),
                "background_fallback_rows": int(np.count_nonzero(background_knn.fallback)),
            }
            for neighbour_index, k in enumerate(self.neighbours):
                release_distance = released_knn.distances[:, neighbour_index]
                background_distance = background_knn.distances[:, neighbour_index]
                sample_correction = np.log(
                    np.clip(background_knn.pool_size, 1.0, None)
                    / np.clip(released_knn.pool_size, 1.0, None)
                )
                log_ratio = sample_correction + dimension * np.log(
                    (background_distance + floor) / (release_distance + floor)
                )
                log_ratio = np.nan_to_num(log_ratio, nan=0.0, posinf=700.0, neginf=-700.0)
                ranked_components.append(v1.strict_rank(log_ratio, self.tie_key))
                view_diagnostics[f"k{k}"] = {
                    "median_log_density_ratio": float(np.median(log_ratio)),
                    "p99_log_density_ratio": float(np.quantile(log_ratio, 0.99)),
                }
            diagnostics[fitted_view.spec.name] = view_diagnostics

        matrix = np.column_stack(ranked_components)
        ordered = np.sort(matrix, axis=1)
        top_two = ordered[:, -min(2, ordered.shape[1]) :].mean(axis=1)
        consensus = (
            0.55 * matrix.mean(axis=1)
            + 0.30 * top_two
            + 0.15 * matrix.max(axis=1)
        )
        return consensus, diagnostics


def _target_specific_affinity(
    raw_affinity: np.ndarray,
    record_ids: np.ndarray,
) -> np.ndarray:
    row_median = np.median(raw_affinity, axis=1, keepdims=True)
    residual = raw_affinity - row_median
    affinity = np.empty_like(raw_affinity)
    for target_index in range(raw_affinity.shape[1]):
        if raw_affinity.shape[1] > 1:
            other_max = np.max(
                np.delete(raw_affinity, target_index, axis=1), axis=1
            )
        else:
            other_max = np.zeros(len(raw_affinity), dtype=float)
        margin = raw_affinity[:, target_index] - other_max
        affinity[:, target_index] = (
            0.55 * v1.strict_rank(residual[:, target_index], record_ids)
            + 0.25 * v1.strict_rank(raw_affinity[:, target_index], record_ids)
            + 0.20 * v1.strict_rank(margin, record_ids)
        )
    return affinity


def _mia_frame(
    record_ids: np.ndarray,
    target_ids: Sequence[str],
    matrix: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"record_id": record_ids})
    for target_index, target_id in enumerate(target_ids):
        frame[target_id] = matrix[:, target_index]
    return frame


def load_candidate_rare_mask(
    dist: Path,
    candidates: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load attacker-owned B_self rare truth without exposing it in outputs."""
    path = dist / "B_self.csv"
    if not path.exists():
        return None, {"available": False, "reason": "B_self.csv not found"}
    frame = pd.read_csv(path, usecols=lambda column: column in {"record_id", "is_rare"})
    if set(frame.columns) != {"record_id", "is_rare"}:
        raise SystemExit("B_self.csv にrecord_id/is_rareがありません")
    if frame["record_id"].duplicated().any():
        raise SystemExit("B_self.csv のrecord_idが一意ではありません")
    indexed = frame.set_index("record_id")
    record_ids = candidates["record_id"].to_numpy(dtype=np.int64)
    missing = set(record_ids) - set(indexed.index.to_numpy(dtype=np.int64))
    extra = set(indexed.index.to_numpy(dtype=np.int64)) - set(record_ids)
    if missing or extra:
        raise SystemExit(
            "B_self.csv とBのrecord_id集合が一致しません: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    values = indexed.loc[record_ids, "is_rare"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise SystemExit("B_self.csv のis_rareは0/1である必要があります")
    mask = values.astype(bool)
    return mask, {"available": True, "rare_rows": int(mask.sum()), "rows": len(mask)}


def _rare_aware_transport(
    general_confidence: np.ndarray,
    hybrid_raw: np.ndarray,
    rare_mask: np.ndarray,
    record_ids: np.ndarray,
    *,
    total_shared: int,
    epsilon: float,
    rare_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add a sparse, target-exclusive transport among known rare candidates."""
    rare_indices = np.flatnonzero(np.asarray(rare_mask, dtype=bool))
    if not len(rare_indices) or rare_weight <= 0.0:
        return general_confidence.copy(), {
            "enabled": False,
            "rare_rows": int(len(rare_indices)),
        }
    n_targets = hybrid_raw.shape[1]
    # The public cohort design gives the overall overlap size.  Under exchangeability,
    # this is the expected rare overlap count; target capacities stay deliberately
    # loose because the realised number of rare members per target is not public.
    expected_total = int(round(total_shared * len(rare_indices) / len(rare_mask)))
    maximum_per_target = min(3, len(rare_indices))
    expected_total = min(
        max(expected_total, 0),
        len(rare_indices),
        n_targets * maximum_per_target,
    )
    rare_raw = hybrid_raw[rare_indices]
    rare_record_ids = record_ids[rare_indices]
    rare_affinity = _target_specific_affinity(rare_raw, rare_record_ids)
    lower = 1.0 / (len(rare_indices) + 1.0)
    upper = len(rare_indices) / (len(rare_indices) + 1.0)
    clipped = np.clip(rare_affinity, lower, upper)
    utility = np.log(clipped) - np.log1p(-clipped)
    assigned, target_counts = v1.constrained_capacity_assignment(
        utility,
        minimum=0,
        maximum=maximum_per_target,
        total=expected_total,
    )
    positive_targets = np.flatnonzero(target_counts > 0)
    plan = np.zeros_like(utility)
    if len(positive_targets):
        positive_plan, transport_diagnostics = v1.sinkhorn_transport(
            utility[:, positive_targets],
            target_counts[positive_targets],
            null_count=len(rare_indices) - expected_total,
            epsilon=epsilon,
        )
        plan[:, positive_targets] = positive_plan
    else:
        transport_diagnostics = {
            "iterations": 0,
            "max_row_error": 0.0,
            "max_column_error": 0.0,
        }
    rare_confidence = np.empty_like(plan)
    for target_index in range(n_targets):
        source = (
            plan[:, target_index]
            if target_counts[target_index] > 0
            else rare_affinity[:, target_index]
        )
        rare_confidence[:, target_index] = v1.strict_rank(source, rare_record_ids)

    combined = general_confidence.copy()
    combined[rare_indices] = (
        (1.0 - rare_weight) * combined[rare_indices]
        + rare_weight * rare_confidence
    )
    for target_index in range(n_targets):
        combined[:, target_index] = v1.strict_rank(
            combined[:, target_index], record_ids
        )
    return combined, {
        "enabled": True,
        "rare_rows": int(len(rare_indices)),
        "expected_rare_shared": expected_total,
        "maximum_per_target": maximum_per_target,
        "assigned_candidates": int((assigned >= 0).sum()),
        "per_target": target_counts.tolist(),
        "transport": transport_diagnostics,
        "weight": rare_weight,
    }


def build_mia_v2(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    *,
    minimum: int,
    maximum: int,
    total: int,
    transport_epsilon: float,
    include_domias: bool,
    density_weight: float = DEFAULT_MIA_DENSITY_WEIGHT,
    rare_mask: np.ndarray | None = None,
    rare_weight: float = DEFAULT_MIA_RARE_WEIGHT,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Blend v1 evidence with local density ratios and rerun transport."""
    _, baseline_variants, baseline_diagnostics = v1.build_mia(
        candidates,
        background,
        targets,
        minimum=minimum,
        maximum=maximum,
        total=total,
        transport_epsilon=transport_epsilon,
        include_domias=include_domias,
    )
    target_ids = list(targets)
    record_ids = candidates["record_id"].to_numpy(dtype=np.int64)
    baseline_unconstrained = baseline_variants["unconstrained"][target_ids].to_numpy(
        dtype=float
    )
    fitted_views = [v1.fit_view(background, spec) for spec in v1.MIA_VIEW_SPECS]
    calibrator = LocalDensityCalibrator(
        background,
        candidates,
        fitted_views,
        MIA_DENSITY_NEIGHBOURS,
        record_ids,
    )
    density_affinity = np.empty_like(baseline_unconstrained)
    density_diagnostics: dict[str, Any] = {}
    for target_index, target_id in enumerate(target_ids):
        score, detail = calibrator.score(targets[target_id])
        density_affinity[:, target_index] = score
        density_diagnostics[target_id] = detail

    hybrid_raw = (
        (1.0 - density_weight) * baseline_unconstrained
        + density_weight * density_affinity
    )
    affinity = _target_specific_affinity(hybrid_raw, record_ids)
    affinity += v1.deterministic_jitter(record_ids, target_ids)
    lower = 1.0 / (len(candidates) + 1.0)
    upper = len(candidates) / (len(candidates) + 1.0)
    clipped = np.clip(affinity, lower, upper)
    utility = np.log(clipped) - np.log1p(-clipped)
    assigned, target_counts = v1.constrained_capacity_assignment(
        utility,
        minimum=minimum,
        maximum=maximum,
        total=total,
    )
    transport, transport_diagnostics = v1.sinkhorn_transport(
        utility,
        target_counts,
        null_count=len(candidates) - total,
        epsilon=transport_epsilon,
    )
    transport_confidence = np.empty_like(transport)
    hybrid_unconstrained = np.empty_like(hybrid_raw)
    density_unconstrained = np.empty_like(density_affinity)
    for target_index in range(len(target_ids)):
        transport_confidence[:, target_index] = v1.strict_rank(
            transport[:, target_index], record_ids
        )
        hybrid_unconstrained[:, target_index] = v1.strict_rank(
            hybrid_raw[:, target_index], record_ids
        )
        density_unconstrained[:, target_index] = v1.strict_rank(
            density_affinity[:, target_index], record_ids
        )

    if rare_mask is not None:
        rare_aware_confidence, rare_diagnostics = _rare_aware_transport(
            transport_confidence,
            hybrid_raw,
            rare_mask,
            record_ids,
            total_shared=total,
            epsilon=transport_epsilon,
            rare_weight=rare_weight,
        )
    else:
        rare_aware_confidence = transport_confidence.copy()
        rare_diagnostics = {"enabled": False, "reason": "rare mask unavailable"}
    # The current shadow suite still ranks v1 transport above the new density
    # experiments.  Version 2 therefore retains it as the selected MIA instead
    # of forcing an unvalidated change into the submission pair.
    main_name = "v2_selected_transport"

    variants = {
        "v2_selected_transport": baseline_variants["transport"].copy(),
        "v2_rare_aware_transport": _mia_frame(
            record_ids, target_ids, rare_aware_confidence
        ),
        "v2_hybrid_transport": _mia_frame(
            record_ids, target_ids, transport_confidence
        ),
        "v2_hybrid_unconstrained": _mia_frame(
            record_ids, target_ids, hybrid_unconstrained
        ),
        "v2_density_unconstrained": _mia_frame(
            record_ids, target_ids, density_unconstrained
        ),
        "v1_transport": baseline_variants["transport"],
        "v1_unconstrained": baseline_variants["unconstrained"],
        "v1_hard_capacity": baseline_variants["hard_capacity"],
    }
    diagnostics = {
        "version": 2,
        "main_variant": main_name,
        "density_weight": density_weight,
        "rare_weight": rare_weight,
        "density_neighbours": list(MIA_DENSITY_NEIGHBOURS),
        "capacity": baseline_diagnostics["capacity"],
        "experimental_hybrid_capacity": {
            "minimum": minimum,
            "maximum": maximum,
            "total_shared": total,
            "unassigned": int((assigned < 0).sum()),
            "per_target": {
                target_id: int(target_counts[index])
                for index, target_id in enumerate(target_ids)
            },
        },
        "transport": baseline_diagnostics["transport"],
        "experimental_hybrid_transport": transport_diagnostics,
        "top10_before_constraint": baseline_diagnostics[
            "top10_before_constraint"
        ],
        "top10_after_transport": baseline_diagnostics["top10_after_transport"],
        "experimental_hybrid_top10_before_constraint": v1.top_concentration(
            hybrid_unconstrained
        ),
        "experimental_hybrid_top10_after_transport": v1.top_concentration(
            transport_confidence
        ),
        "density_targets": density_diagnostics,
        "rare_transport": rare_diagnostics,
        "selection_note": (
            "v1 transport retained as v2 main; density and rare-aware MIA remain comparison variants"
        ),
        "v1": baseline_diagnostics,
    }
    return variants[main_name], variants, diagnostics


def safe_abstain_time(horizon: float, tau: float) -> float:
    """Finite non-negative sentinel guaranteed to miss every valid truth time."""
    return float(horizon + tau + max(tau, 1e-6))


def effective_link_trust_distance(
    released: pd.DataFrame,
    default_distance: float,
    *,
    anomalous_event_rate: float = DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
    anomalous_distance: float = DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
) -> tuple[float, float, bool]:
    """Use a conservative link threshold for outcome-anomalous releases."""
    event = (
        released["onset"].to_numpy(dtype=int)
        + released["death"].to_numpy(dtype=int)
    ) > 0
    event_rate = float(np.mean(event))
    guard_triggered = event_rate > anomalous_event_rate
    distance = (
        min(default_distance, anomalous_distance)
        if guard_triggered
        else default_distance
    )
    return float(distance), event_rate, guard_triggered


def select_and_link(
    pair_cost: np.ndarray,
    membership_score: np.ndarray,
    active_count: int,
    tie_key: np.ndarray,
) -> np.ndarray:
    """Select by membership evidence, then solve a unique row linkage."""
    n_queries, n_released = pair_cost.shape
    if active_count < 0 or active_count > min(n_queries, n_released):
        raise ValueError("invalid AIA active_count")
    assigned = np.full(n_queries, -1, dtype=int)
    if active_count == 0:
        return assigned
    order = np.lexsort((np.asarray(tie_key), -np.asarray(membership_score)))
    active_rows = order[:active_count]
    local_rows, release_columns = linear_sum_assignment(pair_cost[active_rows])
    assigned[active_rows[local_rows]] = release_columns
    if int((assigned >= 0).sum()) != active_count:
        raise RuntimeError("v2 AIA linkage produced the wrong active count")
    if len(np.unique(assigned[assigned >= 0])) != active_count:
        raise RuntimeError("v2 AIA linkage reused a released row")
    return assigned


def decode_linked_times(
    released: pd.DataFrame,
    assigned: np.ndarray,
    pair_cost: np.ndarray,
    strength: np.ndarray,
    *,
    abstain_time: float,
    horizon_time: float,
    trust_distance: float,
) -> dict[str, np.ndarray]:
    """Return linked, censor-aware, horizon and adaptive time predictions."""
    linked = np.full(len(assigned), abstain_time, dtype=float)
    censor_aware = linked.copy()
    horizon = linked.copy()
    adaptive = linked.copy()
    active_rows = np.flatnonzero(assigned >= 0)
    if not len(active_rows):
        return {
            "linked": linked,
            "censor_aware": censor_aware,
            "horizon": horizon,
            "adaptive": adaptive,
        }

    release_rows = assigned[active_rows]
    release_time = released["time"].to_numpy(dtype=float)[release_rows]
    release_event = (
        released["onset"].to_numpy(dtype=int)[release_rows]
        + released["death"].to_numpy(dtype=int)[release_rows]
    ) > 0
    # v1 subtracts 0.35 * strength from every edge of a challenge row.  Add it
    # back so the trust threshold is expressed in the original robust distance
    # plus the published block mismatch penalties.
    penalized_distance = (
        pair_cost[active_rows, release_rows] + 0.35 * strength[active_rows]
    )
    trusted_event = release_event & (penalized_distance <= trust_distance)

    linked[active_rows] = release_time
    censor_aware[active_rows] = np.where(
        release_event, release_time, horizon_time
    )
    horizon[active_rows] = horizon_time
    adaptive[active_rows] = horizon_time
    adaptive[active_rows[trusted_event]] = release_time[trusted_event]
    return {
        "linked": linked,
        "censor_aware": censor_aware,
        "horizon": horizon,
        "adaptive": adaptive,
    }


def _append_aia_frame(
    destination: dict[str, list[pd.DataFrame]],
    name: str,
    target_id: str,
    challenge_ids: np.ndarray,
    time_hat: np.ndarray,
) -> None:
    destination.setdefault(name, []).append(
        pd.DataFrame(
            {
                "target_k": int(target_id),
                "challenge_row_id": challenge_ids,
                "time_hat": time_hat,
            }
        )
    )


def build_aia_v2(
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    main_fraction: float,
    variant_fractions: list[float],
    abstain_time: float,
    horizon_time: float,
    link_trust_distance: float = DEFAULT_AIA_LINK_TRUST_DISTANCE,
    baseline_abstain_time: float | None = None,
    anomalous_event_rate: float = DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
    anomalous_link_trust_distance: float = (
        DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE
    ),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Build v2 AIA variants with separated selection and time decoding."""
    fitted_views = [v1.fit_view(background, spec) for spec in v1.AIA_VIEW_SPECS]
    fractions = sorted(set([main_fraction, *variant_fractions]))
    frames: dict[str, list[pd.DataFrame]] = {}
    target_diagnostics: dict[str, Any] = {}

    for target_id, released in targets.items():
        challenge = challenges[target_id]
        challenge_ids = challenge["challenge_row_id"].to_numpy()
        (
            target_link_trust_distance,
            released_event_rate,
            anomaly_guard_triggered,
        ) = effective_link_trust_distance(
            released,
            link_trust_distance,
            anomalous_event_rate=anomalous_event_rate,
            anomalous_distance=anomalous_link_trust_distance,
        )
        pair_cost, strength, view_summary = v1.aia_pair_cost(
            released, challenge, background, fitted_views
        )
        density_calibrator = LocalDensityCalibrator(
            background,
            challenge,
            fitted_views,
            AIA_DENSITY_NEIGHBOURS,
            challenge_ids,
        )
        density_score, density_summary = density_calibrator.score(released)
        nearest_cost = pair_cost.min(axis=1)
        strength_rank = v1.strict_rank(strength, challenge_ids)
        closeness_rank = v1.strict_rank(-nearest_cost, challenge_ids)
        density_rank = v1.strict_rank(density_score, challenge_ids)
        membership_score = (
            0.50 * strength_rank + 0.30 * density_rank + 0.20 * closeness_rank
        )
        fraction_diagnostics: dict[str, Any] = {}

        for fraction in fractions:
            label = v1.fraction_label(fraction)
            active_count = int(round(len(challenge) * fraction))
            baseline_assigned = v1.cardinality_link_assignment(pair_cost, active_count)
            v2_assigned = select_and_link(
                pair_cost,
                membership_score,
                active_count,
                challenge_ids,
            )
            baseline_linked_decoded = decode_linked_times(
                released,
                baseline_assigned,
                pair_cost,
                strength,
                abstain_time=(
                    abstain_time
                    if baseline_abstain_time is None
                    else baseline_abstain_time
                ),
                horizon_time=horizon_time,
                trust_distance=target_link_trust_distance,
            )
            baseline_adaptive_decoded = decode_linked_times(
                released,
                baseline_assigned,
                pair_cost,
                strength,
                abstain_time=abstain_time,
                horizon_time=horizon_time,
                trust_distance=target_link_trust_distance,
            )
            v2_decoded = decode_linked_times(
                released,
                v2_assigned,
                pair_cost,
                strength,
                abstain_time=abstain_time,
                horizon_time=horizon_time,
                trust_distance=target_link_trust_distance,
            )
            _append_aia_frame(
                frames,
                f"v2_adaptive_{label}",
                target_id,
                challenge_ids,
                v2_decoded["adaptive"],
            )
            _append_aia_frame(
                frames,
                f"v2_horizon_{label}",
                target_id,
                challenge_ids,
                v2_decoded["horizon"],
            )
            _append_aia_frame(
                frames,
                f"v1_linked_{label}",
                target_id,
                challenge_ids,
                baseline_linked_decoded["linked"],
            )
            if np.isclose(fraction, main_fraction):
                _append_aia_frame(
                    frames,
                    f"v2_censor_aware_{label}",
                    target_id,
                    challenge_ids,
                    v2_decoded["censor_aware"],
                )
                _append_aia_frame(
                    frames,
                    f"v2_v1select_adaptive_{label}",
                    target_id,
                    challenge_ids,
                    baseline_adaptive_decoded["adaptive"],
                )

            baseline_active = baseline_assigned >= 0
            baseline_release_rows = baseline_assigned[baseline_active]
            baseline_release_events = (
                released["onset"].to_numpy(dtype=int)[baseline_release_rows]
                + released["death"].to_numpy(dtype=int)[baseline_release_rows]
            ) > 0
            baseline_restored_distance = (
                pair_cost[baseline_active, baseline_release_rows]
                + 0.35 * strength[baseline_active]
            )
            v2_active = v2_assigned >= 0
            v2_release_rows = v2_assigned[v2_active]
            v2_release_events = (
                released["onset"].to_numpy(dtype=int)[v2_release_rows]
                + released["death"].to_numpy(dtype=int)[v2_release_rows]
            ) > 0
            v2_restored_distance = (
                pair_cost[v2_active, v2_release_rows] + 0.35 * strength[v2_active]
            )
            fraction_diagnostics[label] = {
                "active_rows": active_count,
                "v1_v2_active_overlap": int(
                    np.count_nonzero(baseline_active & v2_active)
                ),
                "selected_release_event_rows": int(
                    baseline_release_events.sum()
                ),
                "trusted_event_rows": int(
                    np.count_nonzero(
                        baseline_release_events
                        & (
                            baseline_restored_distance
                            <= target_link_trust_distance
                        )
                    )
                ),
                "median_selected_penalized_distance": (
                    float(np.median(baseline_restored_distance))
                    if len(baseline_restored_distance)
                    else None
                ),
                "experimental_v2_selected_release_event_rows": int(
                    v2_release_events.sum()
                ),
                "experimental_v2_trusted_event_rows": int(
                    np.count_nonzero(
                        v2_release_events
                        & (v2_restored_distance <= target_link_trust_distance)
                    )
                ),
                "experimental_v2_median_selected_penalized_distance": (
                    float(np.median(v2_restored_distance))
                    if len(v2_restored_distance)
                    else None
                ),
            }

        target_diagnostics[target_id] = {
            "challenge_rows": len(challenge),
            "released_rows": len(released),
            "released_censor_fraction": float(
                np.mean(
                    (
                        released["onset"].to_numpy(dtype=int)
                        + released["death"].to_numpy(dtype=int)
                    )
                    == 0
                )
            ),
            "released_event_rate": released_event_rate,
            "effective_link_trust_distance": target_link_trust_distance,
            "anomaly_guard_triggered": anomaly_guard_triggered,
            "views": view_summary,
            "density": density_summary,
            "fractions": fraction_diagnostics,
        }

    variants = {
        name: pd.concat(parts, ignore_index=True).sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        )
        for name, parts in frames.items()
    }
    main_name = f"v2_v1select_adaptive_{v1.fraction_label(main_fraction)}"
    diagnostics = {
        "version": 2,
        "main_variant": main_name,
        "main_fraction": main_fraction,
        "variant_fractions": fractions,
        "abstain_time": abstain_time,
        "baseline_abstain_time": baseline_abstain_time,
        "horizon_time": horizon_time,
        "link_trust_distance": link_trust_distance,
        "anomalous_event_rate": anomalous_event_rate,
        "anomalous_link_trust_distance": anomalous_link_trust_distance,
        "density_neighbours": list(AIA_DENSITY_NEIGHBOURS),
        "targets": target_diagnostics,
    }
    return variants[main_name], variants, diagnostics


def write_outputs(
    out_dir: Path,
    f_mia: pd.DataFrame,
    mia_variants: dict[str, pd.DataFrame],
    f_aia: pd.DataFrame,
    aia_variants: dict[str, pd.DataFrame],
    diagnostics: dict[str, Any],
    *,
    main_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    f_mia.to_csv(out_dir / "F_mia.csv", index=False)
    f_aia.to_csv(out_dir / "F_aia.csv", index=False)
    if main_only:
        return
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in mia_variants.items():
        frame.to_csv(variants_dir / f"F_mia_{name}.csv", index=False)
    for name, frame in aia_variants.items():
        frame.to_csv(variants_dir / f"F_aia_{name}.csv", index=False)
    (out_dir / "diagnostics_v2.json").write_text(
        json.dumps(
            clean_json_v2(diagnostics),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def clean_json_v2(value: Any) -> Any:
    """Convert NumPy values without converting booleans to JSON integers."""
    if isinstance(value, dict):
        return {str(key): clean_json_v2(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_v2(item) for item in value]
    if isinstance(value, np.ndarray):
        return [clean_json_v2(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def validate_output_frames(
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Apply stricter checks than the bundled validator before writing files."""
    target_ids = list(targets)
    expected_mia_columns = ["record_id", *target_ids]
    if list(f_mia.columns) != expected_mia_columns:
        raise ValueError(
            "F_mia columns/order mismatch: "
            f"expected={expected_mia_columns}, got={list(f_mia.columns)}"
        )
    record_ids = pd.to_numeric(f_mia["record_id"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(record_ids).all() or not np.allclose(
        record_ids, np.round(record_ids)
    ):
        raise ValueError("F_mia record_id must contain finite integers")
    if f_mia["record_id"].duplicated().any():
        raise ValueError("F_mia record_id must be unique")
    expected_record_ids = set(candidates["record_id"].astype(int).tolist())
    if set(record_ids.astype(int).tolist()) != expected_record_ids:
        raise ValueError("F_mia record_id set does not match the attack cohort")

    confidence = f_mia[target_ids].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(confidence).all():
        raise ValueError("F_mia confidence contains a non-finite value")
    if ((confidence < 0.0) | (confidence > 1.0)).any():
        raise ValueError("F_mia confidence must be in [0, 1]")
    tied_targets = [
        target_id
        for target_id in target_ids
        if f_mia[target_id].nunique(dropna=False) != len(f_mia)
    ]
    if tied_targets:
        raise ValueError(f"F_mia must be tie-free for every target: {tied_targets}")

    expected_aia_columns = ["target_k", "challenge_row_id", "time_hat"]
    if list(f_aia.columns) != expected_aia_columns:
        raise ValueError(
            "F_aia columns/order mismatch: "
            f"expected={expected_aia_columns}, got={list(f_aia.columns)}"
        )
    target_values = pd.to_numeric(f_aia["target_k"], errors="coerce").to_numpy(
        dtype=float
    )
    challenge_values = pd.to_numeric(
        f_aia["challenge_row_id"], errors="coerce"
    ).to_numpy(dtype=float)
    time_values = pd.to_numeric(f_aia["time_hat"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(target_values).all() or not np.allclose(
        target_values, np.round(target_values)
    ):
        raise ValueError("F_aia target_k must contain finite integers")
    if not np.isfinite(challenge_values).all() or not np.allclose(
        challenge_values, np.round(challenge_values)
    ):
        raise ValueError("F_aia challenge_row_id must contain finite integers")
    if not np.isfinite(time_values).all() or (time_values < 0.0).any():
        raise ValueError("F_aia time_hat must contain finite non-negative values")
    if f_aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("F_aia composite key must be unique")

    actual_targets = {str(value) for value in target_values.astype(int)}
    if actual_targets != set(target_ids):
        raise ValueError(
            "F_aia target set mismatch: "
            f"expected={target_ids}, got={sorted(actual_targets, key=int)}"
        )
    for target_id in target_ids:
        actual_ids = set(
            f_aia.loc[
                target_values.astype(int) == int(target_id), "challenge_row_id"
            ]
            .astype(int)
            .tolist()
        )
        expected_ids = set(
            challenges[target_id]["challenge_row_id"].astype(int).tolist()
        )
        if actual_ids != expected_ids:
            raise ValueError(
                f"F_aia challenge ids do not match target {target_id}: "
                f"missing={len(expected_ids - actual_ids)}, "
                f"extra={len(actual_ids - expected_ids)}"
            )
    return {
        "strict_preflight_passed": True,
        "mia_rows": len(f_mia),
        "mia_target_columns": len(target_ids),
        "mia_tie_free_targets": len(target_ids),
        "aia_rows": len(f_aia),
        "aia_targets": len(target_ids),
    }


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = (args.targets or (dist / "targets")).resolve()
    out_dir = args.out_dir.resolve()
    candidate_dir = (
        args.candidate_dir.resolve() if args.candidate_dir is not None else None
    )
    if candidate_dir == out_dir:
        raise SystemExit("--candidate-dir は --out-dir と別の場所を指定してください")
    candidates, background, targets, challenges, quality = v1.load_inputs(
        dist,
        targets_dir,
        exclude_target_ids=args.exclude_targets,
    )
    rare_mask, rare_source = load_candidate_rare_mask(dist, candidates)
    n_targets = len(targets)
    if not (
        n_targets * args.mia_min_per_target
        <= args.mia_total_shared
        <= n_targets * args.mia_max_per_target
    ):
        raise SystemExit(
            "MIA capacity設定が対象数と両立しません: "
            f"targets={n_targets}, min={args.mia_min_per_target}, "
            f"max={args.mia_max_per_target}, total={args.mia_total_shared}"
        )
    if args.mia_total_shared > len(candidates):
        raise SystemExit(
            "MIA共有総数が候補者数を超えています: "
            f"total={args.mia_total_shared}, candidates={len(candidates)}"
        )
    fractions = [
        float(value.strip())
        for value in args.aia_variant_fractions.split(",")
        if value.strip()
    ]
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise SystemExit("--aia-variant-fractions は0〜1です")
    horizon, tau = v1.load_phase_parameters(dist)
    abstain_time = (
        float(args.aia_abstain_time)
        if args.aia_abstain_time is not None
        else safe_abstain_time(horizon, tau)
    )
    if not np.isfinite(abstain_time) or abstain_time < 0.0:
        raise SystemExit("AIA abstain timeは有限の非負値である必要があります")

    print(
        f"入力確認(v2): candidates={len(candidates)}, "
        f"targets={len(targets)}, A_bg={len(background)}"
    )
    print("MIA v2: v1 transportを採用し、kNN密度比・希少行補正は比較候補として生成")
    f_mia, mia_variants, mia_diagnostics = build_mia_v2(
        candidates,
        background,
        targets,
        minimum=args.mia_min_per_target,
        maximum=args.mia_max_per_target,
        total=args.mia_total_shared,
        transport_epsilon=args.mia_transport_epsilon,
        include_domias=not args.skip_domias,
        density_weight=args.mia_density_weight,
        rare_mask=rare_mask,
        rare_weight=args.mia_rare_weight,
    )
    print("AIA v2: v1会員選別を維持し、不確実な時刻だけ打切り時刻へ適応縮約")
    f_aia, aia_variants, aia_diagnostics = build_aia_v2(
        background,
        targets,
        challenges,
        main_fraction=args.aia_active_fraction,
        variant_fractions=fractions,
        abstain_time=abstain_time,
        horizon_time=horizon,
        link_trust_distance=args.aia_link_trust_distance,
        anomalous_event_rate=args.aia_anomalous_event_rate,
        anomalous_link_trust_distance=(
            args.aia_anomalous_link_trust_distance
        ),
        baseline_abstain_time=v1.choose_low_hit_time(
            background, horizon=horizon, tau=tau
        ),
    )
    preflight = validate_output_frames(
        f_mia, f_aia, candidates, targets, challenges
    )
    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_v2",
        "attack_version": 2,
        "network_access": False,
        "quality": quality,
        "parameters": {
            "horizon_years": horizon,
            "aia_tau_time": tau,
            "aia_active_fraction": args.aia_active_fraction,
            "aia_link_trust_distance": args.aia_link_trust_distance,
            "aia_anomalous_event_rate": args.aia_anomalous_event_rate,
            "aia_anomalous_link_trust_distance": (
                args.aia_anomalous_link_trust_distance
            ),
            "excluded_target_ids": list(args.exclude_targets),
            "mia_min_per_target": args.mia_min_per_target,
            "mia_max_per_target": args.mia_max_per_target,
            "mia_total_shared": args.mia_total_shared,
            "mia_transport_epsilon": args.mia_transport_epsilon,
            "mia_density_weight": args.mia_density_weight,
            "mia_rare_weight": args.mia_rare_weight,
            "domias_included": not args.skip_domias,
        },
        "rare_source": rare_source,
        "preflight": preflight,
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only membership and AIA truth are unavailable locally; official attack scores cannot be computed.",
            "Shadow evaluation is a regression test, not a leaderboard estimate.",
            "The v2 program reads distributed local files only and does not submit anything.",
        ],
    }
    write_outputs(
        out_dir,
        f_mia,
        mia_variants,
        f_aia,
        aia_variants,
        diagnostics,
        main_only=args.main_only,
    )
    if candidate_dir is not None:
        write_outputs(
            candidate_dir,
            f_mia,
            mia_variants,
            f_aia,
            aia_variants,
            diagnostics,
            main_only=True,
        )
    print(f"完了(v2): {out_dir}")
    print(f"  F_mia.csv: {len(f_mia)} rows x {len(f_mia.columns)} columns")
    print(f"  F_aia.csv: {len(f_aia)} rows")
    print(f"  AIA main: {aia_diagnostics['main_variant']}")
    print("  strict preflight: PASS")
    if candidate_dir is not None:
        print(f"  candidate CSVs: {candidate_dir}")


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    main()
