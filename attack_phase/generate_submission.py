"""Generate local PWS Cup 2026 privacy-competition MIA/AIA submissions.

This participant-side program only reads the files distributed for the competition.
It performs statistical record linkage and density comparison locally; it does not
contact, probe, or authenticate to any remote service.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.neighbors import NearestNeighbors


QI_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT")
CONTINUOUS_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
BLOCK_COLUMNS = ("sex", "prefecture")
MIA_REQUIRED_COLUMNS = (
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
AIA_REQUIRED_COLUMNS = ("challenge_row_id", *QI_COLUMNS, *BLOCK_COLUMNS)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    columns: tuple[str, ...]
    log_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FittedView:
    spec: ViewSpec
    center: np.ndarray
    scale: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, self.spec.columns].to_numpy(dtype=float, copy=True)
        for column in self.spec.log_columns:
            index = self.spec.columns.index(column)
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
        return (values - self.center) / self.scale


AIA_VIEW_SPECS = (
    ViewSpec("qi_raw", QI_COLUMNS),
    ViewSpec("qi_log_tails", QI_COLUMNS, ("TG", "ALT")),
    ViewSpec("qi_stable", ("age", "SBP", "HDL")),
)

MIA_VIEW_SPECS = (
    ViewSpec("qi_raw", QI_COLUMNS),
    ViewSpec("qi_log_tails", QI_COLUMNS, ("TG", "ALT")),
    ViewSpec("clinical_log_tails", CONTINUOUS_COLUMNS, ("TG", "ALT", "FPG")),
    ViewSpec(
        "clinical_outcome",
        (*CONTINUOUS_COLUMNS, "time", "smoking", "onset", "death"),
        ("TG", "ALT", "FPG"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配布済みCSVだけを使い、統計的MIA/AIA提出ファイルを生成します。"
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
        help="C_<k>.csv、aia_challenge_<k>.csv、mia_columns.jsonのディレクトリ。既定はDIST/targets",
    )
    parser.add_argument(
        "--exclude-targets",
        default="",
        help="自己攻撃など、処理対象から除外するtarget_kのカンマ区切り（例: 24）",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--aia-active-fraction",
        type=float,
        default=0.5,
        help="AIAで一意リンクを割り当てるchallenge行の割合（既定0.5）",
    )
    parser.add_argument(
        "--aia-variant-fractions",
        default="0.25,0.5,0.75,1.0",
        help="比較用F_aiaを作る選択割合のカンマ区切り",
    )
    parser.add_argument(
        "--aia-abstain-time",
        type=float,
        default=None,
        help="非選択行のtime_hat。省略時はA_bg内で±tau的中率が最小の0〜horizon内の値",
    )
    parser.add_argument("--mia-min-per-target", type=int, default=25)
    parser.add_argument("--mia-max-per-target", type=int, default=27)
    parser.add_argument("--mia-total-shared", type=int, default=736)
    parser.add_argument(
        "--mia-transport-epsilon",
        type=float,
        default=0.35,
        help="全標的同時soft assignmentのエントロピー温度",
    )
    parser.add_argument(
        "--skip-domias",
        action="store_true",
        help="KDE密度比成分を省略して高速化する",
    )
    args = parser.parse_args()
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        parser.error("--aia-active-fraction は0〜1です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    args.exclude_targets = tuple(
        value.strip() for value in args.exclude_targets.split(",") if value.strip()
    )
    if len(args.exclude_targets) != len(set(args.exclude_targets)):
        parser.error("--exclude-targets に重複があります")
    return args


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _tuple_key(raw: Any) -> tuple[Any, ...]:
    return raw if isinstance(raw, tuple) else (raw,)


def _group_indices(frame: pd.DataFrame, columns: Iterable[str]) -> dict[tuple[Any, ...], np.ndarray]:
    groups = frame.groupby(list(columns), sort=False, dropna=False).indices
    return {_tuple_key(key): np.asarray(indices, dtype=int) for key, indices in groups.items()}


def fit_view(background: pd.DataFrame, spec: ViewSpec) -> FittedView:
    values = background.loc[:, spec.columns].to_numpy(dtype=float, copy=True)
    for column in spec.log_columns:
        index = spec.columns.index(column)
        values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = np.std(values, axis=0, ddof=0)
    scale = np.where(robust_scale > 1e-12, robust_scale, standard_scale)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return FittedView(spec=spec, center=center, scale=scale)


def blockwise_nearest(
    released: pd.DataFrame,
    released_x: np.ndarray,
    query: pd.DataFrame,
    query_x: np.ndarray,
    *,
    n_neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return first/second RMS distance, nearest released row, and fallback level."""
    exact_released = _group_indices(released, BLOCK_COLUMNS)
    sex_released = _group_indices(released, (BLOCK_COLUMNS[0],))
    query_groups = _group_indices(query, BLOCK_COLUMNS)
    all_released = np.arange(len(released), dtype=int)
    first = np.empty(len(query), dtype=float)
    second = np.empty(len(query), dtype=float)
    nearest = np.empty(len(query), dtype=int)
    fallback = np.empty(len(query), dtype=int)
    dimension_scale = math.sqrt(released_x.shape[1])

    for key, query_indices in query_groups.items():
        candidates = exact_released.get(key)
        level = 0
        if candidates is None or len(candidates) == 0:
            candidates = sex_released.get((key[0],))
            level = 1
        if candidates is None or len(candidates) == 0:
            candidates = all_released
            level = 2
        candidates = np.asarray(candidates, dtype=int)
        k = min(max(n_neighbors, 1), len(candidates))
        model = NearestNeighbors(n_neighbors=k, metric="euclidean")
        model.fit(released_x[candidates])
        distances, local_indices = model.kneighbors(query_x[query_indices])
        distances = distances / dimension_scale
        first[query_indices] = distances[:, 0]
        second[query_indices] = distances[:, -1]
        nearest[query_indices] = candidates[local_indices[:, 0]]
        fallback[query_indices] = level
    return first, second, nearest, fallback


def conditional_tail_strength(
    query_distance: np.ndarray,
    query: pd.DataFrame,
    calibration_distance: np.ndarray,
    calibration: pd.DataFrame,
    *,
    minimum_group: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Map small distances to -log10 conditional empirical tail probabilities."""
    calibration_exact_indices = _group_indices(calibration, BLOCK_COLUMNS)
    calibration_sex_indices = _group_indices(calibration, (BLOCK_COLUMNS[0],))
    query_groups = _group_indices(query, BLOCK_COLUMNS)
    global_values = np.sort(calibration_distance)
    probability = np.empty(len(query), dtype=float)

    for key, query_indices in query_groups.items():
        calibration_indices = calibration_exact_indices.get(key)
        if calibration_indices is None or len(calibration_indices) < minimum_group:
            calibration_indices = calibration_sex_indices.get((key[0],))
        if calibration_indices is None or len(calibration_indices) < minimum_group:
            values = global_values
        else:
            values = np.sort(calibration_distance[np.asarray(calibration_indices, dtype=int)])
        counts = np.searchsorted(values, query_distance[query_indices], side="right")
        probability[query_indices] = (counts + 1.0) / (len(values) + 1.0)
    strength = -np.log10(np.clip(probability, 1e-12, 1.0))
    return strength, probability


def linkage_component(
    released: pd.DataFrame,
    query: pd.DataFrame,
    background: pd.DataFrame,
    fitted_view: FittedView,
) -> dict[str, np.ndarray]:
    released_x = fitted_view.transform(released)
    query_x = fitted_view.transform(query)
    background_x = fitted_view.transform(background)
    query_first, query_second, query_nearest, query_fallback = blockwise_nearest(
        released, released_x, query, query_x
    )
    background_first, _, _, _ = blockwise_nearest(
        released, released_x, background, background_x, n_neighbors=1
    )
    strength, tail_probability = conditional_tail_strength(
        query_first, query, background_first, background
    )
    distinctiveness = np.log1p(query_second / np.clip(query_first, 1e-12, None))
    return {
        "distance": query_first,
        "second_distance": query_second,
        "nearest": query_nearest,
        "fallback": query_fallback,
        "tail_probability": tail_probability,
        "strength": strength + 0.10 * distinctiveness,
    }


def strict_rank(values: np.ndarray, tie_key: np.ndarray | None = None) -> np.ndarray:
    """Strict, deterministic ranks in (0, 1), with larger values ranked higher."""
    values = np.asarray(values, dtype=float)
    if tie_key is None:
        tie_key = np.arange(len(values), dtype=np.int64)
    tie_key = np.asarray(tie_key)
    order = np.lexsort((tie_key, values))
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks / (len(values) + 1.0)


class KDEEvidence:
    """Cache the background KDE and return stable log p(C)-log p(A_bg) evidence."""

    def __init__(self, background: pd.DataFrame, candidates: pd.DataFrame):
        self.mean = background.loc[:, CONTINUOUS_COLUMNS].mean().to_numpy(dtype=float)
        self.scale = background.loc[:, CONTINUOUS_COLUMNS].std(ddof=0).to_numpy(dtype=float)
        self.scale = np.where(self.scale > 0.0, self.scale, 1.0)
        background_x = self._transform(background)
        self.candidate_x = self._transform(candidates)
        self.reference = gaussian_kde(background_x.T)
        self.reference_logpdf = self.reference.logpdf(self.candidate_x.T)

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame.loc[:, CONTINUOUS_COLUMNS].to_numpy(dtype=float)
        return (values - self.mean) / self.scale

    def score(self, released: pd.DataFrame) -> np.ndarray:
        try:
            synthetic = gaussian_kde(self._transform(released).T)
            evidence = synthetic.logpdf(self.candidate_x.T) - self.reference_logpdf
        except (np.linalg.LinAlgError, ValueError):
            return np.zeros(len(self.candidate_x), dtype=float)
        return np.nan_to_num(evidence, nan=0.0, posinf=700.0, neginf=-700.0)


def deterministic_jitter(record_ids: np.ndarray, target_ids: list[str]) -> np.ndarray:
    record = np.asarray(record_ids, dtype=np.int64)[:, None].astype(np.uint64)
    target_numbers = np.asarray([int(target) for target in target_ids], dtype=np.uint64)[None, :]
    mixed = (record * np.uint64(11400714819323198485)) ^ (
        target_numbers * np.uint64(14029467366897019727)
    )
    return (mixed % np.uint64(1_000_003)).astype(float) / 1_000_003.0 * 1e-9


def constrained_capacity_assignment(
    utility: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact MAP assignment with target capacities and an explicit no-target class."""
    n_candidates, n_targets = utility.shape
    if total < n_targets * minimum or total > n_targets * maximum:
        raise ValueError("MIA total is infeasible for the per-target capacity bounds")
    if total > n_candidates:
        raise ValueError("MIA total cannot exceed the number of candidates")
    n_null = n_candidates - total
    n_vacancies = n_targets * maximum - total
    target_slot_count = n_targets * maximum
    size = n_candidates + n_vacancies
    if target_slot_count + n_null != size:
        raise AssertionError("capacity construction is inconsistent")

    slot_target = np.repeat(np.arange(n_targets), maximum)
    slot_offset = np.tile(np.arange(maximum), n_targets)
    optional_slot = slot_offset >= minimum
    prohibited = 1e9
    cost = np.full((size, size), prohibited, dtype=float)
    cost[:n_candidates, :target_slot_count] = -utility[:, slot_target]
    cost[:n_candidates, target_slot_count:] = 0.0
    if n_vacancies:
        vacancy_rows = slice(n_candidates, size)
        optional_columns = np.flatnonzero(optional_slot)
        cost[vacancy_rows, optional_columns] = 0.0

    row_index, column_index = linear_sum_assignment(cost)
    if np.any(cost[row_index, column_index] >= prohibited / 2):
        raise RuntimeError("constrained MIA assignment used a prohibited edge")
    assigned = np.full(n_candidates, -1, dtype=int)
    for row, column in zip(row_index, column_index, strict=True):
        if row < n_candidates and column < target_slot_count:
            assigned[row] = int(slot_target[column])
    counts = np.bincount(assigned[assigned >= 0], minlength=n_targets)
    if int((assigned >= 0).sum()) != total:
        raise RuntimeError("constrained MIA assignment produced the wrong total")
    if np.any(counts < minimum) or np.any(counts > maximum):
        raise RuntimeError("constrained MIA assignment violated target capacities")
    return assigned, counts


def sinkhorn_transport(
    utility: np.ndarray,
    target_counts: np.ndarray,
    *,
    null_count: int,
    epsilon: float,
    tolerance: float = 1e-9,
    max_iterations: int = 5_000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Soft candidate-to-target/no-target assignment with exact column masses."""
    n_candidates, n_targets = utility.shape
    column_mass = np.concatenate([target_counts.astype(float), [float(null_count)]])
    if not np.isclose(column_mass.sum(), n_candidates):
        raise ValueError("transport column masses must sum to the candidate count")
    augmented_utility = np.column_stack([utility, np.zeros(n_candidates, dtype=float)])
    log_kernel = augmented_utility / epsilon
    log_v = np.zeros(n_targets + 1, dtype=float)
    converged_at = max_iterations

    for iteration in range(max_iterations):
        log_u = -logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = np.log(column_mass) - logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration % 25 == 0 or iteration == max_iterations - 1:
            log_plan = log_kernel + log_u[:, None] + log_v[None, :]
            plan = np.exp(log_plan)
            row_error = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
            column_error = float(np.max(np.abs(plan.sum(axis=0) - column_mass)))
            if max(row_error, column_error) <= tolerance:
                converged_at = iteration + 1
                break
    else:
        log_plan = log_kernel + log_u[:, None] + log_v[None, :]
        plan = np.exp(log_plan)
        row_error = float(np.max(np.abs(plan.sum(axis=1) - 1.0)))
        column_error = float(np.max(np.abs(plan.sum(axis=0) - column_mass)))

    return plan[:, :n_targets], {
        "iterations": converged_at,
        "max_row_error": row_error,
        "max_column_error": column_error,
    }


def exact_match_rate(left: pd.DataFrame, right: pd.DataFrame) -> float:
    columns = [column for column in right.columns if column in left.columns]
    if not columns:
        return 0.0
    right_hash = set(pd.util.hash_pandas_object(right[columns], index=False).to_numpy())
    left_hash = pd.util.hash_pandas_object(left[columns], index=False).to_numpy()
    return float(np.mean([value in right_hash for value in left_hash]))


def top_concentration(matrix: np.ndarray, top_n: int = 10) -> dict[str, int]:
    selected: list[int] = []
    for target_index in range(matrix.shape[1]):
        selected.extend(np.argsort(matrix[:, target_index])[-top_n:].tolist())
    counts = np.bincount(selected, minlength=matrix.shape[0])
    return {
        "top_slots": len(selected),
        "unique_candidates": int(np.count_nonzero(counts)),
        "maximum_target_repetitions": int(counts.max(initial=0)),
    }


def build_mia(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    *,
    minimum: int,
    maximum: int,
    total: int,
    transport_epsilon: float,
    include_domias: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    target_ids = list(targets)
    record_ids = candidates["record_id"].to_numpy(dtype=np.int64)
    fitted_views = [fit_view(background, spec) for spec in MIA_VIEW_SPECS]
    kde = KDEEvidence(background, candidates) if include_domias else None
    raw_affinity = np.empty((len(candidates), len(target_ids)), dtype=float)
    target_diagnostics: dict[str, Any] = {}

    for target_index, target_id in enumerate(target_ids):
        released = targets[target_id]
        components: list[np.ndarray] = []
        weights: list[float] = []
        component_summary: dict[str, Any] = {}
        base_weights = (0.20, 0.20, 0.30, 0.20)
        for fitted_view, weight in zip(fitted_views, base_weights, strict=True):
            component = linkage_component(released, candidates, background, fitted_view)
            ranked = strict_rank(component["strength"], record_ids)
            components.append(ranked)
            weights.append(weight)
            component_summary[fitted_view.spec.name] = {
                "median_distance": float(np.median(component["distance"])),
                "p99_strength": float(np.quantile(component["strength"], 0.99)),
                "sex_or_global_fallback_rows": int(np.count_nonzero(component["fallback"])),
            }
        if kde is not None:
            kde_evidence = kde.score(released)
            if float(np.ptp(kde_evidence)) > 1e-12:
                components.append(strict_rank(kde_evidence, record_ids))
                weights.append(0.10)
                component_summary["domias_log_evidence"] = {
                    "median": float(np.median(kde_evidence)),
                    "p99": float(np.quantile(kde_evidence, 0.99)),
                }
            else:
                component_summary["domias_log_evidence"] = {"neutral": True}
        component_matrix = np.column_stack(components)
        raw_affinity[:, target_index] = np.average(component_matrix, axis=1, weights=weights)
        target_diagnostics[target_id] = {
            "rows": len(released),
            "duplicate_rows": int(released.duplicated().sum()),
            "exact_candidate_match_rate": exact_match_rate(candidates, released),
            "exact_background_match_rate": exact_match_rate(background, released),
            "components": component_summary,
        }

    row_median = np.median(raw_affinity, axis=1, keepdims=True)
    residual = raw_affinity - row_median
    affinity = np.empty_like(raw_affinity)
    for target_index in range(len(target_ids)):
        if len(target_ids) > 1:
            other_max = np.max(np.delete(raw_affinity, target_index, axis=1), axis=1)
        else:
            other_max = np.zeros(len(candidates), dtype=float)
        margin = raw_affinity[:, target_index] - other_max
        residual_rank = strict_rank(residual[:, target_index], record_ids)
        raw_rank = strict_rank(raw_affinity[:, target_index], record_ids)
        margin_rank = strict_rank(margin, record_ids)
        affinity[:, target_index] = 0.55 * residual_rank + 0.25 * raw_rank + 0.20 * margin_rank

    affinity += deterministic_jitter(record_ids, target_ids)
    clipped = np.clip(affinity, 1.0 / (len(candidates) + 1.0), len(candidates) / (len(candidates) + 1.0))
    utility = np.log(clipped) - np.log1p(-clipped)
    assigned, target_counts = constrained_capacity_assignment(
        utility, minimum=minimum, maximum=maximum, total=total
    )
    transport, transport_diagnostics = sinkhorn_transport(
        utility,
        target_counts,
        null_count=len(candidates) - total,
        epsilon=transport_epsilon,
    )

    transport_confidence = np.empty_like(transport)
    unconstrained_confidence = np.empty_like(raw_affinity)
    hard_confidence = np.empty_like(raw_affinity)
    for target_index in range(len(target_ids)):
        transport_confidence[:, target_index] = strict_rank(transport[:, target_index], record_ids)
        unconstrained_confidence[:, target_index] = strict_rank(
            raw_affinity[:, target_index], record_ids
        )
        base = strict_rank(affinity[:, target_index], record_ids)
        is_assigned = assigned == target_index
        hard_confidence[:, target_index] = np.where(
            is_assigned, 0.51 + 0.49 * base, 0.49 * base
        )

    def as_frame(matrix: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame({"record_id": record_ids})
        for index, target_id in enumerate(target_ids):
            frame[target_id] = matrix[:, index]
        return frame

    variants = {
        "transport": as_frame(transport_confidence),
        "hard_capacity": as_frame(hard_confidence),
        "unconstrained": as_frame(unconstrained_confidence),
    }
    diagnostics = {
        "n_candidates": len(candidates),
        "n_targets": len(target_ids),
        "capacity": {
            "minimum": minimum,
            "maximum": maximum,
            "total_shared": total,
            "unassigned": int((assigned < 0).sum()),
            "per_target": {
                target_id: int(target_counts[index]) for index, target_id in enumerate(target_ids)
            },
        },
        "transport": transport_diagnostics,
        "top10_before_constraint": top_concentration(unconstrained_confidence),
        "top10_after_transport": top_concentration(transport_confidence),
        "targets": target_diagnostics,
    }
    return variants["transport"], variants, diagnostics


def choose_low_hit_time(background: pd.DataFrame, *, horizon: float, tau: float) -> float:
    values = np.sort(background["time"].to_numpy(dtype=float))
    grid_size = max(int(round(horizon * 100.0)), 1)
    grid = np.linspace(0.0, horizon, grid_size + 1)
    lower = np.searchsorted(values, grid - tau, side="left")
    upper = np.searchsorted(values, grid + tau, side="right")
    rates = (upper - lower) / max(len(values), 1)
    return float(grid[int(np.argmin(rates))])


def aia_pair_cost(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    fitted_views: list[FittedView],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    strengths: list[np.ndarray] = []
    pair_distance: np.ndarray | None = None
    view_summaries: dict[str, Any] = {}
    for fitted_view in fitted_views:
        component = linkage_component(released, challenge, background, fitted_view)
        strengths.append(component["strength"])
        query_x = fitted_view.transform(challenge)
        released_x = fitted_view.transform(released)
        distance = cdist(query_x, released_x, metric="euclidean") / math.sqrt(query_x.shape[1])
        pair_distance = distance if pair_distance is None else np.minimum(pair_distance, distance)
        view_summaries[fitted_view.spec.name] = {
            "median_nearest_distance": float(np.median(component["distance"])),
            "p05_tail_probability": float(np.quantile(component["tail_probability"], 0.05)),
            "fallback_rows": int(np.count_nonzero(component["fallback"])),
        }
    assert pair_distance is not None
    combined_strength = np.max(np.column_stack(strengths), axis=1)
    sex_mismatch = (
        challenge["sex"].to_numpy()[:, None] != released["sex"].to_numpy()[None, :]
    )
    prefecture_mismatch = (
        challenge["prefecture"].astype(str).to_numpy()[:, None]
        != released["prefecture"].astype(str).to_numpy()[None, :]
    )
    cost = pair_distance + 1.50 * sex_mismatch + 0.75 * prefecture_mismatch
    cost -= 0.35 * combined_strength[:, None]

    challenge_hash = pd.util.hash_pandas_object(
        challenge[[*QI_COLUMNS, *BLOCK_COLUMNS]], index=False
    ).to_numpy(dtype=np.uint64)
    release_hash = pd.util.hash_pandas_object(released[list(MIA_REQUIRED_COLUMNS)], index=False).to_numpy(
        dtype=np.uint64
    )
    jitter = (challenge_hash[:, None] ^ release_hash[None, :]) % np.uint64(1_000_003)
    cost += jitter.astype(float) / 1_000_003.0 * 1e-10
    return cost, combined_strength, view_summaries


def cardinality_link_assignment(pair_cost: np.ndarray, active_count: int) -> np.ndarray:
    """Select exactly active_count unique challenge-to-release links."""
    n_queries, n_released = pair_cost.shape
    if active_count < 0 or active_count > min(n_queries, n_released):
        raise ValueError("invalid AIA active_count")
    n_dummy = n_queries - active_count
    n_vacancies = n_released - active_count
    size = n_queries + n_vacancies
    if n_released + n_dummy != size:
        raise AssertionError("AIA cardinality construction is inconsistent")
    prohibited = 1e9
    augmented = np.full((size, size), prohibited, dtype=float)
    augmented[:n_queries, :n_released] = pair_cost
    if n_dummy:
        augmented[:n_queries, n_released:] = 0.0
    if n_vacancies:
        augmented[n_queries:, :n_released] = 0.0
    rows, columns = linear_sum_assignment(augmented)
    if np.any(augmented[rows, columns] >= prohibited / 2):
        raise RuntimeError("AIA cardinality assignment used a prohibited edge")
    assigned = np.full(n_queries, -1, dtype=int)
    real = (rows < n_queries) & (columns < n_released)
    assigned[rows[real]] = columns[real]
    if int((assigned >= 0).sum()) != active_count:
        raise RuntimeError("AIA cardinality assignment produced the wrong number of links")
    if len(np.unique(assigned[assigned >= 0])) != active_count:
        raise RuntimeError("AIA cardinality assignment reused a released row")
    return assigned


def independent_link_assignment(
    pair_cost: np.ndarray, active_count: int, tie_key: np.ndarray
) -> np.ndarray:
    """Select the best active queries independently, allowing released-row reuse."""
    n_queries, n_released = pair_cost.shape
    if active_count < 0 or active_count > n_queries:
        raise ValueError("invalid independent AIA active_count")
    nearest = np.argmin(pair_cost, axis=1)
    nearest_cost = pair_cost[np.arange(n_queries), nearest]
    order = np.lexsort((np.asarray(tie_key), nearest_cost))
    active_rows = order[:active_count]
    assigned = np.full(n_queries, -1, dtype=int)
    assigned[active_rows] = nearest[active_rows]
    if np.any(assigned[assigned >= 0] >= n_released):
        raise AssertionError("independent AIA assignment produced an invalid row")
    return assigned


def fraction_label(fraction: float) -> str:
    return f"{fraction:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_aia(
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    main_fraction: float,
    variant_fractions: list[float],
    abstain_time: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    fitted_views = [fit_view(background, spec) for spec in AIA_VIEW_SPECS]
    all_fractions = sorted(set([main_fraction, *variant_fractions]))
    frames: dict[float, list[pd.DataFrame]] = {fraction: [] for fraction in all_fractions}
    independent_fractions = sorted(set([main_fraction, 1.0]))
    independent_frames: dict[float, list[pd.DataFrame]] = {
        fraction: [] for fraction in independent_fractions
    }
    target_diagnostics: dict[str, Any] = {}

    for target_id, released in targets.items():
        challenge = challenges[target_id]
        pair_cost, strength, view_summary = aia_pair_cost(
            released, challenge, background, fitted_views
        )
        challenge_ids = challenge["challenge_row_id"].to_numpy()
        target_fraction_summary: dict[str, Any] = {}
        for fraction in all_fractions:
            active_count = int(round(len(challenge) * fraction))
            assigned = cardinality_link_assignment(pair_cost, active_count)
            time_hat = np.full(len(challenge), abstain_time, dtype=float)
            active = assigned >= 0
            if active.any():
                time_hat[active] = released["time"].to_numpy(dtype=float)[assigned[active]]
            frames[fraction].append(
                pd.DataFrame(
                    {
                        "target_k": int(target_id),
                        "challenge_row_id": challenge_ids,
                        "time_hat": time_hat,
                    }
                )
            )
            target_fraction_summary[fraction_label(fraction)] = {
                "active_rows": int(active.sum()),
                "median_selected_link_strength": (
                    float(np.median(strength[active])) if active.any() else None
                ),
            }
        independent_summary: dict[str, Any] = {}
        for fraction in independent_fractions:
            active_count = int(round(len(challenge) * fraction))
            assigned = independent_link_assignment(pair_cost, active_count, challenge_ids)
            time_hat = np.full(len(challenge), abstain_time, dtype=float)
            active = assigned >= 0
            if active.any():
                time_hat[active] = released["time"].to_numpy(dtype=float)[assigned[active]]
            independent_frames[fraction].append(
                pd.DataFrame(
                    {
                        "target_k": int(target_id),
                        "challenge_row_id": challenge_ids,
                        "time_hat": time_hat,
                    }
                )
            )
            independent_summary[fraction_label(fraction)] = {
                "active_rows": int(active.sum()),
                "unique_released_rows": int(np.unique(assigned[active]).size),
            }
        target_diagnostics[target_id] = {
            "challenge_rows": len(challenge),
            "released_rows": len(released),
            "views": view_summary,
            "fractions": target_fraction_summary,
            "independent_fractions": independent_summary,
        }

    variants = {
        fraction_label(fraction): pd.concat(target_frames, ignore_index=True).sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        )
        for fraction, target_frames in frames.items()
    }
    variants.update(
        {
            f"independent_{fraction_label(fraction)}": pd.concat(
                target_frames, ignore_index=True
            ).sort_values(["target_k", "challenge_row_id"], kind="mergesort")
            for fraction, target_frames in independent_frames.items()
        }
    )
    main = variants[fraction_label(main_fraction)]
    diagnostics = {
        "abstain_time": abstain_time,
        "main_fraction": main_fraction,
        "variant_fractions": all_fractions,
        "targets": target_diagnostics,
    }
    return main, variants, diagnostics


def frame_profile(frame: pd.DataFrame, id_column: str | None = None) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    profile = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "null_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
    }
    if id_column is not None:
        profile["duplicate_ids"] = int(frame[id_column].duplicated().sum())
    return profile


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"{label} に必須列がありません: {missing}")


def load_inputs(
    dist: Path,
    targets_dir: Path,
    exclude_target_ids: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    b_path = dist / "B.csv"
    if not b_path.exists():
        b_path = dist / "B_practice.csv"
    if not b_path.exists():
        raise SystemExit(f"B.csv / B_practice.csv が見つかりません: {dist}")
    background_path = dist / "A_bg.csv"
    if not background_path.exists():
        raise SystemExit(f"A_bg.csv が見つかりません: {background_path}")
    columns_path = targets_dir / "mia_columns.json"
    if not columns_path.exists():
        raise SystemExit(f"mia_columns.json が見つかりません: {columns_path}")

    candidates = pd.read_csv(b_path)
    background = pd.read_csv(background_path)
    require_columns(candidates, ("record_id", *MIA_REQUIRED_COLUMNS), b_path.name)
    require_columns(background, ("record_id", *MIA_REQUIRED_COLUMNS), "A_bg.csv")
    if candidates["record_id"].duplicated().any():
        raise SystemExit("Bのrecord_idが一意ではありません")
    target_ids = [
        str(value)
        for value in json.loads(columns_path.read_text(encoding="utf-8"))["target_columns"]
    ]
    if len(target_ids) != len(set(target_ids)):
        raise SystemExit("mia_columns.json のtarget_columnsが一意ではありません")
    excluded = {str(value) for value in exclude_target_ids}
    unknown_exclusions = excluded - set(target_ids)
    if unknown_exclusions:
        raise SystemExit(
            "--exclude-targets にmia_columns.json未掲載の対象があります: "
            f"{sorted(unknown_exclusions, key=int)}"
        )
    target_ids = [target_id for target_id in target_ids if target_id not in excluded]
    if not target_ids:
        raise SystemExit("除外後の攻撃対象がありません")

    targets: dict[str, pd.DataFrame] = {}
    challenges: dict[str, pd.DataFrame] = {}
    target_profiles: dict[str, Any] = {}
    for target_id in target_ids:
        target_path = targets_dir / f"C_{target_id}.csv"
        challenge_path = targets_dir / f"aia_challenge_{target_id}.csv"
        if not target_path.exists() or not challenge_path.exists():
            raise SystemExit(f"対象{target_id}のCまたはchallengeがありません: {targets_dir}")
        released = pd.read_csv(target_path)
        challenge = pd.read_csv(challenge_path)
        require_columns(released, MIA_REQUIRED_COLUMNS, target_path.name)
        require_columns(challenge, AIA_REQUIRED_COLUMNS, challenge_path.name)
        if challenge["challenge_row_id"].duplicated().any():
            raise SystemExit(f"{challenge_path.name} のchallenge_row_idが一意ではありません")
        for label, frame in ((target_path.name, released), (challenge_path.name, challenge)):
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
            if frame.isna().any().any() or not np.isfinite(numeric).all():
                raise SystemExit(f"{label} に欠測または非有限数があります")
        targets[target_id] = released.reset_index(drop=True)
        challenges[target_id] = challenge.reset_index(drop=True)
        target_profiles[target_id] = {
            "released": frame_profile(released),
            "challenge": frame_profile(challenge, "challenge_row_id"),
        }

    quality = {
        "candidate_source": str(b_path.resolve()),
        "targets_source": str(targets_dir.resolve()),
        "excluded_target_ids": sorted(excluded, key=int),
        "candidates": frame_profile(candidates, "record_id"),
        "background": frame_profile(background, "record_id"),
        "targets": target_profiles,
    }
    if quality["candidates"]["null_cells"] or quality["background"]["null_cells"]:
        raise SystemExit("BまたはA_bgに欠測があります")
    return candidates, background, targets, challenges, quality


def load_phase_parameters(dist: Path) -> tuple[float, float]:
    config_path = dist / "scoring_config.yaml"
    if not config_path.exists():
        return 10.0, 0.5
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    horizon = float(config.get("onset", {}).get("horizon_years", 10.0))
    tau = float(config.get("aia", {}).get("tau_time", 0.5))
    return horizon, tau


def write_outputs(
    out_dir: Path,
    f_mia: pd.DataFrame,
    mia_variants: dict[str, pd.DataFrame],
    f_aia: pd.DataFrame,
    aia_variants: dict[str, pd.DataFrame],
    diagnostics: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    f_mia.to_csv(out_dir / "F_mia.csv", index=False)
    f_aia.to_csv(out_dir / "F_aia.csv", index=False)
    for name, frame in mia_variants.items():
        frame.to_csv(variants_dir / f"F_mia_{name}.csv", index=False)
    for name, frame in aia_variants.items():
        frame.to_csv(variants_dir / f"F_aia_fraction_{name}.csv", index=False)
    (out_dir / "diagnostics.json").write_text(
        json.dumps(_clean_json(diagnostics), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = (args.targets or (dist / "targets")).resolve()
    out_dir = args.out_dir.resolve()
    candidates, background, targets, challenges, quality = load_inputs(
        dist,
        targets_dir,
        exclude_target_ids=args.exclude_targets,
    )
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
        raise SystemExit("--mia-total-shared がBの行数を超えています")

    fraction_values = [
        float(value.strip())
        for value in args.aia_variant_fractions.split(",")
        if value.strip()
    ]
    if any(value < 0.0 or value > 1.0 for value in fraction_values):
        raise SystemExit("--aia-variant-fractions は0〜1の値です")
    horizon, tau = load_phase_parameters(dist)
    abstain_time = (
        float(args.aia_abstain_time)
        if args.aia_abstain_time is not None
        else choose_low_hit_time(background, horizon=horizon, tau=tau)
    )
    if not np.isfinite(abstain_time) or abstain_time < 0.0:
        raise SystemExit("AIA abstain timeは有限の非負値である必要があります")

    print(f"入力確認: candidates={len(candidates)}, targets={len(targets)}, A_bg={len(background)}")
    print("MIA: 多視点リンケージと全標的同時制約を計算中...")
    f_mia, mia_variants, mia_diagnostics = build_mia(
        candidates,
        background,
        targets,
        minimum=args.mia_min_per_target,
        maximum=args.mia_max_per_target,
        total=args.mia_total_shared,
        transport_epsilon=args.mia_transport_epsilon,
        include_domias=not args.skip_domias,
    )
    print("AIA: 密度補正リンケージと一意割当を計算中...")
    f_aia, aia_variants, aia_diagnostics = build_aia(
        background,
        targets,
        challenges,
        main_fraction=args.aia_active_fraction,
        variant_fractions=fraction_values,
        abstain_time=abstain_time,
    )
    diagnostics = {
        "report_type": "authorized_local_privacy_competition_statistical_analysis",
        "network_access": False,
        "quality": quality,
        "parameters": {
            "horizon_years": horizon,
            "aia_tau_time": tau,
            "aia_active_fraction": args.aia_active_fraction,
            "excluded_target_ids": list(args.exclude_targets),
            "mia_min_per_target": args.mia_min_per_target,
            "mia_max_per_target": args.mia_max_per_target,
            "mia_total_shared": args.mia_total_shared,
            "mia_transport_epsilon": args.mia_transport_epsilon,
            "domias_included": not args.skip_domias,
        },
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only membership and AIA truth are not present locally, so official attack scores cannot be computed.",
            "Practice targets are toy data and must not be treated as evidence of preliminary-round leaderboard performance.",
            "The program only reads distributed local files and creates participant-side predictions.",
        ],
    }
    write_outputs(out_dir, f_mia, mia_variants, f_aia, aia_variants, diagnostics)
    print(f"完了: {out_dir}")
    print(f"  F_mia.csv: {len(f_mia)} rows x {len(f_mia.columns)} columns")
    print(f"  F_aia.csv: {len(f_aia)} rows")
    print(f"  AIA非選択行の低的中time: {abstain_time:.2f} years")


if __name__ == "__main__":
    main()
