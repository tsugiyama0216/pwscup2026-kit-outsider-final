"""Build an independent Daybreak Blue MIA/AIA candidate for PWS Cup 2026.

This module intentionally does not call the v1, v2, v3, or conservative
Daybreak attack builders.  Its two algorithms are separate:

* MIA: reciprocal nearest-neighbour evidence in four robust views, calibrated
  against the 163 rule-derived known negatives on a target-comparable
  conformal scale.  It uses neither DOMIAS nor transport.
* AIA: a public outcome-integrity router over two reusable multiview codebook
  experts (no Hungarian linkage).  It keeps residual event-time recovery when
  a release preserves the public QI/event relation, and switches the complete
  target block to a censoring-horizon expert when that relation is degraded.

Only organizer-distributed local files are read.  There is no network,
authentication, token, packaging, or submission code in this flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
if str(STARTER_SRC) not in sys.path:
    sys.path.insert(0, str(STARTER_SRC))

from pwscup2026.kit import validate as official_validate  # noqa: E402


MODEL_ID = "gpt-daybreak-blue-latest"
ALGORITHM_ID = "daybreak_blue_reciprocal_conformal_integrity_router_v1"
EXPECTED_SELF_TARGET = "24"
EXPECTED_TARGET_IDS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
EXPECTED_MIA_TOTAL_SHARED = 508
EXPECTED_CANDIDATE_ROWS = 1_049
EXPECTED_CHALLENGE_ROWS = 326
EXPECTED_AIA_MEMBERS = 163
AIA_INTEGRITY_AUC_THRESHOLD = 0.60
MANIFEST_PATTERN = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")

QI_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
MIA_COLUMNS = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
)
CONTINUOUS_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time")
LOG_COLUMNS = frozenset(("TG", "ALT", "FPG"))
BINARY_COLUMNS = ("sex", "smoking", "onset", "death")

@dataclass(frozen=True)
class ReciprocalEvidenceView:
    name: str
    columns: tuple[str, ...]
    log_columns: tuple[str, ...] = ()


MIA_RECIPROCAL_VIEWS = (
    ReciprocalEvidenceView(
        "qi", ("age", "SBP", "TG", "HDL", "ALT"), ("TG", "ALT")
    ),
    ReciprocalEvidenceView("stable", ("age", "SBP", "HDL")),
    ReciprocalEvidenceView(
        "clinical",
        ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"),
        ("TG", "ALT", "FPG"),
    ),
    ReciprocalEvidenceView(
        "outcome",
        (
            "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time",
            "smoking", "onset", "death",
        ),
        ("TG", "ALT", "FPG"),
    ),
)
AIA_SUBSPACES: tuple[tuple[str, ...], ...] = (
    ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"),
    ("age", "SBP", "sex", "prefecture"),
    ("TG", "HDL", "ALT", "sex"),
    ("age", "TG", "ALT", "prefecture"),
    ("SBP", "HDL", "ALT", "sex", "prefecture"),
    ("age", "SBP", "TG", "HDL", "ALT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daybreak Blue独立方式でPWSCup2026予備戦team24のMIA/AIA候補を生成します。"
        )
    )
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--self-target-id", default=EXPECTED_SELF_TARGET)
    parser.add_argument("--exclude-targets", default=EXPECTED_SELF_TARGET)
    parser.add_argument("--mia-total-shared", type=int, default=EXPECTED_MIA_TOTAL_SHARED)
    parser.add_argument("--aia-member-count", type=int, default=EXPECTED_AIA_MEMBERS)
    parser.add_argument("--aia-residual-weight", type=float, default=0.25)
    parser.add_argument("--aia-background-neighbours", type=int, default=48)
    parser.add_argument(
        "--aia-integrity-auc-threshold",
        type=float,
        default=AIA_INTEGRITY_AUC_THRESHOLD,
    )
    parser.add_argument(
        "--acknowledge-prelim-private-block-rule", action="store_true"
    )
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    args.self_target_id = str(args.self_target_id).strip()
    args.exclude_targets = tuple(
        value.strip() for value in str(args.exclude_targets).split(",") if value.strip()
    )
    validate_settings(args)
    return args


def validate_settings(args: argparse.Namespace) -> None:
    if args.self_target_id != EXPECTED_SELF_TARGET:
        raise ValueError("この独立フローは予備戦team24専用です")
    if args.exclude_targets != (EXPECTED_SELF_TARGET,):
        raise ValueError("除外targetは自チーム24だけである必要があります")
    if args.mia_total_shared != EXPECTED_MIA_TOTAL_SHARED:
        raise ValueError("対象20チームの共有総数は508である必要があります")
    if args.aia_member_count != EXPECTED_AIA_MEMBERS:
        raise ValueError("各AIA challengeの推定member数は163である必要があります")
    if not args.acknowledge_prelim_private_block_rule:
        raise ValueError(
            "--acknowledge-prelim-private-block-rule が必要です"
        )
    if not 0.0 <= args.aia_residual_weight <= 1.0:
        raise ValueError("AIA residual weightは0〜1です")
    if args.aia_background_neighbours < 2:
        raise ValueError("AIA background neighboursは2以上です")
    if not 0.0 <= args.aia_integrity_auc_threshold <= 1.0:
        raise ValueError("AIA integrity AUC thresholdは0〜1です")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_manifest(targets_dir: Path) -> dict[str, Any]:
    manifest_path = targets_dir / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        raise ValueError(f"SHA256SUMS.txtが見つかりません: {manifest_path}")
    entries: list[tuple[str, str]] = []
    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = MANIFEST_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"SHA256SUMS.txtの{line_no}行目が不正です")
        entries.append((match.group(1).lower(), match.group(2)))
    missing: list[str] = []
    changed: list[str] = []
    for expected, relative in entries:
        path = targets_dir / relative
        if not path.is_file():
            missing.append(relative)
        elif sha256_file(path) != expected:
            changed.append(relative)
    if not entries or missing or changed:
        raise ValueError(
            f"配布manifest検証失敗: entries={len(entries)}, missing={missing}, changed={changed}"
        )
    return {"manifest": str(manifest_path), "verified_files": len(entries)}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}に必須列がありません: {missing}")


def _validate_frame(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    _require_columns(frame, columns, label)
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{label}に欠測があります")
    numeric = frame[list(columns)].select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if numeric.size and not np.isfinite(numeric).all():
        raise ValueError(f"{label}に非有限数があります")


def load_inputs(
    dist: Path, targets_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]:
    candidates = pd.read_csv(dist / "B.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    _validate_frame(candidates, ("record_id", *MIA_COLUMNS), "B.csv")
    _validate_frame(background, ("record_id", *MIA_COLUMNS), "A_bg.csv")
    if len(candidates) != EXPECTED_CANDIDATE_ROWS:
        raise ValueError(f"B.csvは{EXPECTED_CANDIDATE_ROWS}行である必要があります")
    if candidates["record_id"].duplicated().any():
        raise ValueError("B.csvのrecord_idが重複しています")

    target_config = json.loads((targets_dir / "mia_columns.json").read_text(encoding="utf-8"))
    configured = [str(value) for value in target_config["target_columns"]]
    selected = [value for value in configured if value != EXPECTED_SELF_TARGET]
    selected = sorted(selected, key=int)
    if tuple(selected) != EXPECTED_TARGET_IDS:
        raise ValueError(
            f"target集合が不正です: expected={EXPECTED_TARGET_IDS}, observed={selected}"
        )

    targets: dict[str, pd.DataFrame] = {}
    challenges: dict[str, pd.DataFrame] = {}
    for target_id in selected:
        released = pd.read_csv(targets_dir / f"C_{target_id}.csv").reset_index(drop=True)
        challenge = pd.read_csv(
            targets_dir / f"aia_challenge_{target_id}.csv"
        ).reset_index(drop=True)
        _validate_frame(released, MIA_COLUMNS, f"C_{target_id}.csv")
        _validate_frame(
            challenge, ("challenge_row_id", *QI_COLUMNS), f"aia_challenge_{target_id}.csv"
        )
        if len(challenge) != EXPECTED_CHALLENGE_ROWS:
            raise ValueError(f"target {target_id}のAIA challenge行数が不正です")
        if challenge["challenge_row_id"].duplicated().any():
            raise ValueError(f"target {target_id}のchallenge_row_idが重複しています")
        targets[target_id] = released
        challenges[target_id] = challenge
    quality = {
        "candidate_rows": len(candidates),
        "background_rows": len(background),
        "target_count": len(targets),
        "challenge_rows_per_target": EXPECTED_CHALLENGE_ROWS,
        "candidate_duplicate_ids": int(candidates["record_id"].duplicated().sum()),
        "candidate_null_cells": int(candidates.isna().sum().sum()),
        "background_null_cells": int(background.isna().sum().sum()),
    }
    return candidates, background, targets, challenges, quality


@dataclass(frozen=True)
class RobustAtomEncoder:
    centers: dict[str, float]
    scales: dict[str, float]
    prefectures: tuple[str, ...]

    @classmethod
    def fit(cls, reference: pd.DataFrame) -> "RobustAtomEncoder":
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in CONTINUOUS_COLUMNS:
            values = reference[column].to_numpy(dtype=float)
            if column in LOG_COLUMNS:
                values = np.log1p(np.clip(values, 0.0, None))
            center = float(np.median(values))
            q25, q75 = np.quantile(values, [0.25, 0.75])
            scale = float((q75 - q25) / 1.349)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(values, ddof=0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            centers[column] = center
            scales[column] = scale
        prefectures = tuple(sorted(reference["prefecture"].astype(str).unique()))
        return cls(centers=centers, scales=scales, prefectures=prefectures)

    def atoms(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for column in CONTINUOUS_COLUMNS:
            if column not in frame.columns:
                continue
            values = frame[column].to_numpy(dtype=float)
            if column in LOG_COLUMNS:
                values = np.log1p(np.clip(values, 0.0, None))
            normalized = (values - self.centers[column]) / self.scales[column]
            result[column] = np.clip(normalized, -8.0, 8.0)[:, None]
        for column in BINARY_COLUMNS:
            if column not in frame.columns:
                continue
            result[column] = 0.75 * frame[column].to_numpy(dtype=float)[:, None]
        if "prefecture" in frame.columns:
            lookup = {value: index for index, value in enumerate(self.prefectures)}
            one_hot = np.zeros((len(frame), len(self.prefectures)), dtype=float)
            for row, value in enumerate(frame["prefecture"].astype(str)):
                index = lookup.get(value)
                if index is not None:
                    one_hot[row, index] = 0.70 / math.sqrt(2.0)
            result["prefecture"] = one_hot
        return result

    def matrix(self, frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
        atoms = self.atoms(frame)
        matrix = np.concatenate([atoms[column] for column in columns], axis=1)
        return matrix / math.sqrt(max(len(columns), 1))


def strict_rank(values: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keys = np.asarray(tie_key)
    if values.ndim != 1 or len(values) != len(keys):
        raise ValueError("strict_rank input shape mismatch")
    safe = np.nan_to_num(values, nan=-1e300, posinf=1e300, neginf=-1e300)
    order = np.lexsort((keys, safe))
    ranks = np.empty(len(safe), dtype=float)
    ranks[order] = np.arange(1, len(safe) + 1, dtype=float)
    return ranks / (len(safe) + 1.0)


def _knn_summary(
    reference_x: np.ndarray, query_x: np.ndarray, neighbours: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = min(max(int(neighbours), 1), len(reference_x))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    model.fit(reference_x)
    distance, index = model.kneighbors(query_x)
    return distance[:, 0], distance.mean(axis=1), index[:, 0]


def discover_known_private(
    candidates: pd.DataFrame, self_challenge: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_columns(candidates, QI_COLUMNS, "B.csv")
    _require_columns(self_challenge, QI_COLUMNS, "self AIA challenge")
    candidate_keys = pd.MultiIndex.from_frame(candidates[list(QI_COLUMNS)])
    challenge_keys = pd.MultiIndex.from_frame(self_challenge[list(QI_COLUMNS)])
    if candidate_keys.has_duplicates:
        raise ValueError("B.csvのself-private照合キーが一意ではありません")
    challenge_set = set(challenge_keys.tolist())
    mask = np.asarray([key in challenge_set for key in candidate_keys.tolist()], dtype=bool)
    if int(mask.sum()) != EXPECTED_AIA_MEMBERS:
        raise ValueError(
            f"self-private一致は{EXPECTED_AIA_MEMBERS}件を期待します: {int(mask.sum())}"
        )
    return mask, {
        "match_columns": list(QI_COLUMNS),
        "known_negative_rows": int(mask.sum()),
        "eligible_rows": int((~mask).sum()),
    }


def build_mia_independent(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    known_negative: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    record_ids = candidates["record_id"].to_numpy(dtype=np.int64)
    target_ids = list(targets)
    known_negative = np.asarray(known_negative, dtype=bool)
    if known_negative.shape != (len(candidates),) or int(known_negative.sum()) != EXPECTED_AIA_MEMBERS:
        raise ValueError("MIA known-negative mask must contain exactly 163 rows")

    def fit_view(view: ReciprocalEvidenceView):
        values = background.loc[:, view.columns].to_numpy(dtype=float, copy=True)
        for column in view.log_columns:
            index = view.columns.index(column)
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
        center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        robust = (q75 - q25) / 1.349
        standard = np.std(values, axis=0, ddof=0)
        scale = np.where(robust > 1e-9, robust, standard)
        scale = np.where(scale > 1e-9, scale, 1.0)

        def transform(frame: pd.DataFrame) -> np.ndarray:
            matrix = frame.loc[:, view.columns].to_numpy(dtype=float, copy=True)
            for column in view.log_columns:
                index = view.columns.index(column)
                matrix[:, index] = np.log1p(np.clip(matrix[:, index], 0.0, None))
            return (matrix - center) / scale

        return transform

    def tuple_key(value: Any) -> tuple[Any, ...]:
        return value if isinstance(value, tuple) else (value,)

    def group_indices(frame: pd.DataFrame) -> dict[tuple[Any, ...], np.ndarray]:
        groups = frame.groupby(["sex", "prefecture"], sort=False, dropna=False).indices
        return {tuple_key(key): np.asarray(value, dtype=int) for key, value in groups.items()}

    def sex_indices(frame: pd.DataFrame) -> dict[tuple[Any, ...], np.ndarray]:
        groups = frame.groupby(["sex"], sort=False, dropna=False).indices
        return {tuple_key(key): np.asarray(value, dtype=int) for key, value in groups.items()}

    def blockwise_nearest(
        reference: pd.DataFrame,
        reference_x: np.ndarray,
        query: pd.DataFrame,
        query_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        exact = group_indices(reference)
        by_sex = sex_indices(reference)
        query_groups = group_indices(query)
        all_reference = np.arange(len(reference), dtype=int)
        distance = np.empty(len(query), dtype=float)
        nearest = np.empty(len(query), dtype=int)
        dimension_scale = math.sqrt(reference_x.shape[1])
        for key, query_rows in query_groups.items():
            candidates_in_block = exact.get(key)
            if candidates_in_block is None or not len(candidates_in_block):
                candidates_in_block = by_sex.get((key[0],))
            if candidates_in_block is None or not len(candidates_in_block):
                candidates_in_block = all_reference
            candidates_in_block = np.asarray(candidates_in_block, dtype=int)
            model = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
            model.fit(reference_x[candidates_in_block])
            local_distance, local_index = model.kneighbors(query_x[query_rows])
            distance[query_rows] = local_distance[:, 0] / dimension_scale
            nearest[query_rows] = candidates_in_block[local_index[:, 0]]
        return distance, nearest

    def empirical_lower_tail(values: np.ndarray, null_values: np.ndarray) -> np.ndarray:
        sorted_null = np.sort(np.asarray(null_values, dtype=float))
        counts = np.searchsorted(sorted_null, np.asarray(values, dtype=float), side="right")
        return (counts + 1.0) / (len(sorted_null) + 1.0)

    def empirical_upper_tail(values: np.ndarray, null_values: np.ndarray) -> np.ndarray:
        sorted_null = np.sort(np.asarray(null_values, dtype=float))
        counts_below = np.searchsorted(
            sorted_null, np.asarray(values, dtype=float), side="left"
        )
        return (len(sorted_null) - counts_below + 1.0) / (len(sorted_null) + 1.0)

    fitted = [(view, fit_view(view)) for view in MIA_RECIPROCAL_VIEWS]
    candidate_x = {
        view.name: transform(candidates) for view, transform in fitted
    }
    n_candidates = len(candidates)
    n_targets = len(targets)
    view_count = len(MIA_RECIPROCAL_VIEWS)
    distance_evidence = np.empty((n_candidates, n_targets, view_count), dtype=float)
    reciprocal_evidence = np.empty_like(distance_evidence)
    nearest_rows = np.empty((n_candidates, n_targets, view_count), dtype=int)
    target_diagnostics: dict[str, Any] = {}

    for target_index, (target_id, released) in enumerate(targets.items()):
        target_view_details: dict[str, Any] = {}
        for view_index, (view, transform) in enumerate(fitted):
            released_x = transform(released)
            cand_x = candidate_x[view.name]
            candidate_distance, candidate_nearest = blockwise_nearest(
                released, released_x, candidates, cand_x
            )
            release_distance, release_nearest_candidate = blockwise_nearest(
                candidates, cand_x, released, released_x
            )
            reverse_distance = release_distance[candidate_nearest]
            reciprocal_ratio = candidate_distance / np.clip(
                reverse_distance, 1e-12, None
            )
            reciprocal_ratio *= np.where(
                release_nearest_candidate[candidate_nearest]
                == np.arange(n_candidates),
                0.5,
                1.0,
            )
            distance_p = empirical_lower_tail(
                candidate_distance, candidate_distance[known_negative]
            )
            reciprocal_p = empirical_lower_tail(
                reciprocal_ratio, reciprocal_ratio[known_negative]
            )
            distance_evidence[:, target_index, view_index] = -np.log(distance_p)
            reciprocal_evidence[:, target_index, view_index] = -np.log(reciprocal_p)
            nearest_rows[:, target_index, view_index] = candidate_nearest
            target_view_details[view.name] = {
                "median_distance": float(np.median(candidate_distance)),
                "p01_distance": float(np.quantile(candidate_distance, 0.01)),
                "reciprocal_matches": int(
                    np.count_nonzero(
                        release_nearest_candidate[candidate_nearest]
                        == np.arange(n_candidates)
                    )
                ),
            }
        target_diagnostics[target_id] = {
            "released_rows": len(released),
            "duplicate_rows": int(released.duplicated().sum()),
            "views": target_view_details,
        }

    sorted_distance = np.sort(distance_evidence, axis=2)
    robust_distance = (
        0.55 * sorted_distance[:, :, -2:].mean(axis=2)
        + 0.45 * np.median(distance_evidence, axis=2)
    )
    robust_reciprocal = np.sort(reciprocal_evidence, axis=2)[:, :, -2:].mean(axis=2)
    coherence = np.empty((n_candidates, n_targets), dtype=float)
    for target_index in range(n_targets):
        for row_index in range(n_candidates):
            _, counts = np.unique(
                nearest_rows[row_index, target_index], return_counts=True
            )
            coherence[row_index, target_index] = float(counts.max()) / view_count
    combined = robust_distance + 0.20 * robust_reciprocal + 0.25 * coherence

    log_e = np.empty_like(combined)
    for target_index in range(n_targets):
        p_value = empirical_upper_tail(
            combined[:, target_index], combined[known_negative, target_index]
        )
        log_e[:, target_index] = -np.log(p_value)
    other_max = np.empty_like(log_e)
    for target_index in range(n_targets):
        other_max[:, target_index] = np.max(
            np.delete(log_e, target_index, axis=1), axis=1
        )
    margin = log_e - other_max
    winner = np.argmax(log_e, axis=1)
    winner_gate = winner[:, None] == np.arange(n_targets)[None, :]

    raw_variants = {
        "daybreak_reciprocal_conformal_plain": log_e,
        "daybreak_reciprocal_conformal_margin_0p25": log_e + 0.25 * margin,
        "daybreak_reciprocal_conformal_margin_0p50": log_e + 0.50 * margin,
        "daybreak_reciprocal_conformal_winner_gate": (
            log_e + 2.0 * winner_gate + 0.25 * margin
        ),
        "daybreak_reciprocal_distance_only": (
            robust_distance + 0.35 * robust_reciprocal
        ),
    }
    ranked_variants: dict[str, np.ndarray] = {}
    for name, raw in raw_variants.items():
        ranked = np.empty_like(raw)
        for target_index in range(n_targets):
            values = raw[:, target_index].copy()
            values[known_negative] = values.min() - max(float(np.ptp(values)), 1.0)
            ranked[:, target_index] = strict_rank(values, record_ids)
        ranked_variants[name] = ranked

    def as_frame(matrix: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame({"record_id": record_ids})
        for index, target_id in enumerate(target_ids):
            frame[target_id] = matrix[:, index]
        return frame

    variants = {name: as_frame(values) for name, values in ranked_variants.items()}
    main_name = "daybreak_reciprocal_conformal_margin_0p50"
    main = variants[main_name]
    diagnostics = {
        "algorithm": "private-negative conformal reciprocal-neighbour MIA",
        "main_variant": main_name,
        "views": [view.name for view in MIA_RECIPROCAL_VIEWS],
        "known_negative_rows": int(np.count_nonzero(known_negative)),
        "uses_domias": False,
        "uses_transport": False,
        "uses_capacity_assignment": False,
        "winner_counts": {
            target_id: int(np.count_nonzero(winner == index))
            for index, target_id in enumerate(target_ids)
        },
        "targets": target_diagnostics,
    }
    return main, variants, diagnostics


def _consensus_codebook_link(
    view_nearest: np.ndarray,
    full_pair_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries, n_views = view_nearest.shape
    linked = np.empty(n_queries, dtype=int)
    agreement = np.empty(n_queries, dtype=float)
    for row in range(n_queries):
        choices, counts = np.unique(view_nearest[row], return_counts=True)
        best_count = counts.max()
        finalists = choices[counts == best_count]
        finalist_distance = full_pair_distance[row, finalists]
        linked[row] = int(finalists[np.argmin(finalist_distance)])
        agreement[row] = float(best_count / n_views)
    return linked, agreement


class OutcomeIntegrityFingerprint:
    """Measure whether a release preserves the public QI-to-event relation."""

    columns = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]

    def __init__(self, background: pd.DataFrame) -> None:
        transformer = ColumnTransformer(
            [
                (
                    "numeric",
                    RobustScaler(),
                    ["age", "SBP", "TG", "HDL", "ALT"],
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["sex", "prefecture"],
                ),
            ]
        )
        self.model = make_pipeline(
            transformer,
            LogisticRegression(
                C=0.25,
                max_iter=2_000,
                class_weight="balanced",
                random_state=0,
            ),
        )
        event = (
            background["onset"].to_numpy(dtype=int)
            + background["death"].to_numpy(dtype=int)
        ) > 0
        if np.unique(event).size != 2:
            raise ValueError("A_bgのeventラベルは2クラス必要です")
        self.model.fit(background[self.columns], event.astype(int))

    def measure(self, released: pd.DataFrame) -> dict[str, Any]:
        event = (
            released["onset"].to_numpy(dtype=int)
            + released["death"].to_numpy(dtype=int)
        ) > 0
        probability = self.model.predict_proba(released[self.columns])[:, 1]
        auc = (
            float(roc_auc_score(event, probability))
            if np.unique(event).size == 2
            else float("nan")
        )
        time_risk = 10.0 - released["time"].to_numpy(dtype=float)
        if np.std(probability) > 0.0 and np.std(time_risk) > 0.0:
            correlation = float(np.corrcoef(probability, time_risk)[0, 1])
        else:
            correlation = float("nan")
        return {
            "auc": auc,
            "correlation": correlation,
            "event_rate": float(np.mean(event)),
        }


def _route_outcome_integrity(
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    adaptive: pd.DataFrame,
    horizon_only: pd.DataFrame,
    *,
    auc_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Switch whole target blocks using only public-release integrity."""
    key_columns = ["target_k", "challenge_row_id"]
    adaptive_keys = adaptive[key_columns].to_numpy(dtype=np.int64)
    horizon_keys = horizon_only[key_columns].to_numpy(dtype=np.int64)
    if not np.array_equal(adaptive_keys, horizon_keys):
        raise ValueError("AIA routerのexpert keyが一致しません")

    fingerprint = OutcomeIntegrityFingerprint(background)
    routed = adaptive.copy()
    profiles: dict[str, Any] = {}
    for target_id in sorted(targets, key=int):
        measure = fingerprint.measure(targets[target_id])
        auc = float(measure["auc"])
        use_horizon = not math.isfinite(auc) or auc < auc_threshold
        mask = routed["target_k"].astype(int).eq(int(target_id))
        if use_horizon:
            routed.loc[mask, "time_hat"] = horizon_only.loc[mask, "time_hat"].to_numpy(
                dtype=float
            )
        profiles[target_id] = {
            **measure,
            "decoder": "horizon" if use_horizon else "adaptive_residual",
        }
    return routed, profiles


def build_aia_independent(
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    member_count: int = EXPECTED_AIA_MEMBERS,
    residual_weight: float = 0.25,
    background_neighbours: int = 48,
    integrity_auc_threshold: float = AIA_INTEGRITY_AUC_THRESHOLD,
    horizon: float = 10.0,
    abstain_time: float = 11.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    encoder = RobustAtomEncoder.fit(background)
    full_columns = AIA_SUBSPACES[0]
    background_full = encoder.matrix(background, full_columns)
    bg_time = background["time"].to_numpy(dtype=float)
    k = min(background_neighbours, len(background))
    bg_time_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    bg_time_model.fit(background_full)

    frame_parts: dict[str, list[pd.DataFrame]] = {
        "daybreak_independent_residual_codebook": [],
        "daybreak_independent_direct_codebook": [],
        "daybreak_independent_residual_only": [],
        "daybreak_independent_horizon_codebook": [],
    }
    target_diagnostics: dict[str, Any] = {}
    floor = 1e-10

    for target_id, released in targets.items():
        challenge = challenges[target_id]
        challenge_ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        view_nearest = np.empty((len(challenge), len(AIA_SUBSPACES)), dtype=int)
        ranked_evidence = np.empty((len(challenge), len(AIA_SUBSPACES)), dtype=float)
        view_distances = np.empty((len(challenge), len(AIA_SUBSPACES)), dtype=float)

        for view_index, columns in enumerate(AIA_SUBSPACES):
            challenge_x = encoder.matrix(challenge, columns)
            release_x = encoder.matrix(released, columns)
            background_x = encoder.matrix(background, columns)
            target_first, target_mean, nearest = _knn_summary(release_x, challenge_x)
            bg_first, bg_mean, _ = _knn_summary(background_x, challenge_x)
            evidence = (
                np.log((bg_first + floor) / (target_first + floor))
                + 0.35 * np.log((bg_mean + floor) / (target_mean + floor))
            )
            ranked_evidence[:, view_index] = strict_rank(evidence, challenge_ids)
            view_nearest[:, view_index] = nearest
            view_distances[:, view_index] = target_first

        challenge_full = encoder.matrix(challenge, full_columns)
        release_full = encoder.matrix(released, full_columns)
        full_pair = cdist(challenge_full, release_full, metric="euclidean")
        linked, agreement = _consensus_codebook_link(view_nearest, full_pair)
        linked_distance = full_pair[np.arange(len(challenge)), linked]
        nearest_full = full_pair.min(axis=1)
        release_to_query = np.argmin(full_pair, axis=0)
        mutual = release_to_query[linked] == np.arange(len(challenge))
        evidence_score = (
            0.65 * ranked_evidence.mean(axis=1)
            + 0.20 * strict_rank(-nearest_full, challenge_ids)
            + 0.10 * agreement
            + 0.05 * mutual.astype(float)
        )
        selection_order = np.lexsort((challenge_ids, -evidence_score))
        selected = np.zeros(len(challenge), dtype=bool)
        selected[selection_order[:member_count]] = True

        challenge_bg_index = bg_time_model.kneighbors(
            challenge_full, return_distance=False
        )
        release_bg_index = bg_time_model.kneighbors(
            release_full, return_distance=False
        )
        baseline_challenge = np.median(bg_time[challenge_bg_index], axis=1)
        baseline_release = np.median(bg_time[release_bg_index], axis=1)
        release_time = released["time"].to_numpy(dtype=float)[linked]
        release_event = (
            released["onset"].to_numpy(dtype=int)[linked]
            + released["death"].to_numpy(dtype=int)[linked]
        ) > 0
        residual_prediction = (
            baseline_challenge
            + release_time
            - baseline_release[linked]
        )
        residual_prediction = np.clip(residual_prediction, 0.0, horizon)
        blended_prediction = (
            (1.0 - residual_weight) * release_time
            + residual_weight * residual_prediction
        )
        blended_prediction = np.where(release_event, blended_prediction, horizon)
        direct_prediction = np.where(release_event, release_time, horizon)
        residual_only = np.where(release_event, residual_prediction, horizon)

        predictions = {
            "daybreak_independent_residual_codebook": blended_prediction,
            "daybreak_independent_direct_codebook": direct_prediction,
            "daybreak_independent_residual_only": residual_only,
            "daybreak_independent_horizon_codebook": np.full(
                len(challenge), horizon, dtype=float
            ),
        }
        for name, active_prediction in predictions.items():
            time_hat = np.full(len(challenge), abstain_time, dtype=float)
            time_hat[selected] = active_prediction[selected]
            frame_parts[name].append(
                pd.DataFrame(
                    {
                        "target_k": int(target_id),
                        "challenge_row_id": challenge_ids,
                        "time_hat": time_hat,
                    }
                )
            )

        target_diagnostics[target_id] = {
            "released_rows": len(released),
            "challenge_rows": len(challenge),
            "selected_rows": int(selected.sum()),
            "selected_event_links": int(np.count_nonzero(selected & release_event)),
            "selected_mutual_links": int(np.count_nonzero(selected & mutual)),
            "median_selected_link_distance": float(np.median(linked_distance[selected])),
            "median_selected_agreement": float(np.median(agreement[selected])),
            "reused_release_rows": int(selected.sum() - len(np.unique(linked[selected]))),
        }

    variants = {
        name: pd.concat(parts, ignore_index=True).sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        ).reset_index(drop=True)
        for name, parts in frame_parts.items()
    }
    main_name = "daybreak_outcome_integrity_router_t060"
    routed, integrity_profiles = _route_outcome_integrity(
        background,
        targets,
        variants["daybreak_independent_residual_codebook"],
        variants["daybreak_independent_horizon_codebook"],
        auc_threshold=integrity_auc_threshold,
    )
    variants[main_name] = routed
    diagnostics = {
        "algorithm": (
            "public outcome-integrity router over reusable multiview residual "
            "and horizon codebook experts"
        ),
        "main_variant": main_name,
        "subspaces": [list(value) for value in AIA_SUBSPACES],
        "member_count_per_target": member_count,
        "residual_weight": residual_weight,
        "background_neighbours": background_neighbours,
        "horizon": horizon,
        "abstain_time": abstain_time,
        "integrity_auc_threshold": integrity_auc_threshold,
        "integrity_profiles": integrity_profiles,
        "targets": target_diagnostics,
    }
    return variants[main_name], variants, diagnostics


def strict_output_checks(
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    candidates: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    expected_mia_columns = ["record_id", *EXPECTED_TARGET_IDS]
    if list(f_mia.columns) != expected_mia_columns:
        raise ValueError("F_mia列順またはtarget集合が不正です")
    if len(f_mia) != len(candidates):
        raise ValueError("F_mia行数がB.csvと一致しません")
    if f_mia["record_id"].duplicated().any() or set(f_mia["record_id"]) != set(candidates["record_id"]):
        raise ValueError("F_mia record_id集合または一意性が不正です")
    mia_values = f_mia[list(EXPECTED_TARGET_IDS)].to_numpy(dtype=float)
    if not np.isfinite(mia_values).all() or ((mia_values < 0.0) | (mia_values > 1.0)).any():
        raise ValueError("F_mia confidenceは有限な[0,1]である必要があります")
    if any(f_mia[target_id].nunique(dropna=False) != len(f_mia) for target_id in EXPECTED_TARGET_IDS):
        raise ValueError("F_mia target列にtieがあります")

    if list(f_aia.columns) != ["target_k", "challenge_row_id", "time_hat"]:
        raise ValueError("F_aia列順が不正です")
    target_numeric = pd.to_numeric(f_aia["target_k"], errors="coerce").to_numpy(dtype=float)
    challenge_numeric = pd.to_numeric(
        f_aia["challenge_row_id"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(target_numeric).all()
        or not np.isfinite(challenge_numeric).all()
        or not np.equal(target_numeric, np.floor(target_numeric)).all()
        or not np.equal(challenge_numeric, np.floor(challenge_numeric)).all()
    ):
        raise ValueError("F_aia target_kとchallenge_row_idは有限整数である必要があります")
    if f_aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("F_aia複合キーが重複しています")
    time_hat = f_aia["time_hat"].to_numpy(dtype=float)
    if not np.isfinite(time_hat).all() or (time_hat < 0.0).any():
        raise ValueError("F_aia time_hatは有限非負である必要があります")
    expected_keys = {
        (int(target_id), int(challenge_id))
        for target_id, frame in challenges.items()
        for challenge_id in frame["challenge_row_id"]
    }
    observed_keys = set(
        zip(
            f_aia["target_k"].astype(int),
            f_aia["challenge_row_id"].astype(int),
            strict=True,
        )
    )
    if observed_keys != expected_keys:
        raise ValueError("F_aiaが全challenge keyを一度ずつ含んでいません")
    return {
        "mia_rows": len(f_mia),
        "mia_columns": len(f_mia.columns),
        "mia_tie_free_targets": len(EXPECTED_TARGET_IDS),
        "aia_rows": len(f_aia),
        "aia_unique_keys": len(f_aia),
        "finite": True,
    }


def run_validators(
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    candidates: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
    targets_dir: Path,
) -> dict[str, Any]:
    strict = strict_output_checks(f_mia, f_aia, candidates, challenges)
    official_dist = official_validate.load_dist(targets_dir)
    mia_result = official_validate.validate_mia_submission(f_mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(f_aia, {}, official_dist)
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            f"公式logical validator失敗: MIA={mia_result.errors}, AIA={aia_result.errors}"
        )
    return {"strict": strict, "official_mia": "PASS", "official_aia": "PASS"}


def _write_csvs(
    directory: Path,
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    *,
    candidate: bool,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    if candidate:
        unexpected = sorted(
            path.name
            for path in directory.iterdir()
            if path.name not in {"F_mia.csv", "F_aia.csv"}
        )
        if unexpected:
            raise ValueError(f"candidate-dirに提出外ファイルがあります: {unexpected}")
    mia_path = directory / "F_mia.csv"
    aia_path = directory / "F_aia.csv"
    f_mia.to_csv(mia_path, index=False)
    f_aia.to_csv(aia_path, index=False)
    return {"F_mia.csv": sha256_file(mia_path), "F_aia.csv": sha256_file(aia_path)}


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
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


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = args.targets.resolve()
    out_dir = args.out_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    if out_dir == candidate_dir:
        raise ValueError("--out-dirと--candidate-dirは別にしてください")

    manifest = verify_sha256_manifest(targets_dir)
    candidates, background, targets, challenges, quality = load_inputs(dist, targets_dir)
    self_challenge = pd.read_csv(
        targets_dir / f"aia_challenge_{args.self_target_id}.csv"
    )
    known_negative, private_diagnostics = discover_known_private(
        candidates, self_challenge
    )

    print(f"{MODEL_ID}: independent conformal/integrity-router attack", flush=True)
    f_mia, mia_variants, mia_diagnostics = build_mia_independent(
        candidates, background, targets, known_negative
    )
    f_aia, aia_variants, aia_diagnostics = build_aia_independent(
        background,
        targets,
        challenges,
        member_count=args.aia_member_count,
        residual_weight=args.aia_residual_weight,
        background_neighbours=args.aia_background_neighbours,
        integrity_auc_threshold=args.aia_integrity_auc_threshold,
    )

    validation = run_validators(
        f_mia, f_aia, candidates, challenges, targets_dir
    )
    output_hashes = _write_csvs(out_dir, f_mia, f_aia, candidate=False)
    candidate_hashes = _write_csvs(
        candidate_dir, f_mia, f_aia, candidate=True
    )
    if output_hashes != candidate_hashes:
        raise RuntimeError("local outputとcandidateのhashが一致しません")

    if not args.main_only:
        variants_dir = out_dir / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in mia_variants.items():
            frame.to_csv(variants_dir / f"F_mia_{name}.csv", index=False)
        for name, frame in aia_variants.items():
            frame.to_csv(variants_dir / f"F_aia_{name}.csv", index=False)

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack",
        "model": MODEL_ID,
        "algorithm_id": ALGORITHM_ID,
        "independent_from": ["attack_v1", "attack_v2", "attack_v3", "daybreak_conservative_v3"],
        "algorithm_dependency_on_v1_v2_v3": False,
        "network_access": False,
        "token_read": False,
        "submission_performed": False,
        "manifest": manifest,
        "quality": quality,
        "private_rule_calibration": private_diagnostics,
        "parameters": {
            "target_ids": list(EXPECTED_TARGET_IDS),
            "mia_total_shared": args.mia_total_shared,
            "aia_member_count": args.aia_member_count,
            "aia_residual_weight": args.aia_residual_weight,
            "aia_background_neighbours": args.aia_background_neighbours,
            "aia_integrity_auc_threshold": args.aia_integrity_auc_threshold,
        },
        "validation": validation,
        "hashes": output_hashes,
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only preliminary truth is not distributed, so an official score cannot be computed locally.",
            "Local shadow comparisons are sensitivity tests rather than leaderboard estimates.",
            "No network, token, archive, or submission operation is implemented.",
        ],
    }
    (out_dir / "diagnostics_daybreak_independent.json").write_text(
        json.dumps(_clean_json(diagnostics), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    model_manifest = {
        "model_id": MODEL_ID,
        "algorithm_id": ALGORITHM_ID,
        "competition": "PWSCUP2026 preliminary attack",
        "team": 24,
        "production_legacy_attack_imports": False,
        "legacy_attack_outputs_used_as_inputs": False,
        "candidate_files": ["F_mia.csv", "F_aia.csv"],
        "output_sha256": output_hashes,
        "organizer_manifest_sha256": sha256_file(targets_dir / "SHA256SUMS.txt"),
        "network_access": False,
        "token_read": False,
        "submission_performed": False,
    }
    (out_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    delivered_mia = pd.read_csv(candidate_dir / "F_mia.csv")
    delivered_aia = pd.read_csv(candidate_dir / "F_aia.csv")
    run_validators(
        delivered_mia, delivered_aia, candidates, challenges, targets_dir
    )
    print(f"PASS: F_mia.csv {len(delivered_mia)} x {len(delivered_mia.columns)}")
    print(f"PASS: F_aia.csv {len(delivered_aia)} rows")
    print(f"PASS: manifest {manifest['verified_files']} files")
    print(f"PASS: independent algorithm id={ALGORITHM_ID}")
    print(f"candidate: {candidate_dir}")


if __name__ == "__main__":
    main()
