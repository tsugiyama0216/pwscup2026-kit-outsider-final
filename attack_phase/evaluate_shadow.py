"""Rule-faithful local shadow evaluation for the current MIA/AIA attack.

The organiser-only membership and attribute truth are not distributed to
participants.  This evaluator therefore constructs deterministic, fully local
shadow cohorts with the published competition cardinalities, runs the current
attack unchanged, and delegates scoring to the kit's official scoring helpers.

It never contacts a remote service and must not be interpreted as an official
leaderboard estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STARTER_SRC) not in sys.path:
    sys.path.insert(0, str(STARTER_SRC))

from attack_phase.generate_submission import (  # noqa: E402
    AIA_REQUIRED_COLUMNS,
    BLOCK_COLUMNS,
    CONTINUOUS_COLUMNS,
    MIA_REQUIRED_COLUMNS,
    QI_COLUMNS,
    build_aia,
    build_mia,
    choose_low_hit_time,
)
from pwscup2026.common import submission_schema  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import (  # noqa: E402
    score_aia_power,
    score_mia_power,
)
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.rare.outlier import mahalanobis  # noqa: E402


DEFAULT_SCENARIOS = (
    "raw",
    "noise_0p03",
    "noise_0p10",
    "time_shuffled",
    "independent",
)

SCENARIO_DESCRIPTIONS = {
    "raw": "Positive control: source rows are released exactly, after row shuffling.",
    "noise_0p03": (
        "Light row-wise perturbation: 3% of public marginal SD on seven continuous "
        "attributes and 0.3 years on time."
    ),
    "noise_0p10": (
        "Moderate row-wise perturbation: 10% of public marginal SD on seven continuous "
        "attributes and 1.0 year on time."
    ),
    "time_shuffled": (
        "Static attributes are exact, while (time, onset, death) are jointly shuffled "
        "between rows; this isolates AIA outcome leakage."
    ),
    "independent": (
        "Negative control: C is an unrelated sample from public background data."
    ),
}


@dataclass(frozen=True)
class ShadowSpec:
    n_targets: int = 29
    candidate_rows: int = 1_049
    target_rows: int = 1_049
    shared_total: int = 736
    minimum_shared: int = 25
    maximum_shared: int = 27
    aia_members: int = 163
    aia_controls: int = 163
    aia_onset_per_group: int = 16
    aia_rare_per_group: int = 13
    aia_rare_onset_per_group: int = 4


@dataclass
class ShadowWorld:
    attack_background: pd.DataFrame
    mia_sources: dict[str, pd.DataFrame]
    aia_sources: dict[str, pd.DataFrame]
    challenges: dict[str, pd.DataFrame]
    target_pools: dict[str, pd.DataFrame]
    aia_truth: dict[str, pd.DataFrame]
    attacker_pool: pd.DataFrame
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "既知の疑似真値と公式採点関数を使い、現在のMIA/AIA攻撃を完全ローカルで評価します。"
        )
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=REPO_ROOT / "participant_data",
        help="B_practice.csv、B_self.csv、A_bg.csv、schema.json のディレクトリ",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-shadow-evaluation",
    )
    parser.add_argument(
        "--seeds",
        default="20260826,20260827",
        help="疑似コホートを反復する整数seedのカンマ区切り",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help=f"評価シナリオのカンマ区切り: {','.join(DEFAULT_SCENARIOS)}",
    )
    parser.add_argument("--aia-active-fraction", type=float, default=0.5)
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument(
        "--skip-domias",
        action="store_true",
        help="高速な実装確認用。現在の本命攻撃を評価する場合は指定しません。",
    )
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.scenarios = [
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ]
    unknown = sorted(set(args.scenarios) - set(SCENARIO_DESCRIPTIONS))
    if unknown:
        parser.error(f"未知のscenarioです: {unknown}")
    if not args.seeds:
        parser.error("--seeds は少なくとも1つ必要です")
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        parser.error("--aia-active-fraction は0〜1です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    return args


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [clean_json(item) for item in value.tolist()]
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_source_data(
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    spec: ShadowSpec,
) -> dict[str, Any]:
    required = ["record_id", *MIA_REQUIRED_COLUMNS]
    for name, frame in (("B_practice", candidates), ("A_bg", background)):
        missing = [column for column in required if column not in frame]
        if missing:
            raise SystemExit(f"{name} に必須列がありません: {missing}")
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if frame.isna().any().any() or not np.isfinite(numeric).all():
            raise SystemExit(f"{name} に欠測または非有限数があります")
        if frame["record_id"].duplicated().any():
            raise SystemExit(f"{name} のrecord_idが一意ではありません")
    if len(candidates) != spec.candidate_rows:
        raise SystemExit(
            f"B_practice は{spec.candidate_rows}行を期待しますが{len(candidates)}行です"
        )
    if len(b_self) != len(candidates) or "is_rare" not in b_self:
        raise SystemExit("B_self.csv の行数または is_rare 列が不正です")
    if not candidates[required].equals(b_self[required]):
        raise SystemExit("B_practice.csv と B_self.csv の共有列・行順が一致しません")
    if len(background) < spec.target_rows + spec.aia_controls:
        raise SystemExit("A_bg.csv がshadow target/control構成に足りません")
    if not (
        spec.n_targets * spec.minimum_shared
        <= spec.shared_total
        <= spec.n_targets * spec.maximum_shared
    ):
        raise ValueError("MIA shadow cardinalities are infeasible")
    return {
        "candidate_rows": len(candidates),
        "background_rows": len(background),
        "candidate_rare_truth": int(b_self["is_rare"].astype(bool).sum()),
        "candidate_null_cells": int(candidates.isna().sum().sum()),
        "background_null_cells": int(background.isna().sum().sum()),
        "candidate_duplicate_ids": int(candidates["record_id"].duplicated().sum()),
        "background_duplicate_ids": int(background["record_id"].duplicated().sum()),
        "b_self_aligned": True,
    }


def load_rare_detector(dist: Path) -> dict[str, Any]:
    schema = json.loads((dist / "schema.json").read_text(encoding="utf-8"))
    detector = schema.get("rare_detector")
    if not isinstance(detector, dict):
        raise SystemExit(f"rare_detector がありません: {dist / 'schema.json'}")
    return detector


def public_rare_mask(frame: pd.DataFrame, detector: dict[str, Any]) -> np.ndarray:
    columns = list(detector["columns"])
    distance = mahalanobis(
        frame,
        np.asarray(detector["mu"], dtype=float),
        np.asarray(detector["sigma"], dtype=float),
        columns,
    )
    return distance >= float(detector["band"]["lo"])


def allocate_shared_counts(spec: ShadowSpec, rng: np.random.Generator) -> np.ndarray:
    counts = np.full(spec.n_targets, spec.minimum_shared, dtype=int)
    remaining = spec.shared_total - int(counts.sum())
    for target_index in rng.permutation(spec.n_targets):
        if remaining == 0:
            break
        increment = min(spec.maximum_shared - spec.minimum_shared, remaining)
        counts[target_index] += increment
        remaining -= increment
    if remaining != 0:
        raise RuntimeError("MIA shadow cardinality allocation failed")
    if int(counts.sum()) != spec.shared_total:
        raise AssertionError("MIA shadow shared total is inconsistent")
    return counts


def match_unique_controls(
    members: pd.DataFrame,
    background: pd.DataFrame,
    excluded_indices: np.ndarray,
    *,
    member_rare: np.ndarray | None = None,
    background_rare: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match unique controls on AIA-visible QIs, excluding target members.

    When rare labels are provided, controls are assigned separately within the
    published (rare, onset) strata.  This keeps the 13 rare / 16 onset / 4
    rare-onset composition identical between the shadow member and control arms.
    """
    eligible_mask = np.ones(len(background), dtype=bool)
    eligible_mask[np.asarray(excluded_indices, dtype=int)] = False
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) < len(members):
        raise ValueError("not enough eligible shadow AIA controls")

    if (member_rare is None) != (background_rare is None):
        raise ValueError("member_rare and background_rare must be supplied together")
    if member_rare is None:
        strata = [(np.arange(len(members)), eligible)]
    else:
        member_rare = np.asarray(member_rare, dtype=bool)
        background_rare = np.asarray(background_rare, dtype=bool)
        if len(member_rare) != len(members) or len(background_rare) != len(background):
            raise ValueError("rare mask length mismatch")
        strata = []
        member_onset = members["onset"].to_numpy(dtype=int)
        background_onset = background["onset"].to_numpy(dtype=int)
        for rare_value in (False, True):
            for onset_value in (0, 1):
                member_rows = np.flatnonzero(
                    (member_rare == rare_value) & (member_onset == onset_value)
                )
                if not len(member_rows):
                    continue
                eligible_rows = eligible[
                    (background_rare[eligible] == rare_value)
                    & (background_onset[eligible] == onset_value)
                ]
                if len(eligible_rows) < len(member_rows):
                    raise ValueError(
                        "not enough eligible controls in shadow (rare, onset) stratum"
                    )
                strata.append((member_rows, eligible_rows))

    chosen = np.full(len(members), -1, dtype=int)
    selected_cost = np.full(len(members), np.nan, dtype=float)

    for member_rows, eligible_rows in strata:
        member_age = members.iloc[member_rows]["age"].to_numpy(dtype=float)[:, None]
        eligible_age = (
            background.iloc[eligible_rows]["age"].to_numpy(dtype=float)[None, :]
        )
        # Official controls are matched on sex, prefecture, age band and onset.
        # onset and rare are exact by the stratum split above.  A tiny age-distance
        # term breaks ties without peeking at SBP/TG/HDL/ALT or the attack score.
        cost = 0.01 * np.abs(member_age - eligible_age)
        member_age_band = np.floor((member_age - 40.0) / 5.0)
        eligible_age_band = np.floor((eligible_age - 40.0) / 5.0)
        cost += 5.0 * (member_age_band != eligible_age_band)
        member_sex = members.iloc[member_rows][BLOCK_COLUMNS[0]].to_numpy()[:, None]
        eligible_sex = background.iloc[eligible_rows][BLOCK_COLUMNS[0]].to_numpy()[None, :]
        member_pref = (
            members.iloc[member_rows][BLOCK_COLUMNS[1]].astype(str).to_numpy()[:, None]
        )
        eligible_pref = (
            background.iloc[eligible_rows][BLOCK_COLUMNS[1]]
            .astype(str)
            .to_numpy()[None, :]
        )
        cost += 100.0 * (member_sex != eligible_sex)
        cost += 20.0 * (member_pref != eligible_pref)
        local_members, local_controls = linear_sum_assignment(cost)
        if len(local_members) != len(member_rows):
            raise RuntimeError("shadow AIA control assignment is incomplete")
        global_member_rows = member_rows[local_members]
        chosen[global_member_rows] = eligible_rows[local_controls]
        selected_cost[global_member_rows] = cost[local_members, local_controls]

    if (chosen < 0).any() or not np.isfinite(selected_cost).all():
        raise RuntimeError("shadow AIA control assignment is incomplete")
    controls = background.iloc[chosen]
    return chosen, {
        "unique": int(np.unique(chosen).size),
        "exact_sex": int(
            (members["sex"].to_numpy() == controls["sex"].to_numpy()).sum()
        ),
        "exact_prefecture": int(
            (
                members["prefecture"].astype(str).to_numpy()
                == controls["prefecture"].astype(str).to_numpy()
            ).sum()
        ),
        "exact_age_band": int(
            (
                np.floor((members["age"].to_numpy(dtype=float) - 40.0) / 5.0)
                == np.floor((controls["age"].to_numpy(dtype=float) - 40.0) / 5.0)
            ).sum()
        ),
        "median_penalized_cost": float(np.median(selected_cost)),
        "p95_penalized_cost": float(np.quantile(selected_cost, 0.95)),
    }


def sample_aia_members(
    background: pd.DataFrame,
    background_rare: np.ndarray,
    spec: ShadowSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the published 2x2 (rare, onset) AIA member composition."""
    counts = {
        (True, 1): spec.aia_rare_onset_per_group,
        (True, 0): spec.aia_rare_per_group - spec.aia_rare_onset_per_group,
        (False, 1): spec.aia_onset_per_group - spec.aia_rare_onset_per_group,
        (False, 0): (
            spec.aia_members
            - spec.aia_rare_per_group
            - spec.aia_onset_per_group
            + spec.aia_rare_onset_per_group
        ),
    }
    onset = background["onset"].to_numpy(dtype=int)
    chosen: list[np.ndarray] = []
    for (rare_value, onset_value), count in counts.items():
        available = np.flatnonzero(
            (np.asarray(background_rare, dtype=bool) == rare_value)
            & (onset == onset_value)
        )
        if len(available) < count:
            raise ValueError(
                f"not enough rows for AIA stratum rare={rare_value}, onset={onset_value}"
            )
        chosen.append(rng.choice(available, size=count, replace=False))
    indices = np.concatenate(chosen)
    return indices[rng.permutation(len(indices))]


def split_public_and_hidden_background(
    background: pd.DataFrame,
    *,
    visible_fraction: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a stable, attack-independent split of public rows for shadow simulation.

    The visible arm is passed to the attack as A_bg.  Target fillers and AIA
    controls are drawn only from the hidden arm, so the evaluator never rewards
    an attack for matching the exact same control row in its own reference data.
    The split is fixed independently of cohort seeds and attack outputs.
    """
    if not 0.0 < visible_fraction < 1.0:
        raise ValueError("visible_fraction must be in (0, 1)")
    record_ids = background["record_id"].to_numpy(dtype=np.uint64)
    mixed = (
        record_ids * np.uint64(11400714819323198485)
        + np.uint64(14029467366897019727)
    )
    order = np.argsort(mixed, kind="mergesort")
    visible_count = int(round(len(background) * visible_fraction))
    visible_indices = order[:visible_count]
    hidden_indices = order[visible_count:]
    return (
        background.iloc[visible_indices].reset_index(drop=True),
        background.iloc[hidden_indices].reset_index(drop=True),
    )


def build_shadow_world(
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    detector: dict[str, Any],
    spec: ShadowSpec,
    seed: int,
) -> ShadowWorld:
    rng = np.random.default_rng(seed)
    attack_background, hidden_background = split_public_and_hidden_background(background)
    target_ids = [str(index) for index in range(1, spec.n_targets + 1)]
    shared_counts = allocate_shared_counts(spec, rng)
    candidate_order = rng.permutation(len(candidates))
    candidate_pool_ids = 10_000_000 + np.arange(len(candidates), dtype=int)
    background_pool_ids = 20_000_000 + hidden_background["record_id"].to_numpy(dtype=int)
    background_rare = public_rare_mask(hidden_background, detector)

    attacker_pool = pd.DataFrame(
        {
            "record_id": candidates["record_id"].to_numpy(dtype=int),
            "pool_id": candidate_pool_ids,
            "is_rare": b_self["is_rare"].astype(int).to_numpy(),
        }
    )

    mia_sources: dict[str, pd.DataFrame] = {}
    aia_sources: dict[str, pd.DataFrame] = {}
    challenges: dict[str, pd.DataFrame] = {}
    target_pools: dict[str, pd.DataFrame] = {}
    aia_truth: dict[str, pd.DataFrame] = {}
    target_metadata: dict[str, Any] = {}
    cursor = 0

    for target_id, shared_count in zip(target_ids, shared_counts, strict=True):
        shared_indices = candidate_order[cursor : cursor + int(shared_count)]
        cursor += int(shared_count)
        filler_count = spec.target_rows - int(shared_count)
        filler_indices = rng.choice(
            len(hidden_background), size=filler_count, replace=False
        )
        mia_source = pd.concat(
            [
                candidates.iloc[shared_indices],
                hidden_background.iloc[filler_indices],
            ],
            ignore_index=True,
        )
        mia_source = mia_source.iloc[rng.permutation(len(mia_source))].reset_index(
            drop=True
        )
        mia_sources[target_id] = mia_source

        target_pool_ids = np.concatenate(
            [candidate_pool_ids[shared_indices], background_pool_ids[filler_indices]]
        )
        target_pools[target_id] = pd.DataFrame({"pool_id": target_pool_ids})

        # AIA is a separate shadow suite: B_self gives genuine member rare labels,
        # while MIA requires a different 29-cohort overlap construction.  Keeping
        # the suites separate avoids pretending that undistributed cohorts can be
        # reconstructed jointly from public data.
        member_indices = sample_aia_members(
            candidates,
            b_self["is_rare"].astype(bool).to_numpy(),
            spec,
            rng,
        )
        members = candidates.iloc[member_indices].reset_index(drop=True)
        member_rare = b_self.iloc[member_indices]["is_rare"].astype(bool).to_numpy()
        aia_source = candidates.copy()
        aia_source = aia_source.iloc[rng.permutation(len(aia_source))].reset_index(
            drop=True
        )
        aia_sources[target_id] = aia_source
        control_indices, matching = match_unique_controls(
            members,
            hidden_background,
            excluded_indices=np.empty(0, dtype=int),
            member_rare=member_rare,
            background_rare=background_rare,
        )
        controls = hidden_background.iloc[control_indices].reset_index(drop=True)
        control_rare = background_rare[control_indices]

        challenge_rows = pd.concat(
            [
                members[[*QI_COLUMNS, *BLOCK_COLUMNS]],
                controls[[*QI_COLUMNS, *BLOCK_COLUMNS]],
            ],
            ignore_index=True,
        )
        truth = pd.DataFrame(
            {
                "label": ["member"] * spec.aia_members
                + ["control"] * spec.aia_controls,
                "onset": pd.concat(
                    [members["onset"], controls["onset"]], ignore_index=True
                ),
                "time": pd.concat(
                    [members["time"], controls["time"]], ignore_index=True
                ),
                "is_rare": np.concatenate([member_rare, control_rare]).astype(int),
            }
        )
        shuffle = rng.permutation(len(challenge_rows))
        challenge_rows = challenge_rows.iloc[shuffle].reset_index(drop=True)
        truth = truth.iloc[shuffle].reset_index(drop=True)
        challenge_ids = int(target_id) * 1_000_000 + np.arange(len(challenge_rows))
        challenge_rows.insert(0, "challenge_row_id", challenge_ids)
        truth.insert(0, "challenge_row_id", challenge_ids)
        challenges[target_id] = challenge_rows.loc[:, AIA_REQUIRED_COLUMNS]
        aia_truth[target_id] = truth
        target_metadata[target_id] = {
            "shared_candidates": int(shared_count),
            "mia_target_rows": len(mia_source),
            "aia_target_rows": len(aia_source),
            "aia_members": spec.aia_members,
            "aia_controls": spec.aia_controls,
            "aia_member_rare_public": int(member_rare.sum()),
            "aia_control_rare_public": int(control_rare.sum()),
            "aia_member_onset": int(members["onset"].sum()),
            "aia_control_onset": int(controls["onset"].sum()),
            "aia_member_rare_onset": int(
                (member_rare & members["onset"].astype(bool).to_numpy()).sum()
            ),
            "aia_control_rare_onset": int(
                (control_rare & controls["onset"].astype(bool).to_numpy()).sum()
            ),
            "control_matching": matching,
        }

    if cursor != spec.shared_total:
        raise AssertionError("shadow membership cursor does not equal shared total")
    membership_use = np.zeros(len(candidates), dtype=int)
    membership_use[candidate_order[:cursor]] = 1
    return ShadowWorld(
        attack_background=attack_background,
        mia_sources=mia_sources,
        aia_sources=aia_sources,
        challenges=challenges,
        target_pools=target_pools,
        aia_truth=aia_truth,
        attacker_pool=attacker_pool,
        metadata={
            "seed": seed,
            "shared_counts": {
                target_id: int(count)
                for target_id, count in zip(target_ids, shared_counts, strict=True)
            },
            "shared_total": int(membership_use.sum()),
            "candidate_max_target_memberships": 1,
            "private_candidates": int((membership_use == 0).sum()),
            "background_split": {
                "method": "stable record_id hash, independent of attack and cohort seed",
                "attack_visible_rows": len(attack_background),
                "hidden_donor_control_rows": len(hidden_background),
                "record_id_intersection": int(
                    len(
                        set(attack_background["record_id"])
                        & set(hidden_background["record_id"])
                    )
                ),
            },
            "component_worlds_are_separate": True,
            "aia_member_rare_source": "B_self true is_rare",
            "aia_control_rare_source": "public Mahalanobis proxy on hidden controls",
            "targets": target_metadata,
        },
    )


def coerce_release(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    age_lo, age_hi = (float(value) for value in cfg["population"]["age_range"])
    out["age"] = np.rint(
        np.clip(out["age"].to_numpy(dtype=float), age_lo, age_hi)
    ).astype(int)
    for column in ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"):
        lower, upper = (float(value) for value in cfg["schema"]["ranges"][column])
        out[column] = np.clip(out[column].to_numpy(dtype=float), lower, upper)
    for column in ("sex", "smoking", "onset", "death"):
        out[column] = np.clip(
            np.rint(out[column].to_numpy(dtype=float)), 0, 1
        ).astype(int)
    onset = out["onset"].to_numpy(dtype=int)
    death = out["death"].to_numpy(dtype=int)
    out["death"] = ((onset == 0) & (death == 1)).astype(int)
    horizon = float(cfg["onset"]["horizon_years"])
    out["time"] = np.clip(
        out["time"].to_numpy(dtype=float), 1.0 / 365.25, horizon
    )
    return out.loc[:, submission_schema.SUBMITTED_C_COLUMNS].reset_index(drop=True)


def scenario_release(
    source: pd.DataFrame,
    background: pd.DataFrame,
    scenario: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if scenario == "independent":
        out = background.iloc[
            rng.choice(len(background), size=len(source), replace=False)
        ].copy()
    else:
        out = source.copy()

    if scenario in ("noise_0p03", "noise_0p10"):
        fraction = 0.03 if scenario == "noise_0p03" else 0.10
        scale = background[list(CONTINUOUS_COLUMNS)].std(ddof=0).to_numpy(dtype=float)
        scale = np.where(scale > 0.0, scale, 1.0)
        noise = rng.normal(
            loc=0.0,
            scale=fraction * scale,
            size=(len(out), len(CONTINUOUS_COLUMNS)),
        )
        noisy_values = out.loc[:, CONTINUOUS_COLUMNS].to_numpy(dtype=float) + noise
        for column_index, column in enumerate(CONTINUOUS_COLUMNS):
            # Assign one column at a time so an integer age source can safely become
            # float before coerce_release rounds it back to the submitted schema.
            out[column] = noisy_values[:, column_index]
        horizon = float(cfg["onset"]["horizon_years"])
        out["time"] = out["time"].to_numpy(dtype=float) + rng.normal(
            0.0, fraction * horizon, size=len(out)
        )
    elif scenario == "time_shuffled":
        outcome = out[["time", "onset", "death"]].to_numpy(copy=True)
        out.loc[:, ["time", "onset", "death"]] = outcome[rng.permutation(len(outcome))]
    elif scenario not in ("raw", "independent"):
        raise ValueError(f"unknown scenario: {scenario}")

    out = coerce_release(out, cfg)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def write_shadow_reference(world: ShadowWorld, ref_dir: Path) -> None:
    ref_dir.mkdir(parents=True, exist_ok=True)
    world.attacker_pool.to_csv(ref_dir / "pool_ids_attacker.csv", index=False)
    for target_id, frame in world.target_pools.items():
        frame.to_csv(ref_dir / f"pool_ids_{target_id}.csv", index=False)
    for target_id, frame in world.aia_truth.items():
        frame.to_csv(ref_dir / f"aia_truth_{target_id}.csv", index=False)


def validate_world(world: ShadowWorld, spec: ShadowSpec) -> dict[str, Any]:
    shared_counts = np.asarray(list(world.metadata["shared_counts"].values()), dtype=int)
    challenge_rows = np.asarray([len(frame) for frame in world.challenges.values()])
    mia_source_rows = np.asarray([len(frame) for frame in world.mia_sources.values()])
    aia_source_rows = np.asarray([len(frame) for frame in world.aia_sources.values()])
    member_counts = np.asarray(
        [(frame["label"] == "member").sum() for frame in world.aia_truth.values()]
    )
    control_counts = np.asarray(
        [(frame["label"] == "control").sum() for frame in world.aia_truth.values()]
    )
    group_profiles: list[dict[str, int]] = []
    for frame in world.aia_truth.values():
        for label in ("member", "control"):
            group = frame[frame["label"] == label]
            rare = group["is_rare"].astype(bool)
            onset = group["onset"].astype(bool)
            group_profiles.append(
                {
                    "rows": len(group),
                    "rare": int(rare.sum()),
                    "onset": int(onset.sum()),
                    "rare_onset": int((rare & onset).sum()),
                }
            )
    checks = {
        "target_count": (
            len(world.mia_sources) == spec.n_targets
            and len(world.aia_sources) == spec.n_targets
        ),
        "mia_source_rows": bool(np.all(mia_source_rows == spec.target_rows)),
        "aia_source_rows": bool(np.all(aia_source_rows == spec.target_rows)),
        "shared_total": int(shared_counts.sum()) == spec.shared_total,
        "shared_bounds": bool(
            np.all(
                (shared_counts >= spec.minimum_shared)
                & (shared_counts <= spec.maximum_shared)
            )
        ),
        "candidate_exclusivity": world.metadata["candidate_max_target_memberships"] == 1,
        "challenge_rows": bool(
            np.all(challenge_rows == spec.aia_members + spec.aia_controls)
        ),
        "aia_member_count": bool(np.all(member_counts == spec.aia_members)),
        "aia_control_count": bool(np.all(control_counts == spec.aia_controls)),
        "aia_published_strata": all(
            profile["rows"] == spec.aia_members
            and profile["rare"] == spec.aia_rare_per_group
            and profile["onset"] == spec.aia_onset_per_group
            and profile["rare_onset"] == spec.aia_rare_onset_per_group
            for profile in group_profiles
        ),
        "challenge_ids_unique": all(
            frame["challenge_row_id"].is_unique for frame in world.challenges.values()
        ),
        "challenge_truth_ids_match": all(
            truth["challenge_row_id"].is_unique
            and set(challenge["challenge_row_id"]) == set(truth["challenge_row_id"])
            for target_id, challenge in world.challenges.items()
            for truth in (world.aia_truth[target_id],)
        ),
        "control_matches_unique": all(
            metadata["control_matching"]["unique"] == spec.aia_controls
            for metadata in world.metadata["targets"].values()
        ),
        "attack_background_hidden_disjoint": (
            world.metadata["background_split"]["record_id_intersection"] == 0
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"shadow world validation failed: {failed}")
    return checks


def validate_prediction_frames(
    mia_variants: dict[str, pd.DataFrame],
    aia_variants: dict[str, pd.DataFrame],
    candidates: pd.DataFrame,
    spec: ShadowSpec,
) -> dict[str, Any]:
    expected_mia_columns = {"record_id", *[str(k) for k in range(1, spec.n_targets + 1)]}
    expected_aia_rows = spec.n_targets * (spec.aia_members + spec.aia_controls)
    checks: dict[str, bool] = {}
    for name, frame in mia_variants.items():
        numeric = frame.drop(columns=["record_id"]).to_numpy(dtype=float)
        checks[f"mia_{name}"] = bool(
            len(frame) == len(candidates)
            and set(frame.columns) == expected_mia_columns
            and frame["record_id"].is_unique
            and set(frame["record_id"]) == set(candidates["record_id"])
            and np.isfinite(numeric).all()
            and np.all((numeric >= 0.0) & (numeric <= 1.0))
            and all(frame[column].is_unique for column in frame.columns if column != "record_id")
        )
    for name, frame in aia_variants.items():
        checks[f"aia_{name}"] = bool(
            len(frame) == expected_aia_rows
            and not frame.duplicated(["target_k", "challenge_row_id"]).any()
            and np.isfinite(frame["time_hat"].to_numpy(dtype=float)).all()
            and (frame["time_hat"] >= 0.0).all()
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"attack output validation failed: {failed}")
    return checks


def evaluate_variants(
    mia_variants: dict[str, pd.DataFrame],
    aia_variants: dict[str, pd.DataFrame],
    ref_dir: Path,
    cfg: dict[str, Any],
    *,
    seed: int,
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mia_scores: dict[str, dict[str, Any]] = {}
    aia_scores: dict[str, dict[str, Any]] = {}
    component_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for variant, frame in mia_variants.items():
        result = score_mia_power(frame, ref_dir, cfg)
        mia_scores[variant] = result
        component_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "mia",
                "variant": variant,
                "power": result["power"],
                "all": result["all"],
                "rare": result["rare"],
            }
        )
        target_rows.extend(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "mia",
                "variant": variant,
                "target_k": int(target_id),
                "score": score,
            }
            for target_id, score in result["per_target"].items()
        )

    for variant, frame in aia_variants.items():
        result = score_aia_power(frame, ref_dir, cfg)
        success_profile = aia_success_profile(frame, ref_dir, cfg)
        aia_scores[variant] = result
        component_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "aia",
                "variant": variant,
                "power": result["power"],
                "all": result["all"],
                "rare": result["rare"],
                **success_profile,
            }
        )
        target_rows.extend(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "aia",
                "variant": variant,
                "target_k": int(target_id),
                "score": score,
            }
            for target_id, score in result["per_target"].items()
        )

    score_rows: list[dict[str, Any]] = []
    for mia_variant, mia in mia_scores.items():
        for aia_variant, aia in aia_scores.items():
            score_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "mia_variant": mia_variant,
                    "aia_variant": aia_variant,
                    "mia_power": mia["power"],
                    "aia_power": aia["power"],
                    "attack_power": float(mia["power"] + aia["power"]),
                    "mia_all": mia["all"],
                    "mia_rare": mia["rare"],
                    "aia_all": aia["all"],
                    "aia_rare": aia["rare"],
                }
            )
    return score_rows, component_rows, target_rows


def aia_success_profile(
    predictions: pd.DataFrame,
    ref_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Return interpretable success rates alongside the official AIA score."""
    tau = float(cfg["aia"]["tau_time"])
    per_target: list[dict[str, float]] = []
    for target_k, group in predictions.groupby(predictions["target_k"].astype(int)):
        truth = pd.read_csv(ref_dir / f"aia_truth_{int(target_k)}.csv")
        if group["challenge_row_id"].duplicated().any() or truth[
            "challenge_row_id"
        ].duplicated().any():
            raise RuntimeError("AIA IDs must be unique before scoring")
        if set(group["challenge_row_id"]) != set(truth["challenge_row_id"]):
            raise RuntimeError("AIA prediction/truth ID sets do not match")
        merged = group.merge(
            truth,
            on="challenge_row_id",
            how="inner",
            validate="one_to_one",
        )
        member = merged["label"].to_numpy() == "member"
        control = merged["label"].to_numpy() == "control"
        rare = merged["is_rare"].to_numpy(dtype=bool)
        success = (
            np.abs(
                merged["time_hat"].to_numpy(dtype=float)
                - merged["time"].to_numpy(dtype=float)
            )
            <= tau
        )

        def rate(mask: np.ndarray) -> float:
            return float(success[mask].mean()) if mask.any() else float("nan")

        per_target.append(
            {
                "member_success": rate(member),
                "control_success": rate(control),
                "member_rare_success": rate(member & rare),
                "control_rare_success": rate(control & rare),
            }
        )
    profile = pd.DataFrame(per_target)
    return {
        "member_success": float(profile["member_success"].mean()),
        "control_success": float(profile["control_success"].mean()),
        "success_rate_difference": float(
            (profile["member_success"] - profile["control_success"]).mean()
        ),
        "member_rare_success": float(profile["member_rare_success"].mean()),
        "control_rare_success": float(profile["control_rare_success"].mean()),
    }


def aggregate_records(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_key, group in frame.groupby(group_columns, sort=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        row = dict(zip(group_columns, key, strict=True))
        row["observations"] = len(group)
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_min"] = float(np.min(values))
            row[f"{column}_max"] = float(np.max(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def build_summary(
    scores: pd.DataFrame,
    components: pd.DataFrame,
    target_scores: pd.DataFrame,
    *,
    args: argparse.Namespace,
    spec: ShadowSpec,
    source_quality: dict[str, Any],
    world_metadata: list[dict[str, Any]],
    run_diagnostics: list[dict[str, Any]],
    inputs: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    # fraction_label(0.5) -> 0p5; format explicitly to avoid importing a display helper.
    main_aia = f"{args.aia_active_fraction:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    current = scores[
        (scores["mia_variant"] == "transport")
        & (scores["aia_variant"] == main_aia)
    ]
    current_by_scenario = aggregate_records(
        current,
        ["scenario"],
        ["mia_power", "aia_power", "attack_power", "mia_all", "mia_rare", "aia_all", "aia_rare"],
    )
    current_aia_components = components[
        (components["component"] == "aia") & (components["variant"] == main_aia)
    ]
    aia_success_by_scenario = aggregate_records(
        current_aia_components,
        ["scenario"],
        [
            "member_success",
            "control_success",
            "success_rate_difference",
            "member_rare_success",
            "control_rare_success",
        ],
    )

    stress_scenarios = [
        scenario
        for scenario in ("noise_0p03", "noise_0p10")
        if scenario in args.scenarios
    ]
    ranking_scope = stress_scenarios or [
        scenario for scenario in args.scenarios if scenario != "independent"
    ]
    ranking_components = components[components["scenario"].isin(ranking_scope)]
    component_rankings: dict[str, list[dict[str, Any]]] = {}
    for component in ("mia", "aia"):
        subset = ranking_components[ranking_components["component"] == component]
        ranking = (
            subset.groupby("variant", as_index=False)["power"]
            .agg(["mean", "min", "max", "std"])
            .reset_index()
            .sort_values(["mean", "variant"], ascending=[False, True])
        )
        component_rankings[component] = ranking.to_dict(orient="records")

    combination_ranking = (
        scores[scores["scenario"].isin(ranking_scope)]
        .groupby(["mia_variant", "aia_variant"], as_index=False)["attack_power"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .sort_values(["mean", "mia_variant", "aia_variant"], ascending=[False, True, True])
    )

    def current_mean(scenario: str, column: str) -> float | None:
        values = current.loc[current["scenario"] == scenario, column]
        return float(values.mean()) if len(values) else None

    raw_mia = current_mean("raw", "mia_power")
    raw_aia = current_mean("raw", "aia_power")
    independent_mia = current_mean("independent", "mia_power")
    independent_aia = current_mean("independent", "aia_power")
    shuffled_aia = current_mean("time_shuffled", "aia_power")
    sanity_checks = {
        "official_scoring_helpers_called": True,
        "all_attack_outputs_valid": all(
            all(item["prediction_checks"].values()) for item in run_diagnostics
        ),
        "raw_mia_exceeds_independent": (
            raw_mia is not None
            and independent_mia is not None
            and raw_mia > independent_mia
        ),
        "raw_aia_exceeds_independent": (
            raw_aia is not None
            and independent_aia is not None
            and raw_aia > independent_aia
        ),
        "time_shuffle_reduces_aia": (
            raw_aia is not None and shuffled_aia is not None and raw_aia > shuffled_aia
        ),
    }

    main_target = target_scores[
        (
            (target_scores["component"] == "mia")
            & (target_scores["variant"] == "transport")
        )
        | (
            (target_scores["component"] == "aia")
            & (target_scores["variant"] == main_aia)
        )
    ]
    target_dispersion = aggregate_records(
        main_target,
        ["scenario", "component"],
        ["score"],
    )

    return {
        "report_type": "authorized_local_privacy_competition_shadow_attack_evaluation",
        "report_version": 1,
        "official_score_available": False,
        "official_score_blocker": (
            "Organizer-only pool_ids_<k>.csv and aia_truth_<k>.csv are not distributed. "
            "All reported values use deterministic synthetic shadow truth."
        ),
        "scoring_faithfulness": {
            "functions": ["score_mia_power", "score_aia_power"],
            "mia": "target mean of max(TPR@FPR=0.01 overall, rare)",
            "aia": "target mean of max(control-adjusted time recovery overall, rare)",
            "attack_power": "mia_power + aia_power, range [0, 2]",
            "aia_tau_time_years": float(load_config(KIT_CONFIG_PATH)["aia"]["tau_time"]),
            "rare_approximation": (
                "MIA candidates and AIA members use B_self truth. Shadow AIA controls "
                "use the distributed public Mahalanobis detector because organizer "
                "control labels are unavailable."
            ),
        },
        "parameters": {
            "seeds": args.seeds,
            "scenarios": args.scenarios,
            "scenario_descriptions": {
                scenario: SCENARIO_DESCRIPTIONS[scenario] for scenario in args.scenarios
            },
            "shadow_spec": asdict(spec),
            "aia_active_fraction": args.aia_active_fraction,
            "mia_transport_epsilon": args.mia_transport_epsilon,
            "domias_included": not args.skip_domias,
            "variant_ranking_scope": ranking_scope,
        },
        "inputs": inputs,
        "source_quality": source_quality,
        "shadow_worlds": world_metadata,
        "current_main": {
            "mia_variant": "transport",
            "aia_variant": main_aia,
            "by_scenario": current_by_scenario,
            "aia_success_by_scenario": aia_success_by_scenario,
            "target_dispersion": target_dispersion,
        },
        "component_variant_rankings": component_rankings,
        "combination_ranking": combination_ranking.to_dict(orient="records"),
        "sanity_checks": sanity_checks,
        "confidence_assessment": {
            "status": "share_with_caveats",
            "implementation_regression_and_relative_variant_confidence": "medium",
            "official_leaderboard_prediction_confidence": "low",
            "reason": (
                "The exact official metrics and cardinalities are reproduced, but shadow "
                "target generation and AIA rare labels cannot reproduce undistributed cohorts "
                "or other teams' defenses."
            ),
        },
        "interpretation_limits": [
            "These are local shadow scores, not official or estimated leaderboard scores.",
            "Raw and row-wise-noise scenarios are sensitivity controls, not models of every competing defense.",
            "A_bg supplies shadow filler rows and matched controls; this can alter linkage difficulty.",
            "MIA and AIA use separate shadow target suites because public files cannot reconstruct one coherent 29-target secret world.",
            "Two deterministic cohort seeds show construction sensitivity but are not a population confidence interval.",
            "The independent scenario is an empirical negative control; its nonzero score is finite-sample/null variation, not evidence of leakage.",
            "No remote service, token, submission endpoint, or non-public third-party data is accessed.",
        ],
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "scikit_learn": package_version("scikit-learn"),
        },
    }


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\nPWS Cup 2026 current attack - local rule-faithful shadow evaluation")
    print("official leaderboard score: unavailable (organizer truth is not distributed)")
    rows = pd.DataFrame(summary["current_main"]["by_scenario"])
    display_columns = [
        "scenario",
        "mia_power_mean",
        "aia_power_mean",
        "attack_power_mean",
        "attack_power_min",
        "attack_power_max",
    ]
    print("\nCurrent main: MIA=transport, AIA=" + summary["current_main"]["aia_variant"])
    print(rows[display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSanity checks:")
    for name, passed in summary["sanity_checks"].items():
        print(f"  {'PASS' if passed else 'CHECK'}  {name}")
    print(f"\nresults: {out_dir}")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dist = args.dist.resolve()
    out_dir = args.out_dir.resolve()
    required_paths = {
        "B_practice.csv": dist / "B_practice.csv",
        "B_self.csv": dist / "B_self.csv",
        "A_bg.csv": dist / "A_bg.csv",
        "schema.json": dist / "schema.json",
        "kit_config.yaml": Path(KIT_CONFIG_PATH),
        "attack_implementation": Path(__file__).resolve().parent / "generate_submission.py",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"必要なファイルがありません: {missing}")

    candidates = pd.read_csv(required_paths["B_practice.csv"]).reset_index(drop=True)
    b_self = pd.read_csv(required_paths["B_self.csv"]).reset_index(drop=True)
    background = pd.read_csv(required_paths["A_bg.csv"]).reset_index(drop=True)
    cfg = load_config(KIT_CONFIG_PATH)
    detector = load_rare_detector(dist)
    spec = ShadowSpec()
    source_quality = validate_source_data(candidates, b_self, background, spec)
    candidate_true_rare = b_self["is_rare"].astype(bool).to_numpy()
    candidate_public_rare = public_rare_mask(candidates, detector)
    rare_true_positive = int((candidate_true_rare & candidate_public_rare).sum())
    source_quality["candidate_public_rare"] = int(candidate_public_rare.sum())
    source_quality["candidate_rare_detector_true_positive"] = rare_true_positive
    source_quality["candidate_rare_detector_false_positive"] = int(
        ((~candidate_true_rare) & candidate_public_rare).sum()
    )
    source_quality["candidate_rare_detector_precision"] = (
        rare_true_positive / max(int(candidate_public_rare.sum()), 1)
    )
    source_quality["candidate_rare_detector_recall"] = (
        rare_true_positive / max(int(candidate_true_rare.sum()), 1)
    )
    inputs = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in required_paths.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    all_target_scores: list[dict[str, Any]] = []
    world_metadata: list[dict[str, Any]] = []
    run_diagnostics: list[dict[str, Any]] = []
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    for seed_index, seed in enumerate(args.seeds):
        print(f"\n[seed {seed}] shadow cohort construction")
        world = build_shadow_world(candidates, b_self, background, detector, spec, seed)
        world_checks = validate_world(world, spec)
        world.metadata["validation_checks"] = world_checks
        world_metadata.append(world.metadata)
        abstain_time = choose_low_hit_time(
            world.attack_background, horizon=horizon, tau=tau
        )
        ref_dir = out_dir / "shadow_reference" / f"seed_{seed}"
        write_shadow_reference(world, ref_dir)

        for scenario_index, scenario in enumerate(args.scenarios):
            print(
                f"[seed {seed}] scenario={scenario}: current MIA/AIA variants",
                flush=True,
            )
            mia_targets: dict[str, pd.DataFrame] = {}
            aia_targets: dict[str, pd.DataFrame] = {}
            for target_id in world.mia_sources:
                mia_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, seed_index, scenario_index, int(target_id), 1]
                    )
                )
                aia_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, seed_index, scenario_index, int(target_id), 2]
                    )
                )
                mia_targets[target_id] = scenario_release(
                    world.mia_sources[target_id],
                    world.attack_background,
                    scenario,
                    cfg,
                    mia_rng,
                )
                aia_targets[target_id] = scenario_release(
                    world.aia_sources[target_id],
                    world.attack_background,
                    scenario,
                    cfg,
                    aia_rng,
                )

            _, mia_variants, mia_diagnostics = build_mia(
                candidates,
                world.attack_background,
                mia_targets,
                minimum=spec.minimum_shared,
                maximum=spec.maximum_shared,
                total=spec.shared_total,
                transport_epsilon=args.mia_transport_epsilon,
                include_domias=not args.skip_domias,
            )
            _, aia_variants, aia_diagnostics = build_aia(
                world.attack_background,
                aia_targets,
                world.challenges,
                main_fraction=args.aia_active_fraction,
                variant_fractions=[0.25, 0.5, 0.75, 1.0],
                abstain_time=abstain_time,
            )
            prediction_checks = validate_prediction_frames(
                mia_variants, aia_variants, candidates, spec
            )
            score_rows, component_rows, target_rows = evaluate_variants(
                mia_variants,
                aia_variants,
                ref_dir,
                cfg,
                seed=seed,
                scenario=scenario,
            )
            all_scores.extend(score_rows)
            all_components.extend(component_rows)
            all_target_scores.extend(target_rows)
            run_diagnostics.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "prediction_checks": prediction_checks,
                    "mia_capacity": mia_diagnostics["capacity"],
                    "mia_transport": mia_diagnostics["transport"],
                    "mia_top10_before_constraint": mia_diagnostics[
                        "top10_before_constraint"
                    ],
                    "mia_top10_after_transport": mia_diagnostics[
                        "top10_after_transport"
                    ],
                    "aia_abstain_time": aia_diagnostics["abstain_time"],
                    "aia_main_fraction": aia_diagnostics["main_fraction"],
                }
            )

    scores = pd.DataFrame(all_scores).sort_values(
        ["scenario", "seed", "mia_variant", "aia_variant"], kind="mergesort"
    )
    components = pd.DataFrame(all_components).sort_values(
        ["scenario", "seed", "component", "variant"], kind="mergesort"
    )
    target_scores = pd.DataFrame(all_target_scores).sort_values(
        ["scenario", "seed", "component", "variant", "target_k"], kind="mergesort"
    )
    elapsed = time.perf_counter() - started
    summary = build_summary(
        scores,
        components,
        target_scores,
        args=args,
        spec=spec,
        source_quality=source_quality,
        world_metadata=world_metadata,
        run_diagnostics=run_diagnostics,
        inputs=inputs,
        elapsed_seconds=elapsed,
    )

    scores.to_csv(out_dir / "scores.csv", index=False)
    components.to_csv(out_dir / "component_scores.csv", index=False)
    target_scores.to_csv(out_dir / "target_scores.csv", index=False)
    (out_dir / "run_diagnostics.json").write_text(
        json.dumps(clean_json(run_diagnostics), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(clean_json(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print_summary(summary, out_dir)


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    main()
