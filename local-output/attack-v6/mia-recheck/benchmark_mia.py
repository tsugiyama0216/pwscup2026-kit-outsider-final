"""Independent, truth-known recheck of normal-format MIA rank recipes.

All artifacts are research-only and remain beneath ``local-output``.  The
script never reads a token, packages an archive, changes a production
candidate, or submits.  Candidate recipes are fixed before validation and use
one finite, tie-free score per existing record/target cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_mia_independent,
)
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_v5.generate_submission_v5 import validate_mia  # noqa: E402
from attack_phase_v5.mia_tail_assignment import infer_private_bottom_ids  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_mia  # noqa: E402


OUT = ROOT / "local-output" / "attack-v6" / "mia-recheck"
DIST = ROOT / "participant_data"
TEAM = ROOT / "submissions" / "prelim" / "team24"
DISCOVERY_SEEDS = (20261301, 20261302, 20261303)
VALIDATION_SEEDS = (20261311, 20261312, 20261313)
SCENARIOS = ("noise_0p10", "noise_0p20", "noise_0p40", "independent")
PLAUSIBLE_SCENARIOS = ("noise_0p10", "noise_0p20", "noise_0p40")
OFFICIAL = {
    "v3": {"all": 0.09266381766381768, "rare": 1.0 / 3.0, "auc": 0.6434884498600183},
    "v4": {"all": 0.10236752136752136, "rare": 1.0 / 3.0, "auc": 0.6493386125210912},
    "guided": {"all": 0.10236752136752136, "rare": 1.0 / 3.0, "auc": 0.6493444718960911},
    "daybreak": {"all": 0.1062193732193732, "rare": 1.0 / 6.0, "auc": 0.6356009225541036},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-seeds", default=",".join(str(v) for v in DISCOVERY_SEEDS)
    )
    parser.add_argument(
        "--validation-seeds", default=",".join(str(v) for v in VALIDATION_SEEDS)
    )
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = parser.parse_args()
    args.discovery_seeds = tuple(int(v) for v in args.discovery_seeds.split(",") if v)
    args.validation_seeds = tuple(int(v) for v in args.validation_seeds.split(",") if v)
    args.scenarios = tuple(v for v in args.scenarios.split(",") if v)
    if not args.discovery_seeds or not args.validation_seeds or not args.scenarios:
        parser.error("seeds and scenarios must be non-empty")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def align_frame(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    targets = list(reference.columns[1:])
    if set(frame.columns[1:]) != set(targets):
        raise ValueError("target columns differ")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id duplicates")
    indexed = frame.set_index(frame["record_id"].astype(np.int64))
    ids = reference["record_id"].to_numpy(dtype=np.int64)
    if set(indexed.index) != set(ids):
        raise ValueError("record ids differ")
    output = indexed.loc[ids, targets].reset_index(drop=True)
    output.insert(0, "record_id", ids)
    return output


def validate_shadow_mia(frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Apply the official MIA shape rules to arbitrary shadow target labels."""
    if list(frame.columns) != list(reference.columns) or len(frame) != len(reference):
        raise ValueError("shadow MIA shape differs from reference")
    if frame["record_id"].duplicated().any() or set(frame["record_id"].astype(int)) != set(
        reference["record_id"].astype(int)
    ):
        raise ValueError("shadow MIA record keys differ from reference")
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("shadow MIA values are outside finite [0,1]")
    if any(frame[column].nunique(dropna=False) != len(frame) for column in frame.columns[1:]):
        raise ValueError("shadow MIA contains a target-column tie")


def priority_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = list(frame.columns[1:])
    return np.column_stack(
        [
            frame[column].rank(method="first", ascending=True).to_numpy(dtype=float)
            / (len(frame) + 1.0)
            for column in columns
        ]
    )


def reciprocal_rank_priority(*matrices: np.ndarray, constant: float = 20.0) -> np.ndarray:
    n = matrices[0].shape[0]
    output = np.zeros_like(matrices[0], dtype=float)
    for matrix in matrices:
        descending_rank = n + 1.0 - matrix * (n + 1.0)
        output += 1.0 / (constant + descending_rank)
    return output


def rerank_unlocked(
    baseline: pd.DataFrame,
    priority: np.ndarray,
    locked: np.ndarray,
) -> pd.DataFrame:
    output = baseline.copy()
    ids = output["record_id"].to_numpy(dtype=np.int64)
    targets = list(output.columns[1:])
    unlocked_rows = np.flatnonzero(~locked)
    for target_index, target in enumerate(targets):
        baseline_values = baseline[target].to_numpy(dtype=float)
        available = np.sort(baseline_values[unlocked_rows])
        order = unlocked_rows[
            np.lexsort(
                (
                    ids[unlocked_rows],
                    baseline_values[unlocked_rows],
                    priority[unlocked_rows, target_index],
                )
            )
        ]
        values = baseline_values.copy()
        values[order] = available
        output[target] = values
    return output


def tail_rerank(
    baseline: pd.DataFrame,
    priority: np.ndarray,
    locked: np.ndarray,
    *,
    width: int,
    replace: bool,
) -> pd.DataFrame:
    output = baseline.copy()
    ids = output["record_id"].to_numpy(dtype=np.int64)
    targets = list(output.columns[1:])
    unlocked = np.flatnonzero(~locked)
    for target_index, target in enumerate(targets):
        values = baseline[target].to_numpy(dtype=float)
        descending = np.argsort(-values, kind="mergesort")
        top = descending[:width]
        top_unlocked = top[~locked[top]]
        m = len(top_unlocked)
        if m < 2:
            continue
        if replace:
            chosen_order = unlocked[
                np.lexsort(
                    (
                        ids[unlocked],
                        values[unlocked],
                        -priority[unlocked, target_index],
                    )
                )
            ]
            chosen = chosen_order[:m]
            chosen_set = set(chosen.tolist())
            remainder = np.asarray(
                [row for row in unlocked if int(row) not in chosen_set], dtype=int
            )
            low_values = np.sort(values[unlocked])[:-m]
            high_values = np.sort(values[unlocked])[-m:]
            remainder_order = remainder[
                np.lexsort((ids[remainder], values[remainder]))
            ]
            chosen_ascending = chosen[
                np.lexsort(
                    (
                        ids[chosen],
                        values[chosen],
                        priority[chosen, target_index],
                    )
                )
            ]
            new_values = values.copy()
            new_values[remainder_order] = low_values
            new_values[chosen_ascending] = high_values
            output[target] = new_values
        else:
            available = np.sort(values[top_unlocked])
            order = top_unlocked[
                np.lexsort(
                    (
                        ids[top_unlocked],
                        values[top_unlocked],
                        priority[top_unlocked, target_index],
                    )
                )
            ]
            new_values = values.copy()
            new_values[order] = available
            output[target] = new_values
    return output


def candidate_recipes(
    v4_frame: pd.DataFrame,
    v3_frame: pd.DataFrame,
    daybreak_frame: pd.DataFrame,
    *,
    rare_mask: np.ndarray,
    private_mask: np.ndarray,
) -> dict[str, pd.DataFrame]:
    v3_frame = align_frame(v3_frame, v4_frame)
    daybreak_frame = align_frame(daybreak_frame, v4_frame)
    v4_rank = priority_matrix(v4_frame)
    v3_rank = priority_matrix(v3_frame)
    day_rank = priority_matrix(daybreak_frame)
    locked = np.asarray(rare_mask, dtype=bool) | np.asarray(private_mask, dtype=bool)
    priorities = {
        "rarelock_avg_daybreak10": 0.90 * v4_rank + 0.10 * day_rank,
        "rarelock_avg_daybreak25": 0.75 * v4_rank + 0.25 * day_rank,
        "rarelock_avg_daybreak50": 0.50 * v4_rank + 0.50 * day_rank,
        "rarelock_avg3_equal": (v4_rank + v3_rank + day_rank) / 3.0,
        "rarelock_min_v4_daybreak": np.minimum(v4_rank, day_rank),
        "rarelock_max_v4_daybreak": np.maximum(v4_rank, day_rank),
        "rarelock_rrf_v4_daybreak": reciprocal_rank_priority(v4_rank, day_rank),
    }
    output = {
        "v4": v4_frame.copy(),
        "v3": v3_frame.copy(),
        "daybreak": daybreak_frame.copy(),
    }
    output.update(
        {
            name: rerank_unlocked(v4_frame, priority, locked)
            for name, priority in priorities.items()
        }
    )
    average = priorities["rarelock_avg_daybreak50"]
    minimum = priorities["rarelock_min_v4_daybreak"]
    maximum = priorities["rarelock_max_v4_daybreak"]
    for width in (12, 15, 20):
        output[f"tail{width}_reorder_avg"] = tail_rerank(
            v4_frame, average, locked, width=width, replace=False
        )
    output["tail15_replace_avg"] = tail_rerank(
        v4_frame, average, locked, width=15, replace=True
    )
    output["tail15_replace_min"] = tail_rerank(
        v4_frame, minimum, locked, width=15, replace=True
    )
    output["tail15_replace_max"] = tail_rerank(
        v4_frame, maximum, locked, width=15, replace=True
    )
    return output


def known_private_mask(
    world: shadow.ShadowWorld,
    rare_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    candidate_pool = world.attacker_pool["pool_id"].to_numpy(dtype=np.int64)
    member_pool = set(
        int(value)
        for frame in world.target_pools.values()
        for value in frame["pool_id"].to_numpy(dtype=np.int64)
    )
    nonmember = np.asarray([value not in member_pool for value in candidate_pool])
    rng = np.random.default_rng(seed)
    rare_pool = np.flatnonzero(nonmember & rare_mask)
    common_pool = np.flatnonzero(nonmember & ~rare_mask)
    if len(rare_pool) < 13 or len(common_pool) < 150:
        raise RuntimeError("insufficient known-private rows")
    chosen = np.concatenate(
        [
            rng.choice(rare_pool, 13, replace=False),
            rng.choice(common_pool, 150, replace=False),
        ]
    )
    output = np.zeros(len(candidate_pool), dtype=bool)
    output[chosen] = True
    return output


def count_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 9157)
    values = np.asarray([25] * 13 + [26] * 6 + [27], dtype=int)
    return values[rng.permutation(len(values))]


def scenario_release(
    source: pd.DataFrame,
    background: pd.DataFrame,
    scenario: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if scenario in {"raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent"}:
        return shadow.scenario_release(source, background, scenario, cfg, rng)
    noise_fraction = {"noise_0p20": 0.20, "noise_0p40": 0.40}.get(scenario)
    if noise_fraction is None:
        raise ValueError(f"unknown scenario: {scenario}")
    output = source.copy()
    columns = list(shadow.CONTINUOUS_COLUMNS)
    scale = background[columns].std(ddof=0).to_numpy(dtype=float)
    noise = rng.normal(
        0.0,
        noise_fraction * np.where(scale > 0.0, scale, 1.0),
        size=(len(output), len(columns)),
    )
    for index, column in enumerate(columns):
        output[column] = output[column].to_numpy(dtype=float) + noise[:, index]
    horizon = float(cfg["onset"]["horizon_years"])
    output["time"] = output["time"].to_numpy(dtype=float) + rng.normal(
        0.0, noise_fraction * horizon, size=len(output)
    )
    output = shadow.coerce_release(output, cfg)
    return output.iloc[rng.permutation(len(output))].reset_index(drop=True)


def score_frame(
    frame: pd.DataFrame,
    world: shadow.ShadowWorld,
    cfg: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    aligned_pool = world.attacker_pool.set_index("record_id")
    ids = frame["record_id"].to_numpy(dtype=np.int64)
    pool_ids = aligned_pool.loc[ids, "pool_id"].to_numpy(dtype=np.int64)
    rare = aligned_pool.loc[ids, "is_rare"].to_numpy(dtype=bool)
    target_rows: list[dict[str, Any]] = []
    for target in frame.columns[1:]:
        target_pool = set(
            world.target_pools[str(target)]["pool_id"].to_numpy(dtype=np.int64)
        )
        truth = np.asarray([value in target_pool for value in pool_ids], dtype=int)
        result = score_mia(
            frame[target].to_numpy(dtype=float), truth, cfg, rare_mask=rare
        )
        target_rows.append(
            {
                "target": str(target),
                "power": float(result.score),
                "all": float(result.breakdown["tpr_at_low_fpr"]),
                "rare": float(result.breakdown["tpr_rare"]),
                "auc": float(result.breakdown["auc"]),
            }
        )
    detail = pd.DataFrame(target_rows)
    return (
        {
            "power": float(detail["power"].mean()),
            "all": float(detail["all"].mean()),
            "rare": float(detail["rare"].mean()),
            "auc": float(detail["auc"].mean()),
        },
        target_rows,
    )


def inventory_production() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    unique: dict[str, pd.DataFrame] = {}
    for directory in sorted(TEAM.iterdir()):
        path = directory / "F_mia.csv"
        if not path.is_file():
            continue
        digest = sha256(path)
        frame = pd.read_csv(path)
        rows.append(
            {
                "candidate": directory.name,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "rows": len(frame),
                "targets": len(frame.columns) - 1,
            }
        )
        unique.setdefault(digest, frame)
    return pd.DataFrame(rows), unique


def production_diagnostics(
    variants: dict[str, pd.DataFrame],
    baseline: pd.DataFrame,
    rare_mask: np.ndarray,
    private_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ids = baseline["record_id"].to_numpy(dtype=np.int64)
    for name, frame in variants.items():
        validate_mia(frame, baseline)
        changed_cells = 0
        top15_changes = 0
        rare_order_equal = True
        private_bottom_equal = True
        inversions = 0
        for target in baseline.columns[1:]:
            base_values = baseline[target].to_numpy(dtype=float)
            values = frame[target].to_numpy(dtype=float)
            changed_cells += int(np.count_nonzero(~np.isclose(values, base_values)))
            base_order = np.argsort(-base_values, kind="mergesort")
            order = np.argsort(-values, kind="mergesort")
            top15_changes += len(set(base_order[:15]) ^ set(order[:15])) // 2
            rare_order_equal &= np.array_equal(
                ids[np.flatnonzero(rare_mask)[np.argsort(-values[rare_mask], kind="mergesort")]],
                ids[np.flatnonzero(rare_mask)[np.argsort(-base_values[rare_mask], kind="mergesort")]],
            )
            private_bottom_equal &= set(order[-int(private_mask.sum()) :]) == set(
                np.flatnonzero(private_mask)
            )
            # Count inversions relative to v4 in O(n log n).
            position = np.empty(len(order), dtype=int)
            position[base_order] = np.arange(len(order))
            sequence = position[order]
            bit = np.zeros(len(order) + 1, dtype=int)
            seen = 0
            for value in sequence:
                index = int(value) + 1
                prefix = 0
                cursor = index
                while cursor:
                    prefix += bit[cursor]
                    cursor -= cursor & -cursor
                inversions += seen - prefix
                cursor = index
                while cursor < len(bit):
                    bit[cursor] += 1
                    cursor += cursor & -cursor
                seen += 1
        output_path = OUT / "production_variants" / f"F_mia_{name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        rows.append(
            {
                "variant": name,
                "sha256": sha256(output_path),
                "changed_cells_from_v4": changed_cells,
                "top15_set_changes_total": top15_changes,
                "kendall_inversions_total": inversions,
                "rare_order_equal_v4": bool(rare_order_equal),
                "private_bottom_equal_v4": bool(private_bottom_equal),
                "path": str(output_path.resolve()),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    plausible = results.loc[results["scenario"].isin(PLAUSIBLE_SCENARIOS)].copy()
    base = plausible.loc[
        plausible["variant"].eq("v4"),
        ["split", "seed", "scenario", "power", "all", "rare", "auc"],
    ].rename(
        columns={
            "power": "base_power",
            "all": "base_all",
            "rare": "base_rare",
            "auc": "base_auc",
        }
    )
    plausible = plausible.merge(base, on=["split", "seed", "scenario"], how="left")
    for metric in ("power", "all", "rare", "auc"):
        plausible[f"delta_{metric}"] = plausible[metric] - plausible[f"base_{metric}"]
    return (
        plausible.groupby(["split", "variant"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean_power=("power", "mean"),
            mean_delta_power=("delta_power", "mean"),
            p10_delta_power=("delta_power", lambda v: float(np.quantile(v, 0.10))),
            minimum_delta_power=("delta_power", "min"),
            maximum_delta_power=("delta_power", "max"),
            win_rate=("delta_power", lambda v: float(np.mean(v > 1e-12))),
            loss_rate=("delta_power", lambda v: float(np.mean(v < -1e-12))),
            mean_delta_all=("delta_all", "mean"),
            mean_delta_rare=("delta_rare", "mean"),
            minimum_delta_rare=("delta_rare", "min"),
            mean_delta_auc=("delta_auc", "mean"),
        )
        .sort_values(["split", "mean_delta_power"], ascending=[True, False])
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(DIST / "B_self.csv").set_index("record_id").loc[
        candidates["record_id"]
    ].reset_index()
    background = pd.read_csv(DIST / "A_bg.csv").reset_index(drop=True)
    rare_mask = b_self["is_rare"].astype(bool).to_numpy()
    detector = shadow.load_rare_detector(DIST)
    cfg = load_config(KIT_CONFIG_PATH)
    spec = shadow.ShadowSpec(
        n_targets=20,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=508,
        minimum_shared=25,
        maximum_shared=27,
    )
    shadow.validate_source_data(candidates, b_self, background, spec)

    inventory, unique_files = inventory_production()
    inventory.to_csv(OUT / "production_inventory.csv", index=False)
    production_v3 = pd.read_csv(TEAM / "attack-candidate-v3" / "F_mia.csv")
    production_v4 = pd.read_csv(TEAM / "attack-candidate-v4" / "F_mia.csv")
    production_daybreak = pd.read_csv(
        TEAM / "attack-candidate-daybreak-independent" / "F_mia.csv"
    )
    production_v3 = align_frame(production_v3, production_v4)
    production_daybreak = align_frame(production_daybreak, production_v4)
    private_ids = set(infer_private_bottom_ids(production_v3))
    private_mask_production = candidates["record_id"].isin(private_ids).to_numpy()
    production_variants = candidate_recipes(
        production_v4,
        production_v3,
        production_daybreak,
        rare_mask=rare_mask,
        private_mask=private_mask_production,
    )
    production_checks = production_diagnostics(
        production_variants,
        production_v4,
        rare_mask,
        private_mask_production,
    )
    production_checks.to_csv(OUT / "production_variant_checks.csv", index=False)

    result_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    world_checks: list[dict[str, Any]] = []
    original_allocator = shadow.allocate_shared_counts
    try:
        for split, seeds in (
            ("discovery", args.discovery_seeds),
            ("validation", args.validation_seeds),
        ):
            for seed_index, seed in enumerate(seeds):
                counts = count_vector(seed)
                shadow.allocate_shared_counts = (
                    lambda spec, rng, values=counts: values.copy()
                )
                print(f"{split} seed={seed}: construct world", flush=True)
                world = shadow.build_shadow_world(
                    candidates, b_self, background, detector, spec, seed
                )
                check = shadow.validate_world(world, spec)
                world_checks.append(
                    {
                        "split": split,
                        "seed": seed,
                        "count_25": int(np.count_nonzero(counts == 25)),
                        "count_26": int(np.count_nonzero(counts == 26)),
                        "count_27": int(np.count_nonzero(counts == 27)),
                        **check,
                    }
                )
                known_private = known_private_mask(
                    world, rare_mask, seed=seed * 1000 + 17
                )
                for scenario_index, scenario in enumerate(args.scenarios):
                    print(f"  scenario={scenario}", flush=True)
                    releases = {}
                    for target_id, source in world.mia_sources.items():
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, seed_index, scenario_index, int(target_id), 617]
                            )
                        )
                        releases[target_id] = scenario_release(
                            source,
                            world.attack_background,
                            scenario,
                            cfg,
                            rng,
                        )
                    v4_frame, v4_variants, _ = v4.build_mia_v4(
                        candidates,
                        world.attack_background,
                        releases,
                        known_private,
                        minimum=25,
                        maximum=27,
                        total=508,
                        transport_epsilon=0.35,
                        include_domias=False,
                        density_weight=v2.DEFAULT_MIA_DENSITY_WEIGHT,
                        rare_mask=rare_mask,
                        rare_weight=v2.DEFAULT_MIA_RARE_WEIGHT,
                    )
                    v3_frame = v4_variants[
                        "v3_private_demotion_transport_reference"
                    ]
                    daybreak_frame, _, _ = build_mia_independent(
                        candidates,
                        world.attack_background,
                        releases,
                        known_private,
                    )
                    variants = candidate_recipes(
                        v4_frame,
                        v3_frame,
                        daybreak_frame,
                        rare_mask=rare_mask,
                        private_mask=known_private,
                    )
                    for name, frame in variants.items():
                        try:
                            validate_shadow_mia(frame, v4_frame)
                        except Exception as error:
                            raise ValueError(
                                f"shadow variant {name} failed validation: "
                                f"columns={list(frame.columns)}"
                            ) from error
                        aggregate, details = score_frame(frame, world, cfg)
                        result_rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "scenario": scenario,
                                "variant": name,
                                **aggregate,
                            }
                        )
                        target_rows.extend(
                            {
                                "split": split,
                                "seed": seed,
                                "scenario": scenario,
                                "variant": name,
                                **row,
                            }
                            for row in details
                        )
    finally:
        shadow.allocate_shared_counts = original_allocator

    results = pd.DataFrame(result_rows)
    targets = pd.DataFrame(target_rows)
    summary = summarize(results)
    scenario_summary = (
        results.groupby(["split", "scenario", "variant"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean_power=("power", "mean"),
            minimum_power=("power", "min"),
            maximum_power=("power", "max"),
            mean_all=("all", "mean"),
            mean_rare=("rare", "mean"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["split", "scenario", "mean_power"], ascending=[True, True, False])
    )
    target_summary = (
        targets.loc[targets["scenario"].isin(PLAUSIBLE_SCENARIOS)]
        .groupby(["split", "target", "variant"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean_power=("power", "mean"),
            p10_power=("power", lambda v: float(np.quantile(v, 0.10))),
            mean_all=("all", "mean"),
            mean_rare=("rare", "mean"),
            mean_auc=("auc", "mean"),
        )
        .sort_values(["split", "target", "mean_power"], ascending=[True, True, False])
    )
    results.to_csv(OUT / "scores_by_world.csv", index=False)
    targets.to_csv(OUT / "scores_by_target.csv", index=False)
    summary.to_csv(OUT / "summary_by_split.csv", index=False)
    scenario_summary.to_csv(OUT / "summary_by_scenario.csv", index=False)
    target_summary.to_csv(OUT / "summary_by_target.csv", index=False)
    pd.DataFrame(world_checks).to_csv(OUT / "world_checks.csv", index=False)

    discovery = summary.loc[summary["split"].eq("discovery")].sort_values(
        "mean_delta_power", ascending=False
    )
    validation = summary.loc[summary["split"].eq("validation")]
    discovery_safe = discovery.loc[
        ~discovery["variant"].isin(["v4", "v3", "daybreak"])
    ]
    selected_name = str(discovery_safe.iloc[0]["variant"])
    selected_discovery = discovery_safe.iloc[0]
    selected_validation = validation.loc[
        validation["variant"].eq(selected_name)
    ].iloc[0]
    selected_production = production_checks.loc[
        production_checks["variant"].eq(selected_name)
    ].iloc[0]
    gate = {
        "candidate": selected_name,
        "requirements_fixed_before_validation": {
            "discovery_mean_delta_power_gt": 0.0025,
            "validation_mean_delta_power_gt": 0.0025,
            "validation_win_rate_gte": 0.75,
            "validation_p10_delta_power_gte": -0.001,
            "validation_minimum_delta_rare_gte": 0.0,
            "production_rare_order_equal_v4": True,
            "production_private_bottom_equal_v4": True,
        },
    }
    checks = {
        "discovery_mean": float(selected_discovery["mean_delta_power"]) > 0.0025,
        "validation_mean": float(selected_validation["mean_delta_power"]) > 0.0025,
        "validation_win_rate": float(selected_validation["win_rate"]) >= 0.75,
        "validation_p10": float(selected_validation["p10_delta_power"]) >= -0.001,
        "validation_rare": float(selected_validation["minimum_delta_rare"]) >= -1e-12,
        "production_rare_order": bool(selected_production["rare_order_equal_v4"]),
        "production_private_bottom": bool(
            selected_production["private_bottom_equal_v4"]
        ),
    }
    gate["checks"] = checks
    gate["passed"] = bool(all(checks.values()))
    gate["official_constraint"] = (
        "Even a gate pass is insufficient if the recipe merely re-expresses the guided "
        "tail signal, because guided and v4 were officially tied within display rounding."
    )
    gate["decision"] = (
        "consider_replacement_subject_to_official_constraint"
        if gate["passed"]
        else "retain_v4"
    )

    report = {
        "purpose": "truth-known local MIA recipe recheck; not a leaderboard forecast",
        "parameters": {
            "discovery_seeds": list(args.discovery_seeds),
            "validation_seeds": list(args.validation_seeds),
            "scenarios": list(args.scenarios),
            "plausible_scenarios": list(PLAUSIBLE_SCENARIOS),
            "counts_per_world": {"25": 13, "26": 6, "27": 1, "total": 508},
            "known_private": {"total": 163, "rare": 13, "common": 150},
            "domias": False,
        },
        "official_constraints": OFFICIAL,
        "inventory": {
            "candidate_files": len(inventory),
            "unique_hashes": len(unique_files),
        },
        "selection_gate": gate,
        "discovery_top": discovery.head(10).to_dict(orient="records"),
        "validation_selected": selected_validation.to_dict(),
        "production_selected": selected_production.to_dict(),
        "limitations": [
            "Shadow values compare recipes and do not predict private leaderboard scores.",
            "Official v4-vs-guided equality and v4 rare strength override weak shadow differences.",
            "The independent scenario is a null control and is excluded from selection means.",
            "Candidate selection uses discovery only; validation is read once through the fixed gate.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(discovery.head(8).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("validation selected")
    print(selected_validation.to_string())
    print(json.dumps(gate, indent=2))
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
