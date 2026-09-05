"""Research-only MIA low-FPR benchmark for the final PWSCUP2026 attempt.

This script deliberately keeps discovery and held-out validation worlds apart.
It only creates normal, finite, tie-free F_mia matrices beneath local-output;
it never reads a token, packages an archive, edits a submission candidate, or
contacts CodaBench.

The main hypothesis is channel surgery rather than row-label recovery:

* Daybreak-independent has the best *official aggregate all-channel* TPR.
* v4 has the best observed rare ordering (and the best AUC).
* therefore keep every Daybreak score slot, but permute only the 24 rare rows
  so their within-rare order is exactly v4's.

That recipe uses only complete pre-existing rank matrices and the rare mask
that was already recoverable from the two locally generated submissions.  No
record is declared a member/non-member from leaderboard feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "local-output" / "attack-v7" / "mia-lowfpr"
TEAM = ROOT / "submissions" / "prelim" / "team24"
DIST = ROOT / "participant_data"
LEGACY_PATH = ROOT / "local-output" / "attack-v6" / "mia-recheck" / "benchmark_mia.py"
DIAGNOSTICS_PATH = ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json"

OFFICIAL = {
    "v3": {"all": 0.09266381766381768, "rare": 1.0 / 3.0, "auc": 0.6434884498600183},
    "v4": {"all": 0.10236752136752136, "rare": 1.0 / 3.0, "auc": 0.6493386125210912},
    "daybreak": {"all": 0.1062193732193732, "rare": 1.0 / 6.0, "auc": 0.6356009225541036},
    "guided": {"all": 0.10236752136752136, "rare": 1.0 / 3.0, "auc": 0.6493444718960911},
}

DEFAULT_DISCOVERY = tuple(range(20261701, 20261713))
DEFAULT_VALIDATION = tuple(range(20261801, 20261819))
DEFAULT_SCENARIOS = ("noise_0p10", "noise_0p40")


def _load_legacy():
    spec = importlib.util.spec_from_file_location("mia_recheck_v6", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark helpers: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy()


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def parse_strings(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(",") if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-seeds", default=",".join(map(str, DEFAULT_DISCOVERY)))
    parser.add_argument("--validation-seeds", default=",".join(map(str, DEFAULT_VALIDATION)))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    args = parser.parse_args()
    args.discovery_seeds = parse_ints(args.discovery_seeds)
    args.validation_seeds = parse_ints(args.validation_seeds)
    args.scenarios = parse_strings(args.scenarios)
    if not args.discovery_seeds or not args.validation_seeds or not args.scenarios:
        parser.error("both splits and the scenario set must be non-empty")
    if set(args.discovery_seeds) & set(args.validation_seeds):
        parser.error("discovery and validation seeds must be disjoint")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def align(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return legacy.align_frame(frame, reference)


def mask_from_ids(frame: pd.DataFrame, values: Iterable[int]) -> np.ndarray:
    return frame["record_id"].astype(np.int64).isin(set(map(int, values))).to_numpy()


def lock_mask_order(
    source: pd.DataFrame,
    order_reference: pd.DataFrame,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Keep source score slots but give masked rows reference's relative order.

    This is a pure within-column permutation.  Consequently it neither changes
    the source's score distribution nor moves an unmasked row.  In particular,
    the Daybreak common-row ranking remains byte-for-byte numerically intact.
    """

    source = align(source, order_reference)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(source) or not mask.any():
        raise ValueError("mask has the wrong shape or is empty")
    output = source.copy()
    ids = source["record_id"].to_numpy(dtype=np.int64)
    rows = np.flatnonzero(mask)
    for target in source.columns[1:]:
        source_values = source[target].to_numpy(dtype=float)
        reference_values = order_reference[target].to_numpy(dtype=float)
        # Ascending-to-ascending assignment preserves the full strict order.
        reference_order = rows[
            np.lexsort((ids[rows], reference_values[rows]))
        ]
        values = source_values.copy()
        values[reference_order] = np.sort(source_values[rows])
        output[target] = values
    return output


def rank_fusion(
    v4_frame: pd.DataFrame,
    daybreak_frame: pd.DataFrame,
    locked: np.ndarray,
    daybreak_weight: float,
) -> pd.DataFrame:
    """Borda/rank fusion control, with the same private+rare lock as v6."""

    v4_rank = legacy.priority_matrix(v4_frame)
    day_rank = legacy.priority_matrix(daybreak_frame)
    priority = (1.0 - daybreak_weight) * v4_rank + daybreak_weight * day_rank
    return legacy.rerank_unlocked(v4_frame, priority, locked)


def recipes(
    v4_frame: pd.DataFrame,
    v3_frame: pd.DataFrame,
    daybreak_frame: pd.DataFrame,
    rare_mask: np.ndarray,
    private_mask: np.ndarray,
) -> dict[str, pd.DataFrame]:
    v3_frame = align(v3_frame, v4_frame)
    daybreak_frame = align(daybreak_frame, v4_frame)
    rare_mask = np.asarray(rare_mask, dtype=bool)
    private_mask = np.asarray(private_mask, dtype=bool)

    # Guaranteed-private rows are already below every eligible row in both
    # source attacks.  Only the non-private rare rows can place the first rare
    # false positive, so their order is the score-relevant invariant.
    eligible_rare = rare_mask & ~private_mask
    day_rarelock = lock_mask_order(daybreak_frame, v4_frame, eligible_rare)
    high12 = v4_frame.copy()
    for target in ("1", "2"):
        high12[target] = daybreak_frame[target].to_numpy(dtype=float)
    high12 = lock_mask_order(high12, v4_frame, eligible_rare)

    locked = rare_mask | private_mask
    return {
        "v4": v4_frame.copy(),
        "v3": v3_frame.copy(),
        "daybreak": daybreak_frame.copy(),
        "daybreak_v4_rarelock": day_rarelock,
        "daybreak_v4_rarelock_high12": high12,
        "rankfusion_day25_rarelock": rank_fusion(
            v4_frame, daybreak_frame, locked, 0.25
        ),
        "rankfusion_day50_rarelock": rank_fusion(
            v4_frame, daybreak_frame, locked, 0.50
        ),
    }


def descending_ids(frame: pd.DataFrame, target: str) -> np.ndarray:
    values = frame[target].to_numpy(dtype=float)
    ids = frame["record_id"].to_numpy(dtype=np.int64)
    return ids[np.argsort(-values, kind="mergesort")]


def rank_of_mask(frame: pd.DataFrame, target: str, mask: np.ndarray) -> list[int]:
    order = np.argsort(-frame[target].to_numpy(dtype=float), kind="mergesort")
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(1, len(order) + 1)
    return sorted(map(int, positions[np.flatnonzero(mask)]))


def production_audit(
    variants: dict[str, pd.DataFrame],
    v4_frame: pd.DataFrame,
    daybreak_frame: pd.DataFrame,
    rare_mask: np.ndarray,
    private_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    output_dir = OUT / "production_variants"
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = v4_frame["record_id"].to_numpy(dtype=np.int64)
    rare_rows = np.flatnonzero(rare_mask)
    private_rows = set(np.flatnonzero(private_mask).tolist())

    for name, frame in variants.items():
        legacy.validate_mia(frame, v4_frame)
        rare_equal = True
        private_bottom = True
        changed_v4 = 0
        changed_day = 0
        top16_day_changes = 0
        top37_day_changes = 0
        for target in v4_frame.columns[1:]:
            values = frame[target].to_numpy(dtype=float)
            v4_values = v4_frame[target].to_numpy(dtype=float)
            day_values = daybreak_frame[target].to_numpy(dtype=float)
            changed_v4 += int(np.count_nonzero(values != v4_values))
            changed_day += int(np.count_nonzero(values != day_values))
            rare_order = rare_rows[
                np.argsort(-values[rare_rows], kind="mergesort")
            ]
            v4_rare_order = rare_rows[
                np.argsort(-v4_values[rare_rows], kind="mergesort")
            ]
            rare_equal &= np.array_equal(rare_order, v4_rare_order)
            order = np.argsort(-values, kind="mergesort")
            private_bottom &= set(order[-len(private_rows) :]) == private_rows
            day_order = np.argsort(-day_values, kind="mergesort")
            top16_day_changes += len(set(order[:16]) ^ set(day_order[:16])) // 2
            top37_day_changes += len(set(order[:37]) ^ set(day_order[:37])) // 2

        path = output_dir / f"F_mia_{name}.csv"
        frame.to_csv(path, index=False)
        rows.append(
            {
                "variant": name,
                "sha256": sha256(path),
                "changed_cells_vs_v4": changed_v4,
                "changed_cells_vs_daybreak": changed_day,
                "rare_order_equal_v4": bool(rare_equal),
                "private_bottom_set_preserved": bool(private_bottom),
                "top16_set_changes_vs_daybreak": top16_day_changes,
                "top37_set_changes_vs_daybreak": top37_day_changes,
                "path": str(path.resolve()),
            }
        )

    # Extra evidence specific to the full Daybreak rare-lock hypothesis.
    candidate = variants["daybreak_v4_rarelock"]
    changed_slots: list[dict[str, Any]] = []
    for target in v4_frame.columns[1:]:
        day_order = descending_ids(daybreak_frame, target)
        cand_order = descending_ids(candidate, target)
        for rank, (old, new) in enumerate(zip(day_order, cand_order, strict=True), 1):
            if old != new and (old in set(ids[rare_mask]) or new in set(ids[rare_mask])):
                changed_slots.append(
                    {"target": target, "rank": rank, "daybreak_record": int(old), "candidate_record": int(new)}
                )
    pd.DataFrame(changed_slots).to_csv(OUT / "rarelock_changed_slots.csv", index=False)
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    base = scores.loc[
        scores["variant"].eq("v4"),
        ["split", "seed", "scenario", "power", "all", "rare", "auc"],
    ].rename(columns={metric: f"base_{metric}" for metric in ("power", "all", "rare", "auc")})
    joined = scores.merge(base, on=["split", "seed", "scenario"], how="left")
    for metric in ("power", "all", "rare", "auc"):
        joined[f"delta_{metric}"] = joined[metric] - joined[f"base_{metric}"]
    return (
        joined.groupby(["split", "variant"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean_power=("power", "mean"),
            mean_delta_power=("delta_power", "mean"),
            median_delta_power=("delta_power", "median"),
            p10_delta_power=("delta_power", lambda x: float(np.quantile(x, 0.10))),
            minimum_delta_power=("delta_power", "min"),
            win_rate=("delta_power", lambda x: float(np.mean(x > 1e-12))),
            loss_rate=("delta_power", lambda x: float(np.mean(x < -1e-12))),
            mean_delta_all=("delta_all", "mean"),
            mean_delta_rare=("delta_rare", "mean"),
            minimum_delta_rare=("delta_rare", "min"),
            mean_delta_auc=("delta_auc", "mean"),
        )
        .sort_values(["split", "mean_delta_power"], ascending=[True, False])
    )


def official_structural_bound(
    daybreak_frame: pd.DataFrame,
    candidate: pd.DataFrame,
    rare_mask: np.ndarray,
) -> dict[str, Any]:
    """Bound candidate aggregate all-TPR from public bands and changed slots.

    For targets 1/2 the official band is 4, so the operating prefix contains
    at least ranks 1..29.  For every other target it contains at most 1..16.
    Only changed rare score slots that cross one of those possible prefixes can
    alter the number of true positives before the 11th false positive.
    """

    targets = list(daybreak_frame.columns[1:])
    relevant: dict[str, list[int]] = {}
    safe_inside: dict[str, list[int]] = {}
    for target in targets:
        day_order = descending_ids(daybreak_frame, target)
        cand_order = descending_ids(candidate, target)
        changed = [
            rank
            for rank, (old, new) in enumerate(zip(day_order, cand_order, strict=True), 1)
            if old != new
        ]
        if target in {"1", "2"}:
            # Ranks <=29 are inside every feasible band-4 operating prefix;
            # swapping only among those ranks cannot change the prefix count.
            always_in = [rank for rank in changed if rank <= 29]
            crossing = [rank for rank in changed if 29 < rank <= 37]
            if always_in:
                safe_inside[target] = always_in
        else:
            crossing = [rank for rank in changed if 10 < rank <= 16]
        if crossing:
            relevant[target] = crossing

    # One changed slot can alter a target's TPR by at most 1/n.  Use n=25
    # for the conservative aggregate bound.  (Multiple slots are summed.)
    maximum_aggregate_change = sum(len(v) for v in relevant.values()) / (20.0 * 25.0)
    lower = OFFICIAL["daybreak"]["all"] - maximum_aggregate_change
    upper = OFFICIAL["daybreak"]["all"] + maximum_aggregate_change
    return {
        "daybreak_official_all": OFFICIAL["daybreak"]["all"],
        "v4_official_all": OFFICIAL["v4"]["all"],
        "candidate_changed_slots_that_can_cross_an_official_operating_prefix": relevant,
        "changed_slots_always_inside_band4_prefix": safe_inside,
        "conservative_max_abs_delta_vs_daybreak_all": maximum_aggregate_change,
        "candidate_all_lower_bound": lower,
        "candidate_all_upper_bound": upper,
        "lower_bound_delta_vs_v4_all": lower - OFFICIAL["v4"]["all"],
        "rare_tpr_exactly_v4_by_order_invariance": OFFICIAL["v4"]["rare"],
        "note": "This is an aggregate-all bound, not a bound on mean target-wise max(all, rare).",
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    private_ids = tuple(map(int, diagnostics["mia"]["private_record_ids"]))
    eligible_rare_ids = tuple(map(int, diagnostics["mia"]["eligible_rare_record_ids"]))
    # Recovering the complete production rare mask from the local toy B would
    # be wrong.  The 11 non-private rare rows are already present in the v5/v6
    # diagnostics and are sufficient: all 163 guaranteed-private rows sit
    # below them, so a private rare row cannot be the first rare false positive.
    production_v3 = pd.read_csv(TEAM / "attack-candidate-v3" / "F_mia.csv")
    production_v4 = pd.read_csv(TEAM / "attack-candidate-v4" / "F_mia.csv")
    production_day = pd.read_csv(
        TEAM / "attack-candidate-daybreak-independent" / "F_mia.csv"
    )
    production_v3 = align(production_v3, production_v4)
    production_day = align(production_day, production_v4)
    production_private = mask_from_ids(production_v4, private_ids)
    production_rare = mask_from_ids(production_v4, eligible_rare_ids)
    if int(production_rare.sum()) != 11 or bool((production_rare & production_private).any()):
        raise AssertionError("production eligible-rare/private reconstruction failed")

    production_variants = recipes(
        production_v4,
        production_v3,
        production_day,
        production_rare,
        production_private,
    )
    audit = production_audit(
        production_variants,
        production_v4,
        production_day,
        production_rare,
        production_private,
    )
    audit.to_csv(OUT / "production_audit.csv", index=False)
    structural = official_structural_bound(
        production_day,
        production_variants["daybreak_v4_rarelock"],
        production_rare,
    )
    (OUT / "official_structural_bound.json").write_text(
        json.dumps(structural, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    candidates = pd.read_csv(DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(DIST / "B_self.csv").set_index("record_id").loc[
        candidates["record_id"]
    ].reset_index()
    background = pd.read_csv(DIST / "A_bg.csv").reset_index(drop=True)
    shadow_rare = b_self["is_rare"].astype(bool).to_numpy()
    detector = legacy.shadow.load_rare_detector(DIST)
    cfg = legacy.load_config(legacy.KIT_CONFIG_PATH)
    spec = legacy.shadow.ShadowSpec(
        n_targets=20,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=508,
        minimum_shared=25,
        maximum_shared=27,
    )
    legacy.shadow.validate_source_data(candidates, b_self, background, spec)

    aggregate_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    original_allocator = legacy.shadow.allocate_shared_counts
    try:
        for split, seeds in (
            ("discovery", args.discovery_seeds),
            ("validation", args.validation_seeds),
        ):
            for seed_index, seed in enumerate(seeds):
                counts = legacy.count_vector(seed)
                legacy.shadow.allocate_shared_counts = (
                    lambda spec, rng, values=counts: values.copy()
                )
                print(f"{split} seed={seed}: construct world", flush=True)
                world = legacy.shadow.build_shadow_world(
                    candidates, b_self, background, detector, spec, seed
                )
                world_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "count_25": int(np.count_nonzero(counts == 25)),
                        "count_26": int(np.count_nonzero(counts == 26)),
                        "count_27": int(np.count_nonzero(counts == 27)),
                        **legacy.shadow.validate_world(world, spec),
                    }
                )
                try:
                    private_mask = legacy.known_private_mask(
                        world, shadow_rare, seed=seed * 1000 + 17
                    )
                except RuntimeError as error:
                    # Production reveals 13 rare guaranteed-private rows.  A
                    # shadow world with fewer than 13 rare nonmembers cannot
                    # reproduce that public condition and is rejected before
                    # constructing or scoring any recipe.
                    world_rows[-1]["eligible_for_attack_benchmark"] = False
                    world_rows[-1]["rejection_reason"] = str(error)
                    print(f"  reject world: {error}", flush=True)
                    continue
                world_rows[-1]["eligible_for_attack_benchmark"] = True
                world_rows[-1]["rejection_reason"] = ""
                for scenario_index, scenario in enumerate(args.scenarios):
                    print(f"  scenario={scenario}", flush=True)
                    releases = {}
                    for target, source in world.mia_sources.items():
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [seed, seed_index, scenario_index, int(target), 1707]
                            )
                        )
                        releases[target] = legacy.scenario_release(
                            source, world.attack_background, scenario, cfg, rng
                        )
                    v4_frame, v4_variants, _ = legacy.v4.build_mia_v4(
                        candidates,
                        world.attack_background,
                        releases,
                        private_mask,
                        minimum=25,
                        maximum=27,
                        total=508,
                        transport_epsilon=0.35,
                        include_domias=False,
                        density_weight=legacy.v2.DEFAULT_MIA_DENSITY_WEIGHT,
                        rare_mask=shadow_rare,
                        rare_weight=legacy.v2.DEFAULT_MIA_RARE_WEIGHT,
                    )
                    v3_frame = v4_variants["v3_private_demotion_transport_reference"]
                    day_frame, _, _ = legacy.build_mia_independent(
                        candidates, world.attack_background, releases, private_mask
                    )
                    variants = recipes(
                        v4_frame,
                        v3_frame,
                        day_frame,
                        shadow_rare,
                        private_mask,
                    )
                    for name, frame in variants.items():
                        legacy.validate_shadow_mia(frame, v4_frame)
                        aggregate, details = legacy.score_frame(frame, world, cfg)
                        aggregate_rows.append(
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
                                **detail,
                            }
                            for detail in details
                        )
    finally:
        legacy.shadow.allocate_shared_counts = original_allocator

    scores = pd.DataFrame(aggregate_rows)
    target_scores = pd.DataFrame(target_rows)
    summary = summarize(scores)
    scores.to_csv(OUT / "scores_by_world.csv", index=False)
    target_scores.to_csv(OUT / "scores_by_target.csv", index=False)
    summary.to_csv(OUT / "summary_by_split.csv", index=False)
    pd.DataFrame(world_rows).to_csv(OUT / "world_checks.csv", index=False)

    selected = "daybreak_v4_rarelock"
    discovery = summary.loc[
        (summary["split"] == "discovery") & (summary["variant"] == selected)
    ].iloc[0]
    validation = summary.loc[
        (summary["split"] == "validation") & (summary["variant"] == selected)
    ].iloc[0]
    production = audit.loc[audit["variant"] == selected].iloc[0]
    requirements = {
        "discovery_mean_delta_power_gt_0": float(discovery["mean_delta_power"]) > 0.0,
        "validation_mean_delta_power_gt_0": float(validation["mean_delta_power"]) > 0.0,
        "validation_median_delta_power_gt_0": float(validation["median_delta_power"]) > 0.0,
        "validation_win_rate_gte_0p60": float(validation["win_rate"]) >= 0.60,
        "validation_p10_delta_power_gte_minus_0p002": float(validation["p10_delta_power"]) >= -0.002,
        "validation_minimum_delta_rare_gte_0": float(validation["minimum_delta_rare"]) >= -1e-12,
        "production_rare_order_exact_v4": bool(production["rare_order_equal_v4"]),
        "production_private_bottom_preserved": bool(production["private_bottom_set_preserved"]),
        "official_all_lower_bound_gt_v4": float(structural["lower_bound_delta_vs_v4_all"]) > 0.0,
    }
    gate_passed = all(requirements.values())
    report = {
        "purpose": "normal-format MIA low-FPR research; local shadow values are not leaderboard predictions",
        "predeclared_candidate": selected,
        "official_evidence": OFFICIAL,
        "parameters": {
            "discovery_seeds": list(args.discovery_seeds),
            "validation_seeds": list(args.validation_seeds),
            "scenarios": list(args.scenarios),
            "shadow_counts": {"25": 13, "26": 6, "27": 1, "total": 508},
            "known_private": {"total": 163, "rare": 13, "common": 150},
            "production_eligible_rare": 11,
            "domias": False,
        },
        "selection_gate": {
            "checks": requirements,
            "passed": gate_passed,
            "decision": "candidate_supported" if gate_passed else "retain_v4",
        },
        "discovery": discovery.to_dict(),
        "validation": validation.to_dict(),
        "production": production.to_dict(),
        "official_structural_bound": structural,
        "limitations": [
            "The local B_practice/B_self rows are a toy world, not the production B used to generate the existing rank matrices.",
            "Accordingly production row features were not recomputed from local B; only existing production ranks were transformed.",
            "The official aggregate-all lower bound does not by itself guarantee a lift in mean target-wise max(all, rare).",
            "Discovery and validation seeds are disjoint, and validation does not select a different recipe.",
        ],
        "qa": {
            "aggregate_rows": len(scores),
            "target_rows": len(target_scores),
            "aggregate_key_unique": not scores.duplicated(["split", "seed", "scenario", "variant"]).any(),
            "target_key_unique": not target_scores.duplicated(["split", "seed", "scenario", "variant", "target"]).any(),
            "world_count": len(world_rows),
            "all_world_checks_pass": all(bool(row.get("passed", True)) for row in world_rows),
        },
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
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(report["selection_gate"], indent=2))
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
