"""Truth-known evaluation of the exact team-24 home-prefecture MIA prior.

The preliminary construction has 313 target-private rows and 736 pair-shared
rows.  The self AIA challenge exposes 163 of the private rows exactly, 13 of
which are rare.  No inspected primary source proves that those are all rare
rows in the complete private block.  All private rows have the team's home
prefecture.  In the actual B this leaves:

* 231 eligible home-prefecture rows = 150 private + 81 shared;
* 655 non-home-prefecture rows = all shared;
* 11 eligible rare rows: 10 non-home rows are shared under the documented
  home-private construction; the one eligible home rare row has unknown
  private/shared status.

This benchmark samples only the genuinely unknown 150 private identities,
partitions the resulting exact 736 shared rows into disjoint target blocks,
and compares v4 to bounded common-row prior reranks.  For this sensitivity test
only, latent private sampling excludes the one unresolved eligible home rare
row; this is an explicit model assumption, not an official fact.  The
allocation of 508 rows to the 20 scored targets is a proportional modelling
prior, not known official membership truth.  Discovery and validation seeds
are disjoint.  It reads no token and writes no production submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
for source in (ROOT, ROOT / "starter" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v3 import generate_submission_v3 as v3  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    load_public_targets,
    validate_mia,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit import validate as official_validate  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_mia  # noqa: E402


ACTUAL = Path(
    r"C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
)
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
TEAM = ROOT / "submissions" / "prelim" / "team24"
OUT = ROOT / "local-output" / "attack-v8" / "actual-mia" / "home-prior"
EXPECTED_HASHES = {
    "B.csv": "7A7AA39E16FA5C76EE9C8DC009928373AEC6CF47E7B1BD10DC4E6899B5D2AB3A",
    "B_self.csv": "297354A3C909FBA6BF4D33053CF91EFA9139892BF5269050A5607B86E0F33E35",
    "A_bg.csv": "F4817FBA6C1434D6B74221C2B5ABE90E4A53E80D8FA17E12602D43ADAE0A3126",
}
DISCOVERY_SEEDS = (20268011, 20268012, 20268013)
VALIDATION_SEEDS = (20268101, 20268102, 20268103, 20268104, 20268105)
DEFAULT_SCENARIOS = ("noise_0p03", "noise_0p10", "noise_0p20", "noise_0p40")
TARGET_COUNTS = np.asarray([25] * 13 + [26] * 6 + [27], dtype=int)
THEORETICAL_BETA = math.log((508.0 / 736.0) / ((81.0 / 231.0) * (508.0 / 736.0)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(value) for value in raw.split(",") if value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-seeds", default=",".join(map(str, DISCOVERY_SEEDS))
    )
    parser.add_argument(
        "--validation-seeds", default=",".join(map(str, VALIDATION_SEEDS))
    )
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument(
        "--include-domias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    args.discovery_seeds = parse_values(args.discovery_seeds, int)
    args.validation_seeds = parse_values(args.validation_seeds, int)
    args.scenarios = parse_values(args.scenarios, str)
    if set(args.discovery_seeds) & set(args.validation_seeds):
        parser.error("discovery/validation seeds overlap")
    allowed = {
        "raw",
        "noise_0p03",
        "noise_0p10",
        "noise_0p20",
        "noise_0p40",
        "time_shuffled",
        "independent",
    }
    if not args.discovery_seeds or not args.validation_seeds:
        parser.error("both seed splits are required")
    if not args.scenarios or set(args.scenarios) - allowed:
        parser.error("scenario set is empty or unknown")
    return args


def load_actual() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    observed = {name: sha256(ACTUAL / name) for name in EXPECTED_HASHES}
    if observed != EXPECTED_HASHES:
        raise RuntimeError(f"actual data hash mismatch: {observed}")
    candidates = pd.read_csv(ACTUAL / "B.csv").reset_index(drop=True)
    b_self = pd.read_csv(ACTUAL / "B_self.csv").set_index("record_id").loc[
        candidates["record_id"]
    ].reset_index()
    if not candidates.equals(b_self[candidates.columns]):
        raise RuntimeError("B/B_self alignment failed")
    background = pd.read_csv(ACTUAL / "A_bg.csv").reset_index(drop=True)
    self_challenge = pd.read_csv(TARGETS / "aia_challenge_24.csv")
    known_private, _ = v3.discover_known_private_records(candidates, self_challenge)
    rare = b_self["is_rare"].astype(bool).to_numpy()
    private_prefectures = candidates.loc[known_private, "prefecture"].astype(str).unique()
    if len(private_prefectures) != 1:
        raise RuntimeError("known private rows do not identify one home prefecture")
    home_prefecture = str(private_prefectures[0])
    home = candidates["prefecture"].astype(str).eq(home_prefecture).to_numpy()
    if (
        int(known_private.sum()) != 163
        or int(rare.sum()) != 24
        or int((rare & known_private).sum()) != 13
        or int(home.sum()) != 394
        or int((home & ~known_private).sum()) != 231
        or int((~home).sum()) != 655
    ):
        raise RuntimeError("actual structural cardinality changed")
    # Observed only: 13 matched-private rare rows and 11 eligible rare rows.
    # The one eligible home rare row is not proven shared by any inspected
    # primary source. The latent-world sampler below excludes it as an explicit
    # sensitivity-model assumption so the historical rare order stays fixed.
    if int((rare & ~known_private).sum()) != 11:
        raise RuntimeError("eligible rare count changed")
    if (
        int((rare & ~known_private & home).sum()) != 1
        or int((rare & ~known_private & ~home).sum()) != 10
    ):
        raise RuntimeError("eligible rare home/non-home split changed")
    return candidates, b_self, background, known_private, rare, home, home_prefecture


def target_count_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 72_119)
    return TARGET_COUNTS[rng.permutation(len(TARGET_COUNTS))]


def latent_world(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    known_private: np.ndarray,
    rare: np.ndarray,
    home: np.ndarray,
    *,
    seed: int,
    uniform_private_control: bool,
) -> dict[str, Any]:
    """Construct disjoint MIA truth under the stated sensitivity assumptions."""
    rng = np.random.default_rng(seed)
    if uniform_private_control:
        pool = np.flatnonzero(~known_private & ~rare)
    else:
        pool = np.flatnonzero(home & ~known_private & ~rare)
    latent_private = rng.choice(pool, size=150, replace=False)
    private = known_private.copy()
    private[latent_private] = True
    shared = np.flatnonzero(~private)
    # Excluding eligible rare rows is a model choice. Only the 10 eligible
    # non-home rare rows are structurally forced shared; the eligible home rare
    # row could be among the latent private identities in the real cohort.
    if len(shared) != 736 or bool(np.any(rare & private & ~known_private)):
        raise RuntimeError("latent private/shared construction failed")
    attack_background, hidden_background = shadow.split_public_and_hidden_background(
        background
    )
    counts = target_count_vector(seed)
    order = shared[rng.permutation(len(shared))]
    targets = [str(value) for value in range(1, 21)]
    memberships: dict[str, np.ndarray] = {}
    sources: dict[str, pd.DataFrame] = {}
    cursor = 0
    for target_id, count in zip(targets, counts, strict=True):
        members = order[cursor : cursor + int(count)]
        cursor += int(count)
        truth = np.zeros(len(candidates), dtype=bool)
        truth[members] = True
        memberships[target_id] = truth
        filler_count = 1049 - int(count)
        filler = rng.choice(len(hidden_background), size=filler_count, replace=False)
        source = pd.concat(
            [candidates.iloc[members], hidden_background.iloc[filler]],
            ignore_index=True,
        )
        sources[target_id] = source.iloc[rng.permutation(len(source))].reset_index(
            drop=True
        )
    if cursor != 508 or len(np.unique(order[:cursor])) != 508:
        raise RuntimeError("scored shared assignment failed")
    return {
        "attack_background": attack_background,
        "hidden_background": hidden_background,
        "sources": sources,
        "truth": memberships,
        "counts": counts,
        "private": private,
        "latent_private": latent_private,
        "unscored_shared": order[cursor:],
        "uniform_private_control": uniform_private_control,
    }


def release_scenario(
    source: pd.DataFrame,
    background: pd.DataFrame,
    scenario: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if scenario in {"raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent"}:
        return shadow.scenario_release(source, background, scenario, cfg, rng)
    fraction = {"noise_0p20": 0.20, "noise_0p40": 0.40}[scenario]
    output = source.copy()
    columns = list(shadow.CONTINUOUS_COLUMNS)
    scale = background[columns].std(ddof=0).to_numpy(dtype=float)
    noise = rng.normal(
        0.0,
        fraction * np.where(scale > 0.0, scale, 1.0),
        size=(len(output), len(columns)),
    )
    for index, column in enumerate(columns):
        output[column] = output[column].to_numpy(dtype=float) + noise[:, index]
    horizon = float(cfg["onset"]["horizon_years"])
    output["time"] = output["time"].to_numpy(dtype=float) + rng.normal(
        0.0, fraction * horizon, size=len(output)
    )
    output = shadow.coerce_release(output, cfg)
    return output.iloc[rng.permutation(len(output))].reset_index(drop=True)


def logit_rank(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def rerank_all(
    baseline: pd.DataFrame,
    home: np.ndarray,
    locked: np.ndarray,
    *,
    beta: float,
    permuted_home: np.ndarray | None = None,
) -> pd.DataFrame:
    output = baseline.copy()
    ids = baseline["record_id"].to_numpy(dtype=int)
    active_home = np.asarray(home if permuted_home is None else permuted_home, dtype=bool)
    rows = np.flatnonzero(~locked)
    for target in baseline.columns[1:]:
        values = baseline[target].to_numpy(dtype=float)
        priority = logit_rank(values) + beta * (~active_home)
        destination = rows[np.lexsort((ids[rows], priority[rows]))]
        updated = values.copy()
        updated[destination] = np.sort(values[rows])
        output[target] = updated
    return output


def tail_rerank(
    baseline: pd.DataFrame,
    home: np.ndarray,
    locked: np.ndarray,
    *,
    beta: float,
    widths: dict[str, int],
    replace: bool,
) -> pd.DataFrame:
    output = baseline.copy()
    ids = baseline["record_id"].to_numpy(dtype=int)
    all_unlocked = np.flatnonzero(~locked)
    for target in baseline.columns[1:]:
        values = baseline[target].to_numpy(dtype=float)
        width = int(widths[target])
        top = np.argsort(-values, kind="mergesort")[:width]
        top_unlocked = top[~locked[top]]
        if len(top_unlocked) < 2:
            continue
        priority = logit_rank(values) + beta * (~home)
        if not replace:
            destination = top_unlocked[
                np.lexsort((ids[top_unlocked], priority[top_unlocked]))
            ]
            updated = values.copy()
            updated[destination] = np.sort(values[top_unlocked])
            output[target] = updated
            continue
        count = len(top_unlocked)
        chosen = all_unlocked[
            np.lexsort(
                (ids[all_unlocked], values[all_unlocked], -priority[all_unlocked])
            )
        ][:count]
        chosen_set = set(chosen.tolist())
        remainder = np.asarray(
            [row for row in all_unlocked if int(row) not in chosen_set], dtype=int
        )
        low = np.sort(values[all_unlocked])[:-count]
        high = np.sort(values[all_unlocked])[-count:]
        remainder_order = remainder[np.lexsort((ids[remainder], values[remainder]))]
        chosen_order = chosen[np.lexsort((ids[chosen], priority[chosen]))]
        updated = values.copy()
        updated[remainder_order] = low
        updated[chosen_order] = high
        output[target] = updated
    return output


def variants(
    baseline: pd.DataFrame,
    home: np.ndarray,
    rare: np.ndarray,
    known_private: np.ndarray,
    *,
    negative_seed: int,
) -> dict[str, pd.DataFrame]:
    locked = np.asarray(rare, dtype=bool) | np.asarray(known_private, dtype=bool)
    targets = list(baseline.columns[1:])
    widths_operating = {target: (37 if target in {"1", "2"} else 16) for target in targets}
    widths16 = {target: 16 for target in targets}
    widths37 = {target: 37 for target in targets}
    rng = np.random.default_rng(negative_seed)
    unlocked = np.flatnonzero(~locked)
    permuted = np.asarray(home, dtype=bool).copy()
    permuted[unlocked] = rng.permutation(permuted[unlocked])
    return {
        "v4": baseline.copy(),
        "all_beta0p50": rerank_all(baseline, home, locked, beta=0.50),
        "all_beta_theory": rerank_all(
            baseline, home, locked, beta=THEORETICAL_BETA
        ),
        "all_beta1p50": rerank_all(baseline, home, locked, beta=1.50),
        "tail16_theory_reorder": tail_rerank(
            baseline,
            home,
            locked,
            beta=THEORETICAL_BETA,
            widths=widths16,
            replace=False,
        ),
        "tail37_theory_reorder": tail_rerank(
            baseline,
            home,
            locked,
            beta=THEORETICAL_BETA,
            widths=widths37,
            replace=False,
        ),
        "operating_theory_reorder": tail_rerank(
            baseline,
            home,
            locked,
            beta=THEORETICAL_BETA,
            widths=widths_operating,
            replace=False,
        ),
        "operating_theory_replace": tail_rerank(
            baseline,
            home,
            locked,
            beta=THEORETICAL_BETA,
            widths=widths_operating,
            replace=True,
        ),
        "permuted_prior_negative": rerank_all(
            baseline,
            home,
            locked,
            beta=THEORETICAL_BETA,
            permuted_home=permuted,
        ),
    }


def score_frame(
    frame: pd.DataFrame,
    truth: dict[str, np.ndarray],
    rare: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for target in frame.columns[1:]:
        result = score_mia(
            frame[target].to_numpy(dtype=float),
            truth[target].astype(int),
            cfg,
            rare_mask=rare,
        )
        details.append(
            {
                "target": target,
                "power": float(result.score),
                "all": float(result.breakdown["tpr_at_low_fpr"]),
                "rare": float(result.breakdown["tpr_rare"]),
                "auc": float(result.breakdown["auc"]),
            }
        )
    table = pd.DataFrame(details)
    return {
        "power": float(table["power"].mean()),
        "all": float(table["all"].mean()),
        "rare": float(table["rare"].mean()),
        "auc": float(table["auc"].mean()),
    }, details


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    base = scores.loc[
        scores["variant"].eq("v4"),
        ["world_type", "split", "seed", "scenario", "power", "all", "rare", "auc"],
    ].rename(columns={metric: f"base_{metric}" for metric in ("power", "all", "rare", "auc")})
    joined = scores.merge(
        base, on=["world_type", "split", "seed", "scenario"], how="left"
    )
    for metric in ("power", "all", "rare", "auc"):
        joined[f"delta_{metric}"] = joined[metric] - joined[f"base_{metric}"]
    return (
        joined.groupby(["world_type", "split", "variant"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean_delta_power=("delta_power", "mean"),
            median_delta_power=("delta_power", "median"),
            p10_delta_power=("delta_power", lambda value: float(np.quantile(value, 0.10))),
            minimum_delta_power=("delta_power", "min"),
            win_rate=("delta_power", lambda value: float(np.mean(value > 1e-12))),
            mean_delta_all=("delta_all", "mean"),
            median_delta_all=("delta_all", "median"),
            p10_delta_all=(
                "delta_all", lambda value: float(np.quantile(value, 0.10))
            ),
            minimum_delta_all=("delta_all", "min"),
            all_win_rate=(
                "delta_all", lambda value: float(np.mean(value > 1e-12))
            ),
            mean_delta_rare=("delta_rare", "mean"),
            minimum_delta_rare=("delta_rare", "min"),
            mean_delta_auc=("delta_auc", "mean"),
        )
        .sort_values(
            ["world_type", "split", "mean_delta_power"],
            ascending=[True, True, False],
            kind="mergesort",
        )
    )


def production_variants(
    candidates: pd.DataFrame,
    home: np.ndarray,
    rare: np.ndarray,
    known_private: np.ndarray,
) -> pd.DataFrame:
    baseline = pd.read_csv(TEAM / "attack-candidate-v4" / "F_mia.csv")
    baseline = baseline.set_index("record_id").loc[candidates["record_id"]].reset_index()
    generated = variants(
        baseline, home, rare, known_private, negative_seed=989_101
    )
    rows: list[dict[str, Any]] = []
    official_dist = official_validate.load_dist(TARGETS)
    output_dir = OUT / "production_variants"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in generated.items():
        validate_mia(frame, baseline)
        result = official_validate.validate_mia_submission(frame, {}, official_dist)
        if not result.ok:
            raise RuntimeError(f"official validator failed for {name}: {result.errors}")
        path = output_dir / f"F_mia_{name}.csv"
        frame.to_csv(path, index=False)
        changed = frame.iloc[:, 1:].to_numpy() != baseline.iloc[:, 1:].to_numpy()
        rows.append(
            {
                "variant": name,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "changed_cells": int(changed.sum()),
                "changed_rows": int(np.any(changed, axis=1).sum()),
                "changed_targets": int(np.any(changed, axis=0).sum()),
                "rare_order_locked": all(
                    np.array_equal(
                        np.flatnonzero(rare)[
                            np.argsort(-frame.loc[rare, target].to_numpy(), kind="mergesort")
                        ],
                        np.flatnonzero(rare)[
                            np.argsort(-baseline.loc[rare, target].to_numpy(), kind="mergesort")
                        ],
                    )
                    for target in baseline.columns[1:]
                ),
            }
        )
    return pd.DataFrame(rows)


def run_worlds(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    known_private: np.ndarray,
    rare: np.ndarray,
    home: np.ndarray,
    cfg: dict[str, Any],
    *,
    split: str,
    seeds: Iterable[int],
    scenarios: Iterable[str],
    include_domias: bool,
    uniform_private_control: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    world_type = "uniform_private_negative" if uniform_private_control else "official_home_private"
    for seed in seeds:
        world = latent_world(
            candidates,
            background,
            known_private,
            rare,
            home,
            seed=seed,
            uniform_private_control=uniform_private_control,
        )
        world_rows.append(
            {
                "world_type": world_type,
                "split": split,
                "seed": seed,
                "private": int(world["private"].sum()),
                "shared": int((~world["private"]).sum()),
                "scored_shared": int(sum(world["counts"])),
                "unscored_shared": int(len(world["unscored_shared"])),
                "latent_private_home": int(home[world["latent_private"]].sum()),
                "latent_private_rare": int(rare[world["latent_private"]].sum()),
            }
        )
        for scenario_index, scenario in enumerate(scenarios):
            print(
                f"{world_type} {split} seed={seed} scenario={scenario}",
                flush=True,
            )
            releases = {}
            for target, source in world["sources"].items():
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, scenario_index, int(target), 808_701]
                    )
                )
                releases[target] = release_scenario(
                    source, world["attack_background"], scenario, cfg, rng
                )
            baseline, _, _ = v4.build_mia_v4(
                candidates,
                world["attack_background"],
                releases,
                known_private,
                minimum=25,
                maximum=27,
                total=508,
                transport_epsilon=0.35,
                include_domias=include_domias,
                density_weight=v2.DEFAULT_MIA_DENSITY_WEIGHT,
                rare_mask=rare,
                rare_weight=v2.DEFAULT_MIA_RARE_WEIGHT,
            )
            generated = variants(
                baseline,
                home,
                rare,
                known_private,
                negative_seed=seed * 100 + scenario_index,
            )
            for name, frame in generated.items():
                aggregate, details = score_frame(frame, world["truth"], rare, cfg)
                aggregate_rows.append(
                    {
                        "world_type": world_type,
                        "split": split,
                        "seed": seed,
                        "scenario": scenario,
                        "variant": name,
                        **aggregate,
                    }
                )
                target_rows.extend(
                    {
                        "world_type": world_type,
                        "split": split,
                        "seed": seed,
                        "scenario": scenario,
                        "variant": name,
                        **detail,
                    }
                    for detail in details
                )
    return aggregate_rows, target_rows, world_rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, _b_self, background, known_private, rare, home, home_prefecture = load_actual()
    cfg = load_config(KIT_CONFIG_PATH)
    production = production_variants(candidates, home, rare, known_private)
    production.to_csv(OUT / "production_audit.csv", index=False)

    aggregates: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    worlds: list[dict[str, Any]] = []
    for split, seeds in (
        ("discovery", args.discovery_seeds),
        ("validation", args.validation_seeds),
    ):
        rows, detail, checks = run_worlds(
            candidates,
            background,
            known_private,
            rare,
            home,
            cfg,
            split=split,
            seeds=seeds,
            scenarios=args.scenarios,
            include_domias=args.include_domias,
            uniform_private_control=False,
        )
        aggregates.extend(rows)
        targets.extend(detail)
        worlds.extend(checks)

    # A structurally wrong world is a negative control only.  The discovery
    # winner is frozen before this result is inspected.
    negative_seeds = args.validation_seeds[: min(2, len(args.validation_seeds))]
    rows, detail, checks = run_worlds(
        candidates,
        background,
        known_private,
        rare,
        home,
        cfg,
        split="negative_control",
        seeds=negative_seeds,
        scenarios=("noise_0p10",),
        include_domias=args.include_domias,
        uniform_private_control=True,
    )
    aggregates.extend(rows)
    targets.extend(detail)
    worlds.extend(checks)

    score_table = pd.DataFrame(aggregates)
    target_table = pd.DataFrame(targets)
    world_table = pd.DataFrame(worlds)
    summary = summarize(score_table)
    score_table.to_csv(OUT / "scores_by_world.csv", index=False)
    target_table.to_csv(OUT / "scores_by_target.csv", index=False)
    world_table.to_csv(OUT / "world_checks.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)

    discovery = summary.loc[
        summary["world_type"].eq("official_home_private")
        & summary["split"].eq("discovery")
        & ~summary["variant"].isin(["v4", "permuted_prior_negative"])
    ].sort_values(
        ["mean_delta_power", "p10_delta_power", "mean_delta_auc"],
        ascending=False,
        kind="mergesort",
    )
    selected = str(discovery.iloc[0]["variant"])
    validation = summary.loc[
        summary["world_type"].eq("official_home_private")
        & summary["split"].eq("validation")
        & summary["variant"].eq(selected)
    ].iloc[0]
    negative = summary.loc[
        summary["world_type"].eq("uniform_private_negative")
        & summary["variant"].eq(selected)
    ].iloc[0]
    prod = production.loc[production["variant"].eq(selected)].iloc[0]
    checks = {
        "discovery_positive": float(discovery.iloc[0]["mean_delta_power"]) > 0.0,
        "validation_mean_positive": float(validation["mean_delta_power"]) > 0.0,
        "validation_median_positive": float(validation["median_delta_power"]) > 0.0,
        "validation_win_rate_gte_0p60": float(validation["win_rate"]) >= 0.60,
        "validation_p10_gte_minus_0p002": float(validation["p10_delta_power"]) >= -0.002,
        "validation_all_mean_positive": float(validation["mean_delta_all"]) > 0.0,
        "validation_all_p10_gte_minus_0p002": float(validation["p10_delta_all"])
        >= -0.002,
        "validation_rare_noninferior": float(validation["minimum_delta_rare"]) >= -1e-12,
        "production_rare_locked": bool(prod["rare_order_locked"]),
        "negative_control_not_positive": float(negative["mean_delta_power"]) <= 0.002,
    }
    report = {
        "decision": {
            "selected_on_discovery": selected,
            "gate_pass": all(checks.values()),
            "checks": checks,
        },
        "theoretical_prior": {
            "scored_given_shared": 508.0 / 736.0,
            "shared_given_eligible_home": 81.0 / 231.0,
            "scored_given_eligible_home": (81.0 / 231.0) * (508.0 / 736.0),
            "scored_given_nonhome": 508.0 / 736.0,
            "probability_ratio_nonhome_vs_home": 231.0 / 81.0,
            "log_probability_ratio_beta": THEORETICAL_BETA,
        },
        "actual_structure": {
            "home_prefecture_redacted": hashlib.sha256(
                home_prefecture.encode("utf-8")
            ).hexdigest()[:16],
            "known_private": int(known_private.sum()),
            "eligible_home": int((home & ~known_private).sum()),
            "nonhome": int((~home).sum()),
            "rare": int(rare.sum()),
            "eligible_rare_rows": int((rare & ~known_private).sum()),
            "eligible_nonhome_rare_structurally_shared": int(
                (rare & ~known_private & ~home).sum()
            ),
            "eligible_home_rare_status_unknown": int(
                (rare & ~known_private & home).sum()
            ),
            "latent_private_excludes_eligible_rare_model_assumption": True,
        },
        "parameters": {
            "discovery_seeds": list(args.discovery_seeds),
            "validation_seeds": list(args.validation_seeds),
            "scenarios": list(args.scenarios),
            "include_domias": args.include_domias,
        },
        "discovery_selected": discovery.iloc[0].to_dict(),
        "validation_selected": validation.to_dict(),
        "negative_control_selected": negative.to_dict(),
        "production_selected": prod.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "safety": {
            "token_read": False,
            "archive_created": False,
            "production_changed": False,
            "submission_performed": False,
        },
    }
    (OUT / "report.json").write_text(
        json.dumps(v2.clean_json_v2(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(discovery.to_string(index=False))
    print(json.dumps(report["decision"], indent=2))
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
