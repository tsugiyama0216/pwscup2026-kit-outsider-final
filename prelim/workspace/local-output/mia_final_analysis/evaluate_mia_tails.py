"""Research-only truth-known comparison of narrow MIA tail repairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "starter" / "src"
for value in (ROOT, STARTER):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_v5 import mia_tail_assignment as tail  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_mia  # noqa: E402


COUNT_CONFIGS = (
    ((13, 6, 1), (25, 27)),
    ((13, 6, 1), (26, 27)),
    ((13, 6, 1), (27, 25)),
    ((13, 6, 1), (27, 26)),
    ((14, 4, 2), (25, 27)),
    ((14, 4, 2), (26, 27)),
    ((14, 4, 2), (27, 25)),
    ((14, 4, 2), (27, 26)),
    ((14, 4, 2), (27, 27)),
    ((15, 2, 3), (25, 27)),
    ((15, 2, 3), (26, 27)),
    ((15, 2, 3), (27, 25)),
    ((15, 2, 3), (27, 26)),
    ((15, 2, 3), (27, 27)),
)


def count_vector(config_index: int, rng: np.random.Generator) -> np.ndarray:
    counts, highs = COUNT_CONFIGS[config_index % len(COUNT_CONFIGS)]
    remaining = {n: counts[i] - highs.count(n) for i, n in enumerate((25, 26, 27))}
    low = np.concatenate([np.full(remaining[n], n, dtype=int) for n in (25, 26, 27)])
    low = low[rng.permutation(len(low))]
    return np.concatenate([np.asarray(highs, dtype=int), low])


def release(source: pd.DataFrame, background: pd.DataFrame, scenario: str, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    if scenario in ("raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent"):
        return shadow.scenario_release(source, background, scenario, cfg, rng)
    if scenario.startswith("noise_"):
        fraction = float(scenario.split("_", 1)[1])
        out = source.copy()
        columns = list(shadow.CONTINUOUS_COLUMNS)
        scale = background[columns].std(ddof=0).to_numpy(float)
        noise = rng.normal(0.0, fraction * np.where(scale > 0, scale, 1.0), size=(len(out), len(columns)))
        for index, column in enumerate(columns):
            out[column] = out[column].to_numpy(float) + noise[:, index]
        horizon = float(cfg["onset"]["horizon_years"])
        out["time"] = out["time"].to_numpy(float) + rng.normal(0.0, fraction * horizon, size=len(out))
        return shadow.coerce_release(out, cfg).iloc[rng.permutation(len(out))].reset_index(drop=True)
    raise ValueError(scenario)


def score_frame(frame: pd.DataFrame, world: shadow.ShadowWorld, cfg: dict) -> dict:
    indexed = world.attacker_pool.set_index("record_id")
    ids = frame["record_id"].to_numpy(int)
    pool = indexed.loc[ids, "pool_id"].to_numpy(int)
    rare = indexed.loc[ids, "is_rare"].to_numpy(bool)
    rows = []
    for target in frame.columns[1:]:
        target_pool = set(world.target_pools[str(target)]["pool_id"].astype(int))
        truth = np.asarray([value in target_pool for value in pool], dtype=int)
        result = score_mia(frame[str(target)].to_numpy(float), truth, cfg, rare_mask=rare)
        rows.append({
            "target": str(target),
            "power": float(result.score),
            "all": float(result.breakdown["tpr_at_low_fpr"]),
            "rare": float(result.breakdown["tpr_rare"]),
            "auc": float(result.breakdown["auc"]),
        })
    detail = pd.DataFrame(rows)
    return {
        "power": float(detail.power.mean()),
        "all": float(detail["all"].mean()),
        "rare": float(detail.rare.dropna().mean()) if detail.rare.notna().any() else 0.0,
        "auc": float(detail.auc.mean()),
        "per_target": rows,
    }


def known_private_mask(world: shadow.ShadowWorld, b_self: pd.DataFrame, seed: int) -> np.ndarray | None:
    attacker = world.attacker_pool
    member_pool = set().union(*(set(frame["pool_id"].astype(int)) for frame in world.target_pools.values()))
    nonmember = ~attacker["pool_id"].isin(member_pool).to_numpy()
    rare = b_self["is_rare"].astype(bool).to_numpy()
    rare_pool = np.flatnonzero(nonmember & rare)
    common_pool = np.flatnonzero(nonmember & ~rare)
    if len(rare_pool) < 13 or len(common_pool) < 150:
        return None
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([rng.choice(rare_pool, 13, replace=False), rng.choice(common_pool, 150, replace=False)])
    result = np.zeros(len(attacker), dtype=bool)
    result[chosen] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20261001,20261002,20261003,20261004")
    parser.add_argument("--private-seeds", default="1,2")
    parser.add_argument("--scenarios", default="noise_0p03,noise_0p10,noise_0.20,time_shuffled")
    parser.add_argument("--widths", default="7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--include-domias", action="store_true")
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    private_seeds = [int(v) for v in args.private_seeds.split(",")]
    scenarios = args.scenarios.split(",")
    widths = [int(v) for v in args.widths.split(",")]

    candidates = pd.read_csv(ROOT / "participant_data" / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(ROOT / "participant_data" / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(ROOT / "participant_data")
    cfg = load_config(KIT_CONFIG_PATH)
    spec = shadow.ShadowSpec(n_targets=20, shared_total=508, minimum_shared=25, maximum_shared=27)
    original_allocator = shadow.allocate_shared_counts
    all_rows = []
    target_rows = []
    diagnostics = []
    started = time.perf_counter()
    try:
        for world_index, seed in enumerate(seeds):
            allocator_rng = np.random.default_rng(seed + 991)
            counts = count_vector(world_index, allocator_rng)
            shadow.allocate_shared_counts = lambda spec, rng, values=counts: values.copy()
            print("world", seed, "counts", dict(zip((25, 26, 27), np.bincount(counts, minlength=28)[25:28])), "high", counts[:2].tolist(), flush=True)
            world = shadow.build_shadow_world(candidates, b_self, background, detector, spec, seed)
            shadow.validate_world(world, spec)
            for private_seed in private_seeds:
                private = known_private_mask(world, b_self, seed * 1000 + private_seed)
                if private is None:
                    print("skip: insufficient private rare", seed, private_seed, flush=True)
                    continue
                for scenario_index, scenario in enumerate(scenarios):
                    print("run", seed, private_seed, scenario, flush=True)
                    targets = {}
                    for target, source in world.mia_sources.items():
                        rng = np.random.default_rng(np.random.SeedSequence([seed, private_seed, scenario_index, int(target), 919]))
                        targets[target] = release(source, world.attack_background, scenario, cfg, rng)
                    main, variants, detail = v4.build_mia_v4(
                        candidates,
                        world.attack_background,
                        targets,
                        private,
                        minimum=25,
                        maximum=27,
                        total=508,
                        transport_epsilon=0.35,
                        include_domias=args.include_domias,
                        density_weight=0.35,
                        rare_mask=b_self["is_rare"].astype(bool).to_numpy(),
                        rare_weight=0.65,
                    )
                    v3_frame = variants["v3_private_demotion_transport_reference"]
                    frames = {"v4": main}
                    repair_detail = {}
                    old_min, old_max = tail.MIN_TOP_L, tail.MAX_TOP_L
                    old_recover = tail.recover_eligible_rare_ids
                    eligible_rare_ids = tuple(
                        sorted(
                            candidates.loc[
                                b_self["is_rare"].astype(bool).to_numpy() & ~private,
                                "record_id",
                            ].astype(int).tolist()
                        )
                    )
                    tail.MIN_TOP_L, tail.MAX_TOP_L = min(widths), max(widths)
                    tail.recover_eligible_rare_ids = lambda v3, v4, private_ids, expected_count=11, values=eligible_rare_ids: values
                    try:
                        for width in widths:
                            repaired, d = tail.assign_unique_top_tail(v3_frame, main, top_l=width)
                            frames[f"tail{width}"] = repaired
                            repair_detail[str(width)] = {
                                "duplicates": d["original_duplicate_slots"],
                                "changed": d["changed_tail_slots"],
                                "changed_targets": [t for t, x in d["per_target"].items() if x["removed_record_ids"]],
                            }
                    finally:
                        tail.MIN_TOP_L, tail.MAX_TOP_L = old_min, old_max
                        tail.recover_eligible_rare_ids = old_recover
                    scores = {name: score_frame(frame, world, cfg) for name, frame in frames.items()}
                    base = scores["v4"]
                    run_key = f"{seed}:{private_seed}:{scenario}"
                    for name, result in scores.items():
                        all_rows.append({
                            "world_seed": seed,
                            "private_seed": private_seed,
                            "scenario": scenario,
                            "variant": name,
                            "power": result["power"],
                            "all": result["all"],
                            "rare": result["rare"],
                            "auc": result["auc"],
                            "delta_power": result["power"] - base["power"],
                            "delta_all": result["all"] - base["all"],
                            "delta_rare": result["rare"] - base["rare"],
                            "delta_auc": result["auc"] - base["auc"],
                            "duplicates": 0 if name == "v4" else repair_detail[name.removeprefix("tail")]["duplicates"],
                            "changed_slots": 0 if name == "v4" else repair_detail[name.removeprefix("tail")]["changed"],
                        })
                        base_targets = {row["target"]: row for row in base["per_target"]}
                        for row in result["per_target"]:
                            target_rows.append({
                                "world_seed": seed,
                                "private_seed": private_seed,
                                "scenario": scenario,
                                "variant": name,
                                **row,
                                "delta_power": row["power"] - base_targets[row["target"]]["power"],
                                "delta_all": row["all"] - base_targets[row["target"]]["all"],
                                "delta_auc": row["auc"] - base_targets[row["target"]]["auc"],
                            })
                    diagnostics.append({"run": run_key, "counts": counts.tolist(), "repair": repair_detail})
    finally:
        shadow.allocate_shared_counts = original_allocator

    output_dir = Path(__file__).resolve().parent / "shadow_tail_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(all_rows)
    targets_out = pd.DataFrame(target_rows)
    results.to_csv(output_dir / "runs.csv", index=False)
    targets_out.to_csv(output_dir / "targets.csv", index=False)
    summary = results.groupby("variant", as_index=False).agg(
        runs=("delta_power", "size"),
        mean_delta_power=("delta_power", "mean"),
        min_delta_power=("delta_power", "min"),
        max_delta_power=("delta_power", "max"),
        losses=("delta_power", lambda x: int((x < -1e-12).sum())),
        ties=("delta_power", lambda x: int((abs(x) <= 1e-12).sum())),
        wins=("delta_power", lambda x: int((x > 1e-12).sum())),
        mean_delta_all=("delta_all", "mean"),
        min_delta_all=("delta_all", "min"),
        mean_delta_rare=("delta_rare", "mean"),
        min_delta_rare=("delta_rare", "min"),
        mean_delta_auc=("delta_auc", "mean"),
        mean_changed_slots=("changed_slots", "mean"),
    ).sort_values("mean_delta_power", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "diagnostics.json").write_text(json.dumps({"elapsed": time.perf_counter() - started, "runs": diagnostics}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(output_dir)


if __name__ == "__main__":
    main()
