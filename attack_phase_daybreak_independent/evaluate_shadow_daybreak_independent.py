"""Local shadow comparison for the independent Daybreak Blue attack.

This evaluator is not part of the production attack dependency graph.  It
constructs deterministic local worlds with known truth and compares the new
algorithm with the validated v3 reference through the organizer scoring code.
It performs no network, authentication, token, packaging, or submission work.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for path in (REPO_ROOT, STARTER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v3 import generate_submission_v3 as v3  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
    build_mia_independent,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import (  # noqa: E402
    score_aia_power,
    score_mia_power,
)
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


SCENARIOS = ("raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daybreak独立方式のローカルshadow比較")
    parser.add_argument(
        "--dist",
        type=Path,
        default=REPO_ROOT.parent / "pwscup2026-kit" / "work" / "prelim" / "team24" / "runtime_dist",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "local-output" / "daybreak-v4-research" / "comparison",
    )
    parser.add_argument("--seeds", default="20260903")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--skip-v3-domias", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.scenarios = [value.strip() for value in args.scenarios.split(",") if value.strip()]
    if not args.seeds:
        parser.error("--seeds is empty")
    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    return args


def known_negative_subset(world: shadow.ShadowWorld, count: int = 163) -> np.ndarray:
    candidate_ids = world.attacker_pool["pool_id"].to_numpy(dtype=int)
    member_ids = set(
        int(value)
        for frame in world.target_pools.values()
        for value in frame["pool_id"].to_numpy(dtype=int)
    )
    private = np.asarray([value not in member_ids for value in candidate_ids], dtype=bool)
    indices = np.flatnonzero(private)
    if len(indices) < count:
        raise RuntimeError("shadow private set is smaller than calibration subset")
    record_ids = world.attacker_pool["record_id"].to_numpy(dtype=np.uint64)[indices]
    mixed = record_ids * np.uint64(11400714819323198485) + np.uint64(14029467366897019727)
    selected = indices[np.argsort(mixed, kind="mergesort")[:count]]
    mask = np.zeros(len(candidate_ids), dtype=bool)
    mask[selected] = True
    return mask


def score_component_variants(
    variants: dict[str, pd.DataFrame],
    component: str,
    ref_dir: Path,
    cfg: dict[str, Any],
    seed: int,
    scenario: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, frame in variants.items():
        result = (
            score_mia_power(frame, ref_dir, cfg)
            if component == "mia"
            else score_aia_power(frame, ref_dir, cfg)
        )
        row = {
            "seed": seed,
            "scenario": scenario,
            "component": component,
            "variant": name,
            "power": float(result["power"]),
            "all": float(result["all"]),
            "rare": float(result["rare"]),
        }
        if component == "aia":
            row["p_m"] = float(result["p_m"])
            row["p_c"] = float(result["p_c"])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dist = args.dist.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    spec = shadow.ShadowSpec(
        n_targets=20,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=508,
        minimum_shared=25,
        maximum_shared=26,
    )
    source_quality = shadow.validate_source_data(candidates, b_self, background, spec)
    rare_mask = b_self["is_rare"].astype(bool).to_numpy()
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(args.seeds):
        print(f"seed={seed}: construct local shadow world", flush=True)
        world = shadow.build_shadow_world(candidates, b_self, background, detector, spec, seed)
        shadow.validate_world(world, spec)
        known_negative = known_negative_subset(world)
        ref_dir = out_dir / "shadow_reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)
        abstain_time = v2.safe_abstain_time(horizon, tau)

        for scenario_index, scenario in enumerate(args.scenarios):
            print(f"seed={seed} scenario={scenario}", flush=True)
            mia_targets: dict[str, pd.DataFrame] = {}
            aia_targets: dict[str, pd.DataFrame] = {}
            for target_id in world.mia_sources:
                mia_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, seed_index, scenario_index, int(target_id), 1])
                )
                aia_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, seed_index, scenario_index, int(target_id), 2])
                )
                mia_targets[target_id] = shadow.scenario_release(
                    world.mia_sources[target_id], world.attack_background, scenario, cfg, mia_rng
                )
                aia_targets[target_id] = shadow.scenario_release(
                    world.aia_sources[target_id], world.attack_background, scenario, cfg, aia_rng
                )

            _, new_mia_variants, _ = build_mia_independent(
                candidates, world.attack_background, mia_targets, known_negative
            )
            _, new_aia_variants, _ = build_aia_independent(
                world.attack_background,
                aia_targets,
                world.challenges,
                horizon=horizon,
                abstain_time=abstain_time,
            )
            v3_mia, _, _ = v3.build_mia_v3(
                candidates,
                world.attack_background,
                mia_targets,
                known_negative,
                minimum=spec.minimum_shared,
                maximum=spec.maximum_shared,
                total=spec.shared_total,
                transport_epsilon=0.35,
                include_domias=not args.skip_v3_domias,
                density_weight=v2.DEFAULT_MIA_DENSITY_WEIGHT,
                rare_mask=rare_mask,
                rare_weight=v2.DEFAULT_MIA_RARE_WEIGHT,
            )
            v3_aia, _, _ = v3.build_aia_v3(
                world.attack_background,
                aia_targets,
                world.challenges,
                main_fraction=0.5,
                variant_fractions=[0.5],
                abstain_time=abstain_time,
                horizon_time=horizon,
                link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
                baseline_abstain_time=shadow.choose_low_hit_time(
                    world.attack_background, horizon=horizon, tau=tau
                ),
                anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
                anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
            )
            new_mia_variants["v3_reference"] = v3_mia
            new_aia_variants["v3_reference"] = v3_aia
            shadow.validate_prediction_frames(
                new_mia_variants, new_aia_variants, candidates, spec
            )
            rows.extend(
                score_component_variants(
                    new_mia_variants, "mia", ref_dir, cfg, seed, scenario
                )
            )
            rows.extend(
                score_component_variants(
                    new_aia_variants, "aia", ref_dir, cfg, seed, scenario
                )
            )

    results = pd.DataFrame(rows).sort_values(
        ["component", "scenario", "variant", "seed"], kind="mergesort"
    )
    results.to_csv(out_dir / "component_scores.csv", index=False)
    summary = (
        results.groupby(["component", "variant", "scenario"], as_index=False)["power"]
        .agg(["mean", "min", "max"])
        .reset_index(drop=True)
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    stress = results[results["scenario"].isin(["noise_0p03", "noise_0p10", "time_shuffled"])]
    ranking = (
        stress.groupby(["component", "variant"], as_index=False)["power"]
        .agg(["mean", "min", "max"])
        .sort_values(["component", "mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    ranking.to_csv(out_dir / "stress_ranking.csv", index=False)
    report = {
        "model": "gpt-daybreak-blue-latest",
        "comparison": "independent attack vs v3 reference",
        "official_score_available": False,
        "source_quality": source_quality,
        "parameters": {"seeds": args.seeds, "scenarios": args.scenarios},
        "stress_ranking": ranking.to_dict(orient="records"),
        "elapsed_seconds": time.perf_counter() - started,
        "network_access": False,
        "token_read": False,
        "submission_performed": False,
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
