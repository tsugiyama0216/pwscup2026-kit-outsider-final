"""Truth-known local evaluation of the version-5 structured AIA ensemble."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import build_aia_v5_variants  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import score_aia_power  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


SCENARIOS = ("raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v5 AIAのローカルshadow比較")
    parser.add_argument("--dist", type=Path, default=REPO_ROOT / "participant_data")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-shadow-evaluation-v5",
    )
    parser.add_argument("--seeds", default="20260901,20260902")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument(
        "--soft-matching-world",
        action="store_true",
        help="既存shadowのsoft control matchingをそのまま使う",
    )
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.scenarios = [
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ]
    if not args.seeds:
        parser.error("--seeds is empty")
    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    return args


def exactify_visible_control_matching(world: shadow.ShadowWorld) -> None:
    """Make each shadow control share the official visible block of one member.

    The stock shadow evaluator has a deliberately small hidden control pool and
    therefore often falls back across a visible matching block.  Production
    challenges almost never do so.  This evaluation-only adjustment preserves
    every control's laboratory values and hidden outcome while repairing sex,
    prefecture, and five-year age band before an attack sees the challenge.
    """
    for target_id, source in world.challenges.items():
        challenge = source.reset_index(drop=True).copy()
        truth = world.aia_truth[target_id].reset_index(drop=True)
        if not challenge["challenge_row_id"].equals(truth["challenge_row_id"]):
            raise RuntimeError("shadow challenge and truth rows are not aligned")
        member_rows = np.flatnonzero(truth["label"].eq("member").to_numpy())
        control_rows = np.flatnonzero(truth["label"].eq("control").to_numpy())
        if len(member_rows) != len(control_rows):
            raise RuntimeError("exact matching requires equal member/control counts")

        member_age = challenge.iloc[member_rows]["age"].to_numpy(dtype=float)[:, None]
        control_age = challenge.iloc[control_rows]["age"].to_numpy(dtype=float)[None, :]
        cost = 0.01 * np.abs(member_age - control_age)
        member_sex = challenge.iloc[member_rows]["sex"].to_numpy()[:, None]
        control_sex = challenge.iloc[control_rows]["sex"].to_numpy()[None, :]
        member_pref = (
            challenge.iloc[member_rows]["prefecture"].astype(str).to_numpy()[:, None]
        )
        control_pref = (
            challenge.iloc[control_rows]["prefecture"].astype(str).to_numpy()[None, :]
        )
        member_band = np.floor((member_age - 40.0) / 5.0)
        control_band = np.floor((control_age - 40.0) / 5.0)
        cost += 100.0 * (member_sex != control_sex)
        cost += 20.0 * (member_pref != control_pref)
        cost += 5.0 * (member_band != control_band)
        local_member, local_control = linear_sum_assignment(cost)
        for member_index, control_index in zip(local_member, local_control, strict=True):
            member_row = int(member_rows[member_index])
            control_row = int(control_rows[control_index])
            challenge.loc[control_row, "sex"] = challenge.loc[member_row, "sex"]
            challenge.loc[control_row, "prefecture"] = challenge.loc[
                member_row, "prefecture"
            ]
            band = int(np.floor((float(challenge.loc[member_row, "age"]) - 40.0) / 5.0))
            lower = 40 + 5 * band
            upper = lower + 4
            challenge.loc[control_row, "age"] = int(
                np.clip(round(float(challenge.loc[control_row, "age"])), lower, upper)
            )
        world.challenges[target_id] = challenge


def score_variants(
    variants: dict[str, pd.DataFrame],
    ref_dir: Path,
    cfg: dict[str, Any],
    *,
    seed: int,
    scenario: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, frame in variants.items():
        result = score_aia_power(frame, ref_dir, cfg)
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "variant": name,
                "power": float(result["power"]),
                "all": float(result["all"]),
                "rare": float(result["rare"]),
                "p_m": float(result["p_m"]),
                "p_c": float(result["p_c"]),
            }
        )
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
    shadow.validate_source_data(candidates, b_self, background, spec)
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    abstain = v2.safe_abstain_time(horizon, tau)
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    for seed_index, seed in enumerate(args.seeds):
        print(f"seed={seed}: construct shadow world", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        if not args.soft_matching_world:
            exactify_visible_control_matching(world)
        shadow.validate_world(world, spec)
        ref_dir = out_dir / "shadow_reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)

        for scenario_index, scenario in enumerate(args.scenarios):
            print(f"seed={seed} scenario={scenario}", flush=True)
            targets: dict[str, pd.DataFrame] = {}
            for target_id, source in world.aia_sources.items():
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, seed_index, scenario_index, int(target_id), 5]
                    )
                )
                targets[target_id] = shadow.scenario_release(
                    source, world.attack_background, scenario, cfg, rng
                )

            v4_aia, _, _ = v4.build_aia_v4(
                world.attack_background,
                targets,
                world.challenges,
                main_fraction=0.5,
                variant_fractions=[0.5],
                abstain_time=abstain,
                horizon_time=horizon,
                link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
                baseline_abstain_time=shadow.choose_low_hit_time(
                    world.attack_background, horizon=horizon, tau=tau
                ),
                anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
                anomalous_link_trust_distance=(
                    v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE
                ),
                integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
            )
            independent_aia, _, _ = build_aia_independent(
                world.attack_background,
                targets,
                world.challenges,
                member_count=spec.aia_members,
                horizon=horizon,
                abstain_time=abstain,
            )
            structured, detail = build_aia_v5_variants(
                v4_aia,
                independent_aia,
                targets,
                world.challenges,
                member_count=spec.aia_members,
                abstain_time=abstain,
                horizon_time=horizon,
            )
            variants = {
                "v4_reference": v4_aia,
                "independent_reference": independent_aia,
                **structured,
            }
            rows.extend(
                score_variants(
                    variants, ref_dir, cfg, seed=seed, scenario=scenario
                )
            )
            diagnostics[f"{seed}:{scenario}"] = detail

    results = pd.DataFrame(rows).sort_values(
        ["scenario", "variant", "seed"], kind="mergesort"
    )
    results.to_csv(out_dir / "component_scores.csv", index=False)
    summary = (
        results.groupby(["variant", "scenario"], as_index=False)[
            ["power", "all", "rare", "p_m", "p_c"]
        ]
        .mean()
        .sort_values(["scenario", "power"], ascending=[True, False])
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    positive = results[results["scenario"] != "independent"]
    ranking = (
        positive.groupby("variant", as_index=False)[
            ["power", "all", "rare", "p_m", "p_c"]
        ]
        .mean()
        .sort_values("power", ascending=False)
    )
    ranking.to_csv(out_dir / "positive_ranking.csv", index=False)
    report = {
        "purpose": "truth-known method comparison, not leaderboard estimation",
        "parameters": {
            "seeds": args.seeds,
            "scenarios": args.scenarios,
            "exact_visible_control_matching": not args.soft_matching_world,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "ranking": ranking.to_dict(orient="records"),
        "diagnostics": diagnostics,
        "network_access": False,
        "submission_performed": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(ranking.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
