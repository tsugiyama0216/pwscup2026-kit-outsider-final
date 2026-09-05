"""Rule-faithful local regression comparison of attack v2 against v1.

The organiser-only preliminary truth is not available.  This script therefore
uses the deterministic shadow worlds from ``attack_phase.evaluate_shadow`` and
the official scoring functions.  It is a comparative regression test, not a
leaderboard estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
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
from attack_phase_v2.generate_submission_v2 import (  # noqa: E402
    DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
    DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
    DEFAULT_AIA_LINK_TRUST_DISTANCE,
    DEFAULT_MIA_DENSITY_WEIGHT,
    DEFAULT_MIA_RARE_WEIGHT,
    build_aia_v2,
    build_mia_v2,
    safe_abstain_time,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import (  # noqa: E402
    score_aia_power,
    score_mia_power,
)
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="既知の疑似真値でattack v1/v2を同条件比較します。"
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=REPO_ROOT / "participant_data",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-shadow-evaluation-v2",
    )
    parser.add_argument("--seeds", default="20260826,20260827")
    parser.add_argument(
        "--scenarios",
        default=",".join(shadow.DEFAULT_SCENARIOS),
    )
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument(
        "--mia-density-weight", type=float, default=DEFAULT_MIA_DENSITY_WEIGHT
    )
    parser.add_argument(
        "--mia-rare-weight", type=float, default=DEFAULT_MIA_RARE_WEIGHT
    )
    parser.add_argument(
        "--aia-link-trust-distance",
        type=float,
        default=DEFAULT_AIA_LINK_TRUST_DISTANCE,
    )
    parser.add_argument(
        "--aia-anomalous-event-rate",
        type=float,
        default=DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
    )
    parser.add_argument(
        "--aia-anomalous-link-trust-distance",
        type=float,
        default=DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
    )
    parser.add_argument("--skip-domias", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    args.scenarios = [
        value.strip() for value in args.scenarios.split(",") if value.strip()
    ]
    unknown = sorted(set(args.scenarios) - set(shadow.SCENARIO_DESCRIPTIONS))
    if unknown:
        parser.error(f"未知のscenarioです: {unknown}")
    if not args.seeds:
        parser.error("--seeds は少なくとも1つ必要です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    if not 0.0 <= args.mia_density_weight <= 1.0:
        parser.error("--mia-density-weight は0〜1です")
    if not 0.0 <= args.mia_rare_weight <= 1.0:
        parser.error("--mia-rare-weight は0〜1です")
    if args.aia_link_trust_distance < 0.0:
        parser.error("--aia-link-trust-distance は0以上です")
    if not 0.0 <= args.aia_anomalous_event_rate <= 1.0:
        parser.error("--aia-anomalous-event-rate は0〜1です")
    if args.aia_anomalous_link_trust_distance < 0.0:
        parser.error("--aia-anomalous-link-trust-distance は0以上です")
    return args


def evaluate_variants(
    mia_variants: dict[str, pd.DataFrame],
    aia_variants: dict[str, pd.DataFrame],
    ref_dir: Path,
    cfg: dict[str, Any],
    *,
    seed: int,
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score variants without relying on the removed MIA per-target API."""
    mia_results: dict[str, dict[str, Any]] = {}
    aia_results: dict[str, dict[str, Any]] = {}
    component_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for name, frame in mia_variants.items():
        result = score_mia_power(frame, ref_dir, cfg)
        mia_results[name] = result
        component_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "mia",
                "variant": name,
                "power": result["power"],
                "all": result["all"],
                "rare": result["rare"],
                "auc": result["auc"],
            }
        )

    for name, frame in aia_variants.items():
        result = score_aia_power(frame, ref_dir, cfg)
        aia_results[name] = result
        component_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "aia",
                "variant": name,
                "power": result["power"],
                "all": result["all"],
                "rare": result["rare"],
                "p_m": result["p_m"],
                "p_c": result["p_c"],
                "p_m_rare": result["p_m_rare"],
                "p_c_rare": result["p_c_rare"],
            }
        )
        target_rows.extend(
            {
                "seed": seed,
                "scenario": scenario,
                "component": "aia",
                "variant": name,
                "target_k": int(target_id),
                "score": score,
            }
            for target_id, score in result["per_target"].items()
        )

    score_rows = [
        {
            "seed": seed,
            "scenario": scenario,
            "mia_variant": mia_name,
            "aia_variant": aia_name,
            "mia_power": mia["power"],
            "aia_power": aia["power"],
            "attack_power": mia["power"] + aia["power"],
        }
        for mia_name, mia in mia_results.items()
        for aia_name, aia in aia_results.items()
    ]
    return score_rows, component_rows, target_rows


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [shadow.clean_json(row) for row in frame.to_dict(orient="records")]


def build_summary(
    scores: pd.DataFrame,
    components: pd.DataFrame,
    *,
    args: argparse.Namespace,
    elapsed_seconds: float,
    source_quality: dict[str, Any],
    world_metadata: list[dict[str, Any]],
    run_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    main_mia = "v2_selected_transport"
    base_mia = "v1_transport"
    main_aia = "v2_v1select_adaptive_0p5"
    base_aia = "v1_linked_0p5"

    pair = scores[
        scores["mia_variant"].isin([main_mia, base_mia])
        & scores["aia_variant"].isin([main_aia, base_aia])
    ]
    selected_pairs = pair[
        ((pair["mia_variant"] == main_mia) & (pair["aia_variant"] == main_aia))
        | ((pair["mia_variant"] == base_mia) & (pair["aia_variant"] == base_aia))
    ]
    by_scenario = (
        selected_pairs.groupby(
            ["scenario", "mia_variant", "aia_variant"], as_index=False
        )[["mia_power", "aia_power", "attack_power"]]
        .agg(["mean", "min", "max"])
    )
    by_scenario.columns = [
        "_".join(value for value in column if value)
        if isinstance(column, tuple)
        else column
        for column in by_scenario.columns
    ]
    by_scenario = by_scenario.reset_index(drop=True)

    pivot = selected_pairs.pivot_table(
        index=["seed", "scenario"],
        columns=["mia_variant", "aia_variant"],
        values="attack_power",
    )
    v2_key = (main_mia, main_aia)
    v1_key = (base_mia, base_aia)
    deltas: list[dict[str, Any]] = []
    if v2_key in pivot.columns and v1_key in pivot.columns:
        delta_frame = (pivot[v2_key] - pivot[v1_key]).rename("v2_minus_v1").reset_index()
        deltas = _records(
            delta_frame.groupby("scenario", as_index=False)["v2_minus_v1"]
            .agg(["mean", "min", "max"])
            .reset_index(drop=True)
        )

    ranking_scope = [
        scenario
        for scenario in ("noise_0p03", "noise_0p10", "time_shuffled")
        if scenario in args.scenarios
    ]
    ranking_components = components[components["scenario"].isin(ranking_scope)]
    component_rankings: dict[str, Any] = {}
    for component in ("mia", "aia"):
        subset = ranking_components[ranking_components["component"] == component]
        ranking = (
            subset.groupby("variant", as_index=False)["power"]
            .agg(["mean", "min", "max"])
            .sort_values(["mean", "variant"], ascending=[False, True])
            .reset_index(drop=True)
        )
        component_rankings[component] = _records(ranking)

    sanity_checks = {
        "all_prediction_frames_valid": all(
            all(checks.values())
            for item in run_diagnostics
            for checks in [item["prediction_checks"]]
        ),
        "all_mia_transport_converged": all(
            item["mia_transport"]["max_row_error"] < 1e-7
            and item["mia_transport"]["max_column_error"] < 1e-7
            for item in run_diagnostics
        ),
        "negative_control_included": "independent" in args.scenarios,
    }
    return {
        "report_type": "authorized_local_attack_v2_shadow_comparison",
        "attack_version": 2,
        "official_score_available": False,
        "official_score_blocker": (
            "Organizer-only preliminary membership and AIA truth are not distributed."
        ),
        "interpretation": (
            "Comparative local regression evidence only; not an official score or leaderboard estimate."
        ),
        "parameters": {
            "seeds": args.seeds,
            "scenarios": args.scenarios,
            "mia_transport_epsilon": args.mia_transport_epsilon,
            "mia_density_weight": args.mia_density_weight,
            "mia_rare_weight": args.mia_rare_weight,
            "aia_link_trust_distance": args.aia_link_trust_distance,
            "aia_anomalous_event_rate": args.aia_anomalous_event_rate,
            "aia_anomalous_link_trust_distance": (
                args.aia_anomalous_link_trust_distance
            ),
            "domias_included": not args.skip_domias,
        },
        "main_variants": {
            "v2": {"mia": main_mia, "aia": main_aia},
            "v1": {"mia": base_mia, "aia": base_aia},
        },
        "v1_v2_by_scenario": _records(by_scenario),
        "attack_power_delta_v2_minus_v1": deltas,
        "component_rankings": component_rankings,
        "sanity_checks": sanity_checks,
        "source_quality": source_quality,
        "shadow_worlds": world_metadata,
        "elapsed_seconds": elapsed_seconds,
        "limitations": [
            "The shadow worlds use known toy data and simplified release perturbations.",
            "They do not reproduce every submitted defence or the organiser-only preliminary truth.",
            "Independent is a finite-sample negative control, not a score correction.",
        ],
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    dist = args.dist.resolve()
    out_dir = args.out_dir.resolve()
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    cfg = load_config(KIT_CONFIG_PATH)
    detector = shadow.load_rare_detector(dist)
    spec = shadow.ShadowSpec()
    source_quality = shadow.validate_source_data(
        candidates, b_self, background, spec
    )
    rare_mask = b_self["is_rare"].astype(bool).to_numpy()
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    abstain_time = safe_abstain_time(horizon, tau)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_scores: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    all_target_scores: list[dict[str, Any]] = []
    world_metadata: list[dict[str, Any]] = []
    run_diagnostics: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"[v2 seed {seed}] shadow cohort construction", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        world.metadata["validation_checks"] = shadow.validate_world(world, spec)
        world_metadata.append(world.metadata)
        ref_dir = out_dir / "shadow_reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)

        for scenario in args.scenarios:
            print(f"[v2 seed {seed}] scenario={scenario}", flush=True)
            scenario_code = zlib.crc32(scenario.encode("utf-8"))
            mia_targets: dict[str, pd.DataFrame] = {}
            aia_targets: dict[str, pd.DataFrame] = {}
            for target_id in world.mia_sources:
                mia_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, scenario_code, int(target_id), 1]
                    )
                )
                aia_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, scenario_code, int(target_id), 2]
                    )
                )
                mia_targets[target_id] = shadow.scenario_release(
                    world.mia_sources[target_id],
                    world.attack_background,
                    scenario,
                    cfg,
                    mia_rng,
                )
                aia_targets[target_id] = shadow.scenario_release(
                    world.aia_sources[target_id],
                    world.attack_background,
                    scenario,
                    cfg,
                    aia_rng,
                )

            _, mia_variants, mia_diagnostics = build_mia_v2(
                candidates,
                world.attack_background,
                mia_targets,
                minimum=spec.minimum_shared,
                maximum=spec.maximum_shared,
                total=spec.shared_total,
                transport_epsilon=args.mia_transport_epsilon,
                include_domias=not args.skip_domias,
                density_weight=args.mia_density_weight,
                rare_mask=rare_mask,
                rare_weight=args.mia_rare_weight,
            )
            _, aia_variants, aia_diagnostics = build_aia_v2(
                world.attack_background,
                aia_targets,
                world.challenges,
                main_fraction=0.5,
                variant_fractions=[0.5],
                abstain_time=abstain_time,
                horizon_time=horizon,
                link_trust_distance=args.aia_link_trust_distance,
                anomalous_event_rate=args.aia_anomalous_event_rate,
                anomalous_link_trust_distance=(
                    args.aia_anomalous_link_trust_distance
                ),
                baseline_abstain_time=shadow.choose_low_hit_time(
                    world.attack_background, horizon=horizon, tau=tau
                ),
            )
            prediction_checks = shadow.validate_prediction_frames(
                mia_variants, aia_variants, candidates, spec
            )
            scores, components, target_scores = evaluate_variants(
                mia_variants,
                aia_variants,
                ref_dir,
                cfg,
                seed=seed,
                scenario=scenario,
            )
            all_scores.extend(scores)
            all_components.extend(components)
            all_target_scores.extend(target_scores)
            run_diagnostics.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "prediction_checks": prediction_checks,
                    "mia_transport": mia_diagnostics["transport"],
                    "mia_rare_transport": mia_diagnostics["rare_transport"],
                    "aia_main_variant": aia_diagnostics["main_variant"],
                    "aia_abstain_time": aia_diagnostics["abstain_time"],
                    "aia_baseline_abstain_time": aia_diagnostics[
                        "baseline_abstain_time"
                    ],
                }
            )

    scores_frame = pd.DataFrame(all_scores).sort_values(
        ["scenario", "seed", "mia_variant", "aia_variant"], kind="mergesort"
    )
    components_frame = pd.DataFrame(all_components).sort_values(
        ["scenario", "seed", "component", "variant"], kind="mergesort"
    )
    target_frame = pd.DataFrame(all_target_scores).sort_values(
        ["scenario", "seed", "variant", "target_k"], kind="mergesort"
    )
    elapsed = time.perf_counter() - started
    summary = build_summary(
        scores_frame,
        components_frame,
        args=args,
        elapsed_seconds=elapsed,
        source_quality=source_quality,
        world_metadata=world_metadata,
        run_diagnostics=run_diagnostics,
    )

    scores_frame.to_csv(out_dir / "scores.csv", index=False)
    components_frame.to_csv(out_dir / "component_scores.csv", index=False)
    target_frame.to_csv(out_dir / "aia_target_scores.csv", index=False)
    (out_dir / "run_diagnostics.json").write_text(
        json.dumps(
            shadow.clean_json(run_diagnostics),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            shadow.clean_json(summary),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\nv2 - v1 attack-power delta", flush=True)
    print(
        pd.DataFrame(summary["attack_power_delta_v2_minus_v1"]).to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"results: {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    main()
