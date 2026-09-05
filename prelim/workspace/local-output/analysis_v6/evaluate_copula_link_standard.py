"""Research-only copula-link evaluation under standard shadow scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_SRC = REPO_ROOT / "starter" / "src"
ANALYSIS_DIR = Path(__file__).resolve().parent
for root in (REPO_ROOT, STARTER_SRC, ANALYSIS_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from evaluate_copula_link import linked_frame  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import build_aia_v5_variants  # noqa: E402
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import score_aia_power  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


def main() -> None:
    dist = REPO_ROOT / "participant_data"
    out_dir = REPO_ROOT / "local-output" / "attack-shadow-evaluation-v6-copula-link-standard"
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
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    abstain = v2.safe_abstain_time(horizon, tau)
    scenarios = ("raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent")
    modes = ("nearest_selected", "assignment_selected", "assignment_new")
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate((20260909, 20260910, 20260911, 20260912)):
        print(f"seed={seed}", flush=True)
        world = shadow.build_shadow_world(candidates, b_self, background, detector, spec, seed)
        exactify_visible_control_matching(world)
        ref_dir = out_dir / "reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)
        for scenario_index, scenario in enumerate(scenarios):
            releases = {
                target_id: shadow.scenario_release(
                    source,
                    world.attack_background,
                    scenario,
                    cfg,
                    np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, seed_index, scenario_index, int(target_id), 808]
                        )
                    ),
                )
                for target_id, source in world.aia_sources.items()
            }
            baseline, _, _ = v4.build_aia_v4(
                world.attack_background,
                releases,
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
                anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
                integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
            )
            independent, _, _ = build_aia_independent(
                world.attack_background,
                releases,
                world.challenges,
                member_count=spec.aia_members,
                horizon=horizon,
                abstain_time=abstain,
            )
            structured, _ = build_aia_v5_variants(
                baseline,
                independent,
                releases,
                world.challenges,
                member_count=spec.aia_members,
                abstain_time=abstain,
                horizon_time=horizon,
            )
            template = structured["consensus_v4_copula__horizon"]
            frames = {
                "v4_reference": baseline,
                "independent_reference": independent,
                "copula_v4_only": structured["consensus_v4_copula__v4_only"],
                "copula_horizon": template,
            }
            for mode in modes:
                parts = []
                for target_id in sorted(releases, key=int):
                    parts.append(
                        linked_frame(
                            template.loc[template["target_k"].astype(int).eq(int(target_id))],
                            releases[target_id],
                            world.challenges[target_id],
                            v4_frame=baseline.loc[
                                baseline["target_k"].astype(int).eq(int(target_id))
                            ],
                            mode=mode,
                            abstain=abstain,
                            horizon=horizon,
                        )
                    )
                frames[f"copula_{mode}"] = pd.concat(parts, ignore_index=True)
            for name, frame in frames.items():
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
                        "per_target": json.dumps(result["per_target"], sort_keys=True),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "component_scores.csv", index=False)
    pivot = result.pivot_table(index=["seed", "scenario"], columns="variant", values="power")
    delta = pivot.sub(pivot["v4_reference"], axis=0)
    summary = delta.groupby("scenario").agg(["mean", "min", "max"])
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                scenario: {
                    variant: {
                        stat: float(summary.loc[scenario, (variant, stat)])
                        for stat in ("mean", "min", "max")
                    }
                    for variant in pivot.columns
                }
                for scenario in summary.index
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.round(6).to_string(), flush=True)


if __name__ == "__main__":
    main()
