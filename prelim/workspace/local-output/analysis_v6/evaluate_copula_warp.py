"""Research-only AIA evaluation under production-marginal rank warps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for root in (REPO_ROOT, STARTER_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

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


VISIBLE_CONTINUOUS = ("age", "SBP", "TG", "HDL", "ALT")
PRODUCTION_IDS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
VARIANTS = (
    "v4_reference",
    "independent_reference",
    "stratum_v4__v4_only",
    "stratum_v4_copula__v4_only",
    "consensus_v4_lean__v4_only",
    "consensus_v4_copula__v4_only",
    "consensus_v4_lean__v4_fallback",
    "consensus_v4_lean__independent_fallback",
)


def rank_warp(
    source: pd.DataFrame,
    template: pd.DataFrame,
    *,
    noise_fraction: float,
    noise_scales: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
) -> pd.DataFrame:
    out = source.copy()
    if noise_fraction:
        noise = rng.normal(
            0.0,
            noise_fraction * noise_scales,
            size=(len(out), len(VISIBLE_CONTINUOUS)),
        )
        for index, column in enumerate(VISIBLE_CONTINUOUS):
            out[column] = out[column].to_numpy(dtype=float) + noise[:, index]

    quantile = (np.arange(len(out), dtype=float) + 0.5) / len(out)
    for column in VISIBLE_CONTINUOUS:
        values = out[column].to_numpy(dtype=float)
        order = np.argsort(values, kind="mergesort")
        warped_sorted = np.quantile(
            template[column].to_numpy(dtype=float), quantile, method="linear"
        )
        warped = np.empty(len(out), dtype=float)
        warped[order] = warped_sorted
        out[column] = warped
    out = shadow.coerce_release(out, cfg)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def main() -> None:
    dist = REPO_ROOT / "participant_data"
    target_dir = REPO_ROOT / "team-shared" / "prelim" / "attack-targets"
    out_dir = REPO_ROOT / "local-output" / "attack-shadow-evaluation-v6-copula-warp"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    noise_scales = background[list(VISIBLE_CONTINUOUS)].std(ddof=0).to_numpy(
        dtype=float
    )
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target_id: pd.read_csv(target_dir / f"C_{target_id}.csv")
        for target_id in PRODUCTION_IDS
    }
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
    rows: list[dict] = []
    for seed in (20260909, 20260910, 20260911, 20260912):
        print(f"seed={seed}", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        ref_dir = out_dir / "reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)
        for noise_fraction in (0.0, 0.03):
            scenario = "public_marginal_warp" if not noise_fraction else "public_marginal_warp_noise_0p03"
            releases: dict[str, pd.DataFrame] = {}
            for index, (target_id, source) in enumerate(world.aia_sources.items()):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, index, int(target_id), 606])
                )
                template_id = PRODUCTION_IDS[index]
                releases[target_id] = rank_warp(
                    source,
                    templates[template_id],
                    noise_fraction=noise_fraction,
                    noise_scales=noise_scales,
                    rng=rng,
                    cfg=cfg,
                )
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
            frames = {
                "v4_reference": baseline,
                "independent_reference": independent,
                **structured,
            }
            for name in VARIANTS:
                result = score_aia_power(frames[name], ref_dir, cfg)
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
    pivot = result.pivot_table(
        index=["seed", "scenario"], columns="variant", values="power"
    )
    deltas = pivot.sub(pivot["v4_reference"], axis=0)
    summary = deltas.groupby("scenario").agg(["mean", "min", "max"])
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "purpose": "directional comparison under public marginal warps",
                "deltas": {
                    scenario: {
                        variant: {
                            statistic: float(summary.loc[scenario, (variant, statistic)])
                            for statistic in ("mean", "min", "max")
                        }
                        for variant in pivot.columns
                    }
                    for scenario in summary.index
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.round(6).to_string(), flush=True)


if __name__ == "__main__":
    main()
