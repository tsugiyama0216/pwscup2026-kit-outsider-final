"""Conditional null-shadow comparison of three expert complements.

This intentionally conditions on the same qualitative evidence observed in
production: v4, independent, and routed all have clipped AIA power zero.  It is
a sensitivity analysis, not a private-score estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark as bench  # noqa: E402


DISCOVERY = tuple(range(20261201, 20261207))
VALIDATION = tuple(range(20261211, 20261217))


def complement_active(
    values: np.ndarray, *, horizon: float, abstain: float
) -> np.ndarray:
    active = ~np.isclose(values, abstain)
    output = np.full(len(values), horizon, dtype=float)
    output[active] = abstain
    return output


def run_seed(
    split: str,
    seed: int,
    *,
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    detector: dict,
    spec: bench.shadow.ShadowSpec,
    cfg: dict,
    horizon: float,
    abstain: float,
) -> list[dict]:
    world = bench.shadow.build_shadow_world(
        candidates, b_self, background, detector, spec, seed
    )
    bench.exactify_visible_control_matching(world)
    bench.shadow.validate_world(world, spec)
    releases = {}
    for ordinal, (target_id, source) in enumerate(world.aia_sources.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, ordinal, int(target_id), 8803])
        )
        releases[target_id] = bench.shadow.scenario_release(
            source, world.attack_background, "independent", cfg, rng
        )

    baseline, _, _ = bench.v4.build_aia_v4(
        world.attack_background,
        releases,
        world.challenges,
        main_fraction=0.5,
        variant_fractions=[0.5],
        abstain_time=abstain,
        horizon_time=horizon,
        link_trust_distance=bench.v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
        baseline_abstain_time=bench.shadow.choose_low_hit_time(
            world.attack_background,
            horizon=horizon,
            tau=float(cfg["aia"]["tau_time"]),
        ),
        anomalous_event_rate=bench.v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        anomalous_link_trust_distance=bench.v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        integrity_auc_threshold=bench.v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    independent, _, _ = bench.build_aia_independent(
        world.attack_background,
        releases,
        world.challenges,
        member_count=163,
        horizon=horizon,
        abstain_time=abstain,
    )
    structured, _ = bench.build_aia_v5_variants(
        baseline,
        independent,
        releases,
        world.challenges,
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_all5, _ = bench.build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_stable3, _ = bench.build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        columns=bench.VIEWS["stable3"],
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_stable3_invert, _ = bench.build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        columns=bench.VIEWS["stable3"],
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
        invert=True,
    )
    sex_consensus, _ = bench.build_sex_consensus_horizon(
        baseline,
        independent,
        releases,
        world.challenges,
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    route_sources = {
        "copula": structured["consensus_v4_copula__horizon"],
        "density_all5": density_all5,
        "density_stable3": density_stable3,
        "density_stable3_invert": density_stable3_invert,
        "sex_consensus": sex_consensus,
    }

    rows: list[dict] = []
    for ordinal, internal_id in enumerate(sorted(world.challenges, key=int)):
        official_id = bench.OFFICIAL_ZERO_IDS[ordinal]
        challenge = world.challenges[internal_id].reset_index(drop=True)
        truth = world.aia_truth[internal_id].reset_index(drop=True)
        v4_time = bench.aligned_time(baseline, internal_id, challenge)
        ind_time = bench.aligned_time(independent, internal_id, challenge)
        routed_time = bench.select_routed_times(
            official_id, internal_id, challenge, route_sources
        )
        direct = {
            "v4": bench.score_block(v4_time, truth, cfg)["power"],
            "independent": bench.score_block(ind_time, truth, cfg)["power"],
            "routed": bench.score_block(routed_time, truth, cfg)["power"],
        }
        active = {
            "v4": (~np.isclose(v4_time, abstain)).astype(float),
            "independent": (~np.isclose(ind_time, abstain)).astype(float),
            "routed": (~np.isclose(routed_time, abstain)).astype(float),
        }
        complements = {
            "complement_v4_active": complement_active(
                v4_time, horizon=horizon, abstain=abstain
            ),
            "complement_independent_active": complement_active(
                ind_time, horizon=horizon, abstain=abstain
            ),
            "complement_routed_active": complement_active(
                routed_time, horizon=horizon, abstain=abstain
            ),
            "inverse_consensus": bench.horizon_decode(
                challenge,
                -(active["v4"] + active["independent"] + active["routed"]),
                horizon=horizon,
                abstain=abstain,
            ),
        }
        exact_zero = max(direct.values()) <= 1e-12
        near_zero = max(direct.values()) <= 0.02
        for strategy, predicted in complements.items():
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "official_target": official_id,
                    "internal_target": internal_id,
                    "strategy": strategy,
                    "v4_direct": direct["v4"],
                    "independent_direct": direct["independent"],
                    "routed_direct": direct["routed"],
                    "all_three_exact_zero": exact_zero,
                    "all_three_at_most_0p02": near_zero,
                    **bench.score_block(predicted, truth, cfg),
                }
            )
    return rows


def summarize(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    selected = frame.loc[frame[condition].astype(bool)].copy()
    if selected.empty:
        return pd.DataFrame()
    keys = ["split", "seed", "official_target"]
    baseline = selected.loc[
        selected["strategy"].eq("complement_routed_active"),
        [*keys, "power"],
    ].rename(columns={"power": "routed_complement_power"})
    selected = selected.merge(baseline, on=keys, how="left")
    selected["delta_vs_routed_complement"] = (
        selected["power"] - selected["routed_complement_power"]
    )
    return (
        selected.groupby(["split", "strategy"], as_index=False)
        .agg(
            eligible_blocks=("power", "size"),
            mean=("power", "mean"),
            median=("power", "median"),
            p10=("power", lambda value: float(np.quantile(value, 0.10))),
            p90=("power", lambda value: float(np.quantile(value, 0.90))),
            mean_delta_vs_routed_complement=(
                "delta_vs_routed_complement",
                "mean",
            ),
            win_rate_vs_routed_complement=(
                "delta_vs_routed_complement",
                lambda value: float(np.mean(value > 1e-12)),
            ),
        )
        .sort_values(["split", "mean"], ascending=[True, False])
    )


def main() -> None:
    candidates = pd.read_csv(bench.DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(bench.DIST / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(bench.DIST / "A_bg.csv").reset_index(drop=True)
    detector = bench.shadow.load_rare_detector(bench.DIST)
    cfg = bench.load_config(bench.KIT_CONFIG_PATH)
    spec = bench.shadow.ShadowSpec(
        n_targets=7,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=175,
        minimum_shared=25,
        maximum_shared=26,
    )
    horizon = float(cfg["onset"]["horizon_years"])
    abstain = bench.v2.safe_abstain_time(
        horizon, float(cfg["aia"]["tau_time"])
    )
    rows: list[dict] = []
    for split, seeds in (("discovery", DISCOVERY), ("validation", VALIDATION)):
        for seed in seeds:
            print(f"{split} seed={seed}", flush=True)
            rows.extend(
                run_seed(
                    split,
                    seed,
                    candidates=candidates,
                    b_self=b_self,
                    background=background,
                    detector=detector,
                    spec=spec,
                    cfg=cfg,
                    horizon=horizon,
                    abstain=abstain,
                )
            )
    frame = pd.DataFrame(rows)
    exact = summarize(frame, "all_three_exact_zero")
    near = summarize(frame, "all_three_at_most_0p02")
    frame.to_csv(bench.OUT / "conditional_complement_scores.csv", index=False)
    exact.to_csv(bench.OUT / "conditional_exact_zero_summary.csv", index=False)
    near.to_csv(bench.OUT / "conditional_near_zero_summary.csv", index=False)
    report = {
        "purpose": "conditional null-shadow complement sensitivity; not a leaderboard forecast",
        "discovery_seeds": list(DISCOVERY),
        "validation_seeds": list(VALIDATION),
        "condition_exact": "v4, independent, and routed clipped per-target AIA power all equal 0",
        "condition_near": "v4, independent, and routed per-target AIA power all <= 0.02",
        "exact_summary": exact.to_dict(orient="records"),
        "near_summary": near.to_dict(orient="records"),
        "caveat": "Conditioning under an independent null is post-selection sensitivity analysis and cannot reconstruct private truth.",
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
    }
    (bench.OUT / "conditional_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(exact.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
