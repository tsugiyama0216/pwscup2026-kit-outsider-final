"""Held-out time-decoder check for the official-zero target 19 route.

The membership selector is the exact complement of the production target-19
probe.  Evaluation is conditioned on the probe scoring zero, as observed on the
official leaderboard.  This research script writes only local-output files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NOVEL = ROOT / "local-output" / "attack-v6" / "novel-aia"
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, NOVEL, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import evaluate_horizon_pairing as core  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    filter_linkable_releases,
)
from attack_phase_v5.evaluate_shadow_v5 import (  # noqa: E402
    exactify_visible_control_matching,
)
from local_output_import import load_zero_helpers  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


TARGET = 19
EDGE = 9.498631074607
ABSTAIN = 11.0
DISCOVERY_SEEDS = (20270201, 20270202, 20270203, 20270204)
VALIDATION_SEEDS = (20270211, 20270212, 20270213, 20270214)
SCENARIOS = ("warp", "warp_noise_0p03", "warp_time_shuffled")


def aligned(frame: pd.DataFrame, target_id: str, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(int(target_id))].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def metrics(result: Any) -> dict[str, float]:
    b = result.breakdown
    return {
        "power": float(result.score),
        "all": float(b["r_time_all"]),
        "rare": float(b["r_time_rare"]),
        "p_m": float(b["succ_time_member"]),
        "p_c": float(b["succ_time_control"]),
        "p_m_rare": float(b["succ_time_member_rare"]),
        "p_c_rare": float(b["succ_time_control_rare"]),
    }


def run_case(
    *,
    split: str,
    seed: int,
    scenario: str,
    world: shadow.ShadowWorld,
    releases: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    background = world.attack_background
    baseline, _, _ = v4.build_aia_v4(
        background,
        releases,
        world.challenges,
        main_fraction=0.5,
        variant_fractions=[0.5],
        abstain_time=ABSTAIN,
        horizon_time=10.0,
        link_trust_distance=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
        baseline_abstain_time=shadow.choose_low_hit_time(
            background, horizon=10.0, tau=0.5
        ),
        anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    independent, _, _ = build_aia_independent(
        background,
        releases,
        world.challenges,
        member_count=163,
        horizon=10.0,
        abstain_time=ABSTAIN,
    )
    linkable, _ = filter_linkable_releases(releases)
    rows: list[dict[str, Any]] = []
    for target_id in sorted(world.challenges, key=int):
        challenge = world.challenges[target_id].reset_index(drop=True)
        truth = world.aia_truth[target_id].reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_time = aligned(baseline, target_id, ids)
        independent_time = aligned(independent, target_id, ids)
        probe, selected, _ = core.production_selector(
            TARGET,
            linkable[target_id],
            challenge,
            background,
            ~np.isclose(v4_time, ABSTAIN),
            ~np.isclose(independent_time, ABSTAIN),
        )
        probe_power = core.target_power(
            challenge, truth, core.decoded(probe, 10.0), cfg
        )
        variants = {"edge": core.decoded(selected, EDGE)}
        for q in (0.50, 0.75, 0.90):
            linked_time, linked_gate, _ = copula_event_links(
                linkable[target_id],
                challenge,
                margin_quantile=q,
                horizon_time=10.0,
            )
            values = variants["edge"].copy()
            gate = selected & linked_gate
            values[gate] = linked_time[gate]
            variants[f"event_q{str(q).replace('.', 'p')}"] = values
        for decoder, values in variants.items():
            result = core.target_result(challenge, truth, values, cfg)
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "scenario": scenario,
                    "shadow_target": int(target_id),
                    "probe_power": probe_power,
                    "probe_zero": probe_power <= 1e-15,
                    "decoder": decoder,
                    **metrics(result),
                }
            )
    return rows


def main() -> None:
    dist = ROOT / "participant_data"
    targets_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background_full = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    template = pd.read_csv(targets_dir / f"C_{TARGET}.csv")
    noise_scales = background_full.loc[:, list(core.VISIBLE_NUMERIC)].std(
        ddof=0
    ).to_numpy(dtype=float)
    helper = load_zero_helpers(ROOT)["scenario_releases"]
    spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )
    rows: list[dict[str, Any]] = []
    for split, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("validation", VALIDATION_SEEDS),
    ):
        for seed in seeds:
            world = shadow.build_shadow_world(
                candidates, b_self, background_full, detector, spec, seed
            )
            exactify_visible_control_matching(world)
            shadow.validate_world(world, spec)
            for scenario in SCENARIOS:
                print(f"{split} seed={seed} {scenario}", flush=True)
                releases = helper(
                    world,
                    scenario,
                    seed=seed,
                    cfg=cfg,
                    template=template,
                    noise_scales=noise_scales,
                )
                rows.extend(
                    run_case(
                        split=split,
                        seed=seed,
                        scenario=scenario,
                        world=world,
                        releases=releases,
                        cfg=cfg,
                    )
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(HERE / "zero19_decoder_scores.csv", index=False)
    conditioned = scores.loc[scores["probe_zero"]].copy()
    base = conditioned.loc[
        conditioned["decoder"].eq("edge"),
        ["split", "seed", "scenario", "shadow_target", "power", "all", "rare"],
    ].rename(columns={"power": "base", "all": "base_all", "rare": "base_rare"})
    conditioned = conditioned.merge(
        base,
        on=["split", "seed", "scenario", "shadow_target"],
        how="left",
        validate="many_to_one",
    )
    conditioned["delta"] = conditioned["power"] - conditioned["base"]
    conditioned["delta_all"] = conditioned["all"] - conditioned["base_all"]
    conditioned["delta_rare"] = conditioned["rare"] - conditioned["base_rare"]
    conditioned.to_csv(HERE / "zero19_decoder_conditioned.csv", index=False)
    summary = (
        conditioned.groupby(["split", "scenario", "decoder"], as_index=False)
        .agg(
            cases=("delta", "size"),
            power_mean=("power", "mean"),
            delta_mean=("delta", "mean"),
            delta_p10=("delta", lambda x: float(np.quantile(x, 0.10))),
            delta_min=("delta", "min"),
            negative_rate=("delta", lambda x: float(np.mean(x < -1e-15))),
            positive_rate=("delta", lambda x: float(np.mean(x > 1e-15))),
            delta_rare_mean=("delta_rare", "mean"),
        )
        .sort_values(["split", "scenario", "delta_mean"], ascending=[True, True, False])
    )
    summary.to_csv(HERE / "zero19_decoder_summary.csv", index=False)
    report = {
        "scope": "authorized local shadow evaluation; target 19 only",
        "condition": "production probe direction has exactly zero shadow power",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "summary": summary.to_dict(orient="records"),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "submission_performed": False,
    }
    (HERE / "zero19_decoder_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
