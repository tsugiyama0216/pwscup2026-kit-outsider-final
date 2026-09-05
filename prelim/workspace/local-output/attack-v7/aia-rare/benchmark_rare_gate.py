"""Fresh split/stress benchmark for rare-only v7 AIA overlays.

Research only: no production candidate, archive, token, or submission is touched.
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
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, NOVEL, STARTER, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import evaluate_horizon_pairing as core  # noqa: E402
import evaluate_positive_shift as positive  # noqa: E402
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
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from local_output_import import load_zero_helpers  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from rare_proxy import fit_proxy_models, top_k_mask  # noqa: E402


DISCOVERY = (20270301, 20270302)
VALIDATION = (20270311, 20270312)
SCENARIOS = ("warp", "warp_noise_0p03", "warp_time_shuffled")
PRESERVING = ("warp", "warp_noise_0p03")
TARGETS = (5, 6, 7, 9, 12, 17, 18, 19, 23)
POSITIVE = frozenset(positive.TARGETS)
ABSTAIN = 11.0
EDGE = 9.5


def aligned(frame: pd.DataFrame, target: str, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(int(target))].set_index(
        "challenge_row_id"
    )
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def zero_base_frames(
    official_target: int,
    world: shadow.ShadowWorld,
    releases: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
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
    output: dict[str, np.ndarray] = {}
    probe_power: dict[str, float] = {}
    for target in sorted(world.challenges, key=int):
        challenge = world.challenges[target].reset_index(drop=True)
        truth = world.aia_truth[target].reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_values = aligned(baseline, target, ids)
        ind_values = aligned(independent, target, ids)
        v4_active = ~np.isclose(v4_values, ABSTAIN)
        ind_active = ~np.isclose(ind_values, ABSTAIN)
        routed, _, _ = core.production_selector(
            official_target,
            linkable[target],
            challenge,
            background,
            v4_active,
            ind_active,
        )
        probe_power[target] = max(
            core.target_power(challenge, truth, v4_values, cfg),
            core.target_power(challenge, truth, ind_values, cfg),
            core.target_power(
                challenge, truth, core.decoded(routed, 10.0), cfg
            ),
        )
        current = ~(v4_active & ind_active & routed)
        output[target] = core.decoded(current, EDGE)
    return output, probe_power


def variants(
    base: np.ndarray,
    challenge: pd.DataFrame,
    release: pd.DataFrame,
    proxy_score: np.ndarray,
) -> dict[str, np.ndarray]:
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    rare26 = top_k_mask(proxy_score, ids, 26)
    rare52 = top_k_mask(proxy_score, ids, 52)
    active = ~np.isclose(base, ABSTAIN)
    output = {"baseline": base.copy()}
    links: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for q in (0.50, 0.75, 0.90):
        linked_time, linked_gate, _ = copula_event_links(
            release, challenge, margin_quantile=q, horizon_time=10.0
        )
        links[q] = linked_time, linked_gate
        values = base.copy()
        gate = active & rare26 & linked_gate
        values[gate] = linked_time[gate]
        output[f"link_active_q{str(q).replace('.', 'p')}_k26"] = values
    linked_time, linked_gate = links[0.50]
    values = base.copy()
    gate = active & rare52 & linked_gate
    values[gate] = linked_time[gate]
    output["link_active_q0p5_k52"] = values
    values = base.copy()
    gate = rare26 & linked_gate
    values[gate] = linked_time[gate]
    output["link_expand_q0p5_k26"] = values
    values = base.copy()
    values[rare26 & ~active] = EDGE
    output["activate_edge_k26"] = values
    values = base.copy()
    values[rare26] = EDGE
    output["force_edge_k26"] = values
    values = base.copy()
    values[rare26] = ABSTAIN
    output["abstain_k26"] = values
    return output


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["split", "scenario", "official_target", "variant"], as_index=False)
        .agg(
            cases=("delta", "size"),
            mean_delta=("delta", "mean"),
            p10_delta=("delta", lambda x: float(np.quantile(x, 0.10))),
            min_delta=("delta", "min"),
            win_rate=("delta", lambda x: float(np.mean(x > 1e-12))),
            loss_rate=("delta", lambda x: float(np.mean(x < -1e-12))),
            mean_delta_all=("delta_all", "mean"),
            p10_delta_all=("delta_all", lambda x: float(np.quantile(x, 0.10))),
            mean_delta_rare=("delta_rare", "mean"),
        )
    )


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    proxy = fit_proxy_models(
        background, shadow.public_rare_mask(background, detector), detector
    )
    cfg = load_config(KIT_CONFIG_PATH)
    helper = load_zero_helpers(ROOT)["scenario_releases"]
    noise_scales = background.loc[:, list(core.VISIBLE_NUMERIC)].std(ddof=0).to_numpy(
        dtype=float
    )
    templates = {target: pd.read_csv(target_dir / f"C_{target}.csv") for target in TARGETS}
    spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )
    rows: list[dict[str, Any]] = []
    for split, seeds in (("discovery", DISCOVERY), ("validation", VALIDATION)):
        for seed in seeds:
            world = shadow.build_shadow_world(
                candidates, b_self, background, detector, spec, seed
            )
            exactify_visible_control_matching(world)
            shadow.validate_world(world, spec)
            for official_target in TARGETS:
                for scenario in SCENARIOS:
                    print(f"{split} {seed} C{official_target} {scenario}", flush=True)
                    releases = helper(
                        world,
                        scenario,
                        seed=seed,
                        cfg=cfg,
                        template=templates[official_target],
                        noise_scales=noise_scales,
                    )
                    linkable, _ = filter_linkable_releases(releases)
                    if official_target in POSITIVE:
                        bases = positive.build_base_frames(
                            world, releases, world.attack_background, cfg, official_target
                        )
                        probe = {target: float("nan") for target in bases}
                        for target in bases:
                            exact = np.asarray(bases[target], dtype=float).copy()
                            exact[np.isclose(exact, 10.0)] = EDGE
                            bases[target] = exact
                    else:
                        bases, probe = zero_base_frames(
                            official_target, world, releases, cfg
                        )
                    for target, base in bases.items():
                        challenge = world.challenges[target].reset_index(drop=True)
                        truth = world.aia_truth[target].reset_index(drop=True)
                        proxy_score = proxy.scores(challenge)["extra_trees"]
                        candidate = variants(
                            base, challenge, linkable[target], proxy_score
                        )
                        base_metric = core.target_result(challenge, truth, base, cfg)
                        base_all = float(base_metric.breakdown["r_time_all"])
                        base_rare = float(base_metric.breakdown["r_time_rare"])
                        for name, values in candidate.items():
                            result = core.target_result(challenge, truth, values, cfg)
                            all_score = float(result.breakdown["r_time_all"])
                            rare_score = float(result.breakdown["r_time_rare"])
                            rows.append(
                                {
                                    "split": split,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "official_target": official_target,
                                    "shadow_target": int(target),
                                    "probe_power": probe[target],
                                    "probe_zero": bool(
                                        np.isnan(probe[target]) or probe[target] <= 1e-15
                                    ),
                                    "variant": name,
                                    "power": float(result.score),
                                    "all": all_score,
                                    "rare": rare_score,
                                    "delta": float(result.score - base_metric.score),
                                    "delta_all": all_score - base_all,
                                    "delta_rare": rare_score - base_rare,
                                }
                            )
    scored = pd.DataFrame(rows)
    # Official-zero target sensitivity must match the observed zero probe condition.
    eligible = scored.loc[scored["probe_zero"]].copy()
    eligible.to_csv(HERE / "rare_gate_scores.csv", index=False)
    summary = summarize(eligible)
    summary.to_csv(HERE / "rare_gate_summary.csv", index=False)

    preserving = (
        eligible.loc[eligible["scenario"].isin(PRESERVING)]
        .groupby(["split", "official_target", "variant"], as_index=False)
        .agg(
            cases=("delta", "size"),
            mean_delta=("delta", "mean"),
            p10_delta=("delta", lambda x: float(np.quantile(x, 0.10))),
            win_rate=("delta", lambda x: float(np.mean(x > 1e-12))),
            mean_delta_all=("delta_all", "mean"),
            p10_delta_all=("delta_all", lambda x: float(np.quantile(x, 0.10))),
            mean_delta_rare=("delta_rare", "mean"),
        )
    )
    discovery = preserving.loc[
        preserving["split"].eq("discovery")
        & ~preserving["variant"].eq("baseline")
        & (preserving["mean_delta"] > 0.0)
        & (preserving["win_rate"] >= 0.60)
        & (preserving["p10_delta"] >= -0.01)
        & (preserving["mean_delta_all"] >= -0.002)
        & (preserving["p10_delta_all"] >= -0.01)
    ].copy()
    chosen: dict[int, str] = {}
    for target, group in discovery.groupby("official_target"):
        best = group.sort_values(
            ["mean_delta", "p10_delta", "mean_delta_all"],
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        chosen[int(target)] = str(best["variant"])

    validation_rows = []
    for target, variant in chosen.items():
        block = eligible.loc[
            eligible["split"].eq("validation")
            & eligible["official_target"].eq(target)
            & eligible["variant"].eq(variant)
            & eligible["scenario"].isin(PRESERVING)
        ]
        validation_rows.append(block)
    validation = (
        pd.concat(validation_rows, ignore_index=True)
        if validation_rows
        else eligible.iloc[0:0].copy()
    )
    gate = {
        "candidate_targets": chosen,
        "cases": int(len(validation)),
        "mean_delta": float(validation["delta"].mean()) if len(validation) else 0.0,
        "win_rate": float(np.mean(validation["delta"] > 1e-12)) if len(validation) else 0.0,
        "p10_delta": float(np.quantile(validation["delta"], 0.10)) if len(validation) else 0.0,
        "mean_delta_all": float(validation["delta_all"].mean()) if len(validation) else 0.0,
        "p10_delta_all": float(np.quantile(validation["delta_all"], 0.10)) if len(validation) else 0.0,
        "mean_delta_rare": float(validation["delta_rare"].mean()) if len(validation) else 0.0,
    }
    gate["pass"] = bool(
        gate["cases"] > 0
        and gate["mean_delta"] > 0.0
        and gate["win_rate"] >= 0.60
        and gate["p10_delta"] >= -0.01
        and gate["mean_delta_all"] >= 0.0
        and gate["p10_delta_all"] >= -0.005
    )
    report = {
        "decision": "adopt" if gate["pass"] else "reject",
        "acceptance_gate": {
            "validation_mean_delta_gt": 0.0,
            "validation_win_rate_gte": 0.60,
            "validation_p10_delta_gte": -0.01,
            "validation_mean_delta_all_gte": 0.0,
            "validation_p10_delta_all_gte": -0.005,
        },
        "validation": gate,
        "discovery_seeds": list(DISCOVERY),
        "validation_seeds": list(VALIDATION),
        "preserving_scenarios": list(PRESERVING),
        "stress_scenario": "warp_time_shuffled",
        "tested_targets": list(TARGETS),
        "rare_proxy": "A_bg-trained ExtraTrees, top 26 public-feature scores",
        "official_zero_condition": (
            "maximum direct power across v4, independent, and routed probes "
            "is exactly zero"
        ),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "submission_performed": False,
    }
    (HERE / "rare_gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
