"""Paired shadow sensitivity for replacing existing 10.0 predictions.

Only exact ``time_hat == 10.0`` cells are shifted. Existing event-time
predictions and abstentions are kept unchanged. This is research-only and
writes no submission candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_horizon_pairing as core

from attack_phase import evaluate_shadow as shadow
from attack_phase_v2 import generate_submission_v2 as v2
from attack_phase_v4 import generate_submission_v4 as v4
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (
    copula_event_links,
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
    stratified_half_selection,
)
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching
from attack_phase_v6.aia_global_density import global_density_ratio_score
from local_output_import import load_zero_helpers
from pwscup2026.common.config import load_config
from pwscup2026.kit.validate import KIT_CONFIG_PATH


ROOT = core.ROOT
OUT = core.OUT
TARGETS = (1, 2, 3, 4, 5, 10, 12, 13, 20, 22, 23, 26, 29)
V4_TARGETS = {1, 2, 4, 5, 12, 29}
INDEPENDENT_TARGETS = {3, 13, 22, 23, 26}
ROUTED_TARGETS = {10, 20}
SHIFT_VALUES = (9.498631074607, 9.5, 9.75, 10.0)


def build_base_frames(
    world: shadow.ShadowWorld,
    releases: dict[str, pd.DataFrame],
    background: pd.DataFrame,
    cfg: dict,
    template_target: int,
) -> dict[str, np.ndarray]:
    baseline, _, _ = v4.build_aia_v4(
        background,
        releases,
        world.challenges,
        main_fraction=0.5,
        variant_fractions=[0.5],
        abstain_time=core.ABSTAIN,
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
        abstain_time=core.ABSTAIN,
    )
    linkable, _ = filter_linkable_releases(releases)
    output: dict[str, np.ndarray] = {}
    for target_id in sorted(world.challenges, key=int):
        challenge = world.challenges[target_id].reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_block = baseline.loc[
            baseline["target_k"].astype(int).eq(int(target_id))
        ].set_index("challenge_row_id")
        ind_block = independent.loc[
            independent["target_k"].astype(int).eq(int(target_id))
        ].set_index("challenge_row_id")
        v4_time = v4_block.loc[ids, "time_hat"].to_numpy(dtype=float)
        ind_time = ind_block.loc[ids, "time_hat"].to_numpy(dtype=float)
        if template_target in V4_TARGETS:
            values = v4_time.copy()
        elif template_target in INDEPENDENT_TARGETS:
            values = ind_time.copy()
        elif template_target == 10:
            active_v4 = ~np.isclose(v4_time, core.ABSTAIN)
            active_ind = ~np.isclose(ind_time, core.ABSTAIN)
            prox = proximity_score(linkable[target_id], challenge)
            cop = copula_proximity_score(linkable[target_id], challenge)
            score = 1.10 * active_v4 + 0.90 * active_ind + 0.20 * prox + 0.20 * cop
            selected, _ = stratified_half_selection(challenge, score, member_count=163)
            linked_time, linked_gate, _ = copula_event_links(
                linkable[target_id], challenge, margin_quantile=0.50, horizon_time=10.0
            )
            chosen = np.where(active_v4, v4_time, 10.0)
            chosen = np.where(linked_gate, linked_time, chosen)
            values = np.full(len(challenge), core.ABSTAIN, dtype=float)
            values[selected] = chosen[selected]
        elif template_target == 20:
            score = global_density_ratio_score(
                linkable[target_id], challenge, background
            )
            selected, _ = stratified_half_selection(challenge, score, member_count=163)
            values = core.decoded(selected, 10.0)
        else:
            raise ValueError(template_target)
        output[target_id] = values
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background_full = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target: pd.read_csv(target_dir / f"C_{target}.csv") for target in TARGETS
    }
    noise_scales = background_full.loc[:, list(core.VISIBLE_NUMERIC)].std(
        ddof=0
    ).to_numpy(dtype=float)
    scenario_releases = load_zero_helpers(ROOT)["scenario_releases"]
    spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )
    rows: list[dict[str, object]] = []
    for seed in (20261031, 20261032):
        world = shadow.build_shadow_world(
            candidates, b_self, background_full, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        background = world.attack_background
        for template_target in TARGETS:
            for base_scenario in ("warp", "warp_noise_0p03", "warp_time_shuffled"):
                label = f"{base_scenario}_C{template_target}"
                print(f"seed={seed} {label}", flush=True)
                releases = scenario_releases(
                    world,
                    base_scenario,
                    seed=seed,
                    cfg=cfg,
                    template=templates[template_target],
                    noise_scales=noise_scales,
                )
                frames = build_base_frames(
                    world, releases, background, cfg, template_target
                )
                source = (
                    "v4"
                    if template_target in V4_TARGETS
                    else "independent"
                    if template_target in INDEPENDENT_TARGETS
                    else "routed"
                )
                for target_id, base_values in frames.items():
                    challenge = world.challenges[target_id].reset_index(drop=True)
                    truth = world.aia_truth[target_id].reset_index(drop=True)
                    base_result = core.target_result(
                        challenge, truth, base_values, cfg
                    )
                    for shifted_time in SHIFT_VALUES:
                        shifted = base_values.copy()
                        shifted[np.isclose(shifted, 10.0)] = shifted_time
                        result = core.target_result(challenge, truth, shifted, cfg)
                        rows.append(
                            {
                                "seed": seed,
                                "scenario": label,
                                "template_target": template_target,
                                "shadow_target": int(target_id),
                                "source": source,
                                "shifted_time": shifted_time,
                                "changed_10_rows": int(
                                    np.count_nonzero(np.isclose(base_values, 10.0))
                                ),
                                "base_power": float(base_result.score),
                                "shift_power": float(result.score),
                                "delta_power": float(result.score - base_result.score),
                                "base_all": float(base_result.breakdown["r_time_all"]),
                                "shift_all": float(result.breakdown["r_time_all"]),
                                "base_rare": float(base_result.breakdown["r_time_rare"]),
                                "shift_rare": float(result.breakdown["r_time_rare"]),
                            }
                        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "positive_time_shift_scores.csv", index=False)
    changed = result.loc[~np.isclose(result["shifted_time"], 10.0)].copy()
    summary = (
        changed.groupby(["template_target", "source", "shifted_time"], as_index=False)
        .agg(
            cases=("delta_power", "size"),
            base_power_mean=("base_power", "mean"),
            delta_mean=("delta_power", "mean"),
            delta_median=("delta_power", "median"),
            delta_p10=("delta_power", lambda x: float(np.quantile(x, 0.10))),
            delta_min=("delta_power", "min"),
            delta_max=("delta_power", "max"),
            negative_cases=("delta_power", lambda x: int(np.count_nonzero(x < -1e-15))),
            positive_cases=("delta_power", lambda x: int(np.count_nonzero(x > 1e-15))),
        )
        .sort_values(["template_target", "delta_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT / "positive_time_shift_summary.csv", index=False)
    report = {
        "scope": "authorized local paired shadow sensitivity; no candidate created",
        "rule": "replace only exact 10.0 predictions; preserve event times and 11.0 abstentions",
        "best_by_template": summary.groupby("template_target", as_index=False)
        .head(1)
        .to_dict(orient="records"),
    }
    (OUT / "positive_time_shift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
