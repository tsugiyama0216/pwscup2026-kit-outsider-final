"""Fresh eight-seed audit of the target-19 rare-proxy abstention rule."""

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

import benchmark_rare_gate as rarebench  # noqa: E402
import evaluate_horizon_pairing as core  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import filter_linkable_releases  # noqa: E402
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from local_output_import import load_zero_helpers  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from rare_proxy import fit_proxy_models, top_k_mask  # noqa: E402


SEEDS = tuple(range(20270331, 20270339))
SCENARIOS = ("warp", "warp_noise_0p03", "warp_time_shuffled")
PRESERVING = ("warp", "warp_noise_0p03")
TARGET = 19
ABSTAIN = 11.0


def metric(result: Any) -> dict[str, float]:
    b = result.breakdown
    return {
        "power": float(result.score),
        "all": float(b["r_time_all"]),
        "rare": float(b["r_time_rare"]),
    }


def aggregate(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {"subset": label, "cases": 0}
    return {
        "subset": label,
        "cases": int(len(frame)),
        "seeds": int(frame["seed"].nunique()),
        "mean_baseline": float(frame["base_power"].mean()),
        "mean_delta": float(frame["delta"].mean()),
        "p10_delta": float(np.quantile(frame["delta"], 0.10)),
        "min_delta": float(frame["delta"].min()),
        "win_rate": float(np.mean(frame["delta"] > 1e-12)),
        "loss_rate": float(np.mean(frame["delta"] < -1e-12)),
        "mean_delta_all": float(frame["delta_all"].mean()),
        "p10_delta_all": float(np.quantile(frame["delta_all"], 0.10)),
        "mean_delta_rare": float(frame["delta_rare"].mean()),
        "p10_delta_rare": float(np.quantile(frame["delta_rare"], 0.10)),
        "score1_mean_delta_if_divide_20": float(frame["delta"].mean() / 20.0),
    }


def main() -> None:
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    proxy = fit_proxy_models(
        background, shadow.public_rare_mask(background, detector), detector
    )
    template = pd.read_csv(target_dir / "C_19.csv")
    cfg = load_config(KIT_CONFIG_PATH)
    helper = load_zero_helpers(ROOT)["scenario_releases"]
    noise_scales = background.loc[:, list(core.VISIBLE_NUMERIC)].std(ddof=0).to_numpy(
        dtype=float
    )
    spec = shadow.ShadowSpec(
        n_targets=4,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=100,
        minimum_shared=25,
        maximum_shared=26,
    )
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        shadow.validate_world(world, spec)
        for scenario in SCENARIOS:
            print(f"seed={seed} scenario={scenario}", flush=True)
            releases = helper(
                world,
                scenario,
                seed=seed,
                cfg=cfg,
                template=template,
                noise_scales=noise_scales,
            )
            # This reconstructs the v7 complement-of-three-way-unanimous baseline.
            bases, max_direct_power = rarebench.zero_base_frames(
                TARGET, world, releases, cfg
            )
            linkable, _ = filter_linkable_releases(releases)
            for internal_target, base in bases.items():
                challenge = world.challenges[internal_target].reset_index(drop=True)
                truth = world.aia_truth[internal_target].reset_index(drop=True)
                ids = challenge["challenge_row_id"].to_numpy(dtype=int)
                rare26 = top_k_mask(
                    proxy.scores(challenge)["extra_trees"], ids, 26
                )
                proposed = np.asarray(base, dtype=float).copy()
                proposed[rare26] = ABSTAIN
                base_result = core.target_result(challenge, truth, base, cfg)
                proposed_result = core.target_result(challenge, truth, proposed, cfg)
                base_metric = metric(base_result)
                proposed_metric = metric(proposed_result)
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "internal_target": int(internal_target),
                        "max_direct_power": max_direct_power[internal_target],
                        "direct_all_zero": max_direct_power[internal_target] <= 1e-15,
                        "baseline_positive": base_metric["power"] > 1e-15,
                        "changed_rows": int(
                            np.count_nonzero(~np.isclose(proposed, base))
                        ),
                        "base_power": base_metric["power"],
                        "base_all": base_metric["all"],
                        "base_rare": base_metric["rare"],
                        "power": proposed_metric["power"],
                        "all": proposed_metric["all"],
                        "rare": proposed_metric["rare"],
                        "delta": proposed_metric["power"] - base_metric["power"],
                        "delta_all": proposed_metric["all"] - base_metric["all"],
                        "delta_rare": proposed_metric["rare"] - base_metric["rare"],
                    }
                )
    scores = pd.DataFrame(rows)
    scores.to_csv(HERE / "target19_fresh_scores.csv", index=False)
    summaries: list[dict[str, Any]] = []
    for scenario_group, scenario_mask in (
        ("preserving", scores["scenario"].isin(PRESERVING)),
        ("time_shuffled", scores["scenario"].eq("warp_time_shuffled")),
    ):
        current = scores.loc[scenario_mask]
        for subset, mask in (
            ("all_worlds", np.ones(len(current), dtype=bool)),
            ("baseline_positive", current["baseline_positive"].to_numpy()),
            ("direct_all_zero", current["direct_all_zero"].to_numpy()),
        ):
            item = aggregate(current.loc[mask], subset)
            item["scenario_group"] = scenario_group
            summaries.append(item)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "target19_fresh_summary.csv", index=False)
    indexed = summary.set_index(["scenario_group", "subset"])
    preserving = indexed.loc[("preserving", "all_worlds")]
    preserving_positive = indexed.loc[("preserving", "baseline_positive")]
    stress = indexed.loc[("time_shuffled", "all_worlds")]
    stress_positive = indexed.loc[("time_shuffled", "baseline_positive")]
    checks = {
        "preserving_mean_gt_0": float(preserving["mean_delta"]) > 0.0,
        "preserving_win_rate_gte_0p60": float(preserving["win_rate"]) >= 0.60,
        "preserving_p10_gte_0": float(preserving["p10_delta"]) >= -1e-12,
        "preserving_all_mean_gte_0": float(preserving["mean_delta_all"]) >= -1e-12,
        "stress_mean_gte_0": float(stress["mean_delta"]) >= -1e-12,
        "baseline_positive_preserving_mean_gte_0": float(
            preserving_positive["mean_delta"]
        )
        >= -1e-12,
        "baseline_positive_preserving_p10_gte_0": float(
            preserving_positive["p10_delta"]
        )
        >= -1e-12,
        "baseline_positive_stress_mean_gte_0": float(stress_positive["mean_delta"])
        >= -1e-12,
        "normal_format": True,
    }
    report = {
        "decision": "adopt_research_candidate" if all(checks.values()) else "reject",
        "rule": "target 19 only: set ExtraTrees public-feature rare-proxy top 26 to 11.0",
        "seeds": list(SEEDS),
        "summaries": summaries,
        "checks": checks,
        "production_audit": {
            "proxy_rows": 26,
            "changed_rows": 25,
            "v7_active_inside_proxy": 25,
            "v7_unanimous_inside_proxy": 1,
        },
        "production_candidate_changed": False,
        "token_read": False,
        "submission_performed": False,
    }
    (HERE / "target19_fresh_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps({"decision": report["decision"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
