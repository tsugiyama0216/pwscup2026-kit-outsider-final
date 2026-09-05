"""Weak-source and stress validation for target-3 q90 event additions.

The predeclared production rule keeps every v7 value unchanged except trusted
q90 copula event links that currently equal the normal-format abstention value.
This benchmark deliberately reduces source retention to address the observed
real-vs-shadow strength gap.  It never writes production files.
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
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    QI_COLUMNS,
    copula_event_links,
    filter_linkable_releases,
)
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


TEMPLATE_TARGET = 3
DISCOVERY_SEEDS = (20270421, 20270422)
VALIDATION_SEEDS = tuple(range(20270431, 20270439))
ABSTAIN = 11.0
EDGE = 9.5
HORIZON = 10.0
SCENARIOS: dict[str, dict[str, Any]] = {
    "full_clean": {"source_fraction": 1.0, "noise": 0.0, "shuffle_time": False},
    "weak50_clean": {"source_fraction": 0.50, "noise": 0.0, "shuffle_time": False},
    "weak50_noise": {"source_fraction": 0.50, "noise": 0.03, "shuffle_time": False},
    "weak25_clean": {"source_fraction": 0.25, "noise": 0.0, "shuffle_time": False},
    "weak25_noise": {"source_fraction": 0.25, "noise": 0.03, "shuffle_time": False},
    "weak25_time_shuffle": {
        "source_fraction": 0.25,
        "noise": 0.0,
        "shuffle_time": True,
    },
    "exchangeable": {"source_fraction": 0.0, "noise": 0.0, "shuffle_time": False},
}
PRESERVING = ("weak50_clean", "weak50_noise", "weak25_clean", "weak25_noise")
STRESS = ("weak25_time_shuffle", "exchangeable")
METHODS = ("q75_add_inactive", "q90_add_inactive")


def stable_rng(seed: int, internal_target: int, scenario_index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, internal_target, scenario_index, 3_090_071])
    )


def map_margin_nearest(
    source_values: np.ndarray, destination_values: np.ndarray
) -> np.ndarray:
    source = np.asarray(source_values, dtype=float)
    destination = np.asarray(destination_values, dtype=float)
    if not len(destination):
        return source.copy()
    order = np.argsort(source, kind="mergesort")
    quantiles = (np.arange(len(source), dtype=float) + 0.5) / len(source)
    mapped = np.quantile(destination, quantiles, method="nearest")
    restored = np.empty(len(source), dtype=float)
    restored[order] = mapped
    return restored


def mechanism_release(
    source: pd.DataFrame,
    public_background: pd.DataFrame,
    template: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    source_fraction: float,
    noise: float,
    shuffle_time: bool,
) -> pd.DataFrame:
    rows = len(source)
    source_count = int(round(source_fraction * rows))
    source_rows = (
        source.iloc[rng.choice(len(source), size=source_count, replace=False)]
        if source_count
        else source.iloc[:0]
    )
    decoy_count = rows - source_count
    decoy_rows = (
        public_background.iloc[
            rng.choice(
                len(public_background),
                size=decoy_count,
                replace=decoy_count > len(public_background),
            )
        ]
        if decoy_count
        else public_background.iloc[:0]
    )
    output = pd.concat([source_rows, decoy_rows], ignore_index=True).copy()

    # Replay a piecewise monotone public margin, retaining C3's exact support
    # rather than linearly interpolating between released values.
    for column in QI_COLUMNS:
        output[column] = map_margin_nearest(
            output[column].to_numpy(dtype=float),
            template[column].to_numpy(dtype=float),
        )

    event = (
        output["onset"].to_numpy(dtype=int)
        + output["death"].to_numpy(dtype=int)
    ) > 0
    template_event = (
        template["onset"].to_numpy(dtype=int)
        + template["death"].to_numpy(dtype=int)
    ) > 0
    if event.any() and template_event.any():
        output.loc[event, "time"] = map_margin_nearest(
            output.loc[event, "time"].to_numpy(dtype=float),
            template.loc[template_event, "time"].to_numpy(dtype=float),
        )
    if (~event).any():
        template_censored = template.loc[~template_event, "time"].to_numpy(dtype=float)
        if len(template_censored):
            output.loc[~event, "time"] = map_margin_nearest(
                output.loc[~event, "time"].to_numpy(dtype=float),
                template_censored,
            )

    if noise > 0.0:
        for column in ("SBP", "TG", "HDL", "ALT"):
            scale = max(float(template[column].std(ddof=0)), 1e-9)
            output[column] = output[column].to_numpy(dtype=float) + rng.normal(
                0.0, noise * scale, size=len(output)
            )
    if shuffle_time:
        permuted = rng.permutation(len(output))
        output.loc[:, ["time", "onset", "death"]] = output.loc[
            permuted, ["time", "onset", "death"]
        ].to_numpy()
    output = shadow.coerce_release(output, cfg)
    return output.iloc[rng.permutation(len(output))].reset_index(drop=True)


def make_releases(
    world: shadow.ShadowWorld,
    template: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    seed: int,
    scenario: str,
) -> dict[str, pd.DataFrame]:
    specification = SCENARIOS[scenario]
    scenario_index = list(SCENARIOS).index(scenario)
    output: dict[str, pd.DataFrame] = {}
    for internal_id, source in world.aia_sources.items():
        output[internal_id] = mechanism_release(
            source,
            world.attack_background,
            template,
            cfg,
            stable_rng(seed, int(internal_id), scenario_index),
            **specification,
        )
    return output


def edge_baseline(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[np.isclose(output, HORIZON)] = EDGE
    return output


def score(
    challenge: pd.DataFrame,
    truth: pd.DataFrame,
    prediction: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    result = core.target_result(challenge, truth, prediction, cfg)
    breakdown = result.breakdown
    return {
        "power": float(result.score),
        "all": float(breakdown["r_time_all"]),
        "rare": float(breakdown["r_time_rare"]),
        "p_m": float(breakdown["succ_time_member"]),
        "p_c": float(breakdown["succ_time_control"]),
    }


def aggregate(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {"subset": label, "cases": 0}
    seed_means = frame.groupby("seed")["delta"].mean()
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
        "mean_added_rows": float(frame["changed_rows"].mean()),
        "seed_mean_min": float(seed_means.min()),
        "seed_mean_win_rate": float(np.mean(seed_means > 1e-12)),
        "score1_mean_delta_if_divide_20": float(frame["delta"].mean() / 20.0),
    }


def main() -> None:
    dist = ROOT / "participant_data"
    targets_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    template = pd.read_csv(targets_dir / "C_3.csv")
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
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
                candidates, b_self, background, detector, spec, seed
            )
            exactify_visible_control_matching(world)
            shadow.validate_world(world, spec)
            for scenario in SCENARIOS:
                print(f"{split} seed={seed} scenario={scenario}", flush=True)
                releases = make_releases(
                    world, template, cfg, seed=seed, scenario=scenario
                )
                bases = positive.build_base_frames(
                    world,
                    releases,
                    world.attack_background,
                    cfg,
                    TEMPLATE_TARGET,
                )
                linkable, _ = filter_linkable_releases(releases)
                for internal_id, raw_base in bases.items():
                    challenge = world.challenges[internal_id].reset_index(drop=True)
                    truth = world.aia_truth[internal_id].reset_index(drop=True)
                    base = edge_baseline(raw_base)
                    base_metric = score(challenge, truth, base, cfg)
                    active = ~np.isclose(base, ABSTAIN)
                    for q, method in ((0.75, METHODS[0]), (0.90, METHODS[1])):
                        linked, link_gate, _ = copula_event_links(
                            linkable[internal_id],
                            challenge,
                            margin_quantile=q,
                            horizon_time=HORIZON,
                        )
                        add = (~active) & link_gate
                        proposed = base.copy()
                        proposed[add] = linked[add]
                        result = score(challenge, truth, proposed, cfg)
                        rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "scenario": scenario,
                                "internal_target": int(internal_id),
                                "method": method,
                                "base_power": base_metric["power"],
                                "base_all": base_metric["all"],
                                "base_rare": base_metric["rare"],
                                "power": result["power"],
                                "all": result["all"],
                                "rare": result["rare"],
                                "delta": result["power"] - base_metric["power"],
                                "delta_all": result["all"] - base_metric["all"],
                                "delta_rare": result["rare"] - base_metric["rare"],
                                "changed_rows": int(add.sum()),
                                "baseline_positive": base_metric["power"] > 1e-15,
                            }
                        )
    scores = pd.DataFrame(rows)
    scores.to_csv(HERE / "target3_q90_shadow_scores.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for split in ("discovery", "validation"):
        for method in METHODS:
            selected = scores.loc[
                scores["split"].eq(split) & scores["method"].eq(method)
            ]
            groups = {
                "preserving": selected.loc[selected["scenario"].isin(PRESERVING)],
                "preserving_baseline_positive": selected.loc[
                    selected["scenario"].isin(PRESERVING)
                    & selected["baseline_positive"]
                ],
                "weak25": selected.loc[
                    selected["scenario"].isin(("weak25_clean", "weak25_noise"))
                ],
                "time_shuffle": selected.loc[
                    selected["scenario"].eq("weak25_time_shuffle")
                ],
                "exchangeable": selected.loc[
                    selected["scenario"].eq("exchangeable")
                ],
                "full_clean": selected.loc[selected["scenario"].eq("full_clean")],
            }
            for label, frame in groups.items():
                item = aggregate(frame, label)
                item["split"] = split
                item["method"] = method
                summaries.append(item)
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "target3_q90_shadow_summary.csv", index=False)

    index = summary.set_index(["split", "method", "subset"])
    validation = index.loc[("validation", "q90_add_inactive", "preserving")]
    validation_positive = index.loc[
        ("validation", "q90_add_inactive", "preserving_baseline_positive")
    ]
    time_shuffle = index.loc[("validation", "q90_add_inactive", "time_shuffle")]
    exchangeable = index.loc[("validation", "q90_add_inactive", "exchangeable")]
    checks = {
        "validation_mean_gt_0": float(validation["mean_delta"]) > 0.0,
        "validation_win_rate_gte_0p60": float(validation["win_rate"]) >= 0.60,
        "validation_p10_gte_0": float(validation["p10_delta"]) >= -1e-12,
        "validation_all_mean_gte_0": float(validation["mean_delta_all"]) >= -1e-12,
        "validation_seed_min_gte_0": float(validation["seed_mean_min"]) >= -1e-12,
        "baseline_positive_cases_present": int(validation_positive["cases"]) > 0,
        "baseline_positive_mean_gte_0": float(validation_positive["mean_delta"]) >= -1e-12,
        "time_shuffle_mean_gte_0": float(time_shuffle["mean_delta"]) >= -1e-12,
        "exchangeable_mean_gte_0": float(exchangeable["mean_delta"]) >= -1e-12,
    }
    report = {
        "decision": "shadow_gate_pass" if all(checks.values()) else "reject",
        "predeclared_rule": (
            "target 3 only; keep every v7 value, set q90 trusted event links "
            "that currently equal 11.0 to their released donor time"
        ),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "preserving_scenarios": list(PRESERVING),
        "stress_scenarios": list(STRESS),
        "checks": checks,
        "validation_summaries": summary.loc[
            summary["split"].eq("validation")
            & summary["method"].eq("q90_add_inactive")
        ].to_dict(orient="records"),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (HERE / "target3_q90_shadow_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps({"decision": report["decision"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
