"""Paired shadow benchmark for AIA time decoders on measured-positive targets.

Research only.  This script keeps each production winner's membership mask fixed,
changes only active time predictions, and evaluates on disjoint discovery and
validation shadow seeds.  Production CSVs, tokens, archives, and submissions are
never written.
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
import evaluate_positive_shift as positive  # noqa: E402
from attack_phase import evaluate_shadow as shadow  # noqa: E402
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


OUT = HERE
TARGETS = positive.TARGETS
DISCOVERY_SEEDS = (20270101, 20270102)
VALIDATION_SEEDS = (20270111, 20270112)
EDGE = 9.498631074607
HORIZON = 10.0
ABSTAIN = 11.0
SCENARIOS = (
    "warp",
    "warp_noise_0p03",
    "warp_time_marginal",
    "warp_noise_time_marginal",
    "warp_time_shuffled",
)
PRESERVING_SCENARIOS = tuple(
    value for value in SCENARIOS if value != "warp_time_shuffled"
)


def empirical_quantile_map(
    values: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    """Monotone empirical-CDF map with deterministic midpoint ranks."""
    values = np.asarray(values, dtype=float)
    source = np.sort(np.asarray(source, dtype=float))
    destination = np.sort(np.asarray(destination, dtype=float))
    if not len(source) or not len(destination):
        return values.copy()
    left = np.searchsorted(source, values, side="left")
    right = np.searchsorted(source, values, side="right")
    quantile = (0.5 * (left + right) + 0.5) / (len(source) + 1.0)
    return np.quantile(destination, quantile, method="linear")


def warp_event_time_margin(
    releases: dict[str, pd.DataFrame], template: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Warp event-time margins to a production C while preserving row ranks."""
    template_event = (template["onset"].to_numpy(dtype=int) + template["death"].to_numpy(dtype=int)) > 0
    template_times = template.loc[template_event, "time"].to_numpy(dtype=float)
    output: dict[str, pd.DataFrame] = {}
    for target_id, frame in releases.items():
        changed = frame.copy()
        event = (changed["onset"].to_numpy(dtype=int) + changed["death"].to_numpy(dtype=int)) > 0
        original = changed.loc[event, "time"].to_numpy(dtype=float)
        if len(original) and len(template_times):
            order = np.argsort(original, kind="mergesort")
            q = (np.arange(len(original), dtype=float) + 0.5) / len(original)
            mapped = np.quantile(template_times, q, method="linear")
            restored = np.empty(len(original), dtype=float)
            restored[order] = mapped
            changed.loc[event, "time"] = restored
        output[target_id] = changed
    return output


def make_releases(
    helper: Any,
    world: shadow.ShadowWorld,
    scenario: str,
    *,
    seed: int,
    cfg: dict[str, Any],
    template: pd.DataFrame,
    noise_scales: np.ndarray,
) -> dict[str, pd.DataFrame]:
    if scenario == "warp":
        base = "warp"
    elif scenario == "warp_noise_0p03":
        base = "warp_noise_0p03"
    elif scenario == "warp_time_shuffled":
        base = "warp_time_shuffled"
    elif scenario == "warp_time_marginal":
        base = "warp"
    elif scenario == "warp_noise_time_marginal":
        base = "warp_noise_0p03"
    else:
        raise ValueError(scenario)
    releases = helper(
        world,
        base,
        seed=seed,
        cfg=cfg,
        template=template,
        noise_scales=noise_scales,
    )
    if scenario.endswith("time_marginal"):
        releases = warp_event_time_margin(releases, template)
    return releases


def event_reference(frame: pd.DataFrame) -> np.ndarray:
    event = (frame["onset"].to_numpy(dtype=int) + frame["death"].to_numpy(dtype=int)) > 0
    return frame.loc[event, "time"].to_numpy(dtype=float)


def base_edge(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    output[np.isclose(output, HORIZON)] = EDGE
    return output


def decode_variants(
    base_values: np.ndarray,
    challenge: pd.DataFrame,
    released: pd.DataFrame,
    reference_event_times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build fixed-mask decoder variants; all non-active cells remain 11."""
    base_values = np.asarray(base_values, dtype=float)
    active = ~np.isclose(base_values, ABSTAIN)
    event_prediction = active & (base_values < EDGE - 1e-12)
    output: dict[str, np.ndarray] = {"current_edge": base_edge(base_values)}

    all_edge = np.full(len(base_values), ABSTAIN, dtype=float)
    all_edge[active] = EDGE
    output["all_edge"] = all_edge

    source_event_times = event_reference(released)
    if event_prediction.any() and len(source_event_times):
        mapped = empirical_quantile_map(
            base_values[event_prediction], source_event_times, reference_event_times
        )
        for original_weight in (0.0, 0.25, 0.5, 0.75):
            values = output["current_edge"].copy()
            values[event_prediction] = (
                original_weight * base_values[event_prediction]
                + (1.0 - original_weight) * mapped
            )
            label = str(original_weight).replace(".", "p")
            output[f"event_qmap_original_w{label}"] = values

        for shift in (-0.5, -0.25, 0.25, 0.5):
            values = output["current_edge"].copy()
            values[event_prediction] = np.clip(
                base_values[event_prediction] + shift, 0.0, HORIZON
            )
            label = str(shift).replace("-", "m").replace(".", "p")
            output[f"event_shift_{label}"] = values

    for margin_quantile in (0.50, 0.75, 0.90):
        linked_time, linked_gate, _ = copula_event_links(
            released,
            challenge,
            margin_quantile=margin_quantile,
            horizon_time=HORIZON,
        )
        gate = active & linked_gate
        direct = output["current_edge"].copy()
        direct[gate] = linked_time[gate]
        qlabel = str(margin_quantile).replace(".", "p")
        output[f"copula_q{qlabel}_direct"] = direct
        if gate.any() and len(source_event_times):
            mapped_link = empirical_quantile_map(
                linked_time[gate], source_event_times, reference_event_times
            )
            mapped_values = output["current_edge"].copy()
            mapped_values[gate] = mapped_link
            output[f"copula_q{qlabel}_qmap"] = mapped_values
    return output


def target_score(
    challenge: pd.DataFrame,
    truth: pd.DataFrame,
    values: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    result = core.target_result(challenge, truth, values, cfg)
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


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    baseline = scores.loc[
        scores["decoder"].eq("current_edge"),
        ["split", "seed", "scenario", "template_target", "shadow_target", "power", "all", "rare"],
    ].rename(
        columns={"power": "base_power", "all": "base_all", "rare": "base_rare"}
    )
    merged = scores.merge(
        baseline,
        on=["split", "seed", "scenario", "template_target", "shadow_target"],
        how="left",
        validate="many_to_one",
    )
    merged["delta"] = merged["power"] - merged["base_power"]
    merged["delta_all"] = merged["all"] - merged["base_all"]
    merged["delta_rare"] = merged["rare"] - merged["base_rare"]
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    targets_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background_full = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target: pd.read_csv(targets_dir / f"C_{target}.csv") for target in TARGETS
    }
    noise_scales = background_full.loc[:, list(core.VISIBLE_NUMERIC)].std(
        ddof=0
    ).to_numpy(dtype=float)
    helper = load_zero_helpers(ROOT)["scenario_releases"]
    reference_event_times = np.concatenate(
        [event_reference(b_self), event_reference(background_full)]
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
            background = world.attack_background
            for template_target in TARGETS:
                for scenario in SCENARIOS:
                    print(
                        f"{split} seed={seed} C{template_target} {scenario}",
                        flush=True,
                    )
                    releases = make_releases(
                        helper,
                        world,
                        scenario,
                        seed=seed,
                        cfg=cfg,
                        template=templates[template_target],
                        noise_scales=noise_scales,
                    )
                    frames = positive.build_base_frames(
                        world, releases, background, cfg, template_target
                    )
                    linkable, _ = filter_linkable_releases(releases)
                    for target_id, base_values in frames.items():
                        challenge = world.challenges[target_id].reset_index(drop=True)
                        truth = world.aia_truth[target_id].reset_index(drop=True)
                        variants = decode_variants(
                            base_values,
                            challenge,
                            linkable[target_id],
                            reference_event_times,
                        )
                        for decoder, values in variants.items():
                            rows.append(
                                {
                                    "split": split,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "template_target": template_target,
                                    "shadow_target": int(target_id),
                                    "source": (
                                        "v4"
                                        if template_target in positive.V4_TARGETS
                                        else "independent"
                                        if template_target in positive.INDEPENDENT_TARGETS
                                        else "routed"
                                    ),
                                    "decoder": decoder,
                                    "active_rows": int(
                                        np.count_nonzero(~np.isclose(base_values, ABSTAIN))
                                    ),
                                    "original_event_rows": int(
                                        np.count_nonzero(
                                            (~np.isclose(base_values, ABSTAIN))
                                            & (base_values < EDGE - 1e-12)
                                        )
                                    ),
                                    **target_score(challenge, truth, values, cfg),
                                }
                            )
    scores = pd.DataFrame(rows)
    scored = summarize(scores)
    scores.to_csv(OUT / "positive_decoder_scores.csv", index=False)
    scored.to_csv(OUT / "positive_decoder_deltas.csv", index=False)

    primary = scored.loc[scored["scenario"].isin(PRESERVING_SCENARIOS)].copy()
    summary = (
        primary.groupby(["split", "decoder"], as_index=False)
        .agg(
            cases=("delta", "size"),
            power_mean=("power", "mean"),
            delta_mean=("delta", "mean"),
            delta_median=("delta", "median"),
            delta_p10=("delta", lambda x: float(np.quantile(x, 0.10))),
            delta_min=("delta", "min"),
            negative_rate=("delta", lambda x: float(np.mean(x < -1e-15))),
            positive_rate=("delta", lambda x: float(np.mean(x > 1e-15))),
            delta_all_mean=("delta_all", "mean"),
            delta_rare_mean=("delta_rare", "mean"),
        )
        .sort_values(["split", "delta_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT / "positive_decoder_summary.csv", index=False)
    by_target = (
        primary.groupby(["split", "template_target", "decoder"], as_index=False)
        .agg(
            cases=("delta", "size"),
            delta_mean=("delta", "mean"),
            delta_p10=("delta", lambda x: float(np.quantile(x, 0.10))),
            delta_min=("delta", "min"),
            negative_rate=("delta", lambda x: float(np.mean(x < -1e-15))),
            delta_rare_mean=("delta_rare", "mean"),
        )
        .sort_values(
            ["split", "template_target", "delta_mean"],
            ascending=[True, True, False],
        )
    )
    by_target.to_csv(OUT / "positive_decoder_by_target.csv", index=False)
    stress = (
        scored.loc[scored["scenario"].eq("warp_time_shuffled")]
        .groupby(["split", "decoder"], as_index=False)
        .agg(
            cases=("delta", "size"),
            delta_mean=("delta", "mean"),
            delta_p10=("delta", lambda x: float(np.quantile(x, 0.10))),
            delta_min=("delta", "min"),
            negative_rate=("delta", lambda x: float(np.mean(x < -1e-15))),
        )
        .sort_values(["split", "delta_mean"], ascending=[True, False])
    )
    stress.to_csv(OUT / "positive_decoder_stress.csv", index=False)

    report = {
        "scope": "authorized local shadow evaluation; no production candidate written",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "positive_templates": list(TARGETS),
        "membership_selection": "fixed to the official per-target winner source",
        "current_baseline": "event predictions preserved; exact 10 shifted to 9.498631074607",
        "primary_scenarios": list(PRESERVING_SCENARIOS),
        "stress_scenario": "warp_time_shuffled",
        "validation_ranking": summary.loc[summary["split"].eq("validation")]
        .head(12)
        .to_dict(orient="records"),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "submission_performed": False,
    }
    (OUT / "positive_decoder_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
