"""Conditional shadow test of selective inverse AIA attacks.

The production feedback says three direct selectors scored zero on seven
targets.  This benchmark recreates that post-selection event in independent
shadow worlds, then asks whether using only the strongest part of the routed
selector's exact complement improves the normalized AIA score.  Discovery and
validation seeds are disjoint; production files are never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = ROOT / "local-output" / "attack-v6" / "shadow-benchmark"
for path in (ROOT, BENCH_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark as bench  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    age_band,
    build_aia_v5_variants,
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
)
from attack_phase_v6.aia_global_density import (  # noqa: E402
    build_global_density_horizon,
    build_sex_consensus_horizon,
    global_density_ratio_score,
    sex_copula_proximity_score,
)


OUT = Path(__file__).resolve().parent
DISCOVERY = tuple(range(20261301, 20261313))
VALIDATION = tuple(range(20261401, 20261413))
FRACTIONS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


def proportional_subset(
    challenge: pd.DataFrame,
    allowed: np.ndarray,
    score: np.ndarray,
    count: int,
    *,
    stratified: bool,
) -> np.ndarray:
    """Choose a deterministic nested subset, optionally balanced by public strata."""
    allowed = np.asarray(allowed, dtype=bool)
    score = np.asarray(score, dtype=float)
    ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    if allowed.shape != score.shape or allowed.shape != (len(challenge),):
        raise ValueError("selection arrays are not aligned")
    if count < 0 or count > int(allowed.sum()):
        raise ValueError("requested subset is larger than the allowed mask")
    selected = np.zeros(len(challenge), dtype=bool)
    if not stratified:
        rows = np.flatnonzero(allowed)
        order = rows[np.lexsort((ids[rows], -score[rows]))]
        selected[order[:count]] = True
        return selected

    work = pd.DataFrame(
        {
            "row": np.arange(len(challenge), dtype=int),
            "sex": challenge["sex"].to_numpy(),
            "prefecture": challenge["prefecture"].astype(str).to_numpy(),
            "age_band": age_band(challenge["age"]),
        }
    )
    groups: list[tuple[np.ndarray, int, float]] = []
    allowed_total = int(allowed.sum())
    floor_total = 0
    for group_index, indices in enumerate(
        work.groupby(["sex", "prefecture", "age_band"], sort=True).indices.values()
    ):
        rows = np.asarray(indices, dtype=int)
        rows = rows[allowed[rows]]
        if not len(rows):
            continue
        ideal = count * len(rows) / allowed_total
        quota = int(np.floor(ideal))
        floor_total += quota
        groups.append((rows, quota, ideal - quota))
    extras = count - floor_total
    # Largest-remainder allocation; the next score is only a deterministic tie-break.
    allocation_order = sorted(
        range(len(groups)),
        key=lambda index: (
            groups[index][2],
            float(np.max(score[groups[index][0]])),
            -int(np.min(ids[groups[index][0]])),
        ),
        reverse=True,
    )
    ceil_groups = set(allocation_order[:extras])
    for group_index, (rows, quota, _) in enumerate(groups):
        quota += int(group_index in ceil_groups)
        order = rows[np.lexsort((ids[rows], -score[rows]))]
        selected[order[:quota]] = True
    if int(selected.sum()) != count:
        raise RuntimeError("proportional selection produced wrong count")
    return selected


def predictions(mask: np.ndarray, *, active_time: float, abstain: float) -> np.ndarray:
    result = np.full(len(mask), abstain, dtype=float)
    result[mask] = active_time
    return result


def routed_continuous_score(
    official_id: str,
    internal_id: str,
    challenge: pd.DataFrame,
    release: pd.DataFrame,
    background: pd.DataFrame,
    v4_time: np.ndarray,
    independent_time: np.ndarray,
    *,
    abstain: float,
) -> np.ndarray:
    active_v4 = (~np.isclose(v4_time, abstain)).astype(float)
    active_independent = (~np.isclose(independent_time, abstain)).astype(float)
    kind = bench.ROUTE_KIND[official_id]
    if kind == "copula":
        return (
            1.10 * active_v4
            + 0.90 * active_independent
            + 0.20 * proximity_score(release, challenge)
            + 0.20 * copula_proximity_score(release, challenge)
        )
    if kind == "density_all5":
        return global_density_ratio_score(release, challenge, background)
    if kind == "density_stable3":
        return global_density_ratio_score(
            release, challenge, background, columns=bench.VIEWS["stable3"]
        )
    if kind == "density_stable3_invert":
        return -global_density_ratio_score(
            release, challenge, background, columns=bench.VIEWS["stable3"]
        )
    if kind == "sex_consensus":
        return (
            1.10 * active_v4
            + 0.90 * active_independent
            + 0.55 * sex_copula_proximity_score(release, challenge)
        )
    raise ValueError(f"unknown route kind {kind} for {internal_id}")


def run_seed(
    split: str,
    seed: int,
    *,
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    detector: dict[str, Any],
    spec: bench.shadow.ShadowSpec,
    cfg: dict[str, Any],
    horizon: float,
    abstain: float,
    censor_time: float,
) -> list[dict[str, Any]]:
    world = bench.shadow.build_shadow_world(
        candidates, b_self, background, detector, spec, seed
    )
    bench.exactify_visible_control_matching(world)
    bench.shadow.validate_world(world, spec)
    releases: dict[str, pd.DataFrame] = {}
    for ordinal, (target_id, source) in enumerate(world.aia_sources.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, ordinal, int(target_id), 9959])
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
    structured, _ = build_aia_v5_variants(
        baseline,
        independent,
        releases,
        world.challenges,
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_all5, _ = build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_stable3, _ = build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        columns=bench.VIEWS["stable3"],
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_stable3_invert, _ = build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        columns=bench.VIEWS["stable3"],
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
        invert=True,
    )
    sex_consensus, _ = build_sex_consensus_horizon(
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
    linkable, _ = filter_linkable_releases(releases)

    rows: list[dict[str, Any]] = []
    for ordinal, internal_id in enumerate(sorted(world.challenges, key=int)):
        official_id = bench.OFFICIAL_ZERO_IDS[ordinal]
        challenge = world.challenges[internal_id].reset_index(drop=True)
        truth = world.aia_truth[internal_id].reset_index(drop=True)
        v4_time = bench.aligned_time(baseline, internal_id, challenge)
        independent_time = bench.aligned_time(independent, internal_id, challenge)
        routed_time = bench.select_routed_times(
            official_id, internal_id, challenge, route_sources
        )
        direct = {
            "v4": bench.score_block(v4_time, truth, cfg)["power"],
            "independent": bench.score_block(independent_time, truth, cfg)["power"],
            "routed": bench.score_block(routed_time, truth, cfg)["power"],
        }
        exact_zero = max(direct.values()) <= 1e-12
        near_zero = max(direct.values()) <= 0.02
        direct_active = ~np.isclose(routed_time, abstain)
        v4_active = ~np.isclose(v4_time, abstain)
        independent_active = ~np.isclose(independent_time, abstain)
        complement = ~direct_active
        continuous = routed_continuous_score(
            official_id,
            internal_id,
            challenge,
            linkable[internal_id],
            world.attack_background,
            v4_time,
            independent_time,
            abstain=abstain,
        )
        expert_inverse = -(
            v4_active.astype(float)
            + independent_active.astype(float)
            + direct_active.astype(float)
        ) - 1e-3 * continuous

        candidates_for_block: dict[str, np.ndarray] = {
            "full_complement_h10": predictions(
                complement, active_time=horizon, abstain=abstain
            ),
            "full_complement_edge": predictions(
                complement, active_time=censor_time, abstain=abstain
            ),
            "all_rows_edge": np.full(len(challenge), censor_time, dtype=float),
        }
        vote_count = (
            v4_active.astype(int)
            + independent_active.astype(int)
            + direct_active.astype(int)
        )
        boolean_masks = {
            "complement_v4_edge": ~v4_active,
            "complement_independent_edge": ~independent_active,
            "complement_v4_and_route_edge": ~v4_active & ~direct_active,
            "complement_independent_and_route_edge": ~independent_active & ~direct_active,
            "complement_all_three_edge": vote_count == 0,
            "inverse_votes_at_most_one_edge": vote_count <= 1,
            "inverse_votes_at_most_two_edge": vote_count <= 2,
        }
        for name, mask in boolean_masks.items():
            candidates_for_block[name] = predictions(
                mask, active_time=censor_time, abstain=abstain
            )
        for pattern in range(8):
            mask = (
                (v4_active.astype(int) << 2)
                + (independent_active.astype(int) << 1)
                + direct_active.astype(int)
            ) == pattern
            candidates_for_block[f"expert_pattern_{pattern:03b}_edge"] = predictions(
                mask, active_time=censor_time, abstain=abstain
            )
        for fraction in FRACTIONS:
            count = int(round(fraction * len(challenge)))
            for balance in ("strat", "global"):
                stratified = balance == "strat"
                route_mask = proportional_subset(
                    challenge,
                    complement,
                    -continuous,
                    count,
                    stratified=stratified,
                )
                expert_mask = proportional_subset(
                    challenge,
                    complement,
                    expert_inverse,
                    count,
                    stratified=stratified,
                )
                pct = int(round(100 * fraction))
                candidates_for_block[f"route_q{pct:02d}_{balance}_edge"] = predictions(
                    route_mask, active_time=censor_time, abstain=abstain
                )
                candidates_for_block[f"expert_q{pct:02d}_{balance}_edge"] = predictions(
                    expert_mask, active_time=censor_time, abstain=abstain
                )

        for strategy, predicted in candidates_for_block.items():
            rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "official_target": official_id,
                    "internal_target": internal_id,
                    "strategy": strategy,
                    "active_rows": int(np.count_nonzero(~np.isclose(predicted, abstain))),
                    "all_three_exact_zero": exact_zero,
                    "all_three_at_most_0p02": near_zero,
                    **bench.score_block(predicted, truth, cfg),
                }
            )
    return rows


def summarize(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    chosen = frame.loc[frame[condition]].copy()
    keys = ["split", "seed", "official_target"]
    baseline = chosen.loc[
        chosen["strategy"].eq("full_complement_edge"), [*keys, "power"]
    ].rename(columns={"power": "baseline"})
    chosen = chosen.merge(baseline, on=keys, how="left")
    chosen["delta"] = chosen["power"] - chosen["baseline"]
    return (
        chosen.groupby(["split", "strategy"], as_index=False)
        .agg(
            eligible_blocks=("power", "size"),
            active_rows_mean=("active_rows", "mean"),
            mean=("power", "mean"),
            median=("power", "median"),
            p10=("power", lambda x: float(np.quantile(x, 0.10))),
            p90=("power", lambda x: float(np.quantile(x, 0.90))),
            mean_delta=("delta", "mean"),
            win_rate=("delta", lambda x: float(np.mean(x > 1e-12))),
            loss_rate=("delta", lambda x: float(np.mean(x < -1e-12))),
        )
        .sort_values(["split", "mean"], ascending=[True, False])
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
    censor_time = float(background["time"].max()) - float(cfg["aia"]["tau_time"])

    rows: list[dict[str, Any]] = []
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
                    censor_time=censor_time,
                )
            )
    frame = pd.DataFrame(rows)
    exact = summarize(frame, "all_three_exact_zero")
    near = summarize(frame, "all_three_at_most_0p02")
    discovery = exact.loc[exact["split"].eq("discovery")].sort_values(
        ["mean", "p10", "loss_rate"], ascending=[False, False, True]
    )
    selected = str(discovery.iloc[0]["strategy"])
    validation_selected = exact.loc[
        exact["split"].eq("validation") & exact["strategy"].eq(selected)
    ]
    frame.to_csv(OUT / "scores_long.csv", index=False)
    exact.to_csv(OUT / "exact_zero_summary.csv", index=False)
    near.to_csv(OUT / "near_zero_summary.csv", index=False)
    report = {
        "purpose": "conditional independent-shadow active-fraction selection; not a leaderboard forecast",
        "discovery_seeds": list(DISCOVERY),
        "validation_seeds": list(VALIDATION),
        "fractions": list(FRACTIONS),
        "censor_edge_time": censor_time,
        "selection_rule": "highest discovery mean under the exact-zero condition; validation untouched until lock",
        "selected_strategy": selected,
        "selected_validation": validation_selected.to_dict(orient="records"),
        "exact_zero_eligible_blocks": {
            split: int(group["eligible_blocks"].max())
            for split, group in exact.groupby("split")
        },
        "near_zero_eligible_blocks": {
            split: int(group["eligible_blocks"].max())
            for split, group in near.groupby("split")
        },
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nDISCOVERY TOP\n", discovery.head(15).to_string(index=False))
    print("\nLOCKED VALIDATION\n", validation_selected.to_string(index=False))


if __name__ == "__main__":
    main()
