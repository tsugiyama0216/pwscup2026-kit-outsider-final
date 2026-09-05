"""Truth-known benchmark for final zero-target AIA strategies.

Research-only: all outputs remain beneath ``local-output``.  The benchmark
maps seven independently generated shadow targets to the seven production
targets that scored zero in every measured source.  It preserves the official
challenge construction (163 members + 163 controls, exact matching on
sex/prefecture/five-year age band, and exact onset/rare composition) and
compares alternative selectors using only public-style inputs.

The learned logistic mixture is tuned with leave-one-seed-out discovery
predictions and is evaluated once on disjoint validation seeds.  No production
candidate, token, archive, or remote service is touched.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    build_aia_v5_variants,
    filter_linkable_releases,
    strict_rank,
    stratified_half_selection,
)
from attack_phase_v5.evaluate_shadow_v5 import exactify_visible_control_matching  # noqa: E402
from attack_phase_v6.aia_global_density import (  # noqa: E402
    build_global_density_horizon,
    build_sex_consensus_horizon,
    global_density_ratio_score,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_aia  # noqa: E402


OUT = ROOT / "local-output" / "attack-v6" / "shadow-benchmark"
DIST = ROOT / "participant_data"
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
OFFICIAL_ZERO_IDS = ("6", "7", "8", "9", "17", "18", "19")
DISCOVERY_SEEDS = (20261101, 20261102, 20261103)
VALIDATION_SEEDS = (20261111, 20261112, 20261113)
SCENARIOS = ("raw", "noise_0p03", "time_shuffled", "warp", "independent")
PLAUSIBLE_SCENARIOS = tuple(value for value in SCENARIOS if value != "independent")
QI = ("age", "SBP", "TG", "HDL", "ALT")
VIEWS = {
    "all5": QI,
    "nohdl": ("age", "SBP", "TG", "ALT"),
    "stable3": ("age", "SBP", "TG"),
    "labs": ("SBP", "TG", "ALT"),
}
TREE_VIEWS = ("all5", "stable3")
LOG_COLUMNS = frozenset(("TG", "ALT"))
ROUTE_KIND = {
    "6": "copula",
    "7": "density_all5",
    "8": "copula",
    "9": "density_stable3",
    "17": "copula",
    "18": "sex_consensus",
    "19": "density_stable3_invert",
}
LOCKED_POSITIVE_SUM = 0.15062041254150674 * 20.0


def rank_warp(
    source: pd.DataFrame,
    template: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Warp public QI margins to a production release without using truth."""
    output = source.copy()
    quantile = (np.arange(len(output), dtype=float) + 0.5) / len(output)
    for column in QI:
        order = np.argsort(output[column].to_numpy(dtype=float), kind="mergesort")
        destination = np.quantile(
            template[column].to_numpy(dtype=float), quantile, method="linear"
        )
        warped = np.empty(len(output), dtype=float)
        warped[order] = destination
        output[column] = warped
    output = shadow.coerce_release(output, cfg)
    return output.iloc[rng.permutation(len(output))].reset_index(drop=True)


def scenario_releases(
    world: shadow.ShadowWorld,
    scenario: str,
    *,
    seed: int,
    scenario_index: int,
    cfg: dict[str, Any],
    templates: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for ordinal, (internal_id, source) in enumerate(world.aia_sources.items()):
        official_id = OFFICIAL_ZERO_IDS[ordinal]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, scenario_index, int(internal_id), int(official_id), 7301]
            )
        )
        if scenario == "warp":
            output[internal_id] = rank_warp(
                source, templates[official_id], cfg, rng
            )
        else:
            output[internal_id] = shadow.scenario_release(
                source, world.attack_background, scenario, cfg, rng
            )
    return output


def aligned_time(
    frame: pd.DataFrame, target_id: str, challenge: pd.DataFrame
) -> np.ndarray:
    ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    block = frame.loc[
        frame["target_k"].astype(int).eq(int(target_id)),
        ["challenge_row_id", "time_hat"],
    ].copy()
    block.index = block["challenge_row_id"].astype(np.int64)
    if set(block.index) != set(ids):
        raise RuntimeError(f"prediction keys differ for shadow target {target_id}")
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def horizon_decode(
    challenge: pd.DataFrame,
    score: np.ndarray,
    *,
    horizon: float,
    abstain: float,
) -> np.ndarray:
    selected, _ = stratified_half_selection(challenge, score, member_count=163)
    output = np.full(len(challenge), abstain, dtype=float)
    output[selected] = horizon
    return output


def scaled_matrix(
    frame: pd.DataFrame,
    background: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    reference = background.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
            reference[:, index] = np.log1p(
                np.clip(reference[:, index], 0.0, None)
            )
    center = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-9, scale, np.std(reference, axis=0, ddof=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - center) / scale / math.sqrt(len(columns))


def extra_trees_rank(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    seed: int,
) -> np.ndarray:
    release_x = scaled_matrix(released, background, columns)
    background_x = scaled_matrix(background, background, columns)
    challenge_x = scaled_matrix(challenge, background, columns)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(background_x), size=min(len(background_x), 2 * len(release_x)), replace=False
    )
    x = np.vstack((release_x, background_x[chosen]))
    y = np.r_[np.ones(len(release_x), dtype=int), np.zeros(len(chosen), dtype=int)]
    model = ExtraTreesClassifier(
        n_estimators=64,
        min_samples_leaf=12,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)
    probability = model.predict_proba(challenge_x)[:, 1]
    return strict_rank(
        probability, challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    )


def knn_rank(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    # Keep a local implementation so all four views use identical scaling and
    # are available as features; it reconciles numerically with v6's helper.
    release_x = scaled_matrix(released, background, columns)
    background_x = scaled_matrix(background, background, columns)
    challenge_x = scaled_matrix(challenge, background, columns)
    release_nn = NearestNeighbors(n_neighbors=8, n_jobs=1).fit(release_x)
    background_nn = NearestNeighbors(n_neighbors=8, n_jobs=1).fit(background_x)
    rd = release_nn.kneighbors(challenge_x, return_distance=True)[0]
    bd = background_nn.kneighbors(challenge_x, return_distance=True)[0]
    evidence = np.mean(
        np.column_stack(
            [
                len(columns)
                * np.log((bd[:, value - 1] + 1e-6) / (rd[:, value - 1] + 1e-6))
                for value in (1, 3, 8)
            ]
        ),
        axis=1,
    )
    return strict_rank(
        evidence, challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    )


def score_block(
    predicted: np.ndarray,
    truth: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, float]:
    member = truth["label"].eq("member").to_numpy()
    control = ~member
    rare = truth["is_rare"].astype(bool).to_numpy()
    result = score_aia(
        None,
        predicted[member],
        truth.loc[member, "onset"].to_numpy(dtype=int),
        truth.loc[member, "time"].to_numpy(dtype=float),
        None,
        predicted[control],
        truth.loc[control, "onset"].to_numpy(dtype=int),
        truth.loc[control, "time"].to_numpy(dtype=float),
        cfg,
        member_rare_mask=rare[member],
        control_rare_mask=rare[control],
    )
    breakdown = result.breakdown
    return {
        "power": float(result.score),
        "all": float(breakdown["r_time_all"]),
        "rare": float(breakdown["r_time_rare"]),
        "p_m": float(breakdown["succ_time_member"]),
        "p_c": float(breakdown["succ_time_control"]),
        "p_m_rare": float(breakdown["succ_time_member_rare"]),
        "p_c_rare": float(breakdown["succ_time_control_rare"]),
    }


def select_routed_times(
    official_id: str,
    internal_id: str,
    challenge: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
) -> np.ndarray:
    kind = ROUTE_KIND[official_id]
    source_name = {
        "copula": "copula",
        "density_all5": "density_all5",
        "density_stable3": "density_stable3",
        "density_stable3_invert": "density_stable3_invert",
        "sex_consensus": "sex_consensus",
    }[kind]
    return aligned_time(sources[source_name], internal_id, challenge)


def build_run_records(
    *,
    split: str,
    seed: int,
    scenario: str,
    scenario_index: int,
    world: shadow.ShadowWorld,
    releases: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    horizon: float,
    abstain: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            world.attack_background,
            horizon=horizon,
            tau=float(cfg["aia"]["tau_time"]),
        ),
        anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
        anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
        integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    independent, _, _ = build_aia_independent(
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
        columns=VIEWS["stable3"],
        member_count=163,
        abstain_time=abstain,
        horizon_time=horizon,
    )
    density_stable3_invert, _ = build_global_density_horizon(
        releases,
        world.challenges,
        world.attack_background,
        columns=VIEWS["stable3"],
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
    score_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for ordinal, internal_id in enumerate(sorted(world.challenges, key=int)):
        official_id = OFFICIAL_ZERO_IDS[ordinal]
        challenge = world.challenges[internal_id].reset_index(drop=True)
        truth = world.aia_truth[internal_id].reset_index(drop=True)
        if not challenge["challenge_row_id"].equals(truth["challenge_row_id"]):
            raise RuntimeError("challenge and truth rows are not aligned")
        ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        v4_time = aligned_time(baseline, internal_id, challenge)
        independent_time = aligned_time(independent, internal_id, challenge)
        routed_time = select_routed_times(
            official_id, internal_id, challenge, route_sources
        )
        if not np.isin(routed_time, [horizon, abstain]).all():
            raise RuntimeError("routed zero-target frame is not binary")
        active_v4 = ~np.isclose(v4_time, abstain)
        active_independent = ~np.isclose(independent_time, abstain)
        active_routed = ~np.isclose(routed_time, abstain)

        feature_map: dict[str, np.ndarray] = {
            "expert_v4": active_v4.astype(float),
            "expert_independent": active_independent.astype(float),
            "expert_routed": active_routed.astype(float),
        }
        for view, columns in VIEWS.items():
            feature_map[f"knn_{view}"] = knn_rank(
                linkable[internal_id],
                challenge,
                world.attack_background,
                columns,
            )
        for view in TREE_VIEWS:
            model_seed = int(
                np.random.SeedSequence(
                    [seed, scenario_index, int(internal_id), int(official_id), 991]
                ).generate_state(1)[0]
            )
            feature_map[f"trees_{view}"] = extra_trees_rank(
                linkable[internal_id],
                challenge,
                world.attack_background,
                VIEWS[view],
                seed=model_seed,
            )
        feature_map["expert_sum"] = (
            feature_map["expert_v4"]
            + feature_map["expert_independent"]
            + feature_map["expert_routed"]
        ) / 3.0
        feature_map["density_mean"] = np.mean(
            np.column_stack(
                [
                    feature_map[f"knn_{view}"] for view in VIEWS
                ]
                + [feature_map[f"trees_{view}"] for view in TREE_VIEWS]
            ),
            axis=1,
        )

        strategies: dict[str, np.ndarray] = {
            "safe_v4": v4_time,
            "routed_probe": routed_time,
            "exact_complement_routed": 21.0 - routed_time,
            "inverse_consensus": horizon_decode(
                challenge,
                -feature_map["expert_sum"],
                horizon=horizon,
                abstain=abstain,
            ),
        }
        for view in VIEWS:
            direct = feature_map[f"knn_{view}"]
            strategies[f"knn_{view}_direct"] = horizon_decode(
                challenge, direct, horizon=horizon, abstain=abstain
            )
            strategies[f"knn_{view}_inverse"] = horizon_decode(
                challenge, -direct, horizon=horizon, abstain=abstain
            )
        for view in TREE_VIEWS:
            direct = feature_map[f"trees_{view}"]
            strategies[f"trees_{view}_direct"] = horizon_decode(
                challenge, direct, horizon=horizon, abstain=abstain
            )
            strategies[f"trees_{view}_inverse"] = horizon_decode(
                challenge, -direct, horizon=horizon, abstain=abstain
            )
        strategies["density_mean_direct"] = horizon_decode(
            challenge,
            feature_map["density_mean"],
            horizon=horizon,
            abstain=abstain,
        )
        strategies["density_mean_inverse"] = horizon_decode(
            challenge,
            -feature_map["density_mean"],
            horizon=horizon,
            abstain=abstain,
        )
        strategies["inverse_expert_density_mix"] = horizon_decode(
            challenge,
            -0.65 * feature_map["expert_sum"]
            + 0.35 * feature_map["density_mean"],
            horizon=horizon,
            abstain=abstain,
        )

        for strategy, predicted in strategies.items():
            metrics = score_block(predicted, truth, cfg)
            score_rows.append(
                {
                    "split": split,
                    "seed": seed,
                    "scenario": scenario,
                    "internal_target": internal_id,
                    "official_target": official_id,
                    "strategy": strategy,
                    **metrics,
                }
            )
        feature_names = tuple(feature_map)
        records.append(
            {
                "split": split,
                "seed": seed,
                "scenario": scenario,
                "internal_target": internal_id,
                "official_target": official_id,
                "challenge": challenge,
                "truth": truth,
                "feature_names": feature_names,
                "x": np.column_stack([feature_map[name] for name in feature_names]),
                "y": truth["label"].eq("member").astype(int).to_numpy(),
                "ids": ids,
            }
        )
    return score_rows, records


def fit_logit(
    records: list[dict[str, Any]],
    seeds: set[int],
    *,
    c_value: float,
) -> LogisticRegression:
    chosen = [
        record
        for record in records
        if record["split"] == "discovery"
        and record["scenario"] in PLAUSIBLE_SCENARIOS
        and record["seed"] in seeds
    ]
    if not chosen:
        raise RuntimeError("empty calibration training set")
    x = np.vstack([record["x"] for record in chosen])
    y = np.concatenate([record["y"] for record in chosen])
    return LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        random_state=20260905,
    ).fit(x, y)


def score_calibrated_record(
    record: dict[str, Any],
    model: LogisticRegression,
    cfg: dict[str, Any],
    *,
    horizon: float,
    abstain: float,
) -> dict[str, Any]:
    probability = model.predict_proba(record["x"])[:, 1]
    predicted = horizon_decode(
        record["challenge"], probability, horizon=horizon, abstain=abstain
    )
    return {
        "split": record["split"],
        "seed": record["seed"],
        "scenario": record["scenario"],
        "internal_target": record["internal_target"],
        "official_target": record["official_target"],
        "strategy": "calibrated_logit",
        **score_block(predicted, record["truth"], cfg),
    }


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(["split", "seed", "scenario", "strategy"], as_index=False)
        .agg(
            zero_slice_power=("power", "mean"),
            all=("all", "mean"),
            rare=("rare", "mean"),
            p_m=("p_m", "mean"),
            p_c=("p_c", "mean"),
        )
        .assign(
            directional_whole_aia=lambda frame: (
                LOCKED_POSITIVE_SUM + 7.0 * frame["zero_slice_power"]
            )
            / 20.0
        )
    )


def distribution_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    plausible = aggregate.loc[aggregate["scenario"].isin(PLAUSIBLE_SCENARIOS)].copy()
    exact = plausible.loc[
        plausible["strategy"].eq("exact_complement_routed"),
        ["split", "seed", "scenario", "zero_slice_power"],
    ].rename(columns={"zero_slice_power": "exact_power"})
    plausible = plausible.merge(exact, on=["split", "seed", "scenario"], how="left")
    plausible["delta_vs_exact"] = plausible["zero_slice_power"] - plausible["exact_power"]
    return (
        plausible.groupby(["split", "strategy"], as_index=False)
        .agg(
            observations=("zero_slice_power", "size"),
            mean=("zero_slice_power", "mean"),
            median=("zero_slice_power", "median"),
            std=("zero_slice_power", "std"),
            p10=("zero_slice_power", lambda value: float(np.quantile(value, 0.10))),
            p90=("zero_slice_power", lambda value: float(np.quantile(value, 0.90))),
            minimum=("zero_slice_power", "min"),
            maximum=("zero_slice_power", "max"),
            mean_delta_vs_exact=("delta_vs_exact", "mean"),
            win_rate_vs_exact=("delta_vs_exact", lambda value: float(np.mean(value > 1e-12))),
        )
        .sort_values(["split", "mean"], ascending=[True, False])
    )


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(DIST / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(DIST / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(DIST)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target: pd.read_csv(TARGETS / f"C_{target}.csv")
        for target in OFFICIAL_ZERO_IDS
    }
    spec = shadow.ShadowSpec(
        n_targets=len(OFFICIAL_ZERO_IDS),
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=25 * len(OFFICIAL_ZERO_IDS),
        minimum_shared=25,
        maximum_shared=26,
    )
    shadow.validate_source_data(candidates, b_self, background, spec)
    horizon = float(cfg["onset"]["horizon_years"])
    abstain = v2.safe_abstain_time(horizon, float(cfg["aia"]["tau_time"]))

    score_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    world_checks: list[dict[str, Any]] = []
    for split, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("validation", VALIDATION_SEEDS),
    ):
        for seed in seeds:
            print(f"{split} seed={seed}: build exact-matched shadow world", flush=True)
            world = shadow.build_shadow_world(
                candidates, b_self, background, detector, spec, seed
            )
            exactify_visible_control_matching(world)
            check = shadow.validate_world(world, spec)
            world_checks.append({"split": split, "seed": seed, **check})
            for scenario_index, scenario in enumerate(SCENARIOS):
                print(f"  scenario={scenario}", flush=True)
                releases = scenario_releases(
                    world,
                    scenario,
                    seed=seed,
                    scenario_index=scenario_index,
                    cfg=cfg,
                    templates=templates,
                )
                rows, run_records = build_run_records(
                    split=split,
                    seed=seed,
                    scenario=scenario,
                    scenario_index=scenario_index,
                    world=world,
                    releases=releases,
                    cfg=cfg,
                    horizon=horizon,
                    abstain=abstain,
                )
                score_rows.extend(rows)
                records.extend(run_records)

    # Tune C only with seed-held-out discovery predictions.
    c_grid = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    cv_rows: list[dict[str, Any]] = []
    discovery_seed_set = set(DISCOVERY_SEEDS)
    for c_value in c_grid:
        candidate_rows: list[dict[str, Any]] = []
        for held_seed in DISCOVERY_SEEDS:
            model = fit_logit(
                records, discovery_seed_set - {held_seed}, c_value=c_value
            )
            held_records = [
                record
                for record in records
                if record["split"] == "discovery"
                and record["seed"] == held_seed
                and record["scenario"] in PLAUSIBLE_SCENARIOS
            ]
            candidate_rows.extend(
                score_calibrated_record(
                    record,
                    model,
                    cfg,
                    horizon=horizon,
                    abstain=abstain,
                )
                for record in held_records
            )
        candidate = pd.DataFrame(candidate_rows)
        aggregate = aggregate_scores(candidate)
        cv_rows.append(
            {
                "C": c_value,
                "mean_zero_slice_power": float(aggregate["zero_slice_power"].mean()),
                "minimum_zero_slice_power": float(aggregate["zero_slice_power"].min()),
                "std_zero_slice_power": float(aggregate["zero_slice_power"].std()),
            }
        )
    cv = pd.DataFrame(cv_rows).sort_values(
        ["mean_zero_slice_power", "minimum_zero_slice_power", "C"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    selected_c = float(cv.iloc[0]["C"])

    # Discovery scores stay out-of-fold at the chosen C; validation uses a
    # single model fitted on every plausible discovery record.
    for held_seed in DISCOVERY_SEEDS:
        fold_model = fit_logit(
            records, discovery_seed_set - {held_seed}, c_value=selected_c
        )
        for record in records:
            if record["split"] == "discovery" and record["seed"] == held_seed:
                score_rows.append(
                    score_calibrated_record(
                        record,
                        fold_model,
                        cfg,
                        horizon=horizon,
                        abstain=abstain,
                    )
                )
    final_model = fit_logit(records, discovery_seed_set, c_value=selected_c)
    for record in records:
        if record["split"] == "validation":
            score_rows.append(
                score_calibrated_record(
                    record,
                    final_model,
                    cfg,
                    horizon=horizon,
                    abstain=abstain,
                )
            )

    scores = pd.DataFrame(score_rows)
    aggregate = aggregate_scores(scores)
    distribution = distribution_summary(aggregate)
    per_target = (
        scores.loc[scores["scenario"].isin(PLAUSIBLE_SCENARIOS)]
        .groupby(["split", "official_target", "strategy"], as_index=False)
        .agg(
            observations=("power", "size"),
            mean=("power", "mean"),
            median=("power", "median"),
            p10=("power", lambda value: float(np.quantile(value, 0.10))),
            p90=("power", lambda value: float(np.quantile(value, 0.90))),
            zero_rate=("power", lambda value: float(np.mean(np.isclose(value, 0.0)))),
        )
        .sort_values(
            ["split", "official_target", "mean"], ascending=[True, True, False]
        )
    )
    scenario_summary = (
        aggregate.groupby(["split", "scenario", "strategy"], as_index=False)
        .agg(
            observations=("zero_slice_power", "size"),
            mean=("zero_slice_power", "mean"),
            minimum=("zero_slice_power", "min"),
            maximum=("zero_slice_power", "max"),
        )
        .sort_values(["split", "scenario", "mean"], ascending=[True, True, False])
    )

    scores.to_csv(OUT / "scores_long.csv", index=False)
    aggregate.to_csv(OUT / "aggregate_by_world.csv", index=False)
    distribution.to_csv(OUT / "distribution_summary.csv", index=False)
    per_target.to_csv(OUT / "per_target_summary.csv", index=False)
    scenario_summary.to_csv(OUT / "scenario_summary.csv", index=False)
    cv.to_csv(OUT / "calibration_cv.csv", index=False)
    pd.DataFrame(world_checks).to_csv(OUT / "world_checks.csv", index=False)

    validation = distribution.loc[distribution["split"].eq("validation")].copy()
    best = validation.sort_values("mean", ascending=False).iloc[0]
    exact = validation.loc[
        validation["strategy"].eq("exact_complement_routed")
    ].iloc[0]
    coefficients = {
        name: float(value)
        for name, value in zip(records[0]["feature_names"], final_model.coef_[0], strict=True)
    }
    summary = {
        "purpose": "truth-known local method comparison; not a leaderboard forecast",
        "as_of": "2026-09-05 Asia/Tokyo",
        "scope": {
            "official_zero_targets": list(OFFICIAL_ZERO_IDS),
            "discovery_seeds": list(DISCOVERY_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "scenarios": list(SCENARIOS),
            "plausible_scenarios": list(PLAUSIBLE_SCENARIOS),
            "rows_per_challenge": 326,
            "members_per_challenge": 163,
            "controls_per_challenge": 163,
            "exact_matching": [
                "sex",
                "prefecture",
                "five_year_age_band",
                "onset",
                "rare",
            ],
        },
        "calibration": {
            "method": "L2 logistic regression; C selected by leave-one-discovery-seed-out AIA power",
            "selected_C": selected_c,
            "intercept": float(final_model.intercept_[0]),
            "coefficients": coefficients,
            "cv": cv.to_dict(orient="records"),
        },
        "validation_headline": {
            "best_strategy": str(best["strategy"]),
            "best_zero_slice_mean": float(best["mean"]),
            "best_mean_delta_vs_exact": float(best["mean_delta_vs_exact"]),
            "best_win_rate_vs_exact": float(best["win_rate_vs_exact"]),
            "exact_complement_zero_slice_mean": float(exact["mean"]),
            "exact_complement_p10": float(exact["p10"]),
            "exact_complement_p90": float(exact["p90"]),
            "best_directional_whole_aia": float(
                (LOCKED_POSITIVE_SUM + 7.0 * float(best["mean"])) / 20.0
            ),
            "exact_directional_whole_aia": float(
                (LOCKED_POSITIVE_SUM + 7.0 * float(exact["mean"])) / 20.0
            ),
        },
        "limitations": [
            "Shadow scores compare methods but are not estimates of the private leaderboard score.",
            "The seven official target identifiers are mapped to independently generated shadow targets; target-specific differences are therefore stability checks, not reconstructed private truth.",
            "Warp preserves rank/outcome linkage while matching each public production release's QI margins; it cannot reconstruct the organizer's undisclosed transformation.",
            "Independent is a null control and is excluded from mixture fitting and the headline plausible-scenario mean.",
            "The directional whole-AIA column adds shadow zero-slice scores to the 13-target measured-positive sum only for scale comparison.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    top = validation.sort_values("mean", ascending=False).head(10)
    lines = [
        "# Final zero-target AIA shadow benchmark",
        "",
        "This is a truth-known local comparison, not a leaderboard prediction.",
        "",
        "## Validation headline",
        "",
        f"- Best strategy: `{best['strategy']}`",
        f"- Best zero-target-slice mean: `{float(best['mean']):.6f}`",
        f"- Exact routed complement mean: `{float(exact['mean']):.6f}`",
        f"- Mean delta over exact complement: `{float(best['mean_delta_vs_exact']):+.6f}`",
        f"- Paired validation win rate: `{float(best['win_rate_vs_exact']):.1%}`",
        f"- Selected logistic C: `{selected_c:g}`",
        "",
        "## Top validation strategies",
        "",
        "| strategy | mean | p10 | p90 | delta vs exact | win rate |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            "| {strategy} | {mean:.6f} | {p10:.6f} | {p90:.6f} | {delta:+.6f} | {win:.1%} |".format(
                strategy=row.strategy,
                mean=float(row.mean),
                p10=float(row.p10),
                p90=float(row.p90),
                delta=float(row.mean_delta_vs_exact),
                win=float(row.win_rate_vs_exact),
            )
            for row in top.itertuples(index=False)
        ],
        "",
        "## Design",
        "",
        "Three discovery and three disjoint validation seeds were evaluated under raw, 3% row-noise, time-shuffled, production-margin-warped, and independent-null scenarios. Each 326-row challenge has 163 members and 163 controls matched exactly on sex, prefecture, five-year age band, onset, and rare composition.",
        "",
        "## Required caveats",
        "",
        "- The independent scenario is a null check and is excluded from strategy fitting and headline means.",
        "- The `directional_whole_aia` values are scale translations, not private-score forecasts.",
        "- The private transformation and hidden outcomes cannot be reconstructed locally.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(top.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"selected_C={selected_c:g}")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
