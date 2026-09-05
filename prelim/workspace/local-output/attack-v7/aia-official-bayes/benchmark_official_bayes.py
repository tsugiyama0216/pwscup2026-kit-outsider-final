"""Official-feedback-conditioned AIA selection research.

This is a research-only normal-format benchmark.  It uses public contest data,
the team's three submitted AIA files, and their official aggregate feedback.
Discovery and validation shadow seeds are disjoint.  It never reads a token,
writes a production candidate, creates an archive, or contacts a service.

The central question is whether a strict subset of the 163 rows in the exact
routed complement can beat the current horizon-edge candidate.  Candidate rows
are ranked with the three submitted masks, public release-distance features,
and a reward model trained only on discovery blocks that reproduce the
official condition (all three direct attacks score zero).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
SHADOW_DIR = ROOT / "local-output" / "attack-v6" / "shadow-benchmark"
for source_root in (ROOT, STARTER_SRC, SHADOW_DIR):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import benchmark as bench  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    age_band,
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
    stratified_half_selection,
    strict_rank,
)
from attack_phase_v6.aia_global_density import (  # noqa: E402
    global_density_ratio_score,
    sex_copula_proximity_score,
)


OUT = Path(__file__).resolve().parent
DIST = ROOT / "participant_data"
PRODUCTION_TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
TEAM = ROOT / "submissions" / "prelim" / "team24"
FEEDBACK = ROOT / "attack_phase_v6" / "official_feedback"
ZERO_TARGETS = (6, 7, 8, 9, 17, 18, 19)
N_ARM = 163
N_ROWS = 326
EDGE_TIME = 9.5
ABSTAIN_TIME = 11.0
HORIZON_TIME = 10.0
K_GRID = (32, 48, 64, 80, 96, 112, 128, 144, 163)
RIDGE_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0)
ROUTE_KIND = {
    6: "copula",
    7: "density_all5",
    8: "copula",
    9: "density_stable3",
    17: "copula",
    18: "sex_consensus",
    19: "density_stable3_invert",
}
NUMERIC_FEATURES = (
    "v4_active",
    "independent_active",
    "routed_active",
    "v4_independent_both",
    "reject_votes",
    "inverse_route_rank",
    "proximity",
    "copula_proximity",
    "sex_proximity",
    "density_all5",
    "density_stable3",
    "density_nohdl",
    "density_labs",
    "age_rank",
    "sbp_rank",
    "tg_rank",
    "hdl_rank",
    "alt_rank",
    "visible_stratum_size_rank",
)


@dataclass
class Block:
    split: str
    seed: int
    replicate: int
    official_target: int
    internal_target: str
    route_kind: str
    frame: pd.DataFrame
    truth: pd.DataFrame
    direct: dict[str, float]

    @property
    def key(self) -> str:
        return (
            f"{self.split}:{self.seed}:{self.replicate}:"
            f"{self.official_target}:{self.internal_target}"
        )

    @property
    def exact_zero(self) -> bool:
        return max(self.direct.values()) <= 1e-12

    @property
    def near_zero(self) -> bool:
        return max(self.direct.values()) <= 0.02 + 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=int, default=24)
    parser.add_argument("--validation", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument(
        "--skip-shadow",
        action="store_true",
        help="Reuse saved block_features.csv instead of generating shadows.",
    )
    return parser.parse_args()


def _aligned(frame: pd.DataFrame, target: str, challenge: pd.DataFrame) -> np.ndarray:
    return bench.aligned_time(frame, target, challenge)


def _rank(values: Iterable[float], ids: np.ndarray) -> np.ndarray:
    return strict_rank(np.asarray(values, dtype=float), ids)


def _route_feature_frame(
    *,
    official_target: int,
    internal_target: str,
    challenge: pd.DataFrame,
    release: pd.DataFrame,
    background: pd.DataFrame,
    linkable_release: pd.DataFrame,
    v4_time: np.ndarray,
    independent_time: np.ndarray,
) -> pd.DataFrame:
    ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    v4_active = ~np.isclose(v4_time, ABSTAIN_TIME)
    independent_active = ~np.isclose(independent_time, ABSTAIN_TIME)
    proximity = proximity_score(linkable_release, challenge)
    copula = copula_proximity_score(linkable_release, challenge)
    sex_proximity = sex_copula_proximity_score(linkable_release, challenge)
    density_all5 = global_density_ratio_score(
        linkable_release, challenge, background, columns=bench.VIEWS["all5"]
    )
    density_stable3 = global_density_ratio_score(
        linkable_release, challenge, background, columns=bench.VIEWS["stable3"]
    )
    density_nohdl = global_density_ratio_score(
        linkable_release, challenge, background, columns=bench.VIEWS["nohdl"]
    )
    density_labs = global_density_ratio_score(
        linkable_release, challenge, background, columns=bench.VIEWS["labs"]
    )
    kind = ROUTE_KIND[official_target]
    if kind == "copula":
        route_score = (
            1.10 * v4_active
            + 0.90 * independent_active
            + 0.20 * proximity
            + 0.20 * copula
        )
    elif kind == "density_all5":
        route_score = density_all5
    elif kind == "density_stable3":
        route_score = density_stable3
    elif kind == "density_stable3_invert":
        route_score = -density_stable3
    elif kind == "sex_consensus":
        route_score = (
            1.10 * v4_active + 0.90 * independent_active + 0.55 * sex_proximity
        )
    else:  # pragma: no cover - guarded by the constant map
        raise ValueError(f"unknown route kind: {kind}")
    routed_active, _ = stratified_half_selection(
        challenge, route_score, member_count=N_ARM
    )
    work = challenge.assign(
        age_band=age_band(challenge["age"]),
    )
    stratum_size = work.groupby(
        ["sex", "prefecture", "age_band"], sort=False
    )["challenge_row_id"].transform("size").to_numpy(dtype=float)
    result = pd.DataFrame(
        {
            "challenge_row_id": ids,
            "v4_active": v4_active.astype(float),
            "independent_active": independent_active.astype(float),
            "routed_active": routed_active.astype(float),
            "v4_independent_both": (v4_active & independent_active).astype(float),
            "reject_votes": (
                3 - v4_active.astype(int) - independent_active.astype(int) - routed_active.astype(int)
            ).astype(float),
            "inverse_route_rank": 1.0 - _rank(route_score, ids),
            "proximity": proximity,
            "copula_proximity": copula,
            "sex_proximity": sex_proximity,
            "density_all5": density_all5,
            "density_stable3": density_stable3,
            "density_nohdl": density_nohdl,
            "density_labs": density_labs,
            "age_rank": _rank(challenge["age"], ids),
            "sbp_rank": _rank(challenge["SBP"], ids),
            "tg_rank": _rank(np.log1p(challenge["TG"].clip(lower=0.0)), ids),
            "hdl_rank": _rank(challenge["HDL"], ids),
            "alt_rank": _rank(np.log1p(challenge["ALT"].clip(lower=0.0)), ids),
            "visible_stratum_size_rank": _rank(stratum_size, ids),
            "sex": challenge["sex"].to_numpy(),
            "prefecture": challenge["prefecture"].astype(str).to_numpy(),
            "age_band": age_band(challenge["age"]),
        }
    )
    result["pattern"] = (
        result["v4_active"].astype(int).astype(str)
        + result["independent_active"].astype(int).astype(str)
        + result["routed_active"].astype(int).astype(str)
    )
    result["route_kind"] = kind
    result["official_target"] = official_target
    result["internal_target"] = internal_target
    return result


def _build_base_attacks(
    world: Any,
    releases: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline, _, _ = bench.v4.build_aia_v4(
        world.attack_background,
        releases,
        world.challenges,
        main_fraction=0.5,
        variant_fractions=[0.5],
        abstain_time=ABSTAIN_TIME,
        horizon_time=HORIZON_TIME,
        link_trust_distance=bench.v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
        baseline_abstain_time=bench.shadow.choose_low_hit_time(
            world.attack_background,
            horizon=HORIZON_TIME,
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
        member_count=N_ARM,
        horizon=HORIZON_TIME,
        abstain_time=ABSTAIN_TIME,
    )
    return baseline, independent


def _shadow_releases(
    world: Any, *, seed: int, replicate: int, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    releases: dict[str, pd.DataFrame] = {}
    for ordinal, (target, source) in enumerate(world.aia_sources.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, replicate, ordinal, int(target), 9143])
        )
        releases[target] = bench.shadow.scenario_release(
            source, world.attack_background, "independent", cfg, rng
        )
    return releases


def build_shadow_blocks(
    *, split: str, seeds: Iterable[int], replicates: int, cfg: dict[str, Any]
) -> list[Block]:
    candidates = pd.read_csv(DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(DIST / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(DIST / "A_bg.csv").reset_index(drop=True)
    detector = bench.shadow.load_rare_detector(DIST)
    spec = bench.shadow.ShadowSpec(
        n_targets=len(ZERO_TARGETS),
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=25 * len(ZERO_TARGETS),
        minimum_shared=25,
        maximum_shared=26,
    )
    blocks: list[Block] = []
    for seed in seeds:
        print(f"{split} seed={seed}: build shadow world", flush=True)
        world = bench.shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        bench.exactify_visible_control_matching(world)
        bench.shadow.validate_world(world, spec)
        for replicate in range(replicates):
            releases = _shadow_releases(
                world, seed=seed, replicate=replicate, cfg=cfg
            )
            v4_frame, independent_frame = _build_base_attacks(world, releases, cfg)
            linkable, _ = filter_linkable_releases(releases)
            for ordinal, internal_target in enumerate(
                sorted(world.challenges, key=int)
            ):
                official_target = ZERO_TARGETS[ordinal]
                challenge = world.challenges[internal_target].reset_index(drop=True)
                truth = world.aia_truth[internal_target].reset_index(drop=True)
                v4_time = _aligned(v4_frame, internal_target, challenge)
                independent_time = _aligned(
                    independent_frame, internal_target, challenge
                )
                features = _route_feature_frame(
                    official_target=official_target,
                    internal_target=internal_target,
                    challenge=challenge,
                    release=releases[internal_target],
                    background=world.attack_background,
                    linkable_release=linkable[internal_target],
                    v4_time=v4_time,
                    independent_time=independent_time,
                )
                routed_time = np.where(
                    features["routed_active"].to_numpy(dtype=bool),
                    HORIZON_TIME,
                    ABSTAIN_TIME,
                )
                direct = {
                    "v4": bench.score_block(v4_time, truth, cfg)["power"],
                    "independent": bench.score_block(
                        independent_time, truth, cfg
                    )["power"],
                    "routed": bench.score_block(routed_time, truth, cfg)["power"],
                }
                labels = truth["label"].eq("member").to_numpy()
                edge_hit = (
                    np.abs(EDGE_TIME - truth["time"].to_numpy(dtype=float))
                    <= float(cfg["aia"]["tau_time"])
                )
                features["label_member"] = labels.astype(int)
                features["edge_hit"] = edge_hit.astype(int)
                features["reward"] = edge_hit.astype(int) * np.where(labels, 1, -1)
                features["truth_time"] = truth["time"].to_numpy(dtype=float)
                features["truth_rare"] = truth["is_rare"].astype(int).to_numpy()
                features["split"] = split
                features["seed"] = seed
                features["replicate"] = replicate
                blocks.append(
                    Block(
                        split=split,
                        seed=seed,
                        replicate=replicate,
                        official_target=official_target,
                        internal_target=internal_target,
                        route_kind=ROUTE_KIND[official_target],
                        frame=features,
                        truth=truth,
                        direct=direct,
                    )
                )
    return blocks


def _model_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    base = frame.loc[:, NUMERIC_FEATURES].astype(float)
    categories = pd.get_dummies(
        frame[["route_kind", "pattern"]].astype(str),
        columns=["route_kind", "pattern"],
        dtype=float,
    )
    matrix = pd.concat([base.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)
    if columns is None:
        columns = sorted(matrix.columns)
    return matrix.reindex(columns=columns, fill_value=0.0), columns


def _pattern_scores(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_mean = float(train["reward"].mean())
    global_pattern = train.groupby("pattern")["reward"].agg(["sum", "count"])
    route_pattern = train.groupby(["route_kind", "pattern"])["reward"].agg(
        ["sum", "count"]
    )
    output = np.empty(len(test), dtype=float)
    for index, row in enumerate(test.itertuples(index=False)):
        if row.pattern in global_pattern.index:
            aggregate = global_pattern.loc[row.pattern]
            pattern_mean = (float(aggregate["sum"]) + 64.0 * global_mean) / (
                float(aggregate["count"]) + 64.0
            )
        else:
            pattern_mean = global_mean
        key = (row.route_kind, row.pattern)
        if key in route_pattern.index:
            aggregate = route_pattern.loc[key]
            output[index] = (float(aggregate["sum"]) + 48.0 * pattern_mean) / (
                float(aggregate["count"]) + 48.0
            )
        else:
            output[index] = pattern_mean
    return output


def _fit_predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
) -> np.ndarray:
    train_x, columns = _model_matrix(train)
    test_x, _ = _model_matrix(test, columns)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x.to_numpy(dtype=float), train["reward"].to_numpy(dtype=float))
    return model.predict(test_x.to_numpy(dtype=float))


def add_oof_scores(discovery: list[Block]) -> None:
    eligible = [block for block in discovery if block.exact_zero]
    if not eligible:
        raise RuntimeError("no exact-zero discovery blocks")
    full = pd.concat(
        [block.frame.assign(block_key=block.key) for block in eligible],
        ignore_index=True,
    )
    seeds = full["seed"].to_numpy(dtype=int)
    unique_seeds = np.unique(seeds)
    folds = min(5, len(unique_seeds))
    if folds < 2:
        raise RuntimeError("not enough eligible discovery seeds for OOF fitting")
    for alpha in RIDGE_ALPHA_GRID:
        full[f"ridge_{alpha:g}"] = np.nan
    full["pattern_bayes"] = np.nan
    for train_indices, test_indices in GroupKFold(n_splits=folds).split(
        full, groups=seeds
    ):
        train = full.iloc[train_indices].reset_index(drop=True)
        test = full.iloc[test_indices].reset_index(drop=True)
        full.loc[test_indices, "pattern_bayes"] = _pattern_scores(train, test)
        for alpha in RIDGE_ALPHA_GRID:
            full.loc[test_indices, f"ridge_{alpha:g}"] = _fit_predict_ridge(
                train, test, alpha=alpha
            )
    score_columns = ["pattern_bayes", *[f"ridge_{a:g}" for a in RIDGE_ALPHA_GRID]]
    if full[score_columns].isna().any().any():
        raise RuntimeError("OOF scores contain missing values")
    indexed = full.set_index(["block_key", "challenge_row_id"])
    for block in eligible:
        ids = block.frame["challenge_row_id"].to_numpy(dtype=int)
        for column in score_columns:
            block.frame[column] = indexed.loc[(block.key, ids), column].to_numpy(
                dtype=float
            )


def add_final_scores(discovery: list[Block], validation: list[Block]) -> None:
    train = pd.concat(
        [block.frame for block in discovery if block.exact_zero], ignore_index=True
    )
    for block in validation:
        block.frame["pattern_bayes"] = _pattern_scores(train, block.frame)
        for alpha in RIDGE_ALPHA_GRID:
            block.frame[f"ridge_{alpha:g}"] = _fit_predict_ridge(
                train, block.frame, alpha=alpha
            )


def _proportional_select(block: pd.DataFrame, score: np.ndarray, k: int) -> np.ndarray:
    candidate = block["routed_active"].eq(0).to_numpy()
    candidate_rows = np.flatnonzero(candidate)
    if len(candidate_rows) != N_ARM:
        raise RuntimeError("routed complement does not contain exactly 163 rows")
    if k >= len(candidate_rows):
        return candidate
    keys = list(zip(block["sex"], block["prefecture"], block["age_band"], strict=True))
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in candidate_rows:
        groups[keys[row]].append(int(row))
    quotas: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[float, float, tuple[Any, ...]]] = []
    allocated = 0
    for key, rows in groups.items():
        exact = k * len(rows) / len(candidate_rows)
        floor = int(math.floor(exact))
        quotas[key] = floor
        allocated += floor
        best = float(np.max(score[rows]))
        remainders.append((exact - floor, best, key))
    for _, _, key in sorted(remainders, reverse=True)[: k - allocated]:
        quotas[key] += 1
    selected = np.zeros(len(block), dtype=bool)
    ids = block["challenge_row_id"].to_numpy(dtype=int)
    for key, rows_list in groups.items():
        rows = np.asarray(rows_list, dtype=int)
        order = rows[np.lexsort((ids[rows], -score[rows]))]
        selected[order[: quotas[key]]] = True
    if int(selected.sum()) != k or np.any(selected & ~candidate):
        raise RuntimeError("proportional subset selection failed")
    return selected


def _global_select(block: pd.DataFrame, score: np.ndarray, k: int) -> np.ndarray:
    candidate = block["routed_active"].eq(0).to_numpy()
    rows = np.flatnonzero(candidate)
    if len(rows) != N_ARM:
        raise RuntimeError("routed complement does not contain exactly 163 rows")
    k = min(k, len(rows))
    ids = block["challenge_row_id"].to_numpy(dtype=int)
    order = rows[np.lexsort((ids[rows], -score[rows]))]
    selected = np.zeros(len(block), dtype=bool)
    selected[order[:k]] = True
    return selected


def candidate_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    inverse_route = frame["inverse_route_rank"].to_numpy(dtype=float)
    reject = frame["reject_votes"].to_numpy(dtype=float)
    pattern = frame.get("pattern_bayes", pd.Series(np.zeros(len(frame)))).to_numpy(
        dtype=float
    )
    scores = {
        "inverse_route": inverse_route,
        "reject_vote": reject + 0.01 * inverse_route,
        "pattern_bayes": pattern + 0.01 * inverse_route,
    }
    for alpha in RIDGE_ALPHA_GRID:
        name = f"ridge_{alpha:g}"
        if name in frame:
            ridge = frame[name].to_numpy(dtype=float)
            scores[name] = ridge
            scores[f"hybrid_{alpha:g}"] = (
                _rank(ridge, frame["challenge_row_id"].to_numpy(dtype=int))
                + _rank(pattern, frame["challenge_row_id"].to_numpy(dtype=int))
                + inverse_route
            ) / 3.0
    return scores


def score_policies(
    blocks: list[Block], cfg: dict[str, Any], *, condition: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        if condition == "exact" and not block.exact_zero:
            continue
        if condition == "near" and not block.near_zero:
            continue
        frame = block.frame
        base_selected = frame["routed_active"].eq(0).to_numpy()
        base_predicted = np.where(base_selected, EDGE_TIME, ABSTAIN_TIME)
        base = bench.score_block(base_predicted, block.truth, cfg)
        common = {
            "split": block.split,
            "seed": block.seed,
            "replicate": block.replicate,
            "official_target": block.official_target,
            "internal_target": block.internal_target,
            "route_kind": block.route_kind,
            "condition": condition,
            "direct_v4": block.direct["v4"],
            "direct_independent": block.direct["independent"],
            "direct_routed": block.direct["routed"],
            "baseline_power": base["power"],
        }
        rows.append(
            {
                **common,
                "strategy": "exact_complement_edge",
                "ranking": "none",
                "allocation": "all",
                "active_k": N_ARM,
                "active_count": N_ARM,
                **base,
                "delta_vs_exact": 0.0,
            }
        )
        active_sum = (
            frame[["v4_active", "independent_active", "routed_active"]]
            .sum(axis=1)
            .to_numpy(dtype=int)
        )
        static_masks = {
            # This is the complement of the strongest three-expert control
            # consensus.  Unlike inverse-consensus-at-k=163 it is deliberately
            # allowed to answer more than half the rows.
            "complement_unanimous_active": active_sum <= 2,
            "complement_majority_active": active_sum <= 1,
            "all_three_inactive_only": active_sum == 0,
            "complement_v4_independent_intersection": ~(
                frame["v4_active"].eq(1).to_numpy()
                & frame["independent_active"].eq(1).to_numpy()
            ),
            "complement_v4_routed_intersection": ~(
                frame["v4_active"].eq(1).to_numpy()
                & frame["routed_active"].eq(1).to_numpy()
            ),
            "complement_independent_routed_intersection": ~(
                frame["independent_active"].eq(1).to_numpy()
                & frame["routed_active"].eq(1).to_numpy()
            ),
        }
        for strategy, selected in static_masks.items():
            predicted = np.where(selected, EDGE_TIME, ABSTAIN_TIME)
            metrics = bench.score_block(predicted, block.truth, cfg)
            rows.append(
                {
                    **common,
                    "strategy": strategy,
                    "ranking": "boolean_pattern",
                    "allocation": "pattern",
                    "active_k": -1,
                    "active_count": int(np.count_nonzero(selected)),
                    **metrics,
                    "delta_vs_exact": metrics["power"] - base["power"],
                }
            )
        for ranking, score in candidate_scores(frame).items():
            for allocation, selector in (
                ("global", _global_select),
                ("proportional", _proportional_select),
            ):
                for k in K_GRID:
                    selected = selector(frame, score, k)
                    predicted = np.where(selected, EDGE_TIME, ABSTAIN_TIME)
                    metrics = bench.score_block(predicted, block.truth, cfg)
                    rows.append(
                        {
                            **common,
                            "strategy": f"{ranking}_{allocation}_k{k}",
                            "ranking": ranking,
                            "allocation": allocation,
                            "active_k": k,
                            "active_count": int(np.count_nonzero(selected)),
                            **metrics,
                            "delta_vs_exact": metrics["power"] - base["power"],
                        }
                    )
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    keys = ["split", "condition", "strategy", "ranking", "allocation", "active_k"]
    return (
        scores.groupby(keys, as_index=False)
        .agg(
            blocks=("power", "size"),
            seeds=("seed", "nunique"),
            mean_power=("power", "mean"),
            median_power=("power", "median"),
            p10_power=("power", lambda x: float(np.quantile(x, 0.10))),
            mean_delta=("delta_vs_exact", "mean"),
            median_delta=("delta_vs_exact", "median"),
            p10_delta=("delta_vs_exact", lambda x: float(np.quantile(x, 0.10))),
            p25_delta=("delta_vs_exact", lambda x: float(np.quantile(x, 0.25))),
            win_rate=("delta_vs_exact", lambda x: float(np.mean(x > 1e-12))),
            loss_rate=("delta_vs_exact", lambda x: float(np.mean(x < -1e-12))),
            mean_active=("active_count", "mean"),
            min_active=("active_count", "min"),
            max_active=("active_count", "max"),
        )
        .sort_values(
            ["split", "condition", "mean_delta", "p10_delta"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def choose_discovery_strategy(summary: pd.DataFrame) -> str:
    candidates = summary.loc[
        summary["split"].eq("discovery")
        & summary["condition"].eq("exact")
        & ~summary["strategy"].eq("exact_complement_edge")
    ].copy()
    if candidates.empty:
        return "exact_complement_edge"
    # Discovery-only conservative objective.  A positive lower quartile is
    # preferred; otherwise the current exact complement remains the choice.
    candidates["selection_value"] = (
        candidates["mean_delta"] + 0.5 * candidates["p25_delta"]
    )
    guarded = candidates.loc[
        (candidates["p25_delta"] >= -1e-12)
        & (candidates["loss_rate"] <= 0.25 + 1e-12)
    ]
    if guarded.empty:
        return "exact_complement_edge"
    return str(
        guarded.sort_values(
            ["selection_value", "mean_delta", "active_k"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]["strategy"]
    )


def _production_frames() -> tuple[dict[str, pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    predictions = {
        "v4": pd.read_csv(TEAM / "attack-candidate-v4" / "F_aia.csv"),
        "independent": pd.read_csv(
            TEAM / "attack-candidate-daybreak-independent" / "F_aia.csv"
        ),
        "routed": pd.read_csv(
            TEAM / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv"
        ),
    }
    # The submitted routed attack filtered cross-target duplicate release rows
    # using all 20 targets.  Reconstruct that exact release universe here even
    # though the research policy changes only the common seven zero targets.
    releases = {
        int(path.stem.split("_", 1)[1]): pd.read_csv(path)
        for path in PRODUCTION_TARGETS.glob("C_*.csv")
    }
    challenges = {
        target: pd.read_csv(
            PRODUCTION_TARGETS / f"aia_challenge_{target}.csv"
        )
        for target in ZERO_TARGETS
    }
    return predictions, releases, challenges


def production_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions, releases, challenges = _production_frames()
    background = pd.read_csv(DIST / "A_bg.csv")
    linkable, _ = filter_linkable_releases(
        {str(key): value for key, value in releases.items()}
    )
    parts: list[pd.DataFrame] = []
    reconciliation: dict[str, Any] = {}
    for target in ZERO_TARGETS:
        challenge = challenges[target].reset_index(drop=True)
        v4_time = _aligned(predictions["v4"], str(target), challenge)
        independent_time = _aligned(
            predictions["independent"], str(target), challenge
        )
        frame = _route_feature_frame(
            official_target=target,
            internal_target=str(target),
            challenge=challenge,
            release=releases[target],
            background=background,
            linkable_release=linkable[str(target)],
            v4_time=v4_time,
            independent_time=independent_time,
        )
        submitted = _aligned(predictions["routed"], str(target), challenge)
        submitted_active = ~np.isclose(submitted, ABSTAIN_TIME)
        calculated_active = frame["routed_active"].to_numpy(dtype=bool)
        changed = int(np.count_nonzero(submitted_active != calculated_active))
        reconciliation[str(target)] = {
            "rows": len(frame),
            "submitted_active": int(submitted_active.sum()),
            "calculated_active": int(calculated_active.sum()),
            "changed_rows": changed,
        }
        if changed:
            raise RuntimeError(
                f"production routed reconstruction differs for target {target}: {changed}"
            )
        parts.append(frame)
    return pd.concat(parts, ignore_index=True), reconciliation


def fit_production_scores(
    discovery: list[Block], production: pd.DataFrame
) -> pd.DataFrame:
    train = pd.concat(
        [block.frame for block in discovery if block.exact_zero], ignore_index=True
    )
    output = production.copy()
    output["pattern_bayes"] = _pattern_scores(train, output)
    for alpha in RIDGE_ALPHA_GRID:
        output[f"ridge_{alpha:g}"] = _fit_predict_ridge(
            train, output, alpha=alpha
        )
    return output


def build_research_mask(production: pd.DataFrame, strategy: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if strategy == "exact_complement_edge":
        ranking = "inverse_route"
        allocation = "global"
        k = N_ARM
    elif strategy in {
        "complement_unanimous_active",
        "complement_majority_active",
        "all_three_inactive_only",
        "complement_v4_independent_intersection",
        "complement_v4_routed_intersection",
        "complement_independent_routed_intersection",
    }:
        ranking = "boolean_pattern"
        allocation = "pattern"
        k = -1
    else:
        suffix = strategy.rsplit("_k", 1)
        if len(suffix) != 2:
            raise ValueError(f"cannot parse strategy: {strategy}")
        k = int(suffix[1])
        stem = suffix[0]
        if stem.endswith("_proportional"):
            ranking = stem[: -len("_proportional")]
            allocation = "proportional"
        elif stem.endswith("_global"):
            ranking = stem[: -len("_global")]
            allocation = "global"
        else:
            raise ValueError(f"cannot parse strategy allocation: {strategy}")
    selector = _proportional_select if allocation == "proportional" else _global_select
    for target in ZERO_TARGETS:
        block = production.loc[production["official_target"].eq(target)].reset_index(
            drop=True
        )
        if strategy == "exact_complement_edge":
            selected = block["routed_active"].eq(0).to_numpy()
        elif strategy == "complement_unanimous_active":
            selected = (
                block[["v4_active", "independent_active", "routed_active"]]
                .sum(axis=1)
                .le(2)
                .to_numpy()
            )
        elif strategy == "complement_majority_active":
            selected = (
                block[["v4_active", "independent_active", "routed_active"]]
                .sum(axis=1)
                .le(1)
                .to_numpy()
            )
        elif strategy == "all_three_inactive_only":
            selected = (
                block[["v4_active", "independent_active", "routed_active"]]
                .sum(axis=1)
                .eq(0)
                .to_numpy()
            )
        elif strategy == "complement_v4_independent_intersection":
            selected = ~(
                block["v4_active"].eq(1).to_numpy()
                & block["independent_active"].eq(1).to_numpy()
            )
        elif strategy == "complement_v4_routed_intersection":
            selected = ~(
                block["v4_active"].eq(1).to_numpy()
                & block["routed_active"].eq(1).to_numpy()
            )
        elif strategy == "complement_independent_routed_intersection":
            selected = ~(
                block["independent_active"].eq(1).to_numpy()
                & block["routed_active"].eq(1).to_numpy()
            )
        else:
            scores = candidate_scores(block)
            selected = selector(block, scores[ranking], k)
        parts.append(
            pd.DataFrame(
                {
                    "target_k": target,
                    "challenge_row_id": block["challenge_row_id"].astype(int),
                    "time_hat": np.where(selected, EDGE_TIME, ABSTAIN_TIME),
                    "selected": selected.astype(int),
                    "pattern": block["pattern"],
                    "ranking_score": (
                        np.full(len(block), np.nan)
                        if ranking == "boolean_pattern"
                        else candidate_scores(block).get(
                            ranking, block["inverse_route_rank"].to_numpy(dtype=float)
                        )
                    ),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def robust_raw_bounds(production: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Partial-identification bounds under equal 9.5-hit totals per arm.

    We fix 136 edge-eligible rows in each arm and target, matching the public
    B/A frequency used by the v6 audit.  Every direct submitted mask must have
    nonpositive raw advantage per target, and the routed aggregate control
    excess is constrained to the official 59--68 range.  The result is a
    sensitivity bound, not an unconditional guarantee.
    """
    merged = production.merge(
        candidate[["target_k", "challenge_row_id", "selected"]],
        left_on=["official_target", "challenge_row_id"],
        right_on=["target_k", "challenge_row_id"],
        how="left",
        validate="one_to_one",
    )
    merged["selected"] = merged["selected"].astype(int)
    grouped = (
        merged.groupby(
            [
                "official_target",
                "v4_active",
                "independent_active",
                "routed_active",
                "selected",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "capacity"})
    )
    variable_meta: list[tuple[int, str]] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []
    group_variables: dict[int, tuple[int, int]] = {}
    for row_index, row in grouped.iterrows():
        indices = []
        for arm in ("member", "control"):
            index = len(variable_meta)
            variable_meta.append((row_index, arm))
            lower.append(0.0)
            upper.append(float(row["capacity"]))
            integrality.append(1)
            indices.append(index)
        group_variables[row_index] = (indices[0], indices[1])
    constraints: list[tuple[dict[int, float], float, float]] = []
    for row_index, row in grouped.iterrows():
        m_index, c_index = group_variables[row_index]
        constraints.append(
            ({m_index: 1.0, c_index: 1.0}, 0.0, float(row["capacity"]))
        )
    for target in ZERO_TARGETS:
        target_rows = grouped.index[grouped["official_target"].eq(target)]
        member_total = {group_variables[i][0]: 1.0 for i in target_rows}
        control_total = {group_variables[i][1]: 1.0 for i in target_rows}
        constraints.append((member_total, 136.0, 136.0))
        constraints.append((control_total, 136.0, 136.0))
        for expert in ("v4_active", "independent_active", "routed_active"):
            coefficients: dict[int, float] = {}
            for i in target_rows:
                if int(grouped.loc[i, expert]) == 1:
                    coefficients[group_variables[i][0]] = 1.0
                    coefficients[group_variables[i][1]] = -1.0
            constraints.append((coefficients, -np.inf, 0.0))
    routed_gap: dict[int, float] = {}
    for i, row in grouped.iterrows():
        if int(row["routed_active"]) == 1:
            routed_gap[group_variables[i][0]] = -1.0
            routed_gap[group_variables[i][1]] = 1.0
    constraints.append((routed_gap, 59.0, 68.0))
    objective = np.zeros(len(variable_meta), dtype=float)
    for i, row in grouped.iterrows():
        if int(row["selected"]) == 1:
            objective[group_variables[i][0]] = 1.0
            objective[group_variables[i][1]] = -1.0
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_values: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for row_index, (coefficients, row_lo, row_hi) in enumerate(constraints):
        for column, value in coefficients.items():
            matrix_rows.append(row_index)
            matrix_cols.append(column)
            matrix_values.append(value)
        lo.append(row_lo)
        hi.append(row_hi)
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(len(constraints), len(variable_meta)),
    ).tocsr()
    bounds = Bounds(np.asarray(lower), np.asarray(upper))
    linear = LinearConstraint(matrix, np.asarray(lo), np.asarray(hi))
    results: dict[str, Any] = {}
    for mode, direction in (("min", 1.0), ("max", -1.0)):
        result = milp(
            direction * objective,
            integrality=np.asarray(integrality),
            bounds=bounds,
            constraints=linear,
            options={"time_limit": 120.0},
        )
        if not result.success:
            raise RuntimeError(f"robust MILP {mode} failed: {result.message}")
        results[mode] = float(objective @ result.x)
    return {
        "assumption": "136 edge-time hits in each arm and target",
        "official_constraints": "all three direct raw advantages <= 0 per target; routed aggregate control excess 59..68",
        "candidate_raw_advantage_min": results["min"],
        "candidate_raw_advantage_max": results["max"],
        "variables": len(variable_meta),
        "constraints": len(constraints),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = bench.load_config(bench.KIT_CONFIG_PATH)
    if args.skip_shadow:
        raise SystemExit("--skip-shadow is reserved for a future serialized-block loader")
    discovery_seeds = range(20270101, 20270101 + args.discovery)
    validation_seeds = range(20270201, 20270201 + args.validation)
    discovery = build_shadow_blocks(
        split="discovery",
        seeds=discovery_seeds,
        replicates=args.replicates,
        cfg=cfg,
    )
    validation = build_shadow_blocks(
        split="validation",
        seeds=validation_seeds,
        replicates=args.replicates,
        cfg=cfg,
    )
    add_oof_scores(discovery)
    add_final_scores(discovery, validation)
    scores = pd.concat(
        [
            score_policies(discovery, cfg, condition="exact"),
            score_policies(discovery, cfg, condition="near"),
            score_policies(validation, cfg, condition="exact"),
            score_policies(validation, cfg, condition="near"),
        ],
        ignore_index=True,
    )
    summary = summarize_scores(scores)
    chosen = choose_discovery_strategy(summary)

    production, reconciliation = production_features()
    production = fit_production_scores(discovery, production)
    research_mask = build_research_mask(production, chosen)
    baseline_mask = build_research_mask(production, "exact_complement_edge")
    changed = int(
        np.count_nonzero(
            research_mask["selected"].to_numpy()
            != baseline_mask["selected"].to_numpy()
        )
    )
    robust = robust_raw_bounds(production, research_mask)

    # Persist compact reproducible evidence.  Truth-bearing shadow rows stay in
    # the ignored local-output tree and never enter a submission directory.
    scores.to_csv(OUT / "policy_scores.csv", index=False)
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    research_mask.to_csv(OUT / "research_zero_block_mask.csv", index=False)
    production.to_csv(OUT / "production_features.csv", index=False)
    block_rows = []
    for block in [*discovery, *validation]:
        block_rows.append(
            {
                "split": block.split,
                "seed": block.seed,
                "replicate": block.replicate,
                "official_target": block.official_target,
                "route_kind": block.route_kind,
                "exact_zero": block.exact_zero,
                "near_zero": block.near_zero,
                **{f"direct_{key}": value for key, value in block.direct.items()},
            }
        )
    pd.DataFrame(block_rows).to_csv(OUT / "block_conditions.csv", index=False)

    chosen_validation = summary.loc[
        summary["split"].eq("validation")
        & summary["condition"].eq("exact")
        & summary["strategy"].eq(chosen)
    ]
    baseline_validation = summary.loc[
        summary["split"].eq("validation")
        & summary["condition"].eq("exact")
        & summary["strategy"].eq("exact_complement_edge")
    ]
    chosen_record = (
        chosen_validation.iloc[0].to_dict() if not chosen_validation.empty else None
    )
    baseline_record = (
        baseline_validation.iloc[0].to_dict()
        if not baseline_validation.empty
        else None
    )
    adoption = bool(
        chosen != "exact_complement_edge"
        and chosen_record is not None
        and float(chosen_record["mean_delta"]) >= 0.005
        and float(chosen_record["win_rate"]) >= 0.60
        and float(chosen_record["p10_delta"]) >= -0.02
        and robust["candidate_raw_advantage_min"] >= 0.0
    )
    result = {
        "purpose": "official-feedback-conditioned AIA subset research; not a leaderboard prediction",
        "as_of": "2026-09-05 Asia/Tokyo",
        "normal_format_only": True,
        "official_zero_targets": list(ZERO_TARGETS),
        "shadow": {
            "scenario": "independent public-background release",
            "discovery_seeds": list(discovery_seeds),
            "validation_seeds": list(validation_seeds),
            "release_replicates_per_seed": args.replicates,
            "discovery_exact_zero_blocks": int(
                sum(block.exact_zero for block in discovery)
            ),
            "validation_exact_zero_blocks": int(
                sum(block.exact_zero for block in validation)
            ),
            "discovery_near_zero_blocks": int(
                sum(block.near_zero for block in discovery)
            ),
            "validation_near_zero_blocks": int(
                sum(block.near_zero for block in validation)
            ),
        },
        "strategy_selection": {
            "selected_from_discovery_only": chosen,
            "validation": chosen_record,
            "baseline_validation": baseline_record,
            "changed_rows_from_exact_complement": changed,
            "adoption_gate_passed": adoption,
            "adoption_gate": {
                "mean_delta_min": 0.005,
                "win_rate_min": 0.60,
                "p10_delta_min": -0.02,
                "robust_raw_min": 0.0,
            },
        },
        "production_reconciliation": reconciliation,
        "robust_partial_identification": robust,
        "limitations": [
            "The private row truth is unavailable; shadow scores compare methods and are not leaderboard-score forecasts.",
            "Conditioning on all three direct scores being zero is deliberate but reduces effective sample size.",
            "The robust bound fixes 136 edge-eligible rows per arm and target; it is a sensitivity result, not an unconditional guarantee.",
            "Many discovery policies are compared, so validation performance and the adoption gate control the recommendation.",
        ],
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["shadow"], ensure_ascii=False, indent=2))
    print(json.dumps(result["strategy_selection"], ensure_ascii=False, indent=2))
    print(json.dumps(robust, ensure_ascii=False, indent=2))
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
