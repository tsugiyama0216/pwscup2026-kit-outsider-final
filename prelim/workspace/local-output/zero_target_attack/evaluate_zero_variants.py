"""Research-only truth-known evaluation of zero-target AIA variants.

The script deliberately writes only beneath ``local-output``.  It compares the
current copula-selection/horizon route with alternative linkage and structural
selection rules, including negative controls that destroy row-time linkage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v4 import generate_submission_v4 as v4  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    build_aia_v5_variants,
    stratified_half_selection,
)
from attack_phase_v5.evaluate_shadow_v5 import (  # noqa: E402
    exactify_visible_control_matching,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import score_aia_power  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402


PRODUCTION_TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
QI = ("age", "SBP", "TG", "HDL", "ALT")
NO_HDL = ("age", "SBP", "TG", "ALT")
STABLE3 = ("age", "SBP", "TG")
AGE_SBP_ALT = ("age", "SBP", "ALT")
BLOCK = ("sex", "prefecture")
VISIBLE_CONTINUOUS = QI
LOG_COLUMNS = {"TG", "ALT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20260921,20260922")
    parser.add_argument("--shadow-targets", type=int, default=8)
    parser.add_argument(
        "--scenarios",
        default="raw,noise_0p03,noise_0p10,time_shuffled,independent,warp,warp_noise_0p03,warp_time_shuffled",
    )
    parser.add_argument(
        "--templates",
        default=",".join(str(value) for value in PRODUCTION_TARGETS),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "local-output" / "zero_target_attack" / "shadow",
    )
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",") if value]
    args.scenarios = [value for value in args.scenarios.split(",") if value]
    args.templates = [int(value) for value in args.templates.split(",") if value]
    return args


def strict_rank(values: np.ndarray, tie_key: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.lexsort((np.asarray(tie_key), values))
    output = np.empty(len(values), dtype=float)
    output[order] = np.arange(1, len(values) + 1, dtype=float) / (len(values) + 1.0)
    return output


def midrank(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    output = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        output[:, j] = (
            frame[column].rank(method="average").to_numpy(dtype=float)
            / (len(frame) + 1.0)
        )
    return output / math.sqrt(len(columns))


def background_cdf(
    frame: pd.DataFrame, background: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    output = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        reference = np.sort(background[column].to_numpy(dtype=float))
        values = frame[column].to_numpy(dtype=float)
        lo = np.searchsorted(reference, values, side="left")
        hi = np.searchsorted(reference, values, side="right")
        output[:, j] = (0.5 * (lo + hi) + 0.5) / (len(reference) + 1.0)
    return output / math.sqrt(len(columns))


def raw_embed(
    frame: pd.DataFrame, background: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    reference = background.loc[:, columns].to_numpy(dtype=float, copy=True)
    for j, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, j] = np.log1p(np.clip(values[:, j], 0.0, None))
            reference[:, j] = np.log1p(np.clip(reference[:, j], 0.0, None))
    center = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(reference, axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (values - center) / scale / math.sqrt(len(columns))


def group_indices(frame: pd.DataFrame, columns: Iterable[str]) -> dict[tuple[object, ...], np.ndarray]:
    raw = frame.groupby(list(columns), sort=False, dropna=False).indices
    return {
        key if isinstance(key, tuple) else (key,): np.asarray(rows, dtype=int)
        for key, rows in raw.items()
    }


def block_cost(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    release_x: np.ndarray,
    challenge_x: np.ndarray,
) -> np.ndarray:
    output = cdist(challenge_x, release_x)
    output += 1_000.0 * (
        challenge["sex"].to_numpy()[:, None]
        != release["sex"].to_numpy()[None, :]
    )
    output += 500.0 * (
        challenge["prefecture"].astype(str).to_numpy()[:, None]
        != release["prefecture"].astype(str).to_numpy()[None, :]
    )
    exact = group_indices(release, BLOCK)
    sex_only = group_indices(release, ("sex",))
    all_rows = np.arange(len(release), dtype=int)
    for key, qrows in group_indices(challenge, BLOCK).items():
        pool = exact.get(key)
        if pool is None or not len(pool):
            pool = sex_only.get((key[0],))
        if pool is None or not len(pool):
            pool = all_rows
        # The large mismatch penalties above implement the same exact-block,
        # then sex-only, then global fallback for nearest-neighbour use while
        # leaving a finite matrix for unique assignment.
    return output


def sex_cost(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    release_x: np.ndarray,
    challenge_x: np.ndarray,
) -> np.ndarray:
    """Numeric linkage that trusts sex but deliberately ignores prefecture."""
    output = cdist(challenge_x, release_x)
    output += 1_000.0 * (
        challenge["sex"].to_numpy()[:, None]
        != release["sex"].to_numpy()[None, :]
    )
    return output


def link_details(cost: np.ndarray) -> dict[str, np.ndarray]:
    nearest = np.argmin(cost, axis=1)
    first = cost[np.arange(len(cost)), nearest]
    partition = np.partition(cost, kth=1, axis=1)
    second = partition[:, 1]
    reverse = np.argmin(cost, axis=0)
    mutual = reverse[nearest] == np.arange(len(cost))
    degree = np.bincount(nearest, minlength=cost.shape[1])[nearest]
    return {
        "nearest": nearest,
        "first": first,
        "second": second,
        "mutual": mutual,
        "degree": degree,
    }


def unique_links(cost: np.ndarray, selected: np.ndarray) -> np.ndarray:
    assigned = np.full(cost.shape[0], -1, dtype=int)
    for qrows in group_indices(
        pd.DataFrame({"sex": np.zeros(cost.shape[0])}), ("sex",)
    ).values():
        # This dummy grouping is replaced by caller-side finite-column grouping:
        # each row can only see its valid public block, so Hungarian remains valid.
        del qrows
    rows = np.flatnonzero(selected)
    if len(rows):
        finite_columns = np.flatnonzero(np.isfinite(cost[rows]).any(axis=0))
        local_rows, local_columns = linear_sum_assignment(cost[np.ix_(rows, finite_columns)])
        assigned[rows[local_rows]] = finite_columns[local_columns]
    return assigned


def unique_links_by_block(
    cost: np.ndarray,
    selected: np.ndarray,
    release: pd.DataFrame,
    challenge: pd.DataFrame,
) -> np.ndarray:
    assigned = np.full(len(challenge), -1, dtype=int)
    exact = group_indices(release, BLOCK)
    sex_only = group_indices(release, ("sex",))
    all_release = np.arange(len(release), dtype=int)
    for key, all_qrows in group_indices(challenge, BLOCK).items():
        qrows = all_qrows[selected[all_qrows]]
        if not len(qrows):
            continue
        pool = exact.get(key)
        if pool is None or len(pool) < len(qrows):
            pool = sex_only.get((key[0],))
        if pool is None or len(pool) < len(qrows):
            pool = all_release
        lr, lc = linear_sum_assignment(cost[np.ix_(qrows, pool)])
        assigned[qrows[lr]] = pool[lc]
    if int(np.count_nonzero(assigned >= 0)) != int(np.count_nonzero(selected)):
        raise RuntimeError("unique block linker failed")
    return assigned


def age_pair_selection(challenge: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    work = challenge.copy()
    work["age_band"] = np.floor((work["age"].to_numpy(dtype=float) - 40.0) / 5.0).astype(int)
    ids = work["challenge_row_id"].to_numpy(dtype=int)
    selected = np.zeros(len(work), dtype=bool)
    for rows in group_indices(work, ("sex", "prefecture", "age_band")).values():
        if len(rows) % 2:
            # Production groups are even.  Fall back to the proven half quota in
            # an unexpectedly odd synthetic group.
            order = rows[np.lexsort((ids[rows], -score[rows]))]
            selected[order[: len(rows) // 2]] = True
            continue
        order = rows[np.lexsort((ids[rows], work["age"].to_numpy()[rows]))]
        for index in range(0, len(order), 2):
            pair = order[index : index + 2]
            winner = pair[np.lexsort((ids[pair], -score[pair]))[0]]
            selected[winner] = True
    return selected


def density_ratio_score(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    ids: np.ndarray,
) -> np.ndarray:
    components: list[np.ndarray] = []
    for columns in (NO_HDL, STABLE3, AGE_SBP_ALT):
        rx = raw_embed(release, background, columns)
        qx = raw_embed(challenge, background, columns)
        bx = raw_embed(background, background, columns)
        rcost = block_cost(release, challenge, rx, qx)
        bcost = block_cost(background, challenge, bx, qx)
        for k in (1, 3, 8):
            rd = np.partition(rcost, kth=min(k - 1, rcost.shape[1] - 1), axis=1)[:, k - 1]
            bd = np.partition(bcost, kth=min(k - 1, bcost.shape[1] - 1), axis=1)[:, k - 1]
            evidence = len(columns) * np.log((bd + 1e-8) / (rd + 1e-8))
            components.append(strict_rank(evidence, ids))
    matrix = np.column_stack(components)
    return 0.65 * matrix.mean(axis=1) + 0.35 * np.sort(matrix, axis=1)[:, -2:].mean(axis=1)


def global_density_ratio_scores(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """C-vs-A_bg kNN density ratio without trusting categorical row links.

    This is intentionally global: production challenges have one prefecture and
    a highly skewed sex mix while releases spread both fields broadly.  The
    returned scores are rank-normalized within a challenge so views with
    different dimensions can be combined without hand-tuned scale factors.
    """
    views = {
        "all5": QI,
        "nohdl": NO_HDL,
        "stable3": STABLE3,
        "labs": ("SBP", "TG", "ALT"),
    }
    output: dict[str, np.ndarray] = {}
    ranked_views: list[np.ndarray] = []
    for name, columns in views.items():
        rx = raw_embed(release, background, columns)
        qx = raw_embed(challenge, background, columns)
        bx = raw_embed(background, background, columns)
        release_cost = cdist(qx, rx)
        background_cost = cdist(qx, bx)
        components: list[np.ndarray] = []
        for k in (1, 3, 8):
            rk = min(k - 1, release_cost.shape[1] - 1)
            bk = min(k - 1, background_cost.shape[1] - 1)
            rd = np.partition(release_cost, kth=rk, axis=1)[:, rk]
            bd = np.partition(background_cost, kth=bk, axis=1)[:, bk]
            components.append(
                len(columns) * np.log((bd + 1e-6) / (rd + 1e-6))
            )
        raw_score = np.mean(np.column_stack(components), axis=1)
        ranked = strict_rank(raw_score, ids)
        output[name] = ranked
        ranked_views.append(ranked)
    matrix = np.column_stack(ranked_views)
    output["ensemble"] = (
        0.65 * matrix.mean(axis=1)
        + 0.35 * np.sort(matrix, axis=1)[:, -2:].mean(axis=1)
    )
    return output


def make_frame(target: str, ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_k": int(target),
            "challenge_row_id": ids,
            "time_hat": values,
        }
    )


def mode_links(link_arrays: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(link_arrays)
    output = np.empty(len(matrix), dtype=int)
    for row, values in enumerate(matrix):
        counts = Counter(int(value) for value in values)
        best = max(counts.values())
        choices = {value for value, count in counts.items() if count == best}
        output[row] = next(int(value) for value in values if int(value) in choices)
    return output


def build_variants(
    background: pd.DataFrame,
    releases: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    v4_aia: pd.DataFrame,
    independent_aia: pd.DataFrame,
    structured: dict[str, pd.DataFrame],
    *,
    abstain: float,
    horizon: float,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    names = (
        "current_horizon",
        "current_copula_all_time",
        "current_copula_censor_time",
        "current_copula_mutual_time",
        "current_copula_unique_time",
        "current_bgcdf_all_time",
        "current_vote_all_time",
        "current_vote_mutual_time",
        "current_changed_only_copula",
        "current_changed_mutual_copula",
        "nohdl_horizon",
        "nohdl_copula_all_time",
        "nohdl_copula_mutual_time",
        "nohdl_agepair_horizon",
        "nohdl_agepair_copula_time",
        "density_horizon",
        "density_copula_all_time",
        "density_agepair_horizon",
        "pure_density_horizon",
        "global_consensus_horizon",
        "global_consensus_copula_time",
        "global_consensus_mutual_time",
        "global_consensus_changed_only",
        "global_pure_horizon",
        "global_pure_copula_time",
        "global_bgcdf_horizon",
        "global_bgcdf_copula_time",
        "global_density_ensemble_horizon",
        "global_density_all5_horizon",
        "global_density_nohdl_horizon",
        "global_density_stable3_horizon",
        "global_density_labs_horizon",
        "global_density_ensemble_copula_time",
        "sex_consensus_horizon",
        "sex_pure_horizon",
        "sex_bgcdf_horizon",
        "sex_consensus_copula_time",
        "sex_consensus_censor_time",
        "global_density_consensus_horizon",
        "global_density_consensus_censor_time",
    )
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    diagnostics: list[dict[str, object]] = []
    current_frame = structured["consensus_v4_copula__horizon"]
    for target in sorted(releases, key=int):
        release = releases[target].reset_index(drop=True)
        challenge = challenges[target].reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_block = v4_aia.loc[v4_aia["target_k"].astype(int).eq(int(target))].copy()
        ind_block = independent_aia.loc[independent_aia["target_k"].astype(int).eq(int(target))].copy()
        cur_block = current_frame.loc[current_frame["target_k"].astype(int).eq(int(target))].copy()
        for block in (v4_block, ind_block, cur_block):
            block.index = block["challenge_row_id"].astype(int)
        v4_time = v4_block.loc[ids, "time_hat"].to_numpy(dtype=float)
        ind_time = ind_block.loc[ids, "time_hat"].to_numpy(dtype=float)
        current_time = cur_block.loc[ids, "time_hat"].to_numpy(dtype=float)
        active_v4 = ~np.isclose(v4_time, abstain)
        active_ind = ~np.isclose(ind_time, abstain)
        selected_current = ~np.isclose(current_time, abstain)

        geometries: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {
            "copula_all5": (midrank(release, QI), midrank(challenge, QI), QI),
            "copula_nohdl": (midrank(release, NO_HDL), midrank(challenge, NO_HDL), NO_HDL),
            "copula_stable3": (midrank(release, STABLE3), midrank(challenge, STABLE3), STABLE3),
            "copula_age_sbp_alt": (midrank(release, AGE_SBP_ALT), midrank(challenge, AGE_SBP_ALT), AGE_SBP_ALT),
            "bgcdf_nohdl": (midrank(release, NO_HDL), background_cdf(challenge, background, NO_HDL), NO_HDL),
            "raw_nohdl": (raw_embed(release, background, NO_HDL), raw_embed(challenge, background, NO_HDL), NO_HDL),
        }
        costs = {
            name: block_cost(release, challenge, rx, qx)
            for name, (rx, qx, _) in geometries.items()
        }
        costs["global_copula_nohdl"] = cdist(
            geometries["copula_nohdl"][1], geometries["copula_nohdl"][0]
        )
        costs["global_copula_all5"] = cdist(
            geometries["copula_all5"][1], geometries["copula_all5"][0]
        )
        costs["global_bgcdf_nohdl"] = cdist(
            geometries["bgcdf_nohdl"][1], geometries["bgcdf_nohdl"][0]
        )
        costs["global_raw_nohdl"] = cdist(
            geometries["raw_nohdl"][1], geometries["raw_nohdl"][0]
        )
        costs["sex_copula_nohdl"] = sex_cost(
            release,
            challenge,
            geometries["copula_nohdl"][0],
            geometries["copula_nohdl"][1],
        )
        costs["sex_bgcdf_nohdl"] = sex_cost(
            release,
            challenge,
            geometries["bgcdf_nohdl"][0],
            geometries["bgcdf_nohdl"][1],
        )
        links = {name: link_details(cost) for name, cost in costs.items()}
        copula = links["copula_nohdl"]
        bgcdf = links["bgcdf_nohdl"]
        vote = mode_links(
            [
                links["copula_nohdl"]["nearest"],
                links["copula_all5"]["nearest"],
                links["copula_stable3"]["nearest"],
                links["copula_age_sbp_alt"]["nearest"],
                links["bgcdf_nohdl"]["nearest"],
                links["raw_nohdl"]["nearest"],
            ]
        )
        vote_support = np.max(
            np.apply_along_axis(
                lambda values: np.bincount(values, minlength=len(release)),
                1,
                np.column_stack(
                    [
                        links["copula_nohdl"]["nearest"],
                        links["copula_all5"]["nearest"],
                        links["copula_stable3"]["nearest"],
                        links["copula_age_sbp_alt"]["nearest"],
                        links["bgcdf_nohdl"]["nearest"],
                        links["raw_nohdl"]["nearest"],
                    ]
                ),
            ),
            axis=1,
        )
        release_time = release["time"].to_numpy(dtype=float)
        release_event = (
            release["onset"].to_numpy(dtype=int)
            + release["death"].to_numpy(dtype=int)
        ) > 0

        copula_time = release_time[copula["nearest"]]
        copula_censor = np.where(release_event[copula["nearest"]], copula_time, horizon)
        bgcdf_time = release_time[bgcdf["nearest"]]
        vote_time = release_time[vote]
        unique = unique_links_by_block(
            costs["copula_nohdl"], selected_current, release, challenge
        )
        unique_time = np.full(len(challenge), horizon, dtype=float)
        unique_time[selected_current] = release_time[unique[selected_current]]

        proximity = strict_rank(-copula["first"], ids)
        density = density_ratio_score(release, challenge, background, ids)
        nohdl_score = 1.10 * active_v4 + 0.90 * active_ind + 0.40 * proximity
        density_score = (
            1.10 * active_v4 + 0.90 * active_ind + 0.20 * proximity + 0.35 * density
        )
        selected_nohdl, _ = stratified_half_selection(challenge, nohdl_score)
        selected_density, _ = stratified_half_selection(challenge, density_score)
        selected_pure_density, _ = stratified_half_selection(challenge, density)
        selected_agepair = age_pair_selection(challenge, nohdl_score)
        selected_density_pair = age_pair_selection(challenge, density_score)

        global_copula = links["global_copula_nohdl"]
        global_bgcdf = links["global_bgcdf_nohdl"]
        global_proximity = strict_rank(-global_copula["first"], ids)
        global_bgcdf_proximity = strict_rank(-global_bgcdf["first"], ids)
        global_consensus_score = (
            1.10 * active_v4
            + 0.90 * active_ind
            + 0.55 * global_proximity
        )
        selected_global_consensus, _ = stratified_half_selection(
            challenge, global_consensus_score
        )
        selected_global_pure, _ = stratified_half_selection(
            challenge, global_proximity
        )
        selected_global_bgcdf, _ = stratified_half_selection(
            challenge, global_bgcdf_proximity
        )
        global_density = global_density_ratio_scores(
            release, challenge, background, ids
        )
        selected_global_density = {
            name: stratified_half_selection(challenge, score)[0]
            for name, score in global_density.items()
        }
        selected_global_density_consensus, _ = stratified_half_selection(
            challenge,
            1.10 * active_v4
            + 0.90 * active_ind
            + 0.55 * global_density["ensemble"],
        )
        global_copula_time = release_time[global_copula["nearest"]]
        global_bgcdf_time = release_time[global_bgcdf["nearest"]]
        sex_copula = links["sex_copula_nohdl"]
        sex_bgcdf = links["sex_bgcdf_nohdl"]
        sex_proximity = strict_rank(-sex_copula["first"], ids)
        sex_bgcdf_proximity = strict_rank(-sex_bgcdf["first"], ids)
        selected_sex_consensus, _ = stratified_half_selection(
            challenge,
            1.10 * active_v4 + 0.90 * active_ind + 0.55 * sex_proximity,
        )
        selected_sex_pure, _ = stratified_half_selection(challenge, sex_proximity)
        selected_sex_bgcdf, _ = stratified_half_selection(
            challenge, sex_bgcdf_proximity
        )
        sex_copula_time = release_time[sex_copula["nearest"]]
        sex_copula_censor = np.where(
            release_event[sex_copula["nearest"]], sex_copula_time, horizon
        )

        def decoded(selected: np.ndarray, active_values: np.ndarray) -> np.ndarray:
            output = np.full(len(challenge), abstain, dtype=float)
            output[selected] = active_values[selected]
            return output

        values = {
            "current_horizon": decoded(selected_current, np.full(len(challenge), horizon)),
            "current_copula_all_time": decoded(selected_current, copula_time),
            "current_copula_censor_time": decoded(selected_current, copula_censor),
            "current_copula_mutual_time": decoded(
                selected_current,
                np.where(copula["mutual"], copula_time, horizon),
            ),
            "current_copula_unique_time": decoded(selected_current, unique_time),
            "current_bgcdf_all_time": decoded(selected_current, bgcdf_time),
            "current_vote_all_time": decoded(selected_current, vote_time),
            "current_vote_mutual_time": decoded(
                selected_current,
                np.where(vote_support >= 3, vote_time, horizon),
            ),
            "current_changed_only_copula": decoded(
                selected_current,
                np.where(active_v4, v4_time, copula_time),
            ),
            "current_changed_mutual_copula": decoded(
                selected_current,
                np.where(
                    active_v4,
                    v4_time,
                    np.where(copula["mutual"], copula_time, horizon),
                ),
            ),
            "nohdl_horizon": decoded(selected_nohdl, np.full(len(challenge), horizon)),
            "nohdl_copula_all_time": decoded(selected_nohdl, copula_time),
            "nohdl_copula_mutual_time": decoded(
                selected_nohdl, np.where(copula["mutual"], copula_time, horizon)
            ),
            "nohdl_agepair_horizon": decoded(selected_agepair, np.full(len(challenge), horizon)),
            "nohdl_agepair_copula_time": decoded(selected_agepair, copula_time),
            "density_horizon": decoded(selected_density, np.full(len(challenge), horizon)),
            "density_copula_all_time": decoded(selected_density, copula_time),
            "density_agepair_horizon": decoded(selected_density_pair, np.full(len(challenge), horizon)),
            "pure_density_horizon": decoded(selected_pure_density, np.full(len(challenge), horizon)),
            "global_consensus_horizon": decoded(
                selected_global_consensus, np.full(len(challenge), horizon)
            ),
            "global_consensus_copula_time": decoded(
                selected_global_consensus, global_copula_time
            ),
            "global_consensus_mutual_time": decoded(
                selected_global_consensus,
                np.where(global_copula["mutual"], global_copula_time, horizon),
            ),
            "global_consensus_changed_only": decoded(
                selected_global_consensus,
                np.where(active_v4, v4_time, global_copula_time),
            ),
            "global_pure_horizon": decoded(
                selected_global_pure, np.full(len(challenge), horizon)
            ),
            "global_pure_copula_time": decoded(
                selected_global_pure, global_copula_time
            ),
            "global_bgcdf_horizon": decoded(
                selected_global_bgcdf, np.full(len(challenge), horizon)
            ),
            "global_bgcdf_copula_time": decoded(
                selected_global_bgcdf, global_bgcdf_time
            ),
            "global_density_ensemble_horizon": decoded(
                selected_global_density["ensemble"], np.full(len(challenge), horizon)
            ),
            "global_density_all5_horizon": decoded(
                selected_global_density["all5"], np.full(len(challenge), horizon)
            ),
            "global_density_nohdl_horizon": decoded(
                selected_global_density["nohdl"], np.full(len(challenge), horizon)
            ),
            "global_density_stable3_horizon": decoded(
                selected_global_density["stable3"], np.full(len(challenge), horizon)
            ),
            "global_density_labs_horizon": decoded(
                selected_global_density["labs"], np.full(len(challenge), horizon)
            ),
            "global_density_ensemble_copula_time": decoded(
                selected_global_density["ensemble"], global_copula_time
            ),
            "sex_consensus_horizon": decoded(
                selected_sex_consensus, np.full(len(challenge), horizon)
            ),
            "sex_pure_horizon": decoded(
                selected_sex_pure, np.full(len(challenge), horizon)
            ),
            "sex_bgcdf_horizon": decoded(
                selected_sex_bgcdf, np.full(len(challenge), horizon)
            ),
            "sex_consensus_copula_time": decoded(
                selected_sex_consensus, sex_copula_time
            ),
            "sex_consensus_censor_time": decoded(
                selected_sex_consensus, sex_copula_censor
            ),
            "global_density_consensus_horizon": decoded(
                selected_global_density_consensus,
                np.full(len(challenge), horizon),
            ),
            "global_density_consensus_censor_time": decoded(
                selected_global_density_consensus,
                sex_copula_censor,
            ),
        }
        for name, time_hat in values.items():
            parts[name].append(make_frame(target, ids, time_hat))
        diagnostics.append(
            {
                "target": target,
                "current_nohdl_selection_changes": int(np.count_nonzero(selected_current != selected_nohdl)),
                "nohdl_agepair_changes": int(np.count_nonzero(selected_nohdl != selected_agepair)),
                "density_selection_changes": int(np.count_nonzero(selected_current != selected_density)),
                "copula_mutual": int(copula["mutual"].sum()),
                "copula_unique_nearest": int(np.unique(copula["nearest"]).size),
                "vote_support_ge3": int(np.count_nonzero(vote_support >= 3)),
                "global_copula_mutual": int(global_copula["mutual"].sum()),
                "global_copula_unique_nearest": int(
                    np.unique(global_copula["nearest"]).size
                ),
                "global_consensus_changes": int(
                    np.count_nonzero(selected_current != selected_global_consensus)
                ),
                "global_pure_changes": int(
                    np.count_nonzero(selected_current != selected_global_pure)
                ),
                "global_density_ensemble_changes": int(
                    np.count_nonzero(
                        selected_current != selected_global_density["ensemble"]
                    )
                ),
                "sex_consensus_changes": int(
                    np.count_nonzero(selected_current != selected_sex_consensus)
                ),
                "global_density_consensus_changes": int(
                    np.count_nonzero(
                        selected_current != selected_global_density_consensus
                    )
                ),
            }
        )
    return {
        name: pd.concat(blocks, ignore_index=True).sort_values(
            ["target_k", "challenge_row_id"], kind="mergesort"
        )
        for name, blocks in parts.items()
    }, diagnostics


def rank_warp(
    source: pd.DataFrame,
    template: pd.DataFrame,
    *,
    noise_fraction: float,
    noise_scales: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
    shuffle_time: bool,
) -> pd.DataFrame:
    output = source.copy()
    if noise_fraction:
        noise = rng.normal(
            0.0,
            noise_fraction * noise_scales,
            size=(len(output), len(VISIBLE_CONTINUOUS)),
        )
        for j, column in enumerate(VISIBLE_CONTINUOUS):
            output[column] = output[column].to_numpy(dtype=float) + noise[:, j]
    quantile = (np.arange(len(output), dtype=float) + 0.5) / len(output)
    for column in VISIBLE_CONTINUOUS:
        source_values = output[column].to_numpy(dtype=float)
        order = np.argsort(source_values, kind="mergesort")
        destination = np.quantile(
            template[column].to_numpy(dtype=float), quantile, method="linear"
        )
        warped = np.empty(len(output), dtype=float)
        warped[order] = destination
        output[column] = warped
    if shuffle_time:
        outcomes = output[["time", "onset", "death"]].to_numpy(copy=True)
        output.loc[:, ["time", "onset", "death"]] = outcomes[
            rng.permutation(len(output))
        ]
    output = shadow.coerce_release(output, cfg)
    return output.iloc[rng.permutation(len(output))].reset_index(drop=True)


def scenario_releases(
    world: shadow.ShadowWorld,
    scenario: str,
    *,
    seed: int,
    cfg: dict,
    template: pd.DataFrame | None,
    noise_scales: np.ndarray,
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for index, (target, source) in enumerate(world.aia_sources.items()):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, index, int(target), 9173, hash(scenario) & 0xFFFF])
        )
        if scenario.endswith("_cat_shuffled"):
            base_scenario = scenario.removesuffix("_cat_shuffled")
        elif scenario.endswith("_pref_shuffled"):
            base_scenario = scenario.removesuffix("_pref_shuffled")
        elif scenario.endswith("_sex_shuffled"):
            base_scenario = scenario.removesuffix("_sex_shuffled")
        else:
            base_scenario = scenario
        if base_scenario in {"raw", "noise_0p03", "noise_0p10", "time_shuffled", "independent"}:
            result[target] = shadow.scenario_release(
                source, world.attack_background, base_scenario, cfg, rng
            )
        elif base_scenario.startswith("warp"):
            if template is None:
                raise ValueError("warp scenario requires a production template")
            result[target] = rank_warp(
                source,
                template,
                noise_fraction=0.03 if "noise_0p03" in base_scenario else 0.0,
                noise_scales=noise_scales,
                rng=rng,
                cfg=cfg,
                shuffle_time=base_scenario.endswith("time_shuffled"),
            )
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        if scenario.endswith("_cat_shuffled"):
            # Production releases spread both categorical keys across nearly all
            # prefectures and roughly balance sex, whereas each challenge uses a
            # single prefecture and is strongly sex-skewed.  Destroying these two
            # row links is therefore an essential stress case.
            for column in BLOCK:
                result[target][column] = result[target][column].to_numpy()[
                    rng.permutation(len(result[target]))
                ]
        elif scenario.endswith("_pref_shuffled"):
            result[target]["prefecture"] = result[target]["prefecture"].to_numpy()[
                rng.permutation(len(result[target]))
            ]
        elif scenario.endswith("_sex_shuffled"):
            result[target]["sex"] = result[target]["sex"].to_numpy()[
                rng.permutation(len(result[target]))
            ]
    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidates = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(dist)
    cfg = load_config(KIT_CONFIG_PATH)
    templates = {
        target: pd.read_csv(target_dir / f"C_{target}.csv")
        for target in args.templates
    }
    noise_scales = background.loc[:, VISIBLE_CONTINUOUS].std(ddof=0).to_numpy(dtype=float)
    spec = shadow.ShadowSpec(
        n_targets=args.shadow_targets,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=25 * args.shadow_targets,
        minimum_shared=25,
        maximum_shared=26,
    )
    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    abstain = v2.safe_abstain_time(horizon, tau)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    pair_quality: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"seed={seed}: build world", flush=True)
        world = shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        exactify_visible_control_matching(world)
        ref_dir = args.out_dir / "reference" / f"seed_{seed}"
        shadow.write_shadow_reference(world, ref_dir)
        # Structural diagnostic independent of any attack score.
        for target, challenge in world.challenges.items():
            truth = world.aia_truth[target].set_index("challenge_row_id")
            ids = challenge["challenge_row_id"].to_numpy(dtype=int)
            labels = truth.loc[ids, "label"].eq("member").to_numpy()
            work = challenge.copy()
            work["age_band"] = np.floor((work["age"] - 40.0) / 5.0).astype(int)
            mixed = total = 0
            for block_rows in group_indices(work, ("sex", "prefecture", "age_band")).values():
                order = block_rows[np.lexsort((ids[block_rows], work["age"].to_numpy()[block_rows]))]
                for index in range(0, len(order) - 1, 2):
                    pair = order[index : index + 2]
                    mixed += int(labels[pair].sum() == 1)
                    total += 1
            pair_quality.append(
                {
                    "seed": seed,
                    "target": target,
                    "mixed_pair_fraction": mixed / total,
                    "pairs": total,
                }
            )

        for scenario in args.scenarios:
            template_ids = args.templates if scenario.startswith("warp") else [None]
            for template_id in template_ids:
                label = scenario if template_id is None else f"{scenario}_C{template_id}"
                print(f"seed={seed} scenario={label}", flush=True)
                releases = scenario_releases(
                    world,
                    scenario,
                    seed=seed,
                    cfg=cfg,
                    template=(templates[template_id] if template_id is not None else None),
                    noise_scales=noise_scales,
                )
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
                        world.attack_background, horizon=horizon, tau=tau
                    ),
                    anomalous_event_rate=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
                    anomalous_link_trust_distance=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
                    integrity_auc_threshold=v4.DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
                )
                independent, _, _ = build_aia_independent(
                    world.attack_background,
                    releases,
                    world.challenges,
                    member_count=spec.aia_members,
                    horizon=horizon,
                    abstain_time=abstain,
                )
                structured, _ = build_aia_v5_variants(
                    baseline,
                    independent,
                    releases,
                    world.challenges,
                    member_count=spec.aia_members,
                    abstain_time=abstain,
                    horizon_time=horizon,
                )
                variants, detail = build_variants(
                    world.attack_background,
                    releases,
                    world.challenges,
                    baseline,
                    independent,
                    structured,
                    abstain=abstain,
                    horizon=horizon,
                )
                for name, frame in {
                    "v4_reference": baseline,
                    "independent_reference": independent,
                    **variants,
                }.items():
                    score = score_aia_power(frame, ref_dir, cfg)
                    rows.append(
                        {
                            "seed": seed,
                            "scenario": label,
                            "variant": name,
                            "power": float(score["power"]),
                            "all": float(score["all"]),
                            "rare": float(score["rare"]),
                            "p_m": float(score["p_m"]),
                            "p_c": float(score["p_c"]),
                            "per_target": json.dumps(score["per_target"], sort_keys=True),
                        }
                    )
                for record in detail:
                    diagnostics.append(
                        {"seed": seed, "scenario": label, **record}
                    )

    result = pd.DataFrame(rows)
    result.to_csv(args.out_dir / "scores.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.out_dir / "diagnostics.csv", index=False)
    pair_frame = pd.DataFrame(pair_quality)
    pair_frame.to_csv(args.out_dir / "age_pair_quality.csv", index=False)
    pivot = result.pivot_table(
        index=["seed", "scenario"], columns="variant", values="power"
    )
    delta = pivot.sub(pivot["current_horizon"], axis=0)
    summary = (
        delta.stack(future_stack=True)
        .rename("delta")
        .reset_index()
        .groupby("variant", as_index=False)["delta"]
        .agg(["mean", "min", "max", "std"])
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(args.out_dir / "delta_summary.csv")
    print(summary.head(20).round(6).to_string(), flush=True)
    print(
        "age-pair mixed fraction",
        pair_frame["mixed_pair_fraction"].mean(),
        flush=True,
    )


if __name__ == "__main__":
    main()
