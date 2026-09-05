"""Production-only forensic diagnostics for the eight official-zero AIA targets.

This script is deliberately research-only.  It reads organizer-distributed
files and writes only beneath the ignored ``local-output`` tree.  No submission
artifact is produced.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
PRACTICE = ROOT / "participant_data"
OUT = ROOT / "local-output" / "production-zero-forensics"
TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
QI = ("age", "SBP", "TG", "HDL", "ALT")
VIEWS = {
    "all5": QI,
    "nohdl": ("age", "SBP", "TG", "ALT"),
    "stable3": ("age", "SBP", "TG"),
    "labs": ("SBP", "TG", "ALT"),
}
LOG_COLUMNS = {"TG", "ALT"}
FULL_COLUMNS = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
)


def age_band(frame: pd.DataFrame) -> np.ndarray:
    return np.floor((frame["age"].to_numpy(dtype=float) - 40.0) / 5.0).astype(int)


def robust_parameters(reference: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = reference.loc[:, columns].to_numpy(dtype=float, copy=True)
    for j, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, j] = np.log1p(np.clip(values[:, j], 0.0, None))
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-9, scale, np.std(values, axis=0, ddof=0))
    return center, np.where(scale > 1e-9, scale, 1.0)


def raw_embed(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    for j, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, j] = np.log1p(np.clip(values[:, j], 0.0, None))
    center, scale = robust_parameters(reference, columns)
    return (values - center) / scale / math.sqrt(len(columns))


def empirical_cdf_embed(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    output = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        ref = np.sort(reference[column].to_numpy(dtype=float))
        values = frame[column].to_numpy(dtype=float)
        lo = np.searchsorted(ref, values, side="left")
        hi = np.searchsorted(ref, values, side="right")
        output[:, j] = (0.5 * (lo + hi) + 0.5) / (len(ref) + 1.0)
    return output / math.sqrt(len(columns))


def self_rank_embed(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    output = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        output[:, j] = (
            frame[column].rank(method="average").to_numpy(dtype=float)
            / (len(frame) + 1.0)
        )
    return output / math.sqrt(len(columns))


def inverse_margin_embed(
    frame: pd.DataFrame,
    source_for_ranks: pd.DataFrame,
    reference: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    """Map source empirical ranks through a raw-background inverse CDF."""
    decoded = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        ranks = source_for_ranks[column].rank(method="average").to_numpy(dtype=float)
        probability = (ranks - 0.5) / len(source_for_ranks)
        decoded[:, j] = np.quantile(
            reference[column].to_numpy(dtype=float), probability, method="linear"
        )
    decoded_frame = pd.DataFrame(decoded, columns=columns)
    return raw_embed(decoded_frame, reference, columns)


def category_keys(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "global":
        return np.repeat("_all", len(frame))
    if mode == "pref":
        return frame["prefecture"].astype(str).to_numpy()
    if mode == "sex":
        return frame["sex"].astype(str).to_numpy()
    if mode == "both":
        return (
            frame["sex"].astype(str) + "|" + frame["prefecture"].astype(str)
        ).to_numpy()
    raise ValueError(mode)


def grouped_knn_distance(
    train_x: np.ndarray,
    query_x: np.ndarray,
    train: pd.DataFrame,
    query: pd.DataFrame,
    mode: str,
    *,
    k: int,
) -> np.ndarray:
    train_keys = category_keys(train, mode)
    query_keys = category_keys(query, mode)
    output = np.empty((len(query), k), dtype=float)
    all_rows = np.arange(len(train), dtype=int)
    for key in pd.unique(query_keys):
        qrows = np.flatnonzero(query_keys == key)
        pool = np.flatnonzero(train_keys == key)
        if len(pool) < k:
            pool = all_rows
        use_k = min(k, len(pool))
        distance = NearestNeighbors(n_neighbors=use_k).fit(train_x[pool]).kneighbors(
            query_x[qrows], return_distance=True
        )[0]
        if use_k < k:
            distance = np.pad(distance, ((0, 0), (0, k - use_k)), mode="edge")
        output[qrows] = distance
    return output


def stratified_selection(challenge: pd.DataFrame, score: np.ndarray) -> tuple[np.ndarray, dict[tuple[object, ...], tuple[np.ndarray, int, float]]]:
    work = challenge.copy()
    work["_band"] = age_band(work)
    ids = work["challenge_row_id"].to_numpy(dtype=int)
    selected = np.zeros(len(work), dtype=bool)
    details: dict[tuple[object, ...], tuple[np.ndarray, int, float]] = {}
    floor_total = 0
    odd: list[tuple[float, int, tuple[object, ...], int]] = []
    groups: list[tuple[tuple[object, ...], np.ndarray, np.ndarray]] = []
    for raw_key, raw_rows in work.groupby(["sex", "prefecture", "_band"], sort=True).indices.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        rows = np.asarray(raw_rows, dtype=int)
        order = rows[np.lexsort((ids[rows], -score[rows]))]
        groups.append((key, rows, order))
        quota = len(rows) // 2
        floor_total += quota
        if len(rows) % 2:
            odd.append((float(score[order[quota]]), -int(ids[order[quota]]), key, len(groups) - 1))
    extras = 163 - floor_total
    ceil_indices = {item[3] for item in sorted(odd, reverse=True)[:extras]}
    for index, (key, rows, order) in enumerate(groups):
        quota = len(rows) // 2 + int(index in ceil_indices)
        selected[order[:quota]] = True
        cutoff = float(score[order[quota - 1]]) if quota else math.inf
        details[key] = (rows, quota, cutoff)
    assert selected.sum() == 163
    return selected, details


def calibrated_control_rate(
    challenge: pd.DataFrame,
    details: dict[tuple[object, ...], tuple[np.ndarray, int, float]],
    control: pd.DataFrame,
    control_score: np.ndarray,
) -> tuple[float, float]:
    work = control.copy()
    work["_band"] = age_band(work)
    weighted = 0.0
    weight = 0
    exact_weight = 0
    for key, (_, quota, cutoff) in details.items():
        sex, pref, band = key
        exact = np.flatnonzero(
            work["sex"].eq(sex).to_numpy()
            & work["prefecture"].astype(str).eq(str(pref)).to_numpy()
            & work["_band"].eq(band).to_numpy()
        )
        pool = exact
        if len(pool) < 5:
            pool = np.flatnonzero(
                work["sex"].eq(sex).to_numpy() & work["_band"].eq(band).to_numpy()
            )
        if not len(pool):
            pool = np.arange(len(work), dtype=int)
        weighted += quota * float(np.mean(control_score[pool] >= cutoff))
        weight += quota
        exact_weight += quota * int(len(exact) >= 5)
    return weighted / weight, exact_weight / weight


def make_embeddings(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    control: pd.DataFrame,
    background_train: pd.DataFrame,
    columns: tuple[str, ...],
    geometry: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if geometry == "raw":
        return (
            raw_embed(release, background_train, columns),
            raw_embed(challenge, background_train, columns),
            raw_embed(control, background_train, columns),
            raw_embed(background_train, background_train, columns),
        )
    if geometry == "bgcdf":
        return (
            self_rank_embed(release, columns),
            empirical_cdf_embed(challenge, background_train, columns),
            empirical_cdf_embed(control, background_train, columns),
            empirical_cdf_embed(background_train, background_train, columns),
        )
    if geometry == "inverse":
        return (
            inverse_margin_embed(release, release, background_train, columns),
            raw_embed(challenge, background_train, columns),
            raw_embed(control, background_train, columns),
            raw_embed(background_train, background_train, columns),
        )
    raise ValueError(geometry)


def density_scores(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    control: pd.DataFrame,
    background_train: pd.DataFrame,
    columns: tuple[str, ...],
    geometry: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rx, qx, cx, bx = make_embeddings(
        release, challenge, control, background_train, columns, geometry
    )
    rd_q = grouped_knn_distance(rx, qx, release, challenge, mode, k=8)
    rd_c = grouped_knn_distance(rx, cx, release, control, mode, k=8)
    bd_q = grouped_knn_distance(bx, qx, background_train, challenge, mode, k=8)
    bd_c = grouped_knn_distance(bx, cx, background_train, control, mode, k=8)
    components_q = []
    components_c = []
    dimension = len(columns)
    for k in (1, 3, 8):
        components_q.append(dimension * np.log((bd_q[:, k - 1] + 1e-6) / (rd_q[:, k - 1] + 1e-6)))
        components_c.append(dimension * np.log((bd_c[:, k - 1] + 1e-6) / (rd_c[:, k - 1] + 1e-6)))
    return np.mean(np.column_stack(components_q), axis=1), np.mean(
        np.column_stack(components_c), axis=1
    )


def marginal_and_category_profiles(
    releases: dict[int, pd.DataFrame], challenges: dict[int, pd.DataFrame], background: pd.DataFrame, b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.concat([background, b], ignore_index=True)
    marginal_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for target in TARGETS:
        release = releases[target].copy()
        challenge = challenges[target].copy()
        release["_band"] = age_band(release)
        challenge["_band"] = age_band(challenge)
        for column in QI:
            marginal_rows.append(
                {
                    "target": target,
                    "column": column,
                    "release_unique": int(release[column].nunique()),
                    "release_vs_background_ks": float(ks_2samp(release[column], reference[column]).statistic),
                    "release_vs_challenge_ks": float(ks_2samp(release[column], challenge[column]).statistic),
                    "release_mean": float(release[column].mean()),
                    "challenge_mean": float(challenge[column].mean()),
                }
            )
        for columns in (("prefecture",), ("sex", "prefecture"), ("sex", "prefecture", "_band")):
            need = challenge.groupby(list(columns)).size() / 2.0
            have = release.groupby(list(columns)).size().reindex(need.index, fill_value=0)
            deficit = np.maximum(need - have, 0.0)
            coverage_rows.append(
                {
                    "target": target,
                    "keys": "+".join(columns).replace("_band", "age_band"),
                    "unavoidably_changed_members": float(deficit.sum()),
                    "unavoidably_changed_fraction": float(deficit.sum() / 163.0),
                    "undercovered_strata": int((have < need).sum()),
                    "challenge_strata": int(len(need)),
                    "release_target_pref_rows": int(release["prefecture"].eq(challenge["prefecture"].iloc[0]).sum()),
                }
            )
    return pd.DataFrame(marginal_rows), pd.DataFrame(coverage_rows)


def exact_cross_release_profiles(
    releases: dict[int, pd.DataFrame], challenges: dict[int, pd.DataFrame], background: pd.DataFrame, b: pd.DataFrame
) -> pd.DataFrame:
    all_ids = sorted(releases)
    column_sets = {
        "qi": QI,
        "qi_categories": (*QI, "sex", "prefecture"),
        "static": ("age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG"),
        "full": FULL_COLUMNS,
    }
    hashes: dict[str, dict[int, np.ndarray]] = {}
    sets: dict[str, dict[int, set[int]]] = {}
    for label, columns in column_sets.items():
        hashes[label] = {
            target: pd.util.hash_pandas_object(releases[target].loc[:, columns], index=False).to_numpy(dtype=np.uint64)
            for target in all_ids
        }
        sets[label] = {target: set(value.tolist()) for target, value in hashes[label].items()}
    ref_sets = {
        label: set(
            pd.util.hash_pandas_object(
                pd.concat([background.loc[:, columns], b.loc[:, columns]], ignore_index=True), index=False
            ).to_numpy(dtype=np.uint64).tolist()
        )
        for label, columns in column_sets.items()
        if set(columns).issubset(background.columns)
    }
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        record: dict[str, object] = {"target": target, "release_rows": len(releases[target])}
        for label in column_sets:
            other: set[int] = set()
            for other_target in all_ids:
                if other_target != target:
                    other.update(sets[label][other_target])
            values = hashes[label][target]
            record[f"cross_{label}_rows"] = int(sum(int(value) in other for value in values))
            record[f"internal_{label}_duplicate_occurrences"] = int(pd.Series(values).duplicated().sum())
            if label in ref_sets:
                record[f"practice_exact_{label}_rows"] = int(sum(int(value) in ref_sets[label] for value in values))
        challenge_hash = pd.util.hash_pandas_object(challenges[target].loc[:, QI], index=False).to_numpy(dtype=np.uint64)
        record["challenge_exact_qi_rows"] = int(sum(int(value) in sets["qi"][target] for value in challenge_hash))
        rows.append(record)
    return pd.DataFrame(rows)


def pair_order_profiles(challenges: dict[int, pd.DataFrame], background: pd.DataFrame, simulations: int = 250) -> pd.DataFrame:
    reference_x = raw_embed(background, background, QI)

    def greedy_stat(frame: pd.DataFrame, x: np.ndarray) -> tuple[float, float]:
        work = frame.copy()
        work["_band"] = age_band(work)
        distances: list[float] = []
        age_gaps: list[float] = []
        for raw_rows in work.groupby(["sex", "_band"], sort=True).indices.values():
            rows = np.asarray(raw_rows, dtype=int)
            if len(rows) % 2:
                rows = rows[:-1]
            if not len(rows):
                continue
            matrix = cdist(x[rows], x[rows])
            left, right = np.triu_indices(len(rows), k=1)
            order = np.argsort(matrix[left, right], kind="mergesort")
            unused = np.ones(len(rows), dtype=bool)
            for edge in order:
                a, b = int(left[edge]), int(right[edge])
                if unused[a] and unused[b]:
                    unused[a] = unused[b] = False
                    distances.append(float(matrix[a, b]))
                    age_gaps.append(abs(float(frame.iloc[rows[a]]["age"]) - float(frame.iloc[rows[b]]["age"])))
                    if not unused.any():
                        break
        return float(np.median(distances)), float(np.mean(age_gaps))

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(81731)
    bg_work = background.copy()
    bg_work["_band"] = age_band(bg_work)
    for target in TARGETS:
        challenge = challenges[target].sort_values("challenge_row_id").reset_index(drop=True)
        challenge_x = raw_embed(challenge, background, QI)
        n = len(challenge)
        band = age_band(challenge)
        sex = challenge["sex"].to_numpy()
        adjacent = np.mean((sex[::2] == sex[1::2]) & (band[::2] == band[1::2]))
        half = np.mean((sex[: n // 2] == sex[n // 2 :]) & (band[: n // 2] == band[n // 2 :]))
        observed_distance, observed_age_gap = greedy_stat(challenge, challenge_x)
        null_distance: list[float] = []
        null_age_gap: list[float] = []
        group_counts = challenge.assign(_band=band).groupby(["sex", "_band"]).size()
        for _ in range(simulations):
            chosen: list[int] = []
            for (sex_value, band_value), count in group_counts.items():
                pool = bg_work.index[
                    bg_work["sex"].eq(sex_value) & bg_work["_band"].eq(band_value)
                ].to_numpy(dtype=int)
                chosen.extend(rng.choice(pool, size=int(count), replace=False).tolist())
            sample = background.iloc[chosen].reset_index(drop=True)
            sample_x = reference_x[np.asarray(chosen, dtype=int)]
            d, a = greedy_stat(sample, sample_x)
            null_distance.append(d)
            null_age_gap.append(a)
        rows.append(
            {
                "target": target,
                "rowid_adjacent_same_stratum_fraction": float(adjacent),
                "rowid_halves_same_stratum_fraction": float(half),
                "greedy_pair_median_distance": observed_distance,
                "null_pair_distance_mean": float(np.mean(null_distance)),
                "null_pair_distance_p01": float(np.quantile(null_distance, 0.01)),
                "null_pair_distance_percentile": float(np.mean(np.asarray(null_distance) <= observed_distance)),
                "greedy_pair_mean_age_gap": observed_age_gap,
                "null_pair_age_gap_mean": float(np.mean(null_age_gap)),
            }
        )
    return pd.DataFrame(rows)


def time_link_profiles(releases: dict[int, pd.DataFrame], background: pd.DataFrame, permutations: int = 400) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(81732)
    reference_event_rate = float(((background["onset"] + background["death"]) > 0).mean())
    for target in TARGETS:
        release = releases[target].reset_index(drop=True)
        event = (release["onset"].to_numpy(dtype=int) + release["death"].to_numpy(dtype=int)) > 0
        x = raw_embed(release, background, QI)
        event_rows = np.flatnonzero(event)
        if len(event_rows) >= 3:
            matrix = cdist(x[event_rows], x[event_rows])
            np.fill_diagonal(matrix, np.inf)
            nearest = np.argmin(matrix, axis=1)
            times = release["time"].to_numpy(dtype=float)[event_rows]
            observed = float(np.mean(np.abs(times - times[nearest]) <= 0.5))
            null = np.empty(permutations, dtype=float)
            for index in range(permutations):
                permuted = times[rng.permutation(len(times))]
                null[index] = np.mean(np.abs(permuted - permuted[nearest]) <= 0.5)
            p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
        else:
            observed = math.nan
            null = np.array([math.nan])
            p_value = math.nan
        rows.append(
            {
                "target": target,
                "event_rows": int(event.sum()),
                "event_rate": float(event.mean()),
                "background_event_rate": reference_event_rate,
                "horizon_rate": float(np.mean(release["time"].to_numpy(dtype=float) >= 9.49)),
                "event_loo_nn_time_hit": observed,
                "event_time_shuffle_mean": float(np.nanmean(null)),
                "event_time_shuffle_p95": float(np.nanquantile(null, 0.95)),
                "event_time_link_p_value": p_value,
                "time_unique": int(release["time"].nunique()),
                "death_rate": float(release["death"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run_density_calibration(
    releases: dict[int, pd.DataFrame], challenges: dict[int, pd.DataFrame], background: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, object]] = []
    selections: list[pd.DataFrame] = []
    methods = [
        (view_name, columns, geometry, mode)
        for view_name, columns in VIEWS.items()
        for geometry in ("raw", "bgcdf", "inverse")
        for mode in ("global", "pref", "sex", "both")
    ]
    for target in TARGETS:
        release = releases[target]
        challenge = challenges[target]
        for split_seed in range(6):
            rng = np.random.default_rng(93000 + split_seed)
            held_mask = rng.random(len(background)) < 0.35
            train = background.loc[~held_mask].reset_index(drop=True)
            held = background.loc[held_mask].reset_index(drop=True)
            for view_name, columns, geometry, mode in methods:
                q_score, c_score = density_scores(
                    release, challenge, held, train, columns, geometry, mode
                )
                selected, detail = stratified_selection(challenge, q_score)
                rate, exact_weight = calibrated_control_rate(
                    challenge, detail, held, c_score
                )
                split_rows.append(
                    {
                        "target": target,
                        "split_seed": split_seed,
                        "view": view_name,
                        "geometry": geometry,
                        "category_mode": mode,
                        "null_control_selection_rate": rate,
                        "implied_selection_accuracy": 1.0 - rate,
                        "exact_pref_null_weight": exact_weight,
                    }
                )
                if split_seed == 0:
                    selections.append(
                        pd.DataFrame(
                            {
                                "target": target,
                                "view": view_name,
                                "geometry": geometry,
                                "category_mode": mode,
                                "challenge_row_id": challenge["challenge_row_id"],
                                "score": q_score,
                                "selected": selected.astype(int),
                            }
                        )
                    )
    splits = pd.DataFrame(split_rows)
    summary = (
        splits.groupby(["target", "view", "geometry", "category_mode"], as_index=False)
        .agg(
            null_rate_mean=("null_control_selection_rate", "mean"),
            null_rate_min=("null_control_selection_rate", "min"),
            null_rate_max=("null_control_selection_rate", "max"),
            implied_accuracy_mean=("implied_selection_accuracy", "mean"),
            exact_pref_null_weight=("exact_pref_null_weight", "mean"),
        )
        .sort_values(["target", "null_rate_mean"])
    )
    splits.to_csv(OUT / "density_calibration_splits.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(
        OUT / "density_calibration_selections.csv", index=False
    )
    return summary, splits


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_target_ids = sorted(
        int(path.stem.split("_")[-1]) for path in DATA.glob("C_*.csv")
    )
    releases = {target: pd.read_csv(DATA / f"C_{target}.csv") for target in all_target_ids}
    challenges = {
        target: pd.read_csv(DATA / f"aia_challenge_{target}.csv") for target in TARGETS
    }
    background = pd.read_csv(PRACTICE / "A_bg.csv").reset_index(drop=True)
    b = pd.read_csv(PRACTICE / "B_practice.csv").reset_index(drop=True)

    marginal, coverage = marginal_and_category_profiles(
        releases, challenges, background, b
    )
    exact = exact_cross_release_profiles(releases, challenges, background, b)
    pair = pair_order_profiles(challenges, background)
    time_link = time_link_profiles(releases, background)
    density, _ = run_density_calibration(releases, challenges, background)

    marginal.to_csv(OUT / "marginal_profiles.csv", index=False)
    coverage.to_csv(OUT / "category_coverage.csv", index=False)
    exact.to_csv(OUT / "exact_cross_release.csv", index=False)
    pair.to_csv(OUT / "pair_structure.csv", index=False)
    time_link.to_csv(OUT / "time_linkage.csv", index=False)
    density.to_csv(OUT / "density_calibration_summary.csv", index=False)

    top = density.groupby("target", group_keys=False).head(8)
    report = {
        "scope": "official public zero-score targets; no organizer truth",
        "targets": list(TARGETS),
        "top_null_calibrated_density_methods": top.to_dict(orient="records"),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("CATEGORY COVERAGE")
    print(coverage.round(4).to_string(index=False))
    print("\nEXACT/CROSS-RELEASE")
    print(exact.to_string(index=False))
    print("\nPAIR STRUCTURE")
    print(pair.round(4).to_string(index=False))
    print("\nTIME LINKAGE")
    print(time_link.round(4).to_string(index=False))
    print("\nTOP DENSITY METHODS")
    print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
