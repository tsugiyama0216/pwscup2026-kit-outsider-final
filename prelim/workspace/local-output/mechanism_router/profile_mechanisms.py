"""Profile public preliminary releases for AIA mechanism routing.

This is an exploratory, read-only analysis of organizer-distributed CSVs.  It
writes compact diagnostics under the ignored local-output directory and never
touches submission candidates.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
BACKGROUND_PATH = ROOT / "participant_data" / "A_bg.csv"
OUT = ROOT / "local-output" / "mechanism_router"

TARGET_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
QI = ("age", "SBP", "TG", "HDL", "ALT")
CONT = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
BLOCK = ("sex", "prefecture")
FULL = (*CONT, "smoking", "time", "onset", "death")
ABSTAIN = 11.0


def robust_center_scale(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    return center, np.where(scale > 1e-9, scale, 1.0)


def transform(frame: pd.DataFrame, columns: tuple[str, ...], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in {"TG", "ALT", "FPG"}:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
    return (values - center) / scale / np.sqrt(len(columns))


def transformed_center_scale(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in {"TG", "ALT", "FPG"}:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    return center, np.where(scale > 1e-9, scale, 1.0)


def blockwise_nearest(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    columns: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
    *,
    neighbors: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx = transform(reference, columns, center, scale)
    qx = transform(query, columns, center, scale)
    first = np.empty(len(query), dtype=float)
    second = np.empty(len(query), dtype=float)
    nearest = np.empty(len(query), dtype=int)
    ref_sex = reference["sex"].to_numpy()
    ref_pref = reference["prefecture"].astype(str).to_numpy()
    groups = query.groupby(list(BLOCK), sort=False, dropna=False).indices
    for key, raw_rows in groups.items():
        sex, pref = key if isinstance(key, tuple) else (key, "")
        rows = np.asarray(raw_rows, dtype=int)
        pool = np.flatnonzero((ref_sex == sex) & (ref_pref == str(pref)))
        if not len(pool):
            pool = np.flatnonzero(ref_sex == sex)
        if not len(pool):
            pool = np.arange(len(reference))
        k = min(neighbors, len(pool))
        model = NearestNeighbors(n_neighbors=k).fit(rx[pool])
        distance, index = model.kneighbors(qx[rows])
        first[rows] = distance[:, 0]
        second[rows] = distance[:, -1]
        nearest[rows] = pool[index[:, 0]]
    return first, second, nearest


def strict_rank(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    order = np.lexsort((ids, values))
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(1, len(values) + 1) / (len(values) + 1)
    return result


def lower_tail(values: np.ndarray, null: np.ndarray) -> np.ndarray:
    ordered = np.sort(null)
    return (np.searchsorted(ordered, values, side="right") + 1) / (len(ordered) + 1)


def age_band(values: pd.Series) -> np.ndarray:
    return np.floor((values.to_numpy(dtype=float) - 40.0) / 5.0).astype(int)


def candidate_active(path: Path, target_id: int, ids: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    block = frame.loc[frame["target_k"].astype(int).eq(target_id)].copy()
    block = block.set_index(block["challenge_row_id"].astype(int))
    times = block.loc[ids, "time_hat"].to_numpy(dtype=float)
    return ~np.isclose(times, ABSTAIN)


def quota_violations(challenge: pd.DataFrame, active: np.ndarray) -> tuple[int, int, int]:
    work = challenge.assign(_band=age_band(challenge["age"]), _active=active.astype(int))
    bad = 0
    abs_error = 0
    odd = 0
    for _, group in work.groupby(["sex", "prefecture", "_band"], sort=True):
        n = len(group)
        a = int(group["_active"].sum())
        if n % 2:
            odd += 1
            okay = a in {n // 2, n // 2 + 1}
            abs_error += min(abs(a - n // 2), abs(a - (n // 2 + 1)))
        else:
            okay = a == n // 2
            abs_error += abs(a - n // 2)
        bad += int(not okay)
    return bad, abs_error, odd


def quantile_embed(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    # Dataset-wise empirical ranks remove monotone coordinate transforms.
    values = np.empty((len(frame), len(columns)), dtype=float)
    ids = np.arange(len(frame))
    for j, column in enumerate(columns):
        raw = frame[column].to_numpy(dtype=float)
        values[:, j] = strict_rank(raw, ids)
    return values / np.sqrt(len(columns))


def quantile_block_nearest(reference: pd.DataFrame, query: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rx = quantile_embed(reference, QI)
    qx = quantile_embed(query, QI)
    first = np.empty(len(query), dtype=float)
    nearest = np.empty(len(query), dtype=int)
    ref_sex = reference["sex"].to_numpy()
    ref_pref = reference["prefecture"].astype(str).to_numpy()
    for key, raw_rows in query.groupby(list(BLOCK), sort=False, dropna=False).indices.items():
        sex, pref = key if isinstance(key, tuple) else (key, "")
        rows = np.asarray(raw_rows, dtype=int)
        pool = np.flatnonzero((ref_sex == sex) & (ref_pref == str(pref)))
        if not len(pool):
            pool = np.flatnonzero(ref_sex == sex)
        if not len(pool):
            pool = np.arange(len(reference))
        model = NearestNeighbors(n_neighbors=1).fit(rx[pool])
        dist, idx = model.kneighbors(qx[rows])
        first[rows] = dist[:, 0]
        nearest[rows] = pool[idx[:, 0]]
    return first, nearest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    background = pd.read_csv(BACKGROUND_PATH)
    releases = {target: pd.read_csv(DATA / f"C_{target}.csv") for target in TARGET_IDS}
    challenges = {target: pd.read_csv(DATA / f"aia_challenge_{target}.csv") for target in TARGET_IDS}

    raw_center, raw_scale = transformed_center_scale(background, QI)
    full_center, full_scale = transformed_center_scale(background, CONT)
    v4_path = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v4" / "F_aia.csv"
    ind_path = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v5-probe-independent" / "F_aia.csv"
    struct_path = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v5-structured" / "F_aia.csv"

    profile_rows: list[dict[str, float | int]] = []
    row_scores: list[pd.DataFrame] = []
    for target in TARGET_IDS:
        release = releases[target]
        challenge = challenges[target]
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        q_first, q_second, q_nearest = blockwise_nearest(
            release, challenge, QI, raw_center, raw_scale
        )
        bg_first, _, _ = blockwise_nearest(release, background, QI, raw_center, raw_scale, neighbors=1)
        strength = -np.log10(np.clip(lower_tail(q_first, bg_first), 1e-12, 1.0))
        strength += 0.1 * np.log1p(q_second / np.clip(q_first, 1e-12, None))
        qrank_dist, qrank_nearest = quantile_block_nearest(release, challenge)

        # Cross-view nearest-link agreement gives a public linkage-coherence signal.
        view_nearest = []
        view_dist = []
        for columns in (QI, ("age", "SBP", "HDL"), ("SBP", "TG", "HDL", "ALT")):
            center, scale = transformed_center_scale(background, columns)
            distance, _, nearest = blockwise_nearest(release, challenge, columns, center, scale)
            view_nearest.append(nearest)
            view_dist.append(strict_rank(-distance, ids))
        view_nearest_m = np.column_stack(view_nearest)
        agreement = np.max(
            np.apply_along_axis(lambda x: np.bincount(x, minlength=len(release)), 1, view_nearest_m),
            axis=1,
        ) / len(view_nearest)
        multiview = np.mean(np.column_stack(view_dist), axis=1)
        release_to_query = np.argmin(
            cdist(transform(challenge, QI, raw_center, raw_scale), transform(release, QI, raw_center, raw_scale)),
            axis=0,
        )
        mutual = release_to_query[q_nearest] == np.arange(len(challenge))

        active = {
            "v4": candidate_active(v4_path, target, ids),
            "ind": candidate_active(ind_path, target, ids),
            "struct": candidate_active(struct_path, target, ids),
        }
        scores = pd.DataFrame(
            {
                "target": target,
                "challenge_row_id": ids,
                "raw_distance": q_first,
                "raw_strength": strength,
                "quantile_distance": qrank_dist,
                "multiview_rank": multiview,
                "agreement": agreement,
                "mutual": mutual.astype(int),
                "v4": active["v4"].astype(int),
                "ind": active["ind"].astype(int),
                "struct": active["struct"].astype(int),
            }
        )
        row_scores.append(scores)

        top = np.argsort(-strength)[:163]
        bottom = np.argsort(-strength)[163:]
        # Two-component mixture separation on log distance; this is descriptive,
        # not treated as membership truth.
        log_distance = np.log(np.clip(q_first, 1e-12, None)).reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(log_distance)
        means = np.sort(gm.means_.ravel())
        pooled_sd = np.sqrt(np.mean(gm.covariances_.ravel()))
        mixture_sep = float((means[1] - means[0]) / max(pooled_sd, 1e-9))

        row: dict[str, float | int] = {
            "target": target,
            "release_exact_duplicates": int(release.duplicated().sum()),
            "release_unique_full": int(len(release.drop_duplicates())),
            "cross_exact_any": 0,
            "bg_exact_full": 0,
            "raw_distance_median": float(np.median(q_first)),
            "raw_distance_p10": float(np.quantile(q_first, 0.10)),
            "raw_distance_p50_bg": float(np.median(bg_first)),
            "raw_tail_p_le_001": int(np.count_nonzero(lower_tail(q_first, bg_first) <= 0.01)),
            "raw_tail_p_le_005": int(np.count_nonzero(lower_tail(q_first, bg_first) <= 0.05)),
            "raw_strength_top_bottom_gap": float(np.mean(strength[top]) - np.mean(strength[bottom])),
            "boundary_strength_gap": float(np.sort(strength)[-163] - np.sort(strength)[-164]),
            "mixture_separation": mixture_sep,
            "mutual_rows": int(np.count_nonzero(mutual)),
            "view_agreement_1": int(np.count_nonzero(np.isclose(agreement, 1.0))),
            "quantile_distance_median": float(np.median(qrank_dist)),
            "quantile_distance_p10": float(np.quantile(qrank_dist, 0.10)),
            "raw_quantile_score_rho": float(pd.Series(-q_first).corr(pd.Series(-qrank_dist), method="spearman")),
            "c_event_rate": float(((release["onset"] + release["death"]) > 0).mean()),
            "c_onset_rate": float(release["onset"].mean()),
            "c_death_rate": float(release["death"].mean()),
            "c_horizon_rate": float(np.isclose(release["time"], 10.0, atol=0.01).mean()),
        }
        release_keys = set(map(tuple, release[list(FULL)].to_numpy()))
        bg_keys = set(map(tuple, background[list(FULL)].to_numpy()))
        row["bg_exact_full"] = len(release_keys & bg_keys)
        other_keys: set[tuple[object, ...]] = set()
        for other, other_release in releases.items():
            if other != target:
                other_keys.update(map(tuple, other_release[list(FULL)].to_numpy()))
        row["cross_exact_any"] = len(release_keys & other_keys)

        for column in QI:
            c_values = release[column].to_numpy(dtype=float)
            q_values = challenge[column].to_numpy(dtype=float)
            row[f"unique_{column}"] = int(pd.Series(c_values).nunique())
            row[f"c_q_ks_{column}"] = float(ks_2samp(c_values, q_values).statistic)
            denom = max(np.std(background[column].to_numpy(dtype=float)), 1e-9)
            row[f"c_q_wass_{column}"] = float(wasserstein_distance(c_values, q_values) / denom)
            c_iqr = np.quantile(c_values, 0.75) - np.quantile(c_values, 0.25)
            q_iqr = np.quantile(q_values, 0.75) - np.quantile(q_values, 0.25)
            row[f"c_q_iqr_ratio_{column}"] = float(c_iqr / max(q_iqr, 1e-9))

        for name, mask in active.items():
            bad, error, odd = quota_violations(challenge, mask)
            row[f"{name}_quota_bad_strata"] = bad
            row[f"{name}_quota_abs_error"] = error
            row[f"{name}_raw_strength_mean"] = float(np.mean(strength[mask]))
            row[f"{name}_raw_distance_mean"] = float(np.mean(q_first[mask]))
            row[f"{name}_quantile_distance_mean"] = float(np.mean(qrank_dist[mask]))
            row[f"{name}_multiview_mean"] = float(np.mean(multiview[mask]))
            row[f"{name}_mutual"] = int(np.count_nonzero(mutual & mask))
            row[f"{name}_view_agree"] = float(np.mean(agreement[mask]))
        row["v4_ind_overlap"] = int(np.count_nonzero(active["v4"] & active["ind"]))
        row["v4_struct_overlap"] = int(np.count_nonzero(active["v4"] & active["struct"]))
        row["ind_struct_overlap"] = int(np.count_nonzero(active["ind"] & active["struct"]))
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).sort_values("target")
    profiles.to_csv(OUT / "mechanism_profiles.csv", index=False)
    pd.concat(row_scores, ignore_index=True).to_csv(OUT / "challenge_row_scores.csv", index=False)

    # Pairwise exact and near-match diagnostics across public releases.  Low
    # distances can reveal row-preserving/noise mechanisms and known-shared rows.
    pair_rows: list[dict[str, float | int]] = []
    for first_id, second_id in itertools.combinations(TARGET_IDS, 2):
        first = releases[first_id]
        second = releases[second_id]
        first_distance, _, _ = blockwise_nearest(first, second, CONT, full_center, full_scale, neighbors=1)
        second_distance, _, _ = blockwise_nearest(second, first, CONT, full_center, full_scale, neighbors=1)
        pair_rows.append(
            {
                "first": first_id,
                "second": second_id,
                "exact_full": len(set(map(tuple, first[list(FULL)].to_numpy())) & set(map(tuple, second[list(FULL)].to_numpy()))),
                "second_to_first_p01": float(np.quantile(first_distance, 0.01)),
                "second_to_first_p05": float(np.quantile(first_distance, 0.05)),
                "second_to_first_lt_001": int(np.count_nonzero(first_distance < 0.01)),
                "second_to_first_lt_005": int(np.count_nonzero(first_distance < 0.05)),
                "symmetric_lt_005": int(min(np.count_nonzero(first_distance < 0.05), np.count_nonzero(second_distance < 0.05))),
            }
        )
    pd.DataFrame(pair_rows).to_csv(OUT / "cross_release_pairs.csv", index=False)

    # Small stdout view for rapid inspection.
    keep = [
        "target", "release_unique_full", "cross_exact_any", "bg_exact_full",
        "raw_tail_p_le_001", "raw_tail_p_le_005", "mixture_separation",
        "raw_distance_median", "quantile_distance_median", "v4_ind_overlap",
        "v4_quota_abs_error", "ind_quota_abs_error", "v4_raw_strength_mean",
        "ind_raw_strength_mean", "struct_raw_strength_mean",
    ]
    print(profiles[keep].to_string(index=False))
    print(f"wrote {OUT / 'mechanism_profiles.csv'}")


if __name__ == "__main__":
    main()
