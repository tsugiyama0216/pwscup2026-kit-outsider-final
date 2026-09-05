"""Measure public known-nonmember filtering impact on structured AIA."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "mechanism_router"
TARGET_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
RELEASE_COLUMNS = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
)
NUMERIC_RELEASE_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time")
CATEGORICAL_RELEASE_COLUMNS = ("sex", "prefecture", "smoking", "onset", "death")
VIEWS = (
    ("age", "SBP", "TG", "HDL", "ALT"),
    ("age", "SBP", "HDL"),
    ("SBP", "TG", "HDL", "ALT"),
)


def rounded_keys(frame: pd.DataFrame, decimals: int = 10) -> pd.MultiIndex:
    work = frame.loc[:, RELEASE_COLUMNS].copy()
    work.loc[:, NUMERIC_RELEASE_COLUMNS] = work.loc[:, NUMERIC_RELEASE_COLUMNS].round(decimals)
    return pd.MultiIndex.from_frame(work)


def known_nonmember_masks(releases: dict[int, pd.DataFrame]) -> tuple[dict[int, np.ndarray], dict[int, dict[str, int]]]:
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    background_keys = set(rounded_keys(background).tolist())
    release_keys = {target: rounded_keys(frame) for target, frame in releases.items()}
    masks: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, int]] = {}
    for target, frame in releases.items():
        own = release_keys[target]
        bg = np.fromiter((key in background_keys for key in own), dtype=bool, count=len(frame))
        others: set[tuple[object, ...]] = set()
        for other, keys in release_keys.items():
            if other != target:
                others.update(keys.tolist())
        cross = np.fromiter((key in others for key in own), dtype=bool, count=len(frame))
        masks[target] = bg | cross
        diagnostics[target] = {
            "background_donor_rows": int(bg.sum()),
            "cross_release_rows": int(cross.sum()),
            "known_nonmember_union": int((bg | cross).sum()),
        }
    return masks, diagnostics


def block_nearest_known_margin(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    known: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return min known/nonknown distance and fraction of views preferring known."""
    known_dist_views: list[np.ndarray] = []
    other_dist_views: list[np.ndarray] = []
    release_sex = release["sex"].to_numpy()
    release_pref = release["prefecture"].astype(str).to_numpy()
    groups = challenge.groupby(["sex", "prefecture"], sort=False, dropna=False).indices
    for columns in VIEWS:
        rv = release.loc[:, columns].to_numpy(dtype=float, copy=True)
        qv = challenge.loc[:, columns].to_numpy(dtype=float, copy=True)
        for index, column in enumerate(columns):
            if column in {"TG", "ALT"}:
                rv[:, index] = np.log1p(np.clip(rv[:, index], 0.0, None))
                qv[:, index] = np.log1p(np.clip(qv[:, index], 0.0, None))
        joined = np.vstack((rv, qv))
        center = np.median(joined, axis=0)
        q25, q75 = np.quantile(joined, (0.25, 0.75), axis=0)
        scale = (q75 - q25) / 1.349
        fallback = np.std(joined, axis=0, ddof=0)
        scale = np.where(scale > 1e-9, scale, fallback)
        scale = np.where(scale > 1e-9, scale, 1.0)
        rx = (rv - center) / scale / np.sqrt(len(columns))
        qx = (qv - center) / scale / np.sqrt(len(columns))
        known_d = np.full(len(challenge), np.inf)
        other_d = np.full(len(challenge), np.inf)
        for raw_key, raw_rows in groups.items():
            sex, pref = raw_key if isinstance(raw_key, tuple) else (raw_key, "")
            rows = np.asarray(raw_rows, dtype=int)
            pool = np.flatnonzero((release_sex == sex) & (release_pref == str(pref)))
            if not len(pool):
                pool = np.flatnonzero(release_sex == sex)
            if not len(pool):
                pool = np.arange(len(release))
            known_pool = pool[known[pool]]
            other_pool = pool[~known[pool]]
            if len(known_pool):
                model = NearestNeighbors(n_neighbors=1).fit(rx[known_pool])
                known_d[rows] = model.kneighbors(qx[rows], return_distance=True)[0][:, 0]
            if len(other_pool):
                model = NearestNeighbors(n_neighbors=1).fit(rx[other_pool])
                other_d[rows] = model.kneighbors(qx[rows], return_distance=True)[0][:, 0]
        known_dist_views.append(known_d)
        other_dist_views.append(other_d)
    kd = np.column_stack(known_dist_views)
    od = np.column_stack(other_dist_views)
    ratio = np.min(kd, axis=1) / np.clip(np.min(od, axis=1), 1e-12, None)
    known_preference = np.mean(kd < od, axis=1)
    return np.min(kd, axis=1), ratio, known_preference


def aligned_times(frame: pd.DataFrame, target: int, challenge_ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(target)].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[challenge_ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    from attack_phase_v5.aia_structured_ensemble import (
        proximity_score,
        stratified_half_selection,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    releases = {target: pd.read_csv(DATA / f"C_{target}.csv") for target in TARGET_IDS}
    challenges = {target: pd.read_csv(DATA / f"aia_challenge_{target}.csv") for target in TARGET_IDS}
    masks, diagnostics = known_nonmember_masks(releases)
    base = ROOT / "submissions" / "prelim" / "team24"
    v4 = pd.read_csv(base / "attack-candidate-v4" / "F_aia.csv")
    independent = pd.read_csv(base / "attack-candidate-v5-probe-independent" / "F_aia.csv")
    structured = pd.read_csv(base / "attack-candidate-v5-structured" / "F_aia.csv")

    summary_rows: list[dict[str, float | int]] = []
    row_parts: list[pd.DataFrame] = []
    for target in TARGET_IDS:
        challenge = challenges[target]
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_time = aligned_times(v4, target, ids)
        ind_time = aligned_times(independent, target, ids)
        struct_time = aligned_times(structured, target, ids)
        active_v4 = ~np.isclose(v4_time, 11.0)
        active_ind = ~np.isclose(ind_time, 11.0)
        active_struct = ~np.isclose(struct_time, 11.0)
        known = masks[target]
        filtered = releases[target].loc[~known].drop_duplicates(list(RELEASE_COLUMNS)).reset_index(drop=True)
        original_proximity = proximity_score(releases[target], challenge)
        filtered_proximity = proximity_score(filtered, challenge) if len(filtered) >= 25 else original_proximity
        base_score = 1.10 * active_v4 + 0.90 * active_ind + 0.35 * original_proximity
        filtered_score = 1.10 * active_v4 + 0.90 * active_ind + 0.35 * filtered_proximity
        selected_base, _ = stratified_half_selection(challenge, base_score)
        selected_filtered, _ = stratified_half_selection(challenge, filtered_score)

        known_distance, ratio, preference = block_nearest_known_margin(
            releases[target], challenge, known
        ) if known.any() else (
            np.full(len(challenge), np.inf),
            np.full(len(challenge), np.inf),
            np.zeros(len(challenge)),
        )
        strong_known = (ratio < 0.5) & (preference >= (2 / 3))
        moderate_known = (ratio < 0.8) & (preference >= (2 / 3))
        summary_rows.append(
            {
                "target": target,
                **diagnostics[target],
                "filtered_release_rows": len(filtered),
                "existing_struct_vs_recomputed": int(np.count_nonzero(active_struct != selected_base)),
                "filtered_selection_changes": int(np.count_nonzero(selected_filtered != selected_base)),
                "filtered_vs_existing_struct": int(np.count_nonzero(selected_filtered != active_struct)),
                "strong_known_challenge_rows": int(strong_known.sum()),
                "moderate_known_challenge_rows": int(moderate_known.sum()),
                "v4_active_strong_known": int(np.count_nonzero(active_v4 & strong_known)),
                "ind_active_strong_known": int(np.count_nonzero(active_ind & strong_known)),
                "struct_active_strong_known": int(np.count_nonzero(active_struct & strong_known)),
                "filtered_active_strong_known": int(np.count_nonzero(selected_filtered & strong_known)),
                "v4_active_moderate_known": int(np.count_nonzero(active_v4 & moderate_known)),
                "ind_active_moderate_known": int(np.count_nonzero(active_ind & moderate_known)),
                "struct_active_moderate_known": int(np.count_nonzero(active_struct & moderate_known)),
                "filtered_active_moderate_known": int(np.count_nonzero(selected_filtered & moderate_known)),
                "proximity_rank_rho_before_after": float(
                    pd.Series(original_proximity).corr(pd.Series(filtered_proximity), method="spearman")
                ),
            }
        )
        row_parts.append(
            pd.DataFrame(
                {
                    "target": target,
                    "challenge_row_id": ids,
                    "known_distance": known_distance,
                    "known_to_other_ratio": ratio,
                    "known_view_preference": preference,
                    "strong_known": strong_known.astype(int),
                    "moderate_known": moderate_known.astype(int),
                    "v4": active_v4.astype(int),
                    "ind": active_ind.astype(int),
                    "struct": active_struct.astype(int),
                    "filtered_struct": selected_filtered.astype(int),
                    "proximity_before": original_proximity,
                    "proximity_after": filtered_proximity,
                }
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "nonmember_filter_impact.csv", index=False)
    pd.concat(row_parts, ignore_index=True).to_csv(OUT / "nonmember_filter_rows.csv", index=False)
    print(summary[summary["known_nonmember_union"] > 0].to_string(index=False))


if __name__ == "__main__":
    main()
