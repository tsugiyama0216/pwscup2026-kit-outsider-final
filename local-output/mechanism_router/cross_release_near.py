"""Detect high-precision shared-row candidates across public C releases."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "mechanism_router"
IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
RELEASE_COLUMNS = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
)
VIEWS = {
    "qi": (("sex", "prefecture"), ("age", "SBP", "TG", "HDL", "ALT")),
    "clinical": (
        ("sex", "prefecture", "smoking"),
        ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"),
    ),
    "full": (
        ("sex", "prefecture", "smoking", "onset", "death"),
        ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"),
    ),
}
LOG_COLUMNS = {"TG", "ALT", "FPG"}


def fitted_values(frame: pd.DataFrame, columns: tuple[str, ...], center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
    return (values - center) / scale / np.sqrt(len(columns))


def center_scale(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    for index, column in enumerate(columns):
        if column in LOG_COLUMNS:
            values[:, index] = np.log1p(np.clip(values[:, index], 0.0, None))
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(values, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    return center, np.where(scale > 1e-9, scale, 1.0)


def key_groups(frame: pd.DataFrame, keys: tuple[str, ...]) -> dict[tuple[object, ...], np.ndarray]:
    raw = frame.groupby(list(keys), sort=False, dropna=False).indices
    return {
        key if isinstance(key, tuple) else (key,): np.asarray(rows, dtype=int)
        for key, rows in raw.items()
    }


def pair_mutual(
    left_x: np.ndarray,
    right_x: np.ndarray,
    left_groups: dict[tuple[object, ...], np.ndarray],
    right_groups: dict[tuple[object, ...], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for key in left_groups.keys() & right_groups.keys():
        li = left_groups[key]
        ri = right_groups[key]
        matrix = cdist(left_x[li], right_x[ri])
        l_to_r = np.argmin(matrix, axis=1)
        r_to_l = np.argmin(matrix, axis=0)
        local_l = np.arange(len(li))
        is_mutual = r_to_l[l_to_r] == local_l
        if np.any(is_mutual):
            keep = np.flatnonzero(is_mutual)
            left_rows.append(li[keep])
            right_rows.append(ri[l_to_r[keep]])
            distances.append(matrix[keep, l_to_r[keep]])
    if not distances:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0)
    return np.concatenate(left_rows), np.concatenate(right_rows), np.concatenate(distances)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    releases = {target: pd.read_csv(DATA / f"C_{target}.csv") for target in IDS}
    fitted: dict[str, dict[int, np.ndarray]] = {}
    groups: dict[str, dict[int, dict[tuple[object, ...], np.ndarray]]] = {}
    for name, (keys, columns) in VIEWS.items():
        center, scale = center_scale(background, columns)
        fitted[name] = {
            target: fitted_values(frame, columns, center, scale)
            for target, frame in releases.items()
        }
        groups[name] = {
            target: key_groups(frame, keys) for target, frame in releases.items()
        }

    pair_rows: list[dict[str, float | int]] = []
    link_rows: list[pd.DataFrame] = []
    for left_id, right_id in itertools.combinations(IDS, 2):
        exact = len(
            set(map(tuple, releases[left_id][list(RELEASE_COLUMNS)].to_numpy()))
            & set(map(tuple, releases[right_id][list(RELEASE_COLUMNS)].to_numpy()))
        )
        row: dict[str, float | int] = {"left": left_id, "right": right_id, "exact": exact}
        for name in VIEWS:
            li, ri, distance = pair_mutual(
                fitted[name][left_id], fitted[name][right_id],
                groups[name][left_id], groups[name][right_id],
            )
            row[f"{name}_mutual"] = len(distance)
            for threshold in (0.001, 0.003, 0.01, 0.03, 0.05, 0.10):
                label = str(threshold).replace(".", "p")
                row[f"{name}_lt_{label}"] = int(np.count_nonzero(distance < threshold))
            row[f"{name}_p01"] = float(np.quantile(distance, 0.01)) if len(distance) else np.nan
            row[f"{name}_p05"] = float(np.quantile(distance, 0.05)) if len(distance) else np.nan
            if name == "full" and len(distance):
                link_rows.append(
                    pd.DataFrame(
                        {
                            "left_target": left_id,
                            "left_row": li,
                            "right_target": right_id,
                            "right_row": ri,
                            "distance": distance,
                        }
                    )
                )
        pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows).sort_values(["full_lt_0p03", "clinical_lt_0p03"], ascending=False)
    pairs.to_csv(OUT / "cross_release_pairs.csv", index=False)
    links = pd.concat(link_rows, ignore_index=True) if link_rows else pd.DataFrame()
    links.to_csv(OUT / "cross_release_full_mutual_links.csv", index=False)
    keep = [
        "left", "right", "exact", "full_lt_0p001", "full_lt_0p003",
        "full_lt_0p01", "full_lt_0p03", "clinical_lt_0p01", "clinical_lt_0p03",
        "qi_lt_0p01", "qi_lt_0p03", "full_p01",
    ]
    print(pairs[keep].head(40).to_string(index=False))
    print(f"wrote {OUT / 'cross_release_pairs.csv'}")


if __name__ == "__main__":
    main()
