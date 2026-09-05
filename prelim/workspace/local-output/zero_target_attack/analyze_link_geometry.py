"""Read-only diagnostics for alternative AIA link geometries on zero targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "zero_target_attack"
TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
BLOCK = ("sex", "prefecture")
VIEWS = {
    "all5": ("age", "SBP", "TG", "HDL", "ALT"),
    "no_hdl": ("age", "SBP", "TG", "ALT"),
    "stable3": ("age", "SBP", "TG"),
    "age_sbp_alt": ("age", "SBP", "ALT"),
    "labs_no_hdl": ("SBP", "TG", "ALT"),
    "age_sbp": ("age", "SBP"),
}
LOG_COLUMNS = {"TG", "ALT"}


def midrank(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    result = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        result[:, j] = (
            frame[column].rank(method="average").to_numpy(dtype=float)
            / (len(frame) + 1.0)
        )
    return result / np.sqrt(len(columns))


def background_cdf(
    frame: pd.DataFrame, background: pd.DataFrame, columns: tuple[str, ...]
) -> np.ndarray:
    result = np.empty((len(frame), len(columns)), dtype=float)
    for j, column in enumerate(columns):
        reference = np.sort(background[column].to_numpy(dtype=float))
        values = frame[column].to_numpy(dtype=float)
        lo = np.searchsorted(reference, values, side="left")
        hi = np.searchsorted(reference, values, side="right")
        result[:, j] = (0.5 * (lo + hi) + 0.5) / (len(reference) + 1.0)
    return result / np.sqrt(len(columns))


def robust_raw(
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
    return (values - center) / scale / np.sqrt(len(columns))


def groups(frame: pd.DataFrame) -> dict[tuple[object, ...], np.ndarray]:
    raw = frame.groupby(list(BLOCK), sort=False, dropna=False).indices
    return {
        key if isinstance(key, tuple) else (key,): np.asarray(rows, dtype=int)
        for key, rows in raw.items()
    }


def block_link(
    release_x: np.ndarray,
    challenge_x: np.ndarray,
    release: pd.DataFrame,
    challenge: pd.DataFrame,
) -> dict[str, np.ndarray]:
    rg = groups(release)
    qg = groups(challenge)
    nearest = np.full(len(challenge), -1, dtype=int)
    first = np.full(len(challenge), np.inf)
    second = np.full(len(challenge), np.inf)
    mutual = np.zeros(len(challenge), dtype=bool)
    reverse_degree = np.zeros(len(challenge), dtype=int)
    for key, qrows in qg.items():
        rrows = rg.get(key)
        if rrows is None or not len(rrows):
            rrows = np.arange(len(release), dtype=int)
        matrix = cdist(challenge_x[qrows], release_x[rrows])
        order = np.argpartition(matrix, kth=min(1, len(rrows) - 1), axis=1)
        one = order[:, 0]
        if len(rrows) > 1:
            two = order[:, 1]
            swap = matrix[np.arange(len(qrows)), two] < matrix[np.arange(len(qrows)), one]
            one, two = np.where(swap, two, one), np.where(swap, one, two)
        else:
            two = one
        nearest[qrows] = rrows[one]
        first[qrows] = matrix[np.arange(len(qrows)), one]
        second[qrows] = matrix[np.arange(len(qrows)), two]
        r_to_q = np.argmin(matrix, axis=0)
        local = np.arange(len(qrows))
        mutual[qrows] = r_to_q[one] == local
        degree = np.bincount(one, minlength=len(rrows))
        reverse_degree[qrows] = degree[one]
    return {
        "nearest": nearest,
        "first": first,
        "second": second,
        "mutual": mutual,
        "degree": reverse_degree,
    }


def aligned_time(frame: pd.DataFrame, target: int, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(target)].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    base = ROOT / "submissions" / "prelim" / "team24"
    v4 = pd.read_csv(base / "attack-candidate-v4" / "F_aia.csv")
    independent = pd.read_csv(
        base / "attack-candidate-v5-probe-independent" / "F_aia.csv"
    )
    summary: list[dict[str, object]] = []
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        release = pd.read_csv(DATA / f"C_{target}.csv")
        challenge = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        v4_time = aligned_time(v4, target, ids)
        ind_time = aligned_time(independent, target, ids)
        record = pd.DataFrame({"target": target, "challenge_row_id": ids})
        links: dict[str, dict[str, np.ndarray]] = {}
        for view, columns in VIEWS.items():
            embeddings = {
                "separate": (midrank(release, columns), midrank(challenge, columns)),
                "bgcdf": (midrank(release, columns), background_cdf(challenge, background, columns)),
                "raw": (robust_raw(release, background, columns), robust_raw(challenge, background, columns)),
            }
            for geometry, (rx, qx) in embeddings.items():
                name = f"{geometry}_{view}"
                link = block_link(rx, qx, release, challenge)
                links[name] = link
                nearest = link["nearest"]
                linked_time = release["time"].to_numpy(dtype=float)[nearest]
                linked_onset = release["onset"].to_numpy(dtype=int)[nearest]
                linked_death = release["death"].to_numpy(dtype=int)[nearest]
                record[f"{name}_distance"] = link["first"]
                record[f"{name}_nearest"] = nearest
                record[f"{name}_time"] = linked_time
                record[f"{name}_event"] = ((linked_onset + linked_death) > 0).astype(int)
                record[f"{name}_mutual"] = link["mutual"].astype(int)
                record[f"{name}_margin"] = link["second"] - link["first"]
                active_v4 = ~np.isclose(v4_time, 11.0)
                active_ind = ~np.isclose(ind_time, 11.0)
                summary.append(
                    {
                        "target": target,
                        "link": name,
                        "median_distance": float(np.median(link["first"])),
                        "p10_distance": float(np.quantile(link["first"], 0.1)),
                        "mutual": int(link["mutual"].sum()),
                        "unique_links": int(np.unique(nearest).size),
                        "degree_gt1": int(np.count_nonzero(link["degree"] > 1)),
                        "linked_event": int(np.count_nonzero(linked_onset + linked_death)),
                        "v4_time_agree_active": float(
                            np.mean(np.abs(linked_time[active_v4] - v4_time[active_v4]) <= 0.5)
                        ),
                        "ind_time_agree_active": float(
                            np.mean(np.abs(linked_time[active_ind] - ind_time[active_ind]) <= 0.5)
                        ),
                    }
                )
                global_name = f"global_{geometry}_{view}"
                global_link = block_link(
                    rx,
                    qx,
                    release.assign(sex=0, prefecture="_all"),
                    challenge.assign(sex=0, prefecture="_all"),
                )
                links[global_name] = global_link
                global_nearest = global_link["nearest"]
                global_time = release["time"].to_numpy(dtype=float)[global_nearest]
                record[f"{global_name}_distance"] = global_link["first"]
                record[f"{global_name}_nearest"] = global_nearest
                record[f"{global_name}_time"] = global_time
                record[f"{global_name}_mutual"] = global_link["mutual"].astype(int)
                summary.append(
                    {
                        "target": target,
                        "link": global_name,
                        "median_distance": float(np.median(global_link["first"])),
                        "p10_distance": float(np.quantile(global_link["first"], 0.1)),
                        "mutual": int(global_link["mutual"].sum()),
                        "unique_links": int(np.unique(global_nearest).size),
                        "degree_gt1": int(np.count_nonzero(global_link["degree"] > 1)),
                        "linked_event": int(
                            np.count_nonzero(
                                release["onset"].to_numpy(dtype=int)[global_nearest]
                                + release["death"].to_numpy(dtype=int)[global_nearest]
                            )
                        ),
                        "v4_time_agree_active": float(
                            np.mean(np.abs(global_time[active_v4] - v4_time[active_v4]) <= 0.5)
                        ),
                        "ind_time_agree_active": float(
                            np.mean(np.abs(global_time[active_ind] - ind_time[active_ind]) <= 0.5)
                        ),
                    }
                )
        # Link stability itself is a high-precision confidence proxy.
        chosen = [links[f"separate_{view}"]["nearest"] for view in VIEWS]
        record["separate_link_mode_count"] = [
            int(np.unique(values, return_counts=True)[1].max())
            for values in np.column_stack(chosen)
        ]
        for left, right in (("separate_all5", "separate_no_hdl"), ("separate_no_hdl", "bgcdf_no_hdl"), ("raw_no_hdl", "separate_no_hdl")):
            rho = spearmanr(links[left]["first"], links[right]["first"]).statistic
            summary.append({"target": target, "link": f"comparison:{left}:{right}", "distance_rho": float(rho)})
        record["v4_time"] = v4_time
        record["ind_time"] = ind_time
        rows.append(record)
    pd.DataFrame(summary).to_csv(OUT / "link_geometry_summary.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "link_geometry_rows.csv", index=False)
    report = {
        "purpose": "public-data link diagnostics only; no organizer truth",
        "targets": list(TARGETS),
        "views": {key: list(value) for key, value in VIEWS.items()},
    }
    (OUT / "link_geometry_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
