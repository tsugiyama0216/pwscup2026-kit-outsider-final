"""Estimate production membership lift from global C-vs-background density.

Controls are not public.  For each deterministic split, this script uses one
part of A_bg to fit the density ratio and the disjoint part to estimate the
control selection rate after matching the public sex/prefecture/age-band
composition.  The mixture identity P_Q=(P_member+P_control)/2 then implies the
member rate for a fixed challenge selection.  This is a model diagnostic, not
organizer truth.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "zero_target_attack"
TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
VIEWS = {
    "all5": ("age", "SBP", "TG", "HDL", "ALT"),
    "nohdl": ("age", "SBP", "TG", "ALT"),
    "stable3": ("age", "SBP", "TG"),
    "labs": ("SBP", "TG", "ALT"),
}
LOG = {"TG", "ALT"}


def embed(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    x = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    r = reference.loc[:, columns].to_numpy(dtype=float, copy=True)
    for j, column in enumerate(columns):
        if column in LOG:
            x[:, j] = np.log1p(np.clip(x[:, j], 0.0, None))
            r[:, j] = np.log1p(np.clip(r[:, j], 0.0, None))
    center = np.median(r, axis=0)
    q25, q75 = np.quantile(r, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-9, scale, np.std(r, axis=0))
    return (x - center) / np.where(scale > 1e-9, scale, 1.0)


def density_score(
    release: pd.DataFrame,
    background_train: pd.DataFrame,
    queries: list[pd.DataFrame],
    columns: tuple[str, ...],
) -> list[np.ndarray]:
    release_x = embed(release, background_train, columns)
    background_x = embed(background_train, background_train, columns)
    release_model = NearestNeighbors(n_neighbors=8).fit(release_x)
    background_model = NearestNeighbors(n_neighbors=8).fit(background_x)
    output: list[np.ndarray] = []
    for query in queries:
        query_x = embed(query, background_train, columns)
        rd = release_model.kneighbors(query_x, return_distance=True)[0]
        bd = background_model.kneighbors(query_x, return_distance=True)[0]
        components = []
        for k in (1, 3, 8):
            components.append(
                len(columns)
                * np.log((bd[:, k - 1] + 1e-6) / (rd[:, k - 1] + 1e-6))
            )
        output.append(np.mean(np.column_stack(components), axis=1))
    return output


def tree_score(
    release: pd.DataFrame,
    background_train: pd.DataFrame,
    queries: list[pd.DataFrame],
    columns: tuple[str, ...],
    seed: int,
) -> list[np.ndarray]:
    rx = embed(release, background_train, columns)
    bx = embed(background_train, background_train, columns)
    rng = np.random.default_rng(seed)
    choose = rng.choice(len(bx), size=min(len(bx), len(rx) * 2), replace=False)
    x = np.vstack((rx, bx[choose]))
    y = np.r_[np.ones(len(rx), dtype=int), np.zeros(len(choose), dtype=int)]
    model = ExtraTreesClassifier(
        n_estimators=120,
        min_samples_leaf=12,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)
    return [model.predict_proba(embed(q, background_train, columns))[:, 1] for q in queries]


def age_band(frame: pd.DataFrame) -> np.ndarray:
    return np.floor((frame["age"].to_numpy(dtype=float) - 40.0) / 5.0).astype(int)


def selection_and_control_rate(
    challenge: pd.DataFrame,
    challenge_score: np.ndarray,
    control: pd.DataFrame,
    control_score: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    challenge_work = challenge.assign(_band=age_band(challenge))
    control_work = control.assign(_band=age_band(control))
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    selected = np.zeros(len(challenge), dtype=bool)
    weighted_control = 0.0
    weight = 0
    for key, raw_rows in challenge_work.groupby(
        ["sex", "prefecture", "_band"], sort=True
    ).indices.items():
        rows = np.asarray(raw_rows, dtype=int)
        quota = len(rows) // 2
        order = rows[np.lexsort((ids[rows], -challenge_score[rows]))]
        selected[order[:quota]] = True
        cutoff = challenge_score[order[quota - 1]]
        pool = control_work.index[
            control_work["sex"].eq(key[0])
            & control_work["prefecture"].astype(str).eq(str(key[1]))
            & control_work["_band"].eq(key[2])
        ].to_numpy(dtype=int)
        if len(pool) < 10:
            pool = control_work.index[
                control_work["sex"].eq(key[0])
                & control_work["_band"].eq(key[2])
            ].to_numpy(dtype=int)
        weighted_control += quota * float(np.mean(control_score[pool] >= cutoff))
        weight += quota
    return selected, weighted_control / weight, weight


def implied_horizon_power(control_rate: float, *, censor_rate: float = 147 / 163) -> float:
    member_rate = np.clip(1.0 - control_rate, 0.0, 1.0)
    p_m = censor_rate * member_rate
    p_c = censor_rate * control_rate
    return float(np.clip((p_m - p_c) / max(1.0 - p_c, 1e-9), 0.0, 1.0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv").reset_index(drop=True)
    all_releases = {
        path.stem.split("_")[1]: pd.read_csv(path)
        for path in DATA.glob("C_*.csv")
    }
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from attack_phase_v5.aia_structured_ensemble import filter_linkable_releases

    filtered_releases, filter_diagnostics = filter_linkable_releases(all_releases)
    rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    for target in TARGETS:
        release = filtered_releases[str(target)]
        challenge = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        for split_seed in range(6):
            rng = np.random.default_rng(88000 + split_seed)
            held = rng.random(len(background)) < 0.35
            train = background.loc[~held].reset_index(drop=True)
            control = background.loc[held].reset_index(drop=True)
            for view, columns in VIEWS.items():
                for method, scorer in (("knn", density_score), ("trees", tree_score)):
                    if method == "knn":
                        q_score, c_score = scorer(release, train, [challenge, control], columns)
                    else:
                        q_score, c_score = scorer(
                            release, train, [challenge, control], columns, 99000 + split_seed
                        )
                    selected, control_rate, _ = selection_and_control_rate(
                        challenge, q_score, control, c_score
                    )
                    rows.append(
                        {
                            "target": target,
                            "release_rows": len(release),
                            "split_seed": split_seed,
                            "method": method,
                            "view": view,
                            "control_selection_rate": control_rate,
                            "implied_member_accuracy": 1.0 - control_rate,
                            "implied_horizon_power": implied_horizon_power(control_rate),
                        }
                    )
                    if split_seed == 0:
                        selection_parts.append(
                            pd.DataFrame(
                                {
                                    "target": target,
                                    "method": method,
                                    "view": view,
                                    "challenge_row_id": challenge["challenge_row_id"],
                                    "score": q_score,
                                    "selected": selected.astype(int),
                                }
                            )
                        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "density_signal_filtered_splits.csv", index=False)
    pd.concat(selection_parts, ignore_index=True).to_csv(
        OUT / "density_signal_filtered_selections.csv", index=False
    )
    summary = (
        result.groupby(["target", "method", "view"], as_index=False)
        .agg(
            control_rate_mean=("control_selection_rate", "mean"),
            control_rate_min=("control_selection_rate", "min"),
            control_rate_max=("control_selection_rate", "max"),
            power_mean=("implied_horizon_power", "mean"),
            power_min=("implied_horizon_power", "min"),
            power_max=("implied_horizon_power", "max"),
        )
        .sort_values(["target", "power_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT / "density_signal_filtered_summary.csv", index=False)
    pd.DataFrame.from_dict(filter_diagnostics, orient="index").to_csv(
        OUT / "density_signal_filter_diagnostics.csv", index_label="target"
    )
    print(summary.groupby("target").head(3).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
