"""Null-calibrate the 56 non-A_bg residual rows of official target 19."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
helpers = runpy.run_path(str(Path(__file__).with_name("forensics.py")))
QI = helpers["QI"]
VIEWS = helpers["VIEWS"]
raw_embed = helpers["raw_embed"]
grouped_knn_distance = helpers["grouped_knn_distance"]
stratified_selection = helpers["stratified_selection"]
calibrated_control_rate = helpers["calibrated_control_rate"]


def main() -> None:
    c = pd.read_csv(ROOT / "team-shared/prelim/attack-targets/C_19.csv")
    q = pd.read_csv(ROOT / "team-shared/prelim/attack-targets/aia_challenge_19.csv")
    bg = pd.read_csv(ROOT / "participant_data/A_bg.csv")
    qi_keys = set(map(tuple, bg.loc[:, QI].itertuples(index=False, name=None)))
    is_bg = np.fromiter(
        (tuple(row) in qi_keys for row in c.loc[:, QI].itertuples(index=False, name=None)),
        dtype=bool,
        count=len(c),
    )
    residual = c.loc[~is_bg].reset_index(drop=True)
    donor = c.loc[is_bg].reset_index(drop=True)
    donor_keys = set(map(tuple, donor.loc[:, QI].itertuples(index=False, name=None)))
    available_bg = bg.loc[
        [tuple(row) not in donor_keys for row in bg.loc[:, QI].itertuples(index=False, name=None)]
    ].reset_index(drop=True)

    rows = []
    for seed in range(10):
        rng = np.random.default_rng(94100 + seed)
        held_mask = rng.random(len(available_bg)) < 0.40
        train_bg = available_bg.loc[~held_mask].reset_index(drop=True)
        held = available_bg.loc[held_mask].reset_index(drop=True)
        for view, columns in VIEWS.items():
            rx = raw_embed(residual, train_bg, columns)
            dx = raw_embed(donor, train_bg, columns)
            qx = raw_embed(q, train_bg, columns)
            hx = raw_embed(held, train_bg, columns)
            for mode in ("global", "pref", "sex", "both"):
                r_q = grouped_knn_distance(rx, qx, residual, q, mode, k=3)
                d_q = grouped_knn_distance(dx, qx, donor, q, mode, k=3)
                r_h = grouped_knn_distance(rx, hx, residual, held, mode, k=3)
                d_h = grouped_knn_distance(dx, hx, donor, held, mode, k=3)
                score_q = np.mean(np.log((d_q + 1e-6) / (r_q + 1e-6)), axis=1)
                score_h = np.mean(np.log((d_h + 1e-6) / (r_h + 1e-6)), axis=1)
                selected, details = stratified_selection(q, score_q)
                rate, exact_weight = calibrated_control_rate(q, details, held, score_h)
                rows.append(
                    {
                        "seed": seed,
                        "view": view,
                        "mode": mode,
                        "control_rate": rate,
                        "implied_accuracy": 1.0 - rate,
                        "exact_pref_weight": exact_weight,
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["view", "mode"], as_index=False)
        .agg(
            control_mean=("control_rate", "mean"),
            control_min=("control_rate", "min"),
            control_max=("control_rate", "max"),
            implied_accuracy=("implied_accuracy", "mean"),
        )
        .sort_values("control_mean")
    )
    frame.to_csv(Path(__file__).with_name("target19_residual_splits.csv"), index=False)
    summary.to_csv(Path(__file__).with_name("target19_residual_summary.csv"), index=False)
    print("residual", len(residual), "A_bg donor", len(donor), "available null", len(available_bg))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
