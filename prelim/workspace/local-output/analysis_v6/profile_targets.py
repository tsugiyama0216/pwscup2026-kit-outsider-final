from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
SUB = ROOT / "submissions" / "prelim" / "team24"
VISIBLE = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
FULL = [
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
]


def active_map(name: str) -> dict[int, pd.DataFrame]:
    frame = pd.read_csv(SUB / name / "F_aia.csv")
    frame["active"] = ~np.isclose(frame["time_hat"].to_numpy(float), 11.0)
    return {int(k): g.set_index("challenge_row_id") for k, g in frame.groupby("target_k")}


def rounded_fraction(series: pd.Series, decimals: int = 6) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return float(np.mean(np.isclose(values, np.round(values, decimals), atol=1e-10)))


def main() -> None:
    names = {
        "v1": "attack-candidate-v1",
        "v2": "attack-candidate-v2",
        "v3": "attack-candidate-v3",
        "v4": "attack-candidate-v4",
        "ind": "attack-candidate-daybreak-independent",
        "structured": "attack-candidate-v5-structured",
    }
    experts = {label: active_map(name) for label, name in names.items()}
    target_ids = [
        int(x)
        for x in json.loads((TARGETS / "mia_columns.json").read_text(encoding="utf-8"))["target_columns"]
        if int(x) != 24
    ]
    releases = {k: pd.read_csv(TARGETS / f"C_{k}.csv") for k in target_ids}
    challenges = {k: pd.read_csv(TARGETS / f"aia_challenge_{k}.csv") for k in target_ids}
    other_full_hashes = {}
    for k, frame in releases.items():
        other_full_hashes[k] = set(pd.util.hash_pandas_object(frame[FULL], index=False).astype("uint64"))

    rows = []
    for k in target_ids:
        c = releases[k].reset_index(drop=True)
        q = challenges[k].reset_index(drop=True)
        c_key = pd.MultiIndex.from_frame(c[VISIBLE])
        q_key = pd.MultiIndex.from_frame(q[VISIBLE])
        exact = q_key.isin(c_key)
        c_full_hash = pd.util.hash_pandas_object(c[FULL], index=False).astype("uint64")
        other = set().union(*(v for j, v in other_full_hashes.items() if j != k))
        cross_full = c_full_hash.isin(other)
        row = {
            "target": k,
            "c_event_rate": float(c["onset"].mean()),
            "c_death_rate": float(c["death"].mean()),
            "c_horizon_rate": float(np.mean(np.isclose(c["time"], 10.0, atol=0.51))),
            "c_full_duplicates": int(c.duplicated(FULL).sum()),
            "c_cross_full": int(cross_full.sum()),
            "challenge_exact_visible": int(exact.sum()),
            "challenge_unique_visible": int(q[VISIBLE].drop_duplicates().shape[0]),
            "c_unique_visible": int(c[VISIBLE].drop_duplicates().shape[0]),
            "median_unique_numeric": float(np.median([c[x].nunique() for x in ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]])),
            "mean_integerish": float(np.mean([rounded_fraction(c[x], 0) for x in ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]])),
            "mean_3dp": float(np.mean([rounded_fraction(c[x], 3) for x in ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time"]])),
        }
        for label, blocks in experts.items():
            block = blocks[k].loc[q["challenge_row_id"]]
            active = block["active"].to_numpy(bool)
            row[f"{label}_exact_selected"] = int(np.count_nonzero(active & exact))
            row[f"{label}_event_guesses"] = int(np.count_nonzero(active & (block["time_hat"].to_numpy(float) < 9.5)))
            row[f"{label}_changes_v4"] = int(np.count_nonzero(active != experts["v4"][k].loc[q["challenge_row_id"], "active"].to_numpy(bool)))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("target")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)
    print(out.to_string(index=False))
    out.to_csv(Path(__file__).with_name("target_profile.csv"), index=False)


if __name__ == "__main__":
    main()
