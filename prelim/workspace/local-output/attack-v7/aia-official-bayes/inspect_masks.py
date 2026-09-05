"""Inspect the three official AIA masks on the seven common zero targets.

Research-only.  This script reads public contest artifacts and team feedback;
it does not touch token files, candidate directories, archives, or services.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TEAM = ROOT / "submissions" / "prelim" / "team24"
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
ZERO_TARGETS = (6, 7, 8, 9, 17, 18, 19)
PREDICTIONS = {
    "v4": TEAM / "attack-candidate-v4" / "F_aia.csv",
    "independent": TEAM / "attack-candidate-daybreak-independent" / "F_aia.csv",
    "routed": TEAM / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {name: pd.read_csv(path) for name, path in PREDICTIONS.items()}
    rows: list[dict[str, object]] = []
    patterns: list[dict[str, object]] = []
    for target in ZERO_TARGETS:
        aligned: dict[str, np.ndarray] = {}
        for name, frame in frames.items():
            block = (
                frame.loc[frame["target_k"].astype(int).eq(target)]
                .sort_values("challenge_row_id", kind="mergesort")
                .reset_index(drop=True)
            )
            aligned[name] = block["time_hat"].to_numpy(dtype=float)
        challenge = pd.read_csv(TARGETS / f"aia_challenge_{target}.csv")
        active = {
            name: (~np.isclose(values, 11.0)).astype(int)
            for name, values in aligned.items()
        }
        pattern = list(
            zip(active["v4"], active["independent"], active["routed"], strict=True)
        )
        counts = Counter(pattern)
        rows.append(
            {
                "target": target,
                **{f"{name}_active": int(values.sum()) for name, values in active.items()},
                **{
                    f"{name}_unique_times": json.dumps(
                        sorted(Counter(np.round(values, 12)).items()),
                        ensure_ascii=False,
                    )
                    for name, values in aligned.items()
                },
                "visible_strata": int(
                    challenge.assign(
                        age_band=np.floor((challenge["age"] - 40.0) / 5.0).astype(int)
                    )
                    .groupby(["sex", "prefecture", "age_band"], sort=False)
                    .ngroups
                ),
            }
        )
        for key in sorted(counts):
            patterns.append(
                {
                    "target": target,
                    "v4": key[0],
                    "independent": key[1],
                    "routed": key[2],
                    "rows": counts[key],
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "production_mask_summary.csv", index=False)
    pd.DataFrame(patterns).to_csv(OUT / "production_pattern_counts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(patterns).to_string(index=False))


if __name__ == "__main__":
    main()
