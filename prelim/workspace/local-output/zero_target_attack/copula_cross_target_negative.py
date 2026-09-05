"""Cross-target specificity of separate-margin copula linkage geometry."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "zero_target_attack"
TARGETS = (6, 7, 8, 9, 17, 18, 19, 20)
VIEWS = {
    "all5": ("age", "SBP", "TG", "HDL", "ALT"),
    "nohdl": ("age", "SBP", "TG", "ALT"),
}


def copula(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack(
        [frame[c].rank(method="average").to_numpy(dtype=float) / (len(frame) + 1) for c in columns]
    ) / np.sqrt(len(columns))


def main() -> None:
    releases = {
        int(path.stem.split("_")[1]): pd.read_csv(path)
        for path in DATA.glob("C_*.csv")
        if int(path.stem.split("_")[1]) != 24
    }
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        q = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        for release_target, release in releases.items():
            for view, columns in VIEWS.items():
                qx, rx = copula(q, columns), copula(release, columns)
                base = cdist(qx, rx)
                for block in ("global", "sex", "sex_pref"):
                    cost = base.copy()
                    if block in {"sex", "sex_pref"}:
                        cost += 1000 * (
                            q["sex"].to_numpy()[:, None] != release["sex"].to_numpy()[None, :]
                        )
                    if block == "sex_pref":
                        cost += 500 * (
                            q["prefecture"].astype(str).to_numpy()[:, None]
                            != release["prefecture"].astype(str).to_numpy()[None, :]
                        )
                    nearest = np.argmin(cost, axis=1)
                    first = cost[np.arange(len(q)), nearest]
                    reverse = np.argmin(cost, axis=0)
                    rows.append(
                        {
                            "target": target,
                            "release_target": release_target,
                            "is_own": int(target == release_target),
                            "view": view,
                            "block": block,
                            "mean_nearest": float(first.mean()),
                            "q10_nearest": float(np.quantile(first, 0.1)),
                            "mutual": int(np.count_nonzero(reverse[nearest] == np.arange(len(q)))),
                            "unique": int(np.unique(nearest).size),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "copula_cross_target_negative.csv", index=False)
    summary: list[dict[str, object]] = []
    for (target, view, block), group in frame.groupby(["target", "view", "block"]):
        by_mean = group.sort_values("mean_nearest", kind="mergesort")
        by_mutual = group.sort_values("mutual", ascending=False, kind="mergesort")
        own = group[group.release_target.eq(target)].iloc[0]
        summary.append(
            {
                "target": target,
                "view": view,
                "block": block,
                "own_mean": own.mean_nearest,
                "mean_rank": int(np.flatnonzero(by_mean.release_target.to_numpy() == target)[0] + 1),
                "own_mutual": int(own.mutual),
                "mutual_rank": int(np.flatnonzero(by_mutual.release_target.to_numpy() == target)[0] + 1),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "copula_cross_target_negative_summary.csv", index=False)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
