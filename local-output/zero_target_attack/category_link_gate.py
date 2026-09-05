"""Public permutation diagnostic for whether C categorical row links are usable.

For each target, link challenge rows to C using numeric QIs only, then ask
whether the linked C row agrees on sex/prefecture more often than when the
joint categorical labels are randomly permuted across C rows.  This separates
genuine row-level category/QI dependence from mere marginal overrepresentation.
"""

from __future__ import annotations

import json
import math
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
    "stable3": ("age", "SBP", "TG"),
}
LOG = {"TG", "ALT"}


def raw_embed(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = frame.loc[:, columns].to_numpy(dtype=float, copy=True)
    ref = reference.loc[:, columns].to_numpy(dtype=float, copy=True)
    for j, column in enumerate(columns):
        if column in LOG:
            values[:, j] = np.log1p(np.clip(values[:, j], 0.0, None))
            ref[:, j] = np.log1p(np.clip(ref[:, j], 0.0, None))
    center = np.median(ref, axis=0)
    q25, q75 = np.quantile(ref, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-9, scale, np.std(ref, axis=0))
    return (values - center) / np.where(scale > 1e-9, scale, 1.0) / math.sqrt(len(columns))


def copula_embed(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    values = np.column_stack(
        [frame[c].rank(method="average").to_numpy(dtype=float) / (len(frame) + 1) for c in columns]
    )
    return values / math.sqrt(len(columns))


def permutation_summary(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    nearest: np.ndarray,
    *,
    rng: np.random.Generator,
    permutations: int = 500,
) -> dict[str, float]:
    release_sex = release["sex"].to_numpy()
    release_pref = release["prefecture"].astype(str).to_numpy()
    query_sex = challenge["sex"].to_numpy()
    query_pref = challenge["prefecture"].astype(str).to_numpy()

    def stats(order: np.ndarray) -> np.ndarray:
        linked_sex = release_sex[order][nearest]
        linked_pref = release_pref[order][nearest]
        sex = linked_sex == query_sex
        pref = linked_pref == query_pref
        return np.array([sex.mean(), pref.mean(), (sex & pref).mean()])

    observed = stats(np.arange(len(release)))
    null = np.vstack([stats(rng.permutation(len(release))) for _ in range(permutations)])
    names = ("sex", "pref", "joint")
    output: dict[str, float] = {}
    for j, name in enumerate(names):
        sd = null[:, j].std(ddof=1)
        output[f"{name}_observed"] = float(observed[j])
        output[f"{name}_null_mean"] = float(null[:, j].mean())
        output[f"{name}_z"] = float((observed[j] - null[:, j].mean()) / sd) if sd > 0 else 0.0
        output[f"{name}_upper_p"] = float((1 + np.count_nonzero(null[:, j] >= observed[j])) / (permutations + 1))
    return output


def main() -> None:
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        release = pd.read_csv(DATA / f"C_{target}.csv")
        challenge = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        for view, columns in VIEWS.items():
            for mapping in ("raw", "copula"):
                if mapping == "raw":
                    rx = raw_embed(release, background, columns)
                    qx = raw_embed(challenge, background, columns)
                else:
                    rx = copula_embed(release, columns)
                    qx = copula_embed(challenge, columns)
                distance = cdist(qx, rx)
                nearest = np.argmin(distance, axis=1)
                reverse = np.argmin(distance, axis=0)
                mutual = reverse[nearest] == np.arange(len(challenge))
                base = permutation_summary(
                    release,
                    challenge,
                    nearest,
                    rng=np.random.default_rng(910_000 + target * 10 + len(rows)),
                )
                rows.append(
                    {
                        "target": target,
                        "view": view,
                        "mapping": mapping,
                        "mutual_numeric_links": int(mutual.sum()),
                        "unique_numeric_links": int(np.unique(nearest).size),
                        **base,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "category_link_gate.csv", index=False)
    compact = frame.loc[
        frame.groupby(["target", "mapping"])["joint_z"].idxmax(),
        ["target", "mapping", "view", "joint_observed", "joint_null_mean", "joint_z", "joint_upper_p", "pref_z", "sex_z"],
    ]
    print(compact.round(4).to_string(index=False))
    (OUT / "category_link_gate.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
