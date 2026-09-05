"""Compare public structured AIA variants by target without writing candidates."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    build_aia_v5_variants,
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
)


DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "mechanism_router"
TARGET_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
VARIANTS = (
    "stratum_v4__v4_only",
    "stratum_v4_copula__v4_only",
    "consensus_v4_lean__v4_only",
    "consensus_v4_lean__independent_fallback",
    "consensus_v4_copula__v4_only",
    "consensus_v4_copula__independent_fallback",
    "consensus_independent_lean__independent_fallback",
    "consensus_soft_proximity__independent_fallback",
)


def aligned(frame: pd.DataFrame, target: int, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(target)].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    releases = {str(target): pd.read_csv(DATA / f"C_{target}.csv") for target in TARGET_IDS}
    challenges = {str(target): pd.read_csv(DATA / f"aia_challenge_{target}.csv") for target in TARGET_IDS}
    base = ROOT / "submissions" / "prelim" / "team24"
    v4 = pd.read_csv(base / "attack-candidate-v4" / "F_aia.csv")
    independent = pd.read_csv(base / "attack-candidate-v5-probe-independent" / "F_aia.csv")
    old_structured = pd.read_csv(base / "attack-candidate-v5-structured" / "F_aia.csv")
    generated, _ = build_aia_v5_variants(v4, independent, releases, challenges)
    missing = sorted(set(VARIANTS) - set(generated))
    if missing:
        raise RuntimeError(f"expected variants missing: {missing}")
    linkable, _ = filter_linkable_releases(releases)

    rows: list[dict[str, float | int | str]] = []
    for target in TARGET_IDS:
        challenge = challenges[str(target)]
        ids = challenge["challenge_row_id"].to_numpy(dtype=int)
        ref_times = {
            "v4": aligned(v4, target, ids),
            "independent": aligned(independent, target, ids),
            "old_structured": aligned(old_structured, target, ids),
        }
        ref_active = {name: ~np.isclose(values, 11.0) for name, values in ref_times.items()}
        raw = proximity_score(linkable[str(target)], challenge)
        copula = copula_proximity_score(linkable[str(target)], challenge)
        for name in VARIANTS:
            values = aligned(generated[name], target, ids)
            active = ~np.isclose(values, 11.0)
            rows.append(
                {
                    "target": target,
                    "variant": name,
                    "changes_v4_active": int(np.count_nonzero(active != ref_active["v4"])),
                    "changes_ind_active": int(np.count_nonzero(active != ref_active["independent"])),
                    "changes_old_struct_active": int(np.count_nonzero(active != ref_active["old_structured"])),
                    "changes_v4_time": int(np.count_nonzero(~np.isclose(values, ref_times["v4"]))),
                    "changes_ind_time": int(np.count_nonzero(~np.isclose(values, ref_times["independent"]))),
                    "event_predictions": int(np.count_nonzero(values < 9.5)),
                    "selected_raw_proximity": float(np.mean(raw[active])),
                    "selected_copula_proximity": float(np.mean(copula[active])),
                    "raw_copula_rank_rho": float(pd.Series(raw).corr(pd.Series(copula), method="spearman")),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "variant_diagnostics.csv", index=False)
    for variant in VARIANTS:
        print(f"\n{variant}")
        print(
            result[result["variant"].eq(variant)][
                ["target", "changes_v4_active", "changes_ind_active", "changes_old_struct_active", "changes_v4_time", "event_predictions", "selected_raw_proximity", "selected_copula_proximity", "raw_copula_rank_rho"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
