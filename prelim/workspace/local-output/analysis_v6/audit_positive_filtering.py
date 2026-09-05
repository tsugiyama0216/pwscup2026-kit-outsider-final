"""Quick audit of linkable-release filtering for positive AIA targets."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
research = runpy.run_path(
    str(ROOT / "local-output" / "analysis_v6" / "evaluate_positive_copula_link.py")
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_proximity_score,
    filter_linkable_releases,
    proximity_score,
    stratified_half_selection,
)


def aligned(frame: pd.DataFrame, target: int, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(target)].copy()
    block.index = block["challenge_row_id"].astype(int)
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def build(
    selector_release: pd.DataFrame,
    link_release: pd.DataFrame,
    challenge: pd.DataFrame,
    v4_time: np.ndarray,
    ind_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    active_v4 = ~np.isclose(v4_time, 11.0)
    active_ind = ~np.isclose(ind_time, 11.0)
    raw = proximity_score(selector_release, challenge)
    cop = copula_proximity_score(selector_release, challenge)
    selected, _ = stratified_half_selection(
        challenge,
        1.1 * active_v4 + 0.9 * active_ind + 0.2 * raw + 0.2 * cop,
    )
    links = research["copula_links"](link_release, challenge)
    nearest_time = research["prediction_from_link"](link_release, links["nearest"])
    event = nearest_time < 10.0 - 1e-9
    gate = links["mutual"] & (links["margin_rank"] >= 0.50) & event
    base = np.full(len(challenge), 11.0)
    chosen = np.where(active_v4, v4_time, 10.0)
    base[selected] = chosen[selected]
    output = base.copy()
    output[selected & gate] = nearest_time[selected & gate]
    return selected, output, int(np.count_nonzero(~np.isclose(output, base)))


def main() -> None:
    data = ROOT / "team-shared" / "prelim" / "attack-targets"
    ids_all = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")
    releases = {target: pd.read_csv(data / f"C_{target}.csv") for target in ids_all}
    filtered, diagnostics = filter_linkable_releases(releases)
    changed_positive = {
        target: not releases[target].equals(filtered[target])
        for target in ("3", "4", "5", "10", "12", "13", "22", "23", "26", "29")
    }

    root = ROOT / "submissions" / "prelim" / "team24"
    v4 = pd.read_csv(root / "attack-candidate-v4" / "F_aia.csv")
    ind = pd.read_csv(root / "attack-candidate-v5-probe-independent" / "F_aia.csv")
    current = pd.read_csv(root / "attack-candidate-v6-routed-probe" / "F_aia.csv")
    challenge = pd.read_csv(data / "aia_challenge_29.csv")
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    v4_time = aligned(v4, 29, ids)
    ind_time = aligned(ind, 29, ids)
    current_time = aligned(current, 29, ids)

    configurations = {
        "unfiltered_selector_unfiltered_link": (releases["29"], releases["29"]),
        "filtered_selector_filtered_link": (filtered["29"], filtered["29"]),
        "filtered_selector_unfiltered_link": (filtered["29"], releases["29"]),
    }
    print("changed positive releases", changed_positive)
    print("target29 filter", diagnostics["29"])
    for name, (selector_release, link_release) in configurations.items():
        selected, output, overrides = build(
            selector_release, link_release, challenge, v4_time, ind_time
        )
        active_current = ~np.isclose(current_time, 11.0)
        print(
            name,
            {
                "active_changes_current": int(np.count_nonzero(selected != active_current)),
                "time_changes_current": int(np.count_nonzero(~np.isclose(output, current_time))),
                "event_override_changes_base": overrides,
                "event_predictions": int(np.count_nonzero(output < 9.5)),
            },
        )


if __name__ == "__main__":
    main()
