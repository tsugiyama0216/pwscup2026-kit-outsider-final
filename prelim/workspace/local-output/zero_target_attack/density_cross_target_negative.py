"""Cross-target negative control for production global-density selectors."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
OUT = ROOT / "local-output" / "zero_target_attack"
CONFIG = {
    6: ("age", "SBP", "TG"),
    7: ("age", "SBP", "TG", "HDL", "ALT"),
    8: ("age", "SBP", "TG", "ALT"),
    9: ("age", "SBP", "TG"),
    18: ("age", "SBP", "TG", "HDL", "ALT"),
    19: ("age", "SBP", "TG"),
    20: ("age", "SBP", "TG", "HDL", "ALT"),
}


def load_helpers():
    path = OUT / "estimate_density_signal.py"
    spec = importlib.util.spec_from_file_location("density_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load density helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from attack_phase_v5.aia_structured_ensemble import filter_linkable_releases

    helper = load_helpers()
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv").reset_index(drop=True)
    releases = {
        path.stem.split("_")[1]: pd.read_csv(path)
        for path in DATA.glob("C_*.csv")
        if path.stem.split("_")[1] != "24"
    }
    # Match the production selector's standard cross-target/dedup filter.
    releases, _ = filter_linkable_releases(releases)
    rows: list[dict[str, object]] = []
    for target, columns in CONFIG.items():
        challenge = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        for split_seed in range(6):
            rng = np.random.default_rng(88000 + split_seed)
            held = rng.random(len(background)) < 0.35
            train = background.loc[~held].reset_index(drop=True)
            control = background.loc[held].reset_index(drop=True)
            for release_target, release in releases.items():
                q_score, c_score = helper.density_score(
                    release, train, [challenge, control], columns
                )
                _, control_rate, _ = helper.selection_and_control_rate(
                    challenge, q_score, control, c_score
                )
                rows.append(
                    {
                        "target": target,
                        "split_seed": split_seed,
                        "release_target": int(release_target),
                        "is_own_release": int(int(release_target) == target),
                        "control_rate": control_rate,
                        "implied_power": helper.implied_horizon_power(control_rate),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "density_cross_target_negative.csv", index=False)
    summaries: list[dict[str, object]] = []
    for target, block in frame.groupby("target"):
        means = block.groupby("release_target")["implied_power"].mean().sort_values(ascending=False)
        own = float(means.loc[target])
        other = means.drop(target)
        per_split = block.pivot(index="split_seed", columns="release_target", values="implied_power")
        summaries.append(
            {
                "target": target,
                "own_power": own,
                "own_rank_of_releases": int(means.index.get_loc(target) + 1),
                "release_count": len(means),
                "other_median_power": float(other.median()),
                "other_best_power": float(other.max()),
                "own_minus_other_median": own - float(other.median()),
                "own_beats_other_median_splits": int(
                    np.count_nonzero(
                        per_split[target].to_numpy()
                        > per_split.drop(columns=target).median(axis=1).to_numpy()
                    )
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "density_cross_target_negative_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
