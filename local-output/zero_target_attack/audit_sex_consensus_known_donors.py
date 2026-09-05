"""Compare standard and known-nonmember-filtered sex-consensus selection."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "team-shared" / "prelim" / "attack-targets"
RELEASE_COLUMNS = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
    "smoking", "FPG", "time", "onset", "death",
)
NUMERIC = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time")


def keys(frame: pd.DataFrame) -> pd.MultiIndex:
    work = frame.loc[:, RELEASE_COLUMNS].copy()
    work.loc[:, NUMERIC] = work.loc[:, NUMERIC].round(10)
    return pd.MultiIndex.from_frame(work)


def aligned(frame: pd.DataFrame, target: int, ids: np.ndarray) -> np.ndarray:
    block = frame[frame.target_k.astype(int).eq(target)].set_index("challenge_row_id")
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    from attack_phase_v5.aia_structured_ensemble import (
        filter_linkable_releases,
        stratified_half_selection,
    )
    from attack_phase_v6.aia_global_density import sex_copula_proximity_score

    releases = {
        path.stem.split("_")[1]: pd.read_csv(path)
        for path in DATA.glob("C_*.csv")
        if path.stem.split("_")[1] != "24"
    }
    standard, _ = filter_linkable_releases(releases)
    background_keys = set(keys(pd.read_csv(ROOT / "participant_data" / "A_bg.csv")).tolist())
    release_keys = {target: keys(frame) for target, frame in releases.items()}
    v4 = pd.read_csv(ROOT / "submissions/prelim/team24/attack-candidate-v4/F_aia.csv")
    ind = pd.read_csv(ROOT / "submissions/prelim/team24/attack-candidate-v5-probe-independent/F_aia.csv")
    rows = []
    for target in (18, 19):
        other = set()
        for other_target, other_keys in release_keys.items():
            if other_target != str(target):
                other.update(other_keys.tolist())
        own_keys = release_keys[str(target)]
        known = np.fromiter(
            ((value in background_keys) or (value in other) for value in own_keys),
            dtype=bool,
            count=len(own_keys),
        )
        enhanced = releases[str(target)].loc[~known].drop_duplicates(list(RELEASE_COLUMNS)).reset_index(drop=True)
        challenge = pd.read_csv(DATA / f"aia_challenge_{target}.csv")
        ids = challenge.challenge_row_id.to_numpy(dtype=int)
        av4 = ~np.isclose(aligned(v4, target, ids), 11.0)
        aind = ~np.isclose(aligned(ind, target, ids), 11.0)
        selections = {}
        for name, release in (("standard", standard[str(target)]), ("enhanced", enhanced)):
            proximity = sex_copula_proximity_score(release, challenge)
            score = 1.10 * av4 + 0.90 * aind + 0.55 * proximity
            selections[name] = stratified_half_selection(challenge, score)[0]
        rows.append(
            {
                "target": target,
                "known_rows": int(known.sum()),
                "standard_rows": len(standard[str(target)]),
                "enhanced_rows": len(enhanced),
                "selection_changes": int(np.count_nonzero(selections["standard"] != selections["enhanced"])),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
