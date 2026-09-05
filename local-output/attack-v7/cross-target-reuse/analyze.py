"""Check whether public AIA challenge rows recur across targets or releases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
DIST = ROOT / "participant_data"
OUT = Path(__file__).resolve().parent
VISIBLE = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
LABS = ["age", "SBP", "TG", "HDL", "ALT"]


def signature(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame.loc[:, columns], index=False)


def main() -> None:
    challenge_paths = sorted(
        TARGETS.glob("aia_challenge_*.csv"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    challenges: dict[str, pd.DataFrame] = {
        path.stem.rsplit("_", 1)[1]: pd.read_csv(path) for path in challenge_paths
    }
    rows: list[pd.DataFrame] = []
    for target, frame in challenges.items():
        block = frame.copy()
        block["target"] = target
        block["visible_hash"] = signature(block, VISIBLE).astype(str)
        block["labs_hash"] = signature(block, LABS).astype(str)
        rows.append(block)
    combined = pd.concat(rows, ignore_index=True)

    visible_counts = combined.groupby("visible_hash")["target"].nunique()
    labs_counts = combined.groupby("labs_hash")["target"].nunique()
    combined["visible_target_frequency"] = combined["visible_hash"].map(visible_counts)
    combined["labs_target_frequency"] = combined["labs_hash"].map(labs_counts)

    pair_rows: list[dict[str, object]] = []
    for left_index, left in enumerate(sorted(challenges, key=int)):
        left_visible = set(signature(challenges[left], VISIBLE).astype(str))
        left_labs = set(signature(challenges[left], LABS).astype(str))
        for right in sorted(challenges, key=int)[left_index + 1 :]:
            right_visible = set(signature(challenges[right], VISIBLE).astype(str))
            right_labs = set(signature(challenges[right], LABS).astype(str))
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "visible_overlap": len(left_visible & right_visible),
                    "labs_overlap": len(left_labs & right_labs),
                }
            )

    references: dict[str, pd.DataFrame] = {
        "A_bg": pd.read_csv(DIST / "A_bg.csv"),
        "B_self": pd.read_csv(DIST / "B_self.csv"),
        "B_practice": pd.read_csv(DIST / "B_practice.csv"),
    }
    for path in sorted(TARGETS.glob("C_*.csv"), key=lambda p: int(p.stem[2:])):
        references[path.stem] = pd.read_csv(path)

    match_rows: list[dict[str, object]] = []
    for target, frame in challenges.items():
        challenge_visible = signature(frame, VISIBLE).astype(str)
        challenge_labs = signature(frame, LABS).astype(str)
        for name, reference in references.items():
            visible_set = set(signature(reference, VISIBLE).astype(str))
            labs_set = set(signature(reference, LABS).astype(str))
            match_rows.append(
                {
                    "challenge_target": target,
                    "reference": name,
                    "visible_exact_rows": int(challenge_visible.isin(visible_set).sum()),
                    "labs_exact_rows": int(challenge_labs.isin(labs_set).sum()),
                }
            )

    pair_frame = pd.DataFrame(pair_rows)
    match_frame = pd.DataFrame(match_rows)
    frequency = (
        combined.groupby("target", as_index=False)
        .agg(
            rows=("challenge_row_id", "size"),
            visible_reused_rows=("visible_target_frequency", lambda x: int((x > 1).sum())),
            labs_reused_rows=("labs_target_frequency", lambda x: int((x > 1).sum())),
            max_visible_frequency=("visible_target_frequency", "max"),
            max_labs_frequency=("labs_target_frequency", "max"),
        )
        .sort_values("target", key=lambda s: s.astype(int))
    )

    combined.to_csv(OUT / "row_frequencies.csv", index=False)
    pair_frame.to_csv(OUT / "pair_overlaps.csv", index=False)
    match_frame.to_csv(OUT / "release_exact_matches.csv", index=False)
    frequency.to_csv(OUT / "target_frequency_summary.csv", index=False)
    summary = {
        "challenge_targets": sorted(challenges, key=int),
        "challenge_rows": len(combined),
        "unique_visible_rows": int(combined["visible_hash"].nunique()),
        "visible_rows_reused_across_targets": int((combined["visible_target_frequency"] > 1).sum()),
        "unique_labs_rows": int(combined["labs_hash"].nunique()),
        "labs_rows_reused_across_targets": int((combined["labs_target_frequency"] > 1).sum()),
        "maximum_pair_visible_overlap": int(pair_frame["visible_overlap"].max()),
        "maximum_pair_labs_overlap": int(pair_frame["labs_overlap"].max()),
        "maximum_visible_release_matches": int(match_frame["visible_exact_rows"].max()),
        "maximum_labs_release_matches": int(match_frame["labs_exact_rows"].max()),
        "top_visible_reference_matches": match_frame.nlargest(10, "visible_exact_rows").to_dict(orient="records"),
        "top_labs_reference_matches": match_frame.nlargest(10, "labs_exact_rows").to_dict(orient="records"),
        "visible_frequency_histogram": dict(Counter(combined["visible_target_frequency"].astype(int))),
        "labs_frequency_histogram": dict(Counter(combined["labs_target_frequency"].astype(int))),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
