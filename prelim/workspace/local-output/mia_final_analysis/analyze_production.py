"""Research-only diagnostics for the production MIA rank matrices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "submissions" / "prelim" / "team24" / name / "F_mia.csv").set_index("record_id")


def desc_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(method="first", ascending=False).astype(int)


def main() -> None:
    out = Path(__file__).resolve().parent
    frames = {
        "v1": load("attack-candidate-v1"),
        "v3": load("attack-candidate-v3"),
        "v4": load("attack-candidate-v4"),
        "ind": load("attack-candidate-daybreak-independent"),
        "tail12": load("attack-candidate-v5-tail12"),
        "tail15": load("attack-candidate-v6-feedback-tail15"),
    }
    ranks = {name: desc_rank(frame) for name, frame in frames.items()}

    width_rows = []
    for width in range(1, 41):
        top = {target: set(ranks["v4"].index[ranks["v4"][target] <= width]) for target in TARGETS}
        flat = [rid for target in TARGETS for rid in top[target]]
        counts = pd.Series(flat).value_counts()
        duplicate_ids = counts[counts > 1]
        width_rows.append({
            "width": width,
            "slots": width * len(TARGETS),
            "unique_records": len(counts),
            "duplicate_slots": int(len(flat) - len(counts)),
            "duplicate_records": int(len(duplicate_ids)),
            "max_multiplicity": int(counts.max()),
            "targets_with_conflicts": int(sum(any(rid in duplicate_ids.index for rid in top[t]) for t in TARGETS)),
        })
    pd.DataFrame(width_rows).to_csv(out / "production_width_conflicts.csv", index=False)

    conflict_rows = []
    top15 = {target: set(ranks["v4"].index[ranks["v4"][target] <= 15]) for target in TARGETS}
    flat15 = [rid for values in top15.values() for rid in values]
    duplicate15 = sorted(pd.Series(flat15).value_counts().loc[lambda s: s > 1].index.astype(int))
    for rid in duplicate15:
        occurrences = [target for target in TARGETS if rid in top15[target]]
        tail_owner = [target for target in TARGETS if ranks["tail15"].loc[rid, target] <= 15]
        for target in occurrences:
            row = {
                "record_id": rid,
                "target": target,
                "multiplicity": len(occurrences),
                "tail15_retained": target in tail_owner,
                "v4_rank": int(ranks["v4"].loc[rid, target]),
                "v3_rank": int(ranks["v3"].loc[rid, target]),
                "v1_rank": int(ranks["v1"].loc[rid, target]),
                "ind_rank": int(ranks["ind"].loc[rid, target]),
            }
            for expert in ("v4", "v3", "v1", "ind"):
                rr = ranks[expert].loc[rid, list(TARGETS)].sort_values()
                row[f"{expert}_winner"] = str(rr.index[0])
                row[f"{expert}_winner_rank"] = int(rr.iloc[0])
                row[f"{expert}_second_rank"] = int(rr.iloc[1])
                row[f"{expert}_target_margin"] = int(rr.iloc[1] - ranks[expert].loc[rid, target])
            conflict_rows.append(row)
    pd.DataFrame(conflict_rows).sort_values(["record_id", "v4_rank"]).to_csv(out / "production_tail15_conflicts.csv", index=False)

    change_rows = []
    for variant in ("tail12", "tail15"):
        for target in TARGETS:
            before = ranks["v4"][target]
            after = ranks[variant][target]
            changed = before.ne(after)
            for rid in before.index[changed]:
                old = int(before.loc[rid])
                new = int(after.loc[rid])
                if min(old, new) <= 20:
                    change_rows.append({
                        "variant": variant,
                        "target": target,
                        "record_id": int(rid),
                        "v4_rank": old,
                        "new_rank": new,
                        "delta": new - old,
                        "v3_rank": int(ranks["v3"].loc[rid, target]),
                        "ind_rank": int(ranks["ind"].loc[rid, target]),
                    })
    pd.DataFrame(change_rows).sort_values(["variant", "target", "v4_rank"]).to_csv(out / "production_tail_changes.csv", index=False)

    overlap_rows = []
    for target in TARGETS:
        for width in (10, 11, 12, 13, 14, 15, 16, 20, 25, 30, 50):
            sets = {
                name: set(rank.index[rank[target] <= width])
                for name, rank in ranks.items()
                if name in ("v1", "v3", "v4", "ind")
            }
            overlap_rows.append({
                "target": target,
                "width": width,
                **{f"v4_{name}_overlap": len(sets["v4"] & sets[name]) for name in ("v1", "v3", "ind")},
                "all4_intersection": len(set.intersection(*sets.values())),
                "any4_union": len(set.union(*sets.values())),
            })
    pd.DataFrame(overlap_rows).to_csv(out / "production_expert_overlap.csv", index=False)

    correlations = []
    for target in TARGETS:
        for a in ("v1", "v3", "v4", "ind"):
            for b in ("v1", "v3", "v4", "ind"):
                if a >= b:
                    continue
                rho = float(spearmanr(frames[a][target], frames[b][target]).statistic)
                correlations.append({"target": target, "expert_a": a, "expert_b": b, "spearman": rho})
    pd.DataFrame(correlations).to_csv(out / "production_expert_correlations.csv", index=False)

    summary = {
        "duplicate_tail_widths": {str(row["width"]): row["duplicate_slots"] for row in width_rows},
        "tail15_duplicate_ids": duplicate15,
        "tail15_conflict_count": len(duplicate15),
        "tail15_occurrence_count": len(conflict_rows),
        "tail15_removed_occurrences": int(sum(not row["tail15_retained"] for row in conflict_rows)),
    }
    (out / "production_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(conflict_rows).sort_values(["record_id", "v4_rank"]).to_string(index=False))


if __name__ == "__main__":
    main()
