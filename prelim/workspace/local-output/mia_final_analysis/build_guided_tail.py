"""Prototype Daybreak-guided owner tie-break for the production top-15 tail."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "17", "18", "19", "20", "22", "23", "26", "29")
WIDTH = 15


def load(candidate: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "submissions" / "prelim" / "team24" / candidate / "F_mia.csv").set_index("record_id")


def order(frame: pd.DataFrame, target: str) -> list[int]:
    return frame[target].sort_values(ascending=False, kind="mergesort").index.astype(int).tolist()


def build(
    *,
    minimum_independent_gain: int = 1,
    maximum_v4_penalty: int | None = None,
    equal_rank_fusion: bool = False,
) -> tuple[pd.DataFrame, dict]:
    v3 = load("attack-candidate-v3")
    v4 = load("attack-candidate-v4")
    independent = load("attack-candidate-daybreak-independent")
    current = load("attack-candidate-v6-feedback-tail15")
    diagnostics = json.loads((ROOT / "local-output" / "attack-v6" / "diagnostics_v6.json").read_text(encoding="utf-8"))
    private = set(map(int, diagnostics["mia"]["private_record_ids"]))
    rare = set(map(int, diagnostics["mia"]["eligible_rare_record_ids"]))
    all_ids = set(map(int, v4.index))
    common = all_ids - private - rare
    ranks = {name: frame.rank(method="first", ascending=False).astype(int) for name, frame in {"v3": v3, "v4": v4, "ind": independent, "current": current}.items()}
    original = {target: set(order(v4, target)[:WIDTH]) for target in TARGETS}
    current_selected = {target: set(order(current, target)[:WIDTH]) for target in TARGETS}
    occurrences: dict[int, list[str]] = {}
    for target, ids in original.items():
        for rid in ids:
            occurrences.setdefault(rid, []).append(target)
    conflicts = {rid: targets for rid, targets in occurrences.items() if len(targets) > 1}

    selected = {target: set(values) for target, values in original.items()}
    owner_audit = []
    for rid, candidate_targets in sorted(conflicts.items()):
        old_owner = next(target for target in candidate_targets if rid in current_selected[target])
        independent_ordered = sorted(candidate_targets, key=lambda target: (int(ranks["ind"].loc[rid, target]), int(ranks["v4"].loc[rid, target]), int(target)))
        proposed_independent = independent_ordered[0]
        if equal_rank_fusion:
            # v4 is the strongest broad ranking, while Daybreak is the strongest
            # observed low-FPR ranking.  Their per-column output is rank-scaled,
            # so an equal rank sum is calibration-free across the two systems.
            proposed = min(
                candidate_targets,
                key=lambda target: (
                    int(ranks["v4"].loc[rid, target]) + int(ranks["ind"].loc[rid, target]),
                    int(ranks["v4"].loc[rid, target]),
                    int(ranks["v3"].loc[rid, target]),
                    int(target),
                ),
            )
        else:
            proposed = proposed_independent
        independent_gain = int(ranks["ind"].loc[rid, old_owner] - ranks["ind"].loc[rid, proposed])
        v4_penalty = int(ranks["v4"].loc[rid, proposed] - ranks["v4"].loc[rid, old_owner])
        switch = proposed != old_owner
        if not equal_rank_fusion:
            switch = switch and independent_gain >= minimum_independent_gain
            if maximum_v4_penalty is not None:
                switch = switch and v4_penalty <= maximum_v4_penalty
        owner = proposed if switch else old_owner
        for target in candidate_targets:
            if target != owner:
                selected[target].discard(rid)
        owner_audit.append({
            "record_id": rid,
            "occurrences": candidate_targets,
            "old_owner": old_owner,
            "guided_owner": owner,
            "proposed_independent_owner": proposed_independent,
            "proposed_rule_owner": proposed,
            "switched": switch,
            "independent_gain": independent_gain,
            "v4_penalty": v4_penalty,
            "ranks": {target: {name: int(value.loc[rid, target]) for name, value in ranks.items() if name != "current"} for target in candidate_targets},
        })

    used = set().union(*selected.values())
    slots = [target for target in TARGETS for _ in range(WIDTH - len(selected[target]))]
    available = sorted(common - used)
    utility = np.empty((len(slots), len(available)), dtype=float)
    for row, target in enumerate(slots):
        utility[row] = v4.loc[available, target].to_numpy(float) + 1e-6 * v3.loc[available, target].to_numpy(float)
    rows, columns = linear_sum_assignment(-utility)
    for row, column in zip(rows, columns, strict=True):
        selected[slots[int(row)]].add(available[int(column)])
    if any(len(values) != WIDTH for values in selected.values()):
        raise RuntimeError("unfilled slots")
    flat = [rid for values in selected.values() for rid in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("tail is not unique")

    output = pd.DataFrame(index=v4.index)
    for target in TARGETS:
        original_order = order(v4, target)
        chosen = selected[target]
        descending = [rid for rid in original_order if rid in chosen]
        descending += [rid for rid in original_order if rid not in chosen and rid not in private]
        descending += [rid for rid in original_order if rid in private]
        score = np.empty(len(v4), dtype=float)
        position = {rid: index for index, rid in enumerate(v4.index.astype(int))}
        for descending_rank, rid in enumerate(descending):
            score[position[rid]] = (len(v4) - descending_rank) / (len(v4) + 1.0)
        output[target] = score

        rare_v4 = [rid for rid in original_order if rid in rare]
        rare_out = [rid for rid in order(output, target) if rid in rare]
        if rare_v4 != rare_out:
            raise RuntimeError(f"rare order changed for {target}")
    output.insert(0, "record_id", output.index.astype(int))
    output = output.reset_index(drop=True)
    out_ranks = output.set_index("record_id").rank(method="first", ascending=False).astype(int)
    changes = {}
    for target in TARGETS:
        before = set(order(current, target)[:WIDTH])
        after = set(out_ranks.index[out_ranks[target] <= WIDTH].astype(int))
        if before != after:
            changes[target] = {"removed_vs_current": sorted(before - after), "added_vs_current": sorted(after - before)}
    return output, {
        "width": WIDTH,
        "minimum_independent_gain": minimum_independent_gain,
        "maximum_v4_penalty": maximum_v4_penalty,
        "equal_rank_fusion": equal_rank_fusion,
        "owners": owner_audit,
        "changes_vs_current_tail15": changes,
    }


def main() -> None:
    destination = Path(__file__).resolve().parent
    variants = {
        "guided_all": {"minimum_independent_gain": 1, "maximum_v4_penalty": None},
        "guided_v4_margin3": {"minimum_independent_gain": 5, "maximum_v4_penalty": 3},
        "guided_ind_gain10": {"minimum_independent_gain": 10, "maximum_v4_penalty": None},
        "guided_equal_rank": {"equal_rank_fusion": True},
    }
    summary = {}
    for name, parameters in variants.items():
        frame, detail = build(**parameters)
        frame.to_csv(destination / f"F_mia_{name}.csv", index=False)
        (destination / f"{name}.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        switches = [row for row in detail["owners"] if row["switched"]]
        summary[name] = {"switches": len(switches), "records": [row["record_id"] for row in switches], "targets_changed": sorted(detail["changes_vs_current_tail15"], key=int)}
    (destination / "guided_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
