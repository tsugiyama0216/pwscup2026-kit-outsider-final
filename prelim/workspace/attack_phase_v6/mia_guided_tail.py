"""Conservative Daybreak-guided ownership repair for the MIA top tail.

The v4 ranking is the strongest live MIA base.  Its narrow low-FPR tail,
however, assigns a few candidate records to more than one target even though a
record can be shared with at most one preliminary-round target.  The existing
tail repair removes those duplicate slots.  This module changes only the
tie-break that decides which target keeps a duplicated record.  It minimizes
the equally weighted sum of v4 and independent-expert ranks among the possible
owners; v3 is used only as a deterministic final tie-break and hole filler.

The reconstruction keeps the v4 order of the inferred private block and rare
records.  Consequently the rare-only ROC is identical to v4, while the guided
expert changes only five equal-rank ownership decisions in the saved production
inputs.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from attack_phase_v5.mia_tail_assignment import assign_unique_top_tail


DEFAULT_TOP_L = 15
DEFAULT_PROTECTED_TARGETS = ("1", "2")


def _align_frame(
    frame: pd.DataFrame,
    *,
    record_ids: np.ndarray,
    targets: list[str],
    label: str,
) -> pd.DataFrame:
    if list(frame.columns) != ["record_id", *targets]:
        raise ValueError(f"{label} MIA columns differ from the v4 columns")
    work = frame.copy()
    record_number = pd.to_numeric(work["record_id"], errors="coerce")
    numeric_ids = record_number.to_numpy(dtype=float)
    if not np.isfinite(numeric_ids).all() or not np.allclose(
        numeric_ids, np.round(numeric_ids)
    ):
        raise ValueError(f"{label} record_id must contain finite integers")
    work["record_id"] = record_number.astype(np.int64)
    if work["record_id"].duplicated().any():
        raise ValueError(f"{label} record_id is not unique")
    indexed = work.set_index("record_id", drop=False)
    if set(indexed.index) != set(record_ids.tolist()):
        raise ValueError(f"{label} record_id set differs from v4")
    aligned = indexed.loc[record_ids, ["record_id", *targets]].reset_index(drop=True)
    values = aligned[targets].apply(pd.to_numeric, errors="coerce")
    matrix = values.to_numpy(dtype=float)
    if (
        not np.isfinite(matrix).all()
        or (matrix < 0.0).any()
        or (matrix > 1.0).any()
    ):
        raise ValueError(f"{label} scores must be finite and in [0, 1]")
    if any(values[target].nunique(dropna=False) != len(aligned) for target in targets):
        raise ValueError(f"{label} target columns must be tie-free")
    aligned.loc[:, targets] = values
    return aligned


def _descending_ids(frame: pd.DataFrame, target: str) -> list[int]:
    return (
        frame.sort_values(target, ascending=False, kind="mergesort")["record_id"]
        .astype(np.int64)
        .tolist()
    )


def build_guided_tail(
    v3: pd.DataFrame,
    v4: pd.DataFrame,
    guide: pd.DataFrame,
    *,
    top_l: int = DEFAULT_TOP_L,
    protected_targets: Iterable[str] = DEFAULT_PROTECTED_TARGETS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Repair duplicate ownership using calibration-free equal-rank fusion."""
    targets = [str(column) for column in v4.columns if column != "record_id"]
    if not targets or list(v4.columns) != ["record_id", *targets]:
        raise ValueError("v4 must contain record_id followed by target columns")
    record_number = pd.to_numeric(v4["record_id"], errors="coerce")
    record_values = record_number.to_numpy(dtype=float)
    if not np.isfinite(record_values).all() or not np.allclose(
        record_values, np.round(record_values)
    ):
        raise ValueError("v4 record_id must contain finite integers")
    record_ids = record_number.astype(np.int64).to_numpy()
    if len(record_ids) != len(set(record_ids.tolist())):
        raise ValueError("v4 record_id is not unique")

    v4_aligned = _align_frame(
        v4, record_ids=record_ids, targets=targets, label="v4"
    )
    v3_aligned = _align_frame(
        v3, record_ids=record_ids, targets=targets, label="v3"
    )
    guide_aligned = _align_frame(
        guide, record_ids=record_ids, targets=targets, label="guide"
    )
    baseline, baseline_diagnostics = assign_unique_top_tail(
        v3_aligned, v4_aligned, top_l=top_l
    )

    indexed = {
        "v3": v3_aligned.set_index("record_id"),
        "v4": v4_aligned.set_index("record_id"),
        "guide": guide_aligned.set_index("record_id"),
        "baseline": baseline.set_index("record_id"),
    }
    ranks = {
        name: frame[targets].rank(method="first", ascending=False).astype(int)
        for name, frame in indexed.items()
    }
    private = set(int(value) for value in baseline_diagnostics["private_record_ids"])
    rare = set(
        int(value) for value in baseline_diagnostics["eligible_rare_record_ids"]
    )
    all_ids = set(int(value) for value in record_ids)
    common = all_ids - private - rare
    original_top = {
        target: set(_descending_ids(v4_aligned, target)[:top_l])
        for target in targets
    }
    baseline_top = {
        target: set(_descending_ids(baseline, target)[:top_l])
        for target in targets
    }
    occurrences: dict[int, list[str]] = {}
    for target, values in original_top.items():
        for record_id in values:
            occurrences.setdefault(record_id, []).append(target)
    conflicts = {
        record_id: candidate_targets
        for record_id, candidate_targets in occurrences.items()
        if len(candidate_targets) > 1
    }

    selected = {target: set(values) for target, values in original_top.items()}
    owner_audit: list[dict[str, Any]] = []
    for record_id, candidate_targets in sorted(conflicts.items()):
        if record_id not in common:
            raise ValueError(
                "a private or rare record occurs in multiple original tails: "
                f"record_id={record_id}"
            )
        old_owners = [
            target
            for target in candidate_targets
            if record_id in baseline_top[target]
        ]
        if len(old_owners) != 1:
            raise RuntimeError(
                "baseline repair did not choose one conflict owner: "
                f"record_id={record_id}, owners={old_owners}"
            )
        old_owner = old_owners[0]
        proposed_owner = min(
            candidate_targets,
            key=lambda target: (
                int(ranks["v4"].loc[record_id, target])
                + int(ranks["guide"].loc[record_id, target]),
                int(ranks["v4"].loc[record_id, target]),
                int(ranks["v3"].loc[record_id, target]),
                int(target),
            ),
        )
        guide_gain = int(
            ranks["guide"].loc[record_id, old_owner]
            - ranks["guide"].loc[record_id, proposed_owner]
        )
        v4_penalty = int(
            ranks["v4"].loc[record_id, proposed_owner]
            - ranks["v4"].loc[record_id, old_owner]
        )
        switch = proposed_owner != old_owner
        owner = proposed_owner if switch else old_owner
        for target in candidate_targets:
            if target != owner:
                selected[target].discard(record_id)
        owner_audit.append(
            {
                "record_id": record_id,
                "occurrences": candidate_targets,
                "baseline_owner": old_owner,
                "proposed_guide_owner": proposed_owner,
                "guided_owner": owner,
                "switched": switch,
                "guide_gain": guide_gain,
                "v4_penalty": v4_penalty,
                "ranks": {
                    target: {
                        "v3": int(ranks["v3"].loc[record_id, target]),
                        "v4": int(ranks["v4"].loc[record_id, target]),
                        "guide": int(ranks["guide"].loc[record_id, target]),
                    }
                    for target in candidate_targets
                },
            }
        )

    used = set().union(*selected.values())
    slots = [
        target
        for target in targets
        for _ in range(top_l - len(selected[target]))
    ]
    available = sorted(common - used)
    if len(slots) > len(available):
        raise RuntimeError("not enough unassigned common records to refill the tail")
    if slots:
        utility = np.empty((len(slots), len(available)), dtype=float)
        common_array = np.asarray(available, dtype=np.int64)
        for slot_index, target in enumerate(slots):
            utility[slot_index] = (
                indexed["v4"].loc[common_array, target].to_numpy(dtype=float)
                + 1e-6
                * indexed["v3"].loc[common_array, target].to_numpy(dtype=float)
            )
        assigned_slots, assigned_columns = linear_sum_assignment(-utility)
        if len(assigned_slots) != len(slots):
            raise RuntimeError("guided repair did not fill every target slot")
        for slot_index, column_index in zip(
            assigned_slots, assigned_columns, strict=True
        ):
            selected[slots[int(slot_index)]].add(available[int(column_index)])

    selected_flat = [
        record_id for target in targets for record_id in selected[target]
    ]
    if any(len(selected[target]) != top_l for target in targets):
        raise RuntimeError("guided repair produced an incorrect per-target tail size")
    if len(selected_flat) != len(set(selected_flat)):
        raise RuntimeError("guided repair reused a record across target tails")

    output = v4_aligned[["record_id"]].copy()
    id_to_position = {
        int(record_id): position for position, record_id in enumerate(record_ids)
    }
    for target in targets:
        original_order = _descending_ids(v4_aligned, target)
        chosen = selected[target]
        descending_order = [value for value in original_order if value in chosen]
        descending_order.extend(
            value
            for value in original_order
            if value not in chosen and value not in private
        )
        descending_order.extend(
            value for value in original_order if value in private
        )
        if (
            len(descending_order) != len(record_ids)
            or len(set(descending_order)) != len(record_ids)
        ):
            raise RuntimeError(f"guided reconstruction lost records for target {target}")
        values = np.empty(len(record_ids), dtype=float)
        for descending_rank, record_id in enumerate(descending_order):
            values[id_to_position[record_id]] = (
                len(record_ids) - descending_rank
            ) / (len(record_ids) + 1.0)
        output[target] = values

        if set(_descending_ids(output, target)[:top_l]) != selected[target]:
            raise RuntimeError(f"guided top tail differs for target {target}")
        output_private_order = [
            value for value in _descending_ids(output, target) if value in private
        ]
        v4_private_order = [
            value for value in original_order if value in private
        ]
        if output_private_order != v4_private_order:
            raise RuntimeError(f"private order changed for target {target}")
        output_rare_order = [
            value for value in _descending_ids(output, target) if value in rare
        ]
        v4_rare_order = [value for value in original_order if value in rare]
        if output_rare_order != v4_rare_order:
            raise RuntimeError(f"rare order changed for target {target}")

    protected = tuple(str(int(value)) for value in protected_targets)
    unknown_protected = sorted(set(protected) - set(targets), key=int)
    if unknown_protected:
        raise ValueError(f"unknown protected targets: {unknown_protected}")
    for target in protected:
        if _descending_ids(output, target) != _descending_ids(v4_aligned, target):
            raise RuntimeError(f"protected target {target} ranking changed")

    changes_vs_baseline: dict[str, dict[str, list[int]]] = {}
    for target in targets:
        before = baseline_top[target]
        after = selected[target]
        if before != after:
            changes_vs_baseline[target] = {
                "removed": sorted(before - after),
                "added": sorted(after - before),
            }
    switches = [row for row in owner_audit if row["switched"]]
    diagnostics: dict[str, Any] = {
        "method": "v6_daybreak_guided_low_fpr_unique_tail",
        "top_l": top_l,
        "ownership_rule": "minimum_equal_v4_guide_rank_sum",
        "protected_targets": list(protected),
        "baseline": baseline_diagnostics,
        "owners": owner_audit,
        "switch_count": len(switches),
        "switched_record_ids": [row["record_id"] for row in switches],
        "changes_vs_baseline": changes_vs_baseline,
        "final_unique_tail_records": len(set(selected_flat)),
        "final_duplicate_slots": len(selected_flat) - len(set(selected_flat)),
    }
    return output, diagnostics
