"""Low-FPR, cross-target tail assignment for the version 5 MIA.

The v4 rank fusion improved the live overall MIA channel, but it was applied
after the v3 transport.  Consequently a handful of records occupy the top
low-FPR tail for more than one target even though the preliminary-round cohort
construction permits a record to be shared with at most one target.

This module repairs only that small tail.  It needs no raw participant rows:

* the 163 guaranteed negatives are the records in the bottom-163 block of
  every v3 target column;
* v4 preserved the v3 ordering within the public rare stratum.  After removing
  those guaranteed negatives, the 11 eligible rare records are therefore the
  unique maximum clique in the graph of pairwise order agreements; and
* a rectangular assignment selects ``top_l`` unique records per target while
  lexicographically maximizing retention of the original v4 tail.

The output keeps every guaranteed negative below every eligible record, keeps
the eligible-rare relative order exactly equal to v3/v4, and emits strict ranks
in ``(0, 1)``.  The supported tail width is deliberately narrow (12--15),
covering the operating region around FPR=0.01 without re-solving the full MIA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_PRIVATE_COUNT = 163
DEFAULT_ELIGIBLE_RARE_COUNT = 11
DEFAULT_TOP_L = 12
MIN_TOP_L = 12
MAX_TOP_L = 15


def _target_columns(frame: pd.DataFrame) -> list[str]:
    if "record_id" not in frame.columns:
        raise ValueError("MIA frame is missing record_id")
    targets = [str(column) for column in frame.columns if column != "record_id"]
    if not targets:
        raise ValueError("MIA frame has no target columns")
    if len(targets) != len(set(targets)):
        raise ValueError("MIA target columns are not unique")
    return targets


def _validate_and_align(
    v3: pd.DataFrame,
    v4: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return validated frames in v4 record/column order."""
    v3_targets = _target_columns(v3)
    v4_targets = _target_columns(v4)
    if set(v3_targets) != set(v4_targets):
        raise ValueError("v3 and v4 MIA target columns differ")
    if len(v3) != len(v4) or len(v3) < 1:
        raise ValueError("v3 and v4 MIA row counts differ or are empty")

    def indexed(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        record_number = pd.to_numeric(frame["record_id"], errors="coerce")
        record_values = record_number.to_numpy(dtype=float)
        if not np.isfinite(record_values).all() or not np.allclose(
            record_values, np.round(record_values)
        ):
            raise ValueError(f"{label} record_id must contain finite integers")
        normalized = frame.copy()
        normalized["record_id"] = record_number.astype(np.int64)
        if normalized["record_id"].duplicated().any():
            raise ValueError(f"{label} record_id is not unique")
        return normalized.set_index("record_id", drop=False)

    v3_indexed = indexed(v3, "v3")
    v4_indexed = indexed(v4, "v4")
    if set(v3_indexed.index) != set(v4_indexed.index):
        raise ValueError("v3 and v4 record_id sets differ")

    record_order = v4_indexed.index.to_numpy(dtype=np.int64)
    v3_aligned = v3_indexed.loc[record_order, ["record_id", *v4_targets]].reset_index(
        drop=True
    )
    v4_aligned = v4_indexed.loc[record_order, ["record_id", *v4_targets]].reset_index(
        drop=True
    )
    for label, frame in (("v3", v3_aligned), ("v4", v4_aligned)):
        numeric = frame[v4_targets].apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{label} MIA scores must be finite and in [0, 1]")
        if any(numeric[target].nunique(dropna=False) != len(frame) for target in v4_targets):
            raise ValueError(f"{label} MIA target columns must be tie-free")
        frame.loc[:, v4_targets] = numeric
    return v3_aligned, v4_aligned, v4_targets


def infer_private_bottom_ids(
    v3: pd.DataFrame,
    *,
    private_count: int = DEFAULT_PRIVATE_COUNT,
) -> tuple[int, ...]:
    """Infer records occupying the common bottom block of all v3 targets."""
    targets = _target_columns(v3)
    if not 0 < private_count < len(v3):
        raise ValueError("private_count must be between zero and the row count")
    if v3["record_id"].duplicated().any():
        raise ValueError("v3 record_id is not unique")

    bottom_sets: list[set[int]] = []
    for target in targets:
        values = pd.to_numeric(v3[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise ValueError(f"v3 target {target} must contain finite, tie-free scores")
        bottom_positions = np.argsort(values, kind="mergesort")[:private_count]
        bottom_sets.append(
            set(v3.iloc[bottom_positions]["record_id"].astype(np.int64).tolist())
        )
    common = set.intersection(*bottom_sets)
    if len(common) != private_count:
        raise ValueError(
            "v3 target columns do not share the expected private bottom block: "
            f"expected={private_count}, observed={len(common)}"
        )
    return tuple(sorted(common))


def _order_agreement_adjacency(
    v3_values: np.ndarray,
    v4_values: np.ndarray,
) -> list[int]:
    """Build a bit-set graph of all-target pairwise order agreements."""
    if v3_values.shape != v4_values.shape or v3_values.ndim != 2:
        raise ValueError("order-agreement matrices must have equal 2-D shapes")
    n_rows = len(v3_values)
    adjacency = [0] * n_rows
    for left in range(n_rows - 1):
        # Scores are tie-free, so equality of the less-than vectors is exactly
        # equality of the pairwise order in every target column.
        same_order = np.all(
            (v3_values[left] < v3_values[left + 1 :])
            == (v4_values[left] < v4_values[left + 1 :]),
            axis=1,
        )
        for right in np.flatnonzero(same_order) + left + 1:
            right_int = int(right)
            adjacency[left] |= 1 << right_int
            adjacency[right_int] |= 1 << left
    return adjacency


def _unique_maximum_clique(adjacency: list[int]) -> tuple[tuple[int, ...], int]:
    """Return the unique maximum clique and the number of maximum cliques.

    This is a bit-set Bron--Kerbosch search with pivoting.  The second return
    value lets callers reject an ambiguous order-preserved subset.
    """
    n_vertices = len(adjacency)
    if n_vertices < 1:
        raise ValueError("order-agreement graph is empty")
    best: tuple[int, ...] = ()
    best_count = 0

    def visit(clique: tuple[int, ...], candidates: int, excluded: int) -> None:
        nonlocal best, best_count
        if len(clique) + candidates.bit_count() < len(best):
            return
        if candidates == 0 and excluded == 0:
            size = len(clique)
            if size > len(best):
                best = clique
                best_count = 1
            elif size == len(best):
                best_count += 1
            return

        pivot_pool = candidates | excluded
        pivot = -1
        pivot_degree = -1
        remaining = pivot_pool
        while remaining:
            bit = remaining & -remaining
            vertex = bit.bit_length() - 1
            remaining ^= bit
            degree = (candidates & adjacency[vertex]).bit_count()
            if degree > pivot_degree:
                pivot = vertex
                pivot_degree = degree
        branch = candidates if pivot < 0 else candidates & ~adjacency[pivot]
        while branch:
            bit = branch & -branch
            vertex = bit.bit_length() - 1
            branch ^= bit
            visit(
                (*clique, vertex),
                candidates & adjacency[vertex],
                excluded & adjacency[vertex],
            )
            candidates ^= bit
            excluded |= bit
            if len(clique) + candidates.bit_count() < len(best):
                return

    visit((), (1 << n_vertices) - 1, 0)
    return best, best_count


def recover_eligible_rare_ids(
    v3: pd.DataFrame,
    v4: pd.DataFrame,
    private_ids: Iterable[int],
    *,
    expected_count: int = DEFAULT_ELIGIBLE_RARE_COUNT,
) -> tuple[int, ...]:
    """Recover the unique v3/v4 order-preserved eligible rare subset."""
    v3_aligned, v4_aligned, targets = _validate_and_align(v3, v4)
    if expected_count < 2:
        raise ValueError("expected rare count must be at least two")
    private = {int(value) for value in private_ids}
    record_ids = v4_aligned["record_id"].to_numpy(dtype=np.int64)
    unknown_private = private - set(record_ids.tolist())
    if unknown_private:
        raise ValueError(f"private_ids contain unknown records: {sorted(unknown_private)}")
    eligible_positions = np.flatnonzero(
        ~np.isin(record_ids, np.asarray(sorted(private), dtype=np.int64))
    )
    if len(eligible_positions) < expected_count:
        raise ValueError("not enough eligible records to recover rare subset")
    v3_values = v3_aligned.loc[eligible_positions, targets].to_numpy(dtype=float)
    v4_values = v4_aligned.loc[eligible_positions, targets].to_numpy(dtype=float)
    adjacency = _order_agreement_adjacency(v3_values, v4_values)
    clique, maximum_count = _unique_maximum_clique(adjacency)
    if len(clique) != expected_count:
        raise ValueError(
            "eligible order-agreement maximum clique has unexpected size: "
            f"expected={expected_count}, observed={len(clique)}"
        )
    if maximum_count != 1:
        raise ValueError(
            "eligible rare subset is ambiguous: "
            f"maximum_cliques={maximum_count}, size={len(clique)}"
        )
    recovered = record_ids[eligible_positions[np.asarray(clique, dtype=int)]]
    return tuple(sorted(int(value) for value in recovered))


def _descending_ids(frame: pd.DataFrame, target: str) -> list[int]:
    return (
        frame.sort_values(target, ascending=False, kind="mergesort")["record_id"]
        .astype(np.int64)
        .tolist()
    )


def assign_unique_top_tail(
    v3: pd.DataFrame,
    v4: pd.DataFrame,
    *,
    top_l: int = DEFAULT_TOP_L,
    private_count: int = DEFAULT_PRIVATE_COUNT,
    eligible_rare_count: int = DEFAULT_ELIGIBLE_RARE_COUNT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deconflict the v4 top tail while preserving private and rare invariants."""
    if not MIN_TOP_L <= top_l <= MAX_TOP_L:
        raise ValueError(f"top_l must be in [{MIN_TOP_L}, {MAX_TOP_L}]")
    v3_aligned, v4_aligned, targets = _validate_and_align(v3, v4)
    private_ids = infer_private_bottom_ids(
        v3_aligned, private_count=private_count
    )
    rare_ids = recover_eligible_rare_ids(
        v3_aligned,
        v4_aligned,
        private_ids,
        expected_count=eligible_rare_count,
    )
    private_set = set(private_ids)
    rare_set = set(rare_ids)
    record_ids = v4_aligned["record_id"].to_numpy(dtype=np.int64)
    all_ids = set(int(value) for value in record_ids)
    eligible_set = all_ids - private_set
    common_eligible = sorted(eligible_set - rare_set)

    original_top: dict[str, set[int]] = {
        target: set(_descending_ids(v4_aligned, target)[:top_l])
        for target in targets
    }
    if any(original_top[target] & private_set for target in targets):
        raise ValueError("a guaranteed-private record unexpectedly occupies the v4 top tail")

    # Rare rows already have the validated v3 order.  Lock their current tail
    # membership and solve only for common eligible rows.
    locked_rare: dict[str, set[int]] = {
        target: original_top[target] & rare_set for target in targets
    }
    rare_owners: dict[int, str] = {}
    for target, values in locked_rare.items():
        for record_id in values:
            if record_id in rare_owners:
                raise ValueError(
                    "an eligible rare record occurs in multiple original top tails: "
                    f"record_id={record_id}, targets={rare_owners[record_id]},{target}"
                )
            rare_owners[record_id] = target

    slots: list[str] = []
    for target in targets:
        slots.extend([target] * (top_l - len(locked_rare[target])))
    if len(slots) > len(common_eligible):
        raise ValueError("not enough common eligible rows for unique tail assignment")

    v3_indexed = v3_aligned.set_index("record_id")
    v4_indexed = v4_aligned.set_index("record_id")
    v3_score = v3_indexed.loc[common_eligible, targets].to_numpy(dtype=float).T
    v4_score = v4_indexed.loc[common_eligible, targets].to_numpy(dtype=float).T
    target_index = {target: index for index, target in enumerate(targets)}
    total_slots = len(slots)
    retention_bonus = float(total_slots + 1)
    utility = np.empty((total_slots, len(common_eligible)), dtype=float)
    common_array = np.asarray(common_eligible, dtype=np.int64)
    for slot_index, target in enumerate(slots):
        index = target_index[target]
        was_in_tail = np.isin(
            common_array,
            np.asarray(sorted(original_top[target] - rare_set), dtype=np.int64),
        )
        utility[slot_index] = (
            retention_bonus * was_in_tail.astype(float)
            + v4_score[index]
            + 1e-6 * v3_score[index]
        )
    assigned_slots, assigned_columns = linear_sum_assignment(-utility)
    if len(assigned_slots) != total_slots:
        raise RuntimeError("tail assignment did not fill every target slot")

    selected: dict[str, set[int]] = {
        target: set(locked_rare[target]) for target in targets
    }
    for slot_index, column_index in zip(assigned_slots, assigned_columns, strict=True):
        selected[slots[int(slot_index)]].add(common_eligible[int(column_index)])
    if any(len(selected[target]) != top_l for target in targets):
        raise RuntimeError("tail assignment produced an incorrect per-target count")
    selected_flat = [record_id for target in targets for record_id in selected[target]]
    if len(selected_flat) != len(set(selected_flat)):
        raise RuntimeError("tail assignment reused a record across targets")

    output = v4_aligned[["record_id"]].copy()
    id_to_position = {
        int(record_id): position for position, record_id in enumerate(record_ids)
    }
    n_rows = len(output)
    for target in targets:
        original_order = _descending_ids(v4_aligned, target)
        chosen = selected[target]
        selected_order = [record_id for record_id in original_order if record_id in chosen]
        remaining_eligible = [
            record_id
            for record_id in original_order
            if record_id in eligible_set and record_id not in chosen
        ]
        private_order = [
            record_id for record_id in original_order if record_id in private_set
        ]
        descending_order = selected_order + remaining_eligible + private_order
        if len(descending_order) != n_rows or len(set(descending_order)) != n_rows:
            raise RuntimeError(f"tail reconstruction lost records for target {target}")
        values = np.empty(n_rows, dtype=float)
        for descending_rank, record_id in enumerate(descending_order):
            values[id_to_position[record_id]] = (
                n_rows - descending_rank
            ) / (n_rows + 1.0)
        output[target] = values

    per_target: dict[str, dict[str, Any]] = {}
    retained_slots = 0
    for target in targets:
        new_top = set(_descending_ids(output, target)[:top_l])
        if new_top != selected[target]:
            raise RuntimeError(f"constructed top tail differs for target {target}")
        removed = sorted(original_top[target] - new_top)
        added = sorted(new_top - original_top[target])
        retained = len(original_top[target] & new_top)
        retained_slots += retained
        per_target[target] = {
            "retained": retained,
            "removed_record_ids": removed,
            "added_record_ids": added,
            "locked_rare_record_ids": sorted(locked_rare[target]),
        }

        private_tail = set(_descending_ids(output, target)[-private_count:])
        if private_tail != private_set:
            raise RuntimeError(f"private bottom block changed for target {target}")
        v3_rare_order = [
            record_id
            for record_id in _descending_ids(v3_aligned, target)
            if record_id in rare_set
        ]
        output_rare_order = [
            record_id
            for record_id in _descending_ids(output, target)
            if record_id in rare_set
        ]
        if output_rare_order != v3_rare_order:
            raise RuntimeError(f"eligible rare order changed for target {target}")
        if output[target].nunique(dropna=False) != n_rows:
            raise RuntimeError(f"output contains score ties for target {target}")

    original_flat = [
        record_id for target in targets for record_id in original_top[target]
    ]
    final_flat = [record_id for target in targets for record_id in selected[target]]
    diagnostics: dict[str, Any] = {
        "method": "v5_low_fpr_unique_tail_assignment",
        "top_l": top_l,
        "target_count": len(targets),
        "tail_slots": len(final_flat),
        "private_count": len(private_ids),
        "private_record_ids": list(private_ids),
        "eligible_rare_count": len(rare_ids),
        "eligible_rare_record_ids": list(rare_ids),
        "original_unique_tail_records": len(set(original_flat)),
        "original_duplicate_slots": len(original_flat) - len(set(original_flat)),
        "final_unique_tail_records": len(set(final_flat)),
        "final_duplicate_slots": len(final_flat) - len(set(final_flat)),
        "retained_tail_slots": retained_slots,
        "changed_tail_slots": len(final_flat) - retained_slots,
        "per_target": per_target,
    }
    return output, diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deconflict the v4 MIA low-FPR tail using v3/v4 rank files."
    )
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-l", type=int, default=DEFAULT_TOP_L)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output, diagnostics = assign_unique_top_tail(
        pd.read_csv(args.v3),
        pd.read_csv(args.v4),
        top_l=args.top_l,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(
        "PASS:",
        f"top_l={diagnostics['top_l']}",
        f"changed={diagnostics['changed_tail_slots']}",
        f"duplicates={diagnostics['final_duplicate_slots']}",
    )


if __name__ == "__main__":
    main()
