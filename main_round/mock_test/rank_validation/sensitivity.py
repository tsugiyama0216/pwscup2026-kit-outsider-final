"""Explicit scenario diagnostics for the archived mock_TS design.

Family weights below are not estimated real-world frequencies. They are shown
only to expose the 10/12 weight given to related R defense versions.
"""
from collections import defaultdict
import math

from .core import SCORE_KEYS

DEFENSE_FAMILIES = {"M1_v5": "M", "S1": "S", **{name: "R" for name in ("R1", "R3", "R4", "R5", "R6", "R7", "R9", "R10", "R11", "R12")}}


def mean_delta(pairs):
    return {key: math.fsum(row["delta"][key] for row in pairs) / len(pairs) for key in SCORE_KEYS}


def mock_sensitivity(comparison, source_rows):
    if not source_rows or any(row.get("metadata", {}).get("experiment", {}).get("method_family") != "mock_TS" for row in source_rows):
        return {"available": False, "reason": "Not the archived single-target mock_TS design"}
    pairs = comparison["paired_rows"]
    if {row["release_id"] for row in pairs} != set(DEFENSE_FAMILIES) or len(pairs) != len({row["case_key"] for row in pairs}) * len(DEFENSE_FAMILIES):
        return {"available": False, "reason": "The complete fixed set of twelve defenses is required"}
    families, strata = defaultdict(list), defaultdict(list)
    rare_by_case = {}
    for row in source_rows:
        count = row["metadata"].get("mia_rare_positive_count")
        if count is None:
            return {"available": False, "reason": "Rare member counts unavailable"}
        previous = rare_by_case.setdefault(row["case_key"], count)
        if previous != count:
            raise ValueError("Inconsistent rare member count in same case")
    for row in pairs:
        if row["release_id"] not in DEFENSE_FAMILIES:
            return {"available": False, "reason": "Defense family not explicitly mapped"}
        families[DEFENSE_FAMILIES[row["release_id"]]].append(row)
        strata[rare_by_case[row["case_key"]]].append(row)
    family_means = {name: mean_delta(rows) for name, rows in sorted(families.items())}
    equal_family = {key: math.fsum(row[key] for row in family_means.values()) / len(family_means) for key in SCORE_KEYS}
    omitted = {}
    for name in sorted(families):
        kept = [row for family, rows in families.items() if family != name for row in rows]
        if kept:
            omitted[name] = mean_delta(kept)
    return {"available": True, "family_mapping": DEFENSE_FAMILIES,
            "family_cell_counts": {name: len(rows) for name, rows in sorted(families.items())},
            "family_mean_delta": family_means, "equal_family_mean_delta": equal_family,
            "leave_one_family_out_delta": omitted,
            "rare_positive_strata": {str(count): {"paired_cells": len(rows), "mean_delta": mean_delta(rows),
                                                   "mia_rare_defined": count > 0} for count, rows in sorted(strata.items())},
            "weights_are_official_population_estimates": False,
            "note": "Equal-family and family omission are sensitivity scenarios, not a replacement official ranking. Rare-positive zero means undefined rare TPR, not an observed zero success rate."}
