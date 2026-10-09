"""Pure, preregistered v5 ranking indicator; the official scorer is unchanged.

stable_index = equal-target mean(mia_all + aia_power)
rare_lift    = equal-target mean(mia_power - mia_all)

``aia_power`` is the existing per-target official value AFTER its rare deadzone,
max operation and floor correction/clipping. It is never recalculated here.
Null ``mia_rare`` values are permitted in extra columns because undefined rare
TPR is not an input to this indicator. v4 applications are retrospective design
analyses, not validation of this separately preregistered indicator.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import fsum, isfinite

METRIC_VERSION = "stable-index-v1-20261009-2"
TARGET_IDS = tuple(k for k in (*range(18), 22, 25, 26, 27, 28) if k != 10)
LIFT_TOLERANCE = 1e-12
REQUIRED_COLUMNS = ("target_k", "mia_all", "mia_power", "aia_power")


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(label + ": Boolean values are not metric measurements")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + ": expected a finite number") from exc
    if not isfinite(result):
        raise ValueError(label + ": expected a finite number")
    return result


def _target_id(value):
    number = _number(value, "target_k")
    if not 0 <= number < 2**53 or number != int(number):
        raise ValueError("target_k: expected an exactly representable nonnegative integer")
    return int(number)


def compute_metrics(rows, *, expected_target_ids=TARGET_IDS):
    """Validate complete per-target records and return equal-target means.

    ``rows`` is an iterable of mappings, such as ``frame.to_dict('records')``.
    The default requires all 22 main-round targets exactly once. A leave-one-
    target-out analysis must explicitly supply the remaining expected IDs.
    Neither inputs nor any files are modified; no coefficients are fitted.
    Tiny negative rare lifts within 1e-12 are retained, so every returned mean
    follows its stated arithmetic exactly, and their count is reported.
    """
    try:
        expected = tuple(_target_id(value) for value in expected_target_ids)
    except TypeError as exc:
        raise ValueError("expected_target_ids must be a nonempty iterable of unique IDs") from exc
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_target_ids must be nonempty and unique")
    if isinstance(rows, (Mapping, str, bytes)):
        raise ValueError("rows must be an iterable of per-target mappings")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise ValueError("rows must be an iterable of per-target mappings") from exc
    values = {}
    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("Each row must be a per-target mapping")
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            raise ValueError("Missing metric columns: " + ", ".join(missing))
        target = _target_id(row["target_k"])
        if target in values:
            raise ValueError("Duplicate target_k: " + str(target))
        components = {name: _number(row[name], f"target {target} {name}") for name in REQUIRED_COLUMNS[1:]}
        if any(not 0 <= value <= 1 for value in components.values()):
            raise ValueError(f"target {target}: metric components must be in [0, 1]")
        lift = components["mia_power"] - components["mia_all"]
        if lift < -LIFT_TOLERANCE:
            raise ValueError(f"target {target}: negative rare lift exceeds floating-point tolerance")
        values[target] = {**components, "rare_lift": lift}
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        unexpected = sorted(set(values) - set(expected))
        raise ValueError(f"Incomplete or unexpected targets: missing={missing}, unexpected={unexpected}")
    ordered = [values[target] for target in sorted(expected)]
    count = len(ordered)
    mean = lambda key: fsum(row[key] for row in ordered) / count
    return {
        "metric_version": METRIC_VERSION,
        "target_ids": sorted(expected), "target_count": count,
        "stable_index": fsum(row["mia_all"] + row["aia_power"] for row in ordered) / count,
        "rare_lift": mean("rare_lift"),
        "official_attack_power": fsum(row["mia_power"] + row["aia_power"] for row in ordered) / count,
        "mean_mia_all": mean("mia_all"), "mean_mia_power": mean("mia_power"),
        "mean_aia_power": mean("aia_power"),
        "tolerated_negative_lift_count": sum(row["rare_lift"] < 0 for row in ordered),
        "official_scorer_or_scores_changed": False,
    }