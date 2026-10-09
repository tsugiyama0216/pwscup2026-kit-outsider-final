"""Strict paired diagnostics for archived local scores.

These are descriptive sensitivity checks, not a statistical significance test or
an estimate of official scores. Positive deltas always favour model B. Identical
truth/config fingerprints do not prove that archived public inputs were equal;
that attribution is preserved explicitly in the returned provenance.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

SCORE_KEYS = tuple(f"score_{i}" for i in range(1, 6))
DEFAULT_CONDITION_FIELDS = ("experiment_id", "evaluation_sha256", "scorer_id")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _index(rows, model, condition_fields):
    indexed = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{model} row {position}: expected a mapping")
        item = dict(row)
        for field in ("case_key", "release_id", *condition_fields):
            _text(item.get(field), f"{model} row {position} {field}")
        if "model_id" in item and item["model_id"] != model:
            raise ValueError(f"{model} row {position}: inconsistent model_id")
        for score in SCORE_KEYS:
            value = item.get(score)
            if value is None or isinstance(value, (str, bytes, bool)):
                raise ValueError(f"{model} row {position}: {score} must be a finite number")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{model} row {position}: {score} must be a finite number") from exc
            if not math.isfinite(value):
                raise ValueError(f"{model} row {position}: {score} must be a finite number")
            upper = 2.0 if score == "score_1" else 1.0
            if value < 0.0 or value > upper:
                raise ValueError(f"{model} row {position}: {score} is outside its official range [0, {upper}]")
            item[score] = value
        key = (item["case_key"], item["release_id"])
        if key in indexed:
            raise ValueError(f"{model}: duplicate case/release pair {key!r}")
        indexed[key] = item
    if not indexed:
        raise ValueError(f"{model}: no complete paired score rows")
    return indexed


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def _direction(delta, tolerance):
    return "b_higher" if delta > tolerance else "a_higher" if delta < -tolerance else "tie"


def _metrics(pairs):
    return {score: {
        "mean_a": _mean(pair["a"][score] for pair in pairs),
        "mean_b": _mean(pair["b"][score] for pair in pairs),
        "mean_delta_b_minus_a": _mean(pair["delta"][score] for pair in pairs),
    } for score in SCORE_KEYS}


def _groups(pairs, field, score_key, tolerance):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair[field]].append(pair)
    results = []
    for key, members in sorted(grouped.items()):
        metrics = _metrics(members)
        results.append({field: key, "paired_cells": len(members), "metrics": metrics,
                        "direction": _direction(metrics[score_key]["mean_delta_b_minus_a"], tolerance)})
    return results


def _leave_one_out(pairs, field, score_key, tolerance, full_direction):
    keys = sorted({pair[field] for pair in pairs})
    if len(keys) < 2:
        return {"available": False, "reason": f"Only one {field}; removing it leaves no paired scores",
                "rows": [], "rank_reversal": None, "direction_change_including_tie": None,
                "delta_range": None}
    rows = []
    for key in keys:
        kept = [pair for pair in pairs if pair[field] != key]
        metrics = _metrics(kept)
        delta = metrics[score_key]["mean_delta_b_minus_a"]
        rows.append({"removed": key, "remaining_cells": len(kept), "metrics": metrics,
                     "direction": _direction(delta, tolerance)})
    directions = {row["direction"] for row in rows}
    opposite = {"b_higher": "a_higher", "a_higher": "b_higher"}.get(full_direction)
    values = [row["metrics"][score_key]["mean_delta_b_minus_a"] for row in rows]
    return {"available": True, "rows": rows,
            "rank_reversal": opposite in directions if opposite else False,
            "direction_change_including_tie": any(d != full_direction for d in directions),
            "delta_range": [min(values), max(values)]}


def compare_paired(rows_a, rows_b, *, model_a, model_b, score_key="score_1",
                   condition_fields=DEFAULT_CONDITION_FIELDS, tie_tolerance=1e-12):
    """Compare every matched case/release cell and reject incomplete pairing.

    Each row needs case_key, release_id, score_1..5 and condition_fields.
    experiment_id and scorer_id are always required and must be constant across
    the comparison. Every declared condition must match within each pair.
    evaluation_sha256 identifies evaluation truth/config/scorer, not public C.
    Archived public-input attribution is reported without upgrading it to proof.

    Cell means, case means and release means are separate diagnostics. An
    incomplete case-by-release grid is reported, never silently reweighted.
    Leave-one-out checks remove an entire case/release, preserving dependence
    within the removed group. They are not confidence intervals.
    """
    _text(model_a, "model_a")
    _text(model_b, "model_b")
    if model_a == model_b:
        raise ValueError("model_a and model_b must differ")
    if score_key not in SCORE_KEYS:
        raise ValueError(f"score_key must be one of {SCORE_KEYS}")
    if not isinstance(condition_fields, (tuple, list)) or not condition_fields:
        raise ValueError("condition_fields must be a nonempty list or tuple")
    for field in condition_fields:
        _text(field, "condition field")
    if len(condition_fields) != len(set(condition_fields)):
        raise ValueError("condition_fields contains duplicates")
    if not {"experiment_id", "scorer_id"}.issubset(condition_fields):
        raise ValueError("condition_fields must include experiment_id and scorer_id")
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    a = _index(rows_a, model_a, condition_fields)
    b = _index(rows_b, model_b, condition_fields)
    if set(a) != set(b):
        missing_a = sorted(set(b) - set(a))
        missing_b = sorted(set(a) - set(b))
        raise ValueError(f"Unpaired case/release rows: missing from {model_a}={missing_a!r}; missing from {model_b}={missing_b!r}")
    for field in ("experiment_id", "scorer_id"):
        values = {row[field] for row in (*a.values(), *b.values())}
        if len(values) != 1:
            raise ValueError(f"Cannot mix different {field} values in one comparison")
    pairs = []
    attributions = set()
    for key in sorted(a):
        left, right = a[key], b[key]
        for field in condition_fields:
            if left[field] != right[field]:
                raise ValueError(f"Condition mismatch for {key!r}: {field}")
        # Optional attribution is evidence about archive coverage, not a condition
        # replacing hashes. Keep its absence visible for callers importing CSVs.
        for row in (left, right):
            attribution = row.get("input_attribution", "not_provided")
            attributions.add(_text(attribution, "input_attribution"))
        delta = {score: right[score] - left[score] for score in SCORE_KEYS}
        pairs.append({"case_key": key[0], "release_id": key[1],
                      "conditions": {field: left[field] for field in condition_fields},
                      "a": {score: left[score] for score in SCORE_KEYS},
                      "b": {score: right[score] for score in SCORE_KEYS}, "delta": delta,
                      "direction": _direction(delta[score_key], tie_tolerance)})
    metrics = _metrics(pairs)
    overall_delta = metrics[score_key]["mean_delta_b_minus_a"]
    direction = _direction(overall_delta, tie_tolerance)
    cases = _groups(pairs, "case_key", score_key, tie_tolerance)
    releases = _groups(pairs, "release_id", score_key, tie_tolerance)
    case_values = [row["metrics"][score_key]["mean_delta_b_minus_a"] for row in cases]
    release_values = [row["metrics"][score_key]["mean_delta_b_minus_a"] for row in releases]
    full_grid = len(pairs) == len(cases) * len(releases)
    return {"schema_version": 1, "model_a": model_a, "model_b": model_b,
            "score_key": score_key, "positive_delta_means": "model_b_higher",
            "experiment_id": pairs[0]["conditions"]["experiment_id"],
            "scorer_id": pairs[0]["conditions"]["scorer_id"],
            "condition_fields": list(condition_fields), "tie_tolerance": tie_tolerance,
            "paired_cells": len(pairs), "case_count": len(cases), "release_count": len(releases),
            "complete_case_release_grid": full_grid, "aggregation": "equal_weight_per_matched_cell",
            "metrics": metrics, "direction": direction,
            "cell_counts": {name: sum(pair["direction"] == name for pair in pairs)
                            for name in ("a_higher", "b_higher", "tie")},
            "paired_rows": pairs, "case_means": cases, "release_means": releases,
            "case_mean_delta_range": [min(case_values), max(case_values)],
            "release_mean_delta_range": [min(release_values), max(release_values)],
            "equal_case_mean_delta": _mean(case_values),
            "equal_release_mean_delta": _mean(release_values),
            "leave_one_case_out": _leave_one_out(pairs, "case_key", score_key, tie_tolerance, direction),
            "leave_one_release_out": _leave_one_out(pairs, "release_id", score_key, tie_tolerance, direction),
            "input_attributions": sorted(attributions),
            "statistical_significance_established": False, "official_ranking_validated": False,
            "limitations": [
                "Historical paired observations; repeated cases may reuse people and defense releases.",
                "Evaluation fingerprints verify declared truth/config/scorer, not missing public input bytes.",
                "Leave-one-group-out ranges are sensitivity diagnostics, not confidence intervals.",
                "Official ranking validity requires separate official evidence; local wins alone do not establish it.",
                *( [] if full_grid else ["Incomplete case/release grid: equal-cell, equal-case and equal-release means have different weights."] ),
            ]}