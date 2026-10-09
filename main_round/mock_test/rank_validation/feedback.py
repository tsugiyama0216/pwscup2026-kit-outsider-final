"""Explicit, provenance-preserving supplements to archived official scores."""
import copy
import math

from .core import SCORE_KEYS


def validate_data(data):
    for collection in ("records", "official", "unresolved"):
        if not isinstance(data.get(collection), list):
            raise ValueError(f"{collection} must be a list")
    seen = set()
    for row in data["official"]:
        model = row.get("model_id")
        if not isinstance(model, str) or not model.strip() or model in seen:
            raise ValueError("Official model IDs must be nonempty and unique")
        seen.add(model)
    for row in data["records"]:
        for key in ("model_id", "experiment_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key}")
    for row in data["records"] + data["official"] + data["unresolved"]:
        for key in SCORE_KEYS:
            value = row.get(key)
            limit = 2 if key == "score_1" else 1
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit):
                raise ValueError(f"{key} must be a finite in-range number or null")
        for key, bounds in row.get("score_intervals", {}).items():
            value = row.get(key)
            if key not in SCORE_KEYS or value is None or not isinstance(bounds, list) or len(bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
                raise ValueError("Invalid score interval")
            if not 0 <= bounds[0] <= value <= bounds[1] <= (2 if key == "score_1" else 1):
                raise ValueError("Invalid score interval bounds")


def add_feedback(data, feedback):
    validate_data(data)
    result = copy.deepcopy(data)
    by_model = {row["model_id"]: row for row in result["official"]}
    seen = set()
    for supplement in feedback["rows"]:
        model = supplement["model_id"]
        if not isinstance(model, str) or not model.strip() or model in seen:
            raise ValueError("Duplicate or invalid supplemental model")
        seen.add(model)
        value, decimals = supplement["score_1_display"], supplement["decimal_places"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2 or isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 12:
            raise ValueError("Invalid displayed score or precision")
        half_unit = 0.5 * 10 ** (-decimals)
        low, high = max(0, value - half_unit), min(2, value + half_unit)
        if model not in by_model:
            row = {"model_id": model, "official": True, **{key: None for key in SCORE_KEYS}}
            result["official"].append(row)
            by_model[model] = row
        row = by_model[model]
        exact = row.get("score_1")
        if exact is not None and not low - 1e-12 <= exact <= high + 1e-12:
            raise ValueError("Displayed score contradicts archived official score: " + model)
        if exact is None:
            row["score_1"] = value
            row.setdefault("score_intervals", {})["score_1"] = [low, high]
            row["score_1_precision"] = "four_decimal_display" if decimals == 4 else "rounded_display"
        row["display_evidence"] = {**supplement, "source_filename": feedback.get("source_filename"), "source_sha256": feedback.get("source_sha256")}
    result.setdefault("provenance", {})["supplemental_official_feedback"] = feedback
    validate_data(result)
    return result
