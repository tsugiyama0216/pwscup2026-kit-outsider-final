"""Read-only normalization of archived score handoffs, never new measurements.

All reported metrics are historical observations. Content fingerprints establish
matched scoring inputs, not independent populations or public-input completeness.
"""
from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_METRICS = {
    "score_1": "attack_power", "score_2": "mia_all", "score_3": "mia_rare",
    "score_4": "aia_all", "score_5": "aia_rare",
    "mia_power": "mia_power", "aia_power": "aia_power",
}
INPUT_ATTRIBUTION = "archived_fixture_mapping_public_inputs_not_in_bundle"


def _safe_path(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ValueError(f"Invalid relative bundle reference: {reference!r}")
    part = PurePosixPath(reference)
    if part.is_absolute() or PureWindowsPath(reference).drive or ".." in part.parts:
        raise ValueError(f"Reference escapes bundle: {reference!r}")
    result = (root / reference).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Reference escapes bundle: {reference!r}")
    if not result.exists():
        raise ValueError(f"Missing bundle reference: {reference!r}")
    return result


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(root: Path, reference: str) -> Any:
    def bad_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant in {reference}: {value}")
    return json.loads(_safe_path(root, reference).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=bad_constant)


def _number(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric metric or null: {context}")
    return float(value)


def _metrics(raw: dict[str, Any], context: str) -> dict[str, float | None]:
    result = {}
    for key, alias in _METRICS.items():
        a = _number(raw.get(key), f"{context}/{key}")
        b = _number(raw.get(alias), f"{context}/{alias}")
        if a is not None and b is not None and a != b:
            raise ValueError(f"Conflicting metric aliases: {context}/{key}")
        result[key] = a if a is not None else b
    return result


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _tree_hash(root: Path, reference: str) -> str:
    directory = _safe_path(root, reference)
    if not directory.is_dir():
        raise ValueError(f"Expected directory: {reference}")
    hashes = []
    for path in sorted(directory.rglob("*")):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"File reference escapes bundle: {path.name}")
        if path.is_file():
            hashes.append((path.relative_to(directory).as_posix(), _file_hash(path)))
    if not hashes:
        raise ValueError(f"Empty input directory: {reference}")
    return _digest(hashes)


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty identifier: {context}")
    return value


def _official(root: Path) -> tuple[list[dict], list[dict]]:
    index = _read(root, "official/index.json")
    if not isinstance(index, list):
        raise ValueError("official/index.json must be a list")
    official, unresolved, seen_models, seen_results = [], [], set(), set()
    for entry in index:
        model_id = entry.get("model_id")
        sources = []
        candidates: dict[str, list[tuple[float, str]]] = {k: [] for k in _METRICS}
        sources.append(("official/index.json", entry))
        for key in ("source", "breakdown_file"):
            if entry.get(key):
                sources.append((entry[key], _read(root, entry[key])))
        for reference, source in sources:
            raw = dict(source.get("scores") or {})
            if "score_1" in source:
                top = _number(source["score_1"], f"{reference}/score_1")
                nested = _number(raw.get("score_1"), f"{reference}/scores/score_1")
                if top is not None and nested is not None and top != nested:
                    raise ValueError(f"Conflicting official values: {model_id}/score_1")
                if top is not None:
                    raw["score_1"] = top
            mia, aia = source.get("mia") or {}, source.get("aia") or {}
            for key, value in (("mia_all", mia.get("tpr_at_low_fpr")),
                               ("mia_rare", mia.get("tpr_rare")),
                               ("aia_all", aia.get("r_time_all")),
                               ("aia_rare", aia.get("r_time_rare"))):
                if value is not None:
                    raw[key] = value
            for key, value in _metrics(raw, reference).items():
                if value is not None:
                    candidates[key].append((value, reference))
        row = {"model_id": model_id, "official": True,
               "provenance": entry.get("provenance"),
               "source": [reference for reference, _ in sources],
               "component_provenance": {}, "historical_observation": True}
        for key, values in candidates.items():
            if len({value for value, _ in values}) > 1:
                raise ValueError(f"Conflicting official values: {model_id}/{key}")
            row[key] = values[0][0] if values else None
            row["component_provenance"][key] = [reference for _, reference in values]
        if model_id is None:
            result_id = _identifier(entry.get("result_id"), "unresolved result_id")
            if result_id in seen_results:
                raise ValueError(f"Duplicate unresolved official result: {result_id}")
            seen_results.add(result_id)
            row.update(result_id=result_id, must_not_assume_model=entry.get("must_not_assume_model"))
            unresolved.append(row)
        else:
            model_id = _identifier(model_id, "official model_id")
            if model_id in seen_models:
                raise ValueError(f"Duplicate official model measurement: {model_id}")
            seen_models.add(model_id)
            official.append(row)
    return official, unresolved


def load_handoff(root: str | Path) -> dict[str, Any]:
    """Load matched historical observations without modifying the supplied bundle.

    Missing metrics stay None. Prediction aliases are recorded as aliases, never
    expanded into independent observations. An absent joint22 release is named
    'joint22' explicitly; no per-target official score is used to infer score_1.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Handoff root must be an existing directory")
    experiments = _read(root, "data/experiments.json")
    experiment_map = {}
    for experiment in experiments:
        name = _identifier(experiment.get("experiment_id"), "experiment_id")
        if name in experiment_map:
            raise ValueError(f"Duplicate experiment: {name}")
        experiment_map[name] = experiment
    # Only source files affect scorer identity; machine-generated caches do not.
    scorer_dir = _safe_path(root, "reference/official-kit/src")
    scorer_sources = []
    for path in sorted(scorer_dir.rglob("*")):
        if path.suffix not in (".py", ".yaml", ".yml") or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("Scorer source escapes bundle")
        scorer_sources.append((path.relative_to(scorer_dir).as_posix(), _file_hash(path)))
    if not scorer_sources:
        raise ValueError("Missing pinned official scorer source")
    scorer_id = "sha256:" + _digest(scorer_sources)
    fixtures = _read(root, "data/replay_fixtures.json")
    rare_counts = {}
    records, seen_ids, seen_keys, truth_hashes, config_hashes = [], set(), set(), {}, {}
    for fixture in fixtures:
        fixture_id = _identifier(fixture.get("fixture_id"), "fixture_id")
        if fixture_id in seen_ids:
            raise ValueError(f"Duplicate fixture identifier: {fixture_id}")
        seen_ids.add(fixture_id)
        experiment_id = _identifier(fixture.get("experiment_id"), fixture_id)
        model_id = _identifier(fixture.get("model_id"), fixture_id)
        if experiment_id not in experiment_map:
            raise ValueError(f"Unknown experiment in fixture: {experiment_id}")
        truth_ref, config_ref = fixture["truth_dir"], fixture["config"]
        if truth_ref not in truth_hashes:
            truth_hashes[truth_ref] = _tree_hash(root, truth_ref)
        if experiment_map[experiment_id].get("method_family") == "mock_TS" and truth_ref not in rare_counts:
            truth_path = _safe_path(root, truth_ref)
            with (truth_path / "pool_ids_attacker.csv").open(encoding="utf-8-sig", newline="") as stream:
                attacker_rows = list(csv.DictReader(stream))
            with (truth_path / "pool_ids_0.csv").open(encoding="utf-8-sig", newline="") as stream:
                target_ids = {row["pool_id"] for row in csv.DictReader(stream)}
            rare_counts[truth_ref] = sum(row["is_rare"] == "1" and row["pool_id"] in target_ids for row in attacker_rows)
        if config_ref not in config_hashes:
            config_hashes[config_ref] = _file_hash(_safe_path(root, config_ref))
        case_hash = _digest({"truth": truth_hashes[truth_ref], "config": config_hashes[config_ref]})
        evaluation_hash = _digest({"case": case_hash, "scorer": scorer_id})
        release_id = fixture.get("release_id")
        release_origin = "fixture.release_id"
        if release_id is None:
            if not experiment_id.startswith("joint22_") or fixture.get("attacker_team_id") != 10:
                raise ValueError(f"Missing release_id outside joint22 fixture: {fixture_id}")
            release_id, release_origin = "joint22", "joint22 fixture with release_id=null and attacker_team_id=10"
        release_id = _identifier(release_id, fixture_id)
        key = (experiment_id, model_id, case_hash, release_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate matched observation: {key}")
        seen_keys.add(key)
        prediction_hashes = {key: _file_hash(_safe_path(root, fixture[key])) for key in ("F_mia", "F_aia")}
        source = fixture.get("expected_source")
        if source:
            references = source if isinstance(source, list) else [source]
            for reference in references:
                _safe_path(root, reference)
        row = {"fixture_id": fixture_id, "experiment_id": experiment_id,
               "model_id": model_id, "case_id": fixture.get("case_id"),
               "case_key": case_hash, "release_id": release_id,
               "evaluation_sha256": evaluation_hash, "scorer_id": scorer_id,
               "input_attribution": INPUT_ATTRIBUTION, "source": source,
               **_metrics(fixture["expected_metrics"], fixture_id),
               "metadata": {"historical_observation": True,
                            "independent_new_population": experiment_map[experiment_id].get("independent_new_population", False),
                            "experiment": experiment_map[experiment_id],
                            "release_id_origin": release_origin,
                            "truth_dir": truth_ref, "config": config_ref,
                            "mia_rare_positive_count": rare_counts.get(truth_ref),
                            "prediction_sha256": prediction_hashes,
                            "model_aliases_with_identical_predictions": fixture.get("model_aliases_with_identical_predictions", []),
                            "case_key_scope": "scoring_input_content_not_person_population"}}
        records.append(row)
    official, unresolved = _official(root)
    summaries = []
    for entry in _read(root, "data/summary.json"):
        summaries.append({"experiment_id": entry["experiment_id"], "model_id": entry["model_id"],
                          **_metrics(entry["metrics"], f"summary/{entry['experiment_id']}/{entry['model_id']}"),
                          "source": entry.get("source"), "historical_observation": True,
                          "aggregation_only_not_matched_observation": True})
    return {"records": records, "official": official, "unresolved": unresolved,
            "experiments": experiments, "summary_records": summaries,
            "provenance": {"format_version": 1, "bundle_root": str(root),
                           "fixture_count": len(records), "scorer_id": scorer_id,
                           "input_attribution": INPUT_ATTRIBUTION,
                           "historical_observations_not_new_measurements": True,
                           "official_truth_available": False,
                           "official_submission_csv_identity_independently_verified": False,
                           "official_score_1_not_inferred_from_components": True,
                           "aliases_not_expanded": True,
                           "index_sha256": {p: _file_hash(_safe_path(root, p)) for p in
                                            ("data/replay_fixtures.json", "data/experiments.json",
                                             "data/summary.json", "official/index.json")}}}


