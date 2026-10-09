"""Trusted public-table plugins for new candidates on the same frozen panels.

Optional FITTED_ASSETS maps repository-relative .json/.joblib paths to SHA256.
Only fixed assets inside main_round/attack model directories are eligible.
Declare them at import time and load them inside predict_world. Every mock_test
artifact, evaluation material, prediction and truth file is ineligible even if
its hash is known. This is a trusted-plugin audit, not a security sandbox.
"""
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

from .adapters import ROOT, MOCK, ATTACK, MODELS, _bootstrap, sha, validate_public
from .provenance import DERIVED_MODELS


def candidate_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value) or ".." in value or value.endswith("."):
        raise ValueError("Candidate name must be 1-64 safe ASCII letters/digits/underscore/dot/hyphen and start with a letter")
    if value.upper().split('.')[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Reserved candidate name")
    if value.casefold() in {model.casefold() for model in set(MODELS) | DERIVED_MODELS}:
        raise ValueError("Frozen model names are reserved; choose a distinct candidate name")
    return value


_MODEL_AREAS = frozenset({"attack_m", "attack_r", "attackr", "attak_r", "attak_t", "attackt", "attack_ts", "m"})
_EVALUATION_DIRECTORIES = frozenset({
    "scoring_truth", "private", "ground_truth", "evidence", "participant_data", "targets",
    "public", "case", "cases", "panel", "panels", "result", "results", "scores",
    "fixtures", "fixture", "evaluation", "evaluations", "predictions", "prediction",
    "submission", "submissions", "provenance", "reference_baseline", "official-kit",
    "downloads", "retest", "mock_test", "scoring", "truth", "labels", "credentials",
})
_EVALUATION_TOKENS = frozenset({
    "truth", "evidence", "evaluation", "evaluations", "fixture", "fixtures",
    "panel", "panels", "case", "cases", "results", "scores", "scoring", "predictions",
})
_EVALUATION_FILENAMES = frozenset({
    "roles.json", "case_metadata.json", "panel.json", "manifest.json", "bundle_manifest.json",
    "world_manifest.json", "generation_manifest.json", "aia_floor_metadata.json", "aia_floor.json",
    "truth.json", "labels.json", "scores.json", "score.json", "result.json", "results.json",
    "summary.json", "diagnostics.json", "report.json",
})


def _evaluation_material(path):
    """Classify locations/names without opening their contents."""
    path = Path(path)
    if path.is_relative_to(MOCK):
        return True  # Includes generated public bundles as well as hidden truth.
    if not path.is_relative_to(ROOT):
        return False
    parts = [part.casefold() for part in path.relative_to(ROOT).parts]
    if path.name.casefold() in _EVALUATION_FILENAMES:
        return True
    return any(part in _EVALUATION_DIRECTORIES
               or bool(set(re.split(r"[^a-z0-9]+", part)) & _EVALUATION_TOKENS)
               for part in parts[:-1])


def _model_asset_location(path):
    if not path.is_relative_to(ATTACK) or _evaluation_material(path):
        return False
    parts = path.relative_to(ATTACK).parts
    return len(parts) >= 2 and parts[0].casefold() in _MODEL_AREAS


def _validated_fitted_asset(relative, digest):
    """Reject unsafe declarations before hashing/reading or allowlisting them."""
    if not isinstance(relative, str) or not relative or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid declared fitted asset")
    candidate = Path(relative)
    if (candidate.is_absolute() or candidate.drive or candidate.root or ":" in relative
            or ".." in candidate.parts or candidate.suffix.casefold() not in {".json", ".joblib"}):
        raise ValueError("Invalid declared fitted asset: " + relative)
    original = ROOT / candidate
    if not _model_asset_location(original):
        raise ValueError("Invalid declared fitted asset (outside fixed model assets or evaluation material): " + relative)
    resolved = original.resolve()
    # Checking both paths rejects a model-looking symlink/junction to a panel,
    # public bundle or truth directory, and an evaluation alias to model files.
    if not resolved.is_relative_to(ROOT) or not _model_asset_location(resolved):
        raise ValueError("Invalid declared fitted asset (resolved location): " + relative)
    return resolved


def _python_origin_paths(path):
    """Include both cached bytecode and its source when either is observed."""
    path = Path(path).resolve()
    result = [path]
    if path.suffix.lower() == ".pyc":
        try:
            result.append(Path(importlib.util.source_from_cache(str(path))).resolve())
        except ValueError:
            result.append(path.with_suffix(".py"))
    return result


@contextmanager
def read_guard(source):
    state = {"active": True, "hashing": False, "assets": set(), "reads": set(),
             "blocked": [], "code_hashes": {}, "code_changes": []}
    source = Path(source).resolve()

    def is_repo_code(path):
        return (path.suffix.lower() in {".py", ".pyc"} and path.is_relative_to(ROOT)
                and not path.is_relative_to(ATTACK / ".venv")
                and not path.is_relative_to(ATTACK / ".tools"))

    def remember_code(path):
        if not is_repo_code(path):
            return
        # Reading a hash also emits 'open'. Suppress only this bookkeeping to
        # avoid recursive audit calls; this is a trusted-code consistency audit.
        state["hashing"] = True
        try:
            for origin in _python_origin_paths(path):
                if not is_repo_code(origin) or not origin.is_file():
                    continue  # Imports probe nonexistent .pyc paths normally.
                digest = sha(origin)
                previous = state["code_hashes"].setdefault(origin, digest)
                if previous != digest:
                    state["code_changes"].append(str(origin))
                    raise RuntimeError("Plugin Python dependency changed during inference: " + str(origin))
        finally:
            state["hashing"] = False

    def audit(event, args):
        if event != "open" or not state["active"] or state["hashing"] or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if is_repo_code(path):
            mode, flags = args[1], args[2]
            writes = (isinstance(mode, str) and any(c in mode for c in "wa+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writes:
                state["code_changes"].append(str(path))
                raise RuntimeError("Plugin Python dependencies are read-only during inference: " + str(path))
            remember_code(path)
        if path != source and not is_repo_code(path) and _evaluation_material(path):
            state["blocked"].append(str(path))
            raise RuntimeError("Plugin attempted to read evaluation material: " + str(path))
        if path == source or path in state["assets"]:
            state["reads"].add(str(path))
            return
        if path.name.lower() == "token.txt":
            state["blocked"].append(str(path))
            raise RuntimeError("Plugin attempted to read a credential")
        if not path.is_relative_to(ROOT):
            return
        if path.is_relative_to(ATTACK / ".venv") or path.is_relative_to(ATTACK / ".tools"):
            return
        if any(part.casefold() in {"private", "ground_truth"} for part in path.relative_to(ROOT).parts):
            state["blocked"].append(str(path))
            raise RuntimeError("Plugin attempted to read non-public data: " + str(path))
        if is_repo_code(path):
            return  # Helpers may live beside a plugin under local-output.
        # After source/runtime exceptions, every repository data read must be
        # a validated declared fitted asset. Extension deny-lists alone permit
        # unrecorded .bin/.yaml/renamed data to change a candidate between cases.
        state["blocked"].append(str(path))
        raise RuntimeError("Plugin attempted a non-public or undeclared repository data read: " + str(path))


    # A cached import can return an existing sys.modules entry without opening
    # its source. Conservatively fingerprint all preloaded repository modules.
    # Fresh model_worker processes are still required for candidate execution.
    root_prefix = os.path.normcase(str(ROOT)) + os.sep
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", {})
        for attribute in ("__file__", "__cached__"):
            origin = namespace.get(attribute)
            if isinstance(origin, (str, bytes, os.PathLike)):
                absolute = os.path.abspath(os.fsdecode(origin))
                if os.path.normcase(absolute).startswith(root_prefix):
                    remember_code(Path(absolute).resolve())
    remember_code(source)
    sys.addaudithook(audit)
    try:
        yield state
    finally:
        state["active"] = False


def predict_plugin(path, name, **public):
    from threadpoolctl import threadpool_limits
    name = candidate_name(name)
    validate_public(**public)
    path = Path(path).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("Plugin must be an existing Python file")
    source_hash = sha(path)
    _bootstrap()
    assets = {}
    with read_guard(path) as audit, threadpool_limits(limits=1):
        module_name = "_rank_candidate_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
            if not callable(getattr(module, "predict_world", None)):
                raise ValueError("Plugin must expose predict_world(**public_tables)")
            declared = getattr(module, "FITTED_ASSETS", {})
            if not isinstance(declared, dict):
                raise ValueError("FITTED_ASSETS must be a relative-path/SHA256 dictionary")
            for relative, digest in declared.items():
                asset = _validated_fitted_asset(relative, digest)
                audit["assets"].add(asset)
                if sha(asset) != digest:
                    raise ValueError("Fitted asset hash mismatch: " + relative)
                assets[relative] = digest
            answer = module.predict_world(**copy.deepcopy(public))
            if not isinstance(answer, dict) or not {"F_mia", "F_aia"}.issubset(answer):
                raise ValueError("Plugin must return F_mia and F_aia DataFrames")
        finally:
            sys.modules.pop(module_name, None)
    changed_code = [str(origin) for origin, digest in audit["code_hashes"].items()
                    if not origin.is_file() or sha(origin) != digest]
    if audit["code_changes"] or changed_code:
        raise RuntimeError("Plugin Python dependency changed during inference: " + ", ".join(audit["code_changes"] + changed_code))
    if audit["blocked"] or sha(path) != source_hash or any(sha(ROOT / relative) != digest for relative, digest in assets.items()):
        raise RuntimeError("Plugin read forbidden data or its source/assets changed")
    from sugiyama_v2.scoring import validate_predictions
    validation = validate_predictions(answer["F_mia"], answer["F_aia"], {"team_id": 10, **public})
    answer["diagnostics"] = {"model_id": name, "trusted_plugin": True, "plugin_path": str(path),
        "plugin_sha256": source_hash, "fitted_assets_sha256": assets, "validation": validation,
        "fitted_asset_policy": "fixed_attack_model_directories_only_no_evaluation_material_v2",
        "imported_source_sha256": {origin.relative_to(ROOT).as_posix(): digest
                                   for origin, digest in sorted(audit["code_hashes"].items())},
        "imported_dependency_capture": "repository_python_reads_and_preloaded_module_origins_v1",
        "imported_sources_unchanged": True,
        "public_only_arguments": list(public), "repository_data_read_audit": True,
        "blocked_reads": audit["blocked"], "declared_source_and_asset_reads": sorted(audit["reads"]),
        "plugin_diagnostics": answer.get("diagnostics", {}),
        "limitation": "Trusted local Python, run in a fresh worker. The audit covers repository data and Python file origins, not an OS sandbox, third-party packages, or helper code outside the repository. Preloaded module origins are conservatively fingerprinted; arbitrary dynamically generated code is not fully captured."}
    return answer
