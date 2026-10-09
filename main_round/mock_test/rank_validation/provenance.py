"""Stable scoring and recorded model identities for saved panel comparisons.

This module reads code/runtime metadata, never prediction or truth values.
Signatures distinguish recorded inference evidence from dependencies that older
workers did not capture; they do not authenticate old inference retroactively.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_PACKAGE = ROOT / "main_round/attack/official-kit/starter/src/pwscup2026"
REQUIREMENTS = OFFICIAL_PACKAGE.parents[1] / "requirements-kit.txt"
LOCAL_PACKAGE = ROOT / "main_round/mock_test/sugiyama_v2"
OFFICIAL_MODULES = (
    "pwscup2026.kit.codabench.scoring_io",
    "pwscup2026.scoring.attack_score",
    "pwscup2026.scoring.result",
    "pwscup2026.common.submission_schema",
    "pwscup2026.attack.mia",
)
LOCAL_MODULES = ("sugiyama_v2.scoring", "sugiyama_v2.runner", "sugiyama_v2.io")
GENERATOR_KEYS = ("generator_source_sha256", "inherited_generator_sha256", "inherited_releases_sha256")
FROZEN_MODELS = {"Rv1", "Rv2", "pl1", "Tv105"}
DERIVED_MODELS = {f"Rv{i}" for i in range(6, 12)}


def _validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite provenance value")
        return
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Provenance object keys must be strings")
        for item in value.values():
            _validate_json(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item)
        return
    raise ValueError("Provenance must contain ordinary finite JSON values")


def canonical_sha256(value):
    """Reject NaN/Infinity and key coercion before deterministic JSON encoding."""
    _validate_json(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha(path):
    path = Path(path)
    if path.name.lower() == "token.txt":
        raise ValueError("Credentials are not provenance inputs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Missing or invalid SHA256: " + label)
    return value


def _path_hashes(value, label, *, nonempty=False):
    if not isinstance(value, dict) or (nonempty and not value):
        raise ValueError("Expected a path/SHA256 object: " + label)
    result = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            raise ValueError("Expected repository-relative POSIX source path: " + label)
        part = PurePosixPath(name)
        if part.is_absolute() or part.as_posix() != name or ".." in part.parts or name == ".":
            raise ValueError("Escaping or noncanonical provenance path: " + label)
        if part.name.lower() == "token.txt":
            raise ValueError("Credentials are not provenance inputs")
        result[name] = _digest(digest, label + "/" + name)
    return dict(sorted(result.items()))


def _module_source(name, expected_root):
    module = importlib.import_module(name)
    source = getattr(module, "__file__", None)
    if source is None:
        raise ValueError("Imported scoring module has no source origin: " + name)
    source = Path(source).resolve()
    if (source.suffix != ".py" or not source.is_file()
            or not source.is_relative_to(Path(expected_root).resolve())):
        raise ValueError("Wrong scoring module origin: " + name)
    return {"path": source.relative_to(ROOT).as_posix(), "sha256": _sha(source)}


def _runtime_versions():
    versions = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError("Expected pinned kit requirement")
        name, expected = line.split("==")
        if not name or not expected or name in versions:
            raise ValueError("Invalid or duplicate pinned kit requirement")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError("Runtime differs from pinned scoring kit: " + name)
        versions[name] = actual
    if not versions:
        raise ValueError("Scoring runtime requirements are empty")
    return dict(sorted(versions.items()))


def _signed(value):
    return {**value, "signature_sha256": canonical_sha256(value)}


def evaluation_signature(cfg):
    """Identify the actual imported score code, strict cfg and pinned runtime.

    Call before and after scoring and require equality to detect a source change
    during scoring. It is not enough to hash files only after an old run.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Scoring config must be an object")
    cfg_hash = canonical_sha256(cfg)
    official = {name: _module_source(name, OFFICIAL_PACKAGE) for name in OFFICIAL_MODULES}
    local = {name: _module_source(name, LOCAL_PACKAGE) for name in LOCAL_MODULES}
    return _signed({
        "format": "panel-evaluation-signature-v1",
        "config_sha256": cfg_hash,
        "official_modules": official,
        "local_modules": local,
        "requirements_sha256": _sha(REQUIREMENTS),
        "runtime_versions": _runtime_versions(),
        "python": {"implementation": platform.python_implementation(),
                   "version": platform.python_version(),
                   "system": platform.system(), "machine": platform.machine()},
        "scope": "Scoring/config/runtime identity; public input and private truth/floor hashes must also match.",
    })


def model_signature(diagnostics):
    """Extract stable recorded inference identity, excluding timing and inputs.

    The whole worker diagnostics.json is required. Never replace an old recorded
    hash with a current source hash. Batch and individual workers intentionally
    have different identities, even if numeric parity was separately proven.
    """
    if not isinstance(diagnostics, dict):
        raise ValueError("Expected worker diagnostics object")
    detail = diagnostics.get("diagnostics")
    if not isinstance(detail, dict):
        raise ValueError("Missing model diagnostics")
    name = diagnostics.get("model_id")
    if not isinstance(name, str) or not name or detail.get("model_id") != name:
        raise ValueError("Model identities differ")
    if diagnostics.get("input_files_unchanged") is not True:
        raise ValueError("Missing input preservation check")
    if detail.get("blocked_reads"):
        raise ValueError("Inference recorded a forbidden data read")
    sources = {
        "adapter_sha256": _digest(diagnostics.get("adapter_sha256"), "adapter"),
        "worker_sha256": _digest(diagnostics.get("worker_sha256"), "worker"),
    }
    plugin = detail.get("trusted_plugin") is True
    batch = diagnostics.get("shared_batch") is True
    derived = diagnostics.get("derived_series_batch") is True
    if sum((plugin, batch, derived)) > 1:
        raise ValueError("Inference identity kinds cannot be combined")
    result = {"format": "panel-model-signature-v1", "model_id": name,
              "kind": "derived_series_batch" if derived else "plugin" if plugin else "frozen_batch" if batch else "frozen_individual",
              "inference_sources": sources}
    if plugin:
        if name.casefold() in {n.casefold() for n in FROZEN_MODELS | DERIVED_MODELS}:
            raise ValueError("Plugin uses a reserved frozen model identity")
        sources["plugin_runtime_sha256"] = _digest(diagnostics.get("plugin_runtime_sha256"), "plugin runtime")
        result["plugin_sha256"] = _digest(detail.get("plugin_sha256"), "plugin source")
        result["fitted_assets_sha256"] = _path_hashes(detail.get("fitted_assets_sha256"), "fitted assets")
    elif derived:
        if name not in DERIVED_MODELS:
            raise ValueError("Unrecognized fixed Rv6-series identity")
        series_sources = _path_hashes(diagnostics.get("series_source_sha256"), "series sources", nonempty=True)
        base_path = "main_round/mock_test/rank_validation/"
        if (series_sources.get(base_path + "series_worker.py") != sources["worker_sha256"]
                or series_sources.get(base_path + "adapters.py") != sources["adapter_sha256"]):
            raise ValueError("Series worker source identities differ")
        protocol = _digest(diagnostics.get("series_protocol_sha256"), "series protocol")
        manifest = _digest(diagnostics.get("series_dependency_manifest_sha256"), "series dependency manifest")
        baseline = diagnostics.get("baseline_model_signature")
        if (not isinstance(baseline, dict) or baseline.get("format") != "panel-model-signature-v1"
                or baseline.get("model_id") != "Rv2" or baseline.get("kind") not in {"frozen_batch", "frozen_individual"}
                or baseline.get("signature_sha256") != canonical_sha256({k:v for k,v in baseline.items() if k != "signature_sha256"})):
            raise ValueError("Series requires a valid recorded frozen Rv2 identity")
        baseline_sources = baseline.get("inference_sources")
        if not isinstance(baseline_sources, dict):
            raise ValueError("Missing baseline inference source evidence")
        baseline_detail = {"model_id": "Rv2", "frozen_gates": baseline.get("frozen_gates")}
        if baseline.get("imported_dependency_capture") == "recorded":
            baseline_detail["imported_source_sha256"] = baseline.get("imported_source_sha256")
        baseline_worker = {
            "model_id": "Rv2", "input_files_unchanged": True,
            "adapter_sha256": baseline_sources.get("adapter_sha256"),
            "worker_sha256": baseline_sources.get("worker_sha256"),
            "shared_batch": baseline["kind"] == "frozen_batch",
            "batch_dependency_sha256": baseline.get("batch_dependency_sha256"),
            "diagnostics": baseline_detail,
        }
        # Reapply the existing frozen-model validator, including batch-source
        # consistency. A self-signed but incomplete object is not evidence.
        normalized_baseline = model_signature(baseline_worker)
        if normalized_baseline != baseline:
            raise ValueError("Baseline signature has incomplete or inconsistent frozen evidence")
        output_hashes = _path_hashes(diagnostics.get("baseline_output_sha256"), "baseline outputs")
        if set(output_hashes) != {"F_mia.csv", "F_aia.csv"}:
            raise ValueError("Series requires both baseline prediction hashes")
        _digest(diagnostics.get("baseline_diagnostics_sha256"), "baseline diagnostics")
        gates = detail.get("frozen_gates")
        if (not isinstance(gates, dict) or type(gates.get("series_verified_files")) is not int
                or gates["series_verified_files"] <= 0 or gates.get("manifest_sha256") != manifest
                or gates.get("protocol_sha256") != protocol):
            raise ValueError("Series dependency gate evidence differs")
        result.update(series_source_sha256=series_sources, series_protocol_sha256=protocol,
                      series_dependency_manifest_sha256=manifest, baseline_model_signature=normalized_baseline,
                      frozen_gates=json.loads(json.dumps(gates)))
    else:
        if name not in FROZEN_MODELS:
            raise ValueError("Unrecognized frozen model identity")
        gates = detail.get("frozen_gates")
        if not isinstance(gates, dict) or not gates:
            raise ValueError("Missing recorded frozen dependency gates")
        # Gates are recorded evidence, not newly inferred file hashes.
        canonical_sha256(gates)
        result["frozen_gates"] = json.loads(json.dumps(gates))
        result["frozen_gate_evidence"] = "Recorded successful gate output; counts alone are not complete source hashes."
        if batch:
            dependencies = _path_hashes(diagnostics.get("batch_dependency_sha256"), "batch sources", nonempty=True)
            if (set(dependencies) != {"batch_worker.py", "model_worker.py", "adapters.py"}
                    or dependencies["batch_worker.py"] != sources["worker_sha256"]
                    or dependencies["adapters.py"] != sources["adapter_sha256"]):
                raise ValueError("Batch worker dependency identities differ")
            result["batch_dependency_sha256"] = dependencies
    imports = detail.get("imported_source_sha256")
    if (plugin or derived) and imports is None:
        raise ValueError("Imported dependency hashes were not recorded; rerun with the new worker")
    result["imported_dependency_capture"] = "recorded" if imports is not None else "not_recorded"
    result["imported_source_sha256"] = _path_hashes(imports, "imported sources") if imports is not None else None
    return _signed(result)


def generator_signature(metadata):
    """Use recorded generator hashes so different implementations cannot mix."""
    if not isinstance(metadata, dict):
        raise ValueError("Missing generator metadata")
    return _signed({"format": "panel-generator-signature-v1",
                    **{key: _digest(metadata.get(key), key) for key in GENERATOR_KEYS}})
