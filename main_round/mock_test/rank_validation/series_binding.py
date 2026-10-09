"""Bind derived-series evidence to the actual same-panel frozen Rv2 files.

Only worker diagnostics and the two saved baseline predictions are read. No
panel metadata, scoring truth, scores or actual-production inputs are accessed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .provenance import _digest, _path_hashes, model_signature

FILES = ("F_mia.csv", "F_aia.csv", "diagnostics.json")


def _files(directory):
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(FILES):
        raise ValueError("Baseline must contain only two predictions and worker diagnostics")
    result = {}
    for name in FILES:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Baseline file escapes its directory or is missing")
        result[name] = path
    return result


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_baseline_binding(baseline_directory, derived_diagnostics):
    """Require the exact Rv2 identity, public world and saved prediction bytes.

    Model signatures deliberately omit per-world baseline hashes. This separate
    check preserves their cross-world comparability without allowing a report
    to silently compare against a different baseline on the same world.
    """
    derived_identity = model_signature(derived_diagnostics)
    if derived_identity["kind"] != "derived_series_batch":
        raise ValueError("Baseline binding requires a fixed derived-series worker")
    if derived_diagnostics.get("baseline_files_unchanged") is not True:
        raise ValueError("Missing recorded baseline preservation check")
    directory = Path(baseline_directory).resolve()
    paths = _files(directory)
    raw = paths["diagnostics.json"].read_bytes()
    diagnostic_hash = hashlib.sha256(raw).hexdigest()
    expected_diagnostic = _digest(derived_diagnostics.get("baseline_diagnostics_sha256"), "baseline diagnostics")
    if diagnostic_hash != expected_diagnostic:
        raise ValueError("Baseline diagnostics differ from series inference")
    baseline = json.loads(raw.decode("utf-8-sig"))
    identity = model_signature(baseline)
    if identity["model_id"] != "Rv2" or identity["kind"] not in {"frozen_batch", "frozen_individual"}:
        raise ValueError("Baseline is not a recorded frozen Rv2")
    if "model_signature" in baseline and baseline["model_signature"] != identity:
        raise ValueError("Baseline worker has an inconsistent stored model signature")
    if identity != derived_identity["baseline_model_signature"]:
        raise ValueError("Baseline model identity differs from series inference")
    public_inputs = _path_hashes(baseline.get("public_input_sha256"), "baseline public inputs", nonempty=True)
    derived_inputs = _path_hashes(derived_diagnostics.get("public_input_sha256"), "series public inputs", nonempty=True)
    if public_inputs != derived_inputs:
        raise ValueError("Series and baseline used different public worlds")
    expected = _path_hashes(derived_diagnostics.get("baseline_output_sha256"), "series baseline outputs")
    recorded = _path_hashes(baseline.get("output_sha256"), "baseline outputs")
    if set(expected) != set(FILES[:2]) or recorded != expected:
        raise ValueError("Baseline prediction hash records differ from series inference")
    actual = {name: _sha(paths[name]) for name in FILES[:2]}
    if actual != expected:
        raise ValueError("Baseline prediction bytes changed after series inference")
    before = {**actual, "diagnostics.json": diagnostic_hash}
    after_paths = _files(directory)
    if {name: _sha(after_paths[name]) for name in FILES} != before:
        raise ValueError("Baseline files changed during binding verification")
    return {"baseline_model_signature": identity, "public_input_sha256": public_inputs,
            "baseline_output_sha256": actual, "baseline_diagnostics_sha256": diagnostic_hash}
