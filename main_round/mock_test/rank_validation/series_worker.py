"""Run frozen Rv6--Rv11 once using public tables and same-world Rv2 arrays.

The numeric predictor receives no filesystem paths, generated-case metadata,
truth or score. The production runtime is used only for its frozen hash gate.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import json
from pathlib import Path
import time

from .adapters import ROOT, MOCK, ATTACK, TARGETS, _bootstrap, _load, sha, validate_public
from .model_worker import clean, load_public
from .plugins import read_guard
from .provenance import model_signature
from .replay import read_csv_strict

SERIES = ROOT / "main_round/anonymization/team10/attack/attackR/rv6_series"
MODELS = ("Rv6", "Rv7", "Rv8", "Rv9", "Rv10", "Rv11")
BASELINE_FILES = ("F_mia.csv", "F_aia.csv", "diagnostics.json")
SOURCE_FILES = ("series_worker.py", "adapters.py", "model_worker.py", "plugins.py", "provenance.py", "replay.py")


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_baseline(directory, public, input_hashes):
    """Gate and align an existing Rv2 worker output from exactly this world."""
    _bootstrap()
    from sugiyama_v2.scoring import validate_predictions
    root = Path(directory).resolve()
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != set(BASELINE_FILES):
        raise ValueError("Baseline directory must contain only two predictions and worker diagnostics")
    for name in BASELINE_FILES:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Baseline file escapes its declared directory")
    before = {name: sha(root / name) for name in BASELINE_FILES}
    diagnostic = json_read(root / "diagnostics.json")
    if diagnostic.get("model_id") != "Rv2" or diagnostic.get("public_input_sha256") != input_hashes:
        raise ValueError("Baseline must be Rv2 from the identical public input hashes")
    if diagnostic.get("input_files_unchanged") is not True:
        raise ValueError("Baseline input-preservation verification is missing")
    identity = model_signature(diagnostic)
    if identity["model_id"] != "Rv2" or identity["kind"] not in ("frozen_individual", "frozen_batch"):
        raise ValueError("Baseline is not a recorded frozen Rv2 model")
    if "model_signature" in diagnostic and diagnostic["model_signature"] != identity:
        raise ValueError("Recorded baseline model signature differs")
    expected_outputs = diagnostic.get("output_sha256")
    if not isinstance(expected_outputs, dict) or set(expected_outputs) != set(BASELINE_FILES[:2]):
        raise ValueError("Baseline output hashes are missing or malformed")
    if any(before[name] != expected_outputs[name] for name in BASELINE_FILES[:2]):
        raise ValueError("Baseline output bytes changed")
    mia, aia = (read_csv_strict(root / name) for name in BASELINE_FILES[:2])
    validate_predictions(mia, aia, {"team_id": 10, **public})
    aligned_mia = mia.set_index("record_id", verify_integrity=True).loc[public["candidates"].record_id]
    m = {k: aligned_mia[str(k)].to_numpy(dtype=float, copy=True) for k in TARGETS}
    a = {k: aia.loc[aia.target_k.eq(k)].set_index("challenge_row_id", verify_integrity=True)
              .loc[public["targets"][k]["challenge"].challenge_row_id].time_hat.to_numpy(dtype=float, copy=True)
         for k in TARGETS}
    if {name: sha(root / name) for name in before} != before:
        raise ValueError("Baseline changed while loading")
    return m, a, {"file_sha256": before, "model_signature": identity,
                  "output_sha256": expected_outputs, "diagnostics_sha256": before["diagnostics.json"]}


def verify_baseline_unchanged(directory, record):
    root = Path(directory).resolve()
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != set(BASELINE_FILES) or {name: sha(root / name) for name in BASELINE_FILES} != record["file_sha256"]:
        raise ValueError("Baseline files changed during series inference")
    if model_signature(json_read(root / "diagnostics.json")) != record["model_signature"]:
        raise ValueError("Baseline model identity changed during series inference")


def _sources():
    paths = [Path(__file__).with_name(name) for name in SOURCE_FILES]
    paths += [SERIES / "model.py", SERIES / "runtime.py", ATTACK / "Attack_M/v1/actual.py"]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def series_gate(runtime):
    if Path(runtime.ROOT).resolve() != ROOT or Path(runtime.HERE).resolve() != SERIES:
        raise ValueError("Wrong frozen series runtime origin")
    count = runtime.verify()
    settings = json_read(SERIES / "protocol.json")
    if settings.get("series") != list(MODELS):
        raise ValueError("Unexpected frozen series roster")
    return settings, {"series_verified_files": count,
                      "manifest_sha256": sha(SERIES / "dependencies.json"),
                      "protocol_sha256": sha(SERIES / "protocol.json")}


def frames_from_outputs(outputs, public):
    import numpy as np
    import pandas as pd
    from sugiyama_v2.scoring import validate_predictions
    if not isinstance(outputs, dict) or set(outputs) != set(MODELS):
        raise ValueError("Numeric predictor did not return exactly six models")
    frames = {}
    for name in MODELS:
        answer = outputs[name]
        if not isinstance(answer, dict) or set(answer) != {"mia", "aia"}:
            raise ValueError("Invalid series prediction keys: " + name)
        if set(answer["mia"]) != set(TARGETS) or set(answer["aia"]) != set(TARGETS):
            raise ValueError("Invalid series target roster: " + name)
        mia = pd.DataFrame({"record_id": public["candidates"].record_id.to_numpy()})
        parts = []
        for k in TARGETS:
            m, a = np.asarray(answer["mia"][k], dtype=float), np.asarray(answer["aia"][k], dtype=float)
            if m.shape != (len(public["candidates"]),) or a.shape != (len(public["targets"][k]["challenge"]),):
                raise ValueError("Invalid series vector shape: " + name)
            mia[str(k)] = m
            parts.append(pd.DataFrame({"target_k": k,
                "challenge_row_id": public["targets"][k]["challenge"].challenge_row_id.to_numpy(), "time_hat": a}))
        aia = pd.concat(parts, ignore_index=True)
        validation = validate_predictions(mia, aia, {"team_id": 10, **public})
        frames[name] = {"F_mia": mia, "F_aia": aia, "validation": validation}
    return frames


def predict_series(public, baseline_mia, baseline_aia):
    """Use one frozen numeric call; hash verification remains outside inference."""
    from threadpoolctl import threadpool_limits
    validate_public(**public)
    _bootstrap()
    source_before = _sources()
    runtime = _load("_rank_rv6_series_runtime", SERIES / "runtime.py")
    numeric = _load("_rank_rv6_series_numeric", SERIES / "model.py")
    mask_module = _load("_rank_rv6_series_private_mask", ATTACK / "Attack_M/v1/actual.py")
    settings, gate = series_gate(runtime)
    tables = copy.deepcopy(public)
    with read_guard(SERIES / "model.py") as audit, threadpool_limits(limits=1):
        mask, proof = mask_module.derive_private_mask(tables["candidates"].astype(str), tables["self_labels"].astype(str),
            tables["self_challenge"].astype(str), tables["schema"]["aia_qi_columns"], tables["schema"]["submitted_c_columns"])
        outputs, evidence = numeric.predict_all(
            candidates=tables["candidates"], background=tables["background"],
            targets={k: tables["targets"][k]["release"] for k in TARGETS},
            challenges={k: tables["targets"][k]["challenge"] for k in TARGETS},
            baseline_mia=copy.deepcopy(baseline_mia), baseline_aia=copy.deepcopy(baseline_aia),
            known_negative=mask, settings=copy.deepcopy(settings),
            progress=lambda k: print(f"Rv6--Rv11 public target complete ({k})", flush=True))
    if audit["blocked"] or audit["code_changes"]:
        raise RuntimeError("Series inference attempted a forbidden read or source change")
    if any(not path.is_file() or sha(path) != digest for path, digest in audit["code_hashes"].items()):
        raise RuntimeError("Series imported Python source changed during inference")
    settings_after, gate_after = series_gate(runtime)
    if source_before != _sources() or settings_after != settings or gate_after != gate:
        raise ValueError("Series code, frozen dependencies or settings changed")
    frames = frames_from_outputs(outputs, public)
    provenance = {"series_source_sha256": source_before, "series_protocol_sha256": gate["protocol_sha256"],
        "series_dependency_manifest_sha256": gate["manifest_sha256"], "frozen_gates": gate,
        "private_proof": proof, "blocked_reads": audit["blocked"],
        "imported_source_sha256": {path.relative_to(ROOT).as_posix(): digest
                                   for path, digest in sorted(audit["code_hashes"].items())},
        "imported_dependency_capture": "repository_python_reads_and_preloaded_module_origins_v1"}
    return frames, provenance, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    public_root, baseline_root, output = (path.resolve() for path in (args.public, args.baseline, args.output))
    allowed = (MOCK / "local-output").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        parser.error("Use a new directory under mock_test/local-output")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (public_root, baseline_root)):
        parser.error("Keep output disjoint from public inputs and Rv2 baseline")
    started = time.monotonic()
    public, hashes = load_public(public_root)
    baseline_mia, baseline_aia, baseline = load_baseline(baseline_root, public, hashes)
    frames, recorded, evidence = predict_series(public, baseline_mia, baseline_aia)
    if {name: sha(public_root / name) for name in hashes} != hashes:
        raise ValueError("Public inputs changed during series inference")
    verify_baseline_unchanged(baseline_root, baseline)
    output.mkdir(parents=True, exist_ok=False)
    evidence_path = output / "series_evidence.json"
    evidence_path.write_text(json.dumps(clean(evidence), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from sugiyama_v2.scoring import validate_predictions
    for name, result in frames.items():
        directory = output / name
        directory.mkdir()
        outputs = {}
        for key in ("F_mia", "F_aia"):
            path = directory / (key + ".csv")
            path.write_bytes(result[key].to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8"))
            outputs[path.name] = sha(path)
        validate_predictions(read_csv_strict(directory / "F_mia.csv"), read_csv_strict(directory / "F_aia.csv"), {"team_id": 10, **public})
        diagnostic = {"model_id": name, "pid": os.getpid(), "fresh_process": True,
            "derived_series_batch": True, "public_input_sha256": hashes, "input_files_unchanged": True,
            "output_sha256": outputs,
            "adapter_sha256": recorded["series_source_sha256"][Path(__file__).with_name("adapters.py").relative_to(ROOT).as_posix()],
            "worker_sha256": recorded["series_source_sha256"][Path(__file__).relative_to(ROOT).as_posix()],
            **{key: recorded[key] for key in ("series_source_sha256", "series_protocol_sha256", "series_dependency_manifest_sha256")},
            "baseline_model_signature": baseline["model_signature"], "baseline_output_sha256": baseline["output_sha256"],
            "baseline_diagnostics_sha256": baseline["diagnostics_sha256"], "baseline_files_unchanged": True,
            "shared_evidence_sha256": sha(evidence_path), "seconds": time.monotonic() - started,
            "official_truth_read": False, "official_submission_performed": False,
            "diagnostics": {"model_id": name, "algorithm_changed": False,
                "frozen_gates": recorded["frozen_gates"], "known_negative_proof": recorded["private_proof"],
                "validation": result["validation"], "blocked_reads": recorded["blocked_reads"],
                "imported_source_sha256": recorded["imported_source_sha256"],
                "imported_dependency_capture": recorded["imported_dependency_capture"], "imported_sources_unchanged": True,
                "numeric_api": "rv6_series.model.predict_all", "same_world_baseline_verified": True,
                "public_only_arguments": ["candidates", "background", "targets", "challenges", "baseline_mia", "baseline_aia", "known_negative", "settings"],
                "limitations": ["No official hidden truth is available.", "The trusted Python read audit is not an OS sandbox.",
                                "Baseline dependency identity is recorded, not authenticated retroactively."]}}
        (directory / "diagnostics.json").write_text(json.dumps(clean(diagnostic), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"model": name, "output": str(directory), "seconds": diagnostic["seconds"], "output_sha256": outputs}), flush=True)


if __name__ == "__main__":
    main()