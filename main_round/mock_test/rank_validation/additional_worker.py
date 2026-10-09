"""Run frozen Rv3/Mv1/Mv2 on one public world, in a fresh process."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import importlib
import json
from pathlib import Path
import time

from .adapters import ROOT, MOCK, _bootstrap, sha, validate_public
from .model_worker import clean, load_public
from .plugins import read_guard
from .provenance import model_signature, _path_hashes
from .replay import read_csv_strict

MODELS = ("Rv3", "Mv1", "Mv2")
SOURCES = ("additional_worker.py", "adapters.py", "model_worker.py", "plugins.py", "provenance.py", "replay.py")


def adapter_for(model):
    if model not in MODELS:
        raise ValueError("Unknown additional frozen model")
    return Path(__file__).with_name("additional_r.py" if model == "Rv3" else "additional_m.py")


def source_snapshot(model, fixed):
    fixed = _path_hashes(fixed, "fixed model sources", nonempty=True)
    result = dict(fixed)
    for relative, digest in fixed.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise ValueError("Frozen source mismatch: " + relative)
    for path in [Path(__file__).with_name(name) for name in SOURCES] + [adapter_for(model)]:
        key, digest = path.relative_to(ROOT).as_posix(), sha(path)
        if key in result and result[key] != digest:
            raise ValueError("Conflicting fixed source: " + key)
        result[key] = digest
    return dict(sorted(result.items()))


def predict_frozen(model, public):
    from threadpoolctl import threadpool_limits
    validate_public(**public)
    _bootstrap()
    adapter = adapter_for(model)
    module = importlib.import_module("." + adapter.stem, __package__)
    if Path(module.__file__).resolve() != adapter.resolve():
        raise ValueError("Wrong additional model adapter origin")
    predictor, gate, fixed = module.prepare(model)
    fixed = _path_hashes(fixed, "fixed model sources", nonempty=True)
    sources = source_snapshot(model, fixed)
    before = copy.deepcopy(gate())
    if not isinstance(before, dict) or not before:
        raise ValueError("Missing frozen model gate")
    if "files_sha256" in before and before["files_sha256"] != fixed:
        raise ValueError("Gate and fixed source evidence differ")
    before["files_sha256"] = fixed
    with read_guard(adapter) as audit, threadpool_limits(limits=1):
        answer = predictor(copy.deepcopy(public))
    if audit["blocked"] or audit["code_changes"]:
        raise RuntimeError("Frozen inference attempted a forbidden read or source change")
    if any(not path.is_file() or sha(path) != digest for path, digest in audit["code_hashes"].items()):
        raise RuntimeError("Imported model code changed during inference")
    after = copy.deepcopy(gate())
    after["files_sha256"] = fixed
    if after != before or source_snapshot(model, fixed) != sources:
        raise ValueError("Frozen sources or settings changed during inference")
    if not isinstance(answer, dict) or not {"F_mia", "F_aia"}.issubset(answer):
        raise ValueError("Adapter must return both prediction DataFrames")
    from sugiyama_v2.scoring import validate_predictions
    validation = validate_predictions(answer["F_mia"], answer["F_aia"], {"team_id": 10, **public})
    answer["diagnostics"] = {
        "model_id": model, "algorithm_changed": False, "frozen_gates": before,
        "validation": validation, "public_only_arguments": list(public),
        "blocked_reads": audit["blocked"], "imported_sources_unchanged": True,
        "imported_source_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in sorted(audit["code_hashes"].items())},
        "imported_dependency_capture": "repository_python_reads_and_preloaded_module_origins_v1",
        "numeric_diagnostics": answer.get("diagnostics", {}),
        "limitation": "Frozen production algorithms on public tables. Trusted Python read audit is not an OS sandbox; production parity is checked separately.",
    }
    return answer, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    public_root, output = args.public.resolve(), args.output.resolve()
    allowed = (MOCK / "local-output").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        parser.error("Use a new output directory inside mock_test/local-output")
    if output.is_relative_to(public_root) or public_root.is_relative_to(output):
        parser.error("Keep public inputs and outputs disjoint")
    started = time.monotonic()
    public, input_hashes = load_public(public_root)
    answer, sources = predict_frozen(args.model, public)
    _, after_hashes = load_public(public_root)
    if after_hashes != input_hashes:
        raise ValueError("Public input changed during inference")
    output.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for key in ("F_mia", "F_aia"):
        path = output / (key + ".csv")
        path.write_bytes(answer[key].to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8"))
        output_hashes[path.name] = sha(path)
    from sugiyama_v2.scoring import validate_predictions
    validate_predictions(read_csv_strict(output / "F_mia.csv"), read_csv_strict(output / "F_aia.csv"), {"team_id": 10, **public})
    report = {"model_id": args.model, "additional_frozen": True, "pid": os.getpid(), "fresh_process": True,
        "public_input_sha256": input_hashes, "input_files_unchanged": True, "output_sha256": output_hashes,
        "adapter_sha256": sources[adapter_for(args.model).relative_to(ROOT).as_posix()],
        "worker_sha256": sources[Path(__file__).relative_to(ROOT).as_posix()],
        "additional_source_sha256": sources, "diagnostics": answer["diagnostics"],
        "seconds": time.monotonic() - started, "official_truth_read": False, "official_submission_performed": False}
    report = clean(report)
    model_signature(report)  # Reject incomplete provenance before publishing diagnostics.
    (output / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "seconds": report["seconds"], "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
