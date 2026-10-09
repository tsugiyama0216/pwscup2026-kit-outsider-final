"""Run one frozen model in a fresh process with only a public bundle argument."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
from pathlib import Path
import time

from .adapters import MODELS, MOCK, TARGETS, predict_world, sha, validate_public
from .replay import read_csv_strict


def load_public(directory):
    root = Path(directory).resolve()
    names = ["B.csv", "B_self.csv", "A_bg.csv", "aia_challenge_10.csv", "schema.json", "cfg.json"]
    for k in TARGETS:
        names += [f"targets/C_{k}.csv", f"targets/aia_challenge_{k}.csv"]
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaping public input: " + name)
    # Keep the declared input surface inspectable. Extra private files must never
    # be part of the directory given to this worker.
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if observed != set(names):
        raise ValueError("Public bundle must contain exactly the 50 declared public files")
    read = lambda name: read_csv_strict(root / name)
    data = {"candidates": read("B.csv"), "self_labels": read("B_self.csv"),
            "background": read("A_bg.csv"), "self_challenge": read("aia_challenge_10.csv"),
            "targets": {k: {"release": read(f"targets/C_{k}.csv"),
                             "challenge": read(f"targets/aia_challenge_{k}.csv")} for k in TARGETS},
            "schema": json.loads((root / "schema.json").read_text(encoding="utf-8-sig")),
            "cfg": json.loads((root / "cfg.json").read_text(encoding="utf-8-sig"))}
    validate_public(**data)
    return data, {name: sha(root / name) for name in names}


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", choices=MODELS)
    choice.add_argument("--plugin", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.plugin is not None:
        from .plugins import candidate_name, predict_plugin
        try:
            model_id = candidate_name(args.name)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.name is not None:
            parser.error("--name requires --plugin")
        model_id = args.model
    public_root, output = args.public.resolve(), args.output.resolve()
    allowed = (MOCK / "local-output").resolve()
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        parser.error("Use a new directory beneath mock_test/local-output")
    if output.is_relative_to(public_root) or public_root.is_relative_to(output):
        parser.error("Keep output separate from public inputs")
    adapter_hash = sha(Path(__file__).with_name("adapters.py"))
    worker_hash = sha(Path(__file__))
    plugin_runtime_hash = sha(Path(__file__).with_name("plugins.py")) if args.plugin is not None else None
    started = time.monotonic()
    public, input_hashes = load_public(public_root)
    result = (predict_plugin(args.plugin, model_id, **public) if args.plugin is not None
              else predict_world(args.model, **public))
    if {name: sha(public_root / name) for name in input_hashes} != input_hashes:
        raise RuntimeError("Public input bytes changed during inference")
    if adapter_hash != sha(Path(__file__).with_name("adapters.py")) or worker_hash != sha(Path(__file__)):
        raise RuntimeError("Adapter/worker source changed during execution")
    if plugin_runtime_hash is not None and plugin_runtime_hash != sha(Path(__file__).with_name("plugins.py")):
        raise RuntimeError("Plugin runtime changed during execution")
    output.mkdir(parents=True, exist_ok=False)
    output_hashes = {}
    for key in ("F_mia", "F_aia"):
        path = output / (key + ".csv")
        path.write_bytes(result[key].to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8"))
        output_hashes[path.name] = sha(path)
    from sugiyama_v2.scoring import validate_predictions
    validate_predictions(read_csv_strict(output / "F_mia.csv"), read_csv_strict(output / "F_aia.csv"), {"team_id": 10, **public})
    report = {"model_id": model_id, "pid": os.getpid(), "fresh_process": True,
              "public_input_sha256": input_hashes, "output_sha256": output_hashes,
              "adapter_sha256": adapter_hash,
              "worker_sha256": worker_hash, "plugin_runtime_sha256": plugin_runtime_hash, "input_files_unchanged": True,
              "diagnostics": result["diagnostics"], "seconds": time.monotonic() - started,
              "official_truth_read": False, "official_submission_performed": False}
    (output / "diagnostics.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": model_id, "output": str(output), "seconds": report["seconds"], "output_sha256": output_hashes}), flush=True)


if __name__ == "__main__":
    main()