"""Share frozen v88 results across Rv1/Rv2/Tv105 without changing algorithms.

Only public input tables enter prediction. The independent model_worker remains
available. Adoption requires production parity against its independently saved
outputs. This worker never reads those outputs or private truth itself.
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

from .adapters import MOCK, TARGETS, _prepare, _inference_guard, sha, validate_public
from .model_worker import clean, load_public
from .replay import read_csv_strict

MODELS = ("Rv1", "Rv2", "Tv105")


def finish_shared_rv2(values, private, numeric):
    """Check rank idempotence for every target before reusing the baseline."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    shared = values.copy()
    shared[private] = 0.
    exact = numeric.finish_mia(numeric.ranks(values)[:, None], private)[:, 0]
    if not np.array_equal(shared, exact):
        raise ValueError("Shared Rv2 baseline differs from the frozen double-rank algorithm")
    return exact


def predict_shared(**public):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    validate_public(**public)
    public = copy.deepcopy(public)
    prepared = {name: _prepare(name) for name in MODELS}
    gates = {name: p["gate"]() for name, p in prepared.items()}
    assets = {key: value for p in prepared.values() for key, value in p["assets"].items()}
    b, bg, schema, targets = (public[name] for name in ("candidates", "background", "schema", "targets"))
    v1, v2, tv = (prepared[name] for name in MODELS)
    started = time.monotonic()
    routes = {}
    with _inference_guard(assets, schema) as audit, threadpool_limits(limits=1):
        private, proof = v1["derive_mask"](b.astype(str), public["self_labels"].astype(str),
            public["self_challenge"].astype(str), schema["aia_qi_columns"], schema["submitted_c_columns"])
        # Rv1 needs all target columns at once: retain the original joint median.
        keys, matrix = v1["numeric1"].predict_mia_all(targets={k: targets[k]["release"] for k in TARGETS},
            background=bg, candidates=b, private_mask=private, strength=1.0)
        mia = {"Rv1": pd.DataFrame({"record_id": b.record_id.to_numpy(),
            **{str(k): matrix[:, j] for j, k in enumerate(keys)}})}
        mia.update({name: pd.DataFrame({"record_id": b.record_id.to_numpy()}) for name in ("Rv2", "Tv105")})
        aia_parts = {name: [] for name in ("Rv2", "Tv105")}
        for index, k in enumerate(TARGETS, 1):
            tick = time.monotonic()
            item = targets[k]
            variants, route = tv["model"].predict_all(c=item["release"], background=bg, candidates=b,
                challenge=item["challenge"], cfg=public["cfg"], schema=schema)
            chosen = {"Rv2": variants["Rv2"], "Tv105": tv["synthesis"].predict(variants, tv["selection"])}
            mia["Rv2"][str(k)] = finish_shared_rv2(chosen["Rv2"]["mia"], private, v2["numeric2"])
            confidence = np.asarray(chosen["Tv105"]["mia"], dtype=float).copy()
            confidence[private] = 0.
            mia["Tv105"][str(k)] = confidence
            for name, answer in chosen.items():
                aia_parts[name].append(pd.DataFrame({"target_k": k,
                    "challenge_row_id": item["challenge"].challenge_row_id.to_numpy(),
                    "time_hat": np.asarray(answer["aia"], dtype=float).copy()}))
            routes[str(k)] = route
            print(f"Shared public target {index}/22 complete ({k}); {time.monotonic() - tick:.2f}s", flush=True)
        aia = {name: pd.concat(parts, ignore_index=True) for name, parts in aia_parts.items()}
        # Both original Rv1.predict_aia_one and Rv2 select v88 answers['combined'].
        aia["Rv1"] = aia["Rv2"].copy(deep=True)
    if audit["blocked_reads"]:
        raise RuntimeError("A model suppressed a forbidden data-read error")
    if gates != {name: p["gate"]() for name, p in prepared.items()}:
        raise ValueError("Frozen model changed during shared inference")
    from sugiyama_v2.scoring import validate_predictions
    result = {}
    for name in MODELS:
        validation = validate_predictions(mia[name], aia[name], {"team_id": 10, **public})
        result[name] = {"F_mia": mia[name], "F_aia": aia[name], "diagnostics": {
            "model_id": name, "shared_components": True, "algorithm_changed": False,
            "Rv2_finish_exact_per_target": True, "Rv1_AIA_equals_frozen_Rv2_AIA": True,
            "Rv1_joint_MIA_preserved": True, "public_only_arguments": list(public),
            "known_negative_proof": proof, "frozen_gates": gates[name], "validation": validation,
            "verified_asset_reads": sorted(audit["asset_reads"]), "blocked_reads": audit["blocked_reads"],
            "canonical_public_schema_equal_to_supplied": True, "routes": routes,
            "shared_inference_seconds": time.monotonic() - started,
            "limitations": ["Historical fitted assets retain their original training exposure.",
                            "Trusted Python API and read audit are not an operating-system sandbox.",
                            "Independent production parity must be established before using this optimization."]}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    public_root, output = args.public.resolve(), args.output.resolve()
    allowed = (MOCK / "local-output").resolve()
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        parser.error("Use a new directory beneath mock_test/local-output")
    if output.is_relative_to(public_root) or public_root.is_relative_to(output):
        parser.error("Keep output separate from public inputs")
    source_hashes = {name: sha(Path(__file__).with_name(name))
                     for name in ("batch_worker.py", "model_worker.py", "adapters.py")}
    started = time.monotonic()
    public, input_hashes = load_public(public_root)
    results = predict_shared(**public)
    if {name: sha(public_root / name) for name in input_hashes} != input_hashes:
        raise RuntimeError("Public input bytes changed during inference")
    if source_hashes != {name: sha(Path(__file__).with_name(name)) for name in source_hashes}:
        raise RuntimeError("Batch/adapter/loader source changed during execution")
    from sugiyama_v2.scoring import validate_predictions
    output.mkdir(parents=True, exist_ok=False)
    for model_id, result in results.items():
        directory = output / model_id
        directory.mkdir()
        output_hashes = {}
        for key in ("F_mia", "F_aia"):
            path = directory / (key + ".csv")
            path.write_bytes(result[key].to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8"))
            output_hashes[path.name] = sha(path)
        validate_predictions(read_csv_strict(directory / "F_mia.csv"), read_csv_strict(directory / "F_aia.csv"), {"team_id": 10, **public})
        report = {"model_id": model_id, "pid": os.getpid(), "fresh_process": True, "shared_batch": True,
            "public_input_sha256": input_hashes, "output_sha256": output_hashes,
            "adapter_sha256": source_hashes["adapters.py"], "worker_sha256": source_hashes["batch_worker.py"],
            "batch_dependency_sha256": source_hashes, "input_files_unchanged": True,
            "diagnostics": result["diagnostics"], "seconds": time.monotonic() - started,
            "official_truth_read": False, "official_submission_performed": False}
        (directory / "diagnostics.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"model": model_id, "output": str(directory), "seconds": report["seconds"], "output_sha256": output_hashes}), flush=True)


if __name__ == "__main__":
    main()