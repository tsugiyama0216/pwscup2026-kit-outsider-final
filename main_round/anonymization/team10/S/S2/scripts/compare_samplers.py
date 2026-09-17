"""Compare round/sample on one saved DP model; never read the private cohort.

The round branch is a diagnostic control only. Its CSV is never packaged.
Outputs go to local-output; no AIM measurement or candidate selection occurs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from generate_pwscup import (
    load_json, load_public_background, postprocess_submission, sha256_file,
    synthesize_table, validate_public_configuration,
)
from model_snapshot import load_table

MODEL = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    destination = MODEL / "local-output" / "sampler-comparison"
    destination.mkdir(parents=True, exist_ok=True)
    config_path = MODEL / "config" / "aim_main_team10_S2.json"
    snapshot_path = MODEL / "model_snapshot.npz"
    data = MODEL.parents[1] / "participant_data"
    background_path, schema_path = data / "A_bg.csv", data / "schema.json"
    ledger = load_json(MODEL / "release_ledger.json")
    assert ledger["model_snapshot_sha256"] == sha256_file(snapshot_path)
    config, schema = load_json(config_path), load_json(schema_path)
    validate_public_configuration(config, schema)
    background = load_public_background(background_path, config)
    table, metadata = load_table(snapshot_path)
    for key, path in (
        ("config_sha256", config_path),
        ("public_background_sha256", background_path),
        ("public_schema_sha256", schema_path),
    ):
        assert metadata[key] == sha256_file(path), key
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed-seed diagnostic; not a submission candidate search",
        "private_input_read": False, "additional_dp_measurements": 0,
        "model_snapshot_sha256": sha256_file(snapshot_path),
        "sampling_seed": args.seed, "decoding_seed": args.seed,
        "methods": {},
    }
    for method in ("round", "sample"):
        output_path = destination / f"{method}.csv"
        if output_path.exists():
            raise FileExistsError(f"Diagnostic already exists: {output_path}")
        np.random.seed(args.seed)  # Postprocessing only; never controls DP noise.
        if method == "sample":
            synthetic = synthesize_table(table, config)
        else:
            synthetic = table.synthesize(
                rows=int(config["output_rows"]), method="round"
            ).to_pandas()
        output = postprocess_submission(
            synthetic, background, config, schema,
            rng=np.random.default_rng(args.seed),
        )
        output.to_csv(output_path, index=False, encoding="utf-8")
        report["methods"][method] = {
            "rows": len(output), "sha256": sha256_file(output_path),
            "path": str(output_path.relative_to(MODEL)).replace("\\", "/"),
        }
    (destination / "provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
