"""Compare additional frozen workers with their archived production CSVs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .adapters import ROOT, ATTACK, TARGETS, sha
from .additional_worker import MODELS
from .model_worker import load_public
from .panel import new_output
from .provenance import model_signature
from .replay import read_csv_strict
from .verify_adapter_parity import compare_csv

REFERENCES = {"Rv3": ROOT / "main_round/anonymization/team10/atack/Rv3/submission",
              "Mv1": ATTACK / "Attack_M/local-output/v1/predictions",
              "Mv2": ATTACK / "Attack_M/local-output/v2/predictions"}


def verify(public, produced):
    _, public_hashes = load_public(public)
    rows, identities = {}, {}
    for model in MODELS:
        directory = Path(produced) / model
        diagnostic = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
        identity = model_signature(diagnostic)
        if identity["model_id"] != model or identity["kind"] != "frozen_additional":
            raise ValueError("Wrong production model identity")
        if diagnostic["public_input_sha256"] != public_hashes:
            raise ValueError("Production public input hashes differ")
        identities[model] = identity
        files = {}
        for name in ("F_mia.csv", "F_aia.csv"):
            if sha(directory / name) != diagnostic["output_sha256"][name]:
                raise ValueError("Saved predictions changed")
            files[name] = compare_csv(directory / name, REFERENCES[model] / name)
        actual, expected = (read_csv_strict(p / "F_mia.csv") for p in (directory, REFERENCES[model]))
        if not np.array_equal(actual.record_id, expected.record_id):
            raise ValueError("MIA production IDs differ")
        ranks = {str(k): bool(np.array_equal(rankdata(actual[str(k)], method="average"),
                                             rankdata(expected[str(k)], method="average"))) for k in TARGETS}
        rows[model] = {"files": files, "mia_target_ranks_identical": ranks,
                       "all_mia_target_ranks_identical": all(ranks.values())}
    items = [f for m in rows.values() for f in m["files"].values()]
    return {"format": "additional-frozen-production-parity-v1", "production": rows,
            "model_signatures": identities, "public_input_sha256": public_hashes,
            "all_six_files_numeric_identical": all(f["numeric_identical"] for f in items),
            "byte_identical_files": sum(f["byte_identical"] for f in items),
            "all_mia_ranks_identical": all(m["all_mia_target_ranks_identical"] for m in rows.values()),
            "all_aia_numeric_identical": all(m["files"]["F_aia.csv"]["numeric_identical"] for m in rows.values()),
            "official_submission_performed": False, "read_only_comparison": True,
            "comparator_sha256": sha(__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--produced", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify(args.public, args.produced)
    output = new_output(args.output)
    (output / "verified_parity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("production", "model_signatures", "public_input_sha256")}), flush=True)
    if not result["all_six_files_numeric_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
