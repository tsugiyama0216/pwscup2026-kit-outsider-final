"""Read-only numeric/byte parity checks, kept outside inference processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import ATTACK, MOCK, MODELS, sha
from .replay import read_csv_strict

REFERENCES = {"Rv1": ATTACK / "Attack_R/v1/submission",
              "Rv2": ATTACK / "Attack_R/v2/submission",
              "pl1": ATTACK / "attackT/pl1/submission",
              "Tv105": ATTACK / "attackT/Tv105/predictions"}
NAMES = ("F_mia.csv", "F_aia.csv")


def compare_csv(produced, reference):
    import numpy as np
    produced, reference = Path(produced), Path(reference)
    p, r = produced.read_bytes(), reference.read_bytes()
    pf, rf = read_csv_strict(produced), read_csv_strict(reference)
    same_layout = list(pf.columns) == list(rf.columns) and pf.shape == rf.shape
    pa, ra = pf.to_numpy(dtype=float), rf.to_numpy(dtype=float)
    if not np.isfinite(pa).all() or not np.isfinite(ra).all():
        raise ValueError("Nonfinite predictions cannot establish production parity")
    exact = same_layout and np.array_equal(pa, ra)
    return {"produced": str(produced.resolve()), "reference": str(reference.resolve()),
            "produced_sha256": sha(produced), "reference_sha256": sha(reference),
            "byte_identical": p == r,
            "newline_normalized_byte_identical": p.replace(b"\r\n", b"\n") == r.replace(b"\r\n", b"\n"),
            "same_columns_shape_and_order": same_layout, "numeric_identical": bool(exact),
            "max_absolute_error": float(np.max(np.abs(pa-ra))) if same_layout else None}


def verify(production, batch=None):
    production = Path(production).resolve()
    result = {"production": {}, "batch": {}, "official_submission_performed": False,
              "read_only_comparison": True, "comparator_sha256": sha(__file__)}
    inputs = None
    for model in MODELS:
        directory = production / model
        diagnostic = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
        current = diagnostic["public_input_sha256"]
        if diagnostic["model_id"] != model or diagnostic["input_files_unchanged"] is not True:
            raise ValueError("Worker identity/preservation check failed: " + model)
        if inputs is not None and inputs != current:
            raise ValueError("Independent workers used different public inputs")
        inputs = current
        for name in NAMES:
            if sha(directory / name) != diagnostic["output_sha256"][name]:
                raise ValueError("Worker output changed: " + model + "/" + name)
        result["production"][model] = {name: compare_csv(directory/name, REFERENCES[model]/name) for name in NAMES}
        if batch is not None and model != "pl1":
            shared = Path(batch).resolve() / model
            shared_diag = json.loads((shared / "diagnostics.json").read_text(encoding="utf-8"))
            if (shared_diag["model_id"] != model or shared_diag["public_input_sha256"] != current
                    or shared_diag["input_files_unchanged"] is not True):
                raise ValueError("Shared batch identity/input check failed: " + model)
            for name in NAMES:
                if sha(shared / name) != shared_diag["output_sha256"][name]:
                    raise ValueError("Shared output changed: " + model + "/" + name)
            result["batch"][model] = {name: compare_csv(shared/name, directory/name) for name in NAMES}
    result["public_input_sha256"] = inputs
    for name, expected in inputs.items():
        path = (production / "public" / name).resolve()
        if not path.is_relative_to(production / "public") or sha(path) != expected:
            raise ValueError("Archived public bundle changed: " + name)
    rows = [row for section in ("production", "batch") for files in result[section].values() for row in files.values()]
    result["all_numeric_exact"] = all(row["numeric_identical"] for row in rows)
    result["all_newline_normalized_bytes_exact"] = all(row["newline_normalized_byte_identical"] for row in rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    allowed = (MOCK / "local-output").resolve()
    if output.exists() or not output.is_relative_to(allowed) or output.suffix != ".json":
        parser.error("Use a new JSON report under mock_test/local-output")
    report = verify(args.production, args.batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "all_numeric_exact": report["all_numeric_exact"],
                      "all_newline_normalized_bytes_exact": report["all_newline_normalized_bytes_exact"]}), flush=True)
    if not report["all_numeric_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()