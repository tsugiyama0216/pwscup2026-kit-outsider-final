"""Validate the frozen M1 CSV and its public-only lineage."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODEL = Path(__file__).resolve().parents[1]
TEAM = MODEL.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    c_path = MODEL / "C.csv"
    s1_path = TEAM / "S" / "S1" / "C.csv"
    data = TEAM / "participant_data"
    schema_path = data / "schema.json"
    report_path = MODEL / "generation_report.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    c = pd.read_csv(c_path)
    s1 = pd.read_csv(s1_path)

    errors: list[str] = []
    if list(c.columns) != schema["submitted_c_columns"]:
        errors.append("columns do not match schema.submitted_c_columns")
    if len(c) != len(s1) or len(c) != 1049:
        errors.append("row count is not 1049/S1-aligned")
    if c.isna().any().any():
        errors.append("missing values found")
    numeric = c.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("non-finite values found")
    if not c["age"].between(*schema["age_range"]).all() or not np.equal(c["age"], np.floor(c["age"])).all():
        errors.append("age violates the public range or integer requirement")
    for name, bounds in schema["ranges"].items():
        if not c[name].between(*bounds).all():
            errors.append(f"{name} violates the public range")
    for name in ("sex", "smoking", "onset", "death"):
        if not c[name].isin([0, 1]).all():
            errors.append(f"{name} is not binary")
    events = set(zip(c["onset"].astype(int), c["death"].astype(int)))
    allowed = {tuple(pair) for pair in schema["allowed_onset_death"]}
    if not events <= allowed:
        errors.append("invalid onset/death combination")
    if digest(c_path) != report["output_sha256"]:
        errors.append("C.csv hash differs from generation_report.json")
    expected_inputs = {
        "s1_sha256": s1_path,
        "background_sha256": data / "A_bg.csv",
        "schema_sha256": schema_path,
        "config_sha256": TEAM / "S" / "S1" / "config" / "aim_main_team10_S1.json",
    }
    for key, path in expected_inputs.items():
        if report["inputs"].get(key) != digest(path):
            errors.append(f"input lineage mismatch: {key}")

    detector = schema["rare_detector"]
    values = c[detector["columns"]].to_numpy(dtype=float)
    diff = values - np.asarray(detector["mu"], dtype=float)
    inv_sigma = np.linalg.pinv(np.asarray(detector["sigma"], dtype=float))
    rare_count = int(
        (np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv_sigma, diff), 0.0, None))
         >= float(detector["band"]["lo"])).sum()
    )
    if rare_count != int(report["rare_detected"]):
        errors.append("rare count differs from generation report")

    validation = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not errors,
        "errors": errors,
        "candidate": "M1",
        "rows": len(c),
        "columns": len(c.columns),
        "missing_values": int(c.isna().sum().sum()),
        "duplicate_rows": int(c.duplicated().sum()),
        "rare_detected": rare_count,
        "rare_gate": int(detector["gate"]),
        "c_sha256": digest(c_path),
        "generation_report_sha256": digest(report_path),
        "token_or_submission_zip_created": False,
    }
    (MODEL / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if errors:
        raise SystemExit("; ".join(errors))
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
