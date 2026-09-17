"""Verify the existing S1 release and build its flat defense ZIP without rerunning AIM.

The token is read from team10/participant_data/token.txt directly into ZIP memory.
Its value and digest are never printed or recorded in validation.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import zipfile


COLUMNS = [
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking",
    "FPG", "time", "onset", "death",
]
EXPECTED_ROWS = 1049
ORIGINALS = {"A_bg.csv", "B.csv", "B_self.csv", "README.txt", "schema.json", "utility_ref.json"}
KIT_COMMIT = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
KIT_ARCHIVE_SHA256 = "7d842aec5416f0ebd7d72bcc71d623e4ef8618d3b4308f53c3b8f95d9681d8ca"


class ValidationError(ValueError):
    """A release, input, or package invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def digest(path: Path) -> str:
    require(path.name != "token.txt", "Token digests are prohibited")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(result, dict), f"Expected JSON object: {path.name}")
    return result


def check_originals(team: Path) -> dict:
    manifest_path = team / "data_manifest.json"
    manifest = load_json(manifest_path)
    require(manifest.get("round") == "main" and manifest.get("cohort_id") == 10,
            "Manifest must describe main-round team10")
    expected_paths = {"participant_data/" + name for name in ORIGINALS}
    files = manifest.get("files", [])
    require(len(files) == len(expected_paths)
            and {item["path"] for item in files} == expected_paths,
            "Manifest must contain exactly the six authorized non-credential originals")
    hashes = {}
    for item in files:
        path = team / item["path"]
        require(path.stat().st_size == item["bytes"], f"Original byte size mismatch: {path.name}")
        actual = digest(path)
        require(actual == item["sha256"], f"Original SHA256 mismatch: {path.name}")
        hashes[path.name] = actual
    return {"ok": True, "manifest_sha256": digest(manifest_path), "sha256": hashes}


def check_lineage(root: Path, originals: dict) -> tuple[dict, dict]:
    config_path = root / "config" / "aim_main_team10_S1.json"
    report_path = root / "privacy_report.json"
    ledger_path = root / "release_ledger.json"
    config = load_json(config_path)
    report = load_json(report_path)
    ledger = load_json(ledger_path)
    require(ledger.get("status") == "complete", "DP release ledger is not complete")
    require(report.get("dp_release_count") == 1, "Expected exactly one recorded DP release")
    require(config.get("output_rows") == EXPECTED_ROWS
            and config.get("expected_input_rows") == EXPECTED_ROWS
            and report.get("output_rows_public") == EXPECTED_ROWS,
            "Config/report public row count mismatch")
    hashes = {
        "config_sha256": digest(config_path),
        "generator_sha256": digest(root / "scripts" / "generate_pwscup.py"),
        "public_background_sha256": originals["sha256"]["A_bg.csv"],
        "public_schema_sha256": originals["sha256"]["schema.json"],
        "output_sha256": digest(root / "C.csv"),
    }
    for field, actual in hashes.items():
        require(report.get(field) == actual, f"Privacy report lineage mismatch: {field}")
        require(ledger.get(field) == actual, f"Release ledger lineage mismatch: {field}")
    require(ledger.get("input_sha256") == originals["sha256"]["B.csv"],
            "Release ledger does not match team10 B.csv")
    require(ledger.get("privacy_report_sha256") == digest(report_path),
            "Privacy report bytes do not match release ledger")
    require(ledger.get("opendp_uv_lock_sha256") == digest(root / "uv.lock"),
            "uv.lock does not match release ledger")
    require(report.get("config_version") == config.get("config_version"),
            "Config version does not match privacy report")
    rho = config["privacy"]["rho"]
    require(rho == 0.1 and ledger.get("rho_zcdp") == rho
            and report.get("privacy", {}).get("rho_zcdp") == rho,
            "Privacy budget mismatch")
    provenance = config["public_config_provenance"]
    require(provenance.get("private_inputs_used") is False,
            "Public config records private input usage")
    for field in ("public_background_sha256", "public_schema_sha256"):
        require(provenance.get(field) == hashes[field], f"Public config mismatch: {field}")
    require(provenance.get("base_config_sha256") == digest(root / "config" / "prelim_submitted_v5_base.json"),
            "Preliminary base config hash mismatch")
    require(config["postprocess"]["public_outcome_donor"]["risk"]["source_background_sha256"]
            == hashes["public_background_sha256"], "Outcome donor background hash mismatch")
    return config, {
        "ok": True, **hashes, "input_sha256": ledger["input_sha256"],
        "privacy_report_sha256": digest(report_path), "release_ledger_sha256": digest(ledger_path),
        "release_attempt_id": ledger["release_attempt_id"], "dp_release_count": 1, "rho_zcdp": rho,
        "repackaging_runs_dp": False,
    }


def check_csv(raw: bytes, schema: dict, config: dict, expected_rows: int = EXPECTED_ROWS) -> dict:
    require(schema.get("submitted_c_columns") == COLUMNS, "Unexpected submitted schema")
    reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
    require(next(reader, None) == COLUMNS, "C.csv column names/order must exactly match all 13 columns")
    rows = list(reader)
    require(len(rows) == expected_rows, f"C.csv must contain {expected_rows} data rows")
    numeric_columns = [name for name in COLUMNS if name != "prefecture"]
    ranges = {**schema["ranges"], "age": schema["age_range"], "time": [1.0 / 365.25, 10.0]}
    numeric_rows = []
    extrema = {name: [math.inf, -math.inf] for name in numeric_columns}
    allowed_pairs = {tuple(pair) for pair in schema["allowed_onset_death"]}
    outcome_counts = {f"{pair[0]},{pair[1]}": 0 for pair in sorted(allowed_pairs)}
    for row_index, values in enumerate(rows, start=2):
        require(len(values) == len(COLUMNS), f"C.csv row {row_index} has an incorrect field count")
        row = dict(zip(COLUMNS, values))
        require(all(value.strip() for value in values), f"C.csv row {row_index} has an empty value")
        require(row["prefecture"] in config["categorical_keys"]["prefecture"],
                f"C.csv row {row_index} has an invalid prefecture")
        numbers = {}
        for name in numeric_columns:
            try:
                number = float(row[name])
            except ValueError as exc:
                raise ValidationError(f"C.csv row {row_index}, {name}: nonnumeric value") from exc
            require(math.isfinite(number), f"C.csv row {row_index}, {name}: nonfinite value")
            if name in ranges:
                lower, upper = ranges[name]
                require(lower <= number <= upper, f"C.csv row {row_index}, {name}: out of range")
            if name in ("age", "sex", "smoking", "onset", "death"):
                require(number.is_integer(), f"C.csv row {row_index}, {name}: noninteger value")
            if name in ("sex", "smoking", "onset", "death"):
                require(number in (0, 1), f"C.csv row {row_index}, {name}: nonbinary value")
            numbers[name] = number
            extrema[name][0] = min(extrema[name][0], number)
            extrema[name][1] = max(extrema[name][1], number)
        pair = (int(numbers["onset"]), int(numbers["death"]))
        require(pair in allowed_pairs, f"C.csv row {row_index}: disallowed onset/death combination")
        outcome_counts[f"{pair[0]},{pair[1]}"] += 1
        numeric_rows.append(tuple(row[name] if name == "prefecture" else numbers[name] for name in COLUMNS))
    return {
        "ok": True, "rows": len(rows), "columns": COLUMNS, "column_count": len(COLUMNS),
        "missing_values": 0, "nonfinite_values": 0, "range_violations": 0,
        "duplicate_rows": len(numeric_rows) - len(set(numeric_rows)),
        "numeric_min_max": extrema, "outcome_counts": outcome_counts,
        "time_limits": [1.0 / 365.25, 10.0],
    }


def verify_zip(path: Path, csv_bytes: bytes, token_bytes: bytes) -> dict:
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["C.csv", "token.txt"],
                "ZIP must contain exactly C.csv and token.txt at its root, without duplicate members")
        require(archive.testzip() is None, "ZIP CRC integrity check failed")
        require(archive.read("C.csv") == csv_bytes, "ZIP C.csv bytes differ from release")
        require(archive.read("token.txt") == token_bytes, "ZIP token differs from supplied team10 token")
    return {"ok": True, "members": ["C.csv", "token.txt"], "csv_exact_bytes": True,
            "token_exact_bytes_match_team10_source": True, "token_content_or_hash_recorded": False}


def check_official_kit(kit: Path, archive_path: Path, root: Path) -> dict:
    provenance = load_json(root / "official_kit_provenance.json")
    require(provenance.get("commit") == KIT_COMMIT, "Unexpected official kit commit")
    require(provenance.get("release_tag") == "main-process-20260912", "Unexpected official kit release")
    require(digest(kit.parent / "main-process-20260912.zip") == KIT_ARCHIVE_SHA256,
            "Official source archive SHA256 mismatch")
    source_hashes = {}
    prefix = f"pwscup2026-kit-{KIT_COMMIT}/"
    with zipfile.ZipFile(kit.parent / "main-process-20260912.zip") as source_archive:
        for info in source_archive.infolist():
            if not info.filename.startswith(prefix):
                continue
            relative = info.filename[len(prefix):]
            if (relative.startswith("starter/src/pwscup2026/") and relative.endswith((".py", ".yaml"))) \
                    or relative == "participant_data/scoring_config.yaml":
                actual = digest(kit / relative)
                expected = hashlib.sha256(source_archive.read(info)).hexdigest()
                require(actual == expected, f"Official source differs from verified archive: {relative}")
                source_hashes[relative] = actual
    require(bool(source_hashes), "Official source archive contained no validator sources")
    # Pass only selected result fields. PackageResult also contains the token and must never be serialized.
    worker = r'''
import json, pathlib, sys
kit, submission, participant = (pathlib.Path(x).resolve() for x in sys.argv[1:])
sys.path.insert(0, str(kit / "starter" / "src"))
from pwscup2026.common import config, submission_schema
from pwscup2026.kit import validate
if pathlib.Path(validate.__file__).resolve() != kit / "starter/src/pwscup2026/kit/validate.py":
    raise RuntimeError("Unexpected validator import path")
cfg = config.load_config(kit / "participant_data/scoring_config.yaml")
package = validate.check_package(submission, cfg)
if not package.ok:
    print(json.dumps({"ok": False, "package_ok": False, "error_count": len(package.errors)}))
    sys.exit(0)
expected_rows = len(submission_schema.read_scoring_csv(participant / "B.csv"))
logical = submission_schema.validate_c_submission(package.frames["C.csv"], cfg, n_expected=expected_rows)
combined = validate.validate_submission_zip(submission, cfg, participant, kind_override="defense")
diagnostic = validate.rare_count_diagnostic(package.frames["C.csv"], validate.load_dist(participant))
print(json.dumps({"ok": bool(logical.ok and combined.ok), "package_ok": True,
    "logical_ok": bool(logical.ok), "zip_validator_ok": bool(combined.ok),
    "explicit_expected_rows_from_B": expected_rows,
    "error_count": len(logical.errors) + len(combined.errors), "rare_diagnostic": diagnostic}))
'''
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", worker, str(kit), str(archive_path), str(root.parents[1] / "participant_data")],
        cwd=kit, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    require(completed.returncode == 0, "Official validator process failed (captured output withheld)")
    result = json.loads(completed.stdout)
    require(result.get("ok") is True, "Official kit rejected the package or CSV")
    require(result.get("explicit_expected_rows_from_B") == EXPECTED_ROWS, "Official B row count mismatch")
    return {"status": "passed", **result, "release_tag": provenance["release_tag"],
            "commit": KIT_COMMIT, "archive_sha256": KIT_ARCHIVE_SHA256,
            "release_url": provenance["release_url"], "verified_source_sha256": source_hashes,
            "token_registration_checked": False,
            "note": "Local official format validation only; server-side token registration and scoring are not tested."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-root", type=Path, help="Extracted official main-process-20260912 source root")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    team = root.parents[1]
    validation_path = root / "validation.json"
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "ok": False,
              "round": "main", "team": 10, "candidate": "S1"}
    temporary_zip = None
    try:
        originals = check_originals(team)
        config, lineage = check_lineage(root, originals)
        schema = load_json(team / "participant_data" / "schema.json")
        csv_bytes = (root / "C.csv").read_bytes()
        require(hashlib.sha256(csv_bytes).hexdigest() == lineage["output_sha256"],
                "C.csv changed after lineage verification")
        csv_result = check_csv(csv_bytes, schema, config)
        # The six originals are already hash-verified; obtain the required public row count from B.
        with (team / "participant_data" / "B.csv").open(encoding="utf-8", newline="") as handle:
            require(sum(1 for _ in csv.DictReader(handle)) == EXPECTED_ROWS, "Team10 B row count mismatch")
        token_bytes = (team / "participant_data" / "token.txt").read_bytes()
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError("Supplied token is not UTF-8") from exc
        token_lines = [line.strip() for line in token_text.splitlines() if line.strip()]
        require(len(token_lines) == 1 and re.fullmatch(r"[A-Za-z0-9]{16,}", token_lines[0]) is not None,
                "Supplied token does not satisfy the official token format")
        submission = root / "submission" / "team10_S1_defense.zip"
        submission.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(prefix=".packaging-", suffix=".zip", dir=submission.parent)
        os.close(descriptor)
        temporary_zip = Path(filename)
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, contents in (("C.csv", csv_bytes), ("token.txt", token_bytes)):
                member = zipfile.ZipInfo(name, date_time=(2026, 9, 17, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o600 << 16
                archive.writestr(member, contents)
        package_result = verify_zip(temporary_zip, csv_bytes, token_bytes)
        official = check_official_kit(args.kit_root.resolve(), temporary_zip, root) if args.kit_root else {
            "status": "not_run", "reason": "No --kit-root supplied; only local independent checks ran"}
        temporary_zip.replace(submission)
        temporary_zip = None
        report.update(ok=True, originals=originals, lineage=lineage, csv=csv_result,
                      package={**package_result, "path": "submission/team10_S1_defense.zip",
                               "bytes": submission.stat().st_size, "sha256": digest(submission)},
                      official_kit=official)
    except (ValidationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        # Never serialize arbitrary exceptions: decode/parser errors can contain input bytes.
        report["error"] = str(exc) if isinstance(exc, ValidationError) else type(exc).__name__
    finally:
        if temporary_zip is not None:
            temporary_zip.unlink(missing_ok=True)
        validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        print(f"Validation failed: {report['error']}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified and packaged S1: {report['csv']['rows']} rows x {report['csv']['column_count']} columns")
    print(f"Submission: {root / report['package']['path']}")
    print(f"Validation report: {validation_path}")
    print(f"Official kit: {report['official_kit']['status']}")


if __name__ == "__main__":
    main()
