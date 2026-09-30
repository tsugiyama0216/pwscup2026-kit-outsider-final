"""Local main-round team10 checks and official reference-attack smoke run.

Run with .venv/Scripts/python.exe. Input paths are anchored to this file, so the
current directory cannot accidentally select preliminary or practice data.
This helper never creates a submission ZIP or contacts CodaBench.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KIT = ROOT / "official-kit" / "starter"
PACKAGE = KIT / "src" / "pwscup2026"
TARGETS = ROOT / "targets"
ORIGINALS = ROOT.parent / "anonymization" / "team10" / "participant_data"
OUTPUT = ROOT / "local-output"
TEAM_ID = 10
EXPECTED_TARGETS = set(range(18)) | {22, 25, 26, 27, 28}
ATTACK_TARGETS = sorted(EXPECTED_TARGETS - {TEAM_ID})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_runtime():
    """Fail closed if another editable installation supplies the official code."""
    import numpy as np
    import pandas as pd

    names = ["pwscup2026", "pwscup2026.kit.validate", "pwscup2026.attack.mia",
             "pwscup2026.attack.aia", "pwscup2026.common.config"]
    modules = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None, f"Missing official module: {name}")
        origin = Path(spec.origin).resolve()
        require(origin.is_relative_to(PACKAGE.resolve()), f"Wrong official package origin: {name}: {origin}")
        modules[name] = importlib.import_module(name)
    # Load official reference parameters as supplied, without importing its practice-data helpers.
    spec = importlib.util.spec_from_file_location("main_round_reference_common", KIT / "reference" / "_common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    cfg = common.load_cfg()
    versions = {}
    for line in (KIT / "requirements-kit.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        require(actual == expected, f"Dependency mismatch: {name}: {actual}, expected {expected}")
        versions[name] = actual
    return np, pd, modules, cfg, versions


def verify_hashes():
    manifest = ROOT / "source_manifest.json"
    required = {p.resolve() for p in TARGETS.glob("*.csv")}
    required.add((TARGETS / "mia_columns.json").resolve())
    entries = []
    warnings = []
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        entries = [(ROOT / item["path"], item["sha256"]) for item in payload["files"]]
        required.update((ORIGINALS / name).resolve() for name in ("B.csv", "A_bg.csv", "schema.json"))
    else:
        candidates = [TARGETS / "SHA256SUMS.txt", TARGETS / "SHA256SUMS"]
        manifest = next((p for p in candidates if p.is_file()), None)
        require(manifest is not None, "Neither source_manifest.json nor target SHA256SUMS exists")
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                entries.append((TARGETS / name.lstrip(" *"), digest))
        warnings.append("Source manifest absent: original team10 inputs are not hash-verified")
    verified = set()
    for path, expected in entries:
        path = path.resolve()
        require(path.is_relative_to(ROOT.parent.resolve()), "Manifest path escapes main_round")
        require(path.name.lower() != "token.txt", "Token must not appear in a hash manifest")
        require(path.is_file(), f"Missing manifest input: {path.name}")
        require(re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None, "Invalid SHA256 entry")
        require(sha256(path) == expected.lower(), f"SHA256 mismatch: {path.relative_to(ROOT.parent)}")
        verified.add(path)
    require(required <= verified, "Hash manifest does not cover all required input files")
    return {"manifest": str(manifest), "verified_files": len(verified), "warnings": warnings}


def numeric_values(np, pd, frame, label):
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"{label}: all values must be numeric and finite")
    return values


def integer_ids(np, pd, series, label):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"{label}: IDs must be finite")
    require(np.equal(values, np.floor(values)).all(), f"{label}: IDs must be exact integers")
    return values.astype("int64")


def check_input_values(np, pd, frame, label):
    # Prefecture is a categorical string in the actual distribution.
    prefecture = frame["prefecture"]
    require(prefecture.notna().all() and prefecture.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
            f"{label}: prefecture must contain nonempty strings")
    numeric_values(np, pd, frame.drop(columns="prefecture"), label)


def load_inputs(np, pd, validator):
    schema = json.loads((ORIGINALS / "schema.json").read_text(encoding="utf-8-sig"))
    columns = json.loads((TARGETS / "mia_columns.json").read_text(encoding="utf-8-sig"))["target_columns"]
    require(len(columns) == 23 and set(columns) == {str(k) for k in EXPECTED_TARGETS}, "Unexpected main-round target set")
    for pattern in ("C_*.csv", "aia_challenge_*.csv"):
        found = {int(path.stem.rsplit("_", 1)[1]) for path in TARGETS.glob(pattern)}
        require(found == EXPECTED_TARGETS, f"Unexpected files matching {pattern}")
    b = pd.read_csv(ORIGINALS / "B.csv")
    background = pd.read_csv(ORIGINALS / "A_bg.csv")
    for name, frame, expected_rows in (("B.csv", b, 1049), ("A_bg.csv", background, 5143)):
        require(list(frame.columns) == schema["distributed_columns"], f"{name}: schema columns differ")
        require(len(frame) == expected_rows, f"{name}: expected {expected_rows} rows")
        check_input_values(np, pd, frame, name)
        ids = integer_ids(np, pd, frame["record_id"], name)
        require(len(set(ids)) == len(ids), f"{name}: duplicate record_id")
    targets, challenges = {}, {}
    for k in sorted(EXPECTED_TARGETS):
        c = pd.read_csv(TARGETS / f"C_{k}.csv")
        challenge = pd.read_csv(TARGETS / f"aia_challenge_{k}.csv")
        require(list(c.columns) == schema["submitted_c_columns"], f"C_{k}: schema columns differ")
        require(len(c) == 1049, f"C_{k}: expected 1049 rows")
        require(list(challenge.columns) == ["challenge_row_id"] + schema["aia_qi_columns"],
                f"aia_challenge_{k}: schema columns differ")
        require(len(challenge) > 0, f"aia_challenge_{k}: empty")
        check_input_values(np, pd, c, f"C_{k}")
        check_input_values(np, pd, challenge, f"aia_challenge_{k}")
        ids = integer_ids(np, pd, challenge["challenge_row_id"], f"aia_challenge_{k}")
        require(len(set(ids)) == len(ids), f"aia_challenge_{k}: duplicate ID")
        targets[k], challenges[k] = c, challenge
    # Official loader does not know the actual B.csv filename. Supply its keys explicitly.
    dist = validator.load_dist(TARGETS, team_id=TEAM_ID)
    dist["attacker_record_ids"] = set(b["record_id"].astype(int))
    require(set(dist["challenge"]) == EXPECTED_TARGETS, "Official validator did not load all challenges")
    return b, background, targets, challenges, dist


def inspect_environment():
    hashes = verify_hashes()
    np, pd, modules, cfg, versions = load_runtime()
    validator = modules["pwscup2026.kit.validate"]
    inputs = load_inputs(np, pd, validator)
    token_path = ORIGINALS / "token.txt"
    require(token_path.is_file(), "Local token.txt is missing")
    lines = [line.strip() for line in token_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == 1 and re.fullmatch(cfg["submission"]["token_pattern"], lines[0]) is not None,
            "Local token.txt has invalid format")
    del lines
    b, background, targets, challenges, _ = inputs
    report = {
        "ok": True, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "team_id": TEAM_ID, "python": sys.executable, "python_version": sys.version.split()[0],
        "official_package": str(PACKAGE), "dependency_versions": versions, "input_hashes": hashes,
        "candidate_rows": len(b), "background_rows": len(background),
        "distributed_targets": sorted(targets), "attack_targets": ATTACK_TARGETS,
        "target_rows": {str(k): len(c) for k, c in targets.items()},
        "challenge_rows": {str(k): len(c) for k, c in challenges.items()},
        "token": {"exists": True, "format_valid": True, "registration_checked": False},
    }
    write_report(OUTPUT / "environment_check.json", report)
    return np, pd, modules, cfg, inputs, report


def validate_frames(np, pd, validator, mia, aia, cfg, dist):
    """Official logical validation plus strict checks against the actual distribution."""
    require(mia.columns.is_unique and aia.columns.is_unique, "Duplicate output columns")
    require(set(mia.columns) == {"record_id"} | {str(k) for k in ATTACK_TARGETS},
            "F_mia must contain record_id and exactly the 22 non-self targets")
    require(list(aia.columns) == validator.AIA_COLUMNS, "F_aia has incorrect columns")
    ids = integer_ids(np, pd, mia["record_id"], "F_mia record_id")
    require(len(ids) == 1049 and len(set(ids)) == len(ids) and set(ids) == dist["attacker_record_ids"],
            "F_mia record IDs differ from actual team10 B.csv")
    scores = numeric_values(np, pd, mia.drop(columns="record_id"), "F_mia")
    require(((scores >= 0) & (scores <= 1)).all(), "F_mia confidence must be in [0,1]")
    target_ids = integer_ids(np, pd, aia["target_k"], "F_aia target_k")
    challenge_ids = integer_ids(np, pd, aia["challenge_row_id"], "F_aia challenge_row_id")
    times = numeric_values(np, pd, aia[["time_hat"]], "F_aia time_hat")
    require((times >= 0).all(), "F_aia time_hat must be nonnegative")
    pairs = list(zip(target_ids, challenge_ids))
    expected = {(k, row_id) for k in ATTACK_TARGETS for row_id in dist["challenge"][k]}
    require(len(set(pairs)) == len(pairs), "F_aia has duplicate target/challenge pairs")
    require(set(pairs) == expected, "F_aia target/challenge IDs differ from actual non-self challenges")
    errors = validator.validate_mia_submission(mia, cfg, dist).errors
    errors += validator.validate_aia_submission(aia, cfg, dist).errors
    require(not errors, "Official logical validation failed: " + "; ".join(errors))
    return {"ok": True, "mia_rows": len(mia), "mia_target_columns": len(ATTACK_TARGETS), "aia_rows": len(aia)}


def baseline(runtime):
    np, pd, modules, cfg, inputs, environment = runtime
    b, background, targets, challenges, dist = inputs
    mia = pd.DataFrame({"record_id": b["record_id"].to_numpy()})
    aia_frames = []
    for i, k in enumerate(ATTACK_TARGETS, 1):
        mia[str(k)] = modules["pwscup2026.attack.mia"].overlap_mia(targets[k], background, b, cfg)
        _, time_hat = modules["pwscup2026.attack.aia"].aia_predict(targets[k], challenges[k], background, cfg)
        aia_frames.append(pd.DataFrame({"target_k": k,
                                       "challenge_row_id": challenges[k]["challenge_row_id"].to_numpy(),
                                       "time_hat": time_hat}))
        print(f"Official reference attack: {i}/{len(ATTACK_TARGETS)} targets finished", flush=True)
    aia = pd.concat(aia_frames, ignore_index=True)
    result = validate_frames(np, pd, modules["pwscup2026.kit.validate"], mia, aia, cfg, dist)
    destination = OUTPUT / "reference-baseline"
    destination.mkdir(parents=True, exist_ok=True)
    mia.to_csv(destination / "F_mia.csv", index=False)
    aia.to_csv(destination / "F_aia.csv", index=False)
    # Check CSV round-trip, including that record IDs remain exact.
    result = validate_frames(np, pd, modules["pwscup2026.kit.validate"],
                             pd.read_csv(destination / "F_mia.csv"), pd.read_csv(destination / "F_aia.csv"), cfg, dist)
    report = {**result, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Official reference implementation smoke test; not a competitive candidate or score estimate.",
              "team_id": TEAM_ID, "attack_targets": ATTACK_TARGETS,
              "official_package": environment["official_package"],
              "methods": ["overlap_mia", "aia_predict"], "submission_created": False,
              "output_sha256": {name: sha256(destination / name) for name in ("F_mia.csv", "F_aia.csv")}}
    write_report(destination / "report.json", report)
    print(f"Reference smoke run passed: {destination}")


def validate_zip(runtime, path):
    np, pd, modules, cfg, inputs, _ = runtime
    validator = modules["pwscup2026.kit.validate"]
    require(path.is_file(), "Submission ZIP does not exist")
    require(zipfile.is_zipfile(path), "Input is not a ZIP")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "ZIP contains duplicate member names")
        require(set(names) == validator.ATTACK_FILES, "ZIP must contain exactly F_mia.csv, F_aia.csv, token.txt")
    package = validator.check_package(path, cfg)
    require(package.ok and package.kind == "attack", "Official package validation failed: " + "; ".join(package.errors))
    # Compare only in memory; no token is printed, hashed, or copied to another file.
    local_token = (ORIGINALS / "token.txt").read_text(encoding="utf-8").strip()
    token_matches = hmac.compare_digest((package.token or "").encode("utf-8"), local_token.encode("utf-8"))
    del local_token
    package.token = None
    require(token_matches, "ZIP token does not match the local team10 token")
    result = validate_frames(np, pd, validator, package.frames["F_mia.csv"], package.frames["F_aia.csv"], cfg, inputs[4])
    report = {**result, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "zip": str(path), "team_id": TEAM_ID, "token_matches_local_team10": True,
              "token_registration_checked": False,
              "checks": ["official package", "official logical with team_id=10", "strict actual IDs and finite values"]}
    write_report(OUTPUT / "submission_validation.json", report)
    print("ZIP validation passed for actual team10 inputs; no submission was sent.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Check actual inputs, official imports, dependency pins, and token format")
    subparsers.add_parser("baseline", help="Run the official reference attacks as a local smoke test")
    subparsers.add_parser("validate", help="Read-only strict validation of an existing attack ZIP").add_argument("zip", type=Path)
    args = parser.parse_args()
    try:
        runtime = inspect_environment()
        if args.command == "check":
            print(f"Environment check passed: team10, 1049 candidates, 23 distributed / 22 attack targets.\n{OUTPUT / 'environment_check.json'}")
        elif args.command == "baseline":
            baseline(runtime)
        else:
            validate_zip(runtime, args.zip.resolve())
    except Exception as exc:
        # Our errors contain labels/counts only; never include input rows or token data.
        report = {"ok": False, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "command": args.command, "error": f"{type(exc).__name__}: {exc}"}
        name = "environment_check.json" if args.command == "check" else f"{args.command}_error.json"
        write_report(OUTPUT / name, report)
        print(report["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
