"""Reproduce twice and package the frozen R v2, locally only."""
import argparse
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import runtime as rt
import pandas as pd


def load_validator():
    # Reuse the pinned validator without modifying its original source.
    path = str(rt.HERE.parent/"v1")
    if path not in sys.path:
        sys.path.append(path)
    original = rt.load_file("_rv2_original_run", rt.HERE.parent/"v1/run.py")
    common = rt.load_file("_rv2_package_common", rt.HERE.parent/"v1/common.py")
    return original.validate_package, common.load_actual


def production_check():
    rt.selected_check()
    rt.upstream_check()
    manifest = rt.read_json(rt.HERE/"production_dependencies.json")
    for name, expected in manifest["files_sha256"].items():
        p = (rt.REPO/name).resolve()
        if not p.is_relative_to(rt.REPO) or rt.sha(p) != expected:
            raise ValueError("Production dependency differs: " + name)
    confirmation = rt.read_json(rt.HERE/"confirmation.json")
    if confirmation.get("completed") is not True or confirmation["stage"] != "confirmation":
        raise ValueError("Frozen confirmation is required before packaging")
    return {"verified_files": len(manifest["files_sha256"]),
            "manifest_sha256": rt.sha(rt.HERE/"production_dependencies.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=rt.HERE/"submission")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(rt.HERE) or output.exists():
        raise ValueError("Use a new output directory under Attack_R/v2")
    receipt = rt.HERE/"verification.json" if output == rt.HERE/"submission" else output/"verification.json"
    if receipt.exists():
        raise ValueError("Preserve the existing verification receipt")
    check = production_check()
    work = rt.OUTPUT/"production"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    env = os.environ.copy()
    env["PYTHONUTF8"], env["MPLCONFIGDIR"] = "1", str(rt.OUTPUT/"matplotlib")
    reports = []
    for run in (1, 2):
        destination = work/f"pass{run}"
        subprocess.run([sys.executable, "-B", str(rt.HERE/"infer.py"), "--output", str(destination)], env=env, check=True)
        reports.append(rt.read_json(destination/"report.json"))
    content = {}
    for name in ("F_mia.csv", "F_aia.csv"):
        content[name] = (work/"pass1"/name).read_bytes()
        if content[name] != (work/"pass2"/name).read_bytes():
            raise ValueError("Independent runs differ: " + name)
    validate_package, load_actual = load_validator()
    actual = load_actual()
    w, np, _, validator, cfg, dist, *_ = actual
    frames = {name: pd.read_csv(io.BytesIO(value), float_precision="round_trip") for name, value in content.items()}
    valid = w.validate_frames(np, pd, validator, frames["F_mia.csv"], frames["F_aia.csv"], cfg, dist)
    for name, frame in frames.items():
        if rt.csv_bytes(frame) != content[name]:
            raise ValueError("CSV roundtrip differs: " + name)
    production_check()
    output.mkdir(parents=True)
    for name, value in content.items():
        (output/name).write_bytes(value)
    zipped = output/"team10_R_v2_attack.zip"
    with zipfile.ZipFile(zipped, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in content.items():
            archive.writestr(name, value)
        archive.writestr("token.txt", (w.ORIGINALS/"token.txt").read_bytes())
    package = validate_package(zipped, content, actual)
    preservation = rt.upstream_check()
    changes = {}
    for version in ("v1", "v2f"):
        changes[version] = {}
        for name, frame in frames.items():
            old = pd.read_csv(rt.HERE.parent/version/"submission"/name, float_precision="round_trip")
            cols = [x for x in frame if x not in ("record_id", "target_k", "challenge_row_id")]
            changes[version][name] = int((frame[cols].to_numpy() != old[cols].to_numpy()).sum())
    rt.save_json(receipt, {"model": "Attack_R/v2", "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation": valid,
        "official_zip_and_actual_ID_validation": package,
        "production_dependencies": check, "preservation": preservation,
        "two_independent_process_runs_byte_identical": True, "csv_roundtrip_byte_identical": True,
        "original_token_bytes_match": True, "token_registration_checked": False,
        "zip_members": ["F_mia.csv", "F_aia.csv", "token.txt"], "runs": reports,
        "changed_prediction_cells": changes,
        "file_sha256": {name: rt.sha(output/name) for name in [*content, zipped.name]},
        "official_submission_performed": False, "official_attempts_consumed": 0,
        "official_v2_score_measured": False, "github_push_performed": False})
    print("Validated R v2 local submission: " + str(zipped), flush=True)


if __name__ == "__main__":
    main()
