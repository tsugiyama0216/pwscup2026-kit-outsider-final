"""Generate two independent R v1 passes, validate, and package locally."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hmac
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from common import ATTACK, HERE, csv_bytes, load_actual, read_json, save_json, sha, verify_dependencies


def validate_package(path, expected_csv, runtime):
    w, np, pd, validator, cfg, dist, *_ = runtime
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["F_mia.csv", "F_aia.csv", "token.txt"]:
            raise ValueError("Submission ZIP must contain exactly the three flat files")
        for name, expected in expected_csv.items():
            if archive.read(name) != expected:
                raise ValueError("Packaged CSV bytes differ: " + name)
        if not hmac.compare_digest(archive.read("token.txt"), (w.ORIGINALS / "token.txt").read_bytes()):
            raise ValueError("Packaged credential differs from the original")
    result = validator.validate_submission_zip(path, cfg, w.TARGETS, "attack")
    if not result.ok:
        raise ValueError("Official ZIP validator rejected the package: " + "; ".join(result.errors))
    package = validator.check_package(path, cfg)
    package.token = None
    return w.validate_frames(np, pd, validator, package.frames["F_mia.csv"],
        package.frames["F_aia.csv"], cfg, dist)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "submission")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or (output.exists() and any(output.iterdir())):
        raise ValueError("Use an empty output directory under Attack_R/v1")
    frozen = verify_dependencies()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = HERE / ".work" / stamp
    work.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["MPLCONFIGDIR"] = str(work / "matplotlib")
    reports = {}
    for repetition in (1, 2):
        for component in ("mia", "aia"):
            destination = work / f"pass{repetition}-{component}"
            subprocess.run([sys.executable, "-B", str(HERE / "infer.py"), component,
                "--output", str(destination)], env=environment, check=True)
            reports[f"pass{repetition}_{component}"] = read_json(destination / "report.json")
    content = {}
    for component, name in (("mia", "F_mia.csv"), ("aia", "F_aia.csv")):
        data = (work / f"pass1-{component}" / name).read_bytes()
        if data != (work / f"pass2-{component}" / name).read_bytes():
            raise AssertionError("Independent runs differ: " + name)
        content[name] = data
    runtime = load_actual()
    w, np, pd, validator, cfg, dist, *_ = runtime
    frames = {name: pd.read_csv(io.BytesIO(data), float_precision="round_trip")
              for name, data in content.items()}
    validation = w.validate_frames(np, pd, validator, frames["F_mia.csv"], frames["F_aia.csv"], cfg, dist)
    for name, frame in frames.items():
        if csv_bytes(frame) != content[name]:
            raise AssertionError("CSV roundtrip changed bytes: " + name)
    verify_dependencies()
    output.mkdir(parents=True, exist_ok=True)
    # No plaintext token copy is written; the original bytes go only into this local ZIP.
    for name, data in content.items():
        (output / name).write_bytes(data)
    zipped = output / "team10_R_v1_attack.zip"
    with zipfile.ZipFile(zipped, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            archive.writestr(name, data)
        archive.writestr("token.txt", (w.ORIGINALS / "token.txt").read_bytes())
    package_validation = validate_package(zipped, content, runtime)
    verification = {"model": "Attack_R/v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "mia": "Frozen M v2, selected strength 1.0 and exact known-negative proof",
        "aia": "Frozen T v88 combined AIA; no additional tuning",
        "components": reports, "dependencies": frozen, "validation": validation,
        "official_zip_and_actual_team10_validation": package_validation,
        "two_independent_process_runs_byte_identical": True, "csv_roundtrip_byte_identical": True,
        "zip_members": ["F_mia.csv", "F_aia.csv", "token.txt"],
        "original_token_bytes_match": True, "token_registration_checked": False,
        "file_sha256": {name: sha(output / name) for name in [*content, zipped.name]},
        "official_submission_performed": False, "official_attempts_consumed": 0,
        "official_score_measured": False,
        "rules": ["4.1", "4.2", "5.2", "5.5", "6.3", "7.4", "7.5"]}
    save_json(HERE / "verification.json", verification)
    print(f"R v1 completed and validated: {zipped}", flush=True)


if __name__ == "__main__":
    main()
