"""Generate two identical local submissions and validate the exact ZIP payload."""
import argparse
from datetime import datetime, timezone
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import runtime as rt
import pandas as pd


def production_check():
    manifest = rt.read_json(rt.HERE/"production_dependencies.json")
    for name, digest in manifest["files_sha256"].items():
        path = (rt.REPO/name).resolve()
        if not path.is_relative_to(rt.REPO) or rt.sha(path) != digest:
            raise ValueError("Production dependency changed: " + name)
    rt.selected_check()
    rt.upstream_check()
    confirm = rt.read_json(rt.HERE/"confirmation.json")
    if not confirm["completed"] or confirm["stage"] != "confirm":
        raise ValueError("Frozen candidate confirmation is required")
    return {"verified_files": len(manifest["files_sha256"]),
            "manifest_sha256": rt.sha(rt.HERE/"production_dependencies.json")}


def load_validator():
    sys.path.insert(0, str(rt.HERE.parent/"v1"))
    from common import load_actual
    spec = importlib.util.spec_from_file_location("r_v1_package_validator", rt.HERE.parent/"v1/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_package, load_actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=rt.HERE/"submission")
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(rt.HERE) or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a new empty folder under v2f")
    frozen = production_check()
    selection = rt.selected_check()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = rt.HERE/".work"/stamp
    work.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["MPLCONFIGDIR"] = str(work/"matplotlib")
    reports = []
    for i in (1, 2):
        target = work/f"pass{i}"
        subprocess.run([sys.executable, "-B", str(rt.HERE/"infer.py"), "--output", str(target)],
                       env=environment, check=True)
        reports.append(rt.read_json(target/"report.json"))
    content = {}
    for name in ("F_mia.csv", "F_aia.csv"):
        content[name] = (work/"pass1"/name).read_bytes()
        if content[name] != (work/"pass2"/name).read_bytes():
            raise AssertionError("Independent output differs: " + name)
    validate_package, load_actual = load_validator()
    actual = load_actual()
    w, np, _, validator, cfg, dist, *_ = actual
    frames = {name: pd.read_csv(io.BytesIO(data), float_precision="round_trip") for name, data in content.items()}
    valid = w.validate_frames(np, pd, validator, frames["F_mia.csv"], frames["F_aia.csv"], cfg, dist)
    for name, frame in frames.items():
        if rt.csv_bytes(frame) != content[name]:
            raise AssertionError("CSV roundtrip changed output: " + name)
    production_check()
    output.mkdir(parents=True, exist_ok=True)
    for name, data in content.items():
        (output/name).write_bytes(data)
    archive_path = output/"team10_R_v2f_attack.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            archive.writestr(name, data)
        # Original credential bytes are written only into this ignored local ZIP.
        archive.writestr("token.txt", (w.ORIGINALS/"token.txt").read_bytes())
    package = validate_package(archive_path, content, actual)
    previous = rt.read_json(rt.HERE.parent/"v1/verification.json")
    for name, digest in previous["file_sha256"].items():
        assert rt.sha(rt.HERE.parent/"v1/submission"/name) == digest
    same_aia = content["F_aia.csv"] == (rt.HERE.parent/"v1/submission/F_aia.csv").read_bytes()
    if selection["aia"] == "T" and not same_aia:
        raise AssertionError("Unchanged T AIA should reproduce R v1")
    rt.save_json(rt.HERE/"verification.json", {"model": "Attack_R/v2f", "completed": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
        "production_dependencies": frozen, "validation": valid, "official_zip_validation": package,
        "two_independent_process_runs_byte_identical": True, "csv_roundtrip_byte_identical": True,
        "original_token_bytes_match": True, "token_registration_checked": False,
        "R_v1_artifacts_unchanged": True, "AIA_byte_identical_to_R_v1": same_aia,
        "runs": reports, "file_sha256": {name: rt.sha(output/name) for name in [*content, archive_path.name]},
        "official_submission_performed": False, "official_attempts_consumed": 0,
        "official_v2f_score_measured": False})
    print(f"Validated v2f ZIP: {archive_path}", flush=True)


if __name__ == "__main__":
    main()
