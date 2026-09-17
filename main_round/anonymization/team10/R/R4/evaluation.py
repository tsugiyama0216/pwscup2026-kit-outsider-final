"""Pinned official evaluation and exact-byte packaging for the local R4 release."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TEAM = HERE.parents[1]
DATA = TEAM / "participant_data"
KIT = ROOT / "main_round/local-output/official-kit"
PIN = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
RULEBOOK_SHA256 = "9290c6e98b85214ca691bb90ff961196642c72d85348810f8df36741489885f2"
COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
           "smoking", "FPG", "time", "onset", "death"]
ZIP_NAME = "team10_R4_defense.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment() -> tuple[dict, dict]:
    pin = subprocess.check_output(["git", "-C", str(KIT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(KIT), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if pin != PIN or dirty:
        raise RuntimeError("Official kit must be unchanged at the pinned main-round commit")
    manifest = json.loads((TEAM / "data_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for entry in manifest["files"]:
        path = TEAM / entry["path"]
        hashes[entry["path"]] = digest(path)
        if hashes[entry["path"]] != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Original input changed: {path.name}")
    sys.path.insert(0, str(KIT / "starter/src"))
    import pwscup2026
    if not Path(pwscup2026.__file__).resolve().is_relative_to(KIT):
        raise RuntimeError("Wrong round's official package was imported")
    config_path = KIT / "starter/src/pwscup2026/kit/kit_config.yaml"
    rules_path = ROOT / "main_round/rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf"
    if digest(rules_path) != RULEBOOK_SHA256:
        raise RuntimeError("Rulebook differs from the reviewed main-round version")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")), {
        "official_release": "main-process-20260912", "official_kit_commit": pin,
        "scoring_config_sha256": digest(config_path), "rulebook_sha256": digest(rules_path),
        "inputs": hashes,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "lifelines", "scikit-learn", "statsmodels", "PyYAML"]},
        "python": sys.version.split()[0],
    }


def read_training_data() -> pd.DataFrame:
    from pwscup2026.common.submission_schema import read_scoring_csv
    return read_scoring_csv(DATA / "B.csv")


def evaluate_frame(c: pd.DataFrame, b: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    """Evaluation-only references never enter the generator."""
    from pwscup2026.common.submission_schema import read_scoring_csv, validate_c_submission
    from pwscup2026.kit.selfscore import self_score
    from pwscup2026.kit.validate import rare_count_diagnostic
    from pwscup2026.scoring.utility import UtilityReference
    logical = validate_c_submission(c, cfg, n_expected=len(b))
    if not logical.ok or list(c.columns) != COLUMNS:
        raise ValueError(f"Invalid synthetic table: {logical.errors}")
    censored = (c.onset == 0) & (c.death == 0)
    if not (c.loc[censored, "time"] == cfg["onset"]["horizon_years"]).all():
        raise ValueError("Administrative censoring time differs from the study horizon")
    source_matches = int(c.merge(b[COLUMNS].drop_duplicates(), on=COLUMNS).shape[0])
    duplicates = int(c.duplicated().sum())
    if source_matches or duplicates:
        raise ValueError("Exact source rows or duplicate synthetic rows were produced")
    b_self = read_scoring_csv(DATA / "B_self.csv")
    pd.testing.assert_frame_equal(b, b_self.drop(columns="is_rare"))
    reference_data = json.loads((DATA / "utility_ref.json").read_text(encoding="utf-8"))
    reference = UtilityReference(
        mu=np.asarray(reference_data["mu"]), sigma=np.asarray(reference_data["sigma"]),
        band=reference_data["band"], b_is_rare=b_self.is_rare.to_numpy(dtype=bool),
        km_signal_D=reference_data.get("km_signal_D"), km_floor=reference_data.get("km_floor"),
        a_bg=read_scoring_csv(DATA / "A_bg.csv"))
    score = self_score(c, b, reference, cfg)
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    rare = rare_count_diagnostic(c, schema)
    validation = {
        "logical_ok": True, "rows": len(c), "columns": list(c.columns),
        "no_missing": not bool(c.isna().any().any()),
        "source_exact_matches": source_matches, "synthetic_duplicate_rows": duplicates,
        "censored_time_is_horizon": True, "rare_diagnostic": rare,
        "event_counts": {name: int(c[name].sum()) for name in ("onset", "death")},
        "privacy_note": "No formal differential privacy or official attack-resistance guarantee.",
    }
    return score, validation


def write_package(path: Path, csv_bytes: bytes, token_bytes: bytes) -> None:
    """The only credential copy is inside the ignored local submission ZIP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (("C.csv", csv_bytes), ("token.txt", token_bytes)):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)


def evaluate_release(outdir: Path, b: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    from pwscup2026.kit.selfscore import load_submitted_c
    from pwscup2026.kit.validate import check_package, validate_submission_zip
    zip_path = outdir / "submission" / ZIP_NAME
    package = check_package(zip_path, cfg)
    result = validate_submission_zip(zip_path, cfg, dist_dir=DATA, kind_override="defense")
    if not package.ok or not result.ok or package.kind != "defense":
        raise ValueError("Official submission package validation failed")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["C.csv", "token.txt"]:
            raise ValueError("Submission ZIP must contain exactly two flat files")
        if archive.read("C.csv") != (outdir / "C.csv").read_bytes():
            raise ValueError("ZIP and standalone CSV differ")
        if archive.read("token.txt") != (DATA / "token.txt").read_bytes():
            raise ValueError("Submission credential bytes differ from the original")
    score, validation = evaluate_frame(load_submitted_c(zip_path), b, cfg)
    validation.update({"package_ok": True, "csv_matches_zip_bytes": True,
                       "token_matches_original_bytes": True,
                       "C_sha256": digest(outdir / "C.csv"), "ZIP_sha256": digest(zip_path)})
    if validation["rare_diagnostic"]["below_gate"]:
        raise ValueError("Rare count is below the official utility gate")
    return score, validation
