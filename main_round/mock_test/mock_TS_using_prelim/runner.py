"""Export authorized team24 inputs and run a trusted plugin in a fresh process."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import (ATTACKER_ID, DATA, HERE, clean, code_hashes, read_csv, read_json,
                     reference_config, require, sha256, verify_bundle, within, write_json)


def expected_public_hashes():
    entries = read_json(HERE / "provenance/data_manifest.json")["files"]
    hashes = {item["path"][len("data/public/"):]: item["sha256"]
              for item in entries if item["path"].startswith("data/public/")}
    config_bytes = (json.dumps(clean(reference_config()), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    hashes["cfg.json"] = hashlib.sha256(config_bytes).hexdigest()
    return hashes


def export_public(destination):
    destination = Path(destination).resolve()
    require(not destination.exists(), "Export destination already exists")
    destination.mkdir(parents=True)
    for folder in ("team24_processing", "attack_targets"):
        shutil.copytree(DATA / "public" / folder, destination / folder)
    write_json(destination / "cfg.json", reference_config())
    target_ids = sorted(int(p.stem.split("_")[-1]) for p in (destination / "attack_targets").glob("C_*.csv"))
    require(len(target_ids) == 21 and ATTACKER_ID in target_ids, "Expected 21 preliminary releases including self24")
    files = {p.relative_to(destination).as_posix(): sha256(p)
             for p in sorted(destination.rglob("*")) if p.is_file()}
    require(files == expected_public_hashes(), "Export differs from the copied actual preliminary inputs")
    manifest = {"format_version": 1, "round": "prelim", "attacker_id": ATTACKER_ID,
                "target_ids": [k for k in target_ids if k != ATTACKER_ID], "files": files,
                "description": "Actual authorized preliminary team24 inputs. No target truth or other teams' answers."}
    write_json(destination / "bundle_manifest.json", manifest)
    return manifest


def verify_public(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "bundle_manifest.json")
    require(manifest["round"] == "prelim" and manifest["attacker_id"] == ATTACKER_ID, "Wrong public bundle round/team")
    require(len(manifest["target_ids"]) == 20 and ATTACKER_ID not in manifest["target_ids"], "Wrong target list")
    expected_ids = sorted(int(p.stem.split("_")[-1]) for p in (DATA / "public/attack_targets").glob("C_*.csv")
                          if int(p.stem.split("_")[-1]) != ATTACKER_ID)
    require(manifest["target_ids"] == expected_ids, "Target IDs differ from the preliminary distribution")
    require(manifest["files"] == expected_public_hashes(), "Bundle does not match the actual preliminary distribution")
    for relative, digest in manifest["files"].items():
        path = within(directory, relative)
        require(path.is_file() and sha256(path) == digest, "Public input changed: " + relative)
    expected = set(manifest["files"]) | {"bundle_manifest.json"}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual == expected, "Unexpected files in public bundle")
    return manifest


def public_tables(directory):
    """B_self is legitimately known to the attacker; other teams' truths are absent."""
    directory = Path(directory)
    manifest = verify_public(directory)
    inputs = directory / "team24_processing"
    targets = directory / "attack_targets"
    return {
        "candidates": read_csv(inputs / "B.csv"),
        "self_labels": read_csv(inputs / "B_self.csv"),
        "background": read_csv(inputs / "A_bg.csv"),
        "self_challenge": read_csv(targets / "aia_challenge_24.csv"),
        "targets": {k: {"release": read_csv(targets / f"C_{k}.csv"),
                         "challenge": read_csv(targets / f"aia_challenge_{k}.csv")}
                    for k in manifest["target_ids"]},
        "schema": read_json(inputs / "schema.json"),
        "cfg": read_json(directory / "cfg.json"),
        "self_release": read_csv(targets / "C_24.csv"),
        "utility_reference": read_json(inputs / "utility_ref.json"),
        "attacker_id": ATTACKER_ID,
    }


def run_attack(bundle, plugin, destination):
    bundle, plugin, destination = Path(bundle).resolve(), Path(plugin).resolve(), Path(destination).resolve()
    require(plugin.is_file() and plugin.suffix == ".py", "Plugin must be an existing Python file")
    require(not destination.exists(), "Prediction destination already exists")
    before = verify_bundle()
    public = verify_public(bundle)
    source_hashes = code_hashes()
    plugin_hash = sha256(plugin)
    destination.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    command = [sys.executable, "-I", "-B", str(HERE / "worker.py"), str(bundle), str(plugin), str(destination)]
    with (destination / "attack.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    require(verify_public(bundle) == public and sha256(plugin) == plugin_hash, "Inputs or plugin changed while running")
    require(verify_bundle() == before and code_hashes() == source_hashes, "Copied data or evaluation code changed while running")
    prediction_hashes = {name: sha256(destination / name) for name in ("F_mia.csv", "F_aia.csv")}
    write_json(destination / "prediction_provenance.json", {
        "round": "prelim", "attacker_id": ATTACKER_ID,
        "public_bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json"),
        "public_input_sha256": public["files"], "plugin_path": str(plugin), "plugin_sha256": plugin_hash,
        "prediction_sha256": prediction_hashes, "source_sha256": source_hashes,
        "isolation": "Fresh process and public-table arguments; trusted local code, not an OS filesystem sandbox."})
    return destination
