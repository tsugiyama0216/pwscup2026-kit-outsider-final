"""Paths and provenance checks for the copied preliminary-round materials."""
from __future__ import annotations

from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT_ROOT = HERE.parent / "local-output/mock_TS_using_prelim"
OFFICIAL = HERE / "official-kit/starter"
ATTACKER_ID = 24


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), "Manifest path must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Manifest path escapes its root")
    return path


def new_output(path=None, action="run"):
    path = Path(path) if path else OUTPUT_ROOT / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + action)
    path = path.resolve()
    require(path != OUTPUT_ROOT.resolve() and path.is_relative_to(OUTPUT_ROOT.resolve()),
            "Output must be a new subdirectory of main_round/mock_test/local-output/mock_TS_using_prelim")
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())), "Refusing to overwrite a nonempty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_official():
    for name, module in list(sys.modules.items()):
        if name == "pwscup2026" or name.startswith("pwscup2026."):
            origin = getattr(module, "__file__", None)
            require(origin is None or Path(origin).resolve().is_relative_to(OFFICIAL.resolve()),
                    "Another round's pwscup2026 is already imported; start a fresh process")
    sys.path.insert(0, str(OFFICIAL / "src"))


def reference_config():
    install_official()
    spec = importlib.util.spec_from_file_location("_prelim_reference_config", OFFICIAL / "reference/_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_cfg()


def verify_bundle():
    """Verify all copied bytes; source Downloads directories are not required."""
    checked = {}
    for filename in ("data_manifest.json", "official_source_manifest.json"):
        manifest_path = HERE / "provenance" / filename
        manifest = read_json(manifest_path)
        files = manifest["files"]
        require(bool(files), "Empty manifest: " + filename)
        for entry in files:
            path = within(HERE, entry["path"])
            require(path.name.lower() != "token.txt", "Credential found in evaluation manifest")
            require(path.is_file() and sha256(path) == entry["sha256"], "Copied file missing or changed: " + entry["path"])
        if filename == "data_manifest.json":
            for category in ("public", "private", "reference"):
                prefix = "data/" + category + "/"
                expected_paths = {entry["path"] for entry in files if entry["path"].startswith(prefix)}
                actual_paths = {p.relative_to(HERE).as_posix() for p in (DATA / category).rglob("*") if p.is_file()}
                require(expected_paths == actual_paths, "Unlisted or missing files in data/" + category)
        checked[filename] = {"files": len(files), "manifest_sha256": sha256(manifest_path)}
    processing = DATA / "public/team24_processing"
    pools = read_csv(DATA / "private/ground_truth/pool_ids_24.csv")
    candidates = read_csv(processing / "B.csv")
    require(len(candidates) == 1049 and candidates.record_id.is_unique, "Invalid actual team24 candidates")
    require(set(candidates.record_id) == set(pools.record_id), "B.csv and team24 ground truth IDs differ")
    self_labels = read_csv(processing / "B_self.csv")
    require(set(self_labels.record_id) == set(candidates.record_id), "B_self candidate IDs differ")
    checked["round"] = "prelim-attack-20260825"
    checked["attacker_id"] = ATTACKER_ID
    checked["candidate_rows"] = len(candidates)
    checked["runtime"] = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}
    return checked


def code_hashes():
    return {p.relative_to(HERE).as_posix(): sha256(p) for p in sorted(HERE.glob("*.py"))}


def team_names():
    table = read_csv(DATA / "reference/published_results/team_mapping.csv")
    return {int(row.cohort_id): str(row.team_name) for row in table.itertuples()}
