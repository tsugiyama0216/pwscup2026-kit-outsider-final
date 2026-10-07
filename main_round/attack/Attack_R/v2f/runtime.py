"""Pinned public inputs and upstream gates, without reading live credentials."""
from pathlib import Path
import hashlib
import json
import sys
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
ATTACK = HERE.parents[1]
REPO = ATTACK.parents[1]
MOCK = ATTACK.parent / "mock_test"
sys.path.insert(0, str(ATTACK))
sys.path.insert(0, str(ATTACK / "Attak_T"))
import attack_research
import attack_workspace as w


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def sha(path):
    if Path(path).name.lower() == "token.txt":
        raise ValueError("Never hash the live token")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_check():
    from attack_research.published_v88.check import verify_package
    from M.v2.artifacts import load_selection, check_v1_preserved
    selection = load_selection()
    old = read_json(HERE.parent / "v1/dependencies.json")
    for name, expected in old["files_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError("Frozen R v1/upstream dependency changed: " + name)
    return {"T_v88": verify_package(), "M_v1": check_v1_preserved(),
            "R_v1_files": len(old["files_sha256"]), "M_v2_strength": selection["mia_strength"]}


def selected_check():
    selection = read_json(HERE / "selection.json")
    for name, expected in selection["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Selected v2f code changed: " + name)
    return selection


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
