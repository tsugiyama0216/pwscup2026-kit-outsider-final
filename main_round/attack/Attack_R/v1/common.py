"""Local paths, strict dependency gates and serialization; no token logging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ATTACK = HERE.parents[1]
REPOSITORY = ATTACK.parents[1]
if str(ATTACK) not in sys.path:
    sys.path.insert(0, str(ATTACK))


def sha(path):
    path = Path(path)
    if path.name.lower() == "token.txt":
        raise ValueError("Live token hashing is prohibited")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n").encode("utf-8"))


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def verify_dependencies():
    manifest = read_json(HERE / "dependencies.json")
    for name, expected in manifest["files_sha256"].items():
        path = (REPOSITORY / name).resolve()
        if not path.is_relative_to(REPOSITORY) or not path.is_file() or sha(path) != expected:
            raise ValueError("R v1 frozen dependency mismatch: " + name)
    return {"verified_files": len(manifest["files_sha256"]),
            "manifest_sha256": sha(HERE / "dependencies.json")}


def load_actual():
    import attack_workspace as w

    originals = w.verify_hashes()
    np, pd, modules, cfg, versions = w.load_runtime()
    _, _, _, _, dist = w.load_inputs(np, pd, modules["pwscup2026.kit.validate"])
    read = lambda p: pd.read_csv(p, float_precision="round_trip")
    b = read(w.ORIGINALS / "B.csv")
    background = read(w.ORIGINALS / "A_bg.csv")
    targets = {k: read(w.TARGETS / f"C_{k}.csv") for k in w.ATTACK_TARGETS}
    challenges = {k: read(w.TARGETS / f"aia_challenge_{k}.csv") for k in w.ATTACK_TARGETS}
    return w, np, pd, modules["pwscup2026.kit.validate"], cfg, dist, b, background, targets, challenges, originals, versions
