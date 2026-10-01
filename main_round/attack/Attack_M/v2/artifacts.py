"""Small hash gates for development selection, final checking and prediction."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "M/local-output/v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def dependencies():
    paths = set(HERE.glob("*.py"))
    paths.update(p for p in HERE.glob("*.json") if p.name != "selection.json")
    paths.update((ROOT / "M/v1").glob("*.py"))
    paths.update(ROOT / name for name in
                 ("attack_workspace.py", "attack.ps1", "source_manifest.json", "local_eval/data.py",
                  "local_eval/scoring.py"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(paths)}


def check_snapshot(snapshot):
    changed = [name for name, digest in snapshot.items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise ValueError("Frozen dependencies changed: " + ", ".join(changed))


def check_v1_preserved():
    snapshot = read_json(OUTPUT / "v1_preservation.json")
    check_snapshot(snapshot["files"])
    return {"verified_files": len(snapshot["files"]), "all_bytes_unchanged": True}


def load_selection():
    selection = read_json(HERE / "selection.json")
    if not selection["selected_before_holdout"]:
        raise ValueError("Missing development-only selection declaration")
    check_snapshot(selection["dependency_sha256"])
    return selection
