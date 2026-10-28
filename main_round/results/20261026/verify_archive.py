"""Verify archive bytes, organizer hashes and the saved file manifest (Python 3 stdlib)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
EXPECTED_ATTACK_ZIP_SHA256 = "bdab1804efd39e6e227220ede168d2031fe1275b1b46655ef7891116a3160dc3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Explicitly regenerate the manifest after preparing/reviewing archive contents")
    args = parser.parse_args()
    inventory = read_csv(ROOT / "archive_inventory.csv")
    for item in inventory:
        parts = item["archive"].split("!")
        data = (ROOT / parts[0]).read_bytes()
        for name in parts[1:] + [item["member"]]:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                data = z.read(name)
        assert sha(data) == item["sha256"] and len(data) == int(item["bytes"]), item
        if item["extracted_path"]:
            assert (ROOT / item["extracted_path"]).read_bytes() == data, item

    meta = {r["項目"]: r["値"] for r in read_csv(ROOT / "tables/メタ.csv")}
    external = []
    for key, expected in meta.items():
        if key.startswith("sha256.attack."):
            cohort, name = key[len("sha256.attack."):].split(".", 1)
            path = ROOT / f"submissions/cohort{int(cohort):02}/{name}"
        elif key.startswith("sha256.defense."):
            path = REPO / "main_round/attack/targets" / key[len("sha256.defense."):]
            external.append({"path": path.relative_to(REPO).as_posix(), "sha256": expected})
        elif key == "sha256.ref.mia_columns.json":
            path = REPO / "main_round/attack/targets/mia_columns.json"
            external.append({"path": path.relative_to(REPO).as_posix(), "sha256": expected})
        else:
            continue
        assert sha(path.read_bytes()) == expected, f"Official workbook hash mismatch: {key}"
    assert sum(k.startswith("sha256.attack.") for k in meta) == 46
    assert len(external) == 24, "Expected 23 defense hashes plus mia_columns"
    attack_zip = REPO / "main_round/attack/downloads/PWSCup2026_final_attack.zip"
    assert sha(attack_zip.read_bytes()) == EXPECTED_ATTACK_ZIP_SHA256
    external.append({"path": attack_zip.relative_to(REPO).as_posix(), "sha256": EXPECTED_ATTACK_ZIP_SHA256})
    with zipfile.ZipFile(attack_zip) as z:
        assert z.testzip() is None
        for item in external[:-1]:
            p = REPO / item["path"]
            assert z.read(p.name) == p.read_bytes(), p

    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    for entry in sources["sources"]:
        if "bytes" in entry:
            assert (ROOT / entry["path"]).stat().st_size == entry["bytes"]
    summary = json.loads((ROOT / "tables/workbook_summary.json").read_text(encoding="utf-8"))
    workbook = next((ROOT / "sources").glob("*.xlsx"))
    assert sha(workbook.read_bytes()) == summary["source_sha256"]
    for entry in summary["sheets"]:
        assert sha((ROOT / "tables" / entry["csv"]).read_bytes()) == entry["sha256"]
    self_columns = []
    for path in sorted((ROOT / "submissions").glob("cohort*/F_mia.csv")):
        cohort = int(path.parent.name[6:])
        with path.open(encoding="utf-8-sig", newline="") as f:
            headers = next(csv.reader(f))
        if str(cohort) in headers:
            self_columns.append(cohort)
    assert self_columns == [0, 3, 5, 7, 11, 16, 25]

    manifest_path = ROOT / "manifest.json"
    files = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(ROOT.rglob("*"))
             if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
    checks = {"archive_entries_verified": len(inventory),
              "official_prediction_hashes_verified": sum(k.startswith("sha256.attack.") for k in meta),
              "official_defense_hashes_verified": sum(k.startswith("sha256.defense.") for k in meta),
              "existing_attack_distribution_hash_verified": True,
              "original_self_mia_columns_preserved": self_columns,
              "workbook_and_all_sheet_csv_hashes_verified": True}
    if args.write_manifest:
        manifest = {"saved_on": "2026-10-28", "timezone": "Asia/Tokyo",
                    "official_publication_date": "2026-10-26", "files": files,
                    "external_dependencies_relative_to_repo": external, "checks": checks}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert files == manifest["files"], "File inventory or bytes differ from saved manifest"
        assert checks == manifest["checks"]
        assert external == manifest["external_dependencies_relative_to_repo"]
    print(json.dumps({"status": "passed", "files": len(files), **checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
