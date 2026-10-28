"""Extract public result ZIPs byte-for-byte and inventory their original members."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent
COHORTS = list(range(18)) + [22, 25, 26, 27, 28]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def members(archive):
    assert archive.testzip() is None, "ZIP CRC error"
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate ZIP member"
    for name in names:
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
        assert not any("token" in part.lower() for part in path.parts), "Unexpected credential file"
    return [n for n in names if not n.endswith("/")]


def main():
    inventory = []

    def save(archive_name, member_name, data, destination):
        path = ROOT / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, f"Existing file differs: {destination}"
        else:
            path.write_bytes(data)
        inventory.append({"archive": archive_name, "member": member_name,
                          "extracted_path": destination, "bytes": len(data), "sha256": digest(data)})

    truth_source = next((ROOT / "sources").glob("本戦*.zip"))
    submission_source = next((ROOT / "sources").glob("攻撃*.zip"))
    with zipfile.ZipFile(truth_source) as archive:
        names = members(archive)
        expected = {f"正解データ/{kind}_{k}.csv" for kind in ("pool_ids", "aia_truth") for k in COHORTS}
        expected |= {"正解データ/aia_floor.csv", "正解データ/README_正解データ.md"}
        assert set(names) == expected
        for name in names:
            save(truth_source.relative_to(ROOT).as_posix(), name, archive.read(name),
                 "truth/" + PurePosixPath(name).name)

    with zipfile.ZipFile(submission_source) as archive:
        names = members(archive)
        expected = {f"攻撃提出物/cohort{k:02}_F.zip" for k in COHORTS}
        expected.add("攻撃提出物/README_攻撃フェーズの提出物.md")
        assert set(names) == expected
        readme = "攻撃提出物/README_攻撃フェーズの提出物.md"
        save(submission_source.relative_to(ROOT).as_posix(), readme, archive.read(readme),
             "submissions/README_攻撃フェーズの提出物.md")
        for k in COHORTS:
            name = f"攻撃提出物/cohort{k:02}_F.zip"
            data = archive.read(name)
            inventory.append({"archive": submission_source.relative_to(ROOT).as_posix(),
                              "member": name, "extracted_path": "", "bytes": len(data), "sha256": digest(data)})
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                assert set(members(inner)) == {"F_mia.csv", "F_aia.csv"}
                for filename in sorted(inner.namelist()):
                    save(submission_source.relative_to(ROOT).as_posix() + "!" + name, filename,
                         inner.read(filename), f"submissions/cohort{k:02}/{filename}")

    def rows(path):
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    for k in COHORTS:
        pool = rows(ROOT / f"truth/pool_ids_{k}.csv")
        truth = rows(ROOT / f"truth/aia_truth_{k}.csv")
        assert len(pool) == 1049 and {int(r["record_id"]) for r in pool} == set(range(1049))
        assert len({r["pool_id"] for r in pool}) == 1049
        assert len(truth) == 326 and {int(r["challenge_row_id"]) for r in truth} == set(range(326))
        mia = rows(ROOT / f"submissions/cohort{k:02}/F_mia.csv")
        assert len(mia) == 1049 and {int(r["record_id"]) for r in mia} == set(range(1049))
    assert {int(r["team_id"]) for r in rows(ROOT / "truth/aia_floor.csv")} == set(COHORTS)
    with (ROOT / "archive_inventory.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["archive", "member", "extracted_path", "bytes", "sha256"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps({"cohorts": len(COHORTS), "truth_files": 48,
                      "submission_pairs": 23, "inventory_entries": len(inventory),
                      "zip_crc_and_member_checks": "passed"}))


if __name__ == "__main__":
    main()
