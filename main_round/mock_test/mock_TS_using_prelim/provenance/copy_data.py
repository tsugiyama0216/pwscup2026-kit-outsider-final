"""Create the local preliminary data bundle without reading credentials.

Only allowlisted files are read from the team distribution. Existing files may
be reused only when byte-identical; differing destination files fail closed.
The original XLSX is preserved; sheet CSVs contain the stored cell values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

BUNDLE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RESULTS = REPO / "prelim/results/20260909"
EXPECTED_ACTUAL_HASHES = {
    "B.csv": "7a7aa39e16fa5c76ee9c8dc009928373aec6cf47e7b1bd10dc4e6899b5d2ab3a",
    "B_self.csv": "297354a3c909fba6bf4d33053cf91efa9139892bf5269050a5607b86e0f33e35",
    "A_bg.csv": "f4817fba6c1434d6b74221c2b5abe90e4a53e80d8fa17e12602d43adae0a3126",
    "schema.json": "d0698eb89cde0901b28ddaf72cb57270d744bdb72b45f8315f75b3199f024128",
}
URLS = {
    "targets": "https://github.com/pwscup/pwscup2026-kit/releases/tag/prelim-attack-20260825",
    "results": "https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view",
    "truth": "https://drive.google.com/file/d/1jnYGBEX1vjSgrvzBywHjeBnYru9_H5MQ/view",
    "attacks": "https://drive.google.com/file/d/1x1KGX_XmEMunHqEmjq8ZYMcDqx5BtQs8/view",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_preserving(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing file: {path}")
    else:
        path.write_bytes(data)


def csv_info(data: bytes) -> dict:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    return {"rows_excluding_header": max(0, len(rows) - 1),
            "columns": len(rows[0]) if rows else 0,
            "column_names": rows[0] if rows else [],
            "row_widths": sorted({len(row) for row in rows})}


def export_xlsx_sheets(path: Path) -> list[dict]:
    """Read stored XLSX values with stdlib; do not alter/recalculate workbook."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    doc_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    exports = []
    with zipfile.ZipFile(path) as zf:
        strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in root.findall("m:si", ns)]
        relationships = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {node.attrib["Id"]: node.attrib["Target"]
                for node in relationships.findall(f"{{{rel_ns}}}Relationship")}
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        for index, sheet in enumerate(workbook.findall("m:sheets/m:sheet", ns), 1):
            rel = rels[sheet.attrib[f"{{{doc_rel}}}id"]]
            member = rel.lstrip("/") if rel.startswith("/") else "xl/" + rel
            root = ET.fromstring(zf.read(member))
            cells, max_row, max_col, formulas = {}, 0, 0, 0
            for row in root.findall("m:sheetData/m:row", ns):
                for cell in row.findall("m:c", ns):
                    match = re.fullmatch(r"([A-Z]+)(\d+)", cell.attrib["r"])
                    assert match
                    column = 0
                    for ch in match[1]:
                        column = column * 26 + ord(ch) - ord("A") + 1
                    row_num = int(match[2])
                    max_row, max_col = max(max_row, row_num), max(max_col, column)
                    value_node = cell.find("m:v", ns)
                    value = value_node.text if value_node is not None and value_node.text is not None else ""
                    if cell.attrib.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.attrib.get("t") == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall("m:is//m:t", ns))
                    if cell.find("m:f", ns) is not None:
                        formulas += 1
                    cells[row_num, column] = value
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            for row in range(1, max_row + 1):
                writer.writerow([cells.get((row, column), "") for column in range(1, max_col + 1)])
            exports.append({"sheet": sheet.attrib["name"], "index": index,
                            "rows_including_header": max_row, "columns": max_col,
                            "formula_cells": formulas, "xlsx_member": member,
                            "filename": f"{index:02d}_{sheet.attrib['name']}.csv",
                            "data": output.getvalue().encode("utf-8-sig")})
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processing-directory", type=Path, required=True)
    parser.add_argument("--attack-directory", type=Path,
                        default=REPO / "prelim/workspace/team-shared/prelim/attack-targets")
    args = parser.parse_args()
    processing = args.processing_directory.resolve()
    targets = args.attack_directory.resolve()
    records = []
    checks = {}
    # Validate the old frozen actual-input identity; practice data is never used.
    for name, expected in EXPECTED_ACTUAL_HASHES.items():
        if digest((processing / name).read_bytes()) != expected:
            raise ValueError(f"Actual preliminary team24 source hash mismatch: {name}")
    checks["actual_team24_pinned_hashes"] = len(EXPECTED_ACTUAL_HASHES)
    historical_manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8-sig"))
    for item in historical_manifest["files"]:
        source = RESULTS / item["path"]
        content = source.read_bytes()
        if digest(content) != item["sha256"] or len(content) != item["bytes"]:
            raise ValueError(f"20260909 source manifest mismatch: {item['path']}")
    checks["published_result_source_manifest_files"] = len(historical_manifest["files"])
    target_hashes = {}
    for line in (targets / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if Path(name).name != name:
                raise ValueError("Unsafe target checksum filename")
            if digest((targets / name).read_bytes()) != expected.lower():
                raise ValueError(f"Attack target checksum mismatch: {name}")
            target_hashes[name] = expected.lower()
    checks["distributed_target_checksum_files"] = len(target_hashes)

    def add(source: Path, destination: str, role: str, scope: str,
            reason: str, source_url: str | None = None,
            content: bytes | None = None, derivation: dict | None = None) -> None:
        source = source.resolve()
        data = source.read_bytes() if content is None else content
        dest = BUNDLE / "data" / destination
        write_preserving(dest, data)
        source_repo_relative = None
        try:
            source_repo_relative = source.relative_to(REPO).as_posix()
        except ValueError:
            pass
        record = {
            "path": dest.relative_to(BUNDLE).as_posix(),
            "source_absolute_path": str(source),
            "source_repo_relative_path": source_repo_relative,
            "source_url": source_url,
            "source_sha256": digest(source.read_bytes()),
            "sha256": digest(data), "bytes": len(data),
            "role": role, "scope": scope, "copy_reason": reason,
            "is_byte_identical_copy": content is None,
        }
        if derivation:
            record["derivation"] = derivation
        if dest.suffix.lower() == ".csv":
            record["table"] = csv_info(data)
        records.append(record)

    for name in ["B.csv", "B_self.csv", "A_bg.csv", "schema.json", "utility_ref.json", "README.txt"]:
        add(processing / name, f"public/team24_processing/{name}",
            "own_processing_distribution", "attack_input_available_to_team24",
            "予備戦team24へ実配布された加工フェーズ原本。練習用配布物と置き換えない。")
    for source in sorted(targets.iterdir()):
        if source.is_file() and (re.fullmatch(r"(?:C|aia_challenge)_\d+\.csv", source.name)
                               or source.name in {"mia_columns.json", "README.txt", "SHA256SUMS.txt"}):
            add(source, f"public/attack_targets/{source.name}", "attack_distribution",
                "attack_input_available_to_team24", "実攻撃対象・AIA問題・対象一覧と配布時のチェックサムを保存。",
                URLS["targets"])
    for source in sorted((RESULTS / "sources/ground_truth").iterdir()):
        if source.is_file() and (re.fullmatch(r"(?:pool_ids|aia_truth)_\d+\.csv", source.name)
                               or source.name == "README_予備戦の正解データ.md"):
            add(source, f"private/ground_truth/{source.name}", "published_ground_truth",
                "scorer_only_post_contest_truth", "答え合わせ専用。攻撃プラグインの入力に渡さない。", URLS["truth"])
    submission_dir = RESULTS / "sources/attack_submissions"
    for source in sorted(submission_dir.glob("cohort*_F.zip")):
        with zipfile.ZipFile(source) as zf:
            if set(zf.namelist()) != {"F_mia.csv", "F_aia.csv"}:
                raise ValueError(f"Unexpected public submission archive members: {source.name}")
            add(source, f"reference/attack_submissions/original_zips/{source.name}",
                "published_attack_submission_archive", "reference_only_post_contest",
                "各チームが掲載した1件の公式公開回答。事務局が認証情報を除去した原本ZIP。", URLS["attacks"])
            cohort = re.fullmatch(r"(cohort\d+)_F.zip", source.name)[1]
            for member in ("F_mia.csv", "F_aia.csv"):
                add(source, f"reference/attack_submissions/extracted/{cohort}/{member}",
                    "published_attack_submission_csv", "reference_only_post_contest",
                    "元ZIPのCSVを無改変で展開し、比較・再採点で使いやすくする。", URLS["attacks"],
                    zf.read(member), {"operation": "zip_member_byte_extraction", "archive_member": member})
    add(submission_dir / "README_攻撃フェーズの提出物.md",
        "reference/attack_submissions/README_攻撃フェーズの提出物.md", "organizer_readme",
        "reference_only_post_contest", "公開回答の範囲と自己列の扱いを確認する原本。", URLS["attacks"])
    xlsx = RESULTS / "sources/公開用結果表_20260909.xlsx"
    add(xlsx, "reference/published_results/公開用結果表_20260909.xlsx", "published_result_workbook",
        "reference_only_post_contest", "公表点と対象別の内訳を照合する原本。", URLS["results"])
    for name in ["README.md", "manifest.json", "team_summary.csv", "team24_attack_by_target.csv"]:
        target_name = "historical_" + name if name in {"README.md", "manifest.json"} else name
        add(RESULTS / name, f"reference/published_results/{target_name}", "historical_result_analysis",
            "reference_only_post_contest", "既存の集計と原本照合記録。READMEの相対リンクはコピー元を基準に読む。")
    sheets = export_xlsx_sheets(xlsx)
    for sheet in sheets:
        add(xlsx, f"reference/published_results/sheets/{sheet['filename']}", "workbook_sheet_values",
            "reference_only_post_contest", "Excel原本の保存値を全セル書き出し。数式・書式の代替ではない。", URLS["results"],
            sheet["data"], {"operation": "xlsx_stored_values_to_csv", **{k: v for k, v in sheet.items() if k != "data"}})
    # Stable ASCII-column tables for evaluator comparisons; preserve blank cells.
    ascii_exports = {
        "総括": ("team_mapping.csv", [
            "cohort_id", "team_name", "utility", "utility_gen", "utility_spec", "utility_rare", "utility_valid",
            "beta_raw", "beta_chance", "beta", "mia_component", "aia_component", "anonymity",
            "overall_score", "overall_rank", "official_attack_power", "official_attack_rank"], 22),
        "攻撃_標的別_MIA": ("mia_by_target.csv", [
            "attacker", "target", "n_member", "n_member_rare", "tpr_at_low_fpr", "tpr_rare", "t_j", "auc"], 420),
        "攻撃_標的別_AIA": ("aia_by_target.csv", [
            "attacker", "target", "r_time_all", "r_time_rare", "r_time_rare_unadjusted", "p_m", "p_c", "p_m_rare", "p_c_rare", "r_time"], 419),
    }
    for sheet in sheets:
        if sheet["sheet"] not in ascii_exports:
            continue
        filename, column_names, expected_rows = ascii_exports[sheet["sheet"]]
        rows = list(csv.reader(io.StringIO(sheet["data"].decode("utf-8-sig"), newline="")))
        assert len(rows[0]) == len(column_names) and len(rows) - 1 == expected_rows
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(column_names)
        writer.writerows(rows[1:])
        add(xlsx, f"reference/published_results/{filename}", "workbook_ascii_column_values",
            "reference_only_post_contest", "公表表の保存値を変更せず列名だけASCIIへ対応付け。空欄は空欄のまま保持。", URLS["results"],
            output.getvalue().encode("utf-8-sig"), {"operation": "xlsx_stored_values_to_csv_with_ascii_headers",
                "sheet": sheet["sheet"], "column_mapping": dict(zip(rows[0], column_names)),
                "blank_cells_preserved": True, "rows_excluding_header": expected_rows})

    add(REPO / "prelim/anonymization/anonymization-submitted-v5/C.csv",
        "reference/own_submissions/anonymization_v5/C.csv", "own_final_anonymization",
        "reference_own_submission", "自チームの実提出匿名化v5と配布C_24が同一であることを確認する。")
    for name in ["F_mia.csv", "F_aia.csv"]:
        add(REPO / "prelim/attack/final" / name, f"reference/own_submissions/attack_v9/{name}",
            "own_final_attack", "reference_own_submission", "提出済みv9 guardedを最初の再採点と回帰確認に使う。")
    add(REPO / "prelim/attack/results/guarded_final.json", "reference/own_submissions/attack_v9/guarded_final.json",
        "own_final_reported_feedback", "reference_own_submission", "ユーザー共有の当時の公式表示・詳細を公表確定値と区別して保存。")

    # Independent structural checks, all without displaying data rows.
    by_path = {item["path"]: item for item in records}
    for name in ["B.csv", "B_self.csv"]:
        assert by_path[f"data/public/team24_processing/{name}"]["table"]["rows_excluding_header"] == 1049
    assert by_path["data/public/team24_processing/A_bg.csv"]["table"]["rows_excluding_header"] == 5143
    pool_records = [item for item in records if item["role"] == "published_ground_truth" and Path(item["path"]).name.startswith("pool_ids_")]
    aia_records = [item for item in records if item["role"] == "published_ground_truth" and Path(item["path"]).name.startswith("aia_truth_")]
    assert len(pool_records) == 22 and all(item["table"]["rows_excluding_header"] == 1049 for item in pool_records)
    assert len(aia_records) == 21 and all(item["table"]["rows_excluding_header"] == 326 for item in aia_records)
    assert len([item for item in records if item["role"] == "published_attack_submission_archive"]) == 21
    assert len([item for item in records if item["role"] == "published_attack_submission_csv"]) == 42
    assert len([item for item in records if item["role"] == "attack_distribution"]) == 45
    for name in ["F_mia.csv", "F_aia.csv"]:
        assert by_path[f"data/reference/own_submissions/attack_v9/{name}"]["sha256"] == by_path[f"data/reference/attack_submissions/extracted/cohort24/{name}"]["sha256"]
    assert by_path["data/reference/own_submissions/anonymization_v5/C.csv"]["sha256"] == by_path["data/public/attack_targets/C_24.csv"]["sha256"]
    def read_rows(relative: str) -> list[dict[str, str]]:
        with (BUNDLE / "data" / relative).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    candidates = read_rows("public/team24_processing/B.csv")
    self_rows = read_rows("public/team24_processing/B_self.csv")
    pool24 = read_rows("private/ground_truth/pool_ids_24.csv")
    assert [{key: row[key] for key in candidates[0]} for row in self_rows] == candidates
    assert {row["record_id"] for row in candidates} == {row["record_id"] for row in pool24}
    rare_by_record = {row["record_id"]: row["is_rare"] for row in pool24}
    assert all(row["is_rare"] == rare_by_record[row["record_id"]] for row in self_rows)
    assert sum(int(row["is_rare"]) for row in self_rows) == 24
    for item in pool_records:
        rows = read_rows(item["path"].removeprefix("data/"))
        assert len({row["record_id"] for row in rows}) == 1049
        assert len({row["pool_id"] for row in rows}) == 1049
    for item in aia_records:
        rows = read_rows(item["path"].removeprefix("data/"))
        cohort = re.fullmatch(r"aia_truth_(\d+)\.csv", Path(item["path"]).name)[1]
        challenge = read_rows(f"public/attack_targets/aia_challenge_{cohort}.csv")
        assert len(challenge) == 326
        assert len({row["challenge_row_id"] for row in rows}) == 326
        assert {row["challenge_row_id"] for row in rows} == {row["challenge_row_id"] for row in challenge}
        assert sum(row["label"] == "member" for row in rows) == 163
        assert sum(row["label"] == "control" for row in rows) == 163
    team_mapping = read_rows("reference/published_results/team_mapping.csv")
    assert len(team_mapping) == 22 and len({row["cohort_id"] for row in team_mapping}) == 22
    assert sum(bool(row["official_attack_power"]) for row in team_mapping) == 21
    team21 = next(row for row in team_mapping if row["cohort_id"] == "21")
    assert all(team21[key] == "" for key in ["utility", "anonymity", "overall_score", "official_attack_power"])
    mia_public = read_rows("reference/published_results/mia_by_target.csv")
    aia_public = read_rows("reference/published_results/aia_by_target.csv")
    mia_pairs = {(row["attacker"], row["target"]) for row in mia_public}
    aia_pairs = {(row["attacker"], row["target"]) for row in aia_public}
    assert len(mia_pairs) == 420 and len(aia_pairs) == 419
    assert mia_pairs - aia_pairs == {("4", "23")}
    checks.update({"team24_B_matches_B_self": True, "team24_record_ids_match_pool_truth": True,
                   "team24_rare_labels_match_pool_truth": True, "team24_rare_rows": 24,
                   "pool_ids_unique_per_cohort": True, "aia_challenge_truth_ids_match": 21,
                   "aia_member_control_rows_per_file": [163, 163],
                   "team_mapping_rows": 22, "team_mapping_scored_rows": 21,
                   "cohort21_blank_scores_preserved": True,
                   "published_mia_pairs": 420, "published_aia_pairs": 419,
                   "published_missing_aia_pair_preserved": {"attacker": 4, "target": 23}})

    checks.update({"processing_files": 6, "attack_distribution_files": 45,
                   "pool_files": 22, "pool_rows_per_file": 1049,
                   "aia_truth_files": 21, "aia_truth_rows_per_file": 326,
                   "published_attack_archives": 21, "extracted_attack_csv_files": 42,
                   "own_final_attack_byte_matches": 2, "own_C_24_byte_match": True,
                   "workbook_sheets_exported": len(sheets),
                   "workbook_formula_cells": sum(sheet["formula_cells"] for sheet in sheets)})
    for item in records:
        content = (BUNDLE / item["path"]).read_bytes()
        if digest(content) != item["sha256"] or len(content) != item["bytes"]:
            raise ValueError(f"Copied file mismatch: {item['path']}")
    manifest = {
        "schema_version": 1, "created_date_jst": "2026-10-07", "bundle_name": "mock_TS_using_prelim",
        "source_phase": "preliminary", "attacker_cohort": 24,
        "local_only": True, "token_read_or_copied": False, "practice_fallback_allowed": False,
        "path_base": "mock_TS_using_prelim", "source_urls": URLS,
        "public_directory_meaning": "攻撃時にteam24が利用できた入力。インターネット公開の許可を意味しない。",
        "source_file_count": len([item for item in records if item["is_byte_identical_copy"]]),
        "derived_file_count": len([item for item in records if not item["is_byte_identical_copy"]]),
        "total_file_count": len(records), "total_bytes": sum(item["bytes"] for item in records),
        "verification": checks,
        "known_source_limitations": [
            "公開回答は各チーム1件のみで、全提出履歴ではない。",
            "cohort04のF_aiaには標的23がなく、公式公開結果にも同じ1組が欠落。欠落を補完・0置換しない。",
            "他チームの非公開B/B_selfは含まれない。公開pool_idsはメンバー正解の照合用。",
            "予備戦の実データと正解による評価であり、本戦の隠し正解・AIA控除値を再現しない。",
            "Excel派生CSVは保存値のみ。原本XLSXを保持し、再計算・レイアウト変更を行わない。"],
        "files": records,
    }
    path = BUNDLE / "provenance/data_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "files": len(records), "bytes": manifest["total_bytes"], "verification": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()


