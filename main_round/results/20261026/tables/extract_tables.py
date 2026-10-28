"""Extract published workbook values without rewriting or recalculating it.

Uses only Python's standard library. Numeric cells retain the exact XML value
text, avoiding a round trip through binary floating point. Run from any cwd.
"""

from __future__ import annotations

import csv
import hashlib
import json
import posixpath
import re
from decimal import Decimal, getcontext
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

getcontext().prec = 50
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "sources" / "公開用結果表_本戦_20261026.xlsx"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
TOL = Decimal("1e-12")


def coordinate(ref):
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", ref)
    assert match, ref
    col = 0
    for letter in match[1]:
        col = col * 26 + ord(letter) - 64
    return int(match[2]), col


def column_name(col):
    result = ""
    while col:
        col, rem = divmod(col - 1, 26)
        result = chr(65 + rem) + result
    return result


def extract():
    tables, inventory = {}, []
    with ZipFile(SOURCE) as z:
        assert z.testzip() is None
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        for sheet in workbook.find("s:sheets", NS):
            title = sheet.attrib["name"]
            target = rels[sheet.attrib[REL]]
            xml_path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            xml = ET.fromstring(z.read(xml_path))
            dimension = xml.find("s:dimension", NS).attrib["ref"]
            rows, cols = coordinate(dimension.split(":")[-1])
            values = [[""] * cols for _ in range(rows)]
            formulas, errors = [], []
            for cell in xml.findall(".//s:sheetData/s:row/s:c", NS):
                r, c = coordinate(cell.attrib["r"])
                value = cell.findtext("s:v", default="", namespaces=NS)
                kind = cell.attrib.get("t", "n")
                if kind == "s":
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                elif kind == "b":
                    value = "TRUE" if value == "1" else "FALSE"
                if cell.find("s:f", NS) is not None:
                    formulas.append(cell.attrib["r"])
                if kind == "e":
                    errors.append({"cell": cell.attrib["r"], "value": value})
                values[r - 1][c - 1] = value
            path = ROOT / f"{title}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f, lineterminator="\n").writerows(values)
            with path.open(encoding="utf-8-sig", newline="") as f:
                assert list(csv.reader(f)) == values
            tables[title] = values
            inventory.append({"sheet": title, "range": dimension, "rows_including_header": rows,
                              "columns": cols, "csv": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "formula_cells": formulas, "error_cells": errors})
    return tables, inventory


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    tables, inventory = extract()
    final = tables["最終得点"][1:]
    main_rows = tables["総括"][1:]
    cohorts = set(range(18)) | {22, 25, 26, 27, 28}
    assert {int(r[4]) for r in final} == {int(r[0]) for r in main_rows} == cohorts
    main_by_id = {int(r[0]): r for r in main_rows}
    checks = []

    def check(name, differences, source):
        differences = list(differences)
        maximum = max(map(abs, differences), default=Decimal(0))
        checks.append({"check": name, "count": len(differences), "max_absolute_difference": str(maximum),
                       "tolerance": str(TOL), "passed": maximum <= TOL, "source": source})

    check("final_overall_weighted_0.1_prelim_plus_0.9_main",
          (Decimal(r[7]) - Decimal("0.1") * Decimal(r[5]) - Decimal("0.9") * Decimal(r[6]) for r in final), "最終得点!F2:H24")
    check("final_attack_weighted_0.1_prelim_plus_0.9_main",
          (Decimal(r[12]) - Decimal("0.1") * Decimal(r[10]) - Decimal("0.9") * Decimal(r[11]) for r in final), "最終得点!K2:M24")
    check("main_overall_same_between_sheets",
          (Decimal(r[6]) - Decimal(main_by_id[int(r[4])][15]) for r in final), "最終得点!E2:G24; 総括!A2:P24")
    check("main_attack_same_between_sheets",
          (Decimal(r[11]) - Decimal(main_by_id[int(r[4])][17]) for r in final), "最終得点!E2:L24; 総括!A2:R24")
    for label, rows, score_col, rank_col, source in [
        ("final_overall", final, 7, 0, "最終得点!A2:H24"),
        ("final_attack", final, 12, 9, "最終得点!J2:M24"),
        ("main_overall", main_rows, 15, 16, "総括!P2:Q24"),
        ("main_attack", main_rows, 17, 18, "総括!R2:S24"),
    ]:
        check(label + "_official_ranks_match_competition_ranking",
              (Decimal(r[rank_col]) - (1 + sum(Decimal(o[score_col]) > Decimal(r[score_col]) for o in rows)) for r in rows), source)
    check("main_U_equals_min_of_four_reported_components",
          (Decimal(r[2]) - min(map(Decimal, r[3:7])) for r in main_rows), "総括!C2:G24")
    check("main_Anon_equals_one_minus_max_MIA_AIA_losses",
          (Decimal(r[14]) - (1 - max(Decimal(r[10]), Decimal(r[13]))) for r in main_rows), "総括!K2:O24")
    check("main_overall_equals_min_U_Anon",
          (Decimal(r[15]) - min(Decimal(r[2]), Decimal(r[14])) for r in main_rows), "総括!C2:P24")
    check("main_AIA_loss_equals_max_zero_raw_minus_phi",
          (Decimal(r[13]) - max(Decimal(0), Decimal(r[11]) - Decimal(r[12])) for r in main_rows), "総括!L2:N24")
    floors = {int(r[0]): Decimal(r[2]) for r in tables["AIAの差し引く水準"][1:]}
    components = {}
    for title in ["攻撃_標的別_MIA", "攻撃_標的別_AIA"]:
        records = tables[title][1:]
        assert len(records) == 23 * 22
        assert {(int(r[0]), int(r[1])) for r in records} == {(a, b) for a in cohorts for b in cohorts if a != b}
    for attacker in sorted(cohorts):
        mia = [r for r in tables["攻撃_標的別_MIA"][1:] if int(r[0]) == attacker]
        aia = [r for r in tables["攻撃_標的別_AIA"][1:] if int(r[0]) == attacker]
        m = sum(Decimal(r[6]) for r in mia) / 22
        a = sum(max(Decimal(0), Decimal(r[9]) - floors[int(r[1])]) for r in aia) / 22
        components[attacker] = {"MIA_mean_t_j": str(m), "AIA_mean_after_target_phi": str(a), "sum": str(m + a),
                                "target_count": 22, "source": "攻撃_標的別_MIA!A2:G507; 攻撃_標的別_AIA!A2:J507; AIAの差し引く水準!A2:C24"}
    check("attack_power_reaggregated_from_published_details",
          (Decimal(main_by_id[k][17]) - Decimal(v["sum"]) for k, v in components.items()),
          "総括!A2:R24; 攻撃_標的別_MIA!A2:G507; 攻撃_標的別_AIA!A2:J507; AIAの差し引く水準!A2:C24")
    outsider = {}
    for title, key_col in [("最終得点", 4), ("総括", 0), ("AIAの差し引く水準", 0)]:
        rows = tables[title]
        index, row = next((i, r) for i, r in enumerate(rows[1:], 2) if r[key_col] == "10")
        outsider[title] = {"range": f"{title}!A{index}:{column_name(len(row))}{index}",
                           "values": {label: {"value": value, "cell": f"{title}!{column_name(col)}{index}"}
                                      for col, (label, value) in enumerate(zip(rows[0], row), 1)}}
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    report = {"source": "../sources/" + SOURCE.name, "source_sha256": source_hash,
              "numeric_storage": "All source numeric values are strings copied from XLSX XML; no decimal digits are rounded during extraction.",
              "csv_encoding": "UTF-8 with BOM; LF record separators; blank cells remain empty.",
              "formula_handling": "Cached values only; no formula recalculation or workbook changes.",
              "limits": "Arithmetic reconciliation of published values, not independent rerunning of the official scorer. Utility is the published ledger value.",
              "sheets": inventory, "cohort_ids": sorted(cohorts), "team_count": 23,
              "pairwise_detail_rows_per_metric": 506, "self_targets_excluded": True,
              "outsider": outsider, "attack_components_from_published_details": {str(k): v for k, v in components.items()},
              "checks": checks, "all_checks_passed": all(c["passed"] for c in checks)}
    (ROOT / "workbook_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"sheets": len(inventory), "team_count": 23, "checks": checks, "all_checks_passed": report["all_checks_passed"]}, ensure_ascii=False, indent=2))
    assert report["all_checks_passed"]


if __name__ == "__main__":
    main()
