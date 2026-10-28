"""Re-score the 23 published submissions with the unchanged pinned official kit.

Run from any working directory using main_round/attack/.venv/Scripts/python.exe.
No training, prediction generation, submission, or original-file modification.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
REPO = ARCHIVE.parents[2]
ATTACK = REPO / "main_round/attack"
SRC = ATTACK / "official-kit/starter/src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import yaml
from pwscup2026.common.submission_schema import read_scoring_csv
from pwscup2026.kit.codabench.scoring_io import score_attack
from pwscup2026.scoring import attack_score

COHORTS = list(range(18)) + [22, 25, 26, 27, 28]
TOLERANCE = 1e-12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workbook_sheets(path):
    """Read cached numeric values and strings without an Excel dependency."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    result = {}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for s in ET.fromstring(z.read("xl/workbook.xml")).find("s:sheets", ns):
            target = rels[s.attrib[f"{{{rel_ns}}}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            for row in ET.fromstring(z.read(target)).findall("s:sheetData/s:row", ns):
                values = []
                for cell in row:
                    letters = "".join(c for c in cell.attrib["r"] if c.isalpha())
                    col = 0
                    for c in letters:
                        col = 26 * col + ord(c) - 64
                    values.extend([None] * (col - len(values)))
                    typ = cell.attrib.get("t")
                    v = cell.find("s:v", ns)
                    if typ == "inlineStr":
                        value = "".join(cell.find("s:is", ns).itertext())
                    elif v is None:
                        value = None
                    elif typ == "s":
                        value = strings[int(v.text)]
                    else:
                        value = float(v.text) if typ not in ("str", "e") else v.text
                    values[col - 1] = value
                rows.append(values)
            result[s.attrib["name"]] = rows
    return result


def write_csv(name, rows):
    with (HERE / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    manifest = json.loads((ATTACK / "source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        assert sha(ATTACK / item["path"]) == item["sha256"], item["path"]
    cfg_path = SRC / "pwscup2026/kit/kit_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    workbook = next((ARCHIVE / "sources").glob("*.xlsx"))
    sheets = workbook_sheets(workbook)
    meta = dict(sheets["メタ"][1:])
    for key in ("scoring.low_fpr", "scoring.aia_rare_se_z", "aia.tau_time"):
        a, b = key.split(".")
        assert cfg[a][b] == meta["config." + key], key
    summary = {int(r[0]): r for r in sheets["総括"][1:]}
    mia_reference = {(int(r[0]), int(r[1])): r for r in sheets["攻撃_標的別_MIA"][1:]}
    aia_reference = {(int(r[0]), int(r[1])): r for r in sheets["攻撃_標的別_AIA"][1:]}
    assert sorted(summary) == COHORTS
    assert set(mia_reference) == set(aia_reference) == {(a, t) for a in COHORTS for t in COHORTS if a != t}
    truth_dir = ARCHIVE / "truth"
    pools = {k: read_scoring_csv(truth_dir / f"pool_ids_{k}.csv") for k in COHORTS}
    truths = {k: read_scoring_csv(truth_dir / f"aia_truth_{k}.csv") for k in COHORTS}
    floors = read_scoring_csv(truth_dir / "aia_floor.csv").set_index("team_id")["aia_floor"].to_dict()
    errors, comparisons = [], []
    aggregate, mia_rows, aia_rows, self_columns = [], [], [], []

    def compare(label, calculated, published):
        if published is None:
            assert math.isnan(calculated), (label, calculated, published)
            return
        assert math.isfinite(float(calculated)) and math.isfinite(float(published)), (label, calculated, published)
        delta = abs(float(calculated) - float(published))
        comparisons.append(delta)
        if delta > TOLERANCE:
            errors.append({"field": label, "calculated": calculated, "published": published, "absolute_error": delta})

    for attacker in COHORTS:
        sub = ARCHIVE / "submissions" / f"cohort{attacker:02d}"
        f_mia = read_scoring_csv(sub / "F_mia.csv")
        f_aia = read_scoring_csv(sub / "F_aia.csv")
        targets = set(COHORTS) - {attacker}
        assert f_mia["record_id"].is_unique
        assert set(f_mia["record_id"]) == set(pools[attacker]["record_id"])
        assert set(int(c) for c in f_mia if c != "record_id") - {attacker} == targets
        assert np.isfinite(f_mia.to_numpy(dtype=float)).all()
        assert np.isfinite(f_aia.to_numpy(dtype=float)).all()
        assert not f_aia.duplicated(["target_k", "challenge_row_id"]).any()
        assert set(f_aia["target_k"].astype(int)) - {attacker} == targets
        if str(attacker) in f_mia:
            self_columns.append(attacker)
        result = score_attack({"F_mia.csv": f_mia, "F_aia.csv": f_aia}, truth_dir, cfg, team_id=attacker)
        compare(f"aggregate.{attacker}.attack_power", result["attack_power"], summary[attacker][17])
        aggregate.append({"attacker": attacker, "team": summary[attacker][1], "mia_power": result["mia_power"], "aia_power": result["aia_power"], "attack_power": result["attack_power"], "published_attack_power": summary[attacker][17], "absolute_error": abs(result["attack_power"] - summary[attacker][17])})
        ap = pools[attacker].set_index("record_id").loc[f_mia["record_id"]]
        rare = ap["is_rare"].to_numpy(dtype=bool)
        for target in sorted(targets):
            membership = ap["pool_id"].isin(pools[target]["pool_id"]).to_numpy(dtype=int)
            mr = attack_score.score_mia(f_mia[str(target)].to_numpy(dtype=float), membership, cfg, rare_mask=rare)
            values = [int(membership.sum()), int(membership[rare].sum()), mr.breakdown["tpr_at_low_fpr"], mr.breakdown["tpr_rare"], mr.score, mr.breakdown["auc"]]
            for idx, value in enumerate(values):
                compare(f"MIA.{attacker}.{target}.{idx}", value, mia_reference[(attacker, target)][idx + 2])
            mia_rows.append(dict(zip(["attacker", "target", "member_count", "rare_member_count", "tpr_all", "tpr_rare", "score", "auc"], [attacker, target] + [None if isinstance(v, float) and math.isnan(v) else v for v in values])))
            group = f_aia.loc[f_aia["target_k"] == target]
            assert set(group["challenge_row_id"]) == set(truths[target]["challenge_row_id"])
            joined = group.merge(truths[target], on="challenge_row_id", validate="one_to_one")
            member, control = joined.loc[joined.label == "member"], joined.loc[joined.label == "control"]
            ar = attack_score.score_aia(None, member.time_hat.to_numpy(), member.onset.to_numpy(), member.time.to_numpy(), None, control.time_hat.to_numpy(), control.onset.to_numpy(), control.time.to_numpy(), cfg, member_rare_mask=member.is_rare.to_numpy(dtype=bool), control_rare_mask=control.is_rare.to_numpy(dtype=bool))
            keys = ["r_time_all", "r_time_rare", "r_time_rare_raw", "succ_time_member", "succ_time_control", "succ_time_member_rare", "succ_time_control_rare"]
            values = [ar.breakdown[k] for k in keys] + [ar.score]
            for idx, value in enumerate(values):
                compare(f"AIA.{attacker}.{target}.{idx}", value, aia_reference[(attacker, target)][idx + 2])
            corrected = float(np.clip(ar.score - floors[target], 0, 1))
            compare(f"AIA.{attacker}.{target}.corrected", corrected, result["_breakdown"]["aia_per_target"][str(target)])
            aia_rows.append(dict(zip(["attacker", "target"] + keys + ["raw_score", "phi", "corrected_score"], [attacker, target] + values + [floors[target], corrected])))
    assert self_columns == [0, 3, 5, 7, 11, 16, 25]
    source_files = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None) and Path(m.__file__).resolve().is_relative_to(SRC)})
    report = {
        "status": "passed" if not errors else "failed",
        "official_release": manifest["release"], "official_commit": manifest["official_commit"],
        "conditions": {"low_fpr": cfg["scoring"]["low_fpr"], "tau_time": cfg["aia"]["tau_time"], "aia_rare_se_z": cfg["scoring"]["aia_rare_se_z"], "phi": "published truth/aia_floor.csv, per target", "self_target": "excluded by official score_attack(team_id=attacker)", "aggregation": "22 other targets per attacker; mean MIA plus mean phi-corrected AIA"},
        "verified_source_manifest_files": len(manifest["files"]),
        "published_submissions": len(aggregate), "mia_pairs": len(mia_rows), "aia_pairs": len(aia_rows),
        "numeric_comparisons": len(comparisons), "tolerance": TOLERANCE, "max_absolute_error": max(comparisons),
        "original_self_mia_columns": self_columns, "mismatches": errors,
        "outsider": next(r for r in aggregate if r["attacker"] == 10),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": [{"path": p.relative_to(REPO).as_posix(), "sha256": sha(p)} for p in sorted((ARCHIVE / "sources").iterdir()) if p.suffix in (".zip", ".xlsx")],
        "official_loaded_dependencies": [{"path": p.relative_to(REPO).as_posix(), "sha256": sha(p)} for p in source_files + [cfg_path]],
        "limitations": ["This is a local retrospective verification of published results, not a new organizer-issued result.", "Utility U, anonymity aggregation, and final weighted ranks are not recomputed here; this verification covers attack scoring and published per-target attack details.", "No predictions were regenerated or tuned and no CodaBench submission was made."]
    }
    write_csv("attack_scores.csv", aggregate)
    write_csv("per_target_mia.csv", mia_rows)
    write_csv("per_target_aia.csv", aia_rows)
    (HERE / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ["status", "published_submissions", "mia_pairs", "aia_pairs", "numeric_comparisons", "max_absolute_error", "outsider"]}, ensure_ascii=False))
    if errors:
        raise SystemExit("Published score mismatch; see verification_report.json")


if __name__ == "__main__":
    main()
