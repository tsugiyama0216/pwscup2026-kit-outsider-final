"""Command line for the copied real-data preliminary benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mock_TS_using_prelim.common import (DATA, HERE, clean, code_hashes, new_output, read_csv,
    read_json, require, sha256, team_names, verify_bundle, write_json)
from mock_TS_using_prelim.report import write_report
from mock_TS_using_prelim.runner import expected_public_hashes, export_public, run_attack
from mock_TS_using_prelim.scoring import COHORTS, score_predictions


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.17g")


def save_score(predictions, output, attacker_id=24):
    copied = verify_bundle()
    provenance_path = Path(predictions) / "prediction_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else None
    if provenance:
        require(isinstance(provenance, dict) and {"round", "attacker_id", "public_input_sha256", "prediction_sha256"}.issubset(provenance),
                "Incomplete prediction provenance")
        require(provenance["round"] == "prelim" and provenance["attacker_id"] == attacker_id, "Prediction provenance round/team mismatch")
        require(provenance["public_input_sha256"] == expected_public_hashes(), "Predictions belong to different public inputs")
        for name, digest in provenance["prediction_sha256"].items():
            require(sha256(Path(predictions) / name) == digest, "Prediction changed after generation: " + name)
    result = score_predictions(predictions, attacker_id=attacker_id)
    result.update({"kind": "new_attack", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "copied_source_verification": copied, "source_sha256": code_hashes(),
                   "prediction_provenance": provenance,
                   "input_attribution": "verified_runner_hashes" if provenance else "external_CSV_caller_declares_preliminary_ID_mapping"})
    names = team_names()
    for row in result["by_target"]:
        row["target_name"] = names[row["target_k"]]
    output.mkdir(parents=True, exist_ok=False)
    # Keep the exact externally provided predictions with their scored result.
    for name in ("F_mia.csv", "F_aia.csv"):
        shutil.copyfile(Path(predictions) / name, output / name)
    write_csv(output / "by_target.csv", result["by_target"])
    write_json(output / "result.json", result)
    write_report(result, output)
    return result


def replay(output):
    """Regress all archived submissions against the published final workbook."""
    verified = verify_bundle()
    base = DATA / "reference/published_results"
    mapping = read_csv(base / "team_mapping.csv").set_index("cohort_id")
    mia = read_csv(base / "mia_by_target.csv").set_index(["attacker", "target"])
    aia = read_csv(base / "aia_by_target.csv").set_index(["attacker", "target"])
    mia_columns = {"mia_tpr_all": "tpr_at_low_fpr", "mia_tpr_rare": "tpr_rare", "mia_score": "t_j", "mia_auc": "auc",
                   "mia_positive_count": "n_member", "mia_rare_positive_count": "n_member_rare"}
    aia_columns = {"aia_score": "r_time", "aia_r_all": "r_time_all", "aia_r_rare": "r_time_rare",
                   "aia_r_rare_raw": "r_time_rare_unadjusted", "aia_p_m": "p_m", "aia_p_c": "p_c",
                   "aia_p_m_rare": "p_m_rare", "aia_p_c_rare": "p_c_rare"}
    differences = []

    def compare(actual, expected, label):
        if actual is None or pd.isna(actual):
            require(pd.isna(expected), "Undefined mismatch: " + label)
            return
        require(not pd.isna(expected), "Unexpected published blank: " + label)
        delta = abs(float(actual) - float(expected))
        require(delta <= 1e-10, f"Published result mismatch {label}: calculated={actual}, published={expected}")
        differences.append(delta)

    teams, summaries, details = [], [], []
    seen_mia, seen_aia = set(), set()
    for attacker in COHORTS:
        print(f"Replaying organizer submission: cohort {attacker:02d}", flush=True)
        predictions = DATA / f"reference/attack_submissions/extracted/cohort{attacker:02d}"
        result = score_predictions(predictions, attacker, historical=True)
        expected_power = float(mapping.loc[attacker, "official_attack_power"])
        compare(result["summary"]["attack_power"], expected_power, f"team {attacker} total")
        result.update(published_attack_power=expected_power,
                      attack_power_difference=result["summary"]["attack_power"] - expected_power)
        for row in result["by_target"]:
            pair = (attacker, row["target_k"])
            require(pair in mia.index, "Missing published MIA pair")
            for key, column in mia_columns.items():
                compare(row[key], mia.loc[pair, column], f"MIA {pair} {key}")
            seen_mia.add(pair)
            if row["aia_score"] is not None:
                require(pair in aia.index, "Missing published AIA pair")
                for key, column in aia_columns.items():
                    compare(row[key], aia.loc[pair, column], f"AIA {pair} {key}")
                seen_aia.add(pair)
            else:
                require(pair == (4, 23) and pair not in aia.index, "Unexpected historical omission")
            details.append({"attacker_id": attacker, "attacker_name": mapping.loc[attacker, "team_name"], **row,
                            "target_name": mapping.loc[row["target_k"], "team_name"]})
        teams.append(result)
        summaries.append({"attacker_id": attacker, "team_name": mapping.loc[attacker, "team_name"], **result["summary"],
                          "published_attack_power": expected_power, "difference": result["attack_power_difference"]})
        write_json(output / f"teams/cohort{attacker:02d}.json", result)
    require(seen_mia == set(mia.index) and seen_aia == set(aia.index), "Not all published pair results were checked")
    require(verify_bundle() == verified, "Copied sources changed during replay")
    result = {"kind": "historical_replay", "round": "preliminary", "teams": teams,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "copied_source_verification": verified, "source_sha256": code_hashes(),
              "verification": {"passed": True, "team_count": len(teams), "mia_pairs": len(seen_mia), "aia_pairs": len(seen_aia),
                               "compared_numeric_values": len(differences), "maximum_absolute_error": max(differences), "tolerance": 1e-10}}
    write_csv(output / "comparison.csv", summaries)
    write_csv(output / "all_by_target.csv", details)
    write_json(output / "result.json", result)
    write_report(result, output)
    return result


def main():
    parser = argparse.ArgumentParser(description="予備戦の実データ・公式正解を使うMIA/AIA評価")
    parser.add_argument("action", choices=["Verify", "Replay", "Export", "Attack", "Score", "Run"], nargs="?", default="Replay")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plugin", type=Path, default=HERE / "example_attack.py")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--attacker-id", type=int, default=24)
    args = parser.parse_args()
    if args.action != "Score" and args.attacker_id != 24:
        parser.error("Only Score accepts other attacker IDs; attack inputs are actual team24 data")
    if args.action == "Verify":
        print(__import__("json").dumps(verify_bundle(), ensure_ascii=False, indent=2))
        return
    if args.action == "Attack":
        parser.error("Attack requires --bundle") if args.bundle is None else None
    if args.action == "Score":
        parser.error("Score requires --predictions") if args.predictions is None else None
    output = new_output(args.output, args.action.lower())
    verified = verify_bundle()
    write_json(output / "input_verification.json", verified)
    if args.action == "Replay":
        result = replay(output)
        print("Published results reproduced: " + str(result["verification"]))
        report = output / "report.html"
    elif args.action == "Export":
        export_public(output / "public")
        print("Public attack inputs: " + str(output / "public"))
        return
    elif args.action == "Attack":
        run_attack(args.bundle, args.plugin, output / "predictions")
        print("Predictions: " + str(output / "predictions"))
        return
    elif args.action == "Score":
        result = save_score(args.predictions, output / "evaluation", args.attacker_id)
        print("Preliminary score: " + str(result["summary"]))
        report = output / "evaluation/report.html"
    else:
        require(args.attacker_id == 24, "Run supports only actual team24 inputs")
        export_public(output / "public")
        run_attack(output / "public", args.plugin, output / "predictions")
        result = save_score(output / "predictions", output / "evaluation")
        print("Preliminary score: " + str(result["summary"]))
        report = output / "evaluation/report.html"
    print("Report: " + str(report))


if __name__ == "__main__":
    main()
