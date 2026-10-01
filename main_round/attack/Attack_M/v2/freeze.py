"""Freeze development selections, then separately register final-check results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import attack_workspace as w
from M.v2.artifacts import (HERE, OUTPUT, dependencies, load_selection, read_json,
                            save_json, sha, timestamp, check_v1_preserved, check_snapshot)


def prepare():
    path = HERE / "selection.json"
    if path.exists():
        raise FileExistsError("The combined selection is already frozen")
    for component in ("aia", "mia"):
        holdout = OUTPUT / f"{component}-holdout"
        if holdout.exists() and any(holdout.iterdir()):
            raise ValueError("Final checking has already started before the combined selection")
    a, m = (read_json(HERE / name) for name in ("aia_selection.json", "mia_selection.json"))
    arun = read_json(OUTPUT / "aia-dev/run.json")
    mrun = read_json(OUTPUT / "mia-dev/run.json")
    stress_path = OUTPUT / "mia-stress-dev/run.json"
    if not stress_path.exists():
        raise ValueError("The additional donor-independence development check is required")
    stress = read_json(stress_path)
    if not stress["completed"] or stress["stage"] != "dev" or stress["fixed_strength"] != m["strength"]:
        raise ValueError("The donor-independence check is incomplete or tested another method")
    if stress["mia_selection_sha256"] != sha(HERE / "mia_selection.json"):
        raise ValueError("The donor-independence check used a different MIA selection")
    check_snapshot(stress["model_dependency_sha256"])
    if arun["stage"] != "dev" or mrun["stage"] != "dev" or not arun["completed"]:
        raise ValueError("Both development evaluations must complete first")
    if a["holdout_executed"] or m["holdout_inspected"] or not m["selected_before_holdout"]:
        raise ValueError("Development selections must precede final checking")
    if a["selected_variant"] != arun["selected_variant"] or m != mrun["selected"]:
        raise ValueError("Component selections do not match development records")
    record = {
        "version": "M/v2", "created_utc": timestamp(),
        "aia_variant": a["selected_variant"], "mia_strength": m["strength"],
        "mia_adjustment": m["adjustment"], "selected_before_holdout": True,
        "selection_policy": "Separate predeclared component development rules; no real target labels or final-check scores used",
        "development_aia": a["dev_evidence"], "development_mia": m["development_summary"],
        "additional_development_evidence_sha256": sha(stress_path),
        "development_tests_sha256": sha(OUTPUT / "development-tests.txt"),
        "dependency_sha256": dependencies(),
        "development_evidence_sha256": {
            f"{component}-dev/{name}": sha(OUTPUT / f"{component}-dev" / name)
            for component in ("aia", "mia") for name in ("results.csv", "summary.csv", "run.json")},
        "source_checks": w.verify_hashes(), "v1_preservation": check_v1_preserved(),
        "official_score_measured": False, "official_submission": False,
    }
    save_json(path, record)
    save_json(OUTPUT / "pre_holdout_freeze.json", record)
    print({"frozen": True, "aia_variant": record["aia_variant"], "mia_strength": record["mia_strength"]})


def complete():
    import pandas as pd
    path = OUTPUT / "final_check.json"
    if path.exists():
        raise FileExistsError("Preserve the existing final-check record")
    selection = load_selection()
    arun = read_json(OUTPUT / "aia-holdout/run.json")
    mrun = read_json(OUTPUT / "mia-holdout/run.json")
    if arun["stage"] != "holdout" or not arun["completed"] or mrun["stage"] != "holdout":
        raise ValueError("Both final-check evaluations must complete first")
    if arun["selected_variant"] != selection["aia_variant"]:
        raise ValueError("AIA checked a different selected method")
    if mrun["fixed_selection"] != read_json(HERE / "mia_selection.json"):
        raise ValueError("MIA checked a different component selection")
    a = pd.read_csv(OUTPUT / "aia-holdout/summary.csv")
    m = pd.read_csv(OUTPUT / "mia-holdout/summary.csv")
    af = pd.read_csv(OUTPUT / "aia-holdout/results.csv")
    mf = pd.read_csv(OUTPUT / "mia-holdout/results.csv")
    mname = f"median_{selection['mia_strength']:g}"
    if set(a.variant) != {"v1_reference", selection["aia_variant"]}:
        raise ValueError("Unexpected AIA variants in the final check")
    if set(m.method) != {"median_0", mname}:
        raise ValueError("Unexpected MIA variants in the final check")
    if arun["conditions"] != 189 or mrun["all_target_conditions"] != 396:
        raise ValueError("Incomplete final-check condition coverage")
    if af.groupby("variant").size().min() != 189 or mf.groupby("method").size().min() != 396:
        raise ValueError("Final-check result row counts are incomplete")
    astats = a.set_index("variant").loc[selection["aia_variant"]].to_dict()
    mstats = m.set_index("method").loc[mname].to_dict()
    record = {"completed": True, "created_utc": timestamp(),
        "selection_sha256": sha(HERE / "selection.json"),
        "selected_before_holdout": True, "post_holdout_tuning": False,
        "aia": astats, "mia": mstats,
        "component_scores_are_separate_experiments": True,
        "aia_phi_applied": False, "official_score_measured": False,
        "evidence_sha256": {f"{component}-holdout/{name}": sha(OUTPUT / f"{component}-holdout" / name)
            for component in ("aia", "mia") for name in ("results.csv", "summary.csv", "run.json", "freeze.json")},
        "source_checks": w.verify_hashes(), "v1_preservation": check_v1_preserved()}
    save_json(path, record)
    print({"completed": True, "aia_delta": astats["paired_delta_mean"],
           "mia_delta": mstats["paired_delta_mean"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "complete"), default="prepare")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else complete()
