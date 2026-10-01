"""Registered aggregate-only joint-world evaluation for M/v2 MIA."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from attack_workspace import verify_hashes, load_runtime
from M.v2.mia import ADJUSTMENT, raw_scores, adjust_scores
from M.v2.shadow_joint import build_world
from local_eval.data import QI


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_hashes():
    return {"mia.py": digest(HERE / "mia.py"),
            "shadow_joint.py": digest(HERE / "shadow_joint.py"),
            "evaluate_mia.py": digest(HERE / "evaluate_mia.py"),
            "mia_protocol.json": digest(HERE / "mia_protocol.json"),
            "base_model.py": digest(ROOT / "M/v1/model.py"),
            "base_actual.py": digest(ROOT / "M/v1/actual.py"),
            "generic_data.py": digest(ROOT / "local_eval/data.py")}


def metric(score, truth, rare, low_fpr, tpr_function):
    all_tpr, auc = tpr_function(score, truth.astype(int), low_fpr)
    rare_tpr = None
    if truth[rare].sum() > 0:
        rare_tpr, _ = tpr_function(score[rare], truth[rare].astype(int), low_fpr)
    return {"mia": max(all_tpr, rare_tpr if rare_tpr is not None else 0.),
            "mia_all": all_tpr, "mia_rare": rare_tpr, "auc": auc,
            "rare_max_uplift": max(all_tpr, rare_tpr if rare_tpr is not None else 0.) - all_tpr}


def run(stage, output):
    started = time.monotonic()
    output = Path(output).resolve()
    allowed_root = (ROOT / "M/local-output/v2").resolve()
    if not output.is_relative_to(allowed_root) or output == allowed_root:
        raise ValueError("MIA evaluation output must be a subdirectory of M/local-output/v2")
    protocol = json.loads((HERE / "mia_protocol.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be empty; existing results are preserved")
    selection_path = HERE / "mia_selection.json"
    if stage == "dev" and selection_path.exists():
        raise ValueError("MIA selection already exists; development cannot silently reselect")
    before = verify_hashes()
    _, _, _, cfg, versions = load_runtime()
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    code_hashes = frozen_hashes()
    if stage == "holdout":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if code_hashes != selection["frozen_hashes"]:
            raise ValueError("MIA code/protocol differs from the fixed development selection")
        strengths = sorted(set([0., float(selection["strength"])]))
    else:
        strengths = list(map(float, protocol["strengths"]))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", protocol)
    freeze_record = {"stage": stage, "frozen_hashes": code_hashes}
    if stage == "holdout":
        freeze_record.update({"selected_strength": float(selection["strength"]),
                              "mia_selection_sha256": digest(selection_path)})
    write_json(output / "freeze.json", freeze_record)
    originals = ROOT.parent / "anonymization/team10/participant_data"
    source = pd.read_csv(originals / "A_bg.csv", float_precision="round_trip")
    background = pd.read_csv(originals / "B.csv", float_precision="round_trip")
    schema = json.loads((originals / "schema.json").read_text(encoding="utf-8-sig"))
    source_qi = set(source[QI].itertuples(index=False, name=None))
    bg_qi = set(background[QI].itertuples(index=False, name=None))
    if source_qi & bg_qi:
        raise ValueError("Source/reference QI overlap would compromise the independent-reference protocol")
    seeds = protocol["development_world_seeds" if stage == "dev" else "holdout_world_seeds"]
    split = "dev" if stage == "dev" else "holdout"
    totals = protocol["visible_rare_totals"]
    layouts = protocol["layouts"]
    total_worlds = len(seeds) * len(totals) * len(layouts)
    rows, worlds = [], []
    for seed in seeds:
        for visible_rare_total in totals:
            for layout in layouts:
                world = build_world(source, background, schema, split=split, seed=seed,
                                    visible_rare_total=visible_rare_total, layout=layout)
                raw = np.column_stack([raw_scores(c=c, background=world.background,
                                                candidates=world.candidates) for c in world.releases])
                methods = [(f"median_{strength:g}", strength, True) for strength in strengths]
                if stage == "dev":
                    methods.append(("baseline_unmasked_diagnostic", 0., False))
                for method, strength, masked in methods:
                    score = adjust_scores(raw, world.private_mask if masked else None, strength)
                    for target in range(22):
                        rcount = int(world.truth[world.rare, target].sum())
                        row = {"stage": stage, "seed": seed, "visible_rare_total": visible_rare_total,
                               "layout": layout, "target_slot": target, "style": world.metadata["styles"][target],
                               "method": method, "strength": strength, "private_mask_applied": masked,
                               "mia_members": int(world.truth[:, target].sum()), "mia_rare_members": rcount}
                        row.update(metric(score[:, target], world.truth[:, target], world.rare,
                                          float(cfg["scoring"]["low_fpr"]), _tpr_at_fpr))
                        rows.append(row)
                worlds.append(world.metadata)
                write_json(output / "progress.json", {"stage": stage, "worlds_completed": len(worlds),
                    "worlds_total": total_worlds, "target_conditions_completed": len(worlds) * 22,
                    "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"stage": stage, "worlds_completed": len(worlds), "worlds_total": total_worlds}), flush=True)
    frame = pd.DataFrame(rows)
    primary = frame[~frame["style"].isin(protocol["excluded_sanity_styles_from_selection"])].copy()
    group_keys = ["seed", "visible_rare_total", "layout", "method", "strength", "private_mask_applied"]
    world_scores = primary.groupby(group_keys, dropna=False)[["mia", "mia_all", "rare_max_uplift"]].mean().reset_index()
    baseline = world_scores[world_scores.method.eq("median_0")].set_index(["seed", "visible_rare_total", "layout"]).mia
    world_scores["paired_delta"] = [row.mia - baseline.loc[(row.seed, row.visible_rare_total, row.layout)]
                                    for row in world_scores.itertuples(index=False)]
    summary = []
    for method, group in world_scores.groupby("method"):
        deltas = group.paired_delta.to_numpy(float)
        summary.append({"method": method, "strength": float(group.strength.iloc[0]),
            "private_mask_applied": bool(group.private_mask_applied.iloc[0]),
            "worlds": len(group), "mean_mia": float(group.mia.mean()),
            "mean_all_population_tpr": float(group.mia_all.mean()),
            "mean_rare_max_uplift": float(group.rare_max_uplift.mean()),
            "paired_delta_mean": float(deltas.mean()), "paired_delta_min": float(deltas.min()),
            "wins": int(np.sum(deltas > 1e-12)), "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
            "losses": int(np.sum(deltas < -1e-12))})
    frame.to_csv(output / "results.csv", index=False, lineterminator="\n")
    world_scores.to_csv(output / "world_scores.csv", index=False, lineterminator="\n")
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, lineterminator="\n")
    primary.groupby(["method", "mia_rare_members"])[["mia", "mia_all", "rare_max_uplift"]].agg(["mean", "count"]).to_csv(output / "by_rare_members.csv", lineterminator="\n")
    primary.groupby(["method", "style"])[["mia", "mia_all", "rare_max_uplift"]].mean().to_csv(output / "by_style.csv", lineterminator="\n")
    write_json(output / "world_metadata.json", worlds)
    after = verify_hashes()
    if before != after or frozen_hashes() != code_hashes:
        raise ValueError("Inputs or registered MIA code changed during evaluation")
    report = {"stage": stage, "primary_worlds": total_worlds, "all_target_conditions": total_worlds * 22,
              "summary": summary, "input_verification_before": before, "input_verification_after": after,
              "source_rows": len(source), "reference_rows": len(background), "source_reference_qi_overlap": 0,
              "frozen_hashes": code_hashes, "official_runtime_versions": versions,
              "elapsed_seconds": time.monotonic() - started, "official_submission": False,
              "limitations": protocol["limitations"]}
    if stage == "dev":
        eligible = [row for row in summary if row["private_mask_applied"]]
        best_value = max(row["mean_mia"] for row in eligible)
        selected = min((row for row in eligible if row["mean_mia"] >= best_value - 1e-12), key=lambda row: row["strength"])
        selected_record = {"adjustment": ADJUSTMENT, "strength": selected["strength"],
            "selected_method": selected["method"], "selection_stage": "dev",
            "selected_before_holdout": True, "holdout_inspected": False,
            "model_sha256": code_hashes["mia.py"], "base_model_sha256": code_hashes["base_model.py"],
            "frozen_hashes": code_hashes, "selection_rule": protocol["selection_rule"],
            "development_summary": selected, "development_results_sha256": digest(output / "results.csv"),
            "official_submission": False}
        write_json(selection_path, selected_record)
        report["selected"] = selected_record
    else:
        report["fixed_selection"] = selection
    write_json(output / "run.json", report)
    print(json.dumps({"stage": stage, "summary": summary, "selected_strength": report.get("selected", {}).get("strength")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.stage, args.output.resolve())


if __name__ == "__main__":
    main()
