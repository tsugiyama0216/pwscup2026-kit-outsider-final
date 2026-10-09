"""Score saved Rv12 variants against Rv2 on one immutable synthetic panel.

This process reads private truth only after inference has finished. It never
imports a candidate model or produces predictions, selects a model, or changes
the frozen scoring formula. Existing reports and predictions are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
MOCK = ROOT / "main_round/mock_test"
sys.path.insert(0, str(MOCK))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from rank_validation.panel import check_panel, hashes, new_output, read, valid_name, workspace
from rank_validation.provenance import evaluation_signature, model_signature
from rank_validation.ranking_metric import compute_metrics
from rank_validation.replay import read_csv_strict
from sugiyama_v2.io import csv, save_json, sha
from sugiyama_v2.runner import per_target_diagnostics
from sugiyama_v2.scoring import TARGET_IDS, _load_truth, score_world, validate_predictions


def prediction_evidence(directory, model, state, baseline=None):
    """Bind saved predictions to this exact public input bundle."""
    directory = Path(directory).resolve()
    expected = ("F_mia.csv", "F_aia.csv", "diagnostics.json")
    for name in expected:
        path = directory / name
        if not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError("Missing or escaping prediction file: " + name)
    diagnostic = read(directory / "diagnostics.json")
    if diagnostic.get("model_id") != model:
        raise ValueError("Prediction model identity differs from directory name")
    if diagnostic.get("public_input_sha256") != state["public_sha256"]:
        raise ValueError("Predictions belong to a different public world")
    if diagnostic.get("input_files_unchanged") is not True:
        raise ValueError("Missing input preservation evidence")
    output_hashes = {name: sha(directory / name) for name in expected[:2]}
    if diagnostic.get("output_sha256") != output_hashes:
        raise ValueError("Prediction bytes changed after inference")
    identity = None
    if model == "Rv2":
        identity = model_signature(diagnostic)
        if identity["model_id"] != "Rv2" or identity["kind"] not in {"frozen_batch", "frozen_individual"}:
            raise ValueError("Reference is not a frozen Rv2 inference")
    elif baseline is not None:
        # Rv12 is a derived model; bind both reference files and its diagnostic.
        if diagnostic.get("baseline_files_unchanged") is not True:
            raise ValueError("Missing Rv2 preservation evidence")
        if diagnostic.get("baseline_output_sha256") != baseline["output_sha256"]:
            raise ValueError("Candidate used different Rv2 predictions")
        if diagnostic.get("baseline_diagnostics_sha256") != baseline["diagnostics_sha256"]:
            raise ValueError("Candidate used different Rv2 diagnostics")
    return {"directory": str(directory), "output_sha256": output_hashes,
            "diagnostics_sha256": sha(directory / "diagnostics.json"),
            "model_signature": identity, "diagnostic": diagnostic}


def raw_hits(f_aia, truths, cfg):
    """Raw hit counts preserve standard and rare member/control behavior."""
    rows = []
    tau = float(cfg["aia"]["tau_time"])
    for target in TARGET_IDS:
        prediction = f_aia[f_aia.target_k == target]
        truth = truths[target].set_index("challenge_row_id").loc[prediction.challenge_row_id]
        hit = np.abs(prediction.time_hat.to_numpy(float) - truth.time.to_numpy(float)) <= tau
        rare = truth.is_rare.to_numpy(bool)
        member = truth.label.eq("member").to_numpy()
        row = {"target_k": target}
        for name, mask in (("member", member), ("control", ~member),
                           ("member_rare", member & rare), ("control_rare", ~member & rare)):
            row[name + "_count"] = int(mask.sum())
            row[name + "_hits"] = int(hit[mask].sum())
            row["p_" + name] = float(hit[mask].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def score_saved(directory, model, panel, state, public, truths, baseline=None):
    evidence = prediction_evidence(directory, model, state, baseline)
    before = hashes(directory)
    f_mia = read_csv_strict(Path(directory) / "F_mia.csv")
    f_aia = read_csv_strict(Path(directory) / "F_aia.csv")
    validate_predictions(f_mia, f_aia, public)
    with threadpool_limits(limits=1):
        result = score_world(f_mia, f_aia, public, panel / "scoring_truth", state["config"])
        detail = per_target_diagnostics(f_mia, f_aia, public, panel / "scoring_truth",
                                        state["config"], state["case_metadata"], result)
    detail = detail.merge(raw_hits(f_aia, truths, state["config"]), on="target_k", validate="one_to_one")
    if not np.allclose(detail.p_member, detail.aia_member_success, atol=1e-15, rtol=0):
        raise ValueError("Raw member hits disagree with official scorer")
    if not np.allclose(detail.p_control, detail.aia_control_success, atol=1e-15, rtol=0):
        raise ValueError("Raw control hits disagree with official scorer")
    detail["stable_index"] = detail.mia_all + detail.aia_power
    detail["model"] = model
    metrics = compute_metrics(detail.to_dict("records"))
    if not np.isclose(metrics["official_attack_power"], result["leaderboard"]["score_1"], atol=1e-12, rtol=0):
        raise ValueError("Per-target aggregation disagrees with official score")
    if hashes(directory) != before:
        raise ValueError("Prediction files changed while scoring")
    return result, detail, metrics, evidence


def evaluate(panel_path, predictions_path, output_path, *, role="development"):
    if role not in {"development", "confirmation"}:
        raise ValueError("Role must be development or confirmation")
    panel, state, public = check_panel(panel_path)
    if role == "confirmation" and state["case_metadata"]["seed"] in (2026100961, 2026100962):
        raise ValueError("Previously inspected v5 cases are Rv12 development data")
    if role == "confirmation" and state["case_metadata"].get("reserved_seed_role") != "confirmation":
        raise ValueError("Confirmation requires a reserved confirmation case")
    originals = workspace.verify_hashes()
    predictions = Path(predictions_path).resolve()
    if not predictions.is_dir():
        raise ValueError("Missing variants directory")
    output = Path(output_path).resolve()
    if output.is_relative_to(panel) or output.is_relative_to(predictions):
        raise ValueError("Report must be separate from panel and predictions")
    names = []
    for child in sorted(predictions.iterdir()):
        if not child.is_dir():
            continue
        valid_name(child.name)
        if child.name == "Rv2":
            raise ValueError("Rv2 baseline is read from panel/predictions/Rv2")
        if not child.resolve().is_relative_to(predictions):
            raise ValueError("Variant directory escapes predictions root")
        names.append(child.name)
    if not names or len({name.casefold() for name in names}) != len(names):
        raise ValueError("Need nonempty distinct candidate names")
    signature = evaluation_signature(state["config"])
    harness_sha = sha(__file__)
    _, truths, _ = _load_truth(public, panel / "scoring_truth")
    baseline_dir = panel / "predictions/Rv2"
    base_result, base_frame, base_metrics, base_evidence = score_saved(
        baseline_dir, "Rv2", panel, state, public, truths)
    results = {"Rv2": base_result}
    evidence = {"Rv2": base_evidence}
    frames = [base_frame]
    summaries = []
    all_metrics = {"Rv2": base_metrics}
    for name in names:
        result, detail, metrics, record = score_saved(
            predictions / name, name, panel, state, public, truths, base_evidence)
        results[name], evidence[name], all_metrics[name] = result, record, metrics
        frames.append(detail)
        print(json.dumps({"model": name, "stable_index": metrics["stable_index"],
                          "delta_vs_Rv2": metrics["stable_index"] - base_metrics["stable_index"]}), flush=True)
    for frame in frames:
        name = frame.model.iloc[0]
        aligned = frame.set_index("target_k").sort_index()
        reference = base_frame.set_index("target_k").sort_index()
        differences = aligned.stable_index - reference.stable_index
        for field in ("stable_index", "mia_all", "mia_power", "aia_power", "attack_power", "p_member", "p_control"):
            frame["delta_" + field + "_vs_Rv2"] = frame.target_k.map(aligned[field] - reference[field])
        for suffix in ("member", "control", "member_rare", "control_rare"):
            difference = aligned[suffix + "_hits"] - reference[suffix + "_hits"]
            frame["delta_" + suffix + "_hits_vs_Rv2"] = frame.target_k.map(difference)
        omissions = [float(differences.drop(k).mean()) for k in TARGET_IDS]
        metrics = all_metrics[name]
        summaries.append({"model": name, **metrics,
            "delta_vs_Rv2": metrics["stable_index"] - base_metrics["stable_index"],
            "official_delta_vs_Rv2": metrics["official_attack_power"] - base_metrics["official_attack_power"],
            "mia_all_delta_vs_Rv2": metrics["mean_mia_all"] - base_metrics["mean_mia_all"],
            "aia_delta_vs_Rv2": metrics["mean_aia_power"] - base_metrics["mean_aia_power"],
            "mean_member_hit_rate": float(frame.p_member.mean()),
            "mean_control_hit_rate": float(frame.p_control.mean()),
            "mean_rare_member_hit_rate": float(frame.p_member_rare.mean()),
            "mean_rare_control_hit_rate": float(frame.p_control_rare.mean()),
            "member_hits": int(frame.member_hits.sum()), "control_hits": int(frame.control_hits.sum()),
            "member_hit_delta": int(frame.delta_member_hits_vs_Rv2.sum()),
            "control_hit_delta": int(frame.delta_control_hits_vs_Rv2.sum()),
            "target_wins": int((differences > 1e-12).sum()),
            "target_ties": int((differences.abs() <= 1e-12).sum()),
            "target_losses": int((differences < -1e-12).sum()),
            "leave_one_target_out_delta_min": min(omissions),
            "leave_one_target_out_delta_max": max(omissions)})
    check_panel(panel)
    if evaluation_signature(state["config"]) != signature or sha(__file__) != harness_sha:
        raise ValueError("Evaluation code changed during scoring")
    # Revalidate every prediction at the end, including reference-derived bindings.
    for name, record in evidence.items():
        current = prediction_evidence(Path(record["directory"]), name, state,
                                      None if name == "Rv2" else base_evidence)
        if current != record:
            raise ValueError("Prediction evidence changed during report construction")
    output = new_output(output)
    summary = pd.DataFrame(summaries).sort_values("stable_index", ascending=False, kind="stable")
    csv(output / "summary.csv", summary.drop(columns="target_ids"))
    csv(output / "by_target.csv", pd.concat(frames, ignore_index=True))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "panel": str(panel), "seed": state["case_metadata"]["seed"],
        "evaluation_role": role,
        "source_reserved_role": state["case_metadata"].get("reserved_seed_role"),
        "independent_unseen_people": False,
        "case_manifest_sha256": state["case_snapshot"]["manifest"],
        "public_sha256": state["public_sha256"], "truth_sha256": state["truth_sha256"],
        "config": state["config"], "evaluation_signature": signature,
        "harness_sha256": harness_sha, "prediction_evidence": evidence,
        "official_input_verification": originals,
        "summary": summary.to_dict("records"), "official_results": results,
        "metric": "equal-target mean(overall MIA + floor-corrected AIA), stable-index-v1",
        "output_sha256": {name: sha(output / name) for name in ("summary.csv", "by_target.csv")},
        "official_score_prediction": False, "model_selection_performed": False,
        "inference_performed": False, "all_source_files_preserved": True,
        "limitations": [("Development evidence; do not present it as independent confirmation." if role == "development"
                         else "Confirmation scenario on the same finite donor population; independence requires model selection to have been frozen before inspecting its attack results."),
            "Hidden official truth and floors are unavailable; this is a synthetic paired comparison.",
            "Target deletions are sensitivity checks, not independent samples or confidence intervals.",
            "Lower control hit rate is useful only together with retained member success and corrected AIA."]}
    save_json(output / "result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    args = parser.parse_args(argv)
    report = evaluate(args.panel, args.predictions, args.output, role=args.role)
    print(json.dumps({"output": str(args.output.resolve()), "models": len(report["summary"]),
                      "official_score_prediction": False}), flush=True)


if __name__ == "__main__":
    main()
