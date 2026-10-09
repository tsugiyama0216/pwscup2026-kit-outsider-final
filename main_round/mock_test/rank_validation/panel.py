"""Run frozen attacks on the same saved joint world with a shared AIA floor.

The worker receives only a copied public bundle. Private truth and correction
values are consumed here after predictions have been saved, never by adapters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
MOCK = ROOT / "main_round/mock_test"
OFFICIAL = ROOT / "main_round/attack/official-kit/starter/src"
sys.path.insert(0, str(OFFICIAL))
import sugiyama_v2
import attack_workspace as workspace
from sugiyama_v2.io import load_world, sha, csv, save_json, snapshot
from sugiyama_v2.scoring import estimate_floors, score_world, validate_predictions
from sugiyama_v2.runner import per_target_diagnostics
from .provenance import evaluation_signature, model_signature, DERIVED_MODELS
from .series_binding import validate_baseline_binding

import pandas as pd
from threadpoolctl import threadpool_limits

MODELS = ("Rv1", "Rv2", "pl1", "Tv105")


def hashes(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_output(path):
    path = Path(path).resolve()
    allowed = (MOCK / "local-output").resolve()
    if path == allowed or not path.is_relative_to(allowed) or path.exists():
        raise ValueError("Use a new path inside mock_test/local-output")
    path.mkdir(parents=True)
    return path


def prepare(case, output, repetitions=1000):
    started = time.monotonic()
    case = Path(case).resolve()
    public, private, manifest, before = load_world(case)
    originals = workspace.verify_hashes()
    _, _, _, cfg, versions = workspace.load_runtime()
    output = new_output(output)
    shutil.copytree(case / "public", output / "public")
    save_json(output / "public/cfg.json", cfg)
    truth = output / "scoring_truth"
    shutil.copytree(private, truth)
    floor_seed = int(manifest["metadata"]["seed"]) + 1000003
    print(f"Estimate shared floors once: 22 targets x {repetitions}", flush=True)
    with threadpool_limits(limits=1):
        floors, metadata = estimate_floors(public, truth, cfg, floor_seed, repetitions)
    csv(truth / "aia_floor.csv", floors)
    save_json(truth / "aia_floor_metadata.json", metadata)
    if snapshot(case)[1] != before:
        raise ValueError("Case changed during floor estimation")
    state = {"case": str(case), "case_snapshot": before, "case_metadata": manifest["metadata"],
             "config": cfg, "versions": versions, "original_verification": originals,
             "public_sha256": hashes(output / "public"), "truth_sha256": hashes(truth),
             "floor_repetitions": repetitions, "floor_seed": floor_seed,
             "elapsed_seconds": time.monotonic() - started,
             "model_selection_performed": False, "official_score_prediction": False}
    save_json(output / "panel.json", state)
    print(json.dumps({"panel": str(output), "seconds": state["elapsed_seconds"]}), flush=True)
    return state


def validate_case_binding(state, manifest):
    """Bind prepared copies and seed metadata to the original immutable case."""
    if state["case_metadata"] != manifest["metadata"]:
        raise ValueError("Panel metadata differs from its saved source case")
    expected_public = {name.removeprefix("public/"): digest
                       for name, digest in manifest["files_sha256"].items() if name.startswith("public/")}
    expected_truth = {name.removeprefix("private/"): digest
                      for name, digest in manifest["files_sha256"].items() if name.startswith("private/")}
    public = state["public_sha256"]
    truth = state["truth_sha256"]
    if (set(public) != set(expected_public) | {"cfg.json"}
            or any(public.get(name) != digest for name, digest in expected_public.items())):
        raise ValueError("Public prediction inputs differ from the original case")
    if (set(truth) != set(expected_truth) | {"aia_floor.csv", "aia_floor_metadata.json"}
            or any(truth.get(name) != digest for name, digest in expected_truth.items())):
        raise ValueError("Scoring truth differs from the original case")


def check_panel(panel):
    panel = Path(panel).resolve()
    allowed = (MOCK / "local-output").resolve()
    if panel == allowed or not panel.is_relative_to(allowed):
        raise ValueError("A panel must be inside mock_test/local-output")
    state = read(panel / "panel.json")
    if state["config"] != read(panel / "public/cfg.json"):
        raise ValueError("Panel scoring config differs from the public prediction config")
    if hashes(panel / "public") != state["public_sha256"] or hashes(panel / "scoring_truth") != state["truth_sha256"]:
        raise ValueError("Public inputs or shared scoring truth changed")
    public, _, manifest, snapshot_now = load_world(state["case"])
    if snapshot_now != state["case_snapshot"]:
        raise ValueError("Original saved case changed")
    validate_case_binding(state, manifest)
    floor_meta = read(panel / "scoring_truth/aia_floor_metadata.json")
    if (type(state["floor_repetitions"]) is not int or state["floor_repetitions"] <= 0
            or state["floor_seed"] != int(manifest["metadata"]["seed"]) + 1000003
            or floor_meta["seed"] != state["floor_seed"]
            or floor_meta["repetitions"] != state["floor_repetitions"]):
        raise ValueError("Shared floor metadata differs from the prepared case")
    return panel, state, public


def valid_name(model):
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", model):
        raise ValueError("Use a short model name containing letters, digits, hyphens or underscores")
    if model.upper() in {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)), *("LPT" + str(i) for i in range(1, 10))}:
        raise ValueError("Reserved Windows file name")
    return model


def predict(panel, model, plugin=None):
    valid_name(model)
    if plugin is None and model not in MODELS:
        raise ValueError("A new candidate requires its public-only plugin")
    if plugin is not None and model in set(MODELS) | DERIVED_MODELS:
        raise ValueError("Do not replace a frozen reference model with a plugin")
    panel, state, _ = check_panel(panel)
    output = panel / "predictions" / model
    if output.exists():
        raise ValueError("Existing predictions must be preserved")
    log_dir = panel / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / (model + ".log")
    if log.exists():
        raise ValueError("Existing worker log must be preserved")
    command = [sys.executable, "-B", "-m", "rank_validation.model_worker"]
    command += ["--model", model] if plugin is None else ["--plugin", str(Path(plugin).resolve()), "--name", model]
    command += ["--public", str(panel / "public"), "--output", str(output)]
    print(f"Predicting {model}; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        run = subprocess.run(command, cwd=MOCK, stdout=stream, stderr=subprocess.STDOUT)
    check_panel(panel)
    if run.returncode:
        raise RuntimeError(f"Model {model} failed; inspect {log}")
    print(f"Predictions saved: {model}", flush=True)
    return score(panel, model)


def predict_batch(panel):
    panel, _, _ = check_panel(panel)
    output = panel / "batch_predictions"
    destination = panel / "predictions"
    if output.exists() or any((destination / model).exists() for model in ("Rv1", "Rv2", "Tv105")):
        raise ValueError("Existing batch/reference predictions must be preserved")
    log_dir = panel / "logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / "batch.log"
    if log.exists():
        raise ValueError("Existing batch log must be preserved")
    command = [sys.executable, "-B", "-m", "rank_validation.batch_worker",
               "--public", str(panel / "public"), "--output", str(output)]
    print(f"Predicting frozen Rv1/Rv2/Tv105 components; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        run = subprocess.run(command, cwd=MOCK, stdout=stream, stderr=subprocess.STDOUT)
    check_panel(panel)
    if run.returncode:
        raise RuntimeError(f"Batch prediction failed; inspect {log}")
    destination.mkdir(exist_ok=True)
    for model in ("Rv1", "Rv2", "Tv105"):
        source = (output / model).resolve()
        target = (destination / model).resolve()
        if not source.is_relative_to(panel) or not target.is_relative_to(panel):
            raise ValueError("Batch prediction path escapes the saved panel")
        shutil.copytree(source, target)
    for model in ("Rv1", "Rv2", "Tv105"):
        score(panel, model)


def predict_series(panel):
    """Reuse only the verified Rv2 inference from this same public world."""
    panel, _, _ = check_panel(panel)
    output = panel / "series_predictions"
    destination = panel / "predictions"
    models = tuple(f"Rv{i}" for i in range(6, 12))
    if output.exists() or any((destination / m).exists() or (panel / "scores" / m).exists() for m in models):
        raise ValueError("Existing series predictions and scores must be preserved")
    baseline = destination / "Rv2"
    if not (baseline / "diagnostics.json").is_file():
        raise ValueError("Run the frozen Rv2 reference on this panel first")
    logs = panel / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / "rv6_series.log"
    if log.exists():
        raise ValueError("Existing series log must be preserved")
    command = [sys.executable, "-B", "-m", "rank_validation.series_worker", "--public", str(panel / "public"),
               "--baseline", str(baseline), "--output", str(output)]
    print(f"Predicting all six frozen Rv6--Rv11 models; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        run = subprocess.run(command, cwd=MOCK, stdout=stream, stderr=subprocess.STDOUT)
    check_panel(panel)
    if run.returncode:
        raise RuntimeError(f"Series prediction failed; inspect {log}")
    for model in models:
        source, target = (output / model).resolve(), (destination / model).resolve()
        if not source.is_relative_to(panel) or not target.is_relative_to(panel):
            raise ValueError("Series output escapes its panel")
        shutil.copytree(source, target)
    for model in models:
        score(panel, model)


def score(panel, model):
    valid_name(model)
    panel, state, public = check_panel(panel)
    output = panel / "scores" / model
    if output.exists():
        raise ValueError("Existing scores must be preserved")
    directory = panel / "predictions" / model
    diagnostic = read(directory / "diagnostics.json")
    if diagnostic["public_input_sha256"] != state["public_sha256"]:
        raise ValueError("Predictions were produced from different public inputs")
    if diagnostic["model_id"] != model or not diagnostic.get("input_files_unchanged"):
        raise ValueError("Prediction identity or input preservation check failed")
    for name in ("F_mia.csv", "F_aia.csv"):
        if sha(directory / name) != diagnostic["output_sha256"][name]:
            raise ValueError("Prediction bytes changed after inference")
    prediction_hashes = hashes(directory)
    evaluation = evaluation_signature(state["config"])
    identity = model_signature(diagnostic)
    if identity["kind"] == "derived_series_batch":
        validate_baseline_binding(panel / "predictions" / "Rv2", diagnostic)
    f_mia = pd.read_csv(directory / "F_mia.csv", float_precision="round_trip")
    f_aia = pd.read_csv(directory / "F_aia.csv", float_precision="round_trip")
    validate_predictions(f_mia, f_aia, public)
    with threadpool_limits(limits=1):
        result = score_world(f_mia, f_aia, public, panel / "scoring_truth", state["config"])
        per_target = per_target_diagnostics(f_mia, f_aia, public, panel / "scoring_truth", state["config"], state["case_metadata"], result)
    check_panel(panel)
    if identity["kind"] == "derived_series_batch":
        validate_baseline_binding(panel / "predictions" / "Rv2", diagnostic)
    if evaluation_signature(state["config"]) != evaluation or hashes(directory) != prediction_hashes:
        raise ValueError("Scoring implementation or saved predictions changed during scoring")
    output.mkdir(parents=True)
    result.update({"model_id": model, "panel": str(panel), "case_sha256": state["case_snapshot"]["manifest"],
                   "prediction_sha256": prediction_hashes, "public_sha256": state["public_sha256"],
                   "truth_sha256": state["truth_sha256"], "runtime_versions": state["versions"],
                   "evaluation_signature": evaluation, "model_signature": identity})
    csv(output / "by_target.csv", per_target)
    result["by_target_sha256"] = sha(output / "by_target.csv")
    save_json(output / "result.json", result)
    print(json.dumps({"model": model, "score": result["leaderboard"], "output": str(output)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "predict", "batch", "series", "score"))
    parser.add_argument("--case", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--floor-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.case is None or args.output is None:
            parser.error("prepare requires --case and --output")
        prepare(args.case, args.output, args.floor_repetitions)
    elif args.action in ("batch", "series"):
        if args.panel is None or args.model or args.plugin:
            parser.error("batch requires only --panel")
        (predict_batch if args.action == "batch" else predict_series)(args.panel)
    else:
        if args.panel is None or args.model is None:
            parser.error("predict/score requires --panel and --model")
        if args.action == "predict":
            predict(args.panel, args.model, args.plugin)
        else:
            if args.plugin:
                parser.error("score does not accept --plugin")
            score(args.panel, args.model)


if __name__ == "__main__":
    main()
