"""Run trusted local attacks on public mock inputs and score separate truth."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import attack_workspace as workspace
from .io import (csv, load_world, output_path, prediction_arguments, save_json,
                 sha, snapshot, source_manifest)
from .scoring import (TARGET_IDS, _aia_result, _load_truth, estimate_floors,
                      score_world, validate_predictions)
from pwscup2026.scoring.attack_score import score_mia


DEFAULT_PLUGIN = workspace.ROOT / "Attack_R" / "sugiyama_adapter.py"


def load_plugin(path):
    path = Path(path).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError("Plugin must be a local Python file")
    name = "sugiyama_plugin_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if not callable(getattr(module, "predict_world", None)):
            raise ValueError("Plugin must expose predict_world(**public_tables)")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def per_target_diagnostics(f_mia, f_aia, public, truth_dir, cfg, metadata, result):
    """Detailed numbers are local synthetic diagnostics, not official feedback."""
    pools, truths, _ = _load_truth(public, truth_dir)
    attacker = pools[10].set_index("record_id").loc[f_mia.record_id]
    rare = attacker.is_rare.to_numpy(bool)
    floors = pd.read_csv(Path(truth_dir) / "aia_floor.csv").set_index("team_id").aia_floor
    rows = []
    for k in TARGET_IDS:
        member = np.isin(attacker.pool_id.to_numpy(), pools[k].pool_id.to_numpy())
        mia = score_mia(f_mia[str(k)].to_numpy(), member, cfg, rare_mask=rare)
        prediction = f_aia[f_aia.target_k == k]
        truth = truths[k].set_index("challenge_row_id").loc[prediction.challenge_row_id].reset_index()
        aia = _aia_result(prediction.time_hat.to_numpy(), truth, cfg)
        rows.append({"target_k": k, "style": metadata["target_styles"][str(k)],
            "mia_power": mia.score, "mia_all": mia.breakdown["tpr_at_low_fpr"],
            "mia_rare": mia.breakdown["tpr_rare"],
            "mia_positive": int(member.sum()), "mia_rare_positive": int((member & rare).sum()),
            "aia_before_floor": aia.score,
            "aia_power": result["_breakdown"]["aia_per_target"][str(k)],
            "aia_all": aia.breakdown["r_time_all"], "aia_rare": aia.breakdown["r_time_rare"],
            "aia_member_success": aia.breakdown["succ_time_member"],
            "aia_control_success": aia.breakdown["succ_time_control"], "aia_floor": floors.loc[k]})
    frame = pd.DataFrame(rows)
    frame["attack_power"] = frame.mia_power + frame.aia_power
    return frame


def evaluate(case, plugin, destination, *, floor_seed=None, floor_repetitions=1000):
    started = perf_counter()
    case, plugin = Path(case).resolve(), Path(plugin).resolve()
    destination = output_path(destination)
    public, private, manifest, original = load_world(case)
    if destination.is_relative_to(case) or case.is_relative_to(destination):
        raise ValueError("Evaluation output must be separate from its saved case")
    if floor_repetitions < 2:
        raise ValueError("At least two floor repetitions are required; standard is 1000")
    if floor_seed is not None and (not isinstance(floor_seed, int) or floor_seed < 0):
        raise ValueError("Floor seed must be a nonnegative integer")
    sources = workspace.verify_hashes()
    _, _, _, cfg, versions = workspace.load_runtime()
    before_source = source_manifest()
    before_plugin = sha(plugin)
    module = load_plugin(plugin)
    destination.mkdir(parents=True, exist_ok=True)
    print("Predicting all 22 targets from public inputs only", flush=True)
    with threadpool_limits(limits=1):
        predictions = module.predict_world(**prediction_arguments(public, cfg))
    if not isinstance(predictions, dict) or not {"F_mia", "F_aia"}.issubset(predictions):
        raise ValueError("Plugin must return F_mia and F_aia DataFrames")
    f_mia, f_aia = predictions["F_mia"], predictions["F_aia"]
    validate_predictions(f_mia, f_aia, public)
    csv(destination / "F_mia.csv", f_mia)
    csv(destination / "F_aia.csv", f_aia)
    # Score the bytes actually saved, including their ID ordering.
    f_mia = pd.read_csv(destination / "F_mia.csv", float_precision="round_trip")
    f_aia = pd.read_csv(destination / "F_aia.csv", float_precision="round_trip")
    if snapshot(case)[1] != original or sha(plugin) != before_plugin:
        raise ValueError("Input case or plugin changed during prediction")
    truth = destination / "scoring_truth"
    truth.mkdir()
    truth_names = ["pool_ids_10.csv", "case_metadata.json"]
    truth_names += [f"pool_ids_{k}.csv" for k in TARGET_IDS]
    truth_names += [f"aia_truth_{k}.csv" for k in TARGET_IDS]
    for name in truth_names:
        shutil.copyfile(private / name, truth / name)
    if floor_seed is None:
        floor_seed = int(manifest["metadata"]["seed"]) + 1000003
    print(f"Estimating AIA mock floors: 22 targets x {floor_repetitions} draws", flush=True)
    with threadpool_limits(limits=1):
        floors, floor_metadata = estimate_floors(public, truth, cfg, floor_seed, floor_repetitions)
        csv(truth / "aia_floor.csv", floors)
        save_json(truth / "aia_floor_metadata.json", floor_metadata)
        result = score_world(f_mia, f_aia, public, truth, cfg)
        diagnostics = per_target_diagnostics(f_mia, f_aia, public, truth, cfg, manifest["metadata"], result)
    csv(destination / "by_target.csv", diagnostics)
    summary = diagnostics.groupby("style", sort=True).agg(
        targets=("target_k", "count"), attack_power=("attack_power", "mean"),
        mia_power=("mia_power", "mean"), aia_power=("aia_power", "mean"),
        aia_before_floor=("aia_before_floor", "mean"), aia_floor=("aia_floor", "mean"))
    csv(destination / "by_style.csv", summary.reset_index())
    if snapshot(case)[1] != original or sha(plugin) != before_plugin or source_manifest() != before_source:
        raise ValueError("Inputs or implementation changed during evaluation")
    workspace.verify_hashes()
    result.update({"completed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": getattr(module, "MODEL_NAME", plugin.stem),
        "plugin_path": str(plugin), "plugin_sha256": before_plugin,
        "plugin_diagnostics": predictions.get("diagnostics", {}),
        "case_directory": str(case), "case_manifest_sha256": original["manifest"],
        "case_metadata": manifest["metadata"], "case_files_unchanged": True,
        "source_sha256": before_source, "original_source_verification": sources,
        "runtime_versions": versions, "config": cfg,
        "output_sha256": {p.relative_to(destination).as_posix(): sha(p)
            for p in sorted(destination.rglob("*")) if p.is_file()},
        "elapsed_seconds": perf_counter() - started,
        "limitations": [*manifest["metadata"]["limitations"],
            "The published official scorer is reused exactly; hidden official truths and floors are unavailable.",
            "All 22 targets including independent releases are included. Scores are scenario measurements, not leaderboard estimates.",
            "The plugin API withholds private fields but trusted local Python is not filesystem-sandboxed."]})
    save_json(destination / "result.json", result)
    print("Mock score: " + str(result["leaderboard"]), flush=True)
    print("Saved: " + str(destination / "result.json"), flush=True)
    return result
