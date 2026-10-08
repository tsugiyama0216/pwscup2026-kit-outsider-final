"""Generate answers in a separate process from public main-round mock tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mock_test"))
import sugiyama_v2  # Select the main-round runtime, never the prelim scorer.
from sugiyama_v2.io import csv, save_json
from sugiyama_v2.runner import load_plugin
from sugiyama_v2.scoring import TARGET_IDS, validate_predictions
import pandas as pd
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "config", "plugin", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: pd.read_csv(args.public / name, float_precision="round_trip")
    public = {
        "team_id": 10,
        "candidates": read("B.csv"), "self_labels": read("B_self.csv"),
        "background": read("A_bg.csv"), "self_challenge": read("aia_challenge_10.csv"),
        "schema": json.loads((args.public / "schema.json").read_text(encoding="utf-8")),
        "targets": {k: {"release": read(f"targets/C_{k}.csv"),
                        "challenge": read(f"targets/aia_challenge_{k}.csv")}
                    for k in TARGET_IDS},
    }
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    module = load_plugin(args.plugin)
    arguments = {key: value for key, value in public.items() if key != "team_id"}
    with threadpool_limits(limits=1):
        result = module.predict_world(**arguments, cfg=cfg, attacker_id=10)
    validation = validate_predictions(result["F_mia"], result["F_aia"], public)
    args.output.mkdir(parents=True, exist_ok=True)
    csv(args.output / "F_mia.csv", result["F_mia"])
    csv(args.output / "F_aia.csv", result["F_aia"])
    save_json(args.output / "prediction_metadata.json", {
        "model": getattr(module, "MODEL_NAME", args.plugin.stem),
        "public_arguments_only": True, "attacker_id": 10,
        "validation": validation, "diagnostics": result.get("diagnostics", {}),
        "filesystem_sandbox": False,
    })


if __name__ == "__main__":
    main()
