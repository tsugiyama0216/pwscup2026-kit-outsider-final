"""Record fixed-seed model comparisons; never package or submit experiments."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter

from evaluation import HERE, check_environment, evaluate_frame, read_training_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["diagnose-r1", "conditional"])
    parser.add_argument("--calibration-steps", type=int, default=0)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    cfg, environment = check_environment()
    b = read_training_data()
    from pwscup2026.scoring.result import json_safe
    spec = importlib.util.spec_from_file_location("frozen_r1", HERE.parent / "R1/generate.py")
    r1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r1)
    label = args.label or args.mode
    outdir = HERE / "local-output" / label
    if outdir.exists():
        raise FileExistsError("An ablation record already exists; use a new descriptive label")
    outdir.mkdir(parents=True)
    diagnosis = {}
    if args.mode == "diagnose-r1":
        fits = {}
        train = r1.cox_features(b).assign(time=b.time.to_numpy(), event=b.onset.to_numpy())
        for penalty in (0.01, 0.0):
            fit = CoxPHFitter(penalizer=penalty).fit(train, "time", "event")
            fits[str(penalty)] = {"coefficients": fit.params_.to_dict(),
                                  "standard_errors": fit.standard_errors_.to_dict()}
        diagnosis = {"onset_fits": fits, "B_event_counts": {k: int(b[k].sum()) for k in ("onset", "death")}}
        r1.COX_PENALIZER = 0.0  # In-memory ablation; R1 source and artifacts are never edited.
        c, audit = r1.synthesize(b, cfg)
    else:
        from generate import synthesize
        c, audit = synthesize(b, cfg, calibration_steps=args.calibration_steps)
    raw = c.to_csv(index=False, lineterminator="\n").encode("utf-8")
    c = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    score, validation = evaluate_frame(c, b, cfg)
    (outdir / "C.csv").write_bytes(raw)
    result = json_safe({"label": label, "seed": 20260917, "calibration_steps": args.calibration_steps,
                        "environment": environment, "diagnosis": diagnosis,
                        "model_audit": audit, "score": score, "validation": validation})
    (outdir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"label": label, "utility": score["utility"], "facets": score["facets"],
                      "event_counts": validation["event_counts"],
                      "cox": score["detail"]["U_spec"].get("per_coefficient")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
