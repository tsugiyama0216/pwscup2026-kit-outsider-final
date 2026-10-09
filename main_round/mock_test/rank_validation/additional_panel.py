"""Evaluate unchanged Rv3/Mv1/Mv2 on existing fixed panels."""
import argparse
from pathlib import Path
import subprocess
import sys

from .additional_worker import MODELS
from .panel import MOCK, check_panel, score


def predict(panel, model):
    if model not in MODELS:
        raise ValueError("Unknown additional frozen model")
    panel, _, _ = check_panel(panel)
    output = panel / "predictions" / model
    log = panel / "logs" / (model + "_additional.log")
    if output.exists() or log.exists() or (panel / "scores" / model).exists():
        raise ValueError("Preserve existing predictions, scores and logs")
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-B", "-m", "rank_validation.additional_worker", "--model", model,
               "--public", str(panel / "public"), "--output", str(output)]
    print(f"Predicting frozen {model}; log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        run = subprocess.run(command, cwd=MOCK, stdout=stream, stderr=subprocess.STDOUT)
    check_panel(panel)
    if run.returncode:
        raise RuntimeError(f"Additional model failed; inspect {log}")
    return score(panel, model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    args = parser.parse_args(argv)
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate models")
    for model in args.models:
        predict(args.panel, model)


if __name__ == "__main__":
    main()
