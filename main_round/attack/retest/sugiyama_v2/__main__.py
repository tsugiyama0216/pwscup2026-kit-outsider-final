"""Standard Sugiyama mock evaluation: generate or evaluate one joint world."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from .generate import build_world
from .io import export_world, output_path
from .runner import DEFAULT_PLUGIN, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "evaluate", "run"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", choices=("mixed", "stress"))
    parser.add_argument("--split", choices=("dev", "holdout"))
    parser.add_argument("--case", type=Path)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-seed", type=int)
    parser.add_argument("--floor-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.floor_repetitions < 2 or (args.floor_seed is not None and args.floor_seed < 0):
        parser.error("Floor repetitions must be >= 2 and floor seed must be nonnegative")
    output = output_path(args.output)
    if args.action == "evaluate":
        if args.case is None:
            parser.error("evaluate requires --case")
        if any(value is not None for value in (args.seed, args.profile, args.split)):
            parser.error("evaluate uses the saved seed/profile/split; omit generation options")
        evaluate(args.case, args.plugin, output, floor_seed=args.floor_seed,
                 floor_repetitions=args.floor_repetitions)
        return
    if args.case is not None:
        parser.error("--case is only valid for evaluate")
    import attack_workspace as workspace
    workspace.verify_hashes()
    workspace.load_runtime()
    world = build_world(2026100701 if args.seed is None else args.seed,
                        args.profile or "mixed", args.split or "dev")
    case = output / "case" if args.action == "run" else output
    export_world(world, case)
    print("Saved joint mock world: " + str(case), flush=True)
    if args.action == "run":
        evaluate(case, args.plugin, output / "evaluation", floor_seed=args.floor_seed,
                 floor_repetitions=args.floor_repetitions)


if __name__ == "__main__":
    main()
