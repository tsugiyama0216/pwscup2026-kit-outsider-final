from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pwscup2026.kit.selfscore import load_local_reference, self_score

from experiment_uspec_current_v4 import ROOT, make_candidate


VALIDATION_RUNS = {
    "run1": ROOT
    / "local-output/opendp_aim/experiments/v4_fresh_validation/run1/"
    "C_fixed63_mlt5p5_seed2_risk.csv",
    "run2": ROOT
    / "local-output/opendp_aim/experiments/v4_fresh_validation/run2/"
    "fixed63_mlt5p5_seed2_risk.csv",
    "run3": ROOT
    / "local-output/opendp_aim/experiments/v4_fresh_validation/run3/"
    "m_lt_5_5_seed2_risk_C.csv",
}


def main() -> None:
    experiment_dir = (
        ROOT / "local-output/opendp_aim/experiments/v5_uspec_current_v4"
    )
    output_dir = experiment_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    screen = pd.read_csv(experiment_dir / "screen.csv").head(30)
    baseline = pd.DataFrame(
        [
            {
                "candidate": f"dw{weight:g}_k1_s0",
                "death_weight": weight,
                "neighbors": 1,
                "seed": 0,
            }
            for weight in (0.25, 0.296, 0.31)
        ]
    )
    parameters = (
        pd.concat(
            [
                screen[["candidate", "death_weight", "neighbors", "seed"]],
                baseline,
            ],
            ignore_index=True,
        )
        .drop_duplicates("candidate")
        .reset_index(drop=True)
    )

    background = pd.read_csv(ROOT / "participant_data/A_bg.csv")
    config = json.loads(
        (ROOT / "opendp_aim/config/pwscup_aim_v4.json").read_text(
            encoding="utf-8"
        )
    )
    donor_config = config["postprocess"]["public_outcome_donor"]
    b, reference, score_config, _ = load_local_reference(ROOT / "participant_data")

    rows: list[dict] = []
    for run, path in VALIDATION_RUNS.items():
        base = pd.read_csv(path)
        for parameter in parameters.itertuples(index=False):
            candidate = make_candidate(
                base,
                background,
                donor_config,
                float(parameter.death_weight),
                int(parameter.neighbors),
                int(parameter.seed),
            )
            scored = self_score(candidate, b, reference, score_config)
            spec = scored["detail"]["U_spec"]
            rows.append(
                {
                    "run": run,
                    "candidate": parameter.candidate,
                    "death_weight": parameter.death_weight,
                    "neighbors": parameter.neighbors,
                    "seed": parameter.seed,
                    "U": scored["utility"],
                    **scored["facets"],
                    "io_part": spec.get("io_part"),
                    "tstr_cal": spec.get("tstr_cal"),
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "scores.csv", index=False)
    aggregate = (
        raw.groupby(
            ["candidate", "death_weight", "neighbors", "seed"], as_index=False
        )
        .agg(
            mean_U=("U", "mean"),
            min_U=("U", "min"),
            max_U=("U", "max"),
            mean_U_spec=("U_spec", "mean"),
            min_U_spec=("U_spec", "min"),
            mean_U_rare=("U_rare", "mean"),
            min_U_rare=("U_rare", "min"),
        )
        .sort_values(["mean_U", "min_U"], ascending=False)
    )
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    print(aggregate.to_string(index=False))
    print(f"\nWrote: {output_dir}")


if __name__ == "__main__":
    main()
