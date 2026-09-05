"""Fresh-seed control: compare the locked Boolean rule with predicting every row."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_active_fraction as study  # noqa: E402


CONFIRMATION = tuple(range(20261601, 20261631))
STRATEGIES = (
    "full_complement_edge",
    "inverse_votes_at_most_two_edge",
    "all_rows_edge",
)


def main() -> None:
    candidates = pd.read_csv(study.bench.DIST / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(study.bench.DIST / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(study.bench.DIST / "A_bg.csv").reset_index(drop=True)
    detector = study.bench.shadow.load_rare_detector(study.bench.DIST)
    cfg = study.bench.load_config(study.bench.KIT_CONFIG_PATH)
    spec = study.bench.shadow.ShadowSpec(
        n_targets=7,
        candidate_rows=1_049,
        target_rows=1_049,
        shared_total=175,
        minimum_shared=25,
        maximum_shared=26,
    )
    horizon = float(cfg["onset"]["horizon_years"])
    abstain = study.bench.v2.safe_abstain_time(
        horizon, float(cfg["aia"]["tau_time"])
    )
    censor_time = float(background["time"].max()) - float(cfg["aia"]["tau_time"])
    rows: list[dict[str, object]] = []
    for seed in CONFIRMATION:
        print(f"all-rows control seed={seed}", flush=True)
        generated = study.run_seed(
            "all_rows_control",
            seed,
            candidates=candidates,
            b_self=b_self,
            background=background,
            detector=detector,
            spec=spec,
            cfg=cfg,
            horizon=horizon,
            abstain=abstain,
            censor_time=censor_time,
        )
        rows.extend(row for row in generated if row["strategy"] in STRATEGIES)
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["all_three_exact_zero"]].copy()
    pivot = eligible.pivot(
        index=["seed", "official_target"], columns="strategy", values="power"
    ).reset_index()
    for strategy in ("inverse_votes_at_most_two_edge", "all_rows_edge"):
        pivot[f"{strategy}_delta_vs_complement"] = (
            pivot[strategy] - pivot["full_complement_edge"]
        )
    pivot["locked_delta_vs_all_rows"] = (
        pivot["inverse_votes_at_most_two_edge"] - pivot["all_rows_edge"]
    )

    def stats(values: pd.Series) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p10": float(np.quantile(values, 0.10)),
            "minimum": float(values.min()),
            "win_rate": float(np.mean(values > 1e-12)),
            "loss_rate": float(np.mean(values < -1e-12)),
        }

    summary = {
        "purpose": "fresh-seed all-rows control after the Boolean strategy was locked; not a leaderboard forecast",
        "seeds": list(CONFIRMATION),
        "eligible_exact_zero_blocks": len(pivot),
        "mean_power": {
            strategy: float(pivot[strategy].mean()) for strategy in STRATEGIES
        },
        "delta_vs_full_complement": {
            strategy: stats(pivot[f"{strategy}_delta_vs_complement"])
            for strategy in ("inverse_votes_at_most_two_edge", "all_rows_edge")
        },
        "locked_minus_all_rows": stats(pivot["locked_delta_vs_all_rows"]),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
    }
    frame.to_csv(HERE / "all_rows_control_scores.csv", index=False)
    pivot.to_csv(HERE / "all_rows_control_pairs.csv", index=False)
    (HERE / "all_rows_control_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
