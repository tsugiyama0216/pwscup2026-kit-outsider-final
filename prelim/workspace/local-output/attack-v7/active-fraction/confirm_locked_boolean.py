"""Fresh-seed confirmation of the discovery-locked Boolean AIA rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_active_fraction as study  # noqa: E402


CONFIRMATION = tuple(range(20261501, 20261531))
LOCKED = ("full_complement_edge", "inverse_votes_at_most_two_edge")


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
        print(f"confirmation seed={seed}", flush=True)
        generated = study.run_seed(
            "confirmation",
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
        rows.extend(row for row in generated if row["strategy"] in LOCKED)
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["all_three_exact_zero"]].copy()
    pivot = eligible.pivot(
        index=["seed", "official_target"], columns="strategy", values="power"
    ).reset_index()
    pivot["delta"] = (
        pivot["inverse_votes_at_most_two_edge"] - pivot["full_complement_edge"]
    )
    summary = {
        "purpose": "fresh-seed confirmation after strategy lock; not a leaderboard forecast",
        "locked_candidate": "inverse_votes_at_most_two_edge",
        "baseline": "full_complement_edge",
        "seeds": list(CONFIRMATION),
        "eligible_exact_zero_blocks": len(pivot),
        "candidate_mean": float(pivot["inverse_votes_at_most_two_edge"].mean()),
        "baseline_mean": float(pivot["full_complement_edge"].mean()),
        "mean_delta": float(pivot["delta"].mean()),
        "median_delta": float(pivot["delta"].median()),
        "p10_delta": float(np.quantile(pivot["delta"], 0.10)),
        "p90_delta": float(np.quantile(pivot["delta"], 0.90)),
        "win_rate": float(np.mean(pivot["delta"] > 1e-12)),
        "loss_rate": float(np.mean(pivot["delta"] < -1e-12)),
        "noninferiority_rate": float(np.mean(pivot["delta"] >= -1e-12)),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
    }
    frame.to_csv(HERE / "confirmation_scores.csv", index=False)
    pivot.to_csv(HERE / "confirmation_pairs.csv", index=False)
    (HERE / "confirmation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
