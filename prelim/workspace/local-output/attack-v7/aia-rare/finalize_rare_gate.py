"""Apply the final rare-specific, all-protecting acceptance gate to saved scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRESERVING = ("warp", "warp_noise_0p03")
STRESS = "warp_time_shuffled"


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "official_target", "variant"], as_index=False)
        .agg(
            cases=("delta", "size"),
            mean_delta=("delta", "mean"),
            p10_delta=("delta", lambda x: float(np.quantile(x, 0.10))),
            win_rate=("delta", lambda x: float(np.mean(x > 1e-12))),
            mean_delta_all=("delta_all", "mean"),
            p10_delta_all=("delta_all", lambda x: float(np.quantile(x, 0.10))),
            mean_delta_rare=("delta_rare", "mean"),
            p10_delta_rare=("delta_rare", lambda x: float(np.quantile(x, 0.10))),
            rare_win_rate=("delta_rare", lambda x: float(np.mean(x > 1e-12))),
        )
    )


def bundle(frame: pd.DataFrame, choices: dict[int, str]) -> pd.DataFrame:
    parts = [
        frame.loc[
            frame["official_target"].eq(target)
            & frame["variant"].eq(variant)
        ]
        for target, variant in choices.items()
    ]
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def bundle_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "cases": 0,
            "mean_delta": 0.0,
            "win_rate": 0.0,
            "p10_delta": 0.0,
            "mean_delta_all": 0.0,
            "p10_delta_all": 0.0,
            "mean_delta_rare": 0.0,
            "rare_win_rate": 0.0,
            "p10_delta_rare": 0.0,
        }
    return {
        "cases": int(len(frame)),
        "mean_delta": float(frame["delta"].mean()),
        "win_rate": float(np.mean(frame["delta"] > 1e-12)),
        "p10_delta": float(np.quantile(frame["delta"], 0.10)),
        "mean_delta_all": float(frame["delta_all"].mean()),
        "p10_delta_all": float(np.quantile(frame["delta_all"], 0.10)),
        "mean_delta_rare": float(frame["delta_rare"].mean()),
        "rare_win_rate": float(np.mean(frame["delta_rare"] > 1e-12)),
        "p10_delta_rare": float(np.quantile(frame["delta_rare"], 0.10)),
    }


def main() -> None:
    scored = pd.read_csv(HERE / "rare_gate_scores.csv")
    preserving_rows = scored.loc[scored["scenario"].isin(PRESERVING)].copy()
    summary = aggregate(preserving_rows)
    discovery = summary.loc[
        summary["split"].eq("discovery")
        & ~summary["variant"].eq("baseline")
        & (summary["mean_delta"] > 0.0)
        & (summary["win_rate"] >= 0.60)
        & (summary["p10_delta"] >= -0.01)
        & (summary["mean_delta_all"] >= -0.002)
        & (summary["p10_delta_all"] >= -0.01)
        & (summary["mean_delta_rare"] > 0.0)
        & (summary["rare_win_rate"] >= 0.50)
        & (summary["p10_delta_rare"] >= -0.10)
    ].copy()
    choices: dict[int, str] = {}
    for target, group in discovery.groupby("official_target"):
        group = group.assign(
            objective=group["mean_delta"] + 0.25 * group["mean_delta_rare"]
        )
        best = group.sort_values(
            ["objective", "p10_delta", "mean_delta_all"],
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        choices[int(target)] = str(best["variant"])

    validation_rows = bundle(
        preserving_rows.loc[preserving_rows["split"].eq("validation")], choices
    )
    stress_rows = bundle(
        scored.loc[scored["split"].eq("validation") & scored["scenario"].eq(STRESS)],
        choices,
    )
    validation = bundle_metrics(validation_rows)
    stress = bundle_metrics(stress_rows)
    checks = {
        "has_predeclared_discovery_candidate": bool(choices),
        "validation_mean_delta_gt_0": validation["mean_delta"] > 0.0,
        "validation_win_rate_gte_0p60": validation["win_rate"] >= 0.60,
        "validation_p10_delta_gte_m0p01": validation["p10_delta"] >= -0.01,
        "validation_mean_all_gte_0": validation["mean_delta_all"] >= 0.0,
        "validation_p10_all_gte_m0p005": validation["p10_delta_all"] >= -0.005,
        "validation_mean_rare_gt_0": validation["mean_delta_rare"] > 0.0,
        "validation_rare_win_rate_gte_0p60": validation["rare_win_rate"] >= 0.60,
        "validation_p10_rare_gte_m0p05": validation["p10_delta_rare"] >= -0.05,
        "stress_mean_delta_gte_m0p01": stress["mean_delta"] >= -0.01,
        "stress_p10_delta_gte_m0p05": stress["p10_delta"] >= -0.05,
        "stress_mean_all_gte_m0p005": stress["mean_delta_all"] >= -0.005,
    }
    accepted = all(checks.values())
    if discovery.empty:
        selected_discovery = discovery.copy()
    else:
        selected_discovery = discovery.loc[
            discovery.apply(
                lambda row: choices.get(int(row["official_target"])) == row["variant"],
                axis=1,
            )
        ]
    selected_discovery.to_csv(HERE / "rare_gate_discovery_selection.csv", index=False)
    report = {
        "decision": "adopt" if accepted else "reject",
        "reason": (
            "all rare-purpose and all-protection gates passed"
            if accepted
            else "at least one predeclared rare-purpose/all-protection/stress gate failed"
        ),
        "discovery_choices": {str(k): v for k, v in choices.items()},
        "validation": validation,
        "stress": stress,
        "checks": checks,
        "note": (
            "The earlier power-only gate was invalid for this task because its bundle "
            "improved max(all, rare) while reducing the rare channel."
        ),
        "production_action": "none" if not accepted else "research candidate only",
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "submission_performed": False,
    }
    (HERE / "final_rare_gate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
