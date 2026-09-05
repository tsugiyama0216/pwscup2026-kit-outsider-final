"""Reframe benchmark outputs around the binding official-zero evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent


def table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if name != "strategy" else "---" for name in columns) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for name, value in zip(columns, row, strict=True):
            if name == "strategy":
                values.append(f"`{value}`")
            elif name == "eligible_blocks":
                values.append(str(int(value)))
            elif name.startswith("win_rate"):
                values.append(f"{float(value):.1%}")
            else:
                values.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    unconditioned = summary.pop("validation_headline")
    exact = pd.read_csv(OUT / "conditional_exact_zero_summary.csv")
    near = pd.read_csv(OUT / "conditional_near_zero_summary.csv")
    validation_exact = exact.loc[exact["split"].eq("validation")].sort_values(
        "mean", ascending=False
    )
    validation_near = near.loc[near["split"].eq("validation")].sort_values(
        "mean", ascending=False
    )
    routed = validation_exact.loc[
        validation_exact["strategy"].eq("complement_routed_active")
    ].iloc[0]
    runner_up = validation_exact.iloc[1]
    summary["decision_headline"] = {
        "recommendation": "retain exact complement of routed active mask",
        "reason": "official v4, independent, and routed scores are all zero on these seven targets; unconditioned direct-signal shadows are contradicted by that hard evidence",
        "conditional_exact_zero_validation_blocks": int(routed["eligible_blocks"]),
        "conditional_exact_zero_routed_complement_mean": float(routed["mean"]),
        "conditional_exact_zero_runner_up": str(runner_up["strategy"]),
        "conditional_exact_zero_runner_up_mean": float(runner_up["mean"]),
        "conditional_exact_zero_runner_up_delta": float(
            runner_up["mean_delta_vs_routed_complement"]
        ),
        "replacement_supported": False,
    }
    summary["official_hard_evidence"] = {
        "targets": ["6", "7", "8", "9", "17", "18", "19"],
        "v4_per_target_scores": "all zero",
        "independent_per_target_scores": "all zero",
        "routed_per_target_scores": "all zero",
        "interpretation": "direct-direction shadow performance cannot override measured production failure",
    }
    summary["unconditioned_shadow_diagnostic_rejected_for_selection"] = unconditioned
    summary["conditional_sensitivity"] = {
        "exact_zero": validation_exact.to_dict(orient="records"),
        "near_zero": validation_near.to_dict(orient="records"),
        "warning": "small post-selected independent-null samples; used only to test whether another complement clearly dominates",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    unconditioned_frame = pd.read_csv(OUT / "distribution_summary.csv")
    unconditioned_frame = (
        unconditioned_frame.loc[unconditioned_frame["split"].eq("validation")]
        .sort_values("mean", ascending=False)
        .loc[
            lambda frame: frame["strategy"].isin(
                [
                    "safe_v4",
                    "calibrated_logit",
                    "knn_all5_direct",
                    "routed_probe",
                    "exact_complement_routed",
                    "inverse_consensus",
                ]
            )
        ]
    )
    report = [
        "# Final zero-target AIA shadow benchmark",
        "",
        "## Decision",
        "",
        "**Keep the current exact complement of the routed active mask.** The local evidence does not support replacing it.",
        "",
        "The binding evidence is the team's official result: targets 6, 7, 8, 9, 17, 18, and 19 scored exactly zero under v4, independent, and routed attacks. Therefore the large direct-direction scores in ordinary shadow worlds are known to be externally miscalibrated and are diagnostic only; shadow results do not override official feedback.",
        "",
        "## Official-zero-conditioned sensitivity",
        "",
        "The focused rerun generated 6 discovery and 6 validation null worlds. Only blocks where all three direct experts also scored exactly zero were retained. This reproduces the qualitative official condition, although it is a small post-selected sample.",
        "",
        *table(
            validation_exact,
            [
                "strategy",
                "eligible_blocks",
                "mean",
                "p10",
                "p90",
                "mean_delta_vs_routed_complement",
                "win_rate_vs_routed_complement",
            ],
        ),
        "",
        "Routed complement ranked first in discovery (10 eligible blocks; mean 0.099673) and validation (4 eligible blocks; mean 0.108333). Inverse consensus was close in validation (0.105673) but did not beat routed complement in any eligible validation block. The ≤0.02 sensitivity expanded validation to 7 blocks and preserved the order: routed complement 0.110717, inverse consensus 0.109197, v4 complement 0.081099, independent complement 0.063510.",
        "",
        "## Unconditioned shadow diagnostic",
        "",
        "Ordinary raw/noisy/time-shuffled/warped shadows strongly favor the direct direction because their releases preserve much more member linkage than the private production targets. Those numbers expose the domain gap and are not candidate-selection evidence.",
        "",
        *table(
            unconditioned_frame,
            [
                "strategy",
                "observations",
                "mean",
                "p10",
                "p90",
                "mean_delta_vs_exact",
                "win_rate_vs_exact",
            ],
        ),
        "",
        "In particular, safe v4 scores about 0.94 in these shadows but is officially zero on every target in scope. The calibrated logistic mixture learned positive weights on v4/independent/routed and is therefore also rejected despite its high unconditioned validation mean.",
        "",
        "## Method and QA",
        "",
        "- Main benchmark: 3 discovery and 3 disjoint validation seeds; raw, 3% noise, time-shuffled, production-margin rank-warped, and independent-null scenarios.",
        "- Focused conditional rerun: 6 discovery and 6 disjoint validation seeds under independent-null releases.",
        "- Every challenge contains 163 members and 163 controls, matched on sex, prefecture, five-year age band, onset, and rare status.",
        "- Every evaluated output gives exactly one finite prediction per unique `(target_k, challenge_row_id)` and uses the published half-per-visible-stratum constraint. No duplicate-key or fractional-ID behavior was tested.",
        "- All world consistency checks passed.",
        "",
        "## Caveats",
        "",
        "- This is truth-known local method comparison, not a private leaderboard forecast.",
        "- The exact-zero conditional validation sample is only four blocks; it can reject a clear alternative advantage, but cannot prove the routed complement is optimal.",
        "- Conditioning on zero under an independent null introduces post-selection bias.",
        "- The organizer's undisclosed transformation cannot be reconstructed locally.",
        "- No production candidate, archive, token, or submission was changed.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
