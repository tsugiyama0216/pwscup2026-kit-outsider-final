"""Finalize the discovery-selected complement-of-unanimous AIA mask.

The output is a seven-target research mask and an evidence report, not a
submission candidate.  No token or archive is read or written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_official_bayes as research  # noqa: E402


STRATEGY = "complement_unanimous_active"


def csv_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def one(summary: pd.DataFrame, split: str, condition: str, strategy: str) -> dict:
    row = summary.loc[
        summary["split"].eq(split)
        & summary["condition"].eq(condition)
        & summary["strategy"].eq(strategy)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"expected one summary row for {split}/{condition}/{strategy}, got {len(row)}"
        )
    return row.iloc[0].to_dict()


def main() -> None:
    summary = pd.read_csv(HERE / "policy_summary.csv")
    discovery = one(summary, "discovery", "exact", STRATEGY)
    validation = one(summary, "validation", "exact", STRATEGY)
    validation_near = one(summary, "validation", "near", STRATEGY)
    baseline_validation = one(
        summary, "validation", "exact", "exact_complement_edge"
    )
    partial = (
        summary.loc[
            summary["split"].eq("validation")
            & summary["condition"].eq("exact")
            & summary["active_k"].between(1, 162)
        ]
        .sort_values(["mean_delta", "p10_delta"], ascending=False)
        .iloc[0]
        .to_dict()
    )

    production, reconciliation = research.production_features()
    recommended = research.build_research_mask(production, STRATEGY)
    baseline = research.build_research_mask(
        production, "exact_complement_edge"
    )
    if recommended.duplicated(["target_k", "challenge_row_id"]).any():
        raise RuntimeError("recommended mask has duplicate keys")
    if set(recommended["time_hat"].unique()) != {9.5, 11.0}:
        raise RuntimeError("recommended mask is not binary 9.5/11")
    expected_keys = set()
    for target in research.ZERO_TARGETS:
        challenge = pd.read_csv(
            research.PRODUCTION_TARGETS / f"aia_challenge_{target}.csv"
        )
        expected_keys.update(
            (target, int(value)) for value in challenge["challenge_row_id"]
        )
    observed_keys = set(
        zip(
            recommended["target_k"].astype(int),
            recommended["challenge_row_id"].astype(int),
            strict=True,
        )
    )
    if observed_keys != expected_keys or len(recommended) != 7 * 326:
        raise RuntimeError("recommended mask keys differ from public challenges")
    current_selected = baseline["selected"].to_numpy(dtype=bool)
    new_selected = recommended["selected"].to_numpy(dtype=bool)
    additions = int(np.count_nonzero(new_selected & ~current_selected))
    removals = int(np.count_nonzero(current_selected & ~new_selected))
    if removals != 0:
        raise RuntimeError("complement-of-unanimous must contain the routed complement")

    robust = research.robust_raw_bounds(production, recommended)
    target_spec = (
        recommended.groupby("target_k", as_index=False)
        .agg(
            rows=("challenge_row_id", "size"),
            active_rows=("selected", "sum"),
            unanimous_rejected_rows=("selected", lambda value: int((value == 0).sum())),
        )
        .assign(
            current_exact_complement_active=163,
            added_rows=lambda x: x["active_rows"] - 163,
        )
    )
    target_spec.to_csv(HERE / "production_target_spec.csv", index=False)
    recommended.to_csv(HERE / "recommended_zero_block_mask.csv", index=False)

    score_delta = float(validation["mean_delta"]) * 7.0 / 20.0
    empirical_gate = bool(
        float(discovery["mean_delta"]) > 0.02
        and float(discovery["win_rate"]) >= 0.60
        and float(validation["mean_delta"]) > 0.02
        and float(validation["win_rate"]) >= 0.70
        and float(validation["p10_delta"]) >= -0.05
        and float(validation_near["mean_delta"]) > 0.02
    )
    result = {
        "recommendation": {
            "strategy": STRATEGY,
            "empirical_gate_passed": empirical_gate,
            "transform": "For targets 6,7,8,9,17,18,19 set time_hat=9.5 unless v4, independent, and routed all had time_hat!=11; set unanimous-active rows to 11.0. Leave all other targets and F_mia unchanged.",
            "normal_format": True,
            "active_rows_by_target": {
                str(int(row.target_k)): int(row.active_rows)
                for row in target_spec.itertuples(index=False)
            },
            "added_rows_vs_exact_routed_complement": additions,
            "removed_rows_vs_exact_routed_complement": removals,
            "seven_target_mask_sha256": csv_sha256(
                recommended[
                    ["target_k", "challenge_row_id", "time_hat", "selected", "pattern", "ranking_score"]
                ]
            ),
        },
        "shadow_evidence": {
            "discovery_exact": discovery,
            "validation_exact": validation,
            "validation_near_zero": validation_near,
            "baseline_validation_exact": baseline_validation,
            "score_1_directional_delta": score_delta,
            "partial_active_count_best_validation": partial,
        },
        "official_evidence": {
            "three_masks_reconciled": reconciliation,
            "common_zero_targets": list(research.ZERO_TARGETS),
            "routed_selected_control_minus_member_successes_total_range": [59, 68],
        },
        "partial_identification": {
            **robust,
            "interpretation": "Official aggregates alone do not identify the unanimous intersection; the recommendation relies on the replicated conditional-shadow ordering, not a worst-case guarantee.",
        },
        "qa": {
            "rows": len(recommended),
            "unique_keys": len(observed_keys),
            "expected_keys": len(expected_keys),
            "allowed_times": sorted(recommended["time_hat"].unique().tolist()),
            "token_read": False,
            "production_candidate_changed": False,
            "archive_created": False,
            "submission_performed": False,
        },
    }
    (HERE / "final_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    spec_rows = "\n".join(
        f"| {int(row.target_k)} | {int(row.active_rows)} | {int(row.added_rows)} | {int(row.unanimous_rejected_rows)} |"
        for row in target_spec.itertuples(index=False)
    )
    report = f"""# AIA official-feedback Bayesian/robust re-evaluation

## 結論

正常形式の明確な改善候補は `{STRATEGY}`。共通0点target 6, 7, 8, 9, 17, 18, 19で、
v4・independent・routedの三方式が**全会一致でactive**にした行だけを`11.0`へ退避し、
それ以外を`9.5`にする。現在のrouted完全補集合をすべて含み、541行を追加する。

探索側37 blockでは完全補集合比の対象別power差が平均
`{float(discovery['mean_delta']):+.6f}`、勝率`{float(discovery['win_rate']):.1%}`。
未使用の検証側26 blockでは平均`{float(validation['mean_delta']):+.6f}`、
勝率`{float(validation['win_rate']):.1%}`、p10差`{float(validation['p10_delta']):+.6f}`だった。
score_1への方向的換算は約`{score_delta:+.6f}`。これはprivate score予測ではない。

## Production変換仕様

| target | active rows | exact complementから追加 | unanimous-activeで退避 |
|---:|---:|---:|---:|
{spec_rows}

判定は各公式提出ファイルの`time_hat != 11.0`をactiveとする。3つすべてactiveなら11.0、
それ以外なら9.5。正点13 targetとF_miaは変更しない。

## 比較結果

- exact-zero検証: baseline `{float(baseline_validation['mean_power']):.6f}` → candidate `{float(validation['mean_power']):.6f}`。
- near-zero検証35 blockでも平均差 `{float(validation_near['mean_delta']):+.6f}`、勝率 `{float(validation_near['win_rate']):.1%}`。
- active数を32〜144へ絞る案の検証最良は `{partial['strategy']}` だが、平均差は
  `{float(partial['mean_delta']):+.6f}`、勝率`{float(partial['win_rate']):.1%}`で再現性がなく棄却。
- `000`だけ、active多数決の補集合、release距離ridge/Bayesian tailも完全補集合を安定して上回らなかった。

## 公式制約と限界

三つのproduction maskは提出済みroutedを含め2,282/2,282行で再構成一致した。routedの
公式集計は、7 target合計でactive側のcontrol成功がmember成功より59〜68件多い。
一方、公式集計だけでは三方式全会一致集合の符号は識別できず、136 hits/arm/targetを固定した
MILPでもcandidate raw advantageは`{robust['candidate_raw_advantage_min']:.0f}`〜
`{robust['candidate_raw_advantage_max']:.0f}`。したがって最悪値保証ではなく、独立seedで再現した
条件付きshadowの順位を採用根拠とする。

## QA

- research mask: 2,282行、複合key重複なし、全challenge keyと一致
- time_hat: 9.5 / 11.0のみ
- production candidate・ZIP・token・提出: 変更なし
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result["recommendation"], ensure_ascii=False, indent=2))
    print(json.dumps(result["shadow_evidence"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
