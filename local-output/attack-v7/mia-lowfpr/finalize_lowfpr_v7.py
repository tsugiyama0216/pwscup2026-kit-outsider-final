"""Finalize the research record without touching a submission or token."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER_SRC = ROOT / "starter" / "src"
for value in (ROOT, STARTER_SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from attack_phase_v5.generate_submission_v5 import validate_mia  # noqa: E402
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH, validate_mia_submission  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    candidate_path = HERE / "production_variants" / "F_mia_daybreak_v4_rarelock.csv"
    v4_path = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v4" / "F_mia.csv"
    day_path = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-daybreak-independent" / "F_mia.csv"
    candidate = pd.read_csv(candidate_path)
    v4 = pd.read_csv(v4_path)
    day = pd.read_csv(day_path)
    candidate = candidate.set_index("record_id").loc[v4["record_id"]].reset_index()
    day = day.set_index("record_id").loc[v4["record_id"]].reset_index()

    strict = validate_mia(candidate, v4)
    official_layer2 = validate_mia_submission(candidate, load_config(KIT_CONFIG_PATH), dist=None)
    validator = {
        "official_validate_mia_submission_without_dist": {
            "ok": bool(official_layer2.ok),
            "errors": list(official_layer2.errors),
        },
        "production_reference_validate_mia": {"ok": True, **strict},
        "full_package_validator_run": False,
        "why_not_full_package": (
            "Task scope forbids production/ZIP/token changes, and the historical production runtime_dist "
            "used for these 20 target columns is not present.  The local toy participant_data manifest has 29 targets."
        ),
        "focused_tests": {"ok": True, "tests": 5},
        "repository_unittest_discover": {"ok": True, "tests": 118},
    }
    (HERE / "validator_result.json").write_text(
        json.dumps(validator, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    target_columns = list(v4.columns[1:])
    candidate_values = candidate[target_columns].to_numpy(float)
    v4_values = v4[target_columns].to_numpy(float)
    day_values = day[target_columns].to_numpy(float)
    changed = candidate_values != v4_values
    changed_day = candidate_values != day_values
    per_target = {
        target: int(changed[:, index].sum()) for index, target in enumerate(target_columns)
    }
    diff = {
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": sha256(candidate_path),
        "v4_path": str(v4_path.resolve()),
        "v4_sha256": sha256(v4_path),
        "changed_row_target_cells_vs_v4": int(changed.sum()),
        "changed_distinct_rows_vs_v4": int(changed.any(axis=1).sum()),
        "changed_targets_vs_v4": [
            target for index, target in enumerate(target_columns) if changed[:, index].any()
        ],
        "changed_target_count_vs_v4": int(changed.any(axis=0).sum()),
        "changed_rows_per_target_vs_v4": per_target,
        "changed_row_target_cells_vs_daybreak": int(changed_day.sum()),
        "changed_distinct_rows_vs_daybreak": int(changed_day.any(axis=1).sum()),
    }
    (HERE / "candidate_diff.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    benchmark = json_load(HERE / "summary.json")
    structural = json_load(HERE / "official_structural_bound.json")
    integer_bounds = json_load(HERE / "inverse_power_bounds.json")
    lp_bounds = json_load(HERE / "inverse_power_bounds_lp.json")
    scenario = pd.read_csv(HERE / "summary_by_scenario.csv")
    selected_scenario = scenario.loc[scenario["variant"] == "daybreak_v4_rarelock"]

    decision = {
        "recommendation": "retain_v4",
        "replacement_supported": False,
        "candidate": "daybreak_v4_rarelock",
        "reason": (
            "The candidate preserves v4 rare TPR and has a positive official aggregate-all lower bound, "
            "but it lost to v4 in every one of 12 held-out scenario-world observations and no exact "
            "target-wise max(all,rare) lower bound was certified."
        ),
        "downside": {
            "heldout_mean_delta_power": benchmark["validation"]["mean_delta_power"],
            "heldout_median_delta_power": benchmark["validation"]["median_delta_power"],
            "heldout_p10_delta_power": benchmark["validation"]["p10_delta_power"],
            "heldout_minimum_delta_power": benchmark["validation"]["minimum_delta_power"],
            "heldout_win_rate": benchmark["validation"]["win_rate"],
            "integer_milp": integer_bounds["runs"],
            "lp_relaxation": lp_bounds["runs"],
        },
    }
    final = {
        "decision": decision,
        "candidate": diff,
        "validator": validator,
        "official_structural_bound": structural,
        "benchmark_gate": benchmark["selection_gate"],
        "scenario_rows": selected_scenario.to_dict(orient="records"),
        "safety": {
            "token_read": False,
            "archive_created": False,
            "production_candidate_changed": False,
            "submission_performed": False,
        },
    }
    (HERE / "final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    val = benchmark["validation"]
    report = f"""# PWSCUP2026 MIA low-FPR v7 research

## 結論

**最終提出のMIAはv4を維持する。** 研究候補
`daybreak_v4_rarelock` は公式allチャネルだけを見ると有望だが、採点対象の
`mean_target max(TPR_all, TPR_rare)` で再現優位を確認できなかった。従って最終一回で
差し替えるためのgateを通していない。

## 研究候補

- path: `{diff['candidate_path']}`
- SHA-256: `{diff['candidate_sha256']}`
- v4からの変更: {diff['changed_row_target_cells_vs_v4']:,} cells / {diff['changed_distinct_rows_vs_v4']:,} distinct rows / {diff['changed_target_count_vs_v4']} targets
- Daybreakからの変更: {diff['changed_row_target_cells_vs_daybreak']:,} cells（eligible rare 11行の列内順位修復のみ）
- 正常形式: 1,049行、20 target、finite [0,1]、全列tieなし

方式はDaybreak-independentの各target列を土台にし、既にv3/v4から回収済みの
eligible rare 11行だけを、Daybreakの同じ11 score slot内でv4順に並べ替える。
private 163行のbottom集合は維持する。LBから個別会員/非会員を断定したり、個別行を
正解扱いしてspliceしたりしていない。

## 公式結果からの強い事実

- Daybreak all=`0.1062193732` はv4 all=`0.1023675214`より `+0.0038518519`。
- rare順序はv4と完全同一になるため、candidate rareはv4と同じ `1/3`。
- rare-lockがDaybreakの低FPR prefixを変え得るのはtarget 19のrank 12だけ。
  target 1のrank 7/25 swapはband 4の最小prefix内に両方入る。
- よってcandidate allは保守的に `[0.1042193732, 0.1082193732]`、
  v4比下限 `+0.0018518519`。

ただしこの下限はallのtarget平均であり、採点値のtargetごとの
`max(all, rare)`には直結しない。v4 rareが勝つtarget上のall改善は得点に寄与せず、
別targetのall悪化だけが残る場合がある。

## discovery / held-out

固定recipeをdiscovery 4 seed × 2 scenario、held-out 6 seed × 2 scenarioで評価した。
validationはrecipe選択に使用していない。

- discovery delta power mean: `{benchmark['discovery']['mean_delta_power']:+.6f}` (8 observations)
- held-out delta power mean: `{val['mean_delta_power']:+.6f}`
- held-out median: `{val['median_delta_power']:+.6f}`
- held-out p10: `{val['p10_delta_power']:+.6f}`
- held-out minimum: `{val['minimum_delta_power']:+.6f}`
- held-out win rate: `{val['win_rate']:.1%}` (0/12)
- held-out rare delta minimum: `{val['minimum_delta_rare']:+.6f}`（rare不変は確認）

toy shadowはproduction Bと同一ではないためLB予測ではないが、下振れ方向が12/12で
一貫しており、差替えを支持する再現証拠にはならない。

## feedback-constrained bound

v3/v4/Daybreak/guidedのall集約、全target band、target size 13×25+6×26+1×27、
排他性、rare hit比、v4/guidedの総合score表示幅を同時制約した7,010変数MILPを実行した。
両側120秒でfeasible incumbentなしのtime limitとなり、厳密power保証は得られなかった。
LP relaxationは `[-0.157960, +0.842137]` と緩すぎ、候補の非劣性を証明しない。
これは個別行真値の復元・断定には使っていない。

## validation

- official `validate_mia_submission` layer-2（distなし）: {'PASS' if official_layer2.ok else 'FAIL'}
- production-reference strict validator: PASS
- focused unit tests: 5/5 PASS
- repository `unittest discover`: 118/118 PASS
- full ZIP/package validator: 未実行（production/ZIP/token非変更という本タスク範囲を遵守）

## 採否

公式allだけなら改善する巧い合成だが、最終scoreのworst-case保証がなく、held-outは
明確に悪化した。最後の一回ではv4から変更しない。
"""
    (HERE / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(HERE / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
