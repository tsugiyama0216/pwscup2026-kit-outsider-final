"""Freeze four transparent attacks, run the existing real-prelim evaluator, compare.

This orchestration process may read scoring results. The attack plugins run in
separate public-input-only workers and never import this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVALUATOR_PARENT = REPO / "main_round/mock_test"
sys.path.insert(0, str(EVALUATOR_PARENT))
from mock_TS_using_prelim.common import (OUTPUT_ROOT, code_hashes, new_output,
    read_json, sha256, verify_bundle, write_json)
from mock_TS_using_prelim.runner import export_public, run_attack
from mock_TS_using_prelim.cli import save_score

METHODS = {
    "baseline": ("公式参照 DOMIAS + 1-NN", EVALUATOR_PARENT / "mock_TS_using_prelim/example_attack.py"),
    "explore_1_distance": ("① QI最近傍MIA + 1-NN", HERE / "explore_1_distance/plugin.py"),
    "explore_2_density": ("② 公式DOMIAS + 近傍最頻区間", HERE / "explore_2_density/plugin.py"),
    "explore_3_gen_lra": ("③ 固定カーネルGen-LRA + 近傍最頻区間", HERE / "explore_3_gen_lra/plugin.py"),
    "explore_4_predictor": ("④ 決定木の残差差MIA + time回帰", HERE / "explore_4_predictor/plugin.py"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def algorithm_hashes():
    paths = [Path(__file__)]
    for directory in sorted(HERE.glob("explore_*")):
        if directory.is_dir():
            paths.extend(p for p in directory.rglob("*")
                         if p.is_file() and p.suffix in {".py", ".json"}
                         and "__pycache__" not in p.parts)
    return {p.relative_to(HERE).as_posix(): sha256(p) for p in sorted(paths)}


def write_comparison(output, results, protocol):
    rows, details = [], []
    baseline = results.get("baseline")
    for key in METHODS:
        if key not in results:
            continue
        result = results[key]
        row = {"method": key, "label": METHODS[key][0], **result["summary"],
               "elapsed_seconds": result["exploration_elapsed_seconds"]}
        if baseline:
            for metric in ("attack_power", "mia_power", "aia_power"):
                row["delta_" + metric + "_vs_baseline"] = row[metric] - baseline["summary"][metric]
            comparison = {x["target_k"]: x for x in baseline["by_target"]}
            for task in ("mia", "aia"):
                changes = np.array([x[task + "_score"] - comparison[x["target_k"]][task + "_score"]
                                    for x in result["by_target"]], dtype=float)
                row[task + "_targets_better"] = int((changes > 1e-12).sum())
                row[task + "_targets_equal"] = int((np.abs(changes) <= 1e-12).sum())
                row[task + "_targets_worse"] = int((changes < -1e-12).sum())
        rows.append(row)
        details.extend({"method": key, **x} for x in result["by_target"])
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(details).to_csv(output / "by_target.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    verification = {}
    if baseline and "explore_1_distance" in results:
        base_aia = pd.read_csv(output / "baseline/evaluation/F_aia.csv", float_precision="round_trip")
        distance_aia = pd.read_csv(output / "explore_1_distance/evaluation/F_aia.csv", float_precision="round_trip")
        verification["distance_aia_equals_official"] = base_aia.equals(distance_aia)
    if baseline and "explore_2_density" in results:
        verification["density_mia_bytes_equal_official"] = (
            sha256(output / "baseline/evaluation/F_mia.csv") ==
            sha256(output / "explore_2_density/evaluation/F_mia.csv"))
    if "explore_2_density" in results and "explore_3_gen_lra" in results:
        left = pd.read_csv(output / "explore_2_density/evaluation/F_aia.csv", float_precision="round_trip")
        right = pd.read_csv(output / "explore_3_gen_lra/evaluation/F_aia.csv", float_precision="round_trip")
        index = ["target_k", "challenge_row_id"]
        verification["density_genlra_aia_equal"] = left.set_index(index).sort_index().equals(right.set_index(index).sort_index())
    if not all(verification.values()):
        raise AssertionError("Expected controlled-comparison invariants failed: " + str(verification))
    document = {"protocol": protocol, "completed_utc": utc_now(), "methods": rows,
                "controlled_comparison_checks": verification,
                "interpretation": "Real preliminary retrospective benchmark, no target-truth fitting or tuning. Not unseen main-round performance."}
    write_json(output / "comparison.json", document)

    lines = ["# 4方式の予備戦実データ比較", "", "実team24から自己を除く20標的を、予備戦の固定公式コードで評価しました。",
             "攻撃者の候補は1,049人、AIAは20×326行です。予備戦には本戦のφ控除がありません。",
             "設定を採点前のprotocol.jsonに固定し、正解を使ったパラメーター調整は行っていません。", "",
             "| 手法 | MIA力 | AIA力 | 合計 | 基準との差 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        delta = row.get("delta_attack_power_vs_baseline")
        delta_text = f"{delta:+.6f}" if delta is not None else "—"
        lines.append(f"| {row['label']} | {row['mia_power']:.6f} | {row['aia_power']:.6f} | {row['attack_power']:.6f} | {delta_text} |")
    lines += ["", "## 比較条件", "",
              "- MIA: 標的ごとに全体・希少群のTPR@FPR=1%の大きい方を採用し、20標的平均。",
              "- AIA: ±0.5年以内の成功率について会員・対照差を正規化。希少側は1.64×SE補正し、全体との大きい方を20標的平均。",
              "- 合計はMIA力+AIA力。全体/希少の平均を取った後にmaxする計算ではありません。",
              "- ①と公式のAIAは同じ。②と公式のMIAは同じ。②と③のAIAは同じです。",
              "- ①はQI7列、②③MIAはBMI/FPGを含む連続7列、④MIAはQI7列と候補の既知timeを使います。使用情報の差も含む比較です。",
              "- ③はRで共分散・Silverman帯域を固定するGen-LRA変種です。著者コードのR+x再fit版の完全再現ではありません。",
              "- ④は二乗誤差で学習する固定決定木で、AIA得点を直接最適化していません。",
              "- 予備戦は既に公開・既使用の単一問題です。本戦の未知データ性能、公式本戦得点、統計的な優越性を示す結果ではありません。", "",
              "## 再現・詳細", "", "- [固定条件と全攻撃ソースSHA-256](protocol.json)",
              "- [集約比較CSV](comparison.csv)", "- [20標的別の比較CSV](by_target.csv)",
              "- [比較検証JSON](comparison.json)"]
    for key in METHODS:
        if key in results:
            lines.append(f"- [{METHODS[key][0]}：詳細レポート]({key}/evaluation/report.html)")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tr = []
    for row in rows:
        bar_width = 450 * row["attack_power"] / 2
        tr.append("<tr><th>" + html.escape(row["label"]) + "</th>" +
                  "".join(f"<td>{row[m]:.6f}</td>" for m in ("mia_power", "aia_power", "attack_power")) +
                  f'<td><div class="bar" style="width:{bar_width:.2f}px"></div></td>' +
                  f'<td><a href="{row["method"]}/evaluation/report.html">詳細</a></td></tr>')
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4方式のprelim比較</title><style>body{font:16px system-ui;margin:32px;max-width:1200px;color:#172338;background:#f7f9fc}table{border-collapse:collapse;background:white;width:100%}th,td{padding:14px;border-bottom:1px solid #d9e1ec;text-align:right}th:first-child{text-align:left}.bar{height:18px;background:#3977be;max-width:100%}a{color:#175cac}p{line-height:1.8}.note{padding:18px;background:#e9f0f9;border-left:4px solid #3977be}</style><h1>4方式の予備戦実データ比較</h1><p>team24 → 他20標的。候補1,049人／AIA 6,520行。予備戦の固定公式採点。</p><p class="note">採点前に設定を固定し、正解は採点側のみで使用しました。③は固定カーネルGen-LRA変種です。この結果は本戦での未知データ性能の推定ではありません。</p><table><thead><tr><th>手法</th><th>MIA力</th><th>AIA力</th><th>合計（0〜2）</th><th>合計</th><th>標的別</th></tr></thead><tbody>' + ''.join(tr) + '</tbody></table><p>①と公式のAIA、②と公式のMIA、②と③のAIAは一致を検証しています。各方式は使用属性・推定器も異なるため、すべての差を単一の工夫に帰属することはできません。</p><p><a href="comparison.csv">比較CSV</a> · <a href="by_target.csv">標的別CSV</a> · <a href="protocol.json">事前固定条件</a> · <a href="SUMMARY.md">評価条件の詳細</a></p></html>'
    (output / "report.html").write_text(page, encoding="utf-8")
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="prelim_fixed_20261008_v1")
    parser.add_argument("--workers", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("--name must be a plain directory name")
    keys = list(dict.fromkeys(args.methods))
    output = new_output(OUTPUT_ROOT / args.name)
    original_verification = verify_bundle()
    original_algorithms = algorithm_hashes()
    protocol = {"created_utc": utc_now(), "round": "prelim-attack-20260825", "attacker_id": 24,
                "selection": "one fixed configuration per method, before evaluating target truths",
                "target_truth_used_for_fit_or_tuning": False, "methods": keys,
                "source_sha256": original_algorithms, "evaluator_sha256": code_hashes(),
                "input_verification": original_verification,
                "configs": {key: read_json(HERE / key / "config.json") for key in keys if key != "baseline"},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
                "workers": args.workers, "note": "Prelim is retrospective and previously used in this project; no claim of untouched holdout."}
    write_json(output / "protocol.json", protocol)
    export_public(output / "public")
    print("Frozen protocol: " + str(output / "protocol.json"), flush=True)

    def execute(key):
        start = time.perf_counter()
        destination = output / key
        run_attack(output / "public", METHODS[key][1], destination / "predictions")
        result = save_score(destination / "predictions", destination / "evaluation")
        result["exploration_elapsed_seconds"] = time.perf_counter() - start
        write_json(destination / "timing.json", {"elapsed_seconds": result["exploration_elapsed_seconds"],
                                                  "completed_utc": utc_now()})
        return result

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, key): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
                print(key + ": " + json.dumps(results[key]["summary"]), flush=True)
            except Exception as error:
                write_json(output / "failure.json", {"method": key, "error": repr(error), "utc": utc_now()})
                raise
    if verify_bundle() != original_verification or algorithm_hashes() != original_algorithms:
        raise AssertionError("Inputs or attack source changed during comparison")
    if code_hashes() != protocol["evaluator_sha256"]:
        raise AssertionError("Evaluator source changed during comparison")
    document = write_comparison(output, results, protocol)
    print("Controlled checks: " + json.dumps(document["controlled_comparison_checks"]), flush=True)
    print("Report: " + str(output / "report.html"), flush=True)


if __name__ == "__main__":
    main()
