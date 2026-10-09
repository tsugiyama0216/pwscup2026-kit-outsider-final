"""Compare three additional frozen attacks with the previous seven models."""
from __future__ import annotations

import argparse
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .candidate_report import collect
from .stable_report import build_report as validate_references
from .panel import new_output
from .series_binding import validate_baseline_binding
from sugiyama_v2.io import csv, save_json

ADDITIONAL = ("Rv3", "Mv1", "Mv2")
DERIVED = tuple(f"Rv{i}" for i in range(6, 12))
SERIES = (*ADDITIONAL, *DERIVED)
MODELS = ("Rv2", *SERIES)
TOLERANCE = 1e-12


def validate_reference_binding(references, comparison):
    """Bind every comparison to the reference snapshot read for this report."""
    expected, actual = references["comparison_signatures"], comparison["comparison_signatures"]
    if (any(expected[key] != actual[key] for key in ("evaluation", "generator"))
            or expected["models"]["Rv2"] != actual["models"]["Rv2"]):
        raise ValueError("Candidate comparison differs from the reference evaluation identity")
    manifests = {str(Path(p["path"]).resolve()): (p["protocol_sha256"], p["case_manifest_sha256"])
                 for p in references["panels"]}
    observed = {str(Path(p["panel"]).resolve()): (p["protocol_sha256"], p["case_manifest_sha256"])
                for p in comparison["inputs"]}
    if len(observed) != len(comparison["inputs"]) or observed != manifests:
        raise ValueError("Candidate comparison differs from the reference case snapshots")


def rank_rows(rows):
    """Use shared ranks for numerical ties instead of ordering by model name."""
    if len(rows) != len(MODELS) or {r["model"] for r in rows} != set(MODELS):
        raise ValueError("Require Rv2 and all nine fixed candidate models")
    if not all(np.isfinite(r["index"]) for r in rows):
        raise ValueError("Nonfinite ranking index")
    return sorted((dict(r, rank=1 + sum(s["index"] > r["index"] + TOLERANCE for s in rows))
                   for r in rows), key=lambda r: (r["rank"], MODELS.index(r["model"])))


def build_report(paths):
    references, _ = validate_references(paths)
    comparisons = {model: collect(paths, model, "Rv2") for model in SERIES}
    for model, comparison in comparisons.items():
        validate_reference_binding(references, comparison)
        for item in comparison["inputs"]:
            panel = Path(item["panel"])
            diagnostic = json.loads((panel / "predictions" / model / "diagnostics.json").read_text(encoding="utf-8"))
            if model in DERIVED:
                validate_baseline_binding(panel / "predictions" / "Rv2", diagnostic)
    seeds = tuple(p["seed"] for p in references["panels"])
    baseline = references["confirmation_mean_metrics"]["Rv2"]
    reference_by_seed = {p["seed"]: p["model_metrics"]["Rv2"] for p in references["panels"]}
    rows = [{"model": "Rv2", "index": baseline["stable_index"], "delta_from_Rv2": 0.,
             "mia_all": baseline["mean_mia_all"], "aia_power": baseline["mean_aia_power"],
             "rare_lift": baseline["rare_lift"], "official_simulation_score": baseline["official_attack_power"],
             "case_delta_min": 0., "case_delta_max": 0., "paired_omit_delta_min": 0., "paired_omit_delta_max": 0.,
             "better_in_both_cases": None, "better_in_all_individual_deletions": None}]
    case_rows = [{"seed": seed, "model": "Rv2", "index": reference_by_seed[seed]["stable_index"],
                  "delta_from_Rv2": 0., "official_simulation_score": reference_by_seed[seed]["official_attack_power"]}
                 for seed in seeds]
    omissions = {"Rv2": {r["omitted_target"]: 0. for r in next(iter(comparisons.values()))["confirmation_mean_leave_one_target_out"]}}
    for model, comparison in comparisons.items():
        if comparison["comparison_signatures"]["models"][model]["kind"] != ("frozen_additional" if model in ADDITIONAL else "derived_series_batch"):
            raise ValueError("Require the verified frozen worker for each model, not a namesake plugin")
        cases = comparison["scenarios"]
        if {r["seed"] for r in cases} != set(seeds) or len(cases) != 2 or any(r["role"] != "confirmation" for r in cases):
            raise ValueError("Candidate did not use exactly the fixed two confirmation cases")
        for case in cases:
            if abs(case["reference_score"] - reference_by_seed[case["seed"]]["stable_index"]) > TOLERANCE:
                raise ValueError("Candidate comparison uses a different Rv2 reference")
            case_rows.append({"seed": case["seed"], "model": model, "index": case["candidate_score"],
                              "delta_from_Rv2": case["delta"], "official_simulation_score": case["candidate_official_simulation_score"]})
        mean = lambda key: float(np.mean([r[key] for r in cases]))
        omitted = comparison["confirmation_mean_leave_one_target_out"]
        omissions[model] = {r["omitted_target"]: r["delta"] for r in omitted}
        rows.append({"model": model, "index": mean("candidate_score"),
                     "delta_from_Rv2": comparison["confirmation_mean_delta"],
                     "mia_all": baseline["mean_mia_all"] + mean("mia_all_delta"),
                     "aia_power": baseline["mean_aia_power"] + mean("aia_delta"),
                     "rare_lift": mean("candidate_rare_lift"),
                     "official_simulation_score": mean("candidate_official_simulation_score"),
                     "case_delta_min": min(r["delta"] for r in cases), "case_delta_max": max(r["delta"] for r in cases),
                     "paired_omit_delta_min": min(r["delta"] for r in omitted), "paired_omit_delta_max": max(r["delta"] for r in omitted),
                     "better_in_both_cases": comparison["candidate_wins_all_confirmation_scenarios"],
                     "better_in_all_individual_deletions": comparison["candidate_wins_all_confirmation_target_deletions"]})
    ranking = rank_rows(rows)
    by_model = {r["model"]: r for r in rows}
    by_case = {(r["seed"], r["model"]): r for r in case_rows}
    pairs = []
    for a, b in itertools.combinations(MODELS, 2):
        case_deltas = [by_case[(s,a)]["index"] - by_case[(s,b)]["index"] for s in seeds]
        deletion_deltas = [omissions[a][k] - omissions[b][k] for k in omissions[a]]
        pairs.append({"model_a": a, "model_b": b, "mean_a_minus_b": by_model[a]["index"] - by_model[b]["index"],
                      "case_min": min(case_deltas), "case_max": max(case_deltas),
                      "paired_omit_min": min(deletion_deltas), "paired_omit_max": max(deletion_deltas)})
    return {"format": "frozen-rv3-mv1-mv2-extended-v5-comparison-v1", "seeds": list(seeds),
            "ranking": ranking, "case_metrics": case_rows, "pair_sensitivity": pairs,
            "comparisons_against_Rv2": comparisons, "reference_validation": references,
            "known_reference_rank_reproduced": references["known_reference_rank_reproduced"],
            "indicator": "mean across both fixed cases of mean across 22 targets of mia_all + aia_power",
            "official_rank_prediction": None, "official_scores_measured_for_new_series": False,
            "model_tuning_performed": False, "official_submissions_performed": False,
            "interpretation": "Local selection comparison of nine frozen hypotheses; selecting the maximum is not independent confirmation of future official performance.",
            "limitations": ["The same limited synthetic population and public-profile gaps remain.",
                            "The rank index omits rare MIA uplift; the official-formula score and uplift are retained separately.",
                            "Target deletion ranges are sensitivity checks, not confidence intervals.",
                            "These v5 cases validated the metric using other known models; once used to select among these nine candidates, they are selection evidence for this series."]}


def write_html(path, report):
    h = lambda value: escape(str(value))
    reference_status = ("既知4モデルの本番順位は、この評価条件で再現済みです。"
                        if report["known_reference_rank_reproduced"] else
                        "既知4モデルの本番順位の再現は未達成です。この条件での候補順位を本番順位の推定に採用しないでください。")
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Rv3・Mv1・Mv2と前回7モデル 自己採点</title><style>body{font-family:system-ui,sans-serif;max-width:1160px;margin:36px auto;padding:0 20px;color:#172a3e;background:#f5f7fa}section{padding:24px;background:white;border-radius:12px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{padding:11px;border-bottom:1px solid #dde3ec;text-align:right}td:nth-child(2),th:nth-child(2){text-align:left}.scroll{overflow-x:auto}.notice{background:#fff3d9;padding:16px}h1{font-size:26px}</style>'
    html += '<h1>Rv3・Mv1・Mv2と前回7モデル の自己採点</h1><p class="notice">今回固定した順位比較指標で、同じ2ケース・22対象・同じ正解と補正値を使用。高いほど良い指標です。本番総合点の予測値ではありません。</p><section><h2>2ケースの平均</h2><div class="scroll"><table><tr><th>順位</th><th>モデル</th><th>順位比較指標</th><th>Rv2との差</th><th>全体MIA</th><th>補正済AIA</th><th>希少MIA上乗せ</th><th>公式式の模擬合計</th></tr>'
    for r in report["ranking"]:
        html += f'<tr><td>{r["rank"]}</td><td>{h(r["model"])}</td><td>{r["index"]:.6f}</td><td>{r["delta_from_Rv2"]:+.6f}</td><td>{r["mia_all"]:.6f}</td><td>{r["aia_power"]:.6f}</td><td>{r["rare_lift"]:.6f}</td><td>{r["official_simulation_score"]:.6f}</td></tr>'
    html += '</table></div></section><section><h2>Rv2との差の変動</h2><p>正なら候補が上。範囲は信頼区間ではありません。差が同点のときは改善と扱いません。</p><table><tr><th>モデル</th><th>ケース間の差の範囲</th><th>両ケースから同じ対象を1つ外す差の範囲</th><th>両ケースで改善</th></tr>'
    for r in report["ranking"]:
        if r["model"] == "Rv2": continue
        html += f'<tr><td>{h(r["model"])}</td><td>{r["case_delta_min"]:+.6f} ～ {r["case_delta_max"]:+.6f}</td><td>{r["paired_omit_delta_min"]:+.6f} ～ {r["paired_omit_delta_max"]:+.6f}</td><td>{"はい" if r["better_in_both_cases"] else "いいえ"}</td></tr>'
    html += '</table></section><section><h2>解釈と証跡</h2><p>' + h(reference_status) + '今回の候補モデルの本番順位は未測定です。9候補から良いものを選ぶ今回の結果は候補選択の根拠であり、選んだ候補に対する独立した本番精度の検証ではありません。</p><p>公開特徴の不足11/22対象と、同じ合成母集団を使う制約が残ります。希少者MIAだけの改善は順位比較指標に入らないため、別欄と公式式の模擬合計も併記しています。</p><p>モデルの再調整・公式提出は行っていません。全ケース値・全45ペアの差・対象除外・入力とモデルの署名はJSON/CSVに保存しています。</p></section></html>'
    Path(path).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.panels)
    out = new_output(args.output)
    save_json(out / "result.json", report)
    for name, key in (("ranking", "ranking"), ("case_metrics", "case_metrics"), ("pair_sensitivity", "pair_sensitivity")):
        csv(out / (name + ".csv"), pd.DataFrame(report[key]))
    write_html(out / "report.html", report)
    print(json.dumps({"ranking": report["ranking"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
