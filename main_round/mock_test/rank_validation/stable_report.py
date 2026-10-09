"""Frozen v5 stable-index reference check, with all scenario sensitivities.

The primary rule was frozen before v5 generation. This report does not change
the original official-score joint_report gate or validate unseen candidates.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import joint_report, generation_v5, ranking_metric
from .panel import MODELS, new_output
from sugiyama_v2.io import csv, save_json, sha
from sugiyama_v2.scoring import TARGET_IDS

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/v5"
DEFAULT_METRIC = EVIDENCE / "stable_index_protocol.json"
DEFAULT_FREEZE = EVIDENCE / "freeze_record.json"
DEFAULT_OFFICIAL = HERE / "evidence/official_totals_20261009.json"
METRIC_FIELDS = ("stable_index", "rare_lift", "official_attack_power",
                 "mean_mia_all", "mean_mia_power", "mean_aia_power")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _official_intervals(source):
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != len(MODELS):
        raise ValueError("Expected exactly four official model rows")
    official = {}
    for row in rows:
        name = row.get("model_id")
        value, digits = row.get("score_1_display"), row.get("decimal_places")
        if name in official or name not in MODELS:
            raise ValueError("Duplicate or unknown official model")
        if type(digits) is not int or not 0 <= digits <= 12:
            raise ValueError("Invalid official display precision")
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 2:
            raise ValueError("Invalid official score")
        half = .5 * 10 ** -digits
        official[name] = (value - half, value + half)
    check = joint_report.compare_scores({m: sum(v) / 2 for m, v in official.items()}, official)
    if check["total"] != 6:
        raise ValueError("Official evidence must establish all six strict reference directions")
    return official


def _rank_check(values, official):
    check = joint_report.compare_scores(values, official)
    check["all_six_correct"] = check["total"] == 6 and check["all_correct"]
    return check


def _score_frames(frames, omit=None):
    expected = tuple(k for k in TARGET_IDS if k != omit)
    answers = {}
    for name in MODELS:
        frame = frames[name] if omit is None else frames[name].drop(omit)
        answers[name] = ranking_metric.compute_metrics(
            frame.reset_index().to_dict("records"), expected_target_ids=expected)
    return answers


def _target_details(frames):
    result = {}
    for name in MODELS:
        records = []
        if not {"mia_rare", "mia_positive", "mia_rare_positive"}.issubset(frames[name].columns):
            raise ValueError("Missing retained rare/count diagnostics")
        for row in frames[name].reset_index().to_dict("records"):
            original_counts = [row["mia_positive"], row["mia_rare_positive"]]
            try:
                counts = np.asarray(original_counts, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid retained positive counts") from exc
            if (counts.shape != (2,) or any(isinstance(v, (bool, np.bool_)) for v in original_counts)
                    or not np.isfinite(counts).all() or (counts != np.floor(counts)).any()
                    or (counts < 0).any() or (counts > 1049).any()):
                raise ValueError("Invalid retained positive counts")
            if counts[1] > counts[0]:
                raise ValueError("Rare positives exceed all positives")
            rare = row["mia_rare"]
            if not pd.isna(rare) and (isinstance(rare, bool) or not np.isfinite(float(rare)) or not 0 <= float(rare) <= 1):
                raise ValueError("Invalid retained rare TPR")
            row = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            row.update(stable_index=row["mia_all"] + row["aia_power"],
                       rare_lift=row["mia_power"] - row["mia_all"],
                       official_simulation_score=row["mia_power"] + row["aia_power"])
            records.append(row)
        result[name] = records
    return result


def _scores(metrics, key):
    return {name: metrics[name][key] for name in MODELS}


def _mean_metrics(cases):
    return {name: {key: float(np.mean([case["model_metrics"][name][key] for case in cases]))
                   for key in METRIC_FIELDS} for name in MODELS}


def summarize_cases(cases, official, expected_seeds):
    """Pure aggregation after the caller has checked saved provenance.

    Cases contain seed, frames and role. Useful for deterministic regression
    tests of mean-first, tie, completeness and shared-target deletion behavior.
    """
    expected_seeds = tuple(expected_seeds)
    if len(expected_seeds) != 2 or len(set(expected_seeds)) != 2:
        raise ValueError("The frozen v5 metric requires exactly two distinct confirmation seeds")
    seeds = [case["seed"] for case in cases]
    if len(seeds) != 2 or len(set(seeds)) != 2 or set(seeds) != set(expected_seeds):
        raise ValueError("Both reserved confirmation cases must be present exactly once")
    if any(case.get("role") != "confirmation" for case in cases):
        raise ValueError("Only the two fixed confirmation cases enter the v5 primary mean")
    if any(set(case["frames"]) != set(MODELS) for case in cases):
        raise ValueError("Each case requires the same four reference models")
    panels, omitted_cache = [], {}
    for case in sorted(cases, key=lambda item: item["seed"]):
        metrics = _score_frames(case["frames"])
        omissions = []
        for target in TARGET_IDS:
            values = _score_frames(case["frames"], omit=target)
            omissions.append({"omitted_target": target, "model_metrics": values,
                              "rank_check": _rank_check(_scores(values, "stable_index"), official)})
        omitted_cache[case["seed"]] = {r["omitted_target"]: r for r in omissions}
        panels.append({"seed": case["seed"], "role": case["role"], "model_metrics": metrics,
                       "scores": _scores(metrics, "stable_index"),
                       "by_target": _target_details(case["frames"]),
                       "rank_check": _rank_check(_scores(metrics, "stable_index"), official),
                       "raw_rank_check": _rank_check(_scores(metrics, "official_attack_power"), official),
                       "leave_one_target_out": omissions})
    means = _mean_metrics(panels)
    mean_check = _rank_check(_scores(means, "stable_index"), official)
    paired_omissions = []
    for target in TARGET_IDS:
        omitted_cases = [omitted_cache[case["seed"]][target] for case in panels]
        values = _mean_metrics(omitted_cases)
        paired_omissions.append({"omitted_target": target, "target_count_per_case": 21,
                                "case_count": 2, "model_metrics": values,
                                "rank_check": _rank_check(_scores(values, "stable_index"), official)})
    pairs = []
    for pair in mean_check["pairs"]:
        a, b = pair["model_a"], pair["model_b"]
        differences = {str(case["seed"]): case["scores"][a] - case["scores"][b] for case in panels}
        pairs.append({**pair, "case_differences": differences,
                      "case_difference_min": min(differences.values()),
                      "case_difference_max": max(differences.values()),
                      "sensitivity_range_is_confidence_interval": False})
    case_hits = sum(p["rank_check"]["all_six_correct"] for p in panels)
    return {"panels": panels, "confirmation_mean_metrics": means,
            "confirmation_mean_scores": _scores(means, "stable_index"),
            "fresh_confirmation_mean_rank_check": mean_check,
            "confirmation_mean_rank_check": mean_check,
            "known_reference_rank_reproduced": mean_check["all_six_correct"],
            "all_reserved_scenarios_present": True,
            "individual_case_all_six_count": case_hits, "confirmation_case_count": 2,
            "individual_case_all_six_fraction": case_hits / 2,
            "both_confirmation_cases_match_all_six": case_hits == 2,
            "pair_case_sensitivity": pairs,
            "same_target_leave_one_out": paired_omissions,
            "paired_target_deletion_all_six_count": sum(p["rank_check"]["all_six_correct"] for p in paired_omissions),
            "all_paired_target_deletions_match_known_order": all(p["rank_check"]["all_six_correct"] for p in paired_omissions)}


def _verify_protocol(protocol_path, metric_path, freeze_path):
    protocol_path, metric_path, freeze_path = map(Path, (protocol_path, metric_path, freeze_path))
    protocol = generation_v5.load_protocol()
    if sha(protocol_path) != generation_v5.PROTOCOL_SHA256 or _read(protocol_path) != protocol:
        raise ValueError("Use the unchanged frozen v5 generation protocol")
    metric, freeze = _read(metric_path), _read(freeze_path)
    if sha(metric_path) != protocol["metric_protocol_sha256"] or metric != protocol["metric_validation"]:
        raise ValueError("Metric protocol bytes/content differ from the generation-time freeze")
    if metric["version"] != ranking_metric.METRIC_VERSION:
        raise ValueError("Metric implementation version differs from the fixed protocol")
    seeds = protocol["confirmation_seeds"]
    if protocol["development_seeds"] or len(seeds) != 2 or len(set(seeds)) != 2 or metric["confirmation_seeds"] != seeds:
        raise ValueError("Unexpected fixed v5 seed contract")
    if (freeze["protocol_sha256"] != sha(protocol_path)
            or freeze["metric_protocol_sha256"] != sha(metric_path)
            or freeze["frozen_utc"] != protocol["frozen_utc"]
            or metric["frozen_utc"] != protocol["frozen_utc"]
            or freeze["confirmation_seeds"] != seeds or freeze["development_seeds"]
            or freeze["v5_generation_started"] is not False
            or freeze["new_seed_model_results_exist"] is not False
            or freeze["prior_case_manifest_seed_matches"] != 0):
        raise ValueError("Missing or inconsistent pre-generation freeze evidence")
    if freeze["generator_source_sha256"] != sha(generation_v5.__file__):
        raise ValueError("Frozen v5 generator source changed")
    return protocol, metric, freeze


def build_report(paths, official_path=DEFAULT_OFFICIAL, protocol_path=HERE / "generation_v5_protocol.json",
                 *, metric_protocol_path=DEFAULT_METRIC, freeze_path=DEFAULT_FREEZE):
    """Return (report, case_rows), preserving the joint_report calling convention."""
    paths = [Path(path).resolve() for path in paths]
    protocol, metric, freeze = _verify_protocol(protocol_path, metric_protocol_path, freeze_path)
    evidence_paths = {"generation_protocol": Path(protocol_path), "metric_protocol": Path(metric_protocol_path),
                      "freeze_record": Path(freeze_path), "official_feedback": Path(official_path),
                      "metric_implementation": Path(ranking_metric.__file__), "report_implementation": Path(__file__)}
    before = {name: sha(path) for name, path in evidence_paths.items()}
    official_source = _read(official_path)
    official = _official_intervals(official_source)
    actual_order = sorted(official, key=lambda m: -sum(official[m]))
    if actual_order != metric["known_reference_order_best_first"] or set(metric["reference_models"]) != set(MODELS):
        raise ValueError("Official reference order differs from the preregistered metric")
    raw, _ = joint_report.build_report(paths, official_path, protocol_path)
    if not raw["all_reserved_scenarios_present"]:
        raise ValueError("Incomplete reserved v5 scenarios; a primary result cannot be reported")
    cases, rows, sources = [], [], []
    raw_panels = {p["seed"]: p for p in raw["panels"]}
    for path in paths:
        state, _, frames = joint_report.load_panel(path)
        if state["floor_repetitions"] != 1000:
            raise ValueError("Reference confirmation requires the full 1000 floor repetitions")
        meta, expected = state["case_metadata"], raw_panels[state["case_metadata"]["seed"]]
        if (meta["generator_source_sha256"] != freeze["generator_source_sha256"]
                or state["comparison_signatures"] != raw["comparison_signatures"]
                or state["case_snapshot"]["manifest"] != expected["case_manifest_sha256"]):
            raise ValueError("Panel identity changed during report construction")
        for model, frame in frames.items():
            if abs(float(frame.attack_power.mean()) - expected["scores"][model]) > 1e-12:
                raise ValueError("Raw scores changed while building stable report")
        cases.append({"seed": meta["seed"], "role": meta["reserved_seed_role"], "frames": frames})
        sources.append({"path": str(path), "seed": meta["seed"],
                        "case_manifest_sha256": state["case_snapshot"]["manifest"],
                        "public_sha256": state["public_sha256"], "truth_sha256": state["truth_sha256"],
                        "score_result_sha256": {m: sha(path / "scores" / m / "result.json") for m in MODELS},
                        "prediction_sha256": {m: joint_report.hashes(path / "predictions" / m) for m in MODELS}})
    report = summarize_cases(cases, official, protocol["confirmation_seeds"])
    for panel in report["panels"]:
        saved = raw_panels[panel["seed"]]
        panel.update({key: saved[key] for key in ("path", "protocol_sha256", "case_manifest_sha256",
                      "public_profile_coverage", "public_profile_family_counts")})
        for model in MODELS:
            values = panel["model_metrics"][model]
            rows.append({"panel": panel["path"], "seed": panel["seed"], "role": panel["role"],
                         "model": model, "score": values["stable_index"], "mia": values["mean_mia_all"],
                         "aia": values["mean_aia_power"], "official_simulation_score": values["official_attack_power"],
                         **{key: values[key] for key in METRIC_FIELDS}})
    coverage = [p["public_profile_coverage"]["model_ranking"]["unsupported"] for p in report["panels"]]
    if not all(type(value) is bool for value in coverage):
        raise ValueError("Missing public-profile coverage verdict")
    report.update({"format": "fixed-v5-stable-index-reference-check-v1",
                   "primary_indicator": "equal_case_mean(equal_target_mean(mia_all + aia_power_after_floor))",
                   "primary_gate": "Both frozen confirmation cases present; mean stable index matches all six strict official reference directions; ties fail.",
                   "official_source": official_source, "protocol_sha256": sha(protocol_path),
                   "metric_protocol": metric, "metric_protocol_sha256": sha(metric_protocol_path),
                   "metric_source_sha256": sha(ranking_metric.__file__),
                   "report_source_sha256": sha(__file__),
                   "freeze_record": freeze, "freeze_record_sha256": sha(freeze_path),
                   "comparison_signatures": raw["comparison_signatures"],
                   "input_provenance": sources,
                   "source_sha256": before,
                   "raw_known_reference_rank_reproduced": raw["known_reference_rank_reproduced"],
                   "raw_official_simulation_report": raw,
                   "public_profile_coverage_insufficient": any(coverage),
                   "new_model_ranking_accuracy_validated": False,
                   "official_score_prediction": False,
                   "official_hidden_truth_available": False,
                   "independent_population": False,
                   "v4_was_retrospective_metric_design_data": True,
                   "limitations": [
                       "The stable index is a separate diagnostic, not the official competition score.",
                       "The metric was designed after v4 known-model reversals; v4 is retrospective design evidence.",
                       "The two v5 seeds reuse holdout donors and population construction; they are not independent unseen people.",
                       "Matching the four known official models does not validate future unknown-model rankings.",
                       "Rare MIA max uplift can represent real attack strength; removing it may hide such strength, so it is shown separately.",
                       "Case ranges and target deletion are sensitivities, not confidence intervals.",
                       "Public-profile coverage failures remain unsupported and are not relaxed by a successful known-rank check."]})
    if before != {name: sha(path) for name, path in evidence_paths.items()}:
        raise ValueError("Protocol, metric, report source or official evidence changed during reporting")
    return report, rows


def write_html(path, report, rows):
    h = lambda value: escape(str(value))
    check = report["fresh_confirmation_mean_rank_check"]
    status = "既知4モデルの平均順位が6組すべて一致" if report["known_reference_rank_reproduced"] else "既知4モデルの平均順位一致は未達成"
    observed = {r["model_id"]: r for r in report["official_source"]["rows"]}
    body = []
    for model in check["order"]:
        m, real = report["confirmation_mean_metrics"][model], observed[model]
        body.append(f'<tr><td>{h(model)}</td><td>{real["score_1_display"]:.4f}</td><td>{m["stable_index"]:.6f}</td>'
                    f'<td>{m["mean_mia_all"]:.6f}</td><td>{m["mean_aia_power"]:.6f}</td>'
                    f'<td>{m["rare_lift"]:.6f}</td><td>{m["official_attack_power"]:.6f}</td></tr>')
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>固定指標と本番順位の照合</title>'
    html += '<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;background:#f5f7fa;color:#15273b}h1{font-size:26px}section{background:white;border-radius:12px;padding:22px;margin:20px 0}table{border-collapse:collapse;width:100%}td,th{text-align:right;padding:9px;border-bottom:1px solid #dde3ec}td:first-child,th:first-child{text-align:left}.notice{padding:18px;border-left:5px solid #a36717;background:#fff3da}.scroll{overflow-x:auto}summary{cursor:pointer}</style>'
    html += f'<h1>固定した順位比較指標と本番順位の照合</h1><p class="notice"><strong>{h(status)}</strong>：{check["correct"]}/6 組。未知モデルの本番順位を予測できるかは未検証です。</p>'
    html += '<p>順位比較指標 = 全体 MIA の TPR と補正後 AIA の合計を22対象で平均し、予約済みの確認用2ケースを等しく平均。希少 MIA の上乗せを別欄に示します。指標の絶対値は本番総合点と同じ意味ではありません。</p>'
    html += '<section><h2>本番実測と確認用2ケースの平均</h2><div class="scroll"><table><thead><tr><th>モデル</th><th>本番総合（4桁）</th><th>順位比較指標</th><th>全体 MIA</th><th>補正後 AIA</th><th>希少 MIA 上乗せ</th><th>公式式の模擬総合</th></tr></thead><tbody>'+''.join(body)+'</tbody></table></div></section>'
    html += f'<section><h2>ケース別の確認</h2><p>個別ケースで全6組が一致：{report["individual_case_all_six_count"]}/2 ケース。両ケースでの完全一致：{"あり" if report["both_confirmation_cases_match_all_six"] else "なし"}。これは平均順位の主判定とは別の感度診断です。</p>'
    for panel in report["panels"]:
        rows_here = sorted((r for r in rows if r["seed"] == panel["seed"]), key=lambda r: -r["stable_index"])
        line = ''.join(f'<tr><td>{h(r["model"])}</td><td>{r["stable_index"]:.6f}</td><td>{r["rare_lift"]:.6f}</td><td>{r["official_attack_power"]:.6f}</td></tr>' for r in rows_here)
        rank = panel["rank_check"]
        html += f'<h3>確認用 seed {panel["seed"]}：{rank["correct"]}/6 組</h3><table><tr><th>モデル</th><th>順位比較指標</th><th>希少 MIA 上乗せ</th><th>公式式の模擬総合</th></tr>{line}</table>'
    html += '</section><section><h2>モデル間の差とケースによる変動</h2><p>差は左モデル−右モデル。範囲は2ケース間の感度で、信頼区間ではありません。</p><div class="scroll"><table><tr><th>比較</th><th>ケース1の差</th><th>ケース2の差</th><th>平均差</th><th>差の範囲</th><th>本番方向と一致</th></tr>'
    for pair in report["pair_case_sensitivity"]:
        ds = list(pair["case_differences"].values())
        html += f'<tr><td>{h(pair["model_a"])} − {h(pair["model_b"])}</td><td>{ds[0]:+.6f}</td><td>{ds[1]:+.6f}</td><td>{pair["delta_a_minus_b"]:+.6f}</td><td>{pair["case_difference_min"]:+.6f} ～ {pair["case_difference_max"]:+.6f}</td><td>{"一致" if pair["agrees"] else "不一致"}</td></tr>'
    paired_count = report["paired_target_deletion_all_six_count"]
    html += f'</table></div></section><section><h2>同じ対象を両ケースから外す診断</h2><p><strong>除外後も全6組が一致：{paired_count}/22 通り。</strong>全22通りを保存しています。対象除外の結果は主判定を変更しません。</p><details><summary>22対象の除外結果を表示</summary><table><tr><th>除外対象</th><th>一致ペア数</th><th>順位比較指標の平均順位</th></tr>'
    for item in report["same_target_leave_one_out"]:
        rank = item["rank_check"]
        html += f'<tr><td>{item["omitted_target"]}</td><td>{rank["correct"]}/6</td><td>{h(" > ".join(rank["order"]))}</td></tr>'
    html += '</table></details><p>各ケース単独の22通りの除外結果と全ペア差も JSON・CSV に保存しています。</p></section>'
    coverage_counts = []
    for panel in report["panels"]:
        recorded = panel.get("public_profile_coverage", {})
        count, total = recorded.get("unsupported_target_count"), recorded.get("target_count")
        if type(count) is int and type(total) is int and 0 <= count <= total:
            coverage_counts.append(f'seed {panel["seed"]}: {count}/{total} 対象')
    coverage_detail = "、".join(coverage_counts) if coverage_counts else "対象別の数値は JSON に記録しています。"
    coverage = "不足あり。公開分布を十分に再現できていない対象があります。" if report["public_profile_coverage_insufficient"] else "範囲上の不足は検出されませんでした。ただし順位の妥当性は別問題です。"
    html += f'<section><h2>適用範囲</h2><p><strong>公開プロファイルの網羅性：{h(coverage)}</strong></p><p>{h(coverage_detail)}</p><p>同じ限られた donor 母集団と合成方法を用いる2ケースであり、独立した未知集団の検証ではありません。v4結果を見て考案した指標を、生成前に固定したv5条件で確認しています。既知4モデルの一致から、新しい候補への精度保証はできません。</p><p>公式式の従来の全ケース一致判定：{"達成" if report["raw_known_reference_rank_reproduced"] else "未達成"}。既存の判定条件と公式採点式は変更していません。</p><p>固定日時：{h(report["freeze_record"]["frozen_utc"])}。入力・モデル・採点・生成・指標の署名と、固定条件全文は result.json を参照してください。</p></section></html>'
    Path(path).write_text(html, encoding="utf-8")


def write_outputs(destination, report, rows):
    output = new_output(destination)
    save_json(output / "result.json", report)
    csv(output / "case_metrics.csv", pd.DataFrame(rows))
    observed = {r["model_id"]: r["score_1_display"] for r in report["official_source"]["rows"]}
    csv(output / "confirmation_means.csv", pd.DataFrame([
        {"model": m, "official_score_1_display": observed[m], **values}
        for m, values in report["confirmation_mean_metrics"].items()]))
    target_rows = []
    for panel in report["panels"]:
        for model, values in panel["by_target"].items():
            target_rows.extend({"seed": panel["seed"], "model": model, **row} for row in values)
    csv(output / "target_metrics.csv", pd.DataFrame(target_rows))
    pair_rows = []
    for pair in report["pair_case_sensitivity"]:
        pair_rows.append({k: v for k, v in pair.items() if k != "case_differences"}
                         | {f"delta_seed_{seed}": value for seed, value in pair["case_differences"].items()})
    csv(output / "pair_differences.csv", pd.DataFrame(pair_rows))
    joint_rows, case_rows = [], []
    for item in report["same_target_leave_one_out"]:
        for pair in item["rank_check"]["pairs"]:
            joint_rows.append({"omitted_target": item["omitted_target"], **pair})
    for case in report["panels"]:
        for item in case["leave_one_target_out"]:
            for pair in item["rank_check"]["pairs"]:
                case_rows.append({"seed": case["seed"], "omitted_target": item["omitted_target"], **pair})
    csv(output / "same_target_leave_one_out.csv", pd.DataFrame(joint_rows))
    csv(output / "case_leave_one_out.csv", pd.DataFrame(case_rows))
    write_html(output / "report.html", report, rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-feedback", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--protocol", type=Path, default=HERE / "generation_v5_protocol.json")
    parser.add_argument("--metric-protocol", type=Path, default=DEFAULT_METRIC)
    parser.add_argument("--freeze-record", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()
    report, rows = build_report(args.panels, args.official_feedback, args.protocol,
                                metric_protocol_path=args.metric_protocol, freeze_path=args.freeze_record)
    output = write_outputs(args.output, report, rows)
    print(json.dumps({"known_reference_rank_reproduced": report["known_reference_rank_reproduced"],
                      "fresh_confirmation_mean_rank_check": report["fresh_confirmation_mean_rank_check"],
                      "new_model_ranking_accuracy_validated": False, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
