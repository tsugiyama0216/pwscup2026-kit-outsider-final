"""Compare a new saved candidate with a frozen reference on identical panels."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import MODELS, check_panel, hashes, new_output, valid_name
from .joint_report import validate_target_frame
from .provenance import evaluation_signature, model_signature, generator_signature
from .ranking_metric import compute_metrics
from sugiyama_v2.io import sha, save_json, csv


def collect(paths, candidate, reference):
    if candidate == reference:
        raise ValueError("Candidate and reference must differ")
    rows, inputs, seeds, target_deltas = [], [], set(), []
    comparison_signature = None
    for path in paths:
        panel, state, _ = check_panel(path)
        meta = state["case_metadata"]
        signature = {"evaluation": evaluation_signature(state["config"]),
                     "generator": generator_signature(meta), "models": {}}
        seed = meta["seed"]
        if seed in seeds:
            raise ValueError("Duplicate scenario seed")
        seeds.add(seed)
        frames = {}
        for model in (candidate, reference):
            valid_name(model)
            directory = panel / "scores" / model
            value = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            diagnostic = json.loads((panel / "predictions" / model / "diagnostics.json").read_text(encoding="utf-8"))
            identity = model_signature(diagnostic)
            signature["models"][model] = identity
            if value.get("evaluation_signature") != signature["evaluation"] or value.get("model_signature") != identity:
                raise ValueError("Missing or inconsistent evaluation/model identity")
            if value["runtime_versions"] != state["versions"]:
                raise ValueError("Scoring runtime differs from prepared panel")
            if (value["model_id"] != model or value["public_sha256"] != state["public_sha256"]
                    or value["truth_sha256"] != state["truth_sha256"]
                    or value["prediction_sha256"] != hashes(panel / "predictions" / model)
                    or value["by_target_sha256"] != sha(directory / "by_target.csv")):
                raise ValueError("Mismatched or modified saved comparison")
            frame = pd.read_csv(directory / "by_target.csv", float_precision="round_trip").set_index("target_k").sort_index()
            validate_target_frame(frame, float(value["leaderboard"]["score_1"]))
            frames[model] = frame
        if comparison_signature is None:
            comparison_signature = signature
        elif comparison_signature != signature:
            raise ValueError("Different candidate, reference, scorer or generator versions across scenarios")
        a, b = frames[candidate], frames[reference]
        if not a.index.equals(b.index):
            raise ValueError("Different target sets")
        a_metric = compute_metrics(a.reset_index().to_dict("records"))
        b_metric = compute_metrics(b.reset_index().to_dict("records"))
        delta = (a.mia_all + a.aia_power) - (b.mia_all + b.aia_power)
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite difference")
        deletions = [float(delta.drop(k).mean()) for k in delta.index]
        if meta["reserved_seed_role"] == "confirmation":
            target_deltas.append(delta)
        rows.append({"seed": seed, "role": meta["reserved_seed_role"],
                     "candidate_score": a_metric["stable_index"], "reference_score": b_metric["stable_index"],
                     "delta": float(delta.mean()), "mia_all_delta": float((a.mia_all - b.mia_all).mean()),
                     "aia_delta": float((a.aia_power - b.aia_power).mean()),
                     "candidate_rare_lift": a_metric["rare_lift"], "reference_rare_lift": b_metric["rare_lift"],
                     "rare_lift_delta": a_metric["rare_lift"] - b_metric["rare_lift"],
                     "candidate_official_simulation_score": float(a.attack_power.mean()),
                     "reference_official_simulation_score": float(b.attack_power.mean()),
                     "official_simulation_delta": float((a.attack_power - b.attack_power).mean()),
                     "omit_one_target_min_delta": min(deletions), "omit_one_target_max_delta": max(deletions)})
        inputs.append({"panel": str(panel), "protocol_sha256": meta["protocol_sha256"],
                       "case_manifest_sha256": state["case_snapshot"]["manifest"]})
    if len({r["protocol_sha256"] for r in inputs}) != 1:
        raise ValueError("Do not combine different generation protocols")
    confirmation = [r for r in rows if r["role"] == "confirmation"]
    if not confirmation:
        raise ValueError("Include confirmation scenarios")
    delta = float(np.mean([r["delta"] for r in confirmation]))
    mean_target_deltas = pd.concat(target_deltas, axis=1).mean(axis=1)
    paired_deletions = [{"omitted_target": int(k), "delta": float(mean_target_deltas.drop(k).mean())}
                        for k in mean_target_deltas.index]
    return {"candidate": candidate, "reference": reference, "inputs": inputs, "scenarios": rows,
            "metric": "mean(mia_all + aia_power); fixed stable-index-v1",
            "comparison_signatures": comparison_signature,
            "confirmation_mean_delta": delta,
            "confirmation_mean_leave_one_target_out": paired_deletions,
            "confirmation_case_delta_range": [min(r["delta"] for r in confirmation), max(r["delta"] for r in confirmation)],
            "candidate_wins_all_confirmation_scenarios": all(r["delta"] > 1e-12 for r in confirmation),
            "candidate_wins_all_confirmation_target_deletions": all(r["omit_one_target_min_delta"] > 1e-12 for r in confirmation),
            "official_rank_prediction": None,
            "scope": "Paired fixed-index synthetic comparison; known-reference agreement does not validate unseen-model official ranks.",
            "limitations": ["No actual hidden membership, AIA truth or official floors are available.",
                            "The fixed index omits the rare-subgroup MIA uplift; rare-only improvements need separate assessment.",
                            "All generated scenario scores and losses are retained; no model or seed is selected by this command.",
                            "Repeated synthetic seeds and target deletions are not independent population samples."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", choices=MODELS, default="Rv2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parent / "generation_v5_protocol.json")
    args = parser.parse_args()
    report = collect(args.panels, args.candidate, args.reference)
    from .stable_report import build_report
    validation, _ = build_report(args.panels, Path(__file__).parent / "evidence/official_totals_20261009.json", args.protocol)
    report["known_reference_rank_reproduced"] = validation["known_reference_rank_reproduced"]
    report["reference_validation"] = validation
    report["local_comparison_status"] = (
        "candidate_preferred_on_fixed_panels" if report["known_reference_rank_reproduced"]
        and report["candidate_wins_all_confirmation_scenarios"]
        and report["candidate_wins_all_confirmation_target_deletions"]
        else "reference_preferred_on_fixed_panels" if report["known_reference_rank_reproduced"]
        and all(r["omit_one_target_max_delta"] < -1e-12 for r in report["scenarios"] if r["role"] == "confirmation")
        else "mixed_or_unvalidated")
    output = new_output(args.output)
    save_json(output / "result.json", report)
    csv(output / "comparison.csv", pd.DataFrame(report["scenarios"]))
    body = ''.join(f'<tr><td>{r["seed"]}</td><td>{escape(r["role"])}</td><td>{r["candidate_score"]:.6f}</td><td>{r["reference_score"]:.6f}</td><td>{r["delta"]:+.6f}</td><td>{r["omit_one_target_min_delta"]:+.6f} ～ {r["omit_one_target_max_delta"]:+.6f}</td></tr>' for r in report["scenarios"])
    status = "一致" if report["known_reference_rank_reproduced"] else "未達成"
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>候補モデル比較</title><style>body{font-family:system-ui,sans-serif;margin:40px auto;padding:0 20px;max-width:1080px;color:#18283c}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #dde3ec;text-align:right}td:first-child,th:first-child{text-align:left}.notice{padding:16px;background:#fff4dc}</style>'
    html += f'<h1>{escape(args.candidate)} と {escape(args.reference)} の比較</h1><p class="notice">既知4モデルの確認平均と本番順位の一致：{status}。新候補の本番順位の正しさは、まだ確認できていません。</p>'
    html += '<p>比較指標は全体MIA＋補正済みAIAです。公式方式の合計と希少者MIAの上乗せはCSV/JSONに別保存します。希少者だけの改善はこの指標で判断できません。</p>'
    html += f'<p>確認用ケースの平均差：<strong>{report["confirmation_mean_delta"]:+.6f}</strong>（正なら候補が上）。対象を一つ除いた範囲は感度分析であり、信頼区間ではありません。</p>'
    html += f'<table><thead><tr><th>ケース</th><th>用途</th><th>候補</th><th>基準</th><th>差</th><th>対象を一つ除く差の範囲</th></tr></thead><tbody>{body}</tbody></table><p>入力・予測・正解・補正値のハッシュ、成分別差、既知モデルとの照合は result.json に保存しています。</p></html>'
    (output / "report.html").write_text(html, encoding="utf-8")
    print(json.dumps({"confirmation_mean_delta": report["confirmation_mean_delta"], "official_rank_prediction": None, "output": str(output)}))


if __name__ == "__main__":
    main()
