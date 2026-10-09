"""Compare complete frozen joint panels with the supplied official reference rank.

Agreement on known models is a retrospective check, never a new-model accuracy
estimate. Confirmation seeds are scenario replications, not new human records.
"""
from __future__ import annotations

import argparse
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import MODELS, check_panel, hashes, new_output
from .provenance import evaluation_signature, model_signature, generator_signature
from sugiyama_v2.io import save_json, csv, sha
from sugiyama_v2.scoring import TARGET_IDS


def compare_scores(scores, official):
    """Order checks with exact tie handling and rounded official-score intervals."""
    if set(scores) != set(official):
        raise ValueError("Model sets differ")
    pairs = []
    for a, b in itertools.combinations(sorted(scores), 2):
        low_a, high_a = official[a]
        low_b, high_b = official[b]
        expected = 1 if low_a > high_b else -1 if low_b > high_a else 0
        delta = float(scores[a]) - float(scores[b])
        if not np.isfinite(delta):
            raise ValueError("Nonfinite score difference")
        observed = 1 if delta > 1e-12 else -1 if delta < -1e-12 else 0
        pairs.append({"model_a": a, "model_b": b, "delta_a_minus_b": delta,
                      "official_direction": expected, "local_direction": observed,
                      "agrees": observed == expected if expected else None})
    checked = [p for p in pairs if p["agrees"] is not None]
    return {"pairs": pairs, "correct": sum(p["agrees"] for p in checked),
            "total": len(checked), "all_correct": bool(checked) and all(p["agrees"] for p in checked),
            "order": sorted(scores, key=lambda m: (-scores[m], m))}


def validate_target_frame(frame, score):
    required = ["attack_power", "mia_power", "aia_power"]
    if len(frame) != 22 or frame.index.duplicated().any() or set(frame.index) != set(TARGET_IDS):
        raise ValueError("Expected the complete 22 non-self target IDs")
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing per-target score components")
    values = frame[required].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(score):
        raise ValueError("Nonfinite per-target or aggregate score")
    if (values < 0).any() or (values[:, 0] > 2).any() or (values[:, 1:] > 1).any() or not 0 <= score <= 2:
        raise ValueError("Score outside the official component range")
    if (np.abs(values[:, 0] - values[:, 1] - values[:, 2]) > 1e-12).any():
        raise ValueError("Per-target total differs from MIA plus AIA")
    if abs(frame.attack_power.mean() - score) > 1e-12:
        raise ValueError("Per-target aggregate differs from official scorer")


def load_panel(path):
    path, state, _ = check_panel(path)
    evaluation = evaluation_signature(state["config"])
    identities = {}
    rows, target_frames = [], {}
    for model in MODELS:
        result = json.loads((path / "scores" / model / "result.json").read_text(encoding="utf-8"))
        if result["model_id"] != model or result["public_sha256"] != state["public_sha256"] or result["truth_sha256"] != state["truth_sha256"]:
            raise ValueError("Mismatched scoring conditions")
        if result["prediction_sha256"] != hashes(path / "predictions" / model):
            raise ValueError("Predictions changed since scoring")
        diagnostic = json.loads((path / "predictions" / model / "diagnostics.json").read_text(encoding="utf-8"))
        identities[model] = model_signature(diagnostic)
        if result.get("evaluation_signature") != evaluation or result.get("model_signature") != identities[model]:
            raise ValueError("Missing or inconsistent evaluation/model signature; rescore saved predictions")
        if result["runtime_versions"] != state["versions"]:
            raise ValueError("Scoring runtime differs from its prepared panel")
        if result["by_target_sha256"] != sha(path / "scores" / model / "by_target.csv"):
            raise ValueError("Per-target diagnostics changed since scoring")
        frame = pd.read_csv(path / "scores" / model / "by_target.csv", float_precision="round_trip").set_index("target_k").sort_index()
        value = float(result["leaderboard"]["score_1"])
        validate_target_frame(frame, value)
        target_frames[model] = frame
        meta = state["case_metadata"]
        rows.append({"panel": str(path), "seed": meta["seed"],
                     "role": meta["reserved_seed_role"], "model": model,
                     "score": value, "mia": float(frame.mia_power.mean()),
                     "aia": float(frame.aia_power.mean())})
    if len({tuple(f.index) for f in target_frames.values()}) != 1:
        raise ValueError("Target sets differ")
    state["comparison_signatures"] = {"evaluation": evaluation, "models": identities,
                                       "generator": generator_signature(state["case_metadata"])}
    return state, rows, target_frames


def build_report(paths, official_path, protocol_path):
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    protocol_hash = sha(protocol_path)
    official_source = json.loads(Path(official_path).read_text(encoding="utf-8"))
    official = {}
    for row in official_source["rows"]:
        half = .5 * 10 ** -row["decimal_places"]
        official[row["model_id"]] = (row["score_1_display"] - half, row["score_1_display"] + half)
    if set(official) != set(MODELS):
        raise ValueError("Expected all four official reference models")
    all_rows, panels, seen = [], [], set()
    signatures = None
    for path in paths:
        state, rows, frames = load_panel(path)
        if signatures is None:
            signatures = state["comparison_signatures"]
        elif signatures != state["comparison_signatures"]:
            raise ValueError("Different model, scoring or generator versions across scenarios")
        meta = state["case_metadata"]
        if meta["protocol_sha256"] != protocol_hash:
            raise ValueError("Case does not match the selected protocol")
        seed = meta["seed"]
        expected_role = ("development" if seed in protocol["development_seeds"] else
                         "confirmation" if seed in protocol["confirmation_seeds"] else None)
        if expected_role is None or meta["reserved_seed_role"] != expected_role:
            raise ValueError("Scenario role or seed differs from the fixed protocol")
        if seed in seen:
            raise ValueError("Duplicate scenario seed")
        seen.add(seed)
        all_rows.extend(rows)
        scores = {r["model"]: r["score"] for r in rows}
        omitted = []
        for target in next(iter(frames.values())).index:
            values = {m: float(f.drop(target).attack_power.mean()) for m, f in frames.items()}
            omitted.append({"omitted_target": int(target), **compare_scores(values, official)})
        panels.append({"path": str(Path(path).resolve()), "seed": seed,
                       "role": meta["reserved_seed_role"], "protocol_sha256": meta["protocol_sha256"],
                       "case_manifest_sha256": state["case_snapshot"]["manifest"],
                       "scores": scores, "rank_check": compare_scores(scores, official),
                       "leave_one_target_out": omitted,
                       "public_profile_coverage": meta["public_profile_coverage"],
                       "public_profile_family_counts": meta["public_profile_family_counts"]})
    if len({p["protocol_sha256"] for p in panels}) != 1:
        raise ValueError("Do not combine different generation protocols")
    confirmation = [p for p in panels if p["role"] == "confirmation"]
    pooled = {m: float(np.mean([p["scores"][m] for p in confirmation])) for m in MODELS} if confirmation else None
    pooled_check = compare_scores(pooled, official) if pooled else None
    # This gate is deliberately about the supplied references, not the unknown future leaderboard.
    complete = seen == set(protocol["development_seeds"] + protocol["confirmation_seeds"])
    confirm_ok = bool(confirmation) and all(p["rank_check"]["all_correct"] for p in confirmation)
    every_case_ok = bool(panels) and all(p["rank_check"]["all_correct"] for p in panels)
    report = {"format": "known-reference-joint-check-v1", "official_source": official_source,
              "protocol_sha256": protocol_hash,
              "comparison_signatures": signatures,
              "all_reserved_scenarios_present": complete, "panels": panels,
              "confirmation_mean_scores": pooled, "confirmation_mean_rank_check": pooled_check,
              "all_confirmation_scenarios_match_known_order": confirm_ok,
              "all_scenarios_match_known_order": every_case_ok,
              "known_reference_rank_reproduced": complete and every_case_ok and confirm_ok and pooled_check["all_correct"],
              "new_model_ranking_accuracy_validated": False,
              "official_hidden_truth_available": False,
              "known_models_used_to_assess_protocol": True,
              "limitations": ["Known-model agreement is retrospective; it does not measure prediction accuracy for unseen models.",
                              "Confirmation scenarios use held-out donor rows but still the same limited supplied population.",
                              "A target deletion is a sensitivity check, not an independent trial or a confidence interval.",
                              "Public-profile matching does not identify the hidden defense mechanism or actual membership labels."]}
    return report, all_rows


def write_html(path, report, rows):
    def h(value):
        return escape(str(value))
    ok = report["known_reference_rank_reproduced"]
    status = "既知4モデルの順位が確認用ケースで一致" if ok else "既知4モデルの順位一致は未達成"
    tables = []
    for p in report["panels"]:
        body = "".join(f'<tr><td>{h(r["model"])}</td><td>{r["score"]:.6f}</td><td>{r["mia"]:.6f}</td><td>{r["aia"]:.6f}</td></tr>'
                       for r in sorted((r for r in rows if r["seed"] == p["seed"]), key=lambda r: -r["score"]))
        c = p["rank_check"]
        tables.append(f'<section><h2>{h(p["role"])}・{p["seed"]}</h2><p>大小関係 {c["correct"]}/{c["total"]} 組が一致</p>'
                      '<table><thead><tr><th>モデル</th><th>合計</th><th>MIA</th><th>AIA</th></tr></thead><tbody>'+body+'</tbody></table></section>')
    html = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>本番順位との照合</title>'
    html += '<style>body{font-family:system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;background:#f5f7fa;color:#142235}h1{font-size:26px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{text-align:right;padding:10px;border-bottom:1px solid #dde3ec}td:first-child,th:first-child{text-align:left}.status{border-left:5px solid #a56616;padding:16px;background:#fff4db}</style>'
    html += '<h1>本番順位との照合</h1><p>画像で確認した本番順位：<strong>Rv2 &gt; Rv1 &gt; pl1 &gt; Tv105</strong></p>'
    html += f'<p class="status">{h(status)}</p><p>同じ22対象・同じ正解・同じ補正値で4モデルを比較。公開情報から固定した生成条件を使い、各モデルを本番と同じ処理で実行しています。</p>'
    html += ''.join(tables)
    html += '<section><h2>結果の読み方</h2><p>この照合は、結果が既知の4モデルについての再現確認です。未知のモデルの本番順位を保証するものではありません。確認用の乱数と学習用の乱数は分けていますが、供給元の母集団は共通です。</p><p>順位が一致しない間は、この模擬点を本番の予測順位として採用しません。対象を1つ除く感度分析、公開分布の適合度、全入力ハッシュは result.json に保存しています。</p></section></html>'
    Path(path).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-feedback", type=Path, default=Path(__file__).parent / "evidence/official_totals_20261009.json")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parent / "generation_v3_protocol.json")
    args = parser.parse_args()
    report, rows = build_report(args.panels, args.official_feedback, args.protocol)
    output = new_output(args.output)
    save_json(output / "result.json", report)
    csv(output / "scores.csv", pd.DataFrame(rows))
    write_html(output / "report.html", report, rows)
    print(json.dumps({"known_reference_rank_reproduced": report["known_reference_rank_reproduced"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
