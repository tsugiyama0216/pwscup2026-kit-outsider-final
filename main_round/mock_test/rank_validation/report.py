"""A small, offline Japanese report for descriptive rank validation."""
from __future__ import annotations

import html

LABELS = {"score_1": "合計", "score_2": "MIA全体", "score_3": "MIA希少", "score_4": "AIA全体", "score_5": "AIA希少"}


def esc(value):
    return html.escape(str(value))


def number(value):
    return "未測定" if value is None else f"{value:.6f}"


def loo_label(value):
    if not value["available"]:
        return "確認不能（1群のみ）"
    return "逆転あり" if value["rank_reversal"] else "同点を含む変化あり" if value["direction_change_including_tie"] else "逆転なし"


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def render(report):
    known_failure = any(r["score_key"] == "score_1" and r["status"] == "opposite_direction" for r in report["official_comparisons"])
    notice = "模擬評価と既知の本番総合順位の逆転を検出しました。" if known_failure else "本番の総合順位を保証する評価器ではありません。"
    coverage = report.get("public_profile_check", {}).get("coverage")
    coverage_html = "<p>この実行では公開入力の適用範囲は未検査です。</p>"
    if coverage:
        coverage_html = table(["公開入力の比較指標", "手元12加工の最小", "最大", "本戦22対象中の範囲外"], [[name, number(row["training_min"]), number(row["training_max"]), str(row["outside_count"]) + " / " + str(row["target_count"])] for name, row in coverage["metric_ranges"].items()])
        coverage_html += "<p>QI距離は公開質問からCへの最近傍距離、距離比は背景からCへの距離との比です。元のB・所属正解・公式得点は使っていません。これは手元の評価がカバーする特徴の不足を検出する診断であり、匿名化強度や公式得点の推定値ではありません。</p>"
    official = table(["モデル", *LABELS.values()], [[r["model_id"], *(('約' if k in r.get('score_intervals', {}) else '') + number(r.get(k)) for k in LABELS)] for r in sorted(report["official"], key=lambda x: -(x.get("score_1") or 0))])
    reversals = [r for r in report["official_comparisons"] if r["status"] == "opposite_direction"]
    checks = table(["実験", "比較 A → B", "成分", "手元の差 B−A", "本番の差 B−A", "判定"], [
        [r["experiment_id"], r["model_a"] + " → " + r["model_b"], LABELS[r["score_key"]], number(r["local_delta"]), number(r["official_delta"]),
         {"opposite_direction": "順位逆転", "same_direction": "この既知ペアでは同方向", "tie_mismatch": "同点／差ありの不一致", "both_tied": "両方同点", "missing_official": "公式点不足", "display_precision_inconclusive": "表示精度では判定不能"}[r["status"]]]
        for r in report["official_comparisons"] if r["status"] != "missing_official"])
    local = table(["実験（混ぜて集計しない）", "A → B", "条件数", "問題数", "加工数", "合計差 B−A", "MIA全体差", "MIA希少差", "AIA全体差", "AIA希少差", "1問題除外", "1加工除外"], [
        [r["experiment_id"], r["model_a"] + " → " + r["model_b"], r["paired_cells"], r["case_count"], r["release_count"],
         *(number(r["metrics"][k]["mean_delta_b_minus_a"]) for k in LABELS),
         loo_label(r["leave_one_case_out"]), loo_label(r["leave_one_release_out"])] for r in report["comparisons"]])
    protocol = table(["評価方法・試験", "総合点の照合ペア数", "大小の不一致", "評価方法の判定"], [[r["experiment_id"], r["known_total_pair_count"], r["known_total_contradictions"],
        {"fails_known_official_ranking": "既知の公式順位を再現できない", "limited_known_agreement": "少数の既知ペアで一致・汎化未確認", "official_validation_unavailable": "公式との照合資料なし"}[r["status"]]] for r in report["protocol_checks"]])
    gates = table(["実験", "A → B", "BをAより良い候補とみなせるか", "理由"], [[r["experiment_id"], r["model_a"] + " → " + r["model_b"],
        {"hold": "保留", "no_mean_improvement": "平均で改善なし", "stable_in_observed_conditions_only": "既存条件内では安定・本戦改善は未確認"}[r["adoption_gate"]["status"]], " / ".join(r["adoption_gate"]["reasons"])] for r in report["comparisons"]])
    means = table(["実験", "モデル", "条件数", *LABELS.values()], [[r["experiment_id"], r["model_id"], r["rows"], *(number(r[k]) for k in LABELS)] for r in report["model_summaries"]])
    sensitivity = table(["実験", "A → B", "12加工を等重み", "M/S/R系列を等重み", "R系列を除外", "希少所属0人", "希少所属1人", "希少所属2人"], [
        [r["experiment_id"], r["model_a"] + " → " + r["model_b"], number(r["metrics"]["score_1"]["mean_delta_b_minus_a"]),
         number(r["mock_sensitivity"]["equal_family_mean_delta"]["score_1"]),
         number(r["mock_sensitivity"]["leave_one_family_out_delta"].get("R", {}).get("score_1")),
         *(number(r["mock_sensitivity"]["rare_positive_strata"].get(str(n), {}).get("mean_delta", {}).get("score_1")) for n in (0, 1, 2))]
        for r in report["comparisons"] if r.get("mock_sensitivity", {}).get("available")])
    blocked = table(["実験", "A → B", "比較できない理由"], [[r["experiment_id"], r["model_a"] + " → " + r["model_b"], r["reason"]] for r in report["blocked_comparisons"]])
    return """<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>攻撃モデルの順位検証</title>
<style>body{font-family:system-ui,'Yu Gothic',sans-serif;max-width:1440px;margin:32px auto;padding:0 24px;color:#152d38;background:#f6f8fa;line-height:1.75}h1,h2{line-height:1.3}h1{font-size:30px}h2{margin-top:36px;font-size:21px}p{max-width:1050px}.notice{background:#fff4df;border-left:5px solid #c57c17;padding:16px 22px}.scroll{overflow:auto;background:white;border:1px solid #d9e0e5;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:10px 13px;border-bottom:1px solid #e6ebee;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#e9f0f4}code{background:#e7edf0;padding:2px 5px}li{margin:6px 0}</style>
<h1>攻撃モデルの順位検証</h1><p>2026年10月9日 · 公式採点の再現と、本番順位を予測できるかを分けて確認</p>
<div class='notice'><strong>""" + esc(notice) + """</strong><p>この改善版は、矛盾する模擬平均だけを理由に提出候補を推奨しない仕組みです。未知モデルへの予測精度は未確認です。公式点に合わせる係数の調整はしていません。</p></div>
<h2>確認できた本番結果</h2>""" + official + f"<p>モデル名未確認の結果：{len(report['unresolved'])}件。未知の総合点を内訳から埋めたり、別モデルへ割り当てたりしていません。</p>" + \
        "<h2>本番との大小の照合</h2><p>数値の近さではなく、B−Aの符号を比較します。" + str(len(reversals)) + "件の成分・実験の組み合わせで逆転を検出。複数行は独立した公式検証件数ではありません。</p>" + checks + \
        "<h2>評価方法自体の検査</h2><p>本番で負けた候補を高く評価した試験は、順位推定の根拠として不十分です。既知ペアでの一致も、新しい候補に対する精度の証明にはなりません。</p>" + protocol + \
        "<h2>本戦への適用範囲</h2>" + coverage_html + \
        "<h2>同条件のペア比較</h2><p>差が正ならB、負ならAが高い値です。問題と加工の全組合せが一致する比較だけを表示。問題を繰り返しても独立集団とは扱いません。除外時の範囲・加工別結果・入力対応は comparison.json に保存しています。</p>" + local + \
        "<h2>候補の採用保留チェック</h2><p>平均だけが改善し、希少人数・系列・問題によって符号が反転する候補は保留します。このルールは今回の資料を見た後に作った回顧的な診断であり、将来の本番精度は未検証です。極小の差を実用的改善と認定するものではありません。</p>" + gates + \
        "<h2>実験別の集計</h2><p>予備戦、共同22対象、単一対象、調整用、確認用は別々に表示します。これらを足し合わせた総合順位は作りません。</p>" + means + \
        "<h2>防御系列と希少人数による順位の変化</h2><p>12加工中10加工がR系列です。下表は合計差 B−A を、系列の比重と希少所属人数を変えて表示します。系列等重みは感度診断であり、本戦の出現頻度や新しい正解得点ではありません。希少所属0人では希少TPRが未定義のため、合計では全体MIAを用います。</p>" + sensitivity + \
        "<h2>比較を保留した組み合わせ</h2>" + blocked + \
        "<h2>提出前に見る項目</h2><ol><li>同じ公開入力・同じ正解・同じφで比較する。新モデルは同一の保存問題に実行する。</li><li>合計差に加え、MIA／AIA・希少層のどこで増減したかを見る。</li><li>1加工を除くと順位が変わる候補は、得意な加工への依存が強いと判断する。</li><li>既知の公式比較に逆行する成分は追加調査する。未提出モデルの公式順位が確認できたとは扱わない。</li><li>本戦に近い加工条件を公開データで検証し、事前に固定してから新候補を比べる。</li></ol><p>このZIPには一部の公開入力がなく、保存時の問題・加工対応は引き継ぎ索引に依存します。正解・設定・採点コードの内容ハッシュは照合しますが、それだけで攻撃に渡した公開入力の完全一致や未知人物評価を証明できません。</p></html>"
