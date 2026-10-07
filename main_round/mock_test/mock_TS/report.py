"""Portable Japanese HTML report and machine-readable comparison records."""
from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import mean


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "item"):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _text(value):
    return escape(str(value), quote=True)


def _number(value):
    return "—" if value is None else f"{float(value):.6f}"


def _details(value, label="詳細を表示"):
    return f'<details><summary>{_text(label)}</summary><pre>{_text(json.dumps(_safe_json(value), ensure_ascii=False, indent=2, allow_nan=False))}</pre></details>'


def _result_table(rows):
    if not rows:
        return '<p class="muted">まだ結果がありません。</p>'
    items = []
    for row in rows:
        values = "".join(f'<td class="numeric">{_number(row.get(key))}</td>' for key in
                         ("score_1", "mia_power", "aia_power", "score_2", "score_3", "score_4", "score_5", "phi_mock"))
        values += '<td class="numeric">—</td>'
        floor = row.get("floor_estimate", {})
        mode = "診断回数" if floor.get("diagnostic_reduced_repetitions") else "標準回数"
        detail = {"formula": "score_1 = mia_power + aia_power; aia_power = clip(max(AIA all, AIA rare) - phi_mock, 0, 1)",
                  "denominators": row.get("denominators"), "breakdown": row.get("breakdown"),
                  "floor_estimate": floor, "provided_phi10_sensitivity": row.get("provided_phi10_sensitivity"),
                  "validation": row.get("validation")}
        items.append(f'<tr><td>{_text(row.get("case_id", ""))}</td><td class="model">{_text(row.get("model", ""))}</td>'
                     f'<td>{_text(row.get("method", ""))}</td>{values}'
                     f'<td>{_text(floor.get("repetitions", "—"))}<br><span class="muted">{mode}</span></td></tr>'
                     f'<tr class="detail-row"><td colspan="13">{_details(detail)}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>問題</th><th>C</th><th>攻撃手法</th><th>攻撃力合計<br>score_1</th><th>MIA力</th><th>AIA力<br>φ控除後</th><th>MIA全体<br>score_2</th><th>MIA希少<br>score_3</th><th>AIA全体<br>score_4</th><th>AIA希少<br>score_5</th><th>φ_mock</th><th>score_6</th><th>φ反復</th></tr></thead><tbody>' + "".join(items) + '</tbody></table></div>'


def _group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("model", ""), row.get("method", ""))].append(row)
    out = []
    for (model, method), cases in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append({"model": model, "method": method, "case_count": len(cases),
                    "mean_score_1": mean(float(r["score_1"]) for r in cases),
                    "min_score_1": min(float(r["score_1"]) for r in cases),
                    "max_score_1": max(float(r["score_1"]) for r in cases),
                    "mean_mia_power": mean(float(r["mia_power"]) for r in cases),
                    "mean_aia_power": mean(float(r["aia_power"]) for r in cases)})
    return out


def write_report(output: Path, rows: list[dict], metadata: dict) -> Path:
    """Write detailed_results.html, results.json and summary.csv without remote assets."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, metadata = _safe_json(rows), _safe_json(metadata)
    primary = [r for r in rows if r.get("method") != "background_only"]
    diagnostics = [r for r in rows if r.get("method") == "background_only"]
    grouped = _group_rows(primary)
    payload = {"format": "mock_TS_results_v1", "metadata": metadata, "results": rows,
               "case_means": grouped, "official_score_prediction": False}
    (output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ["case_id", "model", "method", "score_1", "mia_power", "aia_power",
              "score_2", "score_3", "score_4", "score_5", "score_6", "phi_mock", "aia_before_floor"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    grouped_html = "".join('<tr>' + ''.join(f'<td>{_text(value)}</td>' for value in
        (r["model"], r["method"], r["case_count"], _number(r["mean_score_1"]),
         f'{_number(r["min_score_1"])} – {_number(r["max_score_1"])}',
         _number(r["mean_mia_power"]), _number(r["mean_aia_power"]))) + '</tr>' for r in grouped)
    cards = [("Cの種類", len({r.get("model") for r in primary})),
             ("固定問題セット", len({r.get("case_id") for r in rows})),
             ("Cありの実行数", len(primary)), ("Cなしの診断数", len(diagnostics))]
    cards_html = "".join(f'<div class="card"><span>{_text(label)}</span><strong>{value}</strong></div>' for label, value in cards)
    reduced = any(r.get("floor_estimate", {}).get("diagnostic_reduced_repetitions") for r in rows)
    floor_note = ('<div class="notice">φが1,000回未満の結果を含みます。これは動作確認用で、主比較では1,000回で再計算してください。</div>' if reduced else '')
    html = '''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>mock_TS | 攻撃評価 詳細結果</title>
<style>
:root{color-scheme:light;--ink:#18253b;--muted:#66758b;--blue:#245ac3;--border:#dce3ed;--paper:#fff;--bg:#f3f6fb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.7 system-ui,"Yu Gothic",sans-serif}
header{background:#152640;color:white;padding:36px max(24px,calc((100vw - 1440px)/2));border-bottom:5px solid #4ea5db}
.eyebrow{color:#a9cbf5;font-size:12px;letter-spacing:.15em}h1{margin:7px 0;font-size:29px;letter-spacing:.02em}header p{max-width:920px;color:#d9e3f3;margin:8px 0}
main{max-width:1490px;margin:auto;padding:24px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,section{background:var(--paper);border:1px solid var(--border);border-radius:10px}
.card{padding:18px 22px}.card span{color:var(--muted);display:block}.card strong{font-size:30px;font-weight:650}section{padding:22px;margin-top:20px}h2{margin:0 0 10px;font-size:20px}p{margin:8px 0 14px}.muted{color:var(--muted);font-size:12px}
.tag{display:inline-block;border:1px solid #8eadd6;border-radius:20px;padding:2px 11px;font-size:12px;margin-top:8px}.table-wrap{overflow:auto;border:1px solid var(--border);border-radius:7px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e9edf4;font-size:12px}th{background:#ecf2fa;color:#3d506e;font-weight:600;position:sticky;top:0}td.numeric{font-variant-numeric:tabular-nums;text-align:right}.model{font-weight:700}.detail-row td{padding:2px 12px 9px;background:#fbfcfe}summary{cursor:pointer;color:var(--blue);font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5fa;padding:14px;border-radius:6px;max-height:500px;overflow:auto;font-size:12px}
.notice{border-left:4px solid #d78f21;background:#fff6e7;padding:14px 18px;margin-top:20px}.formula{background:#eef4fc;padding:13px 16px;border-radius:6px;font-variant-numeric:tabular-nums}li{margin:6px 0}a{color:var(--blue)}footer{text-align:center;color:var(--muted);font-size:12px;padding:24px}
@media(max-width:700px){main{padding:12px}.cards{grid-template-columns:repeat(2,1fr)}section{padding:14px}header{padding:26px 20px}h1{font-size:24px}}
@media print{body{background:white}header{background:white;color:var(--ink);padding:15px}header p{color:var(--ink)}main{padding:0}.table-wrap{overflow:visible}th,td{padding:5px;font-size:9px}section{break-inside:avoid}details{display:none}}
</style></head><body>
<header><div class="eyebrow">PWS CUP 2026 · LOCAL BENCHMARK</div><h1>mock_TS 攻撃評価 詳細結果</h1>
<p>本選で作成した匿名化データCを、同じ正解付き問題で個別に評価します。各Cは別の実行環境へ渡し、予測や学習状態を次のCへ持ち越しません。</p>
<span class="tag">公式の採点コードを使用 · 模擬問題の比較用</span></header>
<main><div class="cards">''' + cards_html + '</div>' + floor_note + '''
<section><h2>得点の読み方</h2><div class="formula">攻撃力合計（score_1） = MIA力 + AIA力（φ_mock控除後）</div>
<p>MIA力は全体・希少のTPR@FPR 1%の最大値。AIA力は全体・希少の正規化超過成功率の最大値からφ_mockを引き、0〜1へ制限します。希少AIAには公式の1.64×SE補正を適用します。time推定の成功条件は誤差±0.5年以内です。</p>
<p class="muted">score_2〜5は公式と同じ全体・希少の内訳で、AIA内訳はφ控除前です。MIAの希少所属者が0人の問題では希少値は未定義で、公式集約に合わせてscore_3に0を表示し、MIA力には全体値を使います。各群の人数は詳細から確認できます。score_6（公式の最終順位得点）は計算しません。12種類のCを12チームとは扱いません。</p></section>
<section><h2>C・攻撃手法ごとの問題平均</h2><p class="muted">平均・最小・最大は同じB/Cに対する候補抽出の違いです。既定の希少MIA所属者0・1・2人を等しく扱うシナリオ平均で、公式での出現頻度を表しません。独立した人物集団や独立標本の信頼区間ではありません。</p>
<div class="table-wrap"><table><thead><tr><th>C</th><th>攻撃手法</th><th>問題数</th><th>合計平均</th><th>合計 最小〜最大</th><th>MIA力平均</th><th>AIA力平均</th></tr></thead><tbody>''' + grouped_html + '''</tbody></table></div></section>
<section><h2>問題・Cごとの採点結果</h2><p class="muted">Cに含まれる手がかりを使った攻撃の結果。詳細には分母、成功率、φの標準誤差と参照値φ₁₀による感度確認を保存しています。</p>''' + _result_table(primary) + '''</section>
<section><h2>Cなしの診断</h2><p>候補の属性と許可された背景だけで解けてしまわないかを確認します。Cなしでも繰り返し高い値が出る場合、出所や分布の違いによる手がかりを疑います。この値をCありの得点から差し引く補正は行いません。</p>''' + _result_table(diagnostics) + '''</section>
<section><h2>模擬条件と解釈の限界</h2><ul>
<li>MIAの所属者は元のB、非所属者はA_bgの保留部分から選びます。保留した人物を攻撃用の背景から除きます。B側は提供された真の希少ラベル、A_bg側は代理の希少判定です。</li>
<li>AIAはBに一致する163人と、B・A_bgに一致しない合成対照163人です。合成対照のtimeと希少ラベルは模擬上の正解です。見た目や属性関係を整えても実際の非公開対照集団を再現する保証はありません。</li>
<li>合成対照では測定値とtimeのドナーが異なるため、両者の関係は完全には保存されません。近似コピーを避ける制約も分布を変えます。φの再推定だけで、これらに由来する攻撃手法別の偏りをすべて取り除くことはできません。</li>
<li>φ_mockは各問題と縮小した背景に対して公式参照AIAを実行した平均です。標準は1,000回。標準誤差はモンテカルロのばらつきだけを表し、データ分布や生成方法の不確実性を含みません。</li>
<li>提供済みφ₁₀＝0.026887は元の公式質問に対応する参照値です。質問の変わった本mockへ適用した値は感度確認として別掲し、主得点には使用しません。</li>
<li>既存Cや学習済み攻撃が以前にB・A_bg全体を利用した履歴は取り消せません。攻撃には新しい学習状態を用い、過去のモデル・キャッシュの持込み条件を確認してください。</li>
<li>別プロセス・別作業領域による分離は、信頼できる攻撃プログラムの状態持越しを防ぐためです。悪意あるコードがローカルファイルを読むことを防ぐOSサンドボックスではありません。</li>
<li>公式の計算方法を用いるローカル比較実験です。運営の非公開正解・母集団・全標的・最終順位を再現したものでも、公式得点の予測でもありません。</li>
</ul></section><section><h2>実行記録とダウンロード</h2><p><a href="summary.csv">summary.csv</a> · <a href="results.json">results.json</a></p>''' + _details(metadata, "実行メタデータを表示") + '''</section>
</main><footer>mock_TS · 保存された予測CSVを固定公式スコアラーで採点 · 公式提出は行いません</footer></body></html>'''
    path = output / "detailed_results.html"
    path.write_text(html, encoding="utf-8")
    return path
