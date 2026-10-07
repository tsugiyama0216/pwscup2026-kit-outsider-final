"""Self-contained Japanese HTML reports; all values come from saved scoring."""
from __future__ import annotations

from html import escape
from pathlib import Path

from .common import team_names


def number(value):
    return "—" if value is None else f"{float(value):.9f}"


def table(headers, rows):
    head = "".join("<th>" + escape(str(x)) + "</th>" for x in headers)
    body = "".join("<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def write_report(result, directory):
    names = team_names()
    replay = result.get("kind") == "historical_replay"
    title = "予備戦・公開回答の再採点" if replay else "予備戦・自作攻撃の評価"
    content = '<p class="note">予備戦の実配布データと公開正解を使用。予備戦ルールで採点し、本戦のAIA補正値 φ は適用していません。</p>'
    if replay:
        content += f'<div class="cards"><div><b>{len(result["teams"])}</b><span>公開回答の再採点</span></div><div><b>{result["verification"]["mia_pairs"]}</b><span>MIA明細の一致</span></div><div><b>{result["verification"]["aia_pairs"]}</b><span>AIA明細の一致</span></div></div>'
        content += '<p>公表得点・標的別明細と許容差 10⁻¹⁰ で照合済み。表の見出しを押すと並べ替えられます。</p>'
        rows = []
        for item in result["teams"]:
            s = item["summary"]
            rows.append([item["attacker_id"], names[item["attacker_id"]], number(s["mia_power"]), number(s["aia_power"]),
                         number(s["attack_power"]), number(item["published_attack_power"]), f'{item["attack_power_difference"]:.2e}',
                         f'{s["mia_target_count"]} / {s["aia_target_count"]}'])
        content += table(["チーム番号", "チーム名", "MIA力", "AIA力", "攻撃力合計", "公表値", "差", "MIA / AIA標的数"], rows)
        content += '<p class="note">cohort04 → 標的23 のAIA回答は運営公開原本にありません。このチームだけ過去の公表値と同じ19標的平均を再現しています。新作攻撃の評価では欠落を許容しません。</p>'
        content += '<p><a href="comparison.csv">チーム別比較CSV</a> · <a href="all_by_target.csv">全標的の詳細CSV</a> · <a href="result.json">照合結果JSON</a></p>'
    else:
        summary = result["summary"]
        content += '<div class="cards">' + "".join(f'<div><b>{number(summary[key])}</b><span>{label}</span></div>' for key, label in [("attack_power", "攻撃力合計（0〜2）"), ("mia_power", "MIA力（0〜1）"), ("aia_power", "AIA力（0〜1）")]) + '</div>'
        content += '<p>各標的について全体・希少層の大きい値を取り、その後で標的平均を計算します。高いほど攻撃に成功しています。</p>'
        rows = [[r["target_k"], names[r["target_k"]], number(r["mia_score"]), number(r["mia_tpr_all"]), number(r["mia_tpr_rare"]),
                 r["mia_positive_count"], r["mia_rare_positive_count"], number(r["aia_score"]), number(r["aia_p_m"]), number(r["aia_p_c"])]
                for r in result["by_target"]]
        content += table(["標的", "チーム名", "MIA力", "MIA全体", "MIA希少", "所属者数", "希少所属者数", "AIA力", "標的成功率", "対照成功率"], rows)
        content += '<p class="note">「—」は未定義で、0への置換ではありません。AIAは追跡年数が±0.5年以内かで成功を判定し、標的と対照の成功率差を正規化しています。希少側は1.64×SEを控除します。</p>'
        content += '<p><a href="by_target.csv">標的別詳細CSV</a> · <a href="result.json">採点結果・入力ハッシュJSON</a></p>'
    content += '<p class="foot">この値は予備戦の実問題に対する評価です。本戦の得点や順位を予測するものではありません。公開正解で調整したモデルの値は、未知データでの性能とは区別してください。</p>'
    document = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>''' + title + '''</title><style>
body{font:15px/1.75 system-ui,"Yu Gothic",sans-serif;color:#172b3a;background:#f3f6f8;margin:0}main{max-width:1450px;margin:0 auto;padding:40px 28px}h1{font-size:28px;margin:0 0 8px}.eyebrow{color:#316778;font-weight:650;letter-spacing:.1em}.note{background:#e5eff2;padding:14px 18px;border-radius:8px}.cards{display:flex;gap:16px;margin:26px 0}.cards>div{background:white;border:1px solid #d7e1e8;border-radius:10px;padding:20px 26px;flex:1}.cards b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.cards span{color:#52616c}.scroll{overflow-x:auto;background:white;border:1px solid #d7e1e8;border-radius:8px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px 14px;text-align:right;white-space:nowrap;border-bottom:1px solid #edf0f2}th{background:#183b4a;color:white;cursor:pointer}td:nth-child(2),th:nth-child(2){text-align:left;white-space:normal;min-width:140px;max-width:300px}tr:hover td{background:#f0f7f9}a{color:#126d90}.foot{color:#586b78;font-size:13px;margin-top:30px}@media(max-width:650px){main{padding:22px 12px}.cards{display:block}.cards>div{margin-bottom:10px}.cards b{font-size:23px}}
</style><main><div class="eyebrow">PWSCUP 2026 · PRELIMINARY</div><h1>''' + title + '</h1>' + content + '''</main><script>
document.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{const body=th.closest('table').tBodies[0];const asc=th.dataset.asc!=='true';th.dataset.asc=asc;const rows=[...body.rows];rows.sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText;const nx=Number(x),ny=Number(y);const c=Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y,'ja');return asc?c:-c;});rows.forEach(r=>body.appendChild(r));}));
</script></html>'''
    path = Path(directory) / "report.html"
    path.write_text(document, encoding="utf-8")
    return path
