"""Build human-readable and JSON comparisons from resistance reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_BASELINES = ("S1", "S2", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "M1")
MODEL_NOTES = {
    "S1": ("予備戦v5由来のDP AIM合成", "耐性信号は小さいが、相関再現が崩れてU=0のため基準モデル。"),
    "S2": ("S1の条件付きroundを独立sampleへ修正", "相関とUは回復。希少側MIAが増え、Uと耐性がほぼ同時に制約。"),
    "R1": ("層別ガウシアン・コピュラと原因別Cox", "完全合成だが希少側MIAが強い。AIAも測定モデル中で最大。"),
    "R2": ("R1に無正則化Coxとリスク集合事象割当て", "R1よりUは改善したが、MIA βはR1と同値で耐性は改善せず。"),
    "R3": ("R2へ集計Cox係数の部分補正", "解析有用性の改善と同時に希少側MIAがやや増加。"),
    "R4": ("条件付き分布輸送とCoxリスク保持", "U改善は小幅。MIAはR3より増え、耐性がボトルネック。"),
    "R5": ("逐次CART合成とCox事象生成", "R系列で最も良い耐性。有用性は後期R系列より低いがfloorはR系列首位。"),
    "R6": ("R4を土台に希少群コピュラを50%補正", "Uは高いが希少側MIAが最大級。耐性改善を最優先。"),
    "R7": ("多数派・希少群コピュラと集計事象交換", "AIAは小さいがMIAが支配。R6からMIAは少し改善。"),
    "R8": ("非線形分散・希少経験分位点・事象補正", "Uは0.97超。R6/R7よりMIAは改善したが、依然耐性がボトルネック。"),
    "R9": ("地域・年齢・Cox等の集計制約付きR8補正", "Uは最高だがMIAはR6水準へ悪化。集計一致の強化と耐性低下が同時に観測。"),
    "M1": ("S1へ公開A_bgランク・コピュラ補正", "非S1モデルでMIAが最小、floor首位。希少側DOMIAS信号は残る。"),
}
MODEL_CAUSES = {
    "S1": "DP測定と独立的な復号でB固有の7変数密度まで平滑化された一方、列間相関も崩れた可能性があります。低MIAとU=0は同じ強い平滑化の別側面と考えられます。",
    "S2": "sample化で不自然な相関は減りましたが、Bから測定した周辺・低次元構造の再現が戻り、特に希少側でCとBの密度がA_bgより近くなった可能性があります。",
    "R1": "性別・喫煙層別のコピュラとB経験分布が、希少7変数の同時密度を強く再現した可能性があります。完全合成・完全一致0でもKDE密度比は高くなり得ます。",
    "R2": "連続共変量はR1を引き継ぐため、7変数だけを使うDOMIASの値がR1と同じです。事象割当ての変更は主にAIAとU_specへ影響したと考えられます。",
    "R3": "BのCox係数へ近づける連続値補正が、B固有の共変量方向へCを移動させ、希少側密度比をわずかに上げた可能性があります。",
    "R4": "Bの条件付き平均・共分散へ近づける分布輸送が、Coxリスクを保ちながらも7変数のBらしさを増した可能性があります。",
    "R5": "CART葉内の経験分布から再サンプリングする局所平滑化が、後期コピュラ系列ほど希少密度を精密に合わせず、相対的にMIAを抑えた可能性があります。",
    "R6": "B検出希少群の経験分位点と潜在共分散を明示的に移したため、DOMIASが使う希少7変数密度へ最も直接的に近づいた可能性があります。",
    "R7": "多数派・希少群を別々に補正し、希少半径を保護したためBらしい裾構造が残りました。R6からは改善しましたが、希少密度信号は依然強い状態です。",
    "R8": "非線形分散補正と希少経験分位点補正により有用性を高めつつ、区分保護や事象交換がR6/R7よりMIAを弱めた可能性があります。ただしB経験分位点利用の信号は残ります。",
    "R9": "地域件数、年齢分位点、全体・希少分位点、順位相関、Cox統計を同時にBへ合わせたため、Uは上がってもDOMIASが見る7変数密度、とくに希少側が再びBへ近づいた可能性があります。",
    "M1": "依存構造をBではなく公開A_bgから導入したため、B固有の密度比を相対的に抑えた可能性があります。公開A_bg完全一致4行はありますが、B完全一致0であり、希少MIAの残存原因をそれだけでは説明できません。",
}
MODEL_ACTIONS = {
    "S1": "提出候補にはせず、低MIA側の対照として保持します。相関を段階的に戻したときのUとMIAの曲線を比較します。",
    "S2": "希少ドナー処理とDP合成本体を分けてアブレーションし、希少側TPRを増やした工程を特定します。",
    "R1": "層別、経験分位点、共分散縮小率を個別に弱め、希少TPRとU_gen/U_rareの変化を測ります。",
    "R2": "MIA改善には事象生成ではなくR1由来の連続共変量生成を変更します。AIAについてはR1との事象割当て差を比較します。",
    "R3": "Cox補正強度を0から採用値まで段階化し、U_spec改善1単位当たりのMIA増加を記録します。",
    "R4": "条件付き平均補正、共分散補正、Cox直交射影を別々に外し、どの成分が希少密度を強めるか確認します。",
    "R5": "R系列の耐性基準として保持し、葉サイズ・深さ・希少群平滑化を調整してUを上げてもMIAを維持できるか検証します。",
    "R6": "希少コピュラ強度とB経験分位点への写像を弱め、A_bgとの混合またはノイズ付加で希少密度比を下げます。",
    "R7": "多数派補正、希少補正、半径保持、事象交換の4段階を順に評価し、MIAに効く工程だけを緩和します。",
    "R8": "R6/R7より改善した処理を保持しつつ、希少経験分位点強度を下げる候補を優先します。",
    "R9": "地域、年齢、分位点、順位相関、Cox制約を1群ずつR8へ追加するアブレーションを行い、U増分とMIA増分を対応付けます。",
    "M1": "次モデルでは公開A_bg依存構造を維持し、希少ドナー強度だけを下げる候補を作り、U_specを保ちながら希少TPRを下げます。",
}


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def status(summary: dict) -> tuple[str, str]:
    if float(summary["utility_U"]) < 0.5:
        return "utility-reject", "有用性が低く提出候補外"
    if summary["risk_band"] == "high-local-signal":
        return "resistance-hold", "耐性信号が強いため改善・再検証"
    if summary["risk_band"] == "watch":
        return "watch", "提出候補だが耐性を注意して比較"
    return "candidate", "ローカル基準では提出候補"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--diagnostics-out", type=Path)
    args = parser.parse_args()

    models = []
    for report_path in sorted(args.results.glob("*/resistance_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
        state, interpretation = status(summary)
        official_mia = report["mia"]["attacks"]["official_domias_kde"]
        scored_aia = {
            name: attack for name, attack in report["aia"]["attacks"].items()
            if name != "public_horizon_sanity"
        }
        worst_aia_attack = max(scored_aia, key=lambda name: float(scored_aia[name]["risk"]))
        utility_value = float(summary["utility_U"])
        resistance = float(summary["diagnostic_resistance_index"])
        balance_score = 0.0 if utility_value + resistance == 0 else 2 * utility_value * resistance / (utility_value + resistance)
        feature, model_memo = MODEL_NOTES.get(report["model"], ("記録なし", "個別メモ未登録。"))
        item = {
            "model": report["model"],
            **summary,
            "worst_mia_attack": report["mia"]["worst_attack"],
            "official_mia_tpr_all": official_mia["tpr_all"],
            "official_mia_tpr_rare": official_mia["tpr_rare"],
            "team_mia_beta": report["mia"]["attacks"]["team_v4_v9_multiview"]["beta"],
            "worst_aia_attack": worst_aia_attack,
            "exact_B_rows_in_C": report["copy_diagnostics"]["exact_B_rows_in_C"],
            "exact_A_bg_rows_in_C": report["copy_diagnostics"]["exact_A_bg_rows_in_C"],
            "c_duplicate_rows": report["copy_diagnostics"]["c_duplicate_rows"],
            "mia_rare_gap": max(0.0, float(official_mia["tpr_rare"]) - float(official_mia["tpr_all"])),
            "balance_score": balance_score,
            "resistance_margin": resistance - utility_value,
            "primary_attack_driver": "MIA" if float(summary["mia_stress_beta_risk"]) >= float(summary["aia_stress_risk_before_phi"]) else "AIA",
            "feature": feature,
            "model_memo": model_memo,
            "status": state,
            "interpretation": interpretation,
            "report": str(report_path.as_posix()),
        }
        models.append(item)

        detail = [
            f"# {report['model']} 耐性評価",
            "",
            "[詳細な解釈・原因仮説](DETAILS.md) / [指標・攻撃手法の解説](../../METRICS_GUIDE.md)",
            "",
            f"- 有用性U: {fmt(summary['utility_U'])}",
            f"- MIA βリスク（偶然補正後・K正規化前）: {fmt(summary['mia_stress_beta_risk'])}",
            f"- MIA K=12正規化リスク: {fmt(summary['mia_stress_normalized_risk'])}",
            f"- AIAリスク（φ控除前）: {fmt(summary['aia_stress_risk_before_phi'])}",
            f"- 診断用耐性指数: {fmt(summary['diagnostic_resistance_index'])}",
            f"- 提出比較用screening floor: {fmt(summary['screening_floor'])}",
            f"- リスク帯: `{summary['risk_band']}`",
            f"- 判定: **{interpretation}**",
            "",
            "## 攻撃別",
            "",
            "| 攻撃 | 主な値 |",
            "|---|---:|",
        ]
        for name, attack in report["mia"]["attacks"].items():
            detail.append(
                f"| MIA `{name}` β / normalized | {attack['beta']:.6f} / {attack['normalized_risk']:.6f} |"
            )
        for name, attack in report["aia"]["attacks"].items():
            detail.append(f"| AIA `{name}` R | {attack['risk']:.6f} |")
        detail += [
            "",
            "## 解釈上の制約",
            "",
            *[f"- {text}" for text in report["limitations"]],
            "",
            f"詳細JSON: [resistance_report.json](resistance_report.json)",
        ]
        (report_path.parent / "SUMMARY.md").write_text("\n".join(detail) + "\n", encoding="utf-8", newline="\n")

        official = report["mia"]["attacks"]["official_domias_kde"]
        team_mia = report["mia"]["attacks"]["team_v4_v9_multiview"]
        official_aia = report["aia"]["attacks"]["official_qi_nearest"]
        team_aia = report["aia"]["attacks"]["team_multiview_linkage"]
        horizon_aia = report["aia"]["attacks"]["public_horizon_sanity"]
        cause = MODEL_CAUSES.get(report["model"], "原因仮説は未登録です。")
        action = MODEL_ACTIONS.get(report["model"], "追加のアブレーションを設計します。")
        bottleneck = "耐性" if item["resistance_margin"] < 0 else "有用性"
        details = [
            f"# {report['model']} 詳細な攻撃耐性診断",
            "",
            "[全モデル比較](../../MODEL_COMPARISON.md) / [指標・攻撃手法の解説](../../METRICS_GUIDE.md) / [生JSON](resistance_report.json)",
            "",
            "## モデルの位置づけ",
            "",
            f"- 生成・補正の特徴: {feature}",
            f"- 観測された結論: {model_memo}",
            f"- 開発上のボトルネック: **{bottleneck}**",
            "",
            "## 主要指標",
            "",
            "| 指標 | 値 | このモデルでの意味 |",
            "|---|---:|---|",
            f"| U | {fmt(utility_value)} | 4有用性観点の最小値 |",
            f"| MIA β | {fmt(summary['mia_stress_beta_risk'])} | 偶然補正後の最大membership信号。小さいほど良い |",
            f"| AIA R（φ前） | {fmt(summary['aia_stress_risk_before_phi'])} | 対照を超えるtime復元力。小さいほど良い |",
            f"| 診断用耐性 | {fmt(resistance)} | 1−max(MIA β, AIA R) |",
            f"| screening floor | {fmt(summary['screening_floor'])} | min(U, 診断用耐性)。一次比較値 |",
            f"| balance | {fmt(balance_score)} | Uと耐性の調和平均 |",
            f"| 耐性余裕 R−U | {fmt(item['resistance_margin'])} | {'負なので耐性が制約' if item['resistance_margin'] < 0 else '正なので有用性が制約'} |",
            "",
            "## MIAの内訳",
            "",
            "| 攻撃 | 全体TPR | 希少TPR | chance 95% | β | K=12正規化 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| 運営DOMIAS KDE | {fmt(official['tpr_all'])} | {fmt(official['tpr_rare'])} | {fmt(official['chance95'])} | {fmt(official['beta'])} | {fmt(official['normalized_risk'])} |",
            f"| 予備戦由来の多視点MIA | {fmt(team_mia['tpr_all'])} | {fmt(team_mia['tpr_rare'])} | {fmt(team_mia['chance95'])} | {fmt(team_mia['beta'])} | {fmt(team_mia['normalized_risk'])} |",
            "",
            f"希少側MIAギャップは **{fmt(item['mia_rare_gap'])}** です。DOMIASは7連続変数のC対A_bg密度比を使うため、この値が大きい場合は希少群の周辺分布・相関・裾の合わせ込みを優先して疑います。",
            "",
            "## AIAの内訳",
            "",
            "| 攻撃 | R | 役割 |",
            "|---|---:|---|",
            f"| 運営QI最近傍 | {fmt(official_aia['risk'])} | 性別×都道府県内のQI最近傍からtimeを復元 |",
            f"| 予備戦由来の多視点照合 | {fmt(team_aia['risk'])} | 3種類の近傍time予測の中央値 |",
            f"| 10年固定 | {fmt(horizon_aia['risk'])} | データを見ないsanity baseline。最大リスクから除外 |",
            "",
            f"このモデルの主要攻撃は **{item['primary_attack_driver']}** です。AIAはteam10の公式φを控除していないため、絶対値よりモデル間の相対差を重視します。",
            "",
            "## コピー・重複診断",
            "",
            f"- Bとの13列完全一致: {item['exact_B_rows_in_C']}行",
            f"- 公開A_bgとの13列完全一致: {item['exact_A_bg_rows_in_C']}行",
            f"- C内完全重複: {item['c_duplicate_rows']}行",
            "",
            "完全一致が0でも、連続分布や相関がBへ近ければ密度比MIAは成立します。この検査だけで匿名性を判断しません。",
            "",
            "## 考えられる原因",
            "",
            cause,
            "",
            "これはモデル差分と攻撃仕様に基づく仮説であり、因果関係の確定ではありません。候補選択と診断に同じBを使っている点にも注意が必要です。",
            "",
            "## 次に確認すること",
            "",
            action,
            "",
            "各候補ではU、全体TPR、希少TPR、MIA β、AIA R、floorを同時に保存し、1つの処理変更だけで比較します。",
            "",
            "## この結果から言えないこと",
            "",
            "- 公式protection、Anon、順位の予測ではありません。",
            "- K=12正規化値の飽和を、そのまま公式MIA=1とは解釈できません。",
            "- 現在の攻撃で信号が小さくても、未実装の攻撃への耐性は保証しません。",
            "- 完全合成、完全一致0、重複0だけでは匿名性を保証しません。",
        ]
        (report_path.parent / "DETAILS.md").write_text("\n".join(details) + "\n", encoding="utf-8", newline="\n")

    models.sort(key=lambda item: (-float(item["screening_floor"]), item["model"]))
    observed = {item["model"] for item in models}
    missing_models = [model for model in EXPECTED_BASELINES if model not in observed]
    output = {
        "schema_version": "team10-local-resistance-comparison-v1",
        "ranking_key": "screening_floor=min(utility_U, diagnostic_resistance_index)",
        "risk_bands": {
            "low-local-signal": "diagnostic attack risk <= 0.05",
            "watch": "0.05 < diagnostic attack risk <= 0.15",
            "high-local-signal": "diagnostic attack risk > 0.15",
        },
        "models": models,
        "missing_models": missing_models,
        "warning": "Local proxy only; not official protection, Anon, or leaderboard prediction.",
    }
    args.json_out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# team10 有用性・耐性比較",
        "",
        "`screening floor = min(有用性U, 診断用耐性指数)` の降順です。公式順位や公式Anonの予測ではありません。",
        "",
        "対象: S1～S2、R1～R9、M1。" + (
            f"未測定: {', '.join(missing_models)}（ローカルにC.csvと評価記録がありません）。" if missing_models else "全対象を測定済みです。"
        ),
        "",
        "各列の計算方法・目的・攻撃実装・解釈は [METRICS_GUIDE.md](METRICS_GUIDE.md) を参照してください。モデル名から個別の原因仮説と次の検証へ移動できます。",
        "",
        "| モデル | U | MIA β | AIA R（φ前） | 診断用耐性 | floor | balance | 耐性余裕 R-U | 判定 | 特徴・メモ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in models:
        lines.append(
            "| [{model}](results/{model}/DETAILS.md) | {u} | {mb} | {aia} | {res} | {floor} | {balance} | {margin} | {interp} | {memo} |".format(
                model=item["model"], u=fmt(item["utility_U"]), mb=fmt(item["mia_stress_beta_risk"]),
                aia=fmt(item["aia_stress_risk_before_phi"]),
                res=fmt(item["diagnostic_resistance_index"]), floor=fmt(item["screening_floor"]),
                balance=fmt(item["balance_score"]), margin=fmt(item["resistance_margin"]),
                interp=item["interpretation"], memo=item["model_memo"],
            )
        )
    rare_dominant = [
        item["model"] for item in models
        if float(item["official_mia_tpr_rare"]) > float(item["official_mia_tpr_all"])
    ]
    max_public_overlap = max((int(item["exact_A_bg_rows_in_C"]) for item in models), default=0)
    lines += [
        "",
        "## 今回の診断で言えること",
        "",
        f"- 現在のscreening floor最上位は{models[0]['model']}（{fmt(models[0]['screening_floor'])}）です。",
        f"- 運営DOMIASでは希少側TPRが全体TPRを上回ったモデルが{', '.join(rare_dominant)}で、主な耐性課題は希少側MIAです。",
        "- 予備戦v4/v9由来の多視点MIAは全モデルでβ=0.01でした。これはこの攻撃で追加信号を検出できなかったという意味で、耐性の証明ではありません。",
        f"- 全モデルでBとの完全一致行とC内重複行は0でした。公開A_bgとの完全一致は最大{max_public_overlap}行で、直接コピー診断だけでは主要なMIA信号を説明できません。",
        "- AIAは全モデルでMIAより小さく、今回の提出候補比較ではMIAがボトルネックです。",
        "",
        "## 読み方",
        "",
        "- MIA βは、FPR=0.01の全体・希少worst-caseから20,000回の偶然水準を控除した値です。",
        "- MIA正規化はルールブックのK=12を適用した追跡値ですが、代理候補集合の構成が公式と違うため飽和しやすく、モデル順位にはβを使います。",
        "- AIA Rは標的と対照の±0.5年正解率差を正規化した値です。team10のφが手元にないため控除前です。",
        "- 診断用耐性指数は `1 - max(MIA β, AIA R)`。大きいほど良い値です。",
        "- balanceはUと診断用耐性の調和平均です。floorが同じ場合や、一方だけ極端に高いモデルを補助的に比較します。",
        "- 耐性余裕 `R-U` が負なら耐性が、正なら有用性がボトルネックです。",
        "- 希少側MIAギャップ、主要攻撃、完全一致診断はMODEL_DIAGNOSTICS.mdに保存します。",
        "- high-local-signalは提出不可という意味ではなく、攻撃信号が強く、追加改善・確認を優先すべき状態です。",
    ]
    args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    if args.diagnostics_out:
        diagnostic_lines = [
            "# team10 モデル別耐性診断",
            "",
            "同一のローカル代理候補集合で測定した開発用診断です。公式protection、Anon、順位の予測ではありません。",
            "",
        ]
        by_name = {item["model"]: item for item in models}
        for model in EXPECTED_BASELINES:
            if model not in by_name:
                diagnostic_lines += [f"## {model}", "", "未測定。必要なC.csvまたは有用性評価記録がありません。", ""]
                continue
            item = by_name[model]
            bottleneck = "耐性" if item["resistance_margin"] < 0 else "有用性"
            diagnostic_lines += [
                f"## [{model}](results/{model}/DETAILS.md)",
                "",
                f"- 生成・補正の特徴: {item['feature']}",
                f"- U / 診断用耐性 / floor / balance: {fmt(item['utility_U'])} / {fmt(item['diagnostic_resistance_index'])} / {fmt(item['screening_floor'])} / {fmt(item['balance_score'])}",
                f"- MIA β / AIA R / 主要攻撃: {fmt(item['mia_stress_beta_risk'])} / {fmt(item['aia_stress_risk_before_phi'])} / {item['primary_attack_driver']}",
                f"- DOMIAS TPR（全体 / 希少 / 希少ギャップ）: {fmt(item['official_mia_tpr_all'])} / {fmt(item['official_mia_tpr_rare'])} / {fmt(item['mia_rare_gap'])}",
                f"- 完全一致（B / 公開A_bg）・C内重複: {item['exact_B_rows_in_C']} / {item['exact_A_bg_rows_in_C']}・{item['c_duplicate_rows']}",
                f"- ボトルネック: {bottleneck}（耐性余裕 R-U={fmt(item['resistance_margin'])}）",
                f"- 結論: {item['model_memo']}",
                "",
            ]
        diagnostic_lines += [
            "## 指標の使い分け",
            "",
            "- 提出候補の一次順位は、公式のmin(U, Anon)に対応させたscreening floorを使います。",
            "- balanceは滑らかな補助順位です。高いUだけで耐性不足を隠さないため、floorを置き換えません。",
            "- 希少側MIAギャップは、全体評価では見えにくい希少レコードの追加リスクを示します。",
            "- K=12正規化MIAは代理候補集合では飽和するため、モデル順位には使いません。",
        ]
        args.diagnostics_out.write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
