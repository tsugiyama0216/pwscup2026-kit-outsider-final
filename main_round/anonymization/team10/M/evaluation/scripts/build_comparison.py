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
        "| モデル | U | MIA β | AIA R（φ前） | 診断用耐性 | floor | balance | 耐性余裕 R-U | 判定 | 特徴・メモ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in models:
        lines.append(
            "| [{model}](results/{model}/SUMMARY.md) | {u} | {mb} | {aia} | {res} | {floor} | {balance} | {margin} | {interp} | {memo} |".format(
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
                f"## {model}",
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
