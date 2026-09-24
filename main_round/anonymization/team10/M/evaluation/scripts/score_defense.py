"""Build a compact defense-score report from a local attack evaluation.

The underlying evaluator uses the pinned competition attack primitives and the
fixed local proxy in ``evaluate_resistance.py``.  This module only turns its
verbose report into values that are easy to compare between processed C.csv
files.  It deliberately labels the result as a local proxy: organizer attack
candidate sets and team10's phi_i are not available before the attack phase.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}")
    return result


def _optional_bounded(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _bounded(value, name)


def build_summary(report: dict[str, Any], phi: float | None = None) -> dict[str, Any]:
    """Extract defense-oriented scores from ``evaluate_resistance`` output.

    ``anon_proxy`` uses the rulebook-style K-normalized MIA risk.  The
    ``diagnostic_resistance`` value keeps the pre-normalization beta scale for
    development ranking, where the local proxy's larger member pool can make
    K-normalization saturate.
    """
    summary = report.get("summary")
    mia = report.get("mia", {}).get("attacks", {}).get("official_domias_kde")
    aia = report.get("aia", {}).get("stress_risk_before_phi")
    utility = report.get("utility", {}).get("U")
    if not isinstance(summary, dict) or not isinstance(mia, dict):
        raise ValueError("evaluation report is missing summary or MIA results")
    if aia is None:
        raise ValueError("evaluation report is missing AIA results")

    mia_risk = _bounded(mia.get("normalized_risk"), "MIA normalized risk")
    aia_before_phi = _bounded(aia, "AIA risk")
    phi_value = _optional_bounded(phi, "phi_i")
    aia_risk = (
        aia_before_phi
        if phi_value is None
        else max(0.0, min(1.0, aia_before_phi - phi_value))
    )
    attack_power = max(mia_risk, aia_risk)
    anon_proxy = 1.0 - attack_power
    utility_u = _optional_bounded(utility, "utility U")
    overall = None if utility_u is None else min(utility_u, anon_proxy)

    beta = _bounded(report["summary"].get("mia_stress_beta_risk"), "MIA beta")
    diagnostic_attack = max(beta, aia_risk)
    diagnostic_resistance = 1.0 - diagnostic_attack
    diagnostic_floor = None if utility_u is None else min(utility_u, diagnostic_resistance)

    limitations = list(report.get("limitations", []))
    if phi_value is not None:
        limitations = [
            item for item in limitations
            if "phi" not in str(item).lower() and "φ" not in str(item)
        ]
        limitations.append("phi_i was supplied separately and applied only to the AIA component as clip(R - phi_i, 0, 1).")
    return {
        "schema_version": "team10-defense-score-v1",
        "model": report.get("model"),
        "scope": "local attack-phase proxy; not official protection, Anon, or leaderboard score",
        "defense": {
            "mia_risk_normalized": mia_risk,
            "mia_beta_pre_normalization": beta,
            "aia_risk_before_phi": aia_before_phi,
            "phi_i": phi_value,
            "aia_risk_after_phi": aia_risk,
            "attack_power_proxy": attack_power,
            "anon_proxy": anon_proxy,
            "utility_U": utility_u,
            "overall_score_proxy": overall,
            "diagnostic_resistance_pre_normalization": diagnostic_resistance,
            "diagnostic_floor_pre_normalization": diagnostic_floor,
            "primary_attack": "MIA" if mia_risk >= aia_risk else "AIA",
            "bottleneck": None
            if utility_u is None
            else ("utility" if utility_u <= anon_proxy else "defense"),
        },
        "source": {
            "evaluation_report": report.get("inputs", {}).get("c_sha256"),
            "proxy_candidate_set": report.get("mia", {}).get("candidate_proxy"),
            "proxy_challenge_set": report.get("aia", {}).get("challenge_proxy"),
            "phi_source": "user-supplied or organizer-provided phi_i" if phi_value is not None else "not supplied; AIA is before phi_i subtraction",
            "limitations": limitations,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    defense = summary["defense"]

    def fmt(value: Any) -> str:
        return "—" if value is None else f"{float(value):.6f}"

    anon = defense["anon_proxy"]
    if anon >= 0.8:
        assessment = "防御良好"
    elif anon >= 0.5:
        assessment = "要監視"
    else:
        assessment = "要改善"
    primary = defense["primary_attack"]
    bottleneck = defense["bottleneck"]
    bottleneck_label = "防御（攻撃耐性）" if bottleneck == "defense" else "有用性 U" if bottleneck == "utility" else "未判定"

    lines = [
        f"# {summary.get('model', 'model')} 防御評価",
        "",
        "> **結論：" + assessment + "**　主な攻撃経路: **" + primary + "**　ボトルネック: **" + bottleneck_label + "**",
        "",
        "これは公式最終Anonではなく、加工データを同じ条件で比較するためのローカル代理評価です。",
        "",
        "## まず見る数字",
        "",
        "| 防御側の指標 | 値 | 判定上の意味 |",
        "|---|---:|---|",
        f"| **Anon代理（防御点）** | **{fmt(defense['anon_proxy'])}** | 高いほど攻撃に強い |",
        f"| **総合点代理** | **{fmt(defense['overall_score_proxy'])}** | `min(U, Anon代理)` |",
        f"| 有用性 U | {fmt(defense['utility_U'])} | 高いほど加工データが有用 |",
        f"| 主攻撃 | **{primary}** | MIA/AIAのうち大きい側 |",
        "",
        "## 攻撃の内訳",
        "",
        "| 攻撃 | 控除前 | 控除・正規化後 |",
        "|---|---:|---:|",
        f"| MIA（DOMIAS） | β={fmt(defense['mia_beta_pre_normalization'])} | **{fmt(defense['mia_risk_normalized'])}**（K=12） |",
        f"| AIA | R={fmt(defense['aia_risk_before_phi'])} | **{fmt(defense['aia_risk_after_phi'])}**（φ_i={fmt(defense['phi_i'])}） |",
        "",
        "## 開発用の比較値",
        "",
        "| 指標 | 値 |",
        "|---|---:|",
        f"| 攻撃力代理 `max(MIA, AIA)` | {fmt(defense['attack_power_proxy'])} |",
        f"| 開発用耐性（β正規化前） | {fmt(defense['diagnostic_resistance_pre_normalization'])} |",
        f"| 開発用floor（β正規化前） | {fmt(defense['diagnostic_floor_pre_normalization'])} |",
        "",
        "## 読み方",
        "",
        "AIAは `clip(R - φ_i, 0, 1)`、防御点は `1 - max(MIA, AIA)` です。MIAのK=12正規化値はローカル代理集合で飽和することがあるため、候補比較ではβも確認します。",
        "",
        "公式の攻撃候補集合・攻撃者平均・最終Anonが揃うまでは、提出前の比較用スコアとして扱います。",
    ]
    return "\n".join(lines) + "\n"


def run_evaluator(args: argparse.Namespace) -> dict[str, Any]:
    """Run the existing evaluator and return its JSON report."""
    evaluator = Path(__file__).with_name("evaluate_resistance.py")
    output = args.work / "resistance_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(evaluator),
        "--model",
        args.model,
        "--c",
        str(args.c),
        "--data",
        str(args.data),
        "--kit-root",
        str(args.kit_root),
        "--out",
        str(output),
    ]
    if args.score is not None:
        command.extend(["--score", str(args.score)])
    else:
        raise ValueError("the underlying evaluator requires --score; pass a utility score JSON")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return json.loads(output.read_text(encoding="utf-8"))


def load_or_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.resistance_report is not None:
        return json.loads(args.resistance_report.read_text(encoding="utf-8"))
    return run_evaluator(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--c", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path, help="official-kit utility score JSON")
    parser.add_argument("--data", required=True, type=Path, help="directory containing B_self.csv, A_bg.csv, schema.json")
    parser.add_argument("--kit-root", required=True, type=Path)
    parser.add_argument("--phi", type=float, help="organizer-provided team10 phi_i; applies clip(AIA - phi_i, 0, 1)")
    parser.add_argument("--out", required=True, type=Path, help="compact JSON output")
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--work", type=Path, help="temporary evaluator output directory")
    parser.add_argument("--resistance-report", type=Path, help="reuse an existing resistance_report.json")
    args = parser.parse_args()
    args.work = args.work or args.out.parent / ".defense-score-work"
    report = load_or_run(args)
    summary = build_summary(report, args.phi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    print(json.dumps(summary["defense"], ensure_ascii=False))


if __name__ == "__main__":
    main()
