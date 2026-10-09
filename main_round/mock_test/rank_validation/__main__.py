"""Compare matched historical scores and check known official rank directions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
import math
from pathlib import Path

from .core import compare_paired, SCORE_KEYS
from .handoff import load_handoff
from .report import render
from .sensitivity import mock_sensitivity
from .feedback import add_feedback, validate_data


def sign(value, tolerance=1e-12):
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def official_check(comparison, official):
    a, b = comparison["model_a"], comparison["model_b"]
    out = []
    for score in SCORE_KEYS:
        va, vb = official.get(a, {}).get(score), official.get(b, {}).get(score)
        local = comparison["metrics"][score]["mean_delta_b_minus_a"]
        delta = None if va is None or vb is None else vb - va
        delta_interval = None
        if delta is not None:
            ia = official[a].get("score_intervals", {}).get(score, [va, va])
            ib = official[b].get("score_intervals", {}).get(score, [vb, vb])
            delta_interval = [ib[0] - ia[1], ib[1] - ia[0]]
        if delta is None:
            status = "missing_official"
        elif delta_interval[0] <= 0 <= delta_interval[1] and delta_interval[0] != delta_interval[1]:
            status = "display_precision_inconclusive"
        elif sign(delta) == sign(local):
            status = "both_tied" if sign(delta) == 0 else "same_direction"
        elif sign(delta) == 0 or sign(local) == 0:
            status = "tie_mismatch"
        else:
            status = "opposite_direction"
        out.append({"experiment_id": comparison["experiment_id"], "model_a": a, "model_b": b,
                    "score_key": score, "local_delta": local, "official_delta": delta, "official_delta_interval": delta_interval, "status": status})
    return out


def build_report(data):
    validate_data(data)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in data["records"]:
        grouped[row["experiment_id"]][row["model_id"]].append(row)
    official = {r["model_id"]: r for r in data["official"]}
    comparisons, blocked, checks, summaries = [], [], [], []
    for experiment, models in sorted(grouped.items()):
        for model, rows in sorted(models.items()):
            summary = {"experiment_id": experiment, "model_id": model, "rows": len(rows)}
            for key in SCORE_KEYS:
                values = [r.get(key) for r in rows]
                summary[key] = math.fsum(values) / len(values) if all(v is not None and math.isfinite(v) for v in values) else None
            summaries.append(summary)
        for a, b in combinations(sorted(models), 2):
            try:
                result = compare_paired(models[a], models[b], model_a=a, model_b=b)
            except ValueError as exc:
                blocked.append({"experiment_id": experiment, "model_a": a, "model_b": b, "reason": str(exc)})
                continue
            comparisons.append(result)
            result["mock_sensitivity"] = mock_sensitivity(result, models[a])
            checks.extend(official_check(result, official))
    total_pairs = sorted({tuple(sorted((r["model_a"], r["model_b"]))) for r in checks if r["score_key"] == "score_1" and r["official_delta"] is not None})
    protocol_checks = []
    for experiment in sorted(grouped):
        known = [r for r in checks if r["experiment_id"] == experiment and r["score_key"] == "score_1" and r["official_delta"] is not None]
        contradictions = [r for r in known if r["status"] in ("opposite_direction", "tie_mismatch")]
        protocol_checks.append({"experiment_id": experiment, "known_total_pair_count": len(known),
                                "known_total_contradictions": len(contradictions),
                                "status": "fails_known_official_ranking" if contradictions else "limited_known_agreement" if known else "official_validation_unavailable",
                                "new_candidate_ranking_certified": False})
    protocol_by_id = {r["experiment_id"]: r for r in protocol_checks}
    for comparison in comparisons:
        delta = comparison["metrics"]["score_1"]["mean_delta_b_minus_a"]
        reasons = []
        warnings = []
        if protocol_by_id[comparison["experiment_id"]]["status"] == "fails_known_official_ranking":
            reasons.append("この試験は既知の本番総合順位を再現できない")
        if comparison["case_mean_delta_range"][0] < -1e-12 < comparison["case_mean_delta_range"][1]:
            warnings.append("問題ごとに勝敗が反転（単独では不採用理由にしない）")
        if comparison["release_mean_delta_range"][0] < -1e-12 < comparison["release_mean_delta_range"][1]:
            warnings.append("個別加工ごとに勝敗が反転（系列平均も参照）")
        scenario = comparison["mock_sensitivity"]
        if scenario["available"]:
            strata = [r["mean_delta"]["score_1"] for r in scenario["rare_positive_strata"].values()]
            if min(strata) < -1e-12 < max(strata):
                reasons.append("希少所属人数によって勝敗が反転")
            families = [r["score_1"] for r in scenario["family_mean_delta"].values()]
            if min(families) < -1e-12 < max(families):
                reasons.append("防御系列によって勝敗が反転")
        status = "hold" if reasons else "no_mean_improvement" if delta <= 1e-12 else "stable_in_observed_conditions_only"
        comparison["adoption_gate"] = {"status": status, "reasons": reasons or ["本戦への適用範囲と未知候補の精度は未確認"],
                                      "warnings": warnings,
                                      "designed_after_observing_this_archive": True, "prospectively_validated": False}
    return {"schema_version": 1, "source_provenance": data.get("provenance"),
            "experiments": data.get("experiments", []), "official": data["official"], "unresolved": data["unresolved"],
            "comparisons": comparisons, "blocked_comparisons": blocked, "official_comparisons": checks,
            "model_summaries": summaries, "unique_official_total_pairs_available": total_pairs,
            "protocol_checks": protocol_checks,
            "official_ranking_validated": False, "fitted_calibration": False,
            "new_predictions_generated": False, "official_submissions_performed": False}


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--handoff", type=Path)
    source.add_argument("--records", type=Path, help="JSON object with records, official, unresolved, experiments and provenance")
    parser.add_argument("--official-feedback", type=Path, help="Explicit user-supplied displayed totals; preserves exact scores and unknown breakdowns")
    parser.add_argument("--check-public-profile", action="store_true", help="Compare public coverage of the fixed local 12 defenses and actual 22 targets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = Path(__file__).resolve().parents[1] / "local-output"
    if output == allowed.resolve() or not output.is_relative_to(allowed.resolve()) or output.exists():
        parser.error("Use a new output directory below mock_test/local-output")
    data = load_handoff(args.handoff) if args.handoff else json.loads(args.records.read_text(encoding="utf-8-sig"))
    if args.official_feedback:
        data = add_feedback(data, json.loads(args.official_feedback.read_text(encoding="utf-8-sig")))
    report = build_report(data)
    output.mkdir(parents=True, exist_ok=False)
    if args.check_public_profile:
        from .public_profiles import run
        repo = Path(__file__).resolve().parents[3]
        profile = run(repo, output / "public-profiles")
        report["public_profile_check"] = profile
        for comparison in report["comparisons"]:
            if comparison["mock_sensitivity"]["available"]:
                comparison["ranking_scope"] = profile["coverage"]["model_ranking"]
                if comparison["ranking_scope"]["unsupported"]:
                    comparison["adoption_gate"]["status"] = "hold"
                    comparison["adoption_gate"]["reasons"].append("手元12加工の公開特徴の範囲を本戦対象が超える")
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "normalized_records.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.html").write_text(render(report), encoding="utf-8")
    save_csv(output / "official_rank_checks.csv", report["official_comparisons"], ["experiment_id", "model_a", "model_b", "score_key", "local_delta", "official_delta", "status"])
    save_csv(output / "model_summaries.csv", report["model_summaries"], ["experiment_id", "model_id", "rows", *SCORE_KEYS])
    print(json.dumps({"comparisons": len(report["comparisons"]), "blocked": len(report["blocked_comparisons"]),
                      "opposite_directions": sum(r["status"] == "opposite_direction" for r in report["official_comparisons"]),
                      "unique_official_total_pairs_available": report["unique_official_total_pairs_available"],
                      "report": str(output / "report.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
