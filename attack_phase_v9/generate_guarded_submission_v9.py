"""Hash-gated, whole-target guarded composition for the last local candidate.

This is risk management, not a validated new attack uplift: measured positive
blocks are replayed exactly, while seven historically zero blocks retain v7.
The score floor is conditional on unchanged hidden data/scorer and correct
feedback-to-artifact association. No token, ZIP, network, or submission I/O.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS
from attack_phase_v5.blend_feedback_winners import blend_target_winners
from attack_phase_v5.generate_submission_v5 import validate_mia, write_candidate
from attack_phase_v6.generate_final_submission_v6 import (
    EXPECTED_INPUT_HASHES as MEASURED_INPUT_HASHES,
    EXPECTED_OUTPUT_HASHES as MEASURED_OUTPUT_HASHES,
    REPO_ROOT, SOURCE_ORDER, SCORE_ROUNDING_HALF_WIDTH,
    assert_expected_output_hashes, dataframe_csv_sha256, infer_mia_power,
    load_complete_feedback, verify_named_hashes,
)
from attack_phase_v7.generate_submission_v7 import (
    EXPECTED_OUTPUT_HASHES as V7_OUTPUT_HASHES, validate_variable_aia,
)
from attack_phase_v8.actual_inputs import load_actual_inputs
from attack_phase_v8.event_additions import strict_aia_frame
from pwscup2026.kit import validate as official_validate


ZERO_TARGETS = ("6", "7", "8", "9", "17", "18", "19")
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": MEASURED_OUTPUT_HASHES["F_mia.csv"],
    "F_aia.csv": "63944D5FAA2C08E5E2E40415FFBB16B3153B27EBDD052CD68D7ADB59A403A6E0",
}


def canonical_aia(frame: pd.DataFrame) -> pd.DataFrame:
    result = strict_aia_frame(frame).sort_values(
        ["target_k", "challenge_row_id"], kind="mergesort").reset_index(drop=True)
    if {str(t) for t in result.target_k.unique()} != set(EXPECTED_TARGET_IDS):
        raise ValueError("AIA target set differs from the 20 official targets")
    for _, block in result.groupby("target_k"):
        if block.challenge_row_id.tolist() != list(range(326)):
            raise ValueError("AIA block must contain exactly challenge IDs 0..325")
    return result


def compose_guarded_aia(
    sources: Mapping[str, pd.DataFrame],
    score_maps: Mapping[str, Mapping[str, float]],
    v7: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select whole measured winners, substituting v7 only on all-zero blocks."""
    if set(sources) != set(SOURCE_ORDER) or set(score_maps) != set(SOURCE_ORDER):
        raise ValueError("exactly v4, independent and routed sources are required")
    scores = {}
    for label in SOURCE_ORDER:
        if set(score_maps[label]) != set(EXPECTED_TARGET_IDS):
            raise ValueError("feedback target set differs")
        values = np.asarray([score_maps[label][t] for t in EXPECTED_TARGET_IDS], dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("feedback scores must be finite probabilities")
        scores[label] = dict(zip(EXPECTED_TARGET_IDS, values.tolist()))
    actual_zero = {t for t in EXPECTED_TARGET_IDS
                   if all(scores[label][t] == 0 for label in SOURCE_ORDER)}
    if actual_zero != set(ZERO_TARGETS):
        raise ValueError("historical all-zero target set changed; review the recipe")
    normalized = {label: canonical_aia(sources[label]) for label in SOURCE_ORDER}
    original_v7 = canonical_aia(v7)
    measured, blend = blend_target_winners(normalized, scores)
    measured = canonical_aia(measured)
    output = measured.copy()
    zero_mask = output.target_k.isin([int(t) for t in ZERO_TARGETS])
    output.loc[zero_mask, "time_hat"] = original_v7.loc[zero_mask, "time_hat"].to_numpy()
    if not output.loc[~zero_mask].equals(measured.loc[~zero_mask]):
        raise RuntimeError("a measured positive block was modified")
    if not output.loc[zero_mask].equals(original_v7.loc[zero_mask]):
        raise RuntimeError("an experimental block differs from v7")
    details = {
        "strategy": "measured_positive_whole_blocks_plus_v7_zero_blocks",
        "measured_winners": blend["winners"],
        "experimental_zero_targets": list(ZERO_TARGETS),
        "measured_aia_floor": float(sum(blend["winners"][t]["score"]
                                        for t in EXPECTED_TARGET_IDS) / 20),
        "measured_winner_aia_hash": dataframe_csv_sha256(measured),
        "changed_rows_vs_v7": int((output.time_hat != original_v7.time_hat).sum()),
        "changed_targets_vs_v7": [t for t in EXPECTED_TARGET_IDS if
            (output.loc[output.target_k.eq(int(t)), "time_hat"].to_numpy() !=
             original_v7.loc[original_v7.target_k.eq(int(t)), "time_hat"].to_numpy()).any()],
    }
    return output, details


def conditional_floor(independent_feedback: Mapping[str, Any], aia_floor: float) -> dict[str, Any]:
    if not np.isfinite(aia_floor) or not 0 <= aia_floor <= 1:
        raise ValueError("invalid AIA floor")
    mia_nominal = infer_mia_power(independent_feedback)
    if not 0 <= mia_nominal <= 1:
        raise ValueError("inferred MIA power is outside [0,1]")
    nominal = mia_nominal + aia_floor
    return {
        "mia_from_rounded_official_total": mia_nominal,
        "mia_rounding_half_width": SCORE_ROUNDING_HALF_WIDTH,
        "retained_aia_floor": aia_floor,
        "nominal_known_component": nominal,
        "conditional_lower_bound": nominal - SCORE_ROUNDING_HALF_WIDTH,
        "additional_seven_target_contribution": "unknown, nonnegative under the published scorer",
        "not_a_new_official_score": True,
        "not_a_shadow_forecast": True,
        "not_proven_superior_to_v7": True,
        "assumptions": [
            "same hidden challenge rows, labels, times, target set and scoring implementation",
            "the stored official feedback corresponds to the hash-locked submitted artifacts",
            "score_1 display was rounded to four decimal places",
            "per-target AIA scores are clipped nonnegative and averaged over these 20 targets",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path,
                        help="explicit actual team24 prelim directory; no practice fallback")
    parser.add_argument("--targets", type=Path,
                        default=REPO_ROOT / "team-shared/prelim/attack-targets")
    args = parser.parse_args()
    actual = load_actual_inputs(args.dist.resolve(), args.targets.resolve())
    team = REPO_ROOT / "submissions/prelim/team24"
    feedback_dir = REPO_ROOT / "attack_phase_v6/official_feedback"
    paths = {
        "v4_mia": team / "attack-candidate-v4/F_mia.csv",
        "v4_aia": team / "attack-candidate-v4/F_aia.csv",
        "independent_aia": team / "attack-candidate-daybreak-independent/F_aia.csv",
        "routed_aia": team / "attack-candidate-v6-routed-guided-probe/F_aia.csv",
        "feedback_v4": feedback_dir / "v4.json",
        "feedback_independent": feedback_dir / "probe_independent.json",
        "feedback_routed": feedback_dir / "routed_guided_probe.json",
        "targets_manifest": args.targets.resolve() / "SHA256SUMS.txt",
        "v7_aia": team / "attack-candidate-v7-feedback-stack/F_aia.csv",
        "independent_probe_mia": team / "attack-candidate-v5-probe-independent/F_mia.csv",
        "independent_probe_aia": team / "attack-candidate-v5-probe-independent/F_aia.csv",
    }
    expected = {**MEASURED_INPUT_HASHES,
                "v7_aia": V7_OUTPUT_HASHES["F_aia.csv"],
                "independent_probe_mia": MEASURED_INPUT_HASHES["v4_mia"],
                "independent_probe_aia": MEASURED_INPUT_HASHES["independent_aia"]}
    input_hashes = verify_named_hashes(paths, expected=expected)
    feedback = {label: load_complete_feedback(
        paths[f"feedback_{label}"],
        expected_candidate={"v4": "v4", "independent": "probe_independent",
                            "routed": "routed_guided_probe"}[label],
        require_score_1=label != "v4", require_all_displayed_scores=label == "routed")
        for label in SOURCE_ORDER}
    sources = {label: pd.read_csv(paths[f"{label}_aia"]) for label in SOURCE_ORDER}
    aia, composition = compose_guarded_aia(
        sources, {label: feedback[label]["aia_per_target"] for label in SOURCE_ORDER},
        pd.read_csv(paths["v7_aia"]))
    if composition["measured_winner_aia_hash"] != MEASURED_OUTPUT_HASHES["F_aia.csv"]:
        raise ValueError("measured winner reconstruction drifted")
    mia = pd.read_csv(paths["v4_mia"])
    validation = {"mia": validate_mia(mia, mia),
                  "aia": validate_variable_aia(aia, actual.challenges)}
    official_dist = official_validate.load_dist(args.targets.resolve())
    results = [official_validate.validate_mia_submission(mia, {}, official_dist),
               official_validate.validate_aia_submission(aia, {}, official_dist)]
    if not all(result.ok for result in results):
        raise ValueError(f"official logical validator failed: {[r.errors for r in results]}")
    validation["official_logical_validator"] = "PASS"
    output_hashes = {"F_mia.csv": input_hashes["v4_mia"],
                     "F_aia.csv": dataframe_csv_sha256(aia)}
    assert_expected_output_hashes(output_hashes, expected=EXPECTED_OUTPUT_HASHES)
    floor = conditional_floor(feedback["independent"], composition["measured_aia_floor"])
    candidate = team / "attack-candidate-v9-guarded"
    # Never replace a differently populated candidate, including a user's edits.
    if candidate.exists():
        if not candidate.is_dir() or {p.name for p in candidate.iterdir()} != set(EXPECTED_OUTPUT_HASHES):
            raise ValueError("existing candidate directory differs; refusing overwrite")
        verify_named_hashes({name: candidate / name for name in EXPECTED_OUTPUT_HASHES},
                            expected=EXPECTED_OUTPUT_HASHES)
    written = write_candidate(candidate, mia, aia, mia_source=paths["v4_mia"])
    assert_expected_output_hashes(written, expected=EXPECTED_OUTPUT_HASHES)
    report = {"attack_version": "v9-guarded", "candidate_dir": str(candidate),
              "actual_inputs": actual.provenance, "input_hashes": input_hashes,
              "output_hashes": written, "composition": composition,
              "conditional_score_floor": floor, "validation": validation,
              "new_research_methods_promoted": [], "v7_modified": False,
              "token_read": False, "archive_created": False,
              "network_access": False, "submission_performed": False}
    report_path = REPO_ROOT / "local-output/attack-v9/guarded_candidate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                           + "\n", encoding="utf-8")
    print("PASS: guarded v9 CSV candidate; official logical validator PASS")
    print(f"candidate: {candidate}")
    print(f"hashes: {written}")
    print(f"conditional lower bound (not measured): {floor['conditional_lower_bound']:.12f}")


if __name__ == "__main__":
    main()
