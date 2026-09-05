"""Generate the last-attempt candidate from complete official feedback.

The MIA file is conservatively reverted to the byte-identical v4 candidate
because the guided result did not beat the v4 hidden-power estimate beyond the
combined leaderboard rounding interval.  AIA target blocks are copied intact
from the best measured source among v4, independent, and routed v6.

This module never reads the team token, creates an archive, or submits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS  # noqa: E402
from attack_phase_v5.blend_feedback_winners import (  # noqa: E402
    blend_target_winners,
)
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    load_public_targets,
    sha256_file,
    validate_aia,
    validate_mia,
    verify_targets_manifest,
    write_candidate,
)
from attack_phase_v6.generate_submission_v6 import (  # noqa: E402
    dataframe_csv_sha256,
    feedback_winner_power,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


SOURCE_ORDER = ("v4", "independent", "routed")
SCORE_ROUNDING_HALF_WIDTH = 0.00005
COMBINED_MIA_ROUNDING_WIDTH = 2 * SCORE_ROUNDING_HALF_WIDTH
EXPECTED_FINAL_AIA_POWER = 0.1506204125415067
EXPECTED_INPUT_HASHES = {
    "v4_mia": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "v4_aia": "651D754683BCE4B6F22FD7311AB1584B01C2F42FBB73B13996303472411DFA3A",
    "independent_aia": "BDEA52C1FD7F77F9CF7054E8C9D64F3307B455521318FD73495B28A86BCF35E1",
    "routed_aia": "CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F",
    "feedback_v4": "A3579D160B8E0ED9AC2DB5BF140C1B7D73E2A2FCE524493985998D95C7CB753A",
    "feedback_independent": "D08BBD21B101134F61F12DAE27EB1A8A9EEF8454B726440C6FA7037646CA5C2C",
    "feedback_routed": "A4D1B16FAD78EA8C0382D24E12CB02F5D5E8C3FA5B39C3045C422D2B562664B4",
    "targets_manifest": "A7653F34B890F7E1095153AB50C75E5CC30EDF4272A8D83BFE7065D5B81746E5",
}
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": EXPECTED_INPUT_HASHES["v4_mia"],
    "F_aia.csv": "E1D57A40DA372FC8B14E3129E386D0265170CDA7BA06EF0544F8319FF189F465",
}
REQUIRED_MIA_METRICS = ("tpr_at_low_fpr", "tpr_rare", "auc")
REQUIRED_AIA_METRICS = (
    "r_time_all",
    "r_time_rare",
    "p_m",
    "p_c",
    "p_m_rare",
    "p_c_rare",
)
EXPECTED_CONSTANTS = {
    "low_fpr": 0.01,
    "tau_time": 0.5,
    "aia_rare_se_z": 1.64,
    "mia_band_edges": [0.25, 0.5, 0.75],
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    feedback_dir = REPO_ROOT / "attack_phase_v6" / "official_feedback"
    parser = argparse.ArgumentParser(
        description="3回分の公式target別実測winnerから最終候補CSVを生成します。"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--v4-dir", type=Path, default=team_dir / "attack-candidate-v4"
    )
    parser.add_argument(
        "--independent-dir",
        type=Path,
        default=team_dir / "attack-candidate-daybreak-independent",
    )
    parser.add_argument(
        "--routed-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-routed-guided-probe",
    )
    parser.add_argument("--feedback-v4", type=Path, default=feedback_dir / "v4.json")
    parser.add_argument(
        "--feedback-independent",
        type=Path,
        default=feedback_dir / "probe_independent.json",
    )
    parser.add_argument(
        "--feedback-routed",
        type=Path,
        default=feedback_dir / "routed_guided_probe.json",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-feedback-winners",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "local-output"
        / "attack-v6"
        / "final_feedback_winners.json",
    )
    return parser.parse_args()


def verify_named_hashes(
    paths: Mapping[str, Path],
    *,
    expected: Mapping[str, str] = EXPECTED_INPUT_HASHES,
) -> dict[str, str]:
    """Hash-lock every final-candidate input before reading its contents."""
    if set(paths) != set(expected):
        missing = sorted(set(expected) - set(paths))
        extra = sorted(set(paths) - set(expected))
        raise ValueError(f"final input set mismatch: missing={missing}, extra={extra}")
    observed = {name: sha256_file(Path(path)) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"final input hash mismatch: {mismatches}")
    return observed


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _validate_probability_metrics(
    payload: Mapping[str, Any], required: tuple[str, ...], *, label: str
) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"{label} is incomplete: missing={missing}")
    output = {
        key: _finite_number(payload[key], label=f"{label}.{key}") for key in required
    }
    outside = {key: value for key, value in output.items() if not 0.0 <= value <= 1.0}
    if outside:
        raise ValueError(f"{label} probabilities are outside [0,1]: {outside}")
    return output


def _validate_target_map(
    value: Any, *, label: str, probability: bool
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        try:
            key = str(int(raw_key))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} has an invalid target key: {raw_key!r}") from error
        if key in normalized:
            raise ValueError(f"{label} has a duplicate normalized target key: {key}")
        normalized[key] = _finite_number(raw_value, label=f"{label}.{key}")
    expected = set(EXPECTED_TARGET_IDS)
    if set(normalized) != expected:
        missing = sorted(expected - set(normalized), key=int)
        extra = sorted(set(normalized) - expected, key=int)
        raise ValueError(
            f"{label} target set mismatch: missing={missing}, extra={extra}"
        )
    if probability:
        outside = {
            key: score
            for key, score in normalized.items()
            if not 0.0 <= score <= 1.0
        }
        if outside:
            raise ValueError(f"{label} values are outside [0,1]: {outside}")
    elif any(value < 0.0 for value in normalized.values()):
        raise ValueError(f"{label} contains a negative value")
    return normalized


def load_complete_feedback(
    path: Path,
    *,
    expected_candidate: str,
    require_score_1: bool,
    require_all_displayed_scores: bool = False,
) -> dict[str, Any]:
    """Load and validate all fields needed to trust a detailed score result."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read official feedback {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"official feedback {path} must contain an object")
    if payload.get("candidate") != expected_candidate:
        raise ValueError(
            f"feedback candidate mismatch for {path}: "
            f"expected={expected_candidate!r}, observed={payload.get('candidate')!r}"
        )

    normalized = dict(payload)
    normalized["mia"] = _validate_probability_metrics(
        payload.get("mia"), REQUIRED_MIA_METRICS, label=f"{path.name}.mia"
    )
    normalized["aia"] = _validate_probability_metrics(
        payload.get("aia"), REQUIRED_AIA_METRICS, label=f"{path.name}.aia"
    )
    normalized["mia_per_target"] = _validate_target_map(
        payload.get("mia_per_target"),
        label=f"{path.name}.mia_per_target",
        probability=False,
    )
    normalized["aia_per_target"] = _validate_target_map(
        payload.get("aia_per_target"),
        label=f"{path.name}.aia_per_target",
        probability=True,
    )
    if payload.get("constants") != EXPECTED_CONSTANTS:
        raise ValueError(
            f"{path.name}.constants mismatch: "
            f"expected={EXPECTED_CONSTANTS}, observed={payload.get('constants')}"
        )
    if require_score_1:
        normalized["score_1_displayed"] = _finite_number(
            payload.get("score_1_displayed"), label=f"{path.name}.score_1_displayed"
        )
    if require_all_displayed_scores:
        for score_number in range(1, 6):
            key = f"score_{score_number}_displayed"
            normalized[key] = _finite_number(
                payload.get(key), label=f"{path.name}.{key}"
            )
    return normalized


def infer_mia_power(feedback: Mapping[str, Any]) -> float:
    """Reconstruct hidden MIA power from displayed total and exact target AIA."""
    score = _finite_number(feedback.get("score_1_displayed"), label="score_1")
    aia_scores = feedback.get("aia_per_target")
    if not isinstance(aia_scores, Mapping):
        raise ValueError("aia_per_target is required to infer MIA power")
    values = np.asarray(
        [
            _finite_number(aia_scores[target_id], label=f"aia_per_target.{target_id}")
            for target_id in EXPECTED_TARGET_IDS
        ],
        dtype=float,
    )
    return float(score - values.mean())


def choose_final_mia(
    baseline_feedback: Mapping[str, Any],
    guided_feedback: Mapping[str, Any],
    *,
    combined_rounding_width: float = COMBINED_MIA_ROUNDING_WIDTH,
) -> dict[str, Any]:
    """Keep guided only for a lift larger than both display-rounding bands."""
    baseline_power = infer_mia_power(baseline_feedback)
    guided_power = infer_mia_power(guided_feedback)
    delta = guided_power - baseline_power
    selected = "guided" if delta > combined_rounding_width else "v4"
    return {
        "selected": selected,
        "baseline_v4_power": baseline_power,
        "guided_power": guided_power,
        "guided_minus_v4": delta,
        "combined_rounding_width": combined_rounding_width,
        "equivalent_within_rounding": abs(delta) <= combined_rounding_width,
    }


def assert_expected_output_hashes(
    observed: Mapping[str, str],
    *,
    expected: Mapping[str, str] = EXPECTED_OUTPUT_HASHES,
) -> None:
    """Fail before writing if final CSV serialization or routing has drifted."""
    if dict(observed) != dict(expected):
        raise ValueError(
            f"final output hash mismatch before write: "
            f"expected={dict(expected)}, observed={dict(observed)}"
        )


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    v4_dir = args.v4_dir.resolve()
    independent_dir = args.independent_dir.resolve()
    routed_dir = args.routed_dir.resolve()
    feedback_paths = {
        "v4": args.feedback_v4.resolve(),
        "independent": args.feedback_independent.resolve(),
        "routed": args.feedback_routed.resolve(),
    }
    input_paths = {
        "v4_mia": v4_dir / "F_mia.csv",
        "v4_aia": v4_dir / "F_aia.csv",
        "independent_aia": independent_dir / "F_aia.csv",
        "routed_aia": routed_dir / "F_aia.csv",
        "feedback_v4": feedback_paths["v4"],
        "feedback_independent": feedback_paths["independent"],
        "feedback_routed": feedback_paths["routed"],
        "targets_manifest": targets_dir / "SHA256SUMS.txt",
    }

    # Every external input is hash-locked before any candidate path is touched.
    input_hashes = verify_named_hashes(input_paths)
    target_file_hashes = verify_targets_manifest(targets_dir)
    feedback = {
        "v4": load_complete_feedback(
            feedback_paths["v4"], expected_candidate="v4", require_score_1=False
        ),
        "independent": load_complete_feedback(
            feedback_paths["independent"],
            expected_candidate="probe_independent",
            require_score_1=True,
        ),
        "routed": load_complete_feedback(
            feedback_paths["routed"],
            expected_candidate="routed_guided_probe",
            require_score_1=True,
            require_all_displayed_scores=True,
        ),
    }
    mia_decision = choose_final_mia(feedback["independent"], feedback["routed"])
    if mia_decision["selected"] != "v4":
        raise ValueError(
            "final MIA decision no longer selects v4; review new official feedback "
            f"instead of silently changing the locked recipe: {mia_decision}"
        )

    v4_mia = pd.read_csv(input_paths["v4_mia"])
    aia_sources = {
        "v4": pd.read_csv(input_paths["v4_aia"]),
        "independent": pd.read_csv(input_paths["independent_aia"]),
        "routed": pd.read_csv(input_paths["routed_aia"]),
    }
    _, challenges = load_public_targets(targets_dir)
    validation: dict[str, Any] = {
        "v4_mia": validate_mia(v4_mia, v4_mia),
    }
    for label, frame in aia_sources.items():
        validation[f"{label}_aia"] = validate_aia(frame, challenges)

    score_maps = {
        label: feedback[label]["aia_per_target"] for label in SOURCE_ORDER
    }
    final_aia, blend_diagnostics = blend_target_winners(aia_sources, score_maps)
    validation["final_aia"] = validate_aia(final_aia, challenges)
    final_aia_power = feedback_winner_power(score_maps)
    if not math.isclose(
        final_aia_power, EXPECTED_FINAL_AIA_POWER, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError(
            "final AIA power gate failed: "
            f"expected={EXPECTED_FINAL_AIA_POWER}, observed={final_aia_power}"
        )

    official_dist = official_validate.load_dist(targets_dir)
    mia_result = official_validate.validate_mia_submission(v4_mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(final_aia, {}, official_dist)
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            "official logical validator failed: "
            f"MIA={mia_result.errors}, AIA={aia_result.errors}"
        )
    validation["official_validator"] = "PASS"

    # This gate runs after all calculations/validators but before mkdir or CSV writes.
    prewrite_hashes = {
        "F_mia.csv": input_hashes["v4_mia"],
        "F_aia.csv": dataframe_csv_sha256(final_aia),
    }
    assert_expected_output_hashes(prewrite_hashes)

    candidate_dir = args.candidate_dir.resolve()
    written_hashes = write_candidate(
        candidate_dir,
        v4_mia,
        final_aia,
        mia_source=input_paths["v4_mia"],
    )
    if written_hashes != EXPECTED_OUTPUT_HASHES:
        raise ValueError(
            "written final candidate differs from pre-write hash gate: "
            f"expected={EXPECTED_OUTPUT_HASHES}, observed={written_hashes}"
        )

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_final_feedback_blend",
        "attack_version": 6,
        "candidate_dir": str(candidate_dir),
        "mia_decision": mia_decision,
        "aia": {
            "source_order_for_ties": list(SOURCE_ORDER),
            "power": final_aia_power,
            **blend_diagnostics,
        },
        "validation": validation,
        "input_hashes": input_hashes,
        "expected_input_hashes": EXPECTED_INPUT_HASHES,
        "prewrite_output_hashes": prewrite_hashes,
        "candidate_hashes": written_hashes,
        "target_file_hashes": target_file_hashes,
        "network_access": False,
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: final feedback-winner candidate generated")
    print(f"MIA selection: {mia_decision['selected']}")
    print(f"AIA winner power: {final_aia_power:.15f}")
    print(f"winner counts: {blend_diagnostics['winner_counts']}")
    print(f"candidate: {candidate_dir}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
