"""Generate a separate last-attempt candidate with zero-target complements.

All measured-positive target blocks remain exactly equal to the conservative
feedback-winner candidate.  For the seven targets that scored exactly zero in
all measured sources, the submitted routed horizon decision is complemented
(``10.0 <-> 11.0``).  Since each target score is clipped to ``[0, 1]``, these
blocks preserve the known zero lower bound while testing the opposite sign.

This module never reads the team token, creates an archive, or submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import OFFICIAL_COLUMNS  # noqa: E402
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    EXPECTED_AIA_ROWS_PER_TARGET,
    load_public_targets,
    validate_aia,
    validate_mia,
    verify_targets_manifest,
    write_candidate,
)
from attack_phase_v6.generate_final_submission_v6 import (  # noqa: E402
    assert_expected_output_hashes,
    dataframe_csv_sha256,
    load_complete_feedback,
    verify_named_hashes,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


ZERO_TARGET_IDS = ("6", "7", "8", "9", "17", "18", "19")
POSITIVE_TARGET_IDS = tuple(
    target_id for target_id in EXPECTED_TARGET_IDS if target_id not in ZERO_TARGET_IDS
)
EXPECTED_INPUT_HASHES = {
    "safe_mia": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "safe_aia": "E1D57A40DA372FC8B14E3129E386D0265170CDA7BA06EF0544F8319FF189F465",
    "routed_aia": "CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F",
    "feedback_routed": "A4D1B16FAD78EA8C0382D24E12CB02F5D5E8C3FA5B39C3045C422D2B562664B4",
    "targets_manifest": "A7653F34B890F7E1095153AB50C75E5CC30EDF4272A8D83BFE7065D5B81746E5",
}
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": EXPECTED_INPUT_HASHES["safe_mia"],
    "F_aia.csv": "2CFDC03D9EB2A2F096BE3A872BC836CA0AAE16FF170F20F0124F85F14E7A3D82",
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    parser = argparse.ArgumentParser(
        description="公式0点targetだけrouted horizonを完全反転した最終候補を生成します。"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--safe-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-feedback-winners",
    )
    parser.add_argument(
        "--routed-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-routed-guided-probe",
    )
    parser.add_argument(
        "--feedback-routed",
        type=Path,
        default=REPO_ROOT
        / "attack_phase_v6"
        / "official_feedback"
        / "routed_guided_probe.json",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-aggressive-complements",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / "local-output"
        / "attack-v6"
        / "final_aggressive_complements.json",
    )
    return parser.parse_args()


def _normalized_aia(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if list(frame.columns) != list(OFFICIAL_COLUMNS):
        raise ValueError(f"{label} has invalid AIA columns")
    if frame.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError(f"{label} has duplicate AIA keys")
    output = frame.copy()
    output["target_k"] = output["target_k"].astype(int)
    output["challenge_row_id"] = output["challenge_row_id"].astype(int)
    return output


def complement_zero_target_blocks(
    safe: pd.DataFrame,
    routed: pd.DataFrame,
    *,
    target_ids: Iterable[str] = EXPECTED_TARGET_IDS,
    zero_target_ids: Iterable[str] = ZERO_TARGET_IDS,
    rows_per_target: int = EXPECTED_AIA_ROWS_PER_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complement routed binary blocks and preserve every other safe block."""
    safe = _normalized_aia(safe, label="safe")
    routed = _normalized_aia(routed, label="routed")
    target_ids = tuple(str(int(value)) for value in target_ids)
    zero_target_ids = tuple(str(int(value)) for value in zero_target_ids)
    if not zero_target_ids or not set(zero_target_ids) < set(target_ids):
        raise ValueError("zero targets must be a non-empty proper subset of targets")
    if len(set(zero_target_ids)) != len(zero_target_ids):
        raise ValueError("zero targets contain duplicates")

    safe_targets = tuple(
        sorted(safe["target_k"].astype(str).unique().tolist(), key=int)
    )
    routed_targets = tuple(
        sorted(routed["target_k"].astype(str).unique().tolist(), key=int)
    )
    if safe_targets != tuple(sorted(target_ids, key=int)) or routed_targets != tuple(
        sorted(target_ids, key=int)
    ):
        raise ValueError(
            "AIA target set mismatch: "
            f"expected={target_ids}, safe={safe_targets}, routed={routed_targets}"
        )
    safe_keys = safe[["target_k", "challenge_row_id"]].reset_index(drop=True)
    routed_keys = routed[["target_k", "challenge_row_id"]].reset_index(drop=True)
    if not safe_keys.equals(routed_keys):
        raise ValueError("safe and routed AIA key order differs")

    output = safe.copy()
    per_target: dict[str, dict[str, Any]] = {}
    zero_set = set(zero_target_ids)
    for target_id in target_ids:
        target = int(target_id)
        mask = output["target_k"].eq(target)
        if int(mask.sum()) != rows_per_target:
            raise ValueError(
                f"target {target_id} has {int(mask.sum())} rows; "
                f"expected {rows_per_target}"
            )
        if target_id not in zero_set:
            continue
        routed_values = routed.loc[mask, "time_hat"].to_numpy(dtype=float)
        count_10 = int(np.count_nonzero(routed_values == 10.0))
        count_11 = int(np.count_nonzero(routed_values == 11.0))
        if count_10 != rows_per_target // 2 or count_11 != rows_per_target // 2:
            raise ValueError(
                f"target {target_id} is not a balanced 10/11 routed block: "
                f"10={count_10}, 11={count_11}"
            )
        if count_10 + count_11 != rows_per_target:
            unexpected = sorted(set(routed_values) - {10.0, 11.0})
            raise ValueError(
                f"target {target_id} routed block is not purely binary: {unexpected}"
            )
        complemented = 21.0 - routed_values
        output.loc[mask, "time_hat"] = complemented
        safe_values = safe.loc[mask, "time_hat"].to_numpy(dtype=float)
        per_target[target_id] = {
            "rows": rows_per_target,
            "routed_counts": {"10.0": count_10, "11.0": count_11},
            "output_counts": {
                "10.0": int(np.count_nonzero(complemented == 10.0)),
                "11.0": int(np.count_nonzero(complemented == 11.0)),
            },
            "exact_complement_rows": int(
                np.count_nonzero(complemented == 21.0 - routed_values)
            ),
            "changed_rows_from_safe": int(
                np.count_nonzero(complemented != safe_values)
            ),
        }

    changed_targets: list[str] = []
    unchanged_positive: list[str] = []
    for target_id in target_ids:
        mask = output["target_k"].eq(int(target_id))
        changed = not np.array_equal(
            output.loc[mask, OFFICIAL_COLUMNS].to_numpy(),
            safe.loc[mask, OFFICIAL_COLUMNS].to_numpy(),
        )
        if changed:
            changed_targets.append(target_id)
        elif target_id not in zero_set:
            unchanged_positive.append(target_id)
    if set(changed_targets) != zero_set:
        raise ValueError(
            "changed target gate failed: "
            f"expected={sorted(zero_set, key=int)}, observed={changed_targets}"
        )
    expected_positive = set(target_ids) - zero_set
    if set(unchanged_positive) != expected_positive:
        raise ValueError("one or more measured-positive blocks changed")

    output = output[list(OFFICIAL_COLUMNS)].reset_index(drop=True)
    diagnostics = {
        "zero_targets": list(zero_target_ids),
        "positive_targets": [
            target_id for target_id in target_ids if target_id not in zero_set
        ],
        "changed_targets_from_safe": changed_targets,
        "positive_blocks_unchanged": True,
        "per_target": per_target,
    }
    return output, diagnostics


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    safe_dir = args.safe_dir.resolve()
    routed_dir = args.routed_dir.resolve()
    feedback_path = args.feedback_routed.resolve()
    input_paths = {
        "safe_mia": safe_dir / "F_mia.csv",
        "safe_aia": safe_dir / "F_aia.csv",
        "routed_aia": routed_dir / "F_aia.csv",
        "feedback_routed": feedback_path,
        "targets_manifest": targets_dir / "SHA256SUMS.txt",
    }

    input_hashes = verify_named_hashes(input_paths, expected=EXPECTED_INPUT_HASHES)
    target_file_hashes = verify_targets_manifest(targets_dir)
    feedback = load_complete_feedback(
        feedback_path,
        expected_candidate="routed_guided_probe",
        require_score_1=True,
        require_all_displayed_scores=True,
    )
    observed_zero_targets = tuple(
        target_id
        for target_id in EXPECTED_TARGET_IDS
        if feedback["aia_per_target"][target_id] == 0.0
    )
    if observed_zero_targets != ZERO_TARGET_IDS:
        raise ValueError(
            "official routed zero-target gate failed: "
            f"expected={ZERO_TARGET_IDS}, observed={observed_zero_targets}"
        )

    safe_mia = pd.read_csv(input_paths["safe_mia"])
    safe_aia = pd.read_csv(input_paths["safe_aia"])
    routed_aia = pd.read_csv(input_paths["routed_aia"])
    _, challenges = load_public_targets(targets_dir)
    aggressive_aia, complement_diagnostics = complement_zero_target_blocks(
        safe_aia, routed_aia
    )
    validation = {
        "safe_mia": validate_mia(safe_mia, safe_mia),
        "safe_aia": validate_aia(safe_aia, challenges),
        "routed_aia": validate_aia(routed_aia, challenges),
        "aggressive_aia": validate_aia(aggressive_aia, challenges),
    }

    official_dist = official_validate.load_dist(targets_dir)
    mia_result = official_validate.validate_mia_submission(safe_mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(
        aggressive_aia, {}, official_dist
    )
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            "official logical validator failed: "
            f"MIA={mia_result.errors}, AIA={aia_result.errors}"
        )
    validation["official_validator"] = "PASS"

    prewrite_hashes = {
        "F_mia.csv": input_hashes["safe_mia"],
        "F_aia.csv": dataframe_csv_sha256(aggressive_aia),
    }
    assert_expected_output_hashes(prewrite_hashes, expected=EXPECTED_OUTPUT_HASHES)

    candidate_dir = args.candidate_dir.resolve()
    written_hashes = write_candidate(
        candidate_dir,
        safe_mia,
        aggressive_aia,
        mia_source=input_paths["safe_mia"],
    )
    if written_hashes != EXPECTED_OUTPUT_HASHES:
        raise ValueError(
            "written aggressive candidate differs from pre-write hash gate: "
            f"expected={EXPECTED_OUTPUT_HASHES}, observed={written_hashes}"
        )

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_aggressive_final",
        "attack_version": 6,
        "candidate_dir": str(candidate_dir),
        "strategy": "exact_10_11_complement_on_measured_zero_targets",
        "official_zero_target_scores": {
            target_id: feedback["aia_per_target"][target_id]
            for target_id in ZERO_TARGET_IDS
        },
        "complement": complement_diagnostics,
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
    print("PASS: aggressive zero-target-complement candidate generated")
    print(f"complemented targets: {list(ZERO_TARGET_IDS)}")
    print(f"candidate: {candidate_dir}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
