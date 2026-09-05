"""Generate the final normal-format candidate with a robust horizon shift.

The membership selections, decoded event times, and abstentions from the
official-feedback/complement candidate are kept fixed.  Only exact ``10.0``
predictions are moved to ``9.5`` for targets other than the two already at the
official ceiling.  With truth times constrained to ``0 < time <= 10``, the
success window at tolerance 0.5 expands from ``[9.5, 10]`` to ``[9.0, 10]``.

This module never reads the team token, creates an archive, or submits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v4.generate_submission_v4 import EXPECTED_TARGET_IDS  # noqa: E402
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    EXPECTED_AIA_ROWS_PER_TARGET,
    load_public_targets,
    validate_aia,
    validate_mia,
    verify_targets_manifest,
    write_candidate,
)
from attack_phase_v6.generate_aggressive_final_submission_v6 import (  # noqa: E402
    _normalized_aia,
)
from attack_phase_v6.generate_final_submission_v6 import (  # noqa: E402
    assert_expected_output_hashes,
    dataframe_csv_sha256,
    verify_named_hashes,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


SOURCE_TIME = 10.0
EDGE_TIME = 9.5
TAU_TIME = 0.5
HORIZON_TIME = 10.0
UNCHANGED_TARGET_IDS = ("1", "2")
SHIFT_TARGET_IDS = tuple(
    target_id
    for target_id in EXPECTED_TARGET_IDS
    if target_id not in UNCHANGED_TARGET_IDS
)
EXPECTED_INPUT_HASHES = {
    "aggressive_mia": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "aggressive_aia": "2CFDC03D9EB2A2F096BE3A872BC836CA0AAE16FF170F20F0124F85F14E7A3D82",
    "targets_manifest": "A7653F34B890F7E1095153AB50C75E5CC30EDF4272A8D83BFE7065D5B81746E5",
}
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": EXPECTED_INPUT_HASHES["aggressive_mia"],
    "F_aia.csv": "6D4FF74F8FF2278E050A327F8982803F3CEBDFDEF5024F548FB1688FB9F761D6",
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    parser = argparse.ArgumentParser(
        description="最終補集合候補の10年予測だけを9.5年へ移した候補を生成します。"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-aggressive-complements",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-horizon-edge",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-v6" / "final_horizon_edge.json",
    )
    return parser.parse_args()


def shift_exact_horizon_predictions(
    frame: pd.DataFrame,
    *,
    target_ids: Iterable[str] = EXPECTED_TARGET_IDS,
    shift_target_ids: Iterable[str] = SHIFT_TARGET_IDS,
    rows_per_target: int = EXPECTED_AIA_ROWS_PER_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shift exact 10.0 predictions while preserving keys and all other cells."""
    output = _normalized_aia(frame, label="horizon source")
    target_ids = tuple(str(int(value)) for value in target_ids)
    shift_target_ids = tuple(str(int(value)) for value in shift_target_ids)
    if not set(shift_target_ids) <= set(target_ids):
        raise ValueError("shift targets must be a subset of expected targets")
    if len(set(shift_target_ids)) != len(shift_target_ids):
        raise ValueError("shift targets contain duplicates")

    observed_targets = tuple(
        sorted(output["target_k"].astype(str).unique().tolist(), key=int)
    )
    if observed_targets != tuple(sorted(target_ids, key=int)):
        raise ValueError(
            f"AIA target set mismatch: expected={target_ids}, observed={observed_targets}"
        )
    if EDGE_TIME + TAU_TIME < HORIZON_TIME:
        raise ValueError("edge prediction does not cover the full allowed horizon")
    if EDGE_TIME - TAU_TIME > SOURCE_TIME - TAU_TIME:
        raise ValueError("edge prediction does not expand the source success window")

    source = output.copy()
    per_target: dict[str, dict[str, int | float]] = {}
    shift_set = set(shift_target_ids)
    for target_id in target_ids:
        target_mask = output["target_k"].eq(int(target_id))
        rows = int(target_mask.sum())
        if rows != rows_per_target:
            raise ValueError(
                f"target {target_id} has {rows} rows; expected {rows_per_target}"
            )
        shift_mask = target_mask & output["time_hat"].eq(SOURCE_TIME)
        shifted_rows = int(shift_mask.sum()) if target_id in shift_set else 0
        if target_id in shift_set:
            output.loc[shift_mask, "time_hat"] = EDGE_TIME
            if shifted_rows == 0:
                raise ValueError(f"target {target_id} has no exact horizon predictions")
        per_target[target_id] = {
            "rows": rows,
            "shifted_rows": shifted_rows,
            "source_time": SOURCE_TIME,
            "edge_time": EDGE_TIME,
        }

    key_columns = ["target_k", "challenge_row_id"]
    if not output[key_columns].equals(source[key_columns]):
        raise ValueError("horizon shift changed submission keys")
    for target_id in set(target_ids) - shift_set:
        mask = output["target_k"].eq(int(target_id))
        pd.testing.assert_frame_equal(
            output.loc[mask].reset_index(drop=True),
            source.loc[mask].reset_index(drop=True),
        )
    changed = source["time_hat"].ne(output["time_hat"])
    if not np.all(source.loc[changed, "time_hat"].to_numpy() == SOURCE_TIME):
        raise ValueError("a non-horizon prediction changed")
    if not np.all(output.loc[changed, "time_hat"].to_numpy() == EDGE_TIME):
        raise ValueError("a shifted prediction has the wrong value")

    return output.reset_index(drop=True), {
        "shift_targets": list(shift_target_ids),
        "unchanged_targets": [
            target_id for target_id in target_ids if target_id not in shift_set
        ],
        "source_time": SOURCE_TIME,
        "edge_time": EDGE_TIME,
        "tau_time": TAU_TIME,
        "truth_domain": "0 < time <= 10",
        "source_success_interval_on_domain": [9.5, 10.0],
        "edge_success_interval_on_domain": [9.0, 10.0],
        "total_shifted_rows": int(changed.sum()),
        "per_target": per_target,
    }


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    source_dir = args.source_dir.resolve()
    input_paths = {
        "aggressive_mia": source_dir / "F_mia.csv",
        "aggressive_aia": source_dir / "F_aia.csv",
        "targets_manifest": targets_dir / "SHA256SUMS.txt",
    }
    input_hashes = verify_named_hashes(input_paths, expected=EXPECTED_INPUT_HASHES)
    target_file_hashes = verify_targets_manifest(targets_dir)

    mia = pd.read_csv(input_paths["aggressive_mia"])
    source_aia = pd.read_csv(input_paths["aggressive_aia"])
    shifted_aia, shift_diagnostics = shift_exact_horizon_predictions(source_aia)
    _, challenges = load_public_targets(targets_dir)
    validation = {
        "mia": validate_mia(mia, mia),
        "source_aia": validate_aia(source_aia, challenges),
        "shifted_aia": validate_aia(shifted_aia, challenges),
    }

    official_dist = official_validate.load_dist(targets_dir)
    mia_result = official_validate.validate_mia_submission(mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(
        shifted_aia, {}, official_dist
    )
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            "official logical validator failed: "
            f"MIA={mia_result.errors}, AIA={aia_result.errors}"
        )
    validation["official_validator"] = "PASS"

    prewrite_hashes = {
        "F_mia.csv": input_hashes["aggressive_mia"],
        "F_aia.csv": dataframe_csv_sha256(shifted_aia),
    }
    assert_expected_output_hashes(prewrite_hashes, expected=EXPECTED_OUTPUT_HASHES)
    candidate_dir = args.candidate_dir.resolve()
    written_hashes = write_candidate(
        candidate_dir,
        mia,
        shifted_aia,
        mia_source=input_paths["aggressive_mia"],
    )
    if written_hashes != EXPECTED_OUTPUT_HASHES:
        raise ValueError(
            "written horizon-edge candidate differs from hash gate: "
            f"expected={EXPECTED_OUTPUT_HASHES}, observed={written_hashes}"
        )

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_horizon_edge_final",
        "attack_version": 6,
        "candidate_dir": str(candidate_dir),
        "strategy": "fixed_selection_exact_10_to_9p5",
        "shift": shift_diagnostics,
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
    print("PASS: horizon-edge final candidate generated")
    print(f"shifted rows: {shift_diagnostics['total_shifted_rows']}")
    print(f"candidate: {candidate_dir}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
