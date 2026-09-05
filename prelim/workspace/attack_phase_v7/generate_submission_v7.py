"""Generate the final feedback-stacked version-7 attack candidate.

The candidate keeps the measured per-target AIA winners and v4 MIA baseline
from the version-6 horizon-edge candidate.  It changes only two evidence-backed
pieces:

* On the seven targets where v4, independent, and routed selectors all scored
  zero, predict the censoring edge for every row except the unanimous direct
  selection.  Conditional shadow validation found this complement-of-unanimous
  rule stronger than complementing the routed selector alone.
* On target 20, keep the measured winning membership mask but restore seven
  high-confidence public event-time links using mutual copula neighbours.

The module reads no token, creates no archive, and performs no submission.
"""

from __future__ import annotations

import argparse
import json
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
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    OFFICIAL_COLUMNS,
    copula_event_links,
    filter_linkable_releases,
)
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    EXPECTED_AIA_ROWS_PER_TARGET,
    load_public_targets,
    validate_aia,
    validate_mia,
    verify_targets_manifest,
    write_candidate,
)
from attack_phase_v6.generate_aggressive_final_submission_v6 import (  # noqa: E402
    ZERO_TARGET_IDS,
    _normalized_aia,
)
from attack_phase_v6.generate_final_submission_v6 import (  # noqa: E402
    assert_expected_output_hashes,
    dataframe_csv_sha256,
    load_complete_feedback,
    verify_named_hashes,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


ABSTAIN_TIME = 11.0
EDGE_TIME = 9.5
EVENT_LINK_TARGET = "20"
EVENT_LINK_MARGIN_QUANTILE = 0.50
EXPECTED_ACTIVE_ROWS = {
    "6": 217,
    "7": 253,
    "8": 211,
    "9": 274,
    "17": 223,
    "18": 236,
    "19": 268,
}
EXPECTED_INPUT_HASHES = {
    "base_mia": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "base_aia": "6D4FF74F8FF2278E050A327F8982803F3CEBDFDEF5024F548FB1688FB9F761D6",
    "v4_aia": "651D754683BCE4B6F22FD7311AB1584B01C2F42FBB73B13996303472411DFA3A",
    "independent_aia": "BDEA52C1FD7F77F9CF7054E8C9D64F3307B455521318FD73495B28A86BCF35E1",
    "routed_aia": "CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F",
    "feedback_v4": "A3579D160B8E0ED9AC2DB5BF140C1B7D73E2A2FCE524493985998D95C7CB753A",
    "feedback_independent": "D08BBD21B101134F61F12DAE27EB1A8A9EEF8454B726440C6FA7037646CA5C2C",
    "feedback_routed": "A4D1B16FAD78EA8C0382D24E12CB02F5D5E8C3FA5B39C3045C422D2B562664B4",
    "targets_manifest": "A7653F34B890F7E1095153AB50C75E5CC30EDF4272A8D83BFE7065D5B81746E5",
}
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": EXPECTED_INPUT_HASHES["base_mia"],
    "F_aia.csv": "65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0",
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    feedback_dir = REPO_ROOT / "attack_phase_v6" / "official_feedback"
    parser = argparse.ArgumentParser(description="v7 feedback-stacked final candidate")
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=team_dir / "attack-candidate-v6-final-horizon-edge",
    )
    parser.add_argument("--v4-dir", type=Path, default=team_dir / "attack-candidate-v4")
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
        default=team_dir / "attack-candidate-v7-feedback-stack",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "local-output" / "attack-v7" / "candidate_diagnostics.json",
    )
    parser.add_argument(
        "--skip-output-hash-gate",
        action="store_true",
        help="development only: allow computing the initial deterministic output hash",
    )
    return parser.parse_args()


def _aligned_values(
    frame: pd.DataFrame, target_id: str, challenge_ids: np.ndarray, label: str
) -> np.ndarray:
    normalized = _normalized_aia(frame, label=label)
    block = normalized.loc[
        normalized["target_k"].eq(int(target_id)), ["challenge_row_id", "time_hat"]
    ].copy()
    block.index = block["challenge_row_id"].astype(np.int64)
    if len(block) != len(challenge_ids) or set(block.index) != set(challenge_ids):
        raise ValueError(f"{label} target {target_id} keys differ from challenge")
    values = block.loc[challenge_ids, "time_hat"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"{label} target {target_id} has invalid time_hat")
    return values


def assert_zero_feedback(
    feedback: Mapping[str, Mapping[str, Any]],
    *,
    zero_targets: tuple[str, ...] = ZERO_TARGET_IDS,
) -> None:
    for label, report in feedback.items():
        scores = report.get("aia_per_target")
        if not isinstance(scores, Mapping):
            raise ValueError(f"{label} feedback lacks aia_per_target")
        observed = {target: float(scores[target]) for target in zero_targets}
        if any(not np.isclose(score, 0.0) for score in observed.values()):
            raise ValueError(f"{label} does not have zero feedback on every target: {observed}")


def validate_variable_aia(
    frame: pd.DataFrame, challenges: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    """Validate complete normal-format AIA without assuming 163 active rows.

    The official format requires one prediction for every challenge key; it
    does not require an attacker to predict exactly half of the rows.  Earlier
    project helpers enforced 163 only as a strategy invariant, so v7 uses this
    format-level check plus the official validator.
    """
    normalized = _normalized_aia(frame, label="variable-cardinality AIA")
    values = pd.to_numeric(normalized["time_hat"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("variable-cardinality AIA has invalid time_hat")
    diagnostics: dict[str, Any] = {}
    for target_id, challenge in challenges.items():
        block = normalized.loc[normalized["target_k"].eq(int(target_id))]
        expected_ids = set(challenge["challenge_row_id"].astype(int).tolist())
        observed_ids = set(block["challenge_row_id"].astype(int).tolist())
        if len(block) != len(challenge) or observed_ids != expected_ids:
            raise ValueError(f"target {target_id} AIA keys differ from challenge")
        block_values = block["time_hat"].to_numpy(dtype=float)
        diagnostics[target_id] = {
            "rows": len(block),
            "active_rows": int(np.count_nonzero(~np.isclose(block_values, ABSTAIN_TIME))),
            "unique_time_values": int(pd.Series(block_values).nunique()),
        }
    return {"rows": len(normalized), "targets": diagnostics}


def stack_aia_feedback(
    base: pd.DataFrame,
    v4: pd.DataFrame,
    independent: pd.DataFrame,
    routed: pd.DataFrame,
    releases: Mapping[str, pd.DataFrame],
    challenges: Mapping[str, pd.DataFrame],
    *,
    zero_targets: tuple[str, ...] = ZERO_TARGET_IDS,
    event_target: str = EVENT_LINK_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the complement-of-unanimous mask and one locked time decoder."""
    output = _normalized_aia(base, label="horizon-edge base")
    base_normalized = output.copy()
    target_ids = tuple(sorted((str(int(value)) for value in EXPECTED_TARGET_IDS), key=int))
    if set(releases) != set(target_ids) or set(challenges) != set(target_ids):
        raise ValueError("release/challenge target set differs from expected targets")
    observed_targets = tuple(sorted(output["target_k"].astype(str).unique(), key=int))
    if observed_targets != target_ids:
        raise ValueError(f"base target set differs: {observed_targets}")

    zero_set = set(zero_targets)
    per_target: dict[str, Any] = {}
    for target_id in target_ids:
        target_mask = output["target_k"].eq(int(target_id))
        if int(target_mask.sum()) != EXPECTED_AIA_ROWS_PER_TARGET:
            raise ValueError(f"target {target_id} row count differs")
        challenge = challenges[target_id].sort_values(
            "challenge_row_id", kind="mergesort"
        ).reset_index(drop=True)
        ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        output_block = output.loc[target_mask].sort_values(
            "challenge_row_id", kind="mergesort"
        )
        if not np.array_equal(
            output_block["challenge_row_id"].to_numpy(dtype=np.int64), ids
        ):
            raise ValueError(f"base target {target_id} challenge order differs")
        if target_id not in zero_set:
            continue

        source_values = {
            "v4": _aligned_values(v4, target_id, ids, "v4"),
            "independent": _aligned_values(
                independent, target_id, ids, "independent"
            ),
            "routed": _aligned_values(routed, target_id, ids, "routed"),
        }
        active = {
            label: ~np.isclose(values, ABSTAIN_TIME)
            for label, values in source_values.items()
        }
        for label, mask in active.items():
            if int(mask.sum()) != EXPECTED_AIA_ROWS_PER_TARGET // 2:
                raise ValueError(
                    f"{label} target {target_id} does not select exactly half"
                )
        unanimous = active["v4"] & active["independent"] & active["routed"]
        candidate_active = ~unanimous
        expected_active = EXPECTED_ACTIVE_ROWS.get(target_id)
        if expected_active is not None and int(candidate_active.sum()) != expected_active:
            raise ValueError(
                f"target {target_id} active count changed: "
                f"expected={expected_active}, observed={int(candidate_active.sum())}"
            )
        candidate_values = np.full(len(ids), ABSTAIN_TIME, dtype=float)
        candidate_values[candidate_active] = EDGE_TIME
        output.loc[output_block.index, "time_hat"] = candidate_values
        base_values = output_block["time_hat"].to_numpy(dtype=float)
        per_target[target_id] = {
            "source_active_rows": {
                label: int(mask.sum()) for label, mask in active.items()
            },
            "unanimous_direct_rows": int(unanimous.sum()),
            "candidate_active_rows": int(candidate_active.sum()),
            "added_vs_routed_complement": int(
                np.count_nonzero(candidate_active & active["routed"])
            ),
            "changed_rows_from_horizon_edge": int(
                np.count_nonzero(~np.isclose(candidate_values, base_values))
            ),
        }

    # The event decoder is fitted jointly across releases so the same public
    # cross-target duplicate filter used in the tested implementation applies.
    linkable, filter_diagnostics = filter_linkable_releases(dict(releases))
    event_mask = output["target_k"].eq(int(event_target))
    event_block = output.loc[event_mask].sort_values(
        "challenge_row_id", kind="mergesort"
    )
    event_challenge = challenges[event_target].sort_values(
        "challenge_row_id", kind="mergesort"
    ).reset_index(drop=True)
    event_ids = event_challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        event_block["challenge_row_id"].to_numpy(dtype=np.int64), event_ids
    ):
        raise ValueError("event target challenge order differs")
    linked_time, linked_gate, link_details = copula_event_links(
        linkable[event_target],
        event_challenge,
        margin_quantile=EVENT_LINK_MARGIN_QUANTILE,
        horizon_time=10.0,
    )
    current = event_block["time_hat"].to_numpy(dtype=float)
    active_current = ~np.isclose(current, ABSTAIN_TIME)
    use_link = active_current & linked_gate
    decoded = current.copy()
    decoded[use_link] = linked_time[use_link]
    changed_link = ~np.isclose(decoded, current)
    output.loc[event_block.index, "time_hat"] = decoded
    per_target[event_target] = {
        "candidate_active_rows": int(active_current.sum()),
        "trusted_event_links_all_rows": int(linked_gate.sum()),
        "trusted_event_links_in_active_mask": int(use_link.sum()),
        "changed_event_times": int(changed_link.sum()),
        "changed_challenge_row_ids": event_ids[changed_link].astype(int).tolist(),
        "link_details": link_details,
    }

    if not output[["target_k", "challenge_row_id"]].equals(
        base_normalized[["target_k", "challenge_row_id"]]
    ):
        raise RuntimeError("v7 stacking changed AIA keys")
    changed_targets = sorted(
        output.loc[
            ~np.isclose(
                output["time_hat"].to_numpy(dtype=float),
                base_normalized["time_hat"].to_numpy(dtype=float),
            ),
            "target_k",
        ]
        .astype(str)
        .unique()
        .tolist(),
        key=int,
    )
    expected_changed = sorted([*zero_targets, event_target], key=int)
    if changed_targets != expected_changed:
        raise RuntimeError(
            f"unexpected changed targets: expected={expected_changed}, got={changed_targets}"
        )
    return output.reset_index(drop=True), {
        "method": "complement_of_three_way_unanimous_plus_target20_copula_time",
        "zero_targets": list(zero_targets),
        "event_link_target": event_target,
        "event_link_margin_quantile": EVENT_LINK_MARGIN_QUANTILE,
        "edge_time": EDGE_TIME,
        "abstain_time": ABSTAIN_TIME,
        "changed_targets": changed_targets,
        "per_target": per_target,
        "release_filter": filter_diagnostics,
    }


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    paths = {
        "base_mia": args.base_dir.resolve() / "F_mia.csv",
        "base_aia": args.base_dir.resolve() / "F_aia.csv",
        "v4_aia": args.v4_dir.resolve() / "F_aia.csv",
        "independent_aia": args.independent_dir.resolve() / "F_aia.csv",
        "routed_aia": args.routed_dir.resolve() / "F_aia.csv",
        "feedback_v4": args.feedback_v4.resolve(),
        "feedback_independent": args.feedback_independent.resolve(),
        "feedback_routed": args.feedback_routed.resolve(),
        "targets_manifest": targets_dir / "SHA256SUMS.txt",
    }
    input_hashes = verify_named_hashes(paths, expected=EXPECTED_INPUT_HASHES)
    target_file_hashes = verify_targets_manifest(targets_dir)
    feedback = {
        "v4": load_complete_feedback(
            paths["feedback_v4"], expected_candidate="v4", require_score_1=False
        ),
        "independent": load_complete_feedback(
            paths["feedback_independent"],
            expected_candidate="probe_independent",
            require_score_1=True,
        ),
        "routed": load_complete_feedback(
            paths["feedback_routed"],
            expected_candidate="routed_guided_probe",
            require_score_1=True,
            require_all_displayed_scores=True,
        ),
    }
    assert_zero_feedback(feedback)

    mia = pd.read_csv(paths["base_mia"])
    base_aia = pd.read_csv(paths["base_aia"])
    releases, challenges = load_public_targets(targets_dir)
    aia, stacking = stack_aia_feedback(
        base_aia,
        pd.read_csv(paths["v4_aia"]),
        pd.read_csv(paths["independent_aia"]),
        pd.read_csv(paths["routed_aia"]),
        releases,
        challenges,
    )
    validation = {
        "mia": validate_mia(mia, mia),
        "base_aia": validate_aia(base_aia, challenges),
        "v7_aia": validate_variable_aia(aia, challenges),
    }
    official_dist = official_validate.load_dist(targets_dir)
    mia_result = official_validate.validate_mia_submission(mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(aia, {}, official_dist)
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            "official logical validator failed: "
            f"MIA={mia_result.errors}, AIA={aia_result.errors}"
        )
    validation["official_validator"] = "PASS"

    output_hashes = {
        # write_candidate byte-copies the source MIA file to preserve its
        # already validated decimal serialization.
        "F_mia.csv": input_hashes["base_mia"],
        "F_aia.csv": dataframe_csv_sha256(aia),
    }
    if not args.skip_output_hash_gate:
        if not EXPECTED_OUTPUT_HASHES:
            raise ValueError(
                "EXPECTED_OUTPUT_HASHES is not locked; run once with "
                "--skip-output-hash-gate, inspect, then freeze the hashes"
            )
        assert_expected_output_hashes(output_hashes, expected=EXPECTED_OUTPUT_HASHES)
    candidate_dir = args.candidate_dir.resolve()
    written_hashes = write_candidate(
        candidate_dir, mia, aia, mia_source=paths["base_mia"]
    )
    if written_hashes != output_hashes:
        raise ValueError(
            f"written candidate hashes differ: expected={output_hashes}, got={written_hashes}"
        )

    report = {
        "report_type": "authorized_local_privacy_competition_feedback_stacked_final",
        "attack_version": 7,
        "candidate_dir": str(candidate_dir),
        "strategy": stacking,
        "validation": validation,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "target_file_hashes": target_file_hashes,
        "network_access": False,
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: v7 candidate generated")
    print(f"candidate: {candidate_dir}")
    print(f"hashes: {output_hashes}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
