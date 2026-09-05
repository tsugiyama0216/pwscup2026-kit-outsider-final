"""Generate feedback-safe and routed version-6 attack candidates.

This module uses only organizer-distributed public targets, saved attack
outputs, and the per-target feedback returned for the team's own submissions.
It never reads the team token, creates an archive, or performs a submission.
"""

from __future__ import annotations

import argparse
import hashlib
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
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    OFFICIAL_COLUMNS,
    build_aia_v5_variants,
)
from attack_phase_v5.blend_feedback_winners import (  # noqa: E402
    blend_target_winners,
    load_aia_per_target,
)
from attack_phase_v5.generate_submission_v5 import (  # noqa: E402
    EXPECTED_AIA_ROWS_PER_TARGET,
    EXPECTED_INPUT_HASHES,
    load_public_targets,
    sha256_file,
    validate_aia,
    validate_mia,
    verify_input_hashes,
    verify_targets_manifest,
    write_candidate,
)
from attack_phase_v5.mia_tail_assignment import assign_unique_top_tail  # noqa: E402
from attack_phase_v6.aia_global_density import (  # noqa: E402
    build_global_density_horizon,
    build_sex_consensus_horizon,
)
from attack_phase_v6.mia_guided_tail import build_guided_tail  # noqa: E402
from pwscup2026.kit import validate as official_validate  # noqa: E402


COPULA_HORIZON_VARIANT = "consensus_v4_copula__horizon"
EQ_MIX_IND_EVENT_VARIANT = (
    "consensus_equal_copula_mix__independent_event_margin"
)
EQ_COP_V4_EVENT_VARIANT = "consensus_equal_copula__v4_event_margin"
EQ_COP_IND_EVENT_VARIANT = "consensus_equal_copula__independent_event_margin"
V4_MIX_V4_EVENT_VARIANT = "consensus_v4_copula__v4_event_margin"
SOFT_COP_IND_EVENT_VARIANT = "consensus_soft_copula__independent_event_margin"
GLOBAL_DENSITY_ALL5_VARIANT = "global_density_all5__horizon"
GLOBAL_DENSITY_STABLE3_VARIANT = "global_density_stable3__horizon"
INVERTED_DENSITY_STABLE3_VARIANT = "inverted_density_stable3__horizon"
SEX_CONSENSUS_HORIZON_VARIANT = "sex_consensus_nohdl__horizon"
EXPECTED_BACKGROUND_HASH = (
    "304AF8F2EE60BAA37C7968772BFAF604033C9A6336DFA4BE25BA6D7ADA6BCC3D"
)
EXPECTED_DAYBREAK_MIA_HASH = (
    "760D4FE12B4296DE4712D2AF133E6165F93446E9A9794CDA56BADF6868F68294"
)
EXPECTED_GUIDED_MIA_HASH = (
    "70E8E689E476E9EEF54F5821846EC1B93DE04FD6A63B680849D34F720DB78AFA"
)
EXPECTED_ROUTED_AIA_HASH = (
    "CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F"
)
POSITIVE_PROBE_ROUTE: dict[str, str] = {
    "3": EQ_MIX_IND_EVENT_VARIANT,
    "4": EQ_COP_V4_EVENT_VARIANT,
    "5": EQ_COP_V4_EVENT_VARIANT,
    "10": V4_MIX_V4_EVENT_VARIANT,
    "12": EQ_COP_V4_EVENT_VARIANT,
    "13": EQ_COP_IND_EVENT_VARIANT,
    "22": EQ_COP_IND_EVENT_VARIANT,
    "23": EQ_COP_IND_EVENT_VARIANT,
    "26": SOFT_COP_IND_EVENT_VARIANT,
    # The filtered production block diverged from the shadow-tested t29 block.
    # Preserve the measured v4 winner until an exact matching recipe is tested.
    "29": "v4_exact",
}

ZERO_PROBE_ROUTE: dict[str, str] = {
    "6": COPULA_HORIZON_VARIANT,
    "7": GLOBAL_DENSITY_ALL5_VARIANT,
    "8": COPULA_HORIZON_VARIANT,
    "9": GLOBAL_DENSITY_STABLE3_VARIANT,
    "17": COPULA_HORIZON_VARIANT,
    "18": SEX_CONSENSUS_HORIZON_VARIANT,
    "19": INVERTED_DENSITY_STABLE3_VARIANT,
    "20": GLOBAL_DENSITY_ALL5_VARIANT,
}

# Targets 1 and 2 are already at the per-target ceiling of 1.0, so changing
# either block can only be neutral or harmful.  The positive-score route uses
# four fixed selector families plus one conservative
# event gate: only mutual rank-copula links with an above-median nearest-neighbour
# margin may replace an expert time.  Targets where both official experts scored
# zero use either the existing copula selector, a sex-only consensus selector,
# or a global laboratory-density probe.  Density is limited to targets whose
# own release beats cross-target release controls, so a generic distribution
# artifact cannot by itself qualify a target for that route.
DEFAULT_PROBE_ROUTE: dict[str, str] = {
    target_id: (
        "v4_exact"
        if target_id in {"1", "2"}
        else POSITIVE_PROBE_ROUTE[target_id]
        if target_id in POSITIVE_PROBE_ROUTE
        else ZERO_PROBE_ROUTE[target_id]
    )
    for target_id in EXPECTED_TARGET_IDS
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    feedback_dir = REPO_ROOT / "attack_phase_v6" / "official_feedback"
    parser = argparse.ArgumentParser(
        description="公式target別feedbackと公開構造からPWSCUP2026攻撃v6候補を生成します。"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--background",
        type=Path,
        default=REPO_ROOT / "participant_data" / "A_bg.csv",
    )
    parser.add_argument(
        "--v3-dir", type=Path, default=team_dir / "attack-candidate-v3"
    )
    parser.add_argument(
        "--v4-dir", type=Path, default=team_dir / "attack-candidate-v4"
    )
    parser.add_argument(
        "--independent-dir",
        type=Path,
        default=team_dir / "attack-candidate-daybreak-independent",
    )
    parser.add_argument("--feedback-v4", type=Path, default=feedback_dir / "v4.json")
    parser.add_argument(
        "--feedback-independent",
        type=Path,
        default=feedback_dir / "probe_independent.json",
    )
    parser.add_argument(
        "--feedback-daybreak",
        type=Path,
        default=feedback_dir / "daybreak.json",
    )
    parser.add_argument("--candidate-root", type=Path, default=team_dir)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "local-output" / "attack-v6"
    )
    parser.add_argument(
        "--tail-width",
        type=int,
        choices=(15,),
        default=15,
        help="guided production MIA is hash-locked to the low-FPR top-15 tail",
    )
    return parser.parse_args()


def route_target_blocks(
    sources: Mapping[str, pd.DataFrame],
    route: Mapping[str, str],
    *,
    target_ids: Iterable[str] = EXPECTED_TARGET_IDS,
    rows_per_target: int = EXPECTED_AIA_ROWS_PER_TARGET,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copy one complete target block from the source selected by ``route``."""
    if not sources:
        raise ValueError("AIA route sources are empty")
    target_ids = tuple(str(int(value)) for value in target_ids)
    if set(route) != set(target_ids):
        missing = sorted(set(target_ids) - set(route), key=int)
        extra = sorted(set(route) - set(target_ids), key=int)
        raise ValueError(f"AIA route target mismatch: missing={missing}, extra={extra}")
    unknown = sorted(set(route.values()) - set(sources))
    if unknown:
        raise ValueError(f"AIA route contains unknown sources: {unknown}")

    normalized: dict[str, pd.DataFrame] = {}
    expected_keys: dict[str, set[int]] | None = None
    for label, frame in sources.items():
        if list(frame.columns) != list(OFFICIAL_COLUMNS):
            raise ValueError(f"{label} has invalid AIA columns")
        if frame.duplicated(["target_k", "challenge_row_id"]).any():
            raise ValueError(f"{label} has duplicate AIA keys")
        work = frame.copy()
        work["target_k"] = work["target_k"].astype(int)
        keys: dict[str, set[int]] = {}
        for target_id in target_ids:
            block = work.loc[work["target_k"].eq(int(target_id))]
            if len(block) != rows_per_target:
                raise ValueError(
                    f"{label} target {target_id} has {len(block)} rows; "
                    f"expected {rows_per_target}"
                )
            keys[target_id] = set(block["challenge_row_id"].astype(int))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"{label} challenge keys differ from the first source")
        normalized[label] = work

    parts: list[pd.DataFrame] = []
    source_counts = {label: 0 for label in sources}
    for target_id in target_ids:
        label = route[target_id]
        block = normalized[label].loc[
            normalized[label]["target_k"].eq(int(target_id)), OFFICIAL_COLUMNS
        ]
        parts.append(block.copy())
        source_counts[label] += 1
    output = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["target_k", "challenge_row_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return output, {"route": dict(route), "source_counts": source_counts}


def feedback_winner_power(score_maps: Mapping[str, Mapping[str, float]]) -> float:
    """Return the exact AIA power of a whole-target feedback winner blend."""
    if not score_maps:
        raise ValueError("feedback score maps are empty")
    missing = {
        label: sorted(set(EXPECTED_TARGET_IDS) - set(scores), key=int)
        for label, scores in score_maps.items()
        if set(EXPECTED_TARGET_IDS) - set(scores)
    }
    if missing:
        raise ValueError(f"feedback score maps are incomplete: {missing}")
    maxima = [
        max(float(scores[target_id]) for scores in score_maps.values())
        for target_id in EXPECTED_TARGET_IDS
    ]
    if not np.isfinite(maxima).all():
        raise ValueError("feedback scores must be finite")
    return float(np.mean(maxima))


def dataframe_csv_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact UTF-8 CSV bytes that pandas will write for ``frame``."""
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _count_changed_rows(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    left_indexed = left.set_index(["target_k", "challenge_row_id"])["time_hat"]
    right_indexed = right.set_index(["target_k", "challenge_row_id"])["time_hat"]
    if not left_indexed.index.equals(right_indexed.index):
        right_indexed = right_indexed.reindex(left_indexed.index)
    changed = ~np.isclose(
        left_indexed.to_numpy(dtype=float), right_indexed.to_numpy(dtype=float)
    )
    targets = left_indexed.index.get_level_values("target_k").to_numpy(dtype=int)
    return {
        target_id: int(np.count_nonzero(changed & (targets == int(target_id))))
        for target_id in EXPECTED_TARGET_IDS
    }


def _official_validate_candidate(
    mia: pd.DataFrame,
    aia: pd.DataFrame,
    *,
    official_dist: Any,
    label: str,
) -> None:
    mia_result = official_validate.validate_mia_submission(mia, {}, official_dist)
    aia_result = official_validate.validate_aia_submission(aia, {}, official_dist)
    if not mia_result.ok or not aia_result.ok:
        raise ValueError(
            f"official logical validator failed for {label}: "
            f"MIA={mia_result.errors}, AIA={aia_result.errors}"
        )


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    candidate_root = args.candidate_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v3_mia_path = args.v3_dir.resolve() / "F_mia.csv"
    v4_mia_path = args.v4_dir.resolve() / "F_mia.csv"
    v4_aia_path = args.v4_dir.resolve() / "F_aia.csv"
    independent_mia_path = args.independent_dir.resolve() / "F_mia.csv"
    independent_aia_path = args.independent_dir.resolve() / "F_aia.csv"
    background_path = args.background.resolve()
    input_hashes = verify_input_hashes(
        {
            "v3_mia": v3_mia_path,
            "v4_mia": v4_mia_path,
            "v4_aia": v4_aia_path,
            "independent_aia": independent_aia_path,
            "targets_manifest": targets_dir / "SHA256SUMS.txt",
        }
    )
    background_hash = sha256_file(background_path)
    if background_hash != EXPECTED_BACKGROUND_HASH:
        raise ValueError(
            "v6 background hash mismatch: "
            f"expected={EXPECTED_BACKGROUND_HASH}, observed={background_hash}"
        )
    input_hashes["background"] = background_hash
    independent_mia_hash = sha256_file(independent_mia_path)
    if independent_mia_hash != EXPECTED_DAYBREAK_MIA_HASH:
        raise ValueError(
            "v6 Daybreak MIA hash mismatch: "
            f"expected={EXPECTED_DAYBREAK_MIA_HASH}, "
            f"observed={independent_mia_hash}"
        )
    input_hashes["independent_mia"] = independent_mia_hash
    target_file_hashes = verify_targets_manifest(targets_dir)

    v3_mia = pd.read_csv(v3_mia_path)
    v4_mia = pd.read_csv(v4_mia_path)
    independent_mia = pd.read_csv(independent_mia_path)
    v4_aia = pd.read_csv(v4_aia_path)
    independent_aia = pd.read_csv(independent_aia_path)
    releases, challenges = load_public_targets(targets_dir)
    background = pd.read_csv(background_path)

    structured, structured_diagnostics = build_aia_v5_variants(
        v4_aia, independent_aia, releases, challenges
    )
    density_all5, density_all5_diagnostics = build_global_density_horizon(
        releases, challenges, background
    )
    density_stable3, density_stable3_diagnostics = build_global_density_horizon(
        releases,
        challenges,
        background,
        columns=("age", "SBP", "TG"),
    )
    inverted_density_stable3, inverted_density_stable3_diagnostics = (
        build_global_density_horizon(
            releases,
            challenges,
            background,
            columns=("age", "SBP", "TG"),
            invert=True,
        )
    )
    inversion_evidence = inverted_density_stable3_diagnostics["targets"]["19"]
    if inversion_evidence[
        "linkable_release_background_exact_fraction"
    ] < 0.90:
        raise ValueError(
            "target 19 no longer satisfies the background-copy inversion gate"
        )
    sex_consensus, sex_consensus_diagnostics = build_sex_consensus_horizon(
        v4_aia, independent_aia, releases, challenges
    )
    routed_variants = sorted(set(DEFAULT_PROBE_ROUTE.values()) - {"v4_exact"})
    external_variants = {
        GLOBAL_DENSITY_ALL5_VARIANT: density_all5,
        GLOBAL_DENSITY_STABLE3_VARIANT: density_stable3,
        INVERTED_DENSITY_STABLE3_VARIANT: inverted_density_stable3,
        SEX_CONSENSUS_HORIZON_VARIANT: sex_consensus,
    }
    route_sources = {"v4_exact": v4_aia}
    route_sources.update(
        {
            name: external_variants[name] if name in external_variants else structured[name]
            for name in routed_variants
        }
    )
    routed_aia, route_diagnostics = route_target_blocks(
        route_sources, DEFAULT_PROBE_ROUTE
    )

    v4_scores = load_aia_per_target(args.feedback_v4.resolve())
    independent_scores = load_aia_per_target(args.feedback_independent.resolve())
    feedback_aia, feedback_diagnostics = blend_target_winners(
        {"v4": v4_aia, "independent": independent_aia},
        {"v4": v4_scores, "independent": independent_scores},
    )
    feedback_power = feedback_winner_power(
        {"v4": v4_scores, "independent": independent_scores}
    )
    repaired_mia, mia_diagnostics = assign_unique_top_tail(
        v3_mia, v4_mia, top_l=args.tail_width
    )
    guided_mia, guided_mia_diagnostics = build_guided_tail(
        v3_mia,
        v4_mia,
        independent_mia,
        top_l=args.tail_width,
    )

    validation = {
        "v4_mia": validate_mia(v4_mia, v4_mia),
        "tail_mia": validate_mia(repaired_mia, v4_mia),
        "guided_mia": validate_mia(guided_mia, v4_mia),
        "routed_aia": validate_aia(routed_aia, challenges),
        "feedback_aia": validate_aia(feedback_aia, challenges),
    }
    official_dist = official_validate.load_dist(targets_dir)
    _official_validate_candidate(
        v4_mia, routed_aia, official_dist=official_dist, label="routed_probe"
    )
    _official_validate_candidate(
        guided_mia,
        routed_aia,
        official_dist=official_dist,
        label="routed_guided_probe",
    )
    _official_validate_candidate(
        v4_mia, feedback_aia, official_dist=official_dist, label="feedback_v4_mia"
    )
    _official_validate_candidate(
        repaired_mia,
        feedback_aia,
        official_dist=official_dist,
        label=f"feedback_tail{args.tail_width}",
    )

    # Fail before touching any candidate directory if a production recipe or
    # serialization changes.  This prevents a failed generation attempt from
    # replacing a previously validated candidate with drifted CSVs.
    observed_guided_hash = dataframe_csv_sha256(guided_mia)
    if observed_guided_hash != EXPECTED_GUIDED_MIA_HASH:
        raise ValueError(
            "v6 guided MIA hash mismatch before write: "
            f"expected={EXPECTED_GUIDED_MIA_HASH}, observed={observed_guided_hash}"
        )
    observed_routed_hash = dataframe_csv_sha256(routed_aia)
    if observed_routed_hash != EXPECTED_ROUTED_AIA_HASH:
        raise ValueError(
            "v6 routed AIA hash mismatch before write: "
            f"expected={EXPECTED_ROUTED_AIA_HASH}, observed={observed_routed_hash}"
        )

    candidates = {
        "attack-candidate-v6-routed-guided-probe": write_candidate(
            candidate_root / "attack-candidate-v6-routed-guided-probe",
            guided_mia,
            routed_aia,
        ),
        "attack-candidate-v6-routed-probe": write_candidate(
            candidate_root / "attack-candidate-v6-routed-probe",
            v4_mia,
            routed_aia,
            mia_source=v4_mia_path,
        ),
        "attack-candidate-v6-feedback-v4-probe": write_candidate(
            candidate_root / "attack-candidate-v6-feedback-v4-probe",
            v4_mia,
            feedback_aia,
            mia_source=v4_mia_path,
        ),
        f"attack-candidate-v6-feedback-tail{args.tail_width}": write_candidate(
            candidate_root / f"attack-candidate-v6-feedback-tail{args.tail_width}",
            repaired_mia,
            feedback_aia,
        ),
    }
    written_guided = candidates["attack-candidate-v6-routed-guided-probe"]
    if written_guided["F_mia.csv"] != EXPECTED_GUIDED_MIA_HASH:
        raise ValueError("written v6 guided MIA differs from the pre-write bytes")
    if written_guided["F_aia.csv"] != EXPECTED_ROUTED_AIA_HASH:
        raise ValueError("written v6 routed AIA differs from the pre-write bytes")

    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    routed_aia.to_csv(variants_dir / "F_aia_routed_probe.csv", index=False)
    feedback_aia.to_csv(variants_dir / "F_aia_feedback_v4_probe.csv", index=False)
    repaired_mia.to_csv(
        variants_dir / f"F_mia_unique_tail_{args.tail_width}.csv", index=False
    )
    guided_mia.to_csv(
        variants_dir / f"F_mia_guided_tail_{args.tail_width}.csv", index=False
    )

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_v6",
        "attack_version": 6,
        "official_feedback": {
            "v4": {
                "path": str(args.feedback_v4.resolve()),
                "sha256": sha256_file(args.feedback_v4.resolve()),
            },
            "independent_probe": {
                "path": str(args.feedback_independent.resolve()),
                "sha256": sha256_file(args.feedback_independent.resolve()),
            },
            "daybreak": {
                "path": str(args.feedback_daybreak.resolve()),
                "sha256": sha256_file(args.feedback_daybreak.resolve()),
            },
            "winner_aia_power": feedback_power,
            "winner_blend": feedback_diagnostics,
        },
        "routed_probe": {
            **route_diagnostics,
            "changed_rows_from_v4": _count_changed_rows(v4_aia, routed_aia),
        },
        "mia": {**mia_diagnostics, "guided": guided_mia_diagnostics},
        "structured": structured_diagnostics,
        "global_density": {
            "all5": density_all5_diagnostics,
            "stable3": density_stable3_diagnostics,
            "inverted_stable3": inverted_density_stable3_diagnostics,
        },
        "sex_consensus": sex_consensus_diagnostics,
        "validation": validation,
        "candidate_hashes": candidates,
        "input_hashes": input_hashes,
        "expected_input_hashes": {
            **EXPECTED_INPUT_HASHES,
            "background": EXPECTED_BACKGROUND_HASH,
            "independent_mia": EXPECTED_DAYBREAK_MIA_HASH,
        },
        "target_file_hashes": target_file_hashes,
        "network_access": False,
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
    }
    report_path = out_dir / "diagnostics_v6.json"
    report_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: v6 candidate directories generated")
    print(f"known feedback-winner AIA power: {feedback_power:.15f}")
    for name, hashes in candidates.items():
        print(name)
        print(f"  MIA {hashes['F_mia.csv']}")
        print(f"  AIA {hashes['F_aia.csv']}")
    print(f"diagnostics: {report_path}")


if __name__ == "__main__":
    main()
