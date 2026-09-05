"""Read-only independent QA for the v7 attack candidate.

The script writes evidence only under local-output/attack-v7/final-audit.  It
does not alter the candidate, production package, ZIP, or token.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

# The bundled audit Python intentionally has no PyYAML.  validate.py imports
# config.py at module load, although these pure logical validators do not parse
# YAML.  Supply a fail-closed stub rather than installing or reading anything.
try:
    import yaml as _yaml  # noqa: F401
except ModuleNotFoundError:
    yaml_stub = types.ModuleType("yaml")

    def _yaml_unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("YAML parsing is intentionally unavailable in this audit")

    yaml_stub.safe_load = _yaml_unavailable  # type: ignore[attr-defined]
    sys.modules["yaml"] = yaml_stub

from pwscup2026.kit import validate as official_validate
from pwscup2026.scoring.attack_score import score_aia


EXPECTED_TARGET_IDS = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "12",
    "13",
    "17",
    "18",
    "19",
    "20",
    "22",
    "23",
    "26",
    "29",
)
ABSTAIN_TIME = 11.0
EXPECTED_ACTIVE_ROWS = {
    "6": 217,
    "7": 253,
    "8": 211,
    "9": 274,
    "17": 223,
    "18": 236,
    "19": 268,
}
EXPECTED_OUTPUT_HASHES = {
    "F_mia.csv": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "F_aia.csv": "65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0",
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

CANDIDATE = (
    ROOT
    / "submissions"
    / "prelim"
    / "team24"
    / "attack-candidate-v7-feedback-stack"
)
V4 = ROOT / "submissions" / "prelim" / "team24" / "attack-candidate-v4"
BASE = (
    ROOT
    / "submissions"
    / "prelim"
    / "team24"
    / "attack-candidate-v6-final-horizon-edge"
)
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
DIAGNOSTICS = ROOT / "local-output" / "attack-v7" / "candidate_diagnostics.json"
OUT = Path(__file__).with_name("audit.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def strict_variable_cardinality_check(
    frame: pd.DataFrame, challenges: dict[str, pd.DataFrame]
) -> dict[str, object]:
    if list(frame.columns) != ["target_k", "challenge_row_id", "time_hat"]:
        raise ValueError("unexpected AIA columns")
    if frame.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("duplicate AIA composite keys")
    values = pd.to_numeric(frame["time_hat"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("invalid time_hat")
    for target_id, challenge in challenges.items():
        block = frame.loc[frame["target_k"].astype(int).eq(int(target_id))]
        expected_ids = set(challenge["challenge_row_id"].astype(int))
        observed_ids = set(block["challenge_row_id"].astype(int))
        if len(block) != 326 or observed_ids != expected_ids:
            raise ValueError(f"key mismatch for target {target_id}")
    return {"rows": int(len(frame)), "targets": int(len(challenges))}


def main() -> None:
    cfg = {
        "aia": {"tau_time": 0.5},
        "onset": {"horizon_years": 10.0},
        "scoring": {"low_fpr": 0.01, "aia_rare_se_z": 1.64},
    }
    dist = official_validate.load_dist(TARGETS)
    f_mia = pd.read_csv(CANDIDATE / "F_mia.csv")
    f_aia = pd.read_csv(CANDIDATE / "F_aia.csv")
    base_aia = pd.read_csv(BASE / "F_aia.csv")

    challenge_frames = {
        target_id: pd.read_csv(TARGETS / f"aia_challenge_{target_id}.csv")
        for target_id in EXPECTED_TARGET_IDS
    }

    actual_hashes = {
        "F_mia.csv": sha256_file(CANDIDATE / "F_mia.csv"),
        "F_aia.csv": sha256_file(CANDIDATE / "F_aia.csv"),
    }
    v4_mia_hash = sha256_file(V4 / "F_mia.csv")
    base_mia_hash = sha256_file(BASE / "F_mia.csv")
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))

    source_paths = {
        "base_mia": BASE / "F_mia.csv",
        "base_aia": BASE / "F_aia.csv",
        "v4_aia": V4 / "F_aia.csv",
        "independent_aia": ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-daybreak-independent"
        / "F_aia.csv",
        "routed_aia": ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v6-routed-guided-probe"
        / "F_aia.csv",
        "feedback_v4": ROOT / "attack_phase_v6" / "official_feedback" / "v4.json",
        "feedback_independent": ROOT
        / "attack_phase_v6"
        / "official_feedback"
        / "probe_independent.json",
        "feedback_routed": ROOT
        / "attack_phase_v6"
        / "official_feedback"
        / "routed_guided_probe.json",
        "targets_manifest": TARGETS / "SHA256SUMS.txt",
    }
    actual_input_hashes = {
        label: sha256_file(path) for label, path in source_paths.items()
    }

    manifest_expected: dict[str, str] = {}
    for raw_line in (TARGETS / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        if not raw_line.strip():
            continue
        digest, relative_name = raw_line.strip().split(maxsplit=1)
        relative_name = relative_name.lstrip("* ")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe manifest path: {relative_name}")
        manifest_expected[relative_name] = digest.upper()
    manifest_observed = {
        name: sha256_file(TARGETS / name) for name in manifest_expected
    }

    target_ids = sorted(f_aia["target_k"].astype(int).unique().tolist())
    expected_target_ids = sorted(int(target_id) for target_id in EXPECTED_TARGET_IDS)
    duplicate_keys = int(f_aia.duplicated(["target_k", "challenge_row_id"]).sum())
    finite_nonnegative = bool(
        np.isfinite(f_aia["time_hat"].to_numpy(dtype=float)).all()
        and (f_aia["time_hat"].to_numpy(dtype=float) >= 0.0).all()
    )

    key_checks: dict[str, dict[str, object]] = {}
    active_counts: dict[str, int] = {}
    for target_id in EXPECTED_TARGET_IDS:
        target_int = int(target_id)
        block = f_aia.loc[f_aia["target_k"].astype(int).eq(target_int)]
        expected_ids = set(
            challenge_frames[target_id]["challenge_row_id"].astype(int).tolist()
        )
        observed_ids = set(block["challenge_row_id"].astype(int).tolist())
        active = int(
            np.count_nonzero(
                ~np.isclose(block["time_hat"].to_numpy(dtype=float), ABSTAIN_TIME)
            )
        )
        active_counts[target_id] = active
        key_checks[target_id] = {
            "rows": int(len(block)),
            "unique_composite_keys": int(
                block[["target_k", "challenge_row_id"]].drop_duplicates().shape[0]
            ),
            "challenge_set_exact": observed_ids == expected_ids,
        }

    candidate_keys = f_aia[["target_k", "challenge_row_id"]].copy()
    base_keys = base_aia[["target_k", "challenge_row_id"]].copy()
    key_sequence_equal_base = bool(candidate_keys.equals(base_keys))
    changed_rows_by_target: dict[str, int] = {}
    for target_id in EXPECTED_TARGET_IDS:
        mask = f_aia["target_k"].astype(int).eq(int(target_id)).to_numpy()
        changed_rows_by_target[target_id] = int(
            np.count_nonzero(
                ~np.isclose(
                    f_aia.loc[mask, "time_hat"].to_numpy(dtype=float),
                    base_aia.loc[mask, "time_hat"].to_numpy(dtype=float),
                )
            )
        )
    changed_targets = [
        target_id
        for target_id, changed_count in changed_rows_by_target.items()
        if changed_count
    ]

    logical_mia = official_validate.validate_mia_submission(f_mia, cfg, dist)
    logical_aia = official_validate.validate_aia_submission(f_aia, cfg, dist)
    strict_aia = strict_variable_cardinality_check(f_aia, challenge_frames)

    # A synthetic all-active copy proves that the official logical validator
    # imposes no 163-row cap.  It is kept in memory and never written.
    all_active = f_aia.copy()
    all_active.loc[np.isclose(all_active["time_hat"], ABSTAIN_TIME), "time_hat"] = 9.5
    logical_all_active = official_validate.validate_aia_submission(all_active, cfg, dist)
    all_active_counts = {
        str(target_id): int(
            np.count_nonzero(
                ~np.isclose(
                    all_active.loc[
                        all_active["target_k"].astype(int).eq(target_id), "time_hat"
                    ].to_numpy(dtype=float),
                    ABSTAIN_TIME,
                )
            )
        )
        for target_id in target_ids
    }

    # Negative-control probes document what the official logical validator
    # alone does not enforce.  These malformed frames remain in memory.
    duplicate_probe = pd.concat([f_aia, f_aia.iloc[[0]]], ignore_index=True)
    logical_duplicate_probe = official_validate.validate_aia_submission(
        duplicate_probe, cfg, dist
    )
    inf_probe = f_aia.copy()
    inf_probe.loc[0, "time_hat"] = np.inf
    logical_inf_probe = official_validate.validate_aia_submission(inf_probe, cfg, dist)
    fractional_key_probe = f_aia.copy()
    fractional_key_probe["challenge_row_id"] = fractional_key_probe[
        "challenge_row_id"
    ].astype(float)
    fractional_key_probe.loc[0, "challenge_row_id"] = (
        float(fractional_key_probe.loc[0, "challenge_row_id"]) + 0.25
    )
    logical_fractional_key_probe = official_validate.validate_aia_submission(
        fractional_key_probe, cfg, dist
    )

    horizon = float(cfg["onset"]["horizon_years"])
    tau = float(cfg["aia"]["tau_time"])
    minimum_abstain_error = ABSTAIN_TIME - horizon
    endpoint_score = score_aia(
        None,
        np.asarray([ABSTAIN_TIME]),
        np.asarray([0]),
        np.asarray([horizon]),
        None,
        np.asarray([ABSTAIN_TIME]),
        np.asarray([0]),
        np.asarray([horizon]),
        cfg,
    )

    over_163 = {
        target_id: count for target_id, count in active_counts.items() if count > 163
    }
    expected_over_163 = {
        target_id: EXPECTED_ACTIVE_ROWS[target_id]
        for target_id in sorted(EXPECTED_ACTIVE_ROWS, key=int)
    }

    checks = {
        "candidate_file_set_exact": sorted(path.name for path in CANDIDATE.iterdir())
        == ["F_aia.csv", "F_mia.csv"],
        "output_hashes_match_generator_lock": actual_hashes == EXPECTED_OUTPUT_HASHES,
        "output_hashes_match_saved_diagnostics": actual_hashes
        == diagnostics["output_hashes"],
        "all_generator_inputs_match_hash_lock": actual_input_hashes
        == EXPECTED_INPUT_HASHES,
        "all_manifest_entries_match": manifest_observed == manifest_expected,
        "mia_byte_identical_to_v4": actual_hashes["F_mia.csv"] == v4_mia_hash,
        "mia_byte_identical_to_v6_base": actual_hashes["F_mia.csv"]
        == base_mia_hash,
        "aia_columns_exact": list(f_aia.columns)
        == ["target_k", "challenge_row_id", "time_hat"],
        "aia_rows_20_x_326": len(f_aia) == 20 * 326,
        "target_set_exact": target_ids == expected_target_ids,
        "no_duplicate_composite_keys": duplicate_keys == 0,
        "all_target_key_blocks_exact": all(
            item["rows"] == 326
            and item["unique_composite_keys"] == 326
            and item["challenge_set_exact"]
            for item in key_checks.values()
        ),
        "key_sequence_equal_v6_base": key_sequence_equal_base,
        "only_expected_targets_change_values": changed_targets
        == ["6", "7", "8", "9", "17", "18", "19", "20"],
        "time_hat_finite_nonnegative": finite_nonnegative,
        "official_mia_logical_validator_pass": logical_mia.ok,
        "official_aia_logical_validator_pass": logical_aia.ok,
        "strict_variable_cardinality_validator_pass": strict_aia["rows"]
        == 20 * 326,
        "over_163_counts_match_lock": over_163 == expected_over_163,
        "official_validator_accepts_326_active_each_target": logical_all_active.ok
        and all(count == 326 for count in all_active_counts.values()),
        "time_11_is_guaranteed_miss_at_horizon": minimum_abstain_error > tau
        and endpoint_score.breakdown["succ_time_member"] == 0.0
        and endpoint_score.breakdown["succ_time_control"] == 0.0,
        "scoring_target_count_is_20": len(target_ids) == 20,
    }

    report = {
        "audit_scope": "read-only v7 candidate, generator gates, official logical validator and scoring",
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "candidate": {
            "directory": str(CANDIDATE),
            "files": sorted(path.name for path in CANDIDATE.iterdir()),
            "hashes": actual_hashes,
            "rows": {"F_mia.csv": int(len(f_mia)), "F_aia.csv": int(len(f_aia))},
            "target_count": len(target_ids),
            "rows_per_target": key_checks,
            "active_counts": active_counts,
            "targets_over_163": over_163,
            "changed_rows_vs_v6_base": changed_rows_by_target,
        },
        "generator_input_hashes": actual_input_hashes,
        "target_manifest": {
            "entry_count": len(manifest_expected),
            "all_entries_match": manifest_observed == manifest_expected,
        },
        "formal_cardinality": {
            "candidate_official_result": {
                "ok": logical_aia.ok,
                "errors": logical_aia.errors,
            },
            "synthetic_326_active_each_target": {
                "ok": logical_all_active.ok,
                "errors": logical_all_active.errors,
                "active_counts": all_active_counts,
            },
        },
        "official_validator_negative_control_probes": {
            "appended_duplicate_composite_key_is_accepted": {
                "ok": logical_duplicate_probe.ok,
                "errors": logical_duplicate_probe.errors,
            },
            "positive_infinity_time_hat_is_accepted": {
                "ok": logical_inf_probe.ok,
                "errors": logical_inf_probe.errors,
            },
            "fractional_challenge_id_casting_to_same_integer_is_accepted": {
                "ok": logical_fractional_key_probe.ok,
                "errors": logical_fractional_key_probe.errors,
            },
            "candidate_impact": "none; strict generator checks and exact byte hashes exclude all three malformed cases",
        },
        "time_11_semantics": {
            "official_validator_accepts_nonnegative_without_upper_bound": logical_aia.ok,
            "horizon": horizon,
            "tau": tau,
            "minimum_absolute_error_for_truth_in_0_to_horizon": minimum_abstain_error,
            "native_abstention_branch_in_scorer": False,
            "interpretation": "accepted sentinel used as a guaranteed miss, not a special scorer state",
        },
        "score_1_aggregation": {
            "target_count": len(target_ids),
            "mia_power": "mean_k max(TPR_all_k, TPR_rare_k)",
            "aia_power": "mean_k max(R_time_all_k, R_time_rare_k_after_deadzone)",
            "score_1": "mia_power + aia_power",
            "one_target_aia_delta_multiplier": 1.0 / len(target_ids),
        },
        "known_official_validator_limitations_mitigated_by_generator": [
            "validate_aia_submission compares challenge IDs as sets and does not itself reject duplicate composite keys or require 326 rows; _normalized_aia plus validate_variable_aia and the exact output hash cover the candidate.",
            "validate_aia_submission has no finite-value/upper-bound check for positive infinity; validate_variable_aia rejects all nonfinite values and the exact output hash covers the candidate.",
            "validate_aia_submission casts challenge_row_id to int without checking integrality; exact candidate keys and output hashing cover the candidate.",
        ],
        "package_scope_caveat": "The candidate directory intentionally has no token.txt and is not a submission package; no ZIP/token/production artifact was inspected or changed.",
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
