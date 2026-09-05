"""Generate version-5 probe and final-candidate CSV directories.

This generator is intentionally a post-processing layer.  It uses committed
v3/v4 and independent attack candidates plus the organizer-distributed public
targets, so the original private B/A_bg runtime directory is not required.
It creates no archive, reads no token, and performs no submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

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
from attack_phase_v5.mia_tail_assignment import assign_unique_top_tail  # noqa: E402
from pwscup2026.kit import validate as official_validate  # noqa: E402


DEFAULT_STRUCTURED_VARIANT = "consensus_v4_lean__independent_fallback"
EXPECTED_AIA_ROWS_PER_TARGET = 326
EXPECTED_AIA_ACTIVE_PER_TARGET = 163
EXPECTED_MIA_ROWS = 1_049
EXPECTED_INPUT_HASHES = {
    "v3_mia": "8236A9A572306C79BE2D417FCA8F6BA395BCA1F93BA3E90DC99642A2010A4BB2",
    "v4_mia": "D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B",
    "v4_aia": "651D754683BCE4B6F22FD7311AB1584B01C2F42FBB73B13996303472411DFA3A",
    "independent_aia": "BDEA52C1FD7F77F9CF7054E8C9D64F3307B455521318FD73495B28A86BCF35E1",
    "targets_manifest": "A7653F34B890F7E1095153AB50C75E5CC30EDF4272A8D83BFE7065D5B81746E5",
}


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    parser = argparse.ArgumentParser(
        description="保存済み候補と公開標的からPWSCUP2026攻撃v5を生成します。"
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
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
    parser.add_argument(
        "--candidate-root", type=Path, default=team_dir
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "local-output" / "attack-v5"
    )
    parser.add_argument("--structured-variant", default=DEFAULT_STRUCTURED_VARIANT)
    parser.add_argument("--tail-width", type=int, default=12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Reject stale or silently changed saved inputs before generating a candidate."""
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_INPUT_HASHES[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED_INPUT_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"v5 input hash mismatch: {mismatches}")
    return observed


def verify_targets_manifest(directory: Path) -> dict[str, str]:
    """Verify every file named by the organizer-distributed SHA256 manifest."""
    manifest = directory / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pieces = stripped.split(maxsplit=1)
        if len(pieces) != 2:
            raise ValueError(f"invalid SHA256SUMS line: {line!r}")
        digest, relative_name = pieces
        relative_name = relative_name.lstrip("* ")
        if not relative_name or Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
            raise ValueError(f"unsafe SHA256SUMS path: {relative_name!r}")
        expected[relative_name] = digest.upper()
    if not expected:
        raise ValueError("SHA256SUMS.txt is empty")
    observed = {
        relative_name: sha256_file(directory / relative_name)
        for relative_name in expected
    }
    mismatches = {
        name: {"expected": expected[name], "observed": value}
        for name, value in observed.items()
        if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"public target hash mismatch: {mismatches}")
    return observed


def load_public_targets(
    directory: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    config = json.loads((directory / "mia_columns.json").read_text(encoding="utf-8"))
    target_ids = sorted(
        (str(value) for value in config["target_columns"] if str(value) != "24"),
        key=int,
    )
    if tuple(target_ids) != EXPECTED_TARGET_IDS:
        raise ValueError(
            f"target集合が不正です: expected={EXPECTED_TARGET_IDS}, observed={target_ids}"
        )
    releases = {
        target_id: pd.read_csv(directory / f"C_{target_id}.csv").reset_index(drop=True)
        for target_id in target_ids
    }
    challenges = {
        target_id: pd.read_csv(
            directory / f"aia_challenge_{target_id}.csv"
        ).reset_index(drop=True)
        for target_id in target_ids
    }
    return releases, challenges


def validate_mia(frame: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    expected_columns = ["record_id", *EXPECTED_TARGET_IDS]
    if list(frame.columns) != expected_columns:
        raise ValueError("F_miaの列または列順が不正です")
    if len(frame) != EXPECTED_MIA_ROWS or len(reference) != EXPECTED_MIA_ROWS:
        raise ValueError("F_miaの行数が不正です")
    if frame["record_id"].duplicated().any():
        raise ValueError("F_miaのrecord_idが重複しています")
    if set(frame["record_id"].astype(int)) != set(reference["record_id"].astype(int)):
        raise ValueError("F_miaのrecord_id集合がv4と異なります")
    values = frame[list(EXPECTED_TARGET_IDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("F_miaのscoreが有限な[0,1]ではありません")
    ties = {
        target_id: len(frame) - frame[target_id].nunique(dropna=False)
        for target_id in EXPECTED_TARGET_IDS
    }
    if any(ties.values()):
        raise ValueError(f"F_miaにtieがあります: {ties}")
    return {"rows": len(frame), "columns": len(frame.columns), "ties": ties}


def validate_aia(
    frame: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
    *,
    abstain_time: float = 11.0,
) -> dict[str, Any]:
    if list(frame.columns) != list(OFFICIAL_COLUMNS):
        raise ValueError("F_aiaの列または列順が不正です")
    if frame.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("F_aiaの複合キーが重複しています")
    target_ids = sorted(frame["target_k"].astype(int).astype(str).unique(), key=int)
    if tuple(target_ids) != EXPECTED_TARGET_IDS:
        raise ValueError("F_aiaのtarget集合が不正です")
    times = frame["time_hat"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0.0).any():
        raise ValueError("F_aiaのtime_hatが不正です")
    active_counts: dict[str, int] = {}
    for target_id in EXPECTED_TARGET_IDS:
        block = frame.loc[frame["target_k"].astype(int).eq(int(target_id))]
        expected_ids = set(
            challenges[target_id]["challenge_row_id"].astype(int).tolist()
        )
        if len(block) != EXPECTED_AIA_ROWS_PER_TARGET or set(
            block["challenge_row_id"].astype(int)
        ) != expected_ids:
            raise ValueError(f"target {target_id}のchallenge行集合が不正です")
        active = int(
            np.count_nonzero(
                ~np.isclose(block["time_hat"].to_numpy(dtype=float), abstain_time)
            )
        )
        if active != EXPECTED_AIA_ACTIVE_PER_TARGET:
            raise ValueError(
                f"target {target_id}のactive件数が{active}です（163を期待）"
            )
        active_counts[target_id] = active
    return {"rows": len(frame), "active_per_target": active_counts}


def write_candidate(
    directory: Path,
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    *,
    mia_source: Path | None = None,
    aia_source: Path | None = None,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in directory.iterdir()
        if path.name not in {"F_mia.csv", "F_aia.csv"}
    )
    if unexpected:
        raise ValueError(f"candidate directoryに余分なファイルがあります: {unexpected}")
    mia_path = directory / "F_mia.csv"
    aia_path = directory / "F_aia.csv"
    if mia_source is None:
        f_mia.to_csv(mia_path, index=False)
    else:
        shutil.copyfile(mia_source, mia_path)
    if aia_source is None:
        f_aia.to_csv(aia_path, index=False)
    else:
        shutil.copyfile(aia_source, aia_path)
    return {"F_mia.csv": sha256_file(mia_path), "F_aia.csv": sha256_file(aia_path)}


def main() -> None:
    args = parse_args()
    targets_dir = args.targets.resolve()
    candidate_root = args.candidate_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v3_mia_path = args.v3_dir.resolve() / "F_mia.csv"
    v4_mia_path = args.v4_dir.resolve() / "F_mia.csv"
    v4_aia_path = args.v4_dir.resolve() / "F_aia.csv"
    independent_aia_path = args.independent_dir.resolve() / "F_aia.csv"
    input_hashes = verify_input_hashes(
        {
            "v3_mia": v3_mia_path,
            "v4_mia": v4_mia_path,
            "v4_aia": v4_aia_path,
            "independent_aia": independent_aia_path,
            "targets_manifest": targets_dir / "SHA256SUMS.txt",
        }
    )
    target_file_hashes = verify_targets_manifest(targets_dir)
    v3_mia = pd.read_csv(v3_mia_path)
    v4_mia = pd.read_csv(v4_mia_path)
    v4_aia = pd.read_csv(v4_aia_path)
    independent_aia = pd.read_csv(independent_aia_path)
    releases, challenges = load_public_targets(targets_dir)

    repaired_mia, mia_diagnostics = assign_unique_top_tail(
        v3_mia, v4_mia, top_l=args.tail_width
    )
    aia_variants, aia_diagnostics = build_aia_v5_variants(
        v4_aia, independent_aia, releases, challenges
    )
    if args.structured_variant not in aia_variants:
        raise ValueError(
            f"unknown structured AIA variant: {args.structured_variant}; "
            f"available={sorted(aia_variants)}"
        )
    structured_aia = aia_variants[args.structured_variant]

    validation = {
        "v4_mia": validate_mia(v4_mia, v4_mia),
        "repaired_mia": validate_mia(repaired_mia, v4_mia),
        "v4_aia": validate_aia(v4_aia, challenges),
        "independent_aia": validate_aia(independent_aia, challenges),
        "structured_aia": validate_aia(structured_aia, challenges),
    }
    official_dist = official_validate.load_dist(targets_dir)
    for label, frame in (
        ("v4_mia", v4_mia),
        ("repaired_mia", repaired_mia),
    ):
        result = official_validate.validate_mia_submission(frame, {}, official_dist)
        if not result.ok:
            raise ValueError(f"official MIA validator failed for {label}: {result.errors}")
    for label, frame in (
        ("independent_aia", independent_aia),
        ("structured_aia", structured_aia),
    ):
        result = official_validate.validate_aia_submission(frame, {}, official_dist)
        if not result.ok:
            raise ValueError(f"official AIA validator failed for {label}: {result.errors}")

    hashes = {
        "attack-candidate-v5-probe-independent": write_candidate(
            candidate_root / "attack-candidate-v5-probe-independent",
            v4_mia,
            independent_aia,
            mia_source=v4_mia_path,
            aia_source=independent_aia_path,
        ),
        "attack-candidate-v5-structured": write_candidate(
            candidate_root / "attack-candidate-v5-structured",
            v4_mia,
            structured_aia,
            mia_source=v4_mia_path,
        ),
        f"attack-candidate-v5-tail{args.tail_width}": write_candidate(
            candidate_root / f"attack-candidate-v5-tail{args.tail_width}",
            repaired_mia,
            structured_aia,
        ),
    }

    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    repaired_mia.to_csv(
        variants_dir / f"F_mia_unique_tail_{args.tail_width}.csv", index=False
    )
    for name, frame in aia_variants.items():
        frame.to_csv(variants_dir / f"F_aia_{name}.csv", index=False)
    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_v5",
        "attack_version": 5,
        "live_scores": {
            "v4": {
                "attack_power": 0.292,
                "mia_all": 0.1024,
                "mia_rare": 0.3333,
                "aia_all": 0.1171,
                "aia_rare": 0.1,
            },
            "leader": {
                "attack_power": 0.368,
                "mia_all": 0.1039,
                "mia_rare": 0.25,
                "aia_all": 0.1827,
                "aia_rare": 0.0519,
            },
        },
        "structured_aia_variant": args.structured_variant,
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "validation": validation,
        "candidate_hashes": hashes,
        "input_hashes": input_hashes,
        "target_file_hashes": target_file_hashes,
        "network_access": False,
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (out_dir / "diagnostics_v5.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS: v5 candidate directories generated")
    for name, values in hashes.items():
        print(name)
        print(f"  MIA {values['F_mia.csv']}")
        print(f"  AIA {values['F_aia.csv']}")
    print(f"diagnostics: {out_dir / 'diagnostics_v5.json'}")


if __name__ == "__main__":
    main()
