"""Generate the Daybreak Blue reviewed PWS Cup 2026 attack candidate.

This is a competition-only, local-data-only orchestration layer.  Daybreak Blue
independently reviewed the existing v1-v3 evidence and selected the conservative
v3 main pair:

* MIA: v3_private_demotion_transport
* AIA: v3_retained_v2_adaptive

The wrapper deliberately adds operational safety instead of changing ranks
without target-wise truth: it verifies the distributed SHA-256 manifest, locks
the preliminary team-24 target set and the correct shared total (508), and runs
strict finite/key checks in addition to the kit validators.  It performs no
network access, authentication, packaging, or submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from attack_phase import generate_submission as v1
from attack_phase_v2 import generate_submission_v2 as v2
from attack_phase_v3 import generate_submission_v3 as v3
from pwscup2026.kit import validate as official_validate


EXPECTED_SELF_TARGET = "24"
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
EXPECTED_MIA_TOTAL_SHARED = 508
MANIFEST_PATTERN = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daybreak Blueが監査したPWSCup2026予備戦team24攻撃候補を、"
            "配布済みローカルCSVだけから生成します。"
        )
    )
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--self-target-id", default=EXPECTED_SELF_TARGET)
    parser.add_argument("--exclude-targets", default=EXPECTED_SELF_TARGET)
    parser.add_argument(
        "--mia-total-shared", type=int, default=EXPECTED_MIA_TOTAL_SHARED
    )
    parser.add_argument("--mia-min-per-target", type=int, default=25)
    parser.add_argument("--mia-max-per-target", type=int, default=27)
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument("--mia-density-weight", type=float, default=0.35)
    parser.add_argument("--mia-rare-weight", type=float, default=0.65)
    parser.add_argument("--aia-active-fraction", type=float, default=0.5)
    parser.add_argument("--aia-link-trust-distance", type=float, default=0.20)
    parser.add_argument("--aia-anomalous-event-rate", type=float, default=0.50)
    parser.add_argument(
        "--aia-anomalous-link-trust-distance", type=float, default=0.08
    )
    parser.add_argument(
        "--skip-domias",
        action="store_true",
        help="高速な動作確認専用。本命生成では指定しないこと。",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="local outputにも比較variantを保存しない",
    )
    parser.add_argument(
        "--acknowledge-prelim-private-block-rule",
        action="store_true",
        help=(
            "予備戦ルール§3.1/§4.2の『AIA標的は非共有ブロック』を"
            "確認した場合だけself-private降格を有効化する"
        ),
    )
    args = parser.parse_args()
    args.self_target_id = str(args.self_target_id).strip()
    args.exclude_targets = tuple(
        value.strip()
        for value in str(args.exclude_targets).split(",")
        if value.strip()
    )
    validate_preliminary_settings(args, parser.error)
    return args


def validate_preliminary_settings(
    args: argparse.Namespace,
    fail: Any = ValueError,
) -> None:
    """Fail closed if this dedicated flow is pointed at another competition setup."""

    def reject(message: str) -> None:
        if isinstance(fail, type) and issubclass(fail, Exception):
            raise fail(message)
        fail(message)

    if args.self_target_id != EXPECTED_SELF_TARGET:
        reject(f"このフローは予備戦team24専用です: {args.self_target_id!r}")
    if args.exclude_targets != (EXPECTED_SELF_TARGET,):
        reject("除外targetは自チーム24だけである必要があります")
    if len(args.exclude_targets) != len(set(args.exclude_targets)):
        reject("除外targetに重複があります")
    if args.mia_total_shared != EXPECTED_MIA_TOTAL_SHARED:
        reject("予備戦team24の対象20チームに対する共有総数は508です")
    if not args.acknowledge_prelim_private_block_rule:
        reject(
            "--acknowledge-prelim-private-block-rule が必要です。"
            "ルールブック§3.1/§4.2を確認してください"
        )
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        reject("AIA active fractionは0〜1です")
    for name in (
        "aia_link_trust_distance",
        "aia_anomalous_link_trust_distance",
    ):
        if float(getattr(args, name)) < 0.0:
            reject(f"{name}は非負である必要があります")
    if float(args.mia_transport_epsilon) <= 0.0:
        reject("mia_transport_epsilonは正数である必要があります")
    if not 0.0 <= args.aia_anomalous_event_rate <= 1.0:
        reject("AIA anomalous event rateは0〜1です")
    if not 0.0 <= args.mia_density_weight <= 1.0:
        reject("MIA density weightは0〜1です")
    if not 0.0 <= args.mia_rare_weight <= 1.0:
        reject("MIA rare weightは0〜1です")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(targets_dir: Path) -> dict[str, Any]:
    """Verify every file named by the organizer-provided local manifest."""
    manifest_path = targets_dir / "SHA256SUMS.txt"
    if not manifest_path.is_file():
        raise ValueError(f"SHA256SUMS.txtが見つかりません: {manifest_path}")
    entries: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        match = MANIFEST_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(f"SHA256SUMS.txtの{line_number}行目が不正です")
        expected, relative_name = match.groups()
        entries.append((expected.lower(), relative_name))
    if not entries:
        raise ValueError("SHA256SUMS.txtが空です")

    mismatches: list[str] = []
    missing: list[str] = []
    for expected, relative_name in entries:
        candidate = targets_dir / relative_name
        if not candidate.is_file():
            missing.append(relative_name)
            continue
        if sha256_file(candidate) != expected:
            mismatches.append(relative_name)
    if missing or mismatches:
        raise ValueError(
            "配布攻撃データのSHA-256検証に失敗しました: "
            f"missing={missing}, mismatches={mismatches}"
        )
    return {
        "manifest": str(manifest_path),
        "verified_files": len(entries),
        "missing": 0,
        "mismatches": 0,
    }


def require_expected_targets(targets: Iterable[str]) -> None:
    observed = tuple(sorted((str(value) for value in targets), key=int))
    if observed != EXPECTED_TARGET_IDS:
        raise ValueError(
            "team-shared予備戦target集合が期待値と異なります: "
            f"expected={list(EXPECTED_TARGET_IDS)}, observed={list(observed)}"
        )


def strict_extra_checks(f_mia: pd.DataFrame, f_aia: pd.DataFrame) -> dict[str, Any]:
    """Cover finite/key checks that the distributed logical validator omits."""
    expected_mia_columns = ["record_id", *EXPECTED_TARGET_IDS]
    if list(f_mia.columns) != expected_mia_columns:
        raise ValueError("F_mia列順またはtarget集合が不正です")
    mia_values = f_mia[list(EXPECTED_TARGET_IDS)].to_numpy(dtype=float)
    if not np.isfinite(mia_values).all():
        raise ValueError("F_miaにNaNまたは無限値があります")
    if ((mia_values < 0.0) | (mia_values > 1.0)).any():
        raise ValueError("F_miaに[0,1]外の値があります")
    if f_mia["record_id"].duplicated().any():
        raise ValueError("F_miaのrecord_idが重複しています")
    for target_id in EXPECTED_TARGET_IDS:
        if f_mia[target_id].nunique(dropna=False) != len(f_mia):
            raise ValueError(f"F_mia target {target_id}にtieがあります")

    if list(f_aia.columns) != ["target_k", "challenge_row_id", "time_hat"]:
        raise ValueError("F_aia列順が不正です")
    aia_time = f_aia["time_hat"].to_numpy(dtype=float)
    if not np.isfinite(aia_time).all() or (aia_time < 0.0).any():
        raise ValueError("F_aiaのtime_hatは有限非負である必要があります")
    if f_aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("F_aiaの(target_k, challenge_row_id)が重複しています")

    return {
        "mia_rows": len(f_mia),
        "mia_columns": len(f_mia.columns),
        "mia_tie_free_targets": len(EXPECTED_TARGET_IDS),
        "aia_rows": len(f_aia),
        "aia_unique_keys": len(f_aia),
        "all_numeric_values_finite": True,
    }


def run_validators(
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    targets_dir: Path,
) -> dict[str, Any]:
    strict_preflight = v2.validate_output_frames(
        f_mia, f_aia, candidates, targets, challenges
    )
    extra = strict_extra_checks(f_mia, f_aia)
    official_dist = official_validate.load_dist(targets_dir)
    official_mia = official_validate.validate_mia_submission(
        f_mia, {}, official_dist
    )
    official_aia = official_validate.validate_aia_submission(
        f_aia, {}, official_dist
    )
    if not official_mia.ok or not official_aia.ok:
        raise ValueError(
            "公式logical validatorに失敗しました: "
            f"MIA={official_mia.errors}, AIA={official_aia.errors}"
        )
    return {
        "v2_strict_preflight": strict_preflight,
        "daybreak_extra_checks": extra,
        "official_mia": "PASS",
        "official_aia": "PASS",
    }


def write_csv_bundle(
    directory: Path,
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    *,
    allow_only_csvs: bool,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    if allow_only_csvs:
        unexpected = sorted(
            path.name
            for path in directory.iterdir()
            if path.name not in {"F_mia.csv", "F_aia.csv"}
        )
        if unexpected:
            raise ValueError(
                f"candidate-dirに提出外ファイルがあります: {unexpected}"
            )
    mia_path = directory / "F_mia.csv"
    aia_path = directory / "F_aia.csv"
    # Preserve the kit's platform-native CSV convention so a conservative
    # rerun is byte-for-byte comparable with the validated v3 candidate.
    f_mia.to_csv(mia_path, index=False)
    f_aia.to_csv(aia_path, index=False)
    return {
        "F_mia.csv": sha256_file(mia_path),
        "F_aia.csv": sha256_file(aia_path),
    }


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = args.targets.resolve()
    out_dir = args.out_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    if out_dir == candidate_dir:
        raise SystemExit("--out-dirと--candidate-dirは別の場所にしてください")

    manifest = verify_sha256_manifest(targets_dir)
    candidates, background, targets, challenges, quality = v1.load_inputs(
        dist,
        targets_dir,
        exclude_target_ids=args.exclude_targets,
    )
    require_expected_targets(targets)
    require_expected_targets(challenges)

    self_challenge_path = targets_dir / f"aia_challenge_{args.self_target_id}.csv"
    self_challenge = pd.read_csv(self_challenge_path)
    known_private, private_diagnostics = v3.discover_known_private_records(
        candidates, self_challenge
    )
    rare_mask, rare_source = v2.load_candidate_rare_mask(dist, candidates)
    if rare_mask is not None:
        private_diagnostics["matched_rare_rows"] = int(
            np.count_nonzero(known_private & rare_mask)
        )

    target_count = len(targets)
    if not (
        target_count * args.mia_min_per_target
        <= args.mia_total_shared
        <= target_count * args.mia_max_per_target
    ):
        raise SystemExit("MIA capacity設定がtarget数と両立しません")

    horizon, tau = v1.load_phase_parameters(dist)
    abstain_time = v2.safe_abstain_time(horizon, tau)

    print("Daybreak Blue selection: conservative v3 main pair", flush=True)
    f_mia, mia_variants, mia_diagnostics = v3.build_mia_v3(
        candidates,
        background,
        targets,
        known_private,
        minimum=args.mia_min_per_target,
        maximum=args.mia_max_per_target,
        total=args.mia_total_shared,
        transport_epsilon=args.mia_transport_epsilon,
        include_domias=not args.skip_domias,
        density_weight=args.mia_density_weight,
        rare_mask=rare_mask,
        rare_weight=args.mia_rare_weight,
    )
    f_aia, aia_variants, aia_diagnostics = v3.build_aia_v3(
        background,
        targets,
        challenges,
        main_fraction=args.aia_active_fraction,
        variant_fractions=[0.25, 0.5, 0.75, 1.0],
        abstain_time=abstain_time,
        horizon_time=horizon,
        link_trust_distance=args.aia_link_trust_distance,
        baseline_abstain_time=v1.choose_low_hit_time(
            background, horizon=horizon, tau=tau
        ),
        anomalous_event_rate=args.aia_anomalous_event_rate,
        anomalous_link_trust_distance=(
            args.aia_anomalous_link_trust_distance
        ),
    )

    validation = run_validators(
        f_mia,
        f_aia,
        candidates,
        targets,
        challenges,
        targets_dir,
    )
    output_hashes = write_csv_bundle(
        out_dir, f_mia, f_aia, allow_only_csvs=False
    )
    candidate_hashes = write_csv_bundle(
        candidate_dir, f_mia, f_aia, allow_only_csvs=True
    )
    if output_hashes != candidate_hashes:
        raise RuntimeError("local outputとcandidate CSVのhashが一致しません")

    if not args.main_only:
        variants_dir = out_dir / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in mia_variants.items():
            frame.to_csv(
                variants_dir / f"F_mia_{name}.csv",
                index=False,
            )
        for name, frame in aia_variants.items():
            frame.to_csv(
                variants_dir / f"F_aia_{name}.csv",
                index=False,
            )

    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_daybreak_blue",
        "network_access": False,
        "submission_performed": False,
        "token_read": False,
        "model_review": {
            "model": "gpt-daybreak-blue-latest",
            "selected_mia": "v3_private_demotion_transport",
            "selected_aia": "v3_retained_v2_adaptive",
            "reason": (
                "No experimental change had target-wise truth or a non-degradation "
                "guarantee strong enough to displace the validated v3 main pair."
            ),
        },
        "rule_basis": {
            "rulebook_sections": ["3.1", "4.2"],
            "self_private_matches": int(known_private.sum()),
            "acknowledged": True,
        },
        "manifest": manifest,
        "quality": quality,
        "rare_source": rare_source,
        "self_private_detection": private_diagnostics,
        "parameters": {
            "self_target_id": args.self_target_id,
            "target_ids": list(EXPECTED_TARGET_IDS),
            "mia_total_shared": args.mia_total_shared,
            "domias_included": not args.skip_domias,
            "horizon_years": horizon,
            "aia_tau_time": tau,
        },
        "validation": validation,
        "hashes": output_hashes,
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only per-target truth is unavailable locally.",
            "The outputs are generated only from distributed local competition files.",
            "No network, token, packaging, or submission operation is implemented.",
        ],
    }
    (out_dir / "diagnostics_daybreak.json").write_text(
        json.dumps(
            v2.clean_json_v2(diagnostics),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # Re-read the exact delivered bytes and repeat the strict/official checks.
    delivered_mia = pd.read_csv(candidate_dir / "F_mia.csv")
    delivered_aia = pd.read_csv(candidate_dir / "F_aia.csv")
    run_validators(
        delivered_mia,
        delivered_aia,
        candidates,
        targets,
        challenges,
        targets_dir,
    )
    print(f"PASS: F_mia.csv {len(delivered_mia)} rows x {len(delivered_mia.columns)} cols")
    print(f"PASS: F_aia.csv {len(delivered_aia)} rows")
    print(f"PASS: verified manifest files={manifest['verified_files']}")
    print(f"PASS: known private demotion rows={int(known_private.sum())}")
    print(f"candidate: {candidate_dir}")


if __name__ == "__main__":
    main()
