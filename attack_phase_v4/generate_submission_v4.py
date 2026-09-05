"""Generate the PWS Cup 2026 preliminary attack candidate version 4.

V4 is a deliberately separate layer over the validated v3 attack:

* MIA uses the self-private known negatives as an empirical target-wise null,
  combines reciprocal nearest-neighbour evidence across four views, and adds a
  cross-target contrast.  The v3 rare-only ordering is retained inside the
  public rare stratum so the strongest observed leaderboard channel is not
  discarded.
* AIA retains the v2/v3 membership selection.  A background-trained outcome
  model detects target releases whose QI/outcome association is not intact;
  only those target blocks replace active time predictions with the censoring
  horizon.  Abstentions and the selected row set are never changed.

Only organizer-distributed local files are read.  There is no network access,
token handling, archive creation, or submission operation in this module.
"""

from __future__ import annotations

import argparse
import hashlib
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

from attack_phase import generate_submission as v1  # noqa: E402
from attack_phase_v2 import generate_submission_v2 as v2  # noqa: E402
from attack_phase_v3 import generate_submission_v3 as v3  # noqa: E402
from attack_phase_v4.aia_outcome_integrity import (  # noqa: E402
    OutcomeIntegrityFingerprint,
    fixed_selection_horizon,
    route_target_blocks,
)
from attack_phase_v4.mia_private_conformal import (  # noqa: E402
    build_private_conformal_mia,
    demote_known_private,
    preserve_subset_order,
    reciprocal_rank_fusion,
)
from pwscup2026.kit import validate as official_validate  # noqa: E402


EXPECTED_SELF_TARGET = "24"
EXPECTED_TARGET_IDS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
EXPECTED_MIA_TOTAL_SHARED = 508
DEFAULT_MIA_DONOR_RATE = 0.50
DEFAULT_MIA_MIN_SIGNAL_ROWS = 25
DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PWS Cup 2026予備戦team24の攻撃候補v4をローカル生成します。"
    )
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--self-target-id", default=EXPECTED_SELF_TARGET)
    parser.add_argument("--exclude-targets", default=EXPECTED_SELF_TARGET)
    parser.add_argument("--aia-active-fraction", type=float, default=0.5)
    parser.add_argument(
        "--aia-link-trust-distance",
        type=float,
        default=v2.DEFAULT_AIA_LINK_TRUST_DISTANCE,
    )
    parser.add_argument(
        "--aia-anomalous-event-rate",
        type=float,
        default=v2.DEFAULT_AIA_ANOMALOUS_EVENT_RATE,
    )
    parser.add_argument(
        "--aia-anomalous-link-trust-distance",
        type=float,
        default=v2.DEFAULT_AIA_ANOMALOUS_LINK_TRUST_DISTANCE,
    )
    parser.add_argument(
        "--aia-integrity-auc-threshold",
        type=float,
        default=DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
    )
    parser.add_argument("--mia-min-per-target", type=int, default=25)
    parser.add_argument("--mia-max-per-target", type=int, default=27)
    parser.add_argument(
        "--mia-total-shared", type=int, default=EXPECTED_MIA_TOTAL_SHARED
    )
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument(
        "--mia-density-weight", type=float, default=v2.DEFAULT_MIA_DENSITY_WEIGHT
    )
    parser.add_argument(
        "--mia-rare-weight", type=float, default=v2.DEFAULT_MIA_RARE_WEIGHT
    )
    parser.add_argument(
        "--mia-background-donor-rate",
        type=float,
        default=DEFAULT_MIA_DONOR_RATE,
    )
    parser.add_argument(
        "--mia-min-signal-rows", type=int, default=DEFAULT_MIA_MIN_SIGNAL_ROWS
    )
    parser.add_argument("--skip-domias", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    args.self_target_id = str(args.self_target_id).strip()
    args.exclude_targets = tuple(
        value.strip()
        for value in str(args.exclude_targets).split(",")
        if value.strip()
    )
    if args.self_target_id != EXPECTED_SELF_TARGET:
        parser.error("v4は予備戦team24専用です")
    if args.exclude_targets != (EXPECTED_SELF_TARGET,):
        parser.error("--exclude-targets は自チーム24だけにしてください")
    if args.mia_total_shared != EXPECTED_MIA_TOTAL_SHARED:
        parser.error("予備戦team24の共有総数は508です")
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        parser.error("--aia-active-fraction は0〜1です")
    for name in (
        "aia_link_trust_distance",
        "aia_anomalous_link_trust_distance",
    ):
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} は0以上です")
    for name in (
        "aia_anomalous_event_rate",
        "aia_integrity_auc_threshold",
        "mia_density_weight",
        "mia_rare_weight",
        "mia_background_donor_rate",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} は0〜1です")
    if args.mia_min_signal_rows < 1:
        parser.error("--mia-min-signal-rows は1以上です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    return args


def _require_expected_targets(target_ids: Iterable[str]) -> None:
    observed = tuple(sorted((str(value) for value in target_ids), key=int))
    if observed != EXPECTED_TARGET_IDS:
        raise ValueError(
            "予備戦team24のtarget集合が期待値と異なります: "
            f"expected={list(EXPECTED_TARGET_IDS)}, observed={list(observed)}"
        )


def exact_background_donor_mask(
    released: pd.DataFrame,
    background: pd.DataFrame,
    *,
    columns: tuple[str, ...] = tuple(v1.MIA_REQUIRED_COLUMNS),
) -> np.ndarray:
    """Return exact full-row matches against the public background."""
    v1.require_columns(released, columns, "released table")
    v1.require_columns(background, columns, "public background")
    if released[list(columns)].isna().any().any():
        raise ValueError("released tableの背景照合列に欠測があります")
    if background[list(columns)].isna().any().any():
        raise ValueError("public backgroundの背景照合列に欠測があります")
    background_keys = pd.MultiIndex.from_frame(background[list(columns)])
    released_keys = pd.MultiIndex.from_frame(released[list(columns)])
    return np.asarray(released_keys.isin(background_keys), dtype=bool)


def filter_background_donor_targets(
    targets: dict[str, pd.DataFrame],
    background: pd.DataFrame,
    *,
    trigger_rate: float = DEFAULT_MIA_DONOR_RATE,
    minimum_signal_rows: int = DEFAULT_MIA_MIN_SIGNAL_ROWS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Remove exact A_bg donors only from releases dominated by those rows."""
    if not 0.0 <= trigger_rate <= 1.0:
        raise ValueError("MIA donor trigger rate must be in [0, 1]")
    if minimum_signal_rows < 1:
        raise ValueError("MIA minimum signal rows must be positive")
    filtered: dict[str, pd.DataFrame] = {}
    details: dict[str, Any] = {}
    affected: list[str] = []
    for target_id, released in targets.items():
        donor = exact_background_donor_mask(released, background)
        donor_rows = int(donor.sum())
        signal_rows = len(released) - donor_rows
        donor_rate = donor_rows / max(len(released), 1)
        applied = donor_rate >= trigger_rate and signal_rows >= minimum_signal_rows
        filtered[target_id] = (
            released.loc[~donor].reset_index(drop=True) if applied else released
        )
        if applied:
            affected.append(str(target_id))
        details[str(target_id)] = {
            "released_rows": len(released),
            "exact_background_donor_rows": donor_rows,
            "exact_background_donor_rate": donor_rate,
            "remaining_signal_rows": signal_rows,
            "filter_applied": applied,
        }
    return filtered, {"affected_targets": affected, "targets": details}


def build_mia_v4(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    known_private: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    total: int,
    transport_epsilon: float,
    include_domias: bool,
    density_weight: float,
    rare_mask: np.ndarray | None,
    rare_weight: float,
    donor_trigger_rate: float = DEFAULT_MIA_DONOR_RATE,
    donor_minimum_signal_rows: int = DEFAULT_MIA_MIN_SIGNAL_ROWS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    if rare_mask is None:
        raise ValueError("MIA v4 requires the public candidate rare mask")
    rare_mask = np.asarray(rare_mask)
    if (
        rare_mask.ndim != 1
        or len(rare_mask) != len(candidates)
        or rare_mask.dtype.kind != "b"
        or not rare_mask.any()
    ):
        raise ValueError("MIA v4 rare mask must be a non-empty boolean vector")
    rare_mask = rare_mask.astype(bool, copy=False)
    v3_main, _, v3_diagnostics = v3.build_mia_v3(
        candidates,
        background,
        targets,
        known_private,
        minimum=minimum,
        maximum=maximum,
        total=total,
        transport_epsilon=transport_epsilon,
        include_domias=include_domias,
        density_weight=density_weight,
        rare_mask=rare_mask,
        rare_weight=rare_weight,
    )
    conformal, conformal_diagnostics = build_private_conformal_mia(
        candidates,
        background,
        targets,
        known_private,
    )
    filtered_targets, donor_diagnostics = filter_background_donor_targets(
        targets,
        background,
        trigger_rate=donor_trigger_rate,
        minimum_signal_rows=donor_minimum_signal_rows,
    )
    donor_filtered_diagnostics: dict[str, Any] | None = None
    selected = conformal["conformal_margin_0p50"]
    filtered_selected = selected
    if donor_diagnostics["affected_targets"]:
        filtered_conformal, donor_filtered_diagnostics = build_private_conformal_mia(
            candidates,
            background,
            filtered_targets,
            known_private,
        )
        filtered_selected = filtered_conformal["conformal_margin_0p50"]

    fused = reciprocal_rank_fusion(
        v3_main,
        selected,
        known_private,
        rank_constant=10.0,
    )
    filtered_fused = reciprocal_rank_fusion(
        v3_main,
        filtered_selected,
        known_private,
        rank_constant=10.0,
    )
    main = preserve_subset_order(fused, v3_main, rare_mask)
    main = demote_known_private(main, known_private)
    filtered_rrf_rare = preserve_subset_order(
        filtered_fused, v3_main, rare_mask
    )
    filtered_rrf_rare = demote_known_private(
        filtered_rrf_rare, known_private
    )
    conformal_rare = preserve_subset_order(selected, v3_main, rare_mask)
    conformal_rare = demote_known_private(conformal_rare, known_private)
    rare_preservation = {
        "applied": True,
        "rare_rows": int(np.count_nonzero(rare_mask)),
        "reference": "v3_private_demotion_transport",
        "relative_order_exactly_preserved": True,
    }

    variants = {
        "v4_rrf_k10_rare_preserved": main,
        "v4_rrf_k10_donor_filtered_rare_preserved_experimental": (
            filtered_rrf_rare
        ),
        "v4_private_conformal_rare_preserved": conformal_rare,
        "v4_private_conformal_margin_0p50": selected,
        "v4_private_conformal_margin_0p25": conformal[
            "conformal_margin_0p25"
        ],
        "v4_private_conformal_plain": conformal["conformal_plain"],
        "v3_private_demotion_transport_reference": v3_main,
    }
    diagnostics = {
        "version": 4,
        "main_variant": "v4_rrf_k10_rare_preserved",
        "selection_basis": (
            "Fuse v3 with guaranteed-negative conformal evidence using RRF k=10; "
            "retain donor rows after targeted copy-defense regression, restore "
            "the v3 relative order inside the public rare stratum, and demote "
            "guaranteed negatives again after the ensemble."
        ),
        "cross_target_margin_weight": 0.50,
        "rrf_rank_constant": 10.0,
        "rare_preservation": rare_preservation,
        "known_private_top_counts": v3._private_top_counts(main, known_private),
        "conformal": conformal_diagnostics,
        "background_donor_filter_experiment": {
            **donor_diagnostics,
            "conformal": donor_filtered_diagnostics,
            "selected_for_main": False,
        },
        "v3_reference": v3_diagnostics,
    }
    return main, variants, diagnostics


def build_aia_v4(
    background: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
    *,
    main_fraction: float,
    variant_fractions: list[float],
    abstain_time: float,
    horizon_time: float,
    link_trust_distance: float,
    baseline_abstain_time: float,
    anomalous_event_rate: float,
    anomalous_link_trust_distance: float,
    integrity_auc_threshold: float = DEFAULT_AIA_INTEGRITY_AUC_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    base, _, v3_diagnostics = v3.build_aia_v3(
        background,
        targets,
        challenges,
        main_fraction=main_fraction,
        variant_fractions=variant_fractions,
        abstain_time=abstain_time,
        horizon_time=horizon_time,
        link_trust_distance=link_trust_distance,
        baseline_abstain_time=baseline_abstain_time,
        anomalous_event_rate=anomalous_event_rate,
        anomalous_link_trust_distance=anomalous_link_trust_distance,
    )
    base = base.sort_values(
        ["target_k", "challenge_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    horizon = fixed_selection_horizon(
        base,
        abstain_time=abstain_time,
        horizon_time=horizon_time,
    )
    fingerprint = OutcomeIntegrityFingerprint(background)
    main, main_diagnostics = route_target_blocks(
        base,
        horizon,
        targets,
        fingerprint,
        auc_threshold=integrity_auc_threshold,
    )
    comparison_060, comparison_060_diagnostics = route_target_blocks(
        base,
        horizon,
        targets,
        fingerprint,
        auc_threshold=0.60,
    )
    comparison_062, comparison_062_diagnostics = route_target_blocks(
        base,
        horizon,
        targets,
        fingerprint,
        auc_threshold=0.62,
    )
    threshold_label = f"{integrity_auc_threshold:.2f}".replace(".", "")
    main_name = f"v4_auc_integrity_router_v1select_horizon_t{threshold_label}"
    variants = {
        main_name: main,
        "v4_outcome_integrity_auc_0p60_experimental": comparison_060,
        "v4_outcome_integrity_auc_0p62_experimental": comparison_062,
        "v4_fixed_selection_horizon_reference": horizon,
        "v3_retained_v2_adaptive_reference": base.copy(),
    }
    diagnostics = {
        "version": 4,
        "main_variant": main_name,
        "selection_basis": (
            "Route only releases whose public QI/outcome association is below "
            "the shadow-separated AUC threshold. The validated v3 membership "
            "selection is held exactly fixed and only active times are replaced."
        ),
        "main": main_diagnostics,
        "comparison_auc_0p60": comparison_060_diagnostics,
        "comparison_auc_0p62": comparison_062_diagnostics,
        "v3": v3_diagnostics,
    }
    return main, variants, diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def strict_output_checks(
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    challenges: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    preflight = v2.validate_output_frames(
        f_mia, f_aia, candidates, targets, challenges
    )
    values = f_mia[list(EXPECTED_TARGET_IDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("F_miaに非有限値があります")
    if any(f_mia[target].nunique(dropna=False) != len(f_mia) for target in EXPECTED_TARGET_IDS):
        raise ValueError("F_miaにtieがあります")
    times = f_aia["time_hat"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0.0).any():
        raise ValueError("F_aiaのtime_hatが不正です")
    return {"v2_preflight": preflight, "finite": True, "mia_tie_free": True}


def run_official_validators(
    f_mia: pd.DataFrame, f_aia: pd.DataFrame, targets_dir: Path
) -> dict[str, str]:
    official_dist = official_validate.load_dist(targets_dir)
    mia = official_validate.validate_mia_submission(f_mia, {}, official_dist)
    aia = official_validate.validate_aia_submission(f_aia, {}, official_dist)
    if not mia.ok or not aia.ok:
        raise ValueError(
            f"公式logical validator失敗: MIA={mia.errors}, AIA={aia.errors}"
        )
    return {"mia": "PASS", "aia": "PASS"}


def write_outputs(
    directory: Path,
    f_mia: pd.DataFrame,
    f_aia: pd.DataFrame,
    *,
    diagnostics: dict[str, Any] | None,
    variants: dict[str, dict[str, pd.DataFrame]] | None,
    candidate: bool,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    if candidate:
        unexpected = sorted(
            path.name
            for path in directory.iterdir()
            if path.name not in {"F_mia.csv", "F_aia.csv"}
        )
        if unexpected:
            raise ValueError(f"candidate-dirに提出外ファイルがあります: {unexpected}")
    mia_path = directory / "F_mia.csv"
    aia_path = directory / "F_aia.csv"
    f_mia.to_csv(mia_path, index=False)
    f_aia.to_csv(aia_path, index=False)
    if diagnostics is not None:
        (directory / "diagnostics_v4.json").write_text(
            json.dumps(
                v2.clean_json_v2(diagnostics),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if variants is not None:
        variants_dir = directory / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("F_mia_*.csv", "F_aia_*.csv"):
            for stale_path in variants_dir.glob(pattern):
                if stale_path.is_file():
                    stale_path.unlink()
        for component, frames in variants.items():
            for name, frame in frames.items():
                frame.to_csv(variants_dir / f"F_{component}_{name}.csv", index=False)
    return {"F_mia.csv": sha256_file(mia_path), "F_aia.csv": sha256_file(aia_path)}


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = args.targets.resolve()
    out_dir = args.out_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    if out_dir == candidate_dir:
        raise SystemExit("--out-dirと--candidate-dirは別にしてください")
    candidates, background, targets, challenges, quality = v1.load_inputs(
        dist,
        targets_dir,
        exclude_target_ids=args.exclude_targets,
    )
    _require_expected_targets(targets)
    _require_expected_targets(challenges)
    self_challenge = pd.read_csv(
        targets_dir / f"aia_challenge_{args.self_target_id}.csv"
    )
    known_private, private_diagnostics = v3.discover_known_private_records(
        candidates, self_challenge
    )
    rare_mask, rare_source = v2.load_candidate_rare_mask(dist, candidates)
    if rare_mask is not None:
        private_diagnostics["matched_rare_rows"] = int(
            np.count_nonzero(known_private & rare_mask)
        )
    horizon, tau = v1.load_phase_parameters(dist)
    abstain_time = v2.safe_abstain_time(horizon, tau)

    print(
        "MIA v4: private-conformal evidence + RRF + rare preservation",
        flush=True,
    )
    f_mia, mia_variants, mia_diagnostics = build_mia_v4(
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
        donor_trigger_rate=args.mia_background_donor_rate,
        donor_minimum_signal_rows=args.mia_min_signal_rows,
    )
    print("AIA v4: fixed selection + public outcome-integrity router", flush=True)
    f_aia, aia_variants, aia_diagnostics = build_aia_v4(
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
        anomalous_link_trust_distance=args.aia_anomalous_link_trust_distance,
        integrity_auc_threshold=args.aia_integrity_auc_threshold,
    )
    strict = strict_output_checks(f_mia, f_aia, candidates, targets, challenges)
    official = run_official_validators(f_mia, f_aia, targets_dir)
    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_v4",
        "attack_version": 4,
        "network_access": False,
        "token_read": False,
        "submission_performed": False,
        "leaderboard_feedback_used": {
            "source_version": 2,
            "attack_power": 0.28,
            "mia_all": 0.0867,
            "mia_rare": 0.3333,
            "aia_all": 0.1171,
            "aia_rare": 0.1,
        },
        "quality": quality,
        "rare_source": rare_source,
        "self_private_detection": private_diagnostics,
        "parameters": {
            "self_target_id": args.self_target_id,
            "target_ids": list(EXPECTED_TARGET_IDS),
            "horizon_years": horizon,
            "aia_tau_time": tau,
            "aia_active_fraction": args.aia_active_fraction,
            "mia_total_shared": args.mia_total_shared,
            "domias_included": not args.skip_domias,
        },
        "validation": {"strict": strict, "official": official},
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only per-target truth is unavailable locally.",
            "Shadow evaluations compare methods but are not leaderboard estimates.",
            "The program performs no network, token, packaging, or submission operation.",
        ],
    }
    output_hashes = write_outputs(
        out_dir,
        f_mia,
        f_aia,
        diagnostics=diagnostics,
        variants=(None if args.main_only else {"mia": mia_variants, "aia": aia_variants}),
        candidate=False,
    )
    candidate_hashes = write_outputs(
        candidate_dir,
        f_mia,
        f_aia,
        diagnostics=None,
        variants=None,
        candidate=True,
    )
    if output_hashes != candidate_hashes:
        raise RuntimeError("local outputとcandidate CSVのhashが一致しません")
    delivered_mia = pd.read_csv(candidate_dir / "F_mia.csv")
    delivered_aia = pd.read_csv(candidate_dir / "F_aia.csv")
    strict_output_checks(delivered_mia, delivered_aia, candidates, targets, challenges)
    run_official_validators(delivered_mia, delivered_aia, targets_dir)
    print(f"PASS: F_mia.csv {len(delivered_mia)} x {len(delivered_mia.columns)}")
    print(f"PASS: F_aia.csv {len(delivered_aia)} rows")
    print(f"SHA-256 MIA: {candidate_hashes['F_mia.csv']}")
    print(f"SHA-256 AIA: {candidate_hashes['F_aia.csv']}")
    print(f"candidate: {candidate_dir}")


if __name__ == "__main__":
    main()
