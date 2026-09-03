"""Generate the independent PWS Cup 2026 attack candidate version 3.

Version 3 keeps the validated v2 attack intact and adds one piece of
competition-design information that v1/v2 left unused.  The attacker's own
AIA challenge contains rows from the attacker's non-shared block.  Exact
matches between that challenge and the attack cohort are therefore guaranteed
non-members of every *other* target cohort.  V3 moves only those known
negatives below every eligible record in every MIA column while preserving the
relative order of all eligible records.

That post-processing is monotone-safe for MIA ROC performance: no positive is
demoted and no known negative can consume the low-FPR tail.  A stronger
eligible-only transport is emitted as an experimental variant, but the safe
post-processing is the main submission.  AIA is retained from v2 because the
aggregate leaderboard feedback does not identify a target-wise AIA change that
can be guaranteed locally.

Only locally distributed competition files are read.  This module performs no
network, authentication, packaging, or submission operation.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack_phase import generate_submission as v1
from attack_phase_v2 import generate_submission_v2 as v2


PRIVATE_LINK_COLUMNS = (*v1.QI_COLUMNS, *v1.BLOCK_COLUMNS)
DEFAULT_TOP_CUTOFFS = (10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PWS Cup 2026の独立した攻撃候補v3をローカル生成します。"
    )
    parser.add_argument(
        "--dist",
        type=Path,
        required=True,
        help="B.csv（またはB_practice.csv）とA_bg.csvを含む参加者データ",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="C_<k>.csv、aia_challenge_<k>.csv、mia_columns.jsonのディレクトリ",
    )
    parser.add_argument(
        "--exclude-targets",
        default="",
        help="処理対象から除外するtarget_kのカンマ区切り（自チームを含む）",
    )
    parser.add_argument(
        "--self-target-id",
        required=True,
        help="自チームのtarget_k。self challengeから確定非共有行を検出する",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="本命CSV 2枚だけを複製する提出候補ディレクトリ",
    )
    parser.add_argument("--aia-active-fraction", type=float, default=0.5)
    parser.add_argument(
        "--aia-variant-fractions",
        default="0.25,0.5,0.75,1.0",
        help="比較用AIA候補の選択割合",
    )
    parser.add_argument("--aia-abstain-time", type=float, default=None)
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
    parser.add_argument("--mia-min-per-target", type=int, default=25)
    parser.add_argument("--mia-max-per-target", type=int, default=27)
    parser.add_argument("--mia-total-shared", type=int, default=736)
    parser.add_argument("--mia-transport-epsilon", type=float, default=0.35)
    parser.add_argument(
        "--mia-density-weight",
        type=float,
        default=v2.DEFAULT_MIA_DENSITY_WEIGHT,
    )
    parser.add_argument(
        "--mia-rare-weight",
        type=float,
        default=v2.DEFAULT_MIA_RARE_WEIGHT,
    )
    parser.add_argument("--skip-domias", action="store_true")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="F_mia.csvとF_aia.csvだけを書き出す",
    )
    args = parser.parse_args()
    args.exclude_targets = tuple(
        value.strip() for value in args.exclude_targets.split(",") if value.strip()
    )
    if len(args.exclude_targets) != len(set(args.exclude_targets)):
        parser.error("--exclude-targets に重複があります")
    args.self_target_id = str(args.self_target_id).strip()
    if not args.self_target_id:
        parser.error("--self-target-id は空にできません")
    if args.self_target_id not in args.exclude_targets:
        parser.error("--self-target-id は --exclude-targets に含めてください")
    if not 0.0 <= args.aia_active_fraction <= 1.0:
        parser.error("--aia-active-fraction は0〜1です")
    if args.aia_link_trust_distance < 0.0:
        parser.error("--aia-link-trust-distance は0以上です")
    if not 0.0 <= args.aia_anomalous_event_rate <= 1.0:
        parser.error("--aia-anomalous-event-rate は0〜1です")
    if args.aia_anomalous_link_trust_distance < 0.0:
        parser.error("--aia-anomalous-link-trust-distance は0以上です")
    if not 0.0 <= args.mia_density_weight <= 1.0:
        parser.error("--mia-density-weight は0〜1です")
    if not 0.0 <= args.mia_rare_weight <= 1.0:
        parser.error("--mia-rare-weight は0〜1です")
    if args.mia_transport_epsilon <= 0.0:
        parser.error("--mia-transport-epsilon は正数です")
    return args


def _require_unique_keys(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    columns = list(columns)
    v1.require_columns(frame, columns, label)
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        raise ValueError(
            f"{label} のself-private照合キーが一意ではありません: "
            f"duplicate_rows={int(duplicates.sum())}"
        )


def _require_finite_integer(series: pd.Series, label: str) -> None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, np.round(values)):
        raise ValueError(f"{label}は有限整数である必要があります")


def _normalize_private_mask(mask: np.ndarray, expected_length: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError("known-private mask length does not match MIA frame")
    if array.dtype.kind != "b":
        raise ValueError("known-private mask must be a one-dimensional boolean array")
    return array.astype(bool, copy=False)


def discover_known_private_records(
    candidates: pd.DataFrame,
    self_challenge: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = PRIVATE_LINK_COLUMNS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identify the attacker's non-shared rows from its own AIA challenge.

    The preliminary challenge specification uses equal-size member and control
    halves.  For the self target, exact candidate matches are from the
    non-shared private block and hence cannot be members of another target.
    The strict guards deliberately stop generation if that design invariant is
    not visible in the supplied files.
    """
    if len(self_challenge) == 0 or len(self_challenge) % 2:
        raise ValueError("self challengeは正の偶数行である必要があります")
    v1.require_columns(candidates, ("record_id", *key_columns), "attack cohort")
    v1.require_columns(
        self_challenge,
        ("challenge_row_id", *key_columns),
        "self challenge",
    )
    if candidates["record_id"].duplicated().any():
        raise ValueError("attack cohortのrecord_idが一意ではありません")
    if self_challenge["challenge_row_id"].duplicated().any():
        raise ValueError("self challengeのchallenge_row_idが一意ではありません")
    _require_finite_integer(candidates["record_id"], "attack cohortのrecord_id")
    _require_finite_integer(
        self_challenge["challenge_row_id"], "self challengeのchallenge_row_id"
    )
    if candidates[list(key_columns)].isna().any().any():
        raise ValueError("attack cohortのself-private照合キーに欠測があります")
    if self_challenge[list(key_columns)].isna().any().any():
        raise ValueError("self challengeのself-private照合キーに欠測があります")
    numeric_link_columns = [*v1.QI_COLUMNS, "sex"]
    for frame, label in (
        (candidates, "attack cohort"),
        (self_challenge, "self challenge"),
    ):
        numeric = frame[numeric_link_columns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{label}のself-private照合キーに非有限数があります")
    _require_unique_keys(candidates, key_columns, "attack cohort")
    _require_unique_keys(self_challenge, key_columns, "self challenge")

    candidate_lookup = candidates[[*key_columns, "record_id"]]
    linked = self_challenge[["challenge_row_id", *key_columns]].merge(
        candidate_lookup,
        on=list(key_columns),
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    matched = linked["_merge"].eq("both")
    expected = len(self_challenge) // 2
    actual = int(matched.sum())
    if actual != expected:
        raise ValueError(
            "self challengeの候補一致数が設計上の半数と異なります: "
            f"expected={expected}, actual={actual}"
        )
    matched_ids = linked.loc[matched, "record_id"].astype(np.int64)
    if matched_ids.duplicated().any():
        raise ValueError("self challengeから得たrecord_idが一意ではありません")
    candidate_ids = candidates["record_id"].astype(np.int64)
    known_private = candidate_ids.isin(set(matched_ids.tolist())).to_numpy(dtype=bool)
    if int(known_private.sum()) != expected:
        raise ValueError("self-private maskと一致record_id数が整合しません")

    diagnostics = {
        "key_columns": list(key_columns),
        "candidate_rows": len(candidates),
        "challenge_rows": len(self_challenge),
        "expected_matches": expected,
        "matched_rows": actual,
        "unmatched_controls": len(self_challenge) - actual,
        "candidate_key_duplicates": 0,
        "challenge_key_duplicates": 0,
        "matched_record_ids_unique": True,
    }
    return known_private, diagnostics


def _private_top_counts(
    frame: pd.DataFrame,
    known_private: np.ndarray,
    *,
    cutoffs: tuple[int, ...] = DEFAULT_TOP_CUTOFFS,
) -> dict[str, Any]:
    known_private = _normalize_private_mask(known_private, len(frame))
    target_ids = [column for column in frame.columns if column != "record_id"]
    per_target: dict[str, dict[str, int]] = {}
    totals = {str(cutoff): 0 for cutoff in cutoffs}
    for target_id in target_ids:
        values = frame[target_id].to_numpy(dtype=float)
        order = np.argsort(values, kind="mergesort")[::-1]
        target_counts: dict[str, int] = {}
        for cutoff in cutoffs:
            count = int(np.count_nonzero(known_private[order[:cutoff]]))
            target_counts[str(cutoff)] = count
            totals[str(cutoff)] += count
        per_target[target_id] = target_counts
    return {"totals": totals, "per_target": per_target}


def demote_known_private(
    frame: pd.DataFrame,
    known_private: np.ndarray,
) -> pd.DataFrame:
    """Move known negatives below all eligible rows, preserving eligible order."""
    known_private = _normalize_private_mask(known_private, len(frame))
    if not known_private.any():
        raise ValueError("known-private mask is empty")
    result = frame.copy()
    record_ids = result["record_id"].to_numpy(dtype=np.int64)
    for target_id in [column for column in result.columns if column != "record_id"]:
        source = result[target_id].to_numpy(dtype=float, copy=True)
        if not np.isfinite(source).all():
            raise ValueError(f"MIA column {target_id} contains a non-finite value")
        floor = float(np.min(source)) - max(float(np.ptp(source)), 1.0)
        source[known_private] = floor
        result[target_id] = v1.strict_rank(source, record_ids)
    return result


def blend_mia_frames(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    *,
    secondary_weight: float,
) -> pd.DataFrame:
    """Blend two compatible rank matrices and restore strict target ranks."""
    if not 0.0 <= secondary_weight <= 1.0:
        raise ValueError("secondary_weight must be in [0, 1]")
    if list(primary.columns) != list(secondary.columns):
        raise ValueError("MIA frames have different columns")
    if not primary["record_id"].equals(secondary["record_id"]):
        raise ValueError("MIA frames have different record order")
    result = primary.copy()
    record_ids = result["record_id"].to_numpy(dtype=np.int64)
    for target_id in [column for column in result.columns if column != "record_id"]:
        raw = (
            (1.0 - secondary_weight) * primary[target_id].to_numpy(dtype=float)
            + secondary_weight * secondary[target_id].to_numpy(dtype=float)
        )
        result[target_id] = v1.strict_rank(raw, record_ids)
    return result


def eligible_only_retransport(
    unconstrained: pd.DataFrame,
    known_private: np.ndarray,
    *,
    minimum: int,
    maximum: int,
    total: int,
    transport_epsilon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Experimental transport solved only over rows not known to be private."""
    known_private = _normalize_private_mask(known_private, len(unconstrained))
    target_ids = [column for column in unconstrained.columns if column != "record_id"]
    all_record_ids = unconstrained["record_id"].to_numpy(dtype=np.int64)
    eligible = ~known_private
    eligible_ids = all_record_ids[eligible]
    if total > len(eligible_ids):
        raise ValueError("MIA total exceeds eligible candidates after private exclusion")
    raw = unconstrained.loc[eligible, target_ids].to_numpy(dtype=float)
    affinity = v2._target_specific_affinity(raw, eligible_ids)
    affinity += v1.deterministic_jitter(eligible_ids, target_ids)
    lower = 1.0 / (len(eligible_ids) + 1.0)
    upper = len(eligible_ids) / (len(eligible_ids) + 1.0)
    clipped = np.clip(affinity, lower, upper)
    utility = np.log(clipped) - np.log1p(-clipped)
    assigned, target_counts = v1.constrained_capacity_assignment(
        utility,
        minimum=minimum,
        maximum=maximum,
        total=total,
    )
    transport, transport_diagnostics = v1.sinkhorn_transport(
        utility,
        target_counts,
        null_count=len(eligible_ids) - total,
        epsilon=transport_epsilon,
    )

    restored = np.full((len(unconstrained), len(target_ids)), -1.0, dtype=float)
    for target_index in range(len(target_ids)):
        restored[eligible, target_index] = v1.strict_rank(
            transport[:, target_index], eligible_ids
        )
        restored[:, target_index] = v1.strict_rank(
            restored[:, target_index], all_record_ids
        )
    frame = pd.DataFrame({"record_id": all_record_ids})
    for target_index, target_id in enumerate(target_ids):
        frame[target_id] = restored[:, target_index]
    diagnostics = {
        "eligible_rows": int(eligible.sum()),
        "fixed_private_rows": int(known_private.sum()),
        "eligible_null_count": int(eligible.sum()) - total,
        "final_null_count": len(unconstrained) - total,
        "assigned_candidates": int((assigned >= 0).sum()),
        "per_target": {
            target_id: int(target_counts[index])
            for index, target_id in enumerate(target_ids)
        },
        "transport": transport_diagnostics,
    }
    return frame, diagnostics


def build_mia_v3(
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    baseline, v2_variants, v2_diagnostics = v2.build_mia_v2(
        candidates,
        background,
        targets,
        minimum=minimum,
        maximum=maximum,
        total=total,
        transport_epsilon=transport_epsilon,
        include_domias=include_domias,
        density_weight=density_weight,
        rare_mask=rare_mask,
        rare_weight=rare_weight,
    )
    main = demote_known_private(baseline, known_private)
    light_blend = blend_mia_frames(
        baseline,
        v2_variants["v1_unconstrained"],
        secondary_weight=0.10,
    )
    light_blend = demote_known_private(light_blend, known_private)
    rare_aware = demote_known_private(
        v2_variants["v2_rare_aware_transport"], known_private
    )
    retransport, retransport_diagnostics = eligible_only_retransport(
        v2_variants["v1_unconstrained"],
        known_private,
        minimum=minimum,
        maximum=maximum,
        total=total,
        transport_epsilon=transport_epsilon,
    )
    variants = {
        "v3_private_demotion_transport": main,
        "v3_private_demotion_blend_0p10_experimental": light_blend,
        "v3_private_retransport_experimental": retransport,
        "v3_private_demotion_rare_aware_experimental": rare_aware,
        "v2_selected_transport_reference": baseline,
    }
    rare_private = (
        int(np.count_nonzero(np.asarray(rare_mask, dtype=bool) & known_private))
        if rare_mask is not None
        else None
    )
    diagnostics = {
        "version": 3,
        "main_variant": "v3_private_demotion_transport",
        "selection_basis": (
            "Guaranteed non-members from the attacker's non-shared self AIA block "
            "are demoted after v2 transport; all eligible orderings are preserved."
        ),
        "known_private_rows": int(known_private.sum()),
        "known_private_rare_rows": rare_private,
        "eligible_rows": int((~known_private).sum()),
        "before_private_top_counts": _private_top_counts(
            baseline, known_private
        ),
        "after_private_top_counts": _private_top_counts(main, known_private),
        "retransport_experiment": retransport_diagnostics,
        "v2": v2_diagnostics,
    }
    return main, variants, diagnostics


def build_aia_v3(
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    main, v2_variants, v2_diagnostics = v2.build_aia_v2(
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
    anomaly_experiment = main.copy()
    anomaly_targets = [
        str(target_id)
        for target_id, detail in v2_diagnostics["targets"].items()
        if detail["anomaly_guard_triggered"]
    ]
    horizon_name = f"v2_horizon_{v1.fraction_label(main_fraction)}"
    horizon = v2_variants[horizon_name]
    for target_id in anomaly_targets:
        target_mask = anomaly_experiment["target_k"].astype(int).eq(int(target_id))
        replacement = horizon.loc[
            horizon["target_k"].astype(int).eq(int(target_id)), "time_hat"
        ].to_numpy(dtype=float)
        if int(target_mask.sum()) != len(replacement):
            raise ValueError(f"AIA anomaly experiment row mismatch for target {target_id}")
        anomaly_experiment.loc[target_mask, "time_hat"] = replacement
    variants = {
        "v3_retained_v2_adaptive": main,
        "v3_anomalous_horizon_experimental": anomaly_experiment,
        **v2_variants,
    }
    diagnostics = {
        "version": 3,
        "main_variant": "v3_retained_v2_adaptive",
        "selection_basis": (
            "Leaderboard aggregates do not separate target-wise selection and "
            "time-link errors, so the validated v2 AIA is retained."
        ),
        "anomalous_horizon_experiment_targets": anomaly_targets,
        "v2": v2_diagnostics,
    }
    return main.copy(), variants, diagnostics


def write_outputs(
    out_dir: Path,
    f_mia: pd.DataFrame,
    mia_variants: dict[str, pd.DataFrame],
    f_aia: pd.DataFrame,
    aia_variants: dict[str, pd.DataFrame],
    diagnostics: dict[str, Any],
    *,
    main_only: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    f_mia.to_csv(out_dir / "F_mia.csv", index=False)
    f_aia.to_csv(out_dir / "F_aia.csv", index=False)
    if main_only:
        return
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in mia_variants.items():
        frame.to_csv(variants_dir / f"F_mia_{name}.csv", index=False)
    for name, frame in aia_variants.items():
        frame.to_csv(variants_dir / f"F_aia_{name}.csv", index=False)
    (out_dir / "diagnostics_v3.json").write_text(
        json.dumps(
            v2.clean_json_v2(diagnostics),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dist = args.dist.resolve()
    targets_dir = (args.targets or (dist / "targets")).resolve()
    out_dir = args.out_dir.resolve()
    candidate_dir = (
        args.candidate_dir.resolve() if args.candidate_dir is not None else None
    )
    if candidate_dir == out_dir:
        raise SystemExit("--candidate-dir は --out-dir と別の場所を指定してください")
    candidates, background, targets, challenges, quality = v1.load_inputs(
        dist,
        targets_dir,
        exclude_target_ids=args.exclude_targets,
    )
    self_challenge_path = targets_dir / f"aia_challenge_{args.self_target_id}.csv"
    if not self_challenge_path.exists():
        raise SystemExit(f"self challengeが見つかりません: {self_challenge_path}")
    self_challenge = pd.read_csv(self_challenge_path)
    known_private, private_diagnostics = discover_known_private_records(
        candidates, self_challenge
    )
    rare_mask, rare_source = v2.load_candidate_rare_mask(dist, candidates)
    if rare_mask is not None:
        private_diagnostics["matched_rare_rows"] = int(
            np.count_nonzero(known_private & rare_mask)
        )

    n_targets = len(targets)
    if not (
        n_targets * args.mia_min_per_target
        <= args.mia_total_shared
        <= n_targets * args.mia_max_per_target
    ):
        raise SystemExit(
            "MIA capacity設定が対象数と両立しません: "
            f"targets={n_targets}, min={args.mia_min_per_target}, "
            f"max={args.mia_max_per_target}, total={args.mia_total_shared}"
        )
    if args.mia_total_shared > int((~known_private).sum()):
        raise SystemExit("MIA共有総数がself-private除外後の候補数を超えています")
    fractions = [
        float(value.strip())
        for value in args.aia_variant_fractions.split(",")
        if value.strip()
    ]
    if any(value < 0.0 or value > 1.0 for value in fractions):
        raise SystemExit("--aia-variant-fractions は0〜1です")
    horizon, tau = v1.load_phase_parameters(dist)
    abstain_time = (
        float(args.aia_abstain_time)
        if args.aia_abstain_time is not None
        else v2.safe_abstain_time(horizon, tau)
    )
    if not np.isfinite(abstain_time) or abstain_time < 0.0:
        raise SystemExit("AIA abstain timeは有限の非負値である必要があります")

    print(
        f"入力確認(v3): candidates={len(candidates)}, targets={len(targets)}, "
        f"A_bg={len(background)}, known_private={int(known_private.sum())}"
    )
    print("MIA v3: self challengeで確定した非共有行を全標的の最下位へ移動")
    f_mia, mia_variants, mia_diagnostics = build_mia_v3(
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
    print("AIA v3: 実測フィードバックで分離不能なため検証済みv2本命を維持")
    f_aia, aia_variants, aia_diagnostics = build_aia_v3(
        background,
        targets,
        challenges,
        main_fraction=args.aia_active_fraction,
        variant_fractions=fractions,
        abstain_time=abstain_time,
        horizon_time=horizon,
        link_trust_distance=args.aia_link_trust_distance,
        baseline_abstain_time=v1.choose_low_hit_time(
            background, horizon=horizon, tau=tau
        ),
        anomalous_event_rate=args.aia_anomalous_event_rate,
        anomalous_link_trust_distance=args.aia_anomalous_link_trust_distance,
    )
    preflight = v2.validate_output_frames(
        f_mia, f_aia, candidates, targets, challenges
    )
    diagnostics = {
        "report_type": "authorized_local_privacy_competition_attack_v3",
        "attack_version": 3,
        "network_access": False,
        "leaderboard_feedback": {
            "attack_power": 0.28,
            "mia_all": 0.0867,
            "mia_rare": 0.3333,
            "aia_all": 0.1171,
            "aia_rare": 0.1,
        },
        "quality": quality,
        "parameters": {
            "self_target_id": args.self_target_id,
            "horizon_years": horizon,
            "aia_tau_time": tau,
            "aia_active_fraction": args.aia_active_fraction,
            "aia_link_trust_distance": args.aia_link_trust_distance,
            "aia_anomalous_event_rate": args.aia_anomalous_event_rate,
            "aia_anomalous_link_trust_distance": (
                args.aia_anomalous_link_trust_distance
            ),
            "excluded_target_ids": list(args.exclude_targets),
            "mia_min_per_target": args.mia_min_per_target,
            "mia_max_per_target": args.mia_max_per_target,
            "mia_total_shared": args.mia_total_shared,
            "mia_transport_epsilon": args.mia_transport_epsilon,
            "mia_density_weight": args.mia_density_weight,
            "mia_rare_weight": args.mia_rare_weight,
            "domias_included": not args.skip_domias,
        },
        "self_private_detection": private_diagnostics,
        "rare_source": rare_source,
        "preflight": preflight,
        "mia": mia_diagnostics,
        "aia": aia_diagnostics,
        "limitations": [
            "Organizer-only per-target membership and AIA truth are unavailable locally.",
            "The MIA demotion guarantee follows from the published non-shared self-challenge design.",
            "AIA is intentionally unchanged from v2 pending target-wise official feedback.",
            "The program reads distributed local files only and does not submit anything.",
        ],
    }
    write_outputs(
        out_dir,
        f_mia,
        mia_variants,
        f_aia,
        aia_variants,
        diagnostics,
        main_only=args.main_only,
    )
    if candidate_dir is not None:
        write_outputs(
            candidate_dir,
            f_mia,
            {},
            f_aia,
            {},
            diagnostics,
            main_only=True,
        )

    print("完了(v3):", out_dir)
    print(f"  F_mia.csv: {len(f_mia)} rows x {len(f_mia.columns)} columns")
    print(f"  F_aia.csv: {len(f_aia)} rows")
    print(f"  known-private demotion: {int(known_private.sum())} rows")
    print("  strict preflight: PASS")
    if candidate_dir is not None:
        print("  candidate CSVs:", candidate_dir)


if __name__ == "__main__":
    main()
