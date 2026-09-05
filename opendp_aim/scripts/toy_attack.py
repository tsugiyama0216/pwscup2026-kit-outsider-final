"""Run reproducible, unofficial self-attacks against the current PWS Cup C.csv.

This script intentionally reuses an existing synthetic release.  It never calls the
OpenDP generator and therefore does not spend an additional privacy budget.

The official reference attack implementations are used for DOMIAS MIA and nearest-
neighbour AIA.  Because the organiser-only membership and AIA truth files for one's
own submission are unavailable locally, candidate/control cohorts are constructed
from B_practice/B_self and public A_bg.  Results are diagnostics, not official scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "starter" / "reference"
if str(REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DIR))

import _common  # noqa: E402
from pwscup2026.attack import aia as aia_mod  # noqa: E402
from pwscup2026.attack import mia as mia_mod  # noqa: E402
from pwscup2026.common import schema as common_schema  # noqa: E402
from pwscup2026.common import submission_schema  # noqa: E402
from pwscup2026.rare import outlier as outlier_mod  # noqa: E402
from pwscup2026.scoring.attack_score import score_aia, score_mia  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="既存の C.csv に非公式の自己診断用 toy MIA/AIA を実行します。"
    )
    parser.add_argument(
        "--c",
        type=Path,
        default=REPO_ROOT / "local-output" / "opendp_aim" / "C.csv",
        help="攻撃する既存の C.csv",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=REPO_ROOT / "participant_data",
        help="participant_data ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "local-output" / "opendp_aim" / "toy-attack-v4",
        help="集計結果の出力先",
    )
    parser.add_argument(
        "--target-name",
        default="target",
        help="集計CSVで攻撃対象を識別する短い名前",
    )
    parser.add_argument("--seed", type=int, default=7, help="候補抽出・置換検定の乱数 seed")
    parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="MIA の偶然水準を見積もる(membership, rare)ペア置換の回数",
    )
    parser.add_argument(
        "--defense-config",
        type=Path,
        default=REPO_ROOT / "opendp_aim" / "config" / "pwscup_aim_v4.json",
        help="防御設定（ハッシュと来歴の記録だけに使用）",
    )
    parser.add_argument(
        "--privacy-report",
        type=Path,
        default=REPO_ROOT / "local-output" / "opendp_aim" / "privacy_report.json",
        help="privacy/postprocess lineage report（ハッシュと来歴の検証に使用）",
    )
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations は1以上にしてください")
    return args


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_privacy_lineage(
    c_path: Path,
    background_path: Path,
    defense_config_path: Path,
    privacy_report_path: Path,
) -> dict[str, Any]:
    """Fail closed unless C/config/A_bg match a release or postprocess report."""
    for path in (defense_config_path, privacy_report_path):
        if not path.exists():
            raise SystemExit(f"来歴検証ファイルが見つかりません: {path}")
    privacy_report = json.loads(privacy_report_path.read_text(encoding="utf-8"))
    is_postprocess = privacy_report.get("operation") == "public_outcome_donor_repostprocess"
    reported_background_hash = privacy_report.get(
        "background_sha256" if is_postprocess else "public_background_sha256"
    )
    checks: dict[str, tuple[Any, Any]] = {
        "output_sha256": (
            privacy_report.get("output_sha256"),
            sha256_file(c_path),
        ),
        "config_sha256": (
            privacy_report.get("config_sha256"),
            sha256_file(defense_config_path),
        ),
        "public_background_sha256": (
            reported_background_hash,
            sha256_file(background_path),
        ),
    }
    parent_report: dict[str, Any] | None = None
    if is_postprocess:
        required_postprocess_fields = (
            "output_sha256",
            "config_sha256",
            "background_sha256",
            "parent_input",
            "parent_input_sha256",
            "parent_privacy_report",
            "parent_privacy_report_sha256",
            "parent_privacy",
            "parent_dp_release_count",
        )
        missing_fields = [
            name for name in required_postprocess_fields if privacy_report.get(name) is None
        ]
        if missing_fields:
            raise SystemExit(
                "postprocess lineage report の必須項目がありません: "
                + ", ".join(missing_fields)
            )
        parent_input_path = Path(str(privacy_report["parent_input"]))
        parent_report_path = Path(str(privacy_report["parent_privacy_report"]))
        for label, path in (
            ("parent_input", parent_input_path),
            ("parent_privacy_report", parent_report_path),
        ):
            if not path.is_file():
                raise SystemExit(f"postprocess lineage の {label} が見つかりません: {path}")
        checks.update(
            {
                "parent_input_sha256": (
                    privacy_report.get("parent_input_sha256"),
                    sha256_file(parent_input_path),
                ),
                "parent_privacy_report_sha256": (
                    privacy_report.get("parent_privacy_report_sha256"),
                    sha256_file(parent_report_path),
                ),
            }
        )
        parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
        if parent_report.get("output_sha256") is None:
            raise SystemExit("parent privacy report に output_sha256 がありません")
        checks["parent_output_chain"] = (
            privacy_report.get("parent_input_sha256"),
            parent_report.get("output_sha256"),
        )
    mismatches = [name for name, (reported, actual) in checks.items() if reported != actual]
    if mismatches:
        raise SystemExit(
            "privacy report と攻撃対象の来歴が一致しません: " + ", ".join(mismatches)
        )
    if is_postprocess:
        parent_privacy = privacy_report.get("parent_privacy") or {}
        config_version = privacy_report.get("config_version")
        dp_release_count = privacy_report.get("parent_dp_release_count")
        rho_zcdp = parent_privacy.get("rho_zcdp")
    else:
        config_version = privacy_report.get("config_version")
        dp_release_count = privacy_report.get("dp_release_count")
        rho_zcdp = privacy_report.get("privacy", {}).get("rho_zcdp")
    return {
        "validated": True,
        "lineage_report_type": "public_postprocess" if is_postprocess else "dp_release",
        "config_version": config_version,
        "dp_release_count": dp_release_count,
        "rho_zcdp": rho_zcdp,
        "checks": {name: True for name in checks},
    }


def clean_json(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [clean_json(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def load_rare_detector(dist: Path) -> dict[str, Any]:
    schema_path = dist / "schema.json"
    schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
    detector = schema_obj.get("rare_detector")
    if not isinstance(detector, dict):
        raise SystemExit(f"rare_detector がありません: {schema_path}")
    return detector


def public_rare_mask(frame: pd.DataFrame, detector: dict[str, Any]) -> np.ndarray:
    columns = list(detector.get("columns") or common_schema.CONTINUOUS)
    distance = outlier_mod.mahalanobis(
        frame,
        np.asarray(detector["mu"], dtype=float),
        np.asarray(detector["sigma"], dtype=float),
        columns,
    )
    return distance >= float(detector["band"]["lo"])


def validate_inputs(c: pd.DataFrame, b: pd.DataFrame, cfg: dict[str, Any]) -> None:
    result = submission_schema.validate_c_submission(c, cfg, n_expected=len(b))
    if not result.ok:
        joined = "\n  - ".join(result.errors)
        raise SystemExit(f"C.csv の検証に失敗しました:\n  - {joined}")
    numeric = c.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SystemExit("C.csv の数値列に NaN または無限大があります")


def nn_mia_confidence(c: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    """Notebook §7.2's 1-NN distance MIA, reproduced verbatim in substance."""
    columns = list(common_schema.CONTINUOUS)
    mean = candidates[columns].mean().to_numpy(dtype=float)
    std = candidates[columns].std(ddof=0).to_numpy(dtype=float)
    std = np.where(std > 0.0, std, 1.0)
    c_scaled = (c[columns].to_numpy(dtype=float) - mean) / std
    q_scaled = (candidates[columns].to_numpy(dtype=float) - mean) / std
    nn = NearestNeighbors(n_neighbors=1).fit(c_scaled)
    distance, _ = nn.kneighbors(q_scaled)
    return 1.0 / (1.0 + distance.ravel())


def joint_pair_chance_distribution(
    confidence: np.ndarray,
    truth: np.ndarray,
    rare_mask: np.ndarray,
    cfg: dict[str, Any],
    permutations: int,
    seed: int,
) -> np.ndarray:
    """Permute (membership, rare) label pairs together across confidence ranks."""
    rng = np.random.default_rng(seed)
    values = np.empty(permutations, dtype=float)
    for iteration in range(permutations):
        order = rng.permutation(len(truth))
        values[iteration] = float(
            score_mia(
                confidence,
                truth[order],
                cfg,
                rare_mask=rare_mask[order],
            ).score
        )
    return values


def mia_record(
    defense: str,
    method: str,
    confidence: np.ndarray,
    truth: np.ndarray,
    rare_mask: np.ndarray,
    rare_mask_sensitivity: np.ndarray,
    cfg: dict[str, Any],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    result = score_mia(confidence, truth, cfg, rare_mask=rare_mask)
    breakdown = result.breakdown
    beta_worst = float(result.score)
    low_fpr = float(breakdown["low_fpr"])
    chance = joint_pair_chance_distribution(
        confidence, truth, rare_mask, cfg, permutations, seed
    )
    chance_95 = float(np.quantile(chance, 0.95))
    # Local analogue of the rulebook chance correction.  The official value uses
    # organiser cohorts, attackers, and 20,000 permutations, so this remains toy.
    beta_adjusted = float(max(beta_worst - chance_95 + low_fpr, low_fpr))
    fold_k = float(cfg["scoring"]["mia_fold_K"])
    toy_a_mia_adjusted = float(
        np.clip(
            (beta_adjusted / low_fpr - 1.0) / max(fold_k - 1.0, 1e-12),
            0.0,
            1.0,
        )
    )
    toy_a_mia_raw = float(
        np.clip((beta_worst / low_fpr - 1.0) / max(fold_k - 1.0, 1e-12), 0.0, 1.0)
    )
    sensitivity = score_mia(
        confidence, truth, cfg, rare_mask=rare_mask_sensitivity
    )
    members = truth.astype(bool)
    return {
        "defense": defense,
        "method": method,
        "n_candidates": int(len(truth)),
        "n_members": int(members.sum()),
        "n_nonmembers": int((~members).sum()),
        "n_rare_candidates": int(rare_mask.sum()),
        "n_rare_members": int((rare_mask & members).sum()),
        "beta_all": float(breakdown["tpr_at_low_fpr"]),
        "beta_rare": float(breakdown["tpr_rare"]),
        "beta_worst": beta_worst,
        "beta_rare_bself_member_sensitivity": float(
            sensitivity.breakdown["tpr_rare"]
        ),
        "beta_worst_bself_member_sensitivity": float(sensitivity.score),
        "auc": float(breakdown["auc"]),
        "chance_median": float(np.median(chance)),
        "chance_95": chance_95,
        "chance_99": float(np.quantile(chance, 0.99)),
        "beta_adjusted_local": beta_adjusted,
        "toy_a_mia_adjusted_local": toy_a_mia_adjusted,
        "toy_a_mia_raw_no_chance_correction": toy_a_mia_raw,
        "above_chance_95": bool(beta_worst > chance_95),
        "confidence_member_mean": float(confidence[members].mean()),
        "confidence_nonmember_mean": float(confidence[~members].mean()),
    }


def build_qi_matched_controls(
    members: pd.DataFrame,
    background: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one public control per member, matched on the AIA-visible QIs.

    Matching is exact on the configured categorical block and minimum-cost on the
    five standardized continuous QIs.  A control is unique within a block whenever
    possible; only blocks with more members than public controls require reuse.
    """
    continuous = list(cfg["aia"]["qi_continuous"])
    block_keys = list(cfg["aia"]["block_keys"])
    scale = background[continuous].std(ddof=0).to_numpy(dtype=float)
    scale = np.where(scale > 0.0, scale, 1.0)
    member_x = members[continuous].to_numpy(dtype=float) / scale
    background_x = background[continuous].to_numpy(dtype=float) / scale

    chosen = np.full(len(members), -1, dtype=int)
    matched_distance = np.full(len(members), np.nan, dtype=float)
    reused = np.zeros(len(members), dtype=bool)
    fallback_counts = {"exact_block": 0, "sex_fallback": 0, "global_fallback": 0}

    member_groups = members.groupby(block_keys, sort=False).indices
    background_groups = background.groupby(block_keys, sort=False).indices
    sex_key = block_keys[0]
    background_sex_groups = background.groupby([sex_key], sort=False).indices

    for raw_key, raw_member_indices in member_groups.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        member_indices = np.asarray(raw_member_indices, dtype=int)
        available = background_groups.get(key)
        fallback = "exact_block"
        if available is None or len(available) == 0:
            available = background_sex_groups.get((key[0],))
            fallback = "sex_fallback"
        if available is None or len(available) == 0:
            available = np.arange(len(background), dtype=int)
            fallback = "global_fallback"
        available = np.asarray(available, dtype=int)

        diff = member_x[member_indices, None, :] - background_x[available][None, :, :]
        cost = np.sqrt(np.mean(diff * diff, axis=2))
        row_indices, column_indices = linear_sum_assignment(cost)
        chosen[member_indices[row_indices]] = available[column_indices]
        matched_distance[member_indices[row_indices]] = cost[row_indices, column_indices]
        fallback_counts[fallback] += int(len(row_indices))

        assigned_rows = set(int(v) for v in row_indices)
        for local_row in range(len(member_indices)):
            if local_row in assigned_rows:
                continue
            nearest_column = int(np.argmin(cost[local_row]))
            global_row = int(member_indices[local_row])
            chosen[global_row] = int(available[nearest_column])
            matched_distance[global_row] = float(cost[local_row, nearest_column])
            reused[global_row] = True
            fallback_counts[fallback] += 1

    if (chosen < 0).any() or not np.isfinite(matched_distance).all():
        raise RuntimeError("AIA control matching did not assign every member")

    controls = background.iloc[chosen].reset_index(drop=True)
    metadata = {
        "n_controls": int(len(controls)),
        "n_unique_background_rows": int(np.unique(chosen).size),
        "n_reused_assignments": int(reused.sum()),
        "mean_qi_rms_distance": float(matched_distance.mean()),
        "median_qi_rms_distance": float(np.median(matched_distance)),
        "p95_qi_rms_distance": float(np.quantile(matched_distance, 0.95)),
        "fallback_counts": fallback_counts,
    }
    return controls, metadata


def aia_predictions_record(
    defense: str,
    attack_source: str,
    member_onset_conf: np.ndarray | None,
    member_time_hat: np.ndarray,
    control_onset_conf: np.ndarray | None,
    control_time_hat: np.ndarray,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_rare: np.ndarray,
    control_rare: np.ndarray,
    member_rare_public: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    result = score_aia(
        member_onset_conf,
        member_time_hat,
        members["onset"].to_numpy(dtype=int),
        members["time"].to_numpy(dtype=float),
        control_onset_conf,
        control_time_hat,
        controls["onset"].to_numpy(dtype=int),
        controls["time"].to_numpy(dtype=float),
        cfg,
        member_rare_mask=member_rare,
        control_rare_mask=control_rare,
    )
    public_rare_sensitivity = score_aia(
        member_onset_conf,
        member_time_hat,
        members["onset"].to_numpy(dtype=int),
        members["time"].to_numpy(dtype=float),
        control_onset_conf,
        control_time_hat,
        controls["onset"].to_numpy(dtype=int),
        controls["time"].to_numpy(dtype=float),
        cfg,
        member_rare_mask=member_rare_public,
        control_rare_mask=control_rare,
    )
    breakdown = result.breakdown
    member_error = np.abs(member_time_hat - members["time"].to_numpy(dtype=float))
    control_error = np.abs(control_time_hat - controls["time"].to_numpy(dtype=float))
    tau = float(breakdown["tau_time"])
    member_success = member_error <= tau
    control_success = control_error <= tau
    p_member = float(member_success.mean())
    p_control = float(control_success.mean())
    rate_difference = p_member - p_control
    rate_difference_se = float(
        np.sqrt(
            p_member * (1.0 - p_member) / len(member_success)
            + p_control * (1.0 - p_control) / len(control_success)
        )
    )

    def masked_rate(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    return {
        "defense": defense,
        "attack_source": attack_source,
        "n_members": int(len(members)),
        "n_controls": int(len(controls)),
        "n_rare_members": int(member_rare.sum()),
        "n_rare_controls": int(control_rare.sum()),
        "n_rare_members_public_detector": int(member_rare_public.sum()),
        "aia_score_worst": float(result.score),
        "aia_score_worst_public_both_sensitivity": float(
            public_rare_sensitivity.score
        ),
        "r_time_all": float(breakdown["r_time_all"]),
        "r_time_rare": float(breakdown["r_time_rare"]),
        "r_time_rare_raw": float(breakdown["r_time_rare_raw"]),
        "r_time_rare_public_both_sensitivity": float(
            public_rare_sensitivity.breakdown["r_time_rare"]
        ),
        "r_time_rare_raw_public_both_sensitivity": float(
            public_rare_sensitivity.breakdown["r_time_rare_raw"]
        ),
        "r_onset_diagnostic": float(breakdown["r_onset"]),
        "p_time_member": p_member,
        "p_time_control": p_control,
        "p_time_difference": rate_difference,
        "p_time_difference_se_approx": rate_difference_se,
        "p_time_difference_ci95_low_approx": rate_difference - 1.96 * rate_difference_se,
        "p_time_difference_ci95_high_approx": rate_difference + 1.96 * rate_difference_se,
        "p_time_member_rare": masked_rate(member_success, member_rare),
        "p_time_control_rare": masked_rate(control_success, control_rare),
        "tau_time_years": tau,
        "member_time_mae_years": float(member_error.mean()),
        "control_time_mae_years": float(control_error.mean()),
        "member_time_median_abs_error_years": float(np.median(member_error)),
        "control_time_median_abs_error_years": float(np.median(control_error)),
    }


def aia_record(
    defense: str,
    c: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    background: pd.DataFrame,
    member_rare: np.ndarray,
    control_rare: np.ndarray,
    member_rare_public: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    member_onset_conf, member_time_hat = aia_mod.aia_predict(c, members, background, cfg)
    control_onset_conf, control_time_hat = aia_mod.aia_predict(c, controls, background, cfg)
    return aia_predictions_record(
        defense,
        "distributed_reference_algorithm_qi_nearest_neighbor",
        member_onset_conf,
        member_time_hat,
        control_onset_conf,
        control_time_hat,
        members,
        controls,
        member_rare,
        control_rare,
        member_rare_public,
        cfg,
    )


def constant_time_aia_baseline(
    members: pd.DataFrame,
    controls: pd.DataFrame,
    member_rare: np.ndarray,
    control_rare: np.ndarray,
    member_rare_public: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    constant = float(cfg["onset"]["horizon_years"])
    return aia_predictions_record(
        "no_C_baseline_time10",
        "constant_public_horizon_without_C",
        None,
        np.full(len(members), constant, dtype=float),
        None,
        np.full(len(controls), constant, dtype=float),
        members,
        controls,
        member_rare,
        control_rare,
        member_rare_public,
        cfg,
    )


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    mia_rows: list[dict[str, Any]],
    aia_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_report = clean_json(report)
    (output_dir / "toy_attack_report.json").write_text(
        json.dumps(clean_report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(mia_rows).to_csv(output_dir / "mia_summary.csv", index=False)
    pd.DataFrame(aia_rows).to_csv(output_dir / "aia_summary.csv", index=False)


def print_summary(
    c_path: Path,
    output_dir: Path,
    mia_rows: list[dict[str, Any]],
    aia_rows: list[dict[str, Any]],
) -> None:
    print("=" * 72)
    print("PWS Cup 2026 toy self-attack (unofficial)")
    print("=" * 72)
    print(f"target C : {c_path}")
    print("\nMIA: beta=TPR@FPR 1%; worst=max(all, rare)")
    print(
        pd.DataFrame(mia_rows)[
            [
                "defense",
                "method",
                "beta_all",
                "beta_rare",
                "beta_worst",
                "auc",
                "chance_95",
                "beta_adjusted_local",
                "toy_a_mia_adjusted_local",
                "above_chance_95",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nAIA: time recovery within +/-0.5 years; control-adjusted R")
    print(
        pd.DataFrame(aia_rows)[
            [
                "defense",
                "p_time_member",
                "p_time_control",
                "p_time_difference",
                "r_time_all",
                "r_time_rare",
                "aia_score_worst",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nraw_B_sanity が強く破られることは、toy 攻撃器の positive control です。")
    print("この値は organizer-only truth を使う公式保護スコアではありません。")
    print(f"results  : {output_dir}")


def main() -> None:
    args = parse_args()
    if not args.target_name or args.target_name == "raw_B_sanity":
        raise SystemExit("--target-name は空または raw_B_sanity にできません")
    c_path = args.c.resolve()
    dist = args.dist.resolve()
    output_dir = args.output_dir.resolve()
    defense_config_path = args.defense_config.resolve()
    privacy_report_path = args.privacy_report.resolve()
    if not c_path.exists():
        raise SystemExit(f"C.csv が見つかりません: {c_path}")
    for required in ("B_practice.csv", "B_self.csv", "A_bg.csv", "schema.json"):
        if not (dist / required).exists():
            raise SystemExit(f"配布ファイルが見つかりません: {dist / required}")

    cfg = _common.load_cfg()
    c = pd.read_csv(c_path).reset_index(drop=True)
    members = pd.read_csv(dist / "B_practice.csv").reset_index(drop=True)
    b_self = pd.read_csv(dist / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(dist / "A_bg.csv").reset_index(drop=True)
    validate_inputs(c, members, cfg)
    lineage = validate_privacy_lineage(
        c_path,
        dist / "A_bg.csv",
        defense_config_path,
        privacy_report_path,
    )
    if len(b_self) != len(members) or "is_rare" not in b_self.columns:
        raise SystemExit("B_self.csv の行数または is_rare 列が不正です")
    if not members["record_id"].equals(b_self["record_id"]):
        raise SystemExit("B_practice.csv と B_self.csv の record_id 順序が一致しません")
    distributed_columns = list(submission_schema.DISTRIBUTED_COLUMNS)
    if not members[distributed_columns].equals(b_self[distributed_columns]):
        raise SystemExit("B_practice.csv と B_self.csv の共有列が一致しません")

    detector = load_rare_detector(dist)
    member_rare = b_self["is_rare"].astype(bool).to_numpy()
    member_rare_public = public_rare_mask(members, detector)
    raw_b = members[submission_schema.SUBMITTED_C_COLUMNS].copy().reset_index(drop=True)

    # Match the public scoring notebook's candidate construction exactly.
    nonmember_indices = background.sample(
        n=len(members), replace=False, random_state=args.seed
    ).index.to_numpy(dtype=int)
    nonmembers = background.iloc[nonmember_indices].reset_index(drop=True)
    reference_background = background.drop(index=nonmember_indices).reset_index(drop=True)
    candidates = pd.concat([members, nonmembers], ignore_index=True)
    truth = np.concatenate(
        [np.ones(len(members), dtype=int), np.zeros(len(nonmembers), dtype=int)]
    )
    nonmember_rare_public = public_rare_mask(nonmembers, detector)
    candidate_rare = np.concatenate([member_rare_public, nonmember_rare_public])
    candidate_rare_sensitivity = np.concatenate([member_rare, nonmember_rare_public])

    defenses = {args.target_name: c, "raw_B_sanity": raw_b}
    mia_rows: list[dict[str, Any]] = []
    for defense_index, (defense_name, defense_frame) in enumerate(defenses.items()):
        confidences = {
            "1nn_notebook": nn_mia_confidence(defense_frame, candidates),
            "domias_kde_reference_full_abg": mia_mod.overlap_mia(
                defense_frame, background, candidates, cfg
            ),
            "domias_kde_crossfit": mia_mod.overlap_mia(
                defense_frame, reference_background, candidates, cfg
            ),
        }
        for method_index, (method_name, confidence) in enumerate(confidences.items()):
            mia_rows.append(
                mia_record(
                    defense_name,
                    method_name,
                    confidence,
                    truth,
                    candidate_rare,
                    candidate_rare_sensitivity,
                    cfg,
                    args.permutations,
                    args.seed + 10_000 * defense_index + 1_000 * method_index,
                )
            )

    controls, matching = build_qi_matched_controls(members, background, cfg)
    control_rare = public_rare_mask(controls, detector)
    aia_rows = [
        aia_record(
            defense_name,
            defense_frame,
            members,
            controls,
            background,
            member_rare,
            control_rare,
            member_rare_public,
            cfg,
        )
        for defense_name, defense_frame in defenses.items()
    ]
    aia_rows.insert(
        1,
        constant_time_aia_baseline(
            members,
            controls,
            member_rare,
            control_rare,
            member_rare_public,
            cfg,
        ),
    )

    report = {
        "report_type": "unofficial_toy_self_attack",
        "report_version": 2,
        "target_label": lineage.get("config_version") or "v4",
        "privacy_lineage": lineage,
        "inputs": {
            "c_path": str(c_path),
            "c_sha256": sha256_file(c_path),
            "b_practice_sha256": sha256_file(dist / "B_practice.csv"),
            "b_self_sha256": sha256_file(dist / "B_self.csv"),
            "a_bg_sha256": sha256_file(dist / "A_bg.csv"),
            "schema_sha256": sha256_file(dist / "schema.json"),
            "kit_config_sha256": sha256_file(Path(_common.KIT_CONFIG_PATH)),
            "toy_attack_script_sha256": sha256_file(Path(__file__).resolve()),
            "starter_uv_lock_sha256": sha256_file(REPO_ROOT / "starter" / "uv.lock"),
            "defense_config_path": str(defense_config_path),
            "defense_config_sha256": sha256_file(defense_config_path),
            "privacy_report_path": str(privacy_report_path),
            "privacy_report_sha256": sha256_file(privacy_report_path),
        },
        "parameters": {
            "seed": args.seed,
            "mia_permutations": args.permutations,
            "mia_low_fpr": cfg["scoring"]["low_fpr"],
            "mia_fold_k": cfg["scoring"]["mia_fold_K"],
            "mia_chance_method": "joint (membership, rare) pair permutation",
            "mia_rare_primary": "public Mahalanobis detector on members and nonmembers",
            "mia_rare_sensitivity": "B_self truth on members; public detector on nonmembers",
            "aia_tau_time_years": cfg["aia"]["tau_time"],
            "aia_rare_se_z": cfg["scoring"]["aia_rare_se_z"],
            "aia_qi_continuous": cfg["aia"]["qi_continuous"],
            "aia_block_keys": cfg["aia"]["block_keys"],
            "aia_k_nn_onset": cfg["aia"]["k_nn_onset"],
            "rare_detector_columns": detector.get("columns"),
            "rare_detector_lo": detector["band"]["lo"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": distribution_version("numpy"),
            "pandas": distribution_version("pandas"),
            "scipy": distribution_version("scipy"),
            "scikit_learn": distribution_version("scikit-learn"),
        },
        "cohorts": {
            "members": len(members),
            "nonmembers": len(nonmembers),
            "reference_background_full": len(background),
            "reference_background_crossfit": len(reference_background),
            "member_rare_truth": int(member_rare.sum()),
            "member_rare_public_detector": int(member_rare_public.sum()),
            "member_rare_truth_public_intersection": int(
                (member_rare & member_rare_public).sum()
            ),
            "candidate_rare_total": int(candidate_rare.sum()),
            "control_rare_public_detector": int(control_rare.sum()),
        },
        "control_matching": matching,
        "mia": mia_rows,
        "aia": aia_rows,
        "interpretation_limits": [
            "This reuses the existing C.csv and performs no new DP release.",
            "The organiser-only overlap membership truth is unavailable; MIA uses all B rows as members and an equal-size fixed A_bg sample as nonmembers.",
            "The organiser-only AIA truth is unavailable; AIA uses all B rows as members and public A_bg rows matched on visible QIs as controls.",
            "The 1-NN MIA is the notebook self-diagnostic; both DOMIAS variants and AIA call the distributed official reference attack implementations.",
            "The reference DOMIAS variant uses full A_bg like run_mia.py; the cross-fit sensitivity variant removes toy nonmembers from its density reference.",
            "MIA chance quantiles and beta_adjusted_local use 2,000 local joint label-pair permutations, not the official 20,000-permutation cross-team correction.",
            "MIA primary rare masks apply the public detector symmetrically; B_self-member sensitivity fields show the alternative asymmetric definition.",
            "AIA primary rare masks use B_self truth for members and the public detector for controls; public-both sensitivity fields quantify this definition choice.",
            "AIA controls come from A_bg, which the target defense also uses for decoding and outcome donors; this can raise control success and bias the target R downward.",
            "The no_C_baseline_time10 row uses no C and exposes finite-cohort/base-rate differences in this toy construction.",
            "AIA rate-difference confidence intervals are approximate independent-binomial diagnostics and do not model control matching or control reuse.",
            "raw_B_sanity is a positive control for attack sensitivity, not a candidate defense.",
            "Therefore none of these values is an official PWS Cup protection or attack leaderboard score.",
        ],
    }
    write_outputs(output_dir, report, mia_rows, aia_rows)
    print_summary(c_path, output_dir, mia_rows, aia_rows)


if __name__ == "__main__":
    main()
