"""Evaluate local MIA/AIA resistance with official and team-inspired attacks.

This is a defense-development stress test, not an official Anon score.  It uses
the official main-round scoring primitives, the official reference attacks,
and multi-view linkage attacks adapted from the team's preliminary v4/v9 work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors


MIA_VIEWS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("qi", ("age", "SBP", "TG", "HDL", "ALT"), ("TG", "ALT")),
    ("stable", ("age", "SBP", "HDL"), ()),
    ("clinical", ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"), ("TG", "ALT", "FPG")),
    (
        "outcome",
        ("BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "smoking", "onset", "death"),
        ("TG", "ALT", "FPG"),
    ),
)
AIA_VIEWS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("qi", ("age", "SBP", "TG", "HDL", "ALT"), ("TG", "ALT")),
    ("stable", ("age", "SBP", "HDL"), ()),
    ("clinical", ("BMI", "SBP", "TG", "HDL", "ALT", "FPG"), ("TG", "ALT", "FPG")),
)
BLOCK_KEYS = ("sex", "prefecture")
MIA_CHANCE_PERMUTATIONS = 20_000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_utility(path: Path) -> dict[str, Any]:
    """Load either a standalone score.json or an R-series report.json."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    utility = payload.get("score", payload)
    if "utility" not in utility or "facets" not in utility:
        raise ValueError(f"Utility score not found in {path}")
    return utility


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def rare_mask(frame: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    detector = schema["rare_detector"]
    values = frame[detector["columns"]].to_numpy(dtype=float)
    diff = values - np.asarray(detector["mu"], dtype=float)
    inv_sigma = np.linalg.pinv(np.asarray(detector["sigma"], dtype=float))
    distance = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv_sigma, diff), 0.0, None))
    return distance >= float(detector["band"]["lo"])


def transform_view(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    log_columns: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=float)
    for idx, name in enumerate(columns):
        if name in log_columns:
            values[:, idx] = np.log(np.clip(values[:, idx], 1e-6, None))
    return (values - center) / scale


def fit_view(background: pd.DataFrame, columns: tuple[str, ...], log_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    raw = background[list(columns)].to_numpy(dtype=float)
    for idx, name in enumerate(columns):
        if name in log_columns:
            raw[:, idx] = np.log(np.clip(raw[:, idx], 1e-6, None))
    center = np.median(raw, axis=0)
    q25, q75 = np.quantile(raw, [0.25, 0.75], axis=0)
    scale = q75 - q25
    std = raw.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return center, scale


def grouped_nn(
    released: pd.DataFrame,
    released_x: np.ndarray,
    query: pd.DataFrame,
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest row with (sex,prefecture), then sex, then global fallback."""
    distances = np.empty(len(query), dtype=float)
    nearest = np.empty(len(query), dtype=int)
    full = np.arange(len(released))
    pair_groups = {
        tuple(key) if isinstance(key, tuple) else (key,): np.asarray(index, dtype=int)
        for key, index in released.groupby(list(BLOCK_KEYS), sort=False).groups.items()
    }
    sex_groups = {
        (key if isinstance(key, tuple) else (key,)): np.asarray(index, dtype=int)
        for key, index in released.groupby([BLOCK_KEYS[0]], sort=False).groups.items()
    }
    for row, block in enumerate(query[list(BLOCK_KEYS)].itertuples(index=False, name=None)):
        candidates = pair_groups.get(tuple(block))
        if candidates is None or not len(candidates):
            candidates = sex_groups.get((block[0],))
        if candidates is None or not len(candidates):
            candidates = full
        delta = released_x[candidates] - query_x[row]
        distance = np.sqrt(np.mean(delta * delta, axis=1))
        best = int(np.argmin(distance))
        distances[row] = float(distance[best])
        nearest[row] = int(candidates[best])
    return distances, nearest


def empirical_lower_tail(values: np.ndarray, null_values: np.ndarray) -> np.ndarray:
    null = np.sort(np.asarray(null_values, dtype=float))
    counts = np.searchsorted(null, np.asarray(values, dtype=float), side="right")
    return (counts + 1.0) / (len(null) + 1.0)


def team_multiview_mia(
    released: pd.DataFrame,
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    control_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    evidence: list[np.ndarray] = []
    nearest_rows: list[np.ndarray] = []
    view_summary: dict[str, Any] = {}
    for name, columns, logs in MIA_VIEWS:
        center, scale = fit_view(background, columns, logs)
        released_x = transform_view(released, columns, logs, center, scale)
        candidate_x = transform_view(candidates, columns, logs, center, scale)
        distance, nearest = grouped_nn(released, released_x, candidates, candidate_x)
        p_value = empirical_lower_tail(distance, distance[control_mask])
        evidence.append(-np.log(p_value))
        nearest_rows.append(nearest)
        view_summary[name] = {
            "member_distance_median": float(np.median(distance[~control_mask])),
            "control_distance_median": float(np.median(distance[control_mask])),
        }
    matrix = np.column_stack(evidence)
    sorted_evidence = np.sort(matrix, axis=1)
    robust = 0.55 * sorted_evidence[:, -2:].mean(axis=1) + 0.45 * np.median(matrix, axis=1)
    nearest_matrix = np.column_stack(nearest_rows)
    coherence = np.asarray(
        [np.unique(row, return_counts=True)[1].max() / len(row) for row in nearest_matrix], dtype=float
    )
    combined = robust + 0.25 * coherence
    # Strict ranks are stable and sufficient because official MIA scoring is rank based.
    order = np.lexsort((np.arange(len(combined)), combined))
    rank = np.empty(len(combined), dtype=float)
    rank[order] = np.arange(1, len(combined) + 1, dtype=float)
    confidence = rank / (len(combined) + 1.0)
    return confidence, {"views": view_summary, "coherence_mean": float(coherence.mean())}


def ordered_tpr_at_fpr(confidence: np.ndarray, truth: np.ndarray, target_fpr: float) -> float:
    order = np.argsort(-confidence, kind="mergesort")
    score = confidence[order]
    label = truth[order].astype(int)
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if not n_pos or not n_neg:
        return 0.0
    endpoints = np.r_[np.flatnonzero(np.diff(score) != 0), len(score) - 1]
    tp = np.cumsum(label)[endpoints] / n_pos
    fp = np.cumsum(1 - label)[endpoints] / n_neg
    return float(np.interp(target_fpr, np.r_[0.0, fp], np.r_[0.0, tp]))


def mia_raw(confidence: np.ndarray, truth: np.ndarray, rare: np.ndarray, low_fpr: float) -> tuple[float, float, float]:
    overall = ordered_tpr_at_fpr(confidence, truth, low_fpr)
    rare_value = float("nan")
    if rare.any() and int(truth[rare].sum()) and int((1 - truth[rare]).sum()):
        rare_value = ordered_tpr_at_fpr(confidence[rare], truth[rare], low_fpr)
    return max(overall, rare_value) if math.isfinite(rare_value) else overall, overall, rare_value


def chance_correct_mia(
    confidence: np.ndarray,
    truth: np.ndarray,
    rare: np.ndarray,
    low_fpr: float,
    fold_k: float,
) -> dict[str, Any]:
    raw, overall, rare_value = mia_raw(confidence, truth, rare, low_fpr)
    seed_material = np.column_stack((confidence, truth, rare.astype(int))).tobytes()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    pairs = np.column_stack((truth, rare.astype(int)))
    chance = np.empty(MIA_CHANCE_PERMUTATIONS, dtype=float)
    for index in range(MIA_CHANCE_PERMUTATIONS):
        shuffled = pairs[rng.permutation(len(pairs))]
        chance[index] = mia_raw(
            confidence, shuffled[:, 0].astype(int), shuffled[:, 1].astype(bool), low_fpr
        )[0]
    chance95 = float(np.quantile(chance, 0.95))
    beta = max(raw - chance95 + low_fpr, low_fpr)
    normalized = float(np.clip((beta / low_fpr - 1.0) / (fold_k - 1.0), 0.0, 1.0))
    return {
        "raw_worst_tpr": raw,
        "tpr_all": overall,
        "tpr_rare": rare_value,
        "chance95": chance95,
        "beta": beta,
        "normalized_risk": normalized,
        "permutations": MIA_CHANCE_PERMUTATIONS,
        "seed": seed,
    }


def evaluate_mia(
    released: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    background: pd.DataFrame,
    schema: dict[str, Any],
    cfg: dict[str, Any],
    overlap_mia: Any,
) -> dict[str, Any]:
    pool = pd.concat([members, controls], ignore_index=True)
    truth = np.r_[np.ones(len(members), dtype=int), np.zeros(len(controls), dtype=int)]
    detected_rare = rare_mask(pool, schema)
    control_mask = truth == 0
    attacks: dict[str, Any] = {}
    official_confidence = overlap_mia(released, background, pool, cfg)
    attacks["official_domias_kde"] = chance_correct_mia(
        official_confidence, truth, detected_rare, float(cfg["scoring"]["low_fpr"]), float(cfg["scoring"]["mia_fold_K"])
    )
    team_confidence, team_detail = team_multiview_mia(released, pool, background, control_mask)
    attacks["team_v4_v9_multiview"] = {
        **chance_correct_mia(
            team_confidence, truth, detected_rare, float(cfg["scoring"]["low_fpr"]), float(cfg["scoring"]["mia_fold_K"])
        ),
        "detail": team_detail,
    }
    risk = max(float(item["normalized_risk"]) for item in attacks.values())
    beta_risk = max(float(item["beta"]) for item in attacks.values())
    raw_risk = max(float(item["raw_worst_tpr"]) for item in attacks.values())
    worst_attack = max(attacks, key=lambda name: float(attacks[name]["normalized_risk"]))
    return {
        "candidate_proxy": "all 1049 B rows as members and all public A_bg rows as nonmembers",
        "candidate_rows": len(pool),
        "members": len(members),
        "controls": len(controls),
        "attacks": attacks,
        "stress_normalized_risk": risk,
        "stress_beta_risk": beta_risk,
        "stress_raw_tpr": raw_risk,
        "worst_attack": worst_attack,
    }


def team_multiview_aia(
    released: pd.DataFrame,
    query: pd.DataFrame,
    background: pd.DataFrame,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    released_time = released["time"].to_numpy(dtype=float)
    for _name, columns, logs in AIA_VIEWS:
        center, scale = fit_view(background, columns, logs)
        released_x = transform_view(released, columns, logs, center, scale)
        query_x = transform_view(query, columns, logs, center, scale)
        _distance, nearest = grouped_nn(released, released_x, query, query_x)
        predictions.append(released_time[nearest])
    return np.median(np.column_stack(predictions), axis=1)


def summarize_aia_result(result: Any) -> dict[str, Any]:
    return {"risk": float(result.score), "breakdown": json_safe(result.breakdown)}


def evaluate_aia(
    released: pd.DataFrame,
    members: pd.DataFrame,
    controls: pd.DataFrame,
    background: pd.DataFrame,
    schema: dict[str, Any],
    cfg: dict[str, Any],
    aia_predict: Any,
    score_aia: Any,
) -> dict[str, Any]:
    member_rare = rare_mask(members, schema)
    control_rare = rare_mask(controls, schema)
    attacks: dict[str, Any] = {}

    _, member_official = aia_predict(released, members, background, cfg)
    _, control_official = aia_predict(released, controls, background, cfg)
    official_result = score_aia(
        None, member_official, members["onset"].to_numpy(), members["time"].to_numpy(),
        None, control_official, controls["onset"].to_numpy(), controls["time"].to_numpy(),
        cfg, member_rare_mask=member_rare, control_rare_mask=control_rare,
    )
    attacks["official_qi_nearest"] = summarize_aia_result(official_result)

    member_team = team_multiview_aia(released, members, background)
    control_team = team_multiview_aia(released, controls, background)
    team_result = score_aia(
        None, member_team, members["onset"].to_numpy(), members["time"].to_numpy(),
        None, control_team, controls["onset"].to_numpy(), controls["time"].to_numpy(),
        cfg, member_rare_mask=member_rare, control_rare_mask=control_rare,
    )
    attacks["team_multiview_linkage"] = summarize_aia_result(team_result)

    horizon_member = np.full(len(members), 10.0)
    horizon_control = np.full(len(controls), 10.0)
    horizon_result = score_aia(
        None, horizon_member, members["onset"].to_numpy(), members["time"].to_numpy(),
        None, horizon_control, controls["onset"].to_numpy(), controls["time"].to_numpy(),
        cfg, member_rare_mask=member_rare, control_rare_mask=control_rare,
    )
    attacks["public_horizon_sanity"] = summarize_aia_result(horizon_result)
    attack_risks = [float(attacks[name]["risk"]) for name in ("official_qi_nearest", "team_multiview_linkage")]
    return {
        "challenge_proxy": "all B rows as targets and all A_bg rows as controls",
        "members": len(members),
        "controls": len(controls),
        "phi_team10": None,
        "phi_note": "team10 official phi_i is unavailable; no phi subtraction is applied",
        "attacks": attacks,
        "stress_risk_before_phi": max(attack_risks),
    }


def exact_overlap_count(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int:
    keys = set(map(tuple, right[columns].itertuples(index=False, name=None)))
    return sum(tuple(row) in keys for row in left[columns].itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model label, e.g. M1")
    parser.add_argument("--c", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--kit-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    kit_source = args.kit_root / "starter" / "src"
    sys.path.insert(0, str(kit_source.resolve()))
    from pwscup2026.attack.aia import aia_predict
    from pwscup2026.attack.mia import overlap_mia
    from pwscup2026.scoring.attack_score import score_aia

    cfg = yaml.safe_load((args.kit_root / "participant_data" / "scoring_config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("aia", {}).update(
        {"qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"], "block_keys": list(BLOCK_KEYS), "k_nn_onset": 5}
    )
    schema_path = args.data / "schema.json"
    repo_root = Path(__file__).resolve().parents[6]
    prelim_v4_mia = repo_root / "prelim" / "workspace" / "attack_phase_v4" / "mia_private_conformal.py"
    prelim_v4_aia = repo_root / "prelim" / "workspace" / "attack_phase_v4" / "aia_outcome_integrity.py"
    prelim_v9_readme = repo_root / "prelim" / "workspace" / "attack_phase_v9" / "README.md"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    released = pd.read_csv(args.c)
    members = pd.read_csv(args.data / "B_self.csv").drop(columns=["is_rare"])
    controls = pd.read_csv(args.data / "A_bg.csv")
    background = controls.copy()
    utility = load_utility(args.score)

    mia = evaluate_mia(released, members, controls, background, schema, cfg, overlap_mia)
    aia = evaluate_aia(released, members, controls, background, schema, cfg, aia_predict, score_aia)
    official_like_attack_risk = max(
        float(mia["stress_normalized_risk"]), float(aia["stress_risk_before_phi"])
    )
    official_like_resistance = 1.0 - official_like_attack_risk
    diagnostic_attack_risk = max(
        float(mia["stress_beta_risk"]), float(aia["stress_risk_before_phi"])
    )
    diagnostic_resistance = 1.0 - diagnostic_attack_risk
    utility_value = float(utility["utility"])
    decision_floor = min(utility_value, diagnostic_resistance)
    if diagnostic_attack_risk <= 0.05:
        risk_band = "low-local-signal"
    elif diagnostic_attack_risk <= 0.15:
        risk_band = "watch"
    else:
        risk_band = "high-local-signal"
    submitted_columns = list(schema["submitted_c_columns"])

    report = {
        "schema_version": "team10-local-resistance-v1",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "scope": "local stress test; not official protection or final Anon",
        "rulebook_alignment": {
            "mia": "section 6.2.1: TPR at FPR=0.01, rare worst-case, 20,000-permutation chance correction, K=12",
            "aia": "section 6.2.2: normalized excess time accuracy with tau=0.5 and rare SE deadzone",
            "aggregation": "section 6.2: resistance=1-max(MIA,AIA); decision floor mirrors min(U,Anon)",
        },
        "limitations": [
            "Official cross-team MIA candidate sets and AIA target/control challenges are unavailable.",
            "The local proxy uses all B rows as members/targets and public A_bg rows as controls.",
            "The team10 official AIA phi_i baseline is unavailable, so the reported AIA risk is before phi subtraction.",
            "Multiple attacks are aggregated by maximum as a conservative development stress test, not by the official cross-team aggregation.",
            "K=12 normalization is reported for rulebook traceability, but the proxy candidate composition differs from the official 25-27-member pool and may saturate; model ranking therefore uses the chance-corrected beta scale before K normalization.",
            "Results compare models on this fixed local proxy and are not leaderboard predictions.",
        ],
        "inputs": {
            "c_sha256": digest(args.c),
            "utility_score_sha256": digest(args.score),
            "b_self_sha256": digest(args.data / "B_self.csv"),
            "a_bg_sha256": digest(args.data / "A_bg.csv"),
            "schema_sha256": digest(schema_path),
            "official_attack_mia_sha256": digest(kit_source / "pwscup2026" / "attack" / "mia.py"),
            "official_attack_aia_sha256": digest(kit_source / "pwscup2026" / "attack" / "aia.py"),
            "official_attack_score_sha256": digest(kit_source / "pwscup2026" / "scoring" / "attack_score.py"),
            "prelim_v4_mia_source_sha256": digest(prelim_v4_mia),
            "prelim_v4_aia_source_sha256": digest(prelim_v4_aia),
            "prelim_v9_method_record_sha256": digest(prelim_v9_readme),
        },
        "utility": {"U": utility_value, "facets": utility["facets"]},
        "mia": mia,
        "aia": aia,
        "copy_diagnostics": {
            "exact_B_rows_in_C": exact_overlap_count(members, released, submitted_columns),
            "exact_A_bg_rows_in_C": exact_overlap_count(controls, released, submitted_columns),
            "c_duplicate_rows": int(released.duplicated().sum()),
        },
        "summary": {
            "mia_stress_normalized_risk": float(mia["stress_normalized_risk"]),
            "mia_stress_beta_risk": float(mia["stress_beta_risk"]),
            "mia_stress_raw_tpr": float(mia["stress_raw_tpr"]),
            "aia_stress_risk_before_phi": float(aia["stress_risk_before_phi"]),
            "official_like_attack_risk": official_like_attack_risk,
            "official_like_resistance_index": official_like_resistance,
            "diagnostic_attack_risk": diagnostic_attack_risk,
            "diagnostic_resistance_index": diagnostic_resistance,
            "risk_band": risk_band,
            "utility_U": utility_value,
            "screening_floor": decision_floor,
            "bottleneck": "utility" if utility_value <= diagnostic_resistance else "resistance",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
