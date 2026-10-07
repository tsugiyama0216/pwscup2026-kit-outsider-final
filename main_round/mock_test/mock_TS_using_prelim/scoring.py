"""Strict replay of the preliminary-round attack scorer and released truths.

The public final results exclude self attacks.  The historical scorer did not,
so this adapter removes self predictions before calling the untouched scorer.
No main-round AIA floor is applied to this preliminary-round benchmark.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OFFICIAL_SOURCE = ROOT / "official-kit" / "starter" / "src"
COHORTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29)
PREDICTION_NAMES = ("F_mia.csv", "F_aia.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            header = next(csv.reader(stream))
        except StopIteration as exc:
            raise ValueError(f"{path.name}: empty CSV") from exc
    if not header or any(not c for c in header) or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: empty or duplicate column names")
    if columns is not None and header != columns:
        raise ValueError(f"{path.name}: columns must be {columns}, got {header}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if list(frame.columns) != header or frame.empty:
        raise ValueError(f"{path.name}: invalid or empty table")
    return frame


def _number(values: pd.Series, label: str, *, integer: bool = False,
            lower: float | None = None, upper: float | None = None) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: values must be numeric and finite")
    if integer and ((result != np.floor(result)).any() or (np.abs(result) >= 2**53).any()):
        raise ValueError(f"{label}: IDs must be exact integers")
    if lower is not None and (result < lower).any():
        raise ValueError(f"{label}: values must be >= {lower}")
    if upper is not None and (result > upper).any():
        raise ValueError(f"{label}: values must be <= {upper}")
    return result.astype(np.int64) if integer else result


def _same_ids(actual: pd.Series, expected: pd.Series, label: str) -> None:
    if actual.duplicated().any():
        raise ValueError(f"{label}: duplicate IDs")
    missing = set(expected.tolist()) - set(actual.tolist())
    extra = set(actual.tolist()) - set(expected.tolist())
    if missing or extra or len(actual) != len(expected):
        raise ValueError(f"{label}: ID set mismatch; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")


def _load_official():
    expected = (OFFICIAL_SOURCE / "pwscup2026").resolve()
    if not (expected / "kit" / "codabench" / "scoring_io.py").is_file():
        raise FileNotFoundError(f"Copied preliminary official scorer is missing: {expected}")
    for name, module in list(sys.modules.items()):
        if name == "pwscup2026" or name.startswith("pwscup2026."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(expected):
                raise RuntimeError("Another pwscup2026 snapshot is already imported; run this preliminary scorer in a fresh Python process")
    source_path = str(OFFICIAL_SOURCE.resolve())
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    scoring_io = importlib.import_module("pwscup2026.kit.codabench.scoring_io")
    attack_score = importlib.import_module("pwscup2026.scoring.attack_score")
    cfg = yaml.safe_load((expected / "kit" / "kit_config.yaml").read_text(encoding="utf-8-sig"))
    if (float(cfg["scoring"]["low_fpr"]), float(cfg["aia"]["tau_time"]),
            float(cfg["scoring"]["aia_rare_se_z"])) != (0.01, 0.5, 1.64):
        raise ValueError("The copied configuration is not the expected preliminary scoring configuration")
    return scoring_io, attack_score, cfg


def _pool(truth_dir: Path, cohort: int) -> pd.DataFrame:
    frame = _read_csv(truth_dir / f"pool_ids_{cohort}.csv", ["record_id", "pool_id", "is_rare"])
    for key in ("record_id", "pool_id"):
        frame[key] = _number(frame[key], f"pool_ids_{cohort}.{key}", integer=True)
        if frame[key].duplicated().any():
            raise ValueError(f"pool_ids_{cohort}.{key}: duplicate IDs")
    frame["is_rare"] = _number(frame["is_rare"], f"pool_ids_{cohort}.is_rare", integer=True, lower=0, upper=1)
    if len(frame) != 1049:
        raise ValueError(f"pool_ids_{cohort}: expected 1049 rows")
    return frame


def _aia_truth(truth_dir: Path, cohort: int) -> pd.DataFrame:
    frame = _read_csv(truth_dir / f"aia_truth_{cohort}.csv")
    required = {"challenge_row_id", "label", "onset", "time", "is_rare"}
    if not required.issubset(frame.columns):
        raise ValueError(f"aia_truth_{cohort}: missing required truth columns")
    frame["challenge_row_id"] = _number(frame["challenge_row_id"], f"aia_truth_{cohort}.challenge_row_id", integer=True)
    if frame["challenge_row_id"].duplicated().any():
        raise ValueError(f"aia_truth_{cohort}: duplicate challenge IDs")
    for key in ("onset", "is_rare"):
        frame[key] = _number(frame[key], f"aia_truth_{cohort}.{key}", integer=True, lower=0, upper=1)
    frame["time"] = _number(frame["time"], f"aia_truth_{cohort}.time", lower=0)
    if len(frame) != 326 or frame["label"].value_counts().to_dict() != {"member": 163, "control": 163}:
        raise ValueError(f"aia_truth_{cohort}: expected member163/control163")
    return frame


def _finite_or_none(value):
    if isinstance(value, dict):
        return {k: _finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def score_predictions(prediction_dir: Path, attacker_id: int = 24,
                      historical: bool = False) -> dict:
    """Validate predictions, then reproduce the final preliminary attack result.

    ``historical`` is restricted to byte-identical organizer-released CSVs.  It
    permits only the documented cohort04 -> target23 AIA omission; the normal
    route requires every non-self target and every challenge ID.
    """
    prediction_dir = Path(prediction_dir).resolve()
    if isinstance(attacker_id, bool) or not isinstance(attacker_id, (int, np.integer)) or attacker_id not in COHORTS:
        raise ValueError(f"attacker_id must be one of the 21 scored cohorts: {COHORTS}")
    attacker_id = int(attacker_id)
    hashes = {name: _sha256(prediction_dir / name) for name in PREDICTION_NAMES}
    if historical:
        original = DATA_ROOT / "reference" / "attack_submissions" / "extracted" / f"cohort{attacker_id:02d}"
        if any(hashes[name] != _sha256(original / name) for name in PREDICTION_NAMES):
            raise ValueError("Historical replay requires byte-identical organizer-released prediction CSVs")

    truth_dir = DATA_ROOT / "private" / "ground_truth"
    pool = _pool(truth_dir, attacker_id)
    mia = _read_csv(prediction_dir / "F_mia.csv")
    if mia.columns[0] != "record_id":
        raise ValueError("F_mia.csv: first column must be record_id")
    target_columns = list(mia.columns[1:])
    try:
        mia_targets = [int(c) for c in target_columns]
    except ValueError as exc:
        raise ValueError("F_mia.csv: target column names must be canonical integer cohort IDs") from exc
    if any(str(k) != c for k, c in zip(mia_targets, target_columns)):
        raise ValueError("F_mia.csv: target column names must be canonical integer cohort IDs")
    mia["record_id"] = _number(mia["record_id"], "F_mia.record_id", integer=True)
    _same_ids(mia["record_id"], pool["record_id"], "F_mia.record_id")
    for column in target_columns:
        mia[column] = _number(mia[column], f"F_mia.{column}", lower=0, upper=1)

    aia = _read_csv(prediction_dir / "F_aia.csv", ["target_k", "challenge_row_id", "time_hat"])
    for key in ("target_k", "challenge_row_id"):
        aia[key] = _number(aia[key], f"F_aia.{key}", integer=True)
    aia["time_hat"] = _number(aia["time_hat"], "F_aia.time_hat", lower=0)
    if aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("F_aia.csv: duplicate (target_k, challenge_row_id)")

    expected = set(COHORTS) - {attacker_id}
    self_mia = str(attacker_id) in mia.columns
    self_aia_rows = int((aia["target_k"] == attacker_id).sum())
    if self_mia:
        mia = mia.drop(columns=[str(attacker_id)])
    if self_aia_rows:
        self_truth = _aia_truth(truth_dir, attacker_id)
        _same_ids(aia.loc[aia["target_k"] == attacker_id, "challenge_row_id"],
                  self_truth["challenge_row_id"], f"F_aia target {attacker_id}")
        aia = aia.loc[aia["target_k"] != attacker_id].copy()

    if set(map(int, mia.columns[1:])) != expected:
        raise ValueError(f"F_mia.csv: expected all {len(expected)} non-self target columns")
    allowed_missing = {23} if historical and attacker_id == 4 else set()
    aia_targets = set(aia["target_k"].tolist())
    if aia_targets != expected - allowed_missing:
        raise ValueError(f"F_aia.csv: target set mismatch; expected {sorted(expected - allowed_missing)}")

    truths = {}
    for target in sorted(aia_targets):
        truths[target] = _aia_truth(truth_dir, target)
        _same_ids(aia.loc[aia["target_k"] == target, "challenge_row_id"],
                  truths[target]["challenge_row_id"], f"F_aia target {target}")
    # Canonical order makes row- and column-permuted external predictions replay
    # exactly, and supplies the same alignment to official and detailed scoring.
    mia = mia[["record_id", *map(str, sorted(expected))]].sort_values("record_id").reset_index(drop=True)
    aia = aia.sort_values(["target_k", "challenge_row_id"]).reset_index(drop=True)
    scoring_io, attack_score, cfg = _load_official()
    official = scoring_io.score_attack({"F_mia.csv": mia, "F_aia.csv": aia}, truth_dir, cfg, team_id=attacker_id)
    aligned_pool = pool.set_index("record_id").loc[mia["record_id"]]
    rare = aligned_pool["is_rare"].to_numpy(dtype=bool)
    by_target = []
    for target in sorted(expected):
        target_pool = _pool(truth_dir, target)
        membership = np.isin(aligned_pool["pool_id"].to_numpy(), target_pool["pool_id"].to_numpy()).astype(int)
        mr = attack_score.score_mia(mia[str(target)].to_numpy(dtype=float), membership, cfg, rare_mask=rare)
        row = {"target_k": target, "mia_score": float(mr.score),
               "mia_tpr_all": mr.breakdown["tpr_at_low_fpr"], "mia_tpr_rare": mr.breakdown["tpr_rare"],
               "mia_auc": mr.breakdown["auc"], "mia_band": official["_breakdown"]["mia_per_target"][str(target)],
               "mia_positive_count": int(membership.sum()), "mia_negative_count": int((1 - membership).sum()),
               "mia_rare_positive_count": int(membership[rare].sum()),
               "aia_score": None, "aia_r_all": None, "aia_r_rare": None, "aia_r_rare_raw": None,
               "aia_p_m": None, "aia_p_c": None, "aia_p_m_rare": None, "aia_p_c_rare": None,
               "aia_member_count": None, "aia_control_count": None, "aia_status": "missing_historical"}
        if target in aia_targets:
            joined = aia.loc[aia["target_k"] == target].merge(truths[target], on="challenge_row_id", how="left", validate="one_to_one")
            member = joined["label"].to_numpy() == "member"
            control = joined["label"].to_numpy() == "control"
            prediction = joined["time_hat"].to_numpy(dtype=float)
            onset = joined["onset"].to_numpy(dtype=int)
            time = joined["time"].to_numpy(dtype=float)
            rare_aia = joined["is_rare"].to_numpy(dtype=bool)
            ar = attack_score.score_aia(None, prediction[member], onset[member], time[member],
                                       None, prediction[control], onset[control], time[control], cfg,
                                       member_rare_mask=rare_aia[member], control_rare_mask=rare_aia[control])
            row.update(aia_score=float(ar.score), aia_r_all=ar.breakdown["r_time_all"],
                       aia_r_rare=ar.breakdown["r_time_rare"], aia_r_rare_raw=ar.breakdown["r_time_rare_raw"],
                       aia_p_m=ar.breakdown["succ_time_member"], aia_p_c=ar.breakdown["succ_time_control"],
                       aia_p_m_rare=ar.breakdown["succ_time_member_rare"],
                       aia_p_c_rare=ar.breakdown["succ_time_control_rare"],
                       aia_member_count=int(member.sum()), aia_control_count=int(control.sum()), aia_status="scored")
        by_target.append(row)

    for key, values in (("mia_power", [r["mia_score"] for r in by_target]),
                        ("aia_power", [r["aia_score"] for r in by_target if r["aia_score"] is not None])):
        if not math.isclose(float(np.mean(values)), official[key], rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"Per-target details do not reproduce official {key}")
    summary = {key: float(official[key]) for key in
               ("attack_power", "mia_power", "aia_power", "mia_all", "mia_rare", "aia_all", "aia_rare")}
    summary.update(score_1=summary["attack_power"], score_2=summary["mia_all"], score_3=summary["mia_rare"],
                   score_4=summary["aia_all"], score_5=summary["aia_rare"],
                   mia_target_count=len(expected), aia_target_count=len(aia_targets))
    return _finite_or_none({
        "attacker_id": attacker_id, "historical": bool(historical), "scoring_round": "preliminary",
        "aia_floor_applied": False, "prediction_directory": str(prediction_dir),
        "prediction_sha256": hashes, "summary": summary, "by_target": by_target,
        "excluded_self": {"mia_column": self_mia, "aia_rows": self_aia_rows},
        "missing_aia_targets": sorted(allowed_missing), "official_result": official,
        "scorer": {"scoring_io": str(Path(scoring_io.__file__).resolve()),
                   "attack_score": str(Path(attack_score.__file__).resolve()),
                   "low_fpr": .01, "tau_time": .5, "aia_rare_se_z": 1.64},
    })
