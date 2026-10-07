"""Strict single-release scoring using the immutable official attack scorer.

The mock floor belongs to the generated questions, reduced background and truth,
not to any real contest target. No official hidden answer or credential is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import PACKAGE, cfg, read_csv, sha256
from pwscup2026.kit.codabench import scoring_io
from pwscup2026.scoring.attack_score import score_aia

if not Path(scoring_io.__file__).resolve().is_relative_to(PACKAGE.resolve()):
    raise ImportError("mock_TS must use the pinned official scoring package")

FLOOR_METHOD = "mock_TS_background_reference_1049_without_replacement_v1"
SUPPLIED_PHI_10 = 0.026887
SCORE_KEYS = ("attack_power", "mia_all", "mia_rare", "aia_all", "aia_rare")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(values, label):
    a = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(a).all() and (a == np.floor(a)).all() and (np.abs(a) < 2**53).all(),
             f"{label}: IDs must be finite exact integers")
    return a.astype(np.int64)


def _unique_ids(frame, key, label):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique and key in frame,
             f"{label}: missing or duplicate columns")
    a = _ids(frame[key], label)
    _require(len(a) == len(np.unique(a)), f"{label}: duplicate IDs")
    return a


def _numeric(values, label, maximum=None):
    a = np.asarray(values, dtype=float)
    _require(np.isfinite(a).all() and (a >= 0).all(), f"{label}: values must be finite and nonnegative")
    if maximum is not None:
        _require((a <= maximum).all(), f"{label}: values exceed {maximum}")
    return a


def _binary(values, label):
    a = _numeric(values, label, 1)
    _require(np.isin(a, (0, 1)).all(), f"{label}: values must be binary")
    return a.astype(bool)


def _load_case(case_dir):
    case_dir = Path(case_dir)
    candidates = read_csv(case_dir / "public" / "B.csv")
    challenge = read_csv(case_dir / "public" / "aia_challenge_0.csv")
    mia = read_csv(case_dir / "private" / "mia_truth.csv")
    aia = read_csv(case_dir / "private" / "aia_truth.csv")
    rid = _unique_ids(candidates, "record_id", "public B")
    qid = _unique_ids(challenge, "challenge_row_id", "public challenge")
    _require(len(rid) == 1049 and len(qid) == 326, "Expected 1049 MIA candidates and 326 AIA questions")
    mrid = _unique_ids(mia, "record_id", "MIA truth")
    aqid = _unique_ids(aia, "challenge_row_id", "AIA truth")
    _require(set(rid) == set(mrid), "MIA truth IDs differ from public candidates")
    _require(set(qid) == set(aqid), "AIA truth IDs differ from public challenge")
    _require({"member", "is_rare"}.issubset(mia.columns), "MIA truth is incomplete")
    _require({"label", "time", "onset", "is_rare"}.issubset(aia.columns), "AIA truth is incomplete")
    member = _binary(mia.member, "MIA member")
    rare = _binary(mia.is_rare, "MIA rare")
    _require(25 <= member.sum() <= 27, "MIA needs 25 to 27 positive candidates")
    _require(rare.sum() == 24, "MIA needs 24 true/proxy rare candidates")
    _require(set(aia.label) == {"member", "control"}, "AIA truth labels must be member/control")
    arare = _binary(aia.is_rare, "AIA rare")
    _binary(aia.onset, "AIA onset")
    _numeric(aia.time, "AIA truth time")
    for label in ("member", "control"):
        mask = aia.label.eq(label).to_numpy()
        _require(mask.sum() == 163 and arare[mask].sum() == 13,
                 f"AIA needs 163 {label} records with 13 rare records")
    return candidates, challenge, mia, aia


def validate_predictions(f_mia, f_aia, candidates, challenge):
    """Fail before an official inner join can drop missing or duplicate answers."""
    rid = _unique_ids(f_mia, "record_id", "F_mia")
    _require(set(f_mia.columns) == {"record_id", "0"}, "F_mia columns must be record_id and 0")
    _require(set(rid) == set(_ids(candidates.record_id, "public B")), "F_mia IDs differ from public B")
    _numeric(f_mia["0"], "F_mia confidence", 1)
    _require(isinstance(f_aia, pd.DataFrame) and f_aia.columns.is_unique and
             set(f_aia.columns) == {"target_k", "challenge_row_id", "time_hat"},
             "F_aia columns must be target_k, challenge_row_id and time_hat")
    qid = _unique_ids(f_aia, "challenge_row_id", "F_aia")
    target = _ids(f_aia.target_k, "F_aia target")
    _require((target == 0).all(), "F_aia must contain only isolated target 0")
    _require(set(qid) == set(_ids(challenge.challenge_row_id, "public challenge")),
             "F_aia IDs differ from public challenge")
    # The official format permits any finite nonnegative time prediction.
    _numeric(f_aia.time_hat, "F_aia time_hat")
    return {"mia_rows": len(rid), "aia_rows": len(qid), "target_count": 1}


def _aia_result(prediction, truth, config):
    member = truth.label.eq("member").to_numpy()
    rare = truth.is_rare.to_numpy(dtype=bool)
    onset = truth.onset.to_numpy(dtype=int)
    time = truth.time.to_numpy(dtype=float)
    return score_aia(None, prediction[member], onset[member], time[member],
                     None, prediction[~member], onset[~member], time[~member], config,
                     member_rare_mask=rare[member], control_rare_mask=rare[~member])


def _reference_time_samples(background, query, samples, config):
    """Cache official reference distances with identical ties and fallbacks.

    Adapted from the reviewed mock_test/sugiyama_v2/scoring.py algorithm,
    independently tested against pinned pwscup2026.attack.aia.aia_predict.
    Only the unused onset k-NN output is omitted; time is unchanged.
    """
    bg = background.reset_index(drop=True)
    query = query.reset_index(drop=True)
    samples = np.asarray(samples, dtype=np.int64)
    _require(samples.ndim == 2 and samples.shape[1] > 0, "Empty floor samples")
    _require((samples >= 0).all() and (samples < len(bg)).all(), "Floor sample index out of range")
    _require(all(len(np.unique(row)) == len(row) for row in samples), "Floor samples must be without replacement")
    present = np.zeros((len(samples), len(bg)), dtype=bool)
    present[np.arange(len(samples))[:, None], samples] = True
    columns, keys = config["aia"]["qi_continuous"], config["aia"]["block_keys"]
    std = bg[columns].std().to_numpy(dtype=float)
    std = np.where(std > 0, std, 1.)
    x, q = bg[columns].to_numpy(dtype=float) / std, query[columns].to_numpy(dtype=float) / std
    times = bg.time.to_numpy(dtype=float)
    _require(np.isfinite(x).all() and np.isfinite(q).all() and np.isfinite(times).all(),
             "Floor reference inputs must be finite")
    result = np.empty((len(samples), len(query)), dtype=float)
    for j in range(len(query)):
        exact = np.ones(len(bg), dtype=bool)
        for key in keys:
            exact &= bg[key].to_numpy() == query.iloc[j][key]
        first_key = bg[keys[0]].to_numpy() == query.iloc[j][keys[0]]
        pending = np.ones(len(samples), dtype=bool)
        distance = np.sqrt(np.mean((x - q[j]) ** 2, axis=1))
        for allowed in (exact, first_key, np.ones(len(bg), dtype=bool)):
            if not pending.any():
                break
            indices = np.flatnonzero(allowed)
            if not len(indices):
                continue
            rows = np.flatnonzero(pending & present[:, indices].any(axis=1))
            if not len(rows):
                continue
            available = present[np.ix_(rows, indices)]
            masked = np.where(available, distance[indices][None, :], np.inf)
            nearest = np.argmin(masked, axis=1)
            result[rows, j] = times[indices[nearest]]
            ties = available & (distance[indices][None, :] == masked.min(axis=1)[:, None])
            for row_no in np.flatnonzero(ties.sum(axis=1) > 1):
                result[rows[row_no], j] = float(np.mean(np.sort(times[indices[ties[row_no]]])))
            pending[rows] = False
        _require(not pending.any(), "Reference floor could not find a background neighbour")
    return result


def _floor_inputs(case_dir, config):
    paths = ("public/A_bg.csv", "public/aia_challenge_0.csv", "private/aia_truth.csv")
    hashes = {path: sha256(Path(case_dir) / path) for path in paths}
    hashes["official_config"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return hashes


def estimate_floor(case_dir: Path, seed: int, repetitions=1000) -> dict:
    """Average the official AIA max(all, rare-with-SE) on background-only draws."""
    _require(isinstance(repetitions, int) and not isinstance(repetitions, bool) and repetitions >= 2,
             "At least two floor repetitions are required")
    _, query, _, truth = _load_case(case_dir)
    background = read_csv(Path(case_dir) / "public" / "A_bg.csv")
    _require(len(background) >= 1049, "Floor background must contain at least 1049 rows")
    config = cfg()
    truth = truth.set_index("challenge_row_id").loc[query.challenge_row_id].reset_index()
    rng = np.random.default_rng(seed)
    samples = np.stack([rng.choice(len(background), 1049, replace=False) for _ in range(repetitions)])
    prediction = _reference_time_samples(background, query, samples, config)
    scores = np.array([_aia_result(p, truth, config).score for p in prediction], dtype=float)
    _numeric(scores, "Floor reference scores", 1)
    sd = float(scores.std(ddof=1))
    return {"phi": float(scores.mean()), "sd": sd, "se": float(sd / np.sqrt(repetitions)),
            "repetitions": repetitions, "seed": int(seed), "sample_rows": 1049,
            "method": FLOOR_METHOD, "minimum": float(scores.min()), "maximum": float(scores.max()),
            "reference_rulebook_repetitions": 1000, "diagnostic_reduced_repetitions": repetitions < 1000,
            "mock_truth_only": True, "official_hidden_floors_known": False,
            "input_sha256": _floor_inputs(case_dir, config),
            "draw_statistic": "official score_aia: max(all, rare after SE deadzone)",
            "limitation": "Monte Carlo SE excludes population and synthetic-control mismatch."}


def _validate_floor(floor, case_dir, config):
    _require(isinstance(floor, dict), "Estimated floor metadata is required; missing is not zero")
    _require(floor.get("method") == FLOOR_METHOD and floor.get("mock_truth_only") is True and
             floor.get("official_hidden_floors_known") is False, "Invalid mock floor provenance")
    n = floor.get("repetitions")
    _require(isinstance(n, int) and not isinstance(n, bool) and n >= 2 and floor.get("sample_rows") == 1049,
             "Invalid floor repetition/sample metadata")
    _require(isinstance(floor.get("seed"), int), "Floor seed is required")
    _require(floor.get("diagnostic_reduced_repetitions") is (n < 1000), "Invalid floor diagnostic marker")
    _numeric([floor.get("phi", np.nan)], "Mock floor", 1)
    _numeric([floor.get("sd", np.nan), floor.get("se", np.nan)], "Mock floor uncertainty")
    _require(np.isclose(floor["se"], floor["sd"] / np.sqrt(n), rtol=1e-12, atol=1e-15),
             "Floor standard error differs from metadata")
    _require(floor.get("input_sha256") == _floor_inputs(case_dir, config),
             "Floor belongs to different questions, truth, background or config")


def score_predictions(case_dir: Path, f_mia: pd.DataFrame, f_aia: pd.DataFrame,
                      floor: dict, scratch: Path) -> dict:
    """Score one C as a single target, never as another target in a shared world."""
    candidates, query, mia, aia = _load_case(case_dir)
    validation = validate_predictions(f_mia, f_aia, candidates, query)
    config = cfg()
    _validate_floor(floor, case_dir, config)
    scratch = Path(scratch)
    _require(not scratch.exists() or (scratch.is_dir() and not any(scratch.iterdir())),
             "Scoring scratch must be new or empty")
    scratch.mkdir(parents=True, exist_ok=True)
    # Artificial pool identity has no person identifier or relation to row order.
    pool = mia[["record_id", "is_rare"]].copy()
    pool["pool_id"] = np.arange(len(pool), dtype=int)
    pool.to_csv(scratch / "pool_ids_attacker.csv", index=False)
    pool.loc[mia.member.to_numpy(dtype=bool)].to_csv(scratch / "pool_ids_0.csv", index=False)
    aia.to_csv(scratch / "aia_truth_0.csv", index=False)
    pd.DataFrame({"team_id": [0], "aia_floor": [floor["phi"]]}).to_csv(scratch / "aia_floor.csv", index=False)
    frames = {"F_mia.csv": f_mia, "F_aia.csv": f_aia}
    official = scoring_io.score_attack(frames, scratch, config, team_id=None)
    ordered = aia.set_index("challenge_row_id").loc[f_aia.challenge_row_id].reset_index()
    raw_aia = float(_aia_result(f_aia.time_hat.to_numpy(dtype=float), ordered, config).score)
    sensitivity = float(np.clip(raw_aia - SUPPLIED_PHI_10, 0., 1.))
    return {**{f"score_{i+1}": float(official[key]) for i, key in enumerate(SCORE_KEYS)},
            "score_6": None, "mia_power": official["mia_power"], "aia_power": official["aia_power"],
            "breakdown": official["_breakdown"], "official_result": official,
            "aia_before_floor": raw_aia, "phi_mock": floor["phi"], "floor_estimate": floor,
            "provided_phi10_sensitivity": {"phi": SUPPLIED_PHI_10, "aia_power": sensitivity,
                "attack_power": float(official["mia_power"] + sensitivity),
                "label": "Reference sensitivity only; supplied phi does not belong to these mock questions"},
            "denominators": {"mia_positive": int(mia.member.sum()),
                "mia_negative": int((mia.member == 0).sum()),
                "mia_rare_positive": int((mia.member.to_numpy(dtype=bool) & mia.is_rare.to_numpy(dtype=bool)).sum()),
                "mia_rare_negative": int((~mia.member.to_numpy(dtype=bool) & mia.is_rare.to_numpy(dtype=bool)).sum()),
                "mia_rare_defined": bool((mia.member.to_numpy(dtype=bool) & mia.is_rare.to_numpy(dtype=bool)).any()),
                "aia_member": 163, "aia_control": 163, "aia_rare_member": 13, "aia_rare_control": 13},
            "validation": validation, "official_score_prediction": False,
            "score_6_reason": "Mock releases are alternative versions of the same B; no official final ranking score."}
