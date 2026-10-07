"""Strict mock-world validation and the pinned official attack aggregation.

The simulated floor is computed from simulated truth and public background; it
is never an estimate of an actual target's hidden official floor.  No credentials
are read, generated, copied or needed by this module.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from pwscup2026.kit.codabench import scoring_io
from pwscup2026.scoring.attack_score import score_aia, score_mia


_PINNED_PACKAGE = Path(__file__).resolve().parents[2] / "attack" / "official-kit" / "starter" / "src" / "pwscup2026"
if not Path(scoring_io.__file__).resolve().is_relative_to(_PINNED_PACKAGE):
    raise ImportError("Sugiyama scoring must use the repository's pinned official package")


TARGET_IDS = tuple(sorted((set(range(18)) | {22, 25, 26, 27, 28}) - {10}))
FLOOR_METHOD = "mock_background_reference_aia_1049_without_replacement"
FLOOR_METADATA = "aia_floor_metadata.json"
LEADERBOARD = ("attack_power", "mia_all", "mia_rare", "aia_all", "aia_rare")


@contextmanager
def _scratch_directory(prefix):
    # Python's Windows mkdtemp(mode=0700) can create an inaccessible directory
    # under a restricted desktop token. Normal inherited workspace ACLs work.
    parent = Path(__file__).resolve().parents[1] / "local-output" / "sugiyama-score-scratch"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (prefix + uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        _require(path.resolve().parent == parent.resolve(), "Scratch path escaped its workspace")
        # Only flat, explicitly generated mock CSVs are stored here.
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(values, label):
    numbers = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    _require(np.isfinite(numbers).all() and (numbers == np.floor(numbers)).all(),
             f"{label}: IDs must be finite integers")
    _require((np.abs(numbers) < 2**53).all(), f"{label}: IDs exceed exact integer range")
    return numbers.astype(np.int64)


def _unique_ids(frame, column, label):
    _require(isinstance(frame, pd.DataFrame) and frame.columns.is_unique and column in frame,
             f"{label}: missing or duplicate columns")
    ids = _ids(frame[column], label)
    _require(len(np.unique(ids)) == len(ids), f"{label}: duplicate IDs")
    return ids


def _numeric(values, label, *, maximum=None):
    arr = np.asarray(values, dtype=float)
    _require(np.isfinite(arr).all(), f"{label}: values must be finite")
    _require((arr >= 0).all(), f"{label}: values must be nonnegative")
    if maximum is not None:
        _require((arr <= maximum).all(), f"{label}: values exceed {maximum}")
    return arr


def _binary(values, label):
    arr = _numeric(values, label, maximum=1)
    _require(np.isin(arr, [0, 1]).all(), f"{label}: expected binary values")
    return arr.astype(bool)


def _public_layout(public):
    _require(public.get("team_id") == 10, "This mock protocol represents attacker team 10")
    targets = public.get("targets", {})
    _require(set(targets) == set(TARGET_IDS), "Expected exactly the 22 non-self target IDs")
    candidate_ids = _unique_ids(public["candidates"], "record_id", "public B")
    _require(len(candidate_ids) == 1049, "public B must contain exactly 1049 rows")
    challenge_ids = {}
    for k in TARGET_IDS:
        ids = _unique_ids(targets[k]["challenge"], "challenge_row_id", f"challenge {k}")
        _require(len(ids) == 326, f"challenge {k}: expected 326 rows")
        challenge_ids[k] = ids
    return candidate_ids, challenge_ids


def validate_predictions(f_mia, f_aia, public):
    """Validate by keys, allowing harmless row/target-column permutations.

    Self targets are intentionally excluded in generated worlds and predictions.
    The official scorer also excludes self; no zero-valued phantom target is added.
    """
    candidate_ids, challenge_ids = _public_layout(public)
    rid = _unique_ids(f_mia, "record_id", "F_mia")
    expected_cols = {"record_id", *(str(k) for k in TARGET_IDS)}
    _require(set(f_mia.columns) == expected_cols, "F_mia: incorrect target columns")
    _require(set(rid) == set(candidate_ids), "F_mia: candidate IDs differ from public B")
    _numeric(f_mia.drop(columns="record_id"), "F_mia", maximum=1)
    _require(isinstance(f_aia, pd.DataFrame) and f_aia.columns.is_unique and
             set(f_aia.columns) == {"target_k", "challenge_row_id", "time_hat"},
             "F_aia: incorrect or duplicate columns")
    target = _ids(f_aia["target_k"], "F_aia target")
    query = _ids(f_aia["challenge_row_id"], "F_aia challenge")
    pairs = list(zip(target, query))
    expected_pairs = {(k, int(r)) for k, ids in challenge_ids.items() for r in ids}
    _require(len(set(pairs)) == len(pairs), "F_aia: duplicate target/challenge pairs")
    _require(set(pairs) == expected_pairs, "F_aia: target/challenge IDs differ from public input")
    _numeric(f_aia["time_hat"], "F_aia")
    return {"mia_rows": len(rid), "aia_rows": len(pairs), "target_count": len(TARGET_IDS)}


def _load_truth(public, truth_dir):
    """Fail before official inner joins can silently discard malformed truth rows."""
    candidate_ids, challenge_ids = _public_layout(public)
    root = Path(truth_dir)
    paths = [root / "pool_ids_10.csv"]
    paths += [root / f"pool_ids_{k}.csv" for k in TARGET_IDS]
    paths += [root / f"aia_truth_{k}.csv" for k in TARGET_IDS]
    _require(all(p.is_file() for p in paths), "Mock private truth is incomplete")
    pools = {}
    for k in (10, *TARGET_IDS):
        pool = pd.read_csv(root / f"pool_ids_{k}.csv", float_precision="round_trip")
        rid = _unique_ids(pool, "record_id", f"pool {k}")
        _require(len(rid) == 1049 and "pool_id" in pool and "is_rare" in pool,
                 f"pool {k}: expected 1049 rows and explicit rare truth")
        _unique_ids(pool, "pool_id", f"pool {k}")
        rare = _binary(pool["is_rare"], f"pool {k} rare")
        _require(int(rare.sum()) == 24, f"pool {k}: expected exactly 24 true rare rows")
        if k == 10:
            _require(set(rid) == set(candidate_ids), "Private attacker IDs differ from public B")
        pools[k] = pool
    challenges = {}
    for k in TARGET_IDS:
        truth = pd.read_csv(root / f"aia_truth_{k}.csv", float_precision="round_trip")
        rid = _unique_ids(truth, "challenge_row_id", f"truth {k}")
        _require(set(rid) == set(challenge_ids[k]), f"truth {k}: challenge IDs differ")
        _require({"label", "onset", "time", "is_rare"}.issubset(truth.columns),
                 f"truth {k}: missing explicit labels")
        _require(set(truth["label"]) == {"member", "control"}, f"truth {k}: invalid labels")
        rare = _binary(truth["is_rare"], f"truth {k} rare")
        for label in ("member", "control"):
            mask = truth["label"].eq(label).to_numpy()
            _require(mask.sum() == 163 and rare[mask].sum() == 13,
                     f"truth {k}: expected 163 {label} rows including 13 rare")
        _binary(truth["onset"], f"truth {k} onset")
        _numeric(truth["time"], f"truth {k} time")
        challenges[k] = truth
    return pools, challenges, paths


def _aia_result(predictions, truth, cfg):
    member = truth["label"].eq("member").to_numpy()
    rare = truth["is_rare"].to_numpy(dtype=bool)
    onset = truth["onset"].to_numpy(dtype=int)
    time = truth["time"].to_numpy(dtype=float)
    return score_aia(None, predictions[member], onset[member], time[member],
                     None, predictions[~member], onset[~member], time[~member], cfg,
                     member_rare_mask=rare[member], control_rare_mask=rare[~member])


def _reference_time_samples(background, query, samples, cfg):
    """Cache official aia_predict distances, retaining its tie and fallback rules.

    Each row of ``samples`` contains distinct positional indices into background.
    The scoring time prediction is 1-NN; the unused onset k-NN output is omitted.
    """
    bg = background.reset_index(drop=True)
    query = query.reset_index(drop=True)
    samples = np.asarray(samples, dtype=np.int64)
    _require(samples.ndim == 2 and samples.shape[1] > 0, "Empty floor samples")
    _require((samples >= 0).all() and (samples < len(bg)).all(), "Floor sample index out of range")
    _require(all(len(np.unique(row)) == len(row) for row in samples),
             "Floor samples must be drawn without replacement")
    present = np.zeros((len(samples), len(bg)), dtype=bool)
    present[np.arange(len(samples))[:, None], samples] = True
    cols = cfg["aia"]["qi_continuous"]
    keys = cfg["aia"]["block_keys"]
    std = bg[cols].std().to_numpy(dtype=float)
    std = np.where(std > 0, std, 1.)
    x = bg[cols].to_numpy(dtype=float) / std
    q = query[cols].to_numpy(dtype=float) / std
    times = bg["time"].to_numpy(dtype=float)
    _require(np.isfinite(x).all() and np.isfinite(q).all() and np.isfinite(times).all(),
             "Floor reference inputs must be finite")
    result = np.empty((len(samples), len(query)), dtype=float)
    for j in range(len(query)):
        exact = np.ones(len(bg), dtype=bool)
        for key in keys:
            exact &= bg[key].to_numpy() == query.iloc[j][key]
        first_key = bg[keys[0]].to_numpy() == query.iloc[j][keys[0]]
        pending = np.ones(len(samples), dtype=bool)
        distances = np.sqrt(np.mean((x - q[j]) ** 2, axis=1))
        for allowed in (exact, first_key, np.ones(len(bg), dtype=bool)):
            indices = np.flatnonzero(allowed)
            if not len(indices):
                continue
            rows = np.flatnonzero(pending & present[:, indices].any(axis=1))
            if not len(rows):
                continue
            available = present[np.ix_(rows, indices)]
            masked = np.where(available, distances[indices][None, :], np.inf)
            nearest = np.argmin(masked, axis=1)
            result[rows, j] = times[indices[nearest]]
            # np.mean(np.sort(...)) matches the official tie implementation,
            # including floating-point summation order at the tau boundary.
            ties = available & (distances[indices][None, :] == masked.min(axis=1)[:, None])
            for row_no in np.flatnonzero(ties.sum(axis=1) > 1):
                result[rows[row_no], j] = float(np.mean(np.sort(times[indices[ties[row_no]]])))
            pending[rows] = False
        _require(not pending.any(), "Reference floor could not find a background neighbour")
    return result


def estimate_floors(public, truth_dir, cfg, seed, repetitions=1000):
    """Return ``(aia_floor_dataframe, metadata)``; caller saves both privately.

    Per rulebook pp.21-22, sample 1049 background rows without replacement and
    average the official AIA score of the reference attack.  Default 1000 matches
    the published repetition count.  Smaller counts are explicitly diagnostics.
    Common draws across targets reduce irrelevant Monte Carlo comparison noise.
    """
    _require(isinstance(repetitions, int) and repetitions >= 2, "At least two floor repetitions required")
    _, truths, _ = _load_truth(public, truth_dir)
    background = public["background"]
    _require(len(background) >= 1049, "Floor background needs at least 1049 rows")
    rng = np.random.default_rng(seed)
    samples = np.stack([rng.choice(len(background), 1049, replace=False) for _ in range(repetitions)])
    records = []
    diagnostics = {}
    for k in TARGET_IDS:
        query = public["targets"][k]["challenge"]
        truth = truths[k].set_index("challenge_row_id").loc[query.challenge_row_id].reset_index()
        predictions = _reference_time_samples(background, query, samples, cfg)
        scores = np.array([_aia_result(pred, truth, cfg).score for pred in predictions], dtype=float)
        _require(np.isfinite(scores).all(), f"Mock floor {k}: nonfinite score")
        mean, sd = float(scores.mean()), float(scores.std(ddof=1))
        records.append({"team_id": k, "aia_floor": mean})
        diagnostics[str(k)] = {"mean": mean, "sample_sd": sd,
                                "monte_carlo_se": sd / np.sqrt(repetitions),
                                "minimum": float(scores.min()), "maximum": float(scores.max())}
    metadata = {"method": FLOOR_METHOD, "seed": int(seed), "repetitions": repetitions,
                "reference_rulebook_repetitions": 1000, "sample_rows": 1049,
                "diagnostic_reduced_repetitions": repetitions < 1000,
                "official_hidden_floors_known": False, "mock_truth_only": True,
                "cached_reference_distance_calculation": True, "per_target": diagnostics,
                "draw_statistic": "official score_aia score: max(all, rare after SE deadzone)",
                "limitation": "Mock floors only; Monte Carlo SE excludes population and generator mismatch."}
    return pd.DataFrame(records), metadata


def _load_floors(truth_dir):
    root = Path(truth_dir)
    _require((root / "aia_floor.csv").is_file() and (root / FLOOR_METADATA).is_file(),
             "Estimated-floor scoring requires complete mock floor CSV and metadata; missing is not zero")
    frame = pd.read_csv(root / "aia_floor.csv", float_precision="round_trip")
    ids = _unique_ids(frame, "team_id", "mock floor")
    _require(set(ids) == set(TARGET_IDS) and "aia_floor" in frame, "Mock floor targets are incomplete")
    _numeric(frame["aia_floor"], "mock floor", maximum=1)
    meta = json.loads((root / FLOOR_METADATA).read_text(encoding="utf-8"))
    _require(meta.get("method") == FLOOR_METHOD and meta.get("mock_truth_only") is True and
             meta.get("official_hidden_floors_known") is False,
             "Floor provenance must identify a local mock estimate")
    _require(isinstance(meta.get("repetitions"), int) and meta["repetitions"] >= 2 and
             meta.get("sample_rows") == 1049, "Floor sampling metadata is incomplete")
    for _, row in frame.iterrows():
        record = meta.get("per_target", {}).get(str(int(row.team_id)), {})
        _require(np.isclose(record.get("mean", np.nan), row.aia_floor, rtol=0, atol=1e-14),
                 "Floor estimate differs from its recorded mean")
        _numeric([record.get("monte_carlo_se", np.nan)], "Floor Monte Carlo SE")
    return meta


def score_world(f_mia, f_aia, public, truth_dir, cfg, *, floor_mode="estimated"):
    """Use the immutable official score_attack on one complete 22-target world."""
    _require(floor_mode in {"estimated", "none"}, "floor_mode must be estimated or none")
    validation = validate_predictions(f_mia, f_aia, public)
    pools, truths, paths = _load_truth(public, truth_dir)
    frames = {"F_mia.csv": f_mia, "F_aia.csv": f_aia}
    # Explicitly absent floors in this snapshot give an independently computed
    # pre-floor result; never mutate truth or patch the official module globally.
    with _scratch_directory("score-") as scratch:
        for source in paths:
            shutil.copyfile(source, scratch / source.name)
        pre_floor = scoring_io.score_attack(frames, scratch, cfg, team_id=10)
    if floor_mode == "estimated":
        metadata = _load_floors(truth_dir)
        result = scoring_io.score_attack(frames, Path(truth_dir), cfg, team_id=10)
    else:
        metadata = None
        result = dict(pre_floor)
    ap = pools[10].set_index("record_id").loc[f_mia.record_id]
    rare = ap.is_rare.to_numpy(dtype=bool)
    rare_defined = 0
    for k in TARGET_IDS:
        truth = np.isin(ap.pool_id.to_numpy(), pools[k].pool_id.to_numpy())
        component = score_mia(f_mia[str(k)].to_numpy(), truth, cfg, rare_mask=rare)
        rare_defined += int(np.isfinite(component.breakdown["tpr_rare"]))
    result["leaderboard"] = {f"score_{i + 1}": float(result[key]) for i, key in enumerate(LEADERBOARD)}
    result["pre_floor"] = {key: float(pre_floor[key]) for key in (*LEADERBOARD, "mia_power", "aia_power")}
    result["denominators"] = {"target_count": len(TARGET_IDS), "mia_rare_defined_targets": rare_defined,
                              "aia_rare_defined_targets": len(truths)}
    result["validation"] = validation
    result["floor_mode"] = floor_mode
    result["floor_estimate"] = metadata
    result["official_score_prediction"] = False
    return result
