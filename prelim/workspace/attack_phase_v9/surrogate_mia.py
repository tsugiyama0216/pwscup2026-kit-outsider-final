"""Public-surrogate donor occupancy research for MIA.

Some public releases are (within floating-point tolerance) rows selected from
one of the two public background tables.  This module maps such releases back
to their donor rows and asks whether *which donor cells were occupied* carries
membership evidence beyond ordinary distance to the released rows.

The implementation is deliberately research-only.  It never reads a token,
creates a submission, or treats an approximate nearest match as identity.  A
release row is assigned to a donor only when its robust-scaled stable-QI
distance is below an explicit tolerance.  The command-line experiment requires
the hash-gated actual-data directory and writes only aggregate diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from attack_phase_v8.actual_inputs import load_actual_inputs


STABLE_COLUMNS = ("age", "SBP", "TG")
PRIVATE_MATCH_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
SURROGATE_TARGETS = ("18", "19", "29")


def _finite_matrix(frame: pd.DataFrame, columns: Sequence[str], label: str) -> np.ndarray:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")
    matrix = frame[list(columns)].to_numpy(dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} coordinates must be a nonempty finite matrix")
    return matrix


@dataclass(frozen=True)
class RobustScale:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "RobustScale":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
            raise ValueError("scale reference must be a nonempty finite matrix")
        center = np.median(matrix, axis=0)
        q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
        spread = q75 - q25
        fallback = np.std(matrix, axis=0)
        spread = np.where(spread > 1e-12, spread, fallback)
        spread = np.where(spread > 1e-12, spread, 1.0)
        return cls(center=center, scale=spread)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.center):
            raise ValueError("coordinate dimensions differ from robust scale")
        result = (matrix - self.center) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("non-finite robust coordinates")
        return result


@dataclass(frozen=True)
class DonorMap:
    pool: np.ndarray
    donor_index: np.ndarray
    distance: np.ndarray
    matched: np.ndarray
    ambiguous: np.ndarray

    def summary(self) -> dict[str, object]:
        output: dict[str, object] = {
            "rows": int(len(self.pool)),
            "matched": int(self.matched.sum()),
            "unmatched": int((~self.matched).sum()),
            "ambiguous": int(self.ambiguous.sum()),
            "max_matched_distance": (
                float(self.distance[self.matched].max()) if self.matched.any() else None
            ),
        }
        for name in sorted(set(self.pool[self.matched].tolist())):
            indices = self.donor_index[self.matched & (self.pool == name)]
            counts = np.bincount(indices) if len(indices) else np.array([], dtype=int)
            positive = counts[counts > 0]
            output[name] = {
                "mapped_rows": int(len(indices)),
                "unique_donors": int(len(positive)),
                "duplicate_excess": int(len(indices) - len(positive)),
                "max_multiplicity": int(positive.max()) if len(positive) else 0,
            }
        return output


def map_release_to_donor_pools(
    release: pd.DataFrame,
    pools: Mapping[str, pd.DataFrame],
    *,
    columns: Sequence[str] = STABLE_COLUMNS,
    tolerance: float = 1e-8,
) -> DonorMap:
    """Map rows only when a public donor is equal within ``tolerance``.

    Pool names are sorted, making cross-pool exact ties deterministic.  Such
    ties are also marked ambiguous and should not be used as mechanistic proof.
    """
    if not pools or tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("donor pools and a finite nonnegative tolerance are required")
    release_raw = _finite_matrix(release, columns, "release")
    names = sorted(pools)
    pool_raw = {name: _finite_matrix(pools[name], columns, f"pool {name}") for name in names}
    scale = RobustScale.fit(np.vstack(list(pool_raw.values())))
    query = scale.transform(release_raw)
    distances = []
    indices = []
    for name in names:
        distance, index = cKDTree(scale.transform(pool_raw[name])).query(query, k=1)
        distances.append(distance)
        indices.append(index.astype(int))
    distance_matrix = np.column_stack(distances)
    index_matrix = np.column_stack(indices)
    winner = np.argmin(distance_matrix, axis=1)
    best = distance_matrix[np.arange(len(release)), winner]
    matched = best <= tolerance
    # Two pools can contain the same stable-QI point.  Do not hide that fact.
    ambiguous = matched & ((distance_matrix <= tolerance).sum(axis=1) > 1)
    pool_name = np.asarray(names, dtype=object)[winner]
    donor_index = index_matrix[np.arange(len(release)), winner]
    pool_name = np.where(matched, pool_name, "unmatched")
    donor_index = np.where(matched, donor_index, -1)
    return DonorMap(pool_name, donor_index, best, matched, ambiguous)


def donor_counts(mapping: DonorMap, pool_name: str, pool_size: int) -> np.ndarray:
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    selected = mapping.donor_index[mapping.matched & (mapping.pool == pool_name)]
    if len(selected) and (selected.min() < 0 or selected.max() >= pool_size):
        raise ValueError("mapping contains an out-of-range donor index")
    return np.bincount(selected, minlength=pool_size).astype(int)


def _scaled_pair(
    first: pd.DataFrame,
    reference: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, RobustScale]:
    reference_raw = _finite_matrix(reference, columns, "donors")
    first_raw = _finite_matrix(first, columns, "queries")
    scale = RobustScale.fit(reference_raw)
    return scale.transform(first_raw), scale.transform(reference_raw), scale


def assign_surrogate_donors(
    source: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    columns: Sequence[str] = STABLE_COLUMNS,
    without_replacement: bool,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate nearest-public-donor substitution.

    ``without_replacement`` performs a global minimum-cost injection.  This is
    a conditional mechanism model, not a claim about any target's generator.
    Noise is applied in robust-scaled coordinate units before assignment.
    """
    if not np.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("noise_scale must be finite and nonnegative")
    source_x, donor_x, _ = _scaled_pair(source, donors, columns)
    if noise_scale:
        if rng is None:
            raise ValueError("rng is required when noise_scale is nonzero")
        source_x = source_x + rng.normal(scale=noise_scale, size=source_x.shape)
    if without_replacement:
        if len(source_x) > len(donor_x):
            raise ValueError("without-replacement assignment needs at least as many donors")
        # 1049 x 5143 is comfortably bounded for the contest inputs.  float32
        # halves memory while preserving the much coarser empirical distances.
        cost = np.empty((len(source_x), len(donor_x)), dtype=np.float32)
        for start in range(0, len(source_x), 128):
            delta = source_x[start : start + 128, None, :] - donor_x[None, :, :]
            cost[start : start + 128] = np.einsum("ijk,ijk->ij", delta, delta)
        rows, columns_selected = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(len(source_x))):
            raise RuntimeError("assignment did not cover every source row")
        return columns_selected.astype(int)
    return cKDTree(donor_x).query(source_x, k=1)[1].astype(int)


def occupancy_scores(
    queries: pd.DataFrame,
    donors: pd.DataFrame,
    counts: np.ndarray,
    *,
    columns: Sequence[str] = STABLE_COLUMNS,
    neighbours: int = 5,
    inverse_cell: bool = False,
    cell_reference: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Compute donor-occupancy and ordinary released-distance controls."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) != len(donors) or (counts < 0).any():
        raise ValueError("counts must be a nonnegative vector aligned with donors")
    if not 1 <= neighbours <= len(donors):
        raise ValueError("neighbours is outside donor pool")
    query_x, donor_x, scale = _scaled_pair(queries, donors, columns)
    distance, index = cKDTree(donor_x).query(query_x, k=neighbours)
    if neighbours == 1:
        distance = distance[:, None]
        index = index[:, None]
    rank_weight = 1.0 / np.arange(1, neighbours + 1, dtype=float)
    positive_distance = distance[:, -1][distance[:, -1] > 1e-12]
    local_scale = float(np.median(positive_distance)) if len(positive_distance) else 1.0
    weight = rank_weight[None, :] * np.exp(-distance / local_scale)
    selected = counts[index]
    occupancy = np.sum(weight * np.log1p(selected), axis=1) / np.sum(weight, axis=1)
    binary = np.sum(weight * (selected > 0), axis=1) / np.sum(weight, axis=1)

    selected_donors = np.flatnonzero(counts > 0)
    if len(selected_donors):
        nearest_selected = -cKDTree(donor_x[selected_donors]).query(query_x, k=1)[0]
    else:
        nearest_selected = np.full(len(queries), -np.inf)

    if inverse_cell:
        if cell_reference is None:
            raise ValueError("cell_reference is required for inverse-cell scores")
        reference_raw = _finite_matrix(cell_reference, columns, "cell reference")
        reference_x = scale.transform(reference_raw)
        nearest_cell = cKDTree(donor_x).query(reference_x, k=1)[1]
        load = np.bincount(nearest_cell, minlength=len(donors)).astype(float)
        adjusted = np.sum(weight * np.log1p(selected) / np.sqrt(1.0 + load[index]), axis=1)
        adjusted /= np.sum(weight, axis=1)
    else:
        adjusted = occupancy.copy()
    return {
        "occupancy": occupancy,
        "binary_occupancy": binary,
        "inverse_cell": adjusted,
        "nearest_selected": nearest_selected,
        "nearest_donor": -distance[:, 0],
    }


def exact_challenge_membership(
    challenge: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    columns: Sequence[str] = PRIVATE_MATCH_COLUMNS,
) -> np.ndarray:
    """Return exact QI membership labels; record IDs are never joined."""
    for label, frame in (("challenge", challenge), ("candidates", candidates)):
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{label} lacks exact-match columns: {missing}")
    candidate_keys = pd.MultiIndex.from_frame(candidates[list(columns)])
    if candidate_keys.has_duplicates:
        raise ValueError("candidate exact-match key is not unique")
    challenge_keys = pd.MultiIndex.from_frame(challenge[list(columns)])
    return np.asarray(challenge_keys.isin(candidate_keys), dtype=bool)


def tpr_auc(confidence: np.ndarray, truth: np.ndarray, low_fpr: float = 0.01) -> tuple[float, float]:
    """Official tie-grouped TPR interpolation and AUC, reproduced locally."""
    confidence = np.asarray(confidence, dtype=float)
    truth = np.asarray(truth, dtype=int)
    if confidence.ndim != 1 or truth.shape != confidence.shape or not np.isfinite(confidence).all():
        raise ValueError("confidence/truth must be aligned finite vectors")
    if not 0 <= low_fpr <= 1:
        raise ValueError("low_fpr must lie in [0,1]")
    order = np.argsort(-confidence, kind="mergesort")
    c, y = confidence[order], truth[order]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.5
    endpoint = np.r_[np.flatnonzero(np.diff(c) != 0), len(c) - 1]
    tpr = np.r_[0.0, np.cumsum(y)[endpoint] / n_pos]
    fpr = np.r_[0.0, np.cumsum(1 - y)[endpoint] / n_neg]
    return float(np.interp(low_fpr, fpr, tpr)), float(np.trapezoid(tpr, fpr))


def stratified_discovery_mask(keys: Sequence[object], truth: np.ndarray) -> np.ndarray:
    """Deterministically split each class before any attack configuration fit."""
    keys_array = np.asarray(keys)
    truth = np.asarray(truth, dtype=bool)
    if keys_array.shape != truth.shape or truth.ndim != 1 or len(np.unique(keys_array)) != len(keys_array):
        raise ValueError("split keys must be unique and aligned with truth")
    discovery = np.zeros(len(truth), dtype=bool)
    for label in (False, True):
        positions = np.flatnonzero(truth == label)
        digests = np.asarray([
            hashlib.sha256(str(keys_array[position]).encode("utf-8")).hexdigest()
            for position in positions
        ])
        ordered = positions[np.argsort(digests, kind="mergesort")]
        discovery[ordered[: len(ordered) // 2]] = True
    return discovery


def rerank_top_tail(
    baseline: pd.DataFrame,
    signals: Mapping[str, np.ndarray],
    locked: np.ndarray,
    *,
    target_ids: Iterable[str] = SURROGATE_TARGETS,
    width: int = 37,
    weight: float = 0.5,
) -> pd.DataFrame:
    """Reorder only unlocked top-tail score slots; never alter their values."""
    if "record_id" not in baseline or baseline["record_id"].duplicated().any():
        raise ValueError("baseline requires unique record_id")
    locked = np.asarray(locked)
    if locked.shape != (len(baseline),) or locked.dtype.kind != "b":
        raise ValueError("locked must be a boolean mask aligned with baseline")
    if width < 1 or not np.isfinite(weight) or weight < 0:
        raise ValueError("width and weight must be nonnegative/positive")
    output = baseline.copy()
    record_id = baseline["record_id"].to_numpy()
    for target in target_ids:
        if target not in baseline or target not in signals:
            raise ValueError(f"missing target signal/baseline: {target}")
        signal = np.asarray(signals[target], dtype=float)
        if signal.shape != (len(baseline),) or not np.isfinite(signal).all():
            raise ValueError(f"invalid signal for target {target}")
        values = baseline[target].to_numpy(dtype=float)
        eligible_order = np.lexsort((record_id, -values))
        positions = eligible_order[~locked[eligible_order]][:width]
        if len(positions) < 2:
            continue
        # rank 0 is strongest for both baseline and the new signal.
        base_rank = np.empty(len(positions), dtype=float)
        base_rank[np.argsort(-values[positions], kind="mergesort")] = np.arange(len(positions))
        signal_rank = np.empty(len(positions), dtype=float)
        signal_order = np.lexsort((record_id[positions], -signal[positions]))
        signal_rank[signal_order] = np.arange(len(positions))
        fused = base_rank + weight * signal_rank
        destination = positions[np.lexsort((record_id[positions], fused))]
        replacement = np.sort(values[positions])[::-1]
        changed = values.copy()
        changed[destination] = replacement
        output[target] = changed
    if not output["record_id"].equals(baseline["record_id"]):
        raise RuntimeError("candidate keys changed")
    if not output.loc[locked].equals(baseline.loc[locked]):
        raise RuntimeError("locked rows changed")
    for column in baseline.columns:
        if column != "record_id" and not np.array_equal(
            np.sort(output[column].to_numpy()), np.sort(baseline[column].to_numpy())
        ):
            raise RuntimeError(f"score multiset changed for target {column}")
    return output


def _rank_average(*scores: np.ndarray) -> np.ndarray:
    ranks = []
    for score in scores:
        score = np.asarray(score, dtype=float)
        order = np.argsort(score, kind="mergesort")
        rank = np.empty(len(score), dtype=float)
        rank[order] = np.arange(len(score), dtype=float)
        ranks.append(rank / max(len(score) - 1, 1))
    return np.mean(ranks, axis=0)


def _occupancy_shape(counts: np.ndarray) -> dict[str, float | int]:
    counts = np.asarray(counts, dtype=int)
    positive = counts[counts > 0]
    return {
        "mapped_rows": int(counts.sum()),
        "unique_donors": int(len(positive)),
        "duplicate_excess": int(counts.sum() - len(positive)),
        "max_multiplicity": int(positive.max()) if len(positive) else 0,
        "sum_choose_2": int(np.sum(positive * (positive - 1) // 2)),
    }


def _randomized_counts(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute donor identities while preserving the occupancy histogram."""
    result = np.zeros_like(counts)
    positive = counts[counts > 0].copy()
    locations = rng.choice(len(counts), size=len(positive), replace=False)
    rng.shuffle(positive)
    result[locations] = positive
    return result


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown rendering for aggregate diagnostics."""
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---"] * len(columns)]
    for values in frame.itertuples(index=False, name=None):
        rows.append([str(value).replace("|", "\\|") for value in values])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def run_research(
    *,
    dist: Path,
    targets: Path,
    toy_background_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run bounded, predeclared occupancy checks and persist aggregate results."""
    actual = load_actual_inputs(Path(dist), Path(targets))
    toy_background = pd.read_csv(toy_background_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pools = {"actual": actual.background, "toy": toy_background}

    mechanism_rows: list[dict[str, object]] = []
    production_signals: dict[str, np.ndarray] = {}
    mapping_summaries: dict[str, object] = {}
    for target in SURROGATE_TARGETS:
        mapping = map_release_to_donor_pools(actual.releases[target], pools)
        summary = mapping.summary()
        mapping_summaries[target] = summary
        for pool_name, donors in pools.items():
            counts = donor_counts(mapping, pool_name, len(donors))
            shape = _occupancy_shape(counts)
            if counts.sum() == 0:
                continue
            score = occupancy_scores(
                actual.candidates, donors, counts, neighbours=5,
                inverse_cell=True, cell_reference=actual.candidates,
            )
            # Candidate-to-donor demand is a public-only mechanism check.  It is
            # not a hidden membership label.
            candidate_x, donor_x, _ = _scaled_pair(actual.candidates, donors, STABLE_COLUMNS)
            candidate_cell = cKDTree(donor_x).query(candidate_x, k=1)[1]
            demand = np.bincount(candidate_cell, minlength=len(donors))
            selected = counts > 0
            selected_mean = float(demand[selected].mean()) if selected.any() else float("nan")
            null_means = []
            rng = np.random.default_rng(9000 + int(target) + (0 if pool_name == "actual" else 100))
            for _ in range(200):
                null = _randomized_counts(counts, rng) > 0
                null_means.append(float(demand[null].mean()))
            null_means_array = np.asarray(null_means)
            z = (selected_mean - float(null_means_array.mean())) / max(float(null_means_array.std(ddof=1)), 1e-12)
            row = {
                "target": target,
                "pool": pool_name,
                **shape,
                "pool_stable3_duplicate_rows": int(
                    len(donors) - len(donors.drop_duplicates(list(STABLE_COLUMNS)))
                ),
                "candidate_demand_selected_mean": selected_mean,
                "candidate_demand_null_mean": float(null_means_array.mean()),
                "candidate_demand_null_sd": float(null_means_array.std(ddof=1)),
                "candidate_demand_z": float(z),
            }
            mechanism_rows.append(row)
            if target == "29":
                production_signals.setdefault(target, np.zeros(len(actual.candidates)))
                production_signals[target] += score["inverse_cell"] * (counts.sum() / len(actual.releases[target]))
            else:
                production_signals[target] = score["inverse_cell"]

    mechanism = pd.DataFrame(mechanism_rows)
    mechanism.to_csv(output_dir / "mechanism_audit.csv", index=False)

    truth = exact_challenge_membership(actual.self_challenge, actual.candidates)
    if int(truth.sum()) != 163 or int((~truth).sum()) != 163:
        raise ValueError("self challenge exact-QI labels are not 163 members / 163 controls")
    discovery_mask = stratified_discovery_mask(actual.self_challenge["challenge_row_id"], truth)
    validation_mask = ~discovery_mask

    # Discovery and validation differ by both random seeds and perturbation.
    configs = [
        (pool, without, neighbours, inverse)
        for pool, without in (("actual", True), ("toy", False))
        for neighbours in (1, 3, 5, 10)
        for inverse in (False, True)
    ]
    discovery_noise = (0.0, 0.01)
    validation_noise = (0.02, 0.05, 0.10)
    discovery_seeds = (101, 211, 307, 401)
    validation_seeds = (1009, 2017, 3019, 4021, 5003, 6011)
    result_rows: list[dict[str, object]] = []

    def evaluate(split: str, noise_values: Sequence[float], seeds: Sequence[int]) -> None:
        query_mask = discovery_mask if split == "discovery" else validation_mask
        for pool_name, without, neighbours, inverse in configs:
            donors = pools[pool_name]
            for noise in noise_values:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    assigned = assign_surrogate_donors(
                        actual.candidates, donors, without_replacement=without,
                        noise_scale=noise, rng=rng,
                    )
                    counts = np.bincount(assigned, minlength=len(donors))
                    scores = occupancy_scores(
                        actual.self_challenge, donors, counts, neighbours=neighbours,
                        inverse_cell=inverse, cell_reference=actual.candidates,
                    )
                    attack_name = "inverse_cell" if inverse else "occupancy"
                    occupancy = scores[attack_name]
                    # DCR is the conventional control; rank fusion tests only
                    # whether donor identity adds information beyond it.
                    dcr = scores["nearest_selected"]
                    fused = _rank_average(occupancy, dcr)
                    null_counts = _randomized_counts(counts, np.random.default_rng(seed + 70001))
                    null_scores = occupancy_scores(
                        actual.self_challenge, donors, null_counts, neighbours=neighbours,
                        inverse_cell=inverse, cell_reference=actual.candidates,
                    )[attack_name]
                    for name, confidence in (("occupancy", occupancy), ("dcr", dcr), ("fusion", fused), ("null", null_scores)):
                        tpr, auc = tpr_auc(confidence[query_mask], truth[query_mask])
                        result_rows.append({
                            "split": split, "pool": pool_name,
                            "without_replacement": without, "neighbours": neighbours,
                            "inverse_cell": inverse, "noise": noise, "seed": seed,
                            "score": name, "query_rows": int(query_mask.sum()),
                            "positive_rows": int(truth[query_mask].sum()),
                            "tpr_at_0p01": tpr, "auc": auc,
                        })

    evaluate("discovery", discovery_noise, discovery_seeds)
    results = pd.DataFrame(result_rows)
    discovery = results[results.split == "discovery"]
    grouped = (
        discovery[discovery.score == "fusion"]
        .groupby(["pool", "without_replacement", "neighbours", "inverse_cell"], as_index=False)
        .agg(mean_tpr=("tpr_at_0p01", "mean"), mean_auc=("auc", "mean"))
        .sort_values(["mean_tpr", "mean_auc"], ascending=False, kind="mergesort")
    )
    selected_config = grouped.iloc[0].to_dict()
    evaluate("validation", validation_noise, validation_seeds)
    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "self_challenge_simulations.csv", index=False)

    mask = (
        (results.split == "validation")
        & (results.pool == selected_config["pool"])
        & (results.without_replacement == bool(selected_config["without_replacement"]))
        & (results.neighbours == int(selected_config["neighbours"]))
        & (results.inverse_cell == bool(selected_config["inverse_cell"]))
    )
    chosen = results[mask].pivot_table(
        index=["noise", "seed"], columns="score", values=["tpr_at_0p01", "auc"]
    )
    delta_tpr = chosen[("tpr_at_0p01", "fusion")] - chosen[("tpr_at_0p01", "dcr")]
    delta_auc = chosen[("auc", "fusion")] - chosen[("auc", "dcr")]
    null_auc = chosen[("auc", "null")]
    validation_summary = {
        "observations": int(len(chosen)),
        "mean_delta_tpr_vs_dcr": float(delta_tpr.mean()),
        "p10_delta_tpr_vs_dcr": float(delta_tpr.quantile(0.10)),
        "win_rate_tpr_vs_dcr": float((delta_tpr > 0).mean()),
        "mean_delta_auc_vs_dcr": float(delta_auc.mean()),
        "p10_delta_auc_vs_dcr": float(delta_auc.quantile(0.10)),
        "null_mean_auc": float(null_auc.mean()),
        "null_max_abs_auc_from_half": float(np.max(np.abs(null_auc - 0.5))),
    }

    baseline_path = Path("submissions/prelim/team24/attack-candidate-v4/F_mia.csv")
    baseline = pd.read_csv(baseline_path)
    locked = actual.known_private | actual.rare
    candidate = rerank_top_tail(baseline, production_signals, locked)
    from pwscup2026.kit import validate as official_validate
    official_result = official_validate.validate_mia_submission(
        candidate, {}, official_validate.load_dist(Path(targets))
    )
    changed_by_target = {
        target: int(np.count_nonzero(
            candidate[target].to_numpy() != baseline[target].to_numpy()
        )) for target in baseline.columns if target != "record_id"
    }
    invariants = {
        "record_keys_unchanged": bool(candidate.record_id.equals(baseline.record_id)),
        "locked_rows": int(locked.sum()),
        "locked_cells_changed": int(np.count_nonzero(
            candidate.loc[locked].drop(columns="record_id").to_numpy()
            != baseline.loc[locked].drop(columns="record_id").to_numpy()
        )),
        "targets_1_2_changed_cells": int(
            np.count_nonzero(candidate[["1", "2"]].to_numpy() != baseline[["1", "2"]].to_numpy())
        ),
        "official_mia_validator": "PASS" if official_result.ok else "FAIL",
        "official_mia_validator_errors": list(official_result.errors),
        "changed_by_target": changed_by_target,
        "baseline_sha256": _hash_file(baseline_path),
    }
    if not official_result.ok:
        raise ValueError(f"hypothetical normal-format MIA failed validator: {official_result.errors}")

    # Predeclared conservative gate.  Empirical self-challenge performance is
    # necessary but not sufficient: production must also exhibit non-random
    # candidate-dependent donor selection in every modified target/pool.
    self_gate = (
        validation_summary["mean_delta_tpr_vs_dcr"] > 0
        and validation_summary["p10_delta_tpr_vs_dcr"] >= 0
        and validation_summary["win_rate_tpr_vs_dcr"] >= 0.60
        and validation_summary["mean_delta_auc_vs_dcr"] > 0
        and validation_summary["p10_delta_auc_vs_dcr"] >= -0.002
        and abs(validation_summary["null_mean_auc"] - 0.5) <= 0.05
    )
    mechanism_gate = bool(len(mechanism)) and bool((mechanism.candidate_demand_z >= 2.0).all())
    promote = bool(self_gate and mechanism_gate)
    decision = {
        "decision": "PROMOTE" if promote else "REJECT",
        "reason": (
            "all predeclared self-validation and production mechanism gates passed"
            if promote else
            "donor occupancy lacks robust incremental low-FPR and/or production mechanism evidence"
        ),
        "selected_discovery_config": selected_config,
        "validation": validation_summary,
        "gates": {"self_validation": bool(self_gate), "production_mechanism": bool(mechanism_gate)},
        "mapping": mapping_summaries,
        "candidate_invariants": invariants,
        "candidate_written": False,
        "token_read": False,
        "official_submission": False,
        "limitations": [
            "self-challenge query rows are stratified-disjoint across discovery and validation, but both surrogate releases are simulated from the same actual B source",
            "noise seeds are sensitivity replicates, not independent people or official targets",
            "C18 uniqueness is compatible with multiple mechanisms and does not by itself prove nearest-donor substitution",
            "C1/C2 approximate proximity is not used as membership truth",
            "hidden target membership and leaderboard feedback are not used row by row",
        ],
        "provenance": actual.provenance,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Public-surrogate donor occupancy MIA",
        "",
        f"Decision: **{decision['decision']}**.",
        "",
        "This branch maps C18/C19/C29 to public background donors under a strict floating-point tolerance, then tests whether donor-cell occupancy adds low-FPR signal beyond ordinary distance. It is a mechanism-conditional competition experiment, not a leaderboard forecast. The actual and toy pools each have unique stable3 cells; the audit records this rather than assuming row identity.",
        "",
        "## Actual-release mechanism audit",
        "",
        _markdown_table(mechanism),
        "",
        "## Discovery-locked self-challenge validation",
        "",
        f"Selected config: `{selected_config}`.",
        "",
        f"Validation observations: {validation_summary['observations']}; mean/p10 TPR delta vs DCR = {validation_summary['mean_delta_tpr_vs_dcr']:+.6f} / {validation_summary['p10_delta_tpr_vs_dcr']:+.6f}; win rate = {validation_summary['win_rate_tpr_vs_dcr']:.1%}.",
        "",
        f"Mean/p10 AUC delta vs DCR = {validation_summary['mean_delta_auc_vs_dcr']:+.6f} / {validation_summary['p10_delta_auc_vs_dcr']:+.6f}. Null mean AUC = {validation_summary['null_mean_auc']:.6f}.",
        "",
        "## Decision and safeguards",
        "",
        f"Self-validation gate: `{self_gate}`. Production mechanism gate: `{mechanism_gate}`. The research candidate was not written (`candidate_written=false`).",
        "",
        "Any hypothetical rerank is limited to unlocked top-tail slots of targets 18/19/29. It preserves record keys, every known-private/rare row, targets 1/2, and every target's score multiset. Aggregate change counts are recorded in `decision.json`.",
        "",
        "Discovery and validation use disjoint, class-stratified self-challenge query rows and different noise seeds. They still reuse actual B as the simulated source and therefore cannot prove transfer to hidden targets. C18's all-unique donor pattern can be produced by a global no-replacement policy or by unrelated sampling. C1/C2 are never treated as truth-known identities. No official submission, ZIP, production file, or token was touched.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True, help="explicit actual team24 distribution directory")
    parser.add_argument("--targets", type=Path, default=Path("team-shared/prelim/attack-targets"))
    parser.add_argument("--toy-background", type=Path, default=Path("participant_data/A_bg.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("local-output/attack-v9/surrogate-mia"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    decision = run_research(
        dist=args.dist, targets=args.targets, toy_background_path=args.toy_background,
        output_dir=args.output_dir,
    )
    print(json.dumps({"decision": decision["decision"], "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
