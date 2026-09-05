"""Production-aligned contrastive AIA study for the seven official-zero targets.

Research only.  The script learns a normal-format row policy on discovery
shadows, evaluates the locked policy on held-out and fresh seeds, and writes a
research mask below local-output.  It never edits a submission candidate,
reads a token, creates a ZIP, or contacts a remote service.

Unlike a rank-threshold sweep, the policy estimates, for each permitted time,
the two joint probabilities

    q_m(x,t) = P(member and |time-t| <= tau | public x)
    q_c(x,t) = P(control and |time-t| <= tau | public x)

and uses the Dinkelbach marginal utility for the official nonlinear objective

    R = (p_m - p_c) / (1 - p_c),
    utility(x,t; R) = q_m(x,t) - (1-R) q_c(x,t).

The estimator is deliberately low dimensional and hierarchical: the three
submitted active votes, matched public stratum, public rare proxy, release
edge-event rate, and public release/background density are pooled with fixed
Bayesian shrinkage.  Discovery/held-out/fresh seeds are strictly disjoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "local-output" / "attack-v8" / "actual-contrastive"
ACTUAL_DATA = Path(
    r"C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
)
STARTER = ROOT / "starter" / "src"
SHADOW_DIR = ROOT / "local-output" / "attack-v6" / "shadow-benchmark"
OFFICIAL_DIR = ROOT / "local-output" / "attack-v7" / "aia-official-bayes"
RARE_DIR = ROOT / "local-output" / "attack-v7" / "aia-rare"
for source_root in (ROOT, STARTER, SHADOW_DIR, OFFICIAL_DIR, RARE_DIR):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import benchmark as bench  # noqa: E402
import benchmark_official_bayes as official  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    age_band,
    filter_linkable_releases,
    strict_rank,
)
from attack_phase_v8.actual_inputs import load_actual_inputs  # noqa: E402
from rare_proxy import partial_mahalanobis  # noqa: E402


ZERO_TARGETS = tuple(int(value) for value in bench.OFFICIAL_ZERO_IDS)
EDGE = 9.5
ABSTAIN = 11.0
ACTIONS = (9.0, 9.5, 10.0)
# These are Monte-Carlo posterior draws over the *actual* public challenge
# rows.  Keeping the integer ranges disjoint makes the discovery/held-out/
# fresh split mechanically auditable even though the empirical member-outcome
# reservoir is necessarily small (the only truth-known private cohort has 163
# rows).
DISCOVERY = tuple(range(20270401, 20270441))
HELDOUT = tuple(range(20270501, 20270541))
FRESH = tuple(range(20270601, 20270681))
EXCHANGEABLE = tuple(range(20270801, 20270881))
N_ARM = 163
EPS = 1e-12


@dataclass
class StudyBlock:
    split: str
    seed: int
    scenario: str
    official_target: int
    internal_target: str
    frame: pd.DataFrame
    truth: pd.DataFrame
    direct: dict[str, float]

    @property
    def exact_zero(self) -> bool:
        return max(self.direct.values()) <= EPS

    @property
    def near_zero(self) -> bool:
        return max(self.direct.values()) <= 0.02 + EPS

    @property
    def key(self) -> str:
        return (
            f"{self.split}:{self.seed}:{self.scenario}:"
            f"{self.official_target}:{self.internal_target}"
        )


def stable_rank(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    order = np.lexsort((ids, values))
    result = np.empty(len(values), dtype=float)
    result[order] = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    return result


def exact_match_rate(
    source: pd.DataFrame, reference: pd.DataFrame, columns: Sequence[str]
) -> int:
    keys = set(map(tuple, reference.loc[:, columns].to_numpy()))
    return int(sum(tuple(row) in keys for row in source.loc[:, columns].to_numpy()))


def _mixed_release(
    source: pd.DataFrame,
    background: pd.DataFrame,
    toy_background: pd.DataFrame,
    template: pd.DataFrame,
    *,
    official_target: int,
    scenario: str,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Approximate the observed source/background geometry without private truth.

    Production C18 derives completely from actual A_bg and C19 completely from
    the retained toy A_bg under tolerant stable3 matching.  Other zero targets
    have no background-derived rows under that provenance audit.
    exact stable3 matches, but source-preserving shadows give direct powers near
    one and contradict the observed three-way zero feedback.  They therefore
    use an independent background draw rank-warped to the public C_i margins,
    with two independent release draws.  The repeated draw measures release
    instability without injecting private-source signal contradicted by the
    official all-zero condition.
    """
    donor_background = toy_background if official_target == 19 else background
    if official_target in (18, 19):
        background_fraction = 1.0
    elif scenario in ("independent_warp_a", "independent_warp_b"):
        background_fraction = 1.0
    else:
        raise ValueError(f"unknown scenario {scenario}")

    n_background = int(round(background_fraction * len(source)))
    n_source = len(source) - n_background
    pieces: list[pd.DataFrame] = []
    if n_source:
        rows = rng.choice(len(source), size=n_source, replace=False)
        pieces.append(source.iloc[rows])
    if n_background:
        rows = rng.choice(
            len(donor_background), size=n_background, replace=False
        )
        pieces.append(donor_background.iloc[rows])
    mixed = pd.concat(pieces, ignore_index=True)
    mixed = mixed.iloc[rng.permutation(len(mixed))].reset_index(drop=True)

    # C18/C19 background-copy structure is the signal being reproduced;
    # rank warping it would destroy that structure.  For the remaining targets,
    # the public target table supplies the observed marginal geometry only.
    if official_target in (18, 19):
        return bench.shadow.coerce_release(mixed, cfg).iloc[
            rng.permutation(len(mixed))
        ].reset_index(drop=True)
    return bench.rank_warp(mixed, template, cfg, rng)


def _scenario_releases(
    world: Any,
    *,
    seed: int,
    scenario: str,
    cfg: dict[str, Any],
    templates: dict[int, pd.DataFrame],
    toy_background: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    releases: dict[str, pd.DataFrame] = {}
    for ordinal, internal_target in enumerate(sorted(world.aia_sources, key=int)):
        official_target = ZERO_TARGETS[ordinal]
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, ordinal, official_target, SCENARIOS.index(scenario), 8849]
            )
        )
        releases[internal_target] = _mixed_release(
            world.aia_sources[internal_target],
            world.attack_background,
            toy_background,
            templates[official_target],
            official_target=official_target,
            scenario=scenario,
            cfg=cfg,
            rng=rng,
        )
    return releases


def _event_rate_by_stratum(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    center: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    def estimate(reference: pd.DataFrame) -> np.ndarray:
        work = reference.copy()
        work["_age_band"] = age_band(work["age"])
        hit = (
            np.abs(work["time"].to_numpy(dtype=float) - center) <= tau
        ).astype(float)
        work["_hit"] = hit
        global_rate = float(hit.mean())
        grouped = work.groupby(["sex", "_age_band"], sort=False)["_hit"].agg(
            ["sum", "count"]
        )
        output = np.empty(len(challenge), dtype=float)
        challenge_band = age_band(challenge["age"])
        for index, (sex, band) in enumerate(
            zip(challenge["sex"], challenge_band, strict=True)
        ):
            key = (sex, band)
            if key in grouped.index:
                row = grouped.loc[key]
                output[index] = (float(row["sum"]) + 32.0 * global_rate) / (
                    float(row["count"]) + 32.0
                )
            else:
                output[index] = global_rate
        return output

    return estimate(release), estimate(background)


def _knn_event_rate(
    reference: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    center: float,
    tau: float,
    neighbors: int = 32,
) -> np.ndarray:
    columns = bench.VIEWS["all5"]
    reference_x = bench.scaled_matrix(reference, background, columns)
    challenge_x = bench.scaled_matrix(challenge, background, columns)
    count = min(neighbors, len(reference_x))
    nearest = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(reference_x)
    distances, indices = nearest.kneighbors(challenge_x, return_distance=True)
    hit = (
        np.abs(reference["time"].to_numpy(dtype=float) - center) <= tau
    ).astype(float)
    weights = 1.0 / np.maximum(distances, 1e-3)
    return np.sum(weights * hit[indices], axis=1) / np.sum(weights, axis=1)


def _background_occupancy_features(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return public donor-occupancy support around each challenge row.

    C18 and C19 are entirely background-derived under the tolerant provenance
    audit.  A nearest-surrogate mechanism can still reveal which background
    Voronoi cells were requested (and how often) even though it cannot reveal
    the original private outcome.  The feature is also defined for non-copy
    releases so the model cannot infer a target from a missing value.
    """
    scores: list[np.ndarray] = []
    counts_out: list[np.ndarray] = []
    distances_out: list[np.ndarray] = []
    for columns in (bench.VIEWS["all5"], bench.VIEWS["stable3"]):
        background_x = bench.scaled_matrix(background, background, columns)
        release_x = bench.scaled_matrix(release, background, columns)
        challenge_x = bench.scaled_matrix(challenge, background, columns)
        nearest = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(background_x)
        release_distance, release_index = nearest.kneighbors(
            release_x, return_distance=True
        )
        challenge_distance, challenge_index = nearest.kneighbors(
            challenge_x, return_distance=True
        )
        occupancy = np.bincount(
            release_index[:, 0], minlength=len(background_x)
        ).astype(float)
        selected_count = occupancy[challenge_index[:, 0]]
        selected_distance = challenge_distance[:, 0]
        # Exact or repeated donors receive high support; distant nearest cells
        # receive smoothly less.  log1p keeps multiplicities robust.
        score = np.log1p(selected_count) - np.log1p(selected_distance)
        scores.append(score)
        counts_out.append(selected_count)
        distances_out.append(selected_distance)
    return (
        np.mean(np.column_stack(scores), axis=1),
        np.mean(np.column_stack(counts_out), axis=1),
        np.mean(np.column_stack(distances_out), axis=1),
    )


def _augment_public_features(
    frame: pd.DataFrame,
    challenge: pd.DataFrame,
    release: pd.DataFrame,
    background: pd.DataFrame,
    detector: dict[str, Any],
    *,
    tau: float,
) -> pd.DataFrame:
    output = frame.copy()
    ids = challenge["challenge_row_id"].to_numpy(dtype=np.int64)
    release_knn = _knn_event_rate(
        release, challenge, background, center=EDGE, tau=tau
    )
    background_knn = _knn_event_rate(
        background, challenge, background, center=EDGE, tau=tau
    )
    release_stratum, background_stratum = _event_rate_by_stratum(
        release, challenge, background, center=EDGE, tau=tau
    )
    rare_rank = stable_rank(partial_mahalanobis(challenge, detector), ids)
    density = np.mean(
        np.column_stack(
            [
                output["density_all5"].to_numpy(dtype=float),
                output["density_stable3"].to_numpy(dtype=float),
                output["density_nohdl"].to_numpy(dtype=float),
                output["density_labs"].to_numpy(dtype=float),
            ]
        ),
        axis=1,
    )
    density_rank = stable_rank(density, ids)
    event_contrast = 0.70 * (release_knn - background_knn) + 0.30 * (
        release_stratum - background_stratum
    )
    event_rank = stable_rank(event_contrast, ids)
    support, support_count, support_distance = _background_occupancy_features(
        release, challenge, background
    )
    support_rank = stable_rank(support, ids)
    output["release_edge_knn"] = release_knn
    output["background_edge_knn"] = background_knn
    output["release_edge_stratum"] = release_stratum
    output["background_edge_stratum"] = background_stratum
    output["edge_event_contrast"] = event_contrast
    output["background_occupancy_support"] = support
    output["background_occupancy_count"] = support_count
    output["background_occupancy_distance"] = support_distance
    output["rare_proxy_rank"] = rare_rank
    output["density_mean_rank"] = density_rank
    output["rare_bin"] = (rare_rank >= 0.92).astype(int)
    output["edge_bin"] = np.minimum((3.0 * event_rank).astype(int), 2)
    output["density_bin"] = np.minimum((3.0 * density_rank).astype(int), 2)
    output["support_bin"] = np.minimum((3.0 * support_rank).astype(int), 2)
    coarse_age = np.select(
        [challenge["age"].to_numpy(dtype=float) < 55.0,
         challenge["age"].to_numpy(dtype=float) < 70.0],
        ["40_54", "55_69"],
        default="70_plus",
    )
    output["public_stratum"] = (
        challenge["sex"].astype(str).to_numpy() + ":" + coarse_age
    )
    return output


def build_blocks(
    split: str,
    seeds: Iterable[int],
    *,
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    detector: dict[str, Any],
    spec: Any,
    cfg: dict[str, Any],
    templates: dict[int, pd.DataFrame],
    toy_background: pd.DataFrame,
) -> list[StudyBlock]:
    tau = float(cfg["aia"]["tau_time"])
    blocks: list[StudyBlock] = []
    for seed in seeds:
        print(f"{split} seed={seed}: shadow world", flush=True)
        world = bench.shadow.build_shadow_world(
            candidates, b_self, background, detector, spec, seed
        )
        bench.exactify_visible_control_matching(world)
        bench.shadow.validate_world(world, spec)
        for scenario in SCENARIOS:
            releases = _scenario_releases(
                world,
                seed=seed,
                scenario=scenario,
                cfg=cfg,
                templates=templates,
                toy_background=toy_background,
            )
            v4_frame, independent_frame = official._build_base_attacks(
                world, releases, cfg
            )
            linkable, _ = filter_linkable_releases(releases)
            for ordinal, internal_target in enumerate(
                sorted(world.challenges, key=int)
            ):
                official_target = ZERO_TARGETS[ordinal]
                challenge = world.challenges[internal_target].reset_index(drop=True)
                truth = world.aia_truth[internal_target].reset_index(drop=True)
                if not challenge["challenge_row_id"].equals(
                    truth["challenge_row_id"]
                ):
                    raise RuntimeError("challenge/truth order differs")
                v4_time = bench.aligned_time(v4_frame, internal_target, challenge)
                independent_time = bench.aligned_time(
                    independent_frame, internal_target, challenge
                )
                features = official._route_feature_frame(
                    official_target=official_target,
                    internal_target=internal_target,
                    challenge=challenge,
                    release=releases[internal_target],
                    background=world.attack_background,
                    linkable_release=linkable[internal_target],
                    v4_time=v4_time,
                    independent_time=independent_time,
                )
                # The submitted density routes (7, 9, 19) were generated before
                # the actual team background was restored.  Recreate their vote
                # with that retained toy reference, while keeping all continuous
                # contrastive features calibrated to the actual background.
                if official_target in (7, 9, 19):
                    legacy = official._route_feature_frame(
                        official_target=official_target,
                        internal_target=internal_target,
                        challenge=challenge,
                        release=releases[internal_target],
                        background=toy_background,
                        linkable_release=linkable[internal_target],
                        v4_time=v4_time,
                        independent_time=independent_time,
                    )
                    features["routed_active"] = legacy["routed_active"].to_numpy(
                        dtype=float
                    )
                    features["reject_votes"] = (
                        3
                        - features["v4_active"].astype(int)
                        - features["independent_active"].astype(int)
                        - features["routed_active"].astype(int)
                    ).astype(float)
                    features["pattern"] = (
                        features["v4_active"].astype(int).astype(str)
                        + features["independent_active"].astype(int).astype(str)
                        + features["routed_active"].astype(int).astype(str)
                    )
                features = _augment_public_features(
                    features,
                    challenge,
                    linkable[internal_target],
                    world.attack_background,
                    detector,
                    tau=tau,
                )
                routed_time = np.where(
                    features["routed_active"].to_numpy(dtype=bool),
                    10.0,
                    ABSTAIN,
                )
                direct = {
                    "v4": bench.score_block(v4_time, truth, cfg)["power"],
                    "independent": bench.score_block(
                        independent_time, truth, cfg
                    )["power"],
                    "routed": bench.score_block(routed_time, truth, cfg)["power"],
                }
                member = truth["label"].eq("member").to_numpy()
                features["label_member"] = member.astype(int)
                features["truth_rare"] = truth["is_rare"].astype(int).to_numpy()
                truth_time = truth["time"].to_numpy(dtype=float)
                for action in ACTIONS:
                    suffix = str(action).replace(".", "p")
                    hit = np.abs(truth_time - action) <= tau
                    features[f"mhit_{suffix}"] = (member & hit).astype(int)
                    features[f"chit_{suffix}"] = (~member & hit).astype(int)
                features["split"] = split
                features["seed"] = seed
                features["scenario"] = scenario
                blocks.append(
                    StudyBlock(
                        split=split,
                        seed=seed,
                        scenario=scenario,
                        official_target=official_target,
                        internal_target=internal_target,
                        frame=features,
                        truth=truth,
                        direct=direct,
                    )
                )
    return blocks


PUBLISHED_STRATUM_COUNTS = {
    (True, 1): 4,
    (True, 0): 9,
    (False, 1): 12,
    (False, 0): 138,
}


def _stable_fold(frame: pd.DataFrame, folds: int = 3) -> np.ndarray:
    record_ids = frame["record_id"].to_numpy(dtype=np.uint64)
    mixed = (
        record_ids * np.uint64(11400714819323198485)
        + np.uint64(14029467366897019727)
    )
    return np.asarray(mixed % np.uint64(folds), dtype=int)


def _balanced_member_labels(
    challenge: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    """Draw exactly one half-member arm inside every published match block."""
    work = challenge.assign(_age_band=age_band(challenge["age"]))
    member = np.zeros(len(challenge), dtype=bool)
    for _, indices in work.groupby(["sex", "_age_band"], sort=True).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        if len(rows) % 2:
            raise RuntimeError("actual challenge has an odd matched-block size")
        member[rng.choice(rows, size=len(rows) // 2, replace=False)] = True
    if int(member.sum()) != N_ARM:
        raise RuntimeError(f"balanced labels yielded {member.sum()} members")
    return member


def _sample_outcome_donors(
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sample the published rare/onset composition without hidden-label tuning."""
    chosen: list[pd.DataFrame] = []
    fallback_rows = 0
    replacement_rows = 0
    for (rare_value, onset_value), count in PUBLISHED_STRATUM_COUNTS.items():
        eligible = primary.loc[
            primary["is_rare"].astype(bool).eq(rare_value)
            & primary["onset"].astype(int).eq(onset_value)
        ]
        source = eligible
        if len(source) < count:
            source = fallback.loc[
                fallback["is_rare"].astype(bool).eq(rare_value)
                & fallback["onset"].astype(int).eq(onset_value)
            ]
            fallback_rows += count
        if source.empty:
            raise RuntimeError(
                f"no outcome donors for rare={rare_value}, onset={onset_value}"
            )
        replace = len(source) < count
        replacement_rows += count if replace else 0
        indices = rng.choice(len(source), size=count, replace=replace)
        chosen.append(source.iloc[indices].copy())
    donors = pd.concat(chosen, ignore_index=True)
    donors = donors.iloc[rng.permutation(len(donors))].reset_index(drop=True)
    return donors, {
        "fallback_rows": int(fallback_rows),
        "replacement_rows": int(replacement_rows),
    }


def _assign_donor_outcomes(
    challenge_arm: pd.DataFrame,
    donors: pd.DataFrame,
    reference: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Condition empirical hidden outcomes on public QIs by noisy assignment."""
    if len(challenge_arm) != N_ARM or len(donors) != N_ARM:
        raise RuntimeError("outcome assignment requires 163 rows per arm")
    columns = bench.VIEWS["all5"]
    challenge_x = bench.scaled_matrix(challenge_arm, reference, columns)
    donor_x = bench.scaled_matrix(donors, reference, columns)
    delta = challenge_x[:, None, :] - donor_x[None, :, :]
    cost = np.mean(delta * delta, axis=2)
    challenge_sex = challenge_arm["sex"].to_numpy()[:, None]
    donor_sex = donors["sex"].to_numpy()[None, :]
    cost += 1.50 * (challenge_sex != donor_sex)
    challenge_band = age_band(challenge_arm["age"]).astype(str)[:, None]
    donor_band = age_band(donors["age"]).astype(str)[None, :]
    cost += 0.50 * (challenge_band != donor_band)
    # Small independent perturbation integrates assignment uncertainty and
    # prevents a deterministic nearest-neighbour artefact from being learned.
    cost += rng.gumbel(0.0, 0.03, size=cost.shape)
    local_rows, donor_rows = linear_sum_assignment(cost)
    assigned = np.empty(N_ARM, dtype=int)
    assigned[local_rows] = donor_rows
    ordered = donors.iloc[assigned]
    return (
        ordered["time"].to_numpy(dtype=float),
        ordered["onset"].to_numpy(dtype=int),
        ordered["is_rare"].astype(int).to_numpy(),
    )


def _simulate_actual_truth(
    challenge: pd.DataFrame,
    *,
    member_primary: pd.DataFrame,
    member_fallback: pd.DataFrame,
    control_primary: pd.DataFrame,
    control_fallback: pd.DataFrame,
    reference: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, int]]:
    member = _balanced_member_labels(challenge, rng)
    member_rows = np.flatnonzero(member)
    control_rows = np.flatnonzero(~member)
    member_donors, member_diag = _sample_outcome_donors(
        member_primary, member_fallback, rng
    )
    control_donors, control_diag = _sample_outcome_donors(
        control_primary, control_fallback, rng
    )
    member_time, member_onset, member_rare = _assign_donor_outcomes(
        challenge.iloc[member_rows], member_donors, reference, rng
    )
    control_time, control_onset, control_rare = _assign_donor_outcomes(
        challenge.iloc[control_rows], control_donors, reference, rng
    )
    time_truth = np.empty(len(challenge), dtype=float)
    onset_truth = np.empty(len(challenge), dtype=int)
    rare_truth = np.empty(len(challenge), dtype=int)
    time_truth[member_rows] = member_time
    time_truth[control_rows] = control_time
    onset_truth[member_rows] = member_onset
    onset_truth[control_rows] = control_onset
    rare_truth[member_rows] = member_rare
    rare_truth[control_rows] = control_rare
    truth = pd.DataFrame(
        {
            "challenge_row_id": challenge["challenge_row_id"].astype(int),
            "label": np.where(member, "member", "control"),
            "onset": onset_truth,
            "time": time_truth,
            "is_rare": rare_truth,
        }
    )
    for label in ("member", "control"):
        arm = truth["label"].eq(label)
        if (
            int(arm.sum()) != N_ARM
            or int(truth.loc[arm, "is_rare"].sum()) != 13
            or int(truth.loc[arm, "onset"].sum()) != 16
            or int(
                (
                    truth.loc[arm, "is_rare"].astype(bool)
                    & truth.loc[arm, "onset"].astype(bool)
                ).sum()
            )
            != 4
        ):
            raise RuntimeError("simulated truth violates published AIA strata")
    return truth, {
        "member_fallback_rows": member_diag["fallback_rows"],
        "member_replacement_rows": member_diag["replacement_rows"],
        "control_fallback_rows": control_diag["fallback_rows"],
        "control_replacement_rows": control_diag["replacement_rows"],
    }


def _posterior_member_pools(
    candidates: pd.DataFrame,
    b_self: pd.DataFrame,
    known_private: np.ndarray,
    self_challenge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hometown = str(self_challenge["prefecture"].iloc[0])
    full = b_self.copy()
    is_hometown = candidates["prefecture"].astype(str).eq(hometown).to_numpy()
    known_private = np.asarray(known_private, dtype=bool)
    known = full.loc[known_private].reset_index(drop=True)
    # Discovery never uses the 150 non-rare known-private outcomes used by the
    # held-out/fresh reservoir.  The 13 private rare rows must be shared because
    # the remaining hometown cohort contains only one rare row and no
    # rare-onset row; this irreducible limitation is surfaced in the report.
    discovery_mask = (is_hometown & ~known_private) | (
        known_private & full["is_rare"].astype(bool).to_numpy()
    )
    discovery = full.loc[discovery_mask].reset_index(drop=True)
    audit = {
        "hometown": hometown,
        "hometown_rows": int(is_hometown.sum()),
        "known_private_rows": int(known_private.sum()),
        "known_private_rare_rows": int(
            (known_private & full["is_rare"].astype(bool).to_numpy()).sum()
        ),
        "discovery_member_reservoir_rows": len(discovery),
        "discovery_known_nonrare_overlap": int(
            (
                discovery_mask
                & known_private
                & ~full["is_rare"].astype(bool).to_numpy()
            ).sum()
        ),
        "irreducible_rare_overlap_rows": int(
            (discovery_mask & known_private & full["is_rare"].astype(bool).to_numpy()).sum()
        ),
    }
    return discovery, known, audit


def build_actual_posterior_blocks(
    split: str,
    seeds: Iterable[int],
    *,
    production: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
    member_primary: pd.DataFrame,
    member_fallback: pd.DataFrame,
    background_with_rare: pd.DataFrame,
    background_reference: pd.DataFrame,
    cfg: dict[str, Any],
    control_fold: int,
    exchangeable: bool = False,
) -> tuple[list[StudyBlock], dict[str, int]]:
    tau = float(cfg["aia"]["tau_time"])
    folds = _stable_fold(background_with_rare, folds=3)
    fold_pool = background_with_rare.loc[folds == control_fold].reset_index(drop=True)
    blocks: list[StudyBlock] = []
    diagnostics = {
        "worlds": 0,
        "member_fallback_rows": 0,
        "member_replacement_rows": 0,
        "control_fallback_rows": 0,
        "control_replacement_rows": 0,
    }
    for seed in seeds:
        if (seed - min(seeds)) % 10 == 0:
            print(f"{split} posterior seed={seed}", flush=True)
        for target in ZERO_TARGETS:
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, target, 910247])
            )
            challenge = (
                challenges[str(target)]
                .sort_values("challenge_row_id", kind="mergesort")
                .reset_index(drop=True)
            )
            features = production.loc[
                production["official_target"].astype(int).eq(target)
            ].reset_index(drop=True)
            if not challenge["challenge_row_id"].equals(
                features["challenge_row_id"].astype(int)
            ):
                raise RuntimeError("actual challenge/features order differs")
            prefecture = str(challenge["prefecture"].iloc[0])
            local_controls = fold_pool.loc[
                fold_pool["prefecture"].astype(str).eq(prefecture)
            ].reset_index(drop=True)
            if exchangeable:
                local_members = local_controls
                member_all = fold_pool
            else:
                local_members = member_primary
                member_all = member_fallback
            truth, diagnostic = _simulate_actual_truth(
                challenge,
                member_primary=local_members,
                member_fallback=member_all,
                control_primary=local_controls,
                control_fallback=fold_pool,
                reference=background_reference,
                rng=rng,
            )
            for key, value in diagnostic.items():
                diagnostics[key] += int(value)
            diagnostics["worlds"] += 1
            features = features.copy()
            member = truth["label"].eq("member").to_numpy()
            truth_time = truth["time"].to_numpy(dtype=float)
            features["label_member"] = member.astype(int)
            features["truth_rare"] = truth["is_rare"].astype(int).to_numpy()
            for action in ACTIONS:
                suffix = str(action).replace(".", "p")
                hit = np.abs(truth_time - action) <= tau
                features[f"mhit_{suffix}"] = (member & hit).astype(int)
                features[f"chit_{suffix}"] = (~member & hit).astype(int)
            features["split"] = split
            features["seed"] = seed
            features["scenario"] = (
                "exchangeable_actual_bg" if exchangeable else "hometown_private"
            )
            direct = {
                name: bench.score_block(
                    features[f"direct_time_{name}"].to_numpy(dtype=float),
                    truth,
                    cfg,
                )["power"]
                for name in ("v4", "independent", "routed")
            }
            blocks.append(
                StudyBlock(
                    split=split,
                    seed=seed,
                    scenario=(
                        "exchangeable_actual_bg"
                        if exchangeable
                        else "hometown_private"
                    ),
                    official_target=target,
                    internal_target=str(target),
                    frame=features,
                    truth=truth,
                    direct=direct,
                )
            )
    return blocks, diagnostics


def exchangeable_source_data(
    background: pd.DataFrame, detector: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split A_bg into disjoint candidate/reference pools from one population."""
    record_ids = background["record_id"].to_numpy(dtype=np.uint64)
    mixed = (
        record_ids * np.uint64(11400714819323198485)
        + np.uint64(14029467366897019727)
    )
    order = np.argsort(mixed, kind="mergesort")
    candidate_rows = order[:1_049]
    reference_rows = order[1_049:]
    candidates = background.iloc[candidate_rows].reset_index(drop=True).copy()
    reference = background.iloc[reference_rows].reset_index(drop=True).copy()
    b_self = candidates.copy()
    b_self["is_rare"] = bench.shadow.public_rare_mask(candidates, detector).astype(int)
    if set(candidates["record_id"]) & set(reference["record_id"]):
        raise RuntimeError("exchangeable source split overlaps")
    return candidates, b_self, reference


def outcome_marginals(frame: pd.DataFrame, *, tau: float) -> dict[str, float]:
    result = {
        "rows": int(len(frame)),
        "onset_rate": float(frame["onset"].mean()),
        "death_rate": float(frame["death"].mean()),
        "censor_time_10_rate": float(np.mean(np.isclose(frame["time"], 10.0))),
        "mean_time": float(frame["time"].mean()),
    }
    for action in ACTIONS:
        result[f"window_hit_{str(action).replace('.', 'p')}"] = float(
            np.mean(np.abs(frame["time"].to_numpy(dtype=float) - action) <= tau)
        )
    return result


class HierarchicalCellModel:
    """Fixed-shrinkage probability model; no threshold or depth is tuned."""

    LEVELS = (
        (("pattern",), 192.0),
        (("route_kind", "pattern"), 128.0),
        (("route_kind", "pattern", "public_stratum"), 96.0),
        (
            (
                "route_kind",
                "pattern",
                "public_stratum",
                "rare_bin",
                "edge_bin",
                "density_bin",
                "support_bin",
            ),
            64.0,
        ),
    )

    def __init__(self) -> None:
        self.global_rate: dict[str, float] = {}
        self.tables: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

    def fit(self, frame: pd.DataFrame) -> "HierarchicalCellModel":
        if frame.empty:
            raise ValueError("empty cell-model training frame")
        for action in ACTIONS:
            suffix = str(action).replace(".", "p")
            for arm in ("m", "c"):
                column = f"{arm}hit_{suffix}"
                # Jeffreys smoothing avoids exact 0/1 utilities in sparse cells.
                self.global_rate[column] = (
                    float(frame[column].sum()) + 0.5
                ) / (len(frame) + 1.0)
                for keys, _ in self.LEVELS:
                    table = (
                        frame.groupby(list(keys), as_index=False, dropna=False)[column]
                        .agg(["sum", "count"])
                        .reset_index()
                    )
                    self.tables[(column, keys)] = table
        return self

    def predict_column(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        estimate = np.full(len(frame), self.global_rate[column], dtype=float)
        work = frame.reset_index(drop=True)
        for keys, weight in self.LEVELS:
            table = self.tables[(column, keys)]
            lookup = work.loc[:, list(keys)].merge(
                table, on=list(keys), how="left", sort=False
            )
            count = lookup["count"].fillna(0.0).to_numpy(dtype=float)
            total = lookup["sum"].fillna(0.0).to_numpy(dtype=float)
            estimate = (total + weight * estimate) / (count + weight)
        return estimate

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        qm: list[np.ndarray] = []
        qc: list[np.ndarray] = []
        for action in ACTIONS:
            suffix = str(action).replace(".", "p")
            qm.append(self.predict_column(frame, f"mhit_{suffix}"))
            qc.append(self.predict_column(frame, f"chit_{suffix}"))
        return np.stack(qm), np.stack(qc)


def fit_ensemble(
    blocks: list[StudyBlock], *, seed: int, models: int = 12
) -> list[HierarchicalCellModel]:
    eligible = [block for block in blocks if block.exact_zero]
    if not eligible:
        raise RuntimeError("no exact-zero blocks for model fitting")
    available = sorted({block.seed for block in eligible})
    rng = np.random.default_rng(seed)
    output: list[HierarchicalCellModel] = []
    # Model zero is the full-data estimate.  The remaining models are grouped
    # subsamples used only to require stability, not to tune a cutoff.
    full = pd.concat([block.frame for block in eligible], ignore_index=True)
    output.append(HierarchicalCellModel().fit(full))
    for _ in range(models - 1):
        chosen = set(
            rng.choice(
                available,
                size=max(2, int(math.ceil(0.75 * len(available)))),
                replace=False,
            ).tolist()
        )
        train = pd.concat(
            [block.frame for block in eligible if block.seed in chosen],
            ignore_index=True,
        )
        output.append(HierarchicalCellModel().fit(train))
    return output


def _model_predictions(
    ensemble: Sequence[HierarchicalCellModel], frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    predictions = [model.predict(frame) for model in ensemble]
    return (
        np.stack([item[0] for item in predictions]),
        np.stack([item[1] for item in predictions]),
    )


def choose_actions(
    ensemble: Sequence[HierarchicalCellModel],
    frame: pd.DataFrame,
    *,
    times: str,
    robustness: str,
    allow_expand: bool,
    prediction_cache: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    qm, qc = (
        prediction_cache
        if prediction_cache is not None
        else _model_predictions(ensemble, frame)
    )
    action_indices = [1] if times == "edge" else list(range(len(ACTIONS)))
    mean_qm = qm.mean(axis=0)
    mean_qc = qc.mean(axis=0)
    lam = 0.10
    chosen = np.full(len(frame), -1, dtype=int)
    for _ in range(32):
        utility = mean_qm[action_indices] - (1.0 - lam) * mean_qc[action_indices]
        best_local = np.argmax(utility, axis=0)
        best_value = utility[best_local, np.arange(len(frame))]
        proposed = np.where(
            best_value > 0.0,
            np.asarray(action_indices, dtype=int)[best_local],
            -1,
        )
        pm = sum(
            mean_qm[action, proposed == action].sum() for action in action_indices
        ) / N_ARM
        pc = sum(
            mean_qc[action, proposed == action].sum() for action in action_indices
        ) / N_ARM
        updated = max(0.0, (pm - pc) / max(1.0 - pc, 1e-9))
        chosen = proposed
        if abs(updated - lam) < 1e-8:
            lam = updated
            break
        lam = updated

    per_model_utility = qm - (1.0 - lam) * qc
    if robustness == "mean":
        robust = per_model_utility.mean(axis=0)
    elif robustness == "q25":
        robust = np.quantile(per_model_utility, 0.25, axis=0)
    elif robustness == "q10":
        robust = np.quantile(per_model_utility, 0.10, axis=0)
    else:
        raise ValueError(robustness)
    allowed_utility = robust[action_indices]
    best_local = np.argmax(allowed_utility, axis=0)
    best_value = allowed_utility[best_local, np.arange(len(frame))]
    chosen = np.where(
        best_value > 0.0,
        np.asarray(action_indices, dtype=int)[best_local],
        -1,
    )
    if not allow_expand:
        base_active = frame["pattern"].ne("111").to_numpy()
        chosen[~base_active] = -1
    predicted = np.full(len(frame), ABSTAIN, dtype=float)
    for action_index, action in enumerate(ACTIONS):
        predicted[chosen == action_index] = action
    return predicted, {
        "lambda": float(lam),
        "expected_pm": float(
            sum(
                mean_qm[action, chosen == action].sum()
                for action in action_indices
            )
            / N_ARM
        ),
        "expected_pc": float(
            sum(
                mean_qc[action, chosen == action].sum()
                for action in action_indices
            )
            / N_ARM
        ),
    }


VARIANTS = {
    "edge_mean_expand": ("edge", "mean", True),
    "edge_q25_expand": ("edge", "q25", True),
    "edge_q10_expand": ("edge", "q10", True),
    "edge_q25_noexpand": ("edge", "q25", False),
    "multitime_mean_expand": ("multi", "mean", True),
    "multitime_q25_expand": ("multi", "q25", True),
    "multitime_q10_expand": ("multi", "q10", True),
    "multitime_q25_noexpand": ("multi", "q25", False),
}


def score_block_variants(
    block: StudyBlock,
    ensemble: Sequence[HierarchicalCellModel],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    base_predicted = np.where(block.frame["pattern"].ne("111"), EDGE, ABSTAIN)
    base = bench.score_block(base_predicted, block.truth, cfg)
    common = {
        "split": block.split,
        "seed": block.seed,
        "scenario": block.scenario,
        "official_target": block.official_target,
        "internal_target": block.internal_target,
        "exact_zero": block.exact_zero,
        "near_zero": block.near_zero,
        "direct_max": max(block.direct.values()),
        "base_power": base["power"],
        "base_all": base["all"],
        "base_rare": base["rare"],
    }
    rows = [
        {
            **common,
            "strategy": "v7_unanimous_complement",
            "active_rows": int(np.count_nonzero(base_predicted != ABSTAIN)),
            "changed_rows": 0,
            "lambda": np.nan,
            **base,
            "delta": 0.0,
            "delta_all": 0.0,
            "delta_rare": 0.0,
        }
    ]
    prediction_cache = _model_predictions(ensemble, block.frame)
    for strategy, (times, robustness, allow_expand) in VARIANTS.items():
        predicted, diagnostic = choose_actions(
            ensemble,
            block.frame,
            times=times,
            robustness=robustness,
            allow_expand=allow_expand,
            prediction_cache=prediction_cache,
        )
        metrics = bench.score_block(predicted, block.truth, cfg)
        rows.append(
            {
                **common,
                "strategy": strategy,
                "active_rows": int(np.count_nonzero(predicted != ABSTAIN)),
                "changed_rows": int(np.count_nonzero(~np.isclose(predicted, base_predicted))),
                "lambda": diagnostic["lambda"],
                **metrics,
                "delta": metrics["power"] - base["power"],
                "delta_all": metrics["all"] - base["all"],
                "delta_rare": metrics["rare"] - base["rare"],
            }
        )
    return rows


def discovery_oof_scores(
    blocks: list[StudyBlock], cfg: dict[str, Any]
) -> pd.DataFrame:
    seeds = sorted({block.seed for block in blocks})
    rows: list[dict[str, Any]] = []
    for fold in range(4):
        test_seeds = {seed for index, seed in enumerate(seeds) if index % 4 == fold}
        train = [block for block in blocks if block.seed not in test_seeds]
        test = [block for block in blocks if block.seed in test_seeds]
        ensemble = fit_ensemble(train, seed=20270800 + fold, models=8)
        for block in test:
            rows.extend(score_block_variants(block, ensemble, cfg))
    return pd.DataFrame(rows)


def direct_scores(
    train: list[StudyBlock],
    test: list[StudyBlock],
    cfg: dict[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[HierarchicalCellModel]]:
    ensemble = fit_ensemble(train, seed=seed, models=12)
    rows: list[dict[str, Any]] = []
    for block in test:
        rows.extend(score_block_variants(block, ensemble, cfg))
    return pd.DataFrame(rows), ensemble


def paired_summary(scores: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition == "exact_zero":
        chosen = scores.loc[scores["exact_zero"]].copy()
    elif condition == "near_zero":
        chosen = scores.loc[scores["near_zero"]].copy()
    elif condition == "all_blocks":
        chosen = scores.copy()
    elif condition == "baseline_positive_exact":
        chosen = scores.loc[scores["exact_zero"] & (scores["base_power"] > EPS)].copy()
    elif condition == "baseline_band_exact":
        chosen = scores.loc[
            scores["exact_zero"]
            & scores["base_power"].between(0.02, 0.25, inclusive="both")
        ].copy()
    else:
        raise ValueError(condition)
    if chosen.empty:
        return pd.DataFrame()
    # Two release scenarios share the same challenge truth.  Average them at
    # seed-target grain before uncertainty summaries to avoid pseudo-replication.
    cases = (
        chosen.groupby(
            ["split", "seed", "official_target", "internal_target", "strategy"],
            as_index=False,
        )
        .agg(
            scenarios=("scenario", "nunique"),
            power=("power", "mean"),
            all=("all", "mean"),
            rare=("rare", "mean"),
            delta=("delta", "mean"),
            delta_all=("delta_all", "mean"),
            delta_rare=("delta_rare", "mean"),
            base_power=("base_power", "mean"),
            active_rows=("active_rows", "mean"),
            changed_rows=("changed_rows", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (split, strategy), group in cases.groupby(["split", "strategy"], sort=True):
        # The leaderboard weights the seven targets equally.  Exact-zero
        # posterior acceptance rates differ by target, so a plain case mean
        # would over-weight the easiest target to condition.  Give every target
        # equal total mass and every accepted draw within a target equal mass.
        per_target_n = group.groupby("official_target")["delta"].transform("size")
        weight = 1.0 / per_target_n.to_numpy(dtype=float)
        weight /= weight.sum()

        def weighted(column: str) -> float:
            return float(np.sum(weight * group[column].to_numpy(dtype=float)))

        rows.append(
            {
                "split": split,
                "strategy": strategy,
                "cases": len(group),
                "targets": int(group["official_target"].nunique()),
                "seeds": int(group["seed"].nunique()),
                "mean_power": weighted("power"),
                "mean_base": weighted("base_power"),
                "mean_delta": weighted("delta"),
                "median_delta": float(group["delta"].median()),
                "p10_delta": float(np.quantile(group["delta"], 0.10)),
                "p25_delta": float(np.quantile(group["delta"], 0.25)),
                "min_delta": float(group["delta"].min()),
                "win_rate": float(np.sum(weight * (group["delta"] > EPS))),
                "loss_rate": float(np.sum(weight * (group["delta"] < -EPS))),
                "mean_delta_all": weighted("delta_all"),
                "mean_delta_rare": weighted("delta_rare"),
                "mean_active": weighted("active_rows"),
                "mean_changed": weighted("changed_rows"),
                "condition": condition,
            }
        )
    return pd.DataFrame(rows)


def summarize_all(scores: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        paired_summary(scores, condition)
        for condition in (
            "exact_zero",
            "near_zero",
            "all_blocks",
            "baseline_positive_exact",
            "baseline_band_exact",
        )
    ]
    return pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)


def choose_discovery(summary: pd.DataFrame) -> str:
    candidates = summary.loc[
        summary["split"].eq("discovery")
        & summary["condition"].eq("exact_zero")
        & summary["strategy"].ne("v7_unanimous_complement")
    ].copy()
    candidates["objective"] = candidates["mean_delta"] + 0.50 * candidates["p25_delta"]
    guarded = candidates.loc[
        (candidates["p25_delta"] >= -0.02)
        & (candidates["loss_rate"] <= 0.35)
    ]
    if guarded.empty:
        return "v7_unanimous_complement"
    return str(
        guarded.sort_values(
            ["objective", "mean_delta", "p10_delta"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]["strategy"]
    )


def _actual_production_features(
    background: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = {
        "v4": pd.read_csv(
            official.TEAM / "attack-candidate-v4" / "F_aia.csv"
        ),
        "independent": pd.read_csv(
            official.TEAM
            / "attack-candidate-daybreak-independent"
            / "F_aia.csv"
        ),
        "routed": pd.read_csv(
            official.TEAM
            / "attack-candidate-v6-routed-guided-probe"
            / "F_aia.csv"
        ),
    }
    releases = {
        target: pd.read_csv(official.PRODUCTION_TARGETS / f"C_{target}.csv")
        for target in ZERO_TARGETS
    }
    linkable_all, _ = filter_linkable_releases(
        {str(key): value for key, value in releases.items()}
    )
    pieces: list[pd.DataFrame] = []
    reconciliation: dict[str, Any] = {}
    for target in ZERO_TARGETS:
        challenge = pd.read_csv(
            official.PRODUCTION_TARGETS / f"aia_challenge_{target}.csv"
        ).sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
        v4_time = bench.aligned_time(predictions["v4"], str(target), challenge)
        independent_time = bench.aligned_time(
            predictions["independent"], str(target), challenge
        )
        routed_time = bench.aligned_time(
            predictions["routed"], str(target), challenge
        )
        frame = official._route_feature_frame(
            official_target=target,
            internal_target=str(target),
            challenge=challenge,
            release=releases[target],
            background=background,
            linkable_release=linkable_all[str(target)],
            v4_time=v4_time,
            independent_time=independent_time,
        )
        submitted_routed = ~np.isclose(routed_time, ABSTAIN)
        calculated_routed = frame["routed_active"].to_numpy(dtype=bool)
        # The existing production file was built against the retained toy
        # background.  Preserve its submitted vote exactly, while recomputing
        # only public continuous features against the now-confirmed actual A_bg.
        frame["routed_active"] = submitted_routed.astype(float)
        frame["reject_votes"] = (
            3
            - frame["v4_active"].astype(int)
            - frame["independent_active"].astype(int)
            - frame["routed_active"].astype(int)
        ).astype(float)
        frame["pattern"] = (
            frame["v4_active"].astype(int).astype(str)
            + frame["independent_active"].astype(int).astype(str)
            + frame["routed_active"].astype(int).astype(str)
        )
        # Retain the exact submitted time predictions for conditioning the
        # posterior worlds on the three observed official zero scores.  These
        # are public attack outputs, not hidden truth.
        frame["direct_time_v4"] = v4_time
        frame["direct_time_independent"] = independent_time
        frame["direct_time_routed"] = routed_time
        reconciliation[str(target)] = {
            "submitted_routed_active": int(submitted_routed.sum()),
            "actual_bg_recalculated_active": int(calculated_routed.sum()),
            "mask_difference_if_rebuilt": int(
                np.count_nonzero(submitted_routed != calculated_routed)
            ),
            "policy_uses_submitted_vote": True,
        }
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True), reconciliation


def _production_augmented(
    detector: dict[str, Any], cfg: dict[str, Any], background: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    production, reconciliation = _actual_production_features(background)
    releases = {
        target: pd.read_csv(official.PRODUCTION_TARGETS / f"C_{target}.csv")
        for target in ZERO_TARGETS
    }
    linkable_all, _ = filter_linkable_releases(
        {str(key): value for key, value in releases.items()}
    )
    pieces: list[pd.DataFrame] = []
    tau = float(cfg["aia"]["tau_time"])
    for target in ZERO_TARGETS:
        block = production.loc[production["official_target"].eq(target)].reset_index(
            drop=True
        )
        challenge = pd.read_csv(
            official.PRODUCTION_TARGETS / f"aia_challenge_{target}.csv"
        ).sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
        augmented = _augment_public_features(
            block,
            challenge,
            linkable_all[str(target)],
            background,
            detector,
            tau=tau,
        )
        # Response columns are required by the model's schema but never read at
        # prediction time.  Keep explicit zeros to make accidental truth use
        # detectable in code review.
        for action in ACTIONS:
            suffix = str(action).replace(".", "p")
            augmented[f"mhit_{suffix}"] = 0
            augmented[f"chit_{suffix}"] = 0
        pieces.append(augmented)
    return pd.concat(pieces, ignore_index=True), reconciliation


def production_mask(
    production: pd.DataFrame,
    ensemble: Sequence[HierarchicalCellModel],
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strategy == "v7_unanimous_complement":
        spec = None
    else:
        spec = VARIANTS[strategy]
    pieces: list[pd.DataFrame] = []
    changes: list[pd.DataFrame] = []
    for target in ZERO_TARGETS:
        block = production.loc[production["official_target"].eq(target)].reset_index(
            drop=True
        )
        base = np.where(block["pattern"].ne("111"), EDGE, ABSTAIN)
        if spec is None:
            predicted = base.copy()
        else:
            predicted, _ = choose_actions(
                ensemble,
                block,
                times=spec[0],
                robustness=spec[1],
                allow_expand=spec[2],
            )
        part = pd.DataFrame(
            {
                "target_k": target,
                "challenge_row_id": block["challenge_row_id"].astype(int),
                "time_hat": predicted,
                "v7_time_hat": base,
                "pattern": block["pattern"],
                "public_stratum": block["public_stratum"],
                "rare_bin": block["rare_bin"].astype(int),
                "edge_bin": block["edge_bin"].astype(int),
                "density_bin": block["density_bin"].astype(int),
            }
        )
        pieces.append(part)
        changes.append(part.loc[~np.isclose(predicted, base)].copy())
    return pd.concat(pieces, ignore_index=True), pd.concat(changes, ignore_index=True)


def own_challenge_validation(
    ensemble: Sequence[HierarchicalCellModel],
    strategy: str,
    *,
    background: pd.DataFrame,
    b_self: pd.DataFrame,
    detector: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate q_m only on the 163 publicly identifiable own-cohort members.

    Target 24 is not a scored attack target for team 24 and its 163 controls
    have no distributed outcomes.  Consequently this is a one-arm transport
    diagnostic, never a substitute for an R or p_c evaluation.
    """
    target = 24
    challenge = pd.read_csv(
        official.PRODUCTION_TARGETS / f"aia_challenge_{target}.csv"
    ).sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
    releases = {
        int(path.stem.split("_", 1)[1]): pd.read_csv(path)
        for path in official.PRODUCTION_TARGETS.glob("C_*.csv")
    }
    linkable, _ = filter_linkable_releases(
        {str(key): value for key, value in releases.items()}
    )
    source_frames = {
        "v4": pd.read_csv(official.TEAM / "attack-candidate-v4" / "F_aia.csv"),
        "independent": pd.read_csv(
            official.TEAM
            / "attack-candidate-daybreak-independent"
            / "F_aia.csv"
        ),
        "routed": pd.read_csv(
            official.TEAM
            / "attack-candidate-v6-routed-guided-probe"
            / "F_aia.csv"
        ),
    }
    v4_time = bench.aligned_time(source_frames["v4"], str(target), challenge)
    independent_time = bench.aligned_time(
        source_frames["independent"], str(target), challenge
    )
    routed_time = bench.aligned_time(
        source_frames["routed"], str(target), challenge
    )
    # Official target 6 selects the generic copula feature calculation.  The
    # submitted routed vote is restored immediately below; no target-6 mask is
    # transported into this diagnostic.
    frame = official._route_feature_frame(
        official_target=6,
        internal_target=str(target),
        challenge=challenge,
        release=releases[target],
        background=background,
        linkable_release=linkable[str(target)],
        v4_time=v4_time,
        independent_time=independent_time,
    )
    frame["routed_active"] = (~np.isclose(routed_time, ABSTAIN)).astype(float)
    frame["reject_votes"] = (
        3
        - frame["v4_active"].astype(int)
        - frame["independent_active"].astype(int)
        - frame["routed_active"].astype(int)
    ).astype(float)
    frame["pattern"] = (
        frame["v4_active"].astype(int).astype(str)
        + frame["independent_active"].astype(int).astype(str)
        + frame["routed_active"].astype(int).astype(str)
    )
    frame["route_kind"] = "own24"
    frame = _augment_public_features(
        frame,
        challenge,
        linkable[str(target)],
        background,
        detector,
        tau=float(cfg["aia"]["tau_time"]),
    )
    for action in ACTIONS:
        suffix = str(action).replace(".", "p")
        frame[f"mhit_{suffix}"] = 0
        frame[f"chit_{suffix}"] = 0
    base = np.where(frame["pattern"].ne("111"), EDGE, ABSTAIN)
    if strategy == "v7_unanimous_complement":
        candidate = base.copy()
    else:
        times, robustness, allow_expand = VARIANTS[strategy]
        candidate, _ = choose_actions(
            ensemble,
            frame,
            times=times,
            robustness=robustness,
            allow_expand=allow_expand,
        )

    qi = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
    own = b_self.copy()
    lookup = {tuple(row): index for index, row in enumerate(own[qi].to_numpy())}
    matched = np.asarray(
        [lookup.get(tuple(row), -1) for row in challenge[qi].to_numpy()], dtype=int
    )
    member = matched >= 0
    if int(member.sum()) != N_ARM:
        raise RuntimeError(f"own challenge expected 163 known members, got {member.sum()}")
    truth_time = np.full(len(challenge), np.nan, dtype=float)
    truth_rare = np.zeros(len(challenge), dtype=bool)
    truth_time[member] = own.iloc[matched[member]]["time"].to_numpy(dtype=float)
    truth_rare[member] = (
        own.iloc[matched[member]]["is_rare"].astype(bool).to_numpy()
    )
    tau = float(cfg["aia"]["tau_time"])
    base_hit = member & (np.abs(base - truth_time) <= tau)
    candidate_hit = member & (np.abs(candidate - truth_time) <= tau)
    known_rare = member & truth_rare
    rows = pd.DataFrame(
        {
            "challenge_row_id": challenge["challenge_row_id"].astype(int),
            "known_member": member.astype(int),
            "known_rare": truth_rare.astype(int),
            "pattern": frame["pattern"],
            "v7_zero_policy_time": base,
            "candidate_time": candidate,
            "changed": (~np.isclose(base, candidate)).astype(int),
            "base_member_hit": base_hit.astype(int),
            "candidate_member_hit": candidate_hit.astype(int),
        }
    )
    report = {
        "scope": "one-arm q_m transport diagnostic on own target 24; controls unknown",
        "known_members": int(member.sum()),
        "known_rare_members": int(known_rare.sum()),
        "base_member_window_rate": float(base_hit[member].mean()),
        "candidate_member_window_rate": float(candidate_hit[member].mean()),
        "member_window_rate_delta": float(
            candidate_hit[member].mean() - base_hit[member].mean()
        ),
        "base_rare_member_window_rate": float(base_hit[known_rare].mean()),
        "candidate_rare_member_window_rate": float(
            candidate_hit[known_rare].mean()
        ),
        "rare_member_window_rate_delta": float(
            candidate_hit[known_rare].mean() - base_hit[known_rare].mean()
        ),
        "changed_all_rows": int(np.count_nonzero(~np.isclose(base, candidate))),
        "changed_known_members": int(
            np.count_nonzero(~np.isclose(base[member], candidate[member]))
        ),
        "limitation": (
            "p_c and R cannot be computed because the 163 target-24 control outcomes "
            "are not distributed"
        ),
    }
    return report, rows


def _source_audit(
    background: pd.DataFrame,
    detector: dict[str, Any],
    production: pd.DataFrame,
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    release_copy: dict[str, Any] = {}
    provenance_path = (
        ROOT / "local-output" / "attack-v8" / "actual-data" / "background_provenance.csv"
    )
    provenance = pd.read_csv(provenance_path)
    for target in ZERO_TARGETS:
        subset = provenance.loc[
            provenance["target"].astype(int).eq(target)
            & provenance["view"].eq("stable3_numeric")
        ]
        release_copy[str(target)] = {
            str(row.reference): int(row.matched_rows)
            for row in subset.itertuples(index=False)
        }
        release_copy[str(target)]["rows"] = 1049
    v7 = pd.read_csv(
        ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v7-feedback-stack"
        / "F_aia.csv"
    )
    mismatches: dict[str, int] = {}
    for target in ZERO_TARGETS:
        block = production.loc[production["official_target"].eq(target)].copy()
        block = block.sort_values("challenge_row_id", kind="mergesort")
        expected = np.where(block["pattern"].ne("111"), EDGE, ABSTAIN)
        actual = (
            v7.loc[v7["target_k"].astype(int).eq(target)]
            .sort_values("challenge_row_id", kind="mergesort")["time_hat"]
            .to_numpy(dtype=float)
        )
        mismatches[str(target)] = int(np.count_nonzero(~np.isclose(expected, actual)))
    challenge_columns = set(
        pd.read_csv(
            official.PRODUCTION_TARGETS / "aia_challenge_19.csv", nrows=1
        ).columns
    )
    detector_columns = list(detector["columns"])
    return {
        "official_exact_zero_definition": (
            "max of official score_aia power for v4, independent, routed equals 0; "
            "posterior draws use the same score function and each source's exact "
            "submitted time predictions"
        ),
        "three_mask_route_reconciliation": reconciliation,
        "v7_boolean_reconstruction_mismatches": mismatches,
        "production_release_tolerant_stable3_provenance": release_copy,
        "fixed_public_release_design": {
            "actual_public_challenge_rows_used": True,
            "actual_public_release_rows_used": True,
            "background_reference": "hash-verified actual A_bg",
            "background_occupancy_features": "all5 and stable3 inverse-Voronoi support",
            "reason": (
                "holding actual QI/release geometry fixed preserves C18/C19 donor "
                "occupancy and avoids inventing seven other hometown cohorts from B24"
            ),
        },
        "rare_mask": {
            "truth_detector_columns": detector_columns,
            "missing_from_challenge": sorted(set(detector_columns) - challenge_columns),
            "policy_feature": "visible-submatrix Mahalanobis rank only",
            "caveat": (
                "imputed member truth uses B_self and imputed control truth uses the "
                "full detector; rare_bin is only a public proxy and is never treated "
                "as observed target truth"
            ),
        },
    }


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    actual = load_actual_inputs(ACTUAL_DATA, official.PRODUCTION_TARGETS)
    candidates = actual.candidates.reset_index(drop=True)
    b_self = actual.self_frame.reset_index(drop=True)
    background = actual.background.reset_index(drop=True)
    toy_background = pd.read_csv(bench.DIST / "A_bg.csv").reset_index(drop=True)
    detector = actual.schema["rare_detector"]
    cfg = bench.load_config(bench.KIT_CONFIG_PATH)
    background_with_rare = background.copy()
    background_with_rare["is_rare"] = bench.shadow.public_rare_mask(
        background, detector
    ).astype(int)
    discovery_members, private_members, member_pool_audit = _posterior_member_pools(
        candidates,
        b_self,
        actual.known_private,
        actual.self_challenge,
    )
    production, reconciliation = _production_augmented(detector, cfg, background)

    # The seven public challenges themselves are held fixed.  Truth is drawn
    # from the officially published matched-block design and empirical
    # hometown-private/control outcome reservoirs.  This avoids pretending
    # that the nationwide B table is seven other teams' private cohorts.
    discovery, discovery_diag = build_actual_posterior_blocks(
        "discovery",
        DISCOVERY,
        production=production,
        challenges=actual.challenges,
        member_primary=discovery_members,
        member_fallback=discovery_members,
        background_with_rare=background_with_rare,
        background_reference=background,
        cfg=cfg,
        control_fold=0,
    )
    heldout, heldout_diag = build_actual_posterior_blocks(
        "heldout",
        HELDOUT,
        production=production,
        challenges=actual.challenges,
        member_primary=private_members,
        member_fallback=private_members,
        background_with_rare=background_with_rare,
        background_reference=background,
        cfg=cfg,
        control_fold=1,
    )

    discovery_scores = discovery_oof_scores(discovery, cfg)
    heldout_scores, _ = direct_scores(
        discovery, heldout, cfg, seed=20270891
    )
    initial_scores = pd.concat([discovery_scores, heldout_scores], ignore_index=True)
    initial_scores.to_csv(OUT / "checkpoint_discovery_heldout_scores.csv", index=False)
    initial_summary = summarize_all(initial_scores)
    initial_summary.to_csv(
        OUT / "checkpoint_discovery_heldout_summary.csv", index=False
    )
    selected = choose_discovery(initial_summary)

    fresh, fresh_diag = build_actual_posterior_blocks(
        "fresh",
        FRESH,
        production=production,
        challenges=actual.challenges,
        member_primary=private_members,
        member_fallback=private_members,
        background_with_rare=background_with_rare,
        background_reference=background,
        cfg=cfg,
        control_fold=2,
    )
    fresh_scores, production_ensemble = direct_scores(
        [*discovery, *heldout], fresh, cfg, seed=20270991
    )
    fresh_scores.to_csv(OUT / "checkpoint_fresh_scores.csv", index=False)
    exchangeable, exchangeable_diag = build_actual_posterior_blocks(
        "exchangeable_fresh",
        EXCHANGEABLE,
        production=production,
        challenges=actual.challenges,
        member_primary=background_with_rare,
        member_fallback=background_with_rare,
        background_with_rare=background_with_rare,
        background_reference=background,
        cfg=cfg,
        control_fold=2,
        exchangeable=True,
    )
    exchangeable_scores, _ = direct_scores(
        [*discovery, *heldout], exchangeable, cfg, seed=20271091
    )
    exchangeable_scores.to_csv(
        OUT / "checkpoint_exchangeable_scores.csv", index=False
    )
    scores = pd.concat(
        [initial_scores, fresh_scores, exchangeable_scores], ignore_index=True
    )
    summary = summarize_all(scores)
    scores.to_csv(OUT / "checkpoint_all_scores.csv", index=False)
    summary.to_csv(OUT / "checkpoint_all_summary.csv", index=False)

    mask, changes = production_mask(production, production_ensemble, selected)
    # The released attack files omit the attacker's own target 24, so a
    # three-vote target-24 transport audit cannot be reconstructed from those
    # files.  Root's independent, grouped-CV target-24 membership audit covers
    # that question and found no above-null classifier.  Do not synthesize a
    # routed vote merely to make this optional diagnostic run.
    own_report = {
        "status": "not_run",
        "reason": "submitted attack files do not contain self target 24",
        "independent_root_audit": (
            "a separate membership-classifier CV was indistinguishable from null; "
            "it is not validation of this attack policy"
        ),
    }
    own_rows = pd.DataFrame(
        columns=["challenge_row_id", "known_member", "candidate_time"]
    )
    audit = _source_audit(background, detector, production, reconciliation)
    allowed = {9.0, 9.5, 10.0, 11.0}
    qa = {
        "rows": len(mask),
        "unique_keys": int(mask[["target_k", "challenge_row_id"]].drop_duplicates().shape[0]),
        "finite": bool(np.isfinite(mask["time_hat"].to_numpy(dtype=float)).all()),
        "nonnegative": bool((mask["time_hat"].to_numpy(dtype=float) >= 0.0).all()),
        "allowed_values": sorted(mask["time_hat"].astype(float).unique().tolist()),
        "allowed_values_pass": set(mask["time_hat"].astype(float).unique()).issubset(allowed),
        "changed_rows_from_v7": len(changes),
        "changed_by_target": {
            str(int(key)): int(value)
            for key, value in changes.groupby("target_k").size().items()
        },
    }

    def record(split: str, condition: str) -> dict[str, Any] | None:
        selected_rows = summary.loc[
            summary["split"].eq(split)
            & summary["condition"].eq(condition)
            & summary["strategy"].eq(selected)
        ]
        return selected_rows.iloc[0].to_dict() if len(selected_rows) else None

    held = record("heldout", "exact_zero")
    held_positive = record("heldout", "baseline_positive_exact")
    fresh_record = record("fresh", "exact_zero")
    fresh_positive = record("fresh", "baseline_positive_exact")
    exchange_record = record("exchangeable_fresh", "exact_zero")
    exchange_positive = record(
        "exchangeable_fresh", "baseline_positive_exact"
    )
    held_pass = bool(
        held
        and held_positive
        and held["mean_delta"] >= 0.005
        and held["win_rate"] >= 0.60
        and held["p10_delta"] >= -0.03
        and held_positive["mean_delta"] >= 0.0
        and held_positive["p10_delta"] >= -0.05
    )
    fresh_pass = bool(
        fresh_record
        and fresh_positive
        and fresh_record["mean_delta"] > 0.0
        and fresh_record["win_rate"] >= 0.55
        and fresh_record["p10_delta"] >= -0.03
        and fresh_positive["mean_delta"] >= 0.0
        and fresh_positive["p10_delta"] >= -0.05
    )
    exchange_pass = bool(
        exchange_record
        and exchange_positive
        and exchange_record["mean_delta"] >= 0.0
        and exchange_record["p10_delta"] >= -0.05
        and exchange_positive["mean_delta"] >= 0.0
        and exchange_positive["p10_delta"] >= -0.05
    )
    adoption = bool(
        selected != "v7_unanimous_complement"
        and held_pass
        and fresh_pass
        and exchange_pass
    )
    report = {
        "purpose": (
            "posterior-predictive sensitivity analysis for nonlinear contrastive "
            "zero-target policy; not empirical target validation or a leaderboard forecast"
        ),
        "as_of": "2026-09-05 Asia/Tokyo",
        "selection": {
            "locked_from_discovery_oof": selected,
            "heldout_exact": held,
            "heldout_baseline_positive": held_positive,
            "fresh_exact": fresh_record,
            "fresh_baseline_positive": fresh_positive,
            "exchangeable_exact": exchange_record,
            "exchangeable_baseline_positive": exchange_positive,
            "heldout_gate_pass": held_pass,
            "fresh_gate_pass": fresh_pass,
            "exchangeable_gate_pass": exchange_pass,
            "adopt": adoption,
            "posterior_score1_directional_delta_if_all_seven_changed": (
                float(fresh_record["mean_delta"]) * 7.0 / 20.0
                if fresh_record
                else None
            ),
        },
        "design": {
            "objective": "Dinkelbach q_m-(1-R)q_c for R=(p_m-p_c)/(1-p_c)",
            "actions": [*ACTIONS, ABSTAIN],
            "truth_model": (
                "actual public challenge rows; half-member draw within every "
                "sex x 5-year-age block; fixed 13 rare / 16 onset / 4 rare-onset "
                "per arm; public-QI-conditioned empirical outcome assignment"
            ),
            "discovery_seeds": list(DISCOVERY),
            "heldout_seeds": list(HELDOUT),
            "fresh_seeds": list(FRESH),
            "exchangeable_fresh_seeds": list(EXCHANGEABLE),
            "exact_zero_blocks": {
                split: int(sum(block.exact_zero for block in blocks))
                for split, blocks in (
                    ("discovery", discovery),
                    ("heldout", heldout),
                    ("fresh", fresh),
                    ("exchangeable_fresh", exchangeable),
                )
            },
            "feedback_conditioning_sensitivity": {
                condition: record("fresh", condition)
                for condition in ("exact_zero", "near_zero", "all_blocks")
            },
            "population_outcome_marginals": {
                "B_actual_full": outcome_marginals(
                    candidates, tau=float(cfg["aia"]["tau_time"])
                ),
                "known_private_163": outcome_marginals(
                    private_members, tau=float(cfg["aia"]["tau_time"])
                ),
                "discovery_hometown_reservoir": outcome_marginals(
                    discovery_members, tau=float(cfg["aia"]["tau_time"])
                ),
                "A_bg_control_source": outcome_marginals(
                    background, tau=float(cfg["aia"]["tau_time"])
                ),
            },
            "member_pool_audit": member_pool_audit,
            "posterior_donor_diagnostics": {
                "discovery": discovery_diag,
                "heldout": heldout_diag,
                "fresh": fresh_diag,
                "exchangeable_fresh": exchangeable_diag,
            },
            "target_weighting": (
                "each of the seven targets receives equal total weight after "
                "conditioning, regardless of exact-zero acceptance rate"
            ),
        },
        "actual_input_provenance": actual.provenance,
        "source_audit": audit,
        "own_challenge_diagnostic": own_report,
        "production_research_mask": qa,
        "limitations": [
            "Private production truth is unavailable; shadow deltas are directional evidence only.",
            "Conditioning on the three exact-zero observed attacks is faithful but reduces effective sample size.",
            "The 13 truth-known private rare rows are the only hometown reservoir supporting the published rare composition, so rare outcomes cannot be person-disjoint across splits.",
            "Actual QIs and releases are fixed across streams; membership and outcomes are imputed from common empirical reservoirs.",
            "Held-out and fresh only separate Monte-Carlo draws and control-donor folds; they are not held-out features, real target outcomes, or empirical leaderboard validation.",
            "The public rare proxy omits BMI and FPG and cannot reproduce the scoring rare mask exactly.",
            "The exchangeable A_bg world is a stress test for fixed member/control outcome differences; it is not claimed to reproduce each private production generator.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }

    scores.to_csv(OUT / "scores.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    mask.to_csv(OUT / "production_research_mask.csv", index=False)
    changes.to_csv(OUT / "production_changes.csv", index=False)
    own_rows.to_csv(OUT / "own24_known_member_audit.csv", index=False)
    conditions = pd.DataFrame(
        [
            {
                "split": block.split,
                "seed": block.seed,
                "scenario": block.scenario,
                "official_target": block.official_target,
                "internal_target": block.internal_target,
                "exact_zero": block.exact_zero,
                "near_zero": block.near_zero,
                "direct_v4": block.direct["v4"],
                "direct_independent": block.direct["independent"],
                "direct_routed": block.direct["routed"],
            }
            for block in [*discovery, *heldout, *fresh, *exchangeable]
        ]
    )
    conditions.to_csv(OUT / "block_conditions.csv", index=False)
    report = bench.shadow.clean_json(report)
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(
        mask[["target_k", "challenge_row_id", "time_hat"]]
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()
    print(
        json.dumps(
            {
                "selected": selected,
                "adopt": adoption,
                "heldout": held,
                "fresh": fresh_record,
                "changed_rows": len(changes),
                "mask_sha256": digest,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
