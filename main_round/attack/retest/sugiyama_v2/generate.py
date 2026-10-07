"""Generate one joint, synthetic 30-cohort world with the public contest layout.

No attack implementation is imported here. ``is_rare`` is a birth label inherited
from labeled B_self donor strata, never a Mahalanobis detector prediction. The
synthetic identities are distinct; their finite empirical donors are reused.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
            "smoking", "FPG", "time", "onset", "death"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
NUMERIC_KERNEL = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
ACTIVE_TARGETS = tuple(range(10)) + tuple(range(11, 18)) + (22, 25, 26, 27, 28)
TEAM_ID = 10
COHORT_COUNT = 30
MEMBER_QUOTAS = {(0, 0): 138, (0, 1): 12, (1, 0): 9, (1, 1): 4}
DONOR_PARTITION_SEED = 2026100701


def _digest(frame):
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n",
                                      float_format="%.17g").encode("utf-8")).hexdigest()


def _load_sources():
    import attack_workspace as workspace
    originals = workspace.ORIGINALS
    read = lambda name: pd.read_csv(originals / name, float_precision="round_trip")
    return (read("B_self.csv"), read("A_bg.csv"),
            json.loads((originals / "schema.json").read_text(encoding="utf-8-sig")))


def _partition_donors(donors, background, split):
    if split not in ("dev", "holdout"):
        raise ValueError("split must be dev or holdout")
    rng = np.random.default_rng(DONOR_PARTITION_SEED)
    selected = []
    for rare, onset in MEMBER_QUOTAS:
        indices = np.flatnonzero((donors.is_rare.to_numpy(int) == rare)
                                 & (donors.onset.to_numpy(int) == onset))
        if len(indices) < 2:
            raise ValueError(f"At least two labeled donors needed in {(rare, onset)}")
        indices = rng.permutation(indices)
        half = len(indices) // 2
        selected.extend(indices[:half] if split == "dev" else indices[half:])
    selected = np.asarray(sorted(selected), dtype=int)
    background_order = rng.permutation(len(background))
    half = len(background_order) // 2
    background_indices = np.sort(background_order[:half] if split == "dev"
                                 else background_order[half:])
    if len(background_indices) < 2:
        raise ValueError("At least four background source rows required")
    return (donors.iloc[selected].reset_index(drop=True),
            background.iloc[background_indices].reset_index(drop=True),
            selected.tolist(), background_indices.tolist())


def _clean(frame, schema):
    frame = frame[FEATURES].copy()
    for column, bounds in schema["ranges"].items():
        frame[column] = frame[column].clip(*bounds)
    frame["age"] = np.rint(frame.age.clip(*schema["age_range"])).astype(int)
    frame["time"] = frame.time.clip(1 / 365.25, 10)
    for column in ("sex", "smoking", "onset", "death"):
        frame[column] = frame[column].round().clip(0, 1).astype(int)
    frame["death"] = ((frame.onset == 0) & (frame.death == 1)).astype(int)
    if not np.isfinite(frame.drop(columns="prefecture").to_numpy(float)).all():
        raise ValueError("Synthetic population contains nonfinite features")
    return frame


def _kernel(donors, strata, background, rng, schema, *, matching=None):
    """Independent draws from smoothed empirical labeled strata.

    Mix two same-key donors and add correlated continuous noise. This gives
    new feature vectors without pretending to create an independent population.
    The fixed kernel is not selected using attack scores or official feedback.
    """
    strata = np.asarray(strata, dtype=int)
    first = np.empty(len(strata), dtype=int)
    second = np.empty(len(strata), dtype=int)
    donor_keys = list(zip(donors.is_rare.astype(int), donors.onset.astype(int),
                          donors.sex.astype(int), donors.prefecture,
                          ((donors.age - 40) // 5).astype(int)))
    groups = {}
    for index, key in enumerate(donor_keys):
        groups.setdefault(key, []).append(index)
    if matching is None:
        for rare, onset in MEMBER_QUOTAS:
            positions = np.flatnonzero((strata[:, 0] == rare) & (strata[:, 1] == onset))
            choices = np.flatnonzero((donors.is_rare.to_numpy(int) == rare)
                                    & (donors.onset.to_numpy(int) == onset))
            if len(positions) and not len(choices):
                raise ValueError(f"No labeled donors in {(rare, onset)}")
            first[positions] = rng.choice(choices, len(positions), replace=True)
    else:
        if len(matching) != len(strata):
            raise ValueError("Matching rows and synthetic strata differ")
        wanted = list(zip(strata[:, 0], strata[:, 1], matching.sex.astype(int),
                          matching.prefecture, ((matching.age - 40) // 5).astype(int)))
        requested = {}
        for index, key in enumerate(wanted):
            requested.setdefault(key, []).append(index)
        for key, positions in requested.items():
            if key not in groups:
                raise ValueError("No exact conditional donors; matching is never relaxed")
            first[positions] = rng.choice(groups[key], len(positions), replace=True)
    selected = {}
    for index, donor in enumerate(first):
        selected.setdefault(donor_keys[donor], []).append(index)
    for key, positions in selected.items():
        second[positions] = rng.choice(groups[key], len(positions), replace=True)
    frame = donors.iloc[first][FEATURES].reset_index(drop=True).copy()
    other = donors.iloc[second][FEATURES].reset_index(drop=True)
    # Smooth health measurements while preserving categorical donor structure.
    values = background[NUMERIC_KERNEL].to_numpy(float)
    covariance = np.cov(values, rowvar=False)
    covariance = .95 * covariance + .05 * np.diag(np.maximum(np.diag(covariance), 1e-6))
    noise = rng.multivariate_normal(np.zeros(len(NUMERIC_KERNEL)), covariance,
                                    size=len(frame), check_valid="raise")
    frame[NUMERIC_KERNEL] = (.8 * frame[NUMERIC_KERNEL].to_numpy(float)
                            + .2 * other[NUMERIC_KERNEL].to_numpy(float) + .18 * noise)
    low = 40 + ((frame.age.to_numpy(int) - 40) // 5) * 5
    frame["age"] = np.clip(frame.age.to_numpy(float) + rng.normal(0, 1.2, len(frame)), low, low + 4)
    frame["time"] = (.85 * frame.time.to_numpy(float) + .15 * other.time.to_numpy(float)
                     + rng.normal(0, .18, len(frame)))
    frame["onset"] = strata[:, 1]
    frame = _clean(frame, schema)
    frame["is_rare"] = strata[:, 0]
    return frame, first


def _background_draw(background, rng, schema):
    choices = rng.choice(len(background), 5143, replace=True)
    frame = background.iloc[choices][FEATURES].reset_index(drop=True).copy()
    scale = background[NUMERIC_KERNEL].std(ddof=0).to_numpy(float)
    frame[NUMERIC_KERNEL] = frame[NUMERIC_KERNEL].to_numpy(float) + rng.normal(
        size=(len(frame), len(NUMERIC_KERNEL))) * scale * .12
    frame["age"] = frame.age.to_numpy(float) + rng.normal(0, 1.0, len(frame))
    frame["time"] = frame.time.to_numpy(float) + rng.normal(0, .15, len(frame))
    return _clean(frame, schema)


def _topology(rng, rare_onset_shared):
    """25 ordinary rows per pair plus two independent rare regular graphs.

    The two colored graphs preserve each cohort's shared rare onset count and
    allow zero, one or two rare rows per pair, hence overlaps of 25--27 people.
    This is one plausible topology, not a reconstruction of hidden assignments.
    """
    if not 0 <= rare_onset_shared <= 11:
        raise ValueError("Shared rare onset degree must lie in 0..11")
    pairs = {(left, right): [] for left in range(COHORT_COUNT)
             for right in range(left + 1, COHORT_COUNT)}
    for onset, degree in ((1, rare_onset_shared), (0, 11 - rare_onset_shared)):
        permutation = rng.permutation(COHORT_COUNT)
        positions = np.empty(COHORT_COUNT, dtype=int)
        positions[permutation] = np.arange(COHORT_COUNT)
        distances = set(range(1, degree // 2 + 1))
        distances |= {COHORT_COUNT - value for value in distances}
        if degree % 2:
            distances.add(COHORT_COUNT // 2)
        for left, right in pairs:
            distance = int((positions[right] - positions[left]) % COHORT_COUNT)
            if distance in distances:
                pairs[(left, right)].append(onset)
    return pairs


def build_world(seed, profile="mixed", split="dev", *, donors=None,
                background_source=None, schema=None, release_factory=None):
    """Build public inputs plus separate private truth for a full evaluation.

    ``public`` alone is the attack API. Source labels, styles, cohort memberships,
    donor indices and AIA truths stay outside that object. No tokens are read.
    """
    from .releases import PROFILES, make_release

    if profile not in PROFILES:
        raise ValueError(f"Unknown release profile: {profile}")
    if int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if donors is None or background_source is None or schema is None:
        defaults = _load_sources()
        donors = defaults[0] if donors is None else donors
        background_source = defaults[1] if background_source is None else background_source
        schema = defaults[2] if schema is None else schema
    donors = donors.copy()
    if not donors.is_rare.isin([0, 1]).all():
        raise ValueError("Donor is_rare must be a supplied binary birth label")
    labeled, empirical_background, donor_indices, bg_indices = _partition_donors(
        donors, background_source, split)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20261007]))
    onset_rates = {rare: float(labeled.loc[labeled.is_rare.eq(rare), "onset"].mean())
                   for rare in (0, 1)}
    cohorts = {team: [] for team in range(COHORT_COUNT)}
    aia_members = {team: [] for team in range(COHORT_COUNT)}
    private_members = {team: [] for team in range(COHORT_COUNT)}
    strata = []

    def allocate(count, rare, onset=None):
        start = len(strata)
        outcomes = (rng.binomial(1, onset_rates[rare], count) if onset is None
                    else np.repeat(onset, count))
        strata.extend((rare, int(value)) for value in outcomes)
        return list(range(start, start + count))

    rare_onset_per_cohort = int(donors.loc[donors.is_rare.eq(1), "onset"].sum())
    topology = _topology(rng, rare_onset_per_cohort - MEMBER_QUOTAS[(1, 1)])
    for (left, right), rare_onsets in topology.items():
        ids = allocate(25, 0)
        for onset in rare_onsets:
            ids.extend(allocate(1, 1, onset))
        cohorts[left].extend(ids)
        cohorts[right].extend(ids)
    for team in range(COHORT_COUNT):
        members = []
        for (rare, onset), count in MEMBER_QUOTAS.items():
            members.extend(allocate(count, rare, onset))
        other_private = allocate(150, 0)
        aia_members[team] = members
        private_members[team] = members + other_private
        cohorts[team].extend(private_members[team])
        cohorts[team] = rng.permutation(cohorts[team]).astype(int).tolist()
    target_pool_size = len(strata)
    controls = {}
    for team in range(COHORT_COUNT):
        controls[team] = list(range(len(strata), len(strata) + 163))
        strata.extend(strata[index] for index in aia_members[team])
    pool, donor_use = _kernel(labeled, strata[:target_pool_size], empirical_background, rng, schema)
    # Draw controls from the identical kernel conditional on exact matching keys.
    # Their donor choices and noise do not depend on the member's measurements or
    # donor index. Member keys are supported because age stays in its donor bin.
    control_matching = pd.concat([pool.iloc[aia_members[team]] for team in range(COHORT_COUNT)],
                                 ignore_index=True)
    control_frame, control_donor_use = _kernel(labeled, strata[target_pool_size:],
        empirical_background, rng, schema, matching=control_matching)
    pool = pd.concat([pool, control_frame], ignore_index=True)
    donor_use = np.r_[donor_use, control_donor_use]
    for team in range(COHORT_COUNT):
        members = pool.iloc[aia_members[team]]
        ids = controls[team]
        controls_frame = pool.iloc[ids]
        for key in ("sex", "prefecture", "is_rare", "onset"):
            if not np.array_equal(members[key].to_numpy(), controls_frame[key].to_numpy()):
                raise AssertionError("Conditional control key differs")
        if not np.array_equal((members.age.to_numpy() - 40) // 5,
                              (controls_frame.age.to_numpy() - 40) // 5):
            raise AssertionError("Conditional control age band differs")
    pool.insert(0, "pool_id", np.arange(len(pool), dtype=int))
    if pool[FEATURES].duplicated().any() or pool[QI].duplicated().any():
        raise ValueError("Synthetic feature collision; change simulation seed")

    candidate_ids = cohorts[TEAM_ID]
    candidates = pool.iloc[candidate_ids][FEATURES].reset_index(drop=True)
    candidates.insert(0, "record_id", np.arange(1049, dtype=int))
    self_labels = candidates.copy()
    self_labels["is_rare"] = pool.iloc[candidate_ids].is_rare.to_numpy(int)
    background = _background_draw(empirical_background, rng, schema)
    challenge_pool_ids, challenges, aia_truth = {}, {}, {}
    for team in range(COHORT_COUNT):
        order = rng.permutation(326)
        ids = np.asarray(aia_members[team] + controls[team], dtype=int)[order]
        challenge = pool.iloc[ids][QI].reset_index(drop=True)
        challenge.insert(0, "challenge_row_id", np.arange(326, dtype=int))
        truth = pool.iloc[ids][["onset", "time", "is_rare"]].reset_index(drop=True)
        truth.insert(0, "label", np.asarray(["member"] * 163 + ["control"] * 163)[order])
        truth.insert(0, "challenge_row_id", np.arange(326, dtype=int))
        challenges[team], aia_truth[team] = challenge, truth
        challenge_pool_ids[team] = ids.tolist()
    styles = tuple(PROFILES[profile])
    if not styles:
        raise ValueError("Release profile must contain at least one style")
    assignments = [styles[index % len(styles)] for index in range(len(ACTIVE_TARGETS))]
    assignments = rng.permutation(assignments).tolist()
    release = make_release if release_factory is None else release_factory
    targets = {}
    for team, style in zip(ACTIVE_TARGETS, assignments):
        frame = pool.iloc[cohorts[team]][FEATURES].reset_index(drop=True)
        target_seed = int(np.random.SeedSequence([seed, team, 6026]).generate_state(1)[0])
        targets[team] = {"release": release(frame, background, style, target_seed, schema),
                         "challenge": challenges[team]}
    known = set(candidate_ids) & set(challenge_pool_ids[TEAM_ID])
    if known != set(aia_members[TEAM_ID]) or len(known) != 163:
        raise AssertionError("Public self challenge must prove exactly 163 private rows")
    reuse = Counter(donor_use.tolist())
    metadata = {
        "generator": "sugiyama_v2_joint", "version": 1, "seed": int(seed),
        "profile": profile, "split": split, "team_id": TEAM_ID,
        "active_targets": list(ACTIVE_TARGETS), "cohort_count": COHORT_COUNT,
        "target_pool_rows": target_pool_size, "control_pool_rows": len(pool) - target_pool_size,
        "background_rows": len(background), "cohort_rows": 1049,
        "cohort_rare": 24, "cohort_shared": 736, "cohort_private": 313,
        "cohort_rare_onset": rare_onset_per_cohort, "pair_shared_range": [25, 27],
        "pair_rare_range": [0, 2],
        "aia_members": 163, "aia_controls": 163, "aia_rare_per_side": 13,
        "aia_onset_per_side": 16, "aia_rare_onset_per_side": 4,
        "self_known_private": 163,
        "matching_keys": ["is_rare", "onset", "sex", "prefecture", "age_5year"],
        "matching_relaxed": False,
        "control_generation": "Identical exact-key conditional donor kernel; no post-generation covariate replacement",
        "target_styles": {str(team): style for team, style in zip(ACTIVE_TARGETS, assignments)},
        "rare_label_source": "Synthetic birth labels inherited from supplied B_self is_rare strata; not detector predictions",
        "donor_partition_seed": DONOR_PARTITION_SEED,
        "labeled_donor_rows": len(labeled), "background_donor_rows": len(empirical_background),
        "labeled_source_sha256": _digest(donors),
        "background_source_sha256": _digest(background_source[FEATURES]),
        "donor_rows_used": len(reuse), "max_labeled_donor_reuse": max(reuse.values()),
        "limitations": [
            "Synthetic joint topology, birth labels and matching quotas reproduce public structure, not hidden official records.",
            "Finite empirical donor profiles are reused with smoothing; synthetic identities are not independent people.",
            "Dev and holdout use disjoint donor rows but are synthetic draws from the same supplied sources.",
            "Release family frequencies are stress scenarios, not estimates of actual competitors' methods.",
            "Private prefecture concentration is not modeled; this topology is one plausible structural scenario.",
        ],
    }
    return {"public": {"team_id": TEAM_ID, "candidates": candidates,
                       "self_labels": self_labels, "background": background,
                       "self_challenge": challenges[TEAM_ID], "targets": targets, "schema": schema},
            "private": {"pool": pool, "cohorts": cohorts, "aia_truth": aia_truth,
                        "candidate_pool_ids": candidate_ids,
                        "challenge_pool_ids": challenge_pool_ids,
                        "background_identity_ids": list(range(-5143, 0)),
                        "self_known_private_pool_ids": sorted(known),
                        "labeled_donor_indices": donor_indices,
                        "background_donor_indices": bg_indices},
            "metadata": metadata}
