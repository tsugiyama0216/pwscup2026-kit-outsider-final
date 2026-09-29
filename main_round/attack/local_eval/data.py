"""Disjoint-person development/holdout shadow cohorts for local attack testing."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
            "smoking", "FPG", "time", "onset", "death"]
QUOTAS = {(False, 0): 138, (False, 1): 12, (True, 0): 9, (True, 1): 4}
PARTITION_SEED = 2026092701
STYLES = ("copy", "rounded", "jitter", "bootstrap", "independent", "strong_jitter", "copula")
PRIMARY_STYLES = ("rounded", "jitter", "strong_jitter", "bootstrap", "copula")


def rare_mask(frame, schema):
    detector = schema["rare_detector"]
    diff = frame[detector["columns"]].to_numpy(float) - np.asarray(detector["mu"])
    inv = np.linalg.pinv(np.asarray(detector["sigma"]))
    distance = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv, diff), 0, None))
    return distance >= float(detector["band"]["lo"])


def matching_keys(frame, rare):
    return list(zip(rare.astype(bool), frame.onset.astype(int), frame.sex.astype(int),
                    frame.prefecture.astype(str), ((frame.age - 40) // 5).astype(int)))


def partition_source(source, schema):
    """Keep matching pairs in each split; no source row appears in both splits."""
    rng = np.random.default_rng(PARTITION_SEED)
    groups = defaultdict(list)
    for idx, key in enumerate(matching_keys(source, rare_mask(source, schema))):
        groups[key].append(idx)
    partitions = {"dev": [], "holdout": []}
    pair_counts = {key: {"dev": 0, "holdout": 0} for key in QUOTAS}
    leftovers = []
    for key in sorted(groups):
        indices = rng.permutation(groups[key])
        stratum = key[:2]
        for offset in range(0, len(indices) - 1, 2):
            side = min(partitions, key=lambda s: (pair_counts[stratum][s], len(partitions[s]), s))
            partitions[side].extend(indices[offset:offset + 2].tolist())
            pair_counts[stratum][side] += 1
        if len(indices) % 2:
            leftovers.append(int(indices[-1]))
    for idx in rng.permutation(leftovers):
        side = min(partitions, key=lambda s: (len(partitions[s]), s))
        partitions[side].append(int(idx))
    for side in partitions:
        if len(partitions[side]) < 2235:
            raise ValueError(f"{side} has insufficient distinct rows for the fixed protocol")
        for key, quota in QUOTAS.items():
            if pair_counts[key][side] < quota:
                raise ValueError(f"{side} lacks exact matched pairs in stratum {key}")
    assert not set(partitions["dev"]) & set(partitions["holdout"])
    assert len(partitions["dev"]) + len(partitions["holdout"]) == len(source)
    return {side: np.asarray(sorted(indices), dtype=int) for side, indices in partitions.items()}


@dataclass
class ShadowCase:
    target: pd.DataFrame
    candidates: pd.DataFrame
    challenge: pd.DataFrame
    background: pd.DataFrame
    mia_truth: np.ndarray
    mia_rare: np.ndarray
    aia_member: np.ndarray
    aia_rare: np.ndarray
    aia_time: np.ndarray
    aia_onset: np.ndarray
    aia_pair: np.ndarray
    metadata: dict
    private_roles: dict


def build_case(source, background, schema, split, seed):
    rng = np.random.default_rng(seed)
    partitions = partition_source(source, schema)
    if split not in partitions:
        raise ValueError("split must be dev or holdout")
    available = set(partitions[split].tolist())
    rare = rare_mask(source, schema)
    keys = matching_keys(source, rare)
    groups = defaultdict(list)
    for idx in sorted(available):
        groups[keys[idx]].append(idx)
    pairs = defaultdict(list)
    for key in sorted(groups):
        indices = rng.permutation(groups[key])
        pairs[key[:2]].extend((int(indices[i]), int(indices[i + 1]))
                             for i in range(0, len(indices) - 1, 2))
    members, controls = [], []
    for key, quota in QUOTAS.items():
        if len(pairs[key]) < quota:
            raise ValueError(f"No exact AIA matching capacity for {key}; no silent relaxation")
        for idx in rng.choice(len(pairs[key]), size=quota, replace=False):
            left, right = pairs[key][idx]
            if rng.integers(2):
                left, right = right, left
            members.append(left)
            controls.append(right)

    def draw(pool, n, is_rare):
        values = [idx for idx in sorted(pool) if bool(rare[idx]) == is_rare]
        if len(values) < n:
            raise ValueError(f"Not enough rows for rare={is_rare}, requested={n}")
        return rng.choice(values, size=n, replace=False).tolist()

    unused = available - set(members) - set(controls)
    target = members + draw(unused, 11, True) + draw(unused, 875, False)
    target = rng.permutation(target).tolist()
    shared_pool = set(target) - set(members)
    rare_shared = int(seed % 3)
    shared = draw(shared_pool, rare_shared, True) + draw(shared_pool, 26 - rare_shared, False)
    negative_pool = available - set(target) - set(controls)
    negatives = draw(negative_pool, 24 - rare_shared, True)
    negatives += draw(negative_pool, 1023 - (24 - rare_shared), False)
    candidate_order = rng.permutation(1049)
    candidates = np.asarray(shared + negatives)[candidate_order]
    mia_truth = np.r_[np.ones(26, dtype=int), np.zeros(1023, dtype=int)][candidate_order]
    challenge_order = rng.permutation(326)
    challenge_indices = np.asarray(members + controls)[challenge_order]
    aia_member = np.r_[np.ones(163, dtype=bool), np.zeros(163, dtype=bool)][challenge_order]
    aia_pair = np.tile(np.arange(163), 2)[challenge_order]
    assert not set(target) & set(negatives)
    assert not (set(target) | set(candidates)) & set(controls)
    assert not set(shared) & set(members)
    assert len(set(target)) == 1049 and len(set(candidates)) == 1049
    assert int(rare[target].sum()) == int(rare[candidates].sum()) == 24
    assert all(keys[m] == keys[c] for m, c in zip(members, controls))
    candidate_frame = source.iloc[candidates][FEATURES].reset_index(drop=True)
    candidate_frame.insert(0, "record_id", np.arange(len(candidate_frame)))
    challenge = source.iloc[challenge_indices][QI].reset_index(drop=True)
    challenge.insert(0, "challenge_row_id", np.arange(len(challenge)))
    role_digest = hashlib.sha256()
    roles = {"target": target, "mia_shared": shared, "mia_nonmember": negatives,
             "aia_member": members, "aia_control": controls}
    for name, indices in sorted(roles.items()):
        role_digest.update(name.encode())
        role_digest.update(np.asarray(indices, dtype="<i8").tobytes())
    metadata = {"split": split, "seed": seed, "partition_seed": PARTITION_SEED,
                "partition_rows": len(available), "target_rows": 1049,
                "mia_candidates": 1049, "mia_members": 26, "mia_rare_candidates": 24,
                "mia_rare_members": rare_shared, "aia_members": 163, "aia_controls": 163,
                "aia_rare_per_side": 13, "aia_onset_per_side": 16, "aia_rare_onset_per_side": 4,
                "matching_keys": ["detected_rare", "onset", "sex", "prefecture", "age_5year"],
                "matching_relaxed": False, "role_assignment_sha256": role_digest.hexdigest(),
                "rare_label_source": "Public Mahalanobis detector surrogate; not true is_rare",
                "background_source": "Actual B used only as an independent reference; no B_self labels"}
    return ShadowCase(source.iloc[target][FEATURES].reset_index(drop=True), candidate_frame,
        challenge, background[FEATURES].reset_index(drop=True), mia_truth, rare[candidates],
        aia_member, rare[challenge_indices], source.iloc[challenge_indices].time.to_numpy(float),
        source.iloc[challenge_indices].onset.to_numpy(int), aia_pair, metadata, roles)


def make_release(target, background, style, seed, schema):
    if style not in STYLES:
        raise ValueError(f"Unknown shadow style: {style}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, STYLES.index(style), 2026]))
    if style == "independent":
        c = background.iloc[rng.choice(len(background), 1049, replace=True)][FEATURES].copy()
    elif style == "bootstrap":
        c = target.iloc[rng.choice(len(target), len(target), replace=True)][FEATURES].copy()
    else:
        c = target[FEATURES].copy()
    c = c.reset_index(drop=True)
    if style == "rounded":
        widths = {"age": 5, "BMI": 1, "SBP": 10, "TG": 10, "HDL": 5, "ALT": 5, "FPG": 10, "time": .5}
        for col, width in widths.items():
            c[col] = np.round(c[col].to_numpy(float) / width) * width
    if style == "copula":
        from scipy.stats import norm, rankdata
        numeric = [col for col in FEATURES if col != "prefecture"]
        values = target[numeric].to_numpy(float)
        ranks = np.column_stack([rankdata(values[:, i], method="average") for i in range(values.shape[1])])
        z = norm.ppf((ranks - .5) / len(values))
        cov = np.cov(z, rowvar=False)
        # Small regularization is fixed in advance, not selected from attack results.
        cov = .95 * cov + .05 * np.diag(np.maximum(np.diag(cov), 1e-6))
        draw = rng.multivariate_normal(np.zeros(len(numeric)), cov, size=1049, check_valid="raise")
        latent_scale = np.sqrt(np.maximum(np.diag(cov), 1e-6))
        u = norm.cdf(draw / latent_scale)
        for i, col in enumerate(numeric):
            c[col] = np.quantile(values[:, i], u[:, i])
        for col in ("sex", "smoking", "onset", "death"):
            c[col] = np.rint(c[col]).astype(int)
        c["death"] = ((c.onset == 0) & (c.death == 1)).astype(int)
        for sex in (0, 1):
            positions = np.flatnonzero(c.sex.to_numpy() == sex)
            donors = target.loc[target.sex.eq(sex), "prefecture"]
            if donors.empty:
                donors = target.prefecture
            c.loc[positions, "prefecture"] = rng.choice(donors.to_numpy(), len(positions), replace=True)
    if style in ("jitter", "strong_jitter", "independent"):
        scale = background[CONTINUOUS].std(ddof=0).to_numpy(float)
        strength, time_sd = (.50, 1.0) if style == "strong_jitter" else (.10, .20)
        c[CONTINUOUS] = c[CONTINUOUS].to_numpy(float) + rng.normal(size=(len(c), 7)) * scale * strength
        c["time"] = c.time.to_numpy(float) + rng.normal(0, time_sd, len(c))
    for col, bounds in schema["ranges"].items():
        c[col] = c[col].clip(*bounds)
    c["age"] = np.rint(c.age.clip(*schema["age_range"])).astype(int)
    c["time"] = c.time.clip(1 / 365.25, 10)
    c = c.iloc[rng.permutation(len(c))].reset_index(drop=True)
    assert list(c.columns) == FEATURES and len(c) == 1049
    assert np.isfinite(c.drop(columns="prefecture").to_numpy(float)).all()
    return c
