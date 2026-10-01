"""Joint shadow worlds with exclusive candidate membership across 22 targets.

Only transient memory holds source positions or record-level truth. Shared
noncandidate donors may recur in multiple targets because the public source
is too small to reconstruct all 30 original-sized cohorts without reuse.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from local_eval.data import FEATURES, QI, QUOTAS, make_release, partition_source, rare_mask

STYLES = ("copy", "rounded", "jitter", "strong_jitter", "bootstrap", "copula",
          "independent", "category_perturb", "background_mix")


@dataclass
class JointWorld:
    candidates: pd.DataFrame
    background: pd.DataFrame
    releases: list[pd.DataFrame]
    truth: np.ndarray
    rare: np.ndarray
    private_mask: np.ndarray
    metadata: dict


def _choose(rng, indices, n, weights=None):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < n:
        raise ValueError("Insufficient disjoint source capacity; no silent replacement")
    probability = None if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    return rng.choice(indices, n, replace=False, p=probability).tolist()


def build_world(source, background, schema, *, split, seed, visible_rare_total, layout):
    if visible_rare_total not in (0, 5, 11):
        raise ValueError("Unregistered rare-count configuration")
    if layout not in ("mixed", "uniform_jitter"):
        raise ValueError("Unregistered world layout")
    available = partition_source(source, schema)[split]
    detected = rare_mask(source, schema)
    onset = source.onset.to_numpy(int)
    candidate_rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    known = []
    for (is_rare, event), quota in QUOTAS.items():
        pool = available[(detected[available] == is_rare) & (onset[available] == event)]
        known.extend(_choose(candidate_rng, pool, quota))
    unused = sorted(set(available) - set(known))
    extra_rare = _choose(candidate_rng, [i for i in unused if detected[i]], 11)
    extra_normal = _choose(candidate_rng, [i for i in unused if not detected[i]], 875)
    ordered = np.asarray(known + extra_rare + extra_normal)
    order = candidate_rng.permutation(1049)
    chosen = ordered[order]
    mask = np.isin(chosen, known)
    candidate_rare = detected[chosen]
    candidates = source.iloc[chosen][FEATURES].reset_index(drop=True)
    candidates.insert(0, "record_id", np.arange(1049))

    # All 29 pair blocks total 736; private rows total 313. Counts are 25-27.
    topology_rng = np.random.default_rng(np.random.SeedSequence([seed, 20]))
    counts = np.full(29, 25, dtype=int)
    for _ in range(11):
        eligible = np.flatnonzero(counts < 27)
        counts[topology_rng.choice(eligible)] += 1
    rare_counts = np.zeros(29, dtype=int)
    for _ in range(visible_rare_total):
        eligible = np.flatnonzero(rare_counts[:22] < 2)
        rare_counts[topology_rng.choice(eligible)] += 1
    invisible_rare = min(2, 11 - visible_rare_total)
    for _ in range(invisible_rare):
        rare_counts[22 + topology_rng.integers(7)] += 1
    unknown_rare = topology_rng.permutation(np.flatnonzero(candidate_rare & ~mask))
    unknown_normal = topology_rng.permutation(np.flatnonzero(~candidate_rare & ~mask))
    role_matrix = np.zeros((1049, 29), dtype=bool)
    rare_offset = normal_offset = 0
    for j, count in enumerate(counts):
        nr = int(rare_counts[j])
        nn = int(count) - nr
        role_matrix[unknown_rare[rare_offset:rare_offset + nr], j] = True
        role_matrix[unknown_normal[normal_offset:normal_offset + nn], j] = True
        rare_offset += nr
        normal_offset += nn
    assert role_matrix.sum() == 736 and not role_matrix[mask].any()
    assert np.all(role_matrix.sum(axis=1) <= 1)
    assert int((~role_matrix.any(axis=1)).sum()) == 313
    assert np.array_equal(role_matrix.sum(axis=0), counts)
    assert np.array_equal(role_matrix[candidate_rare].sum(axis=0), rare_counts)

    filler_pool = np.asarray(sorted(set(available) - set(chosen)), dtype=int)
    filler_rare = filler_pool[detected[filler_pool]]
    filler_normal = filler_pool[~detected[filler_pool]]
    style_rng = np.random.default_rng(np.random.SeedSequence([seed, 30]))
    styles = [STYLES[i % len(STYLES)] for i in range(22)]
    styles = [styles[i] for i in style_rng.permutation(22)] if layout == "mixed" else ["jitter"] * 22
    releases = []
    filler_sets = []
    for j in range(22):
        draw_rng = np.random.default_rng(np.random.SeedSequence([seed, j, 40]))
        preferred = draw_rng.choice(source.iloc[filler_pool].prefecture.astype(str).unique())
        def donor_weights(indices):
            return np.where(source.iloc[indices].prefecture.astype(str).to_numpy() == preferred, 3., 1.)
        fillers = _choose(draw_rng, filler_rare, 24 - int(rare_counts[j]), donor_weights(filler_rare))
        fillers += _choose(draw_rng, filler_normal, 1025 - (int(counts[j]) - int(rare_counts[j])), donor_weights(filler_normal))
        filler_sets.append(set(fillers))
        target = pd.concat([candidates.loc[role_matrix[:, j], FEATURES], source.iloc[fillers][FEATURES]], ignore_index=True)
        assert len(target) == 1049
        assert int(detected[fillers].sum()) + int(rare_counts[j]) == 24
        release_seed = int(np.random.SeedSequence([seed, j, 50]).generate_state(1)[0])
        style = styles[j]
        base_style = "jitter" if style in ("category_perturb", "background_mix") else style
        c = make_release(target, background, base_style, release_seed, schema)
        perturb_rng = np.random.default_rng(np.random.SeedSequence([seed, j, 60]))
        if style == "category_perturb":
            for column, probability in [("sex", .15), ("smoking", .25)]:
                flip = perturb_rng.random(len(c)) < probability
                c.loc[flip, column] = 1 - c.loc[flip, column]
            replace = perturb_rng.random(len(c)) < .30
            c.loc[replace, "prefecture"] = perturb_rng.choice(background.prefecture.to_numpy(), int(replace.sum()))
        elif style == "background_mix":
            replace = perturb_rng.choice(len(c), round(.35 * len(c)), replace=False)
            donors = perturb_rng.choice(len(background), len(replace), replace=False)
            c.loc[replace, FEATURES] = background.iloc[donors][FEATURES].to_numpy()
        releases.append(c[FEATURES].reset_index(drop=True))
    union_fillers = set().union(*filler_sets)
    metadata = {
        "split": split, "seed": seed, "visible_rare_total": visible_rare_total, "layout": layout,
        "candidate_rows": 1049, "known_private_proxy_rows": int(mask.sum()),
        "known_private_proxy_rare": int(candidate_rare[mask].sum()),
        "candidate_true_rare_proxy": int(candidate_rare.sum()),
        "visible_target_count": 22, "visible_shared_counts": counts[:22].tolist(),
        "visible_rare_shared_counts": rare_counts[:22].tolist(), "styles": styles,
        "all29_shared_total": int(counts.sum()), "private_total": 313,
        "invisible7_shared_total": int(counts[22:].sum()), "invisible7_rare_total": int(rare_counts[22:].sum()),
        "visible_rare_shared_total": int(rare_counts[:22].sum()),
        "targets_without_rare_shared": int(np.count_nonzero(rare_counts[:22] == 0)),
        "distinct_noncandidate_donors": len(union_fillers),
        "noncandidate_donor_uses": sum(map(len, filler_sets)),
        "limitations": [
            "Rare labels use the public Mahalanobis proxy, not true is_rare.",
            "Noncandidate donors recur across cohorts; only query-person membership is globally exclusive.",
            "Dev/holdout person partitions reuse the prior generic partition; new seeds are not fresh people.",
            "Target filler region weighting is a proxy, not reconstruction of the organizer allocation.",
        ],
    }
    return JointWorld(candidates, background[FEATURES].reset_index(drop=True), releases,
                      role_matrix[:, :22], candidate_rare, mask, metadata)
