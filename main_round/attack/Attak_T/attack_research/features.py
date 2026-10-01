"""Features computed only from the public attack view; no IDs or labels are used."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from local_eval.data import FEATURES, QI

CLINICAL = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
QI_CONT = ["age", "SBP", "TG", "HDL", "ALT"]
WIDTHS = {"age": 5., "BMI": 1., "SBP": 10., "TG": 10., "HDL": 5.,
          "ALT": 5., "FPG": 10., "time": .5}


def canonical(frame, columns):
    return frame[columns].sort_values(columns, kind="mergesort").reset_index(drop=True)


def transformed(frame, background, columns, logs=False):
    values = frame[columns].to_numpy(float).copy()
    ref = background[columns].to_numpy(float).copy()
    if logs:
        for i, col in enumerate(columns):
            if col in ("TG", "ALT", "FPG"):
                values[:, i] = np.log1p(np.maximum(values[:, i], 0.))
                ref[:, i] = np.log1p(np.maximum(ref[:, i], 0.))
    scale = np.maximum(ref.std(axis=0), 1e-6)
    return values / scale


def distance(query, reference, background, columns, logs=False, extra_blocks=()):
    q = transformed(query, background, columns, logs)
    r = transformed(reference, background, columns, logs)
    d = cdist(q, r, "sqeuclidean") / len(columns)
    for col in ("sex", "prefecture", *extra_blocks):
        d += (query[col].to_numpy()[:, None] != reference[col].to_numpy()[None, :]) * 100.
    return d


def top_indices(d, count=8):
    count = min(count, d.shape[1])
    indices = np.argpartition(d, count-1, axis=1)[:, :count]
    values = np.take_along_axis(d, indices, axis=1)
    order = np.argsort(values, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1)


def release_summary(c):
    numbers = [float(c.duplicated(FEATURES).mean()), float(c.duplicated(QI).mean()),
               float(c.onset.mean()), float(c.death.mean()), float((c.time >= 9.5).mean())]
    for col, width in WIDTHS.items():
        x = c[col].to_numpy(float)
        numbers += [float(np.isclose(x / width, np.rint(x / width), atol=1e-8).mean())]
    return np.asarray(numbers, dtype=float)


def _collect(query, c, background, views):
    columns, nearest, matrices = [], [], []
    for names, logs, blocks in views:
        dc = distance(query, c, background, names, logs, blocks)
        db = distance(query, background, background, names, logs, blocks)
        ci, bi = top_indices(dc), top_indices(db)
        cv, bv = np.take_along_axis(dc, ci, 1), np.take_along_axis(db, bi, 1)
        for j in (0, 1, 3, 7):
            columns += [np.log1p(cv[:, j]), np.log1p(bv[:, j]),
                        np.log((bv[:, j] + 1e-9) / (cv[:, j] + 1e-9))]
        columns += [cv[:, 0] / np.maximum(cv[:, 1], 1e-9),
                    rankdata(-cv[:, 0], method="average") / len(query)]
        nearest.append(ci)
        matrices.append(dc)
    x = np.column_stack(columns)
    x = np.column_stack([x, np.tile(release_summary(c), (len(query), 1))])
    return x, nearest, matrices


def extract_mia(c, background, candidates):
    c, background = canonical(c, FEATURES), canonical(background, FEATURES)
    query = candidates[FEATURES].reset_index(drop=True)
    views = [(CLINICAL + ["time"], False, ("smoking", "onset", "death")),
             (CLINICAL + ["time"], True, ("smoking", "onset", "death")),
             (CLINICAL, True, ()), (QI_CONT, True, ())]
    x, nearest, distances = _collect(query, c, background, views)
    rounded = query.copy()
    for col, width in WIDTHS.items():
        rounded[col] = np.rint(rounded[col] / width) * width
    keys = pd.MultiIndex.from_frame(c[FEATURES])
    exact = pd.MultiIndex.from_frame(query[FEATURES]).isin(keys)
    rounded_match = pd.MultiIndex.from_frame(rounded[FEATURES]).isin(keys)
    # Exact attribute evidence is distinct from ID or source-row memorization.
    x = np.column_stack([x, exact.astype(float), rounded_match.astype(float)])
    simple = np.mean([rankdata(-d.min(axis=1)) / len(query) for d in distances[:2]], axis=0)
    simple = np.where(exact | rounded_match, 1., simple * .95)
    return x.astype(np.float32), simple


def extract_aia(c, background, challenge):
    c, background = canonical(c, FEATURES), canonical(background, FEATURES)
    query = challenge[QI].reset_index(drop=True)
    views = [(QI_CONT, False, ()), (QI_CONT, True, ()),
             (["age", "SBP", "HDL"], False, ())]
    x, nearest, distances = _collect(query, c, background, views)
    c_time, event = c.time.to_numpy(float), (c.onset.to_numpy() + c.death.to_numpy()) > 0
    proposals = []
    extra = []
    for ci in nearest:
        linked = ci[:, 0]
        proposals.append(np.where(event[linked], c_time[linked], 10.))
        extra += [c_time[linked], event[linked].astype(float), c_time[ci].mean(axis=1),
                  c_time[ci].std(axis=1)]
    # One-to-one global matching is an additional hypothesis, not forced on every release.
    rows, cols = linear_sum_assignment(distances[1])
    assignment = np.empty(len(query), dtype=int)
    assignment[rows] = cols
    proposals += [np.where(event[assignment], c_time[assignment], 10.),
                  np.full(len(query), 10.), np.full(len(query), 9.5)]
    extra += [distances[1][rows, cols], c_time[assignment], event[assignment].astype(float)]
    proposals = np.clip(np.column_stack(proposals), 0., 10.)
    # A deterministic interval match recovers rounded releases without needing style labels.
    rounded = query.copy()
    for col in QI_CONT:
        rounded[col] = np.rint(rounded[col] / WIDTHS[col]) * WIDTHS[col]
    rounded_keys = pd.MultiIndex.from_frame(rounded[QI])
    exact_keys = pd.MultiIndex.from_frame(query[QI])
    release_keys = pd.MultiIndex.from_frame(c[QI])
    exact = exact_keys.isin(release_keys)
    rounded_match = rounded_keys.isin(release_keys)
    lookup = c.groupby(QI, dropna=False, sort=False).time.median()
    exact_time = lookup.reindex(exact_keys).to_numpy(float)
    rounded_time = lookup.reindex(rounded_keys).to_numpy(float)
    exact_time = np.where(exact, exact_time, proposals[:, 0])
    rounded_time = np.where(rounded_match, rounded_time, proposals[:, 0])
    proposals = np.column_stack([proposals, exact_time, rounded_time])
    x = np.column_stack([x, *extra, exact.astype(float), rounded_match.astype(float)])
    simple_membership = np.mean([rankdata(-d.min(axis=1)) / len(query) for d in distances[:2]], axis=0)
    simple_membership = np.where(exact | rounded_match, 1., simple_membership * .95)
    simple_time = np.where(exact, exact_time, np.where(rounded_match, rounded_time, proposals[:, 0]))
    return x.astype(np.float32), proposals.astype(np.float32), simple_membership, simple_time


def proposal_features(x, proposals):
    n, k = proposals.shape
    base = np.repeat(x, k, axis=0)
    times = proposals.reshape(-1)
    differences = np.abs(times[:, None] - np.repeat(proposals, k, axis=0))
    return np.column_stack([base, times, differences, (times >= 9.5).astype(float)]).astype(np.float32)
