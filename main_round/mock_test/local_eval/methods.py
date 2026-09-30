"""Prediction-only examples. No evaluation labels or source partition IDs enter here."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from pwscup2026.attack.aia import aia_predict
from pwscup2026.attack.mia import overlap_mia
from .data import QI


def official(*, c, background, candidates, challenge, cfg):
    _, times = aia_predict(c, challenge, background, cfg)
    return {"mia": overlap_mia(c, background, candidates, cfg), "aia": times}


def constant(*, c, background, candidates, challenge, cfg):
    return {"mia": np.full(len(candidates), .5), "aia": np.full(len(challenge), 10.)}


def background_only(*, c, background, candidates, challenge, cfg):
    _, times = aia_predict(background, challenge, background, cfg)
    return {"mia": np.full(len(candidates), .5), "aia": times}


def transform(frame, background, columns, logs):
    x = frame[columns].to_numpy(float).copy()
    b = background[columns].to_numpy(float).copy()
    for col in logs:
        pos = columns.index(col)
        x[:, pos] = np.log(np.clip(x[:, pos], 1e-6, None))
        b[:, pos] = np.log(np.clip(b[:, pos], 1e-6, None))
    scale = b.std(axis=0)
    return (x - b.mean(axis=0)) / np.where(scale > 0, scale, 1)


def grouped_times(c, query, cx, qx):
    groups = {key: np.asarray(indices) for key, indices in c.groupby(["sex", "prefecture"]).groups.items()}
    sexes = {key: np.asarray(indices) for key, indices in c.groupby("sex").groups.items()}
    all_indices = np.arange(len(c))
    result = []
    for i, (sex, region) in enumerate(query[["sex", "prefecture"]].itertuples(index=False, name=None)):
        indices = groups.get((sex, region), sexes.get(sex, all_indices))
        d = np.mean((cx[indices] - qx[i]) ** 2, axis=1)
        nearest = indices[d <= d.min()]
        result.append(float(np.mean(np.sort(c.time.to_numpy()[nearest]))))
    return np.asarray(result)


def multiview(*, c, background, candidates, challenge, cfg):
    """Illustrative multiview baseline, not the tuned preliminary winner."""
    views = [(["age", "SBP", "TG", "HDL", "ALT"], ["TG", "ALT"]),
             (["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"], ["TG", "ALT", "FPG"]),
             (["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time", "onset", "death"], ["TG", "ALT", "FPG"])]
    ranks = []
    for columns, logs in views:
        cx = transform(c, background, columns, logs)
        qx = transform(candidates, background, columns, logs)
        bx = transform(background, background, columns, logs)
        dc = cdist(qx, cx, metric="sqeuclidean").min(axis=1)
        db = cdist(qx, bx, metric="sqeuclidean").min(axis=1)
        raw = 1 / (1 + dc / np.maximum(db, 1e-12))
        ranks.append(rankdata(raw, method="average") / (len(raw) + 1))
    q = challenge.loc[:, QI]
    time_views = [views[0], (["age", "SBP", "HDL"], []),
                  (["SBP", "TG", "HDL", "ALT"], ["TG", "ALT"])]
    times = []
    for columns, logs in time_views:
        times.append(grouped_times(c, q, transform(c, background, columns, logs),
                                   transform(q, background, columns, logs)))
    return {"mia": np.mean(ranks, axis=0), "aia": np.median(times, axis=0)}


BUILTINS = {"official": official, "multiview": multiview,
            "constant": constant, "background_only": background_only}
