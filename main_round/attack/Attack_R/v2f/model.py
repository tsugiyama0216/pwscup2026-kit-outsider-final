"""Public-input candidate components for R v2f; no truth, IDs, or file I/O."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

MIA_OPTIONS = ("T", "T90_M10", "T75_M25", "T50_M50", "M")
AIA_OPTIONS = ("T", "activate_025", "activate_040", "retime_040", "nearest_guard")
QI = ["age", "SBP", "TG", "HDL", "ALT"]
GRID = np.arange(.25, 10.01, .25)


def rank(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError("Expected a nonempty finite score vector")
    return (rankdata(values, method="average") - .5) / len(values)


def mia_variants(t, m):
    t, m = rank(t), rank(m)
    return {"T": t, "T90_M10": .9*t+.1*m, "T75_M25": .75*t+.25*m,
            "T50_M50": .5*t+.5*m, "M": m}


def apply_private_mask(values, mask):
    values = np.asarray(values, float).copy()
    mask = np.asarray(mask)
    if mask.dtype != np.dtype(bool) or mask.shape != values.shape:
        raise ValueError("The known-negative mask must align exactly")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid MIA confidence")
    values[mask] = 0.
    return values


def time_evidence(c, background, challenge):
    """Compare local time-window mass against independent background mass."""
    columns = QI + ["sex", "prefecture", "time"]
    c = c[columns].sort_values(columns, kind="mergesort").reset_index(drop=True)
    b = background[columns].sort_values(columns, kind="mergesort").reset_index(drop=True)
    query = challenge[QI + ["sex", "prefecture"]].reset_index(drop=True)
    scale = np.maximum(b[QI].std(ddof=0).to_numpy(float), 1e-6)
    q = query[QI].to_numpy(float) / scale
    distances = []
    for frame in (c, b):
        d = cdist(q, frame[QI].to_numpy(float)/scale, "sqeuclidean") / len(QI)
        for column, penalty in (("sex", 4.), ("prefecture", 4.)):
            d += penalty * (query[column].to_numpy()[:, None] != frame[column].to_numpy()[None, :])
        distances.append(d)
    dc, db = distances
    bandwidth = np.maximum(np.partition(db, 15, axis=1)[:, 15], 1e-6)
    distributions = []
    for d, frame in zip(distances, (c, b)):
        shifted = d - d.min(axis=1, keepdims=True)
        weights = np.exp(-shifted / bandwidth[:, None])
        weights /= weights.sum(axis=1, keepdims=True)
        windows = (abs(frame.time.to_numpy(float)[:, None] - GRID[None, :]) <= .5).astype(float)
        distributions.append(weights @ windows)
    pc, pb = distributions
    excess = (pc-pb) / np.maximum(1-pb, .05)
    best = np.argmax(excess, axis=1)
    gain = excess[np.arange(len(query)), best]
    nearest = dc.argmin(axis=1)
    close_ratio = dc.min(axis=1) / np.maximum(db.min(axis=1), 1e-9)
    nearest_time = c.time.to_numpy(float)[nearest]
    second = np.partition(dc, 1, axis=1)[:, 1]
    unique = dc.min(axis=1) <= .1*np.maximum(second, 1e-9)
    return GRID[best], gain, close_ratio, nearest_time, unique


def aia_variants(t, evidence):
    t = np.asarray(t, float)
    best, gain, ratio, nearest, unique = evidence
    if t.shape != best.shape or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError("Invalid AIA input")
    active = t <= 10.
    options = {"T": t.copy()}
    for name, threshold in (("activate_025", .25), ("activate_040", .4)):
        use = (~active) & (gain >= threshold) & (ratio < .5)
        options[name] = np.where(use, best, t)
    use = active & (gain >= .4) & (ratio < .5) & (abs(best-t) > .5)
    options["retime_040"] = np.where(use, best, t)
    use = unique & (ratio < .05) & (abs(nearest-t) > .5)
    options["nearest_guard"] = np.where(use, nearest, t)
    return options


def compute_candidates(t_attack, *, c, background, candidates, challenge, cfg):
    from M.v2.mia import raw_scores
    answers, route = t_attack.predict(c=c, background=background, candidates=candidates,
                                      challenge=challenge, cfg=cfg)
    old = answers["combined"]
    m = raw_scores(c=c, background=background, candidates=candidates)
    return mia_variants(old["mia"], m), aia_variants(old["aia"], time_evidence(c, background, challenge)), str(route)


def predict_selected(t_attack, selection, *, c, background, candidates, challenge, cfg, private_mask=None):
    mia, aia, route = compute_candidates(t_attack, c=c, background=background,
        candidates=candidates, challenge=challenge, cfg=cfg)
    membership, times = mia[selection["mia"]], aia[selection["aia"]]
    if private_mask is not None:
        membership = apply_private_mask(membership, private_mask)
    return {"mia": membership, "aia": times}, route
