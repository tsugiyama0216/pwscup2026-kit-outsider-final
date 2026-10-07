"""Public-input R v2 candidates; no paths, truth, seeds or target-style labels.

Compare frozen R v1/v2f with geometry that does not depend on a mock-trained
release classifier. All options are global; no target-ID-specific parameters.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
            "smoking", "FPG", "time", "onset", "death"]
QI_NUM = ["age", "SBP", "TG", "HDL", "ALT"]
FULL_NUM = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
TARGETS = tuple(k for k in (*range(18), 22, 25, 26, 27, 28) if k != 10)
MIA_OPTIONS = ("R_v1", "T", "T_cross", "TM", "geometry", "geometry_half", "geometry_cross")
AIA_OPTIONS = ("T", "nearest", "window5", "window15", "contrast15", "gated15", "T_rescue")
GRID = np.linspace(0., 10., 101)


def ranks(values, axis=0):
    values = np.asarray(values, float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite scores")
    return (rankdata(values, axis=axis, method="average") - .5) / values.shape[axis]


def finish_mia(raw, private, strength=0.):
    raw = np.asarray(raw, float)
    private = np.asarray(private)
    if raw.ndim != 2 or private.dtype != np.dtype(bool) or private.shape != (len(raw),):
        raise ValueError("Invalid score matrix or known-negative mask")
    if not np.isfinite(raw).all() or not 0 <= strength <= 1:
        raise ValueError("Invalid scores or correction strength")
    adjusted = raw - strength * np.median(raw, axis=1, keepdims=True)
    result = ranks(adjusted)
    result[private] = 0.
    return result


def _numeric(frame, columns):
    x = frame[columns].to_numpy(float).copy()
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite public feature")
    for column in ("TG", "ALT"):
        if column in columns:
            pos = columns.index(column)
            if (x[:, pos] < 0).any():
                raise ValueError("Negative laboratory value")
            x[:, pos] = np.log1p(x[:, pos])
    return x


def distances(query, frame, background, columns):
    scale = np.maximum(_numeric(background, columns).std(axis=0), 1e-6)
    d = cdist(_numeric(query, columns) / scale, _numeric(frame, columns) / scale,
              "sqeuclidean") / len(columns)
    for column, penalty in (("sex", 1.), ("prefecture", .25)):
        d += penalty * (query[column].to_numpy()[:, None] != frame[column].to_numpy()[None, :])
    return d


def geometry_mia(c, background, candidates):
    """Density ratio from seven numeric attributes, without outcome variables."""
    dc = distances(candidates, c, background, FULL_NUM)
    db = distances(candidates, background, background, FULL_NUM)
    dc = np.column_stack([dc.min(axis=1), np.partition(dc, 4, axis=1)[:, 4]])
    db = np.column_stack([db.min(axis=1), np.partition(db, 4, axis=1)[:, 4]])
    ratio = np.log(np.maximum(db, 1e-12)) - np.log(np.maximum(dc, 1e-12))
    return ratio @ np.array([.7, .3])


def _canonical(frame):
    return frame.loc[:, FEATURES].sort_values(FEATURES, kind="mergesort").reset_index(drop=True)


def _window_probabilities(d, times, neighbors):
    indices = np.argsort(d, axis=1, kind="stable")[:, :neighbors]
    local = np.take_along_axis(d, indices, axis=1)
    bandwidth = np.maximum(local[:, -1:], 1e-9)
    weights = np.exp(-(local - local[:, :1]) / bandwidth)
    weights /= weights.sum(axis=1, keepdims=True)
    hits = np.abs(times[indices, None] - GRID[None, None, :]) <= .5
    return np.sum(weights[:, :, None] * hits, axis=1)


def geometry_aia(c, background, challenge, old):
    """Local time-window estimates and release/background contrast.

    Neighborhoods are soft in sex/prefecture rather than empty-stratum rules.
    No constraint imposes half-membership within sparse challenge strata.
    """
    c, background = _canonical(c), _canonical(background)
    dc = distances(challenge, c, background, QI_NUM)
    db = distances(challenge, background, background, QI_NUM)
    ct, bt = c.time.to_numpy(float), background.time.to_numpy(float)
    nearest = ct[dc.argmin(axis=1)]
    p5 = _window_probabilities(dc, ct, 5)
    pc = _window_probabilities(dc, ct, 15)
    pb = _window_probabilities(db, bt, 15)
    contrast = (pc - pb) / np.maximum(1. - pb, .05)
    best = contrast.argmax(axis=1)
    gain = contrast[np.arange(len(challenge)), best]
    ratio = dc.min(axis=1) / np.maximum(db.min(axis=1), 1e-12)
    contrasted = GRID[best]
    return {"T": np.asarray(old, float).copy(), "nearest": nearest,
            "window5": GRID[p5.argmax(axis=1)], "window15": GRID[pc.argmax(axis=1)],
            "contrast15": contrasted,
            "gated15": np.where((gain > .1) & (ratio < 1.), contrasted, 11.),
            "T_rescue": np.where((np.asarray(old) > 10.) & (ratio < .25) & (gain > .25), contrasted, old)}


def compute_world(*, candidates, self_labels, background, self_challenge, targets, schema, cfg):
    """Calculate the predeclared candidate set from public inputs only."""
    from M.v1.actual import derive_private_mask
    from M.v2.mia import raw_scores, adjust_scores
    from attack_research.v88.predict import CombinedAttack
    if set(targets) != set(TARGETS) or len(candidates) != 1049:
        raise ValueError("Expected one complete 22-target world")
    if candidates.record_id.duplicated().any():
        raise ValueError("Duplicate public record ID")
    private, proof = derive_private_mask(candidates.astype(str), self_labels.astype(str),
        self_challenge.astype(str), schema["aia_qi_columns"], schema["submitted_c_columns"])
    raw_m, raw_t, raw_g = [], [], []
    aia = {name: [] for name in AIA_OPTIONS}
    routes = {}
    attack = CombinedAttack()
    for k in TARGETS:
        item = targets[k]
        if set(item) != {"release", "challenge"}:
            raise ValueError("Only public release and challenge are accepted")
        c, challenge = item["release"], item["challenge"]
        answers, route = attack.predict(c=c, background=background, candidates=candidates,
                                        challenge=challenge, cfg=cfg)
        old = answers["combined"]
        raw_t.append(ranks(old["mia"]))
        raw_m.append(raw_scores(c=c, background=background, candidates=candidates))
        raw_g.append(geometry_mia(c, background, candidates))
        for name, times in geometry_aia(c, background, challenge, old["aia"]).items():
            if not np.isfinite(times).all() or (times < 0).any():
                raise ValueError("Invalid AIA prediction")
            aia[name].append(pd.DataFrame({"target_k": k,
                "challenge_row_id": challenge.challenge_row_id.to_numpy(), "time_hat": times}))
        routes[str(k)] = str(route)
        print(f"R v2 public prediction: target {k} complete", flush=True)
    m, t, g = map(np.column_stack, (raw_m, raw_t, raw_g))
    baseline = adjust_scores(m, private, 1.)
    scores = {"R_v1": baseline, "T": finish_mia(t, private),
              "T_cross": finish_mia(t, private, .5),
              "TM": finish_mia(.5 * t + .5 * baseline, private),
              "geometry": finish_mia(g, private), "geometry_half": finish_mia(g, private, .5),
              "geometry_cross": finish_mia(g, private, 1.)}
    mia = {name: pd.DataFrame({"record_id": candidates.record_id.to_numpy(),
            **{str(k): values[:, j] for j, k in enumerate(TARGETS)}}) for name, values in scores.items()}
    return {"mia": mia, "aia": {name: pd.concat(parts, ignore_index=True) for name, parts in aia.items()},
            "diagnostics": {"private_proof": proof, "inferred_T_routes": routes,
                            "input_scope": "Public tables only; no target truth, style, seed or path"}}


def selected_world(selection, **public):
    answer = compute_world(**public)
    return {"F_mia": answer["mia"][selection["mia"]],
            "F_aia": answer["aia"][selection["aia"]], "diagnostics": answer["diagnostics"]}
