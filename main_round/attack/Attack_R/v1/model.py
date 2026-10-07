"""R v1: unchanged M v2 membership and unchanged T v88 time inference."""
from __future__ import annotations

import numpy as np


def predict_mia_all(*, targets, background, candidates, private_mask, strength):
    """Keep M v2's joint, cross-target calculation and selected strength."""
    from M.v2.mia import raw_scores, adjust_scores

    keys = sorted(targets)
    if not keys:
        raise ValueError("No attack targets")
    raw = np.column_stack([
        raw_scores(c=targets[k], background=background, candidates=candidates)
        for k in keys
    ])
    scores = adjust_scores(raw, private_mask=private_mask, strength=strength)
    return keys, scores


def predict_aia_one(attack, *, c, background, candidates, challenge, cfg):
    """Select only the AIA component of the original, complete v88 prediction."""
    answers, inferred = attack.predict(c=c, background=background,
        candidates=candidates, challenge=challenge, cfg=cfg)
    times = np.asarray(answers["combined"]["aia"], dtype=float)
    if times.shape != (len(challenge),) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Invalid T v88 AIA output")
    return times, str(inferred)
