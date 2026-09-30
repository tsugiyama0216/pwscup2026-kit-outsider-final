"""Compose public-input AIA with cross-target MIA; no disk I/O or truth labels."""
from __future__ import annotations

import numpy as np
from . import aia, mia


def predict_all(*, targets, background, candidates, challenges, cfg,
                mia_strength, aia_variant, private_mask=None):
    """Return membership columns in sorted target order and times in query order.

    Target keys are used only for alignment, never as model features. The
    optional mask must be independently established negative membership data.
    It is shared across all target columns, with no forced positive allocation.
    """
    keys = sorted(targets)
    if not keys or set(keys) != set(challenges):
        raise ValueError("Targets and challenges must have the same nonempty keys")
    raw = np.column_stack([mia.raw_scores(c=targets[k], background=background,
                                        candidates=candidates) for k in keys])
    membership = mia.adjust_scores(raw, private_mask=private_mask,
                                   strength=mia_strength)
    times = {k: aia.predict(c=targets[k], background=background,
                            challenge=challenges[k], cfg=cfg,
                            variant=aia_variant) for k in keys}
    if membership.shape != (len(candidates), len(keys)):
        raise ValueError("Wrong MIA dimensions")
    if not np.isfinite(membership).all() or np.any((membership < 0) | (membership > 1)):
        raise ValueError("Invalid membership output")
    horizon = float(cfg.get("onset", {}).get("horizon_years", 10.))
    for k in keys:
        values = np.asarray(times[k], float)
        if values.shape != (len(challenges[k]),) or not np.isfinite(values).all():
            raise ValueError("Invalid AIA output shape or values")
        if np.any(values < 0) or np.any(values > horizon):
            raise ValueError("AIA time outside the public horizon")
    return {"target_order": keys, "mia": membership, "aia": times}
