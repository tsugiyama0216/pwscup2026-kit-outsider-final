"""Pure MIA scoring and soft cross-target adjustment for independent M/v2."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from M.v1 import model as base
from M.v1.actual import apply_private_mask

STRENGTHS = (0.0, 0.25, 0.5, 1.0)
ADJUSTMENT = "subtract_candidate_cross_target_median"


def raw_scores(*, c, background, candidates):
    """Return the frozen v1 full-feature local-density ranking statistic."""
    base._validate(c, base.PUBLIC_COLUMNS, "C")
    base._validate(background, base.PUBLIC_COLUMNS, "background")
    base._validate(candidates, base.PUBLIC_COLUMNS, "candidates", allow_empty=True)
    c, background = base._canonical(c), base._canonical(background)
    candidates = candidates.loc[:, base.PUBLIC_COLUMNS].copy(deep=True).reset_index(drop=True)
    if len(candidates) == 0:
        return np.empty(0, dtype=float)
    arrays = base._full_features((c, background, candidates), background)
    result = base._local_density_score(c, background, candidates, arrays, 7.)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite MIA raw statistic")
    return np.asarray(result, dtype=float)


def adjust_scores(raw_matrix, private_mask=None, strength=0.0):
    """Return [candidate, target] scores without assigning membership labels.

    A candidate's median across targets is a nuisance-propensity estimate, not
    a calibrated membership probability. No row/column quota is enforced.
    All known-private rows become zero only after ranking. They do not supply
    fitted calibration parameters. At strength zero this is exactly v1's
    per-target average-rank transformation followed by the same private mask.
    """
    values = np.asarray(raw_matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a finite candidate-by-target matrix")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Adjustment strength must be finite and in [0,1]")
    n, targets = values.shape
    if private_mask is None:
        mask = None
    else:
        mask = np.asarray(private_mask)
        if mask.shape != (n,) or mask.dtype != np.dtype(bool):
            raise ValueError("Private mask must be a Boolean candidate vector")
    if n == 0:
        return np.empty_like(values)
    adjusted = values.copy()
    if strength != 0 and targets > 1:
        adjusted -= float(strength) * np.median(values, axis=1, keepdims=True)
    scores = (rankdata(adjusted, method="average", axis=0) - .5) / n
    if mask is not None:
        scores = np.column_stack([apply_private_mask(scores[:, j], mask) for j in range(targets)])
    return scores
