"""Actual-distribution checks, kept outside the public-input prediction model."""
from collections import Counter
from decimal import Decimal

import numpy as np


def _keys(frame, columns):
    return [tuple(str(v) if col == "prefecture" else Decimal(str(v))
                  for col, v in zip(columns, row))
            for row in frame[columns].itertuples(index=False, name=None)]


def derive_private_mask(b, b_self, self_challenge, qi, columns):
    """Fail closed unless the exact self-challenge deduction is fully supported.

    Prefer CSV frames loaded as strings, avoiding any tolerance or rounding.
    The returned mask follows B's supplied order, not its record_id values.
    Organizer truth is not read; the inference also relies on the rule that
    all 163 member queries come from the private block of this cohort.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(len(b) == len(b_self) == 1049 and len(self_challenge) == 326,
            "Unexpected dimensions for actual team10 private-mask proof")
    require(len(qi) == 7 and len(columns) == 13, "Unexpected public schema")
    b_ids = _keys(b, ["record_id"])
    bs_ids = _keys(b_self, ["record_id"])
    require(len(set(b_ids)) == len(set(bs_ids)) == 1049,
            "Candidate or B_self IDs are not unique")
    require(set(b_ids) == set(bs_ids), "B_self ID set differs from B")
    bs_position = {key: pos for pos, key in enumerate(bs_ids)}
    aligned = b_self.iloc[[bs_position[key] for key in b_ids]]
    require(_keys(b, columns) == _keys(aligned, columns),
            "B and ID-aligned B_self public values differ")
    b_keys = _keys(b, qi)
    query_keys = _keys(self_challenge, qi)
    require(len(set(b_keys)) == len(b_keys), "B public QI are not unique")
    require(len(set(query_keys)) == len(query_keys), "Self challenge has duplicate QI")
    counts = Counter(query_keys)
    mask = np.array([key in counts for key in b_keys], dtype=bool)
    pool = set(b_keys)
    matched_queries = sum(key in pool for key in query_keys)
    require(int(mask.sum()) == matched_queries == 163,
            "Exact public-QI evidence does not identify precisely 163 private rows")
    rare = aligned["is_rare"].to_numpy(dtype=int)
    onset = aligned["onset"].to_numpy(dtype=int)
    death = aligned["death"].to_numpy(dtype=int)
    require(set(rare).issubset({0, 1}) and int(rare.sum()) == 24,
            "Unexpected true rare labels in B_self")
    require(int(rare[mask].sum()) == 13 and int(onset[mask].sum()) == 16
            and int((rare[mask] * onset[mask]).sum()) == 4,
            "Self-match group does not satisfy the official member quotas")
    return mask, {
        "exact_private_rows": 163, "remaining_unknown_rows": 886,
        "private_true_rare": int(rare[mask].sum()),
        "private_onset": int(onset[mask].sum()),
        "private_rare_onset": int((rare[mask] * onset[mask]).sum()),
        "private_death": int(death[mask].sum()),
        "comparison": "Exact decimal public-QI equality; unique matches; no tolerance",
        "deduction": "All 163 self members are private by the contest definition; exactly 163 unique B matches exhaust them.",
        "limitations": "Other 886 rows remain unknown. This selected negative set does not estimate population FPR.",
    }


def apply_private_mask(scores, mask):
    """Known negatives rank below every unknown; no count or team assignment."""
    values = np.asarray(scores, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Invalid prediction or mask shape/values")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("MIA values must be in [0,1]")
    # A shared positive can never be in mask if the exact deduction holds.
    values = 1e-12 + (1 - 1e-12) * values
    values[mask] = 0.0
    return values
