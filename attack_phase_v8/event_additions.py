"""Add public high-confidence event links without replacing active predictions.

This policy alone makes no claim of improved private score.  In particular,
additional false hits on controls can reduce AIA power.  The caller must select
and validate its target-level recipe on held-out, truth-known local worlds.
No token, archive, or network functionality is present in this module.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from attack_phase_v5.aia_structured_ensemble import (
    OFFICIAL_COLUMNS,
    copula_event_links,
    filter_linkable_releases,
)


def strict_aia_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject malformed keys before conversion; never truncate fractional IDs."""
    if list(frame.columns) != list(OFFICIAL_COLUMNS):
        raise ValueError("AIA columns differ from official format")
    out = frame.copy()
    for key in ("target_k", "challenge_row_id"):
        values = pd.to_numeric(out[key], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**53).any() or not np.equal(values, np.floor(values)).all()):
            raise ValueError(f"invalid integer key: {key}")
        out[key] = values.astype(np.int64)
    if out.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("duplicate normalized AIA keys")
    times = pd.to_numeric(out["time_hat"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("invalid time_hat")
    out["time_hat"] = times
    return out.reset_index(drop=True)


def add_trusted_events(
    base: pd.DataFrame,
    releases: Mapping[str, pd.DataFrame],
    challenges: Mapping[str, pd.DataFrame],
    *,
    target_ids: Iterable[str],
    margin_quantile: float = .90,
    horizon: float = 10.,
    abstain: float = 11.,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overlay only abstention rows on explicitly selected whole target blocks.

    Always filter using every supplied release, not only selected targets.
    Targets 1 and 2 are protected because the measured AIA scores are maximal.
    Input row order may differ across tables; alignment uses validated keys.
    """
    output = strict_aia_frame(base)
    original = output.copy()
    selected = tuple(target_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected targets must be nonempty and unique")
    if any(not isinstance(t, str) or not t.isdigit() or str(int(t)) != t for t in selected):
        raise ValueError("target IDs must be canonical integer strings")
    if set(selected) & {"1", "2"}:
        raise ValueError("ceiling targets 1 and 2 are protected")
    observed = {str(t) for t in output.target_k.unique()}
    if set(releases) != observed or set(challenges) != observed:
        raise ValueError("all release/challenge/base target sets must agree")
    if not set(selected) <= observed:
        raise ValueError("selected target is absent")
    if (not np.isfinite([horizon, abstain, margin_quantile]).all()
            or horizon <= 0 or abstain <= horizon + .5
            or not 0 <= margin_quantile <= 1):
        raise ValueError("invalid horizon, abstention, or quantile")
    # Validate every challenge, including unmodified blocks.
    for target, challenge in challenges.items():
        if "challenge_row_id" not in challenge:
            raise ValueError("challenge lacks row IDs")
        ids = pd.to_numeric(challenge.challenge_row_id, errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(ids).all() or (ids < 0).any() or (ids >= 2**53).any()
                or not np.equal(ids, np.floor(ids)).all() or len(np.unique(ids)) != len(ids)):
            raise ValueError("challenge contains invalid/duplicate keys")
        block_ids = output.loc[output.target_k.eq(int(target)), "challenge_row_id"]
        if len(block_ids) != len(ids) or set(block_ids) != set(ids):
            raise ValueError("challenge and base keys differ")
    linkable, filtering = filter_linkable_releases(dict(releases))
    details = {}
    for target in selected:
        challenge = challenges[target].sort_values("challenge_row_id", kind="mergesort").reset_index(drop=True)
        block = output.loc[output.target_k.eq(int(target))].sort_values("challenge_row_id", kind="mergesort")
        linked, gate, link_info = copula_event_links(
            linkable[target], challenge, margin_quantile=margin_quantile, horizon_time=horizon)
        linked, gate = np.asarray(linked, dtype=float), np.asarray(gate, dtype=bool)
        if linked.shape != (len(block),) or gate.shape != (len(block),):
            raise ValueError("decoder output shape mismatch")
        if not np.isfinite(linked).all() or (linked < 0).any() or (linked > horizon).any():
            raise ValueError("decoder emitted invalid event time")
        values = block.time_hat.to_numpy(dtype=float)
        # Exact equality deliberately avoids changing near-11 active values.
        additions = (values == abstain) & gate & (linked < horizon)
        values[additions] = linked[additions]
        output.loc[block.index, "time_hat"] = values
        details[target] = {"added_rows": int(additions.sum()),
            "added_challenge_row_ids": challenge.loc[additions, "challenge_row_id"].astype(int).tolist(),
            "link_details": link_info}
    changed = output.time_hat.to_numpy() != original.time_hat.to_numpy()
    if not (original.loc[changed, "time_hat"] == abstain).all():
        raise RuntimeError("active predictions were modified")
    if not output[["target_k", "challenge_row_id"]].equals(original[["target_k", "challenge_row_id"]]):
        raise RuntimeError("keys were changed")
    return output, {"policy": "abstention_only_trusted_event_additions",
        "selected_targets": list(selected), "margin_quantile": margin_quantile,
        "changed_rows": int(changed.sum()), "targets": details, "release_filter": filtering}
