"""Audit the proposed target-20-only time overlay without writing production files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    filter_linkable_releases,
)


TARGET = "20"
HORIZON = 10.0
ABSTAIN = 11.0
MARGIN_QUANTILE = 0.50


def main() -> None:
    release_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    releases = {
        path.stem.removeprefix("C_"): pd.read_csv(path)
        for path in sorted(release_dir.glob("C_*.csv"))
    }
    linkable, filter_diagnostics = filter_linkable_releases(releases)

    challenge = pd.read_csv(release_dir / f"aia_challenge_{TARGET}.csv")
    challenge = challenge.reset_index(drop=True).copy()
    if "challenge_row_id" not in challenge.columns:
        challenge["challenge_row_id"] = np.arange(len(challenge), dtype=int)

    baseline = pd.read_csv(
        ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v6-final-horizon-edge"
        / "F_aia.csv"
    )
    block = baseline.loc[baseline["target_k"].astype(str).eq(TARGET)].copy()
    block = block.sort_values("challenge_row_id", kind="mergesort")
    if not np.array_equal(
        block["challenge_row_id"].to_numpy(dtype=int),
        np.arange(len(challenge), dtype=int),
    ):
        raise ValueError("target 20 block is not aligned to its AIA challenge")

    linked_time, linked_gate, link_details = copula_event_links(
        linkable[TARGET],
        challenge,
        margin_quantile=MARGIN_QUANTILE,
        horizon_time=HORIZON,
    )
    old = block["time_hat"].to_numpy(dtype=float)
    active = old < ABSTAIN - 1e-12
    gate = active & linked_gate
    new = old.copy()
    new[gate] = linked_time[gate]
    changed = ~np.isclose(new, old, atol=0.0, rtol=0.0)

    audit = pd.DataFrame(
        {
            "target_k": int(TARGET),
            "challenge_row_id": challenge["challenge_row_id"].to_numpy(dtype=int),
            "old_time_hat": old,
            "new_time_hat": new,
            "active": active,
            "trusted_event_link": linked_gate,
            "applied": gate,
            "changed": changed,
        }
    )
    audit.loc[audit["changed"]].to_csv(
        HERE / "production_target20_changed_rows.csv", index=False
    )

    report = {
        "target": int(TARGET),
        "margin_quantile": MARGIN_QUANTILE,
        "filter_target20": filter_diagnostics[TARGET],
        "link_details": link_details,
        "baseline_active_rows": int(active.sum()),
        "trusted_event_links_all_challenge": int(linked_gate.sum()),
        "applied_active_links": int(gate.sum()),
        "changed_rows": int(changed.sum()),
        "membership_mask_unchanged": bool(
            np.array_equal(old < ABSTAIN - 1e-12, new < ABSTAIN - 1e-12)
        ),
        "changed_challenge_row_ids": audit.loc[
            audit["changed"], "challenge_row_id"
        ].astype(int).tolist(),
        "changed_time_hat": audit.loc[audit["changed"], "new_time_hat"].tolist(),
    }
    (HERE / "production_target20_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
