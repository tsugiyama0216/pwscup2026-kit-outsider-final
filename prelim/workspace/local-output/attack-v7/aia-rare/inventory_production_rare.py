"""Inventory public rare rows and conservative link opportunities in v7 AIA."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    filter_linkable_releases,
)
from rare_proxy import fit_proxy_models, top_k_mask  # noqa: E402


ABSTAIN = 11.0
EDGE = 9.5


def aligned_values(
    predictions: pd.DataFrame, target: str, challenge: pd.DataFrame
) -> np.ndarray:
    block = predictions.loc[
        predictions["target_k"].astype(int).eq(int(target)),
        ["challenge_row_id", "time_hat"],
    ].set_index("challenge_row_id")
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    if set(block.index.astype(int)) != set(ids):
        raise ValueError(f"prediction/challenge key mismatch for target {target}")
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    predictions = pd.read_csv(
        ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v7-feedback-stack"
        / "F_aia.csv"
    )
    releases = {
        path.stem.removeprefix("C_"): pd.read_csv(path)
        for path in sorted(target_dir.glob("C_*.csv"))
    }
    linkable, filter_details = filter_linkable_releases(releases)
    detector = shadow.load_rare_detector(ROOT / "participant_data")
    background = pd.read_csv(ROOT / "participant_data" / "A_bg.csv")
    background_rare = shadow.public_rare_mask(background, detector)
    proxy = fit_proxy_models(background, background_rare, detector)
    rows: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    attacked_targets = set(predictions["target_k"].astype(int).astype(str))
    for target in sorted(attacked_targets, key=int):
        challenge = pd.read_csv(target_dir / f"aia_challenge_{target}.csv")
        challenge = challenge.sort_values("challenge_row_id", kind="mergesort").reset_index(
            drop=True
        )
        values = aligned_values(predictions, target, challenge)
        rare_score = proxy.scores(challenge)["extra_trees"]
        rare = top_k_mask(
            rare_score,
            challenge["challenge_row_id"].to_numpy(dtype=int),
            26,
        )
        release_rare = shadow.public_rare_mask(releases[target], detector)
        active = ~np.isclose(values, ABSTAIN)
        event_prediction = active & (values < EDGE - 1e-12)
        release_event = (
            releases[target]["onset"].to_numpy(dtype=int)
            + releases[target]["death"].to_numpy(dtype=int)
        ) > 0
        item: dict[str, Any] = {
            "target": int(target),
            "challenge_rows": len(challenge),
            "public_rare_rows": int(rare.sum()),
            "active_rows": int(active.sum()),
            "active_rare_rows": int((active & rare).sum()),
            "inactive_rare_rows": int((~active & rare).sum()),
            "event_prediction_rows": int(event_prediction.sum()),
            "event_prediction_rare_rows": int((event_prediction & rare).sum()),
            "release_rows": len(releases[target]),
            "release_public_rare_rows": int(release_rare.sum()),
            "release_event_rate": float(release_event.mean()),
            "release_rare_event_rate": float(release_event[release_rare].mean())
            if release_rare.any()
            else float("nan"),
            "release_common_event_rate": float(release_event[~release_rare].mean())
            if (~release_rare).any()
            else float("nan"),
            "cross_target_filter_applied": bool(
                filter_details[target]["cross_target_filter_applied"]
            ),
            "linkable_rows": int(filter_details[target]["linkable_rows"]),
        }
        for quantile in (0.50, 0.75, 0.90):
            linked_time, linked_gate, details = copula_event_links(
                linkable[target],
                challenge,
                margin_quantile=quantile,
                horizon_time=10.0,
            )
            label = str(quantile).replace(".", "p")
            item[f"q{label}_trusted_event_links"] = int(linked_gate.sum())
            item[f"q{label}_rare_links"] = int((linked_gate & rare).sum())
            item[f"q{label}_rare_active_links"] = int(
                (linked_gate & rare & active).sum()
            )
            item[f"q{label}_rare_inactive_links"] = int(
                (linked_gate & rare & ~active).sum()
            )
            changed = linked_gate & rare & active & ~np.isclose(linked_time, values)
            for index in np.flatnonzero(changed):
                opportunities.append(
                    {
                        "target": int(target),
                        "challenge_row_id": int(challenge.iloc[index]["challenge_row_id"]),
                        "margin_quantile": quantile,
                        "old_time_hat": float(values[index]),
                        "linked_time_hat": float(linked_time[index]),
                    }
                )
            if not np.isfinite(linked_time).all():
                raise ValueError(f"non-finite linked time for target {target}")
            if details["trusted_event_links"] != int(linked_gate.sum()):
                raise AssertionError("link diagnostics mismatch")
        rows.append(item)

    inventory = pd.DataFrame(rows).sort_values("target")
    inventory.to_csv(HERE / "production_rare_inventory.csv", index=False)
    pd.DataFrame(opportunities).drop_duplicates().to_csv(
        HERE / "production_rare_link_opportunities.csv", index=False
    )
    report = {
        "scope": "read-only public rare inventory for v7 AIA",
        "candidate": "attack-candidate-v7-feedback-stack/F_aia.csv",
        "rare_rule": "published Mahalanobis detector from participant_data/schema.json",
        "challenge_rare_proxy": "top 26 ExtraTrees scores trained on A_bg full-detector labels",
        "targets": inventory.to_dict(orient="records"),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "submission_performed": False,
    }
    (HERE / "production_rare_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        inventory[
            [
                "target",
                "public_rare_rows",
                "active_rare_rows",
                "event_prediction_rare_rows",
                "q0p5_rare_active_links",
                "q0p5_rare_inactive_links",
                "release_rare_event_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
