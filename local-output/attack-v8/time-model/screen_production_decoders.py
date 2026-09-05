"""Screen mechanism-aware time overlays on public production challenges.

Outputs only diagnostics.  The v7 candidate is read-only and no submission
artifact is created.
"""

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

from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    copula_event_links,
    filter_linkable_releases,
)
from mechanism_decoders import (  # noqa: E402
    interval_link_geometry,
    interval_links_from_geometry,
)


TARGETS = (3, 4, 5, 10, 12, 13, 20, 22, 23, 26, 29)
ALL_TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
ABSTAIN = 11.0
EDGE = 9.5
VISIBLE = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")


def challenge_from_background(background: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    query = background.iloc[rows].loc[:, list(VISIBLE)].reset_index(drop=True)
    query.insert(0, "challenge_row_id", np.arange(len(query), dtype=int))
    return query


def current_links(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    *,
    q: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del background
    return copula_event_links(
        released, challenge, margin_quantile=q, horizon_time=10.0
    )


INTERVAL_SPECS = (
    (0.25, 0.75),
    (0.50, 0.50),
    (0.75, 0.00),
    (0.75, 0.75),
    (0.90, 0.75),
)


def all_decoders(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for q in (0.25, 0.50, 0.75, 0.90):
        output[f"current_q{int(q * 100)}"] = current_links(
            released, challenge, background, q=q
        )
    for query_cdf in ("challenge", "background"):
        label = "self" if query_cdf == "challenge" else "bg"
        geometry = interval_link_geometry(
            released, challenge, background, query_cdf=query_cdf
        )
        for q, likelihood in INTERVAL_SPECS:
            output[f"interval_{label}_q{int(q * 100)}_l{int(likelihood * 100)}"] = (
                interval_links_from_geometry(
                    geometry,
                    query_cdf=query_cdf,
                    margin_quantile=q,
                    likelihood_quantile=likelihood,
                    horizon_time=10.0,
                )
            )
    return output


def main() -> None:
    dist = ROOT / "participant_data"
    targets_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    candidate_path = (
        ROOT
        / "submissions"
        / "prelim"
        / "team24"
        / "attack-candidate-v7-feedback-stack"
        / "F_aia.csv"
    )
    background = pd.read_csv(dist / "A_bg.csv")
    v7 = pd.read_csv(candidate_path)
    releases = {
        str(target): pd.read_csv(targets_dir / f"C_{target}.csv")
        for target in ALL_TARGETS
    }
    # Match the production all-target filtering scope exactly.
    linkable, _ = filter_linkable_releases(releases)

    method_names = [
        *(f"current_q{int(q * 100)}" for q in (0.25, 0.50, 0.75, 0.90)),
        *(
            f"interval_{label}_q{int(q * 100)}_l{int(likelihood * 100)}"
            for label in ("self", "bg")
            for q, likelihood in INTERVAL_SPECS
        ),
    ]

    production_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_id = str(target)
        challenge = pd.read_csv(targets_dir / f"aia_challenge_{target}.csv")
        block = (
            v7.loc[v7["target_k"].astype(int).eq(target)]
            .set_index("challenge_row_id")
            .loc[challenge["challenge_row_id"].to_numpy(dtype=int), "time_hat"]
            .to_numpy(dtype=float)
        )
        active = ~np.isclose(block, ABSTAIN)
        for method, (linked, gate, details) in all_decoders(
            linkable[target_id], challenge, background
        ).items():
            apply = active & gate
            changed = apply & ~np.isclose(block, linked)
            production_rows.append(
                {
                    "target": target,
                    "method": method,
                    "active_rows": int(active.sum()),
                    "gate_rows": int(gate.sum()),
                    "gate_active_rows": int(apply.sum()),
                    "changed_rows": int(changed.sum()),
                    "changed_row_ids": ";".join(
                        map(str, challenge.loc[changed, "challenge_row_id"].astype(int))
                    ),
                    "details": json.dumps(details, ensure_ascii=False, sort_keys=True),
                }
            )

        for fold, seed in enumerate((881_001, 881_002, 881_003, 881_004)):
                rng = np.random.default_rng(seed + target)
                indices = rng.choice(len(background), size=326, replace=False)
                query = challenge_from_background(background, indices)
                truth = background.iloc[indices]["time"].to_numpy(dtype=float)
                calibration_background = background.drop(
                    index=background.index[indices]
                ).reset_index(drop=True)
                decoded = all_decoders(
                    linkable[target_id], query, calibration_background
                )
                for method, (predicted, control_gate, _) in decoded.items():
                    old_hit = np.abs(EDGE - truth) <= 0.5
                    new_hit = np.abs(predicted - truth) <= 0.5
                    control_rows.append(
                        {
                            "target": target,
                            "method": method,
                            "fold": fold,
                            "gate_rows": int(control_gate.sum()),
                            "new_hits": int(np.count_nonzero(control_gate & new_hit)),
                            "old_hits_replaced": int(
                                np.count_nonzero(control_gate & old_hit)
                            ),
                            "mean_control_hit_delta_on_all_rows": float(
                                np.mean(control_gate * (new_hit.astype(float) - old_hit.astype(float)))
                            ),
                        }
                    )

    production = pd.DataFrame(production_rows)
    controls = pd.DataFrame(control_rows)
    production.to_csv(HERE / "production_decoder_screen.csv", index=False)
    controls.to_csv(HERE / "background_control_screen.csv", index=False)
    summary = (
        controls.groupby(["target", "method"], as_index=False)
        .agg(
            control_folds=("fold", "size"),
            mean_gate_rows=("gate_rows", "mean"),
            mean_new_hits=("new_hits", "mean"),
            mean_old_hits_replaced=("old_hits_replaced", "mean"),
            mean_control_hit_delta=("mean_control_hit_delta_on_all_rows", "mean"),
            max_control_hit_delta=("mean_control_hit_delta_on_all_rows", "max"),
        )
        .merge(
            production.drop(columns="details"), on=["target", "method"], how="left"
        )
    )
    summary.to_csv(HERE / "production_decoder_screen_summary.csv", index=False)
    print(
        summary.sort_values(
            ["target", "mean_control_hit_delta", "changed_rows"],
            ascending=[True, True, False],
        )
        .groupby("target")
        .head(5)
        .to_string(index=False)
    )
    report = {
        "scope": "public production screening only; no private truth",
        "targets": list(TARGETS),
        "methods": method_names,
        "background_control_folds": 4,
        "interpretation": (
            "negative control-hit delta means the event overlay reduces A_bg "
            "control successes relative to an edge prediction; it is not proof "
            "of member-time gain"
        ),
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (HERE / "production_decoder_screen.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
