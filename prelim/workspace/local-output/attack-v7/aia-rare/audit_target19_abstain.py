"""Read-only production audit for the target-19 rare-proxy abstention rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from rare_proxy import fit_proxy_models, top_k_mask  # noqa: E402


TARGET = 19
ABSTAIN = 11.0


def aligned(frame: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    block = frame.loc[frame["target_k"].astype(int).eq(TARGET)].set_index(
        "challenge_row_id"
    )
    return block.loc[ids, "time_hat"].to_numpy(dtype=float)


def main() -> None:
    dist = ROOT / "participant_data"
    target_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    team = ROOT / "submissions" / "prelim" / "team24"
    challenge = pd.read_csv(target_dir / f"aia_challenge_{TARGET}.csv")
    challenge = challenge.sort_values("challenge_row_id", kind="mergesort").reset_index(
        drop=True
    )
    ids = challenge["challenge_row_id"].to_numpy(dtype=int)
    detector = shadow.load_rare_detector(dist)
    background = pd.read_csv(dist / "A_bg.csv")
    proxy = fit_proxy_models(
        background, shadow.public_rare_mask(background, detector), detector
    )
    proxy_score = proxy.scores(challenge)["extra_trees"]
    rare26 = top_k_mask(proxy_score, ids, 26)

    current_frame = pd.read_csv(
        team / "attack-candidate-v7-feedback-stack" / "F_aia.csv"
    )
    source_frames = {
        "v4": pd.read_csv(team / "attack-candidate-v4" / "F_aia.csv"),
        "independent": pd.read_csv(
            team / "attack-candidate-daybreak-independent" / "F_aia.csv"
        ),
        "routed": pd.read_csv(
            team / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv"
        ),
    }
    current = aligned(current_frame, ids)
    source_active = {
        name: ~np.isclose(aligned(frame, ids), ABSTAIN)
        for name, frame in source_frames.items()
    }
    unanimous = (
        source_active["v4"] & source_active["independent"] & source_active["routed"]
    )
    expected_current = ~unanimous
    current_active = ~np.isclose(current, ABSTAIN)
    if not np.array_equal(current_active, expected_current):
        raise ValueError("v7 target 19 is not complement-of-unanimous")

    proposed = current.copy()
    proposed[rare26] = ABSTAIN
    changed = ~np.isclose(proposed, current)
    pattern = np.array(
        [
            f"{int(source_active['v4'][i])}{int(source_active['independent'][i])}"
            f"{int(source_active['routed'][i])}"
            for i in range(len(ids))
        ]
    )
    rows = pd.DataFrame(
        {
            "target_k": TARGET,
            "challenge_row_id": ids,
            "rare_proxy_score": proxy_score,
            "rare_proxy_top26": rare26,
            "source_pattern_v4_independent_routed": pattern,
            "unanimous_active": unanimous,
            "v7_active": current_active,
            "old_time_hat": current,
            "research_time_hat": proposed,
            "changed": changed,
        }
    )
    rows.loc[rows["changed"]].to_csv(
        HERE / "target19_abstain_changed_rows.csv", index=False
    )
    report = {
        "target": TARGET,
        "rule": "set ExtraTrees rare-proxy top 26 to normal-format abstain time 11.0",
        "proxy_top26": int(rare26.sum()),
        "v7_active_rows": int(current_active.sum()),
        "v7_active_inside_proxy": int((current_active & rare26).sum()),
        "v7_inactive_inside_proxy": int((~current_active & rare26).sum()),
        "changed_rows": int(changed.sum()),
        "unanimous_inside_proxy": int((unanimous & rare26).sum()),
        "changed_challenge_row_ids": ids[changed].astype(int).tolist(),
        "membership_mask_changed": True,
        "output_format_remains_complete": True,
        "production_candidate_changed": False,
        "token_read": False,
        "submission_performed": False,
    }
    (HERE / "target19_abstain_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
