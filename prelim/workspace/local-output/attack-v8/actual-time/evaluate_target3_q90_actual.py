"""Actual-data audit of a target-3 q90 abstention-only event-time addition.

The rule was selected before these truth-known data were made available.  This
script therefore treats the actual team-24 B/A_bg world as validation, not as a
second discovery screen.  It never reads token.txt and never writes a
submission directory or archive.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER = ROOT / "starter" / "src"
TIME_MODEL = ROOT / "local-output" / "attack-v8" / "time-model"
for source_root in (ROOT, STARTER, TIME_MODEL):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase_daybreak_independent.generate_submission_daybreak_independent import (  # noqa: E402
    build_aia_independent,
)
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    QI_COLUMNS,
    copula_event_links,
    filter_linkable_releases,
)
from attack_phase_v8.event_additions import add_trusted_events  # noqa: E402
from mechanism_decoders import (  # noqa: E402
    conditional_hazard_prediction,
    interval_event_links,
)
from block_copula import (  # noqa: E402
    block_conditional_copula_event_links,
    block_extra_trees_event_prediction,
    block_knn_event_window_prediction,
)
from pwscup2026.common.config import load_config  # noqa: E402
from pwscup2026.kit.validate import KIT_CONFIG_PATH  # noqa: E402
from pwscup2026.scoring.attack_score import score_aia  # noqa: E402


ACTUAL = Path(
    r"C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
)
TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
TOY_DIST = ROOT / "participant_data"
V7 = (
    ROOT
    / "submissions"
    / "prelim"
    / "team24"
    / "attack-candidate-v7-feedback-stack"
    / "F_aia.csv"
)
EXPECTED_HASHES = {
    "B.csv": "7A7AA39E16FA5C76EE9C8DC009928373AEC6CF47E7B1BD10DC4E6899B5D2AB3A",
    "B_self.csv": "297354A3C909FBA6BF4D33053CF91EFA9139892BF5269050A5607B86E0F33E35",
    "A_bg.csv": "F4817FBA6C1434D6B74221C2B5ABE90E4A53E80D8FA17E12602D43ADAE0A3126",
    "schema.json": "D0698EB89CDE0901B28DDAF72CB57270D744BDB72B45F8315F75B3199F024128",
}
ATTACKED_TARGETS = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12",
    "13", "17", "18", "19", "20", "22", "23", "26", "29",
)
ALL_PUBLIC_TARGETS = (*ATTACKED_TARGETS, "24")
VALIDATION_SEEDS = tuple(range(20271031, 20271039))
ABSTAIN = 11.0
HORIZON = 10.0
EDGE = 9.5
OFFICIAL_MEMBER_STRATA = {
    (False, 0): 138,
    (False, 1): 12,
    (True, 0): 9,
    (True, 1): 4,
}
CHALLENGE_COLUMNS = (*QI_COLUMNS, "sex", "prefecture")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_inputs() -> None:
    for name, expected in EXPECTED_HASHES.items():
        observed = sha256(ACTUAL / name)
        if observed != expected:
            raise ValueError(f"actual input hash mismatch: {name}: {observed}")


def load_public_frames(
    target_ids: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    releases = {
        target: pd.read_csv(TARGETS / f"C_{target}.csv").reset_index(drop=True)
        for target in target_ids
    }
    challenges = {
        target: pd.read_csv(
            TARGETS / f"aia_challenge_{target}.csv"
        ).reset_index(drop=True)
        for target in target_ids
    }
    return releases, challenges


def select_official_members(
    b_self: pd.DataFrame, rng: np.random.Generator, home_prefecture: str
) -> np.ndarray:
    selected: list[int] = []
    rare = b_self["is_rare"].astype(bool).to_numpy()
    onset = b_self["onset"].astype(int).to_numpy()
    home = b_self["prefecture"].astype(str).to_numpy() == str(home_prefecture)
    for (rare_value, onset_value), count in OFFICIAL_MEMBER_STRATA.items():
        eligible = np.flatnonzero(
            home & (rare == rare_value) & (onset == onset_value)
        )
        if len(eligible) < count:
            raise ValueError("actual B cannot supply the official member strata")
        selected.extend(rng.choice(eligible, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def make_truth_known_world(
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    background_rare: np.ndarray,
    seed: int,
    home_prefecture: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    member_indices = select_official_members(b_self, rng, home_prefecture)
    members = b_self.iloc[member_indices].reset_index(drop=True)
    member_rare = members["is_rare"].astype(bool).to_numpy()
    eligible_control_indices = np.flatnonzero(
        background["prefecture"].astype(str).to_numpy() == str(home_prefecture)
    )
    home_background = background.iloc[eligible_control_indices].reset_index(drop=True)
    home_background_rare = background_rare[eligible_control_indices]
    local_control_indices, _ = shadow.match_unique_controls(
        members,
        home_background,
        np.empty(0, dtype=int),
        member_rare=member_rare,
        background_rare=home_background_rare,
    )
    control_indices = eligible_control_indices[local_control_indices]
    controls = background.iloc[control_indices].reset_index(drop=True)
    combined = pd.concat([members, controls], ignore_index=True)
    labels = np.r_[np.ones(len(members), dtype=bool), np.zeros(len(controls), dtype=bool)]
    rare = np.r_[member_rare, background_rare[control_indices]]
    truth = {
        "member": labels,
        "rare": rare.astype(bool),
        "time": combined["time"].to_numpy(dtype=float),
        "onset": combined["onset"].to_numpy(dtype=int),
    }
    order = rng.permutation(len(combined))
    challenge = combined.iloc[order].loc[:, CHALLENGE_COLUMNS].reset_index(drop=True)
    challenge.insert(0, "challenge_row_id", np.arange(len(challenge), dtype=int))
    permuted_truth = {name: values[order] for name, values in truth.items()}
    return challenge, permuted_truth, control_indices


def known_member_mask_in_b(
    b_self: pd.DataFrame, own_challenge: pd.DataFrame
) -> np.ndarray:
    known_keys = pd.MultiIndex.from_frame(own_challenge.loc[:, CHALLENGE_COLUMNS])
    candidate_keys = pd.MultiIndex.from_frame(b_self.loc[:, CHALLENGE_COLUMNS])
    return np.asarray(candidate_keys.isin(known_keys), dtype=bool)


def make_discovery_disjoint_world(
    b_self: pd.DataFrame,
    background: pd.DataFrame,
    background_rare: np.ndarray,
    known_member_mask: np.ndarray,
    seed: int,
    home_prefecture: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Sample only B rows absent from the actual self challenge."""
    rng = np.random.default_rng(seed)
    home = b_self["prefecture"].astype(str).to_numpy() == str(home_prefecture)
    onset = b_self["onset"].astype(int).to_numpy()
    rare = b_self["is_rare"].astype(bool).to_numpy()
    heldout = home & ~known_member_mask
    heldout_rare = np.flatnonzero(heldout & rare)
    chosen: list[int] = heldout_rare.tolist()
    for onset_value, count in ((0, 147), (1, 16)):
        already = int(np.count_nonzero(onset[heldout_rare] == onset_value))
        required = count - already
        eligible = np.flatnonzero(
            heldout & ~rare & (onset == onset_value)
        )
        if required < 0 or len(eligible) < required:
            raise ValueError("insufficient discovery-disjoint home-pref B rows")
        chosen.extend(rng.choice(eligible, size=required, replace=False).tolist())
    members = b_self.iloc[np.asarray(chosen, dtype=int)].reset_index(drop=True)
    member_rare = members["is_rare"].astype(bool).to_numpy()
    eligible_control_indices = np.flatnonzero(
        background["prefecture"].astype(str).to_numpy() == str(home_prefecture)
    )
    home_background = background.iloc[eligible_control_indices].reset_index(drop=True)
    home_background_rare = background_rare[eligible_control_indices]
    local_control_indices, _ = shadow.match_unique_controls(
        members,
        home_background,
        np.empty(0, dtype=int),
        member_rare=member_rare,
        background_rare=home_background_rare,
    )
    control_indices = eligible_control_indices[local_control_indices]
    controls = background.iloc[control_indices].reset_index(drop=True)
    combined = pd.concat([members, controls], ignore_index=True)
    member = np.r_[
        np.ones(len(members), dtype=bool), np.zeros(len(controls), dtype=bool)
    ]
    rare = np.r_[member_rare, background_rare[control_indices]]
    truth = {
        "member": member,
        "rare": rare.astype(bool),
        "time": combined["time"].to_numpy(dtype=float),
        "onset": combined["onset"].to_numpy(dtype=int),
    }
    order = rng.permutation(len(combined))
    challenge = combined.iloc[order].loc[:, CHALLENGE_COLUMNS].reset_index(drop=True)
    challenge.insert(0, "challenge_row_id", np.arange(len(challenge), dtype=int))
    return challenge, {name: value[order] for name, value in truth.items()}, control_indices


def independent_edge_baseline(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    toy_background: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame, _, diagnostics = build_aia_independent(
        toy_background,
        {"24": release},
        {"24": challenge},
        member_count=163,
        horizon=HORIZON,
        abstain_time=ABSTAIN,
    )
    values = (
        frame.loc[frame["target_k"].astype(int).eq(24)]
        .set_index("challenge_row_id")
        .loc[challenge["challenge_row_id"].to_numpy(dtype=int), "time_hat"]
        .to_numpy(dtype=float)
    )
    values[np.isclose(values, HORIZON)] = EDGE
    if int(np.count_nonzero(~np.isclose(values, ABSTAIN))) != 163:
        raise RuntimeError("independent baseline did not select exactly half")
    return values, diagnostics


def score_prediction(
    prediction: np.ndarray, truth: dict[str, np.ndarray], cfg: dict[str, Any]
) -> dict[str, float]:
    member = truth["member"]
    rare = truth["rare"]
    result = score_aia(
        None,
        prediction[member],
        truth["onset"][member],
        truth["time"][member],
        None,
        prediction[~member],
        truth["onset"][~member],
        truth["time"][~member],
        cfg,
        member_rare_mask=rare[member],
        control_rare_mask=rare[~member],
    )
    values = result.breakdown
    return {
        "power": float(result.score),
        "all": float(values["r_time_all"]),
        "rare": float(values["r_time_rare"]),
        "p_m": float(values["succ_time_member"]),
        "p_c": float(values["succ_time_control"]),
        "p_m_rare": float(values["succ_time_member_rare"]),
        "p_c_rare": float(values["succ_time_control_rare"]),
    }


def evaluate_method(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    method: str,
    challenge: pd.DataFrame,
    truth: dict[str, np.ndarray],
    base: np.ndarray,
    linked: np.ndarray,
    gate: np.ndarray,
    cfg: dict[str, Any],
) -> None:
    inactive = np.isclose(base, ABSTAIN)
    add = inactive & np.asarray(gate, dtype=bool) & (linked < HORIZON)
    proposal = base.copy()
    proposal[add] = linked[add]
    base_result = score_prediction(base, truth, cfg)
    result = score_prediction(proposal, truth, cfg)
    member = truth["member"]
    rare = truth["rare"]
    success = np.abs(linked - truth["time"]) <= 0.5
    rows.append(
        {
            "seed": seed,
            "method": method,
            "gate_rows": int(gate.sum()),
            "inactive_gate_rows": int(add.sum()),
            "inactive_member_gate_rows": int(np.count_nonzero(add & member)),
            "inactive_control_gate_rows": int(np.count_nonzero(add & ~member)),
            "inactive_member_hits": int(np.count_nonzero(add & member & success)),
            "inactive_control_hits": int(np.count_nonzero(add & ~member & success)),
            "inactive_rare_member_gate_rows": int(
                np.count_nonzero(add & member & rare)
            ),
            "inactive_rare_control_gate_rows": int(
                np.count_nonzero(add & ~member & rare)
            ),
            "base_power": base_result["power"],
            "power": result["power"],
            "delta": result["power"] - base_result["power"],
            "delta_all": result["all"] - base_result["all"],
            "delta_rare": result["rare"] - base_result["rare"],
            "base_all": base_result["all"],
            "all": result["all"],
            "base_rare": base_result["rare"],
            "rare": result["rare"],
            "base_p_m": base_result["p_m"],
            "p_m": result["p_m"],
            "base_p_c": base_result["p_c"],
            "p_c": result["p_c"],
            "base_p_m_rare": base_result["p_m_rare"],
            "p_m_rare": result["p_m_rare"],
            "base_p_c_rare": base_result["p_c_rare"],
            "p_c_rare": result["p_c_rare"],
        }
    )


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, frame in scores.groupby("method", sort=False):
        records.append(
            {
                "method": method,
                "cases": len(frame),
                "mean_delta": float(frame["delta"].mean()),
                "win_rate": float(np.mean(frame["delta"] > 1e-12)),
                "loss_rate": float(np.mean(frame["delta"] < -1e-12)),
                "p10_delta": float(np.quantile(frame["delta"], 0.10)),
                "min_delta": float(frame["delta"].min()),
                "mean_delta_all": float(frame["delta_all"].mean()),
                "mean_delta_rare": float(frame["delta_rare"].mean()),
                "mean_inactive_gate_rows": float(frame["inactive_gate_rows"].mean()),
                "inactive_member_hits": int(frame["inactive_member_hits"].sum()),
                "inactive_control_hits": int(frame["inactive_control_hits"].sum()),
                "inactive_member_gate_rows": int(
                    frame["inactive_member_gate_rows"].sum()
                ),
                "inactive_control_gate_rows": int(
                    frame["inactive_control_gate_rows"].sum()
                ),
                "score1_delta_if_one_of_20": float(frame["delta"].mean() / 20.0),
            }
        )
    return pd.DataFrame(records)


def actual_background_blocks(
    release: pd.DataFrame,
    background: pd.DataFrame,
    *,
    home_prefecture: str,
) -> pd.DataFrame:
    """Home-prefecture actual-A_bg false-success audit.

    The first two folds are disjoint 163-row blocks.  Eight 326-row stress
    samples are also reported separately; those intentionally overlap because
    only 389 actual target-3-home-prefecture background rows are available.
    """
    rng = np.random.default_rng(20271109)
    home_rows = np.flatnonzero(
        background["prefecture"].astype(str).to_numpy() == str(home_prefecture)
    )
    order = rng.permutation(home_rows)
    samples: list[tuple[str, int, np.ndarray]] = []
    for fold in range(2):
        samples.append(("disjoint_163", fold, order[fold * 163 : (fold + 1) * 163]))
    for fold in range(8):
        sample_rng = np.random.default_rng(20271200 + fold)
        samples.append(
            (
                "bootstrap_326",
                fold,
                sample_rng.choice(home_rows, size=326, replace=False),
            )
        )
    rows: list[dict[str, Any]] = []
    for sample_kind, fold, indices in samples:
        query = background.iloc[indices].loc[:, CHALLENGE_COLUMNS].reset_index(drop=True)
        query.insert(0, "challenge_row_id", np.arange(len(query), dtype=int))
        truth_time = background.iloc[indices]["time"].to_numpy(dtype=float)
        decoders = {
            "global_q90": copula_event_links(
                release, query, margin_quantile=0.90, horizon_time=HORIZON
            ),
            "block_pref_q90": block_conditional_copula_event_links(
                release,
                query,
                condition_on_sex=False,
                margin_quantile=0.90,
                horizon_time=HORIZON,
            ),
            "block_pref_sex_q90": block_conditional_copula_event_links(
                release,
                query,
                condition_on_sex=True,
                margin_quantile=0.90,
                horizon_time=HORIZON,
            ),
        }
        for method, (linked, gate, _) in decoders.items():
            success = np.abs(linked - truth_time) <= 0.5
            rows.append(
                {
                    "sample_kind": sample_kind,
                    "fold": fold,
                    "method": method,
                    "rows": len(query),
                    "gate_rows": int(gate.sum()),
                    "false_successes": int(np.count_nonzero(gate & success)),
                    "false_success_rate_all": float(np.mean(gate & success)),
                    "false_success_rate_gated": (
                        float(np.mean(success[gate])) if gate.any() else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def known_member_diagnostic(
    release: pd.DataFrame,
    challenge: pd.DataFrame,
    b_self: pd.DataFrame,
    quantile: float,
    *,
    decoder: str,
) -> dict[str, Any]:
    keys = list(CHALLENGE_COLUMNS)
    lookup = b_self.loc[:, [*keys, "time", "onset", "death", "is_rare"]]
    merged = challenge.merge(lookup, on=keys, how="left", validate="one_to_one")
    member = merged["time"].notna().to_numpy()
    if decoder == "global":
        linked, gate, details = copula_event_links(
            release, challenge, margin_quantile=quantile, horizon_time=HORIZON
        )
    elif decoder in {"block_pref", "block_pref_sex"}:
        linked, gate, details = block_conditional_copula_event_links(
            release,
            challenge,
            condition_on_sex=decoder == "block_pref_sex",
            margin_quantile=quantile,
            horizon_time=HORIZON,
        )
    else:
        raise ValueError(f"unknown decoder: {decoder}")
    known_time = merged["time"].fillna(-999.0).to_numpy(dtype=float)
    success = np.abs(linked - known_time) <= 0.5
    rare = merged["is_rare"].fillna(0).astype(bool).to_numpy()
    return {
        "decoder": decoder,
        "known_members": int(member.sum()),
        "known_rare_members": int(np.count_nonzero(member & rare)),
        "gate_rows": int(gate.sum()),
        "member_gate_rows": int(np.count_nonzero(gate & member)),
        "member_successes": int(np.count_nonzero(gate & member & success)),
        "rare_member_gate_rows": int(np.count_nonzero(gate & member & rare)),
        "rare_member_successes": int(
            np.count_nonzero(gate & member & rare & success)
        ),
        "details": details,
    }


def main() -> None:
    verify_inputs()
    cfg = load_config(KIT_CONFIG_PATH)
    b_self = pd.read_csv(ACTUAL / "B_self.csv").reset_index(drop=True)
    background = pd.read_csv(ACTUAL / "A_bg.csv").reset_index(drop=True)
    detector = shadow.load_rare_detector(ACTUAL)
    background_rare = shadow.public_rare_mask(background, detector)
    toy_background = pd.read_csv(TOY_DIST / "A_bg.csv").reset_index(drop=True)

    public_releases, public_challenges = load_public_frames(ALL_PUBLIC_TARGETS)
    linkable_all, filter_details = filter_linkable_releases(public_releases)
    release24 = linkable_all["24"]
    home24 = str(public_challenges["24"]["prefecture"].iloc[0])

    # Predeclared q90 validation on eight new truth-known team-24 worlds.
    validation_rows: list[dict[str, Any]] = []
    for seed in VALIDATION_SEEDS:
        print(f"actual validation seed={seed}", flush=True)
        challenge, truth, control_indices = make_truth_known_world(
            b_self, background, background_rare, seed, home24
        )
        base, _ = independent_edge_baseline(release24, challenge, toy_background)
        calibration_background = background.drop(index=control_indices).reset_index(
            drop=True
        )

        for quantile in (0.75, 0.90):
            linked, gate, _ = copula_event_links(
                release24,
                challenge,
                margin_quantile=quantile,
                horizon_time=HORIZON,
            )
            evaluate_method(
                validation_rows,
                seed=seed,
                method=f"copula_q{int(100 * quantile)}_full",
                challenge=challenge,
                truth=truth,
                base=base,
                linked=linked,
                gate=gate,
                cfg=cfg,
            )

        for condition_on_sex, method in (
            (False, "block_pref_q90_full"),
            (True, "block_pref_sex_q90_full"),
        ):
            linked, gate, _ = block_conditional_copula_event_links(
                release24,
                challenge,
                condition_on_sex=condition_on_sex,
                margin_quantile=0.90,
                horizon_time=HORIZON,
            )
            evaluate_method(
                validation_rows,
                seed=seed,
                method=method,
                challenge=challenge,
                truth=truth,
                base=base,
                linked=linked,
                gate=gate,
                cfg=cfg,
            )

        for quantile, nearest_time, method in (
            (0.95, False, "block_knn_mode_q95_full"),
            (0.98, False, "block_knn_mode_q98_full"),
            (0.98, True, "block_knn_nearest_q98_full"),
        ):
            linked, gate, _ = block_knn_event_window_prediction(
                release24,
                challenge,
                gate_quantile=quantile,
                nearest_event_time=nearest_time,
                horizon_time=HORIZON,
            )
            evaluate_method(
                validation_rows,
                seed=seed,
                method=method,
                challenge=challenge,
                truth=truth,
                base=base,
                linked=linked,
                gate=gate,
                cfg=cfg,
            )

        linked, gate, _ = block_extra_trees_event_prediction(
            release24,
            challenge,
            seed=seed + 6_000_000,
            gate_quantile=0.98,
            horizon_time=HORIZON,
        )
        evaluate_method(
            validation_rows,
            seed=seed,
            method="block_extra_trees_q98_full",
            challenge=challenge,
            truth=truth,
            base=base,
            linked=linked,
            gate=gate,
            cfg=cfg,
        )

        # Missing-release stress preserves the baseline/mask and removes only
        # donor evidence.  This is intentionally not a mechanism-tuned sweep.
        for fraction in (0.75, 0.50, 0.25):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, int(100 * fraction), 24, 3090])
            )
            chosen = np.sort(
                rng.choice(
                    len(release24),
                    size=max(25, int(round(fraction * len(release24)))),
                    replace=False,
                )
            )
            weak_release = release24.iloc[chosen].reset_index(drop=True)
            linked, gate, _ = copula_event_links(
                weak_release,
                challenge,
                margin_quantile=0.90,
                horizon_time=HORIZON,
            )
            evaluate_method(
                validation_rows,
                seed=seed,
                method=f"copula_q90_release_{int(100 * fraction)}pct",
                challenge=challenge,
                truth=truth,
                base=base,
                linked=linked,
                gate=gate,
                cfg=cfg,
            )
            if np.isclose(fraction, 0.50):
                linked, gate, _ = block_conditional_copula_event_links(
                    weak_release,
                    challenge,
                    condition_on_sex=False,
                    margin_quantile=0.90,
                    horizon_time=HORIZON,
                )
                evaluate_method(
                    validation_rows,
                    seed=seed,
                    method="block_pref_q90_release_50pct",
                    challenge=challenge,
                    truth=truth,
                    base=base,
                    linked=linked,
                    gate=gate,
                    cfg=cfg,
                )

        shuffled = release24.copy()
        event = (
            shuffled["onset"].to_numpy(dtype=int)
            + shuffled["death"].to_numpy(dtype=int)
        ) > 0
        event_rows = np.flatnonzero(event)
        rng = np.random.default_rng(seed + 8_888_000)
        shuffled.loc[event_rows, "time"] = shuffled.loc[
            rng.permutation(event_rows), "time"
        ].to_numpy(dtype=float)
        linked, gate, _ = copula_event_links(
            shuffled, challenge, margin_quantile=0.90, horizon_time=HORIZON
        )
        evaluate_method(
            validation_rows,
            seed=seed,
            method="copula_q90_event_time_shuffle",
            challenge=challenge,
            truth=truth,
            base=base,
            linked=linked,
            gate=gate,
            cfg=cfg,
        )
        linked, gate, _ = block_conditional_copula_event_links(
            shuffled,
            challenge,
            condition_on_sex=False,
            margin_quantile=0.90,
            horizon_time=HORIZON,
        )
        evaluate_method(
            validation_rows,
            seed=seed,
            method="block_pref_q90_event_time_shuffle",
            challenge=challenge,
            truth=truth,
            base=base,
            linked=linked,
            gate=gate,
            cfg=cfg,
        )

        linked, gate, _ = interval_event_links(
            release24,
            challenge,
            calibration_background,
            query_cdf="challenge",
            margin_quantile=0.75,
            likelihood_quantile=0.75,
            horizon_time=HORIZON,
        )
        evaluate_method(
            validation_rows,
            seed=seed,
            method="interval_q75_l75_full",
            challenge=challenge,
            truth=truth,
            base=base,
            linked=linked,
            gate=gate,
            cfg=cfg,
        )

        linked, gate, _ = conditional_hazard_prediction(
            release24,
            challenge,
            calibration_background,
            seed=seed + 7_000_000,
            event_probability_quantile=0.90,
            horizon_time=HORIZON,
        )
        evaluate_method(
            validation_rows,
            seed=seed,
            method="conditional_hazard_q90_full",
            challenge=challenge,
            truth=truth,
            base=base,
            linked=linked,
            gate=gate,
            cfg=cfg,
        )

    # A second audit excludes all 163 actual self-challenge members.  It cannot
    # reproduce the 13-rare composition (only one home-pref rare B row remains),
    # so it is an all-channel robustness check rather than a rare-score estimate.
    heldout_rows: list[dict[str, Any]] = []
    known_b = known_member_mask_in_b(b_self, public_challenges["24"])
    for seed in VALIDATION_SEEDS:
        challenge, truth, _ = make_discovery_disjoint_world(
            b_self,
            background,
            background_rare,
            known_b,
            seed + 100_000,
            home24,
        )
        base, _ = independent_edge_baseline(release24, challenge, toy_background)
        method_outputs = {
            "global_q90_discovery_disjoint": copula_event_links(
                release24,
                challenge,
                margin_quantile=0.90,
                horizon_time=HORIZON,
            ),
            "block_pref_q90_discovery_disjoint": (
                block_conditional_copula_event_links(
                    release24,
                    challenge,
                    condition_on_sex=False,
                    margin_quantile=0.90,
                    horizon_time=HORIZON,
                )
            ),
            "block_pref_sex_q90_discovery_disjoint": (
                block_conditional_copula_event_links(
                    release24,
                    challenge,
                    condition_on_sex=True,
                    margin_quantile=0.90,
                    horizon_time=HORIZON,
                )
            ),
            "block_knn_mode_q95_discovery_disjoint": (
                block_knn_event_window_prediction(
                    release24,
                    challenge,
                    gate_quantile=0.95,
                    nearest_event_time=False,
                    horizon_time=HORIZON,
                )
            ),
        }
        for method, (linked, gate, _) in method_outputs.items():
            evaluate_method(
                heldout_rows,
                seed=seed,
                method=method,
                challenge=challenge,
                truth=truth,
                base=base,
                linked=linked,
                gate=gate,
                cfg=cfg,
            )

    heldout_scores = pd.DataFrame(heldout_rows)
    heldout_summary = summarize(heldout_scores)
    heldout_scores.to_csv(
        HERE / "target3_discovery_disjoint_scores.csv", index=False
    )
    heldout_summary.to_csv(
        HERE / "target3_discovery_disjoint_summary.csv", index=False
    )

    scores = pd.DataFrame(validation_rows)
    summary = summarize(scores)
    scores.to_csv(HERE / "target3_q90_actual_validation_scores.csv", index=False)
    summary.to_csv(HERE / "target3_q90_actual_validation_summary.csv", index=False)

    # Production target-3 candidate: exact tested helper, written only to this
    # research directory.  The source v7 CSV remains unchanged.
    releases20, challenges20 = load_public_frames(ATTACKED_TARGETS)
    v7 = pd.read_csv(V7)
    research_candidate, candidate_details = add_trusted_events(
        v7,
        releases20,
        challenges20,
        target_ids=("3",),
        margin_quantile=0.90,
        horizon=HORIZON,
        abstain=ABSTAIN,
    )
    research_candidate.to_csv(HERE / "F_aia_target3_q90_research.csv", index=False)
    before = v7.sort_values(["target_k", "challenge_row_id"], kind="mergesort")
    after = research_candidate.sort_values(
        ["target_k", "challenge_row_id"], kind="mergesort"
    )
    changed = ~np.isclose(
        before["time_hat"].to_numpy(dtype=float),
        after["time_hat"].to_numpy(dtype=float),
    )
    diff = after.loc[changed, ["target_k", "challenge_row_id", "time_hat"]].copy()
    diff.insert(
        2, "v7_time_hat", before.loc[changed, "time_hat"].to_numpy(dtype=float)
    )
    diff.to_csv(HERE / "target3_global_q90_production_diff.csv", index=False)

    linkable20, filter20 = filter_linkable_releases(releases20)
    challenge3 = challenges20["3"].sort_values(
        "challenge_row_id", kind="mergesort"
    ).reset_index(drop=True)
    block_candidate = v7.copy()
    block_index = (
        block_candidate.loc[block_candidate["target_k"].astype(int).eq(3)]
        .sort_values("challenge_row_id", kind="mergesort")
        .index
    )
    block_values = block_candidate.loc[block_index, "time_hat"].to_numpy(
        dtype=float
    )
    block_linked, block_gate, block_details = block_conditional_copula_event_links(
        linkable20["3"],
        challenge3,
        condition_on_sex=False,
        margin_quantile=0.90,
        horizon_time=HORIZON,
    )
    block_add = (
        np.isclose(block_values, ABSTAIN) & block_gate & (block_linked < HORIZON)
    )
    block_values[block_add] = block_linked[block_add]
    block_candidate.loc[block_index, "time_hat"] = block_values
    block_candidate.to_csv(
        HERE / "F_aia_target3_block_pref_q90_research.csv", index=False
    )
    block_after = block_candidate.sort_values(
        ["target_k", "challenge_row_id"], kind="mergesort"
    )
    block_changed = ~np.isclose(
        before["time_hat"].to_numpy(dtype=float),
        block_after["time_hat"].to_numpy(dtype=float),
    )
    block_diff = block_after.loc[
        block_changed, ["target_k", "challenge_row_id", "time_hat"]
    ].copy()
    block_diff.insert(
        2,
        "v7_time_hat",
        before.loc[block_changed, "time_hat"].to_numpy(dtype=float),
    )
    block_diff.to_csv(
        HERE / "target3_block_pref_q90_production_diff.csv", index=False
    )

    background_controls = actual_background_blocks(
        linkable20["3"],
        background,
        home_prefecture=str(challenge3["prefecture"].iloc[0]),
    )
    background_controls.to_csv(
        HERE / "target3_actual_bg_disjoint_controls.csv", index=False
    )

    known24 = {
        decoder: known_member_diagnostic(
            release24,
            public_challenges["24"],
            b_self,
            0.90,
            decoder=decoder,
        )
        for decoder in ("global", "block_pref", "block_pref_sex")
    }
    known_curve_rows: list[dict[str, Any]] = []
    for quantile in (0.0, 0.25, 0.50, 0.75, 0.90):
        for decoder in ("global", "block_pref", "block_pref_sex"):
            item = known_member_diagnostic(
                release24,
                public_challenges["24"],
                b_self,
                quantile,
                decoder=decoder,
            )
            item.pop("details")
            item["margin_quantile"] = quantile
            known_curve_rows.append(item)
    known_curve = pd.DataFrame(known_curve_rows)
    known_curve.to_csv(
        HERE / "target24_known_member_link_curve.csv", index=False
    )
    chosen_summary = summary.loc[
        summary["method"].eq("block_pref_q90_full")
    ].iloc[0]
    weak_summary = summary.loc[
        summary["method"].eq("block_pref_q90_release_50pct")
    ].iloc[0]
    shuffle_summary = summary.loc[
        summary["method"].eq("block_pref_q90_event_time_shuffle")
    ].iloc[0]
    control_q90 = background_controls.loc[
        background_controls["method"].eq("block_pref_q90")
    ]
    checks = {
        "validation_mean_delta_gt_0": float(chosen_summary["mean_delta"]) > 0.0,
        "validation_win_rate_gte_0p60": float(chosen_summary["win_rate"]) >= 0.60,
        "validation_p10_gte_0": float(chosen_summary["p10_delta"]) >= -1e-12,
        "validation_all_mean_gte_0": float(chosen_summary["mean_delta_all"]) >= -1e-12,
        "validation_min_gte_0": float(chosen_summary["min_delta"]) >= -1e-12,
        "weak50_mean_gte_0": float(weak_summary["mean_delta"]) >= -1e-12,
        "disjoint_actual_bg_false_successes_zero": int(
            control_q90["false_successes"].sum()
        )
        == 0,
        "normal_format_abstention_only": bool(
            len(block_diff) > 0
            and block_diff["target_k"].eq(3).all()
            and block_diff["v7_time_hat"].eq(ABSTAIN).all()
            and block_diff["time_hat"].between(
                0.0, HORIZON, inclusive="left"
            ).all()
        ),
    }
    report = {
        "decision": "adopt_candidate" if all(checks.values()) else "reject",
        "predeclared_rule": (
            "target 3 only; fit ranks within its single challenge prefecture; "
            "match within prefecture+sex; preserve all v7 cells; on q90 mutual "
            "event links replace exact 11.0 abstentions with public donor time"
        ),
        "actual_input_hashes": EXPECTED_HASHES,
        "validation_seeds": list(VALIDATION_SEEDS),
        "validation_design": (
            "eight fresh official-composition 163-member/163-control worlds; "
            "members from actual B_self, matched controls from actual A_bg; "
            "fixed public C24 release (not a B-to-C3 warp); v7-like independent "
            "mask fitted with the retained toy A_bg"
        ),
        "inference_boundary": (
            "C24 is a truth-known mechanism check. Its negative result does not "
            "prove C3 links fail, but supplies no evidence that changing the last "
            "submission's C3 abstentions is beneficial."
        ),
        "checks": checks,
        "selected_validation_summary": chosen_summary.to_dict(),
        "discovery_disjoint_all_channel_summary": heldout_summary.to_dict(
            orient="records"
        ),
        "known_target24_member_diagnostic": known24,
        "known_target24_link_curve_max_member_successes": int(
            known_curve["member_successes"].max()
        ),
        "event_time_shuffle_summary": shuffle_summary.to_dict(),
        "actual_bg_disjoint_q90": {
            "folds": int(len(control_q90)),
            "rows": int(control_q90["rows"].sum()),
            "gate_rows": int(control_q90["gate_rows"].sum()),
            "false_successes": int(control_q90["false_successes"].sum()),
            "false_success_rate_all": float(
                control_q90["false_successes"].sum()
                / control_q90["rows"].sum()
            ),
        },
        "production_research_diff": block_diff.to_dict(orient="records"),
        "production_global_comparator_diff": diff.to_dict(orient="records"),
        "candidate_details": {
            "block_pref": {
                "decoder": block_details,
                "added_rows": int(block_add.sum()),
                "release_filter": filter20["3"],
            },
            "global_comparator": candidate_details,
        },
        "target24_release_filter": filter_details["24"],
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (HERE / "target3_q90_actual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps({"decision": report["decision"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
