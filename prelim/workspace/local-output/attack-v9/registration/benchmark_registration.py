"""Truth-known and public-only audit of the v9 affine-registration feature.

This script deliberately separates:

* controlled planted-anchor worlds (membership truth is known);
* actual public releases (mechanism evidence only; membership truth is hidden);
* actual-background/source-swap nulls; and
* the truth-known self-target AIA challenge (member identity/time only).

It never writes a submission, archive, token, or production CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STARTER_SRC = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v3.generate_submission_v3 import PRIVATE_LINK_COLUMNS
from attack_phase_v8.actual_inputs import load_actual_inputs
from attack_phase_v9.registered_linkage import (
    DEFAULT_VIEW_A,
    DEFAULT_VIEW_B,
    RegistrationConfig,
    cross_view_anchor_audit,
    fit_affine_icp,
    registered_event_times,
    registered_linkage_scores,
)


DEFAULT_DIST = Path(
    r"C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider"
)
DEFAULT_TARGETS = ROOT / "team-shared" / "prelim" / "attack-targets"
DEFAULT_OUT = ROOT / "local-output" / "attack-v9" / "registration"
BASE_MIA = (
    ROOT
    / "submissions"
    / "prelim"
    / "team24"
    / "attack-candidate-v7-feedback-stack"
    / "F_mia.csv"
)
NUMERIC = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
CATEGORICAL = ("sex", "prefecture", "smoking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v9 robust registration research")
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _actual_row(
    target: str,
    result,
    base_mia: pd.DataFrame,
    rare: np.ndarray,
    background_null_counts: list[int],
    source,
    release,
) -> dict[str, Any]:
    selected = np.flatnonzero(result.accepted)
    percentile = base_mia[target].rank(method="average", pct=True).to_numpy(float)
    hold_a = cross_view_anchor_audit(
        result.model_a,
        source,
        release,
        validation_columns=DEFAULT_VIEW_B,
        permutations=64,
        random_state=1200 + int(target),
    )
    hold_b = cross_view_anchor_audit(
        result.model_b,
        source,
        release,
        validation_columns=DEFAULT_VIEW_A,
        permutations=64,
        random_state=2200 + int(target),
    )
    potential = selected[percentile[selected] < 0.95]
    row = {
        "target": int(target),
        "accepted_rows": int(len(selected)),
        "accepted_record_ids": base_mia.loc[selected, "record_id"].astype(int).tolist(),
        "audit_false_links": int(result.diagnostics["audit_false_links"]),
        "audit_false_link_rate": float(result.diagnostics["audit_false_link_rate"]),
        "background_null_mean_accepted": float(np.mean(background_null_counts)),
        "background_null_max_accepted": int(np.max(background_null_counts)),
        "base_percentile_mean": float(np.mean(percentile[selected])) if len(selected) else None,
        "accepted_above_base_p95": int(np.count_nonzero(percentile[selected] >= 0.95)),
        "potential_new_rows_below_p95": int(len(potential)),
        "potential_new_record_ids": base_mia.loc[potential, "record_id"].astype(int).tolist(),
        "rare_selected": int(rare[selected].sum()),
        "holdout_a_ratio": hold_a.get("residual_ratio"),
        "holdout_a_p": hold_a.get("permutation_p_lower"),
        "holdout_b_ratio": hold_b.get("residual_ratio"),
        "holdout_b_p": hold_b.get("permutation_p_lower"),
        "model_a_min_inlier": float(np.min(result.model_a.inlier_fractions)),
        "model_b_min_inlier": float(np.min(result.model_b.inlier_fractions)),
        "model_a": result.model_a.as_dict(),
        "model_b": result.model_b.as_dict(),
    }
    # Strict empirical target gate.  C1/C2 are positive mechanism controls but
    # cannot be changed because their measured AIA is already at the ceiling.
    row["target_gate"] = bool(
        target not in {"1", "2"}
        and len(selected) >= np.max(background_null_counts) + 3
        and result.diagnostics["audit_false_link_rate"] <= 0.02
        and hold_a.get("status") == "ok"
        and hold_b.get("status") == "ok"
        and hold_a["residual_ratio"] <= 0.75
        and hold_b["residual_ratio"] <= 0.75
        and hold_a["permutation_p_lower"] <= 0.05
        and hold_b["permutation_p_lower"] <= 0.05
        and len(potential) > 0
    )
    return row


def _controlled_release(
    candidates: pd.DataFrame,
    background: pd.DataFrame,
    eligible: np.ndarray,
    *,
    seed: int,
    noise: float,
    planted: bool,
    anchor_count: int = 26,
) -> tuple[pd.DataFrame, set[int]]:
    rng = np.random.default_rng(seed)
    release = background.sample(n=len(candidates), replace=False, random_state=seed).reset_index(drop=True)
    release = release.astype({column: float for column in NUMERIC})
    slope = rng.uniform(0.88, 1.12, len(NUMERIC))
    source_median = candidates.loc[:, NUMERIC].median().to_numpy(float)
    intercept = rng.uniform(-0.08, 0.08, len(NUMERIC)) * source_median
    scale = release.loc[:, NUMERIC].std().replace(0, 1).to_numpy(float)
    release.loc[:, NUMERIC] = release.loc[:, NUMERIC].to_numpy(float) * slope + intercept
    release.loc[:, NUMERIC] += rng.normal(0, noise * scale, (len(release), len(NUMERIC)))
    planted_indices: set[int] = set()
    if planted:
        source_indices = rng.choice(np.flatnonzero(eligible), size=anchor_count, replace=False)
        release_indices = rng.choice(len(release), size=anchor_count, replace=False)
        release.loc[release_indices, NUMERIC] = (
            candidates.loc[source_indices, NUMERIC].to_numpy(float) * slope
            + intercept
            + rng.normal(0, noise * scale, (anchor_count, len(NUMERIC)))
        )
        release.loc[release_indices, CATEGORICAL] = candidates.loc[
            source_indices, CATEGORICAL
        ].to_numpy()
        planted_indices = set(map(int, source_indices))
    return release, planted_indices


def _controlled_row(
    inputs,
    *,
    seed: int,
    noise: float,
    stream: str,
    planted: bool,
    config: RegistrationConfig,
) -> dict[str, Any]:
    eligible = ~(inputs.known_private | inputs.rare)
    release, truth = _controlled_release(
        inputs.candidates,
        inputs.background,
        eligible,
        seed=seed,
        noise=noise,
        planted=planted,
    )
    result = registered_linkage_scores(
        inputs.candidates,
        release,
        inputs.known_private,
        protected_mask=inputs.rare,
        config=RegistrationConfig(**{**config.__dict__, "random_state": seed}),
    )
    selected = set(map(int, np.flatnonzero(result.accepted)))
    true_positive = len(selected & truth)
    return {
        "stream": stream,
        "seed": seed,
        "noise_fraction": noise,
        "planted": planted,
        "truth_anchors": len(truth),
        "accepted": len(selected),
        "true_positive": true_positive,
        "false_positive": len(selected - truth),
        "precision": true_positive / len(selected) if selected else (1.0 if not truth else 0.0),
        "recall": true_positive / len(truth) if truth else None,
        "audit_false_link_rate": result.diagnostics["audit_false_link_rate"],
    }


def _self_aia_audit(inputs, targets: Path, config: RegistrationConfig) -> dict[str, Any]:
    release = pd.read_csv(targets / "C_24.csv")
    challenge = inputs.self_challenge.reset_index(drop=True)
    numeric = ("age", "SBP", "TG", "HDL", "ALT")
    categorical = ("sex", "prefecture")
    fit_mask = np.ones(len(inputs.candidates), dtype=bool)
    models = {
        "view_a": fit_affine_icp(
            inputs.candidates,
            release,
            fit_mask,
            fit_view=("age", "SBP", "TG"),
            numeric_columns=numeric,
            categorical_columns=categorical,
            config=config,
        ),
        "view_b": fit_affine_icp(
            inputs.candidates,
            release,
            fit_mask,
            fit_view=("HDL", "ALT"),
            numeric_columns=numeric,
            categorical_columns=categorical,
            config=RegistrationConfig(**{**config.__dict__, "random_state": 18181}),
        ),
    }
    linked = {
        name: registered_event_times(challenge, release, model)
        for name, model in models.items()
    }
    truth = challenge.merge(
        inputs.candidates[[*PRIVATE_LINK_COLUMNS, "time", "onset", "death"]],
        on=list(PRIVATE_LINK_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    member = truth["time"].notna().to_numpy()
    truth_time = truth["time"].fillna(-999).to_numpy(float)
    onset_event = truth["onset"].fillna(0).to_numpy(float) > 0
    death_event = truth["death"].fillna(0).to_numpy(float) > 0
    death_only_event = death_event & ~onset_event
    any_event = onset_event | death_event
    truth_counts = {
        "known_member_rows": int(member.sum()),
        "known_member_onset_events": int(np.count_nonzero(member & onset_event)),
        "known_member_death_only_events": int(np.count_nonzero(member & death_only_event)),
        "known_member_any_events": int(np.count_nonzero(member & any_event)),
    }
    rows: dict[str, Any] = {}
    for name, (prediction, audit) in linked.items():
        active = prediction < 11.0
        correct_all = member & active & (np.abs(prediction - truth_time) <= 0.5)
        rows[name] = {
            **audit,
            **truth_counts,
            "member_active": int(np.count_nonzero(member & active)),
            "control_active": int(np.count_nonzero(~member & active)),
            "correct_member_times_all": int(correct_all.sum()),
            "correct_member_any_event_times": int(np.count_nonzero(correct_all & any_event)),
            "correct_member_onset_times": int(np.count_nonzero(correct_all & onset_event)),
            "correct_member_death_only_times": int(
                np.count_nonzero(correct_all & death_only_event)
            ),
        }
    pred_a, pred_b = linked["view_a"][0], linked["view_b"][0]
    agreement = (pred_a < 11.0) & (pred_b < 11.0) & (np.abs(pred_a - pred_b) <= 0.5)
    agreement_time = np.where(agreement, 0.5 * (pred_a + pred_b), 11.0)
    correct_all = member & agreement & (np.abs(agreement_time - truth_time) <= 0.5)
    rows["two_model_agreement"] = {
        **truth_counts,
        "decoded_event_rows": int(agreement.sum()),
        "member_active": int(np.count_nonzero(member & agreement)),
        "control_active": int(np.count_nonzero(~member & agreement)),
        "correct_member_times_all": int(correct_all.sum()),
        "correct_member_any_event_times": int(np.count_nonzero(correct_all & any_event)),
        "correct_member_onset_times": int(np.count_nonzero(correct_all & onset_event)),
        "correct_member_death_only_times": int(
            np.count_nonzero(correct_all & death_only_event)
        ),
    }
    return rows


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_actual_inputs(args.dist.resolve(), args.targets.resolve())
    base_mia = pd.read_csv(BASE_MIA)
    if not np.array_equal(
        base_mia["record_id"].to_numpy(dtype=int),
        inputs.candidates["record_id"].to_numpy(dtype=int),
    ):
        raise ValueError("base MIA does not align to actual B record_id")
    config = RegistrationConfig(iterations=2, max_tentative_anchors=120, ransac_trials=180)

    null_results = []
    for seed in (7001, 7003, 7009):
        null_release = inputs.background.sample(
            n=len(inputs.candidates), replace=False, random_state=seed
        ).reset_index(drop=True)
        null_results.append(
            registered_linkage_scores(
                inputs.candidates,
                null_release,
                inputs.known_private,
                protected_mask=inputs.rare,
                config=RegistrationConfig(**{**config.__dict__, "random_state": seed}),
            )
        )
    null_counts = [int(result.accepted.sum()) for result in null_results]

    actual_rows = []
    actual_details = {}
    for target in sorted(inputs.releases, key=int):
        result = registered_linkage_scores(
            inputs.candidates,
            inputs.releases[target],
            inputs.known_private,
            protected_mask=inputs.rare,
            config=RegistrationConfig(
                **{**config.__dict__, "random_state": config.random_state + int(target)}
            ),
        )
        row = _actual_row(
            target,
            result,
            base_mia,
            inputs.rare,
            null_counts,
            inputs.candidates,
            inputs.releases[target],
        )
        actual_details[target] = row
        actual_rows.append({key: value for key, value in row.items() if key not in {"model_a", "model_b", "accepted_record_ids", "potential_new_record_ids"}})
        print(f"actual target {target}: accepted={row['accepted_rows']} gate={row['target_gate']}", flush=True)

    discovery_specs = [(101, 0.005), (103, 0.015), (107, 0.030), (109, 0.050)]
    validation_specs = [
        (211, 0.008),
        (223, 0.012),
        (227, 0.020),
        (229, 0.028),
        (233, 0.035),
        (239, 0.045),
        (241, 0.055),
        (251, 0.065),
    ]
    synthetic_rows = []
    for stream, specs in (("discovery", discovery_specs), ("fresh_validation", validation_specs)):
        for seed, noise in specs:
            for planted in (True, False):
                synthetic_rows.append(
                    _controlled_row(
                        inputs,
                        seed=seed,
                        noise=noise,
                        stream=stream,
                        planted=planted,
                        config=config,
                    )
                )
                print(f"{stream} seed={seed} planted={planted}", flush=True)
    synthetic = pd.DataFrame(synthetic_rows)
    validation_positive = synthetic.loc[
        synthetic["stream"].eq("fresh_validation") & synthetic["planted"]
    ]
    validation_null = synthetic.loc[
        synthetic["stream"].eq("fresh_validation") & ~synthetic["planted"]
    ]
    synthetic_gate = bool(
        validation_positive["precision"].mean() >= 0.80
        and validation_positive["precision"].quantile(0.10) >= 0.60
        and validation_positive["recall"].mean() >= 0.25
        and validation_null["accepted"].max() <= 5
        and validation_positive["audit_false_link_rate"].quantile(0.90) <= 0.03
    )
    gated_targets = [str(row["target"]) for row in actual_rows if row["target_gate"]]
    self_aia = _self_aia_audit(inputs, args.targets.resolve(), config)
    self_aia_internal_signal = any(
        values["correct_member_times_all"] > 0 and values["control_active"] == 0
        for values in self_aia.values()
    )
    promotion = bool(synthetic_gate and gated_targets)

    pd.DataFrame(actual_rows).to_csv(output / "actual_target_diagnostics.csv", index=False)
    synthetic.to_csv(output / "synthetic_worlds.csv", index=False)
    report = {
        "report_type": "authorized_local_registration_research",
        "data_provenance": inputs.provenance,
        "configuration": config.__dict__,
        "negative_controls": {
            "actual_background_seeds": [7001, 7003, 7009],
            "accepted_rows": null_counts,
            "meaning": "independent actual A_bg cohorts; no B24 membership anchors",
        },
        "synthetic_summary": {
            "gate": synthetic_gate,
            "fresh_validation_positive_precision_mean": float(validation_positive["precision"].mean()),
            "fresh_validation_positive_precision_p10": float(validation_positive["precision"].quantile(0.10)),
            "fresh_validation_positive_recall_mean": float(validation_positive["recall"].mean()),
            "fresh_validation_null_accepted_max": int(validation_null["accepted"].max()),
        },
        "actual_targets": actual_details,
        "self_target_aia": self_aia,
        "decision": {
            "synthetic_gate": synthetic_gate,
            "empirically_gated_non_ceiling_targets": gated_targets,
            "self_aia_internal_signal": self_aia_internal_signal,
            "promote_mia_overlay": promotion,
            # A self-target mechanism check can never by itself authorize
            # transfer to another target's generator.
            "promote_aia_decoder": False,
            "future_transfer_validation_required": True,
            "production_files_changed": 0,
            "reason": (
                "Only public-mechanism evidence is available per attacked target; "
                "promotion additionally requires strict anchor-view/null gates. "
                "Self-target evidence cannot approve cross-target AIA transfer."
            ),
        },
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (output / "registration_report.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "candidate_plan.json").write_text(
        json.dumps(_jsonable(report["decision"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(report["decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
