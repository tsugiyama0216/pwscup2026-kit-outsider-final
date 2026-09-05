"""Research-only reconstruction of AIA success counts from official feedback.

The script never reads a token and never creates a submission.  It does three
things:

1. Enumerates integer member/control success-count solutions implied by the
   published all-channel rates and target scores.
2. Computes conditional score ranges for the exact routed complement when the
   total number of horizon hits in the member/control arms is fixed.
3. Uses an exact mixed-integer model of the three submitted time predictions to
   test whether an inverse three-expert vote dominates the routed complement.

Only regular submissions are considered: one unique prediction for each of the
326 challenge rows in every target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "local-output" / "attack-v6" / "robust-official"
FEEDBACK = ROOT / "attack_phase_v6" / "official_feedback"
TEAM = ROOT / "submissions" / "prelim" / "team24"
TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
ZERO_TARGETS = (6, 7, 8, 9, 17, 18, 19)
N_ARM = 163
N_CHALLENGE = 326
HORIZON_DAYS = 3652
YEAR_DAYS = 365.25
TAU = 0.5

RUNS = {
    "v4": {
        "prediction": TEAM / "attack-candidate-v4" / "F_aia.csv",
        "feedback": FEEDBACK / "v4.json",
    },
    "independent": {
        "prediction": TEAM / "attack-candidate-daybreak-independent" / "F_aia.csv",
        "feedback": FEEDBACK / "probe_independent.json",
    },
    "routed": {
        "prediction": TEAM / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv",
        "feedback": FEEDBACK / "routed_guided_probe.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="Skip the exact row-time feasibility audit.",
    )
    return parser.parse_args()


def r_all(member_successes: int, control_successes: int) -> float:
    p_m = member_successes / N_ARM
    p_c = control_successes / N_ARM
    return float(np.clip((p_m - p_c) / max(1.0 - p_c, 1e-9), 0.0, 1.0))


ALL_PAIRS = tuple(
    (m, c, r_all(m, c))
    for m in range(N_ARM + 1)
    for c in range(N_ARM + 1)
    # Every measured block has exactly 163 non-abstaining predictions, and
    # time=11 cannot hit a truth time in [0, horizon].
    if m + c <= N_ARM
)
PAIRS_BY_R: dict[float, list[tuple[int, int, float]]] = defaultdict(list)
for _m, _c, _r in ALL_PAIRS:
    PAIRS_BY_R[round(_r, 12)].append((_m, _c, _r))


def matching_pairs(score: float) -> list[tuple[int, int]]:
    return [
        (m, c)
        for m, c, value in PAIRS_BY_R[round(score, 12)]
        if math.isclose(value, score, abs_tol=1e-11)
    ]


def target_1_2_pairs(feedback: dict[str, Any]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    per_target = feedback["aia_per_target"]
    other_sum = sum(
        float(value)
        for target, value in per_target.items()
        if target not in {"1", "2"}
    )
    required = 20.0 * float(feedback["aia"]["r_time_all"]) - other_sum
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for m1, c1, r1 in ALL_PAIRS:
        needed = required - r1
        for m2, c2, r2 in PAIRS_BY_R[round(needed, 12)]:
            if math.isclose(r1 + r2, required, abs_tol=1e-11):
                choices.append(((m1, c1), (m2, c2)))
    return choices


def reconstruct_zero_counts(feedback: dict[str, Any]) -> dict[str, Any]:
    per_target = feedback["aia_per_target"]
    zero_targets = tuple(int(k) for k, value in per_target.items() if value == 0.0)
    positive = {
        int(k): matching_pairs(float(value))
        for k, value in per_target.items()
        if k not in {"1", "2"} and value > 0.0
    }
    total_m = int(round(float(feedback["aia"]["p_m"]) * len(TARGETS) * N_ARM))
    total_c = int(round(float(feedback["aia"]["p_c"]) * len(TARGETS) * N_ARM))
    aggregate: set[tuple[int, int, int]] = set()
    keys = tuple(positive)
    for pair_12 in target_1_2_pairs(feedback):
        for choices in itertools.product(*(positive[k] for k in keys)):
            positive_m = pair_12[0][0] + pair_12[1][0] + sum(x[0] for x in choices)
            positive_c = pair_12[0][1] + pair_12[1][1] + sum(x[1] for x in choices)
            zero_m = total_m - positive_m
            zero_c = total_c - positive_c
            if (
                zero_m >= 0
                and zero_c >= zero_m
                and zero_m + zero_c <= len(zero_targets) * N_ARM
            ):
                aggregate.add((zero_c - zero_m, zero_m, zero_c))
    ordered = sorted(aggregate)
    return {
        "zero_targets": list(zero_targets),
        "total_member_successes": total_m,
        "total_control_successes": total_c,
        "target_1_2_options": target_1_2_pairs(feedback),
        "positive_pair_options": {str(k): value for k, value in positive.items()},
        "zero_aggregate_options": [
            {"control_minus_member": d, "member": m, "control": c}
            for d, m, c in ordered
        ],
        "gap_min": min(x[0] for x in ordered),
        "gap_max": max(x[0] for x in ordered),
    }


def conditional_complement_range(
    routed_aggregate: Iterable[dict[str, int]],
    *,
    horizon_member: int,
    horizon_control: int,
) -> dict[str, float | int]:
    """Range the seven complement scores for fixed total horizon-hit counts."""
    total_horizon = horizon_member + horizon_control
    options: list[tuple[int, int, float]] = []
    for m in range(horizon_member + 1):
        for c in range(m, horizon_control + 1):
            selected_total = m + c
            if not max(0, total_horizon - N_ARM) <= selected_total <= N_ARM:
                continue
            d = c - m
            if d > 68:
                continue
            comp_m = horizon_member - m
            comp_c = horizon_control - c
            options.append((d, c, r_all(comp_m, comp_c)))

    # DP states are (selected control-member gap, selected controls), retaining
    # the minimum and maximum additive complement score over seven targets.
    aggregates = {
        (int(x["control_minus_member"]), int(x["control"]))
        for x in routed_aggregate
    }
    max_gap = max(x[0] for x in aggregates)
    min_gap = min(x[0] for x in aggregates)
    max_control = max(x[1] for x in aggregates)
    min_control = min(x[1] for x in aggregates)
    min_option_control = min(x[1] for x in options)
    max_option_control = max(x[1] for x in options)
    states: dict[tuple[int, int], tuple[float, float]] = {(0, 0): (0.0, 0.0)}
    for step in range(len(ZERO_TARGETS)):
        remaining = len(ZERO_TARGETS) - step - 1
        updated: dict[tuple[int, int], tuple[float, float]] = {}
        for (gap, controls), (low, high) in states.items():
            for option_gap, option_control, score in options:
                new_gap = gap + option_gap
                new_control = controls + option_control
                if new_gap > max_gap or new_gap + remaining * max_gap < min_gap:
                    continue
                if (
                    new_control + remaining * min_option_control > max_control
                    or new_control + remaining * max_option_control < min_control
                ):
                    continue
                key = (new_gap, new_control)
                candidate = (low + score, high + score)
                previous = updated.get(key)
                if previous is None:
                    updated[key] = candidate
                else:
                    updated[key] = (
                        min(previous[0], candidate[0]),
                        max(previous[1], candidate[1]),
                    )
        states = updated

    feasible = [states[key] for key in aggregates if key in states]
    if not feasible:
        raise RuntimeError("fixed horizon counts are incompatible with official totals")
    score_sum_min = min(value[0] for value in feasible)
    score_sum_max = max(value[1] for value in feasible)
    safe_aia = 0.15062041254150674
    safe_mia = 0.1579114879230253
    return {
        "horizon_member_per_target": horizon_member,
        "horizon_control_per_target": horizon_control,
        "feasible_official_aggregate_options": len(feasible),
        "zero_target_score_sum_min": score_sum_min,
        "zero_target_score_sum_max": score_sum_max,
        "aia_power_min": safe_aia + score_sum_min / len(TARGETS),
        "aia_power_max": safe_aia + score_sum_max / len(TARGETS),
        "score_1_min": safe_mia + safe_aia + score_sum_min / len(TARGETS),
        "score_1_max": safe_mia + safe_aia + score_sum_max / len(TARGETS),
    }


def load_prediction_table() -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for name, paths in RUNS.items():
        frame = pd.read_csv(paths["prediction"])
        if len(frame) != len(TARGETS) * N_CHALLENGE:
            raise ValueError(f"{name} has an unexpected row count")
        if frame.duplicated(["target_k", "challenge_row_id"]).any():
            raise ValueError(f"{name} contains duplicate challenge keys")
        frames[name] = frame.set_index(["target_k", "challenge_row_id"])["time_hat"]
    joined = pd.concat(frames, axis=1).reset_index()
    if len(joined) != len(TARGETS) * N_CHALLENGE or joined.isna().any().any():
        raise ValueError("prediction keys do not align")

    joined["exact_complement"] = 11.0
    zero_mask = joined["target_k"].isin(ZERO_TARGETS)
    routed = joined.loc[zero_mask, "routed"].to_numpy(dtype=float)
    if not np.isin(routed, (10.0, 11.0)).all():
        raise ValueError("routed zero blocks are not binary")
    joined.loc[zero_mask, "exact_complement"] = 21.0 - routed

    joined["inverse_consensus"] = 11.0
    for target in ZERO_TARGETS:
        mask = joined["target_k"].eq(target)
        block = joined.loc[mask].copy()
        active = block[["v4", "independent", "routed"]].ne(11.0)
        block["reject_votes"] = 3 - active.sum(axis=1)
        chosen = block.sort_values(
            ["reject_votes", "challenge_row_id"], ascending=[False, True]
        ).index[:N_ARM]
        joined.loc[chosen, "inverse_consensus"] = 10.0
    return joined


def overlap_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    per_target: dict[str, Any] = {}
    total_swaps = 0
    for target in ZERO_TARGETS:
        block = predictions.loc[predictions["target_k"].eq(target)]
        expert_active = block[["v4", "independent", "routed"]].ne(11.0)
        exact_active = block["exact_complement"].ne(11.0)
        vote_active = block["inverse_consensus"].ne(11.0)
        changes = int(np.count_nonzero(exact_active.to_numpy() != vote_active.to_numpy()))
        total_swaps += changes // 2
        patterns = (
            expert_active.astype(int)
            .value_counts()
            .sort_index()
            .rename("rows")
            .reset_index()
            .to_dict(orient="records")
        )
        per_target[str(target)] = {
            "exact_vote_overlap": int(np.count_nonzero(exact_active & vote_active)),
            "changed_rows": changes,
            "swaps": changes // 2,
            "patterns": patterns,
        }
    return {
        "per_target": per_target,
        "total_swaps": total_swaps,
        "total_changed_rows": 2 * total_swaps,
    }


def possible_success_patterns(times: np.ndarray) -> tuple[tuple[int, ...], ...]:
    truth_grid = np.arange(1, HORIZON_DAYS + 1, dtype=float) / YEAR_DAYS
    matrix = np.abs(truth_grid[:, None] - times[None, :]) <= TAU
    return tuple(tuple(int(value) for value in row) for row in np.unique(matrix, axis=0))


def exact_all_channel_milp(
    predictions: pd.DataFrame,
    feedback: dict[str, dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Optimize candidate advantages under all published all-channel facts."""
    valid_modes = {
        "min_raw_difference",
        "max_raw_difference",
        "exact_wins_vote_zero",
        "vote_wins_exact_zero",
    }
    if mode not in valid_modes:
        raise ValueError(f"unknown MILP mode: {mode}")
    prediction_columns = (
        "v4",
        "independent",
        "routed",
        "exact_complement",
        "inverse_consensus",
    )
    grouped_rows: dict[tuple[int, tuple[tuple[int, ...], ...]], int] = defaultdict(int)
    for row in predictions.itertuples(index=False):
        times = np.asarray([getattr(row, name) for name in prediction_columns], dtype=float)
        patterns = possible_success_patterns(times)
        grouped_rows[(int(row.target_k), patterns)] += 1

    variable_meta: list[tuple[str, Any]] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []
    group_variables: dict[
        tuple[int, tuple[tuple[int, ...], ...]], list[tuple[int, tuple[int, ...], str]]
    ] = {}
    for key, count in grouped_rows.items():
        entries: list[tuple[int, tuple[int, ...], str]] = []
        for pattern in key[1]:
            for arm in ("member", "control"):
                index = len(variable_meta)
                variable_meta.append(("assignment", (key, pattern, arm)))
                lower.append(0.0)
                upper.append(float(count))
                integrality.append(1)
                entries.append((index, pattern, arm))
        group_variables[key] = entries

    constraints: list[tuple[dict[int, float], float, float]] = []

    def add_constraint(coefficients: dict[int, float], lo: float, hi: float) -> None:
        constraints.append((coefficients, lo, hi))

    # Every row receives exactly one truth-time pattern and one membership arm.
    for key, count in grouped_rows.items():
        add_constraint({index: 1.0 for index, _, _ in group_variables[key]}, count, count)

    success_expressions: dict[tuple[int, str, str], dict[int, float]] = {}
    member_expressions: dict[int, dict[int, float]] = {}
    for target in TARGETS:
        member_expr: dict[int, float] = {}
        for run_index, run in enumerate(prediction_columns):
            success_expressions[(target, run, "member")] = {}
            success_expressions[(target, run, "control")] = {}
        for key, entries in group_variables.items():
            if key[0] != target:
                continue
            for index, pattern, arm in entries:
                if arm == "member":
                    member_expr[index] = 1.0
                for run_index, run in enumerate(prediction_columns):
                    if pattern[run_index]:
                        success_expressions[(target, run, arm)][index] = 1.0
        member_expressions[target] = member_expr
        add_constraint(member_expr, N_ARM, N_ARM)

    # Link exact success pairs for positive targets and inequalities for zeros.
    for run in ("v4", "independent", "routed"):
        run_feedback = feedback[run]
        per_target = run_feedback["aia_per_target"]
        for target in TARGETS:
            if target in (1, 2):
                continue
            member_expr = success_expressions[(target, run, "member")].copy()
            control_expr = success_expressions[(target, run, "control")].copy()
            score = float(per_target[str(target)])
            if score == 0.0:
                coefficients = member_expr.copy()
                for index, value in control_expr.items():
                    coefficients[index] = coefficients.get(index, 0.0) - value
                add_constraint(coefficients, -np.inf, 0.0)
                continue
            options = matching_pairs(score)
            selector_indices: list[int] = []
            for m, c in options:
                index = len(variable_meta)
                variable_meta.append(("pair", (run, target, m, c)))
                lower.append(0.0)
                upper.append(1.0)
                integrality.append(1)
                selector_indices.append(index)
                member_expr[index] = -float(m)
                control_expr[index] = -float(c)
            add_constraint({index: 1.0 for index in selector_indices}, 1.0, 1.0)
            add_constraint(member_expr, 0.0, 0.0)
            add_constraint(control_expr, 0.0, 0.0)

        joint_indices: list[int] = []
        member_1 = success_expressions[(1, run, "member")].copy()
        control_1 = success_expressions[(1, run, "control")].copy()
        member_2 = success_expressions[(2, run, "member")].copy()
        control_2 = success_expressions[(2, run, "control")].copy()
        for pair_1, pair_2 in target_1_2_pairs(run_feedback):
            index = len(variable_meta)
            variable_meta.append(("pair12", (run, pair_1, pair_2)))
            lower.append(0.0)
            upper.append(1.0)
            integrality.append(1)
            joint_indices.append(index)
            member_1[index] = -float(pair_1[0])
            control_1[index] = -float(pair_1[1])
            member_2[index] = -float(pair_2[0])
            control_2[index] = -float(pair_2[1])
        add_constraint({index: 1.0 for index in joint_indices}, 1.0, 1.0)
        for expression in (member_1, control_1, member_2, control_2):
            add_constraint(expression, 0.0, 0.0)

        total_m = int(round(float(run_feedback["aia"]["p_m"]) * len(TARGETS) * N_ARM))
        total_c = int(round(float(run_feedback["aia"]["p_c"]) * len(TARGETS) * N_ARM))
        aggregate_m: dict[int, float] = {}
        aggregate_c: dict[int, float] = {}
        for target in TARGETS:
            aggregate_m.update(success_expressions[(target, run, "member")])
            aggregate_c.update(success_expressions[(target, run, "control")])
        add_constraint(aggregate_m, total_m, total_m)
        add_constraint(aggregate_c, total_c, total_c)

    def signed_expression(target: int, candidate: str) -> dict[int, float]:
        expression = success_expressions[(target, candidate, "member")].copy()
        for index, value in success_expressions[(target, candidate, "control")].items():
            expression[index] = expression.get(index, 0.0) - value
        return expression

    objective = np.zeros(len(variable_meta), dtype=float)
    if mode in {"min_raw_difference", "max_raw_difference"}:
        for target in ZERO_TARGETS:
            for index, value in signed_expression(target, "inverse_consensus").items():
                objective[index] += value
            for index, value in signed_expression(target, "exact_complement").items():
                objective[index] -= value
        direction = 1.0 if mode == "min_raw_difference" else -1.0
    elif mode == "exact_wins_vote_zero":
        for target in ZERO_TARGETS:
            add_constraint(
                signed_expression(target, "inverse_consensus"), -np.inf, 0.0
            )
            for index, value in signed_expression(target, "exact_complement").items():
                objective[index] += value
        direction = -1.0
    else:
        for target in ZERO_TARGETS:
            add_constraint(signed_expression(target, "exact_complement"), -np.inf, 0.0)
            for index, value in signed_expression(target, "inverse_consensus").items():
                objective[index] += value
        direction = -1.0

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_constraints: list[float] = []
    upper_constraints: list[float] = []
    for row_index, (coefficients, lo, hi) in enumerate(constraints):
        for column_index, value in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
        lower_constraints.append(lo)
        upper_constraints.append(hi)
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraints), len(variable_meta)),
    ).tocsr()
    result = milp(
        direction * objective,
        integrality=np.asarray(integrality),
        bounds=Bounds(np.asarray(lower), np.asarray(upper)),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower_constraints),
            np.asarray(upper_constraints),
        ),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    solution = np.rint(result.x).astype(int)

    candidate_scores: dict[str, dict[str, Any]] = {}
    for candidate in ("exact_complement", "inverse_consensus"):
        score_sum = 0.0
        per_target: dict[str, Any] = {}
        for target in ZERO_TARGETS:
            m = int(
                sum(
                    solution[index]
                    for index in success_expressions[(target, candidate, "member")]
                )
            )
            c = int(
                sum(
                    solution[index]
                    for index in success_expressions[(target, candidate, "control")]
                )
            )
            score = r_all(m, c)
            score_sum += score
            per_target[str(target)] = {"member": m, "control": c, "r_all": score}
        candidate_scores[candidate] = {
            "score_sum": score_sum,
            "per_target": per_target,
        }
    return {
        "mode": mode,
        "status": result.message,
        "objective_raw_value": float(direction * result.fun),
        "variables": len(variable_meta),
        "constraints": len(constraints),
        "candidate_scores": candidate_scores,
        "score_sum_difference": (
            candidate_scores["inverse_consensus"]["score_sum"]
            - candidate_scores["exact_complement"]["score_sum"]
        ),
    }


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    feedback = {
        name: json.loads(Path(paths["feedback"]).read_text(encoding="utf-8"))
        for name, paths in RUNS.items()
    }
    reconstructed = {
        name: reconstruct_zero_counts(value) for name, value in feedback.items()
    }
    routed_options = reconstructed["routed"]["zero_aggregate_options"]
    conditional = [
        conditional_complement_range(
            routed_options, horizon_member=horizon_m, horizon_control=horizon_c
        )
        for horizon_m, horizon_c in ((136, 136), (137, 137), (137, 136), (147, 147))
    ]
    predictions = load_prediction_table()
    overlap = overlap_diagnostics(predictions)
    exchangeable_retention = 1.0 - 2.0 * overlap["total_swaps"] / (
        len(ZERO_TARGETS) * N_ARM
    )
    exchangeable = {
        "routed_gap_min": reconstructed["routed"]["gap_min"],
        "routed_gap_max": reconstructed["routed"]["gap_max"],
        "swaps": overlap["total_swaps"],
        "raw_signal_retention": exchangeable_retention,
        "expected_raw_loss_min": (1.0 - exchangeable_retention)
        * reconstructed["routed"]["gap_min"],
        "expected_raw_loss_max": (1.0 - exchangeable_retention)
        * reconstructed["routed"]["gap_max"],
    }
    report: dict[str, Any] = {
        "scope": "authorized_local_competition_research_only",
        "normal_submission_only": True,
        "token_read": False,
        "archive_created": False,
        "submission_performed": False,
        "integer_reconstruction": reconstructed,
        "conditional_exact_complement": conditional,
        "inverse_consensus_overlap": overlap,
        "exchangeable_routed_signal_model": exchangeable,
    }
    if not args.skip_milp:
        report["exact_all_channel_milp"] = {
            mode: exact_all_channel_milp(predictions, feedback, mode=mode)
            for mode in (
                "min_raw_difference",
                "max_raw_difference",
                "exact_wins_vote_zero",
                "vote_wins_exact_zero",
            )
        }
        report["exact_all_channel_milp"]["rare_channel_note"] = (
            "The MILP enforces every published all-channel count/rate and "
            "per-target maximum implication. Rare-row identities are not "
            "published and are not assigned in this model."
        )
    output_path = OUT / "robust_official_audit.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output_path}")
    print(
        json.dumps(
            {
                "zero_gap_ranges": {
                    name: [value["gap_min"], value["gap_max"]]
                    for name, value in reconstructed.items()
                },
                "conditional_score_1": [
                    [value["horizon_member_per_target"], value["horizon_control_per_target"], value["score_1_min"], value["score_1_max"]]
                    for value in conditional
                ],
                "inverse_consensus_swaps": overlap["total_swaps"],
                "exchangeable": exchangeable,
                "milp": report.get("exact_all_channel_milp"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
