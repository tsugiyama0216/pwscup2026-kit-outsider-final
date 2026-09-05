"""Audit cross-target AIA construction-order and shuffle artifacts.

The audit is deliberately limited to three public-input membership labels:

* targets 1 and 2: 163 rows selected by unique multiview QI-to-release
  assignment (the same public match recovered by the existing v1/v4 attack),
* target 24: 163 challenge rows with an exact seven-QI match in the attacker's
  hash-verified B table.

These are not organizer truth files.  ``record_id`` is table-local and is never
joined across cohorts.  It is used only to describe ordering inside the one
owned B table.  The script neither reads token.txt nor edits a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase import evaluate_shadow as shadow  # noqa: E402
from attack_phase import generate_submission as v1  # noqa: E402
from attack_phase_v8.actual_inputs import load_actual_inputs  # noqa: E402


QI = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
KNOWN_TARGETS = (1, 2, 24)
N_ROWS = 326
N_MEMBERS = 163
PERIODS = (2, 4, 8, 16, 32, 64)
FEATURE_SETS = {
    "challenge_order": (
        "challenge_position",
        "challenge_position_sq",
        *tuple(
            name
            for period in PERIODS
            for name in (f"challenge_sin_{period}", f"challenge_cos_{period}")
        ),
    ),
    "source_order": (
        "source_position",
        "source_position_sq",
        "source_minus_challenge",
        *tuple(
            name
            for period in PERIODS
            for name in (f"source_sin_{period}", f"source_cos_{period}")
        ),
    ),
    "combined": (
        "challenge_position",
        "challenge_position_sq",
        "source_position",
        "source_position_sq",
        "source_minus_challenge",
        "sex_order_position",
        "stratum_order_position",
        "stratum_order_parity",
        *tuple(
            name
            for period in PERIODS
            for prefix in ("challenge", "source")
            for name in (f"{prefix}_sin_{period}", f"{prefix}_cos_{period}")
        ),
    ),
}


@dataclass
class KnownTarget:
    target: int
    challenge: pd.DataFrame
    source: pd.DataFrame
    member: np.ndarray
    source_index: np.ndarray
    member_source_index: np.ndarray
    onset: np.ndarray
    rare: np.ndarray
    features: pd.DataFrame
    match_diagnostics: dict[str, Any]


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _rank01(values: Iterable[float]) -> np.ndarray:
    series = pd.Series(np.asarray(list(values)))
    return (series.rank(method="average").to_numpy(dtype=float) - 1.0) / max(
        len(series) - 1, 1
    )


def exact_qi_membership(
    source: pd.DataFrame, challenge: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map challenge rows to a source only when all seven public QIs match."""
    lookup: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(source.loc[:, QI].itertuples(index=False, name=None)):
        lookup.setdefault(tuple(row), []).append(index)
    member = np.zeros(len(challenge), dtype=bool)
    source_index = np.full(len(challenge), -1, dtype=int)
    ambiguous = 0
    for index, row in enumerate(
        challenge.loc[:, QI].itertuples(index=False, name=None)
    ):
        matches = lookup.get(tuple(row), [])
        if matches:
            member[index] = True
            source_index[index] = matches[0]
            ambiguous += int(len(matches) > 1)
    return member, source_index, {
        "method": "exact seven-QI tuple lookup",
        "matched_challenge_rows": int(member.sum()),
        "unique_matched_source_rows": int(np.unique(source_index[member]).size),
        "ambiguous_challenge_rows": int(ambiguous),
    }


def release_qi_membership(
    source: pd.DataFrame,
    challenge: pd.DataFrame,
    background: pd.DataFrame,
    fitted_views: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover the 163-row public QI assignment used for near-raw C1/C2."""
    pair_cost, _, view_summary = v1.aia_pair_cost(
        source, challenge, background, fitted_views
    )
    assigned = v1.cardinality_link_assignment(pair_cost, N_MEMBERS)
    member = assigned >= 0
    # Positional features must be label-independent.  Use the same unconditional
    # nearest-row rule for every challenge row; ``assigned`` is retained only
    # for constructing/evaluating the public proxy label and member outcomes.
    nearest = np.argmin(pair_cost, axis=1).astype(int)
    source_index = nearest
    member_cost = pair_cost[np.flatnonzero(member), assigned[member]]
    inactive_cost = pair_cost[~member].min(axis=1)
    diagnostics = {
        "method": "unique multiview QI assignment with public cardinality 163",
        "matched_challenge_rows": int(member.sum()),
        "unique_matched_source_rows": int(np.unique(assigned[member]).size),
        "selected_cost_median": float(np.median(member_cost)),
        "selected_cost_p95": float(np.quantile(member_cost, 0.95)),
        "unselected_nearest_cost_median": float(np.median(inactive_cost)),
        "unselected_nearest_cost_p05": float(np.quantile(inactive_cost, 0.05)),
        "view_summary": view_summary,
    }
    return member, source_index, assigned, diagnostics


def positional_feature_frame(
    challenge: pd.DataFrame, source_index: np.ndarray, source_rows: int
) -> pd.DataFrame:
    """Build ordering-only features; no distance, lab value, time, or label."""
    if len(challenge) != N_ROWS or len(source_index) != N_ROWS:
        raise ValueError("positional features require 326 aligned rows")
    challenge_position = np.arange(N_ROWS, dtype=float) / (N_ROWS - 1)
    source_position = np.asarray(source_index, dtype=float) / max(source_rows - 1, 1)
    output = pd.DataFrame(
        {
            "challenge_position": challenge_position,
            "challenge_position_sq": challenge_position**2,
            "source_position": source_position,
            "source_position_sq": source_position**2,
            "source_minus_challenge": source_position - challenge_position,
        }
    )
    for prefix, raw in (
        ("challenge", np.arange(N_ROWS, dtype=float)),
        ("source", np.asarray(source_index, dtype=float)),
    ):
        for period in PERIODS:
            phase = 2.0 * math.pi * raw / period
            output[f"{prefix}_sin_{period}"] = np.sin(phase)
            output[f"{prefix}_cos_{period}"] = np.cos(phase)

    work = challenge.copy()
    work["_row"] = np.arange(N_ROWS)
    work["_age_band"] = np.floor(
        (work["age"].to_numpy(dtype=float) - 40.0) / 5.0
    ).astype(int)
    output["sex_order_position"] = 0.0
    output["stratum_order_position"] = 0.0
    output["stratum_order_parity"] = 0.0
    for _, indices in work.groupby(["sex"], sort=True).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        output.loc[rows, "sex_order_position"] = np.arange(len(rows)) / max(
            len(rows) - 1, 1
        )
    for _, indices in work.groupby(["sex", "_age_band"], sort=True).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        ordinal = np.arange(len(rows))
        output.loc[rows, "stratum_order_position"] = ordinal / max(
            len(rows) - 1, 1
        )
        output.loc[rows, "stratum_order_parity"] = ordinal % 2
    return output


def build_known_targets(actual: Any) -> dict[int, KnownTarget]:
    fitted_views = [
        v1.fit_view(actual.background, spec) for spec in v1.AIA_VIEW_SPECS
    ]
    known: dict[int, KnownTarget] = {}
    detector = actual.schema["rare_detector"]
    for target in (1, 2):
        challenge = actual.challenges[str(target)].reset_index(drop=True)
        source = actual.releases[str(target)].reset_index(drop=True)
        member, source_index, assigned, diagnostics = release_qi_membership(
            source, challenge, actual.background, fitted_views
        )
        onset = np.full(N_ROWS, -1, dtype=int)
        rare = np.full(N_ROWS, -1, dtype=int)
        onset[member] = source.iloc[assigned[member]]["onset"].to_numpy(dtype=int)
        source_rare = shadow.public_rare_mask(source, detector).astype(int)
        rare[member] = source_rare[assigned[member]]
        known[target] = KnownTarget(
            target=target,
            challenge=challenge,
            source=source,
            member=member,
            source_index=source_index,
            member_source_index=assigned[member],
            onset=onset,
            rare=rare,
            features=positional_feature_frame(challenge, source_index, len(source)),
            match_diagnostics=diagnostics,
        )

    challenge = actual.self_challenge.reset_index(drop=True)
    source = actual.self_frame.reset_index(drop=True)
    member, exact_source, diagnostics = exact_qi_membership(source, challenge)
    # Use one unconditional nearest-row function for members and controls.
    # Substituting exact source indices only for known members would leak the
    # evaluation label into the positional feature construction.
    pair_cost, _, _ = v1.aia_pair_cost(
        source.drop(columns=["is_rare"]), challenge, actual.background, fitted_views
    )
    source_index = np.argmin(pair_cost, axis=1).astype(int)
    onset = np.full(N_ROWS, -1, dtype=int)
    rare = np.full(N_ROWS, -1, dtype=int)
    onset[member] = source.iloc[exact_source[member]]["onset"].to_numpy(dtype=int)
    rare[member] = source.iloc[exact_source[member]]["is_rare"].to_numpy(dtype=int)
    known[24] = KnownTarget(
        target=24,
        challenge=challenge,
        source=source,
        member=member,
        source_index=source_index,
        member_source_index=exact_source[member],
        onset=onset,
        rare=rare,
        features=positional_feature_frame(challenge, source_index, len(source)),
        match_diagnostics=diagnostics,
    )
    for target, item in known.items():
        if len(item.challenge) != N_ROWS or int(item.member.sum()) != N_MEMBERS:
            raise RuntimeError(f"target {target} did not recover 163/326 labels")
        if item.challenge["challenge_row_id"].duplicated().any():
            raise RuntimeError(f"target {target} challenge ids are not unique")
    return known


def _fit_predict(
    known: dict[int, KnownTarget],
    training: Iterable[int],
    heldout: int,
    columns: tuple[str, ...],
    labels: dict[int, np.ndarray],
) -> np.ndarray:
    train_targets = list(training)
    x_train = np.vstack(
        [known[target].features.loc[:, columns].to_numpy(dtype=float) for target in train_targets]
    )
    y_train = np.concatenate([labels[target].astype(int) for target in train_targets])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            solver="liblinear",
            max_iter=2_000,
            random_state=0,
        ),
    )
    model.fit(x_train, y_train)
    return model.predict_proba(
        known[heldout].features.loc[:, columns].to_numpy(dtype=float)
    )[:, 1]


def prediction_metrics(score: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    ids = np.arange(len(score), dtype=int)
    order = np.lexsort((ids, -np.asarray(score, dtype=float)))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:N_MEMBERS]] = True
    return {
        "auc": float(roc_auc_score(truth.astype(int), score)),
        "selected_members": int(np.sum(selected & truth)),
        "top163_accuracy": float(np.mean(selected == truth)),
        "top163_jaccard": float(
            np.sum(selected & truth) / np.sum(selected | truth)
        ),
    }


def permute_within_public_strata(
    labels: np.ndarray, challenge: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    output = labels.copy()
    age_band = np.floor(
        (challenge["age"].to_numpy(dtype=float) - 40.0) / 5.0
    ).astype(int)
    frame = pd.DataFrame({"sex": challenge["sex"], "age_band": age_band})
    for _, indices in frame.groupby(["sex", "age_band"], sort=True).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        output[rows] = labels[rng.permutation(rows)]
    return output


def loto_permutation_audit(
    known: dict[int, KnownTarget], *, permutations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    labels = {target: item.member.copy() for target, item in known.items()}
    observed_rows: list[dict[str, Any]] = []
    for heldout in KNOWN_TARGETS:
        training = tuple(target for target in KNOWN_TARGETS if target != heldout)
        for feature_set, columns in FEATURE_SETS.items():
            score = _fit_predict(known, training, heldout, columns, labels)
            observed_rows.append(
                {
                    "heldout_target": heldout,
                    "training_targets": "+".join(map(str, training)),
                    "feature_set": feature_set,
                    **prediction_metrics(score, labels[heldout]),
                }
            )
    observed = pd.DataFrame(observed_rows)

    rng = np.random.default_rng(seed)
    null_rows: list[dict[str, Any]] = []
    for replicate in range(permutations):
        permuted = {
            target: permute_within_public_strata(
                labels[target], known[target].challenge, rng
            )
            for target in KNOWN_TARGETS
        }
        for heldout in KNOWN_TARGETS:
            training = tuple(target for target in KNOWN_TARGETS if target != heldout)
            for feature_set, columns in FEATURE_SETS.items():
                score = _fit_predict(known, training, heldout, columns, permuted)
                metrics = prediction_metrics(score, permuted[heldout])
                null_rows.append(
                    {
                        "replicate": replicate,
                        "heldout_target": heldout,
                        "feature_set": feature_set,
                        "auc": metrics["auc"],
                        "selected_members": metrics["selected_members"],
                        "top163_accuracy": metrics["top163_accuracy"],
                    }
                )
    null = pd.DataFrame(null_rows)
    summaries: list[dict[str, Any]] = []
    for row in observed.itertuples(index=False):
        subset = null.loc[
            null["heldout_target"].eq(row.heldout_target)
            & null["feature_set"].eq(row.feature_set)
        ]
        same_holdout = null.loc[null["heldout_target"].eq(row.heldout_target)]
        max_auc = same_holdout.groupby("replicate")["auc"].max().to_numpy()
        summaries.append(
            {
                **row._asdict(),
                "null_auc_mean": float(subset["auc"].mean()),
                "null_auc_p95": float(subset["auc"].quantile(0.95)),
                "auc_p_upper": float(
                    (1 + np.sum(subset["auc"].to_numpy() >= row.auc))
                    / (len(subset) + 1)
                ),
                "auc_p_fwer_within_holdout": float(
                    (1 + np.sum(max_auc >= row.auc)) / (len(max_auc) + 1)
                ),
                "null_selected_members_p95": float(
                    subset["selected_members"].quantile(0.95)
                ),
                "selected_members_p_upper": float(
                    (
                        1
                        + np.sum(
                            subset["selected_members"].to_numpy()
                            >= row.selected_members
                        )
                    )
                    / (len(subset) + 1)
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    primary = summary.loc[
        summary["heldout_target"].eq(24)
        & summary["feature_set"].eq("combined")
    ].iloc[0]
    all_combined = summary.loc[summary["feature_set"].eq("combined")]
    gate = bool(
        primary["auc"] > primary["null_auc_p95"]
        and primary["auc_p_fwer_within_holdout"] < 0.05
        and primary["selected_members"] > primary["null_selected_members_p95"]
        and (all_combined["auc"] > 0.50).all()
    )
    return summary, null, {
        "primary": primary.to_dict(),
        "all_combined_auc": {
            str(int(row.heldout_target)): float(row.auc)
            for row in all_combined.itertuples(index=False)
        },
        "promotion_gate": {
            "heldout24_auc_above_stratified_null_p95": bool(
                primary["auc"] > primary["null_auc_p95"]
            ),
            "heldout24_fwer_p_below_0p05": bool(
                primary["auc_p_fwer_within_holdout"] < 0.05
            ),
            "heldout24_selected_members_above_null_p95": bool(
                primary["selected_members"]
                > primary["null_selected_members_p95"]
            ),
            "all_loto_directions_positive": bool((all_combined["auc"] > 0.50).all()),
            "pass": gate,
        },
    }


def descriptive_position_stats(known: dict[int, KnownTarget]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, item in known.items():
        y = item.member.astype(int)
        position = np.arange(N_ROWS, dtype=float)
        transitions = int(np.count_nonzero(y[1:] != y[:-1]))
        rows.append(
            {
                "target": target,
                "members": int(y.sum()),
                "challenge_id_min": int(item.challenge["challenge_row_id"].min()),
                "challenge_id_max": int(item.challenge["challenge_row_id"].max()),
                "challenge_ids_monotone": bool(
                    item.challenge["challenge_row_id"].is_monotonic_increasing
                ),
                "position_label_pearson": float(pearsonr(position, y).statistic),
                "first_163_members": int(y[:N_MEMBERS].sum()),
                "even_position_member_rate": float(y[::2].mean()),
                "odd_position_member_rate": float(y[1::2].mean()),
                "adjacent_same_rate": float(np.mean(y[1:] == y[:-1])),
                "runs": transitions + 1,
                "source_position_label_spearman": float(
                    spearmanr(item.source_index, y).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_target_position_stats(
    known: dict[int, KnownTarget], *, permutations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(KNOWN_TARGETS):
        for right in KNOWN_TARGETS[left_index + 1 :]:
            a = known[left].member.astype(int)
            b = known[right].member.astype(int)
            observed_agreement = float(np.mean(a == b))
            observed_phi = float(pearsonr(a, b).statistic)
            null_agreement = np.empty(permutations, dtype=float)
            null_abs_phi = np.empty(permutations, dtype=float)
            for replicate in range(permutations):
                shuffled = permute_within_public_strata(
                    b, known[right].challenge, rng
                )
                null_agreement[replicate] = np.mean(a == shuffled)
                null_abs_phi[replicate] = abs(float(pearsonr(a, shuffled).statistic))
            rows.append(
                {
                    "left_target": left,
                    "right_target": right,
                    "label_agreement": observed_agreement,
                    "label_phi": observed_phi,
                    "agreement_null_p95": float(np.quantile(null_agreement, 0.95)),
                    "agreement_p_upper": float(
                        (1 + np.sum(null_agreement >= observed_agreement))
                        / (permutations + 1)
                    ),
                    "abs_phi_p_two_sided": float(
                        (1 + np.sum(null_abs_phi >= abs(observed_phi)))
                        / (permutations + 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def member_order_correlations(known: dict[int, KnownTarget]) -> pd.DataFrame:
    sequences: dict[int, np.ndarray] = {}
    for target, item in known.items():
        member_positions = np.flatnonzero(item.member)
        order = np.argsort(item.member_source_index, kind="mergesort")
        sequences[target] = member_positions[order]
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(KNOWN_TARGETS):
        for right in KNOWN_TARGETS[left_index + 1 :]:
            rows.append(
                {
                    "left_target": left,
                    "right_target": right,
                    "source_order_to_challenge_order_spearman": float(
                        spearmanr(sequences[left], sequences[right]).statistic
                    ),
                    "identical_challenge_slots": int(
                        np.sum(sequences[left] == sequences[right])
                    ),
                }
            )
    return pd.DataFrame(rows)


def outcome_position_stats(
    known: dict[int, KnownTarget], *, permutations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for outcome in ("onset", "rare"):
        sets = {
            target: set(np.flatnonzero(getattr(item, outcome) == 1).tolist())
            for target, item in known.items()
        }
        for left_index, left in enumerate(KNOWN_TARGETS):
            for right in KNOWN_TARGETS[left_index + 1 :]:
                observed = len(sets[left] & sets[right])
                left_member = np.flatnonzero(known[left].member)
                right_member = np.flatnonzero(known[right].member)
                left_count = len(sets[left])
                right_count = len(sets[right])
                null = np.empty(permutations, dtype=int)
                for replicate in range(permutations):
                    left_slots = set(
                        rng.choice(left_member, size=left_count, replace=False).tolist()
                    )
                    right_slots = set(
                        rng.choice(right_member, size=right_count, replace=False).tolist()
                    )
                    null[replicate] = len(left_slots & right_slots)
                rows.append(
                    {
                        "outcome": outcome,
                        "left_target": left,
                        "right_target": right,
                        "left_positive_slots": left_count,
                        "right_positive_slots": right_count,
                        "observed_slot_intersection": observed,
                        "null_intersection_mean": float(null.mean()),
                        "null_intersection_p95": float(np.quantile(null, 0.95)),
                        "intersection_p_upper": float(
                            (1 + np.sum(null >= observed)) / (permutations + 1)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "local-output" / "attack-v9" / "positional",
    )
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026090501)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.permutations < 199:
        raise SystemExit("--permutations must be at least 199")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    actual = load_actual_inputs(args.dist, args.targets_dir)
    known = build_known_targets(actual)

    positions = descriptive_position_stats(known)
    pairwise = cross_target_position_stats(
        known, permutations=args.permutations, seed=args.seed + 1
    )
    member_order = member_order_correlations(known)
    outcomes = outcome_position_stats(
        known, permutations=args.permutations, seed=args.seed + 2
    )
    loto, null, model_report = loto_permutation_audit(
        known, permutations=args.permutations, seed=args.seed + 3
    )

    # A positional policy is promotable only if the fully predeclared combined
    # model clears the family-wise null on held-out target 24 and points the same
    # way in every LOTO fold.  No alternate feature set is selected post hoc.
    adoption = bool(model_report["promotion_gate"]["pass"])
    report = {
        "as_of": "2026-09-05 Asia/Tokyo",
        "decision": "ADOPT_POSITIONAL_SIGNATURE" if adoption else "REJECT_POSITIONAL_SIGNATURE",
        "adopt": adoption,
        "scope": (
            "ordering/shuffle artifacts on three independently public-QI-matched "
            "targets; not organizer truth and not an LB forecast"
        ),
        "matching": {
            str(target): item.match_diagnostics for target, item in known.items()
        },
        "model": model_report,
        "data_quality": {
            "rows_per_challenge": {
                str(target): len(item.challenge) for target, item in known.items()
            },
            "members_per_challenge": {
                str(target): int(item.member.sum()) for target, item in known.items()
            },
            "challenge_ids_unique": {
                str(target): bool(item.challenge["challenge_row_id"].is_unique)
                for target, item in known.items()
            },
            "challenge_ids_exact_0_to_325": {
                str(target): bool(
                    np.array_equal(
                        item.challenge["challenge_row_id"].to_numpy(dtype=int),
                        np.arange(N_ROWS),
                    )
                )
                for target, item in known.items()
            },
            "record_id_handling": (
                "table-local; never joined across cohorts; own-B values used only "
                "to establish within-table source order after exact QI proof"
            ),
        },
        "method": {
            "feature_sets": {
                key: list(value) for key, value in FEATURE_SETS.items()
            },
            "classifier": "StandardScaler + fixed L2 logistic(C=0.25)",
            "validation": "three-target LOTO; target24 is the primary holdout",
            "null": (
                "labels independently permuted within each target's sex x 5-year "
                "age band, preserving every stratum member count exactly"
            ),
            "permutations": args.permutations,
            "familywise_control": (
                "heldout-specific max AUC across the three predeclared feature sets"
            ),
        },
        "limitations": [
            "Targets 1 and 2 use public QI-assignment labels, not organizer truth files.",
            "Only three construction instances are truth-known enough for this audit.",
            "C1/C2 expose source row order but not record_id; target24 record_id is table-local.",
            "Nearest-source multiplicity was excluded because it is a QI linkage-density signal, not an ordering feature.",
            "Nearest-source row position is computed by the identical unconditional rule for members and controls; member assignments never overwrite a feature.",
            "Source position may still proxy public-QI sorting; it is tested as a construction-order hypothesis, not treated as an independent membership signal.",
            "A null result cannot exclude a target-specific positional artifact absent from all three known targets.",
        ],
        "safety": {
            "token_read": False,
            "secret_or_salt_search": False,
            "organizer_hidden_files_read": False,
            "production_changed": False,
            "zip_created": False,
            "submitted": False,
        },
        "input_provenance": actual.provenance,
    }

    positions.to_csv(args.out_dir / "position_stats.csv", index=False)
    pairwise.to_csv(args.out_dir / "pairwise_position_stats.csv", index=False)
    member_order.to_csv(args.out_dir / "member_order_stats.csv", index=False)
    outcomes.to_csv(args.out_dir / "outcome_position_stats.csv", index=False)
    loto.to_csv(args.out_dir / "loto_results.csv", index=False)
    null.to_csv(args.out_dir / "permutation_null.csv", index=False)
    cleaned = _clean_json(report)
    (args.out_dir / "audit.json").write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(
        loto.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    print(
        json.dumps(
            {
                "decision": cleaned["decision"],
                "matching_counts": {
                    str(target): int(item.member.sum()) for target, item in known.items()
                },
                "primary": cleaned["model"]["primary"],
                "combined_auc": cleaned["model"]["all_combined_auc"],
                "loto_sha256": digest,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
