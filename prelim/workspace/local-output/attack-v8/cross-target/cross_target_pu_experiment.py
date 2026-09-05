"""Public-only cross-target AIA positive-unlabelled experiment.

The official challenge for target k is a 50:50 mixture of target-private
members and controls.  Controls are sampled outside every cohort and matched
to the members on public demographic blocks and hidden onset.  Within one
prefecture, challenges for other targets therefore provide a contaminated but
identity-disjoint control reference.  A_bg is a second, clean but finite
population reference.

This script tests whether a cross-fitted density-ratio classifier can
distinguish one target's challenge mixture from both references.  It never
reads organiser truth and never writes a submission.  Target/prefecture groups
used for method discovery and validation are fixed before any metric is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack_phase.evaluate_shadow import load_rare_detector, public_rare_mask  # noqa: E402
from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    stratified_half_selection,
)


TARGET_DIR = REPO_ROOT / "team-shared" / "prelim" / "attack-targets"
BACKGROUND_PATH = REPO_ROOT / "participant_data" / "A_bg.csv"
TEAM_DIR = REPO_ROOT / "submissions" / "prelim" / "team24"
FEEDBACK_DIR = REPO_ROOT / "attack_phase_v6" / "official_feedback"
OUT_DIR = REPO_ROOT / "local-output" / "attack-v8" / "cross-target"

SELF_TARGET = "24"
ABSTAIN_TIME = 11.0
LAB_COLUMNS = ("SBP", "TG", "HDL", "ALT")
TARGET_FEATURE_COLUMNS = ("SBP", "log_TG", "HDL", "log_ALT")
DESIRED_LATENT_PROPORTIONS = {
    # Per arm: 138 common/non-onset, 12 common/onset,
    #          9 rare/non-onset, 4 rare/onset.
    (False, 0): 138.0 / 163.0,
    (False, 1): 12.0 / 163.0,
    (True, 0): 9.0 / 163.0,
    (True, 1): 4.0 / 163.0,
}

# Prefecture groups are keyed by their minimum target id after loading.  This
# fixed split is declared before metrics are calculated.  Singleton groups are
# diagnostics only because they have no same-prefecture challenge reference.
DISCOVERY_GROUP_ROOTS = frozenset((1, 5, 12, 22))
VALIDATION_GROUP_ROOTS = frozenset((3, 9, 13))

SOURCE_PATHS = {
    "v4": TEAM_DIR / "attack-candidate-v4" / "F_aia.csv",
    "independent": TEAM_DIR / "attack-candidate-daybreak-independent" / "F_aia.csv",
    "routed": TEAM_DIR / "attack-candidate-v6-routed-guided-probe" / "F_aia.csv",
    "v7": TEAM_DIR / "attack-candidate-v7-feedback-stack" / "F_aia.csv",
}
FEEDBACK_PATHS = {
    "v4": FEEDBACK_DIR / "v4.json",
    "independent": FEEDBACK_DIR / "probe_independent.json",
    "routed": FEEDBACK_DIR / "routed_guided_probe.json",
}


def csv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = pd.DataFrame(
        {
            "SBP": pd.to_numeric(frame["SBP"], errors="raise"),
            "log_TG": np.log1p(
                np.clip(pd.to_numeric(frame["TG"], errors="raise"), 0.0, None)
            ),
            "HDL": pd.to_numeric(frame["HDL"], errors="raise"),
            "log_ALT": np.log1p(
                np.clip(pd.to_numeric(frame["ALT"], errors="raise"), 0.0, None)
            ),
        }
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite public AIA feature")
    return values


def age_band(frame: pd.DataFrame) -> np.ndarray:
    return np.floor((frame["age"].to_numpy(dtype=float) - 40.0) / 5.0).astype(int)


def stable_rank(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.asarray(keys), np.asarray(values, dtype=float)))
    result = np.empty(len(order), dtype=float)
    result[order] = np.arange(1, len(order) + 1, dtype=float)
    return result / (len(order) + 1.0)


def rake_reference_weights(
    frame: pd.DataFrame,
    target: pd.DataFrame,
    *,
    rare: np.ndarray | None,
    iterations: int = 25,
) -> np.ndarray:
    """Rake a public reference to target sex/age-band and known latent totals."""
    work = pd.DataFrame(
        {
            "sex": frame["sex"].to_numpy(),
            "band": age_band(frame),
        }
    )
    target_blocks = pd.DataFrame(
        {"sex": target["sex"].to_numpy(), "band": age_band(target)}
    ).value_counts(normalize=True)
    weights = np.ones(len(frame), dtype=float)
    latent = None
    if rare is not None:
        latent = pd.DataFrame(
            {
                "rare": np.asarray(rare, dtype=bool),
                "onset": frame["onset"].to_numpy(dtype=int),
            }
        )

    for _ in range(iterations):
        for block, desired in target_blocks.items():
            mask = (work["sex"].to_numpy() == block[0]) & (
                work["band"].to_numpy() == block[1]
            )
            observed = float(weights[mask].sum())
            if observed > 0.0:
                weights[mask] *= float(desired) / observed
        if latent is not None:
            rare_values = latent["rare"].to_numpy()
            onset_values = latent["onset"].to_numpy()
            for cell, desired in DESIRED_LATENT_PROPORTIONS.items():
                mask = (rare_values == cell[0]) & (onset_values == cell[1])
                observed = float(weights[mask].sum())
                if observed > 0.0:
                    weights[mask] *= float(desired) / observed
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError("reference raking collapsed")
        weights /= total
    return weights


def model_factories() -> dict[str, Callable[[], BaseEstimator]]:
    def linear(c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=0,
                    ),
                ),
            ]
        )

    def quadratic(c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=0,
                    ),
                ),
            ]
        )

    def rbf(c_value: float, gamma: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", SVC(C=c_value, gamma=gamma, kernel="rbf")),
            ]
        )

    return {
        "linear_c0p03": lambda: linear(0.03),
        "linear_c0p3": lambda: linear(0.3),
        "quadratic_c0p01": lambda: quadratic(0.01),
        "quadratic_c0p1": lambda: quadratic(0.1),
        "rbf_c0p3_g0p1": lambda: rbf(0.3, 0.1),
        "rbf_c1_g0p1": lambda: rbf(1.0, 0.1),
    }


def crossfit_density_ratio(
    target: pd.DataFrame,
    siblings: pd.DataFrame,
    background: pd.DataFrame,
    sibling_weights: np.ndarray,
    background_weights: np.ndarray,
    factory: Callable[[], BaseEstimator],
    *,
    bg_ratio: float,
    seed: int,
    permute_labels: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return OOF target scores and held-out mixture/reference AUCs."""
    parts = [target]
    labels = [np.ones(len(target), dtype=int)]
    source = [np.full(len(target), "target", dtype=object)]
    raw_weights = [np.full(len(target), 1.0 / len(target), dtype=float)]

    if len(siblings):
        parts.append(siblings)
        labels.append(np.zeros(len(siblings), dtype=int))
        source.append(np.full(len(siblings), "sibling", dtype=object))
        sibling_total = 1.0 if bg_ratio == 0.0 else 1.0 / (1.0 + bg_ratio)
        raw_weights.append(np.asarray(sibling_weights, dtype=float) * sibling_total)
    if bg_ratio > 0.0 or not len(siblings):
        parts.append(background)
        labels.append(np.zeros(len(background), dtype=int))
        source.append(np.full(len(background), "background", dtype=object))
        bg_total = 1.0 if not len(siblings) else bg_ratio / (1.0 + bg_ratio)
        raw_weights.append(np.asarray(background_weights, dtype=float) * bg_total)

    combined = pd.concat(parts, ignore_index=True)
    y = np.concatenate(labels)
    source_array = np.concatenate(source)
    sample_weight = np.concatenate(raw_weights)
    if permute_labels:
        rng = np.random.default_rng(seed + 999_983)
        y = rng.permutation(y)
    x = feature_matrix(combined)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(combined), np.nan, dtype=float)
    for train, test in splitter.split(x, y):
        model = clone(factory())
        model.fit(x[train], y[train], clf__sample_weight=sample_weight[train])
        oof[test] = model.decision_function(x[test])
    if not np.isfinite(oof).all():
        raise RuntimeError("cross-fitted scores are incomplete")

    target_mask = source_array == "target"
    metrics: dict[str, float] = {}
    for name, negative_source in (
        ("sibling", "sibling"),
        ("background", "background"),
    ):
        negative_mask = source_array == negative_source
        if not negative_mask.any():
            metrics[f"auc_{name}"] = float("nan")
            continue
        keep = target_mask | negative_mask
        metrics[f"auc_{name}"] = float(
            roc_auc_score(y[keep], oof[keep], sample_weight=sample_weight[keep])
        )
    negative_mask = ~target_mask
    metrics["auc_combined"] = float(
        roc_auc_score(y, oof, sample_weight=sample_weight)
    )
    return oof[target_mask], metrics


def load_public_data() -> tuple[
    dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame, np.ndarray
]:
    challenges: dict[str, pd.DataFrame] = {}
    releases: dict[str, pd.DataFrame] = {}
    for path in TARGET_DIR.glob("aia_challenge_*.csv"):
        target_id = str(int(path.stem.rsplit("_", 1)[1]))
        challenges[target_id] = pd.read_csv(path).sort_values(
            "challenge_row_id", kind="mergesort"
        ).reset_index(drop=True)
        releases[target_id] = pd.read_csv(TARGET_DIR / f"C_{target_id}.csv")
    background = pd.read_csv(BACKGROUND_PATH).reset_index(drop=True)
    rare = public_rare_mask(
        background, load_rare_detector(REPO_ROOT / "participant_data")
    )
    return challenges, releases, background, rare


def build_groups(challenges: dict[str, pd.DataFrame]) -> tuple[dict[str, list[str]], dict[str, str]]:
    by_prefecture: dict[str, list[str]] = defaultdict(list)
    for target_id, challenge in challenges.items():
        prefectures = challenge["prefecture"].astype(str).unique()
        if len(prefectures) != 1:
            raise ValueError(f"target {target_id} challenge has multiple prefectures")
        by_prefecture[prefectures[0]].append(target_id)
    groups = {
        prefecture: sorted(targets, key=int)
        for prefecture, targets in by_prefecture.items()
    }
    split: dict[str, str] = {}
    for targets in groups.values():
        root = min(map(int, targets))
        label = (
            "discovery"
            if root in DISCOVERY_GROUP_ROOTS
            else "validation"
            if root in VALIDATION_GROUP_ROOTS
            else "singleton"
        )
        for target_id in targets:
            split[target_id] = label
    return groups, split


def collision_audit(challenges: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows = []
    for target_id, challenge in challenges.items():
        block = challenge.copy()
        block["target_id"] = target_id
        rows.append(block)
    combined = pd.concat(rows, ignore_index=True)
    definitions = {
        "full_qi": ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"],
        "labs4": ["SBP", "TG", "HDL", "ALT"],
        "stable3_block": ["age", "SBP", "TG", "sex", "prefecture"],
        "age_sex_prefecture": ["age", "sex", "prefecture"],
        "sex_prefecture": ["sex", "prefecture"],
    }
    result: dict[str, Any] = {}
    for name, columns in definitions.items():
        counts = combined.groupby(columns, dropna=False)["target_id"].nunique()
        cross = counts[counts > 1]
        keys = combined.set_index(columns).index
        cross_keys = set(cross.index.tolist())
        participating = int(sum(key in cross_keys for key in keys))
        result[name] = {
            "cross_target_groups": int(len(cross)),
            "participating_rows": participating,
            "maximum_targets": int(cross.max()) if len(cross) else 0,
        }
    return result


def source_near_audit(
    challenges: dict[str, pd.DataFrame],
    releases: dict[str, pd.DataFrame],
    background: pd.DataFrame,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Audit exact and standardized near matches to every public source family."""
    practice = pd.read_csv(REPO_ROOT / "participant_data" / "B_practice.csv")
    reference_x = feature_matrix(background)
    center = np.median(reference_x, axis=0)
    q25, q75 = np.quantile(reference_x, (0.25, 0.75), axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.std(reference_x, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    full_qi = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]

    def one_source(query: pd.DataFrame, source: pd.DataFrame) -> dict[str, float | int]:
        if not len(source):
            return {
                "rows": 0,
                "exact_full_qi": 0,
                "nearest_p05": float("nan"),
                "nearest_median": float("nan"),
            }
        query_keys = pd.MultiIndex.from_frame(query[full_qi])
        source_keys = pd.MultiIndex.from_frame(source[full_qi])
        source_x = (feature_matrix(source) - center) / scale / np.sqrt(len(scale))
        query_x = (feature_matrix(query) - center) / scale / np.sqrt(len(scale))
        distance = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(source_x).kneighbors(
            query_x, return_distance=True
        )[0][:, 0]
        return {
            "rows": int(len(source)),
            "exact_full_qi": int(query_keys.isin(source_keys).sum()),
            "nearest_p05": float(np.quantile(distance, 0.05)),
            "nearest_median": float(np.median(distance)),
        }

    rows: list[dict[str, Any]] = []
    for target_id, query in challenges.items():
        prefecture = str(query["prefecture"].iloc[0])
        sibling_ids = [value for value in groups[prefecture] if value != target_id]
        sources = {
            "own_C": releases[target_id],
            "other_same_pref_C": (
                pd.concat([releases[value] for value in sibling_ids], ignore_index=True)
                if sibling_ids
                else query.iloc[0:0].copy()
            ),
            "other_same_pref_challenge": (
                pd.concat([challenges[value] for value in sibling_ids], ignore_index=True)
                if sibling_ids
                else query.iloc[0:0].copy()
            ),
            "A_bg_same_pref": background.loc[
                background["prefecture"].astype(str).eq(prefecture)
            ],
            "B_practice_same_pref": practice.loc[
                practice["prefecture"].astype(str).eq(prefecture)
            ],
        }
        for label, source in sources.items():
            rows.append(
                {
                    "target_id": target_id,
                    "source": label,
                    **one_source(query, source),
                }
            )
    return pd.DataFrame(rows)


def source_enrichment(
    score_table: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feedback = {
        label: json.loads(path.read_text(encoding="utf-8"))["aia_per_target"]
        for label, path in FEEDBACK_PATHS.items()
    }
    for source_label, path in SOURCE_PATHS.items():
        if source_label == "v7":
            continue
        source = pd.read_csv(path)
        for target_id, challenge in challenges.items():
            if target_id == SELF_TARGET:
                continue
            scores = score_table.loc[
                score_table["target_id"].eq(target_id)
            ].set_index("challenge_row_id")["score"]
            ids = challenge["challenge_row_id"].astype(int).to_numpy()
            values = scores.loc[ids].to_numpy(dtype=float)
            block = source.loc[source["target_k"].astype(int).eq(int(target_id))]
            times = block.set_index(block["challenge_row_id"].astype(int)).loc[
                ids, "time_hat"
            ].to_numpy(dtype=float)
            active = ~np.isclose(times, ABSTAIN_TIME)
            delta = float(values[active].mean() - values[~active].mean())
            rows.append(
                {
                    "source": source_label,
                    "target_id": target_id,
                    "score": float(feedback[source_label][target_id]),
                    "pu_active_minus_inactive": delta,
                    "active_rows": int(active.sum()),
                }
            )
    return pd.DataFrame(rows)


def hypothetical_swaps(
    score_table: pd.DataFrame,
    challenges: dict[str, pd.DataFrame],
    split: dict[str, str],
    *,
    swaps_per_target: int = 4,
) -> pd.DataFrame:
    """Measure a bounded rerank against v7; do not emit an F_aia candidate."""
    v7 = pd.read_csv(SOURCE_PATHS["v7"])
    rows: list[dict[str, Any]] = []
    for target_id, challenge in challenges.items():
        if target_id == SELF_TARGET:
            continue
        ids = challenge["challenge_row_id"].astype(int).to_numpy()
        score = score_table.loc[score_table["target_id"].eq(target_id)].set_index(
            "challenge_row_id"
        )["score"].loc[ids].to_numpy(dtype=float)
        block = v7.loc[v7["target_k"].astype(int).eq(int(target_id))].copy()
        times = block.set_index(block["challenge_row_id"].astype(int)).loc[
            ids, "time_hat"
        ].to_numpy(dtype=float)
        active = ~np.isclose(times, ABSTAIN_TIME)
        # A bounded half-mask swap is only defined where v7 has the official
        # 163-row cardinality.  Expanded zero-target masks are left untouched.
        if int(active.sum()) != 163 or split[target_id] == "singleton":
            rows.append(
                {
                    "target_id": target_id,
                    "split": split[target_id],
                    "v7_active": int(active.sum()),
                    "swaps": 0,
                    "changed_rows": 0,
                    "pu_mean_lift": 0.0,
                    "eligible": False,
                }
            )
            continue
        work = pd.DataFrame(
            {
                "row": np.arange(len(challenge)),
                "sex": challenge["sex"].to_numpy(),
                "band": age_band(challenge),
            }
        )
        removed: list[int] = []
        added: list[int] = []
        for indices in work.groupby(["sex", "band"], sort=True).indices.values():
            local = np.asarray(indices, dtype=int)
            old = local[active[local]]
            new = local[~active[local]]
            if not len(old) or not len(new):
                continue
            remove = old[np.argmin(score[old])]
            add = new[np.argmax(score[new])]
            if score[add] > score[remove]:
                removed.append(int(remove))
                added.append(int(add))
        if len(removed) > swaps_per_target:
            gains = score[np.asarray(added)] - score[np.asarray(removed)]
            keep = np.argsort(-gains, kind="mergesort")[:swaps_per_target]
            removed = [removed[index] for index in keep]
            added = [added[index] for index in keep]
        lift = (
            float(np.mean(score[np.asarray(added)] - score[np.asarray(removed)]))
            if added
            else 0.0
        )
        rows.append(
            {
                "target_id": target_id,
                "split": split[target_id],
                "v7_active": int(active.sum()),
                "swaps": len(added),
                "changed_rows": 2 * len(added),
                "pu_mean_lift": lift,
                "eligible": True,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    challenges, releases, background, background_rare = load_public_data()
    groups, split = build_groups(challenges)
    factories = model_factories()
    bg_ratios = {
        "sibling_only": 0.0,
        "sibling_bg_0p25": 0.25,
        "sibling_bg_1": 1.0,
        "bg_only": float("inf"),
    }
    metric_rows: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for target_id in sorted(challenges, key=int):
        target = challenges[target_id]
        prefecture = str(target["prefecture"].iloc[0])
        sibling_ids = [value for value in groups[prefecture] if value != target_id]
        siblings = (
            pd.concat([challenges[value] for value in sibling_ids], ignore_index=True)
            if sibling_ids
            else target.iloc[0:0].copy()
        )
        bg = background.loc[
            background["prefecture"].astype(str).eq(prefecture)
        ].reset_index(drop=True)
        bg_rare = background_rare[
            background["prefecture"].astype(str).to_numpy() == prefecture
        ]
        bg_weights = rake_reference_weights(bg, target, rare=bg_rare)
        sibling_weights = (
            rake_reference_weights(siblings, target, rare=None)
            if len(siblings)
            else np.asarray([], dtype=float)
        )

        for model_name, factory in factories.items():
            for mode, raw_ratio in bg_ratios.items():
                if mode != "bg_only" and not len(siblings):
                    continue
                effective_siblings = siblings if mode != "bg_only" else siblings.iloc[0:0]
                effective_sibling_weights = (
                    sibling_weights if mode != "bg_only" else np.asarray([], dtype=float)
                )
                bg_ratio = 1.0 if mode == "bg_only" else raw_ratio
                scores, metrics = crossfit_density_ratio(
                    target,
                    effective_siblings,
                    bg,
                    effective_sibling_weights,
                    bg_weights,
                    factory,
                    bg_ratio=bg_ratio,
                    seed=810_000 + int(target_id),
                )
                score_cache[(target_id, model_name, mode)] = stable_rank(
                    scores, target["challenge_row_id"].to_numpy(dtype=int)
                )
                metric_rows.append(
                    {
                        "target_id": target_id,
                        "split": split[target_id],
                        "group_root": min(map(int, groups[prefecture])),
                        "siblings": len(sibling_ids),
                        "background_rows": len(bg),
                        "model": model_name,
                        "mode": mode,
                        **metrics,
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "public_crossfit_metrics.csv", index=False)
    grouped = metrics.loc[metrics["siblings"].gt(0)].copy()
    discovery = (
        grouped.loc[grouped["split"].eq("discovery")]
        .groupby(["model", "mode"], as_index=False)[
            ["auc_sibling", "auc_background", "auc_combined"]
        ]
        .mean()
    )
    discovery["gate_metric"] = np.minimum(
        discovery["auc_sibling"], discovery["auc_background"]
    )
    discovery = discovery.sort_values(
        ["gate_metric", "auc_sibling"], ascending=False, kind="mergesort"
    )
    discovery.to_csv(OUT_DIR / "discovery_model_ranking.csv", index=False)
    winner = discovery.iloc[0]
    winner_model = str(winner["model"])
    winner_mode = str(winner["mode"])

    validation = grouped.loc[
        grouped["split"].eq("validation")
        & grouped["model"].eq(winner_model)
        & grouped["mode"].eq(winner_mode)
    ].copy()
    validation.to_csv(OUT_DIR / "heldout_prefecture_validation.csv", index=False)

    # Label-permutation null, run only after the discovery winner is frozen.
    # The permutation changes the weak training labels, never target membership
    # truth (which is not distributed).
    null_rows: list[dict[str, Any]] = []
    selected_bg_ratio = bg_ratios[winner_mode]
    for target_id in validation["target_id"].astype(str).tolist():
        target = challenges[target_id]
        prefecture = str(target["prefecture"].iloc[0])
        sibling_ids = [value for value in groups[prefecture] if value != target_id]
        siblings = pd.concat(
            [challenges[value] for value in sibling_ids], ignore_index=True
        )
        bg = background.loc[
            background["prefecture"].astype(str).eq(prefecture)
        ].reset_index(drop=True)
        bg_rare = background_rare[
            background["prefecture"].astype(str).to_numpy() == prefecture
        ]
        bg_weights = rake_reference_weights(bg, target, rare=bg_rare)
        sibling_weights = rake_reference_weights(siblings, target, rare=None)
        if winner_mode == "bg_only":
            siblings = siblings.iloc[0:0]
            sibling_weights = np.asarray([], dtype=float)
            selected_bg_ratio = 1.0
        for repeat in range(12):
            _, null_metrics = crossfit_density_ratio(
                target,
                siblings,
                bg,
                sibling_weights,
                bg_weights,
                factories[winner_model],
                bg_ratio=selected_bg_ratio,
                seed=910_000 + 100 * int(target_id) + repeat,
                permute_labels=True,
            )
            null_rows.append(
                {
                    "target_id": target_id,
                    "repeat": repeat,
                    **null_metrics,
                }
            )
    null = pd.DataFrame(null_rows)
    null.to_csv(OUT_DIR / "heldout_label_permutation_null.csv", index=False)

    score_parts = []
    for target_id, challenge in challenges.items():
        key = (target_id, winner_model, winner_mode)
        if key not in score_cache:
            # The selected cross-target method is intentionally undefined for
            # singleton prefectures.  A bg-only linear score is included only
            # for diagnostics and never enters the candidate gate.
            key = (target_id, "linear_c0p3", "bg_only")
        score_parts.append(
            pd.DataFrame(
                {
                    "target_id": target_id,
                    "challenge_row_id": challenge["challenge_row_id"].astype(int),
                    "score": score_cache[key],
                    "score_source": (
                        f"{winner_model}:{winner_mode}"
                        if split[target_id] != "singleton"
                        else "linear_c0p3:bg_only_diagnostic"
                    ),
                }
            )
        )
    score_table = pd.concat(score_parts, ignore_index=True)
    score_table.to_csv(OUT_DIR / "production_pu_scores.csv", index=False)

    enrichment = source_enrichment(score_table, challenges)
    enrichment["split"] = enrichment["target_id"].map(split)
    enrichment.to_csv(OUT_DIR / "official_feedback_enrichment.csv", index=False)
    validation_enrichment = enrichment.loc[
        enrichment["split"].eq("validation") & enrichment["source"].eq("routed")
    ]
    correlation = spearmanr(
        validation_enrichment["score"],
        validation_enrichment["pu_active_minus_inactive"],
    )

    swaps = hypothetical_swaps(score_table, challenges, split)
    swaps.to_csv(OUT_DIR / "hypothetical_v7_swaps.csv", index=False)
    near_audit = source_near_audit(challenges, releases, background, groups)
    near_audit.to_csv(OUT_DIR / "public_source_near_matches.csv", index=False)

    validation_summary = {
        column: float(validation[column].mean())
        for column in ("auc_sibling", "auc_background", "auc_combined")
    }
    null_summary = {
        column: {
            "mean": float(null[column].mean()),
            "p95": float(null[column].quantile(0.95)),
        }
        for column in ("auc_sibling", "auc_background", "auc_combined")
    }
    # This is deliberately stringent: both held-out weak-label views must beat
    # 0.53, every validation target must beat chance against siblings, and the
    # aggregate official feedback association must have the expected sign.
    public_gate = bool(
        len(validation) >= 4
        and validation_summary["auc_sibling"] >= 0.53
        and validation_summary["auc_background"] >= 0.53
        and validation_summary["auc_sibling"] > null_summary["auc_sibling"]["p95"]
        and bool((validation["auc_sibling"] > 0.5).all())
        and np.isfinite(correlation.statistic)
        and float(correlation.statistic) > 0.0
    )

    report = {
        "purpose": "public-only weak-label evidence; not organiser-truth scoring",
        "identity_model": {
            "members": "target-private P_i; no cross-target person recurrence",
            "controls": "outside all cohorts/A_bg; unique per target",
            "legacy_shadow_warning": (
                "attack_phase/evaluate_shadow.py reuses B_practice as every target's "
                "AIA source and is invalid for cross-target identity features"
            ),
        },
        "collision_audit": collision_audit(challenges),
        "prefecture_groups": [
            {
                "targets": values,
                "split": split[values[0]],
            }
            for values in sorted(groups.values(), key=lambda item: min(map(int, item)))
        ],
        "method_selection": {
            "winner_model": winner_model,
            "winner_mode": winner_mode,
            "discovery_gate_metric": float(winner["gate_metric"]),
            "discovery_auc_sibling": float(winner["auc_sibling"]),
            "discovery_auc_background": float(winner["auc_background"]),
        },
        "heldout_prefecture_validation": validation_summary,
        "heldout_label_permutation_null": null_summary,
        "heldout_official_feedback_gate": {
            "source": "routed only",
            "observations": int(len(validation_enrichment)),
            "spearman": (
                float(correlation.statistic)
                if np.isfinite(correlation.statistic)
                else None
            ),
            "pvalue": float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None,
        },
        "public_gate_pass": public_gate,
        "candidate_emitted": False,
        "candidate_reason": (
            "No F_aia is emitted by this experiment. A truth-known, no-reuse "
            "validation gate must also pass before production use."
        ),
        "hypothetical_v7_changes": {
            "eligible_targets": int(swaps["eligible"].sum()),
            "total_changed_rows_if_four_swaps": int(swaps["changed_rows"].sum()),
        },
        "input_hashes": {
            "A_bg.csv": csv_sha256(BACKGROUND_PATH),
            **{path.name + ":" + label: csv_sha256(path) for label, path in SOURCE_PATHS.items()},
        },
        "network_access": False,
        "submission_performed": False,
    }
    (OUT_DIR / "public_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["method_selection"], indent=2))
    print(json.dumps(report["heldout_prefecture_validation"], indent=2))
    print(f"public_gate_pass={public_gate}")


if __name__ == "__main__":
    main()
