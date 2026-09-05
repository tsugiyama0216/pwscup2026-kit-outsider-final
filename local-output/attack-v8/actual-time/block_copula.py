"""Single-home-prefecture copula event links for preliminary AIA challenges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


QI = ("age", "SBP", "TG", "HDL", "ALT")


def _rank_matrix(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    subset = frame.iloc[rows]
    return np.column_stack(
        [
            subset[column].rank(method="average", pct=True).to_numpy(dtype=float)
            for column in QI
        ]
    )


def block_conditional_copula_event_links(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    condition_on_sex: bool = False,
    margin_quantile: float = 0.90,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Link in CDF coordinates fitted inside the challenge home prefecture.

    Preliminary AIA challenges contain one prefecture, while the corresponding
    C release contains a home-prefecture block plus nationwide rows.  Global
    marginal ranks therefore compare different mixtures.  This decoder fits
    release and query CDFs within prefecture (or prefecture/sex), then retains
    exact sex/prefecture mutual-nearest-neighbour and event gates.
    """
    required_release = (*QI, "sex", "prefecture", "time", "onset", "death")
    required_challenge = (*QI, "sex", "prefecture")
    for label, frame, required in (
        ("release", released, required_release),
        ("challenge", challenge, required_challenge),
    ):
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
        if frame.loc[:, list(required)].isna().any().any():
            raise ValueError(f"{label} contains missing values")
    if not len(released) or not len(challenge):
        raise ValueError("release and challenge must be nonempty")
    if not 0.0 <= margin_quantile <= 1.0:
        raise ValueError("margin_quantile must be in [0,1]")

    r_pref = released["prefecture"].astype(str).to_numpy()
    c_pref = challenge["prefecture"].astype(str).to_numpy()
    r_sex = released["sex"].to_numpy()
    c_sex = challenge["sex"].to_numpy()
    nearest = np.full(len(challenge), -1, dtype=int)
    margin = np.zeros(len(challenge), dtype=float)
    mutual = np.zeros(len(challenge), dtype=bool)
    fallback_rows = 0
    conditional_groups = 0

    block_table = pd.DataFrame({"prefecture": c_pref, "sex": c_sex})
    for raw_key, query_indices in block_table.groupby(
        ["prefecture", "sex"], sort=False
    ).indices.items():
        prefecture, sex = raw_key
        query_rows = np.asarray(query_indices, dtype=int)
        donor_rows = np.flatnonzero((r_pref == str(prefecture)) & (r_sex == sex))
        if not len(donor_rows):
            donor_rows = np.flatnonzero(r_sex == sex)
            fallback_rows += len(query_rows)
        if not len(donor_rows):
            donor_rows = np.arange(len(released), dtype=int)
            fallback_rows += len(query_rows)

        if condition_on_sex:
            release_reference = donor_rows
            challenge_reference = query_rows
        else:
            release_reference = np.flatnonzero(r_pref == str(prefecture))
            challenge_reference = np.flatnonzero(c_pref == str(prefecture))
            if not len(release_reference):
                release_reference = np.arange(len(released), dtype=int)
            if not len(challenge_reference):
                challenge_reference = np.arange(len(challenge), dtype=int)

        release_rank_reference = _rank_matrix(released, release_reference)
        challenge_rank_reference = _rank_matrix(challenge, challenge_reference)
        release_position = {int(row): pos for pos, row in enumerate(release_reference)}
        challenge_position = {
            int(row): pos for pos, row in enumerate(challenge_reference)
        }
        release_x = release_rank_reference[
            np.fromiter(
                (release_position[int(row)] for row in donor_rows),
                dtype=int,
                count=len(donor_rows),
            )
        ]
        challenge_x = challenge_rank_reference[
            np.fromiter(
                (challenge_position[int(row)] for row in query_rows),
                dtype=int,
                count=len(query_rows),
            )
        ]
        pair = cdist(challenge_x, release_x, metric="euclidean")
        order = np.argsort(pair, axis=1, kind="mergesort")
        first_local = order[:, 0]
        nearest[query_rows] = donor_rows[first_local]
        if len(donor_rows) > 1:
            margin[query_rows] = (
                np.take_along_axis(pair, order[:, 1:2], axis=1)[:, 0]
                - pair[np.arange(len(query_rows)), first_local]
            )
        else:
            margin[query_rows] = np.inf
        reverse = np.argmin(pair, axis=0)
        mutual[query_rows] = reverse[first_local] == np.arange(len(query_rows))
        conditional_groups += 1

    if (nearest < 0).any():
        raise RuntimeError("some challenge rows were not linked")
    margin_rank = pd.Series(margin).rank(method="average", pct=True).to_numpy(
        dtype=float
    )
    event = (
        released["onset"].to_numpy(dtype=int)[nearest]
        + released["death"].to_numpy(dtype=int)[nearest]
    ) > 0
    gate = mutual & event & (margin_rank >= margin_quantile)
    linked = np.full(len(challenge), horizon_time, dtype=float)
    linked[gate] = released["time"].to_numpy(dtype=float)[nearest[gate]]
    return linked, gate, {
        "decoder": "block_conditional_copula_event_links",
        "cdf_conditioning": "prefecture+sex" if condition_on_sex else "prefecture",
        "matching_block": "prefecture+sex",
        "margin_quantile": margin_quantile,
        "release_rows": len(released),
        "challenge_rows": len(challenge),
        "conditional_groups": conditional_groups,
        "fallback_query_rows": int(fallback_rows),
        "mutual_links": int(mutual.sum()),
        "event_links": int(event.sum()),
        "trusted_event_links": int(gate.sum()),
    }


def _top_fraction_gate(
    score: np.ndarray, challenge: pd.DataFrame, quantile: float
) -> np.ndarray:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("gate quantile must be in [0,1]")
    count = int(np.ceil((1.0 - quantile) * len(score)))
    count = int(np.clip(count, 0, len(score)))
    ids = (
        challenge["challenge_row_id"].to_numpy(dtype=np.int64)
        if "challenge_row_id" in challenge
        else np.arange(len(challenge), dtype=np.int64)
    )
    order = np.lexsort((ids, -np.asarray(score, dtype=float)))
    gate = np.zeros(len(score), dtype=bool)
    gate[order[:count]] = True
    return gate


def block_knn_event_window_prediction(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    gate_quantile: float = 0.98,
    risk_neighbours: int = 32,
    time_neighbours: int = 16,
    nearest_event_time: bool = False,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Conditional event-risk and narrow-window time mode inside home blocks."""
    r_pref = released["prefecture"].astype(str).to_numpy()
    c_pref = challenge["prefecture"].astype(str).to_numpy()
    r_sex = released["sex"].to_numpy()
    c_sex = challenge["sex"].to_numpy()
    event_all = (
        released["onset"].to_numpy(dtype=int)
        + released["death"].to_numpy(dtype=int)
    ) > 0
    time_all = released["time"].to_numpy(dtype=float)
    risk = np.zeros(len(challenge), dtype=float)
    predicted = np.full(len(challenge), horizon_time, dtype=float)
    nearest_event_distance = np.full(len(challenge), np.inf, dtype=float)

    blocks = pd.DataFrame({"prefecture": c_pref, "sex": c_sex})
    for raw_key, query_indices in blocks.groupby(
        ["prefecture", "sex"], sort=False
    ).indices.items():
        prefecture, sex = raw_key
        query_rows = np.asarray(query_indices, dtype=int)
        release_rows = np.flatnonzero(
            (r_pref == str(prefecture)) & (r_sex == sex)
        )
        if not len(release_rows):
            release_rows = np.flatnonzero(r_sex == sex)
        if not len(release_rows):
            release_rows = np.arange(len(released), dtype=int)
        rx = _rank_matrix(released, release_rows)
        qx = _rank_matrix(challenge, query_rows)
        distance = cdist(qx, rx, metric="euclidean")
        order = np.argsort(distance, axis=1, kind="mergesort")
        k_risk = min(risk_neighbours, len(release_rows))
        local_events = event_all[release_rows]
        risk[query_rows] = local_events[order[:, :k_risk]].mean(axis=1)
        event_positions = np.flatnonzero(local_events)
        if not len(event_positions):
            continue
        for local_query, query_row in enumerate(query_rows):
            event_order = event_positions[
                np.argsort(
                    distance[local_query, event_positions], kind="mergesort"
                )
            ]
            selected = event_order[: min(time_neighbours, len(event_order))]
            selected_distance = distance[local_query, selected]
            selected_time = time_all[release_rows[selected]]
            nearest_event_distance[query_row] = float(selected_distance[0])
            if nearest_event_time:
                predicted[query_row] = float(selected_time[0])
                continue
            bandwidth = max(float(np.median(selected_distance)), 1e-6)
            weights = np.exp(-selected_distance / bandwidth)
            support = np.asarray(
                [
                    weights[np.abs(selected_time - center) <= 0.5].sum()
                    for center in selected_time
                ],
                dtype=float,
            )
            best = np.lexsort((selected_distance, -support))[0]
            predicted[query_row] = float(selected_time[best])

    proximity_rank = (
        pd.Series(-nearest_event_distance)
        .rank(method="average", pct=True)
        .to_numpy(dtype=float)
    )
    score = risk + 0.10 * proximity_rank
    gate = _top_fraction_gate(score, challenge, gate_quantile)
    gate &= np.isfinite(nearest_event_distance)
    return predicted, gate, {
        "decoder": "block_knn_event_window_prediction",
        "gate_quantile": gate_quantile,
        "risk_neighbours": risk_neighbours,
        "time_neighbours": time_neighbours,
        "time_rule": "nearest_event" if nearest_event_time else "weighted_window_mode",
        "gate_rows": int(gate.sum()),
        "home_release_event_rows": int(event_all.sum()),
        "median_gate_risk": float(np.median(risk[gate])) if gate.any() else None,
    }


def block_extra_trees_event_prediction(
    released: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    seed: int,
    gate_quantile: float = 0.98,
    horizon_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Smooth home-prefecture conditional event/time comparison model."""
    prefectures = challenge["prefecture"].astype(str).unique()
    if len(prefectures) != 1:
        raise ValueError("expected a single challenge prefecture")
    home = str(prefectures[0])
    release_rows = np.flatnonzero(
        released["prefecture"].astype(str).to_numpy() == home
    )
    if len(release_rows) < 25:
        release_rows = np.arange(len(released), dtype=int)
    release_x = _rank_matrix(released, release_rows)
    challenge_x = _rank_matrix(challenge, np.arange(len(challenge), dtype=int))
    release_x = np.column_stack(
        [release_x, released.iloc[release_rows]["sex"].to_numpy(dtype=float)]
    )
    challenge_x = np.column_stack(
        [challenge_x, challenge["sex"].to_numpy(dtype=float)]
    )
    event = (
        released.iloc[release_rows]["onset"].to_numpy(dtype=int)
        + released.iloc[release_rows]["death"].to_numpy(dtype=int)
    ) > 0
    classifier = ExtraTreesClassifier(
        n_estimators=256,
        min_samples_leaf=8,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    ).fit(release_x, event.astype(int))
    probability = classifier.predict_proba(challenge_x)[:, 1]
    gate = _top_fraction_gate(probability, challenge, gate_quantile)
    prediction = np.full(len(challenge), horizon_time, dtype=float)
    if event.any():
        regressor = ExtraTreesRegressor(
            n_estimators=256,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=seed + 1,
            n_jobs=1,
        ).fit(
            release_x[event],
            released.iloc[release_rows].loc[event, "time"].to_numpy(dtype=float),
        )
        prediction[gate] = np.clip(
            regressor.predict(challenge_x[gate]), 0.0, horizon_time
        )
    return prediction, gate, {
        "decoder": "block_extra_trees_event_prediction",
        "gate_quantile": gate_quantile,
        "release_rows": int(len(release_rows)),
        "release_event_rows": int(event.sum()),
        "gate_rows": int(gate.sum()),
        "median_gate_probability": float(np.median(probability[gate]))
        if gate.any()
        else None,
    }
