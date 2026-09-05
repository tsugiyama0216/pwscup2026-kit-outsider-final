"""Public-feature-only proxies for the undistributed AIA rare label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


VISIBLE_RARE_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT")


def stable_rank(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Return deterministic ranks in [0, 1], with larger input ranked higher."""
    values = np.asarray(values, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    order = np.lexsort((ids, values))
    rank = np.empty(len(values), dtype=float)
    rank[order] = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    return rank


def partial_mahalanobis(frame: pd.DataFrame, detector: Mapping[str, Any]) -> np.ndarray:
    """Mahalanobis distance using the published covariance's visible submatrix."""
    detector_columns = list(detector["columns"])
    positions = [detector_columns.index(column) for column in VISIBLE_RARE_COLUMNS]
    mu = np.asarray(detector["mu"], dtype=float)[positions]
    covariance = np.asarray(detector["sigma"], dtype=float)[np.ix_(positions, positions)]
    precision = np.linalg.pinv(covariance)
    x = frame.loc[:, VISIBLE_RARE_COLUMNS].to_numpy(dtype=float) - mu
    squared = np.einsum("ij,jk,ik->i", x, precision, x)
    return np.sqrt(np.clip(squared, 0.0, None))


@dataclass
class RareProxyModels:
    quadratic_logit: Any
    extra_trees: ExtraTreesClassifier
    detector: Mapping[str, Any]

    def scores(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        x = frame.loc[:, VISIBLE_RARE_COLUMNS].to_numpy(dtype=float)
        ids = (
            frame["challenge_row_id"].to_numpy(dtype=np.int64)
            if "challenge_row_id" in frame
            else np.arange(len(frame), dtype=np.int64)
        )
        partial = partial_mahalanobis(frame, self.detector)
        logit = self.quadratic_logit.predict_proba(x)[:, 1]
        trees = self.extra_trees.predict_proba(x)[:, 1]
        ranks = {
            "partial_mahal": stable_rank(partial, ids),
            "quadratic_logit": stable_rank(logit, ids),
            "extra_trees": stable_rank(trees, ids),
        }
        ranks["ensemble"] = np.mean(np.column_stack(list(ranks.values())), axis=1)
        return ranks


def fit_proxy_models(
    background: pd.DataFrame,
    labels: np.ndarray,
    detector: Mapping[str, Any],
    *,
    seed: int = 20270300,
) -> RareProxyModels:
    labels = np.asarray(labels, dtype=int)
    if len(labels) != len(background) or len(np.unique(labels)) != 2:
        raise ValueError("rare proxy training labels are invalid")
    x = background.loc[:, VISIBLE_RARE_COLUMNS].to_numpy(dtype=float)
    quadratic = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegression(
            C=0.10,
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        ),
    ).fit(x, labels)
    trees = ExtraTreesClassifier(
        n_estimators=256,
        min_samples_leaf=8,
        max_features=0.8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, labels)
    return RareProxyModels(quadratic, trees, detector)


def top_k_mask(score: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    ids = np.asarray(ids, dtype=np.int64)
    if not 0 <= k <= len(score):
        raise ValueError("top-k value is outside the frame")
    order = np.lexsort((ids, -score))
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:k]] = True
    return selected
