"""Outcome-integrity routing for attack version 4 AIA predictions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


FINGERPRINT_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
NUMERIC_COLUMNS = ("age", "SBP", "TG", "HDL", "ALT")
CATEGORICAL_COLUMNS = ("sex", "prefecture")


class OutcomeIntegrityFingerprint:
    """Measure whether a release retains the public QI/outcome association."""

    def __init__(self, background: pd.DataFrame) -> None:
        required = (*FINGERPRINT_COLUMNS, "onset", "death")
        missing = [column for column in required if column not in background.columns]
        if missing:
            raise ValueError(f"background is missing outcome fingerprint columns: {missing}")
        event = (
            background["onset"].to_numpy(dtype=int)
            + background["death"].to_numpy(dtype=int)
        ) > 0
        if len(np.unique(event)) != 2:
            raise ValueError("background event indicator must contain two classes")
        transformer = ColumnTransformer(
            [
                ("numeric", RobustScaler(), list(NUMERIC_COLUMNS)),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(CATEGORICAL_COLUMNS),
                ),
            ]
        )
        self.model = make_pipeline(
            transformer,
            LogisticRegression(
                C=0.25,
                max_iter=2000,
                class_weight="balanced",
                random_state=0,
            ),
        )
        self.model.fit(background[list(FINGERPRINT_COLUMNS)], event.astype(int))

    def measure(self, released: pd.DataFrame) -> dict[str, Any]:
        required = (*FINGERPRINT_COLUMNS, "time", "onset", "death")
        missing = [column for column in required if column not in released.columns]
        if missing:
            raise ValueError(f"release is missing outcome fingerprint columns: {missing}")
        if not len(released):
            raise ValueError("release is empty")
        event = (
            released["onset"].to_numpy(dtype=int)
            + released["death"].to_numpy(dtype=int)
        ) > 0
        probability = self.model.predict_proba(
            released[list(FINGERPRINT_COLUMNS)]
        )[:, 1]
        auc = (
            float(roc_auc_score(event, probability))
            if 0 < int(event.sum()) < len(event)
            else None
        )
        time_risk = 10.0 - released["time"].to_numpy(dtype=float)
        correlation = float(np.corrcoef(probability, time_risk)[0, 1])
        if not math.isfinite(correlation):
            correlation = None
        return {
            "auc": auc,
            "correlation": correlation,
            "event_rate": float(event.mean()),
        }


def fixed_selection_horizon(
    adaptive: pd.DataFrame,
    *,
    abstain_time: float,
    horizon_time: float,
) -> pd.DataFrame:
    """Replace only active predictions, retaining the exact selected key set."""
    required = ("target_k", "challenge_row_id", "time_hat")
    missing = [column for column in required if column not in adaptive.columns]
    if missing:
        raise ValueError(f"adaptive AIA frame is missing columns: {missing}")
    output = adaptive.reset_index(drop=True).copy()
    values = output["time_hat"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("adaptive AIA times must be finite")
    active = ~np.isclose(values, abstain_time)
    output.loc[active, "time_hat"] = horizon_time
    if not np.array_equal(
        ~np.isclose(output["time_hat"].to_numpy(dtype=float), abstain_time),
        active,
    ):
        raise RuntimeError("horizon decoder changed AIA membership selection")
    return output


def route_target_blocks(
    adaptive: pd.DataFrame,
    horizon: pd.DataFrame,
    releases: dict[str, pd.DataFrame],
    fingerprint: OutcomeIntegrityFingerprint,
    *,
    auc_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Route complete target blocks while requiring identical challenge keys."""
    if not 0.0 <= auc_threshold <= 1.0:
        raise ValueError("AIA integrity AUC threshold must be in [0, 1]")
    required = ["target_k", "challenge_row_id", "time_hat"]
    if list(adaptive.columns) != required or list(horizon.columns) != required:
        raise ValueError("AIA expert frames must have the official three columns")
    if adaptive.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("adaptive AIA frame has duplicate challenge keys")
    if horizon.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("horizon AIA frame has duplicate challenge keys")
    adaptive = adaptive.reset_index(drop=True)
    horizon = horizon.reset_index(drop=True)
    output = adaptive.copy()
    diagnostics: dict[str, Any] = {}
    routed_targets: list[str] = []
    for target_id in sorted(releases, key=int):
        measure = fingerprint.measure(releases[target_id])
        auc = measure["auc"]
        use_horizon = auc is None or float(auc) < auc_threshold
        adaptive_mask = output["target_k"].astype(int).eq(int(target_id))
        horizon_mask = horizon["target_k"].astype(int).eq(int(target_id))
        adaptive_ids = output.loc[
            adaptive_mask, "challenge_row_id"
        ].to_numpy(dtype=np.int64)
        horizon_ids = horizon.loc[
            horizon_mask, "challenge_row_id"
        ].to_numpy(dtype=np.int64)
        if not np.array_equal(adaptive_ids, horizon_ids):
            raise ValueError(f"AIA expert challenge keys differ for target {target_id}")
        if use_horizon:
            output.loc[adaptive_mask, "time_hat"] = horizon.loc[
                horizon_mask, "time_hat"
            ].to_numpy(dtype=float)
            routed_targets.append(str(target_id))
        diagnostics[str(target_id)] = {
            **measure,
            "decoder": "fixed_selection_horizon" if use_horizon else "adaptive",
            "changed_rows": int(
                np.count_nonzero(
                    ~np.isclose(
                        output.loc[adaptive_mask, "time_hat"].to_numpy(dtype=float),
                        adaptive.loc[adaptive_mask, "time_hat"].to_numpy(dtype=float),
                    )
                )
            ),
        }
    return output, {
        "auc_threshold": auc_threshold,
        "routed_targets": routed_targets,
        "targets": diagnostics,
    }
