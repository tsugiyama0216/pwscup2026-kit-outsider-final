"""A single time predictor supplies both attribute and residual membership inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# The evaluator imports this file by absolute path in an isolated Python process.
_ATTACK_DIRECTORY = str(Path(__file__).resolve().parent.parent)
if _ATTACK_DIRECTORY not in sys.path:
    sys.path.insert(0, _ATTACK_DIRECTORY)
from explore_common.core import score_to_confidence

MODEL_NAME = "explore_4_predictor: time CART residual difference"
CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
NUMERIC_QI = tuple(CONFIG["numeric_qi"])
CATEGORICAL_QI = tuple(CONFIG["categorical_qi"])


def qi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Read only the seven public QI; never use time as a feature."""
    result = pd.DataFrame(index=frame.index)
    for column in NUMERIC_QI:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"QI {column} contains a non-finite value")
        result[column] = values
    for column in CATEGORICAL_QI:
        if frame[column].isna().any():
            raise ValueError(f"QI {column} contains a missing value")
        # Sex is a numeric category; normalize 0, 0.0 and '0' identically.
        if column == "sex":
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
            if not np.isin(values, [0.0, 1.0]).all():
                raise ValueError("sex must be 0 or 1")
            result[column] = values.astype(int).astype(str)
        else:
            result[column] = frame[column].astype(str).to_numpy()
    return result


def observed_time(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("time contains a non-finite value")
    if ((values <= 0) | (values > CONFIG["horizon_years"])).any():
        raise ValueError("time must be in (0, 10] years")
    return values


def fit_preprocessor(background: pd.DataFrame) -> ColumnTransformer:
    """Fit categorical vocabulary on background only and keep numeric QI unchanged."""
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         list(CATEGORICAL_QI)),
        ("numeric", "passthrough", list(NUMERIC_QI)),
    ], remainder="drop")
    preprocessor.fit(qi_frame(background))
    return preprocessor


class SyntheticPredictor:
    """Fixed CART regression trained on one table's QI and observed time."""

    def __init__(self, preprocessor: ColumnTransformer):
        self.preprocessor = preprocessor
        self.regressor = DecisionTreeRegressor(**CONFIG["tree"])

    def fit(self, training: pd.DataFrame) -> "SyntheticPredictor":
        if len(training) == 0:
            raise ValueError("Predictor training data must be nonempty")
        features = self.preprocessor.transform(qi_frame(training))
        self.regressor.fit(features, observed_time(training))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        if len(rows) == 0:
            return np.empty(0, dtype=float)
        return self.regressor.predict(self.preprocessor.transform(qi_frame(rows)))


def residual_difference(time: np.ndarray, background_prediction: np.ndarray,
                        synthetic_prediction: np.ndarray) -> np.ndarray:
    """Positive means the synthetic model predicts known time better than background."""
    return np.abs(background_prediction - time) - np.abs(synthetic_prediction - time)


def predict_world(*, candidates, self_labels=None, background, self_challenge=None,
                  targets, schema=None, cfg=None, attacker_id=24, self_release=None,
                  utility_reference=None):
    """Public-table API. Candidate time is used only after fitting, in MIA residuals."""
    preprocessor = fit_preprocessor(background)
    background_model = SyntheticPredictor(preprocessor).fit(background)
    candidate_time = observed_time(candidates)
    background_prediction = background_model.predict(candidates)
    mia = pd.DataFrame({"record_id": candidates["record_id"].to_numpy(copy=True)})
    aia_frames = []
    target_diagnostics = {}
    for target_id, target in sorted(targets.items(), key=lambda item: int(item[0])):
        target_id = int(target_id)
        if target_id == int(attacker_id):
            continue
        model = SyntheticPredictor(preprocessor).fit(target["release"])
        synthetic_prediction = model.predict(candidates)
        raw = residual_difference(candidate_time, background_prediction, synthetic_prediction)
        mia[str(target_id)] = score_to_confidence(raw)
        challenge = target["challenge"]
        aia_frames.append(pd.DataFrame({
            "target_k": target_id,
            "challenge_row_id": challenge["challenge_row_id"].to_numpy(copy=True),
            "time_hat": model.predict(challenge),
        }))
        target_diagnostics[str(target_id)] = {
            "training_rows": int(len(target["release"])),
            "challenge_rows": int(len(challenge)),
            "tree_depth": int(model.regressor.get_depth()),
            "tree_leaves": int(model.regressor.get_n_leaves()),
            "raw_score_min": float(raw.min()) if raw.size else None,
            "raw_score_max": float(raw.max()) if raw.size else None,
        }
    aia = (pd.concat(aia_frames, ignore_index=True) if aia_frames else
           pd.DataFrame(columns=["target_k", "challenge_row_id", "time_hat"]))
    return {
        "F_mia": mia,
        "F_aia": aia,
        "diagnostics": {
            "model_name": MODEL_NAME,
            "configuration": CONFIG,
            "preprocessor_fit_on": "background only",
            "candidate_time_use": "MIA absolute residuals only; never fitted",
            "background_rows": int(len(background)),
            "transformed_feature_count": int(background_model.regressor.n_features_in_),
            "targets": target_diagnostics,
        },
    }
