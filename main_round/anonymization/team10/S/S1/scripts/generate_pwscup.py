"""Generate a PWS Cup 2026 defense table with one OpenDP AIM release.

The private cohort is converted to categorical bins using only public settings.
OpenDP measures those bins with AIM. Numeric values and event times are then
decoded using public background data and public intervals as DP postprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import opendp.prelude as dp
import pandas as pd
import polars as pl


OUTCOME_CENSORED = 0
CLINICAL_COLUMNS = ("BMI", "SBP", "TG", "HDL", "ALT", "FPG")
BINARY_COLUMNS = ("sex", "smoking", "onset", "death")
COX_RISK_FEATURES = (
    "age",
    "sex",
    "BMI",
    "SBP",
    "log_TG",
    "HDL",
    "log_ALT",
    "FPG",
    "smoking",
)
COX_LOG_FEATURE_SOURCES = {"log_TG": "TG", "log_ALT": "ALT"}


class ConfigurationError(ValueError):
    """Raised when a public configuration is internally inconsistent."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a differentially private PWSCUP C.csv with OpenDP AIM"
    )
    parser.add_argument("--input", required=True, help="Path to the private B cohort CSV")
    parser.add_argument(
        "--background",
        help="Path to public A_bg.csv (default: A_bg.csv next to --input)",
    )
    parser.add_argument("--schema", required=True, help="Path to public schema.json")
    parser.add_argument("--config", required=True, help="Path to public AIM config JSON")
    parser.add_argument("--output", required=True, help="Destination C.csv")
    parser.add_argument(
        "--privacy-report", required=True, help="Destination privacy metadata JSON"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all public/private inputs and exit before any DP release",
    )
    parser.add_argument(
        "--release-ledger",
        help=(
            "Local-only JSON ledger created immediately before the DP release; "
            "an existing ledger blocks a new release"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON: {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def initialize_release_ledger(
    path: Path,
    *,
    input_path: Path,
    background_path: Path,
    schema_path: Path,
    config_path: Path,
    output_path: Path,
    report_path: Path,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uv_lock = Path(__file__).resolve().parents[1] / "uv.lock"
    ledger: dict[str, Any] = {
        "ledger_version": 1,
        "release_attempt_id": str(uuid4()),
        "status": "release_started",
        "release_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "rho_zcdp": float(config["privacy"]["rho"]),
        "input_sha256": sha256_file(input_path),
        "public_background_sha256": sha256_file(background_path),
        "public_schema_sha256": sha256_file(schema_path),
        "config_sha256": sha256_file(config_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "output_path": str(output_path),
        "privacy_report_path": str(report_path),
    }
    if uv_lock.exists():
        ledger["opendp_uv_lock_sha256"] = sha256_file(uv_lock)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(
            "release ledger already exists; refusing another DP release: " f"{path}"
        ) from exc


def update_release_ledger(path: Path, status: str, **updates: Any) -> None:
    ledger = load_json(path)
    ledger["status"] = status
    ledger.update(updates)
    write_json_atomic(path, ledger)


def _strictly_increasing(values: list[float]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def validate_public_configuration(config: dict[str, Any], schema: dict[str, Any]) -> None:
    model_columns = config["model_columns"]
    categorical = config["categorical_keys"]
    numeric = config["numeric_bin_edges"]

    if len(model_columns) != len(set(model_columns)):
        raise ConfigurationError("model_columns contains duplicates")
    if set(model_columns) != set(categorical) | set(numeric):
        raise ConfigurationError(
            "every model column must have categorical keys or numeric bin edges"
        )

    for name, values in numeric.items():
        if len(values) < 2 or not _strictly_increasing(values):
            raise ConfigurationError(f"bin edges must be strictly increasing: {name}")

    expected_bounds = {"age": schema["age_range"], **schema["ranges"]}
    for name, edges in numeric.items():
        lo, hi = expected_bounds[name]
        if not math.isclose(float(edges[0]), float(lo)) or not math.isclose(
            float(edges[-1]), float(hi)
        ):
            raise ConfigurationError(f"bin edges must span the public range: {name}")

    time_bins = config["postprocess"]["event_time_bins"]
    time_floor = float(config["postprocess"]["time_floor"])
    time_horizon = float(config["postprocess"]["time_horizon"])
    if not time_bins or not math.isclose(float(time_bins[0][0]), time_floor):
        raise ConfigurationError("event_time_bins must start at time_floor")
    if not math.isclose(float(time_bins[-1][1]), time_horizon):
        raise ConfigurationError("event_time_bins must end at time_horizon")
    for idx, (lo, hi) in enumerate(time_bins):
        if not lo < hi:
            raise ConfigurationError("event_time_bins must have positive width")
        if idx and not math.isclose(float(time_bins[idx - 1][1]), float(lo)):
            raise ConfigurationError("event_time_bins must be contiguous")

    expected_states = list(range(1 + 2 * len(time_bins)))
    if categorical["outcome_state"] != expected_states:
        raise ConfigurationError("outcome_state keys do not match event_time_bins")

    donor_config = config["postprocess"].get("public_rare_donor", {})
    if donor_config.get("enabled", False):
        detector_columns = list(schema["rare_detector"]["columns"])
        if list(donor_config.get("columns", detector_columns)) != detector_columns:
            raise ConfigurationError(
                "public_rare_donor.columns must match rare_detector.columns in order"
            )
        target = donor_config.get("target", "detected_output")
        if target not in {"detected_output", "public_background_rate"}:
            raise ConfigurationError(
                "public_rare_donor.target must be 'detected_output' or "
                "'public_background_rate'"
            )
        selection = donor_config.get("selection", "without_replacement")
        if selection not in {"without_replacement", "fixed_permutation"}:
            raise ConfigurationError(
                "public_rare_donor.selection must be 'without_replacement' or "
                "'fixed_permutation'"
            )
        if target == "public_background_rate":
            if donor_config.get("target_row_selection") != "largest_mahalanobis":
                raise ConfigurationError(
                    "public-rate rare targets require target_row_selection="
                    "'largest_mahalanobis'"
                )
            if donor_config.get("demote_excess") is not True:
                raise ConfigurationError(
                    "public-rate rare targets require demote_excess=true"
                )
            if donor_config.get("demote_selection") != "nearest_below_threshold":
                raise ConfigurationError(
                    "public-rate rare targets require demote_selection="
                    "'nearest_below_threshold'"
                )
        maximum_distance = donor_config.get("maximum_distance_exclusive")
        if maximum_distance is not None:
            maximum_distance = float(maximum_distance)
            detector_lo = float(schema["rare_detector"]["band"]["lo"])
            if not math.isfinite(maximum_distance) or maximum_distance <= detector_lo:
                raise ConfigurationError(
                    "public_rare_donor.maximum_distance_exclusive must be finite "
                    "and greater than the rare threshold"
                )
        donor_seed = donor_config.get("seed")
        if donor_seed is not None and (
            isinstance(donor_seed, bool)
            or not isinstance(donor_seed, int)
            or donor_seed < 0
        ):
            raise ConfigurationError(
                "public_rare_donor.seed must be a nonnegative integer or null"
            )

    outcome_donor = config["postprocess"].get("public_outcome_donor", {})
    if outcome_donor.get("enabled", False):
        expected_replacements = ["time", "onset", "death"]
        if outcome_donor.get("selection") != "nearest":
            raise ConfigurationError(
                "public_outcome_donor.selection must be 'nearest'"
            )
        if outcome_donor.get("neighbors") != 1:
            raise ConfigurationError("public_outcome_donor.neighbors must be 1")
        if list(outcome_donor.get("replace_columns", [])) != expected_replacements:
            raise ConfigurationError(
                "public_outcome_donor.replace_columns must be "
                "['time', 'onset', 'death']"
            )

        distance = outcome_donor.get("distance")
        if distance == "standardized_numeric_plus_onehot":
            expected_numeric = list(schema["rare_detector"]["columns"])
            expected_categorical = ["sex", "prefecture", "smoking"]
            if list(outcome_donor.get("numeric_columns", [])) != expected_numeric:
                raise ConfigurationError(
                    "public_outcome_donor.numeric_columns must match "
                    "rare_detector.columns in order"
                )
            if (
                list(outcome_donor.get("categorical_columns", []))
                != expected_categorical
            ):
                raise ConfigurationError(
                    "public_outcome_donor.categorical_columns must be "
                    "['sex', 'prefecture', 'smoking']"
                )
        elif distance == "public_cox_risk":
            risk = outcome_donor.get("risk", {})
            source_hash = risk.get("source_background_sha256")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ConfigurationError(
                    "public_outcome_donor.risk.source_background_sha256 "
                    "must be a SHA-256 hex digest"
                )
            try:
                int(source_hash, 16)
            except ValueError as exc:
                raise ConfigurationError(
                    "public_outcome_donor.risk.source_background_sha256 "
                    "must be a SHA-256 hex digest"
                ) from exc
            if list(risk.get("features", [])) != list(COX_RISK_FEATURES):
                raise ConfigurationError(
                    "public_outcome_donor.risk.features does not match "
                    "the supported Cox design"
                )
            for event in ("onset", "death"):
                coefficients = risk.get(f"{event}_coefficients", {})
                if set(coefficients) != set(COX_RISK_FEATURES):
                    raise ConfigurationError(
                        f"public_outcome_donor.risk.{event}_coefficients must "
                        "contain every Cox feature"
                    )
                if not all(
                    math.isfinite(float(value)) for value in coefficients.values()
                ):
                    raise ConfigurationError(
                        f"public_outcome_donor.risk.{event}_coefficients must be finite"
                    )
            death_weight = float(risk.get("death_weight", math.nan))
            if not math.isfinite(death_weight) or death_weight < 0:
                raise ConfigurationError(
                    "public_outcome_donor.risk.death_weight must be finite and nonnegative"
                )
        else:
            raise ConfigurationError(
                "public_outcome_donor.distance must be "
                "'standardized_numeric_plus_onehot' or 'public_cox_risk'"
            )

        chunk_size = outcome_donor.get("chunk_size", 128)
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ConfigurationError(
                "public_outcome_donor.chunk_size must be a positive integer"
            )

    expected_output = [
        "age",
        "sex",
        "prefecture",
        "BMI",
        "SBP",
        "TG",
        "HDL",
        "ALT",
        "smoking",
        "FPG",
        "time",
        "onset",
        "death",
    ]
    if schema["submitted_c_columns"] != expected_output:
        raise ConfigurationError("unexpected submitted_c_columns in schema.json")

    if config["privacy"]["definition"] != "rho-zCDP":
        raise ConfigurationError("generator supports only rho-zCDP")
    rho = float(config["privacy"]["rho"])
    if not math.isfinite(rho) or rho <= 0:
        raise ConfigurationError("privacy.rho must be positive and finite")


def validate_private_input(
    source: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    if list(source.columns) != schema["distributed_columns"]:
        raise ValueError(
            "input columns do not match the public distributed schema; "
            f"got {list(source.columns)}"
        )
    expected_rows = int(config["expected_input_rows"])
    if len(source) != expected_rows:
        raise ValueError(f"input must contain {expected_rows} rows; got {len(source)}")

    numeric_input = ["age", *CLINICAL_COLUMNS, "time", *BINARY_COLUMNS]
    for name in numeric_input:
        source[name] = pd.to_numeric(source[name], errors="raise")
        if not np.isfinite(source[name].to_numpy(dtype=float)).all():
            raise ValueError(f"input contains a non-finite value: {name}")

    age_lo, age_hi = schema["age_range"]
    if not source["age"].between(age_lo, age_hi, inclusive="both").all():
        raise ValueError("input age is outside the public range")
    if not np.equal(source["age"], np.floor(source["age"])).all():
        raise ValueError("input age must be integral")
    for name, (lo, hi) in schema["ranges"].items():
        if not source[name].between(lo, hi, inclusive="both").all():
            raise ValueError(f"input {name} is outside the public range")
    for name in BINARY_COLUMNS:
        if not source[name].isin([0, 1]).all():
            raise ValueError(f"input {name} must be binary")

    actual_events = set(zip(source["onset"].astype(int), source["death"].astype(int)))
    if not actual_events <= {(0, 0), (1, 0), (0, 1)}:
        raise ValueError("input contains an invalid onset/death combination")

    time_floor = float(config["postprocess"]["time_floor"])
    time_horizon = float(config["postprocess"]["time_horizon"])
    if not source["time"].between(time_floor, time_horizon, inclusive="both").all():
        raise ValueError("input time is outside the public range")

    for name in ("sex", "prefecture", "smoking"):
        if not source[name].isin(config["categorical_keys"][name]).all():
            raise ValueError(f"input contains a value outside the public key-set: {name}")


def bin_indices(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Map public-range values to left-open bins represented by integer keys."""

    indices = np.searchsorted(np.asarray(edges[1:], dtype=float), values, side="left")
    return indices.astype(int)


def encode_outcome_state(source: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    time = source["time"].to_numpy(dtype=float)
    bins = config["postprocess"]["event_time_bins"]
    upper_edges = np.asarray([interval[1] for interval in bins], dtype=float)
    time_bin = np.searchsorted(upper_edges, time, side="left")
    time_bin = np.clip(time_bin, 0, len(bins) - 1)

    onset = source["onset"].to_numpy(dtype=int)
    death = source["death"].to_numpy(dtype=int)
    state = np.zeros(len(source), dtype=int)
    state[onset == 1] = 1 + time_bin[onset == 1]
    state[death == 1] = 1 + len(bins) + time_bin[death == 1]
    return state


def rare_distances(source: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    detector = schema["rare_detector"]
    columns = detector["columns"]
    values = source[columns].to_numpy(dtype=float)
    mu = np.asarray(detector["mu"], dtype=float)
    sigma = np.asarray(detector["sigma"], dtype=float)
    diff = values - mu
    inv_sigma = np.linalg.pinv(sigma)
    distance_sq = np.einsum("ij,jk,ik->i", diff, inv_sigma, diff)
    return np.sqrt(np.clip(distance_sq, 0.0, None))


def encode_rare_flag(source: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
    distance = rare_distances(source, schema)
    return (distance >= float(schema["rare_detector"]["band"]["lo"])).astype(int)


def replace_detected_rare_with_public_donors(
    output: pd.DataFrame,
    background: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Restore rare-group joint structure from public rows as DP postprocessing.

    The target mask is computed only from the already-DP synthetic output, and
    donor rows come only from public A_bg. Therefore this step consumes no
    additional privacy budget. All rare-marker values are copied from the same
    public row so that column-wise decoding does not destroy their dependence.
    """

    donor_config = config["postprocess"].get("public_rare_donor", {})
    if not bool(donor_config.get("enabled", False)):
        return

    detector_columns = list(schema["rare_detector"]["columns"])
    configured_columns = list(donor_config.get("columns", detector_columns))
    if configured_columns != detector_columns:
        raise ConfigurationError(
            "public_rare_donor.columns must match rare_detector.columns in order"
        )
    target = donor_config.get("target", "detected_output")
    if target not in {"detected_output", "public_background_rate"}:
        raise ConfigurationError("unsupported public_rare_donor.target")
    selection = donor_config.get("selection", "without_replacement")
    if selection not in {"without_replacement", "fixed_permutation"}:
        raise ConfigurationError("unsupported public_rare_donor.selection")
    donor_seed = donor_config.get("seed")
    donor_rng = rng if donor_seed is None else np.random.default_rng(int(donor_seed))

    detector_lo = float(schema["rare_detector"]["band"]["lo"])
    output_distances = rare_distances(output, schema)
    public_distances = rare_distances(background, schema)
    public_rare_mask = public_distances >= detector_lo

    if target == "detected_output":
        target_indices = np.flatnonzero(output_distances >= detector_lo)
    else:
        target_count = int(round(float(public_rare_mask.mean()) * len(output)))
        target_indices = np.argsort(-output_distances, kind="stable")[:target_count]

        target_mask = np.zeros(len(output), dtype=bool)
        target_mask[target_indices] = True
        demote_indices = np.flatnonzero(
            (output_distances >= detector_lo) & ~target_mask
        )
        if len(demote_indices):
            nonrare_indices = np.flatnonzero(public_distances < detector_lo)
            if not len(nonrare_indices):
                raise ValueError(
                    "public A_bg contains no non-rare rows for rare-count repair"
                )
            nearest_order = nonrare_indices[
                np.argsort(-public_distances[nonrare_indices], kind="stable")
            ]
            demoter_indices = np.resize(nearest_order, len(demote_indices))
            marker_positions = [
                output.columns.get_loc(name) for name in detector_columns
            ]
            output.iloc[demote_indices, marker_positions] = background.iloc[
                demoter_indices
            ][detector_columns].to_numpy()

    target_count = len(target_indices)
    if target_count == 0:
        if encode_rare_flag(output, schema).any():
            raise ValueError("rare-count repair did not remove every rare row")
        return

    maximum_distance = donor_config.get("maximum_distance_exclusive")
    if maximum_distance is not None:
        public_rare_mask &= public_distances < float(maximum_distance)
    donors = background.loc[public_rare_mask].reset_index(drop=True)
    if donors.empty:
        raise ValueError("public A_bg contains no rows matching the public rare detector")

    if selection == "fixed_permutation":
        donor_indices = np.resize(donor_rng.permutation(len(donors)), target_count)
    else:
        donor_indices = donor_rng.choice(
            len(donors),
            size=target_count,
            replace=target_count > len(donors),
        )
    selected_donors = donors.iloc[donor_indices]
    marker_positions = [output.columns.get_loc(name) for name in detector_columns]
    output.iloc[target_indices, marker_positions] = selected_donors[
        detector_columns
    ].to_numpy()

    final_rare = encode_rare_flag(output, schema).astype(bool)
    if not final_rare[target_indices].all():
        raise ValueError("public rare-donor replacement produced a non-rare target row")
    if target == "public_background_rate" and int(final_rare.sum()) != target_count:
        raise ValueError(
            "public rare-donor replacement did not produce the public-rate count"
        )


def public_cox_design_matrix(
    frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature in features:
        source = COX_LOG_FEATURE_SOURCES.get(feature, feature)
        values = frame[source].to_numpy(dtype=float)
        if feature in COX_LOG_FEATURE_SOURCES:
            values = np.log(np.clip(values, 1e-6, None))
        columns.append(values)
    return np.column_stack(columns)


def outcome_donor_features(
    frame: pd.DataFrame,
    background: pd.DataFrame,
    donor_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Build public-background and DP-output features for donor matching."""

    distance = donor_config["distance"]
    if distance == "public_cox_risk":
        risk = donor_config["risk"]
        features = list(risk["features"])
        public_design = public_cox_design_matrix(background, features)
        output_design = public_cox_design_matrix(frame, features)
        onset_coefficients = np.asarray(
            [risk["onset_coefficients"][name] for name in features], dtype=float
        )
        death_coefficients = np.asarray(
            [risk["death_coefficients"][name] for name in features], dtype=float
        )
        public_raw = np.column_stack(
            [public_design @ onset_coefficients, public_design @ death_coefficients]
        )
        output_raw = np.column_stack(
            [output_design @ onset_coefficients, output_design @ death_coefficients]
        )
        public_mean = public_raw.mean(axis=0)
        public_scale = public_raw.std(axis=0, ddof=0)
        public_scale[public_scale == 0.0] = 1.0
        weights = np.asarray([1.0, float(risk["death_weight"])])
        return (
            ((public_raw - public_mean) / public_scale) * weights,
            ((output_raw - public_mean) / public_scale) * weights,
        )

    numeric_columns = list(donor_config["numeric_columns"])
    public_numeric = background[numeric_columns].to_numpy(dtype=float)
    output_numeric = frame[numeric_columns].to_numpy(dtype=float)
    public_mean = public_numeric.mean(axis=0)
    public_scale = public_numeric.std(axis=0, ddof=0)
    public_scale[public_scale == 0.0] = 1.0
    return (
        (public_numeric - public_mean) / public_scale,
        (output_numeric - public_mean) / public_scale,
    )


def replace_outcomes_with_nearest_public_donors(
    output: pd.DataFrame,
    background: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Copy a coherent public outcome from the nearest public covariate row.

    The configured feature space is derived only from public A_bg. The nearest
    row is selected using only public data and the already-DP synthetic
    covariates, and its coherent outcome triplet is copied as a unit.
    """

    donor_config = config["postprocess"].get("public_outcome_donor", {})
    if not bool(donor_config.get("enabled", False)):
        return

    replace_columns = list(donor_config["replace_columns"])
    if donor_config["distance"] == "public_cox_risk":
        risk_features = list(donor_config["risk"]["features"])
        feature_columns = [
            COX_LOG_FEATURE_SOURCES.get(name, name) for name in risk_features
        ]
        categorical_columns: list[str] = []
    else:
        feature_columns = list(donor_config["numeric_columns"])
        categorical_columns = list(donor_config["categorical_columns"])
    required = set(feature_columns + categorical_columns + replace_columns)
    for label, frame in (("synthetic output", output), ("public A_bg", background)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing outcome-donor columns: {sorted(missing)}")
    if background.empty:
        raise ValueError("public A_bg must contain at least one outcome donor row")

    public_z, output_z = outcome_donor_features(output, background, donor_config)

    public_categorical = {
        name: background[name].to_numpy() for name in categorical_columns
    }
    output_categorical = {
        name: output[name].to_numpy() for name in categorical_columns
    }
    donor_indices = np.empty(len(output), dtype=int)
    chunk_size = int(donor_config.get("chunk_size", 128))
    for start in range(0, len(output), chunk_size):
        stop = min(start + chunk_size, len(output))
        delta = output_z[start:stop, None, :] - public_z[None, :, :]
        distance_sq = np.einsum("ijk,ijk->ij", delta, delta)
        if donor_config["distance"] == "standardized_numeric_plus_onehot":
            for name in categorical_columns:
                mismatched = (
                    output_categorical[name][start:stop, None]
                    != public_categorical[name][None, :]
                )
                distance_sq += 2.0 * mismatched
        donor_indices[start:stop] = np.argmin(distance_sq, axis=1)

    output.loc[:, replace_columns] = background.iloc[donor_indices][
        replace_columns
    ].to_numpy()
    output["time"] = pd.to_numeric(output["time"], errors="raise").astype(float)
    output["onset"] = pd.to_numeric(output["onset"], errors="raise").astype(int)
    output["death"] = pd.to_numeric(output["death"], errors="raise").astype(int)

    if not np.isfinite(output["time"].to_numpy(dtype=float)).all():
        raise ValueError("public outcome donors produced a non-finite time")
    for name in ("onset", "death"):
        if not output[name].isin([0, 1]).all():
            raise ValueError(f"public outcome donors produced non-binary {name}")
    events = set(zip(output["onset"], output["death"]))
    if not events <= {(0, 0), (1, 0), (0, 1)}:
        raise ValueError("public outcome donors produced an invalid event combination")


def prepare_private_data(
    input_path: Path,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> pl.LazyFrame:
    source = pd.read_csv(input_path, encoding="utf-8")
    validate_private_input(source, config, schema)

    encoded: dict[str, Any] = {}
    outcome_state = encode_outcome_state(source, config)
    rare_flag = encode_rare_flag(source, schema)
    for name in config["model_columns"]:
        if name in config["numeric_bin_edges"]:
            encoded[name] = bin_indices(
                source[name].to_numpy(dtype=float),
                config["numeric_bin_edges"][name],
            )
        elif name == "outcome_state":
            encoded[name] = outcome_state
        elif name == "rare_flag":
            encoded[name] = rare_flag
        else:
            encoded[name] = source[name].to_numpy()

    return pl.from_pandas(pd.DataFrame(encoded)).lazy()


def load_public_background(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    background = pd.read_csv(path, encoding="utf-8")
    required = set(config["numeric_bin_edges"])
    public_numeric_columns = set(config["numeric_bin_edges"])
    public_categorical_columns: list[str] = []
    outcome_donor = config["postprocess"].get("public_outcome_donor", {})
    if outcome_donor.get("enabled", False):
        if outcome_donor["distance"] == "public_cox_risk":
            expected_hash = outcome_donor["risk"]["source_background_sha256"]
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash.lower():
                raise ValueError(
                    "public A_bg does not match the source used for the fixed "
                    "public Cox coefficients"
                )
            sources = {
                COX_LOG_FEATURE_SOURCES.get(name, name)
                for name in outcome_donor["risk"]["features"]
            }
            required.update(sources)
            public_numeric_columns.update(sources)
        else:
            required.update(outcome_donor["numeric_columns"])
            required.update(outcome_donor["categorical_columns"])
            public_numeric_columns.update(outcome_donor["numeric_columns"])
            public_categorical_columns = list(outcome_donor["categorical_columns"])
        required.update(outcome_donor["replace_columns"])
    missing = required - set(background.columns)
    if missing:
        raise ValueError(f"public background data is missing columns: {sorted(missing)}")
    for name in public_numeric_columns:
        background[name] = pd.to_numeric(background[name], errors="raise")
        if not np.isfinite(background[name].to_numpy(dtype=float)).all():
            raise ValueError(f"public background contains a non-finite value: {name}")
    if outcome_donor.get("enabled", False):
        for name in ("time", "onset", "death"):
            background[name] = pd.to_numeric(background[name], errors="raise")
            if not np.isfinite(background[name].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"public background contains a non-finite value: {name}"
                )
        for name in public_categorical_columns:
            if not background[name].isin(config["categorical_keys"][name]).all():
                raise ValueError(
                    f"public background contains a value outside the public key-set: {name}"
                )
        for name in ("sex", "smoking"):
            if name in public_numeric_columns and not background[name].isin(
                config["categorical_keys"][name]
            ).all():
                raise ValueError(
                    f"public background contains a value outside the public key-set: {name}"
                )
        for name in ("onset", "death"):
            if not background[name].isin([0, 1]).all():
                raise ValueError(f"public background {name} must be binary")
        events = set(zip(background["onset"].astype(int), background["death"].astype(int)))
        if not events <= {(0, 0), (1, 0), (0, 1)}:
            raise ValueError("public background contains an invalid event combination")
        time_floor = float(config["postprocess"]["time_floor"])
        time_horizon = float(config["postprocess"]["time_horizon"])
        if not background["time"].between(
            time_floor, time_horizon, inclusive="both"
        ).all():
            raise ValueError("public background time is outside the public range")
    return background


def model_keys(config: dict[str, Any]) -> dict[str, list[Any]]:
    keys = {name: list(values) for name, values in config["categorical_keys"].items()}
    for name, edges in config["numeric_bin_edges"].items():
        keys[name] = list(range(len(edges) - 1))
    return {name: keys[name] for name in config["model_columns"]}


def canonical_query(
    columns: list[str] | tuple[str, ...], model_order: dict[str, int]
) -> tuple[str, ...]:
    unique = set(columns)
    if len(unique) != len(columns):
        raise ConfigurationError(f"workload query contains duplicates: {columns}")
    missing = unique - set(model_order)
    if missing:
        raise ConfigurationError(f"unknown workload columns: {sorted(missing)}")
    return tuple(sorted(unique, key=model_order.__getitem__))


def build_workload(config: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    spec = config["workload"]
    model_order = {name: idx for idx, name in enumerate(config["model_columns"])}
    weights: dict[tuple[str, ...], float] = {}
    fixed_queries: set[tuple[str, ...]] = set()

    def add(
        columns: list[str] | tuple[str, ...], weight: float, *, fixed: bool = False
    ) -> None:
        query = canonical_query(columns, model_order)
        if not query:
            raise ConfigurationError("workload queries cannot be empty")
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ConfigurationError(f"workload weight must be in (0, 1]: {query}")
        weights[query] = max(weights.get(query, 0.0), weight)
        if fixed:
            fixed_queries.add(query)

    for query in spec["queries"]:
        add(
            query["columns"],
            float(query["weight"]),
            fixed=bool(query.get("fixed", False)),
        )
    for group in spec.get("pairwise_groups", []):
        for pair in combinations(group["columns"], 2):
            add(pair, float(group["weight"]))

    adaptive = [
        dp.mbi.Count(columns, weight=weight) for columns, weight in weights.items()
    ]
    fixed = [
        dp.mbi.Count(columns, weight=weights[columns])
        for columns in weights
        if columns in fixed_queries
    ]
    return adaptive, fixed


def run_aim(
    private_frame: pl.LazyFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, int, int]:
    dp.enable_features("contrib")
    rho = float(config["privacy"]["rho"])
    aim = config["aim"]
    workload, fixed_workload = build_workload(config)

    adaptive_algorithm = dp.mbi.AIM(
        queries=workload,
        measure_split=float(aim["measure_split"]),
        rounds=int(aim["rounds"]),
        max_size=float(aim["max_size_mb"]),
    )
    if fixed_workload:
        algorithm = dp.mbi.Sequential(
            algorithms=[
                dp.mbi.Fixed(queries=fixed_workload),
                adaptive_algorithm,
            ],
            weights=[
                float(aim["fixed_budget_weight"]),
                float(aim["adaptive_budget_weight"]),
            ],
            oneway=aim["oneway"],
            oneway_split=float(aim["oneway_split"]),
        )
    else:
        algorithm = dp.mbi.AIM(
            queries=workload,
            oneway=aim["oneway"],
            oneway_split=float(aim["oneway_split"]),
            measure_split=float(aim["measure_split"]),
            rounds=int(aim["rounds"]),
            max_size=float(aim["max_size_mb"]),
        )

    context = dp.Context.compositor(
        data=private_frame,
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(rho=rho),
    )
    table = (
        context.query(rho=rho)
        .select(*config["model_columns"])
        .contingency_table(
            keys=model_keys(config),
            algorithm=algorithm,
        )
        .release()
    )
    synthetic = table.synthesize(
        rows=int(config["output_rows"]),
        method=config["synthesis"]["method"],
    )
    return synthetic.to_pandas(), len(workload), len(fixed_workload)


def fill_from_dp_output(
    frame: pd.DataFrame,
    name: str,
    allowed: list[Any],
    public_default: Any,
) -> None:
    valid = frame[name].isin(allowed)
    valid_values = frame.loc[valid, name]
    modes = valid_values.mode(dropna=True)
    replacement = modes.iloc[0] if not modes.empty else public_default
    frame.loc[~valid, name] = replacement


def decode_numeric_column(
    bin_values: np.ndarray,
    public_values: np.ndarray,
    edges: list[float],
    rng: np.random.Generator,
    integral: bool,
) -> np.ndarray:
    public_bins = bin_indices(public_values.astype(float), edges)
    decoded = np.empty(len(bin_values), dtype=float)

    for bin_id in range(len(edges) - 1):
        output_mask = bin_values == bin_id
        count = int(output_mask.sum())
        if count == 0:
            continue
        candidates = public_values[public_bins == bin_id]
        if len(candidates):
            decoded[output_mask] = rng.choice(candidates, size=count, replace=True)
            continue

        lower, upper = float(edges[bin_id]), float(edges[bin_id + 1])
        if integral:
            lo_int, hi_int = math.ceil(lower), math.floor(upper)
            decoded[output_mask] = rng.integers(lo_int, hi_int + 1, size=count)
        else:
            decoded[output_mask] = rng.uniform(lower, upper, size=count)
    return decoded


def decode_outcome_state(
    states: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = config["postprocess"]["event_time_bins"]
    n_bins = len(bins)
    horizon = float(config["postprocess"]["time_horizon"])
    time = np.full(len(states), horizon, dtype=float)
    onset = np.zeros(len(states), dtype=int)
    death = np.zeros(len(states), dtype=int)

    for state in range(1, 1 + 2 * n_bins):
        mask = states == state
        count = int(mask.sum())
        if count == 0:
            continue
        if state <= n_bins:
            onset[mask] = 1
            bin_id = state - 1
        else:
            death[mask] = 1
            bin_id = state - 1 - n_bins
        lower, upper = (float(value) for value in bins[bin_id])
        time[mask] = rng.uniform(lower, upper, size=count)
    return time, onset, death


def postprocess_submission(
    synthetic: pd.DataFrame,
    background: pd.DataFrame,
    config: dict[str, Any],
    schema: dict[str, Any],
) -> pd.DataFrame:
    keys = model_keys(config)
    defaults = config["categorical_defaults"]
    for name in config["model_columns"]:
        if name in defaults:
            default = defaults[name]
        else:
            default = len(keys[name]) // 2
        fill_from_dp_output(synthetic, name, keys[name], default)

    for name in config["model_columns"]:
        if name != "prefecture":
            synthetic[name] = pd.to_numeric(synthetic[name], errors="raise").astype(int)
    synthetic["prefecture"] = synthetic["prefecture"].astype(str)

    rng = np.random.default_rng()
    decoded: dict[str, Any] = {
        "sex": synthetic["sex"].to_numpy(dtype=int),
        "prefecture": synthetic["prefecture"].to_numpy(dtype=str),
        "smoking": synthetic["smoking"].to_numpy(dtype=int),
    }
    for name, edges in config["numeric_bin_edges"].items():
        decoded[name] = decode_numeric_column(
            synthetic[name].to_numpy(dtype=int),
            background[name].to_numpy(dtype=float),
            edges,
            rng,
            integral=name == "age",
        )

    states = synthetic["outcome_state"].to_numpy(dtype=int)
    time, onset, death = decode_outcome_state(states, config, rng)
    decoded["time"] = time
    decoded["onset"] = onset
    decoded["death"] = death

    output = pd.DataFrame(decoded)
    age_lo, age_hi = schema["age_range"]
    output["age"] = np.floor(
        np.clip(output["age"].to_numpy(dtype=float), age_lo, age_hi)
    ).astype(int)
    for name, (lo, hi) in schema["ranges"].items():
        output[name] = np.clip(output[name].to_numpy(dtype=float), lo, hi)

    time_floor = float(config["postprocess"]["time_floor"])
    time_horizon = float(config["postprocess"]["time_horizon"])
    output["time"] = np.clip(output["time"].to_numpy(dtype=float), time_floor, time_horizon)

    replace_detected_rare_with_public_donors(
        output,
        background,
        config,
        schema,
        rng,
    )
    output["age"] = np.floor(
        np.clip(output["age"].to_numpy(dtype=float), age_lo, age_hi)
    ).astype(int)
    for name, (lo, hi) in schema["ranges"].items():
        output[name] = np.clip(output[name].to_numpy(dtype=float), lo, hi)

    replace_outcomes_with_nearest_public_donors(output, background, config)
    output["time"] = np.clip(
        output["time"].to_numpy(dtype=float), time_floor, time_horizon
    )

    output = output[schema["submitted_c_columns"]].reset_index(drop=True)
    if len(output) != int(config["output_rows"]):
        raise ValueError("AIM output row count changed during postprocessing")
    if output.isna().any().any():
        raise ValueError("AIM output contains missing values after postprocessing")
    return output


def make_privacy_report(
    config: dict[str, Any],
    config_path: Path,
    background_path: Path,
    schema_path: Path,
    output_path: Path,
    workload_count: int,
    fixed_query_count: int,
) -> dict[str, Any]:
    rho = float(config["privacy"]["rho"])
    conversions = []
    for raw_delta in config["privacy"]["epsilon_report_deltas"]:
        delta = float(raw_delta)
        epsilon = rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
        conversions.append({"epsilon": epsilon, "delta": delta})

    notes = [
        "record_id is excluded before the DP release",
        "all key-sets and numeric bin edges are fixed by public inputs",
        "rare_flag is derived row-wise from the public rare detector and is not released",
        "numeric decoding uses only the public A_bg distribution",
        "sampling, decoding, clipping, and schema formatting are postprocessing",
        "this report includes no statistics computed from the private cohort",
    ]
    rare_donor = config["postprocess"].get("public_rare_donor", {})
    if rare_donor.get("enabled", False):
        if rare_donor.get("target") == "public_background_rate":
            notes.append(
                "the rare-row target count is the public A_bg rare rate times the "
                "public output size, and target rows are chosen only from the DP output"
            )
        else:
            notes.append("rare targets are detected only in the DP synthetic output")
        notes.append(
            "rare-marker values are copied together from one public A_bg donor row"
        )
        if rare_donor.get("seed") is not None:
            notes.append(
                "the fixed donor seed is used only after the randomized DP release"
            )
    if config["postprocess"].get("public_outcome_donor", {}).get("enabled", False):
        outcome_donor = config["postprocess"]["public_outcome_donor"]
        if outcome_donor.get("distance") == "public_cox_risk":
            notes.append(
                "time, onset, and death are copied together from the nearest "
                "public A_bg row in a fixed A_bg-derived Cox risk space"
            )
        else:
            notes.append(
                "time, onset, and death are copied together from the nearest public "
                "A_bg row using only DP synthetic covariates"
            )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": "OpenDP AIM with Private-PGM postprocessing",
        "config_version": config["config_version"],
        "config_sha256": sha256_file(config_path),
        "public_background_sha256": sha256_file(background_path),
        "public_schema_sha256": sha256_file(schema_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "output_sha256": sha256_file(output_path),
        "privacy": {
            "rho_zcdp": rho,
            "privacy_unit": config["privacy"]["privacy_unit"],
            "adjacency": config["privacy"]["adjacency"],
            "approx_dp_conversions": conversions,
        },
        "dp_release_count": 1,
        "output_rows_public": int(config["output_rows"]),
        "workload_query_count": workload_count,
        "fixed_query_count": fixed_query_count,
        "versions": {
            "opendp": version("opendp"),
            "mbi": version("mbi"),
            "pandas": version("pandas"),
            "polars": version("polars"),
        },
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    background_path = (
        Path(args.background).resolve()
        if args.background
        else input_path.parent / "A_bg.csv"
    )
    schema_path = Path(args.schema).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.privacy_report).resolve()

    schema = load_json(schema_path)
    config = load_json(config_path)
    validate_public_configuration(config, schema)
    background = load_public_background(background_path, config)
    donor_config = config["postprocess"].get("public_rare_donor", {})
    if donor_config.get("enabled", False) and not encode_rare_flag(background, schema).any():
        raise ValueError("public A_bg contains no rows matching the public rare detector")
    private_frame = prepare_private_data(input_path, config, schema)

    if args.validate_only:
        print("Preflight validation passed; no DP release was performed.")
        print(f"  input:      {input_path}")
        print(f"  background: {background_path}")
        print(f"  schema:     {schema_path}")
        print(f"  config:     {config_path}")
        return

    ledger_path = Path(args.release_ledger).resolve() if args.release_ledger else None
    if ledger_path is not None:
        for destination in (output_path, report_path):
            if destination.exists():
                raise FileExistsError(
                    "refusing to overwrite an existing release artifact: "
                    f"{destination}"
                )
        initialize_release_ledger(
            ledger_path,
            input_path=input_path,
            background_path=background_path,
            schema_path=schema_path,
            config_path=config_path,
            output_path=output_path,
            report_path=report_path,
            config=config,
        )

    print(
        "Running one OpenDP AIM release "
        f"(rho={config['privacy']['rho']}, rows={config['output_rows']})...",
        flush=True,
    )
    synthetic, workload_count, fixed_query_count = run_aim(private_frame, config)
    if ledger_path is not None:
        update_release_ledger(
            ledger_path,
            "release_completed",
            release_completed_at_utc=datetime.now(timezone.utc).isoformat(),
            workload_query_count=workload_count,
            fixed_query_count=fixed_query_count,
        )
    output = postprocess_submission(synthetic, background, config, schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    report = make_privacy_report(
        config,
        config_path,
        background_path,
        schema_path,
        output_path,
        workload_count,
        fixed_query_count,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if ledger_path is not None:
        update_release_ledger(
            ledger_path,
            "complete",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256=sha256_file(output_path),
            privacy_report_sha256=sha256_file(report_path),
        )
    print(f"Wrote C.csv: {output_path} ({len(output)} rows x {len(output.columns)} columns)")
    print(f"Wrote privacy report: {report_path}")


if __name__ == "__main__":
    main()
