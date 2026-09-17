"""Build an AIM config using only public PWS Cup inputs.

The mechanism design and default privacy budget are inherited from a reviewed
base config.  An explicit positive finite rho-zCDP value may override that
default.  Data-world-specific bin edges, categorical defaults, and public Cox
coefficients are recomputed from ``A_bg.csv`` and ``schema.json`` only.
No private cohort file is accepted by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning


QUANTILES = (0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
NUMERIC_COLUMNS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
MODE_COLUMNS = ("sex", "prefecture", "smoking")
COX_FEATURES = (
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
COX_LOG_SOURCES = {"log_TG": "TG", "log_ALT": "ALT"}


def positive_finite_rho(value: str | float) -> float:
    try:
        rho = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rho must be a number") from exc
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be positive and finite")
    return rho


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a public-data-only AIM config for a PWS Cup data world"
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-version", required=True)
    parser.add_argument(
        "--rho",
        type=positive_finite_rho,
        help=(
            "Explicit total rho-zCDP budget; defaults to privacy.rho in "
            "--base-config"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc


def public_bounds(schema: dict[str, Any], name: str) -> tuple[float, float]:
    bounds = schema["age_range"] if name == "age" else schema["ranges"][name]
    return float(bounds[0]), float(bounds[1])


def as_json_number(value: float, *, integral: bool = False) -> int | float:
    if integral:
        return int(round(float(value)))
    rounded = round(float(value), 2)
    return int(rounded) if rounded.is_integer() else rounded


def public_quantile_edges(
    background: pd.DataFrame, schema: dict[str, Any], name: str
) -> list[int | float]:
    lo, hi = public_bounds(schema, name)
    internal = background[name].quantile(
        list(QUANTILES), interpolation="linear"
    ).tolist()
    candidates = [lo, *internal, hi]
    edges: list[int | float] = []
    for candidate in candidates:
        value = as_json_number(candidate, integral=name == "age")
        if not edges or float(value) > float(edges[-1]):
            edges.append(value)
    if len(edges) < 2 or not math.isclose(float(edges[0]), lo) or not math.isclose(
        float(edges[-1]), hi
    ):
        raise ValueError(f"could not build public-range bin edges for {name}")
    return edges


def public_mode(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if modes.empty:
        raise ValueError(f"cannot determine a public mode for {series.name}")
    value = modes.iloc[0]
    return value.item() if isinstance(value, np.generic) else value


def cox_design(background: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=background.index)
    for feature in COX_FEATURES:
        source = COX_LOG_SOURCES.get(feature, feature)
        values = pd.to_numeric(background[source], errors="raise").to_numpy(dtype=float)
        if feature in COX_LOG_SOURCES:
            values = np.log(np.clip(values, 1e-6, None))
        design[feature] = values
    return design


def fit_public_cox_coefficients(
    background: pd.DataFrame, event: str
) -> dict[str, float]:
    fit_frame = cox_design(background)
    fit_frame["time"] = pd.to_numeric(background["time"], errors="raise")
    fit_frame[event] = pd.to_numeric(background[event], errors="raise")
    model = CoxPHFitter(penalizer=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(fit_frame, duration_col="time", event_col=event)
    coefficients = {name: float(model.params_[name]) for name in COX_FEATURES}
    if not all(math.isfinite(value) for value in coefficients.values()):
        raise ValueError(f"public {event} Cox fit produced a non-finite coefficient")
    standard_errors = model.standard_errors_.reindex(COX_FEATURES).to_numpy(dtype=float)
    if not np.isfinite(standard_errors).all() or not (standard_errors > 0.0).all():
        raise ValueError(f"public {event} Cox fit produced an invalid standard error")
    return coefficients


def validate_public_background(
    background: pd.DataFrame, schema: dict[str, Any]
) -> None:
    required = set(schema["distributed_columns"])
    missing = sorted(required - set(background.columns))
    if missing:
        raise ValueError(f"public A_bg.csv is missing columns: {missing}")
    if background.empty:
        raise ValueError("public A_bg.csv is empty")
    if background[list(required)].isna().any().any():
        raise ValueError("public A_bg.csv contains missing values")

    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(background[name], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"public A_bg.csv contains non-finite values: {name}")
        lo, hi = public_bounds(schema, name)
        if not ((values >= lo) & (values <= hi)).all():
            raise ValueError(f"public A_bg.csv is outside the public range: {name}")

    time = pd.to_numeric(background["time"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(time).all() or not ((time > 0.0) & (time <= 10.0)).all():
        raise ValueError("public A_bg.csv contains an invalid time value")
    for name in ("sex", "smoking", "onset", "death"):
        if not background[name].isin([0, 1]).all():
            raise ValueError(f"public A_bg.csv contains a non-binary value: {name}")
    event_pairs = set(
        zip(background["onset"].astype(int), background["death"].astype(int))
    )
    if not event_pairs <= {(0, 0), (1, 0), (0, 1)}:
        raise ValueError("public A_bg.csv contains an invalid onset/death pair")


def build_config(
    base_config: dict[str, Any],
    background: pd.DataFrame,
    schema: dict[str, Any],
    *,
    config_version: str,
    base_config_path: Path,
    background_path: Path,
    schema_path: Path,
    rho: float | None = None,
) -> dict[str, Any]:
    validate_public_background(background, schema)
    config = deepcopy(base_config)
    config["config_version"] = config_version
    config["privacy"]["rho"] = positive_finite_rho(
        config["privacy"]["rho"] if rho is None else rho
    )
    config["numeric_bin_edges"] = {
        name: public_quantile_edges(background, schema, name)
        for name in NUMERIC_COLUMNS
    }
    for name in MODE_COLUMNS:
        config["categorical_defaults"][name] = public_mode(background[name])

    rare_columns = list(schema["rare_detector"]["columns"])
    config["postprocess"]["public_rare_donor"]["columns"] = rare_columns
    risk = config["postprocess"]["public_outcome_donor"]["risk"]
    risk["source_background_sha256"] = sha256_file(background_path)
    risk["coefficient_fit"] = (
        "CoxPHFitter(penalizer=0.0) on public A_bg.csv; "
        "cause-specific onset/death fits"
    )
    risk["features"] = list(COX_FEATURES)
    risk["onset_coefficients"] = fit_public_cox_coefficients(background, "onset")
    risk["death_coefficients"] = fit_public_cox_coefficients(background, "death")

    config["public_config_provenance"] = {
        "builder": "opendp_aim/scripts/build_public_config.py",
        "base_config_sha256": sha256_file(base_config_path),
        "public_background_sha256": sha256_file(background_path),
        "public_schema_sha256": sha256_file(schema_path),
        "numeric_quantiles": list(QUANTILES),
        "private_inputs_used": False,
    }
    return config


def write_new_or_verify_identical(path: Path, config: dict[str, Any]) -> str:
    serialized = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"refusing to overwrite a different public config: {path}"
            )
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return "created"


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config).resolve()
    background_path = Path(args.background).resolve()
    schema_path = Path(args.schema).resolve()
    output_path = Path(args.output).resolve()

    base_config = load_json(base_path)
    schema = load_json(schema_path)
    background = pd.read_csv(background_path, encoding="utf-8")
    config = build_config(
        base_config,
        background,
        schema,
        config_version=args.config_version,
        base_config_path=base_path,
        background_path=background_path,
        schema_path=schema_path,
        rho=args.rho,
    )
    disposition = write_new_or_verify_identical(output_path, config)
    print(f"Public AIM config {disposition}: {output_path}")
    print(f"  background_sha256: {sha256_file(background_path)}")
    print(f"  schema_sha256:     {sha256_file(schema_path)}")
    print("  private inputs used: false")


if __name__ == "__main__":
    main()
