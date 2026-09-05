"""Inventory public C transformation and event-link mechanisms.

This is research-only code for the authorized PWSCUP2026 attack phase.  It
reads public contest files and writes diagnostics under local-output.  It never
reads a token, edits a candidate, builds an archive, or submits anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STARTER = ROOT / "starter" / "src"
for source_root in (ROOT, STARTER):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from attack_phase_v5.aia_structured_ensemble import (  # noqa: E402
    QI_COLUMNS,
    RELEASE_COLUMNS,
    copula_event_links,
    filter_linkable_releases,
)


TARGETS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 26, 29)
NUMERIC_STATIC = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
STATIC = (
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
)
VISIBLE = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
OUTCOME = ("time", "onset", "death")


def hashes(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(frame.loc[:, list(columns)], index=False).to_numpy(
        dtype=np.uint64
    )


def exact_row_count(frame: pd.DataFrame, reference: pd.DataFrame, columns: tuple[str, ...]) -> int:
    ref = set(hashes(reference, columns).tolist())
    return int(np.fromiter((int(v) in ref for v in hashes(frame, columns)), dtype=bool).sum())


def exact_value_fraction(values: pd.Series, reference: pd.Series) -> float:
    pool = set(reference.dropna().tolist())
    return float(values.map(lambda value: value in pool).mean())


def rounding_digits(values: np.ndarray, threshold: float = 0.95) -> int | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    for digits in range(0, 9):
        tolerance = max(1e-10, 10.0 ** (-digits - 7))
        if float(np.mean(np.abs(finite - np.round(finite, digits)) <= tolerance)) >= threshold:
            return digits
    return None


def affine_quantile_fit(source: np.ndarray, released: np.ndarray) -> dict[str, float]:
    probabilities = np.linspace(0.02, 0.98, 97)
    x = np.quantile(np.asarray(source, dtype=float), probabilities)
    y = np.quantile(np.asarray(released, dtype=float), probabilities)
    design = np.column_stack((np.ones(len(x)), x))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coef
    scale = max(float(np.quantile(y, 0.75) - np.quantile(y, 0.25)), 1e-9)
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "nrmse_iqr": float(np.sqrt(np.mean((y - fitted) ** 2)) / scale),
        "max_abs_iqr": float(np.max(np.abs(y - fitted)) / scale),
    }


def nearest_source_column(
    frame: pd.DataFrame, references: dict[str, pd.DataFrame], column: str
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    released = frame[column].to_numpy(dtype=float)
    for source_name, reference in references.items():
        for source_column in NUMERIC_STATIC:
            source = reference[source_column].to_numpy(dtype=float)
            pooled = np.concatenate((released, source))
            scale = max(float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25)), 1e-9)
            raw_w1 = float(
                np.mean(
                    np.abs(
                        np.quantile(released, np.linspace(0.01, 0.99, 99))
                        - np.quantile(source, np.linspace(0.01, 0.99, 99))
                    )
                )
                / scale
            )
            candidates.append(
                {
                    "source": source_name,
                    "source_column": source_column,
                    "raw_quantile_l1_iqr": raw_w1,
                }
            )
    return min(candidates, key=lambda item: item["raw_quantile_l1_iqr"])


def background_control_links(
    released: pd.DataFrame,
    background: pd.DataFrame,
    *,
    seed: int,
    margin_quantile: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(background), size=326, replace=False)
    query = background.iloc[rows].loc[:, list(VISIBLE)].reset_index(drop=True)
    query.insert(0, "challenge_row_id", np.arange(len(query), dtype=int))
    truth = background.iloc[rows]["time"].to_numpy(dtype=float)
    linked, gate, details = copula_event_links(
        released, query, margin_quantile=margin_quantile, horizon_time=10.0
    )
    hit = gate & (np.abs(linked - truth) <= 0.5)
    accepted = int(gate.sum())
    return {
        **details,
        "accepted": accepted,
        "false_successes": int(hit.sum()),
        "false_success_rate_all": float(hit.mean()),
        "precision_given_accepted": float(hit.sum() / accepted) if accepted else 0.0,
    }


def main() -> None:
    dist = ROOT / "participant_data"
    targets_dir = ROOT / "team-shared" / "prelim" / "attack-targets"
    references = {
        "A_bg": pd.read_csv(dist / "A_bg.csv"),
        "B_practice": pd.read_csv(dist / "B_practice.csv"),
    }
    releases = {
        str(target): pd.read_csv(targets_dir / f"C_{target}.csv") for target in TARGETS
    }
    filtered, filter_details = filter_linkable_releases(releases)
    rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}

    for target in TARGETS:
        target_id = str(target)
        frame = releases[target_id]
        challenge = pd.read_csv(targets_dir / f"aia_challenge_{target}.csv")
        event = (frame["onset"].to_numpy(dtype=int) + frame["death"].to_numpy(dtype=int)) > 0
        event_times = frame.loc[event, "time"].to_numpy(dtype=float)
        day_error = np.abs(frame["time"].to_numpy(dtype=float) * 365.25 - np.round(frame["time"].to_numpy(dtype=float) * 365.25))
        base = {
            "target": target,
            "rows": int(len(frame)),
            "events": int(event.sum()),
            "event_rate": float(event.mean()),
            "event_time_median": float(np.median(event_times)) if len(event_times) else np.nan,
            "time_unique": int(frame["time"].nunique()),
            "time_day_grid_fraction": float(np.mean(day_error <= 1e-7)),
            "time_rounding_digits_95": rounding_digits(frame["time"].to_numpy(dtype=float)),
            "outcome_inconsistent": int(np.sum(event & (frame["time"].to_numpy(dtype=float) >= 10.0))),
            "duplicate_full_rows": int(frame.duplicated(list(RELEASE_COLUMNS)).sum()),
            "exact_full_A_bg": exact_row_count(frame, references["A_bg"], RELEASE_COLUMNS),
            "exact_static_A_bg": exact_row_count(frame, references["A_bg"], STATIC),
            "exact_visible_A_bg": exact_row_count(frame, references["A_bg"], VISIBLE),
            "exact_full_B_practice": exact_row_count(frame, references["B_practice"], RELEASE_COLUMNS),
            "exact_static_B_practice": exact_row_count(frame, references["B_practice"], STATIC),
            "exact_visible_B_practice": exact_row_count(frame, references["B_practice"], VISIBLE),
            "challenge_exact_visible_C": exact_row_count(challenge, frame, VISIBLE),
            "linkable_rows": int(len(filtered[target_id])),
        }
        link_results: dict[str, Any] = {}
        for q in (0.25, 0.50, 0.75, 0.90):
            _, gate, info = copula_event_links(
                filtered[target_id], challenge, margin_quantile=q, horizon_time=10.0
            )
            control = background_control_links(
                filtered[target_id], references["A_bg"], seed=910_000 + target, margin_quantile=q
            )
            label = f"q{int(q * 100):02d}"
            base[f"challenge_links_{label}"] = int(gate.sum())
            base[f"background_false_success_all_{label}"] = control["false_success_rate_all"]
            link_results[label] = {"challenge": info, "background_control": control}
        rows.append(base)

        column_details: dict[str, Any] = {}
        for column in NUMERIC_STATIC:
            values = frame[column].to_numpy(dtype=float)
            item: dict[str, Any] = {
                "target": target,
                "column": column,
                "unique": int(frame[column].nunique()),
                "rounding_digits_95": rounding_digits(values),
            }
            for source_name, reference in references.items():
                item[f"exact_value_fraction_{source_name}"] = exact_value_fraction(
                    frame[column], reference[column]
                )
                item[f"ks_{source_name}"] = float(
                    ks_2samp(values, reference[column].to_numpy(dtype=float)).statistic
                )
                fit = affine_quantile_fit(reference[column].to_numpy(dtype=float), values)
                for key, value in fit.items():
                    item[f"affine_{source_name}_{key}"] = value
            item.update(
                {
                    f"nearest_distribution_{key}": value
                    for key, value in nearest_source_column(frame, references, column).items()
                }
            )
            column_rows.append(item)
            column_details[column] = item

        corr_columns = list(NUMERIC_STATIC)
        corr = frame[corr_columns].corr(method="spearman").to_numpy(dtype=float)
        corr_bg = references["A_bg"][corr_columns].corr(method="spearman").to_numpy(dtype=float)
        corr_b = references["B_practice"][corr_columns].corr(method="spearman").to_numpy(dtype=float)
        detailed[target_id] = {
            "summary": base,
            "columns": column_details,
            "spearman_frobenius_to_A_bg": float(np.linalg.norm(corr - corr_bg)),
            "spearman_frobenius_to_B_practice": float(np.linalg.norm(corr - corr_b)),
            "links": link_results,
            "cross_target_filter": filter_details[target_id],
        }

    summary = pd.DataFrame(rows)
    columns = pd.DataFrame(column_rows)
    summary.to_csv(HERE / "mechanism_inventory.csv", index=False)
    columns.to_csv(HERE / "column_inventory.csv", index=False)
    report = {
        "scope": "public C mechanism inventory and A_bg false-link control",
        "targets": list(TARGETS),
        "target_details": detailed,
        "network_access": False,
        "token_read": False,
        "production_candidate_changed": False,
        "archive_created": False,
        "submission_performed": False,
    }
    (HERE / "mechanism_inventory.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    display = summary[
        [
            "target",
            "events",
            "event_rate",
            "exact_full_A_bg",
            "exact_full_B_practice",
            "challenge_exact_visible_C",
            "challenge_links_q50",
            "background_false_success_all_q50",
        ]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
