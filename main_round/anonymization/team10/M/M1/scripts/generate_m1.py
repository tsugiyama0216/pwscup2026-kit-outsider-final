"""Repair S1 dependence with a public-reference rank copula.

The rank rearrangement preserves every S1 univariate marginal exactly.  It uses
a fixed sample of public A_bg rows to rearrange the nine scoring covariates,
then reapplies S1's public rare-row and public outcome donor postprocessing.
The rare-row repair intentionally changes a small set of marker values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "smoking"]
LOG_SOURCES = {"log_TG": "TG", "log_ALT": "ALT"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return (ranks + 0.5) / len(values)


def rearrange_marginal(
    original: np.ndarray,
    reference: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Jitter resolves ties in binary variables without changing their values.
    old_key = rank01(original.astype(float) + rng.normal(0.0, 1e-9, len(original)))
    ref_key = rank01(reference.astype(float) + rng.normal(0.0, 1e-9, len(reference)))
    key = (1.0 - strength) * old_key + strength * ref_key
    destination = np.argsort(key, kind="stable")
    result = np.empty(len(original), dtype=original.dtype)
    result[destination] = np.sort(original, kind="stable")
    return result


def assign_prefecture(
    original: pd.Series,
    reference: pd.Series,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preserve S1 counts while matching public categories whenever possible."""
    result = np.empty(len(original), dtype=object)
    result[:] = None
    remaining_rows = set(range(len(original)))
    counts = original.value_counts().to_dict()
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        preferred = [i for i in remaining_rows if reference.iloc[i] == category]
        rng.shuffle(preferred)
        chosen = preferred[:count]
        if len(chosen) < count:
            fallback = list(remaining_rows.difference(chosen))
            rng.shuffle(fallback)
            chosen.extend(fallback[: count - len(chosen)])
        for index in chosen:
            result[index] = category
            remaining_rows.remove(index)
    if remaining_rows or any(value is None for value in result):
        raise RuntimeError("prefecture assignment did not consume every row")
    return result


def rare_distances(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    detector = schema["rare_detector"]
    values = frame[detector["columns"]].to_numpy(dtype=float)
    diff = values - np.asarray(detector["mu"], dtype=float)
    inv_sigma = np.linalg.pinv(np.asarray(detector["sigma"], dtype=float))
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", diff, inv_sigma, diff), 0.0, None))


def repair_rare_rows(
    output: pd.DataFrame,
    background: pd.DataFrame,
    config: dict,
    schema: dict,
) -> int:
    donor = config["postprocess"]["public_rare_donor"]
    columns = list(schema["rare_detector"]["columns"])
    threshold = float(schema["rare_detector"]["band"]["lo"])
    output_distance = rare_distances(output, schema)
    public_distance = rare_distances(background, schema)
    public_rare = public_distance >= threshold
    target_count = int(round(float(public_rare.mean()) * len(output)))
    targets = np.argsort(-output_distance, kind="stable")[:target_count]
    target_mask = np.zeros(len(output), dtype=bool)
    target_mask[targets] = True

    excess = np.flatnonzero((output_distance >= threshold) & ~target_mask)
    if len(excess):
        nonrare = np.flatnonzero(public_distance < threshold)
        nearest = nonrare[np.argsort(-public_distance[nonrare], kind="stable")]
        output.loc[excess, columns] = background.iloc[np.resize(nearest, len(excess))][columns].to_numpy()

    maximum = donor.get("maximum_distance_exclusive")
    if maximum is not None:
        public_rare &= public_distance < float(maximum)
    donors = background.loc[public_rare, columns].reset_index(drop=True)
    donor_rng = np.random.default_rng(int(donor["seed"]))
    selected = np.resize(donor_rng.permutation(len(donors)), target_count)
    output.loc[targets, columns] = donors.iloc[selected].to_numpy()
    final_count = int((rare_distances(output, schema) >= threshold).sum())
    if final_count != target_count:
        raise RuntimeError(f"rare repair expected {target_count}, got {final_count}")
    return final_count


def cox_design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    columns = []
    for feature in features:
        source = LOG_SOURCES.get(feature, feature)
        values = frame[source].to_numpy(dtype=float)
        if feature in LOG_SOURCES:
            values = np.log(np.clip(values, 1e-6, None))
        columns.append(values)
    return np.column_stack(columns)


def repair_outcomes(
    output: pd.DataFrame,
    background: pd.DataFrame,
    config: dict,
    death_weight: float | None = None,
) -> None:
    donor = config["postprocess"]["public_outcome_donor"]
    risk = donor["risk"]
    features = list(risk["features"])
    public_x = cox_design(background, features)
    output_x = cox_design(output, features)
    onset_beta = np.asarray([risk["onset_coefficients"][name] for name in features])
    death_beta = np.asarray([risk["death_coefficients"][name] for name in features])
    public_raw = np.column_stack((public_x @ onset_beta, public_x @ death_beta))
    output_raw = np.column_stack((output_x @ onset_beta, output_x @ death_beta))
    mean = public_raw.mean(axis=0)
    scale = public_raw.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    effective_death_weight = (
        float(risk["death_weight"]) if death_weight is None else float(death_weight)
    )
    weights = np.asarray([1.0, effective_death_weight])
    public_z = ((public_raw - mean) / scale) * weights
    output_z = ((output_raw - mean) / scale) * weights
    selected = np.empty(len(output), dtype=int)
    chunk = int(donor.get("chunk_size", 128))
    for start in range(0, len(output), chunk):
        stop = min(start + chunk, len(output))
        delta = output_z[start:stop, None, :] - public_z[None, :, :]
        selected[start:stop] = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
    output.loc[:, ["time", "onset", "death"]] = background.iloc[selected][
        ["time", "onset", "death"]
    ].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", required=True, type=Path)
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--strength", required=True, type=float)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--death-weight", type=float)
    parser.add_argument(
        "--outcome-method",
        choices=("nearest-risk", "sampled-public-row", "preserve-s1"),
        default="nearest-risk",
    )
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    source = pd.read_csv(args.s1)
    background = pd.read_csv(args.background)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if len(source) > len(background):
        raise ValueError("public background is smaller than S1")
    rng = np.random.default_rng(args.seed)
    sampled = background.iloc[rng.choice(len(background), len(source), replace=False)].reset_index(drop=True)
    output = source.copy()
    for column in FEATURES:
        output[column] = rearrange_marginal(
            source[column].to_numpy(), sampled[column].to_numpy(), args.strength, rng
        )
    output["prefecture"] = assign_prefecture(source["prefecture"], sampled["prefecture"], rng)
    rare_count = repair_rare_rows(output, background, config, schema)
    if args.outcome_method == "nearest-risk":
        repair_outcomes(output, background, config, args.death_weight)
    elif args.outcome_method == "sampled-public-row":
        output.loc[:, ["time", "onset", "death"]] = sampled[
            ["time", "onset", "death"]
        ].to_numpy()
    output = output[schema["submitted_c_columns"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    report = {
        "method": (
            "public-reference rank-copula rearrangement of frozen S1 marginals, "
            "followed by S1 public rare-row and outcome-donor repairs"
        ),
        "strength": args.strength,
        "seed": args.seed,
        "death_weight": (
            float(config["postprocess"]["public_outcome_donor"]["risk"]["death_weight"])
            if args.death_weight is None
            else args.death_weight
        ),
        "outcome_method": args.outcome_method,
        "rows": len(output),
        "rare_detected": rare_count,
        "inputs": {
            "s1_sha256": sha256(args.s1),
            "background_sha256": sha256(args.background),
            "schema_sha256": sha256(args.schema),
            "config_sha256": sha256(args.config),
        },
        "output_sha256": sha256(args.output),
        "privacy_note": (
            "Uses only frozen S1 and public A_bg/schema/config. It is postprocessing "
            "of S1 and does not access B or B_self."
        ),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
