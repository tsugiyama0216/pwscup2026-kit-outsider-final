from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from pwscup2026.kit.selfscore import load_local_reference, self_score
from pwscup2026.scoring.utility import _score_u_spec

ROOT = Path(__file__).resolve().parents[2]
OUTCOME_COLUMNS = ["time", "onset", "death"]
LOG_FEATURE_SOURCES = {"log_TG": "TG", "log_ALT": "ALT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "現行v4の共変量を固定し、公開A_bgのCox-risk転帰ドナー設定を探索する。"
            "B_practiceによる順位付けは大会向けローカル診断であり、DP保証の一部ではない。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "local-output/opendp_aim/C.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "opendp_aim/config/pwscup_aim_v4.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "participant_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "local-output/opendp_aim/experiments/v5_uspec_current_v4",
    )
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def public_cox_design_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    columns: list[np.ndarray] = []
    for feature in features:
        source = LOG_FEATURE_SOURCES.get(feature, feature)
        values = frame[source].to_numpy(dtype=float)
        if feature in LOG_FEATURE_SOURCES:
            values = np.log(np.clip(values, 1e-6, None))
        columns.append(values)
    return np.column_stack(columns)


def outcome_donor_features(
    frame: pd.DataFrame,
    background: pd.DataFrame,
    donor_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
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
    center = public_raw.mean(axis=0)
    scale = public_raw.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    weights = np.asarray([1.0, float(risk["death_weight"])])
    return (
        ((public_raw - center) / scale) * weights,
        ((output_raw - center) / scale) * weights,
    )


def make_candidate(
    base: pd.DataFrame,
    background: pd.DataFrame,
    donor_config: dict,
    death_weight: float,
    k: int,
    seed: int,
) -> pd.DataFrame:
    candidate = base.copy()
    configured = json.loads(json.dumps(donor_config))
    configured["risk"]["death_weight"] = death_weight
    public_features, output_features = outcome_donor_features(
        candidate, background, configured
    )
    model = NearestNeighbors(metric="euclidean").fit(public_features)
    _, neighbors = model.kneighbors(output_features, n_neighbors=k)
    if k == 1:
        donor_indices = neighbors[:, 0]
    else:
        rng = np.random.default_rng(seed)
        donor_indices = neighbors[
            np.arange(len(candidate)), rng.integers(0, k, size=len(candidate))
        ]
    candidate.loc[:, OUTCOME_COLUMNS] = background.iloc[donor_indices][
        OUTCOME_COLUMNS
    ].to_numpy()
    candidate["time"] = candidate["time"].astype(float)
    candidate["onset"] = candidate["onset"].astype(int)
    candidate["death"] = candidate["death"].astype(int)
    return candidate


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.input)
    background_path = args.data_dir / "A_bg.csv"
    background = pd.read_csv(background_path)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    donor_config = config["postprocess"]["public_outcome_donor"]
    b, reference, score_config, _ = load_local_reference(args.data_dir)

    death_weights = (
        0.0,
        0.1,
        0.2,
        0.25,
        0.28,
        0.29,
        0.296,
        0.30,
        0.31,
        0.32,
        0.35,
        0.4,
        0.5,
        1.0,
    )
    neighbor_counts = (1, 2, 3, 5, 8, 15)
    seeds = range(5)

    screened: list[dict] = []
    candidates: dict[str, pd.DataFrame] = {}
    for death_weight in death_weights:
        for k in neighbor_counts:
            current_seeds = (0,) if k == 1 else seeds
            for seed in current_seeds:
                name = f"dw{death_weight:g}_k{k}_s{seed}"
                candidate = make_candidate(
                    base,
                    background,
                    donor_config,
                    death_weight,
                    k,
                    seed,
                )
                score, detail = _score_u_spec(candidate, b)
                screened.append(
                    {
                        "candidate": name,
                        "death_weight": death_weight,
                        "neighbors": k,
                        "seed": seed,
                        "U_spec": score,
                        "io_part": detail.get("io_part"),
                        "tstr_cal": detail.get("tstr_cal"),
                        "onset_count": int(candidate["onset"].sum()),
                        "death_count": int(candidate["death"].sum()),
                    }
                )
                candidates[name] = candidate

    screen = pd.DataFrame(screened).sort_values(
        ["U_spec", "tstr_cal"], ascending=False
    )
    screen.to_csv(args.output_dir / "screen.csv", index=False)

    full_rows: list[dict] = []
    full_details: dict[str, dict] = {}
    for row in screen.head(args.top).itertuples(index=False):
        candidate = candidates[row.candidate]
        scored = self_score(candidate, b, reference, score_config)
        spec = scored["detail"]["U_spec"]
        rare = scored["detail"]["U_rare"]
        full_rows.append(
            {
                "candidate": row.candidate,
                "death_weight": row.death_weight,
                "neighbors": row.neighbors,
                "seed": row.seed,
                "U": scored["utility"],
                **scored["facets"],
                "io_part": spec.get("io_part"),
                "tstr_cal": spec.get("tstr_cal"),
                "rare_detected_C": rare.get("rare_detected_C"),
                "rare_onset_cal": rare.get("onset_cal"),
            }
        )
        full_details[row.candidate] = scored
        candidate.to_csv(args.output_dir / f"{row.candidate}.csv", index=False)

    full = pd.DataFrame(full_rows).sort_values(["U", "U_spec"], ascending=False)
    full.to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "scores.json").write_text(
        json.dumps(full_details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "purpose": "practice-only U_spec tuning on the fixed v4 covariates",
        "selection_warning": (
            "Candidates are ranked against B_practice. This ranking is not part of the "
            "rho=0.1 DP mechanism and must not be described as privacy-preserving tuning."
        ),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "background": str(background_path.resolve()),
        "background_sha256": sha256(background_path),
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "death_weights": list(death_weights),
        "neighbor_counts": list(neighbor_counts),
        "seeds": list(seeds),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(full.to_string(index=False))
    print(f"\nWrote: {args.output_dir}")


if __name__ == "__main__":
    main()
