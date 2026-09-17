"""R2: stratified copula covariates and conditional Cox risk-set synthesis.

The generator learns from B only. Official self-score references are used
exclusively after synthesis. No source-row linkage, seed search or formal DP.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from evaluation import (COLUMNS, DATA, HERE, ZIP_NAME, check_environment, digest,
                        evaluate_release, read_training_data, write_package)
from survival import synthesize_outcomes

CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
SEED = 20260917
MIN_STRATUM = 30
SHRINKAGE = 0.03
CALIBRATION_STEPS = 0


def correlation(x: np.ndarray) -> np.ndarray:
    z = norm.ppf(rankdata(x, axis=0, method="average") / (len(x) + 1))
    z -= z.mean(axis=0)
    scale = np.sqrt(np.mean(z * z, axis=0))
    z = np.divide(z, scale, out=np.zeros_like(z), where=scale > 1e-12)
    corr = z.T @ z / len(z)
    np.fill_diagonal(corr, 1.0)
    corr = (1 - SHRINKAGE) * corr + SHRINKAGE * np.eye(x.shape[1])
    values, vectors = np.linalg.eigh(corr)
    corr = (vectors * np.maximum(values, 1e-8)) @ vectors.T
    diagonal = np.sqrt(np.diag(corr))
    return corr / np.outer(diagonal, diagonal)


def sample_copula(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    x = pool[CONT].to_numpy(dtype=float)
    latent = rng.multivariate_normal(np.zeros(len(CONT)), correlation(x), size=n)
    uniforms = norm.cdf(latent)
    out = pd.DataFrame(index=range(n))
    for j, column in enumerate(CONT):
        out[column] = np.quantile(x[:, j], uniforms[:, j],
                                  method="inverted_cdf" if column == "age" else "linear")
    return out


def synthesize_covariates(b: pd.DataFrame, cfg: dict, seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    """Keep R1's tested covariate model to isolate the outcome-model improvement."""
    b = b[COLUMNS].reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    pieces, strata = [], []
    global_pref = b.prefecture.value_counts(normalize=True).sort_index()
    for (sex, smoking), group in b.groupby(["sex", "smoking"], sort=True):
        pool, fallback = group, "none"
        if len(pool) < MIN_STRATUM:
            pool, fallback = b[b.sex == sex], "sex"
        if len(pool) < MIN_STRATUM:
            pool, fallback = b, "all"
        out = sample_copula(pool, len(group), rng)
        out["sex"], out["smoking"] = int(sex), int(smoking)
        counts = group.prefecture.value_counts().reindex(global_pref.index, fill_value=0)
        probabilities = (counts + 5 * global_pref) / (len(group) + 5)
        out["prefecture"] = rng.choice(global_pref.index.to_numpy(), len(group), p=probabilities)
        pieces.append(out)
        strata.append({"sex": int(sex), "smoking": int(smoking), "n": len(group),
                       "fit_n": len(pool), "fallback": fallback})
    c = pd.concat(pieces, ignore_index=True)
    lo, hi = cfg["population"]["age_range"]
    c["age"] = np.clip(c.age, lo, hi).astype(int)
    for column, (lo, hi) in cfg["schema"]["ranges"].items():
        c[column] = np.clip(c[column], lo, hi)
    return c, strata


def synthesize(b: pd.DataFrame, cfg: dict, seed: int = SEED,
               calibration_steps: int = CALIBRATION_STEPS) -> tuple[pd.DataFrame, dict]:
    b = b[COLUMNS].reset_index(drop=True).copy()
    c, strata = synthesize_covariates(b, cfg, seed)
    outcomes, survival_audit = synthesize_outcomes(
        b, c, float(cfg["onset"]["horizon_years"]), seed=seed,
        calibration_steps=calibration_steps)
    for column in ("time", "onset", "death"):
        c[column] = outcomes[column].to_numpy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2026, 2]))
    c = c[COLUMNS].iloc[rng.permutation(len(c))].reset_index(drop=True)
    return c, {"strata": strata, "survival": survival_audit}


def source_hashes() -> dict:
    return {name: digest(HERE / name) for name in ("generate.py", "survival.py", "evaluation.py")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reproduce-check", action="store_true",
                        help="Regenerate in memory and compare CSV bytes; never overwrite artifacts")
    args = parser.parse_args()
    cfg, provenance = check_environment()
    from pwscup2026.scoring.result import json_safe
    b = read_training_data()
    outdir = args.output.resolve()
    if not outdir.is_relative_to(HERE):
        raise ValueError("R2 output must remain inside its model directory")
    if args.verify_only or args.reproduce_check:
        saved = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
        if saved["provenance"]["source_sha256"] != source_hashes():
            raise RuntimeError("R2 implementation changed after the release")
        for field in ("official_kit_commit", "scoring_config_sha256", "rulebook_sha256", "inputs",
                      "dependencies", "python"):
            if saved["provenance"].get(field) != provenance[field]:
                raise RuntimeError(f"R2 reproduction provenance changed: {field}")
        score, validation = evaluate_release(outdir, b, cfg)
        for field in ("C_sha256", "ZIP_sha256"):
            if validation[field] != saved["validation"][field]:
                raise RuntimeError(f"Frozen R2 artifact changed: {field}")
        if score["utility"] != saved["score"]["utility"] or score["facets"] != saved["score"]["facets"]:
            raise RuntimeError("R2 self-score changed in this environment")
        if args.reproduce_check:
            c, _ = synthesize(b, cfg)
            if c.to_csv(index=False, lineterminator="\n").encode("utf-8") != (outdir / "C.csv").read_bytes():
                raise RuntimeError("R2 fixed-seed reproduction differs")
        print(json.dumps({"verified": True, "reproduced": args.reproduce_check,
                          "utility": score["utility"], "facets": score["facets"]}))
        return
    targets = [outdir / "C.csv", outdir / "report.json", outdir / "submission" / ZIP_NAME]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite R2; use --verify-only or a new local-output directory")
    c, model_audit = synthesize(b, cfg)
    csv_bytes = c.to_csv(index=False, lineterminator="\n").encode("utf-8")
    # Validate the serialized representation before creating any credential-bearing ZIP.
    from evaluation import evaluate_frame
    preflight_score, _ = evaluate_frame(pd.read_csv(io.BytesIO(csv_bytes), float_precision="round_trip"), b, cfg)
    baseline = json.loads((HERE.parent / "R1/report.json").read_text(encoding="utf-8"))
    if (preflight_score["utility"] <= baseline["score"]["utility"]
            or any(preflight_score["facets"][name] < value - 1e-12
                   for name, value in baseline["score"]["facets"].items())):
        raise RuntimeError("R2 release must improve utility without regressing any R1 utility facet")
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "C.csv").open("xb") as stream:
        stream.write(csv_bytes)
    write_package(outdir / "submission" / ZIP_NAME, csv_bytes, (DATA / "token.txt").read_bytes())
    score, validation = evaluate_release(outdir, b, cfg)
    provenance.update({"model": "R2", "source_sha256": source_hashes(), "seed": SEED,
                       "training_inputs": ["participant_data/B.csv"],
                       "evaluation_only": ["participant_data/B_self.csv", "participant_data/A_bg.csv",
                                           "participant_data/utility_ref.json", "participant_data/schema.json"],
                       "method": "stratified Gaussian copula + conditional Cox risk-set sampling",
                       "correlation_shrinkage": SHRINKAGE, "min_stratum": MIN_STRATUM,
                       "score_based_seed_search": False, "calibration_steps": CALIBRATION_STEPS,
                       "candidate_selection": "conditional uncalibrated model after three documented fixed-seed ablations",
                       "development_record": "EXPERIMENTS.md",
                       "formal_differential_privacy": False,
                       "official_submission_performed": False})
    comparison = {"R1": baseline["score"]["facets"], "R2": score["facets"],
                  "utility_R1": baseline["score"]["utility"], "utility_R2": score["utility"],
                  "utility_delta": score["utility"] - baseline["score"]["utility"]}
    result = json_safe({"provenance": provenance, "model_audit": model_audit,
                        "validation": validation, "score": score, "comparison": comparison})
    with (outdir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"created": str(outdir), "rows": len(c), "utility": score["utility"],
                      "facets": score["facets"], "delta_R1": comparison["utility_delta"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
