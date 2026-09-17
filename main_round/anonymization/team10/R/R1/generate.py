"""R1: stratified Gaussian copula + cause-specific Cox synthesis.

Only B.csv trains the synthesizer. B_self/A_bg/utility_ref are evaluation-only.
No S-series imports, record linkage, row perturbation, or formal DP claim.
The design follows the official kit's public copula_survival reference; this
implementation adds sex/smoking strata, correlation shrinkage, discrete age
quantiles, log-TG/log-ALT Cox predictors, and strict artifact verification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm, rankdata
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TEAM = HERE.parents[1]
DATA = TEAM / "participant_data"
KIT = ROOT / "main_round/local-output/official-kit"
PIN = "3dd71bd3436f2abfdbb64da97e860f22729ea054"
CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
           "smoking", "FPG", "time", "onset", "death"]
SEED = 20260917
MIN_STRATUM = 30
SHRINKAGE = 0.03
COX_PENALIZER = 0.01
TIME_FLOOR = 1.0 / 365.25


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correlation(x: np.ndarray) -> np.ndarray:
    """Rank-normal correlation, regularized and positive definite."""
    z = norm.ppf(rankdata(x, axis=0, method="average") / (len(x) + 1))
    z -= z.mean(axis=0)
    sd = np.sqrt(np.mean(z * z, axis=0))
    z = np.divide(z, sd, out=np.zeros_like(z), where=sd > 1e-12)
    corr = z.T @ z / len(z)
    np.fill_diagonal(corr, 1.0)
    corr = (1.0 - SHRINKAGE) * corr + SHRINKAGE * np.eye(x.shape[1])
    vals, vecs = np.linalg.eigh(corr)
    corr = (vecs * np.maximum(vals, 1e-8)) @ vecs.T
    scale = np.sqrt(np.diag(corr))
    return corr / np.outer(scale, scale)


def sample_copula(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    x = pool[CONT].to_numpy(dtype=float)
    latent = rng.multivariate_normal(np.zeros(len(CONT)), correlation(x), size=n)
    u = norm.cdf(latent)
    out = pd.DataFrame(index=range(n))
    for j, col in enumerate(CONT):
        # Age is discrete: interpolating then flooring would bias it downward.
        method = "inverted_cdf" if col == "age" else "linear"
        out[col] = np.quantile(x[:, j], u[:, j], method=method)
    return out


def cox_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["age", "sex", "BMI", "SBP", "HDL", "FPG", "smoking"]].astype(float).copy()
    out["log_TG"] = np.log(df["TG"].to_numpy(dtype=float))
    out["log_ALT"] = np.log(df["ALT"].to_numpy(dtype=float))
    return out


def invert_hazard(hazard: np.ndarray, times: np.ndarray, cumulative: np.ndarray,
                  uniforms: np.ndarray) -> np.ndarray:
    """Invert a piecewise-constant fitted cumulative hazard (no extrapolation)."""
    if not len(cumulative) or cumulative[-1] <= 0:
        return np.full(len(hazard), np.inf)
    target = -np.log(np.clip(uniforms, 1e-12, 1.0 - 1e-12)) / np.maximum(hazard, 1e-12)
    # Generalized inverse works even at flat stretches of the baseline curve.
    index = np.searchsorted(cumulative, target, side="left")
    result = np.full(len(target), np.inf)
    observed = index < len(times)
    result[observed] = times[index[observed]]
    return result


def sample_cause(b: pd.DataFrame, c: pd.DataFrame, cause: str,
                 rng: np.random.Generator) -> np.ndarray:
    if int(b[cause].sum()) == 0:
        return np.full(len(c), np.inf)
    train = cox_features(b)
    # Exclude constant predictors; never silently replace a failed Cox fit.
    features = train.columns[train.std() > 1e-12].tolist()
    train = train[features].assign(time=b["time"].to_numpy(), event=b[cause].to_numpy())
    model = CoxPHFitter(penalizer=COX_PENALIZER)
    model.fit(train, duration_col="time", event_col="event", show_progress=False)
    h0 = model.baseline_cumulative_hazard_.iloc[:, 0]
    hazard = model.predict_partial_hazard(cox_features(c)[features]).to_numpy()
    if not np.isfinite(hazard).all() or (hazard <= 0).any():
        raise ValueError("Non-finite or non-positive fitted partial hazard")
    return invert_hazard(hazard, h0.index.to_numpy(), h0.to_numpy(), rng.uniform(size=len(c)))


def competing_outcomes(onset_time: np.ndarray, death_time: np.ndarray,
                       horizon: float) -> pd.DataFrame:
    observed = np.minimum(onset_time, death_time) <= horizon
    onset = observed & (onset_time <= death_time)
    death = observed & (death_time < onset_time)
    time = np.where(observed, np.minimum(onset_time, death_time), horizon)
    return pd.DataFrame({"time": np.clip(time, TIME_FLOOR, horizon),
                         "onset": onset.astype(int), "death": death.astype(int)})


def synthesize(b: pd.DataFrame, cfg: dict, seed: int = SEED) -> tuple[pd.DataFrame, list[dict]]:
    # Whitelist inputs: neither record_id nor is_rare is a modeling feature.
    b = b[COLUMNS].reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    pieces, strata = [], []
    global_pref = b["prefecture"].value_counts(normalize=True).sort_index()
    for (sex, smoking), group in b.groupby(["sex", "smoking"], sort=True):
        pool, fallback = group, "none"
        if len(pool) < MIN_STRATUM:
            pool, fallback = b[b["sex"] == sex], "sex"
        if len(pool) < MIN_STRATUM:
            pool, fallback = b, "all"
        out = sample_copula(pool, len(group), rng)
        out["sex"], out["smoking"] = int(sex), int(smoking)
        # Five pseudo-observations shrink sparse prefecture cells to cohort marginals.
        counts = group["prefecture"].value_counts().reindex(global_pref.index, fill_value=0)
        probabilities = (counts + 5 * global_pref) / (len(group) + 5)
        out["prefecture"] = rng.choice(global_pref.index.to_numpy(), len(group), p=probabilities)
        pieces.append(out)
        strata.append({"sex": int(sex), "smoking": int(smoking), "n": len(group),
                       "fit_n": len(pool), "fallback": fallback})
    c = pd.concat(pieces, ignore_index=True)
    lo, hi = cfg["population"]["age_range"]
    c["age"] = np.clip(c["age"], lo, hi).astype(int)
    for col, (lo, hi) in cfg["schema"]["ranges"].items():
        c[col] = np.clip(c[col], lo, hi)
    onset = sample_cause(b, c, "onset", rng)
    death = sample_cause(b, c, "death", rng)
    outcomes = competing_outcomes(onset, death, float(cfg["onset"]["horizon_years"]))
    for col in outcomes:
        c[col] = outcomes[col]
    return c[COLUMNS].iloc[rng.permutation(len(c))].reset_index(drop=True), strata


def write_package(path: Path, csv_bytes: bytes, token_bytes: bytes) -> None:
    # Deterministic ZIP metadata, exact original token bytes, no extracted token copy.
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in (("C.csv", csv_bytes), ("token.txt", token_bytes)):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload)


def check_environment() -> tuple[dict, dict]:
    pin = subprocess.check_output(["git", "-C", str(KIT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(KIT), "status", "--porcelain",
                                     "--untracked-files=no"], text=True).strip()
    if pin != PIN or dirty:
        raise RuntimeError("Official kit must be clean at the pinned main-round commit")
    manifest = json.loads((TEAM / "data_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for entry in manifest["files"]:
        path = TEAM / entry["path"]
        actual = digest(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"Input digest mismatch: {path.name}")
        hashes[entry["path"]] = actual
    sys.path.insert(0, str(KIT / "starter/src"))
    import pwscup2026
    if not Path(pwscup2026.__file__).resolve().is_relative_to(KIT):
        raise RuntimeError("Wrong pwscup2026 package imported")
    cfg_path = KIT / "starter/src/pwscup2026/kit/kit_config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")), {
        "official_kit_commit": pin, "scoring_config_sha256": digest(cfg_path), "inputs": hashes}


def evaluate(outdir: Path, b: pd.DataFrame, cfg: dict) -> tuple[dict, dict]:
    from pwscup2026.common.submission_schema import read_scoring_csv, validate_c_submission
    from pwscup2026.kit.selfscore import load_submitted_c, self_score
    from pwscup2026.kit.validate import check_package, check_token_registered, rare_count_diagnostic
    from pwscup2026.scoring.utility import UtilityReference

    zip_path = outdir / "submission/team10_R1_defense.zip"
    package = check_package(zip_path, cfg)
    errors = list(package.errors)
    errors += check_token_registered(package.token, DATA)
    if errors or package.kind != "defense":
        raise RuntimeError(f"Package validation failed: {errors}")
    c = load_submitted_c(zip_path)
    validation = validate_c_submission(c, cfg, n_expected=len(b))
    if not validation.ok:
        raise RuntimeError(f"Logical validation failed: {validation.errors}")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["C.csv", "token.txt"]:
            raise RuntimeError("ZIP must contain exactly two flat files")
        if archive.read("C.csv") != (outdir / "C.csv").read_bytes():
            raise RuntimeError("CSV bytes differ between standalone file and ZIP")
        if archive.read("token.txt") != (DATA / "token.txt").read_bytes():
            raise RuntimeError("Token bytes changed")
    censored = (c["onset"] == 0) & (c["death"] == 0)
    if not (c.loc[censored, "time"] == cfg["onset"]["horizon_years"]).all():
        raise RuntimeError("Invalid administrative censoring time")
    exact_matches = int(c.merge(b[COLUMNS].drop_duplicates(), on=COLUMNS, how="inner").shape[0])
    duplicates = int(c.duplicated().sum())
    if exact_matches or duplicates:
        raise RuntimeError("Exact source rows or duplicate synthetic rows detected")

    b_self = read_scoring_csv(DATA / "B_self.csv")
    pd.testing.assert_frame_equal(b, b_self.drop(columns="is_rare"))
    ref_data = json.loads((DATA / "utility_ref.json").read_text(encoding="utf-8"))
    ref = UtilityReference(mu=np.asarray(ref_data["mu"]), sigma=np.asarray(ref_data["sigma"]),
                           band=ref_data["band"], b_is_rare=b_self["is_rare"].to_numpy(dtype=bool),
                           km_signal_D=ref_data.get("km_signal_D"), km_floor=ref_data.get("km_floor"),
                           a_bg=read_scoring_csv(DATA / "A_bg.csv"))
    score = self_score(c, b, ref, cfg)
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    report = {"package_ok": True, "logical_ok": validation.ok, "errors": [],
              "rows": len(c), "columns": c.columns.tolist(), "no_missing": not c.isna().any().any(),
              "token_matches_original_bytes": True, "csv_matches_zip_bytes": True,
              "censored_time_is_horizon": True, "source_exact_matches": exact_matches,
              "synthetic_duplicate_rows": duplicates, "rare_diagnostic": rare_count_diagnostic(c, schema),
              "event_counts": {name: int(c[name].sum()) for name in ("onset", "death")},
              "C_sha256": digest(outdir / "C.csv"), "ZIP_sha256": digest(zip_path),
              "privacy_note": "Exact non-duplication is not a privacy guarantee. No formal DP. No official protection score."}
    return score, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Read and verify frozen artifacts; no regeneration")
    parser.add_argument("--output", type=Path, default=HERE,
                        help="New output directory for reproducibility checks; existing artifacts are never overwritten")
    args = parser.parse_args()
    cfg, provenance = check_environment()
    from pwscup2026.common.submission_schema import read_scoring_csv
    from pwscup2026.scoring.result import json_safe
    b = read_scoring_csv(DATA / "B.csv")
    outdir = args.output.resolve()
    if not outdir.is_relative_to(HERE):
        raise ValueError("Output must stay inside R1")
    if args.verify_only:
        score, report = evaluate(outdir, b, cfg)
        saved = json.loads((outdir / "report.json").read_text(encoding="utf-8"))
        for key in ("C_sha256", "ZIP_sha256"):
            if report[key] != saved["validation"][key]:
                raise RuntimeError(f"Frozen artifact hash changed: {key}")
        if digest(Path(__file__)) != saved["provenance"]["generator_sha256"]:
            raise RuntimeError("Generator changed after release")
        if score["utility"] != saved["score"]["utility"] or score["facets"] != saved["score"]["facets"]:
            raise RuntimeError("Self-score changed in this environment")
        print(json.dumps({"verified": True, "utility": score["utility"], "facets": score["facets"]}))
        return
    targets = [outdir / "C.csv", outdir / "report.json", outdir / "submission/team10_R1_defense.zip"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite R1 artifacts; use --verify-only or a new local-output directory")
    c, strata = synthesize(b, cfg)
    csv_bytes = c.to_csv(index=False, lineterminator="\n").encode("utf-8")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "submission").mkdir(exist_ok=True)
    with (outdir / "C.csv").open("xb") as stream:
        stream.write(csv_bytes)
    write_package(outdir / "submission/team10_R1_defense.zip", csv_bytes, (DATA / "token.txt").read_bytes())
    score, report = evaluate(outdir, b, cfg)
    provenance.update({"generator_sha256": digest(Path(__file__)), "seed": SEED,
                       "training_inputs": ["participant_data/B.csv"],
                       "evaluation_only": ["participant_data/B_self.csv", "participant_data/A_bg.csv",
                                           "participant_data/utility_ref.json", "participant_data/schema.json"],
                       "method": "sex/smoking-stratified Gaussian copula + competing-risk Cox",
                       "correlation_shrinkage": SHRINKAGE, "cox_penalizer": COX_PENALIZER,
                       "min_stratum": MIN_STRATUM, "strata": strata,
                       "generation_runs": 1, "score_based_seed_search": False,
                       "python": sys.version.split()[0],
                       "dependencies": {name: importlib.metadata.version(name) for name in
                                        ["numpy", "pandas", "scipy", "lifelines", "scikit-learn", "statsmodels", "PyYAML"]}})
    result = json_safe({"provenance": provenance, "validation": report, "score": score})
    with (outdir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"created": str(outdir), "rows": len(c), "utility": score["utility"],
                      "facets": score["facets"], "rare": report["rare_diagnostic"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
