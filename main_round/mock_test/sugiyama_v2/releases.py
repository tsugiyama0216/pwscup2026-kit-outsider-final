"""Fixed, diverse release scenarios for joint Sugiyama v2 mock worlds.

These are sensitivity scenarios, not fitted reconstructions of other teams'
defenses. Parameters and profile counts are fixed without fitting leaderboard
feedback. The independent scenario never reads the target cohort.
"""
from __future__ import annotations

from collections import Counter
from numbers import Integral

import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
            "smoking", "FPG", "time", "onset", "death"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
STYLES = ("copy", "rounded", "jitter", "strong_jitter", "copula",
          "conditional_resample", "outcome_shuffle", "background_mix", "independent")
ROWS = 1049

# These contain all 22 targets, including independent/null releases. Consumers
# must not remove selected styles from the principal whole-world aggregation.
PROFILES = {
    "mixed": (
        "rounded", "jitter", "conditional_resample", "copula", "strong_jitter",
        "outcome_shuffle", "background_mix", "independent", "conditional_resample",
        "jitter", "copula", "strong_jitter", "background_mix", "outcome_shuffle",
        "conditional_resample", "copula", "jitter", "independent", "strong_jitter",
        "conditional_resample", "outcome_shuffle", "background_mix",
    ),
    "stress": (
        "strong_jitter", "copula", "conditional_resample", "background_mix",
        "independent", "outcome_shuffle", "copula", "strong_jitter",
        "conditional_resample", "background_mix", "independent", "outcome_shuffle",
        "strong_jitter", "copula", "conditional_resample", "background_mix",
        "independent", "outcome_shuffle", "strong_jitter", "copula",
        "conditional_resample", "background_mix",
    ),
}


def describe_styles():
    """Return JSON-compatible provenance and predeclared scenario metadata."""
    return {
        "version": "sugiyama-v2-release-scenarios-1",
        "fit_to_official_feedback": False,
        "parameter_selection": "Fixed ranges; realized parameters depend only on seed and style",
        "aggregation": "Include every target, including independent; copy is optional sanity only",
        "profiles": {name: {"targets": len(styles), "counts": dict(Counter(styles)),
                             "styles": list(styles)} for name, styles in PROFILES.items()},
        "styles": {
            "copy": "Unmodified public columns, shuffled row order; sanity scenario only",
            "rounded": "Fixed coarse rounding of numeric columns and time",
            "jitter": "Background-SD noise 0.06-0.28; time SD 0.12-0.70; categorical redraw 0.015-0.09",
            "strong_jitter": "Background-SD noise 0.30-0.90; time SD 0.60-2.30; categorical redraw 0.08-0.28",
            "copula": "New Gaussian rank draws; covariance shrinkage 0.03-0.25; joint event category; conditional prefecture",
            "conditional_resample": "Bootstrap and perturb QI; resample non-QI/outcome blocks from 12-50 QI neighbors",
            "outcome_shuffle": "Shuffle complete time/onset/death blocks within sex and 10-year age strata",
            "background_mix": "Perturbed target release with 25%-75% whole rows replaced by perturbed background draws",
            "independent": "Target-invariant background bootstrap, numeric/time noise and categorical redraw",
        },
        "limitations": [
            "Profiles are sensitivity panels, not estimates of the 22 actual defenses or their prevalence.",
            "No differential-privacy or official utility guarantee is claimed for these synthetic releases.",
            "An independent release remains a null target in the world and can score above zero by chance.",
        ],
    }


def _public(frame, name, *, target=False):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if target and len(frame) != ROWS:
        raise ValueError(f"{name} must have {ROWS} rows")
    if not set(FEATURES).issubset(frame.columns):
        raise ValueError(f"{name} is missing public columns")
    result = frame.loc[:, FEATURES].copy(deep=True).reset_index(drop=True)
    if result.prefecture.isna().any():
        raise ValueError(f"{name} has missing prefecture values")
    numeric = result.drop(columns="prefecture").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} has nonfinite values")
    for column in ("sex", "smoking", "onset", "death"):
        if not result[column].isin((0, 1)).all():
            raise ValueError(f"{name} has invalid {column}")
    if ((result.onset == 1) & (result.death == 1)).any():
        raise ValueError(f"{name} has simultaneous onset and death")
    return result


def _jitter(frame, background, rng, *, strong=False, independent=False):
    c = frame.copy(deep=True)
    scale = background[CONTINUOUS].std(ddof=0).to_numpy(float)
    strength = rng.uniform(.30, .90) if strong else rng.uniform(.06, .28)
    time_sd = rng.uniform(.60, 2.30) if strong else rng.uniform(.12, .70)
    category_rate = rng.uniform(.08, .28) if strong else rng.uniform(.015, .09)
    if independent:
        strength, time_sd = rng.uniform(.05, .35), rng.uniform(.20, 1.20)
    c[CONTINUOUS] = c[CONTINUOUS].to_numpy(float) + rng.normal(size=(len(c), len(CONTINUOUS))) * scale * strength
    c["time"] = c.time.to_numpy(float) + rng.normal(0, time_sd, len(c))
    for column in ("sex", "smoking", "prefecture"):
        change = rng.random(len(c)) < category_rate
        c.loc[change, column] = rng.choice(background[column].to_numpy(), int(change.sum()))
    return c


def _copula(target, rng):
    from scipy.stats import norm, rankdata

    # Represent onset/death as a single categorical variable so invalid (1,1)
    # event combinations cannot be introduced by independent inverse marginals.
    columns = CONTINUOUS + ["time", "sex", "smoking", "event"]
    fitting = target.assign(event=2 * target.onset.astype(int) + target.death.astype(int))
    values = fitting[columns].to_numpy(float)
    ranks = np.column_stack([rankdata(values[:, j], method="average") for j in range(len(columns))])
    latent = norm.ppf((ranks - .5) / len(values))
    covariance = np.cov(latent, rowvar=False)
    diagonal = np.diag(np.maximum(np.diag(covariance), 1e-6))
    shrink = rng.uniform(.03, .25)
    covariance = (1 - shrink) * covariance + shrink * diagonal
    draws = rng.multivariate_normal(np.zeros(len(columns)), covariance, size=ROWS, check_valid="raise")
    uniform = norm.cdf(draws / np.sqrt(np.maximum(np.diag(covariance), 1e-6)))
    c = pd.DataFrame(index=np.arange(ROWS))
    for j, column in enumerate(columns):
        if column in ("sex", "smoking", "event"):
            positions = np.minimum((uniform[:, j] * len(values)).astype(int), len(values) - 1)
            c[column] = np.sort(values[:, j])[positions].astype(int)
        else:
            c[column] = np.quantile(values[:, j], uniform[:, j])
    c["onset"], c["death"] = (c.event == 2).astype(int), (c.event == 1).astype(int)
    del c["event"]
    c["prefecture"] = ""
    # Preserve broad demographic dependence without copying a synthetic row's
    # numeric values or taking any member/query labels as inputs.
    for sex in (0, 1):
        for age_band in range(3, 9):
            selected = (c.sex == sex) & (np.floor(c.age / 10) == age_band)
            positions = np.flatnonzero(selected)
            if not len(positions):
                continue
            donors = target[(target.sex == sex) & (np.floor(target.age / 10) == age_band)]
            if len(donors) < 12:
                donors = target[target.sex == sex]
            if donors.empty:
                donors = target
            c.loc[positions, "prefecture"] = rng.choice(donors.prefecture.to_numpy(), len(positions))
    if c.prefecture.eq("").any():
        raise ValueError("Copula age distribution is outside supported contest age strata")
    return c.loc[:, FEATURES]


def _conditional(target, background, rng):
    rows = rng.integers(len(target), size=ROWS)
    c = target.iloc[rows].copy(deep=True).reset_index(drop=True)
    numeric_qi = [column for column in QI if column not in ("sex", "prefecture")]
    scale = np.maximum(background[numeric_qi].std(ddof=0).to_numpy(float), 1e-6)
    c[numeric_qi] = c[numeric_qi].to_numpy(float) + rng.normal(size=(ROWS, len(numeric_qi))) * scale * rng.uniform(.05, .18)
    query = c[numeric_qi].to_numpy(float) / scale
    source = target[numeric_qi].to_numpy(float) / scale
    squared = np.maximum(np.sum(query ** 2, axis=1)[:, None] + np.sum(source ** 2, axis=1)[None, :] - 2 * query @ source.T, 0)
    squared += 2 * (c.sex.to_numpy()[:, None] != target.sex.to_numpy()[None, :])
    squared += 2 * (c.prefecture.to_numpy()[:, None] != target.prefecture.to_numpy()[None, :])
    neighbors = int(rng.integers(12, 51))
    nearest = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
    distance = np.take_along_axis(squared, nearest, axis=1)
    weights = 1 / (.25 + distance)
    cumulative = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
    picked = np.minimum(np.sum(rng.random(ROWS)[:, None] > cumulative, axis=1), neighbors - 1)
    donors = nearest[np.arange(ROWS), picked]
    other_columns = [column for column in FEATURES if column not in QI]
    c[other_columns] = target.iloc[donors][other_columns].to_numpy()
    c["time"] = c.time.to_numpy(float) + rng.normal(0, rng.uniform(.10, .60), ROWS)
    return c


def _outcome_shuffle(target, rng):
    c = target.copy(deep=True)
    strata = [target.sex.to_numpy(), np.floor(target.age.to_numpy(float) / 10).astype(int)]
    for indices in target.groupby(strata, sort=True).indices.values():
        indices = np.asarray(indices, dtype=int)
        shuffled = rng.permutation(indices)
        c.loc[indices, ["time", "onset", "death"]] = target.iloc[shuffled][["time", "onset", "death"]].to_numpy()
    return c


def _finish(c, rng, schema):
    c = c.loc[:, FEATURES].copy(deep=True)
    for column, bounds in schema["ranges"].items():
        if column in c:
            c[column] = c[column].clip(*bounds)
    c["age"] = np.rint(c.age.clip(*schema["age_range"])).astype(int)
    c["time"] = c.time.clip(1 / 365.25, 10)
    for column in ("sex", "smoking", "onset", "death"):
        c[column] = c[column].astype(int)
    c = c.iloc[rng.permutation(len(c))].reset_index(drop=True)
    if c.shape != (ROWS, len(FEATURES)) or not np.isfinite(c.drop(columns="prefecture").to_numpy(float)).all():
        raise ValueError("Generated release has invalid shape or values")
    allowed = {tuple(pair) for pair in schema.get("allowed_onset_death", [(0, 0), (0, 1), (1, 0)])}
    if not set(map(tuple, c[["onset", "death"]].to_numpy())).issubset(allowed):
        raise ValueError("Generated release has invalid outcome combinations")
    return c


def make_release(target, background, style, seed, schema):
    """Generate a reproducible 1049-row public C without changing its inputs."""
    if style not in STYLES:
        raise ValueError(f"Unknown Sugiyama v2 release style: {style}")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), STYLES.index(style), 2026100702]))
    background = _public(background, "background")
    if style == "independent":
        # Do not even validate target here: neither its size nor its values may
        # change an independent/null release or consume random numbers.
        c = background.iloc[rng.integers(len(background), size=ROWS)].reset_index(drop=True)
        c = _jitter(c, background, rng, independent=True)
    else:
        target = _public(target, "target", target=True)
        if style == "copula":
            c = _copula(target, rng)
        elif style == "conditional_resample":
            c = _conditional(target, background, rng)
        elif style == "outcome_shuffle":
            c = _outcome_shuffle(target, rng)
        elif style in ("jitter", "strong_jitter", "background_mix"):
            c = _jitter(target, background, rng, strong=style == "strong_jitter")
            if style == "background_mix":
                count = int(round(ROWS * rng.uniform(.25, .75)))
                positions = rng.choice(ROWS, count, replace=False)
                draws = background.iloc[rng.integers(len(background), size=count)].reset_index(drop=True)
                c.iloc[positions] = _jitter(draws, background, rng, independent=True).to_numpy()
        else:
            c = target.copy(deep=True)
            if style == "rounded":
                widths = {"age": 5, "BMI": 1, "SBP": 10, "TG": 10, "HDL": 5, "ALT": 5, "FPG": 10, "time": .5}
                for column, width in widths.items():
                    c[column] = np.round(c[column].to_numpy(float) / width) * width
    return _finish(c, rng, schema)
