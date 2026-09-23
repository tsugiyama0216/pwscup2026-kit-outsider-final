"""Public-correlation-assisted DP Gaussian copula for team10.

Only ``fit`` accesses private records. Its outputs are budgeted integer DP
histograms plus public statistics. ``sample`` is reusable postprocessing.
Noise uses secret OS randomness; the sampling seed is deliberately separate.
See PRIVACY_DESIGN.md for the fixed-size replace-one privacy contract.
"""
from __future__ import annotations

from fractions import Fraction
import itertools
import math
import secrets
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

VERSION = "S_Copula_public_correlation_geometric_v1"
PRIVATE_N = 1049  # Public team allocation size; never inferred or released by fit.
CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
NUMERIC = CONT + ["time"]
GROUPS = list(itertools.product((0, 1), (0, 1), (0, 1), (0, 1, 2)))
DEFAULT_SAMPLE = {
    "group_prior": 0.0,
    "marginal_prior_noise_multiplier": 3.0,
    "prefecture_prior": 0.0,
    "public_correlation_strength": 1.0,
    "public_radial_strength": 0.0,
    "enforce_public_support": True,
    "enforce_rare_partition": True,
    "stratified": True,
}


def _fraction(value: Any) -> Fraction:
    value = value if isinstance(value, Fraction) else Fraction(str(value))
    if value <= 0:
        raise ValueError("epsilon must be positive")
    return value


def privacy_plan(epsilon: Any) -> dict:
    """Data-independent exact budget plan; each query has L1 sensitivity two."""
    eps = _fraction(epsilon)
    allocations = [("group_counts", eps / 5),
                   ("prefecture_counts", eps * 3 / 20)]
    allocations += [(f"marginal_{name}", eps * 11 / 140) for name in CONT]
    allocations += [("marginal_time", eps / 10)]
    assert sum(v for _, v in allocations) == eps
    queries = []
    for name, alloc in allocations:
        q = Fraction(2, 1) / (Fraction(2, 1) + alloc)
        queries.append({"name": name, "epsilon_allocated_exact": str(alloc),
                        "l1_sensitivity": 2,
                        "geometric_q_numerator": q.numerator,
                        "geometric_q_denominator": q.denominator,
                        "epsilon_effective_upper": float(alloc),
                        "epsilon_effective_formula": "2*log(1+epsilon_allocated/2)"})
    return {"adjacency": "replace_one_fixed_public_size", "public_n": PRIVATE_N,
            "epsilon_exact": str(eps), "epsilon": float(eps), "delta": 0,
            "composition": "sequential_all_queries", "queries": queries}


def _geometric(q_numerator: int, q_denominator: int) -> int:
    """Exact unbounded G with Pr[G=k]=(1-q)q**k using rational Bernoulli."""
    count = 0
    while secrets.randbelow(q_denominator) < q_numerator:
        count += 1
    return count


def _noise_counts(counts: np.ndarray, query: dict) -> list:
    # No truncation, floating-point inverse CDF, deterministic seed or retries.
    num, den = query["geometric_q_numerator"], query["geometric_q_denominator"]
    shape = np.asarray(counts).shape
    result = [int(c) + _geometric(num, den) - _geometric(num, den)
              for c in np.asarray(counts).flat]
    # Python integers keep the noise support unbounded; int64 would overflow.
    return np.asarray(result, dtype=object).reshape(shape).tolist()


def release_integer_vector(counts: Any, epsilon: Any, sensitivity: int) -> tuple[list, dict]:
    """General integer-vector mechanism for separately accounted DP queries.

    The caller must prove the supplied integer L1 sensitivity and compose
    this query with every other release. This helper does not infer budgets.
    """
    eps = _fraction(epsilon)
    if not isinstance(sensitivity, int) or sensitivity <= 0:
        raise ValueError("sensitivity must be a positive integer")
    q = Fraction(sensitivity, 1) / (Fraction(sensitivity, 1) + eps)
    query = {"epsilon_allocated_exact": str(eps), "l1_sensitivity": sensitivity,
             "geometric_q_numerator": q.numerator, "geometric_q_denominator": q.denominator,
             "epsilon_effective_upper": float(eps),
             "epsilon_effective_formula": "sensitivity*log(1+epsilon_allocated/sensitivity)"}
    return _noise_counts(np.asarray(counts, dtype=object), query), query


def _released_float(values: Any) -> np.ndarray:
    """Public saturation before floating postprocessing; exact model keeps ints."""
    a = np.asarray(values, dtype=object)
    return np.asarray([float(max(-10**12, min(10**12, int(v)))) for v in a.flat]).reshape(a.shape)


def _bounds(schema: dict) -> dict:
    bounds = {k: list(map(float, v)) for k, v in schema["ranges"].items()}
    bounds["age"] = list(map(float, schema["age_range"]))
    # Official public kit follow-up horizon is ten years. A tiny strictly
    # positive bound is a public validity convention, never a private min.
    bounds["time"] = [1e-6, 10.0]
    return bounds


def _prepare(frame: pd.DataFrame, schema: dict, categories: list,
             public_defaults: dict) -> pd.DataFrame:
    """Deterministic row-local clipping/imputation to a fixed public domain."""
    out = pd.DataFrame(index=range(len(frame)))
    for col in NUMERIC:
        lo, hi = _bounds(schema)[col]
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        values = np.nan_to_num(values, nan=public_defaults[col], posinf=hi, neginf=lo)
        out[col] = np.clip(values, lo, hi)
    out["age"] = np.rint(out["age"]).astype(int)
    for col in ["sex", "smoking", "onset", "death"]:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        out[col] = (np.nan_to_num(values, nan=0.0) >= 0.5).astype(int)
    out["death"] = out["death"] * (1 - out["onset"])
    out["prefecture"] = [v if isinstance(v, str) and v in categories else categories[0]
                         for v in frame["prefecture"].tolist()]
    return out


def _radii(frame: pd.DataFrame, detector: dict) -> np.ndarray:
    cols = detector["columns"]
    d = frame[cols].to_numpy(float) - np.asarray(detector["mu"], float)
    inv = np.linalg.pinv(np.asarray(detector["sigma"], float))
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", d, inv, d)))


def _group_ids(frame: pd.DataFrame, detector: dict) -> np.ndarray:
    rare = (_radii(frame, detector) >= float(detector["band"]["lo"])).astype(int)
    event = np.where(frame.onset.to_numpy() == 1, 2, frame.death.to_numpy())
    return (((rare * 2 + frame.sex.to_numpy()) * 2 + frame.smoking.to_numpy()) * 3
            + event).astype(int)


def _simplex(values: np.ndarray, total: float) -> np.ndarray:
    """Euclidean projection onto nonnegative vectors with fixed sum."""
    v = np.asarray(values, dtype=float)
    if total <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    active = np.flatnonzero(u - css / np.arange(1, len(u) + 1) > 0)
    theta = css[active[-1]] / (active[-1] + 1)
    result = np.maximum(v - theta, 0.0)
    return result * (total / result.sum())


def _correlation(frame: pd.DataFrame) -> list:
    x = frame[NUMERIC].to_numpy(float)
    z = norm.ppf((rankdata(x, axis=0, method="average") - 0.5) / len(x))
    z -= z.mean(axis=0)
    sd = np.sqrt(np.mean(z * z, axis=0))
    z /= np.where(sd > 1e-12, sd, 1.0)
    corr = z.T @ z / len(x)
    np.fill_diagonal(corr, 1.0)
    # Public ridge stabilizes small public subgroups and tied event times.
    corr = 0.97 * corr + 0.03 * np.eye(len(NUMERIC))
    return corr.tolist()


def _public_spec(public: pd.DataFrame, schema: dict, bins: int) -> dict:
    pg = _group_ids(public, schema["rare_detector"])
    public_radii = _radii(public, schema["rare_detector"])
    specs = []
    bounds = _bounds(schema)
    for gid, (rare, sex, smoking, event) in enumerate(GROUPS):
        subset = public.loc[pg == gid]
        if len(subset) < 16:
            # The fallback depends ONLY on public sample sizes.
            all_event = np.where(public.onset.to_numpy() == 1, 2, public.death.to_numpy())
            public_rare = public_radii >= float(schema["rare_detector"]["band"]["lo"])
            subset = public.loc[(all_event == event) & (public_rare == bool(rare))]
            if len(subset) < 16:
                subset = public.loc[public_rare == bool(rare)]
                if len(subset) < 16:
                    subset = public
        margins = {}
        for col in NUMERIC:
            values = subset[col].to_numpy(float)
            lo, hi = bounds[col]
            edges = np.unique(np.r_[lo, np.quantile(values, np.linspace(0, 1, bins + 1)), hi])
            counts, _ = np.histogram(values, bins=edges)
            # Full support ensures a well-defined inverse even in public gaps.
            p = (counts + 0.05) / (counts.sum() + 0.05 * len(counts))
            grids = []
            for k in range(len(edges) - 1):
                take = values[(values >= edges[k]) &
                              ((values < edges[k + 1]) if k < len(edges)-2 else
                               (values <= edges[k + 1]))]
                if len(take):
                    grid = np.quantile(take, np.linspace(0, 1, 17))
                else:
                    # DP noise may give mass to a public-empty boundary bin.
                    # Its inverse must not invent huge uniform tails extending
                    # to schema endpoints (e.g. TG3000) absent from A_bg.
                    midpoint = (edges[k] + edges[k + 1]) / 2
                    nearest = values[np.argmin(np.abs(values - midpoint))]
                    grid = np.full(17, nearest)
                grids.append(grid.tolist())
            margins[col] = {"edges": edges.tolist(), "public_probabilities": p.tolist(),
                            "public_within_bin_quantiles": grids}
        specs.append({"group": list(GROUPS[gid]), "public_n": int((pg == gid).sum()),
                      "public_fit_n": len(subset), "public_correlation": _correlation(subset),
                      "public_radius_quantiles": np.quantile(
                          _radii(subset, schema["rare_detector"]), np.linspace(0, 1, 257)).tolist(),
                      "margins": margins})
    return {"groups": specs, "public_group_counts": np.bincount(pg, minlength=len(GROUPS)).tolist()}


def fit(private_frame: pd.DataFrame, public_frame: pd.DataFrame,
        schema: dict, epsilon: Any, *, bins: int = 8) -> dict:
    """Release one DP model; each call spends its full planned epsilon.

    ``private_frame`` must have the public fixed size 1049. Public bin count
    and all other design choices must be frozen without private evaluation.
    No caller-supplied randomness is accepted for the DP mechanism.
    """
    plan = privacy_plan(epsilon)
    if len(private_frame) != PRIVATE_N:
        raise ValueError(f"This fixed-size privacy contract requires {PRIVATE_N} rows")
    if not isinstance(bins, int) or not 2 <= bins <= 32:
        raise ValueError("public bins must be an integer from 2 through 32")
    categories = sorted(str(v) for v in public_frame.prefecture.dropna().unique())
    if not categories:
        raise ValueError("Public background must define prefecture categories")
    defaults = {col: float(pd.to_numeric(public_frame[col], errors="coerce").median())
                for col in NUMERIC}
    if not all(math.isfinite(v) for v in defaults.values()):
        raise ValueError("Public background has invalid numeric defaults")
    public = _prepare(public_frame, schema, categories, defaults)
    private = _prepare(private_frame, schema, categories, defaults)
    model = _public_spec(public, schema, bins)
    model.update({"version": VERSION, "privacy": plan, "schema": schema,
                  "public_rows": len(public), "categories": categories,
                  "public_defaults": defaults, "bins": bins,
                  "public_support": {col: [float(public[col].min()), float(public[col].max())]
                                     for col in NUMERIC},
                  "default_sample_options": dict(DEFAULT_SAMPLE)})
    queries = {q["name"]: q for q in plan["queries"]}
    ids = _group_ids(private, schema["rare_detector"])
    model["dp_group_counts"] = _noise_counts(np.bincount(ids, minlength=len(GROUPS)),
                                             queries["group_counts"])
    public_pref = public.prefecture.value_counts().reindex(categories, fill_value=0).to_numpy()
    model["public_prefecture_probabilities"] = (public_pref / public_pref.sum()).tolist()
    private_pref = private.prefecture.value_counts().reindex(categories, fill_value=0).to_numpy()
    model["dp_prefecture_counts"] = _noise_counts(private_pref, queries["prefecture_counts"])
    for col in NUMERIC:
        # One vector query spans ALL groups for one feature. A replacement
        # changes at most two cells, even when the public group label changes.
        packed, lengths = [], []
        for gid, spec in enumerate(model["groups"]):
            counts, _ = np.histogram(private.loc[ids == gid, col].to_numpy(float),
                                     bins=spec["margins"][col]["edges"])
            packed.extend(counts.tolist())
            lengths.append(len(counts))
        released = _noise_counts(np.asarray(packed, dtype=np.int64), queries[f"marginal_{col}"])
        offset = 0
        for spec, length in zip(model["groups"], lengths):
            spec["margins"][col]["dp_counts"] = released[offset:offset + length]
            offset += length
    return model


def _allocate(probabilities: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    exact = np.asarray(probabilities) * n
    counts = np.floor(exact).astype(int)
    left = n - counts.sum()
    # Random tie breaker is pure postprocessing, not secret DP noise.
    order = np.lexsort((rng.random(len(counts)), -(exact - counts)))
    counts[order[:left]] += 1
    return counts


def _inverse_margin(u: np.ndarray, probabilities: np.ndarray, margin: dict) -> np.ndarray:
    cdf = np.r_[0.0, np.cumsum(probabilities)]
    cdf[-1] = 1.0
    idx = np.minimum(np.searchsorted(cdf[1:], u, side="right"), len(probabilities)-1)
    local = (u - cdf[idx]) / np.maximum(probabilities[idx], 1e-300)
    result = np.empty(len(u))
    for k in np.unique(idx):
        take = idx == k
        grid = np.asarray(margin["public_within_bin_quantiles"][k], float)
        result[take] = np.interp(local[take], np.linspace(0, 1, len(grid)), grid)
    return result


def _project_rare(frame: pd.DataFrame, target_rare: np.ndarray, detector: dict,
                  bounds: dict) -> pd.DataFrame:
    """Postprocessing to preserve the public majority/rare partition."""
    cols = detector["columns"]
    values = frame[cols].to_numpy(float)
    mu = np.asarray(detector["mu"], float)
    gate = float(detector["band"]["lo"])
    # Public deterministic projection, using only DP-generated output.
    for _ in range(3):
        radii = _radii(frame, detector)
        wanted = np.where(target_rare == 1, np.maximum(radii, gate + 0.025),
                          np.minimum(radii, gate - 0.025))
        delta = values - mu
        zero = radii < 1e-12
        if zero.any():
            delta[zero, 0] = 1.0
            radii[zero] = 1.0 / np.sqrt(float(detector["sigma"][0][0]))
        values = mu + delta * (wanted / np.maximum(radii, 1e-12))[:, None]
        for j, col in enumerate(cols):
            values[:, j] = np.clip(values[:, j], *bounds[col])
            if col == "age":
                values[:, j] = np.rint(values[:, j])
            frame[col] = values[:, j]
    return frame


def _public_radial_adjust(frame: pd.DataFrame, spec: dict, strength: float,
                          detector: dict, bounds: dict) -> pd.DataFrame:
    """Match public radial quantiles, using only synthetic ranks/public grids."""
    if strength <= 0:
        return frame
    radius = _radii(frame, detector)
    quantiles = (rankdata(radius) - 0.5) / len(frame)
    grid = np.asarray(spec["public_radius_quantiles"], float)
    target = np.interp(quantiles, np.linspace(0, 1, len(grid)), grid)
    target = (1-strength)*radius + strength*target
    columns = detector["columns"]
    mu = np.asarray(detector["mu"], float)
    delta = frame[columns].to_numpy(float) - mu
    ratio = np.divide(target, radius, out=np.ones_like(target), where=radius > 1e-12)
    # Public rectangular bounds cap rays before projection to avoid creating
    # implausible boundary masses by independently clipping coordinates.
    lower = np.array([bounds[col][0] for col in columns])
    upper = np.array([bounds[col][1] for col in columns])
    ray_limit = np.min(np.divide(np.where(delta >= 0, upper-mu, lower-mu), delta,
                                out=np.full_like(delta, np.inf), where=np.abs(delta) > 1e-12), axis=1)
    ratio = np.minimum(ratio, ray_limit)
    adjusted = mu + ratio[:, None]*delta
    for j, col in enumerate(columns):
        frame[col] = np.clip(adjusted[:, j], *bounds[col])
    return frame


def sample(model: dict, n: int = PRIVATE_N, seed: int | None = None,
           **options: Any) -> pd.DataFrame:
    """Generate 13-column C from a saved DP model, with no private input.

    Options are postprocessing and can be selected using public proxy data or
    DP model statistics. Selecting them using unprotected B metrics is outside
    this mechanism's guarantee and must not drive a public release decision.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    unknown = set(options) - set(DEFAULT_SAMPLE)
    if unknown:
        raise ValueError(f"Unknown sample options: {sorted(unknown)}")
    opt = dict(DEFAULT_SAMPLE)
    opt.update(model.get("default_sample_options", {}))
    opt.update(options)
    for name in ["group_prior", "marginal_prior_noise_multiplier", "prefecture_prior"]:
        if not math.isfinite(float(opt[name])) or opt[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    strength = float(opt["public_correlation_strength"])
    if not 0 <= strength <= 1:
        raise ValueError("public_correlation_strength must lie in [0,1]")
    radial_strength = float(opt["public_radial_strength"])
    if not 0 <= radial_strength <= 1:
        raise ValueError("public_radial_strength must lie in [0,1]")
    rng = np.random.default_rng(seed)
    effective_bounds = _bounds(model["schema"])
    if opt["enforce_public_support"]:
        for col, limits in model["public_support"].items():
            effective_bounds[col] = [max(effective_bounds[col][0], limits[0]),
                                     min(effective_bounds[col][1], limits[1])]
    public_counts = np.asarray(model["public_group_counts"], float)
    gp = public_counts / public_counts.sum()
    group_total = PRIVATE_N + float(opt["group_prior"])
    group_mass = _simplex(_released_float(model["dp_group_counts"]) +
                          float(opt["group_prior"]) * gp, group_total)
    group_p = group_mass / group_mass.sum()
    counts = _allocate(group_p, n, rng) if opt["stratified"] else rng.multinomial(n, group_p)
    queries = {q["name"]: q for q in model["privacy"]["queries"]}
    chunks, target_groups = [], []
    for gid, (spec, count) in enumerate(zip(model["groups"], counts)):
        if count == 0:
            continue
        corr = strength * np.asarray(spec["public_correlation"], float) + (1-strength)*np.eye(len(NUMERIC))
        z = rng.multivariate_normal(np.zeros(len(NUMERIC)), corr, size=int(count), check_valid="raise")
        u = (rankdata(z, axis=0) - 0.5) / count if opt["stratified"] else norm.cdf(z)
        data = {}
        for j, col in enumerate(NUMERIC):
            margin = spec["margins"][col]
            ppublic = np.asarray(margin["public_probabilities"], float)
            alloc = float(Fraction(queries[f"marginal_{col}"]["epsilon_allocated_exact"]))
            scale = 2.0 / alloc  # mean of each exact geometric variable
            prior = float(opt["marginal_prior_noise_multiplier"]) * scale * np.sqrt(len(ppublic))
            ngroup = max(0.0, group_mass[gid] - opt["group_prior"]*gp[gid])
            total = ngroup + prior
            mass = _simplex(_released_float(margin["dp_counts"]), ngroup) + prior*ppublic
            p = mass / total if total > 0 else ppublic
            data[col] = _inverse_margin(u[:, j], p, margin)
        rare, sex, smoking, event = spec["group"]
        data.update({"sex": np.full(count, sex), "smoking": np.full(count, smoking),
                     "onset": np.full(count, int(event == 2)), "death": np.full(count, int(event == 1))})
        chunk = pd.DataFrame(data)
        chunk = _public_radial_adjust(chunk, spec, radial_strength,
                                      model["schema"]["rare_detector"], effective_bounds)
        chunks.append(chunk)
        target_groups.extend([rare] * count)
    frame = pd.concat(chunks, ignore_index=True)
    if opt["enforce_rare_partition"]:
        frame = _project_rare(frame, np.asarray(target_groups), model["schema"]["rare_detector"],
                              effective_bounds)
    for col, limits in effective_bounds.items():
        frame[col] = np.clip(frame[col], *limits)
    pp = np.asarray(model["public_prefecture_probabilities"], float)
    pmass = _simplex(_released_float(model["dp_prefecture_counts"]) + opt["prefecture_prior"]*pp,
                     PRIVATE_N + opt["prefecture_prior"])
    pc = _allocate(pmass / pmass.sum(), n, rng) if opt["stratified"] else rng.multinomial(n, pmass/pmass.sum())
    prefs = np.repeat(np.asarray(model["categories"], dtype=object), pc)
    rng.shuffle(prefs)
    frame["prefecture"] = prefs
    for col in ["age", "sex", "smoking", "onset", "death"]:
        frame[col] = np.rint(frame[col]).astype(int)
    return frame.iloc[rng.permutation(n)].reset_index(drop=True)[model["schema"]["submitted_c_columns"]]
