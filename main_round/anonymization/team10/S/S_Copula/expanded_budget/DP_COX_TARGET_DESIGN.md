# Whole-table bounded Cox coefficient / standard-error target

`dp_cox_target.py` releases a jointly protected 18-dimensional target: the nine
Cox coefficients and their nine fitted standard errors. It replaces the
one-step score approximation with a numerically fitted full-cohort Cox target,
then protects the entire estimator by a fixed public range bound. It does not
assume that the raw Cox estimator is stable under replacing one record.

No actual protected cohort has been fitted by this implementation task. All
tests used A_bg public rows or mocked numerical fits/noise. Existing core,
postprocessing_v2, and previous release artifacts are unchanged.

## Public constants and estimator

`build_public_target(public, public_n=1049)` standardizes the nine features using
public means/scales and fits an unpenalized Cox model using lifelines/Efron ties.
The features are age, sex, BMI, SBP, log TG, HDL, log ALT, FPG, and smoking.
Onset is the event; death and other non-onset endings are censored.

The public center is the fitted coefficient vector `beta0`. The expected
cohort standard error is

`expected_se = public_fit_se * sqrt(public_rows / public_n)`.

This scale is an approximation for utility purposes, not a sensitivity
assumption. The defaults are public `R=16`, `Q=65536`, and `se_weight=2`.
The caller must select/freeze these and epsilon using public studies before
the protected invocation.

The protected estimator fits coefficients `beta` and standard errors `se`
on the fixed-size sanitized cohort and forms

`x = concatenate((beta-beta0)/expected_se, se_weight*log(se/expected_se))`.

It projects x into a public L1 ball of radius R, scales by Q/R, rounds to
Python integers, and repairs any rounding excess by deterministic subtraction
from the largest absolute coordinates. The final integer vector h therefore
has **exact integer L1 norm at most Q**. Floating-point projection error cannot
invalidate that final integer bound. Invalid/nonfinite fits, warnings and
exceptions all map to the same public zero vector, which is inside the ball.
The fallback is followed by the same noise invocation. No failure flag,
raw coefficient/SE, private norm, or clipping statistic is returned or logged.

## Pure DP proof

Fix all public parameters and public preprocessing. For any pair of fixed-size
replacement-neighbor tables D and D', the integer outputs satisfy

`||h(D)-h(D')||_1 <= ||h(D)||_1 + ||h(D')||_1 <= 2Q`.

This is a range proof. It holds even if changing one row changes every raw Cox
coefficient arbitrarily, changes convergence, or switches to the public
fallback. No private Hessian bound, per-row decomposition, local sensitivity,
or differentiability is needed.

The existing `dp_copula.release_integer_vector` is called exactly once with
epsilon and sensitivity S=2Q. Each coordinate receives an independent
difference of two exact geometric variables, with rational

`q = S/(S+epsilon)`.

The integer noise probability is proportional to `q**abs(z)`. Hence the
likelihood ratio between neighboring table outputs is at most

`q**(-S) = (1+epsilon/S)**S <= exp(epsilon)`.

Thus the released vector is **epsilon-DP with delta=0** under fixed-size
replace-one adjacency. Subsequent conversion, clipping and coefficient/SE
reconstruction are pure postprocessing. Quantization is not a source of delta.
The protected fit's finite-machine behavior is included in the bounded map;
the proof relies on the existing exact rational-Bernoulli, unbounded-integer
noise kernel and its randomness assumptions. Timing, process crashes, resource
consumption, or separately published logs are not additional protected outputs
of this mathematical mechanism.

The runner must reserve this new expenditure before fitting and compose it
with all prior releases. It must not infer a new size, suppress output or retry
because of a private fit result, publish private diagnostics, or select the
new output/seed on private utility. A second protected invocation spends again.
Using this new target together with the prior epsilon-14 releases has total
accounting epsilon **14 + new_target_epsilon**, unless another newly budgeted
component is also introduced.

## Reconstruction and API

The DP integer vector is kept exactly in `noisy_target_integer`. Only its
floating conversion saturates at the fixed public limits ±10^12. The decoded
beta coordinates are clipped to ±R expected-SE units. Reconstructed SE ratios
are clipped to the fixed public interval [0.1, 10]. These limits are
postprocessing and make extreme noisy values finite under the intended public
settings. They do not release the actual clipping status.

```python
from dp_cox_target import build_public_target, measure_target

public_target = build_public_target(sanitized_public, public_n=1049)
# Freeze public_target/settings, reserve epsilon, and sanitize B row-locally
# using fixed public schema bounds in the controlled runner before this call.
released = measure_target(
    sanitized_fixed_size_cohort, public_target, epsilon,
    radius=16, quantization=65536, se_weight=2,
)
```

Important returned fields are `beta_standardized`,
`standard_errors_standardized`, `beta_original_units`,
`standard_errors_original_units`, `noisy_target_integer`, `mechanism`,
`epsilon_exact`, `delta`, and `public_target_sha256`. The public model stores
`center`, `scale`, `beta_standardized`, and `expected_se_standardized`.

`postprocess_target(noisy_integer, public_target, epsilon=..., ...)` supports
reusing an already released vector. Its epsilon argument labels the original
measurement budget; calling this function does not make another measurement.

The optional `noise_callable(values, epsilon, sensitivity)` injection is for
public tests and returns `(noisy_integer_vector, mechanism_metadata)`, matching
the existing exact kernel. Production should use the default kernel. Internal
`_fit_summary` and `_bounded_integer_target` expose sensitive intermediates and
must never be called to create unaccounted protected artifacts or diagnostics.

## Verification

`test_dp_cox_target.py` passed all six public-only tests:

1. Exact integer L1 bounds across small/default Q, huge positive/negative
   vectors, and random inputs; pairwise global bound and attained 2Q endpoint.
2. Numerical failure/warning maps to zero before the same single noise call,
   with no warning/exception text or data-dependent diagnostic output.
3. Public-proxy beta/SE reconstruction without noise is accurate to the
   quantization bound and leaves input rows unchanged.
4. The production-default exact noise function is invoked once with S=131072.
5. Unbounded integer DP draws retain exact storage and reconstruct finite
   targets through public saturation/clipping.
6. Invalid public settings or violating the fixed public row count are rejected
   before any noise call.

These verify implementation/privacy invariants, not an actual-cohort accuracy
or utility threshold. Radius, budget and final calibration quality are to be
checked in the separate public-only study before any protected invocation.
