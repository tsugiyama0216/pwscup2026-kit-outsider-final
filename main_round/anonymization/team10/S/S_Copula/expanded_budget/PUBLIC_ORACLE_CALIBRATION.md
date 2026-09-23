# Joint Cox coefficient and standard-error calibration

`target_calibration.calibrate_target(synthetic, target, public_target,
detector, ...) -> (C, trace)` accepts saved DP targets named
`beta_standardized` and `standard_errors_standardized`, in the units defined
by `public_target.center` and `public_target.scale`.

The optimizer permutes `(time, onset, death)` triples within the actual
synthetic rare/nonrare partition. All predictor rows remain unchanged.
Therefore predictor marginals, categorical distributions, correlations,
Mahalanobis radii and complete predictor geometry remain exactly unchanged.
Each stratum also retains its complete multiset of survival triples and
censoring logic. This costs no additional privacy budget.

## Objective and derivatives

At the DP coefficient target, compute the exact Efron partial-likelihood score
S and observed information I using the actual synthetic risk sets. The
objective combines mean squared `(I^-1 S)/target_SE` with mean squared
`log(sqrt(diag(I^-1))/target_SE)`. Both terms have weight one by default.

The score Jacobian and information derivatives guide pair-swap proposals.
For a displacement h of one predictor row, the information derivative is
`h B^T + B h^T + (beta^T h) C`; B and C are computed from cumulative Efron risk
moments, including tied event sets. Candidates are ranked using approximated
new coefficient and log-SE errors. Every accepted swap must improve the
objective under exact Efron reevaluation. Final diagnostics use an independent
unpenalized lifelines fit on the resulting synthetic table.

The target information is evaluated at the target coefficient, so its SE
approximation becomes accurate as the score approaches zero. Fixed predictor
distributions and survival multisets can make an arbitrary noisy SE target
unattainable; the algorithm does not promise exact SE matching.

## Public oracle control

The two existing public A_bg cohort seeds 2026091811 and 2026091827 each reserve
1049 public records as a proxy and use the disjoint remainder for nuisance
models. Starting synthetic tables come from the frozen v2 construction using
the corresponding **public** noise-0 cache. The oracle targets are exact
unpenalized beta and SE fitted to these PUBLIC held-out records. They are only
a noise-free capability control and are never private release targets.

| Public cohort | Official U_spec | beta standardized L2 error | log-SE RMSE |
|---|---:|---:|---:|
| 2026091811 | 0.9628695 | 0.0053043 | 0.0900104 |
| 2026091827 | 0.9715620 | 0.0168254 | 0.0663235 |

Both controls exceed 0.9. The previous coefficient-only optimizer also exceeded
0.9 on these same controls (0.96363 and 0.96504). Joint calibration improves the
coefficient residual and explicitly handles the released SE target; its first
cohort U_spec is slightly lower and the second is higher. The evidence does
not support claiming universal superiority over coefficient-only calibration.

These are Cox-facet results, not whole-utility or actual-private performance.
Fresh finite-epsilon DP target noise, whole-estimator projection, and changes
in other utility facets remain separate concerns for the parent's public
budget study.

## Validation and isolation

`test_target_calibration.py` passes two tests: analytic information derivatives
match central finite differences with tied events; calibration is
deterministic, does not mutate its input, preserves static rows and per-stratum
survival triples exactly, and has a strictly improving accepted objective.
`test_target_calibration.py --oracle` reproduces the two oracle controls and
writes `public_oracle_calibration.json`.

No actual B/B_self, actual saved-release numeric values, private reference,
private score or R files were read for this work. No actual target measurement
was taken, and no v2 or frozen core file was modified. The production function
reads no files and accepts no private fitting or scoring callback.

## Direct bounded-target review

The separately implemented `dp_cox_target.py` projects the entire 18-coordinate
beta/SE estimator into a fixed public integer L1 ball. Its replace-one global
L1 sensitivity is at most twice that radius in integer coordinates, even if
the unpenalized fit is unstable. Its constant fallback is in the same range
and receives the same DP noise. These properties support the stated global
sensitivity construction independently of asymptotic Cox stability.

Public expected-SE normalization matches the units of the clinical coefficient
error, while log-SE coordinates ensure positive decoded errors. Statistical
limits include clipping when the true estimator is far from its public
reference, public normalization mismatch, and multiplicative SE noise.
Approximately, a beta coordinate has DP noise standard deviation
`sqrt(8) * radius / epsilon` in public-SE units. Thus a large epsilon may be
needed even though the mechanism has a valid pure-DP accounting proof.
