# Public influence regression: findings and limits

This study reads only team10's public `A_bg.csv`, schema and explicitly marked
public proxy caches. It never reads actual B, actual saved release values,
private scores or reference files. Regressions are fitted solely on the 4094
public background records complementary to each held-out public cohort.

The new estimator differentiates the weighted Efron Cox score with respect to
public case weights, including tied-event average weights. It regresses this
true public Cox influence onto the clipped integer row scores and conditions
on the known independent geometric noise covariance. An optional extension
adds the existing 24 rare/sex/smoking/event DP group counts. These are separate
postprocessing estimators and do not spend additional privacy budget.

## Four fixed public validation draws

| Estimator | Mean standardized beta L2 error | Mean official IO contribution |
|---|---:|---:|
| Public coefficients only | 0.43033 | 0.55452 |
| Frozen v2 fixed-hazard mapping | 0.36363 | 0.60107 |
| Public influence, clipped score | 0.36136 | 0.60309 |
| Public influence, score plus group counts | 0.37315 | 0.66481 |

The IO column uses the official significance/sign/interval-overlap formula and
public-information standard errors scaled to 1049 rows. It measures target
quality, not whole synthetic utility. Influence-only gains are very small and
split dependent. Group counts help interval conclusions on average but worsen
mean coefficient L2 error. These four draws do not establish a robust superior
release method. No replacement was frozen or applied to an actual candidate.

## Noise-free public control

Replacing the DP measurements by the exact clipped-score and group-count
statistics of the PUBLIC held-out cohort isolates the estimator's limiting
bias. It is an oracle control, never an actual release operation.

| Public cohort seed | Inputs | beta L2 error | Official IO with public SE |
|---|---|---:|---:|
| 2026091811 | Score only | 0.11063 | 0.83835 |
| 2026091811 | Score and counts | 0.11003 | 0.83927 |
| 2026091827 | Score only | 0.16454 | 0.79913 |
| 2026091827 | Score and counts | 0.15597 | 0.80129 |

Thus simply increasing epsilon for this same one-step target estimator does
not address its public noise-free error. A direct bounded Cox estimator and
explicit SE calibration merit investigation. The reported IO also depends on
public SE estimation; it is not an impossibility bound for every possible
synthetic-data method.

## Validation and status

`tests_test_influence.py` passes two tests: analytic case-weight derivatives
match refitted weighted Efron fits with explicit event ties; posterior
covariances are positive semidefinite; estimation is deterministic and does
not mutate parent models. `public_influence_study.py` reproduces the four-draw
comparison from fixed public caches. Machine-readable results are
`public_influence_study.json` and `public_noise_free_floor.json`.

The code remains an experimental public-only study, with no deployment freeze.
No epsilon 30/100/300 grid was run because the noise-free control already
showed that a different target mechanism should be investigated first.
