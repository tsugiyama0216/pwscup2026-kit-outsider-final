# DP histogram refinement: public-only evidence

`histogram_refine.py` performs deterministic postprocessing of the three saved
independent releases. Using all three has composed parent privacy **epsilon 14,
delta 0**, with **zero additional privacy expenditure**. The original releases,
frozen configuration and mechanism code are unchanged. This work did not read
actual B, B_self, private utility references, R models, or private diagnostics.

## Estimator

1. Check identical public specifications, group definitions, histogram edges,
   query identities, adjacency and public cohort size 1049. Reject a duplicated
   saved release. Independence/provenance of distinct releases remains the
   caller's responsibility.
2. Combine corresponding cells with inverse variance weights, using the exact
   geometric variance `2*q/(1-q)^2` recorded in each query.
3. Pool the independent direct group-count measurement with the sum of each
   corresponding feature histogram, weighted by its known noise variance.
4. Estimate nonnegative, mutually consistent group and marginal counts using a
   Gaussian posterior moment approximation. Its public covariance is
   `multiplier * n * (1+n/public_fit_n) * (diag(p)-p*p.T)`. The public finite-sample
   correction accounts for uncertainty in the reference probabilities. All
   probabilities and reference sample sizes come from the released public
   specifications. A fixed Jeffreys half-count per public group avoids treating
   a publicly unobserved group as impossible. The multiplier is **1.0**.
5. Return continuous posterior probabilities for synthesis, plus deterministic
   largest-remainder integer counts compatible with the frozen sampler.

This is an approximate Bayes estimator with exact noise variances, not an exact
geometric-likelihood posterior. The public multinomial covariance, conditioning
and nonnegativity constraints avoid the original estimator's large generic
pseudocount and the distortions from projecting highly noisy counts first.

## Public experiment

Only the 5,143 rows of A_bg and public schema were used. Each seed (471, 472)
defines a 1,049-row proxy cohort and a **disjoint** public remainder. The two
splits overlap each other, so six trials do not mean six independent cohorts.
For each split, three independent seeded geometric noise simulations create
epsilon 1, 3, 10 release proxies with the same 80% histogram allocation as the
saved release design. Simulation seeding is restricted to these public proxies;
the production mechanism's secret randomness is unchanged.

The comparator is the frozen epsilon-10 histogram sampler before its separate
Cox calibration, with marginal prior multiplier 15 and radial strength 0.
The refined estimator combines all three already-accounted releases. Each
estimator generates 10,490 synthetic rows with fixed sampling seed 917 to reduce
sampling error in event means and Cox estimates. Cox errors use the disjoint
public reference's feature standard deviations and log TG/log ALT.

| Metric, mean over six public trials | Frozen epsilon-10 histograms | Refined epsilon-14 histograms | Reduction |
|---|---:|---:|---:|
| Histogram cell count RMSE | 1.64826 | 1.57162 | 4.7% |
| Group count RMSE | 2.37132 | 1.73355 | 26.9% |
| Event-conditional standardized mean RMSE | 0.07137 | 0.06502 | 8.9% |
| Standardized Cox coefficient L2 error | 0.49984 | 0.38767 | 22.4% |

Multiplier 1 had the lowest histogram error among 0.5, 1, 2 and 4. Multiplier 2
slightly reduced Cox error but worsened histogram and event-mean errors, so the
theoretical multiplier 1 is retained. These are reconstruction measures on
public proxies, **not official utility scores or a claim of utility 0.9**.
Full per-trial values and standard deviations are in
`public_histogram_experiment.json`.

## API and checks

```python
from histogram_refine import refine_models, sample_refined

# saved_wrappers are the three existing release model.json dictionaries.
posterior = refine_models(saved_wrappers, prior_multiplier=1.0)
c = sample_refined(posterior, n=1049, seed=20260918,
                   public_radial_strength=0)
```

`posterior_group_counts`, `posterior_prefecture_counts`, and each group's
`margins[column].posterior_counts` / `posterior_probabilities` expose continuous
results for downstream targets. `histogram_postprocessing.parent_epsilon_exact`
records composed privacy. The original `privacy.queries` metadata is retained
for compatibility and describes the source query, **not the composed output's
epsilon**; use the enclosing postprocessing metadata for accounting.

Run `tests_test_histogram.py` for unit tests and add `--experiment` to reproduce
the public experiment. Three tests verify variance, consistency, deterministic
behavior, input immutability, valid synthesis, identity checks and duplicate
rejection. They passed on 2026-09-18.
