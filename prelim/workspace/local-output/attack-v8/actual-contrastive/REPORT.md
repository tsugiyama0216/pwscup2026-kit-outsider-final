# Actual-data contrastive zero-target audit

## Decision

**REJECT the discovery-selected `multitime_mean_expand` policy. Keep the v7
zero-target complement-of-unanimous blocks unchanged.** Every required gate
failed, including the nonzero-v7 protection subset and the exchangeable-world
stress test. No production candidate, ZIP, token, or submission was changed.

This is a posterior-predictive sensitivity analysis, not empirical validation
on another team's hidden truth and not a leaderboard-score forecast.

## Locked-policy results

The policy was selected once from discovery OOF simulations, then held fixed.
Results below condition on all three submitted direct attacks (v4,
independent, routed) scoring exactly zero and give each of the seven official
zero targets equal total weight.

| Evaluation stream | Accepted cases | Mean v7 power | Mean candidate power | Candidate − v7 | Win rate | p10 delta |
|---|---:|---:|---:|---:|---:|---:|
| Discovery OOF | 97 | 0.04813 | 0.08150 | +0.03338 | 55.46% | -0.05706 |
| Held-out imputation | 54 | 0.12621 | 0.11094 | **-0.01527** | 43.45% | -0.19814 |
| Fresh imputation | 78 | 0.17823 | 0.16647 | **-0.01176** | 42.86% | -0.10669 |
| Exchangeable A_bg stress | 123 | 0.09441 | 0.07484 | **-0.01957** | 34.62% | -0.13039 |

The baseline-positive exact-zero subset is worse as well:

- held-out: mean delta **-0.05909**, loss rate 70.32%, p10 -0.20166;
- fresh: mean delta **-0.01269**, loss rate 58.12%, p10 -0.10938;
- exchangeable: mean delta **-0.04022**, loss rate 68.46%, p10 -0.13784.

If the rejected policy were applied to all seven zero targets, its fresh
posterior mean maps mechanically to `7/20 × -0.01176 = -0.00412` in the
score_1 direction. This is only a sensitivity delta, not an LB prediction.
The official rare dead-zone (`z=1.64`) was used by the exact scorer throughout;
the fresh mean rare-component delta was +0.00809, but it did not offset the
negative all-group power or rescue the failed overall/downside gates.

## Evaluation design

- Inputs came through the fail-closed actual-data loader with the four required
  hashes and target manifest checks. Practice-data fallback was disabled and
  the token was not read.
- Actual public challenge rows, releases, submitted three-vote patterns, event
  geometry, density, and background donor-occupancy features were fixed.
- Within every public `sex × five-year age band`, half the rows were drawn as
  members, preserving 163 members and 163 controls. Uniform assignment within
  those blocks is a modelling assumption. The separate actual-self classifier
  did not establish useful discrimination, but failure to find discrimination
  does not prove uniform membership and is not validation of this policy.
- Every imputed arm preserved the published 13 rare, 16 onset, and 4
  rare-onset counts. Member outcomes came from the hometown/private empirical
  reservoir; controls came from hash-separated actual-A_bg donor folds and
  were assigned conditional on public QIs.
- The learned action maximized the Dinkelbach utility
  `q_m(x,t) - (1-R) q_c(x,t)` for `R=(p_m-p_c)/(1-p_c)`, comparing times 9.0,
  9.5, 10.0, and abstention at 11.0. This directly targets member-minus-control
  window success rather than generic classification accuracy.
- C18/C19 public background-copy geometry was not regenerated: the actual
  release rows and inverse-Voronoi occupancy/count features were used directly.

## Limitations and risk interpretation

All streams reuse the same actual public feature rows. Membership and hidden
time/rare outcomes are imputed from common empirical reservoirs. Therefore,
“held-out” and “fresh” mean disjoint Monte-Carlo streams and control-donor
folds—not held-out QIs, real hidden outcomes, or independent private cohorts.
The 13 known-private rare rows must overlap discovery and later streams because
the remaining Tokyo cohort contains only one rare row and no rare-onset row.

These limitations make a positive result insufficient for promotion. Here the
result is negative even before applying that evidentiary discount. The proposed
research mask would change 1,450 of 2,282 zero-target rows (130–243 per target),
so it is also too expansive for a last-submission gamble. Keeping v7 avoids
adopting this unsupported change; it does not establish a leaderboard score
floor for the still-unsubmitted v7. The rejected candidate has materially
negative tails even when v7 itself is positive in these imputed worlds.

## Artifacts

- `report.json`: gate decisions, provenance, population diagnostics, and QA.
- `summary.csv`: target-equal summaries for every predeclared policy.
- `scores.csv`: case-level exact official-score comparisons.
- `block_conditions.csv`: direct-score conditioning inventory.
- `production_research_mask.csv` and `production_changes.csv`: normal-format
  research-only transformation, retained solely for audit and explicitly not
  recommended for production.
- Source: `../contrastive-zero/benchmark_contrastive_zero.py`.

Normal-format QA for the research mask passed (2,282 unique keys, finite
nonnegative times, values only in `{9.0, 9.5, 10.0, 11.0}`), but format validity
does not override the failed efficacy gates.
