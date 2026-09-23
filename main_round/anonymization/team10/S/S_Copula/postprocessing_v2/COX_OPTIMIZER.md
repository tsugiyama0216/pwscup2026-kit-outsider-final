# Synthetic-only Cox optimizer

`cox_optimizer.calibrate(C, targetBeta, public_model, strata=rare)` rearranges
whole `(time, onset, death)` triples within the specified synthetic strata.
It leaves every predictor row unchanged. Thus all predictor marginal values,
categorical counts, pairwise correlations, Mahalanobis radii and their complete
joint distribution remain identical. Each stratum retains the exact multiset
of survival triples, including event/death counts, follow-up distributions,
and censoring logic. The default rare/nonrare stratification must be computed
from the already-synthetic table and the public detector.

The objective is the squared norm of the Efron partial-likelihood score at the
target coefficient, preconditioned by the initial synthetic observed
information. The risk sets come from the actual synthetic survival triples.
An analytic score Jacobian ranks candidate pair swaps using their approximate
new objective; each accepted swap is reevaluated with the exact Efron score.
Acceptance is strictly monotone. A deterministic seed controls proposals.
All nine Cox coefficients can change through survival assignment even though
age, sex, smoking and every continuous predictor remain unchanged rowwise.

Optional `target_information` must be public or DP, with the scale matching the
output row count. A nonzero `information_weight` adds squared log-SE error to
the acceptance objective; the default is zero because no exact private
information matrix has been released. The score Jacobian only proposes swaps,
so the optional information objective is evaluated exactly but is not itself
used as a proposal derivative. There is no promise of exact SE matching.

## Privacy and scope

The function reads no files and makes no measurements. Its only inputs are
synthetic C, the saved DP/public target, public nuisance parameters, and
public/synthetic options. Diagnostics also use only those inputs. For a joint
candidate depending on the epsilon 1, 3 and 10 releases, the inherited bound
is epsilon 14, delta 0; postprocessing costs no additional budget. It is not
valid to label that combined candidate epsilon 10. Exact private Cox
coefficients, private information, unprotected private utility or validation
results must never enter this API or choose its settings.

## Validation

`tests_test_optimizer.py` has two tests covering:

- Exact Efron observed information and score Jacobian against central finite
  differences, including tied event times; score near zero at the independent
  lifelines Cox fit.
- Determinism, no input mutation, identical predictor rows, identical survival
  triple multisets within strata, strict accepted-objective improvement, and
  coefficient error reduction on public A_bg data.

Both tests passed. `tests_test_optimizer.py --experiment` performs a single
public-only control experiment. Seed 471 splits public A_bg into 1049 proxy
records and disjoint public nuisance records; seeded public simulation 47100
creates a refined histogram synthetic table. Exact Cox coefficients computed
from the PUBLIC proxy are used only as an oracle control to measure optimizer
capability. They are never candidates for a private release target.

The official kit's U_spec improved from **0.6571976 to 0.9783521**; standardized
coefficient L2 error fell from **0.4585170 to 0.0050935**. Machine-readable
results are in `public_optimizer_experiment.json`. This checks the solver's
ceiling, not deployed utility: the saved DP target's estimation error remains
and may dominate. The parent pipeline independently checks its actual target
estimator and other utility facets on public proxy experiments.

As a separate synthetic-only residual check, the unchanged saved epsilon-10
C and its own saved DP Cox target had coefficient L2 error 0.1928345 before
calibration and 0.0050623 after 150 requested iterations. No private utility
was evaluated, and no existing saved C or model was overwritten.
