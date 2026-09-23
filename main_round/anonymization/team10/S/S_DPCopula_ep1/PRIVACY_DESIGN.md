# S_Copula: public-correlation-assisted DP Gaussian copula

This module is a Gaussian copula synthesizer with public correlation structure
and differentially private marginal measurements. It is **not** an implementation
of Li et al.'s original private Kendall-correlation DPCopula. Its architecture is
chosen to spend scarce private budget on the cohort's margins, categories and
event composition while using the authorized public background for dependence.

## Privacy contract

The private database contains the publicly specified **1,049 rows**. Neighbors
replace one complete row, keeping this size fixed. This is record-level bounded
(replace-one) epsilon-DP, with delta = 0. It is not an add/remove guarantee for
private dataset size, and it is not a multi-record/person guarantee when one
person can contribute multiple rows. All distributed columns except record_id
are processed; record_id never enters a query or the released model.

The threat model conditions on the public schema and **the main-round team10
A_bg.csv**. It assumes A_bg and its provenance are authorized public information,
independent of the protected release decision. The schema's ranges, rare
detector, allowed columns, binary domains, public prefecture category list,
ten-year follow-up horizon and n=1,049 are treated as fixed public constants.
An adversary seeing those constants, code, model and generated C is allowed.

Invalid numeric values are clipped or imputed row by row from public constants.
Unknown prefectures map to the first public category. This preprocessing does not
fit anything on private records. The guarantee applies to fixed-column inputs
within the declared size contract; schema/size violations are rejected before
the noise stage. The private B_self, utility_ref, tokens, evaluation results,
private empirical ranks, private minima/maxima and raw Cox fits are never inputs
to this module.

## Public copula and private queries

Twenty-four groups are specified publicly: majority/rare (public Mahalanobis
radius threshold 4), sex, smoking, and outcome (censored, death, onset). An
individual changing any attribute may cross groups. Each clinical feature has
a **single histogram-vector query over all groups**. We do not use an invalid
parallel-composition argument that assumes replacements stay in their group.

The public background defines per-group histogram edges and within-bin quantile
grids, fallback distributions for small public groups, and rank-normal Gaussian
correlations. These public statistics do not spend private budget. Private data
contributes only the following vectors, each with L1 replace-one sensitivity 2:

| Query | Fraction of this module's epsilon |
|---|---:|
| Group counts | 1/5 |
| Global prefecture counts | 3/20 |
| Seven clinical group-by-bin histograms, age included | 11/140 each |
| Time group-by-bin histogram | 1/10 |
| Total | 1 |

Each row increments one cell per query. Replacing it removes at most one count
and adds at most one count, so sensitivity is at most 2, including group
crossings. Histograms include clipping endpoints. Although the public group
partition depends on the row, the list of groups and bins is public and fixed.

The API `release_integer_vector` supports separately accounted integer queries
such as the companion DP Cox score. Its caller must prove its own sensitivity,
reserve its epsilon separately and compose it with this module. Allocating
epsilon to a Cox extension **in addition** to a full-budget call to `fit` would
exceed the intended total. A wrapper must instead split the total budget first.

## Exact integer noise

For an integer-vector query f with L1 sensitivity D and allocated budget e,
choose the exact rational q = D/(D+e). For each coordinate, independently draw
G1,G2 with probability P(G=k) = (1-q)q^k, k=0,1,..., and release f+G1-G2.
The noise probability at integer z is `(1-q)/(1+q) * q**abs(z)`. The ratio for
neighboring query vectors is bounded by `q**(-D)`, hence effective epsilon is
`D*log(1+e/D) <= e`. Sequential composition of the table above is at most the
requested epsilon. Budget arithmetic and Bernoulli probabilities use Python
Fraction/integer arithmetic, not rounded floating-point exponentials.

`secrets.randbelow(q.denominator)` implements the exact rational Bernoulli
trial; sampling has no cap, rejection based on private utility, fixed private
seed or disclosed private seed. Integer outputs use arbitrary-precision Python
integers, avoiding the finite support and overflow of float inverse-CDF Laplace
and fixed-width integer noise. Runtime is variable. This implementation does not
claim side-channel privacy against observing process timing, memory or failures.
Its mathematical DP proof uses the standard ideal-randomness/unbounded-arithmetic
model; on an actual computer the OS CSPRNG, finite resources and Python runtime
are implementation assumptions, not an information-theoretic physical source.

Extremely large DP integers are clipped to a public +/-10^12 interval before
floating-point postprocessing. This only occurs after noise and does not alter
the DP guarantee. The saved model retains the arbitrary-precision DP integers.

## Sampling and budget reuse

`fit(B, A_bg, schema, epsilon)` returns a JSON-serializable model containing
public statistics and DP integer measurements. No diagnostic raw private
statistics are attached. Each invocation is a fresh release and consumes its
assigned epsilon, including a release later discarded or overwritten.

`sample(model, n=1049, seed=...)` has no private-data argument. It projects noisy
counts to nonnegative totals, shrinks marginals toward public distributions,
draws Gaussian latent variables, applies inverse marginals, maintains valid
exclusive event codes, and enforces the public rare partition using generated
records only. Its ordinary sampling seed may be saved. Repeated sampling,
plotting, count projection, model serialization, public-prior changes and
postprocessing can reuse the same released model without additional privacy
cost, provided decisions depend only on public information or DP output.

The optional `public_radial_strength` maps generated within-group radius ranks
toward radius quantiles measured exclusively in A_bg. It can reduce an artificial
tail shell introduced by Gaussian approximations and the majority/rare boundary.
Its strength must be selected on public proxies or DP output. Public range
constraints cap radial rays, and final public-domain enforcement follows the map.
Fallback public fits retain the same majority/rare partition whenever A_bg has
at least 16 rows in that partition; private sample sizes never control fallback.

Public-empty histogram bins map to the nearest observed public value instead
of generating a uniform distribution across an otherwise empty schema-wide
tail. By default `enforce_public_support=True` also intersects output/radial
bounds with **public A_bg** minima and maxima. Both safeguards use public data
only; private clipping extrema are never fitted. These choices intentionally
trade away extrapolation: a true private-cohort tail beyond public observed
support may be suppressed. Public proxy checks exposed tail-score instability
without these safeguards, before any real private release was run.

Model selection must be frozen using public surrogate experiments or made from
DP output. Private B utility scores can be locally audited, but using those
unprotected scores to choose which C/model to release, iteratively tune it,
publish exact scores, or decide whether to retry is not covered by this proof.
The release runner must maintain a budget ledger across all private fits and
related mechanisms. For the authorized three total releases epsilon=1,3,10,
sequential composition on the same B is **epsilon <= 14, delta=0**; a component
inside each release receives only its share of that release's budget.

## Limits and verification

DP does not guarantee a particular contest anonymity score, utility >=0.9, or
that sensitive attributes cannot be predicted from population correlations.
This design intentionally preserves public correlations; AIA may still exploit
them. Public/prior mismatch, small event-by-tail groups, histogram noise,
projected counts and limited private dependence information may reduce utility.

Tests use fabricated records and public data only. They check exact budget
composition, the integer noise likelihood ratio, group-crossing sensitivity,
unbounded integer support, model output channels, no record IDs, deterministic
postprocessing, domain validity and extreme DP-count saturation. Tests are
engineering evidence and do not replace the mathematical sensitivity proof.

Reference: [Li, Xiong and Jiang, EDBT 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4232968/)
motivates separating private margins from copula dependence. Here dependence
is learned from public A_bg, and the exact integer count mechanism is proved
above rather than claiming to reproduce that paper's private estimator.
