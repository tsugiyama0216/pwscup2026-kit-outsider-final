# Final zero-target AIA shadow benchmark

## Decision

**Keep the current exact complement of the routed active mask.** The local evidence does not support replacing it.

The binding evidence is the team's official result: targets 6, 7, 8, 9, 17, 18, and 19 scored exactly zero under v4, independent, and routed attacks. Therefore the large direct-direction scores in ordinary shadow worlds are known to be externally miscalibrated and are diagnostic only; shadow results do not override official feedback.

## Official-zero-conditioned sensitivity

The focused rerun generated 6 discovery and 6 validation null worlds. Only blocks where all three direct experts also scored exactly zero were retained. This reproduces the qualitative official condition, although it is a small post-selected sample.

| strategy | eligible_blocks | mean | p10 | p90 | mean_delta_vs_routed_complement | win_rate_vs_routed_complement |
|---|---:|---:|---:|---:|---:|---:|
| `complement_routed_active` | 4 | 0.108333 | 0.040447 | 0.166885 | 0.000000 | 0.0% |
| `inverse_consensus` | 4 | 0.105673 | 0.033000 | 0.166885 | -0.002660 | 0.0% |
| `complement_v4_active` | 4 | 0.085874 | 0.062086 | 0.103883 | -0.022458 | 25.0% |
| `complement_independent_active` | 4 | 0.061842 | 0.009184 | 0.108591 | -0.046491 | 0.0% |

Routed complement ranked first in discovery (10 eligible blocks; mean 0.099673) and validation (4 eligible blocks; mean 0.108333). Inverse consensus was close in validation (0.105673) but did not beat routed complement in any eligible validation block. The ≤0.02 sensitivity expanded validation to 7 blocks and preserved the order: routed complement 0.110717, inverse consensus 0.109197, v4 complement 0.081099, independent complement 0.063510.

## Unconditioned shadow diagnostic

Ordinary raw/noisy/time-shuffled/warped shadows strongly favor the direct direction because their releases preserve much more member linkage than the private production targets. Those numbers expose the domain gap and are not candidate-selection evidence.

| strategy | observations | mean | p10 | p90 | mean_delta_vs_exact | win_rate_vs_exact |
|---|---:|---:|---:|---:|---:|---:|
| `safe_v4` | 12.000000 | 0.937594 | 0.837686 | 1.000000 | 0.838013 | 100.0% |
| `calibrated_logit` | 12.000000 | 0.829519 | 0.816239 | 0.843895 | 0.729938 | 100.0% |
| `knn_all5_direct` | 12.000000 | 0.792889 | 0.709225 | 0.843295 | 0.693308 | 100.0% |
| `routed_probe` | 12.000000 | 0.682352 | 0.611277 | 0.722962 | 0.582771 | 100.0% |
| `exact_complement_routed` | 12.000000 | 0.099581 | 0.067100 | 0.122436 | 0.000000 | 0.0% |
| `inverse_consensus` | 12.000000 | 0.000000 | 0.000000 | 0.000000 | -0.099581 | 0.0% |

In particular, safe v4 scores about 0.94 in these shadows but is officially zero on every target in scope. The calibrated logistic mixture learned positive weights on v4/independent/routed and is therefore also rejected despite its high unconditioned validation mean.

## Method and QA

- Main benchmark: 3 discovery and 3 disjoint validation seeds; raw, 3% noise, time-shuffled, production-margin rank-warped, and independent-null scenarios.
- Focused conditional rerun: 6 discovery and 6 disjoint validation seeds under independent-null releases.
- Every challenge contains 163 members and 163 controls, matched on sex, prefecture, five-year age band, onset, and rare status.
- Every evaluated output gives exactly one finite prediction per unique `(target_k, challenge_row_id)` and uses the published half-per-visible-stratum constraint. No duplicate-key or fractional-ID behavior was tested.
- All world consistency checks passed.

## Caveats

- This is truth-known local method comparison, not a private leaderboard forecast.
- The exact-zero conditional validation sample is only four blocks; it can reject a clear alternative advantage, but cannot prove the routed complement is optimal.
- Conditioning on zero under an independent null introduces post-selection bias.
- The organizer's undisclosed transformation cannot be reconstructed locally.
- No production candidate, archive, token, or submission was changed.
