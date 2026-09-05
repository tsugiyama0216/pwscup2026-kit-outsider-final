# Positive-target mask research, v8

Decision: **do not replace the v7 positive-target masks** with any tested rule.

The new implementation is `benchmark_positive_masks.py`. It builds noisy
releases retaining 0%, 5%, 15%, or 35% of a known source cohort, with independent
background replacements. This avoids relying only on near-perfect full-source
shadows. For each shadow target, the baseline is the whole-target winner among
v4, independent, and a routed consensus; its existing event estimates are kept
and exact 10.0 predictions shifted to 9.5, matching the horizon-edge baseline.
Targets 1, 2, and 20 are not proposed for production changes by this study.

We implemented coverage rules using the union, majority, intersection, a
two-expert union, and all rows, with separate expert-time variants. Each keeps
the baseline's surviving active time estimates. The rule is applied uniformly;
no official individual membership or outcome label is inferred or used.

## Paired results

The discovery winner was `all_rows`, locked in `locked_rule.json` before fresh
validation. The weak-positive subset has baseline target power in [0.02,0.25].
These are target-level differences, **not leaderboard forecasts**.

| World / split | Independent seed groups | Weak-positive observations | All-rows mean difference | Wins | 10th percentile |
|---|---:|---:|---:|---:|---:|
| B_practice / discovery | 8 | 75 | +0.062786 | 74.7% | -0.078973 |
| B_practice / fresh validation | 8 | 75 | +0.100313 | 80.0% | -0.044476 |
| Fresh A_bg cohort / stress validation | 8 | 68 | -0.004660 | 44.1% | -0.144802 |

In the stress world, the member cohort and reference/control donor populations
are disjoint random subsets of A_bg instead of a single fixed B_practice cohort.
The all-rows mean difference over **all 128** stress observations is -0.040238,
and over its 123 baseline-positive observations is -0.046354. Thus the initial
gain is not robust to changing the source population. It cannot justify risking
the final submission. The other six rules also have negative weak-positive
mean differences in fresh original and stress validation.

## Caveats and integrity

- 384 observations = 24 seed groups × 4 challenge samples × 4 release fractions;
  the observations within a seed are correlated, not 384 independent trials.
- The best-of-three baseline uses shadow truth only to reproduce whole-target
  feedback selection. Selection of a low-score subset changes the estimand;
  unconditioned and all-positive results are reported alongside it.
- Control matching is the legacy approximate matcher followed by visible-block
  repair. Its source/controls are disjoint, but its reuse and sampling are not an
  exact simulator of the organizer's complete population. This is a sensitivity
  test, not a faithful reconstruction of hidden production truth.
- The background-cohort rare labels are the public detector's proxy labels.
- No production CSV, token, archive, submission, or network service was changed.
- Repository regression check after these additions: 118 tests PASS.

## Reproduce

Run with the project's configured Python:

```
python local-output/attack-v8/positive-mask/benchmark_positive_masks.py --split discovery --count 8
python local-output/attack-v8/positive-mask/benchmark_positive_masks.py --split validation --count 8
python local-output/attack-v8/positive-mask/benchmark_positive_masks.py --split validation --count 8 --world bg_cohort
```

The script saves all row-level *shadow scores*, denominators, predictions'
active counts, and summary tables beside this report. It does not save hidden
production labels or treat local scores as such labels.
