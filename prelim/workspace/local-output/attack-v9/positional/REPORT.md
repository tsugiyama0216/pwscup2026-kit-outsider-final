# v9 positional/shuffle artifact audit

## Decision

**REJECT.**  After removing two feature-construction leaks found during the
audit, no transferable challenge-row or source-row ordering signature remains.
The primary leave-one-target-out test on the only exact-label holdout (target
24) is below chance and far below its stratified permutation threshold.  This
branch must not change the v7 candidate.

This result concerns construction position only.  It is not an evaluation of
the separate matched sex/age balancing work, and it is not a leaderboard score
forecast.

## Label provenance and data quality

| target | evaluation label | selected | proof strength |
|---:|---|---:|---|
| 1 | fixed 163-row multiview QI-to-release assignment | 163 | approximate public proxy; exact seven-QI matches = 0 |
| 2 | fixed 163-row multiview QI-to-release assignment | 163 | approximate public proxy; exact seven-QI matches = 0 |
| 24 | exact seven-QI match from self challenge into hash-gated own B | 163 | exact public-input label; 163 unique source matches, 0 ambiguous |

The target 1/2 assignment masks equal the historical v4 masks, but this is the
same attack lineage and is not independent validation.  Their labels are never
described as organizer truth.  `record_id` is table-local and was not joined
across tables.  Every challenge has 326 unique IDs, exactly `0..325` in file
order.

## Predeclared model and null

- Features were restricted to normalized challenge row position, unconditional
  nearest-source row position, their difference, fixed harmonics, and ordinal
  position within public sex and sex-by-five-year-age cells.
- Nearest-source position uses the same `argmin(pair_cost)` function for all 326
  rows.  Member assignments or exact labels never overwrite a feature.
- QI values, linkage distance, outcome/time, rare status, and nearest-source
  multiplicity are excluded.
- Classifier: fixed `StandardScaler` plus L2 logistic regression (`C=0.25`).
- Validation: leave one of targets 1, 2, and 24 out.  Target 24 is the primary
  holdout because its 163 labels are exact.
- Null: 1,000 independent label permutations within each target's public
  sex-by-five-year-age cells, preserving the member count of every cell.  The
  heldout p-value is also corrected using the maximum AUC over the three
  predeclared feature sets.

## Results

| held out | challenge-order AUC | source-order AUC | combined AUC | combined top-163 members |
|---:|---:|---:|---:|---:|
| 1 | 0.46110 | 0.51952 | 0.46084 | 75 |
| 2 | 0.51699 | 0.52260 | 0.52136 | 82 |
| **24 (primary)** | 0.49753 | 0.48195 | **0.48180** | **79** |

For target 24 combined, the stratified-null mean AUC is 0.50082 and its 95th
percentile is 0.55547.  The one-sided raw p-value is 0.7183, family-wise p-value
is 0.9081, and the null 95th percentile for recovered members is 89.  The
observed 79/163 does not clear any gate.  Combined AUCs are not positive in all
folds (`0.46084, 0.52136, 0.48180`).

Descriptive checks agree with the predictive rejection:

- Same-position label agreement is 0.4356 for targets 1/2 (phi `-0.1288`),
  0.5031 for 1/24, and 0.5215 for 2/24.  The only nominal association is in the
  wrong, target-pair-specific direction and does not recur on target 24.
- Member source-order to challenge-order sequence correlations are `-0.1172`,
  `0.0080`, and `-0.0164`; identical slots are 0, 1, and 1 out of 163.
- Rare-slot intersections are 1, 1, and 0; onset-slot intersections are 0, 1,
  and 2.  All are within their exact Monte Carlo nulls.

## Audit correction

An initial diagnostic appeared strongly positive because it included
nearest-source multiplicity and replaced source indices only on labeled member
rows.  Target 24 then compared exact unique own-B links for members with nearest
own-B collisions for controls, which is a direct linkage/label-construction
artifact rather than a positional signature.  That run is invalid and is not
used anywhere above.  Both mechanisms were removed before the locked 1,000-null
run: multiplicity is absent, and all rows use the identical unconditional
nearest-source mapping.

## Limits

Targets 1 and 2 provide approximate assignment labels, and only target 24 has
exact labels, so this is a small construction-instance audit.  A target-specific
artifact absent from all three can never be excluded.  Conversely, the primary
negative heldout result is sufficient to reject deployment: there is no
defensible cross-target positional effect to transfer.

## Reproduction and artifacts

Run with the repository venv:

```powershell
python -m attack_phase_v9.positional_audit `
  --dist C:\Users\takut\Downloads\PWSCup2026_prelim_team24_アウトサイダー_outsider `
  --permutations 1000
```

The script uses the mandatory hash-gated actual-input loader.  Five focused
unit tests pass.  Main artifacts are `audit.json`, `loto_results.csv`,
`permutation_null.csv`, `position_stats.csv`, `pairwise_position_stats.csv`,
`member_order_stats.csv`, and `outcome_position_stats.csv` in this directory.
No production CSV, archive, token, or submission was touched.

