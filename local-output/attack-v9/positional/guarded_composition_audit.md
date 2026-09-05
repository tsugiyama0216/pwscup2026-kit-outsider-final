# Guarded whole-target composition audit

## Decision

**The construction and floor argument are sound, subject to one precise
interpretation:** copy all 13 positive-target blocks byte-for-value from the
measured v4/independent/routed target winner, copy only the seven all-measured-
zero blocks (`6,7,8,9,17,18,19`) from v7, and use the exact v4 MIA file.

This is a conservative alternative to v7, not evidence that it dominates v7.
It restores all unmeasured `10.0 -> 9.5` horizon shifts on positive targets and
restores measured routed target 20 instead of the unmeasured seven-row v7 event
decoder.  It changes 1,544 rows versus v7, on targets
`3,4,5,10,12,13,20,22,23,26,29`; targets 1/2 already equal their measured
winners.

## Exact whole-block recipe

| source | targets |
|---|---|
| v4 | 1, 2, 4, 5, 12, 29 |
| independent | 3, 13, 22, 23, 26 |
| routed | 10, 20 |
| v7 zero-target research blocks | 6, 7, 8, 9, 17, 18, 19 |

Ties follow the already locked v6 winner recipe.  Every measured winner block
in `attack-candidate-v6-final-feedback-winners/F_aia.csv` was rechecked against
its named source and matched exactly.

## Score aggregation and lower bound

The official scorer computes a clipped nonnegative score independently for
each target and averages the 20 values.  Therefore changing a zero-target block
cannot reduce another target, and each unmeasured v7 zero block contributes at
least zero.

The 13 exact measured winner blocks contribute AIA power
`0.15062041254150674`.  The byte-identical v4 MIA power inferred from the
displayed independent-probe total is `0.1579114879230253 +/- 0.00005`.
Consequently:

- nominal total before any zero-target gain: `0.30853190046453205`;
- rounding-aware lower endpoint: `0.30848190046453206`;
- rounding-aware upper endpoint: `0.30858190046453204`.

This is about 0.00803 above the upper rounding endpoint of the latest displayed
0.3004.  The statement is a reconstruction for the fixed official truth and
fixed scorer, not a forecast from shadows.  It relies on the competition's
score function remaining unchanged, as indicated by identical feedback
constants.

There is no double counting: the AIA value is the mean of the 20 per-target
scores, while the MIA value is added once.  The target-20 measured routed score
`0.05208333333333332` is already included in the `0.1506204125` AIA value and
must not be supplemented with the v7 decoder estimate.

## Integrity checks

- v4 MIA current SHA-256: `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`.
- measured winner AIA current SHA-256: `E1D57A40DA372FC8B14E3129E386D0265170CDA7BA06EF0544F8319FF189F465`.
- v7 AIA current SHA-256: `65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0`.
- In-memory hybrid canonical AIA SHA-256:
  `63944D5FAA2C08E5E2E40415FFBB16B3153B27EBDD052CD68D7ADB59A403A6E0`.
- Rows: 6,520; keys complete; official logical validator: PASS.
- Active row counts retain v7's `217,253,211,274,223,236,268` on the seven
  zero targets and 163 on every measured positive target.
- All three feedback JSONs, all three measured AIA sources, v4 MIA, and the
  target manifest match the existing frozen hashes.

This audit constructed the hybrid only in memory.  It did not write a candidate,
make a ZIP, read the token, or submit.

