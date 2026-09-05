# Robust registration attack: no safe v9 promotion

## Technical summary

The new two-anchor-view affine ICP/RANSAC linkage is implemented and works on
controlled planted-anchor releases, but it is **not reliable enough for the final official
attempt**. The fresh validation gate failed: mean precision was 0.7032, its 10th
percentile was 0.5167, and independent null releases produced as many as 14
accepted rows. No non-ceiling public target passed the predeclared combination
of held-out-view, known-private, and actual-background gates. Therefore this
study changes zero production cells and recommends neither a MIA rerank nor an
AIA time overlay.

## Public releases do not show a new target that clears the null

Three independently sampled actual-A_bg releases yielded 2, 3, and 11 accepted
rows (mean 5.33). Against that reference, none of the 18 non-ceiling targets
passed both anchor-view held-out residual checks and the target-level excess-count
gate.

Targets 1 and 2 produced 16 and 21 accepted rows. Every one was already in the
top 5% of the v7 MIA column, so registration mostly recovers an existing strong
signal and supplies no new cell. These targets are protected in any case.

The largest superficially interesting non-ceiling results also fail the
cross-view checks:

| Target | Accepted | Below v7 top 5% | View-A residual ratio / p | View-B residual ratio / p | Decision |
|---:|---:|---:|---:|---:|---|
| 12 | 10 | 7 | 0.868 / 0.031 | 0.898 / 0.092 | Reject |
| 19 | 16 | 12 | 0.835 / 0.015 | 0.908 / 0.031 | Reject; known background-copy mechanism |
| 20 | 10 | 5 | 0.924 / 0.046 | 1.044 / 0.800 | Reject |
| 7 | 10 | 6 | 1.015 / 0.554 | 1.026 / 0.769 | Reject |

All known-private audit false-link counts happened to be zero for these actual
targets, but that check alone is insufficient: a target can look safe simply
because its fitted score threshold is conservative, while the apparent new
links remain indistinguishable from background or fail the other feature view.
Full target results are in `actual_target_diagnostics.csv`.

## Fresh planted-anchor validation is useful but fails the precision gate

Controlled worlds used the actual B24 and A_bg feature distributions, planted
26 truth-known B anchors into a 1,049-row independently sampled background
release, and then applied a fresh positive affine map plus 0.8%–6.5% feature
noise. Four discovery seeds were kept separate from eight fresh validation
seeds. Paired null worlds used the same marginal mechanism without planted B
anchors.

Across the eight fresh positive worlds:

- mean precision: 0.7032;
- precision 10th percentile: 0.5167;
- mean recall: 0.7500;
- known-private audit false-link rate 90th percentile: 0.0122.

Precision deteriorated as noise rose: individual validation precision ranged
from 0.4444 to 0.8947. Paired nulls accepted as many as 14 rows. The fixed gate
required mean precision at least 0.80, 10th-percentile precision at least 0.60,
mean recall at least 0.25, and no more than five null acceptances. It therefore
fails without post-hoc threshold tuning. Raw world-level results are in
`synthetic_worlds.csv`.

## The self-target AIA audit finds no correct event-time link

For the truth-known self challenge, the raw-to-C24 affine registration was fit
without treating the 163 self-private rows as negatives (they are part of the
self source). The challenge has 16 onset members plus 6 death-only members, for
22 members with either event. The age/SBP/TG view decoded one released event,
but it was on a control row; all 163 member rows were inactive, so correct
member times were zero under the official all-member-time definition as well as
for onset-only and any-event subsets. The HDL/ALT view decoded none, and the
correlated two-model agreement decoder decoded none. This is direct negative
evidence against using the registration as a new AIA time overlay.

## Scope and definitions

- Source inputs are loaded only through the hash-gated
  `attack_phase_v8.actual_inputs.load_actual_inputs` path: actual B has 1,049
  rows and actual A_bg has 5,143 rows.
- For attacked-target MIA calibration, the 163 self-private rows are guaranteed
  non-members. A deterministic split uses 81 for threshold calibration and 82
  untouched rows for the false-link audit.
- All 24 rare rows are excluded from fitting and promotion. This is a guard for
  the measured rare channel, not an assumption that the 11 remaining rare rows
  are shared or private.
- Targets 1 and 2 are never eligible for an overlay.
- The baseline comparison is the unsubmitted v7 MIA file. Public target results
  are mechanism diagnostics, not hidden target truth and not leaderboard-score
  estimates.

## Model and validation design

The implementation initializes a positive affine transform for each numeric
column from marginal quartiles. It then alternates mutual nearest-neighbour
anchor nomination with per-column RANSAC refits. Age/SBP/TG/ALT and
BMI/HDL/FPG are disjoint only when nominating the two tentative-anchor sets.
Each resulting affine model then computes its final neighbour using the same
full seven numeric columns and the same categories. Final model agreement is
therefore correlated corroboration, not two independent linkage tests. A row
can be accepted only when both fitted models choose the same release row, both
links are mutual, its score is above the 99th-percentile split-negative
threshold, and it is neither known private nor rare.

To catch circular pseudo-anchors, each model is audited on columns not used to
nominate its anchors. The validation-column affine map is fit on one half of the
tentative pairs and evaluated on the other half. Its scaled residual is compared
with 64 joint row permutations. Actual-A_bg cohort swaps provide a separate
null for the full pipeline.

The module also contains normal-format-shaped research helpers for a guarded
MIA overlay and an affine nearest-event decoder. The overlay refuses target 1/2
and enforces value preservation of known-private and rare rows. A complete
output from either helper still requires the separate official logical
validator. The AIA helper additionally requires target-transfer validation;
self-target evidence alone never authorizes use on another generator. These
helpers were not invoked on the v7 candidate because the evidence gates failed.

## Limitations and robustness interpretation

- Planted-anchor experiments establish behavior under the simulated affine
  mechanism only. They are not empirical membership truth for another team.
- C1/C2 consistency with high v7 ranks is not truth attribution; it only shows
  that the method can rediscover the same strong public geometry.
- Actual public targets can use nonlinear, row-dependent, suppressive, or
  synthetic mechanisms outside the affine model. Rejection does not prove that
  no registration attack exists.
- C24 is a truth-known self-mechanism check, not proof about C3 or any other
  team's generator.
- Actual-A_bg is the strongest public independent null available, but its
  transformation need not exactly match every target release. This is why the
  target decision also requires anchor-view held-out residual evidence.

## Recommended action

Do not add registered MIA rows or registered AIA times to the final candidate.
Retain the module and results as a bounded negative experiment, and keep the
final decision anchored to methods with target-level measured or stronger
held-out evidence. Any future AIA use requires validation on a genuinely
separate target mechanism, even if a future self-target check is positive.
There is no supported score_1 uplift to report from this method.

## Reproduction

```powershell
$python = "C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe"
& $python .\local-output\attack-v9\registration\benchmark_registration.py
& $python -m unittest -v attack_phase_v9.tests.test_registered_linkage
```

The detailed machine-readable decision is in `registration_report.json`; the
short production decision is in `candidate_plan.json`. No token, ZIP, network,
or submission path is accessed, and the read-only v7 baseline candidate is not
modified by the benchmark.

Final verification: all 157 repository tests pass; artifact assertions cover
20 public target rows and 24 controlled worlds. The v7 MIA/AIA SHA-256 values
remain `D4D66585...0128B` and `65AD898E...EAA0` respectively.
