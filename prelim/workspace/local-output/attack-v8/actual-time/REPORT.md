# AIA actual-time mechanism audit

## Decision

Reject every new time-model change tested here. In particular, do not add the
target-3 q90 event links to v7. Keep the existing v7 target-20 seven-cell time
decoder unchanged.

The target-3 rule is normal-format and safely limited to abstention cells, but
it has no positive actual-data evidence. The predeclared gate requires positive
mean delta, at least 60% wins, nonnegative p10/all-channel delta, and no actual
background false success. The measured result was mean delta `0`, win rate
`0%`, and one home-prefecture A_bg false success in the repeated 326-row stress.

No v8 score, 0.5 score, or leaderboard improvement is claimed. The only
measured official score used here is the current `0.3004`.

## Structural correction tested

Every public AIA challenge is restricted to one home prefecture. Target 3 has
326 Kanagawa rows, while C3 contains 382 Kanagawa rows plus nationwide rows.
The self target has 326 Tokyo challenge rows and 386 Tokyo C24 rows. The legacy
copula decoder computed release ranks over the nationwide mixture and only then
matched inside prefecture/sex, so its release and challenge CDF coordinates
were structurally misaligned.

The new decoder computes ranks within the challenge prefecture, retains exact
prefecture/sex matching, requires mutual nearest neighbours and a released
event, and keeps only the top q90 distance margins. A prefecture+sex CDF variant
was also tested. Both preserve every active v7 prediction and can only replace
exact `11.0` abstentions with a public released event time.

## Actual-data validation

The eight-world experiment uses the fixed public C24 release against samples
from the actual B24 and actual A_bg, with the official 163-member/163-control
composition and matched home-prefecture controls. It is not a synthetic
B-to-C3 warp. The independent v7-like active mask is preserved before the
addition is evaluated.

| Method | Mean target delta | Win rate | p10 | Added member hits | Added control hits |
|---|---:|---:|---:|---:|---:|
| Legacy global copula q90 | 0 | 0% | 0 | 0 / 3 gated | 0 / 1 gated |
| Prefecture-CDF copula q90 | 0 | 0% | 0 | 0 / 5 gated | 0 / 2 gated |
| Prefecture+sex-CDF copula q90 | 0 | 0% | 0 | 0 / 9 gated | 0 / 0 gated |
| Home-block kNN window mode q95 | +0.001126 | 12.5% | 0 | 2 / 24 gated | 0 / 14 gated |
| Home-block ExtraTrees q98 | -0.001106 | 0% | -0.002654 | 0 / 4 gated | 1 / 14 gated |
| Global conditional hazard q90 | -0.001065 | 12.5% | -0.002654 | 1 / 36 gated | 5 / 48 gated |

The small kNN mean came from one of eight worlds and fails the 60% win gate.
At q98 it produced zero incremental hits. Smooth conditional models increased
control coincidences and had negative mean deltas.

On the actual self challenge, all 163 member identities and times are known.
Across global, prefecture, and prefecture+sex mutual event links, sweeping the
margin threshold from q0 through q90 produced zero correct member times within
the official ±0.5 window. At q90, the prefecture decoder gated three rows, one
known member, and zero correct member times.

A second audit removed all 163 known self-challenge members from B before
forming eight new query worlds from the remaining 231 home-prefecture B rows.
Global q90, prefecture q90, prefecture+sex q90, and kNN q95 all had mean delta
`0` and win rate `0%`. KNN found six donor times that were within tolerance, but
the unchanged baseline already succeeded on those rows, so it added no score.

The weak-release checks retained only 75%, 50%, or 25% of C24 donor evidence.
The target rule still had zero incremental success. Event-time shuffling also
gave zero, because the gated abstention cells had no successes even before the
shuffle. These zeros are absence of attack signal, not evidence of a robust
gain.

## Background calibration and downside

Actual A_bg calibration was restricted to target 3's home prefecture. Two
non-overlapping 163-row blocks gave zero false successes among four
prefecture-q90 gates. Eight 326-row without-replacement stress samples gave one
false success among 29 gates. Because only 389 actual Kanagawa background rows
exist, the 326-row stress samples necessarily overlap; they are reported
separately from the disjoint blocks.

The C24 result cannot prove that C3 uses the same mechanism. It does show that
the proposed record-level inverse is not validated on the only public release
for which private member times are known. With one official attempt remaining,
changing four or five C3 abstentions on that evidence has an asymmetric lower
risk and is not justified.

## Production impact if forced (not recommended)

The prefecture-CDF candidate would change four target-3 rows only:

| challenge_row_id | v7 | proposed |
|---:|---:|---:|
| 105 | 11.0 | 2.872507143978301 |
| 118 | 11.0 | 5.579579552242734 |
| 252 | 11.0 | 3.258708207248944 |
| 322 | 11.0 | 8.741235354882003 |

The legacy global candidate changes those four plus row 267 to
`1.9428355105061923`. Both CSVs exist only as research artifacts in this
directory. The v7 F_aia and F_mia hashes remain respectively
`65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0` and
`D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`.

## Integration audit

`attack_phase_v8.actual_inputs` requires the four exact actual-data hashes,
does not permit a practice-data fallback, loads 1,049 B rows, 5,143 A_bg rows,
20 attack targets, 163 known private rows, and 24 rare rows, and does not
enumerate or read the token. `attack_phase_v8.event_additions` validates exact
integer keys and finite times, filters the complete target set, protects targets
1 and 2, and modifies exact `11.0` cells only. Its documentation correctly makes
no private-score claim.

The focused integration suite passed 10/10 tests:

```powershell
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' -m unittest attack_phase_v8.tests.test_actual_inputs attack_phase_v8.tests.test_event_additions -v
```

Reproduce the time audit with:

```powershell
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v8/actual-time/evaluate_target3_q90_actual.py'
```

Primary machine-readable outputs are `target3_q90_actual_report.json`,
`target3_q90_actual_validation_summary.csv`,
`target3_discovery_disjoint_summary.csv`,
`target24_known_member_link_curve.csv`,
`target3_actual_bg_disjoint_controls.csv`, and `integration_audit.json`.

