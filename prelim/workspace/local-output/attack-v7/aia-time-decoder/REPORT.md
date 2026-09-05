# AIA time decoder research report

## Decision

Use the existing v6 final horizon-edge `F_aia.csv` and change only seven
`time_hat` cells for target 20.  Keep its routed membership mask fixed.  For
those rows, use the released event time from a high-confidence mutual
nearest-neighbour link in separate-margin rank space, with margin quantile
`0.50`.

Do not apply the decoder globally and do not activate the four trusted target
20 links that are outside the existing active mask.  Leave targets 1 and 2,
all official-zero complement blocks, and every other target unchanged.

## Exact production rule

1. Read every public `C_k` and call `filter_linkable_releases` once across the
   complete release dictionary.
2. Call `copula_event_links(filtered["20"], aia_challenge_20,
   margin_quantile=0.50, horizon_time=10.0)`.
3. Align the existing target 20 block by `challenge_row_id`.
4. Define `active = old_time_hat < 11.0` and `apply = active & linked_gate`.
5. Set `time_hat[apply] = linked_time[apply]`; change nothing else.

The production audit finds 105 mutual links, 47 links whose nearest released
row is an event, 11 trusted event links, and seven trusted links inside the
existing 163-row active mask.  The seven changes are:

| challenge_row_id | old | new |
| ---: | ---: | ---: |
| 37 | 9.5 | 4.226513697922147 |
| 102 | 9.5 | 6.842896965107362 |
| 104 | 9.5 | 0.1402432388095489 |
| 114 | 9.5 | 3.627287354895649 |
| 243 | 9.5 | 3.161249780077052 |
| 255 | 9.5 | 5.624217697676642 |
| 290 | 9.5 | 7.588880289669058 |

## Validation result

The primary paired benchmark fixed the official per-target winner's member
selection.  Discovery and held-out validation used disjoint seeds, four
shadow targets per seed, and four mechanism-preserving scenarios: warp,
warp plus numeric noise, event-time marginal warp, and both perturbations.

For target 20, an exact-production-edge (`9.5`) replication improved all 32
discovery cases and all 32 held-out validation cases:

| Split | Mean target delta | p10 | Minimum | Negative rate | Mean rare delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | +0.08222 | +0.02125 | +0.00680 | 0% | +0.26970 |
| Validation | +0.09656 | +0.03349 | +0.02632 | 0% | +0.30855 |

If the leaderboard's AIA target aggregation is the arithmetic mean of 20
targets, the observed mean contribution to `score_1` is approximately
`+0.00411` in discovery and `+0.00483` in validation.  This is an estimate,
not a leaderboard guarantee.

The broader all-positive-target run used the nearby day-grid boundary
`9.498631074607` and independently obtained target 20 means of `+0.07686`
(discovery) and `+0.10659` (validation), again with zero negative cases.  The
conclusion is not sensitive to that edge convention.

Across all positive targets, applying q=0.50 direct globally had a non-zero
downside (validation minimum `-0.03425`, negative rate `5.53%`).  The strong
average was dominated by target 20, so global application is rejected.

## Fixed versus changed member selection

A separate paired benchmark compared the recommended fixed mask with also
activating trusted links previously marked `11.0`.  Expansion adds about six
rows per shadow target and would add four rows in production.

Under preserving scenarios, expansion was non-negative in all cases, but its
incremental held-out mean over the fixed-mask rule was only about `0.00702` at
the target level (roughly `0.00035` after division by 20).  It introduces a
second, unverified membership-selection change.  Both variants fail when
release event times are deliberately shuffled.  The small incremental upside
does not justify that extra risk, so the fixed mask remains the recommendation.

## Official-zero and rare sensitivity

Target 19 was evaluated separately while conditioning on its production probe
direction having exactly zero shadow power.  Its q=0.50 event overlay improved
every preserving validation case (`+0.09854` to `+0.11102` mean depending on
noise), including the rare channel.  But it worsened every time-shuffled case
(`-0.10181` validation mean) and contradicts the direct official score of zero.
It is therefore excluded.  The seven official-zero complement blocks remain
untouched.

## Other decoder findings

- Replacing every selected time by the censoring edge was strongly harmful;
  existing event-time predictions must be retained.
- Empirical-CDF/quantile mapping, constant shifts, and q=0.75/q=0.90 link
  thresholds were less stable or weaker than target 20 q=0.50 direct.
- Target-specific additions for 5, 12, 26, and 29 offered only about
  `0.0006` to `0.0011` further estimated total-score gain and relied more on
  marginal-warp assumptions.  They are excluded.
- Public observed times lie on an integer-day grid (`time * 365.25`).  Moving
  the censoring edge from 9.5 to the half-year boundary can cover one extra
  integer day, but shadow evidence showed no material gain and an added hit can
  increase the control rate as well as the member rate.  It is not bundled
  with the target 20 change.

## Downside and interpretation

The rule assumes target 20's public QI-to-event-time ordering survives the
release mechanism.  When event times are shuffled independently of QIs, the
fixed rule is harmful (held-out mean around `-0.113`).  This stress case is the
main lower-tail risk.  The recommendation limits exposure to seven cells on
the one target for which every mechanism-preserving discovery and validation
case improved.

No token was read, no network access was used, no submission was performed,
and these scripts do not write a production candidate or archive.

## Reproduction

From the repository root, using the starter virtual environment:

```powershell
$env:PYTHONHASHSEED = '0'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-time-decoder/benchmark_positive_decoders.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-time-decoder/benchmark_target20_membership_expansion.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-time-decoder/benchmark_zero_target19_decoders.py'
& 'C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe' 'local-output/attack-v7/aia-time-decoder/audit_production_target20.py'
```

The compact machine-readable decisions are in
`positive_decoder_report.json`, `target20_membership_expansion_report.json`,
`zero19_decoder_report.json`, and `production_target20_audit.json`.
