# PWSCUP2026 attack v5 — Codex handoff

Last updated: 2026-09-04 (Asia/Tokyo)

## Purpose

Continue the authorized PWSCUP2026 preliminary attack phase from the current local repository without relying on the original chat history. The objective is to use the three remaining official submissions efficiently and aim for an attack score of 0.4. A score of 0.4 is an aspiration, not a locally validated prediction.

No CodaBench submission has been performed by Codex. Do not submit or push without an explicit user instruction.

## Repository state

- Remote: `https://github.com/tsugiyama0216/pwscup2026-kit-private.git`
- Working branch: `codex/attack-v5`
- Branch starting commit before v5: `dc0b01ac8672cf4d6d1e122b6894f084b89d9bf1`
- Python: `starter/.venv/Scripts/python.exe`
- v5 implementation: `attack_phase_v5/`
- Candidate CSVs: `submissions/prelim/team24/attack-candidate-v5-*`
- Token-bearing ZIPs and shadow outputs: `local-output/` (ignored by Git)

The repository may be opened at a different absolute path in a future Codex project. Use paths relative to the repository root. If this handoff is being read from the temporary ChatGPT-project mirror, add or clone the repository into a stable Codex project directory before long-term work.

### Portable local transfer

This session creates an ignored, token-free Git bundle at:

`local-output/codex-handoff/pwscup2026-attack-v5.bundle`

To make a stable local clone without pushing the branch first, run the following from the parent directory where the new repository should live:

```powershell
git clone -b codex/attack-v5 <absolute-path-to-pwscup2026-attack-v5.bundle> pwscup2026-kit-private
git -C .\pwscup2026-kit-private remote set-url origin https://github.com/tsugiyama0216/pwscup2026-kit-private.git
```

Then add the cloned folder as a Codex project and start with `CODEX_START_PROMPT.md`. The bundle contains committed source and candidate CSVs, but intentionally excludes `local-output/`, submission ZIPs, the virtual environment, and token material.

## Official baseline supplied by the user

| Candidate | attack score | MIA all | MIA rare | AIA all | AIA rare |
|---|---:|---:|---:|---:|---:|
| v4 actual | 0.292 | 0.1024 | 0.3333 | 0.1171 | 0.1000 |
| current leader | 0.368 | 0.1039 | 0.2500 | 0.1827 | 0.0519 |

Three official submissions remain. The primary gap is AIA all; v4 MIA all is already close to the leader and v4 MIA rare is stronger.

Important scoring detail: the total is based on per-target maxima between all and rare channels, then target averaging. The four leaderboard breakdowns cannot simply be added or maximized to reconstruct the MIA/AIA contribution.

## Implemented v5 attack

### AIA structured ensemble

`attack_phase_v5/aia_structured_ensemble.py` combines:

1. the v4 active set,
2. the independent AIA active set,
3. multiview proximity to the public release, and
4. the published 163-member / 163-control matching structure.

For 18 of 20 production targets, every visible `(sex, prefecture, five-year age band)` stratum has even cardinality, so exactly half is selected in each stratum. Targets 10 and 23 use score-based allocation for odd strata while retaining exactly 163 active rows.

The selected decoder prefers the independent candidate's time, then v4, then the 10-year horizon. Cross-target exact rows are filtered only from proximity evidence. This changes eight production active decisions in total (four each for targets 18 and 29).

### MIA low-FPR tail assignment

`attack_phase_v5/mia_tail_assignment.py` preserves the v4 rare ordering and known private-negative bottom block, then uses a global assignment to remove impossible cross-target duplication in the top-12 low-FPR tail.

- Original duplicated top-12 slots: 9
- Final duplicated top-12 slots: 0
- Changed slots: 9 of 240
- Shadow data had no corresponding duplicate slots, so production benefit size is unknown.

### Feedback winner blend

`attack_phase_v5/blend_feedback_winners.py` accepts official `aia_per_target` maps and copies each complete 326-row target block from the source with the highest official target score. It does not perform row-level leaderboard probing. The resulting AIA target mean is at least as high as each included source's target mean, subject to using the exact official per-target values.

### Integrity and packaging

- `generate_submission_v5.py` verifies fixed SHA-256 values for v3/v4/independent inputs and the organizer target manifest before generation.
- `package_submission_v5.py` creates deterministic, flat ZIPs containing only `F_mia.csv`, `F_aia.csv`, and `token.txt`.
- The packager checks archived CSV hashes against the candidate files and runs the official two-layer validator before replacing an output ZIP.
- The token is read only by the packager and is never printed.

## Local evaluation evidence

### Structured AIA

Training shadow: four seeds × `raw`, noise 3%, noise 10%, and time-shuffled.

- structured power: 0.919577
- v4 power: 0.911017
- mean structured − v4: +0.008560
- minimum difference across 16 conditions: 0

Independent holdout seeds 20260905–20260908:

- mean structured − v4 across 16 conditions: +0.008367
- minimum: 0
- maximum: +0.032186
- informative noise conditions: 8 wins, 0 losses; mean +0.016733

These are directional comparisons, not leaderboard forecasts.

Additional onset-count, rare-count, joint-integer, and age-pair constraints all lost to the current structured method in every tested condition and were rejected. A pair constraint recovered about 83% of true shadow member-control pairs but flipped 64–98 strong decisions per target and substantially reduced power.

Cross-target exact-row filter ablations were injection-design-sensitive: a localized production-like injection improved all eight comparisons (mean +0.009493), while a global 160-row injection averaged -0.000962 across four comparisons. Because the public non-shared-member construction supports the exclusion and production impact is only eight decisions, the filter remains enabled, but no expected gain is assigned to it.

### MIA tail

When the shadow had no top-tail conflicts, tail12 and tail15 were byte-equivalent in ranking effect to v4 and produced zero score change. This verifies no-conflict non-regression and rare-order preservation, but does not estimate the nine production changes.

## Generated candidates and archives

Candidate CSV hashes:

| Candidate | MIA SHA-256 | AIA SHA-256 |
|---|---|---|
| probe-independent | `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B` | `BDEA52C1FD7F77F9CF7054E8C9D64F3307B455521318FD73495B28A86BCF35E1` |
| structured-v4mia | `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B` | `A72B5B668C965DA359102EBAF7051228940C05D467C3945C920832C7F43B51DC` |
| stage2-tail12 | `D47457F5A7CA3BB31599649496963F1D3EE5F05B587F1A4DC7E3B99F3F269872` | `A72B5B668C965DA359102EBAF7051228940C05D467C3945C920832C7F43B51DC` |

Current deterministic local ZIPs, all official-validator PASS with the current team token:

| Intended use | Relative path | ZIP SHA-256 |
|---|---|---|
| First AIA probe | `local-output/attack-v5/attack-v5-probe-independent.zip` | `1A79C9567E122C7222D7B1D0860EFC73179A128B3AE8B215980564D005FF7BC8` |
| Isolated structured fallback | `local-output/attack-v5/attack-v5-structured-v4mia.zip` | `509E9829394FAF2360619F5E54EE08547865B3EF6F010732E5FB63E87DB858F5` |
| Structured plus MIA tail12 | `local-output/attack-v5/attack-v5-stage2-tail12.zip` | `5914787A0476239EB6732DF18D7FDB9A55338C1174FDAEC0F01E239FF95BB168` |

If a ZIP is absent after cloning, regenerate it with the packager. Do not commit it because it contains the team token.

## Verification commands

Run from the repository root:

```powershell
.\starter\.venv\Scripts\python.exe -m attack_phase_v5.generate_submission_v5
.\starter\.venv\Scripts\python.exe -m unittest discover -v
```

The last completed full run passed 77 tests.

Package the first probe:

```powershell
.\starter\.venv\Scripts\python.exe -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v5-probe-independent `
  --out .\local-output\attack-v5\attack-v5-probe-independent.zip
```

## Remaining submission plan

1. Submit `attack-v5-probe-independent.zip` first. It keeps v4 MIA byte-for-byte and changes only AIA. This is an information probe; it is not claimed to be the best standalone local candidate.
2. Save the complete official detailed result, including:
   - attack score and the four leaderboard breakdowns,
   - all 20 `aia_per_target` values,
   - `p_m`, `p_c`, `p_m_rare`, and `p_c_rare`,
   - preferably the same detailed values from the earlier v4 submission.
3. If detailed per-target feedback is available, use `attack-v5-stage2-tail12.zip` second. The per-target AIA mean separates its AIA contribution; total minus AIA power gives its MIA contribution.
4. If detailed feedback cannot be saved, use `attack-v5-structured-v4mia.zip` second so only AIA changes.
5. Build the third submission with the empirically better MIA and the whole-target AIA winners from v4/probe/structured. Package and validate it before asking for explicit submission approval.

Do not spend the third attempt on the cross-target-filter ablation alone.

## Key documentation

- `attack_phase_v5/README.md`
- `attack_phase_v5/EXPERIMENT_RESULTS.md`
- `rulebook/PWS_Cup2026_ルールブック_prelim-attack-20260825.pdf`
- `starter/src/pwscup2026/scoring/attack_score.py`
- `starter/src/pwscup2026/kit/codabench/scoring_io.py`
