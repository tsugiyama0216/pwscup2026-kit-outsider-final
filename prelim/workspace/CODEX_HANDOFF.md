# PWSCUP2026 attack research — Codex handoff

## Current state: preliminary round complete, final score 0.3407

Historical commands and implementation paths in this document now use `prelim/workspace/` as their working-directory base. The old execution layout moved there; see the root `REPOSITORY_LAYOUT.md`. The anonymization submission is now `prelim/anonymization/anonymization-submitted-v5/` (Git-root-relative), with its live token ignored. Old v1 study documents are in `prelim/old_attack_study_materials/`.

Organized preliminary-attack files are under `../attack/`: `final/`
contains the submitted v9 pair, `candidates/` has the other 17 candidate pairs,
and `results/` contains six feedback JSONs and the final report. The 43 copied
files are byte-identical to originals, with source/hash mapping in
`../attack/manifest.json`. Code and original files now live in this workspace. The
folder README links the implementation and archived research logs.

This section supersedes all old remaining-attempt counts, pending-submission
instructions and numerical forecasts below. The user submitted v9 guarded and
reported its full official result. **No preliminary attempts remain.** The next
work is the main round after a break, once new rules/data and a user request are
available; no monitoring automation or new attack run has been scheduled.

- Latest user-reported official total: **0.3407**, versus v6 **0.3004**:
  +0.0403 (about +13.4%). Full JSON and analysis:
  `attack_phase_v9/official_feedback/guarded_final.json`,
  `attack_phase_v9/FINAL_RESULT.md`.
- MIA CSV and all reported MIA diagnostics are unchanged from v4. Actual final
  AIA power = mean(20 target scores) = **0.18274506769014637**, consisting of
  retained positive13 contribution **0.1506204125415067** plus formerly zero7
  contribution **0.032124655148639644**. New gains: targets6/8/17/18/19;
  targets7/9 remain0. Positive13 measured target scores all match their sources.
- Score display rounding reconciles the total. AIA power is not score4+score5;
  MIA power is not score2+score3. v7 was not submitted, and the four new v9
  methods stayed rejected. Neither v7 superiority nor main-round transfer is
  established. AIA source-block equality uses the generator's canonical pandas
  numeric interpretation, not byte identity of original AIA text.
- The user requested private upload of all completed work. Archive all code,
  tests, candidate CSVs and selected research text at original `local-output/`
  paths, including failed/invalidated experiments and shadow truth. See
  `research_archive/README.md` and `prelim-2026-manifest.json` for exact scope.
  Keep actual raw inputs at their original Downloads path; no copying to Git.
- Security decision explicitly confirmed by user: the historical submitted-v5
  token is not rotated, but the owner accepts retaining existing history in the
  private repository. Do not repeat the old blanket push blocker below as if
  this choice were missing. Do not rewrite history/force-push. Remove the live
  token from current tracking only while preserving its local file; do not add
  token-bearing ZIPs. History retention is not a claim of zero credential risk.
- Destination was verified private:
  `https://github.com/tsugiyama0216/pwscup2026-kit-private`.
  The previous archive used `codex/prelim-final-archive`. The current organization uses `codex/organize-prelim-phases`; the old branch instruction is historical, not permission to overwrite main.
- Archive validation: **170 tests PASS**, including the new official-result
  reconciliation and archive secret-selection tests. Exact-byte `.gitattributes`
  is intentional: Windows autocrlf previously differed between working/index
  bytes, which would break the input/output SHA-256 gates after another checkout.

## Latest update: v9 exploration complete, guarded candidate prepared

This section supersedes candidate recommendations below. One official attempt
still remains; measured baseline is routed-guided v6 **0.3004**. No submission,
push or commit was performed. New research and the guarded composition are
documented in `attack_phase_v9/README.md`.

- Four new implementations: Gen-LRA fixed-KDE influence, public-donor occupancy
  MIA, two-view affine registration, positional/shuffle audit. **All failed
  adoption checks.** Their held-out/null results are retained under ignored
  `local-output/attack-v9/`; none was promoted into the production candidate.
- A risk-managed local candidate is prepared at
  `submissions/prelim/team24/attack-candidate-v9-guarded/` and packaged at
  `local-output/attack-v9/attack-v9-guarded.zip`. This is a composition change,
  not evidence that a new attack method succeeded.
- MIA is exact v4. AIA whole-target recipe: v4={1,2,4,5,12,29},
  independent={3,13,22,23,26}, routed={10,20}, v7={6,7,8,9,17,18,19}.
  All measured positive blocks are exact replays; v7's zero-target strategy
  remains. The unmeasured horizon shifts and seven target20 event-time changes
  on positive blocks are removed. Relative to v7, 1,544 rows in 11 targets differ.
- Preserved measured AIA contribution is 0.1506204125415067. With v4 MIA
  inferred from the rounded independent-probe total, the conditional lower bound
  is 0.308481900464532. This assumes unchanged hidden data/scorer, correct stored
  feedback-to-artifact association, and four-decimal rounding. It adds no
  unmeasured zero-target gain and is NOT an official score, shadow forecast,
  generalization bound, proof of superiority to v7, or support for 0.5.
- New generator requires explicit actual `--dist` and the v8 hash-gated loader.
  It also locks historical feedback/source/probe/v7/manifest/output hashes and
  refuses to overwrite a differently populated existing candidate. Independent
  code and block/score-floor audits found no blocking issue.
- Full157 tests PASS; official logical and ZIP validators PASS. v9 hashes:
  MIA `D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`,
  AIA `63944D5FAA2C08E5E2E40415FFBB16B3153B27EBDD052CD68D7ADB59A403A6E0`,
  ZIP `B5297F0761A88DE0C5A9B0205903138613FCD2B7A501A231AE35B391A83D0506`.
  Existing v7 CSV and ZIP hashes remain unchanged. The safe v5 packager read
  the user-supplied token only to place it inside the ignored local ZIP; no
  token content was printed. CSV generation itself does not read the token.
- Position audit's initial optimistic run was invalidated: source multiplicity
  was confounded and labeled-member source indices had been overwritten.
  Both leaks were removed; final exact-label self24 AUC=.48180, FWER p=.9081.
  C1/C2 labels remain approximate, never organizer truth.
- Registration cross-review: anchor-fit numeric views differ, but final links
  use common full features, so their agreement is correlated, not independent
  evidence. Self24 has 16 onset plus 6 death-only events (22 total), with zero
  active/correct member decoder additions. A self-only AIA pass must not authorize
  cross-target promotion; transfer validation remains required.

## Current update: actual inputs supplied during v8 research

This section supersedes stale submission counts and recommendations below.

- **One official attempt remains.** Latest measured score: **0.3004**, detailed
  JSON at `attack_phase_v6/official_feedback/routed_guided_probe.json`.
- v7 is an unsubmitted local candidate. No claimed shadow-to-LB numerical range
  should be used as a forecast. Current v8 status is `attack_phase_v8/README.md`.
- User supplied actual data at
  `C:/Users/takut/Downloads/PWSCup2026_prelim_team24_アウトサイダー_outsider/`.
  B/B_self/A_bg/schema differ from the practice world retained in the repository.
  Do not overwrite or confuse the two worlds. Use the explicit, hash-gated
  `attack_phase_v8.actual_inputs.load_actual_inputs` loader for new production work.
- Actual B has 1049 rows, A_bg 5143; own challenge matches 163 B rows including
  13 true rare rows. 11 other rare rows remain eligible for MIA. `record_id` is
  table-local and must not be used to join B with A_bg.
- Actual inputs reproduce historical v4 (with DOMIAS), v3 and Daybreak MIA.
  Previous submissions were real-data attacks; recent exploratory work lacked
  the original local input files and used historical CSVs plus practice shadows.
- New research outputs and provenance: ignored `local-output/attack-v8/`.
  Actual B/B_self/token must not be copied into tracked/shared directories.
- v8 research is now complete with **no new candidate promoted**. Actual-data
  hometown-prior MIA, additional AIA time links, control classification, and
  nonlinear contrastive AIA all failed their adoption checks. Preserve existing
  v7 CSV/ZIP; do not describe v7 as a measured improvement or 0.5 as supported.
  The previous v7 score ranges are explicitly withdrawn as LB forecasts.
- Full tests: 128 PASS; actual-input smoke and unchanged v7 hashes confirmed.
  Read `attack_phase_v8/README.md` for results and limitations. In particular,
  AIA sample rare count 13 does not prove private-block total rare count 13;
  the MIA shadow's all-remaining-rare-shared rule is an extra model assumption.
- Neither official submission nor push is authorized. Do not consume the last
  attempt. Token-bearing archives still require the existing safe packager.

## Earlier v6 handoff (historical context)

Last updated: 2026-09-05 (Asia/Tokyo)

## Scope and authority

Continue the authorized PWSCUP2026 preliminary attack using organizer data,
local shadow worlds, and the team's own official feedback. Two submissions
remain. Codex has generated and validated archives but has **not** submitted to
CodaBench, pushed, or consumed an attempt. Do none of those without explicit
user instruction. A score of 0.4 remains an aspiration, not a validated forecast.

Repository: `C:/Users/takut/Documents/ChatGPT/PWSCUP2026/pwscup2026-kit-private`

- branch: `codex/attack-v6`
- starting commit: `68ebda230bd12fee5fc159d427c927429ab0b0d1`
- implementation: `attack_phase_v6/`
- detailed research: ignored `local-output/`
- candidates: `submissions/prelim/team24/attack-candidate-v6-*`
- stable Python:
  `C:/Users/takut/.codex/.chatgpt-projects/g-p-6a853f607498819195a7ef70756f71c9/pwscup2026-kit-private/starter/.venv/Scripts/python.exe`

## Official feedback

| Submission | MIA all | MIA rare | MIA AUC | AIA all | AIA rare |
|---|---:|---:|---:|---:|---:|
| v3 | .0926638 | .3333333 | .6434884 | .1171384 | .1000000 |
| v4 | .1023675 | .3333333 | .6493386 | .1171384 | .1000000 |
| latest independent-AIA probe | .1023675 | .3333333 | .6493386 | .1123375 | .0903025 |
| Daybreak independent | **.1062194** | .1666667 | .6356009 | .1123375 | .0903025 |

The latest probe's MIA is byte-identical to v4. Daybreak's AIA is byte-identical
to the latest probe, so the duplicate official AIA results reconcile exactly.
All supplied JSON is saved under `attack_phase_v6/official_feedback/`.
The displayed latest-probe `score_1` was `0.2771`, giving the v4-MIA power
baseline `0.2771 - 0.1191885120769747 = 0.1579114879230253`. Because score_1
was displayed to four decimals, treat that baseline as having about `±0.00005`
rounding uncertainty.

Exact AIA powers computed from the 20 official target values:

- v4: `0.1340658382149063`
- independent: `0.1191885120769747`
- complete-target winner blend: `0.1472268438242094`
- guaranteed blend lift over v4: `+0.0131610056093031`

Independent wins targets `3,13,22,23,26`; v4 wins
`1,4,5,10,12,29`; target 2 ties at 1; targets
`6,7,8,9,17,18,19,20` tie at zero.

## Recommended penultimate submission

Submit only after explicit user instruction:

`local-output/attack-v6/attack-v6-routed-guided-probe.zip`

- ZIP SHA-256: `64B845B5EB17954B4DD52A264B28733E190CC998CBDF5FB1880558464990657A`
- MIA SHA-256: `70E8E689E476E9EEF54F5821846EC1B93DE04FD6A63B680849D34F720DB78AFA`
- AIA SHA-256: `CCAA0A4EE58797F263E1281D5BC65E763B129A2F77DE00A1A4579DD60566129F`
- flat three-member archive and official package validator: PASS

Immediate MIA-revert archive, with the same routed AIA:

`local-output/attack-v6/attack-v6-routed-v4-probe.zip`

- ZIP SHA-256: `3D2D5967055D3134E67A30A32C93EEC51427B70C30746821DECFEFD8D97A9FDB`
- MIA SHA-256: v4 `D4D66585...0128B`
- AIA SHA-256: same `CCAA0A4E...6129F`
- official package validator: PASS

## Routed AIA

`attack_phase_v6/generate_submission_v6.py` protects target 1/2 and routes:

- 3: equal copula mix selector + independent/event decoder
- 4,5,12: equal copula selector + v4/event decoder
- 10: v4-lean copula selector + v4/event decoder
- 29: exact v4 block (measured official winner)
- 13,22,23: equal copula selector + independent/event decoder
- 26: soft copula selector + independent/event decoder
- 6,8,17: existing copula selector + horizon
- 7,20: global all-five density + horizon
- 9: global stable3 `(age,SBP,TG)` density + horizon
- 18: sex-only no-HDL consensus + horizon
- 19: **inverted** stable3 density + horizon

Positive-target route selection used discovery/validation splits over eight
shadow seeds. The original ten-target shadow route had a validation mean
AIA-power delta of `+0.00690`, but its shadow-tested target-29 block differed
from the filter-applied generator block by nine rows. Target 29 was therefore
reverted to exact v4; the other nine positive production blocks match their
tested artifacts. The shadow figure is directional evidence only, not a score
prediction for the final archive.

Density is admitted only after cross-target C substitution. Targets 7, 9, and
20 ranked 2/20, 2/20, and 1/20 respectively and beat the other-C split median
6/6 times. Target 6 was rejected after ranking only 7/20. Weak target 8 and
generic target 18 density signals were also rejected.

Target 19 is a sign reversal. Exact A_bg matches are:

- all-five: `993/1049`
- stable3: `1028/1049`
- all non-prefecture columns: `941/1049`
- after release filtering, stable3: `979/999`

Ordinary stable3 density selected held-out controls at rates
`.571632,.542806,.629910,.590552,.542333,.607928`; mean `.580860`, all six
above chance. It ranked first among all 20 C substitutions with a `.099243`
margin over the other-C median. Therefore the selector is complemented and
times remain horizon-only. The generator enforces a 90% background-copy gate.

## Guided MIA

`attack_phase_v6/mia_guided_tail.py` starts from v4 and the existing unique
top-15 assignment. For each duplicated top-15 record it chooses the possible
owner minimizing equal-weight `v4 rank + Daybreak rank`; v3 is only a final
tie-break and refill signal.

Production invariants:

- retained top-15 slots: theoretical maximum `284/300`
- duplicate slots: `16 -> 0`
- protected target 1/2 full rankings unchanged
- inferred private bottom 163 and eligible-rare order unchanged
- rare ROC therefore unchanged at exactly `1/3`
- total v4-to-guided Kendall inversions: 98
- five owner switches:
  `410 29->8`, `647 10->29`, `807 9->6`, `998 5->22`, `1001 3->20`

This remains an experiment. Daybreak's all-channel lift `+0.00385185185` can be
explained entirely by one extra TP in each of targets 1 and 2 when their sizes
are 25 and 27. Thus it does not prove the five low-band switches. If all five
direct owner changes are wrong, the MIA-all downside is roughly
`.0093-.0100` before refill/threshold effects. The penultimate attempt is the
only appropriate time to test it; an unmeasured last attempt must use v4.

## Final feedback gate

After the penultimate result, save the total score and full details. Compute:

`new MIA power = score_1 - mean(aia_per_target)`

Compare it to the v4-MIA baseline `0.1579114879230253`, allowing for the source
score's `±0.00005` display rounding (and the new score's rounding too). Keep
guided only if it is clearly higher beyond the combined rounding band; revert
to v4 if lower or effectively tied. The hidden MIA
power, not MIA-all alone, is decisive because scoring averages target-wise
`max(all, rare)`.

Sanity checks for the guided file:

- MIA rare must remain `0.3333333333333333`
- AUC must lie in `[0.6491472063, 0.6495300188]`

Otherwise suspect a file/result mismatch. For AIA, build the final candidate by
copying each complete 326-row winner among v4, independent, and this routed run.

## Reproduction and validation

```powershell
$python = "C:\Users\takut\.codex\.chatgpt-projects\g-p-6a853f607498819195a7ef70756f71c9\pwscup2026-kit-private\starter\.venv\Scripts\python.exe"
& $python -m attack_phase_v6.generate_submission_v6
& $python -m unittest discover -v
& $python -m attack_phase_v5.package_submission_v5 `
  --candidate-dir .\submissions\prelim\team24\attack-candidate-v6-routed-guided-probe `
  --out .\local-output\attack-v6\attack-v6-routed-guided-probe.zip
```

The generator never reads the token. The packager reads it without printing it,
checks CSV hashes, creates a flat archive, and runs the official validator.
Never commit token-bearing ZIPs.

Historical security note: the old `submissions/prelim/team24/submitted-v5/token.txt` existed in earlier commits. The owner subsequently accepted retaining the existing private history without rotation. Its live file is now ignored in `../anonymization/anonymization-submitted-v5/`. Follow the current security decision above; do not newly track it or rewrite history.
