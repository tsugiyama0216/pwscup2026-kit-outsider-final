# v8: actual-data revalidation and attack research

Status: bounded research completed; no v8 submission promoted or submitted.
The user has **one official attempt remaining**. The current measured score is
`0.3004`; v7 is a generated, unsubmitted fallback, not a measured improvement.

Decision: no tested v8 policy has sufficient evidence to replace the existing
candidate. Preserve the v7 CSV/ZIP without claiming it is proven to improve
0.3004. A score of 0.5 is an aspiration, not a supported forecast. The old v7
shadow-to-LB ranges have been explicitly withdrawn as forecasts in its README
and local evaluation document.

## Actual data, supplied during this research turn

The user supplied:

`C:/Users/takut/Downloads/PWSCup2026_prelim_team24_アウトサイダー_outsider/`

The supplied README explicitly says B/A_bg/schema replace the starter's
practice-world inputs. B, A_bg, and the rare detector all differ from the retained
practice data. B has 1,049 rows; A_bg has 5,143. B_self agrees with B and has 24
true rare rows. The actual self challenge matches 163 own-B rows, including its
13 known-private rare rows, leaving 11 other rare rows not ruled out for MIA.
This does not by itself prove that every one of those 11 rows is shared.

`actual_inputs.py` requires an explicit directory, hashes the four allowlisted
data files, verifies the target manifest and self-challenge invariants, and
refuses practice-data fallback. It never opens a token. Do not copy actual B or
B_self into tracked/shared directories. Detailed audits are ignored beneath
`local-output/attack-v8/actual-data/`.

The historical v4 MIA, including its DOMIAS features, was reproduced
byte-for-byte from actual B/A_bg. Its SHA-256 remains
`D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`.
The historical Daybreak MIA was also reproduced byte-for-byte; its SHA-256 is
`760D4FE12B4296DE4712D2AF133E6165F93446E9A9794CDA56BADF6868F68294`.
Thus historical submissions were not merely practice-data submissions. The
limitation was that recent local research only had historical production CSVs,
public target files, and the retained practice source/reference population.
Old shadow-to-leaderboard numerical ranges must not be used as forecasts.

## Confirmed structure

The official rulebook's section 3.1 specifies 313 non-shared home-prefecture rows
and 736 pair-specific shared rows per cohort. Shared blocks are NOT drawn from
another target's private block; a candidate's prefecture does not identify its
partner target. Actual B24 contains 394 home-prefecture rows and 655 other rows.
After the 163 publicly identified private rows are removed, the remaining 231
home-prefecture rows contain exactly 150 private and 81 shared rows. This is a
valid membership *prior*, not a per-row ground-truth assignment.

Actual AIA challenge rows for each target share one home prefecture. Legacy
shadows sampling members across all B prefectures do not reproduce that
structure. New AIA validation must model the hometown private block explicitly.

Public-background provenance using age/SBP/TG and floating-point tolerance:

- C18: all 1,049 rows map to actual A_bg.
- C19: all 1,049 rows map to practice A_bg; some other columns were changed.
- C29: 885 rows map to actual A_bg and 164 to practice A_bg.

This does not prove that these releases are independent of private input:
selection of public surrogates or their multiplicities can still leak information.
It does prevent mistaking a few CSV floating-point differences for private donors.

## Implemented research policies

- `event_additions.py`: an abstention-only high-confidence event-link overlay.
  It preserves all currently active predictions and protects targets 1/2, but
  can still reduce AIA score by creating control hits. It is not promoted by
  merely passing format checks.
- Actual-data MIA re-estimation and hometown-prior experiments are under
  `local-output/attack-v8/actual-mia/`.
- Actual-data AIA timing and contrastive policy experiments are isolated under
  `local-output/attack-v8/actual-time/` and `actual-contrastive/`.

Rejected exploratory ideas remain reproducible in ignored local-output:
coverage union/intersection, blanket activation, challenge-vs-background PU
classification, and a supervised own-challenge control classifier. The latter
has grouped OOF best AUC only 0.51594 and does not support cross-target transfer.

## Completed actual-data decisions

### MIA: reject the hometown-prior rerank

The discovery-selected top-37 rerank was evaluated on 20 validation worlds
(five new seeds across four noise scenarios). Mean change in max-channel power
was +0.003465, but only 50% of worlds improved, below the predeclared 60% gate.
The all-channel p10 change was -0.002362, also failing the explicit lower-tail
audit. Seven worlds tied in power, and rare locking can hide all-channel losses.
Do not promote the 609-cell research change; retain historical v4 MIA.

These worlds reuse empirical data and assume random pair assignment and a soft
508-row allocation to scored targets. They are model-based sensitivity tests,
not independent official identities. Runtime-bounded shadows omitted DOMIAS;
they therefore do not directly validate the exact production recipe. Detailed
results are in `local-output/attack-v8/actual-mia/home-prior/FINAL_REPORT.md`.
The shadow additionally excluded rare rows from the 150 latent private rows;
the observed 13 rare AIA members alone does not establish that exclusion.

### AIA timing: reject the additional event links

The new decoder accounts for each challenge's single home prefecture when
computing ranks. On the truth-known self target, all tested mutual event links
had zero correct member event times. In eight actual-data query worlds, the
selected q90 additions had zero incremental power. A separate target-3
background stress produced one false hit. A kNN alternative's +0.001126 mean
came from only one of eight worlds and did not pass the win-rate gate.

This is absence of supporting evidence, not proof that target 3 cannot be
attacked: validation used public C24, not the unknown C3 generator. Do not add
the four/five target-3 times to the submission. Detailed results and limitations
are in `local-output/attack-v8/actual-time/REPORT.md`.

### Contrastive AIA: reject the nonlinear multi-time policy

Implemented a member/control contrastive objective, Bayesian shrinkage, public
background occupancy features, and a choice among 9, 9.5, 10, and 11 years.
Discovery selected `multitime_mean_expand` for the seven official-zero targets.
Every subsequent gate failed:

| Evaluation stream | Accepted exact-zero observations | Mean target-power change vs v7 | Target-weighted win rate |
|---|---:|---:|---:|
| Held-out | 54 | -0.015274 | 43.5% |
| Fresh | 78 | -0.011760 | 42.9% |
| Exchangeable-background stress | 123 | -0.019570 | 34.6% |

Restricting the comparison to observations where v7 already had positive power
also gave negative means: -0.059086, -0.012691, and -0.040218 respectively.
Thus the rejection is not just an artifact of the zero score floor.

The experiment fixes the actual public QI rows and releases, then imputes
membership and hidden outcomes from empirical reservoirs under the published
matched-block and rare/onset counts. The discovery/held-out/fresh streams are
not independent real cohorts: feature rows recur and the 13 known rare donors
overlap. This is posterior-predictive sensitivity analysis, not empirical
validation against hidden production outcomes. It does not prove v7 is optimal.
Raw scores, per-block conditions, and the rejected research mask are isolated
under `local-output/attack-v8/actual-contrastive/`.

## Final verification

- v8 policy/input tests: 10 PASS.
- Full repository tests after adding these modules: 128 PASS.
- Actual input loader smoke: PASS, 20 target releases, no token read.
- Existing v7 MIA/AIA/ZIP SHA-256 values are unchanged.
- No official submission, push, or new token-bearing archive has been made.

Existing artifacts, preserved rather than newly recommended as proven stronger:

- MIA: `submissions/prelim/team24/attack-candidate-v7-feedback-stack/F_mia.csv`
  (`D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`).
- AIA: `submissions/prelim/team24/attack-candidate-v7-feedback-stack/F_aia.csv`
  (`65AD898E8E24497AA1C415BBCBE14D8360BC35676903A069162F4F20F115EAA0`).
- ZIP: `local-output/attack-v7/attack-v7-feedback-stack.zip`
  (`CB2A8889C50B4ACEEB9C96D2A892CA640AF7A0DF0896781021D823C5964635D6`).
