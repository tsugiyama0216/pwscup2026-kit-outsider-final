# PWSCUP2026 Codex instructions

## Scope and authorization

- This repository is the private working repository for the authorized PWSCUP2026 privacy competition.
- Attack work must remain within the contest datasets, rules, local shadow worlds, and official submission format.
- Do not repurpose techniques or data against real people or systems outside the competition.

## Start every task

- Read `CODEX_HANDOFF.md` before changing the attack implementation or advising on a submission.
- Treat the live-score values and remaining-submission count in that handoff as the current baseline until the user supplies newer official feedback.
- Inspect `git status` before editing. Preserve unrelated user changes.

## Operational rules

- Use `starter/.venv/Scripts/python.exe`; a separate system Python installation is not required.
- Never print, commit, or duplicate the contents of `token.txt` outside ignored local submission archives.
- Do not submit to CodaBench, push to GitHub, or consume an official submission attempt without the user's explicit instruction.
- Keep candidate directories under `submissions/prelim/team24/` limited to `F_mia.csv` and `F_aia.csv`.
- Build token-bearing ZIPs only with `python -m attack_phase_v5.package_submission_v5` and require the official validator to pass.
- After code changes, run `python -m unittest discover -v`. Rebuild and revalidate any affected candidate and ZIP.
- Do not claim a local shadow score as a predicted leaderboard score.

## Current workflow

- Preliminary submissions are exhausted. The user submitted v9 guarded and reported a final measured total of 0.3407, up from routed-guided v6's 0.3004. Read `attack_phase_v9/FINAL_RESULT.md` and `official_feedback/guarded_final.json`. v7 itself is unsubmitted, so its comparative performance is unknown. Await the main-round distribution/rules before new attack work.
- v9's four new research branches failed adoption checks. The deployed guarded composition retained v4 MIA and the measured positive AIA winners, with v7 on seven formerly zero targets. Five of those targets (6/8/17/18/19) scored positively; 7/9 stayed zero. Do not attribute the gain to rejected new methods or forecast main-round performance from it.
- The user explicitly authorized upload of all work to the existing private repository and accepted retention of the pre-existing live-token history after being informed it was unrotated. Do not rewrite history or force-push. Exclude live token files, token-bearing ZIPs, caches, and actual raw B/B_self/A_bg from new archive content; preserve local copies. Selected research text under ignored `local-output/` is intentionally archived; consult `research_archive/README.md` and its manifest.
- The user supplied actual preliminary B/B_self/A_bg/schema in `C:/Users/takut/Downloads/PWSCup2026_prelim_team24_アウトサイダー_outsider/`. `participant_data/` is a different practice world; never silently substitute it for actual data. New production generation must use the hash-gated `attack_phase_v8.actual_inputs` loader with an explicit actual directory.
- Historical v4 and Daybreak submissions are actual-data outputs; they can be reproduced from the newly supplied inputs. Preserve their measured baselines and rare/ceiling guards unless a change has evidence.
- Do not call correlated shadow observations independent trials, or convert shadow improvements into a leaderboard prediction. AIA shadows must respect the home-prefecture private-block sampling. MIA has exactly 313 private and 736 shared rows across 29 peers.
- Never splice individual rows using leaderboard-derived private truth. Use public contest inputs and local truth-known experiments; save and report negative validation results as well as positive ones.
