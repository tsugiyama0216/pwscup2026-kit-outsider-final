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

- The first pending official run is the isolated AIA probe documented in `CODEX_HANDOFF.md`.
- Save the official detailed result, especially all 20 `aia_per_target` values and aggregate `p_m`, `p_c`, `p_m_rare`, and `p_c_rare`.
- Use `attack_phase_v5/blend_feedback_winners.py` for the final whole-target AIA blend; never splice individual rows using leaderboard feedback.
