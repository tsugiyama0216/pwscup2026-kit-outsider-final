# PWSCUP2026 repository instructions

## Scope and authorization

- This is the private repository for the authorized PWSCUP2026 privacy competition. Keep all work within contest datasets and rules; do not target real people or systems outside it.
- Inspect Git status before edits and preserve unrelated user changes.
- Read `CODEX_HANDOFF.md` before attack work or submission advice. Read `prelim/workspace/AGENTS.md` and its handoff before changing preliminary code.
- Do not submit to CodaBench, push to GitHub, merge into main, or consume an official attempt without the user's explicit instruction. Never rewrite history or force-push for this reorganization.
- Never print live token contents. Do not newly commit live tokens except the team10 files and S1 submission ZIP explicitly authorized below. Other token-bearing ZIPs, preliminary actual downloaded B/B_self/A_bg, virtual environments and caches remain excluded. Existing private token history does not authorize other credential copies.
- On 2026-09-16 the user explicitly authorized local allocation and upload of the supplied main-round processing data to this existing private GitHub repository. The six non-credential originals belong in `main_round/anonymization/team10/participant_data/`.
- On 2026-09-17 the user explicitly authorized uploading the supplied team10 token to the existing private repository `tsugiyama0216/pwscup2026-kit-private` at exactly `main_round/anonymization/team10/participant_data/token.txt` and `main_round/anonymization/team10/模擬提出データ/token.txt`. Keep the repository private. Do not print token values or copy them into documentation, logs or other files.
- On 2026-09-17 the user additionally explicitly authorized uploading the token-bearing S1 submission ZIP to that same private repository. The authorized artifact is `main_round/anonymization/team10/S/S1/submission/team10_S1_defense.zip`; preserve its verified bytes and keep token values out of documentation and logs.

## Repository boundaries

- `prelim/` contains all preliminary-round artifacts, historical implementations, research and the exact starter/practice snapshot used for that round.
- `prelim/anonymization/anonymization-submitted-v5/` is the anonymization submission, NOT attack v5. Its live token stays local and ignored.
- `prelim/attack/` contains the final v9 pair, 17 earlier candidate pairs and six official result JSONs. It is a handoff collection; canonical reproduction inputs/code remain in `prelim/workspace/`.
- `prelim/old_attack_study_materials/` explains initial v1, not final v9. Preserve the DOCX originals as historical evidence.
- `prelim/workspace/` preserves the old execution layout. Run its historical module commands from there. Relative paths in historical reports use that directory as their base unless stated otherwise.
- `main_round/rulebook/` contains the official main-process-20260912 PDF and a change summary verified on 2026-09-09. Main-round processing data for cohort/team10 was allocated on 2026-09-16 under `main_round/anonymization/team10/participant_data/`. The main-round scoring kit has not been added or verified, and there is no main-round result yet. Preliminary team24 and main-round team10 are different cohort assignments.
- See `REPOSITORY_LAYOUT.md` for path migration and the file tree. Do not move source packages individually without checking cross-version imports and hash-gated input paths.

## Verification

- From the repository root, `python -m unittest discover -v` runs the preliminary suite through `test_prelim_workspace.py`. Alternatively run the same command inside `prelim/workspace/`.
- Use the project's existing Python environment; a new system-wide installation is unnecessary. New-clone setup runs inside `prelim/workspace/`.
- Rebuild and revalidate affected candidates when changing algorithms. Path-only archival organization must preserve submitted CSV, feedback, target manifest and DOCX bytes.
- `prelim/workspace/archive_prelim_work.py` prepares the secret-scanned Git-root-relative manifest and supports `--verify-staged`. Keep the root `.gitattributes` exact-byte policy.

## Final preliminary result

- The user submitted v9 guarded. The 2026-09-09 workbook confirms attack power 0.3406681446132233 (rank 4) and overall score 0.4223063973063974 (rank 17). The previously reported 0.3407 is rounded attack power. See `prelim/results/20260909/README.md`; no preliminary attempts remain.
- Read `prelim/workspace/attack_phase_v9/FINAL_RESULT.md`; local shadow scores are not leaderboard predictions. v7 itself was not submitted and v9's four new research methods were rejected.
- Actual preliminary inputs remain in the user-supplied Downloads directory. `prelim/workspace/participant_data/` is practice data, not a substitute. Main-round processing inputs are now available in the main-round directory; main-round attack-phase inputs have not been added.
