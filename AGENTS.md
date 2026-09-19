# PWSCUP2026 repository instructions

## Scope and authorization

- This is the private repository for the authorized PWSCUP2026 privacy competition. Keep all work within contest datasets and rules; do not target real people or systems outside it.
- Inspect Git status before edits and preserve unrelated user changes.
- Read `CODEX_HANDOFF.md` before attack work or submission advice. Read `prelim/workspace/AGENTS.md` and its handoff before changing preliminary code.
- Do not submit to CodaBench, push to GitHub, merge into main, or consume an official attempt without the user's explicit instruction. Never rewrite history or force-push for this reorganization.
- Never print live token contents. Do not newly commit live tokens except the team10 files and S1/S2/R1 submission ZIPs explicitly authorized below. Other token-bearing ZIPs, preliminary actual downloaded B/B_self/A_bg, virtual environments and caches remain excluded. Existing private token history does not authorize other credential copies.
- On 2026-09-16 the user explicitly authorized local allocation and upload of the supplied main-round processing data to this existing private GitHub repository. The six non-credential originals belong in `main_round/anonymization/team10/participant_data/`.
- On 2026-09-17 the user explicitly authorized uploading the supplied team10 token to the existing private repository `tsugiyama0216/pwscup2026-kit-private` at exactly `main_round/anonymization/team10/participant_data/token.txt` and `main_round/anonymization/team10/模擬提出データ/token.txt`. Keep the repository private. Do not print token values or copy them into documentation, logs or other files.
- On 2026-09-17 the user additionally explicitly authorized uploading the token-bearing S1 submission ZIP to that same private repository. The authorized artifact is `main_round/anonymization/team10/S/S1/submission/team10_S1_defense.zip`; preserve its verified bytes and keep token values out of documentation and logs.
- On 2026-09-17, after the S2 repair and submission ZIP were completed, the user authorized uploading the current state and making it main. Together with the prior explicit permission for token-bearing ZIPs in this private repository, this includes exactly `main_round/anonymization/team10/S/S2/submission/team10_S2_defense.zip`. Upload S2 and its diagnostic records only to the existing private origin and fast-forward main without rewriting history. CodaBench submission remains the user's responsibility.
- On 2026-09-17 the user explicitly requested uploading R1 and separately confirmed inclusion of its token-bearing ZIP in the same private repository. This authorizes exactly `main_round/anonymization/team10/R/R1/submission/team10_R1_defense.zip`. Preserve the verified ZIP bytes and never print its token. No CodaBench submission is authorized.
- On 2026-09-17 the user requested storing the submitted R2 files under team10/R on GitHub, then specified the directory name `R2` with the same organization as R1. This authorizes uploading the R2 contents and exactly `main_round/anonymization/team10/R/R2/submission/team10_R2_defense.zip` to this same private repository. Preserve the existing CSV and ZIP bytes and model name R2, never print the token, and do not send anything to CodaBench. The user stopped the proposed additional regeneration check; this upload does not resume that work.

## Repository boundaries

- On 2026-09-19 the user requested a more understandable R12 that preserves R11's score, storing submission data under team10/R on this PC and GitHub. This authorizes R12 contents and exactly `main_round/anonymization/team10/R/R12/submission/team10_R12_defense.zip` in the existing private origin. Preserve R1-R11 and other models; distinguish verified local utility from unmeasured official anonymity/final scores. Never print the token or submit to CodaBench.

- On 2026-09-19 the user requested an R9-based R11 aimed at improving the score, saving submission data under team10/R on this PC and GitHub. This authorizes the R11 model, evaluation records and exactly `main_round/anonymization/team10/R/R11/submission/team10_R11_defense.zip` in the existing private origin. Select for higher utility rather than the attack-AUC increase previously requested for R10. Preserve R1-R10 and other models, never print tokens, distinguish local utility from official final scores, and do not submit to CodaBench.

- On 2026-09-19, after deleting R10, the user explicitly requested storing R10 submission data under team10/R on this PC and GitHub. This supersedes the deletion request and authorizes restoring the verified R10 from commit 19148a6 and uploading exactly `main_round/anonymization/team10/R/R10/submission/team10_R10_defense.zip` with its model and verification records to the existing private origin. Preserve the frozen CSV bytes, never print the token, distinguish proxy AUC from official AUC, and do not submit to CodaBench.

- On 2026-09-18 the user supplied R8 official feedback and requested an improved R9 aiming above 0.99, saving submission data under team10/R on this PC and GitHub. This authorizes R9 contents and exactly `main_round/anonymization/team10/R/R9/submission/team10_R9_defense.zip` in the existing private origin. Preserve verified artifact bytes, R1-R8 and other workers' models, never print the token, and do not submit to CodaBench.

- On 2026-09-18 the user requested uploading worker M's S1-based M1 artifacts to the existing private GitHub repository. This authorizes the tracked contents under `main_round/anonymization/team10/M/M1/` and the corresponding documentation updates. It does not authorize a token-bearing M1 submission ZIP or a CodaBench submission.

- On 2026-09-18 the user supplied R7 official feedback and requested an improved R8, storing submission data under team10/R on this PC and GitHub. This authorizes R8 contents and exactly `main_round/anonymization/team10/R/R8/submission/team10_R8_defense.zip` in the existing private origin. Preserve verified artifact bytes and R1-R7, never print the token, and do not submit to CodaBench.

- On 2026-09-18 the user requested improving R6 into R7 and saving the submission data under team10/R on this PC and GitHub. This authorizes R7 contents and exactly `main_round/anonymization/team10/R/R7/submission/team10_R7_defense.zip` in the existing private origin. Preserve verified artifact bytes and R1-R6, never print the token, and do not submit to CodaBench.

- On 2026-09-18, after cancelling and deleting the previous R5-based R6, the user requested a new R4-based copula R6, storing its submission data on this PC and GitHub. This authorizes the new R6 contents and exactly `main_round/anonymization/team10/R/R6/submission/team10_R6_defense.zip` in the existing private origin. Preserve verified artifact bytes and R1-R5, never print the token, and do not submit to CodaBench.

- On 2026-09-18 the user requested creating sequential conditional synthesis R5, storing its submission data under team10/R/R5 on this PC and GitHub, and showing both locations. This authorizes R5 contents and exactly `main_round/anonymization/team10/R/R5/submission/team10_R5_defense.zip` in the existing private origin. Preserve verified artifact bytes, never print the token, and do not submit to CodaBench.

- On 2026-09-17, after receiving the completed R4 and its local paths, the user explicitly requested saving it to GitHub too. This authorizes uploading the R4 contents and exactly `main_round/anonymization/team10/R/R4/submission/team10_R4_defense.zip` to the existing private repository. Preserve the verified artifact bytes, never print the token, and do not submit to CodaBench.

- On 2026-09-17, after receiving the completed R3 submission ZIP, the user requested storing R3 under team10/R. This authorizes uploading the R3 model, evaluation records, and exactly `main_round/anonymization/team10/R/R3/submission/team10_R3_defense.zip` to the existing private repository. Preserve the verified artifact bytes and never print the token. No CodaBench submission is authorized.

- `prelim/` contains all preliminary-round artifacts, historical implementations, research and the exact starter/practice snapshot used for that round.
- `prelim/anonymization/anonymization-submitted-v5/` is the anonymization submission, NOT attack v5. Its live token stays local and ignored.
- `prelim/attack/` contains the final v9 pair, 17 earlier candidate pairs and six official result JSONs. It is a handoff collection; canonical reproduction inputs/code remain in `prelim/workspace/`.
- `prelim/old_attack_study_materials/` explains initial v1, not final v9. Preserve the DOCX originals as historical evidence.
- `prelim/workspace/` preserves the old execution layout. Run its historical module commands from there. Relative paths in historical reports use that directory as their base unless stated otherwise.
- `main_round/rulebook/` contains the official main-process-20260912 PDF and a change summary verified on 2026-09-09. Main-round processing data for cohort/team10 was allocated on 2026-09-16 under `main_round/anonymization/team10/participant_data/`. The pinned main-round kit is verified locally. S1's user-supplied official utility result matches local U=0; S2 fixes conditional-round sampling and has local U=0.5484134662543412, with no S2 CodaBench submission performed here. See `main_round/anonymization/team10/S/README.md`. Preliminary team24 and main-round team10 are different cohort assignments.
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
