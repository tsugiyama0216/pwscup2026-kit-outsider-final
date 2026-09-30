# Main-round attack workspace

- Worker M's independent v1 lives in `M/v1/`, with archived outputs in `M/local-output/v1/` and initial planning in `M/strategy/`. Use `attack.ps1 M-v1-*`; the shorter `v1-*` aliases are retained for compatibility. Old local copies outside M are not canonical. Preserve ongoing research and other workers' files.

- Read `README.md`, `source_manifest.json`, and the repository root instructions first.
- This workspace is for authorized PWSCup2026 contest data only. Main-round cohort is 10, not preliminary cohort 24.
- Use `.venv/Scripts/python.exe` through `attack.ps1`. The official release is `main-attack-20260928`, commit `a490490e11a0ec460a5df9fdefb17c72ad459c1e`.
- Keep `official-kit/` and `targets/` as exact downloaded snapshots. Verify `source_manifest.json` before execution. Do not replace actual inputs with official-kit practice data.
- Actual B/B_self/A_bg/schema/utility_ref/token are in `../anonymization/team10/participant_data/`. Reference these originals; do not print or hash the live token.
- The target manifest contains 23 cohorts including self10. Work on the 22 other cohorts. MIA must cover the 1049 record IDs in actual team10 B.csv; AIA must cover all 326 challenge rows for each of the 22 targets.
- Official CLI `--dist` alone does not load IDs from B.csv. Use the wrapper's explicit team_id=10 and attacker_record_ids checks, including finite values and integer IDs.
- `baseline` outputs are smoke-test/reference predictions, not an optimized candidate or an estimate of official attack power.
- On 2026-09-27 the user explicitly requested uploading this prepared state to the existing private repository. The verified public release downloads, raw targets, official runtime source, setup/docs and frozen reference_baseline are included. The public upstream source archive contains public practice/example tokens only and was recursively checked against the live team10 credential. Virtual environments, caches, live credentials and live-token submission ZIPs are not newly included. Do not submit or consume an official attempt without the user's explicit instruction.
- Put new algorithm code outside `official-kit/`; place experiment outputs under `local-output/` and record inputs/method/config.
