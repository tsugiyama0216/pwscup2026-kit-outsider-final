# retest saved evaluation

- On 2026-10-07 the user corrected the requested location to the existing attack directory. This folder, `main_round/attack/retest/`, is the runnable saved version; the earlier `team10/attack/retest` location is obsolete.
- Use `run.ps1`. The copied package remains named `sugiyama_v2` internally. Its own bootstrap must select this saved package while reusing the canonical shared `main_round/mock_test/local_eval` bootstrap and pinned runtime from `main_round/attack`.
- Keep source population, official kit, model assets and the R v1 adapter in their existing canonical locations. Do not duplicate tokens, submission ZIPs, virtual environments or caches here.
- Evaluation outputs use the shared ignored `main_round/mock_test/local-output/retest/` area. Preserve nonempty outputs and frozen attack dependencies.
- `sugiyama_v2/verification_20261007.json` and `save_verification_before_attack_move_20261007.json` are byte-preserved historical evidence; their paths and hashes intentionally identify earlier locations. Record current layout checks separately in `save_verification_20261007.json`.
- On 2026-10-07 the user explicitly requested GitHub upload. This authorizes publishing retest and its necessary non-credential dependencies and documentation to the existing private origin/main. Exclude unrelated local work, generated mock outputs, tokens and submission ZIPs. No official contest submission is authorized.
