# TS's main-round attack work

- This directory contains TS's (the repository owner's) attack implementations, fitted models and archived selection records. Worker M's independent attacks remain in `../M/`. Shared local evaluation tools are in `../../mock_test/`, outside both workers' attacks.
- Read the parent `AGENTS.md` and `README.md`. Official inputs, pinned runtime, environment and validation stay in the parent shared workspace.
- Use `../.venv/Scripts/python.exe` from this directory. Keep the Python module names `attack_research` and `local_eval` for existing fitted-model compatibility.
- Put TS-specific outputs under `local-output/`; M outputs remain under `../M/local-output/`. New shared mock cases and comparisons belong in `../../mock_test/local-output/`.
- Preserve archived models, predictions and historical verification JSON bytes. `manifest_before_attack_TS.json` preserves the pre-move v88 manifest; active `manifest.json` verifies the relocated runtime. Record layout changes separately in `layout_migration.json`.
- `manifest_before_mock_test.json` preserves the manifest before extracting the shared toolkit. The active manifest uses `main_round` as its base to cover both `attack/` and `mock_test/`; the new toolkit's `layout_migration.json` records this extraction.
- Before cross-PC v88 execution, read `../../../CODEX_HANDOFF.md` section `V88_CROSS_PC_HANDOFF`. The missing `prelim_transfer.methods` is an original project-code dependency; `setup.ps1` and successful manifest checks do not restore it. Recover the original module and its transitive dependencies before claiming reproducibility.
- The initial organization was local. The user's follow-up requests sharing the TS layout and v88 recovery guidance through the existing private GitHub for other PCs. It does not request reconstructing missing algorithms, official submission or model retraining.
