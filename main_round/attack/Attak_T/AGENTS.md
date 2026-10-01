# TS's main-round attack work

- Current folder name is `Attak_T` (the exact user-requested spelling). The previous v88 manifest is preserved as `attack_research/published_v88/manifest_before_Attak_T.json`; active paths and bootstrap source hashes are updated. Keep historical manifests, result JSON and fitted assets byte-identical.


- This directory contains TS's (the repository owner's) attack implementations, fitted models and archived selection records. Worker M's independent attacks remain in `../Attack_M/`. Shared local evaluation tools are in `../../mock_test/`, outside both workers' attacks.
- Read the parent `AGENTS.md` and `README.md`. Official inputs, pinned runtime, environment and validation stay in the parent shared workspace.
- Use `../.venv/Scripts/python.exe` from this directory. Keep the Python module names `attack_research` and `local_eval` for existing fitted-model compatibility.
- Put TS-specific outputs under `local-output/`; M outputs remain under `../Attack_M/local-output/`. New shared mock cases and comparisons belong in `../../mock_test/local-output/`.
- Preserve archived models, predictions and historical verification JSON bytes. `manifest_before_attack_TS.json` preserves the pre-move v88 manifest; active `manifest.json` verifies the relocated runtime. Record layout changes separately in `layout_migration.json`.
- `manifest_before_mock_test.json` preserves the manifest before extracting the shared toolkit; the toolkit's `layout_migration.json` records that extraction. The active manifest now uses the repository root as its base to cover both main-round runtime files and the recovered adapter's historical dependencies under `prelim/workspace/`. Preserve historical manifests and verification records.
- Before cross-PC v88 execution, read `../../../CODEX_HANDOFF.md` section `V88_CROSS_PC_HANDOFF`. The original `prelim_transfer` adapter was recovered byte-for-byte on 2026-09-30 into this TS directory. Its source dependencies are covered by the active manifest. Run `../setup.ps1`, `published_v88.check --runtime`, and the frozen 25-case benchmark before claiming reproducibility. Plain `check` verifies hashes only. Record validation scope in `attack_research/published_v88/dependency_recovery_20260930.json`; another physical PC has not been tested.
- The initial organization was local. The user's follow-up requests sharing the TS layout and v88 recovery guidance through the existing private GitHub for other PCs. It does not request reconstructing missing algorithms, official submission or model retraining.
