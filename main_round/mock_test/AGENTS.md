# Shared main-round mock tests

- This toolkit is independent of TS's and M's attack implementations. Keep reusable case generation, scoring and comparisons here; attack methods stay in `../attack/<worker>/` and are supplied through the public `predict()` API.
- Read the repository root instructions and `../attack/AGENTS.md` for the shared official inputs and pinned runtime. Use `../attack/.venv/Scripts/python.exe` and preserve the official snapshots.
- Keep the package name `local_eval` for existing import compatibility. Do not import TS attack algorithms into the generator or scorer.
- `mock_workspace.py` supplies import paths and workspace attributes without editing M v2's frozen `attack_workspace.py`, `attack.ps1`, M source or `attack/local_eval/` snapshot. Preserve those bytes and hash gates. Run shared tests and frozen M tests in separate Python processes; do not silently replace an already imported evaluator package.
- Save generated suites and comparison results only under `local-output/`; it is ignored by Git. Preserve nonempty output directories. Give attack code only public inputs, never private truth tables.
- Multiple seeds resample the same fixed person partitions; they are not independent populations. The existing holdout partition has already been used. Preserve these limitations in reports and use matched cases to compare attacks.
- Preserve existing generation/scoring formulas and archived model/prediction/result bytes when changing layout. Record the extraction in `layout_migration.json` and keep previous v88 manifests unchanged.
- This toolkit does not upload to GitHub, submit to CodaBench or create credential-bearing submission ZIPs.
