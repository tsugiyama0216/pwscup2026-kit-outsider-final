# Worker M's attack series

- M identifies the user/worker; v1 identifies the version. Do not rename this version to M1.
- Keep M's versioned implementations in `v1/`, `v2/`, and so on, and their outputs under `local-output/<version>/`.
- Reuse the shared pinned runtime, official data, and validators from the parent attack workspace.
- `v1/model.py`, `v1/actual.py`, and the selected prediction CSVs were preserved byte-for-byte during organization. Read `layout_migration.json` for old-to-new paths.
- Existing JSON verification records describe their creation-time layout and hashes. Keep those records unchanged; record later organization or validation separately.
- The old root-level v1 and initial-strategy files may remain as ignored local copies for ongoing work. Use the M paths as canonical; do not add old copies back to Git.
- Do not include another worker's models, in-progress follow-up research, credentials, caches, or submission ZIPs in a layout-only update.
