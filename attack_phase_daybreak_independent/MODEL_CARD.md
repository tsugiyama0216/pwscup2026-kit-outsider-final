# Model card: Daybreak Blue independent attack

- Model: `gpt-daybreak-blue-latest`
- Algorithm: `daybreak_blue_reciprocal_conformal_integrity_router_v1`
- Scope: PWSCUP2026 preliminary attack, team 24 only
- Inputs: organizer-distributed `A_bg.csv`, `B.csv`, `C_k.csv`, `aia_challenge_k.csv`
- Outputs: official-shape `F_mia.csv`, `F_aia.csv`
- Legacy production dependency: none
- MIA family: reciprocal-neighbour private-negative conformal margin
- AIA family: public outcome-integrity router over reusable residual/horizon codebooks
- AIA router threshold: release event-prediction ROC AUC `< 0.60`
- Remote access: none
- Submission operation: none

This is a deterministic competition analysis model.  It is deliberately stored
outside the v1/v2/v3 directories, and its output hashes are recorded in a local
manifest so that it cannot be confused with the conservative Daybreak wrapper
whose outputs are identical to v3.
