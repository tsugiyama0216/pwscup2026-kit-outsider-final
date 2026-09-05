# Actual-data MIA home-prior audit

## Decision

**REJECT the home-prefecture rerank. Keep the historical v4 MIA unchanged.**

The discovery-selected `tail37_theory_reorder` has a positive mean in the
model-based shadow, but it fails the predeclared validation win-rate gate and
the explicit all-channel lower-tail audit. With one official attempt left,
the evidence does not justify changing 609 cells across all 20 targets.

This is a rejection of the new rerank, not a failure to recover the submitted
baseline. Using the newly supplied actual B/B_self/A_bg and DOMIAS reproduces
the historical v4 F_mia byte-for-byte:

`D4D66585ACAB6713531A01B9B3BA673E860FFAC8C84536BD7FF0C46240E0128B`

The older `rebuild_comparison.csv` values `changed_cells=7440` and
`changed_rows=997` were calculated on the in-memory builder frame before CSV
serialization. A re-read comparison of the written artifact gives zero changed
cells, zero changed rows, equal keys, and the identical SHA above. The rebuild
script now labels the pre-serialization diagnostic separately and compares the
submitted artifact after re-reading it.

## Fixed design

- Actual inputs were SHA-pinned; no token was read.
- The public self challenge identifies 163 known-private rows, including 13
  observed private rare rows. No inspected primary source proves that 13 is the
  rare quota of the complete 313-row private block.
- The rule-derived latent population has 313 private and 736 pair-shared rows
  across **29** peer blocks. After removing known-private rows, 150 latent
  private identities were sampled only from eligible home-prefecture common
  rows. This deliberately excludes all 11 eligible rare rows. Ten are non-home
  and therefore structurally shared, but the one eligible home rare row has
  unknown private/shared status. Treating it as shared is an explicit **model
  assumption**, not an official fact. All rare rows were rank-locked.
- The 508 rows assigned to 20 scored target blocks are a proportional modelling
  prior (`13×25 + 6×26 + 1×27`), not known official membership truth. The
  remaining 228 shared rows were unscored.
- Discovery used 3 seeds × 4 release-noise scenarios = 12 worlds. Validation
  used 5 disjoint seeds × 4 scenarios = 20 untouched worlds. Two structurally
  wrong worlds were used only as negative controls.
- For bounded runtime, this sensitivity run used v4 without DOMIAS. It is a
  high-rank-overlap surrogate, not direct validation of the exact production
  DOMIAS policy; this weakens, rather than strengthens, any promotion claim.

## Selected rerank results versus v4

| Split | n | mean Δpower | median | p10 | min | power wins | mean Δall | p10 Δall | min Δall | Δrare | mean ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Discovery | 12 | +0.005583 | +0.002000 | -0.005400 | -0.008000 | 58.3% | +0.007865 | -0.002008 | -0.006000 | 0 | -0.000100 |
| Validation | 20 | +0.003465 | +0.000962 | -0.000269 | -0.003692 | **50.0%** | +0.005238 | **-0.002362** | -0.006000 | 0 | -0.000069 |
| Wrong-prior control | 2 | -0.007808 | — | — | — | 0% | -0.017654 | — | — | 0 | -0.001269 |

Seven of 20 validation worlds tied in max-channel power. The all-channel win
rate was only 45%; rare locking can otherwise make `max(all, rare)` conceal an
all-channel loss.

Failed checks:

- required validation power win rate ≥ 60%; observed 50%;
- required validation all-channel p10 ≥ -0.002; observed -0.002362.

The negative control behaves in the expected direction, but that does not
repair the failed held-out gates.

## Research artifact (do not promote)

- File: `production_variants/F_mia_tail37_theory_reorder.csv`
- SHA-256:
  `697CA4D79CA0C05ECFA580D57479A4C52DE2C0609E67533E705323FBAF48B4EA`
- Changes versus v4: 609 cells, 482 rows, all 20 targets.
- The 174 locked rows (`known_private OR rare`) have zero changed cells.
- Record keys and normal-format official validation pass.
- Promotion flag: **false**.

## Interpretation limits

The home/shared asymmetry is rule-derived and real under the documented
construction, but the partner target of each shared row is unknown. Shadow
truth and the rerank both use the same home-private assumption, so disjoint
Monte Carlo seeds measure stability inside that model; they are not independent
official row labels. The v4 values are ranking scores rather than calibrated
posteriors, making the log-ratio adjustment heuristic. These limits, plus the
failed lower-tail gates, make unchanged v4 the defensible MIA decision.

Post-audit correction: rulebook §4.2 establishes the 13 rare rows in the AIA
sample, not a complete-private-block rare quota of 13. The benchmark forced the
remaining eligible home rare row to be shared. Results are therefore
conditional on that additional assumption. Because the rerank was already
rejected, this correction does not trigger a rerun or change the decision.

No production file, ZIP, token, or submission was modified.
