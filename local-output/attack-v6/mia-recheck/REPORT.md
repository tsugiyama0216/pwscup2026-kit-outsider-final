# MIA local recheck — time-boxed final decision

## Decision

**Keep the v4 MIA payload unchanged.** No normal-format alternative has a reproducible held-out advantage, and the strongest available evidence is still the official comparison: v4 and the later guided/routed submission have identical TPR at 1% FPR (`0.1023675`) and rare TPR (`0.3333333`), while their AUC differs by only `0.0000059`.

This run was deliberately stopped after one qualified discovery world and one disjoint validation world. The sample is too small to claim optimality, but it is sufficient to reject a last-minute replacement when the predeclared gate is not met. Shadow evidence does not override official results.

## Existing payload inventory

- 15 production candidate directories contain `F_mia.csv`, collapsing to 7 byte-distinct payloads.
- `attack-candidate-v4`, `v5-probe-independent`, `v5-structured`, `v6-feedback-v4-probe`, `v6-routed-probe`, `v6-final-feedback-winners`, and `v6-final-aggressive-complements` use the exact same v4 MIA bytes.
- `attack-candidate-v6-routed-guided-probe` is byte-distinct, but its official low-FPR and rare scores tie v4; only AUC moves from `0.6493386` to `0.6493445`.
- `attack-candidate-v3` and `attack-candidate-daybreak-blue` share one payload. The independent Daybreak payload is separate.

The complete hash grouping and rank-distance diagnostics are saved in `production_hash_groups.csv` and `production_existing_diagnostics.csv`.

## Official evidence controls the decision

| payload/result | TPR @ 1% FPR | rare TPR | AUC | implication |
|---|---:|---:|---:|---|
| v3 | 0.092664 | 0.333333 | 0.643488 | below v4 on overall tail and AUC |
| v4 | 0.102368 | 0.333333 | 0.649339 | baseline to retain |
| guided/routed | 0.102368 | 0.333333 | 0.649344 | no measurable tail lift over v4 |
| Daybreak independent | 0.106219 | 0.166667 | 0.635601 | small overall-tail gain, but rare TPR halves and AUC falls |

## Normal-format alternatives tested

Sixteen payloads were scored: v4, v3, Daybreak, seven rare/private-locked rank combinations (weighted rank averages, min/max, and reciprocal-rank fusion), three top-tail reorders, and three top-tail replacements. Every research payload has one finite, tie-free score per record and target. The safer alternatives lock the v4 rare ordering and the 163 inferred private-bottom records.

The comparison used a truth-known shadow world with the production shared-count pattern (13 targets × 25, 6 × 26, 1 × 27), 20 targets, and a severe 40% noise release. Seed `20261301` was discovery; disjoint seed `20261311` was held-out validation.

| candidate | discovery power | Δ vs v4 | validation power | Δ vs v4 | validation Δ AUC | interpretation |
|---|---:|---:|---:|---:|---:|---|
| v4 | 0.527661 | 0.000000 | 0.587462 | 0.000000 | 0.000000 | baseline |
| tail12 reorder (discovery-selected safe candidate) | 0.527661 | +0.000000 | 0.587462 | +0.000000 | -0.000149 | ties tail power, slightly lowers AUC |
| equal 3-way locked average | 0.517738 | -0.009923 | 0.591308 | +0.003846 | +0.003080 | sign reversal; not reproducible |
| v3 | 0.539507 | +0.011846 | 0.594923 | +0.007462 | +0.005414 | shadow gain contradicted by official result |
| Daybreak | 0.443632 | -0.084028 | 0.450154 | -0.137308 | -0.023304 | clearly below v4 in this shadow |

Among the 13 safe alternatives, none beat v4 in discovery: two tied and eleven lost. One unselected blend was positive in validation (`+0.003846`) after losing in discovery (`-0.009923`), so it fails the required replication test.

## Predeclared replacement gate

The discovery-selected `tail12_reorder_avg` failed the two essential checks: discovery mean lift had to exceed `0.0025`, and held-out validation mean lift had to exceed `0.0025`. Its observed lift was `0` in both splits; validation win rate was `0%`. Rare order and private-bottom preservation passed, but format safety alone is not efficacy.

## QA and limitations

- 32 aggregate rows and 640 per-target rows were checked; keys are unique and each variant/world has 20 targets.
- Aggregate metrics recompute from target rows with maximum absolute error `2.22e-16`.
- All generated-world consistency checks passed, and all 16 research candidates passed normal-format validation.
- **Limitation:** one discovery and one validation world cannot measure a stable distribution. The broader multi-seed run was stopped by the task's time limit after an additional seed could not supply the fixed 13 rare known-private controls. Therefore the conclusion is conservative: there is no evidence to justify a change, not proof that v4 is globally optimal.
- Production MIA files were read for comparison; no production candidate, archive, token, or submission was changed, and no token was read.
