# Public-surrogate donor occupancy MIA

Decision: **REJECT**.

This branch maps C18/C19/C29 to public background donors under a strict floating-point tolerance, then tests whether donor-cell occupancy adds low-FPR signal beyond ordinary distance. It is a mechanism-conditional competition experiment, not a leaderboard forecast. The actual and toy pools each have unique stable3 cells; the audit records this rather than assuming row identity.

## Actual-release mechanism audit

| target | pool | mapped_rows | unique_donors | duplicate_excess | max_multiplicity | sum_choose_2 | pool_stable3_duplicate_rows | candidate_demand_selected_mean | candidate_demand_null_mean | candidate_demand_null_sd | candidate_demand_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | actual | 1049 | 1049 | 0 | 1 | 0 | 0 | 0.21258341277407056 | 0.20588655862726404 | 0.012466578983593617 | 0.537184592149921 |
| 19 | toy | 1049 | 893 | 156 | 8 | 196 | 0 | 0.19260918253079506 | 0.2039137737961926 | 0.013159512768277625 | -0.8590432992814471 |
| 29 | actual | 885 | 812 | 73 | 4 | 85 | 0 | 0.18719211822660098 | 0.2037130541871921 | 0.014437782808137949 | -1.1442848379239359 |
| 29 | toy | 164 | 115 | 49 | 5 | 68 | 0 | 0.26956521739130435 | 0.20426086956521738 | 0.04017601459384599 | 1.625456095789303 |

## Discovery-locked self-challenge validation

Selected config: `{'pool': 'toy', 'without_replacement': False, 'neighbours': 3, 'inverse_cell': False, 'mean_tpr': 0.10956790123456789, 'mean_auc': 0.8786484529797287}`.

Validation observations: 18; mean/p10 TPR delta vs DCR = -0.000921 / -0.048780; win rate = 27.8%.

Mean/p10 AUC delta vs DCR = +0.075509 / +0.035225. Null mean AUC = 0.498141.

## Decision and safeguards

Self-validation gate: `False`. Production mechanism gate: `False`. The research candidate was not written (`candidate_written=false`).

Any hypothetical rerank is limited to unlocked top-tail slots of targets 18/19/29. It preserves record keys, every known-private/rare row, targets 1/2, and every target's score multiset. Aggregate change counts are recorded in `decision.json`.

Discovery and validation use disjoint, class-stratified self-challenge query rows and different noise seeds. They still reuse actual B as the simulated source and therefore cannot prove transfer to hidden targets. C18's all-unique donor pattern can be produced by a global no-replacement policy or by unrelated sampling. C1/C2 are never treated as truth-known identities. No official submission, ZIP, production file, or token was touched.
