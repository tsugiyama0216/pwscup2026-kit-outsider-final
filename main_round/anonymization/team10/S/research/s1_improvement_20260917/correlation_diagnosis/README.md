# S1 correlation diagnosis (2026-09-17)

Read-only analysis of existing S1 `C.csv`, team10 `B.csv`, and public `A_bg.csv`. No new DP measurements, synthetic rows, model, or submission ZIP were generated. These aggregates use B for local diagnosis and are **not a DP release**. They are not evidence of final anonymity or leaderboard performance.

`diagnose.py` reproduces the official 9-feature / 36-pair Spearman distances and records input hashes in `aggregate_diagnosis.json`.

| Comparison | Mean absolute correlation difference | 90th percentile |
|---|---:|---:|
| S1 C vs B | 0.489979 | 0.743677 |
| Public A_bg vs B | 0.028380 | 0.049533 |
| Rebinned C vs rebinned B | 0.509369 | 0.754583 |
| Nonrare C vs nonrare B | 0.536741 | 0.810014 |

The S1 error is 17.3 times the public/private population difference. Rebinning uses S1's own public edges and its `searchsorted(..., side='left')` rule. The comparable error before within-bin reconstruction means independent numeric decoding is not the main explanation. Excluding rows identified by the public rare detector worsens the discrepancy, so the rare-donor repair is not the principal explanation either.

| Pair | B | S1 C | A_bg |
|---|---:|---:|---:|
| BMI, HDL | -0.033419 | 0.773282 | -0.070005 |
| TG, HDL | -0.048813 | 0.736517 | -0.066983 |
| age, HDL | -0.008482 | 0.772316 | 0.000104 |
| age, sex | -0.054017 | 0.689827 | -0.040484 |
| age, ALT | -0.029244 | 0.714266 | -0.032439 |

All 9 C features have large positive Spearman correlations with physical row index, ranging from 0.613851 (smoking) to 0.837636 (ALT). In contrast all B/A_bg correlations with row index are below 0.037 in absolute value. This is strong evidence of an ordering artifact requiring a synthesis-code check; this analysis alone does not establish exactly where the artifact was introduced. Shuffling all rows together would leave feature-to-feature correlations unchanged and would not solve it.

The excessive association exists within categorical groups, not only between them. Across 21 numeric-numeric pairs, mean absolute correlation in sex=0 rows is 0.760 in C vs 0.232 in B and 0.243 in A_bg. Among smoking=0 rows it is 0.765 in C vs 0.210 in B and 0.195 in A_bg.

S1's workload includes 21 numeric-numeric pairs and sex-smoking but omits **14 of the 36 evaluated pairs**: all 7 numeric features crossed with sex and all 7 crossed with smoking. None of the 36 evaluated pairs are fixed queries; the 9 fixed queries are each feature against outcome_state. S1 allocates 0.35 of rho to oneways and splits the remainder 3:1 between fixed and adaptive stages. Thus the adaptive pool receives only rho=0.01625 out of total rho=0.1. This is a second design problem after the ordering artifact, especially because final time/onset/death are overwritten with public outcome donors.

For reference only, feeding A_bg's observed correlations into the same reported scales yields a mean component of approximately 0.929 and a tail component of 0.938. This is not a projected candidate score: row counts, disclosure constraints, other utility components, and privacy must still be checked for any future generator. It shows that public correlation structure is a promising source for a diagnostic target or postprocessing prior.

Relevant generator evidence: `S1/scripts/generate_pwscup.py` lines 794–825 build the workload; 842–853 allocate fixed/adaptive stages; 879–883 call `table.synthesize(method='round')`; 899–925 independently sample each numeric value within its bin; 1011–1022 apply rare and outcome donor processing. Workload details are in `S1/config/aim_main_team10_S1.json`, and `S1/score.json` contains the official local evaluation.

Reproduce from the repository root with the existing scoring Python environment:

```powershell
& '../pwscup2026-kit/starter/.venv/Scripts/python.exe' 'main_round/anonymization/team10/S/research/s1_improvement_20260917/correlation_diagnosis/diagnose.py'
```
