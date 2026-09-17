# S1 synthesis audit (2026-09-17)

This audit used installed source code and an artificial public uniform distribution only. It did not read private B, rerun the DP mechanism, change S1, change installed libraries, or create a submission.

## Confirmed defect in installed `mbi==1.1.0` rounding sampler

`mbi/markov_random_field.py:133-147` assigns stratified uniforms in original row order within each conditional parent group. When several child columns share their parent group, they receive the same increasing rank schedule. They become strongly positively correlated even if the model says they are conditionally independent.

The initial column and independent-component columns use `synthetic_col` (`:77-95`, `:174-176`), which explicitly shuffles rounded values. Thus disconnected components alone did not cause this toy failure. The failure is in the conditional branch.

`opendp/extras/mbi/_table/__init__.py:111` forwards `synthesize` to `model.synthetic_data(rows, method)`.

S1 selects `"synthesis": {"method": "round"}` and forwards it at `S1/scripts/generate_pwscup.py:879-882`. Its nine fixed edges all connect `outcome_state` with a covariate, which is a directly relevant star structure. Adaptive edges may change the actual sampling structure. S1 does not persist the released graphical model, so this audit cannot quantify exactly how much of S1's observed error came from this defect.

## Public toy reproduction

`toy_sampler_audit.py` constructs a model whose H, A, B and C variables are all independent and uniform. H has two categories; A/B/C have eight. The exact model correlations are zero. It samples 1,049 rows for each seed 0..9. `toy_sampler_results.json` contains every run.

| Model structure | Sampler | Range of A/B/C pairwise Pearson correlations across 10 runs |
|---|---|---:|
| Uniform star H-A/H-B/H-C | Installed round | 0.999275 to 0.999818 |
| Uniform star H-A/H-B/H-C | Installed sample | -0.060351 to 0.047254 |
| Uniform star H-A/H-B/H-C | Research round with shuffled group ranks | -0.055851 to 0.060954 |
| Independent singletons | Installed round | -0.068670 to 0.063372 |

Spearman results confirm the same failure: star/round gives 0.999276 to 0.999819, while star/sample gives -0.059863 to 0.047529. Both correlation types are recorded in the JSON with the seed and row-count metadata.

The research variant is constructed in memory from the installed function. It replaces the first stable group ordering with a fresh random row permutation followed by a stable group ordering, independently for each generated child. It preserves conditional count rounding while preventing reuse of deterministic rank assignments. This is a small diagnostic, not sufficient validation for production adoption.

## Safest next experiment

Use the already supported `method="sample"` in a new candidate, keeping all other settings fixed, and examine all utility metrics. This avoids modifying dependency internals and directly bypasses the confirmed defect. It introduces ordinary sampling variation in marginal counts, so utility improvement is not guaranteed.

A single final row shuffle cannot fix the defect: it leaves every inter-column correlation unchanged. Independently shuffling final columns would destroy legitimate modeled relationships. If preserving rounded conditional counts is valuable, evaluate a locally versioned corrected rounding implementation after the simple `sample` baseline.

For future DP runs, preserve the released DP model and/or DP measured marginals and the intermediate synthetic binned data. Synthesis and downstream postprocessing can then be compared on a single DP release without another access to B. The current S1 function returns only the synthetic frame (`generate_pwscup.py:883`) and writes only the postprocessed output (`:1180-1184`). Reconstructing the current unsaved model by rerunning OpenDP would be a new DP release.

## Other postprocessing observations

- Numeric columns are decoded independently within each bin by public-value sampling (`generate_pwscup.py:899-925`). This retains the sampler's between-bin row associations; random jitter within bins does not repair them. It can additionally weaken legitimate within-bin covariance.
- Public rare-row repair replaces all seven marker values together (`:502-504`, `:530-532`). It preserves joint structure inside that marker block, but may change its relationship with sex/smoking/outcomes. The demotion donor order is deterministic by public distance and targets ascending output row indices (`:486-504`). Quantifying this stage needs pre-repair output, which is not saved.
- Public outcome replacement changes time/onset/death only (`:658-660`); it cannot repair wrong correlations among age/BMI/SBP/TG/HDL/ALT/FPG. It overwrites the synthesized outcome state after AIM has spent budget on nine covariate/outcome fixed marginals. This is a separate potential budget-allocation improvement to test after fixing synthesis.

Runtime used: OpenDP 0.15.1, mbi 1.1.0, existing `pwscup2026-kit/opendp_aim/.venv`.
