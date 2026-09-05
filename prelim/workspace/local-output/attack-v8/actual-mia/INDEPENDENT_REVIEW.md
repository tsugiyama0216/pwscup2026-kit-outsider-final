# Independent review: home-prior MIA shadow

## Bottom line

The implementation is internally consistent with the stated home-private
model and does not use hidden per-row feedback truth. It is suitable as a
model-based sensitivity test, not as independent evidence that the unknown
official memberships follow the sampled worlds. The reviewed result did not
pass validation and should not be promoted.

Any later candidate should remain rejected unless fresh validation passes both
the max-channel MIA power gate and explicit all-channel gates. Rare ordering is
correctly locked, but that lock can cause `max(all, rare)` to hide deterioration
in the all channel.

## What is correct

- The actual inputs are hash-pinned. The code reads B, B_self, A_bg, and public
  targets; it does not enumerate or read the token.
- The self AIA challenge identifies 163 known private rows, including 13 true
  rare rows. These are valid public/own-data facts, not leaderboard row truth.
- In the benchmark's official-model world, 150 latent private rows are sampled
  only from eligible home-prefecture, non-rare rows. This gives exactly 313
  private and 736 shared rows, with 655 non-home rows all shared and 81 eligible
  home rows shared **under the additional assumption described below**.
- The shared rows are assigned once without replacement to 20 target blocks,
  with per-target counts 25–27 and a total of 508; 228 shared rows remain
  unscored. No candidate is assigned to two scored target blocks.
- Known-private and rare rows are locked in every rerank. An independent check
  of the reviewed production artifact found zero changed cells in all 174
  locked rows, unchanged keys, and 17,293 changed cells only among unlocked
  rows.
- The uniformly sampled private world is a reasonable deliberately wrong
  negative control after locked rows are excluded: home status carries no
  private enrichment among its unlocked population.
- Production outputs are written only beneath local-output. The official v4
  input, v7 candidate, ZIPs, and token are not modified.

## Material caveats

### 508 is a soft prior, not hidden truth

Repository documentation explicitly describes 508 as a proportional prior
obtained by scaling 736 shared rows across the 20 observed targets. It is not a
validator constraint or known official total. The benchmark also assigns
shared rows to target identities uniformly at random. Its results therefore
measure performance under those assumptions.

The central home-private fact is stronger: under the stated construction,
non-home rows are shared and only 81 of 231 eligible home rows are shared. But
the official target partner of each shared row is unknown. Positive shadow
gains cannot by themselves validate that assignment.

### Truth generation and candidate use the same prior

The official-model shadow makes all 150 unknown private rows home-prefecture,
then evaluates a rerank that boosts non-home rows. This is an appropriate
sensitivity test for the rule-derived prior, but the advantage is partly built
into the data-generating model. Discovery/validation seed separation controls
Monte Carlo over that model; it does not supply new empirical official
membership labels.

### “All eligible rare rows are shared” is an extra assumption

The primary rule presently identified says that the AIA sample contains 13 rare
members. It does not by itself establish that the complete private block P_i
contains exactly 13 rare rows. The actual self challenge proves that 13 known
private rows are rare, while B_self contains 24 rare rows in total. Of the 11
remaining rare rows, ten are non-home and therefore shared if the home-private
construction is correct; one is an eligible home row whose private/shared
status is not proven by the AIA sampling quota.

The benchmark excludes every remaining rare row from the 150 latent-private
draw and therefore forces that one home rare row, as well as the ten non-home
rare rows, to be shared. Without a primary generator/rule proving that the AIA
sample exhausts private rare rows, `eligible_rare_all_shared=11` must be labeled
a modeling assumption rather than a structural invariant. This affects at most
one candidate's latent private/shared status, and rare scores are locked between
variants, so it does not overturn the reviewed rejection; it does narrow what
the shadow can claim to reproduce exactly.

### Probability transform is heuristic

`beta = log(231 / 81) = 1.048` is the log probability ratio implied by the
home/shared model. For an individual target with count `c` and partner
assignment independent of home, the corresponding probabilities are roughly
`p_nonhome = c / 736` and
`p_home = (81 / 231) * c / 736`; their exact log-odds difference is about
1.07–1.08 for c=25–27. The much larger ~1.94 log-odds difference applies to
membership in any of the 20 targets, not to each target column.

More importantly, v4 confidence values are strict ranking scores, not
calibrated posterior probabilities. Adding beta on their logits and then
reassigning the original score multiset is a bounded rerank heuristic, not a
calibrated Bayesian posterior update. Selection among beta variants must rely
on held-out behavior, with lower-tail controls.

### The shadow release mechanism is approximate

Each target source contains the sampled B24 member rows plus hidden A_bg
fillers, followed by generic noise scenarios. Real C targets contain their own
cohorts and heterogeneous anonymization mechanisms, including public-background
surrogates. Paired comparison against the same v4 baseline helps isolate the
prior rerank, but absolute and interaction effects may not transfer.

### Gate must protect the all channel

The original gate checked mean/median/win/p10 for `power=max(all, rare)` and
rare noninferiority, but did not explicitly require all-channel improvement.
Because rare rows are unchanged, rare can dominate and mask an all-channel
loss. A promotion gate should require at minimum:

- validation mean delta power > 0;
- validation median delta power > 0 and win rate >= 0.60;
- validation p10 delta power no worse than the declared small tolerance;
- validation mean delta all > 0 and a declared p10 all-channel lower bound;
- rare delta exactly/noninferior to zero in every validation observation;
- no favorable effect in the structurally wrong negative control;
- production locked-row equality and normal-format validation.

Given one official attempt, a broad rerank that changes hundreds of rows and
thousands of cells warrants a stricter lower-tail bar than a small targeted
change.

## Reviewed result snapshot

The available report at review time was an intentionally shortened run: one
discovery seed, one validation seed, one 10% noise scenario, and no DOMIAS.
Discovery selected `all_beta0p50` at +0.004 mean power, but validation was
-0.002 with 0% wins and -0.002 all-channel delta. The report correctly set
`gate_pass=false`. Its research production form changed 17,293 cells across
875 rows and all 20 targets, while preserving rare order.

This snapshot is an unambiguous rejection. A later broader run may refine the
sensitivity estimate, but it does not remove the soft-508, target-assignment,
uncalibrated-score, or common-prior caveats above.
