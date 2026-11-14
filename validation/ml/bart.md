# BART (regression and probit)

## Method

`p2a_core::ml::bart` samples a sum of trees (Chipman, George & McCulloch 2010) as `BART::wbart` (regression)
and `BART::pbart` (probit) do.

**Priors (regression).** The response is centred at its mean `fmean`; the trees start at 0 and fit `y - fmean`.
The leaf prior SD is `tau = (max y - min y) / (2 k sqrt(n_trees))`. `sigest` is the residual SD of the linear
fit of `y` on the non-constant covariates (or `sd(y)` with at least as many covariates as rows; p2a also falls
back to `sd(y)` when that fit is exact), and `lambda = sigest^2 qchisq(1 - q, nu) / nu`. A constant `y` is
refused. `sigest`, `lambda` and `leaf_prior_sd` are reported.

**Probit.** With `classification = true` (`y` coded 0/1, both classes present):

- `offset = qnorm(mean(y))`; trees start at 0;
- the latent SD is fixed at 1 and the leaf prior SD at `3 / (k sqrt(n_trees))`;
- each sweep draws `z_i ~ N(offset + f(x_i), 1)` truncated by `y_i` (Albert & Chib 1993), by inversion, or by
  Robert's (1995) exponential sampler when the side's probability is below 1e-100, and fits the trees to
  `z - offset`;
- each kept draw stores `Phi(offset + f)`; `predictions` are the posterior mean probabilities (pbart's
  `prob.train.mean`). `Phi` is R's `pnorm` (r_mathlib).

**Kernel.** Split rules come from a fixed grid of `numcut` (default 100) cutpoints per covariate, built as
`BART::bartModelMatrix` builds it (an even grid between the extremes, or midpoints between the unique values
when there are fewer than `numcut`); rows with `x < cut` go left. Each tree gets one birth or death proposal
per sweep with the Chipman-George-McCulloch acceptance ratio (tree-prior term
`p_d (1 - p_l)(1 - p_r) / (1 - p_d)`, the move and node-choice probabilities, the leaf likelihood with the
leaf value integrated out), then fresh leaf values from their conjugate posteriors. The split variable is
drawn among the covariates with cutpoints; a child with fewer than `min_node_size` rows (default 5, wbart's
fixed value) rejects the birth, and a node at `max_depth` (default 10) cannot split.

**Importance.** `variable_count_mean` is the mean number of splits on each covariate per kept draw, over all
trees (wbart's `colMeans(varcount)`); `variable_importance` holds its shares, all zeros when no kept draw has a
split.

**Prediction.** With `keep_trees: true` every kept draw's trees are stored compactly (about
`n_mcmc x n_trees x (2 s + 1) x 24` bytes for `s` splits per tree, under the allocation ceiling) and
`bart_predict` scores new rows; on the training rows it repeats `predictions` and the interval bit for bit.
Without kept trees it says how to keep them. Intervals in both modes are percentiles of the draws, R's
`quantile(type = 7)`.

## R references

`validation/r/cases/ml_bart_t4.R` (BART 2.9.10) and `validation/r/cases/ml_boosting_t3.R`.

- `ml_bart_wbart_calibration` (`generated/bart_friedman.csv`, Friedman design, n = 200, six covariates): wbart's
  prior quantities and the `bartModelMatrix` cutpoint grid, exactly.
- `ml_bart_wbart_posterior` (same data): `wbart` with 50 trees, 300 burn-in and 1,000 kept draws, run as 100
  seeded chains; each value is the mean over the chains of a per-chain summary and `*_mc_sd` its SD across them.
- `ml_bart_pbart_posterior` (`generated/bart_probit_sim.csv`, n = 300): `pbart` with 50 trees, 200 burn-in and
  300 kept draws, the same kind of summaries over 100 chains.
- `ml_bart_probit_single_leaf` (`generated/bart_probit_small.csv`, n = 40): with one tree that cannot split, the
  posterior of `Phi(offset + f)`, `f ~ N(0, 9 / k^2)`, by `stats::integrate`.
- `ml_bart_probit_glm` (`generated/bart_probit_sim.csv`): `glm(binomial(link = "probit"))` probabilities, a
  property target.

## Decisive values

| Quantity | R | Monte Carlo SD |
|---|---:|---:|
| wbart posterior mean of sigma | 0.9093 | 0.0453 |
| wbart mean posterior SD of f | 0.6619 | 0.0318 |
| wbart RMSE of the posterior mean against the true f | 0.7780 | 0.0220 |
| wbart 95% coverage of the true f | 0.895 | 0.029 |
| pbart offset | -0.1256613 | exact |
| pbart mean posterior probability | 0.45096 | 0.00167 |
| Single leaf: posterior mean probability | 0.3515426 | |
| Single leaf: posterior SD of the probability | 0.0735486 | |

## Tests

- `ml_bart_regression` (`validation/methods/ml_bart_t4.json`): `test_validate_bart_wbart_calibration`,
  `test_validate_bart_wbart_posterior`.
- `ml_bart_probit_pbart` (same file): `test_validate_bart_pbart_posterior`.
- `ml_bart_probit` (`validation/methods/ml_boosting_t3.json`): `test_validate_bart_probit_single_leaf_posterior`,
  `test_validate_bart_probit_tracks_glm_probit`.
- Properties: `bart_samples_the_tree_prior_when_the_data_are_silent` (the chain's split count against the
  restricted tree prior by rejection sampling), `bart_grow_prior_ratio_matches_cgm`,
  `bart_integrated_leaf_likelihood_matches_quadrature`, `bart_birth_death_ratio_is_reversible`,
  `bart_birth_respects_min_node_size_and_max_depth`, `bart_trees_hold_only_reachable_nodes`,
  `bart_importance_counts_splits_in_every_kept_draw`, `bart_centres_the_response_and_recovers_sigma`,
  `bart_refuses_a_constant_response`, `bart_predict_on_the_training_rows_repeats_the_fit`,
  `bart_predict_without_kept_trees_says_how`, `bart_predict_checks_columns_and_finiteness`,
  `bart_keep_trees_respects_the_allocation_ceiling`, `test_truncated_normal_draws_stay_on_their_side`,
  `bart_probit_rejects_non_binary_and_single_class`, `bart_intervals_are_type7_percentiles`.

## Tolerances

Calibration `Tol::coef(200)`, `numcut` exact. Posterior summaries: one p2a chain (seed 46) within 5 Monte Carlo
SDs of the mean of the 100 R chains, absolute; each split share by its own SD. pbart offset `Tol::coef(300)`.
Single leaf: offset `Tol::coef(40)`, posterior mean 0.009 and SD 0.006 absolute (5 SDs of 200 R replications).
glm property: mean absolute difference below 0.085 and correlation above 0.90.

## Wall-clock

`release`, `ci` features, base `f80b0131` against this tree: one fit per process (a warm-up call, then the median
of 3 calls, 1 at n = 10,000 and for the default configuration), five rounds with base and this tree interleaved per
row and the order alternating, one core (`taskset -c 2`), `RAYON_NUM_THREADS=1`, 2026-11-13, load average 26-46
(five lanes building on the other cores). Data: n x 5 uniform design, y = 2 x1 + 0.5 x2 + noise (probit: y above
its median). Cells are medians; "ratio" is this tree over the base, "min ratio" the ratio of the minima.

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| BART regression, 50 trees, 20 burn-in, 50 draws | 100 | 44.2 | 20.8 | 0.47 | 0.47 |
|  | 1,000 | 343.9 | 156.5 | 0.46 | 0.45 |
|  | 10,000 | 4,678.2 | 1,393.5 | 0.30 | 0.30 |
| Probit BART, same configuration | 100 | 49.3 | 22.8 | 0.46 | 0.44 |
|  | 1,000 | 433.7 | 173.3 | 0.40 | 0.41 |
|  | 10,000 | 5,479.9 | 1,593.6 | 0.29 | 0.30 |
| BART regression, defaults (200 trees, 1,000 + 1,000 sweeps) | 100 | 3,530.7 | 1,508.4 | 0.43 | 0.43 |
|  | 1,000 | 14,688.0 | 11,842.5 | 0.81 | 0.83 |

The single traversal per tree and sweep, the kept leaf index of every row and the integer cutpoint grid make each
sweep cheaper even though the corrected kernel grows somewhat larger trees. The default configuration at
n = 10,000 was not timed (several minutes per fit).

**`keep_trees = true`** (opt-in, not gated): defaults at n = 1,000, this tree only, three fits each in alternation
(2026-11-14): 12.0 s with kept trees against 12.4 s without (the same within the noise), peak RSS 30,508 kB against
10,820 kB.

**Peak RSS** (`/usr/bin/time -v`, one fit per process, defaults, 2026-11-14):

| Row | n | base (kB) | this tree (kB) | ratio |
|---|---:|---:|---:|---:|
| BART regression, defaults | 1,000 | 11,636 | 10,820 | 0.93 |
|  | 10,000 | 83,672 | 82,996 | 0.99 |

The `n_mcmc x n` draw matrix dominates on both sides.

## Deviations

BART 2.9.10's birth ratio (`bprop`) asks whether the parent of the leaf is a nog node with `tree::ntype()`, which
returns `'t'` for the root before testing for `'n'`. A birth at a child of a root with two leaves therefore takes
`Pnog = 1 / (nnogs + 1)` instead of `1 / nnogs`, and BART's trees are smaller than the prior implies: with the
data silenced (`k = 1e6`) wbart averages 1.19 splits per tree on the Friedman design, against 1.38 for the
restricted Chipman-George-McCulloch prior, which p2a's chain reproduces. p2a keeps the prior. So
`splits_per_tree` is not compared for either case, nor are pbart's `post_sd_avg` and `width95_avg`, which sit
about 3 Monte Carlo SDs above pbart's on average over seeds (registry deviations, finding ML-16). That the
quirk explains both gaps was checked by running p2a once with the root-nog quirk emulated: on the silenced design wbart's
splits per tree came out at R's 1.19, and on the pbart case splits per tree (1.224 against 1.214), `post_sd_avg`
(0.1184 against 0.1182) and `width95_avg` (0.4481 against 0.4471) all fell within 0.7 Monte Carlo SDs of pbart.
No other difference remains. The other summaries, the split shares among them, agree.

## Not pinned

The posterior summaries are distributional pins of one chain, not of the draws. The leaf values, latent draws and
the RNG are p2a's own.
