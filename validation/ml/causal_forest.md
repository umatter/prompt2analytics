# Causal Forest Validation

Reference: grf 2.x `causal_forest`, `predict` and `average_treatment_effect` (grf is not installed;
formula pins and property tests). The R code of every case is `validation/r/cases/ml_causal_forest.R`;
the registry entry is `validation/methods/ml_causal_forest.json`.

## Method

`p2a_core::ml::causal_forest` (`causal_forest`, `causal_forest_arrays`,
`causal_forest_arrays_with_nuisance`, `causal_forest_predict(_arrays)`, `average_treatment_effect(_with)`)
follows `grf::causal_forest(X, Y, W)`:

1. **Local centering.** `Y.hat = E[Y | X]` and `W.hat = P(W = 1 | X)` are out-of-bag predictions of two
   honest regression forests (`max(50, n_trees / 4)` trees each, min.node.size 5, honesty fraction 0.5,
   no depth cap), unless supplied (`causal_forest_arrays_with_nuisance`, grf's `Y.hat =`, `W.hat =`).
2. **Causal trees on the residuals** `Y - Y.hat`, `W - W.hat` (the R-learner moment), grown in groups of
   `ci_group_size` on shared half-samples, honest, with grf's gradient split: pseudo-outcomes
   `rho_i = (W~_i - mean W~)((Y~_i - mean Y~) - tau_node (W~_i - mean W~))`, criterion
   `sum_l^2 / n_l + sum_r^2 / n_r`, stabilised splits (alpha = 0.05). Empty honest leaves are pruned.
3. **Prediction** pools the leaves a unit falls into (grf's forest weights):
   `tau(x) = (avg(YW) - avg(Y) avg(W)) / (avg(WW) - avg(W)^2)`; training units are predicted out of bag.
4. **Variance** of each prediction by little bags with objective-Bayes debiasing (grf `variance.estimates`).
5. **ATE** from the AIPW scores
   `Gamma_i = tau_i + (W_i - W.hat_i) / (W.hat_i (1 - W.hat_i)) (Y_i - Y.hat_i - tau_i (W_i - W.hat_i))`:
   `mean(Gamma)` with SE `sqrt(sum((Gamma - mean)^2) / (n (n - 1)))`, a z statistic, a normal p-value and a
   normal interval (grf `average_treatment_effect`, `target.sample = "all"`, `method = "AIPW"`).

`oob_error` is the mean out-of-bag R-loss `(Y~_i - avg_i(Y~) - tau_i (W~_i - avg_i(W~)))^2` (grf's raw
error, `debiased.error + excess.error`).

| Setting | Default | Meaning |
|---|---|---|
| `n_trees` | 2000 | causal trees, rounded down to a multiple of `ci_group_size` (with a warning) |
| `min_node_size` | 5 | grf `min.node.size` of the causal trees |
| `mtry` | `ceil(sqrt(p))` | mean of the Poisson number of candidate covariates per node, clamped to [1, p] (grf: `min(ceil(sqrt(p) + 20), p)`; see Differences) |
| `honesty` | true | honest trees (estimation units repopulate the leaves) |
| `honesty_fraction` | 0.5 | share of each subsample that **estimates** the leaves |
| `sample_fraction` | 0.5 | subsample per tree; at most 0.5 when `ci_group_size > 1` (grf's rule) |
| `ci_group_size` | 2 | trees per little bag; 1 gives no variance estimates (NaN, with a warning) |
| `nuisance_trees` | `max(50, n_trees / 4)` | trees of each nuisance forest |
| `max_depth` | 10 | depth cap of the causal trees (p2a extension; the nuisance forests are not capped) |
| `seed` | 42 | one random stream: the Y.hat forest, the W.hat forest, then the causal trees |

## What is validated and how

grf is not a case dependency. Validation has four parts.

### (a) Formula pins

grf's formulas are written out in R from its sources (`InstrumentalRelabelingStrategy`,
`InstrumentalSplittingRule`, `RegressionSplittingRule`, `InstrumentalPredictionStrategy`,
`ObjectiveBayesDebiaser`, `get_scores.causal_forest`, `average_treatment_effect`). The Rust functions are
compared with `p2a_validation::Check`: `Tol::coef` for scores, estimates, tau and first stage, `Tol::se` for
SEs, and `Tol::stat` for decreases and variances, as in `validation/README.md`. Features and thresholds must
match exactly.

| Case | Values (R 4.3.3) | Test |
|---|---|---|
| `cf_split_node` (30 units; features x distinct, x2 tied) | node tau 0.7483450387030217; causal split at `min_node_size` 3: feature 0, threshold 0.5415, decrease 1.2856676822142568; at 5 (stabilisation binds): 0.485, 1.1280410596313275; x2 only: 0.45, 0.10702418809721084 | `test_validate_cf_split_node_causal` |
| `cf_split_node` | regression split (min child 2): feature 1, threshold 0.35, decrease 177.20196551311298 | `test_validate_cf_split_node_regression` |
| `cf_little_bags` (12 leaves, 6 groups) | tau 0.7800916208575865, first stage 0.17064544830360812, var_between 0.027432638634080634, group_noise 0.2200666810106443, debiased variance 0.0554310669510403; with group 0 incomplete: tau 0.7973513441446286, variance 0.06539969187496175 | `test_validate_cf_little_bags` |
| `cf_little_bags` | debiaser on six (vb, gn, G), including ratio -38.69 (continued fraction) | `test_validate_cf_bayes_debiaser` |
| `cf_aipw_formula` (n = 40) | estimate 1.2640632615045366, SE 0.538829230473634, subset rows 0..15 0.9631570086899984 (1.0059294613874787), 95% and 90% intervals, all 40 scores | `test_validate_cf_aipw_formula` |

### (b) Property tests on R-generated designs

The forest outputs (ATE, CATEs, variances) are **not R-pinned**. The tests hold seeded Rust fits to
properties of R values: the oracle AIPW with the true nuisances, naive differences, and Welch SEs. The
thresholds were calibrated on an R emulation of the algorithm (R's RNG, so the emulation is not
bit-comparable) and are at least twice the emulated deviations. The Rust column below is the seeded fit at the
time of writing.

| Case | Test (seed, trees) | Threshold | Emulation | Rust |
|---|---|---|---|---|
| `cf_confounded` y0 (tau = 0; naive 2.536) | `test_validate_cf_confounded_zero_effect_covers_zero` (4701, 400) | \|ATE\| < 0.15, CI covers 0, p > 0.05, SE within [0.8, 1.25] of the oracle 0.0738, \|mean CATE\| < 0.1 | ATE [-0.039, 0.009], SE [0.071, 0.076] | -0.0158 (0.0755) |
| `cf_confounded` y1 (tau = 1; naive 3.536) | `test_validate_cf_confounded_constant_effect_recovered` (4702, 400) | \|ATE - 1\| <= 0.15 and within 2.576 SE; naive - ATE > 20 SE; \|mean CATE - 1\| <= 0.15 | ATE [0.961, 0.997] | 0.9813 (0.0752) |
| `cf_confounded` yh (tau = 1 + 2 x1) | `test_validate_cf_confounded_heterogeneous_effect_tracks_tau` (4703, 400) | \|ATE - SATE\| <= 0.15; cor(CATE, tau) >= 0.8; x1 most important; tau(0.5, 0.1, 0.5) and tau(0.5, 0.9, 0.5) within 0.4 of 1.2 and 2.8, at least 1 apart | ATE [1.937, 2.005]; cor [0.940, 0.955] | 1.9527 (0.0799); cor 0.954; 1.113, 2.758 |
| `cf_confounded` y1, true Y.hat and W.hat | `test_validate_cf_confounded_supplied_nuisance_matches_oracle_aipw` (4704, 400) | \|ATE - oracle\| <= 0.25 oracle SE; SE within 5% of the oracle SE | ATE [0.9937, 0.9979], SE [0.0742, 0.0745] | 0.9932 (0.0742) |
| `cf_rct` first 500 / all 2000 rows | `test_validate_cf_rct_ate_se_shrinks_with_n` (4705, 200) | SE within [0.85, 1.3] of the Welch SE; \|ATE - oracle\| <= 0.75 oracle SE; SE(2000) / SE(500) in [0.4, 0.6] | SE [0.0905, 0.0933] / [0.0452, 0.0460] | 0.0932 / 0.0470 |
| `cf_rct` first 500 rows | `test_validate_cf_rct_average_treatment_effect_uses_the_aipw_scores` (4708, 200) | `average_treatment_effect` equals the fit's ATE; 95% half-width within [0.85, 1.3] of 1.96 Welch SE; a subset and level 0.9 match `aipw_ate` on the subset's scores | | |
| `cf_rare` (n = 200, 10 treated, tau = 5) | `test_validate_cf_rare_treatment_is_not_shrunk` (4706, 400) | \|mean CATE - 5\| <= 0.1, min CATE >= 4.85 (the old shrinkage gave 4.718), \|ATE - 5\| <= 0.15 and within 2.576 SE; low-propensity warning | mean CATE [4.973, 5.059], min >= 4.918 | mean 5.0035, min 4.949, ATE 4.935 (0.0996) |
| `cf_confounded` yh, depth 10 vs 0 | `test_validate_cf_confounded_oob_error_is_the_r_loss` (4707, 200) | same Y.hat; the deeper forest has the lower R-loss; R-loss in [0.9, 1.25] | 1.043-1.075 vs 1.118-1.143 | 1.0667 vs 1.1266 |

### (c) Release-only replication test

`test_causal_forest_ate_ci_coverage_over_replications` is ignored in debug builds. Run it with
`cargo test -p p2a-core --lib --release --no-default-features --features ci -- ml::causal_forest::tests::test_causal_forest_ate_ci_coverage_over_replications`.

The test fits 50 seeded designs with n = 500, x ~ U(0,1)^3, e = 0.1 + 0.8 x0, Y = 10 x0 + W + N(0, 1)
(tau = 1) and 200 trees each. It requires:

- the 95% AIPW interval covers in at least 42 of 50 (`pbinom(41, 50, 0.95) = 0.00076`);
- |mean ATE - 1| <= 0.08;
- mean(SE) / sd(ATE) in [0.6, 1.6];
- mean little-bags variance / CATE MSE in [0.5, 3];
- CATE interval coverage in [0.85, 0.995].

A replication refused for an out-of-bag `W.hat` of exactly 0 or 1 counts as not covered, and at most 2 may be
refused. The emulation, 3 batches of 50, gave:

- coverage 50, 45 and 48;
- ratios 1.023, 0.799 and 0.994;
- variance/MSE 1.71, 1.39 and 1.48;
- CATE coverage 0.959, 0.939 and 0.946.

The Rust run gave coverage 48/50 (1 refused), mean ATE 0.9998, ratio 1.106, variance/MSE 1.570 and CATE
coverage 0.952.

### (d) Guards and unit tests (not registered: they load no case)

- **Split search:** `causal_split_sweep_matches_naive_reference` checks the sweep against an O(n^2) reference
  on noisy data. `causal_split_tie_takes_one_of_the_tied_splits` and
  `causal_split_sweep_on_tied_data_matches_or_ties_the_reference` cover ties: either tied split is accepted.
- **Tree and draws:** `test_honest_tree_prune_promotes_the_non_empty_sibling` checks pruning;
  `test_poisson_count_matches_clamped_poisson_mean` checks the clamped Poisson draws.
- **Input checks:**
  - `test_aipw_scores_refuse_invalid_inputs`
  - `test_causal_forest_rejects_invalid_arrays`
  - `test_causal_forest_predict_checks_columns`
  - `test_causal_forest_dataset_wrapper_reports_missing_values`
  - `test_average_treatment_effect_rejects_bad_subsets`
  - `causal_forest_validates_counts_and_fractions`
  - `test_insufficient_data`
- **Fits and determinism:**
  - `test_causal_forest_basic`
  - `test_causal_forest_no_honesty`
  - `test_causal_forest_is_deterministic_per_seed`
  - `test_causal_forest_ci_group_size_one_has_no_variance_estimates`
  - `test_causal_forest_default_mtry`
- **Resource guards:**
  - `memory_bound_counts_at_least_the_nodes_every_tree_grows`
  - `memory_bound_grows_with_its_arguments_and_saturates`
  - `index_vectors_are_allocated_at_their_lengths`
  - `a_min_node_size_above_half_of_usize_neither_panics_nor_wraps`
  - `causal_forest_huge_n_trees_is_an_error_not_an_abort`
  - `causal_forest_honours_cancellation_and_rejects_non_finite_inputs`

## Differences from grf

- **Thresholds** are midpoints between consecutive distinct values. grf splits at the left value; both give
  the same partition of the training data, but new points in the gap are routed differently.
- **`honesty_fraction`** keeps p2a's meaning, the share of each subsample that **estimates** the leaves.
  grf's `honesty.fraction` is the share that grows the tree; the two are equal at 0.5. The growing part has
  `ceil(m (1 - honesty_fraction))` units, clamped so that both parts are non-empty.
- **`max_depth`** (default 10) caps the causal trees only.
- **The debiaser** returns 0 only for a zero spread; grf uses an absolute `|s| < 1e-10`, which is not
  scale-free. Its normal ratio uses a continued fraction below -30, where grf's `exp`/`erfc` underflow to NaN.
- **Not implemented:** sample weights, clusters, tuning, `orthog.boosting`, and `target.sample` other than
  `"all"`.
- **The default `mtry`** stays p2a's `ceil(sqrt(p))` as the mean of the Poisson candidate count; grf uses
  `min(ceil(sqrt(p) + 20), p)`. This is the registry deviation under ML-20. The tests set `mtry = p`.
- **Undefined estimates are errors** where grf returns NaN or Inf:
  - a unit that is in the subsample of every tree of a forest, so it has no out-of-bag estimate;
  - an estimated or supplied `W.hat` of 0 or 1;
  - a unit whose pooled first stage `avg(WW) - avg(W)^2` is below 1e-10.

  `causal_forest_predict_arrays` refuses a column count different from the fit, and non-finite rows. W must
  be 0/1.
- **Overlap warnings** follow grf's three warnings (`min(W.hat) <= 0.05` and/or `max(W.hat) >= 0.95`). The
  `target.sample` advice is replaced by a pointer to Crump, Hotz, Imbens and Mitnik (2009).
- **`n_trees` below `ci_group_size`** (fewer trees than one group) is an error.

## Findings closed

Each finding below names the test that failed before the change.

- **ML-20** (confounding ignored: no local centering). Before, `test_validate_cf_confounded_zero_effect_covers_zero`
  failed with a mean CATE of 4.69 on tau = 0, and `test_validate_cf_confounded_constant_effect_recovered`
  failed with an ATE of 3.49.
- **ML-21** (the ATE SE was not a sampling SE). Before, `test_validate_cf_rct_ate_se_shrinks_with_n` failed
  with an SE of 0.491 against a Welch SE of 0.088 at n = 500. The replication test (c) calibrates the
  interval.
- **ML-22** (`average_treatment_effect` used sd(CATE) / sqrt(n), and a subset index >= n panicked). Before,
  `test_validate_cf_rct_average_treatment_effect_uses_the_aipw_scores` and
  `test_average_treatment_effect_rejects_bad_subsets` failed (index out of bounds).
- **ML-74** (one-arm leaves shrank the CATE toward 0). Before, `test_validate_cf_rare_treatment_is_not_shrunk`
  failed with a mean CATE of 4.45 on tau = 5.
- **ML-76** (`oob_error` was not an out-of-bag loss of the estimator), closed as a side effect: `oob_error` is
  the R-loss (`test_validate_cf_confounded_oob_error_is_the_r_loss`; before, 0.0915 against 1.87).
- **ML-126** (split criterion), closed as a side effect: grf's gradient criterion
  (`test_validate_cf_split_node_causal`).

## References

- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random
  Forests. *Journal of the American Statistical Association*, 113(523), 1228-1242.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2),
  1148-1178.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika*,
  108(2), 299-319.
- Sexton, J., & Laake, P. (2009). Standard errors for bagged and random forest estimators. *Computational
  Statistics & Data Analysis*, 53(3), 801-811.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in
  estimation of average treatment effects. *Biometrika*, 96(1), 187-199.
- R package `grf` (Tibshirani, Athey & Wager): https://grf-labs.github.io/grf/

R code: `validation/r/cases/ml_causal_forest.R`.
