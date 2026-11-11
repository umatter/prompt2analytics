# twang - GBM Propensity Score Estimation Validation

## Method Overview

The `twang` module (`run_twang`) estimates propensity scores by gradient boosting as R's twang
does with `gbm`, and picks the number of trees that minimises a covariate-balance criterion.

### Algorithm

`gbm(distribution = "bernoulli", bag.fraction = 1, train.fraction = 1)`:

1. **Initialization**: F_0 = log(n_treated / n_control) (gbm's `initF`).
2. **Boosting iterations** (m = 1 to `n_trees`):
   - Residuals r_i = t_i - p_i, p_i = sigmoid(F_{m-1}(x_i)) (unclipped).
   - Grow a regression tree on r best-first, as gbm: each terminal node keeps its best split
     (improvement n_L n_R / n (mean_L - mean_R)^2 over the midpoints of consecutive distinct
     values, left `x < midpoint`, both children with at least `min_node_size` rows,
     `n.minobsinnode`), and the node with the largest improvement splits, `interaction_depth`
     times or until no node can split. Every feature is sorted once per run; a split
     partitions the sorted lists stably, so a tree costs O(depth k n).
   - Newton leaf values: sum_leaf r / sum_leaf p (1 - p).
   - F_m = F_{m-1} + shrinkage * tree(x).
   - Evaluate the stop method's criterion with the weights of p_m (only that statistic per
     iteration; KS swept along the presorted orders).
3. **Optimal iteration**: the first minimum of the criterion over iterations
   max(1, `min_iterations`)..=`n_trees`. An early stop is opt-in: with `balance_threshold > 0`
   the run stops at the first iteration >= `min_iterations` below the threshold, and the
   optimum is taken over the iterations run (with a warning). An optimum at the last
   iteration gives the warning "the balance criterion was still decreasing at n_trees = N;
   increase n_trees".

The criterion follows twang's `ps.summary.new2`: ES = weighted mean difference / an SD fixed
from the unweighted sample (treated SD for ATT, SD of the full sample for ATE, control SD for
ATC; n - 1, every covariate continuous), KS = the weighted two-sample Kolmogorov-Smirnov
statistic; es.mean / es.max / ks.mean / ks.max over the covariates.

### Key Differences from R twang

| Feature | R twang | p2a twang |
|---------|---------|-----------------|
| Base learner | gbm trees, Bernoulli, Newton leaves | the same (R-pinned to gbm's link) |
| Interaction depth | default 3 | honoured; default 1 |
| Number of trees | default 10000 | default 3000 |
| Estimand | default ATE | default ATT |
| Iteration search | 25-point grid, then refined between neighbours | global argmin over every iteration (never worse) |
| Early stop | none | opt-in (`balance_threshold > 0`) |
| Stop methods | es.mean, es.max, ks.mean, ks.max | all four |
| bag.fraction | default 1 | 1 (no subsampling) |

Until v0.5.0 the leaves were mean residuals (about a quarter of gbm's step at the same
shrinkage), `interaction_depth` was ignored, the run stopped at the first iteration after
`min_iterations = 100` whose criterion fell below 0.1, and ES divided by the pooled SD of the
weighted groups (CAUS-93, CAUS-108).

## Reference Implementation

- **Packages**: gbm 2.3.1 (R 4.3.3). twang itself is not installed here: its criterion is
  computed in R from twang's definitions on gbm's path.
- R code: `validation/r/cases/causal_weighting_t4.R`, case `twang_gbm_sim` (weighting_sim,
  n = 300; shrinkage 0.01, n.minobsinnode 10, 3000 trees; depth 1 ATT es.mean, depth 3 ATE
  ks.max, depth 1 ATC es.max).
- Registry: `validation/methods/causal_weighting_t4.json`, id `twang_gbm`.

**R-pinned** (`test_validate_twang_gbm_path`, `test_validate_twang_selection`, `Tol::coef(300)`):
`initF`; gbm's link at iterations 1, 10, 100, 1000 and 3000 (rows 1-10, depth 1 and 3); the
optimal iteration (exact: 3000, 389, 2980; the runner-up criterion values are at least 2e-5
away); the criterion at the optimum, at iteration 1 and at 3000; the propensity scores at the
optimum; ES and KS after weighting; ES before; ESS per group. The depth-1 ATT run has its
optimum at the last iteration and carries the boundary warning.

## Test Cases

### Test Case 1: Basic Propensity Score Estimation

**Data**: Synthetic dataset with known imbalance
- 40 observations (20 treated, 20 control)
- 2 covariates with different means by treatment group
- Treated group has higher mean on both covariates

**Configuration**:
```rust
TwangConfig {
    n_trees: 500,
    shrinkage: 0.05,
    stop_method: StopMethod::ESMean,
    estimand: TwangEstimand::ATT,
    min_iterations: 50,
    ..Default::default()
}
```

**Expected Behavior**:
- Propensity scores in (0, 1)
- Positive weights for all observations
- Balance (ES.Mean) should improve or remain stable after weighting

### Test Case 2: Stopping Methods

**Test**: All four stopping methods produce valid results

| Method | Metric Optimized |
|--------|------------------|
| ESMean | Mean absolute standardized effect size |
| ESMax | Maximum absolute standardized effect size |
| KSMean | Mean Kolmogorov-Smirnov statistic |
| KSMax | Maximum KS statistic |

### Test Case 3: Estimands

**Test**: ATT, ATE, ATC weight computation

| Estimand | Treated Weights | Control Weights |
|----------|-----------------|-----------------|
| ATT | 1 (normalized) | ps / (1-ps) |
| ATE | 1 / ps | 1 / (1-ps) |
| ATC | (1-ps) / ps | 1 (normalized) |

### Test Case 4: KS Statistic

**Data**: Perfect separation
- Treated: values 1, 2, 3
- Control: values 4, 5, 6

**Expected**: KS statistic = 1.0 (no overlap)

### Test Case 5: Effective Sample Size

**Data**: Uniform weights [1, 1, 1, 1]

**Expected**: ESS = n = 4

## Validation Results

| Test | Status | Notes |
|------|--------|-------|
| test_twang_basic | PASS | PS in (0,1), positive weights |
| test_twang_stop_methods | PASS | All 4 methods work |
| test_twang_estimands | PASS | ATT/ATE/ATC weights correct |
| test_decision_stump | PASS | Correct split selection (depth-1 tree, Newton leaves) |
| test_stump_sweep_matches_naive_search | PASS | Prefix-sum sweep equals the exhaustive stump search |
| test_tree_growth_matches_naive_best_first | PASS | Best-first growth equals a port of gbm's growth at depth 1-6 |
| test_twang_ks_presorted_matches_naive | PASS | Presorted KS equals the per-call sort, bit for bit |
| test_twang_es_uses_fixed_sd | PASS | ES over the unweighted treated / full / control SD |
| test_twang_early_stop_is_opt_in | PASS | Early stop only with balance_threshold > 0, with a warning |
| test_ks_statistic | PASS | KS = 1.0 for separated groups |
| test_ess_computation | PASS | ESS = n for uniform weights |
| test_balance_history | PASS | Balance tracked per iteration |
| test_twang_display | PASS | Formatted output |
| test_convenience_function | PASS | Wrapper function |
| test_twang_missing_column | PASS | Error handling |
| test_twang_no_treated | PASS | Error handling |

## Limitations

1. **No subsampling**: `bag.fraction = 1` only.
2. **No cross-validation**: the iteration is chosen by balance, as twang does.

## Missing values and column types

Missing values (null or NaN) in the treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: twang is not installed here; the missing-value refusals are not verified against it.

Tests: `test_validate_na_twang_columns` (case `causal_mediation_na`, registry `validation/methods/causal_na.json`, id
`weighting_na_refusal`).

## References

- Ridgeway, G., McCaffrey, D., Morral, A., Burgette, L., & Griffin, B.A. (2017).
  "Toolkit for Weighting and Analysis of Nonequivalent Groups: A Tutorial".
  RAND Corporation.

- McCaffrey, D.F., Ridgeway, G., & Morral, A.R. (2004). "Propensity Score
  Estimation with Boosted Regression for Evaluating Causal Effects in
  Observational Studies". *Psychological Methods*, 9(4), 403-425.

- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting
  Machine". *Annals of Statistics*, 29(5), 1189-1232.

## Rust Test Location

`crates/p2a-core/src/econometrics/twang.rs` (tests module)
