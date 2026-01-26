# twang - GBM Propensity Score Estimation Validation

## Method Overview

The `twang` module implements propensity score estimation using gradient boosted decision stumps with automatic tuning for covariate balance. This is a simplified version of the R twang package approach.

### Algorithm

1. **Initialization**: F_0(x) = log(n_treated / n_control) (log-odds of being treated)
2. **Boosting iterations** (m = 1 to M):
   - Compute pseudo-residuals: r_i = y_i - sigmoid(F_{m-1}(x_i))
   - Fit decision stump (single-split tree) to residuals
   - Update: F_m(x) = F_{m-1}(x) + shrinkage * stump(x)
   - Compute balance metrics with current propensity-based weights
   - Track optimal stopping point based on balance metric
3. **Return** propensity scores from optimal iteration

### Key Differences from R twang

| Feature | R twang | p2a-stats twang |
|---------|---------|-----------------|
| Base learner | Full GBM trees | Decision stumps only |
| Interaction depth | Configurable (default 3) | Fixed at 1 |
| Stop methods | es.mean, es.max, ks.mean, ks.max | All four supported |
| Cross-validation | For optimal trees | Not implemented |

## Reference Implementation

- **Package**: twang 2.6 (R)
- **URL**: https://cran.r-project.org/package=twang
- **Version validated against**: Conceptual alignment; simplified implementation

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
| test_decision_stump | PASS | Correct split selection |
| test_ks_statistic | PASS | KS = 1.0 for separated groups |
| test_ess_computation | PASS | ESS = n for uniform weights |
| test_balance_history | PASS | Balance tracked per iteration |
| test_twang_display | PASS | Formatted output |
| test_convenience_function | PASS | Wrapper function |
| test_twang_missing_column | PASS | Error handling |
| test_twang_no_treated | PASS | Error handling |

## Limitations

1. **Simplified base learner**: Uses decision stumps (depth=1) instead of full GBM trees
2. **No cross-validation**: Optimal iteration selected by simple tracking, not CV
3. **Fixed interaction depth**: Cannot capture complex covariate interactions
4. **No bag fraction**: No subsampling for variance reduction

## Usage Notes

For production-quality GBM propensity scores with complex interactions, consider:
- R twang package (full GBM)
- Python `causalml` or `econml` packages
- This implementation is suitable for educational purposes and simple use cases

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
