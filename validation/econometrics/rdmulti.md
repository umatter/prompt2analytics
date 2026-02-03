# Multi-Cutoff Regression Discontinuity (rdmulti) Validation

## Method Overview

Multi-cutoff RD extends standard regression discontinuity designs to handle multiple cutoffs sharing the same running variable. It estimates treatment effects at each cutoff and optionally pools them into a single weighted estimate.

### Key Features

1. **Multiple cutoffs**: Different thresholds c1, c2, ..., cJ with same running variable
2. **Pooled estimation**: Combine estimates across cutoffs using inverse-variance or sample-size weighting
3. **Cutoff-specific effects**: Allow heterogeneous effects at each cutoff
4. **Heterogeneity test**: Chi-squared test for whether effects differ across cutoffs
5. **Flexible bandwidth**: Global, per-cutoff optimal, or user-specified bandwidths

## Reference Implementation

- **R Package**: `rdmulti` (Cattaneo, Titiunik, Vazquez-Bare)
- **CRAN**: https://cran.r-project.org/package=rdmulti
- **Documentation**: https://rdpackages.github.io/rdmulti/
- **Version Used**: 1.1 (2023)

## Mathematical Framework

For multiple cutoffs c1, ..., cJ with running variable X:

**Cutoff-specific effects:**
```
tau_j = E[Y(1) - Y(0) | X = c_j] for each cutoff j
```

**Pooled estimate (inverse-variance weighted):**
```
tau_pooled = sum_j(w_j * tau_j) where w_j = (1/se_j^2) / sum_k(1/se_k^2)
```

**Pooled standard error (assuming independence):**
```
se_pooled = sqrt(sum_j(w_j^2 * se_j^2))
```

**Heterogeneity test statistic:**
```
Q = sum_j((tau_j - tau_pooled)^2 / se_j^2) ~ chi^2(J-1)
```

## Test Cases

### Test Case 1: Two Cutoffs with Different Effects

**Synthetic Data Generation:**
```r
set.seed(42)
n <- 120

# Cutoff 1: c = 0, true effect = 1.5
x1_left <- runif(30, -1.5, 0)
y1_left <- 1.0 + 0.3 * x1_left + rnorm(30, 0, 0.2)
x1_right <- runif(30, 0, 1.5)
y1_right <- 1.0 + 0.3 * x1_right + 1.5 + rnorm(30, 0, 0.2)

# Cutoff 2: c = 2, true effect = 2.5
x2_left <- runif(30, 0.5, 2)
y2_left <- 2.0 + 0.3 * x2_left + rnorm(30, 0, 0.2)
x2_right <- runif(30, 2, 3.5)
y2_right <- 2.0 + 0.3 * x2_right + 2.5 + rnorm(30, 0, 0.2)

# Combine data
y <- c(y1_left, y1_right, y2_left, y2_right)
x <- c(x1_left, x1_right, x2_left, x2_right)
cutoff_group <- c(rep(0, 60), rep(1, 60))
```

**Expected Results (approximate, due to noise):**
- Cutoff 1 effect: ~1.5 (tolerance: +/- 0.5)
- Cutoff 2 effect: ~2.5 (tolerance: +/- 0.5)
- Heterogeneity test: Likely significant if effects differ substantially

### Test Case 2: Pooling Weights Verification

**Unit Test for Weighting:**

Sample size weighting (n_eff = [80, 40]):
```
w1 = 80 / 120 = 0.667
w2 = 40 / 120 = 0.333
```

Inverse variance weighting (se = [0.1, 0.2]):
```
1/var1 = 1/0.01 = 100
1/var2 = 1/0.04 = 25
total = 125

w1 = 100 / 125 = 0.8
w2 = 25 / 125 = 0.2
```

### Test Case 3: Heterogeneity Test

**Setup:**
- Effect 1: 1.0 with SE 0.1
- Effect 2: 3.0 with SE 0.1

**Expected:**
- Q statistic should be very large (> 50)
- p-value should be very small (< 0.001)
- Test should indicate significant heterogeneity

## Rust Implementation Validation

### Test Results Summary

| Test Name | Status | Notes |
|-----------|--------|-------|
| `test_pooling_weights_sample_size` | PASS | Weights computed correctly |
| `test_pooling_weights_inverse_variance` | PASS | Weights computed correctly |
| `test_pooling_weights_equal` | PASS | Equal weights = 1/J |
| `test_pooled_estimate` | PASS | Weighted average correct |
| `test_heterogeneity_test` | PASS | Detects significant heterogeneity |
| `test_heterogeneity_test_homogeneous` | PASS | No false positive |
| `test_assign_to_nearest_cutoff` | PASS | Correct assignment |
| `test_rd_multi_dataset` | PASS | Full pipeline works |
| `test_rd_multi_auto_assignment` | PASS | Auto-assign by nearest cutoff |
| `test_rd_multi_insufficient_data` | PASS | Proper error handling |
| `test_rd_multi_global_bandwidth` | PASS | Global bandwidth applied |
| `test_display_formatting` | PASS | Output formatting correct |

### Tolerance Guidelines

| Parameter | Tolerance |
|-----------|-----------|
| Weights | < 0.01 |
| Pooled effect (synthetic data) | +/- 1.0 |
| Cutoff-specific effects (noisy) | +/- 0.5 |
| Heterogeneity p-value (significant) | < 0.05 |

## Known Differences from R Implementation

1. **Bandwidth selection**: Our implementation uses MSE-optimal bandwidth from the base `run_rd` function, which follows CCT (2014). The R `rdmulti` package may use slightly different tuning parameters.

2. **Bias correction**: We use the robust bias-corrected estimate from the base RD estimator for pooling, consistent with CCT recommendations.

3. **Standard errors**: The pooled SE assumes independence across cutoffs, which is appropriate when cutoffs define non-overlapping populations.

## References

- Cattaneo, M. D., Titiunik, R., Vazquez-Bare, G., & Keele, L. (2016). "Interpreting Regression Discontinuity Designs with Multiple Cutoffs". *Journal of Politics*, 78(4), 1229-1248.
- Cattaneo, M. D., Titiunik, R., & Vazquez-Bare, G. (2020). "Analysis of Regression Discontinuity Designs with Multiple Cutoffs or Multiple Scores". *Stata Journal*, 20(4), 866-891.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs". *Econometrica*, 82(6), 2295-2326.
- R package `rdmulti`: https://cran.r-project.org/package=rdmulti
