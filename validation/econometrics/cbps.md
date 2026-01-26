# Covariate Balancing Propensity Score (CBPS) Validation

## Method Overview

Covariate Balancing Propensity Score (CBPS) uses Generalized Method of Moments (GMM) to simultaneously estimate propensity scores and achieve covariate balance. Unlike standard logistic regression, CBPS explicitly targets balance as part of estimation by incorporating covariate balance conditions into the GMM moment conditions.

## Mathematical Formulation

### Moment Conditions

CBPS combines two sets of moment conditions:

1. **Score equations** (from logistic regression):
   ```
   E[X_i * (T_i - p(X_i; beta))] = 0
   ```

2. **Balance conditions** (explicit covariate balance):
   ```
   E[T_i * X_i / p(X_i; beta) - (1-T_i) * X_i / (1-p(X_i; beta))] = 0
   ```

### GMM Estimation

The combined moment conditions are solved using two-step GMM:
- Step 1: Use identity weighting matrix (2SLS-style)
- Step 2: Update with optimal weighting matrix (inverse of moment condition variance)

### J-Test for Overidentification

When using `ExactBalance` method, the model is overidentified (2k moment conditions for k parameters). The J-statistic tests model specification:
```
J = n * g_bar' * W * g_bar ~ chi^2(k)
```

## Reference Implementation

### R Package: CBPS

The canonical implementation is the R `CBPS` package by Fong, Ratkovic, and Imai.

```r
# Installation
install.packages("CBPS")

# Usage
library(CBPS)

# Generate test data
set.seed(42)
n <- 200
x1 <- rnorm(n)
x2 <- rnorm(n)
# Treatment probability depends on x1 and x2
prob <- plogis(-0.5 + 0.8*x1 + 0.6*x2)
treatment <- rbinom(n, 1, prob)

# Run CBPS
fit <- CBPS(treatment ~ x1 + x2, ATT = FALSE)

# Results
summary(fit)
balance(fit)  # Balance statistics
```

### Expected Results

For the test data in `crates/p2a-core/src/econometrics/cbps.rs`:

| Statistic | R CBPS | Rust CBPS | Tolerance |
|-----------|--------|-----------|-----------|
| Converged | TRUE | TRUE | - |
| Max Std.Diff Before | >0.5 | ~0.5 | 0.1 |
| Max Std.Diff After | <0.1 | <0.15 | 0.05 |
| J-test df | k | k | - |

## Test Cases

### Test Case 1: Basic Binary Treatment

**Data**: 40 observations with clear imbalance between treated/control groups.

```r
# R reproduction
library(CBPS)

# Create imbalanced data
df <- data.frame(
  treatment = c(rep(1, 20), rep(0, 20)),
  x1 = c(0.8, 0.9, 1.0, 1.1, 1.2, 0.7, 0.85, 0.95, 1.05, 1.15,
         0.9, 1.0, 1.1, 1.2, 0.8, 0.75, 0.88, 0.92, 1.08, 1.12,
         0.2, 0.3, 0.4, 0.5, 0.6, 0.15, 0.25, 0.35, 0.45, 0.55,
         0.3, 0.4, 0.5, 0.6, 0.25, 0.18, 0.32, 0.42, 0.52, 0.62),
  x2 = c(0.6, 0.7, 0.8, 0.9, 1.0, 0.55, 0.65, 0.75, 0.85, 0.95,
         0.7, 0.8, 0.9, 1.0, 0.65, 0.58, 0.72, 0.78, 0.88, 0.98,
         0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.15, 0.25, 0.35, 0.45,
         0.2, 0.3, 0.4, 0.5, 0.15, 0.08, 0.22, 0.32, 0.42, 0.52)
)

# Fit CBPS
fit <- CBPS(treatment ~ x1 + x2, data = df, ATT = FALSE)
summary(fit)

# Check balance
balance(fit)
```

**Expected behavior**:
- CBPS should converge
- Balance should improve (lower max standardized difference after weighting)
- Propensity scores should be in (0, 1)
- IPW weights should be positive

### Test Case 2: Just-Identified (Logit Baseline)

When using `JustIdentified` method, CBPS reduces to standard logistic regression.

**Expected**:
- Coefficients should match `glm(family=binomial)` results
- No J-test (model is exactly identified)
- Balance may not be as good as `ExactBalance`

### Test Case 3: Balance Improvement

The key benefit of CBPS is improved covariate balance.

**Validation criteria**:
- `max_std_diff_after <= max_std_diff_before` (balance should not get worse)
- With typical imbalanced data, `max_std_diff_after < 0.15` (good balance)

## Rust Implementation Tests

Located in: `crates/p2a-core/src/econometrics/cbps.rs`

```bash
# Run CBPS tests
cargo test -p p2a-core cbps
```

### Test Functions

| Test | Description | Validates |
|------|-------------|-----------|
| `test_cbps_exact_balance` | Default method | Convergence, balance improvement |
| `test_cbps_just_identified` | Logit baseline | No J-test, valid propensity scores |
| `test_balance_table` | Balance diagnostics | Std diff calculation |
| `test_ipw_weights_normalization` | Weight computation | Sum to n_treated, n_control |
| `test_cbps_missing_column` | Error handling | Column not found error |
| `test_cbps_constant_treatment` | Edge case | Proper error for all-treated data |

## Tolerances

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Coefficients | 1e-4 | Some variation due to GMM optimization |
| Standard errors | 1e-3 | Sandwich estimator may differ slightly |
| Propensity scores | 1e-4 | Should be very close |
| Balance std diff | 0.05 | Accept small differences |
| J-statistic | 0.1 | Chi-squared approximation |

## Performance Notes

- CBPS uses iterative GMM optimization which is slower than simple logit
- For large datasets (n > 10000), consider using `JustIdentified` for initial exploration
- The `ExactBalance` method provides better balance at cost of computation

## References

### Primary Reference

- Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
  *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
  DOI: 10.1111/rssb.12027

### R Implementation

- Fong, C., Ratkovic, M., & Imai, K. (2022). CBPS: Covariate Balancing Propensity Score.
  R package version 0.23. https://CRAN.R-project.org/package=CBPS

### GMM Theory

- Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators."
  *Econometrica*, 50(4), 1029-1054.

### Propensity Score Weighting

- Rosenbaum, P.R. & Rubin, D.B. (1983). "The Central Role of the Propensity Score in Observational Studies for Causal Effects."
  *Biometrika*, 70(1), 41-55.

## Status

- [x] Core algorithm implemented
- [x] Unit tests passing (modulo pre-existing errors in other modules)
- [x] MCP tool added
- [x] Documentation complete
- [ ] Benchmark vs R CBPS package
- [ ] Large-scale validation study
