# Validation: Hausman Specification Test

## Method Overview

The Hausman test compares Fixed Effects (FE) and Random Effects (RE) estimators to determine which is more appropriate. It tests whether the entity effects are correlated with the regressors.

**Null Hypothesis**: H₀: RE is consistent (use RE)
**Alternative**: H₁: RE is inconsistent, FE is consistent (use FE)

**Test Statistic**:
```
H = (β_FE - β_RE)' [Var(β_FE) - Var(β_RE)]⁻¹ (β_FE - β_RE) ~ χ²(k)
```

where k is the number of time-varying regressors.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `phtest()` | 2.6-x |

## Test Cases

### Test 1: Grunfeld Dataset - Hausman Test

**Dataset**: `validation/datasets/grunfeld.csv` (n=200)

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

# Estimate both models
fe_fit <- plm(inv ~ value + capital, data = pdata, model = "within")
re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")

# Hausman test
phtest(fe_fit, re_fit)

# Expected output:
#         Hausman Test
# data:  inv ~ value + capital
# chisq = 2.3304, df = 2, p-value = 0.3119
# alternative hypothesis: one model is inconsistent
```

**Results Comparison**:

| Statistic | R's phtest | p2a Rust | Tolerance |
|-----------|------------|----------|-----------|
| χ² statistic | 2.33 | ~2.33 | 0.1 |
| df | 2 | 2 | exact |
| p-value | 0.312 | ~0.312 | 0.01 |

**Interpretation**: p-value > 0.05 → fail to reject H₀ → RE is acceptable

**Rust Test**: `crates/p2a-core/src/econometrics/panel.rs::tests::test_validate_hausman_grunfeld`

---

### Test 2: Synthetic Data Where FE Is Required

Create data where entity effects are correlated with regressors (RE invalid).

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 100
n_periods <- 10
n <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)

# Entity effect correlated with x
alpha <- rnorm(n_entities)
x <- alpha[entity] + rnorm(n)  # x is correlated with entity effect!
y <- alpha[entity] + 2.0 * x + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x = x, entity = factor(entity), time = factor(time))
pdata <- pdata.frame(data, index = c("entity", "time"))

fe_fit <- plm(y ~ x, data = pdata, model = "within")
re_fit <- plm(y ~ x, data = pdata, model = "random")

phtest(fe_fit, re_fit)
# Should reject H0 (low p-value) → use FE
```

**Validation Criteria**:
- p-value < 0.05 (reject H₀)
- Large χ² statistic

---

### Test 3: Synthetic Data Where RE Is Valid

Create data where entity effects are uncorrelated with regressors (RE valid).

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 100
n_periods <- 10
n <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)

# Entity effect uncorrelated with x
alpha <- rnorm(n_entities)
x <- rnorm(n)  # x is independent of entity effect
y <- alpha[entity] + 2.0 * x + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x = x, entity = factor(entity), time = factor(time))
pdata <- pdata.frame(data, index = c("entity", "time"))

fe_fit <- plm(y ~ x, data = pdata, model = "within")
re_fit <- plm(y ~ x, data = pdata, model = "random")

phtest(fe_fit, re_fit)
# Should fail to reject H0 (high p-value) → RE is acceptable
```

**Validation Criteria**:
- p-value > 0.05 (fail to reject H₀)
- Small χ² statistic

---

### Test 4: Multiple Regressors

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

# With multiple regressors
fe_fit <- plm(inv ~ value + capital, data = pdata, model = "within")
re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")

phtest(fe_fit, re_fit)
# df = 2 (number of time-varying regressors)
```

**Validation Criteria**:
- Degrees of freedom = number of regressors in FE model
- Test uses difference in coefficients for all regressors

---

## Test Statistic Details

The Hausman statistic is computed as:
```
H = (β̂_FE - β̂_RE)' [V̂(β̂_FE) - V̂(β̂_RE)]⁻¹ (β̂_FE - β̂_RE)
```

Under H₀, the difference in variance matrices is positive semi-definite.

**Note**: In practice, the variance difference matrix may not be positive definite due to numerical issues. Some implementations use a generalized inverse.

## Numerical Precision Summary

| Dataset | χ² Precision | p-value Precision |
|---------|-------------|-------------------|
| Grunfeld | < 0.1 | < 0.01 |
| Synthetic | < 0.5 | < 0.02 |

## Known Differences

1. **Variance estimation**: Different variance estimators may give different results.
2. **Regularization**: Some implementations add small diagonal to avoid singularity.
3. **One-sided test**: Standard Hausman is two-sided; some argue for one-sided.

## Running the Tests

```bash
# Run Hausman test validation
cargo test -p p2a-core -- hausman

# Run with output
cargo test -p p2a-core -- test_hausman --nocapture
```

## References

- Hausman, J.A. (1978). "Specification Tests in Econometrics". *Econometrica*, 46(6), 1251-1271.
- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*, 5th ed. Wiley. Chapter 4.
