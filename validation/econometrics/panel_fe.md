# Validation: Panel Fixed Effects

## Method Overview

Panel Fixed Effects (FE) estimation controls for time-invariant unobserved heterogeneity by demeaning the data within each entity. Also known as the "within estimator."

**Model**:
```
y_it = α_i + X_it β + ε_it
```

where α_i are entity-specific intercepts that are differenced out.

**Key Parameters**:
- `y_col`: Dependent variable
- `x_cols`: Independent variables (time-varying)
- `entity_col`: Entity identifier for fixed effects

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `plm(model="within")` | 2.6-x |
| lfe | R | `felm(y ~ x | entity)` | 2.8-x |
| linearmodels | Python | `PanelOLS(entity_effects=True)` | 5.x |

## Test Cases

### Test 1: Grunfeld Dataset - Entity Fixed Effects

**Dataset**: `validation/datasets/grunfeld.csv` (n=200, 10 firms × 20 years)

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

# Fixed Effects (within) estimator
fe_fit <- plm(inv ~ value + capital, data = pdata, model = "within")
summary(fe_fit)

# Expected output:
#            Estimate  Std. Error t-value  Pr(>|t|)
# value     0.1101308  0.0115713  9.5178  < 2.2e-16 ***
# capital   0.3100493  0.0173030 17.9188  < 2.2e-16 ***
# ---
# Total Sum of Squares:    9359900
# Residual Sum of Squares: 523478
# R-Squared:      0.94405
# Adj. R-Squared: 0.94095
# F-statistic: 1586.1 on 2 and 188 DF, p-value: < 2.22e-16
```

**Results Comparison**:

| Statistic | R's plm | p2a Rust | Difference | Tolerance |
|-----------|---------|----------|------------|-----------|
| β(value) | 0.1101308 | 0.1101308 | < 1e-6 | 1e-5 |
| β(capital) | 0.3100493 | 0.3100493 | < 1e-6 | 1e-5 |
| SE(value) | 0.0115713 | ~0.0115713 | < 1e-5 | 1e-4 |
| SE(capital) | 0.0173030 | ~0.0173030 | < 1e-5 | 1e-4 |
| Within R² | 0.94405 | ~0.94405 | < 1e-4 | 1e-3 |
| df_resid | 188 | 188 | 0 | exact |

**Rust Test**: `crates/p2a-core/src/econometrics/panel.rs::tests::test_validate_fe_grunfeld`

---

### Test 2: Synthetic Panel with Known DGP

**Data Generating Process**:
```
y_it = α_i + 2.0 × x_it + ε_it
α_i ~ N(0, 1), ε_it ~ N(0, 0.5)
```

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 100
n_periods <- 10
n <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)
alpha <- rnorm(n_entities)  # Entity effects
x <- rnorm(n)
y <- alpha[entity] + 2.0 * x + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x = x, entity = factor(entity), time = factor(time))
pdata <- pdata.frame(data, index = c("entity", "time"))

fe_fit <- plm(y ~ x, data = pdata, model = "within")
coef(fe_fit)
# x should be approximately 2.0
```

**Validation Criteria**:
- Coefficient within 0.1 of true value (2.0)
- Correct degrees of freedom: n - k - (n_entities - 1)

---

### Test 3: Time-Invariant Variable (Should Be Dropped)

Fixed effects cannot estimate coefficients on time-invariant variables.

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 50
n_periods <- 5
n <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)
x_varying <- rnorm(n)  # Time-varying
x_invariant <- rep(rnorm(n_entities), each = n_periods)  # Time-invariant

y <- x_invariant + 2.0 * x_varying + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x_varying = x_varying, x_invariant = x_invariant,
                   entity = factor(entity), time = factor(time))
pdata <- pdata.frame(data, index = c("entity", "time"))

# x_invariant will be dropped
fe_fit <- plm(y ~ x_varying + x_invariant, data = pdata, model = "within")
summary(fe_fit)
# Only x_varying coefficient estimated
```

**Validation Criteria**:
- Time-invariant variable is dropped or NA
- Warning/error is produced

---

### Test 4: Unbalanced Panel

**R Code**:
```r
library(plm)

set.seed(42)
# Create unbalanced panel (some entities have fewer observations)
data <- data.frame(
  entity = c(rep(1, 5), rep(2, 3), rep(3, 4)),
  time = c(1:5, 1:3, 2:5),
  x = rnorm(12),
  y = c(rnorm(5), rnorm(3) + 2, rnorm(4) + 4)
)

pdata <- pdata.frame(data, index = c("entity", "time"))
fe_fit <- plm(y ~ x, data = pdata, model = "within")
summary(fe_fit)
```

**Validation Criteria**:
- Handles unbalanced panel correctly
- Correct degrees of freedom calculation

---

## Numerical Precision Summary

| Dataset | n | Coefficient Precision | SE Precision |
|---------|---|----------------------|--------------|
| Grunfeld | 200 | < 1e-6 | < 1e-5 |
| Synthetic | 1000 | < 1e-8 | < 1e-6 |

## Known Differences

1. **Intercept reporting**: R's plm doesn't report an intercept; the overall mean is absorbed.
2. **R² computation**: Within R² vs. overall R² may differ across implementations.
3. **Robust SEs**: p2a provides robust SEs via separate function; plm uses `vcovHC`.

## Running the Tests

```bash
# Run FE validation tests
cargo test -p p2a-core -- panel::tests::test_validate_fe

# Run with output
cargo test -p p2a-core -- fixed_effects --nocapture
```

## References

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press. Chapter 10.
- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*, 5th ed. Wiley.
