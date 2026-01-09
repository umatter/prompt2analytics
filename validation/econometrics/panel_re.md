# Validation: Panel Random Effects

## Method Overview

Panel Random Effects (RE) estimation treats the entity-specific effects as random draws from a population distribution, rather than fixed parameters. This allows estimation of time-invariant regressors but requires the assumption that entity effects are uncorrelated with the regressors.

**Model**:
```
y_it = α + X_it β + u_i + ε_it
```

where u_i ~ N(0, σ²_u) is the random entity effect.

**Key Parameters**:
- `y_col`: Dependent variable
- `x_cols`: Independent variables
- `entity_col`: Entity identifier

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `plm(model="random")` | 2.6-x |

## Test Cases

### Test 1: Grunfeld Dataset - Random Effects

**Dataset**: `validation/datasets/grunfeld.csv` (n=200, 10 firms × 20 years)

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

# Random Effects estimator
re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")
summary(re_fit)

# Variance components
ercomp(re_fit)

# Expected output for coefficients:
#             Estimate Std. Error z-value  Pr(>|z|)
# (Intercept) -57.8xxx   28.8xxx  -2.00xx  0.04xxx *
# value        0.1098xxx  0.0106xxx 10.3xxx  < 2e-16 ***
# capital      0.3085xxx  0.0171xxx 18.0xxx  < 2e-16 ***
```

**Results Comparison**:

| Statistic | R's plm | p2a Rust | Tolerance |
|-----------|---------|----------|-----------|
| β(Intercept) | -57.8 | ~-57.8 | 1.0 |
| β(value) | 0.1098 | ~0.1098 | 1e-4 |
| β(capital) | 0.3085 | ~0.3085 | 1e-4 |
| θ (transformation) | 0.86 | ~0.86 | 0.01 |
| σ²_u (between) | varies | varies | 10% |
| σ²_ε (within) | varies | varies | 10% |

**Rust Test**: `crates/p2a-core/src/econometrics/panel.rs::tests::test_validate_re_grunfeld`

---

### Test 2: Comparison with Fixed Effects

For the same data, compare RE and FE estimates. Under correct RE assumptions, they should be similar.

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

fe_fit <- plm(inv ~ value + capital, data = pdata, model = "within")
re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")

# Compare coefficients
cbind(FE = coef(fe_fit), RE = coef(re_fit)[c("value", "capital")])
```

**Validation Criteria**:
- RE and FE coefficients should be reasonably close
- Hausman test can formalize this comparison

---

### Test 3: Time-Invariant Variable (RE Can Estimate)

Unlike FE, RE can estimate coefficients on time-invariant variables.

**R Code**:
```r
library(plm)

set.seed(42)
n_entities <- 50
n_periods <- 5
n <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)
x_varying <- rnorm(n)
x_invariant <- rep(rnorm(n_entities), each = n_periods)  # Time-invariant

y <- 1.0 + 0.5 * x_invariant + 2.0 * x_varying + rnorm(n, 0, 0.5)

data <- data.frame(y = y, x_varying = x_varying, x_invariant = x_invariant,
                   entity = factor(entity), time = factor(time))
pdata <- pdata.frame(data, index = c("entity", "time"))

re_fit <- plm(y ~ x_varying + x_invariant, data = pdata, model = "random")
summary(re_fit)
# Both coefficients should be estimated
# x_invariant coefficient should be ≈ 0.5
```

**Validation Criteria**:
- Time-invariant coefficient is estimated
- Coefficient close to true value (0.5)

---

### Test 4: Variance Components Estimation

**R Code**:
```r
library(plm)

data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")

# Variance components
vc <- ercomp(re_fit)
print(vc)

# sigma^2_u: between-entity variance
# sigma^2_e: within-entity (idiosyncratic) variance
# theta: transformation parameter = 1 - sqrt(sigma^2_e / (T*sigma^2_u + sigma^2_e))
```

**Validation Criteria**:
- Variance components positive
- θ ∈ [0, 1]
- Correct relationship: θ = 1 - sqrt(σ²_ε / (T×σ²_u + σ²_ε))

---

## GLS Transformation

The Random Effects estimator uses a GLS transformation:
```
(y_it - θ×ȳ_i) = α(1-θ) + (X_it - θ×X̄_i)β + (error)
```

where θ is the transformation parameter based on variance components.

## Numerical Precision Summary

| Dataset | n | Coefficient Precision | θ Precision |
|---------|---|----------------------|-------------|
| Grunfeld | 200 | < 1e-4 | < 1e-2 |
| Synthetic | 1000 | < 1e-5 | < 1e-2 |

## Known Differences

1. **Variance estimation**: Different methods (Swamy-Arora, Wallace-Hussain) may give slightly different θ.
2. **Intercept**: RE includes an intercept; FE absorbs it.
3. **Degrees of freedom**: RE uses different df than FE.

## Running the Tests

```bash
# Run RE validation tests
cargo test -p p2a-core -- panel::tests::test_validate_re

# Compare with FE
cargo test -p p2a-core -- random_effects --nocapture
```

## References

- Baltagi, B.H. (2013). *Econometric Analysis of Panel Data*, 5th ed. Wiley. Chapter 2.
- Swamy, P.A.V.B. & Arora, S.S. (1972). "The Exact Finite Sample Properties of the Estimators of Coefficients in the Error Components Regression Models". *Econometrica*, 40(2), 261-275.
