# Validation: Fixed Effects Generalized Linear Model (FEGLM)

## Method Overview

Fixed Effects Generalized Linear Model (FEGLM) combines GLM estimation with high-dimensional fixed effects absorption. It uses Iteratively Reweighted Least Squares (IRLS) with weighted Method of Alternating Projections (MAP) for efficient estimation without creating dummy variables.

**Key Parameters**:
- `y_col`: Outcome variable
- `x_cols`: Predictor variables
- `fe_cols`: Fixed effect variables to absorb
- `family`: GLM family (Logit, Probit, Poisson, Gaussian)
- `config`: Optional configuration (IRLS tolerance, MAP settings)

**Supported Families**:
- **Logit**: Binary outcomes with logistic link
- **Probit**: Binary outcomes with probit link
- **Poisson**: Count data with log link
- **Gaussian**: Continuous outcomes with identity link (reduces to HDFE)

**Use Cases**:
- Binary outcomes with firm + year fixed effects
- Count data with high-dimensional FE
- Multinomial/conditional logit approximations
- Trade data with exporter + importer FE

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| alpaca | R | `feglm()` | 0.3.4 |
| fixest | R | `feglm()`/`fepois()` | 0.11.x |
| ppmlhdfe | Stata | `ppmlhdfe` | 2.x |

**Primary Reference**:
> Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects". arXiv:1707.01815.
> https://arxiv.org/abs/1707.01815

**R Package Reference**:
> Czarnowske, D. & Stammann, A. "alpaca: Fit GLM's with High-Dimensional k-Way Fixed Effects". CRAN.
> https://cran.r-project.org/package=alpaca

## Algorithm

The FEGLM algorithm iterates between:

1. **IRLS Update**: Compute working weights and working response
   ```
   w_i = (dmu/deta)^2 / V(mu_i)
   z_i = eta_i + (y_i - mu_i) * (deta/dmu)
   ```

2. **Weighted MAP**: Demean z and X using weighted alternating projections
   ```
   z_tilde = weighted_demean(z, FE, w)
   X_tilde = weighted_demean(X, FE, w)
   ```

3. **WLS Step**: Solve weighted least squares
   ```
   beta_new = (X_tilde' W X_tilde)^-1 X_tilde' W z_tilde
   ```

4. **Convergence**: Check coefficient change < tolerance

## Test Cases

### Test 1: Logit with Single Fixed Effect

**Data Generating Process**:
```
P(Y=1|X,FE) = Λ(1.0*x + α_id)
```

Where Λ(·) is the logistic CDF.

**R Code**:
```r
library(alpaca)

# Panel: 3 entities, 4 time periods
df <- data.frame(
  y = c(1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0),
  x = c(1.5, 2.0, 1.8, 0.3, 0.5, 1.5, 0.2, 0.4, 1.6, 1.9, 1.7, 0.4),
  id = factor(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3))
)

est <- feglm(y ~ x | id, data = df, family = binomial())
summary(est)

# Coefficient should recover true beta ≈ 1.0
coef(est)
```

**Results Comparison**:

| Parameter | True Value | R's feglm | p2a Rust | Tolerance |
|-----------|------------|-----------|----------|-----------|
| β(x) | 1.0 | ~1.0 | ~1.0 | 0.3 |

**Rust Test**: `crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_logit_basic`

---

### Test 2: Probit with Two-Way Fixed Effects

**Data Generating Process**:
```
P(Y=1|X,FE) = Φ(0.5*x + α_id + γ_time)
```

Where Φ(·) is the standard normal CDF.

**R Code**:
```r
library(alpaca)

# Simulated panel
set.seed(42)
n <- 100
id <- factor(rep(1:10, each = 10))
time <- factor(rep(1:10, times = 10))
x <- rnorm(n)
id_eff <- rnorm(10)[as.numeric(id)]
time_eff <- rnorm(10)[as.numeric(time)]
latent <- 0.5 * x + id_eff + time_eff
y <- as.integer(pnorm(latent) > runif(n))

df <- data.frame(y = y, x = x, id = id, time = time)

est <- feglm(y ~ x | id + time, data = df, family = binomial("probit"))
summary(est)
```

**Validation Criteria**:
- Coefficient within 0.5 of true value (0.5)
- Correct sign
- Converges within max iterations

**Rust Test**: `crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_probit_twoway`

---

### Test 3: Poisson with Fixed Effects (Trade Model)

**Data Generating Process**:
```
E[Y|X,FE] = exp(1.0*x + α_exporter + γ_importer)
```

This is the canonical gravity model specification.

**R Code**:
```r
library(alpaca)

# Simulated trade data
df <- data.frame(
  y = c(5, 10, 3, 8, 12, 4, 6, 15, 2, 9, 11, 5),
  x = c(0.5, 1.0, 0.2, 0.8, 1.2, 0.3, 0.6, 1.5, 0.1, 0.9, 1.1, 0.5),
  exporter = factor(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
  importer = factor(c(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4))
)

est <- feglm(y ~ x | exporter + importer, data = df, family = poisson())
summary(est)

# Check log-likelihood and coefficient
logLik(est)
coef(est)
```

**Results Comparison**:

| Statistic | R's feglm | p2a Rust | Tolerance |
|-----------|-----------|----------|-----------|
| β(x) | varies | varies | 0.3 |
| Converged | TRUE | TRUE | - |
| Iterations | varies | varies | - |

**Rust Test**: `crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_poisson_basic`

---

### Test 4: Gaussian with HDFE (Equivalence Test)

**Data Generating Process**:
```
y = 2.0*x1 + 1.0*x2 + α_id + γ_firm + ε
```

Gaussian FEGLM should produce identical results to linear HDFE.

**R Code**:
```r
library(alpaca)
library(lfe)

set.seed(42)
n <- 200
id <- factor(rep(1:10, 20))
firm <- factor(rep(1:5, 40))
x1 <- rnorm(n)
x2 <- rnorm(n)
id_eff <- rnorm(10)[as.numeric(id)]
firm_eff <- rnorm(5)[as.numeric(firm)]
y <- 2.0 * x1 + 1.0 * x2 + id_eff + firm_eff + rnorm(n, sd = 0.5)

df <- data.frame(y = y, x1 = x1, x2 = x2, id = id, firm = firm)

# FEGLM with Gaussian
est_feglm <- feglm(y ~ x1 + x2 | id + firm, data = df, family = gaussian())

# Linear HDFE for comparison
est_felm <- felm(y ~ x1 + x2 | id + firm, data = df)

# Coefficients should match
coef(est_feglm)
coef(est_felm)
```

**Validation Criteria**:
- FEGLM Gaussian coefficients match HDFE within 1e-6
- Standard errors match within 1e-5
- Converges in 1 IRLS iteration (linear case)

**Rust Test**: `crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_gaussian_matches_hdfe`

---

### Test 5: Perfect Separation Detection

**R Code**:
```r
library(alpaca)

# Perfect separation: x perfectly predicts y within each group
df <- data.frame(
  y = c(0, 0, 1, 1, 0, 0, 1, 1),
  x = c(-1, -1, 1, 1, -1, -1, 1, 1),
  id = factor(c(1, 1, 1, 1, 2, 2, 2, 2))
)

# Should warn about separation or produce very large coefficients
est <- feglm(y ~ x | id, data = df, family = binomial())
```

**Validation Criteria**:
- Warning about separation or non-convergence
- Very large coefficient estimates (diverging)

---

### Test 6: Coefficient Recovery Test

**Data Generating Process**:
```
P(Y=1|X) = Λ(2.0*x1 - 1.0*x2)
```

Generate data from known DGP and verify coefficient recovery.

**R Code**:
```r
library(alpaca)

set.seed(123)
n <- 1000
id <- factor(rep(1:50, 20))
x1 <- rnorm(n)
x2 <- rnorm(n)
id_eff <- rnorm(50)[as.numeric(id)]
latent <- 2.0 * x1 - 1.0 * x2 + id_eff
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

df <- data.frame(y = y, x1 = x1, x2 = x2, id = id)

est <- feglm(y ~ x1 + x2 | id, data = df, family = binomial())
coef(est)

# Should recover:
# beta_x1 ≈ 2.0
# beta_x2 ≈ -1.0
```

**Results Comparison**:

| Parameter | True Value | R's feglm | p2a Rust | Tolerance |
|-----------|------------|-----------|----------|-----------|
| β(x1) | 2.0 | ~2.0 | ~2.0 | 0.3 |
| β(x2) | -1.0 | ~-1.0 | ~-1.0 | 0.3 |

**Rust Test**: `crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_logit_coefficient_recovery`

---

## Link Functions

| Family | Link | η = g(μ) | μ = g⁻¹(η) | V(μ) |
|--------|------|----------|------------|------|
| Logit | logit | log(μ/(1-μ)) | exp(η)/(1+exp(η)) | μ(1-μ) |
| Probit | probit | Φ⁻¹(μ) | Φ(η) | φ(η)²/Φ(η)(1-Φ(η)) |
| Poisson | log | log(μ) | exp(η) | μ |
| Gaussian | identity | μ | η | 1 |

## Working Weights and Response

For IRLS, we compute:

**Working weights**: `w = (∂μ/∂η)² / V(μ)`
**Working response**: `z = η + (y - μ) × (∂η/∂μ)`

| Family | w | z |
|--------|---|---|
| Logit | μ(1-μ) | η + (y-μ)/(μ(1-μ)) |
| Probit | φ(η)²/[Φ(η)(1-Φ(η))] | η + (y-μ)×(1-Φ)Φ/φ |
| Poisson | μ | η + (y-μ)/μ |
| Gaussian | 1 | y |

## Numerical Precision Summary

| Family | n | Coefficient Precision | LL Precision |
|--------|---|----------------------|--------------|
| Logit | 100+ | < 0.1 | < 0.01 |
| Probit | 100+ | < 0.1 | < 0.01 |
| Poisson | 100+ | < 0.1 | < 0.01 |
| Gaussian | 100+ | < 1e-5 | N/A |

## Known Differences

1. **Acceleration**: p2a uses optional Gearhart-Koshy acceleration in MAP; alpaca may use different acceleration.

2. **Standard Errors**: p2a currently computes standard SEs from information matrix; alpaca supports additional SE estimators.

3. **Separated Data**: Different handling of quasi/complete separation in binary models.

4. **Convergence Criteria**: IRLS convergence based on max coefficient change; alpaca may use different criterion.

5. **Degrees of Freedom**: Both use: `df = n - k - sum(FE levels) + absorbed`

## Running the Tests

```bash
# Run all FEGLM validation tests
cargo test -p p2a-core -- feglm::tests

# Run specific family tests
cargo test -p p2a-core -- feglm::tests::test_feglm_logit
cargo test -p p2a-core -- feglm::tests::test_feglm_probit
cargo test -p p2a-core -- feglm::tests::test_feglm_poisson

# Run with output to see computed values
cargo test -p p2a-core -- feglm::tests --nocapture
```

## References

- Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects". arXiv:1707.01815.
- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity". *Review of Economics and Statistics*, 88(4), 641-658.
- Fernández-Val, I. & Weidner, M. (2016). "Individual and Time Effects in Nonlinear Panel Models with Large N, T". *Journal of Econometrics*, 192(1), 291-312.
- Czarnowske, D. & Stammann, A. (2020). "alpaca: Fit GLM's with High-Dimensional k-Way Fixed Effects". *R package*.
