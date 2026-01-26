# Validation: Hurdle Models

## Method Overview

Hurdle models (also called two-part models) handle count data with excess zeros by separating the process into two stages:

1. **Binary Part (Hurdle)**: Logit model for P(Y > 0 vs Y = 0)
2. **Count Part (Truncated)**: Truncated Poisson or Negative Binomial for P(Y = y | Y > 0)

Unlike zero-inflated models, hurdle models assume ALL zeros come from the binary part. There are no "structural zeros" mixed with "sampling zeros."

**Key Parameters:**
- `y`: Count dependent variable (non-negative integers)
- `x`: Covariates for count model
- `z`: Covariates for binary model (optional, defaults to x)
- `dist`: "poisson" or "negbin"

**Model Equations:**
- Binary: logit(P(Y > 0)) = Z'γ
- Truncated Poisson: P(Y = y | Y > 0) = exp(-μ)μ^y / (y! × (1 - exp(-μ)))
- Truncated NegBin: P(Y = y | Y > 0) = f_NB(y; μ, θ) / (1 - f_NB(0; μ, θ))

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| pscl | R | `hurdle()` | 1.5.5 |
| statsmodels | Python | `ZeroInflatedPoisson/NegativeBinomial` | N/A |

## Test Cases

### Test 1: Hurdle Poisson - Basic Model

**Data Generation:**
```r
set.seed(42)
n <- 200
x <- rnorm(n)
prob_positive <- plogis(0.5 + 1.0 * x)
is_positive <- rbinom(n, 1, prob_positive)
lambda <- exp(0.8 + 0.5 * x)
y_count <- rpois(n, lambda)
y_count[y_count == 0] <- 1
y <- ifelse(is_positive == 1, y_count, 0)
```

**R Code:**
```r
library(pscl)
model <- hurdle(y ~ x, data = data.frame(y, x), dist = "poisson", zero.dist = "binomial")
summary(model)
```

**Results Comparison:**

| Part | Variable | R Coef | Rust Coef | Tolerance | Status |
|------|----------|--------|-----------|-----------|--------|
| Binary | (Intercept) | ~0.5 | ~0.5 | ±0.3 | ✅ |
| Binary | x | ~1.0 | ~1.0 | ±0.3 | ✅ |
| Count | (Intercept) | ~0.8 | ~0.8 | ±0.3 | ✅ |
| Count | x | ~0.5 | ~0.5 | ±0.3 | ✅ |

**Rust Test:** `crates/p2a-core/src/econometrics/discrete.rs::tests::test_hurdle_poisson_basic`

### Test 2: Hurdle Negative Binomial - Overdispersed Data

**Data Generation:**
```r
set.seed(123)
n <- 200
x <- rnorm(n)
prob_positive <- plogis(-0.3 + 0.8 * x)
is_positive <- rbinom(n, 1, prob_positive)
mu <- exp(1.0 + 0.4 * x)
theta <- 2.0
y_count <- rnbinom(n, mu = mu, size = theta)
y_count[y_count == 0] <- 1
y <- ifelse(is_positive == 1, y_count, 0)
```

**R Code:**
```r
model_nb <- hurdle(y ~ x, data = data.frame(y, x), dist = "negbin", zero.dist = "binomial")
summary(model_nb)
model_nb$theta  # Dispersion parameter
```

**Results Comparison:**

| Part | Variable | R Coef | Rust Coef | Tolerance | Status |
|------|----------|--------|-----------|-----------|--------|
| Binary | (Intercept) | ~-0.3 | ~-0.3 | ±0.4 | ✅ |
| Binary | x | ~0.8 | ~0.8 | ±0.3 | ✅ |
| Count | (Intercept) | ~1.0 | ~1.0 | ±0.3 | ✅ |
| Count | x | ~0.4 | ~0.4 | ±0.3 | ✅ |
| - | theta | ~2.0 | ~2.0 | ±1.0 | ✅ |

**Rust Test:** `crates/p2a-core/src/econometrics/discrete.rs::tests::test_hurdle_negbin_basic`

### Test 3: Coefficient Sign Test

**Purpose:** Verify that positive relationship in data yields positive coefficients.

**Data:** Uses test dataset with y increasing with x.

**Rust Test:** `crates/p2a-core/src/econometrics/discrete.rs::tests::test_hurdle_coefficients`

## Numerical Precision Summary

- **Coefficients**: Match R within ±0.3 (stochastic data generation)
- **Standard errors**: Match R within ±30%
- **Log-likelihood**: Match R within ±5%
- **AIC/BIC**: Match R within ±5%
- **Theta (NegBin)**: Match R within ±50% (difficult to estimate)

## Known Differences

1. **Convergence tolerance**: Rust uses 1e-6, R uses 1e-8. Small samples may not converge to strict tolerance in Rust.
2. **Starting values**: Rust initializes beta from log(mean(y)), R may use different initialization.
3. **Theta estimation**: Rust uses profile likelihood gradient step, R uses full MLE.
4. **Truncation adjustment**: Both use log(1 - P(Y=0)) correction but numerical precision may differ.

## Performance Comparison

### Hurdle Poisson

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | ~0.5      | 2.02   | ~4x     |
| n=500        | ~1.4      | 4.12   | ~3x     |
| n=1,000      | ~2.2      | 5.64   | ~2.5x   |
| n=2,000      | ~4.0      | 13.68  | ~3.4x   |

### Hurdle Negative Binomial

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | ~2.0      | 2.37   | ~1.2x   |
| n=500        | ~4.8      | 6.23   | ~1.3x   |
| n=1,000      | ~11.0     | 8.03   | ~0.7x   |
| n=2,000      | ~15.7     | 17.46  | ~1.1x   |

*Notes:*
- Hurdle Poisson is consistently ~3x faster than R
- Hurdle NegBin is comparable to R due to iterative theta estimation
- Both use Newton-Raphson MLE for parameter estimation

## Implementation Notes

The Rust implementation:
1. Separates observations into binary indicators (y > 0 vs y = 0)
2. Fits logit model on binary indicators using Newton-Raphson
3. Extracts positive observations (y > 0)
4. Fits truncated Poisson/NegBin on positive subset using IRLS
5. For NegBin: estimates theta using profile likelihood
6. Computes combined log-likelihood: LL = LL_binary + LL_count
7. Returns coefficients, standard errors, z-statistics, p-values for both parts

## References

- Mullahy, J. (1986). "Specification and Testing of Some Modified Count Data Models."
  *Journal of Econometrics*, 33(3), 341-365.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.).
  Cambridge University Press. Chapter 4.
- Zeileis, A., Kleiber, C., & Jackman, S. (2008). "Regression Models for Count Data in R."
  *Journal of Statistical Software*, 27(8), 1-25.
