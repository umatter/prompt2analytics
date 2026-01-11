# Doubly Robust (AIPW) Treatment Effect Validation

## Method Overview

Augmented Inverse Probability Weighting (AIPW) combines propensity score weighting with outcome regression for doubly robust treatment effect estimation.

**p2a Function**: `run_doubly_robust()`

**Methods**:
- AIPW (Augmented IPW - doubly robust, default)
- IPW (IPW component only)
- Regression (Outcome regression only)

**Estimands**:
- ATE (Average Treatment Effect)
- ATT (Average Treatment Effect on Treated)

## Reference Implementations

| Package | Function | Language | Notes |
|---------|----------|----------|-------|
| causalweight | `treatweight()` | R | With AIPW option |
| AIPW | `AIPW()` | R | Dedicated AIPW package |
| DoubleML | `DoubleMLPLR()` | Python | Cross-fitted |
| econml | `DRLearner` | Python | |

## Test Case 1: Synthetic Data with Known DGP

### Data Generating Process

```
X1, X2 ~ Uniform(-1, 1)
D ~ Bernoulli(logit(0.5 + 0.3*X1 + 0.2*X2))
Y(0) = 1.0*X1 + 0.5*X2 + epsilon
Y(1) = Y(0) + 2.0  (constant treatment effect)
Y = D*Y(1) + (1-D)*Y(0)

True ATE = 2.0
```

### R Code (causalweight)

```r
library(causalweight)

set.seed(42)
n <- 1000

x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)

# Propensity score
ps <- plogis(0.5 + 0.3*x1 + 0.2*x2)
d <- rbinom(n, 1, ps)

# Potential outcomes
y0 <- 1.0*x1 + 0.5*x2 + rnorm(n, 0, 0.5)
y1 <- y0 + 2.0  # True ATE = 2.0
y <- d*y1 + (1-d)*y0

data <- data.frame(y = y, d = d, x1 = x1, x2 = x2)

# AIPW estimation
result <- treatweight(
  y = data$y,
  d = data$d,
  x = data[, c("x1", "x2")],
  boot = 499,
  trim = 0.05,
  ATET = FALSE  # ATE
)

cat("Method: AIPW (doubly robust)\n")
cat("ATE:", result$effect, "\n")
cat("SE:", result$se, "\n")
```

### R Code (AIPW package)

```r
library(AIPW)

# Using AIPW package for comparison
aipw <- AIPW$new(
  Y = data$y,
  A = data$d,
  W = data[, c("x1", "x2")],
  Q.SL.library = "SL.glm",
  g.SL.library = "SL.glm"
)
aipw$stratified_fit()$summary()
```

### Expected Results

| Statistic | R (causalweight) | R (AIPW) | p2a (Rust) | Tolerance |
|-----------|------------------|----------|------------|-----------|
| ATE | 2.0 ± 0.15 | 2.0 ± 0.15 | 2.0 ± 0.15 | 0.1 |
| SE | 0.05-0.08 | 0.05-0.08 | 0.05-0.08 | 0.02 |

## Test Case 2: Misspecified Propensity Model

AIPW should be robust when either the propensity or outcome model is correct.

### DGP with Nonlinear Propensity

```
True PS: logit(0.5 + 0.5*X1^2 + 0.3*X2)  (nonlinear in X1)
Fitted PS: logit(a + b*X1 + c*X2)  (linear, misspecified)
Outcome model: correctly specified as linear
```

### R Code

```r
set.seed(42)
n <- 1000

x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)

# TRUE nonlinear propensity (model is misspecified)
ps_true <- plogis(0.5 + 0.5*x1^2 + 0.3*x2)
d <- rbinom(n, 1, ps_true)

# Linear outcome (model is correct)
y0 <- 1.0*x1 + 0.5*x2 + rnorm(n, 0, 0.5)
y1 <- y0 + 2.0
y <- d*y1 + (1-d)*y0

# AIPW should still work due to correct outcome model
result_aipw <- treatweight(
  y = y, d = d,
  x = cbind(x1, x2),
  boot = 499
)

# IPW alone will be biased
result_ipw <- treatweight(
  y = y, d = d,
  x = cbind(x1, x2),
  boot = 499
  # Note: causalweight always uses AIPW internally
)
```

## Test Case 3: Different Methods Comparison

Compare IPW-only, Regression-only, and AIPW on the same data.

### R Code

```r
set.seed(42)
n <- 1000

# Well-specified case
x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)
ps <- plogis(0.5 + 0.3*x1 + 0.2*x2)
d <- rbinom(n, 1, ps)
y <- 2.0*d + 1.0*x1 + 0.5*x2 + rnorm(n, 0, 0.5)

# All three methods should give similar results
# (causalweight uses AIPW by default)

# For pure IPW comparison, use manual calculation:
ps_hat <- predict(glm(d ~ x1 + x2, family = binomial), type = "response")
w1 <- d / ps_hat
w0 <- (1-d) / (1 - ps_hat)
ate_ipw <- sum(w1 * y) / sum(w1) - sum(w0 * y) / sum(w0)

# For pure regression:
lm_treated <- lm(y ~ x1 + x2, subset = d == 1)
lm_control <- lm(y ~ x1 + x2, subset = d == 0)
mu1 <- predict(lm_treated, newdata = data.frame(x1 = x1, x2 = x2))
mu0 <- predict(lm_control, newdata = data.frame(x1 = x1, x2 = x2))
ate_reg <- mean(mu1 - mu0)
```

### Expected Results

| Method | Estimate | Notes |
|--------|----------|-------|
| IPW | 2.0 ± 0.2 | Higher variance |
| Regression | 2.0 ± 0.1 | Lower variance, not doubly robust |
| AIPW | 2.0 ± 0.1 | Best of both, doubly robust |

## Numerical Precision

| Sample Size | Effect Tolerance | SE Tolerance |
|-------------|------------------|--------------|
| n < 500 | 0.15 | 0.03 |
| n = 500-2000 | 0.08 | 0.015 |
| n > 2000 | 0.04 | 0.008 |

## Known Differences

1. **Outcome model**: p2a uses OLS for outcome regression; some R packages use machine learning
2. **Bootstrap**: Implementation details may differ slightly
3. **Cross-fitting**: p2a does not use cross-fitting; DoubleML does

## References

- Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed."
- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models."
- Glynn, A.N. & Quinn, K.M. (2010). "An Introduction to the Augmented Inverse Propensity Weighted Estimator."
