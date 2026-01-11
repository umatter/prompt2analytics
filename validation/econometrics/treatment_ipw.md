# IPW Treatment Effect Validation

## Method Overview

Inverse Probability Weighting (IPW) for treatment effect estimation using propensity scores.

**p2a Function**: `run_ipw_treatment()`

**Estimands**:
- ATE (Average Treatment Effect)
- ATT (Average Treatment Effect on Treated)

## Reference Implementations

| Package | Function | Language | Notes |
|---------|----------|----------|-------|
| causalweight | `treatweight()` | R | Primary reference |
| WeightIt | `weightit()` | R | Alternative |
| statsmodels | `PropensityScoreIPW` | Python | |

## Test Case 1: Lalonde Experimental Data

The LaLonde (1986) job training dataset is a standard benchmark for treatment effect methods.

### R Code (causalweight)

```r
library(causalweight)

# Load LaLonde data (or create synthetic version)
# Synthetic version with similar structure:
set.seed(42)
n <- 500

# Covariates
age <- runif(n, 18, 55)
education <- sample(8:16, n, replace = TRUE)
married <- rbinom(n, 1, 0.4)

# Treatment assignment (selection on observables)
ps_true <- plogis(-2 + 0.05*age + 0.1*education + 0.5*married)
treatment <- rbinom(n, 1, ps_true)

# Outcome (with treatment effect of 1500)
earnings <- 5000 + 100*age + 500*education + 2000*married + 1500*treatment + rnorm(n, 0, 2000)

data <- data.frame(
  earnings = earnings,
  treatment = treatment,
  age = age,
  education = education,
  married = married
)

# IPW estimation
result <- treatweight(
  y = data$earnings,
  d = data$treatment,
  x = data[, c("age", "education", "married")],
  boot = 999,
  trim = 0.05
)

cat("ATE:", result$effect, "\n")
cat("SE:", result$se, "\n")
cat("95% CI:", result$effect - 1.96*result$se, ",", result$effect + 1.96*result$se, "\n")
```

### Expected Results

| Statistic | R (causalweight) | p2a (Rust) | Tolerance |
|-----------|------------------|------------|-----------|
| ATE | ~1500 | ~1500 | 200 |
| SE | ~150-250 | ~150-250 | 50 |
| n_trimmed | Varies | Varies | - |

### Rust Test

```rust
#[test]
fn test_validate_ipw_ate_lalonde() {
    // See crates/p2a-core/src/econometrics/treatment.rs
    // test_ipw_ate_basic
}
```

## Test Case 2: Synthetic Data with Known DGP

### Data Generating Process

```
X1 ~ Uniform(-1, 1)
X2 ~ Uniform(-1, 1)
D ~ Bernoulli(logit(0.5 + 0.3*X1 + 0.2*X2))
Y = 2.0*D + 1.0*X1 + 0.5*X2 + epsilon, epsilon ~ N(0, 1)

True ATE = 2.0
```

### R Code

```r
library(causalweight)

set.seed(42)
n <- 1000

x1 <- runif(n, -1, 1)
x2 <- runif(n, -1, 1)

# Propensity score model
ps <- plogis(0.5 + 0.3*x1 + 0.2*x2)
d <- rbinom(n, 1, ps)

# Outcome with true ATE = 2.0
y <- 2.0*d + 1.0*x1 + 0.5*x2 + rnorm(n, 0, 1)

data <- data.frame(y = y, d = d, x1 = x1, x2 = x2)

result <- treatweight(
  y = data$y,
  d = data$d,
  x = data[, c("x1", "x2")],
  boot = 499,
  trim = 0.05
)

print(result)
```

### Expected Results

| Statistic | R (causalweight) | p2a (Rust) | Tolerance |
|-----------|------------------|------------|-----------|
| ATE | 2.0 ± 0.2 | 2.0 ± 0.2 | 0.1 |
| SE | 0.08-0.12 | 0.08-0.12 | 0.02 |

## Test Case 3: ATT Estimation

### R Code

```r
# Same data as Test Case 2
result_att <- treatweight(
  y = data$y,
  d = data$d,
  x = data[, c("x1", "x2")],
  boot = 499,
  trim = 0.05,
  estimand = "ATT"
)
```

## Numerical Precision

| Sample Size | Effect Tolerance | SE Tolerance |
|-------------|------------------|--------------|
| n < 500 | 0.2 | 0.05 |
| n = 500-2000 | 0.1 | 0.02 |
| n > 2000 | 0.05 | 0.01 |

Bootstrap standard errors will have inherent randomness (±5-10% variation).

## Known Differences

1. **Bootstrap implementation**: causalweight uses stratified bootstrap; p2a uses simple bootstrap
2. **Propensity score trimming**: Both use symmetric trimming but may handle boundary cases differently
3. **Normalization**: Both use Hajek (normalized) estimator

## References

- Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe."
- Bodory, H. & Huber, M. (2018). "causalweight: Estimation Methods for Causal Inference Based on Inverse Probability Weighting."
- LaLonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training Programs."
