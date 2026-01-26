# CausalImpact Validation

## Method Overview

CausalImpact is a Bayesian structural time series approach for estimating the causal effect of an intervention. The method was introduced by Google and uses state-space models to construct a counterfactual prediction of what would have happened in the absence of the intervention.

**Key Features:**
- Uses pre-intervention data to fit a structural time series model
- Predicts counterfactual for post-intervention period
- Provides Bayesian credible intervals for causal effects
- Supports control time series as covariates
- Returns cumulative, average, and relative effects

## Implementation Details

**File:** `crates/p2a-core/src/forecasting/causal_impact.rs`

**Algorithm:**
1. **Model Fitting (Pre-period only):**
   - Fit a local level model (random walk + noise)
   - Optionally include local linear trend
   - Optionally include seasonality
   - Optionally include regression on control series

2. **Counterfactual Prediction:**
   - Use Kalman filter to predict post-intervention observations
   - Generate prediction intervals from Kalman variance estimates

3. **Causal Effect Estimation:**
   - Point effect = observed - predicted
   - Cumulative effect = sum of point effects over post-period
   - Relative effect = cumulative effect / sum of predicted values
   - Bayesian p-value from normal approximation

## Reference Implementation

**R Package:** `CausalImpact` by Google
- Version: 1.3.0
- CRAN: https://cran.r-project.org/package=CausalImpact
- GitHub: https://google.github.io/CausalImpact/

## Test Cases

### Test Case 1: Positive Treatment Effect

**Setup:**
```r
# R code to generate reference results
library(CausalImpact)
set.seed(42)

# Generate data with known treatment effect
n_pre <- 70
n_post <- 30
n <- n_pre + n_post

# Local level random walk + treatment effect
y <- numeric(n)
level <- 100
for (t in 1:n) {
  level <- level + rnorm(1, 0, 1)
  effect <- if (t > n_pre) 10 else 0
  y[t] <- level + effect + rnorm(1, 0, 0.5)
}

# Run CausalImpact
data <- zoo(y, 1:n)
impact <- CausalImpact(data, c(1, n_pre), c(n_pre + 1, n))
summary(impact)
```

**Expected Results:**
- Cumulative effect: ~300 (30 * 10)
- Average effect: ~10
- Effect should be statistically significant

### Test Case 2: No Treatment Effect

**Setup:**
```r
set.seed(123)
n_pre <- 70
n_post <- 30

y <- numeric(n_pre + n_post)
level <- 100
for (t in 1:(n_pre + n_post)) {
  level <- level + rnorm(1, 0, 1)
  y[t] <- level + rnorm(1, 0, 0.5)
}

data <- zoo(y, 1:(n_pre + n_post))
impact <- CausalImpact(data, c(1, n_pre), c(n_pre + 1, n_pre + n_post))
summary(impact)
```

**Expected Results:**
- Cumulative effect: close to 0
- Effect should NOT be statistically significant
- p-value > 0.05

### Test Case 3: With Control Series

**Setup:**
```r
set.seed(456)
n_pre <- 70
n_post <- 30
n <- n_pre + n_post

# Control series (unaffected by treatment)
control <- 50 + cumsum(rnorm(n, 0, 1))

# Response driven by control + treatment effect
y <- 0.5 * control + rnorm(n, 0, 0.5)
y[(n_pre + 1):n] <- y[(n_pre + 1):n] + 8  # Treatment effect of 8

data <- cbind(y, control)
data <- zoo(data, 1:n)
impact <- CausalImpact(data, c(1, n_pre), c(n_pre + 1, n))
summary(impact)
```

**Expected Results:**
- Cumulative effect: ~240 (30 * 8)
- Effect should be statistically significant
- Control series should improve prediction accuracy

## Comparison with R CausalImpact

| Metric | R CausalImpact | Rust p2a-core | Tolerance |
|--------|----------------|---------------|-----------|
| Cumulative Effect | Baseline | Match within 20% | 0.20 |
| Average Effect | Baseline | Match within 20% | 0.20 |
| Effect Direction | Baseline | Must match | Exact |
| Significance | Baseline | Should match | - |

Note: Higher tolerance is acceptable because:
1. Different random number generators
2. Different optimization algorithms for MLE
3. Bayesian methods can have larger variance in point estimates

## Rust Test Functions

Located in `crates/p2a-core/src/forecasting/causal_impact.rs`:
- `test_causal_impact_positive_effect`
- `test_causal_impact_no_effect`
- `test_causal_impact_with_controls`
- `test_causal_impact_invalid_periods`
- `test_causal_impact_insufficient_pre_period`
- `test_causal_impact_with_trend`
- `test_causal_impact_model_info`
- `test_causal_impact_inference`

## References

1. Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015).
   "Inferring causal impact using Bayesian structural time series models".
   *Annals of Applied Statistics*, 9(1), 247-274.
   https://doi.org/10.1214/14-AOAS788

2. Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the
   Kalman Filter*. Cambridge University Press.

3. Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
   Methods* (2nd ed.). Oxford Statistical Science Series.

4. R package `CausalImpact`:
   - CRAN: https://cran.r-project.org/package=CausalImpact
   - Documentation: https://google.github.io/CausalImpact/

## Status

- [x] Core algorithm implemented
- [x] Local level model support
- [x] Local linear trend support
- [x] Control series support
- [x] Unit tests written
- [x] MCP tool added
- [ ] Seasonal model validation (basic support implemented)
- [ ] Cross-validation with R package on real datasets
