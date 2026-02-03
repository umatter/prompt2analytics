# LTMLE (Longitudinal Targeted Maximum Likelihood Estimation) Validation

## Method Overview

LTMLE extends standard TMLE to longitudinal settings with time-varying treatments and
confounders. It estimates causal effects under dynamic treatment regimes using sequential
regression combined with targeting steps for double robustness.

### Key Features

- Sequential g-computation iterating backward through time points
- Targeting step at each time point using cumulative IPW weights (clever covariate)
- Double robustness: consistent if either outcome models OR propensity score models are correct
- Influence curve-based standard errors for valid inference

### Mathematical Framework

For T time points with:
- L_t: time-varying covariates at time t
- A_t: treatment at time t (binary: 0 or 1)
- Y_T: final outcome

Target parameter: E[Y^{a_1,...,a_T}] under intervention regime (e.g., always treat: a_t = 1 for all t)

**Algorithm:**
1. Estimate propensity scores: g_t(L_t) = P(A_t = 1 | L_t) for each t
2. Backward iteration from t = T to t = 1:
   - Fit outcome model Q_t(A_t, L_t) using pseudo-outcome from t+1
   - Predict under counterfactual treatment: Q_t^d where A_t = d
   - Target: Q_t^* = Q_t + epsilon * H_t where H_t = cumulative 1/g weights
3. Final estimate: psi = mean(Q_1^*)

## Reference Implementation

**R Package:** `ltmle` (Schwab, Lendle, Petersen, van der Laan)
- CRAN: https://cran.r-project.org/package=ltmle
- GitHub: https://github.com/joshuaschwab/ltmle

**R Version:** 4.3.x
**Package Version:** 1.3-0

## Test Cases

### Test Case 1: Two Time Points with Linear Outcome

**Data Generating Process:**
```
L_1 ~ Uniform(0, 1)
A_1 | L_1 ~ Bernoulli(expit(0.5 * L_1))
L_2 | A_1, L_1 = L_1 + 0.3 * A_1 + noise
A_2 | L_2 ~ Bernoulli(expit(0.5 * L_2))
Y | A_1, A_2, L_1, L_2 = 0.2*L_1 + 0.3*L_2 + 0.4*A_1 + 0.5*A_2 + noise
```

True ATE (always treat vs never treat): approximately 0.4 + 0.5 = 0.9

**Rust Implementation:**
```rust
use p2a_core::econometrics::{run_ltmle, LtmleConfig, LtmleData, LtmleQModel};

let data = LtmleData::new(outcomes, treatments, covariates)?;
let config = LtmleConfig {
    q_model: LtmleQModel::Linear,
    gbounds: (0.01, 0.99),
    ..Default::default()
};
let result = run_ltmle(&data, config)?;
```

**R Reproduction:**
```r
library(ltmle)

# Generate data
set.seed(123)
n <- 1000

# Time 1
L1 <- runif(n)
g1 <- plogis(0.5 * L1)
A1 <- rbinom(n, 1, g1)

# Time 2
L2 <- L1 + 0.3 * A1 + rnorm(n, 0, 0.1)
g2 <- plogis(0.5 * L2)
A2 <- rbinom(n, 1, g2)

# Outcome
Y <- 0.2*L1 + 0.3*L2 + 0.4*A1 + 0.5*A2 + rnorm(n, 0, 0.1)

# Create data frame
data <- data.frame(L1 = L1, A1 = A1, L2 = L2, A2 = A2, Y = Y)

# Run LTMLE
result <- ltmle(
  data = data,
  Anodes = c("A1", "A2"),
  Lnodes = c("L1", "L2"),
  Ynodes = "Y",
  abar = list(treatment = c(1, 1), control = c(0, 0)),
  SL.library = "glm"
)

summary(result)
```

### Results Comparison

| Metric | Rust Implementation | R ltmle | Tolerance |
|--------|---------------------|---------|-----------|
| ATE | (varies with data) | ~ 0.9 | 0.2 (due to noise) |
| SE | > 0 | > 0 | qualitative |
| CI includes true value | Yes | Yes | qualitative |

## Validation Tests

### Test: Data Validation
- Validates number of time points >= 2
- Validates treatment is binary (0 or 1)
- Validates consistent sample sizes across time points

### Test: Basic Functionality
- ATE estimate is finite and in reasonable range
- Standard error is positive and finite
- Confidence intervals are finite
- Counterfactual means satisfy E[Y^1] > E[Y^0] when treatment effect is positive

### Test: Propensity Score Truncation
- All propensity scores are within specified bounds
- Truncation counts are recorded

### Test: Influence Curve
- Influence curve has approximately zero mean
- Standard errors computed from influence curve variance

## Limitations

1. **Static interventions only:** Current implementation supports only "always treat" vs "never treat" regimes. Dynamic treatment regimes (where A_t depends on history) are not yet implemented.

2. **No censoring:** The implementation does not handle censoring nodes (C_t). All observations must be observed at all time points.

3. **Logistic outcome model:** For binary outcomes, the logistic model targeting may have numerical issues with extreme probabilities.

4. **Single outcome:** Only final outcome Y_T is supported, not intermediate outcomes at each time point.

## References

- van der Laan, M.J. & Gruber, S. (2012). "Targeted Minimum Loss Based Estimation of
  Causal Effects of Multiple Time Point Interventions." *The International Journal
  of Biostatistics*, 8(1), Article 9. https://doi.org/10.1515/1557-4679.1370

- Lendle, S.D., Schwab, J., Petersen, M.L., & van der Laan, M.J. (2017). "ltmle:
  An R Package Implementing Targeted Minimum Loss-Based Estimation for Longitudinal
  Data." *Journal of Statistical Software*, 81(1), 1-21. https://doi.org/10.18637/jss.v081.i01

- van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference for
  Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and
  Causal Inference Models." *Biometrics*, 61(4), 962-973.

## Rust Test Functions

- `test_ltmle_data_validation` - Tests data structure validation
- `test_ltmle_data_validation_errors` - Tests error handling for invalid data
- `test_ltmle_basic` - Tests basic LTMLE functionality with linear outcome
- `test_ltmle_with_linear_outcome` - Tests linear outcome model explicitly
- `test_ltmle_propensity_truncation` - Tests propensity score truncation
- `test_ltmle_influence_curve` - Tests influence curve computation
- `test_ltmle_fluctuation_coefs` - Tests targeting step coefficients
- `test_ltmle_display` - Tests result formatting
- `test_intervention_type_display` - Tests intervention type formatting
- `test_add_intercept` - Tests design matrix construction
- `test_logit` - Tests logit function
- `test_ltmle_single_time_point_error` - Tests error for insufficient time points

## Status

**Implementation Status:** Complete (core algorithm with static interventions)

**Validation Status:** Partial (tests pass, R comparison pending full run)

**Last Updated:** 2026-01-25
