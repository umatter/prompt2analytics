# Validation: Regression Standardization (G-computation) - stdReg

## Method Overview

Regression standardization (also called G-computation or the parametric g-formula for single time point) estimates causal effects by fitting an outcome model and averaging predictions under different treatment values over the covariate distribution.

## Mathematical Framework

Under the assumptions of consistency, no unmeasured confounding (conditional exchangeability), and positivity:

```
E[Y(a)] = sum_x E[Y|A=a, X=x] * P(X=x)
        = (1/n) * sum_i E[Y|A=a, X_i]
```

For ATE (Average Treatment Effect):
```
ATE = E[Y(1)] - E[Y(0)]
    = (1/n) * sum_i [Y_hat_i(1) - Y_hat_i(0)]
```

For ATT (Average Treatment Effect on Treated):
```
ATT = E[Y(1) - Y(0) | A=1]
    = (1/n1) * sum_{i:A_i=1} [Y_hat_i(1) - Y_hat_i(0)]
```

For ATC (Average Treatment Effect on Controls):
```
ATC = E[Y(1) - Y(0) | A=0]
    = (1/n0) * sum_{i:A_i=0} [Y_hat_i(1) - Y_hat_i(0)]
```

## Reference Implementation

**R package**: `stdReg` (Sjolander, 2016)
- CRAN: https://cran.r-project.org/package=stdReg
- Documentation: https://cran.r-project.org/web/packages/stdReg/stdReg.pdf

**Alternative R packages**:
- `margins` (Leeper, 2021): https://cran.r-project.org/package=margins
- `Zelig` (Imai et al.): https://cran.r-project.org/package=Zelig

## Implementation Details

### Location
`crates/p2a-core/src/econometrics/stdreg.rs`

### Features Implemented
1. **Outcome Models**:
   - Linear (OLS) for continuous outcomes
   - Logistic (MLE) for binary outcomes
   - Poisson (MLE) for count outcomes

2. **Estimands**:
   - ATE (Average Treatment Effect)
   - ATT (Average Treatment Effect on Treated)
   - ATC (Average Treatment Effect on Controls)
   - Levels (E[Y(1)] and E[Y(0)] separately)

3. **Standard Error Methods**:
   - Bootstrap (default, recommended)
   - Delta method (analytical)
   - Sandwich (robust)

4. **Additional Effect Measures (Binary Outcomes)**:
   - Risk Ratio with CI
   - Odds Ratio with CI
   - Number Needed to Treat (NNT)

5. **Optional Features**:
   - Treatment-covariate interactions in outcome model
   - Individual treatment effects (CATE)

## Test Cases

### Test Case 1: Linear Outcome, ATE

**Data Generating Process**:
```
X ~ Uniform(0, 1)
A | X ~ Bernoulli(P(A=1) depending on X)
Y = 0.5 + 0.3*X + 0.4*A + noise

True ATE = 0.4
```

**Rust test**: `test_stdreg_linear_ate`

**Expected Results**:
- ATE should be approximately 0.4
- SE should be positive and finite
- E[Y(1)] > E[Y(0)]
- R-squared > 0.5

### Test Case 2: Binary Outcome, Logistic Model

**Data**: 40 observations with binary outcome

**Expected Results**:
- ATE should be positive (treatment increases P(Y=1))
- Risk Ratio > 1
- Odds Ratio > 1
- Risk Ratio and Odds Ratio CIs should bracket point estimates

### Test Case 3: ATT and ATC Estimands

**Expected**:
- ATT averages effect over treated only
- ATC averages effect over controls only
- Both should be positive for the test DGP

## R Validation Script

```r
# Validation script for stdReg comparison
library(stdReg)

# Generate test data
set.seed(42)
n <- 1000
x1 <- runif(n)
x2 <- runif(n)
# Treatment assignment depends on covariates
ps <- plogis(0.5 * x1 + 0.3 * x2)
treatment <- rbinom(n, 1, ps)
# Outcome: Y = 0.5 + 0.3*x1 + 0.2*x2 + 0.4*treatment + noise
y <- 0.5 + 0.3 * x1 + 0.2 * x2 + 0.4 * treatment + rnorm(n, 0, 0.3)

df <- data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2)

# Fit outcome model
model <- glm(y ~ treatment + x1 + x2, data = df)

# Standardize
std_fit <- stdGlm(model, data = df, X = "treatment")
summary(std_fit)

# Results should show:
# - E[Y(1)] - E[Y(0)] approximately 0.4
# - SE via delta method
```

## Tolerance Guidelines

| Metric | Tolerance |
|--------|-----------|
| Coefficients | < 1e-6 |
| Standard Errors | < 1e-4 |
| p-values | < 0.01 |
| Effect Estimates | < 0.05 (bootstrap variability) |

## MCP Tool

**Tool name**: `regression_standardization`

**Description**: Estimate causal effects using regression standardization (G-computation/parametric g-formula). Fits an outcome model and averages predictions under different treatment values over the covariate distribution.

**Parameters**:
- `dataset`: Name of loaded dataset
- `outcome`: Outcome variable column
- `treatment`: Binary treatment indicator column
- `covariates`: Covariate column names
- `model_type`: 'linear' (default), 'logistic', or 'poisson'
- `estimand`: 'ate' (default), 'att', 'atc', or 'levels'
- `se_method`: 'bootstrap' (default), 'delta', or 'sandwich'
- `n_bootstrap`: Number of bootstrap replications (default: 999)
- `interactions`: Include treatment-covariate interactions (default: false)
- `confidence_level`: Confidence level for intervals (default: 0.95)

## References

1. Robins, J.M. (1986). "A new approach to causal inference in mortality studies with a sustained exposure period -- application to control of the healthy worker survivor effect." *Mathematical Modelling*, 7(9-12), 1393-1512.

2. Snowden, J.M., Rose, S., & Mortimer, K.M. (2011). "Implementation of G-computation on a simulated dataset: Demonstration of a causal inference technique." *American Journal of Epidemiology*, 173(7), 731-738.

3. Hernan, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC. Chapter 13.

4. Sjolander, A. (2016). "Regression standardization with the R package stdReg." *European Journal of Epidemiology*, 31(6), 563-574.

## Status

- [x] Core implementation
- [x] Linear outcome model
- [x] Logistic outcome model
- [x] Poisson outcome model
- [x] ATE/ATT/ATC estimands
- [x] Bootstrap standard errors
- [x] Delta method standard errors
- [x] Sandwich standard errors
- [x] Risk ratio and odds ratio for binary outcomes
- [x] Unit tests (12 tests passing)
- [x] MCP tool integration
- [x] Documentation

## Notes

- Bootstrap SE is recommended as it accounts for the uncertainty in the outcome model estimation
- Delta method SE is faster but assumes asymptotic normality
- For binary outcomes, logistic model is recommended over linear to ensure predictions stay in [0,1]
- Treatment-covariate interactions allow for heterogeneous treatment effects
