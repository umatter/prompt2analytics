# Average Marginal Effects (AME) Validation

## Method Overview

Marginal effects measure the change in the expected value of the dependent variable
for a one-unit change in an independent variable, holding all other variables constant.

For **OLS**: Marginal effects equal the coefficients (constant for all observations).

For **Logit/Probit**: Marginal effects depend on the covariate values:
- Logit: dP/dx_j = beta_j * Lambda(X'beta) * (1 - Lambda(X'beta))
- Probit: dP/dx_j = beta_j * phi(X'beta)

Average Marginal Effects (AME) average these over all observations:
- AME_j = (1/n) * sum_{i=1}^n beta_j * f(X_i'beta)

where f is the PDF of the link function.

## Implementation Location

- **Core module**: `crates/p2a-core/src/regression/marginal_effects.rs`
- **Public exports**: `crates/p2a-core/src/regression/mod.rs`

## Key Functions

| Function | Description |
|----------|-------------|
| `marginal_effects_ols` | Compute ME from OLS result (equals coefficients) |
| `marginal_effects_discrete` | Compute AME/MEM from Logit/Probit result |
| `marginal_effects` | Dispatcher: estimate model and compute ME |
| `contrasts` | Compute contrasts between specified values |

## Result Structures

```rust
pub struct MarginalEffectsResult {
    pub average_marginal: Vec<MarginalEffect>,  // AME for each variable
    pub at_means: Option<Vec<MarginalEffect>>,  // MEM (marginal effect at means)
    pub model_type: ModelType,
    pub n_obs: usize,
    pub variables: Vec<String>,
    pub has_intercept: bool,
}

pub struct MarginalEffect {
    pub variable: String,
    pub estimate: f64,       // Point estimate
    pub std_error: f64,      // Standard error (delta method)
    pub z_value: f64,
    pub p_value: f64,
    pub ci_lower: f64,       // 95% CI lower bound
    pub ci_upper: f64,       // 95% CI upper bound
    pub dy_dx: f64,          // Same as estimate
    pub significance: SignificanceLevel,
}
```

## Reference Implementation

R package: `marginaleffects` (Arel-Bundock, 2023)
https://vincentarelbundock.github.io/marginaleffects/

Alternative: `margins` (Leeper, 2021)
https://cran.r-project.org/package=margins

## Test Case 1: OLS Marginal Effects

For OLS, marginal effects equal coefficients exactly.

### R Code

```r
library(marginaleffects)

# Generate data
set.seed(42)
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- 1 + 2 * x1 + 0.5 * x2 + rnorm(n, sd = 0.5)
data <- data.frame(y = y, x1 = x1, x2 = x2)

# Fit OLS
model <- lm(y ~ x1 + x2, data = data)

# Compute marginal effects
me <- avg_slopes(model)
print(me)
```

### Expected Results

For OLS, AME should equal coefficients:
- beta_1 (x1): approximately 2.0
- beta_2 (x2): approximately 0.5

### Rust Test

```rust
#[test]
fn test_marginal_effects_ols() {
    let dataset = create_test_dataset();
    let ols = run_ols(&dataset, "y", &["x1", "x2"], true, CovarianceType::Standard)?;
    let me = marginal_effects_ols(&ols)?;

    // For OLS, ME = coefficients
    for (i, coef) in ols.coefficients.iter().enumerate() {
        assert!((me.average_marginal[i].estimate - coef.estimate).abs() < 1e-10);
    }
}
```

## Test Case 2: Logit AME

### R Code

```r
library(marginaleffects)

# Generate binary outcome data
set.seed(123)
n <- 500
x1 <- rnorm(n)
x2 <- rnorm(n)
z <- 0.5 + 0.8 * x1 + 0.3 * x2
p <- plogis(z)
y <- rbinom(n, 1, p)
data <- data.frame(y = y, x1 = x1, x2 = x2)

# Fit logit
model <- glm(y ~ x1 + x2, data = data, family = binomial(link = "logit"))

# Compute AME
me <- avg_slopes(model)
print(me)

# Compare with coefficients (AME should be smaller)
print(coef(model))
```

### Expected Results

For Logit with true coefficients beta_1 = 0.8, beta_2 = 0.3:
- AME_x1: approximately 0.15-0.20 (scaled by mean PDF)
- AME_x2: approximately 0.05-0.08

Key property: |AME| < |coefficient| for all variables.

### Rust Test

```rust
#[test]
fn test_marginal_effects_logit() {
    let dataset = create_binary_dataset();
    let logit = run_logit(&dataset, "y", &["x1", "x2"])?;
    let me = marginal_effects_discrete(&logit, &dataset, &["x1", "x2"])?;

    // AME should be smaller than coefficients
    for (i, me_i) in me.average_marginal.iter().enumerate().skip(1) {
        assert!(me_i.estimate.abs() < logit.coefficients[i].abs());
    }
}
```

## Test Case 3: Probit AME

### R Code

```r
library(marginaleffects)

# Use same data as logit
model <- glm(y ~ x1 + x2, data = data, family = binomial(link = "probit"))
me <- avg_slopes(model)
print(me)
```

### Expected Results

Similar pattern to logit, but different scaling:
- Probit coefficients are typically about 0.625x logit coefficients
- AME_probit ~ AME_logit (approximately equal)

## Standard Error Computation

Standard errors are computed using the delta method:

SE(AME_j) = sqrt(G_j' * Var(beta) * G_j)

where G_j is the gradient of AME_j with respect to beta.

For Logit:
G_j[l] = (1/n) * sum_i [I(j=l) * lambda(z_i) + beta_j * lambda'(z_i) * x_il]

where:
- lambda(z) = Lambda(z) * (1 - Lambda(z)) is the logistic PDF
- lambda'(z) = lambda(z) * (1 - 2*Lambda(z)) is the derivative

For Probit:
G_j[l] = (1/n) * sum_i [I(j=l) * phi(z_i) + beta_j * (-z_i * phi(z_i)) * x_il]

where phi is the standard normal PDF.

## Tolerance Guidelines

| Statistic | Tolerance |
|-----------|-----------|
| AME estimate | 1e-4 (relative to coefficient) |
| Standard error | 1e-3 (relative) |
| z-value | 1e-2 |
| p-value | 0.01 |

Note: Some discrepancy is expected due to:
1. Different optimization algorithms
2. Different starting values
3. Numerical precision in delta method computation

## References

- Bartus, T. (2005). Estimation of marginal effects using margeff. *The Stata Journal*, 5(3), 309-329.

- Cameron, A.C., & Trivedi, P.K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 15.

- Greene, W.H. (2018). *Econometric Analysis* (8th ed.). Pearson. Chapter 14.

- Leeper, T.J. (2021). margins: Marginal effects for model objects. R package version 0.3.26. https://CRAN.R-project.org/package=margins

- Arel-Bundock, V. (2023). marginaleffects: Predictions, Comparisons, Slopes, Marginal Means, and Hypothesis Tests. R package. https://vincentarelbundock.github.io/marginaleffects/

## Status

- [x] Core implementation complete
- [x] OLS marginal effects
- [x] Logit/Probit AME with delta method SEs
- [x] Marginal effects at the mean (MEM)
- [x] Contrasts
- [x] Unit tests
- [ ] MCP tool integration (pending server.rs stabilization)
- [ ] Cross-validation with R marginaleffects package

## Change Log

- 2026-01-24: Initial implementation with OLS, Logit, and Probit support
