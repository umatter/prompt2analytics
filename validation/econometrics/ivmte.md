# Marginal Treatment Effects (MTE) Validation

## Method Overview

The Marginal Treatment Effects (MTE) framework connects IV estimation to a choice-theoretic model of treatment selection. It reveals heterogeneity in treatment effects by showing that different IV estimands (LATE, ATT, ATE) are weighted averages of the MTE curve.

**Key concepts:**
- **MTE(u)**: Treatment effect for individuals at the margin of selection (indifferent at unobserved U=u)
- **Selection equation**: D = 1{P(Z) >= U} where P(Z) is propensity given instrument
- **LATE, ATT, ATE**: All are integrals of MTE with different weights

## Implementation Details

**File:** `crates/p2a-core/src/econometrics/ivmte.rs`

**Rust API:**
```rust
pub fn run_ivmte(
    y: &ArrayView1<f64>,           // Outcome
    d: &ArrayView1<f64>,           // Treatment (binary 0/1)
    z: &ArrayView1<f64>,           // Instrument
    x: Option<&ArrayView2<f64>>,   // Covariates (optional)
    config: IVMTEConfig,
) -> EconResult<IVMTEResult>
```

**MCP Tool:** `iv_mte`

## Reference Implementation

**R Package:** `ivmte` (Shea & Torgovitsky, 2021)
- CRAN: https://cran.r-project.org/package=ivmte
- Version: 1.4.0

### R Reference Code

```r
# Install if needed
# install.packages("ivmte")

library(ivmte)

# Generate test data with known MTE
set.seed(12345)
n <- 500

# Instrument
z <- rnorm(n)

# Unobserved heterogeneity (resistance to treatment)
u <- runif(n)

# Propensity score: P(Z) = Phi(0.5 + 0.8*Z)
p_z <- pnorm(0.5 + 0.8 * z)

# Treatment assignment: D = 1{P(Z) >= U}
d <- as.numeric(p_z >= u)

# Potential outcomes with heterogeneous treatment effects
# Y(0) = 1 + noise
# Y(1) = 1 + (1.5 + 0.5*U) + noise  (MTE = 1.5 + 0.5*U)
y0 <- 1 + rnorm(n, 0, 0.5)
y1 <- 1 + (1.5 + 0.5 * u) + rnorm(n, 0, 0.5)

# Observed outcome
y <- d * y1 + (1 - d) * y0

# Create data frame
data <- data.frame(y = y, d = d, z = z)

# Run ivmte
result <- ivmte(
  data = data,
  target = "ate",
  m0 = ~ 1,
  m1 = ~ 1 + u + I(u^2),
  propensity = d ~ z,
  instrument.name = "z",
  outcome.name = "y",
  treatment.name = "d",
  monocap.m0 = FALSE,
  monocap.m1 = FALSE
)

# Print results
print(result)
```

## Test Cases

### Test Case 1: Synthetic Data with Known MTE

**Data Generation:**
- n = 200 observations
- True MTE(u) = 1.5 + 0.5*u (linear, increasing in u)
- Propensity: P(Z) = 0.5 + 0.3*Z
- Selection: D = 1{P(Z) >= U}

**Expected Results:**
- ATE should be approximately 1.75 (integral of MTE)
- MTE curve should be increasing

### Test Case 2: Propensity Model Comparison

Compare results across different propensity models:
- Probit (default, as in Heckman-Vytlacil)
- Logit
- Linear probability model

All should give similar ATE estimates with well-specified instruments.

## Validation Results

### Rust vs R Comparison

| Estimand | Rust | R (ivmte) | Absolute Diff | Relative Diff |
|----------|------|-----------|---------------|---------------|
| ATE | TBD | TBD | TBD | TBD |
| ATT | TBD | TBD | TBD | TBD |
| ATU | TBD | TBD | TBD | TBD |
| LATE | TBD | TBD | TBD | TBD |

### Tolerance Thresholds

| Sample Size | Coefficients | Standard Errors | p-values |
|-------------|--------------|-----------------|----------|
| n = 200 | 0.05 | 0.02 | 0.05 |
| n = 1000 | 0.01 | 0.005 | 0.01 |
| n = 5000 | 0.001 | 0.001 | 0.001 |

**Note:** MTE estimation is inherently more variable than standard IV due to the
polynomial approximation and propensity score estimation stages.

## Rust Test Functions

Location: `crates/p2a-core/src/econometrics/ivmte.rs`

```rust
#[test]
fn test_ivmte_basic()

#[test]
fn test_ivmte_propensity_models()

#[test]
fn test_ivmte_polynomial_degree()

#[test]
fn test_treatment_effect_relationships()

#[test]
fn test_mte_curve()

#[test]
fn test_weights_sum_to_one()
```

## References

### Primary Sources

- Heckman, J.J., & Vytlacil, E. (2005). Structural equations, treatment effects,
  and econometric policy evaluation. *Econometrica*, 73(3), 669-738.
  https://doi.org/10.1111/j.1468-0262.2005.00594.x

- Heckman, J.J., & Vytlacil, E.J. (2007). Econometric evaluation of social
  programs, Part I: Causal models, structural models and econometric policy
  evaluation. *Handbook of Econometrics*, 6, 4779-4874.
  https://doi.org/10.1016/S1573-4412(07)06070-9

### Implementation References

- Cornelissen, T., Dustmann, C., Raute, A., & Schonberg, U. (2016). From LATE
  to MTE: Alternative methods for the evaluation of policy interventions.
  *Labour Economics*, 41, 47-60.
  https://doi.org/10.1016/j.labeco.2016.06.004

- Mogstad, M., Santos, A., & Torgovitsky, A. (2018). Using instrumental
  variables for inference about policy relevant treatment parameters.
  *Econometrica*, 86(5), 1589-1619.
  https://doi.org/10.3982/ECTA15463

### R Package

- Shea, J., & Torgovitsky, A. (2021). ivmte: Instrumental Variables for
  Marginal Treatment Effects. R package version 1.4.0.
  https://CRAN.R-project.org/package=ivmte

## Status

| Aspect | Status |
|--------|--------|
| Implementation | Complete |
| Unit Tests | Complete |
| MCP Tool | Complete |
| R Validation | Pending |
| Documentation | Complete |
