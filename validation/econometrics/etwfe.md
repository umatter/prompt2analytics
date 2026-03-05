# Validation: Extended Two-Way Fixed Effects (ETWFE)

## Method Overview

Extended Two-Way Fixed Effects implements the Wooldridge (2021) approach to heterogeneous treatment effects in staggered adoption designs. It estimates cohort-specific treatment effects by including cohort-by-time interactions and then aggregates to an overall ATT.

**Key Parameters**:
- `entity_col`, `time_col`: Panel identifiers
- `cohort_col`: Treatment cohort indicator
- `y_col`: Outcome variable
- `x_cols`: Optional covariates

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| etwfe | R | `etwfe()`, `emfx()` | 0.4.x |
| fixest | R | Underlying estimation engine | 0.11.x |

## Test Cases

### Test 1: Recovers Cohort-Specific Effects

**Data Generating Process**:
```
- 2 treatment cohorts with different effect sizes
- Cohort 1 (treated at t=3): effect = 2.0
- Cohort 2 (treated at t=5): effect = 4.0
- Never-treated control group
```

**R Code**:
```r
library(etwfe)
library(fixest)

set.seed(42)
n <- 300
T_max <- 8

# Panel data with staggered adoption
data <- expand.grid(id = 1:n, time = 1:T_max)
data$cohort <- rep(c(3, 5, Inf), each = n/3)[data$id]

# Generate outcome
data$treated <- data$time >= data$cohort & is.finite(data$cohort)
data$effect <- ifelse(data$cohort == 3, 2.0,
               ifelse(data$cohort == 5, 4.0, 0))
data$y <- rnorm(nrow(data)) + data$effect * data$treated

# ETWFE estimation
mod <- etwfe(y ~ 1, tvar = time, gvar = cohort,
             data = data, vcov = ~id)
emfx(mod)

# Cohort-specific effects should recover 2.0 and 4.0
```

**Rust Tests**:
- `test_validate_etwfe_recovers_cohort_specific_effects`
- `test_validate_etwfe_att_positive_with_positive_treatment`
- `test_validate_etwfe_event_study_pattern`
- `test_validate_etwfe_heterogeneous_vs_homogeneous`
- `test_validate_etwfe_structural_counts`

## Tolerance Levels

| Statistic | Tolerance | Notes |
|-----------|-----------|-------|
| Cohort ATT | 1.0 | DGP-based with noise |
| Overall ATT | 0.5 | Weighted average |
| Event-study coefficients | directional | Pre-treatment near 0, post positive |

## Running the Tests

```bash
cargo test -p p2a-core -- etwfe::tests::test_validate
```

## References

- Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators." Working Paper.
