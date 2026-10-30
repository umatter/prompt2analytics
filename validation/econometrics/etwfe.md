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

## Missing values and column types

`run_etwfe` (MCP `etwfe`) reads the outcome, unit, time, treatment, first treatment period and
controls through one `ColumnRequest`:

- a row with a missing value (null or NaN) in any of them is dropped from all of them, as
  fixest and etwfe drop incomplete rows, and `EtwfeResult::missing` and the last Display line
  report it. Nothing is imputed any more: a missing `first_treat` was read as 0 (never treated,
  so a treated unit became a control), a missing time as a period 0, a missing unit id as one
  pooled unit "0" (or ""), a missing treatment or control as 0.0;
- unit ids may be integers, fractional numbers, strings, categoricals or dates; time and
  `first_treat` hold whole numbers or dates (a fractional or infinite value is an error);
  never-treated units are coded 0, and an infinite `first_treat` (a common never-treated code,
  which used to run as a cohort without post cells) is refused with an error that says so;
- a text outcome is `NonNumericColumn`, an absent column `ColumnNotFound`.

Fixture tests, not R-pinned (`crates/p2a-core/src/econometrics/etwfe.rs`):
`test_na_etwfe_null_first_treat_drops_rows`, `test_na_etwfe_null_ids_and_time_drop_rows`,
`test_na_etwfe_errors_name_the_column`.

## Running the Tests

```bash
cargo test -p p2a-core -- etwfe::tests::test_validate
```

## References

- Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators." Working Paper.
