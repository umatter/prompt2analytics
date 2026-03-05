# Validation: Staggered Difference-in-Differences (Callaway-Sant'Anna)

## Method Overview

Staggered DiD estimates group-time average treatment effects (ATT(g,t)) when treatment timing varies across units. The Callaway and Sant'Anna (2021) estimator avoids the bias from two-way fixed effects under heterogeneous treatment effects.

**Key Parameters**:
- `entity_col`, `time_col`: Panel identifiers
- `treatment_time_col`: Period when each unit first receives treatment
- `y_col`: Outcome variable
- `comparison_group`: Never-treated or not-yet-treated
- `estimation_method`: IPW, outcome regression, or doubly robust
- `aggregation`: Simple, group, calendar, or event-study

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| did | R | `att_gt()`, `aggte()` | 2.1.x |

## Test Cases

### Test 1: Group-Time ATTs Recover True Effects

**Data Generating Process**:
```
- 3 cohorts: treated at t=3, t=5, t=7
- Never-treated control group
- Constant treatment effect = 2.0 per cohort
- n = 400 units, T = 10 periods
```

**R Code**:
```r
library(did)

set.seed(42)
n_units <- 400
n_periods <- 10

id <- rep(1:n_units, each = n_periods)
time <- rep(1:n_periods, times = n_units)

# Assign cohorts
cohort <- rep(NA, n_units)
cohort[1:100] <- 3
cohort[101:200] <- 5
cohort[201:300] <- 7
cohort[301:400] <- 0  # never treated
cohort_expanded <- rep(cohort, each = n_periods)

# Generate outcome
treat_effect <- 2.0
y <- rnorm(n_units * n_periods) +
     treat_effect * (time >= cohort_expanded & cohort_expanded > 0)

data <- data.frame(id = id, time = time, y = y,
                   first_treat = cohort_expanded)

result <- att_gt(yname = "y", tname = "time", idname = "id",
                 gname = "first_treat", data = data,
                 control_group = "nevertreated")
summary(result)

# Post-treatment ATT(g,t) estimates should be near 2.0
# Pre-treatment ATT(g,t) estimates should be near 0
```

**Validation Criteria**:
- Post-treatment ATT(g,t) estimates within 0.5 of 2.0
- Pre-treatment ATT(g,t) estimates not significantly different from 0

**Rust Tests**:
- `crates/p2a-core/src/econometrics/staggered_did.rs::tests::test_validate_staggered_did_group_time_atts_recover_true_effects`

### Test 2: Event-Study Pre-Treatment Near Zero

**Validation Criteria**:
- Event-study estimates at negative event-times close to 0
- Supports parallel trends assumption

**Rust Test**: `test_validate_staggered_did_event_study_pre_treatment_near_zero`

### Test 3: Event-Study Post-Treatment Positive

**Rust Test**: `test_validate_staggered_did_event_study_post_treatment_positive`

### Test 4: Overall ATT is Weighted Average

**Rust Test**: `test_validate_staggered_did_overall_att_weighted_average`

### Test 5: Group Effects Heterogeneity

**Rust Test**: `test_validate_staggered_did_group_effects_heterogeneity`

### Test 6: Never-Treated Comparison

**Rust Test**: `test_validate_staggered_did_never_treated_comparison`

### Test 7: Homogeneous Effect

**Rust Test**: `test_validate_staggered_did_homogeneous_effect`

## Tolerance Levels

| Statistic | Tolerance | Notes |
|-----------|-----------|-------|
| ATT(g,t) post-treatment | 0.5 | DGP-based, stochastic |
| ATT(g,t) pre-treatment | not sig. at 0.05 | Parallel trends check |
| Overall ATT | 0.5 | Weighted average of group-time ATTs |
| Standard errors | 0.3 | Approximate due to bootstrap |

## Running the Tests

```bash
cargo test -p p2a-core -- staggered_did::tests::test_validate
```

## References

- Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with multiple time periods." *Journal of Econometrics*, 225(2), 200-230.
