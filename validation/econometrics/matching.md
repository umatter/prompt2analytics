# Propensity Score Matching (MatchIt) Validation

## Method Overview

Propensity score matching creates balanced comparison groups for causal inference from observational data by matching treated units to similar control units.

### Supported Methods

| Method | Description | R Equivalent |
|--------|-------------|--------------|
| Nearest Neighbor | Match to closest control by PS distance | `matchit(..., method = "nearest")` |
| CEM | Coarsened exact matching within strata | `matchit(..., method = "cem")` |
| Full/Optimal | Optimal stratification | `matchit(..., method = "full")` |
| Subclassification | PS stratification | `matchit(..., method = "subclass")` |

### Distance Metrics

| Distance | Description |
|----------|-------------|
| Logit | Propensity score via logistic regression (default) |
| Probit | Propensity score via probit regression |
| Mahalanobis | Mahalanobis distance on covariates |
| Euclidean | Euclidean distance on covariates |

## Reference Implementation

R package `MatchIt` version 4.5.5 (Ho et al., 2011)

```r
library(MatchIt)
library(cobalt)  # For balance diagnostics
```

## Test Cases

### Test Case 1: Nearest Neighbor 1:1 Matching

**Data Generation (R):**
```r
set.seed(42)
n <- 500

# Generate covariates
x1 <- rnorm(n)
x2 <- rnorm(n)

# Treatment assignment depends on covariates (creates imbalance)
ps_true <- plogis(-1 + 0.5*x1 + 0.5*x2)
treatment <- rbinom(n, 1, ps_true)

# Create dataset
df <- data.frame(
  treatment = treatment,
  x1 = x1,
  x2 = x2
)

# Run MatchIt
m.out <- matchit(treatment ~ x1 + x2, data = df, method = "nearest",
                 distance = "logit", replace = FALSE)

# Get balance statistics
bal <- bal.tab(m.out, un = TRUE)
print(summary(m.out))
```

**Expected Results:**
- Before matching: SMD for x1 and x2 > 0.1 (imbalanced)
- After matching: SMD for x1 and x2 < 0.1 (balanced)
- All treated units matched (n_matched_treated = n_treated)
- No replacement: each control matched at most once

### Test Case 2: Nearest Neighbor with Caliper

**R Code:**
```r
m.out <- matchit(treatment ~ x1 + x2, data = df, method = "nearest",
                 distance = "logit", caliper = 0.2, replace = FALSE)
summary(m.out)
```

**Expected Results:**
- Some treated units may be discarded if no valid match within caliper
- Better balance than without caliper
- Caliper = 0.2 SD of propensity score

### Test Case 3: Coarsened Exact Matching (CEM)

**R Code:**
```r
m.out <- matchit(treatment ~ x1 + x2, data = df, method = "cem")
summary(m.out)
```

**Expected Results:**
- Creates strata based on covariate bins
- Exact matching within strata
- Some units may be pruned (no match in stratum)
- Weights assigned to maintain balance

### Test Case 4: Full Matching

**R Code:**
```r
m.out <- matchit(treatment ~ x1 + x2, data = df, method = "full")
summary(m.out)
```

**Expected Results:**
- All units matched (no discarding)
- Creates optimal strata containing treated and control
- Weights sum to n_treated within each stratum

### Test Case 5: Subclassification

**R Code:**
```r
m.out <- matchit(treatment ~ x1 + x2, data = df, method = "subclass",
                 subclass = 5)
summary(m.out)
```

**Expected Results:**
- Creates 5 propensity score subclasses
- Within-stratum balance
- All units receive a subclass assignment

## Balance Diagnostics

### Standardized Mean Difference (SMD)

Formula: `(mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)`

**Rule of thumb:** |SMD| < 0.1 indicates good balance (Rosenbaum & Rubin, 1985)

### Variance Ratio

Formula: `var_treated / var_control`

**Rule of thumb:** 0.5 < ratio < 2 indicates acceptable balance

### Kolmogorov-Smirnov Statistic

Maximum absolute difference between treated and control CDFs.

**Rule of thumb:** KS < 0.1 indicates similar distributions

## Rust Test Functions

```rust
#[test]
fn test_validate_nearest_neighbor_against_r() {
    // ... test implementation
}

#[test]
fn test_validate_cem_against_r() {
    // ... test implementation
}

#[test]
fn test_balance_improvement() {
    // ... test implementation
}
```

## Tolerances

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Propensity scores | 1e-4 | Optimization may vary |
| SMD | 0.02 | Small differences due to random tie-breaking |
| Weights | 1e-6 | Exact for most methods |
| n_matched | 0 | Exact match expected |

## Known Differences from R

1. **Tie-breaking:** When multiple controls have the same distance, the order may differ between R and Rust implementations.

2. **Full matching:** Our implementation uses a greedy approximation rather than optimal matching due to computational complexity.

3. **CEM cutpoints:** Default cutpoints may differ slightly; we use quartiles by default.

## References

- Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2007). Matching as Nonparametric
  Preprocessing for Reducing Model Dependence in Parametric Causal Inference.
  *Political Analysis*, 15(3), 199-236. https://doi.org/10.1093/pan/mpl013

- Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity Score
  in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41-55.
  https://doi.org/10.1093/biomet/70.1.41

- Iacus, S.M., King, G., & Porro, G. (2012). Causal Inference without Balance
  Checking: Coarsened Exact Matching. *Political Analysis*, 20(1), 1-24.
  https://doi.org/10.1093/pan/mpr013

- Hansen, B.B. (2004). Full Matching in an Observational Study of Coaching for
  the SAT. *Journal of the American Statistical Association*, 99(467), 609-618.
  https://doi.org/10.1198/016214504000000647

- R package `MatchIt`: Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2011).
  MatchIt: Nonparametric Preprocessing for Parametric Causal Inference.
  *Journal of Statistical Software*, 42(8), 1-28.
  https://cran.r-project.org/package=MatchIt

## Validation Status

| Method | Validation Status | Notes |
|--------|-------------------|-------|
| Nearest Neighbor | Implemented | Unit tests pass |
| CEM | Implemented | Unit tests pass |
| Full Matching | Implemented (greedy) | Unit tests pass |
| Subclassification | Implemented | Unit tests pass |
| Balance diagnostics | Implemented | SMD, variance ratio, KS |
