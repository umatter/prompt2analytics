# Validation: Generalized Synthetic Control (gsynth)

## Method Overview

The Generalized Synthetic Control Method extends traditional synthetic control to handle multiple treated units with staggered treatment adoption. It uses an Interactive Fixed Effects (IFE) model:

```
Y_it = α_i + λ_i'f_t + X_it'β + τ_it·D_it + ε_it
```

Where:
- `α_i` = unit fixed effects
- `λ_i` = unit-specific factor loadings (N × r)
- `f_t` = time-varying latent factors (T × r)
- `β` = covariate coefficients
- `τ_it` = treatment effect for unit i at time t
- `D_it` = treatment indicator (0/1)

The method uses control units in the pre-treatment period to estimate factors and loadings, then constructs counterfactuals for treated units.

**Key Parameters**:
- `n_factors`: Number of latent factors (can be selected via cross-validation)
- `force`: Fixed effects specification ("none", "unit", "time", "twoWay")
- `estimator`: "ife" (interactive fixed effects) or "mc" (matrix completion)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| gsynth | R | `gsynth()` | 1.2.1 |

## Test Cases

### Test 1: Synthetic Panel Data with Known Effects

**Data Generating Process**:
```
3 control units (C1, C2, C3), 2 treated units (T1, T2)
10 time periods, treatment starts at t=7 for T1, t=8 for T2
Pre-treatment periods: 1-6 (6 periods)
Treatment effects: 5.0 for T1, 3.0 for T2
```

**R Code (gsynth)**:
```r
library(gsynth)

# Create panel data
set.seed(42)
n_control <- 3
n_treated <- 2
n_times <- 10

# Generate data frame
units <- c()
times <- c()
outcomes <- c()
treatment <- c()

# Control units (never treated)
for (unit in c("C1", "C2", "C3")) {
  for (t in 1:10) {
    units <- c(units, unit)
    times <- c(times, t)
    base <- 10 + t + ifelse(unit == "C2", 2, 0)
    outcomes <- c(outcomes, base)
    treatment <- c(treatment, 0)
  }
}

# Treated unit 1: treatment at t=7
for (t in 1:10) {
  units <- c(units, "T1")
  times <- c(times, t)
  base <- 10 + t + 1
  effect <- ifelse(t >= 7, 5, 0)
  outcomes <- c(outcomes, base + effect)
  treatment <- c(treatment, ifelse(t >= 7, 1, 0))
}

# Treated unit 2: treatment at t=8
for (t in 1:10) {
  units <- c(units, "T2")
  times <- c(times, t)
  base <- 10 + t - 0.5
  effect <- ifelse(t >= 8, 3, 0)
  outcomes <- c(outcomes, base + effect)
  treatment <- c(treatment, ifelse(t >= 8, 1, 0))
}

panel <- data.frame(
  unit = units,
  time = times,
  outcome = outcomes,
  treated = treatment
)

# Run gsynth
result <- gsynth(
  outcome ~ treated,
  data = panel,
  index = c("unit", "time"),
  force = "unit",
  r = 1,  # 1 factor
  CV = FALSE,
  se = FALSE
)

print(result)
# ATT should be positive, approximately (5*4 + 3*3) / 7 ≈ 4.14
```

**Results Comparison**:

| Statistic | R (gsynth) | p2a Rust | Tolerance |
|-----------|------------|----------|-----------|
| ATT | ~4.0 | ~4.0 | 1.0 |
| n_treated | 2 | 2 | exact |
| n_control | 3 | 3 | exact |
| n_factors | 1 | 1 | exact |

**Rust Test**: `crates/p2a-core/src/econometrics/synth.rs::gsynth_tests::test_gsynth_basic`

---

### Test 2: Cross-Validation Factor Selection

**Description**: Test that cross-validation selects appropriate number of factors.

**R Code**:
```r
# Same data as Test 1, but with CV
result_cv <- gsynth(
  outcome ~ treated,
  data = panel,
  index = c("unit", "time"),
  force = "unit",
  r = c(0, 3),  # CV from 0 to 3 factors
  CV = TRUE,
  se = FALSE
)

print(result_cv$r.cv)  # Selected number of factors
```

**Rust Test**: `crates/p2a-core/src/econometrics/synth.rs::gsynth_tests::test_gsynth_cv`

---

## Numerical Precision Summary

| Component | Expected Precision |
|-----------|-------------------|
| ATT | Within 1.0 of R |
| Unit ATTs | Within 1.0 of R |
| Factor values | Sign and order of magnitude |
| n_factors (CV) | Exact or ±1 |

## Known Differences

1. **Factor normalization**: Different PCA conventions may lead to different factor/loading scales (but same fitted values)
2. **CV fold assignment**: Random fold assignment may differ from R
3. **Small sample behavior**: With few control units, factor estimation may be less stable

## Performance Comparison

Benchmarks run on criterion (Rust) with 100 samples each, compared against R gsynth package (v1.3+, fect wrapper).

| Dataset Size | Rust (µs) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=12, T=15 (180 obs) | 31 | ~4.2 | **~135x** |
| n=25, T=20 (500 obs) | 79 | ~6.6 | **~84x** |
| n=60, T=25 (1500 obs) | 206 | ~15 | **~73x** |

*R comparison uses similar panel sizes (n=20/T=10, n=50/T=20, n=100/T=50).*

The Rust implementation achieves significant speedups due to:
- Eigenvalue decomposition via power iteration (avoids full SVD)
- Pre-allocated matrices for factor computation
- Regularized matrix inversions for numerical stability
- Minimal memory allocations during counterfactual construction

## References

- Xu, Y. (2017). "Generalized Synthetic Control Method: Causal Inference with Interactive Fixed Effects Models." *Political Analysis*, 25(1), 57-76.
- R Package: `gsynth` (Yiqing Xu). https://yiqingxu.org/packages/gsynth/
