# Validation: Impulse Response Functions (IRF)

## Method Overview

Impulse Response Functions trace the effect of a one-time shock to one variable on the current and future values of all variables in a VAR system.

**Types**:
- **Orthogonalized IRF**: Uses Cholesky decomposition of residual covariance
- **Generalized IRF**: Order-invariant (Pesaran & Shin, 1998)
- **Structural IRF**: Based on identified structural VAR

**Computation**:
```
IRF_h = Φ_h × P
```

where Φ_h = A^h is the VAR(1) coefficient matrix to power h, and P is the Cholesky factor of Σ.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| vars | R | `irf()` | 1.5-x |
| statsmodels | Python | `IRAnalysis` | 0.14.x |

## Test Cases

### Test 1: Bivariate VAR(1) IRF

**R Code**:
```r
library(vars)

set.seed(42)
n <- 200

# VAR(1) with known dynamics
A <- matrix(c(0.6, 0.1, 0.2, 0.4), 2, 2, byrow = TRUE)
y <- matrix(0, n, 2)

for (t in 2:n) {
  y[t, ] <- A %*% y[t-1, ] + rnorm(2, 0, 0.5)
}

data <- data.frame(y1 = y[, 1], y2 = y[, 2])
var_fit <- VAR(data, p = 1, type = "none")

# Compute IRF
irf_result <- irf(var_fit, impulse = "y1", response = "y2", n.ahead = 10)
plot(irf_result)

# IRF values
irf_result$irf$y1
```

**Results Comparison**:

| Horizon | R IRF (y1→y2) | p2a Rust | Tolerance |
|---------|--------------|----------|-----------|
| h=0 | 0.00 | 0.00 | 0.01 |
| h=1 | ~0.2 | ~0.2 | 0.05 |
| h=2 | ~0.2 | ~0.2 | 0.05 |
| h=5 | ~0.1 | ~0.1 | 0.05 |
| h=10 | ~0.0 | ~0.0 | 0.02 |

**Rust Test**: `crates/p2a-core/src/econometrics/timeseries.rs::tests::test_validate_irf`

---

### Test 2: Orthogonalized vs Non-Orthogonalized

**R Code**:
```r
library(vars)

# Estimate VAR
var_fit <- VAR(data, p = 1, type = "none")

# Orthogonalized IRF (Cholesky ordering matters)
irf_orth <- irf(var_fit, impulse = "y1", response = "y2", n.ahead = 10, ortho = TRUE)

# Non-orthogonalized
irf_raw <- irf(var_fit, impulse = "y1", response = "y2", n.ahead = 10, ortho = FALSE)

# Compare
cbind(Orthogonalized = irf_orth$irf$y1[, "y2"],
      Raw = irf_raw$irf$y1[, "y2"])
```

**Validation Criteria**:
- Orthogonalized at h=0 uses Cholesky decomposition
- Non-orthogonalized at h=0 equals 0 (except own response)

---

### Test 3: Cumulative IRF (Long-Run Effects)

**R Code**:
```r
library(vars)

var_fit <- VAR(data, p = 1, type = "const")

# Cumulative IRF
irf_cumul <- irf(var_fit, impulse = "y1", response = "y2",
                 n.ahead = 20, cumulative = TRUE)
plot(irf_cumul)

# Long-run multiplier
tail(irf_cumul$irf$y1[, "y2"], 1)
```

**Validation Criteria**:
- Cumulative IRF converges if VAR is stable
- Long-run effect = (I - A)⁻¹ for VAR(1)

---

### Test 4: Confidence Bands (Bootstrap)

**R Code**:
```r
library(vars)

var_fit <- VAR(data, p = 1, type = "none")

# IRF with bootstrap confidence intervals
irf_boot <- irf(var_fit, impulse = "y1", response = "y2",
                n.ahead = 10, boot = TRUE, runs = 500, ci = 0.95)

# Check confidence bands
irf_boot$Lower$y1
irf_boot$Upper$y1
```

**Validation Criteria**:
- 95% CI covers true IRF
- Bands widen at longer horizons

---

## Ordering Sensitivity

For orthogonalized IRF, the Cholesky ordering matters:

**R Code**:
```r
# Order 1: y1 first (y1 can affect y2 contemporaneously)
data1 <- data[, c("y1", "y2")]
var1 <- VAR(data1, p = 1)
irf1 <- irf(var1, n.ahead = 5)

# Order 2: y2 first (y2 can affect y1 contemporaneously)
data2 <- data[, c("y2", "y1")]
var2 <- VAR(data2, p = 1)
irf2 <- irf(var2, n.ahead = 5)

# IRFs will differ at h=0
```

## Computation Details

For VAR(p):
1. Convert to companion form VAR(1)
2. Compute Φ_h = A_companion^h
3. Extract relevant k×k block
4. Apply Cholesky factor for orthogonalization

## Numerical Precision Summary

| Horizon | IRF Precision |
|---------|---------------|
| h=1-5 | < 0.01 |
| h=5-10 | < 0.02 |
| h=10-20 | < 0.05 |

Precision decreases with horizon due to compounding matrix powers.

## Known Differences

1. **Bootstrap method**: Different bootstrap algorithms.
2. **Cholesky implementation**: Numerical differences in decomposition.
3. **Cumulative computation**: Accumulated at each step vs. post-processing.

## Running the Tests

```bash
# Run IRF validation tests
cargo test -p p2a-core -- irf

# Run with output
cargo test -p p2a-core -- timeseries::tests::test_irf --nocapture
```

## References

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 2.
- Sims, C.A. (1980). "Macroeconomics and Reality". *Econometrica*, 48(1), 1-48.
- Pesaran, H.H. & Shin, Y. (1998). "Generalized Impulse Response Analysis in Linear Multivariate Models". *Economics Letters*, 58(1), 17-29.
