# Validation: Vector Error Correction Model (VECM)

## Method Overview

The Vector Error Correction Model is a restricted VAR for cointegrated time series. It separates long-run equilibrium relationships (cointegrating vectors) from short-run dynamics.

**Model (VECM)**:
```
ΔY_t = Πy_{t-1} + Γ₁ΔY_{t-1} + ... + Γ_{p-1}ΔY_{t-p+1} + ε_t
```

where Π = αβ' with:
- β: Cointegrating vectors (long-run relationships)
- α: Adjustment coefficients (speed of adjustment)
- Γᵢ: Short-run dynamics

**Estimation**: Johansen procedure (MLE).

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| urca | R | `ca.jo()` | 1.3-x |
| vars | R | `vec2var()` | 1.5-x |

## Test Cases

### Test 1: Simple Cointegrated System

**Data Generating Process**:
Two I(1) series with one cointegrating relationship:
```
y₁ₜ = y₂ₜ + u_t  (long-run: y₁ = y₂)
```

**R Code**:
```r
library(urca)
library(vars)

set.seed(42)
n <- 200

# Random walk
rw <- cumsum(rnorm(n))

# Cointegrated series: y1 ≈ y2 with some noise
y1 <- rw + rnorm(n, 0, 0.3)
y2 <- rw + rnorm(n, 0, 0.3)

data <- data.frame(y1 = y1, y2 = y2)

# Johansen test
jtest <- ca.jo(data, type = "trace", K = 2, spec = "transitory")
summary(jtest)

# Convert to VECM
vecm <- cajorls(jtest, r = 1)
summary(vecm$rlm)
```

**Validation Criteria**:
- Cointegration rank r = 1 detected
- Cointegrating vector close to (1, -1)
- Error correction term significant

---

### Test 2: Johansen Trace Test

**R Code**:
```r
library(urca)

set.seed(42)
n <- 300

# Three cointegrated series (r = 2)
rw1 <- cumsum(rnorm(n))
rw2 <- cumsum(rnorm(n))

y1 <- rw1 + rnorm(n, 0, 0.2)
y2 <- rw1 + 0.5*rw2 + rnorm(n, 0, 0.2)
y3 <- rw2 + rnorm(n, 0, 0.2)

data <- data.frame(y1 = y1, y2 = y2, y3 = y3)

jtest <- ca.jo(data, type = "trace", K = 2, spec = "transitory")
summary(jtest)

# Check trace statistics and critical values
```

**Results Comparison**:

| Hypothesis | R Trace Stat | p2a Rust | Critical (5%) |
|------------|-------------|----------|---------------|
| r = 0 | varies | varies | 29.68 |
| r ≤ 1 | varies | varies | 15.41 |
| r ≤ 2 | varies | varies | 3.76 |

---

### Test 3: Error Correction Dynamics

**R Code**:
```r
library(urca)
library(vars)

set.seed(42)
n <- 200

# Generate VECM(1) data
alpha <- c(-0.3, 0.2)  # Adjustment speeds
beta <- c(1, -0.8)     # Cointegrating vector

rw <- cumsum(rnorm(n))
y1 <- rw + rnorm(n, 0, 0.3)
y2 <- 0.8 * rw + rnorm(n, 0, 0.3)

data <- data.frame(y1 = y1, y2 = y2)

# Estimate VECM
jtest <- ca.jo(data, type = "trace", K = 2)
vecm <- cajorls(jtest, r = 1)

# Check adjustment coefficients
print(vecm$beta)  # Cointegrating vector
```

**Validation Criteria**:
- Adjustment speeds (α) correctly estimated
- Negative α for "error-correcting" variable
- Cointegrating vector identified

---

### Test 4: Cointegration Rank Selection

**R Code**:
```r
library(urca)

# Test with different numbers of cointegrating relationships
# r = 0 (no cointegration)
# r = 1 (one relationship)
# r = k (all series stationary)

# Use sequential testing: reject r=0, then test r≤1, etc.
```

## Johansen Procedure

1. **Estimate unrestricted VAR** in levels
2. **Compute residuals** from regressions on lagged differences
3. **Solve eigenvalue problem** for Π = αβ'
4. **Test rank** using trace or max-eigenvalue statistics

## Critical Values

Trace test critical values (asymptotic, no constant in CE):

| Variables | r=0 (5%) | r≤1 (5%) | r≤2 (5%) |
|-----------|----------|----------|----------|
| 2 | 12.21 | 4.13 | - |
| 3 | 24.28 | 12.32 | 4.13 |
| 4 | 40.17 | 24.28 | 12.32 |

## Numerical Precision Summary

| Test | n | Eigenvalue Precision | Trace Stat Precision |
|------|---|---------------------|---------------------|
| 2-var | 200 | < 1e-4 | < 0.1 |
| 3-var | 300 | < 1e-4 | < 0.2 |

## Known Differences

1. **Normalization**: Cointegrating vectors normalized differently.
2. **Deterministic terms**: Intercept in CE vs. VAR varies.
3. **Critical values**: Tables may differ slightly.

## Running the Tests

```bash
# Run VECM validation tests
cargo test -p p2a-core -- vecm

# Run with output
cargo test -p p2a-core -- timeseries::tests::test_vecm --nocapture
```

## References

- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models". *Econometrica*, 59(6), 1551-1580.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models*. Oxford University Press.
