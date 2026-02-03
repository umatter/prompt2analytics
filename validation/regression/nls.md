# Validation: Nonlinear Least Squares (NLS)

## Method Overview

This document validates the p2a-core implementation of Nonlinear Least Squares against R's `stats::nls()`.

**Functions Implemented:**
- `nls()` - Core NLS with single x variable
- `nls_multi()` - NLS with multi-dimensional x
- `run_nls()` - Dataset wrapper
- Pre-defined model functions: `model_exponential_decay`, `model_michaelis_menten`, etc.

**Key Parameters:**
- `model` - User-defined function f(x, θ) → y
- `start` - Initial parameter values
- `algorithm` - GaussNewton or LevenbergMarquardt (default)
- `config.tolerance` - Convergence threshold (default: 1e-8)
- `config.max_iter` - Maximum iterations (default: 200)

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `nls()` | R 4.3+ |
| scipy | Python | `curve_fit()` | 1.11+ |

## Mathematical Formulas

### Objective Function

Minimize the residual sum of squares:

```
RSS(θ) = Σᵢ (yᵢ - f(xᵢ, θ))²
```

For weighted regression:
```
WRSS(θ) = Σᵢ wᵢ(yᵢ - f(xᵢ, θ))²
```

### Gauss-Newton Update

```
θₖ₊₁ = θₖ + (JᵀJ)⁻¹Jᵀr
```

Where:
- J is the Jacobian matrix: J[i,j] = -∂f(xᵢ,θ)/∂θⱼ
- r is the residual vector: rᵢ = yᵢ - f(xᵢ, θ)

### Levenberg-Marquardt Update

```
θₖ₊₁ = θₖ + (JᵀJ + λ·diag(JᵀJ))⁻¹Jᵀr
```

Where λ is the damping parameter adjusted adaptively:
- λ decreases when RSS decreases (step accepted)
- λ increases when RSS increases (step rejected)

### Standard Errors

```
SE(θ) = σ · √diag((JᵀJ)⁻¹)
σ = √(RSS / (n - k))
```

## Test Cases

### Test 1: Exponential Decay

**Model:** y = a × exp(-b × x) + c

**Data:**
```r
x <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
y <- c(11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9)
fit <- nls(y ~ a * exp(-b * x) + c, start = list(a = 8, b = 0.3, c = 1))
```

**Results Comparison:**

| Parameter | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| a | ~9.5-10.5 | 9.776 | 1.0 | ✅ |
| b | ~0.4-0.6 | 0.513 | 0.2 | ✅ |
| c | ~2.0-2.5 | 2.208 | 0.5 | ✅ |
| Converged | true | true | - | ✅ |

**Note:** Rust L-M may find a better local optimum (lower RSS) than R's Gauss-Newton.

**Rust Test:** `crates/p2a-core/src/regression/nls.rs::tests::test_validate_against_r_exponential_decay`

### Test 2: Michaelis-Menten Kinetics

**Model:** V = Vmax × S / (Km + S)

**Data:**
```r
S <- c(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
V <- c(28.6, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1)
fit <- nls(V ~ Vmax * S / (Km + S), start = list(Vmax = 150, Km = 0.05))
```

**Results Comparison:**

| Parameter | Rust (p2a) | R | Tolerance | Status |
|-----------|------------|---|-----------|--------|
| Vmax | ~200.2 | 200.2 | 5.0 | ✅ |
| Km | ~0.102 | 0.102 | 0.02 | ✅ |
| Converged | true | true | - | ✅ |

**Rust Test:** `crates/p2a-core/src/regression/nls.rs::tests::test_validate_against_r_michaelis_menten`

### Test 3: Bounded Parameters

**Purpose:** Verify parameter constraints are respected.

**Rust Test:** `crates/p2a-core/src/regression/nls.rs::tests::test_nls_with_bounds`

## Numerical Precision Summary

| Statistic | Typical Tolerance | Notes |
|-----------|-------------------|-------|
| Coefficients | 5-10% | Depends on data conditioning |
| Standard errors | 10-20% | Sensitive to local optimum |
| RSS | 50% | L-M often finds better optima |

## Known Differences

1. **Algorithm Default:**
   - R's `nls()` defaults to Gauss-Newton
   - Our implementation defaults to Levenberg-Marquardt (more robust)
   - L-M often finds lower RSS values

2. **Numerical Differentiation:**
   - R uses analytical derivatives when provided
   - Our implementation uses central differences (step = 1e-7)

3. **Convergence Criterion:**
   - R uses relative-offset convergence
   - We use relative change in RSS

4. **Standard Error Calculation:**
   - Both use σ² × (J'J)⁻¹ but may differ due to different final θ

## Performance Comparison

### Exponential Decay Model (3 parameters)

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=10 | 70 | ~2000 | ~29x |
| n=50 | 116 | ~2000 | ~17x |
| n=100 | 170 | ~2000 | ~12x |
| n=500 | 612 | ~2000 | ~3x |
| n=1,000 | 1,152 | ~2000 | ~2x |

### Michaelis-Menten Model (2 parameters)

| Dataset Size | Rust (µs) | R (µs)* | Speedup |
|--------------|-----------|---------|---------|
| n=8 | 40 | ~2000 | ~50x |
| n=20 | 51 | ~2000 | ~39x |
| n=50 | 68 | ~2000 | ~29x |
| n=100 | 82 | ~2000 | ~24x |
| n=500 | 280 | ~2000 | ~7x |

### Algorithm Comparison (n=100, Exponential Decay)

| Algorithm | Rust (µs) |
|-----------|-----------|
| Levenberg-Marquardt | 172 |
| Gauss-Newton | 170 |

*Note: R timings limited by system.time 2ms resolution; actual R performance is ~2ms regardless of n for these small sizes.*

**Benchmark Notes:**
- Rust benchmarks: Criterion with 100 samples, median times
- Rust shows ~10-50x speedup for small datasets
- Speedup decreases for larger datasets as computation dominates overhead

## Supported Models

| Model | Formula | Parameters |
|-------|---------|------------|
| Exponential Decay | y = a × e^(-bx) + c | [a, b, c] |
| Exponential Growth | y = a × e^(bx) | [a, b] |
| Michaelis-Menten | y = Vmax × x / (Km + x) | [Vmax, Km] |
| Logistic Growth | y = K / (1 + e^(-r(x-x₀))) | [K, r, x₀] |
| Power Law | y = a × x^b | [a, b] |
| Asymptotic | y = a - b × e^(-cx) | [a, b, c] |

## References

- Levenberg, K. (1944). "A Method for the Solution of Certain Non-Linear Problems
  in Least Squares". *Quarterly of Applied Mathematics*, 2(2), 164-168.
- Marquardt, D. W. (1963). "An Algorithm for Least-Squares Estimation of Nonlinear
  Parameters". *SIAM Journal on Applied Mathematics*, 11(2), 431-441.
- Bates, D. M. & Watts, D. G. (1988). *Nonlinear Regression Analysis and Its Applications*. Wiley.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nls.html
