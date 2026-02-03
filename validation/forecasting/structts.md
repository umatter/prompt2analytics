# Validation: StructTS (Structural Time Series)

## Method Overview

StructTS fits structural time series models by maximum likelihood estimation using the Kalman filter. The models decompose a time series into unobserved components:

- **Local Level** (`level`): y_t = μ_t + ε_t, μ_{t+1} = μ_t + η_t
- **Local Linear Trend** (`trend`): y_t = μ_t + ε_t, μ_{t+1} = μ_t + β_t + η_t, β_{t+1} = β_t + ζ_t
- **Basic Structural Model** (`BSM`): y_t = μ_t + γ_t + ε_t (adds stochastic seasonality)

Key parameters:
- `model_type`: Type of structural model (Level, Trend, or BSM)
- `frequency`: Seasonal period for BSM (required for BSM)
- `fixed`: Optional vector of fixed variance parameters

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `StructTS()` | R 4.3.2 |

## Test Cases

### Test 1: Nile River Flow - Local Level Model

This classic dataset is often used to demonstrate local level models.

**R Code**:
```r
# Nile River annual flow data (1871-1970)
data(Nile)

# Fit local level model
fit <- StructTS(Nile, type = "level")
print(fit)

# Expected output:
# Variances:
#   level    epsilon
#  1469.072 15099.803
```

**Results Comparison**:

| Parameter | R (StructTS) | Rust (p2a) | Tolerance |
|-----------|--------------|------------|-----------|
| level variance | 1469.07 | ~1469 | 10.0 |
| epsilon variance | 15099.80 | ~15100 | 100.0 |
| log-likelihood | ~-632.5 | ~-632.5 | 1.0 |

**Rust Test**: `crates/p2a-core/src/forecasting/structts.rs::tests::test_struct_ts_local_level`

### Test 2: Local Linear Trend Model

**R Code**:
```r
set.seed(42)
n <- 100
# Generate trend + noise
level <- cumsum(0.5 + rnorm(n, 0, 0.3))
y <- ts(level + rnorm(n, 0, 2))

fit <- StructTS(y, type = "trend")
print(fit)
```

**Results Comparison**:

| Parameter | R (StructTS) | Rust (p2a) | Tolerance |
|-----------|--------------|------------|-----------|
| level variance | varies | varies | 20% |
| slope variance | varies | varies | 20% |
| epsilon variance | varies | varies | 20% |

**Rust Test**: `crates/p2a-core/src/forecasting/structts.rs::tests::test_struct_ts_local_trend`

### Test 3: Basic Structural Model (BSM)

**R Code**:
```r
# UKgas dataset - quarterly UK gas consumption
data(UKgas)

fit <- StructTS(UKgas, type = "BSM")
print(fit)

# Components
tsdiag(fit)
```

**Results Comparison**:

| Parameter | R (StructTS) | Rust (p2a) | Tolerance |
|-----------|--------------|------------|-----------|
| level variance | varies | varies | 20% |
| slope variance | varies | varies | 20% |
| seas variance | varies | varies | 20% |
| epsilon variance | varies | varies | 20% |

**Rust Test**: `crates/p2a-core/src/forecasting/structts.rs::tests::test_struct_ts_bsm`

## Numerical Precision Summary

- Variance parameters: Within 20% of R results (MLE can have multiple local optima)
- Log-likelihood: Within 1% of R results
- Filtered states: Within 5% of R results
- Smoothed states: Within 5% of R results

## Known Differences

1. **Optimization Algorithm**: R uses `optim()` with BFGS; Rust uses Nelder-Mead with log-transformed parameters.

2. **Default Initialization**: R uses diffuse initialization with very large variance; Rust uses a configurable approach.

3. **Convergence Criteria**: Slight differences in convergence thresholds can lead to slightly different parameter estimates.

## Performance Comparison

**Note**: R's StructTS uses highly optimized C/Fortran code from the stats package, including BFGS optimization from the `optim()` function. The Rust implementation uses pure Rust with Nelder-Mead optimization on log-transformed variance parameters.

### Local Level Model

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=50 | 4.4 | 2.5 | 0.57x |
| n=100 | 9.6 | 1.7 | 0.18x |
| n=200 | 19.0 | 1.9 | 0.10x |
| n=500 | 46.7 | 4.7 | 0.10x |

### Local Linear Trend Model

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=50 | 11.8 | 3.0 | 0.25x |
| n=100 | 29.8 | 4.3 | 0.14x |
| n=200 | 61.7 | 4.4 | 0.07x |
| n=500 | 167.0 | 6.8 | 0.04x |

### Basic Structural Model (BSM)

| Dataset Size | Rust (ms) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| n=48 | 78.0 | 76.6 | 0.98x |
| n=96 | 154.1 | 283.6 | 1.84x |
| n=144 | 231.4 | 146.3 | 0.63x |
| n=240 | 368.7 | 443.8 | 1.20x |

### Analysis

R's StructTS is faster for simple models (Level, Trend) but Rust is competitive for the BSM model.

**Why R is faster for simple models:**

1. **BFGS vs Nelder-Mead**: R's BFGS uses gradient information for faster convergence, while Nelder-Mead is derivative-free and requires more function evaluations.

2. **C Implementation**: R's Kalman filter and optimizer are implemented in highly optimized C/Fortran code with BLAS routines.

3. **PORT Library**: R uses the L-BFGS-B algorithm from the PORT library, which is highly tuned for this type of problem.

**Why Rust is competitive for BSM:**

For BSM (4 parameters), the optimization overhead is a smaller fraction of total time. The per-iteration cost of the Kalman filter dominates, and Rust's Kalman filter is reasonably efficient.

### Optimization History

| Iteration | Algorithm | Level n=100 | Improvement |
|-----------|-----------|-------------|-------------|
| Initial | Coordinate descent | 56.2 ms | baseline |
| v2 | Nelder-Mead | 12.0 ms | 4.7x |
| v3 | NM + log-transform | 9.6 ms | 5.9x |

**Future Optimizations**:
- Add the `argmin` crate with L-BFGS and analytical gradients
- Use BLAS-optimized Kalman filter operations
- Pre-compute partial derivatives for faster gradient estimation

## Kalman Filter Performance (Stand-alone)

The Kalman filter alone (without MLE optimization) is fast:

| Dataset Size | Rust Filter (µs) | Rust Smoother (µs) | Rust Forecast (µs) |
|--------------|------------------|--------------------|--------------------|
| n=100 | 147 | 519 | 11 |
| n=500 | 778 | 2,556 | 11 |
| n=1000 | 1,495 | 5,159 | 11 |
| n=5000 | 8,000 | 25,004 | 11 |

The Kalman filter shows linear O(n) scaling as expected. The smoother is about 3-4x slower due to the backward pass. Forecasting is O(h) where h is the horizon, independent of input length.

## References

- Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- R Core Team. `StructTS` documentation. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/StructTS.html
