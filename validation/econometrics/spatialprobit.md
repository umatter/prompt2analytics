# Spatial Probit Models Validation

## Method Overview

The `spatialprobit` module implements Bayesian spatial probit models for binary dependent variables with spatial dependence. These models are equivalent to R's `spatialprobit` package (Wilhelm & de Matos, 2013).

### Models Implemented

1. **SAR Probit (Spatial Autoregressive Probit)**
   - Model: y* = rho * W * y* + X * beta + epsilon, epsilon ~ N(0, I)
   - y = 1 if y* > 0, else y = 0
   - Captures spatial spillover effects in binary outcomes

2. **SEM Probit (Spatial Error Probit)**
   - Model: y* = X * beta + u, u = lambda * W * u + epsilon, epsilon ~ N(0, I)
   - y = 1 if y* > 0, else y = 0
   - Accounts for spatial correlation in errors (nuisance dependence)

## Estimation Method

Uses Bayesian MCMC with data augmentation following LeSage & Pace (2009):

1. **Data Augmentation**: Albert & Chib (1993) method for latent y*
2. **Sampling Steps**:
   - Sample y* from truncated normal given (beta, rho) and observed y
   - Sample beta from conjugate multivariate normal posterior
   - Sample rho using Griddy Gibbs (Ritter & Tanner, 1992)
3. **Posterior Inference**: Means, standard deviations, and credible intervals from MCMC draws

## Reference Implementation

### R Code (spatialprobit package)
```r
library(spatialprobit)
library(spdep)

# Create spatial weights
coords <- cbind(lon = runif(100), lat = runif(100))
nb <- knn2nb(knearneigh(coords, k = 5))
listw <- nb2listw(nb, style = "W")
W <- listw2mat(listw)

# Generate data with spatial structure
set.seed(12345)
n <- 100
X <- cbind(1, rnorm(n))
beta_true <- c(0.5, 0.5)
rho_true <- 0.3

# y* = rho*W*y* + X*beta + epsilon
A <- diag(n) - rho_true * W
epsilon <- rnorm(n)
y_star <- solve(A) %*% (X %*% beta_true + epsilon)
y <- as.numeric(y_star > 0)

# Estimate SAR probit
result <- sarprobit(y ~ X[,2], W = W, ndraw = 1000, burn.in = 200)
summary(result)

# Estimate SEM probit
result_sem <- semprobit(y ~ X[,2], W = W, ndraw = 1000, burn.in = 200)
summary(result_sem)
```

## Test Cases

### Test Case 1: SAR Probit with Simulated Data

**Setup:**
- n = 25 observations (5x5 grid)
- k = 4 nearest neighbors, row-standardized weights
- True model: P(y=1) = Phi(0.5 + 0.5*x + spatial_effect)
- Seed: 12345

**Rust Implementation:**
```rust
use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle};
use p2a_core::econometrics::spatialprobit::{run_sar_probit, SpatialProbitConfig};

let config = SpatialProbitConfig {
    n_draws: 1000,
    burn_in: 200,
    seed: Some(12345),
    ..Default::default()
};

let result = run_sar_probit(&dataset, "y", &["x"], &mut listw, config)?;
```

**Expected Results:**
| Parameter | Estimate Range | Notes |
|-----------|---------------|-------|
| rho | -0.5 to 0.5 | Should capture spatial dependence |
| Intercept | 0.0 to 1.0 | Varies with data |
| x coefficient | 0.3 to 0.7 | Around true value 0.5 |

### Test Case 2: SEM Probit Validation

**Setup:**
- Same data as Test Case 1
- SEM model specification

**Expected Behavior:**
- Lambda should capture spatial error correlation
- Coefficient estimates similar to non-spatial probit when lambda is near 0

### Test Case 3: Marginal Effects (Spatial Impacts)

For SAR probit, marginal effects are:
- **Direct effect**: Average diagonal of S_k * phi(A^{-1}*X*beta)
- **Indirect effect**: Average off-diagonal of S_k * phi(A^{-1}*X*beta)
- **Total effect**: Direct + Indirect

where S_k = A^{-1} * beta_k, A = I - rho*W

**Validation:**
- Total effect > Direct effect when rho > 0 (positive spillovers)
- Direct + Indirect = Total (additive decomposition)

## Comparison with R Results

### Tolerance Guidelines

| Statistic | Tolerance | Notes |
|-----------|-----------|-------|
| Posterior means | 0.1 | MCMC sampling variation |
| Posterior SDs | 0.05 | Depends on draws |
| Spatial parameter | 0.1 | Inherent MCMC variation |
| Marginal effects | 0.05 | Depends on draws and approximation |

Note: Spatial probit models use MCMC estimation, so exact replication requires matching:
- Random seed
- Number of draws
- Burn-in period
- Same data and weights matrix

### Differences from R spatialprobit

1. **Approximation method**: Our implementation uses Neumann series approximation for (I - rho*W)^{-1} which may differ slightly from direct matrix inversion
2. **Truncated normal sampling**: Uses inverse CDF method
3. **Griddy Gibbs**: Uses 50-point grid for rho sampling

## MCMC Diagnostics

The result includes:
- `acceptance_rate`: Should be between 0.2-0.8 for good mixing
- `beta_draws`: Full chain for convergence diagnostics
- `rho_draws`: Full chain for spatial parameter
- `credible_interval_*`: 95% credible intervals

### Convergence Checks

1. **Trace plots**: Examine beta_draws and rho_draws
2. **Autocorrelation**: Should decay quickly
3. **Effective sample size**: Should be substantial fraction of n_draws

## Implementation Details

### File Location
`crates/p2a-core/src/econometrics/spatialprobit.rs`

### Key Functions
- `run_sar_probit()`: SAR probit estimation
- `run_sem_probit()`: SEM probit estimation
- `sample_latent_y()`: Truncated normal sampling for y*
- `sample_beta()`: Conjugate normal posterior for beta
- `sample_rho_griddy_gibbs()`: Griddy Gibbs for spatial parameter
- `compute_spatial_probit_impacts()`: Direct/indirect/total effects

### Dependencies
- `ndarray`: Matrix operations
- `rand`, `rand_distr`: Random number generation
- `statrs`: Normal distribution functions

## References

### Primary Sources

1. LeSage, J.P. & Pace, R.K. (2009). "Introduction to Spatial Econometrics". CRC Press, Chapter 10.
   ISBN: 978-1420064247.

2. Albert, J.H. & Chib, S. (1993). "Bayesian analysis of binary and polychotomous response data".
   *Journal of the American Statistical Association*, 88(422), 669-679.
   https://doi.org/10.1080/01621459.1993.10476321

3. Geweke, J. (1991). "Efficient simulation from the multivariate normal and Student-t distributions
   subject to linear constraints". *Computing Science and Statistics: Proc. 23rd Symposium on the
   Interface*, 571-578.

4. Beron, K.J. & Vijverberg, W.P.M. (2004). "Probit in a Spatial Context: A Monte Carlo Analysis".
   In Anselin, L., Florax, R.J.G.M. & Rey, S.J. (Eds.), *Advances in Spatial Econometrics*
   (pp. 169-195). Springer.

### R Package Reference

5. Wilhelm, S. & de Matos, M.G. (2013). "Estimating Spatial Probit Models in R".
   *The R Journal*, 5(1), 130-143.
   https://cran.r-project.org/package=spatialprobit

### Related Implementations

- R package `spatialprobit`: https://cran.r-project.org/package=spatialprobit
- R package `spdep`: Spatial dependence and weights
- R package `spatialreg`: Spatial regression models (continuous outcomes)

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| test_sar_probit_basic | PASS | Basic SAR estimation works |
| test_sem_probit_basic | PASS | Basic SEM estimation works |
| test_spatial_probit_impacts | PASS | Direct/indirect/total computed correctly |
| test_credible_intervals | PASS | 95% CI contains posterior mean |
| test_truncated_normal_sampling | PASS | Samples correctly constrained |
| test_inverse_normal_cdf | PASS | Quantile function accurate |

## MCP Tool Integration

Tools exposed:
- `sar_probit_model`: SAR probit for binary spatial outcomes
- `sem_probit_model`: SEM probit for binary outcomes with spatial error

Both tools require spatial weights to be created first using `spatial_neighbors`.
