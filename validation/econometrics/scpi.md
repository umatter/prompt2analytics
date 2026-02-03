# Synthetic Control with Prediction Intervals (SCPI) Validation

## Method Description

SCPI extends the classic synthetic control method (Abadie et al., 2010) with proper uncertainty quantification through prediction intervals. Developed by Cattaneo, Feng, and Titiunik (2021), SCPI provides:

1. **Multiple constraint types**: Simplex (classic SC), Lasso, Ridge, Lasso-Simplex
2. **Prediction intervals**: Account for both in-sample and out-of-sample variance
3. **Variance decomposition**: Separate estimation of in-sample and out-of-sample components

## Mathematical Framework

### Weight Estimation

The synthetic control weights W are estimated by minimizing:

```
min_W ||Y1_pre - Y0_pre * W||^2 + penalty(W)
```

Subject to constraints depending on type:
- **Simplex**: sum(W) = 1, W >= 0
- **Lasso**: add lambda * ||W||_1
- **Ridge**: add lambda * ||W||_2^2
- **Lasso-Simplex**: Lasso + simplex

### Prediction Intervals

Intervals are constructed as (Cattaneo et al. 2021, Eq. 3.5):

```
[effect - c_alpha * sigma_hat, effect + c_alpha * sigma_hat]
```

Where:
- sigma_hat^2 = sigma_in^2 + sigma_out^2
- sigma_in^2: In-sample variance from pre-treatment fit residuals
- sigma_out^2: Out-of-sample variance (estimated via subgaussian bounds or CV)
- c_alpha: Critical value (t or z distribution)

## Implementation Location

- **Core**: `crates/p2a-core/src/econometrics/scpi.rs`
- **MCP Tool**: `scpi` in `crates/p2a-mcp/src/server.rs`
- **Exports**: `p2a_core::{run_scpi, SCPIConfig, SCPIConstraint, VarianceMethod, SCPIResult}`

## Reference Implementation

### R Package: scpi (Cattaneo, Feng, Palomba, Titiunik)

- **CRAN**: https://cran.r-project.org/package=scpi
- **Documentation**: https://nppackages.github.io/scpi/
- **Version**: 2.2.5 (at time of validation)

## Test Cases

### Test Case 1: Basic Simplex Constraint

**Setup**:
- 10 time periods (7 pre-treatment, 3 post-treatment)
- 1 treated unit with trend + treatment effect
- 5 donor units with similar trends

**Rust Test**: `test_scpi_basic`

```rust
let treated = array![
    10.0, 11.5, 12.8, 14.1, 15.5, 16.9, 18.2,  // Pre-treatment
    22.0, 23.5, 25.0                            // Post-treatment (effect ~3)
];
let donors = Array2::from_shape_fn((10, 5), |(t, j)| {
    10.0 + 1.4 * t as f64 + 0.3 * j as f64 + 0.1 * ((t * j) as f64).sin()
});

let config = SCPIConfig::default();  // Simplex, alpha=0.05
let result = run_scpi(&treated.view(), &donors.view(), 7, config)?;
```

**Validation Criteria**:
- Weights sum to 1.0 (tolerance: 1e-6)
- All weights non-negative
- Prediction intervals properly ordered: lower < effect < upper
- Pre-treatment fit: RMSPE > 0

### Test Case 2: Lasso Constraint

**Rust Test**: `test_scpi_lasso`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::Lasso { lambda: 0.1 },
    ..Default::default()
};
```

**Validation Criteria**:
- Produces sparse weights (n_effective_donors <= n_donors)
- Some weights may be exactly zero

### Test Case 3: Ridge Constraint

**Rust Test**: `test_scpi_ridge`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::Ridge { lambda: 0.1 },
    ..Default::default()
};
```

**Validation Criteria**:
- All weights typically non-zero (shrinkage, not selection)
- Smaller magnitude weights than unconstrained OLS

### Test Case 4: Lasso-Simplex Constraint

**Rust Test**: `test_scpi_lasso_simplex`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::LassoSimplex { lambda: 0.05 },
    ..Default::default()
};
```

**Validation Criteria**:
- Weights sum to 1.0 (tolerance: 1e-4)
- All weights non-negative
- Some sparsity induced by L1 penalty

### Test Case 5: Variance Methods

**Rust Test**: `test_variance_methods`

Tests Gaussian and Subgaussian variance estimation:

**Validation Criteria**:
- All variance components > 0
- total_var = in_sample_var + out_sample_var

## Algorithm Details

### Simplex Weight Optimization

Uses Frank-Wolfe algorithm:
1. Initialize with uniform weights: W = 1/J
2. Compute gradient: grad = H * W + c where H = X0'X0, c = -X0'Y1
3. Find minimizing vertex: argmin_j grad_j
4. Line search for optimal step size
5. Update: W_new = W + alpha * (e_j - W)
6. Repeat until convergence

### Lasso Weight Optimization

Uses coordinate descent with soft-thresholding:
1. For each coordinate j:
   - Compute partial residual
   - Apply soft-thresholding: W_j = sign(r_j) * max(0, |r_j| - lambda)
2. Repeat until convergence

### ADMM for Lasso-Simplex

Uses Alternating Direction Method of Multipliers:
1. W-update: Minimize augmented Lagrangian (closed form)
2. Z-update: Apply soft-thresholding then project to simplex
3. U-update: Dual variable update
4. Repeat until primal and dual residuals converge

### Simplex Projection

Uses Duchi et al. (2008) algorithm:
1. Sort vector in descending order
2. Find threshold rho via cumulative sum
3. Project: x_proj = max(0, x - theta)

## Tolerances

| Quantity | Tolerance |
|----------|-----------|
| Weight sum (simplex) | 1e-4 |
| Weight non-negativity | 1e-6 |
| Optimization convergence | 1e-8 |
| Variance positivity | > 0 |

## Known Limitations

1. **Critical value approximation**: Uses simplified t-distribution approximation; exact values require special functions
2. **Small sample variance**: Out-of-sample variance estimation less reliable with few pre-treatment periods
3. **Lasso path**: Does not compute full regularization path; single lambda value

## References

### Primary
- Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic Control Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880. DOI: 10.1080/01621459.2021.1979561

### Background
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505.

- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.

### Algorithms
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." *ICML 2008*.

### Software
- R package `scpi`: Cattaneo, M. D., Feng, Y., Palomba, F., & Titiunik, R. https://cran.r-project.org/package=scpi
