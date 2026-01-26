# Spatial Panel Data Models (splm) Validation

## Method Overview

The `splm` module provides ML and GMM estimation for spatial panel data models, combining panel data methods (fixed effects, random effects) with spatial dependence structures (spatial lag, spatial error).

### Implemented Functions

1. **`run_spml()`** - Spatial panel ML estimation
   - Spatial lag panel: y = rho*W*y + X*beta + alpha + epsilon
   - Spatial error panel: y = X*beta + alpha + u, u = lambda*W*u + epsilon
   - Fixed effects (within transformation)
   - Random effects (GLS)
   - Pooled estimation

2. **`run_spgm()`** - Spatial panel GMM estimation
   - W2SLS (within/fixed effects)
   - G2SLS (GLS random effects)
   - B2SLS (between effects)
   - EC2SLS (Baltagi's error components)

## Reference Implementation

R package: `splm` version 1.6-5
- CRAN: https://cran.r-project.org/package=splm
- Functions: `spml()`, `spgm()`, `spreml()`

### Key References

- Baltagi, B.H., Song, S.H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150.
- Kapoor, M., Kelejian, H.H., & Prucha, I.R. (2007). Panel data models with spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130.
- Millo, G., & Piras, G. (2012). splm: Spatial Panel Data Models in R. *Journal of Statistical Software*, 47(1), 1-38.

## Test Cases

### Test Case 1: Fixed Effects with Spatial Lag

**Setup:**
- 4 entities (2x2 spatial grid)
- 3 time periods
- Spatial lag model with within transformation

**R Code:**
```r
library(splm)
library(spdep)

# Create panel data
set.seed(42)
n_ent <- 4
n_t <- 3
data <- expand.grid(entity = 1:n_ent, time = 1:n_t)
data$x1 <- runif(nrow(data))
data$x2 <- runif(nrow(data))
data$y <- 1 + 0.5 * data$x1 + 0.3 * data$x2 +
          as.numeric(data$entity) * 0.2 + rnorm(nrow(data), sd = 0.1)

# Create spatial weights (4 locations in 2x2 grid)
coords <- matrix(c(0,0, 1,0, 0,1, 1,1), ncol=2, byrow=TRUE)
nb <- knn2nb(knearneigh(coords, k=2))
listw <- nb2listw(nb, style="W")

# Spatial lag model with fixed effects
model <- spml(y ~ x1 + x2, data = data,
              index = c("entity", "time"),
              listw = listw,
              model = "within",
              lag = TRUE,
              spatial.error = "none")
summary(model)
```

**Rust Test:**
```rust
#[test]
fn test_spml_within_spatial_lag() {
    let (dataset, mut listw) = create_spatial_panel_data();

    let config = SpmlConfig {
        model: SpatialPanelModel::Within,
        lag: true,
        spatial_error: SpatialErrorType::None,
        ..Default::default()
    };

    let result = run_spml(
        &dataset, "y", &["x1", "x2"],
        "entity", "time", &mut listw, config
    ).unwrap();

    assert!(result.has_lag);
    assert!(result.rho.is_some());
    let rho = result.rho.unwrap();
    assert!(rho > -1.0 && rho < 1.0);
}
```

### Test Case 2: Random Effects Model

**Setup:**
- 6 entities (2x3 spatial grid)
- 4 time periods
- Random effects GLS estimation

**R Code:**
```r
# Random effects spatial panel model
model_re <- spml(y ~ x1 + x2, data = data,
                 index = c("entity", "time"),
                 listw = listw,
                 model = "random",
                 lag = FALSE,
                 spatial.error = "none")
summary(model_re)
```

**Rust Test:**
```rust
#[test]
fn test_spml_random_effects() {
    let (dataset, mut listw) = create_larger_panel_data();

    let config = SpmlConfig {
        model: SpatialPanelModel::Random,
        lag: false,
        spatial_error: SpatialErrorType::None,
        ..Default::default()
    };

    let result = run_spml(
        &dataset, "y", &["x1"],
        "entity", "time", &mut listw, config
    ).unwrap();

    assert_eq!(result.model, SpatialPanelModel::Random);
    assert!(result.variance.sigma_mu.is_some());
}
```

### Test Case 3: GMM Estimation (W2SLS)

**R Code:**
```r
library(splm)

# GMM estimation with fixed effects
model_gmm <- spgm(y ~ x1 + x2, data = data,
                  index = c("entity", "time"),
                  listw = listw,
                  method = "w2sls",
                  lag = FALSE,
                  spatial.error = TRUE)
summary(model_gmm)
```

**Rust Test:**
```rust
#[test]
fn test_spgm_basic() {
    let (dataset, mut listw) = create_spatial_panel_data();

    let config = SpgmConfig {
        method: SpgmMethod::W2sls,
        lag: false,
        spatial_error: true,
        ..Default::default()
    };

    let result = run_spgm(
        &dataset, "y", &["x1", "x2"],
        "entity", "time", &mut listw, config
    ).unwrap();

    assert_eq!(result.method, SpgmMethod::W2sls);
}
```

## Validation Results

### Coefficients Comparison

| Parameter | R (splm) | Rust | Tolerance | Status |
|-----------|----------|------|-----------|--------|
| beta_1 | - | - | 1e-4 | Pending |
| beta_2 | - | - | 1e-4 | Pending |
| rho | - | - | 1e-4 | Pending |
| sigma2 | - | - | 1e-4 | Pending |

**Note:** Full numerical validation requires running R comparison scripts with identical data.

## Implementation Notes

### Panel Structure
- Spatial weights matrix W must be N x N where N is the number of cross-sectional units
- Panel data is assumed to be sorted by entity-time
- Time dimension T can vary across entities (unbalanced panels supported)

### Estimation Methods
1. **Fixed Effects (Within):** Removes entity-specific means, estimates by OLS/ML on demeaned data
2. **Random Effects:** Uses GLS with quasi-demeaning based on variance components
3. **Pooled:** Standard spatial models ignoring panel structure

### Spatial Lag Estimation
- Uses concentrated likelihood approach
- Optimizes rho using golden section search over valid range
- Valid rho range determined by eigenvalues of W: (1/lambda_min, 1/lambda_max)

### Log-Determinant Computation
- log|I - rho*W| = sum(log(1 - rho * lambda_i))
- For panel: T * sum(log(1 - rho * lambda_i))

## Known Limitations

1. Two-way effects (individual + time) not fully implemented
2. Serial correlation components (spreml) not yet available
3. Spatial Durbin specification pending

## MCP Tool Exposure

Two tools exposed:
- `spatial_panel_ml` - ML estimation via `run_spml()`
- `spatial_panel_gmm` - GMM estimation via `run_spgm()`

## References

1. Baltagi, B.H., Song, S.H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150. https://doi.org/10.1016/S0304-4076(03)00120-9

2. Kapoor, M., Kelejian, H.H., & Prucha, I.R. (2007). Panel data models with spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130. https://doi.org/10.1016/j.jeconom.2006.09.004

3. Millo, G., & Piras, G. (2012). splm: Spatial Panel Data Models in R. *Journal of Statistical Software*, 47(1), 1-38. https://www.jstatsoft.org/v47/i01/

4. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
