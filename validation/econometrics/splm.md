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
- W must be N x N, and row i of W belongs to the i-th entity in sorted id order: numbers
  ascending, text in byte order (R `factor()` under `LC_COLLATE=C`; an R session in a
  UTF-8 locale sorts text case-insensitively, `a A b B` instead of `A B a b`). Weights
  whose neighbours carry region ids must list the sorted entity labels, in order.
- The panel must be balanced, with each (entity, time) pair exactly once. Otherwise
  `run_spml` and `run_spgm` return an error naming the first missing or duplicated cell;
  `splm::spml` refuses unbalanced panels too.
- Rows may be in any order. The estimates do not depend on it (the data are put in a
  time-major layout with the entities in sorted id order inside each period), and
  residuals and fitted values are returned in the dataset's row order.
- Missing values (null or NaN) in y, X, the entity or the time column are refused
  (`NullValues`, naming the column). Integer and Boolean columns are accepted. Time must
  hold whole numbers or dates; a fractional time is an error naming the row.
- In the MCP tools, build the weights with `spatial_neighbors` from a coordinate dataset
  with one row per entity, sorted by the entity id.

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

## Tests (T2-WP14): panel layout and missing values

These tests are not R-pinned in the registry's sense: splm is not installed, so the
reference values come from a base-R re-implementation of the spml within-lag concentrated
likelihood (`ll(rho) = -NT/2 (1 + log 2 pi + log(RSS(rho)/NT)) + T sum log(1 - rho ev)`,
the form `optimize_rho_panel` maximises), and they are pinned inline in the tests.

- `test_spml_entity_order_r_cross_check` (`econometrics::splm::tests`): the 12-entity,
  5-period panel below, fitted with the rows entity-major (A), shuffled (B, row i = row
  (7 i + 3) mod 60 of A) and shuffled with text labels u1..u12 (C). All three give
  rho = 0.41522977605328182, beta_x = 1.4535946650208815, logLik = 13.520649774963919 and
  sigma2 = RSS/(n - N - k) = 0.044630452161004024. Tolerances: `Tol::iterative(1e-6)` for
  rho, beta and sigma2 (the golden section stops at tol = 1e-8) and `Tol::iterative(1e-9)`
  for logLik. With W rows in first-appearance order (the code before T2-WP14), B gives
  rho = 0.17868516481819932.
- `test_spml_spgm_row_order_invariant`: A and B give identical estimates for the within
  lag, within error, random-effects lag and pooled lag ML models and for three GMM
  configurations; residuals and fitted values follow each dataset's row order.
- `test_spatial_lag_panel_follows_sorted_entity_ids`: three entities b, a, c over two
  periods; W y = 101 10 10 20 202 20 in row order (R: the same).
- `test_spml_unbalanced_and_duplicated_panels_refused`: one cell removed or one row
  repeated is an error naming the cell.
- `test_na_spml_missing_and_invalid_ids`: a missing y, entity or time is `NullValues`; a
  fractional time and a text outcome are refused.
- `test_na_spml_int64_equals_float64`: Int64 y and x give the same results as their
  Float64 copies.
- `test_spml_weights_must_follow_sorted_entities`: an 11 x 11 W for 12 entities is
  refused with the sorted-order message; region ids equal to the sorted labels are
  accepted, reversed ones are refused.
- MCP (`tools::handlers::spatial::tests`): `test_mcp_spatial_panel_ml_sorted_ids_match_r`
  (dataset C through `spatial_neighbors` and `spatial_panel_ml`) and
  `test_mcp_spatial_panel_unbalanced_duplicated_and_w_size_refused`.

R script (R 4.3.3, base R):

```r
options(digits = 17)
invisible(Sys.setlocale("LC_COLLATE", "C"))
N <- 12L; TT <- 5L
k <- 1:N
gx <- (k - 1L) %% 4L; gy <- (k - 1L) %/% 4L                    # 3 x 4 integer grid, entity k
D <- as.matrix(dist(cbind(gx, gy)))
B <- (D > 0 & D <= 1) * 1                                     # rook: Neighbors::from_distance(0, 1)
W <- B / rowSums(B)                                           # WeightStyle::RowStd
stopifnot(max(abs(Im(eigen(W, only.values = TRUE)$values))) < 1e-12)
x <- outer(k, 1:TT, function(kk, tt) sin(0.7 * kk + 1.3 * tt) + 0.1 * kk)
alpha <- 0.5 * cos(1.1 * k)
e <- outer(k, 1:TT, function(kk, tt) 0.3 * sin(2.3 * kk * tt + 0.4))
y <- sapply(1:TT, function(tt) solve(diag(N) - 0.5 * W, 1.5 * x[, tt] + alpha + e[, tt]))
A <- data.frame(entity = rep(k, each = TT), time = rep(2000L + 1:TT, times = N),
                y = as.vector(t(y)), x = as.vector(t(x)))
fit_within_lag <- function(d, Wm) {
  ent <- factor(d$entity)                                     # levels = W rows
  stopifnot(nlevels(ent) == nrow(Wm))
  tt <- factor(d$time)
  n <- nrow(d); TT <- nlevels(tt)
  wy <- numeric(n)
  for (p in levels(tt)) {
    rows <- which(d$time == p); o <- rows[order(as.integer(ent[rows]))]
    wy[o] <- Wm %*% d$y[o]
  }
  dm <- function(v) v - ave(v, ent)
  yw <- dm(d$y); wyw <- dm(wy); xw <- dm(d$x)
  evm <- Re(eigen(Wm, only.values = TRUE)$values)
  rss <- function(rho) { r <- yw - rho * wyw; b <- sum(xw * r) / sum(xw^2); sum((r - b * xw)^2) }
  nll <- function(rho) 0.5 * n * (1 + log(2 * pi) + log(rss(rho) / n)) - TT * sum(log(1 - rho * evm))
  lo <- max(-0.99, 1 / min(evm)); hi <- min(0.99, 1 / max(evm))
  g <- seq(lo, hi, length.out = 2001); g0 <- g[which.min(sapply(g, nll))]
  opt <- optimize(nll, c(max(lo, g0 - 0.01), min(hi, g0 + 0.01)), tol = 1e-15)
  rho <- opt$minimum
  r <- yw - rho * wyw; beta <- sum(xw * r) / sum(xw^2)
  rs <- sum((r - beta * xw)^2)
  list(rho = rho, beta = beta, ll = -nll(rho), sigma2_df = rs / (n - nlevels(ent) - 1), interval = c(lo, hi))
}
ref <- fit_within_lag(A, W)
perm <- (7L * (0:59) + 3L) %% 60L
Bd <- A[perm + 1L, ]
first <- unique(Bd$entity)
old <- fit_within_lag(transform(Bd, entity = factor(entity, levels = first)), W)
newB <- fit_within_lag(Bd, W)
labs <- levels(factor(paste0("u", 1:12)))
Cd <- Bd; Cd$entity <- labs[Bd$entity]
newC <- fit_within_lag(Cd, W)
lab3 <- c("b", "a", "c", "a", "b", "c"); t3 <- c(1, 1, 1, 2, 2, 2); y3 <- c(10, 1, 100, 2, 20, 200)
W3 <- matrix(c(0, 1, 0,  1, 0, 1,  0, 1, 0), 3, byrow = TRUE)
wy3 <- numeric(6)
for (p in 1:2) { r <- which(t3 == p); o <- r[order(lab3[r])]; wy3[o] <- W3 %*% y3[o] }
```

`A` begins `1 2001 2.313251294567376881 1.009297426825681798`; the Rust fixture solves
the same system by 200 Neumann iterations and reproduces it to about 1e-15.

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
