# Spatial Panel Data Models (splm) Validation

## Method Overview

`p2a_core::econometrics::splm` estimates spatial panel models on a balanced panel of N
entities over T periods, stacked by period (`W_NT = I_T (x) W`):

1. **`run_spml()`**, maximum likelihood (splm's `spml`):
   - within models with individual, time or two-way effects (y and X demeaned, then W
     applied to the demeaned data) and pooled models, each with a spatial lag
     (`y = rho W y + X b + u`), a spatial error (`u = lambda W u + e`) or both (SARAR,
     individual effects and pooled);
   - individual random effects, `u = iota_T (x) mu + eps`, `phi = sigma_mu^2 / sigma^2`,
     without spatial error, with Baltagi, Song and Koh's error (`b`: mu not spatially
     correlated) or with Kapoor, Kelejian and Prucha's (`kkp`: mu spatially correlated),
     each with or without a spatial lag.

   Within and pooled likelihood, with `A = I - rho W`, `B = I - lambda W`,
   `e = B_NT (A_NT y - X b)`, b and `s2 = e'e / (NT)` concentrated out:
   `l = -NT/2 log(2 pi s2) + T log|A| + T log|B| - e'e / (2 s2)`. Random effects, with
   `c = T phi`, `K = B'B` (`K = I` without spatial error):
   `Omega / s2 = Jbar_T (x) M^-1 + E_T (x) K^-1` with `M = (c I + K^-1)^-1` (b), `K / (c + 1)`
   (kkp) or `I / (c + 1)` (none), and
   `l = -NT/2 log(2 pi s2) + T log|A| - ldet / 2 - (T ubar'M ubar + sum_t (u_t - ubar)'K(u_t - ubar)) / (2 s2)`,
   `ldet = log|cK + I| - T log|K|` (b), `N log(c + 1) - T log|K|` (kkp), `N log(c + 1)` (none).

2. **`run_spgm()`**, GM and 2SLS (splm's `spgm`):
   - the spatial-lag model by within 2SLS (`w2sls`), between 2SLS on the entity means
     (`b2sls`), Baltagi's EC2SLS (`ec2sls`) or GLS 2SLS (`g2sls`), instruments
     `[X, WX, W^2X]`;
   - the within spatial-error model by the KKP within GM moments and within FGLS
     (`w2sls`);
   - the within lag-plus-error model (GS2SLS) by w2sls, the KKP within GM on its residuals
     and 2SLS of the spatially filtered model (`w2sls`).

**Naming.** p2a follows spatialreg: `rho` is the spatial-lag coefficient and `lambda` the
spatial-error coefficient. splm's own output uses lambda for the lag and rho for the
error. Within and pooled models give the same estimate for `b` and `kkp`, because mu is
swept out or absent.

## Reference

splm is not installed, so the references are the estimators written out from the published
likelihoods and moment conditions (Elhorst 2003; Baltagi, Song and Koh 2003; Kapoor,
Kelejian and Prucha 2007; Millo and Piras 2012; Baltagi 1981; Kelejian and Prucha 1998,
1999) in `validation/r/cases/spatial_splm.R`, case `splm_spml_spgm`
(`validation/expected/splm_spml_spgm.json`, data
`validation/datasets/generated/splm_sim.csv`: N = 49, T = 6, W = 4 nearest neighbours,
row-standardised, with 10 complex eigenvalues). R packages anchor the non-spatial special
cases: `plm::plm(model = "within", effect = ...)` 2.6.7 for the within coefficients and
`nlme::lme(method = "ML")` 3.1.164 for the random-effects fit; numDeriv 2016.8.1.1 for the
Jacobian of the analytic score. R 4.3.3.

## Estimation

- **Profile Newton.** The concentrated log-likelihood is maximised over
  psi = (rho, log phi, lambda) (those in the model) by damped Newton from rho = lambda = 0,
  phi = 1, on a central-difference Hessian of the profile (step 1e-5). rho and lambda stay
  inside the interval the complex spectrum of W allows; log phi stays in [-30, 30].
  Convergence: max |change| / max(1, |psi|) <= `tol` (default 1e-8), at most `max_iter`
  iterations. log|I - r W| = Re sum ln(1 - r w_i) over the complex eigenvalues, which
  are computed once per weights (the last four spectra are kept, matched on the full
  weight lists).
- **phi at its boundary.** When log phi reaches -30, phi is reported as 0 (sigma_mu = 0),
  the fit equals the pooled model, phi has no standard error (NaN) and a warning says so.
- **Standard errors** are the inverse of minus the observed information of the full
  log-likelihood, computed as the central-difference Jacobian of the analytic score.
  splm itself may report expected-information or GLS-type SEs for some models; p2a's are
  observed-information SEs, as pinned. sigma2 is the ML estimate e'e / (NT).
- **2SLS** (`spgm`): one QR of the instruments H per fit. With Q1 its first rank(H)
  columns and C = Q1'Z, P_H Z = Q1 C, so the coefficients and (P'P)^-1 come from least
  squares of Q1'y on the small matrix C; P_H Z is never formed. w2sls: Z = [X~, W y~],
  H = [X~, W X~, W^2 X~], sigma2 = u'u / (N (T - 1)), SEs with u'u / (N (T - 1) - p),
  Sargan |P_H u|^2 / sigma2. b2sls: Z = [1, Xbar, W ybar], H = [1, Xbar, W Xbar, W^2 Xbar]
  on the N entity means, sigma2 = u'u / N, sigma_1^2 = T u'u / N. ec2sls and g2sls: 2SLS of
  om(y) on om(Z), om(v) = Q0 v / sigma_nu + P0 v / sigma_1, Z = [X, W y], with instruments
  [Q0 H, P0 H] (ec2sls) or om(H) (g2sls), sigma_nu^2 from w2sls and sigma_1^2 from b2sls;
  SEs sqrt(diag((P'P)^-1)), sigma_mu^2 = (sigma_1^2 - sigma_nu^2) / T (a warning when
  negative).
- **Within GM** (spatial error): the moments of Kelejian and Prucha (1999) on within
  residuals with n = N (T - 1) and trace term (T - 1) tr(W'W). sigma^2 enters linearly and
  is concentrated out, so lambda is the root in (-1, 1) of a cubic first-order condition
  with the smallest objective, found exactly by safeguarded Newton-bisection on each
  monotone piece (`tol`, `max_iter`). Then within FGLS of (I - lambda W) y~ on
  (I - lambda W) X~.
- **Within GS2SLS** (lag and error): w2sls, the within GM on its residuals, then 2SLS of
  (I - lambda W) y~ on (I - lambda W) Z~ with the original instruments H (Kelejian and
  Prucha 1998); sigma^2 is the GM estimate, lambda has no standard error. Not compared with
  splm (not installed).

## Supported combinations

| Function | model / method | effect | lag | error | Notes |
|---|---|---|---|---|---|
| `run_spml` | within | individual | yes / no | none, b, kkp (identical) | SARAR with both |
| `run_spml` | within | time, twoways | yes / no | none, b, kkp | |
| `run_spml` | pooling | individual only (E1) | yes / no | none, b, kkp | |
| `run_spml` | random | individual only (E2) | yes / no | none, b, kkp | |
| `run_spgm` | w2sls | individual (within) | yes | no | |
| `run_spgm` | w2sls | individual (within) | no | yes | within GM + FGLS |
| `run_spgm` | w2sls | individual (within) | yes | yes | GS2SLS |
| `run_spgm` | b2sls, ec2sls, g2sls | - | yes | no (E5 otherwise) | |

`run_spgm` accepts only `moments = initial`. Errors (`InvalidSpecification` unless
stated):

- **E1:** "model = 'pooling' estimates no individual or time effects, so effect = '...' does
  not apply; use model = 'within' with effect = '...', or leave effect at 'individual'"
- **E2:** "random effects are individual effects (as splm::spml); effect = '...' is not
  available with model = 'random'; use model = 'within' for time or two-way fixed effects"
- **E3:** "spgm estimates a spatial lag (lag = true) or a spatial error (spatial_error =
  true); with neither there is no spatial parameter, so use a non-spatial panel estimator"
- **E5:** "the spgm spatial error model is implemented for method = 'w2sls' (the within GM
  moments of Kapoor, Kelejian and Prucha 2007, then within FGLS); method = '...' with
  spatial_error = true needs the random-effects GM, which is not implemented"
- **E6:** "moments = '...' weights the six random-effects GM moments of Kapoor, Kelejian and
  Prucha (2007), which are not implemented; the within GM uses the three within moments
  unweighted (moments = 'initial')"
- **E7:** "the regressors fit y exactly after the ... (residual sum of squares ... against
  ...); the Gaussian likelihood is unbounded, so the spatial panel model cannot be
  estimated"
- Too-small panels are `InsufficientData` (T < 2 for within, random effects and spgm;
  N < 2 for period demeaning; no residual degrees of freedom). Aliased regressors are
  `PerfectMulticollinearity`; too few instruments `UnderIdentified`; a GM without a
  stationary point in (-1, 1) or with a non-positive sigma^2 `Computation`.

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

## Tests

R-pinned (registry `validation/methods/spatial_splm.json`), all on case `splm_spml_spgm`,
in `econometrics::splm::tests`:

| Test | Keys | Tolerance |
|---|---|---|
| `test_validate_spml_individual_lag`, `_individual_error`, `_individual_sarar`, `_time_lag`, `_time_error`, `_twoways_lag`, `_twoways_error`, `_pooling_lag`, `_pooling_error` | `<tag>_coef`, `_rho`, `_lambda`, `_s2`, `_se`, `_loglik` | parameters `Tol::iterative(1e-8)`, SEs `Tol::iterative(1e-6)`, logLik 1e-10 |
| `test_validate_spml_within_none_plm` | `individual_none_*`, plm within coefficients | the same; plm coef 1e-8 |
| `test_validate_spml_random_baltagi`, `_random_kkp`, `_random_none_lme`, `_random_lag`, `_random_sarar_baltagi` | `random_<tag>_coef`, `_phi`, `_s2`, `_sigma2_mu`, `_se`, `_loglik`, `_rho`, `_lambda`; `random_none_lme_*` (nlme) | the same |
| `test_validate_spgm_w2sls`, `_b2sls`, `_ec2sls`, `_g2sls` | `<method>_coef` (rho last), `_se`, `_sargan`, `_n_instruments`; `sigma2_nu`, `sigma2_1` | `Tol::coef(n)`, `Tol::se(n)`, `Tol::stat(n)`; n = 294, 49 for b2sls |
| `test_validate_spgm_within_error_gm` | `spgm_within_error_rho` (p2a's lambda), `_sigma2`, `_coef`, `_se` | lambda and sigma2 `Tol::iterative(1e-10)`; coef and SE at n = 294 |
| `test_validate_spgm_within_gs2sls` | `gs2sls_coef`, `_se`, `_lambda`, `_sigma2`, `_sargan` | the same |

Newton converges quadratically to the root of the analytic profile gradient (R's optimum
is Newton-polished to |gradient| < 1e-9), hence 1e-8 on the parameters; the SEs compare a
central difference of the analytic score with R's Richardson Jacobian. The 2SLS tests
also check that the exogenous columns of Z are their own projection on H within 1e-10.

Property tests (no R value; the same data, read from the CSV): lambda is reported for an
error model and rho is not (`test_spml_error_lambda_not_rho`); individual, time and
two-way effects give different estimates (`test_spml_effects_change_the_estimate`); the
iteration count and the estimates do not change when y and X are scaled by 1e-6 or 1e6
(`test_spml_convergence_is_scale_free`, `test_spml_random_convergence_is_scale_free`);
`b`, `kkp` and no spatial error differ under random effects
(`test_spml_random_error_types_differ`); the four 2SLS methods give different rho
(`test_spgm_methods_differ`: R 0.46613, 0.33570, 0.48136, 0.46514); E1, E2, E3, E5 and E6
(`test_spml_pooling_rejects_effects`, `test_spml_random_rejects_time_effects`,
`test_spgm_unsupported_combinations_are_errors`); exact fits
(`test_spml_perfect_fit_is_refused`) and too-small panels
(`test_spml_rejects_too_small_panels`, `test_spgm_rejects_too_small_panels`); phi at its
boundary on a hand panel, where the fit is pooled OLS (`test_spml_random_phi_at_boundary`,
R `lm` and `nlme::lme`); the complex log-determinant
(`test_panel_spectrum_log_det_uses_complex_eigenvalues`); option parsing.

### Decisive values

| Quantity | R | Before (the code these tests replaced) |
|---|---|---|
| within individual lag rho | 0.430224471611295 | 0.42992867087595732 (real parts of the spectrum only) |
| within individual error lambda | 0.430180548080978 | 0.42947781330776158, reported as rho |
| within time-effect lag rho | 0.3738995396467167 | the individual-effect value |
| random effects, Baltagi error: lambda, sigma_mu^2 | 0.3855693594443952, 0.45760628939589354 | lambda not estimated |
| random effects without spatial error, sigma_mu (y_re) | sqrt(0.4386478403391463) = 0.6623 | 0.92927640426594027 |
| w2sls coefficients x1, x2, rho | 0.8433299096071644, -0.5480186543829472, 0.466133393075115 | rho 0.45499286485906043 (instruments [X, WX]) |
| ec2sls coefficients (Intercept), x1, x2, rho | 0.0688995602817346, 0.8838002525590327, -0.5257883597831456, 0.48136146160724036 | the w2sls estimate |
| within GM lambda | 0.4334753248240874 | not estimated |

## Tests (T2-WP14): panel layout and missing values

These tests are not R-pinned in the registry's sense: splm is not installed, so the
reference values come from a base-R re-implementation of the spml within-lag concentrated
likelihood (`ll(rho) = -NT/2 (1 + log 2 pi + log(RSS(rho)/NT)) + T sum log(1 - rho ev)`),
and they are pinned inline in the tests.

- `test_spml_entity_order_r_cross_check` (`econometrics::splm::tests`): the 12-entity,
  5-period panel below, fitted with the rows entity-major (A), shuffled (B, row i = row
  (7 i + 3) mod 60 of A) and shuffled with text labels u1..u12 (C). All three give
  rho = 0.41522977605328182, beta_x = 1.4535946650208815 and logLik = 13.520649774963919.
  sigma² is the ML estimate RSS/(NT) = 0.03496052085945315 (the R script prints
  RSS/(n - N - k) = 0.044630452161004024; times 47/60). Tolerances: `Tol::iterative(1e-6)`
  for rho, beta and sigma2 and `Tol::iterative(1e-9)` for logLik. With W rows in first-appearance order (the code before T2-WP14), B gives
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
- MCP (`tools::handlers::spatial::tests`): `test_mcp_spatial_panel_ml_sorted_ids_base_r_cross_check`
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

1. The random-effects GM (six moments, `moments = weights | fullweights`) is not
   implemented, so spgm's spatial-error and GS2SLS models are within (`w2sls`) only.
2. No time or two-way random effects.
3. spreml (serial correlation): `SpremlConfig` exists without an estimator.
4. The random-effects model with Baltagi error does a dense N x N eigen decomposition of
   B'B for every lambda it evaluates (O(N^3) time, N x N memory); the KKP and no-error
   paths are sparse.
5. No spatial Durbin specification.

## MCP Tool Exposure

Two tools exposed:
- `spatial_panel_ml` - ML estimation via `run_spml()`
- `spatial_panel_gmm` - GMM estimation via `run_spgm()`

## References

1. Baltagi, B.H., Song, S.H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150. https://doi.org/10.1016/S0304-4076(03)00120-9

2. Kapoor, M., Kelejian, H.H., & Prucha, I.R. (2007). Panel data models with spatially correlated error components. *Journal of Econometrics*, 140(1), 97-130. https://doi.org/10.1016/j.jeconom.2006.09.004

3. Millo, G., & Piras, G. (2012). splm: Spatial Panel Data Models in R. *Journal of Statistical Software*, 47(1), 1-38. https://www.jstatsoft.org/v47/i01/

4. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.

5. Baltagi, B.H. (1981). Simultaneous equations with error components. *Journal of Econometrics*, 17(2), 189-200.

6. Kelejian, H.H., & Prucha, I.R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics*, 17(1), 99-121.

7. Kelejian, H.H., & Prucha, I.R. (1999). A generalized moments estimator for the autoregressive parameter in a spatial model. *International Economic Review*, 40(2), 509-533.

8. Elhorst, J.P. (2003). Specification and estimation of spatial panel data models. *International Regional Science Review*, 26(3), 244-268.

