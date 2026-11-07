# validation/r/cases/ml_pdp.R -- partial dependence and ICE references (T3-WP49: ML-137, ML-133, WT-18,
# DATA-83). pdp::partial (brute force, pdp 0.8.3) on fixed lm fits. The Rust tests rebuild each fit's
# prediction function from the committed coefficients, so these cases pin pdp's grid rules, the averaging,
# the two-predictor layout and ICE centring, not a model:
#  * pdp:::pred_grid's default grid, min(length(unique(x)), 51) uniform points (no grid.resolution), which
#    is also the default of PdpConfig and IceConfig;
#  * its uniform rule seq(min(x), max(x), length.out = n) (grid.resolution = n) and its quantile rule
#    quantile(x, probs, type = 7, na.rm = TRUE) (quantiles = TRUE), at pdp's default probs = 1:9/10 and at
#    probs = seq(0, 1, length.out = 20);
#  * two predictors on a user pred.grid of 5 x 4 points, stored as a row-major matrix (x1 slowest);
#  * ICE (ice = TRUE) and centred ICE (center = TRUE: every curve minus its value at the smallest grid
#    value) on an unsorted user grid, which pdp sorts (order_grid); an 8-point uniform ICE grid with the
#    per-point population SD, range and IQR (type 7) of the ICE values;
#  * a feature with ties: its quantile grid repeats values (pdp repeats the row; p2a lists each value
#    once, so the case also stores unique() and the yhat of the first occurrence);
#  * pdp_small_scale: a feature on the 1e-12 scale whose grid points are distinct and must stay so.
# randomForest is fitted on pdp_lm only to assert that pred_grid does not depend on the model.
# pdp and randomForest are called through their namespaces (no library()), so that pdp::partial does not
# mask purrr::partial and randomForest does not mask combine/margin/importance for the other case files.
stopifnot(requireNamespace("pdp", quietly = TRUE), requireNamespace("randomForest", quietly = TRUE))

case("pdp_lm", "ml/pdp",
     "Partial dependence and ICE grid rules (pdp::partial) on a fixed lm fit, n = 60",
     function(ctx) {
  set.seed(20261022)
  n <- 60
  x1 <- runif(n, -2, 2)
  x2 <- rnorm(n)
  x3 <- sample(c(0.5, 1.5, 2.5, 3.5), n, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1))
  y <- 1 + 2 * x1 - x2 + 0.5 * x1^2 + 0.8 * x1 * x2 + 0.3 * x3 + rnorm(n, sd = 0.4)
  d <- ctx$gen_dataset("pdp_lm", data.frame(x1 = x1, x2 = x2, x3 = x3, y = y))
  fit <- lm(y ~ x1 * x2 + I(x1^2) + x3, data = d)
  stopifnot(identical(names(coef(fit)), c("(Intercept)", "x1", "x2", "I(x1^2)", "x3", "x1:x2")))

  # The default grid: 60 distinct x1 values give 51 points, the 4 distinct x3 values 4 points
  pdef <- pdp::partial(fit, pred.var = "x1", train = d)
  pdef3 <- pdp::partial(fit, pred.var = "x3", train = d)
  stopifnot(nrow(pdef) == 51, nrow(pdef3) == 4,
            identical(pdef$x1, seq(min(d$x1), max(d$x1), length.out = 51)))

  # 1-D, uniform and quantile grids
  pu <- pdp::partial(fit, pred.var = "x1", grid.resolution = 25, train = d)
  stopifnot(nrow(pu) == 25, pu$x1[1] == min(d$x1), pu$x1[25] == max(d$x1),
            identical(pu$x1, seq(min(d$x1), max(d$x1), length.out = 25)))
  pq <- pdp::partial(fit, pred.var = "x1", quantiles = TRUE, train = d)
  stopifnot(identical(pq$x1, unname(quantile(d$x1, probs = 1:9 / 10, type = 7))))
  probs20 <- seq(0, 1, length.out = 20)
  pq20 <- pdp::partial(fit, pred.var = "x1", quantiles = TRUE, probs = probs20, train = d)

  # pred_grid does not depend on the model
  set.seed(1)
  rf <- randomForest::randomForest(y ~ x1 + x2 + x3, data = d, ntree = 25)
  stopifnot(identical(pdp::partial(rf, pred.var = "x1", grid.resolution = 25, train = d)$x1, pu$x1),
            identical(pdp::partial(rf, pred.var = "x1", quantiles = TRUE, train = d)$x1, pq$x1))

  # 2-D on a 5 x 4 pred.grid: M[i, j] = PD(g1[i], g2[j]), stored row-major
  g1 <- seq(min(d$x1), max(d$x1), length.out = 5)
  g2 <- seq(min(d$x2), max(d$x2), length.out = 4)
  p2 <- pdp::partial(fit, pred.var = c("x1", "x2"), pred.grid = expand.grid(x1 = g1, x2 = g2), train = d)
  M <- matrix(NA_real_, 5, 4)
  for (r in seq_len(nrow(p2))) M[match(p2$x1[r], g1), match(p2$x2[r], g2)] <- p2$yhat[r]
  stopifnot(!anyNA(M))

  # ICE and centred ICE on an unsorted user grid (pdp sorts it); matrices row-major (observation slowest)
  gc <- c(1.3, -1.7, 0.25, -0.6, 1.9)
  ice_mat <- function(p, k) {
    p <- p[order(p$yhat.id), ]
    m <- matrix(p$yhat, nrow = n, ncol = k, byrow = TRUE)
    stopifnot(all(p$yhat.id == rep(seq_len(n), each = k)))
    m
  }
  ic <- pdp::partial(fit, pred.var = "x1", pred.grid = data.frame(x1 = gc), ice = TRUE, train = d)
  stopifnot(identical(ic$x1[1:5], sort(gc)))
  Mi <- ice_mat(ic, 5)
  cic <- pdp::partial(fit, pred.var = "x1", pred.grid = data.frame(x1 = gc), ice = TRUE, center = TRUE, train = d)
  Mc <- ice_mat(cic, 5)
  stopifnot(all(Mc[, 1] == 0), max(abs(Mc - (Mi - Mi[, 1]))) < 1e-12)

  # ICE on the 8-point uniform grid, with the spread statistics p2a reports per grid point
  iu <- pdp::partial(fit, pred.var = "x1", grid.resolution = 8, ice = TRUE, train = d)
  Mu <- ice_mat(iu, 8)
  pop_sd <- function(v) sqrt(mean((v - mean(v))^2))

  # A feature with ties: pdp repeats grid values; p2a keeps the first occurrence of each
  pt <- pdp::partial(fit, pred.var = "x3", quantiles = TRUE, train = d)
  stopifnot(anyDuplicated(pt$x3) > 0)
  keep <- !duplicated(pt$x3)

  list(values = list(
    coef = named(coef(fit)),
    pd_default_grid = vec(pdef$x1), pd_default_yhat = vec(pdef$yhat),
    pd_default_x3_grid = vec(pdef3$x3), pd_default_x3_yhat = vec(pdef3$yhat),
    pd_uniform_grid = vec(pu$x1), pd_uniform_yhat = vec(pu$yhat),
    pdp_default_probs = vec(1:9 / 10),
    pd_quantile_grid = vec(pq$x1), pd_quantile_yhat = vec(pq$yhat),
    quantile20_probs = vec(probs20),
    pd_quantile20_grid = vec(pq20$x1), pd_quantile20_yhat = vec(pq20$yhat),
    pd2_grid1 = vec(g1), pd2_grid2 = vec(g2), pd2_yhat = vec(as.vector(t(M))),
    ice_user_grid = vec(gc), ice_sorted_grid = vec(sort(gc)),
    ice_custom_yhat = vec(as.vector(t(Mi))), ice_custom_pd = vec(colMeans(Mi)),
    cice_custom_yhat = vec(as.vector(t(Mc))), cice_custom_pd = vec(colMeans(Mc)),
    ice_uniform_grid = vec(iu$x1[1:8]), ice_uniform_yhat = vec(as.vector(t(Mu))),
    ice_uniform_pd = vec(colMeans(Mu)),
    ice_uniform_sd_pop = vec(apply(Mu, 2, pop_sd)),
    ice_uniform_range = vec(apply(Mu, 2, function(v) diff(range(v)))),
    ice_uniform_iqr7 = vec(apply(Mu, 2, IQR)),
    pd_ties_grid_r = vec(pt$x3), pd_ties_grid_unique = vec(pt$x3[keep]), pd_ties_yhat_unique = vec(pt$yhat[keep])
  ),
  reference = "pdp::partial(object, pred.var, grid.resolution | quantiles = TRUE, probs | pred.grid, ice, center, train) (pdp 0.8.3, brute force); stats::lm; stats::quantile(type = 7); stats::IQR",
  call = "pdp::partial(lm(y ~ x1 * x2 + I(x1^2) + x3, d), 'x1', grid.resolution = 25 | quantiles = TRUE | pred.grid, ice = TRUE, center = TRUE, train = d)",
  packages = c("pdp", "randomForest"))
})

case("pdp_small_scale", "ml/pdp",
     "Partial dependence and centred ICE grids for a feature on the 1e-12 scale (pdp::partial), n = 40",
     function(ctx) {
  set.seed(20261023)
  n <- 40
  x <- 1e-12 * runif(n, 1, 3)
  z <- rnorm(n)
  y <- 2 + 4e11 * x - 0.5 * z + rnorm(n, sd = 0.3)
  d <- ctx$gen_dataset("pdp_small_scale", data.frame(x = x, z = z, y = y))
  fit <- lm(y ~ x + z, data = d)
  pq <- pdp::partial(fit, pred.var = "x", quantiles = TRUE, train = d)
  pu <- pdp::partial(fit, pred.var = "x", grid.resolution = 9, train = d)
  ci <- pdp::partial(fit, pred.var = "x", grid.resolution = 6, ice = TRUE, center = TRUE, train = d)
  ci <- ci[order(ci$yhat.id), ]
  Mc <- matrix(ci$yhat, nrow = n, ncol = 6, byrow = TRUE)
  stopifnot(length(unique(pq$x)) == 9, length(unique(pu$x)) == 9, diff(range(d$x)) < 1e-10, all(Mc[, 1] == 0))
  list(values = list(
    coef = named(coef(fit)),
    pd_quantile_grid = vec(pq$x), pd_quantile_yhat = vec(pq$yhat),
    pd_uniform_grid = vec(pu$x), pd_uniform_yhat = vec(pu$yhat),
    cice_uniform_grid = vec(ci$x[1:6]), cice_uniform_yhat = vec(as.vector(t(Mc)))
  ),
  reference = "pdp::partial(lm, pred.var, quantiles = TRUE | grid.resolution, ice = TRUE, center = TRUE, train) (pdp 0.8.3)",
  call = "pdp::partial(lm(y ~ x + z, d), 'x', quantiles = TRUE | grid.resolution = 9 | grid.resolution = 6, ice = TRUE, center = TRUE, train = d)",
  packages = c("pdp"))
})
