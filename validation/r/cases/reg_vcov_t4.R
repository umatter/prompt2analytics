# T4-WP06: OLS variance estimators. HAC with VAR(1) prewhitening (REG-22), one lag convention
# for every kernel and the untruncated Quadratic Spectral kernel (REG-50), Driscoll-Kraay from
# per-period score sums (REG-24), the robust Wald F (REG-49), and indefinite covariances
# (REG-62). sandwich 3.1.3, plm 2.6.7, lmtest 0.9.40, car (linearHypothesis) on R 4.3.3.
#
# p2a's HAC and Driscoll-Kraay `bandwidth` is a lag count L. Every kernel is evaluated at
# j / (L + 1), as sandwich::NeweyWest(lag = L) and sandwich::vcovPL(lag = L) do, so p2a's
# lag L equals kernHAC(bw = L + 1) for every kernel. The HAC lag set is weightsAndrews'
# (weights of lags 0..n-1, cut after the last one above 1e-7 in absolute value); the
# Driscoll-Kraay lag set is meatPL's (cut after the last non-zero weight).

suppressPackageStartupMessages({ library(lmtest); library(sandwich); library(plm) })

.t4_se <- function(V) { s <- sqrt(diag(V)); names(s) <- colnames(V); named(s) }
.t4_vcov <- function(V) vec(as.vector(t(V)))   # row-major
.t4_ar <- function(n, phi) { e <- numeric(n); e[1] <- rnorm(1); for (i in 2:n) e[i] <- phi * e[i - 1] + rnorm(1); e }
.t4_nweights <- function(m, bw, kernel, prewhite) length(weightsAndrews(m, bw = bw, kernel = kernel, prewhite = prewhite))

# REG-22 and REG-50: three regressors with different autocorrelation, AR(0.7) errors.
case("vcov_hac_prewhite_ts", "regression/hac", "HAC with VAR(1) prewhitening, y ~ x1 + x2 (x1 AR(0.9), x2 iid, errors AR(0.7)), n = 300, lag 5", function(ctx) {
  set.seed(20261111); n <- 300
  x1 <- .t4_ar(n, 0.9); x2 <- rnorm(n); e <- .t4_ar(n, 0.7)
  d <- ctx$gen_dataset("vcov_hac_prewhite_ts", data.frame(y = 1 + 0.5 * x1 - 0.3 * x2 + e, x1 = x1, x2 = x2))
  m <- lm(y ~ x1 + x2, data = d)
  v <- list(coef = named(coef(m)), n_obs = nobs(m), lag = 5)
  V <- NeweyWest(m, lag = 5, prewhite = FALSE); v$nw_lag5 <- .t4_se(V); v$nw_lag5_vcov <- .t4_vcov(V)
  V <- NeweyWest(m, lag = 5, prewhite = TRUE); v$nw_lag5_prewhite <- .t4_se(V); v$nw_lag5_prewhite_vcov <- .t4_vcov(V)
  kern <- c(parzen = "Parzen", qs = "Quadratic Spectral", truncated = "Truncated", tukey_hanning = "Tukey-Hanning")
  for (k in names(kern)) {
    V <- kernHAC(m, kernel = kern[[k]], bw = 6, prewhite = TRUE, adjust = FALSE)
    v[[sprintf("kernhac_%s_bw6_prewhite", k)]] <- .t4_se(V)
    v[[sprintf("kernhac_%s_bw6_prewhite_vcov", k)]] <- .t4_vcov(V)
  }
  V <- kernHAC(m, kernel = "Quadratic Spectral", bw = 6, prewhite = FALSE, adjust = FALSE)
  v$kernhac_qs_bw6 <- .t4_se(V); v$kernhac_qs_bw6_vcov <- .t4_vcov(V)
  v$qs_bw6_n_weights <- .t4_nweights(m, 6, "Quadratic Spectral", 0)
  v$qs_bw6_prewhite_n_weights <- .t4_nweights(m, 6, "Quadratic Spectral", 1)
  stopifnot(v$qs_bw6_n_weights > 0.98 * n, v$qs_bw6_prewhite_n_weights > 0.98 * (n - 1))
  list(values = v,
       reference = "sandwich::NeweyWest(lag = 5, prewhite = FALSE | TRUE, adjust = FALSE); sandwich::kernHAC(kernel =, bw = 6, prewhite = TRUE | FALSE, adjust = FALSE); sandwich::weightsAndrews",
       call = "NeweyWest(lm(y ~ x1 + x2, data = vcov_hac_prewhite_ts), lag = 5, prewhite = TRUE)", packages = "sandwich")
})

# REG-50: the Quadratic Spectral kernel over a long series. At bw = 1 (lag 0) weightsAndrews
# stops at the last weight above 1e-7 (far below n - 1); at bw = 3 (lag 2) it keeps all n lags.
case("vcov_hac_qs_long", "regression/hac", "HAC, Quadratic Spectral kernel at lag 0 and 2 (all weights above 1e-7), y ~ x, errors AR(0.5), n = 3000", function(ctx) {
  set.seed(31); n <- 3000
  x <- .t4_ar(n, 0.3); e <- .t4_ar(n, 0.5)
  d <- ctx$gen_dataset("vcov_hac_qs_long", data.frame(y = 2 + x + e, x = x))
  m <- lm(y ~ x, data = d)
  v <- list(coef = named(coef(m)), n_obs = nobs(m))
  for (bw in c(1, 3)) {
    V <- kernHAC(m, kernel = "Quadratic Spectral", bw = bw, prewhite = FALSE, adjust = FALSE)
    v[[sprintf("kernhac_qs_bw%d", bw)]] <- .t4_se(V)
    v[[sprintf("kernhac_qs_bw%d_vcov", bw)]] <- .t4_vcov(V)
    v[[sprintf("qs_bw%d_n_weights", bw)]] <- .t4_nweights(m, bw, "Quadratic Spectral", 0)
  }
  V <- kernHAC(m, kernel = "Quadratic Spectral", bw = 3, prewhite = TRUE, adjust = FALSE)
  v$kernhac_qs_bw3_prewhite <- .t4_se(V); v$kernhac_qs_bw3_prewhite_vcov <- .t4_vcov(V)
  V <- NeweyWest(m, lag = 2, prewhite = FALSE); v$nw_lag2 <- .t4_se(V)
  stopifnot(v$qs_bw1_n_weights < n / 2, v$qs_bw3_n_weights > 0.99 * n)
  list(values = v,
       reference = "sandwich::kernHAC(kernel = 'Quadratic Spectral', bw = 1 | 3, prewhite = FALSE | TRUE, adjust = FALSE); sandwich::weightsAndrews; sandwich::NeweyWest(lag = 2, prewhite = FALSE)",
       call = "kernHAC(lm(y ~ x, data = vcov_hac_qs_long), kernel = 'Quadratic Spectral', bw = 1, prewhite = FALSE, adjust = FALSE)", packages = "sandwich")
})

# REG-22, REG-50: the reviewer's Grunfeld numbers (rows in file order, as a time series).
case("vcov_hac_grunfeld", "regression/hac", "HAC on Grunfeld (inv ~ value + capital, file order): Newey-West lag 3 with and without prewhitening, QS at lag 2 and 3", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = d)
  v <- list(coef = named(coef(m)), n_obs = nobs(m), bw_rule = floor(4 * (nobs(m) / 100)^(2 / 9)))
  V <- NeweyWest(m, lag = 3, prewhite = FALSE); v$nw_lag3 <- .t4_se(V)
  V <- NeweyWest(m, lag = 3, prewhite = TRUE); v$nw_lag3_prewhite <- .t4_se(V); v$nw_lag3_prewhite_vcov <- .t4_vcov(V)
  for (bw in c(3, 4)) v[[sprintf("kernhac_qs_bw%d", bw)]] <- .t4_se(kernHAC(m, kernel = "Quadratic Spectral", bw = bw, prewhite = FALSE, adjust = FALSE))
  V <- NeweyWest(m, lag = v$bw_rule, prewhite = FALSE); v$nw_lag_rule <- .t4_se(V)
  list(values = v,
       reference = "sandwich::NeweyWest(lag =, prewhite = FALSE | TRUE, adjust = FALSE); sandwich::kernHAC(kernel = 'Quadratic Spectral', bw = 3 | 4, prewhite = FALSE, adjust = FALSE)",
       call = "NeweyWest(lm(inv ~ value + capital, data = grunfeld), lag = 3, prewhite = TRUE)", packages = "sandwich")
})

# REG-24, REG-50: Driscoll-Kraay as vcovPL(cluster = ~firm, order.by = ~year, adjust = FALSE),
# every kernel at lag 2 (= the default floor(T^(1/4)) for T = 20); plm::vcovSCC cross-check.
.t4_dk <- function(m, d, lag, tag) {
  v <- list()
  kern <- c(bartlett = "Bartlett", parzen = "Parzen", qs = "Quadratic Spectral", truncated = "Truncated", tukey_hanning = "Tukey-Hanning")
  for (k in names(kern)) {
    V <- vcovPL(m, cluster = ~firm, order.by = ~year, lag = lag, kernel = kern[[k]], adjust = FALSE)
    v[[sprintf("vcovpl_%s_lag%d", k, lag)]] <- .t4_se(V)
    v[[sprintf("vcovpl_%s_lag%d_vcov", k, lag)]] <- .t4_vcov(V)
  }
  pm <- plm(inv ~ value + capital, data = pdata.frame(d, index = c("firm", "year")), model = "pooling")
  v[[sprintf("vcovscc_lag%d", lag)]] <- .t4_se(vcovSCC(pm, maxlag = lag))
  stopifnot(isTRUE(all.equal(v[[sprintf("vcovscc_lag%d", lag)]]$values, v[[sprintf("vcovpl_bartlett_lag%d", lag)]]$values, tolerance = 1e-12)))
  v
}
case("vcov_dk_grunfeld", "regression/hac", "Driscoll-Kraay on Grunfeld (pooled inv ~ value + capital, N = 10, T = 20), five kernels at lag 2", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = d)
  T <- length(unique(d$year))
  v <- c(list(coef = named(coef(m)), n_obs = nobs(m), n_periods = T, lag_rule = floor(T^(1 / 4)),
              rdf_periods = T - length(coef(m))), .t4_dk(m, d, 2, ""))
  list(values = v,
       reference = "sandwich::vcovPL(cluster = ~firm, order.by = ~year, lag = 2, kernel =, adjust = FALSE); plm::vcovSCC(maxlag = 2)",
       call = "vcovPL(lm(inv ~ value + capital, data = grunfeld), cluster = ~firm, order.by = ~year, lag = 2, adjust = FALSE)", packages = c("sandwich", "plm"))
})
case("vcov_dk_grunfeld_unbal", "regression/hac", "Driscoll-Kraay on an unbalanced Grunfeld (10 rows dropped), five kernels at lag 2", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  d <- ctx$gen_dataset("vcov_dk_grunfeld_unbal", g[-c(3, 17, 25, 48, 66, 91, 120, 150, 177, 199), ])
  m <- lm(inv ~ value + capital, data = d)
  v <- c(list(coef = named(coef(m)), n_obs = nobs(m), n_periods = length(unique(d$year))), .t4_dk(m, d, 2, ""))
  list(values = v,
       reference = "sandwich::vcovPL(cluster = ~firm, order.by = ~year, lag = 2, kernel =, adjust = FALSE); plm::vcovSCC(maxlag = 2)",
       call = "vcovPL(lm(inv ~ value + capital, data = vcov_dk_grunfeld_unbal), cluster = ~firm, order.by = ~year, lag = 2, adjust = FALSE)", packages = c("sandwich", "plm"))
})
case("vcov_dk_grunfeld_t10", "regression/hac", "Driscoll-Kraay on Grunfeld 1935-1944 (T = 10): default lag floor(T^(1/4)) = 1, five kernels", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  d <- ctx$gen_dataset("vcov_dk_grunfeld_t10", g[g$year <= 1944, ])
  m <- lm(inv ~ value + capital, data = d)
  T <- length(unique(d$year))
  v <- c(list(coef = named(coef(m)), n_obs = nobs(m), n_periods = T, lag_rule = floor(T^(1 / 4))), .t4_dk(m, d, 1, ""))
  stopifnot(v$lag_rule == 1,
            isTRUE(all.equal(v$vcovpl_tukey_hanning_lag1$values, v$vcovpl_bartlett_lag1$values, tolerance = 1e-12)))
  list(values = v,
       reference = "sandwich::vcovPL(cluster = ~firm, order.by = ~year, lag = 1, kernel =, adjust = FALSE); plm::vcovSCC(maxlag = 1)",
       call = "vcovPL(lm(inv ~ value + capital, data = vcov_dk_grunfeld_t10), cluster = ~firm, order.by = ~year, lag = 1, kernel = 'Parzen', adjust = FALSE)", packages = c("sandwich", "plm"))
})

# REG-49: the overall F test with the reported covariance, as lmtest::waldtest(m, vcov = V)
# (F on the slopes, denominator df n - k) and car::linearHypothesis for a model through the origin.
case("ols_wald_f_grunfeld", "regression/ols", "Robust Wald F of inv ~ value + capital on Grunfeld: HC0-HC3, CR1 by firm, two-way firm + year; through the origin with HC1", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = d)
  s <- summary(m)$fstatistic
  v <- list(n_obs = nobs(m), f_classical = unname(s[1]), f_df1 = unname(s[2]), f_df2 = unname(s[3]),
            f_classical_p = unname(pf(s[1], s[2], s[3], lower.tail = FALSE)))
  wt <- function(V) { w <- waldtest(m, vcov = V, test = "F"); c(w$F[2], w$`Pr(>F)`[2], w$Res.Df[1]) }
  for (t in c("HC0", "HC1", "HC2", "HC3")) {
    r <- wt(vcovHC(m, type = t)); tl <- tolower(t)
    v[[paste0("wald_f_", tl)]] <- r[1]; v[[paste0("wald_f_", tl, "_p")]] <- r[2]
  }
  r <- wt(vcovCL(m, cluster = ~firm, type = "HC1")); v$wald_f_cl_firm <- r[1]; v$wald_f_cl_firm_p <- r[2]
  r <- wt(vcovCL(m, cluster = ~firm + year, type = "HC1")); v$wald_f_cl_firm_year <- r[1]; v$wald_f_cl_firm_year_p <- r[2]
  v$wald_df2 <- r[3]
  m0 <- lm(inv ~ 0 + value + capital, data = d)
  lh <- car::linearHypothesis(m0, c("value = 0", "capital = 0"), vcov. = vcovHC(m0, type = "HC1"), test = "F")
  v$wald_f_hc1_origin <- lh$F[2]; v$wald_f_hc1_origin_p <- lh$`Pr(>F)`[2]; v$wald_origin_df2 <- lh$Res.Df[2]
  b <- coef(m)[-1]; V <- vcovHC(m, type = "HC1")[-1, -1]
  stopifnot(abs(drop(t(b) %*% solve(V, b)) / 2 - v$wald_f_hc1) < 1e-9 * v$wald_f_hc1)
  list(values = v,
       reference = "summary.lm F; lmtest::waldtest(m, vcov = sandwich::vcovHC(type =) | vcovCL(cluster =, type = 'HC1'), test = 'F'); car::linearHypothesis(vcov. = vcovHC(type = 'HC1'), test = 'F') through the origin",
       call = "waldtest(lm(inv ~ value + capital, data = grunfeld), vcov = vcovHC(m, type = 'HC1'), test = 'F')", packages = c("sandwich", "lmtest", "car"))
})

# REG-62: a two-way cluster-robust covariance that is not positive semi-definite (3 x 3
# clusters), and sandwich's eigenvalue fix (vcovCL(fix = TRUE)).
case("ols_twoway_indefinite", "regression/ols", "Two-way clustered OLS whose CGM covariance has a negative variance (3 x 3 clusters, n = 36), with and without vcovCL(fix = TRUE)", function(ctx) {
  set.seed(1); a <- rep(1:3, each = 12); b <- rep(rep(1:3, each = 4), 3)
  x <- rnorm(36); y <- 1 + 0.5 * x + rnorm(36)
  d <- ctx$gen_dataset("ols_twoway_indefinite", data.frame(y = y, x = x, a = a, b = b))
  m <- lm(y ~ x, data = d)
  V <- vcovCL(m, cluster = ~a + b, type = "HC1")
  Vf <- vcovCL(m, cluster = ~a + b, type = "HC1", fix = TRUE)
  ev <- eigen(V, symmetric = TRUE)$values
  stopifnot(any(diag(V) < 0), min(ev) < 0)
  list(values = list(coef = named(coef(m)), n_obs = nobs(m), cl_twoway_diag = named(setNames(diag(V), colnames(V))),
                     cl_twoway_vcov = .t4_vcov(V), cl_twoway_eigenvalues = vec(ev),
                     cl_twoway_fix_se = .t4_se(Vf), cl_twoway_fix_vcov = .t4_vcov(Vf)),
       reference = "sandwich::vcovCL(cluster = ~a + b, type = 'HC1', fix = FALSE | TRUE)",
       call = "vcovCL(lm(y ~ x, data = ols_twoway_indefinite), cluster = ~a + b, type = 'HC1', fix = TRUE)", packages = "sandwich")
})

# REG-62: the truncated kernel can give a negative variance; R reports sqrt(negative) = NaN.
case("hac_truncated_indefinite", "regression/hac", "HAC, truncated kernel, lag 5 (kernHAC bw = 6), errors AR(-0.7), n = 40: a negative variance", function(ctx) {
  set.seed(5); n <- 40
  e <- as.numeric(arima.sim(list(ar = -0.7), n)); x <- rnorm(n)
  d <- ctx$gen_dataset("hac_truncated_indefinite", data.frame(y = 1 + x + e, x = x))
  m <- lm(y ~ x, data = d)
  V <- kernHAC(m, kernel = "Truncated", bw = 6, prewhite = FALSE, adjust = FALSE)
  stopifnot(any(diag(V) < 0))
  list(values = list(coef = named(coef(m)), n_obs = nobs(m), kernhac_truncated_bw6_diag = named(setNames(diag(V), colnames(V))),
                     kernhac_truncated_bw6_vcov = .t4_vcov(V)),
       reference = "sandwich::kernHAC(kernel = 'Truncated', bw = 6, prewhite = FALSE, adjust = FALSE)",
       call = "kernHAC(lm(y ~ x, data = hac_truncated_indefinite), kernel = 'Truncated', bw = 6, prewhite = FALSE, adjust = FALSE)", packages = "sandwich")
})
