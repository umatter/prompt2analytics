# validation/r/cases/stats_misc_t3.R -- references: KS with distribution parameters,
# PP.test internals, cancor, prop.test, poisson.test, Mahalanobis (full rank and singular),
# two-way model tables with an empty cell, t.test at other confidence levels, acf/pacf/ccf,
# spline's default n, spec.pgram's grid and spec.ci (STAT-26, STAT-38, STAT-54, STAT-83, STAT-67,
# X-3, STAT-6, STAT-35, STAT-60; STAT-62, TOOL-11 and TOOL-25 core parts).
#
# Some keys are not pinned by the tests: they are registry deviations of open findings
# (STAT-16, STAT-27, STAT-39, STAT-58, STAT-59, STAT-63, STAT-68), named in a
# `// deviation <id>` comment next to the Check. Their R values are recorded here so a later fix
# can pin them without recomputing.

# One column per sample; NA padding keeps a rectangular CSV (Rust drops it with NaPolicy::Omit,
# the KS wrappers do the same since T2-WP06).
.pad <- function(cols) {
  len <- max(lengths(cols))
  df <- data.frame(id = seq_len(len))
  for (k in names(cols)) { v <- rep(NA_real_, len); v[seq_along(cols[[k]])] <- cols[[k]]; df[[k]] <- v }
  df
}

case("stat_ks_t3", "stats/ks", "ks.test: one-sample against pnorm/punif/pexp with parameters, two-sample; D, D+, D-, asymptotic and exact p", function(ctx) {
  set.seed(20261022)
  cols <- list(xn = round(rnorm(30, 3, 2), 4), xu = round(runif(25, -1, 3), 4), xe = round(rexp(40, 0.5), 4),
               x5 = c(1.2, 1.5, 1.8, 2.1, 2.4), y5 = c(2.0, 2.5, 3.0, 3.5, 4.0))
  set.seed(77)
  cols$xa <- round(rnorm(60), 5); cols$ya <- round(rnorm(45, 0.4, 1.2), 5)
  d <- ctx$gen_dataset("ks_t3", .pad(cols))
  g <- function(k) d[[k]][!is.na(d[[k]])]
  xn <- g("xn"); xu <- g("xu"); xe <- g("xe"); x5 <- g("x5"); y5 <- g("y5"); xa <- g("xa"); ya <- g("ya")
  stopifnot(!anyDuplicated(xn), !anyDuplicated(xu), !anyDuplicated(xe), !anyDuplicated(c(xa, ya)), !anyDuplicated(c(x5, y5)))
  one <- function(x, ...) {
    a <- ks.test(x, ..., exact = FALSE); e <- ks.test(x, ...)
    gt <- ks.test(x, ..., alternative = "greater", exact = FALSE); lt <- ks.test(x, ..., alternative = "less", exact = FALSE)
    stopifnot(sqrt(length(x)) * a$statistic >= 1)   # R's pKS2 (tol 1e-6) is exact to ~1e-15 only for sqrt(n) D >= 1
    list(d = unname(a$statistic), p_asym = a$p.value, p_exact = e$p.value,
         dplus = unname(gt$statistic), p_greater_asym = gt$p.value, dminus = unname(lt$statistic), p_less_asym = lt$p.value)
  }
  two <- function(x, y) {
    a <- ks.test(x, y, exact = FALSE); e <- ks.test(x, y)
    gt <- ks.test(x, y, alternative = "greater", exact = FALSE); lt <- ks.test(x, y, alternative = "less", exact = FALSE)
    list(d = unname(a$statistic), p_asym = a$p.value, p_exact = e$p.value,
         dplus = unname(gt$statistic), p_greater_asym = gt$p.value, dminus = unname(lt$statistic), p_less_asym = lt$p.value)
  }
  v <- list()
  add <- function(prefix, l) for (k in names(l)) v[[paste0(prefix, "_", k)]] <<- l[[k]]
  add("norm", one(xn, "pnorm", 2, 2))
  add("unif", one(xu, "punif", -1.5, 3))
  add("exp", one(xe, "pexp", 0.8))
  add("two5", two(x5, y5))
  add("twol", two(xa, ya))
  v$n_xn <- length(xn); v$n_xu <- length(xu); v$n_xe <- length(xe); v$n_xa <- length(xa); v$n_ya <- length(ya)
  list(values = v, reference = "stats::ks.test (exact = FALSE: C_pKS2 tol 1e-6 and exp(-2 n D^2); exact: pkolmogorov2x / psmirnov)",
       call = "ks.test(x, 'pnorm', 2, 2[, exact = FALSE, alternative]); ks.test(x, y[, exact = FALSE, alternative])",
       packages = character(0))
})

case("stat_ts_acf_spline_t3", "stats/acf", "acf, pacf, ccf (lag.max defaults and 5) on an AR(1) pair; spline(method = 'natural') with the default n = 3 length(x) and n = 1", function(ctx) {
  set.seed(1111)
  n <- 64
  e <- rnorm(n); x <- numeric(n); x[1] <- e[1]; for (i in 2:n) x[i] <- 0.6 * x[i - 1] + e[i]
  yv <- c(0, x[-n]) * 0.7 + rnorm(n, 0, 0.5)                 # y_t depends on x_{t-1}: ccf(x, y) peaks at lag -1
  kx <- c(0.3, 1.1, 1.9, 3.2, 4.0, 5.5, 6.1, 7.4, 8.0, 9.6, rep(NA, n - 10))
  ky <- c(round(sin(kx[1:10]) + 0.1 * kx[1:10] + c(0.03, -0.02, 0.05, 0.01, -0.04, 0.02, 0.0, -0.03, 0.04, -0.01), 6), rep(NA, n - 10))
  d <- ctx$gen_dataset("ts_acf_spline_t3", data.frame(x = round(x, 8), y = round(yv, 8), kx = kx, ky = ky))
  x <- d$x; y <- d$y
  a <- acf(x, plot = FALSE); pa <- pacf(x, plot = FALSE); cv <- acf(x, type = "covariance", plot = FALSE)
  cc <- ccf(x, y, lag.max = 5, plot = FALSE)
  kx <- d$kx[!is.na(d$kx)]; ky <- d$ky[!is.na(d$ky)]
  s <- spline(kx, ky, method = "natural"); s1 <- spline(kx, ky, n = 1, method = "natural")
  stopifnot(length(s$x) == 3 * length(kx))
  list(values = list(acf = vec(a$acf), acf_lag_max = max(a$lag), acvf = vec(cv$acf), pacf = vec(pa$acf),
                     ccf = vec(cc$acf), ccf_lags = vec(cc$lag),
                     acf_ci_bound = qnorm(0.975) / sqrt(n),
                     spline_x = vec(s$x), spline_y = vec(s$y), spline_n = length(s$x),
                     spline1_x = s1$x, spline1_y = s1$y),
       reference = "stats::acf, pacf, ccf (lag k of ccf(x, y) is cor(x[t+k], y[t])); stats::spline(method = 'natural', n = 3 * length(x))",
       call = "acf(x); pacf(x); acf(x, type = 'covariance'); ccf(x, y, lag.max = 5); spline(kx, ky, method = 'natural'[, n = 1])",
       packages = character(0))
})

case("stat_spectrum_t3", "timeseries/spectrum", "spec.pgram grid and peak (n = 96, spans = c(3, 3), taper 0.1) and plot.spec's spec.ci multipliers", function(ctx) {
  set.seed(9696)
  n <- 96
  tt <- 1:n
  x <- sin(2 * pi * tt / 8) + 0.5 * arima.sim(list(ar = 0.4), n)
  d <- ctx$gen_dataset("spectrum_t3", data.frame(t = tt, x = round(as.numeric(x), 8)))
  stopifnot(nextn(n) == n)
  sp <- spec.pgram(d$x, spans = c(3, 3), taper = 0.1, detrend = TRUE, plot = FALSE)
  raw <- spec.pgram(d$x, taper = 0, detrend = TRUE, plot = FALSE)
  # plot.spec's private spec.ci, verbatim (R 4.3.3)
  spec_ci <- function(df, coverage = 0.95) {
    tail <- (1 - coverage)
    upper.quantile <- 1 - tail * pchisq(df, df, lower.tail = FALSE)
    lower.quantile <- tail * pchisq(df, df)
    1 / (qchisq(c(upper.quantile, lower.quantile), df) / df)
  }
  list(values = list(freq = vec(sp$freq), peak_freq = sp$freq[which.max(sp$spec)], spec = vec(sp$spec),
                     df = sp$df, bandwidth = sp$bandwidth, raw_spec = vec(raw$spec), raw_df = raw$df, raw_bandwidth = raw$bandwidth,
                     ci_df = vec(c(sp$df, 2, 18, 7.5)),
                     ci95_at_df = vec(c(spec_ci(sp$df), spec_ci(2), spec_ci(18), spec_ci(7.5))),
                     ci90_at_df = vec(c(spec_ci(sp$df, 0.9), spec_ci(2, 0.9), spec_ci(18, 0.9), spec_ci(7.5, 0.9)))),
       reference = "stats::spec.pgram (frequencies k / N, modified Daniell spans, taper); stats:::plot.spec's spec.ci (unequal tails proportional to P(X <= df))",
       call = "spec.pgram(x, spans = c(3, 3), taper = 0.1, detrend = TRUE); spec_ci(df, coverage)", packages = character(0))
})
