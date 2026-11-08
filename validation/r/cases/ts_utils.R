# validation/r/cases/ts_utils.R -- time-series utility references (T3-WP37: TS-13, TS-16, TS-18, TS-22; planner
# move-ins TS-4, TS-7, TS-9, TS-15, TS-31).
#  * stats::filter(method = "convolution", sides = 2 | 1) with even and odd filters (C cfilter: nshift = p %/% 2,
#    NA where the window leaves the series) and method = "recursive";
#  * stats::ar(method = "ols", intercept = FALSE): a short series (n = 24, default order.max 13 > n / 2; R fits up to
#    the saturated order 12 and stops at 13 with "singularities"), a longer one (n = 60, every order estimable) and
#    fixed orders 2 and 3; ar(method = "burg") (TS-4) and ar(method = "yule-walker") prediction variances (TS-15);
#  * changepoint::cpt.var / cpt.meanvar / cpt.mean (PELT, MBIC) on a series at level 1e7; cpt.var with a manual
#    penalty; cpt.mean(method = "BinSeg") recorded only (TS-8, Tier 4);
#  * stats::diffinv (differences >= 2 with non-zero xi recorded only: TS-28, Tier 4);
#  * stats::ARMAacf: pure AR and MA pinned; mixed ARMA and a persistent AR(2) recorded only (TS-14, Tier 4);
#  * the body of stats::cpgram, which returns NULL: spec.taper(scale(x, TRUE, FALSE), p = taper), Mod(fft(x))^2 / n,
#    R's frequency set (the point (0, 0) dropped), the band 1.358 / (sqrt(mp) + 0.12 + 0.11 / sqrt(mp)), on white
#    noise with mean 3 and on an AR(1) around 10 (TS-9). The KS p-value is the asymptotic Kolmogorov survival function
#    (1 - stats:::C_pKS2 below 1, its alternating series from 1 on).
suppressPackageStartupMessages(library(changepoint))
case("ts_filter_ar_cpt", "forecasting/tsutils", "stats::filter two-sided even/odd; ar(method = 'ols') short series; changepoint at level 1e7", function(ctx) {
  set.seed(20261123)
  x <- cumsum(rnorm(60)); s <- round(arima.sim(list(ar = c(0.5, -0.3)), 24), 10)
  lev <- c(rnorm(80, 0, 1), rnorm(70, 0, 3), rnorm(60, 0, 1)) + 1e7
  wn <- rnorm(100, 3, 1); ar1 <- arima.sim(list(ar = 0.9), 120) + 10       # drawn after lev: x, s and lev unchanged
  na <- function(v) c(v, rep(NA, 210 - length(v)))
  d <- ctx$gen_dataset("ts_utils", data.frame(t = 1:210, x = na(round(x, 10)), s = na(s), lev = round(lev, 6),
                                              wn = na(round(wn, 10)), ar1 = na(round(as.numeric(ar1), 10))))
  x <- d$x[1:60]; s <- d$s[1:24]; lev <- d$lev; wn <- d$wn[1:100]; ar1 <- d$ar1[1:120]
  v <- list()
  # --- filter (TS-13) ---
  for (f in list(c(0.5, 0.5), rep(0.25, 4), c(0.2, 0.3, 0.5), c(1, 2, 3, 2, 1) / 9)) {
    tag <- paste0("filter_p", length(f), "_", gsub("\\.", "", format(f[1])))
    v[[paste0(tag)]] <- vec(as.numeric(stats::filter(x, f, method = "convolution", sides = 2)))
    v[[paste0(tag, "_sides1")]] <- vec(as.numeric(stats::filter(x, f, method = "convolution", sides = 1)))
  }
  v$filter_recursive <- vec(as.numeric(stats::filter(x, c(0.5, -0.2), method = "recursive")))
  stopifnot(inherits(try(stats::filter(x[1:3], rep(0.1, 10)), silent = TRUE), "try-error"))   # 'filter' is longer than time series
  # --- ar(method = "ols") (TS-16, TS-15) ---
  a <- suppressWarnings(ar(s, method = "ols", demean = TRUE, intercept = FALSE))
  stopifnot(a$order == 12, is.infinite(a$aic[14]))
  v$ar_ols_order <- a$order; v$ar_ols_coef <- vec(as.numeric(a$ar)); v$ar_ols_aic <- vec(ifelse(is.infinite(a$aic), NA, as.numeric(a$aic))); v$ar_ols_aic_infinite_orders <- vec(which(is.infinite(a$aic)) - 1); v$ar_ols_var <- as.numeric(a$var.pred)
  v$ar_ols_aic_diff <- vec(as.numeric(a$aic[1:12] - a$aic[1]))        # orders 0..11 relative to order 0: free of the saturated fit's rounding
  w <- tryCatch(ar(s, method = "ols", demean = TRUE, intercept = FALSE), warning = function(w) conditionMessage(w))
  stopifnot(grepl("model order:  13 singularities", w))
  a60 <- ar(x, method = "ols", demean = TRUE, intercept = FALSE)
  v$ar_ols_n60_order <- a60$order; v$ar_ols_n60_coef <- vec(as.numeric(a60$ar)); v$ar_ols_n60_aic <- vec(as.numeric(a60$aic)); v$ar_ols_n60_var <- as.numeric(a60$var.pred); v$ar_ols_n60_mean <- a60$x.mean
  for (m in 2:3) {
    f <- ar(s, method = "ols", demean = TRUE, intercept = FALSE, aic = FALSE, order.max = m)
    v[[paste0("ar_ols_fixed", m, "_coef")]] <- vec(as.numeric(f$ar)); v[[paste0("ar_ols_fixed", m, "_var")]] <- as.numeric(f$var.pred)
  }
  b <- ar(x, method = "burg")
  v$ar_burg_order <- b$order; v$ar_burg_coef <- vec(as.numeric(b$ar)); v$ar_burg_var <- b$var.pred; v$ar_burg_aic <- vec(as.numeric(b$aic)); v$ar_burg_pacf <- vec(as.numeric(b$partialacf))
  b3 <- ar(x, method = "burg", aic = FALSE, order.max = 3)
  v$ar_burg_fixed3_coef <- vec(as.numeric(b3$ar)); v$ar_burg_fixed3_var <- b3$var.pred
  y <- ar(x, method = "yule-walker")
  v$ar_yw_order <- y$order; v$ar_yw_coef <- vec(as.numeric(y$ar)); v$ar_yw_var <- y$var.pred
  y3 <- ar(x, method = "yule-walker", aic = FALSE, order.max = 3)
  v$ar_yw_fixed3_coef <- vec(as.numeric(y3$ar)); v$ar_yw_fixed3_var <- y3$var.pred
  # --- changepoint (TS-18, TS-7, TS-31) ---
  cv <- cpt.var(lev, method = "PELT", penalty = "MBIC"); cm <- cpt.meanvar(lev, method = "PELT", penalty = "MBIC")
  v$cpt_var_cpts <- vec(cpts(cv)); v$cpt_meanvar_cpts <- vec(cpts(cm))      # cpt_meanvar_cpts: R's raw-sum cancellation, recorded only
  cv0 <- cpt.var(lev - mean(lev), method = "PELT", penalty = "MBIC"); v$cpt_var_centred_cpts <- vec(cpts(cv0))
  v$cpt_var_pen <- pen.value(cv)
  v$cpt_var_cost <- logLik(cv)[[1]] + sum(log(seg.len(cv)))               # sum of the mbic var.norm segment costs
  cvm <- cpt.var(lev, method = "PELT", penalty = "Manual", pen.value = 10)
  v$cpt_var_manual10_cpts <- vec(cpts(cvm)); v$cpt_var_manual10_cost <- logLik(cvm)[[1]]
  z <- lev - mean(lev)
  cmz <- cpt.meanvar(z, method = "PELT", penalty = "MBIC")
  v$cpt_meanvar_centred_cpts <- vec(cpts(cmz)); v$cpt_meanvar_pen <- pen.value(cmz)
  cmean <- cpt.mean(z, method = "PELT", penalty = "MBIC", minseglen = 2)
  v$cpt_mean_centred_cpts <- vec(cpts(cmean)); v$cpt_mean_pen <- pen.value(cmean)
  bs <- cpt.mean(z, method = "BinSeg", penalty = "MBIC", Q = 10, minseglen = 2); v$cpt_mean_binseg_ncpts <- length(cpts(bs))   # TS-8, recorded only
  # --- diffinv (TS-22; TS-28 recorded) ---
  x12 <- x[1:12]
  v$diffinv_lag1 <- vec(diffinv(x12, xi = 1.5)); v$diffinv_lag2 <- vec(diffinv(x12, lag = 2, xi = c(1, -2)))
  v$diffinv_d2_zero <- vec(diffinv(x12, differences = 2)); v$diffinv_d2_xi <- vec(diffinv(x12, differences = 2, xi = c(1, 3)))
  # --- ARMAacf (TS-22; TS-14 recorded) ---
  v$armaacf_ar1 <- vec(ARMAacf(ar = 0.8, lag.max = 10)); v$armaacf_ar2 <- vec(ARMAacf(ar = c(0.5, -0.3), lag.max = 10))
  v$armaacf_ar2_pacf <- vec(ARMAacf(ar = c(0.5, -0.3), lag.max = 10, pacf = TRUE)); v$armaacf_ma2 <- vec(ARMAacf(ma = c(0.4, 0.25), lag.max = 6))
  v$armaacf_arma11 <- vec(ARMAacf(ar = 0.5, ma = 0.3, lag.max = 6)); v$armaacf_ar2_persistent <- vec(ARMAacf(ar = c(1.5, -0.56), lag.max = 10))
  # --- cpgram (TS-9) ---
  cpg <- function(series, taper = 0.1) {
    xx <- spec.taper(scale(series, TRUE, FALSE), p = taper)
    yy <- Mod(fft(xx))^2 / length(xx); yy[1L] <- 0
    nn <- length(xx); ff <- (0:(nn / 2)) / nn
    if (length(ff) %% 2 == 0) { nn <- length(ff) - 1; yy <- yy[1L:nn]; ff <- ff[1L:nn] } else yy <- yy[seq_along(ff)]
    mp <- length(ff) - 1; k <- sqrt(mp) + 0.12 + 0.11 / sqrt(mp); crit <- 1.358 / k
    cum <- cumsum(yy) / sum(yy); dev <- max(abs(cum - ff / 0.5)); stat <- dev * k
    ksurv <- if (stat < 1) 1 - .Call(stats:::C_pKS2, stat, 1e-15) else { j <- 1:100; 2 * sum((-1)^(j - 1) * exp(-2 * j^2 * stat^2)) }
    list(freq = vec(ff[-1]), cpgram = vec(cum[-1]), crit = crit, max_dev = dev, stat = stat, p = ksurv, white = as.numeric(dev <= crit))
  }
  for (nm in c("wn", "ar1")) { r <- cpg(get(nm)); for (k in names(r)) v[[paste0("cpgram_", nm, "_", k)]] <- r[[k]] }
  stopifnot(v$cpgram_wn_white == 1, v$cpgram_ar1_white == 0)
  list(values = v, reference = "stats::filter(convolution, sides = 2 | 1; recursive); stats::ar(method = 'ols' | 'burg' | 'yule-walker'); changepoint::cpt.var / cpt.meanvar / cpt.mean (PELT, MBIC); stats::diffinv; stats::ARMAacf; the body of stats::cpgram",
       call = "filter(x, f, sides = 2); ar(s, method = 'ols', intercept = FALSE); cpt.var(lev, method = 'PELT'); diffinv(x[1:12], xi = 1.5); ARMAacf(ar = c(0.5, -0.3)); cpgram(wn)", packages = "changepoint")
})
