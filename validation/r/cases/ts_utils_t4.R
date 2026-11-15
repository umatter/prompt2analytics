# validation/r/cases/ts_utils_t4.R -- T4-WP24 references for the time-series utilities.
#  * stats::ARMAacf for mixed ARMA(p, q) (ACF and PACF), persistent AR, lag.max below p, pure MA past q
#    (TS-14). ARMAacf does not check stationarity; p2a refuses a non-stationary AR part (not a case value).
#  * stats::diffinv with differences >= 2 and non-zero xi, lag 1, 2 and 3 (TS-28).
#  * stats::runmed(endrule = "median") = smoothEnds(runmed(endrule = "keep"), k) with Tukey's end-point rule,
#    k = 3, 5, 7, 9 and k = n (TS-34).
#  * stats::arima.sim with given innovations (start.innov, innov, n.start), ARMA(2, 1) and ARIMA(1, 1, 1),
#    and R's default burn-in n.start = p + q + ceiling(6 / log(minroots)) (TS-33).
#  * stats::ar(aic = FALSE, order.max = 20) on n = 50, above the default order.max 16 (TS-30); ols with
#    intercept = FALSE, p2a's convention (validation/timeseries/ar.md).
#  * stats::stl(s.window = "periodic") and stl(s.window = 13) on a monthly series and on the same series
#    at level 1e8 (TS-26).
case("ts_utils_t4", "forecasting/tsutils", "ARMAacf (mixed, persistent), diffinv (d >= 2 with xi), runmed endrule median, arima.sim with given innovations and its burn-in, ar with a fixed order above the default order.max, stl periodic and s.window = 13", function(ctx) {
  set.seed(20261115)
  x50 <- round(as.numeric(arima.sim(list(ar = c(0.6, -0.2)), 50)), 6)
  e <- round(rnorm(40), 6)
  tt <- 1:144
  y <- round(1000 + 2 * tt + 50 * sin(2 * pi * tt / 12) + 20 * cos(2 * pi * tt / 6) + rnorm(144, 0, 8), 6)
  na <- function(v) c(v, rep(NA, 144 - length(v)))
  d <- ctx$gen_dataset("ts_utils_t4", data.frame(t = tt, x50 = na(x50), e = na(e), y = y))
  x50 <- d$x50[1:50]; e <- d$e[1:40]; y <- d$y
  v <- list()
  # --- ARMAacf (TS-14) ---
  v$acf_arma11 <- vec(ARMAacf(ar = 0.5, ma = 0.3, lag.max = 6))
  v$acf_arma12 <- vec(ARMAacf(ar = 0.9, ma = c(-0.5, 0.4), lag.max = 6))
  v$acf_arma21 <- vec(ARMAacf(ar = c(0.5, 0.3), ma = 0.4, lag.max = 6))
  v$pacf_arma21 <- vec(ARMAacf(ar = c(0.5, 0.3), ma = 0.4, lag.max = 6, pacf = TRUE))          # lags 1..6
  v$acf_ar2_persistent <- vec(ARMAacf(ar = c(1.5, -0.56), lag.max = 10))
  v$acf_ar1_099 <- vec(ARMAacf(ar = 0.99, lag.max = 5))
  v$acf_arma32 <- vec(ARMAacf(ar = c(0.4, -0.2, 0.3), ma = c(0.5, 0.25), lag.max = 8))
  v$pacf_arma32 <- vec(ARMAacf(ar = c(0.4, -0.2, 0.3), ma = c(0.5, 0.25), lag.max = 8, pacf = TRUE))
  v$acf_arma13_lag2 <- vec(ARMAacf(ar = 0.6, ma = c(0.3, -0.2, 0.1), lag.max = 2))        # lag.max < q + 1
  v$acf_ar3_lag1 <- vec(ARMAacf(ar = c(0.3, 0.2, 0.1), lag.max = 1))                      # lag.max < p
  v$acf_ma3_lag5 <- vec(ARMAacf(ma = c(0.6, -0.3, 0.2), lag.max = 5))                     # zeros past q
  # --- diffinv (TS-28) ---
  v$dinv_d2_xi <- vec(diffinv(c(2, 3, 4), lag = 1, differences = 2, xi = c(1, 3)))
  v$dinv_l2_d2 <- vec(diffinv(c(1, 2, 3, 4), lag = 2, differences = 2, xi = c(1, 2, 3, 4)))
  v$dinv_d3 <- vec(diffinv(c(0.5, -1, 2, 0.25, 3), lag = 1, differences = 3, xi = c(1, -2, 0.5)))
  v$dinv_l3_d2 <- vec(diffinv(c(0.5, -1, 2, 0.25, 3, 1.5, -0.75), lag = 3, differences = 2, xi = c(1, -2, 0.5, 4, 0, 2.5)))
  stopifnot(inherits(try(diffinv(c(2, 3, 4), 1, 2, xi = 1), silent = TRUE), "try-error"))   # 'xi' does not have the right length
  # --- runmed endrule = "median" (TS-34) ---
  z <- c(3.1, -0.4, 2.2, 5.0, 1.7, 0.3, 4.4, 2.9, -1.2, 0.8, 3.3, 2.0, 6.1)
  v$rm_k3_a <- vec(runmed(c(5, 1, 2, 3, 10, 4, 6), 3, endrule = "median"))
  v$rm_k5_b <- vec(runmed(c(5, 1, 2, 3, 10, 4, 6, 8, 0), 5, endrule = "median"))
  v$rm_k3_doc <- vec(runmed(c(1, 2, 100, 4, 5, 6, 7), 3, endrule = "median"))
  for (k in c(3, 7, 9, 13)) v[[paste0("rm_z_k", k)]] <- vec(runmed(z, k, endrule = "median"))
  v$rm_n5_k5 <- vec(runmed(c(3, 1, 2, 9, 4), 5, endrule = "median"))
  # --- arima.sim (TS-33) ---
  ns <- function(ar, ma) { p <- length(ar); q <- length(ma); mr <- if (p) min(Mod(polyroot(c(1, -ar)))) else Inf
                           p + q + ifelse(p > 0, ceiling(6 / log(mr)), 0) }
  v$nstart <- vec(c(ns(0.5, numeric()), ns(0.99, numeric()), ns(0.95, numeric()), ns(c(0.5, -0.3), 0.4),
                    ns(numeric(), c(0.4, 0.2)), ns(c(1.5, -0.56), numeric()), ns(c(0.5, 0), numeric())))
  stopifnot(all(as.numeric(v$nstart) == c(10, 598, 118, 13, 2, 29, 11)))
  m <- 13                                                                                  # = ns(c(0.5, -0.3), 0.4)
  v$sim_arma21 <- vec(arima.sim(list(ar = c(0.5, -0.3), ma = 0.4), n = 20, innov = e[(m + 1):(m + 20)], n.start = m, start.innov = e[1:m]))
  m1 <- ns(0.5, 0.4)                                                                       # 11
  v$sim_arima111_nstart <- m1
  v$sim_arima111 <- vec(arima.sim(list(order = c(1, 1, 1), ar = 0.5, ma = 0.4), n = 20, innov = e[(m1 + 1):(m1 + 20)], n.start = m1, start.innov = e[1:m1]))   # n + 1 values, the first 0
  for (bad in list(1.2, 1, c(0.5, 0.5)))
    stopifnot(inherits(try(arima.sim(list(ar = bad), 10), silent = TRUE), "try-error"))  # 'ar' part of model is not stationary
  # --- ar with a fixed order above the default order.max (TS-30) ---
  stopifnot(min(49, floor(10 * log10(50))) == 16)
  for (meth in c("yule-walker", "burg", "ols")) {
    # intercept = FALSE for ols (ar.ols defaults intercept = demean); p2a's ar.ols fits no constant after demeaning
    f <- if (meth == "ols") ar(x50, aic = FALSE, order.max = 20, method = meth, demean = TRUE, intercept = FALSE)
         else ar(x50, aic = FALSE, order.max = 20, method = meth, demean = TRUE)
    tag <- c("yule-walker" = "yw", burg = "burg", ols = "ols")[[meth]]
    stopifnot(f$order == 20)
    v[[paste0("ar_", tag, "_fixed20_coef")]] <- vec(as.numeric(f$ar)); v[[paste0("ar_", tag, "_fixed20_var")]] <- as.numeric(f$var.pred)
  }
  v$ar_x_mean <- mean(x50)
  stopifnot(inherits(try(ar(x50, aic = FALSE, order.max = 50, method = "burg"), silent = TRUE), "try-error"))   # 'order.max' must be < 'n.used'
  # --- stl (TS-26) ---
  for (lv in c(0, 1e8)) {
    yy <- ts(y + lv, frequency = 12)
    tag <- if (lv == 0) "" else "_big"
    p <- stl(yy, s.window = "periodic"); w <- stl(yy, s.window = 13)
    v[[paste0("stl_periodic", tag, "_seasonal")]] <- vec(p$time.series[, "seasonal"])
    v[[paste0("stl_periodic", tag, "_trend")]] <- vec(p$time.series[, "trend"])
    v[[paste0("stl_periodic", tag, "_remainder")]] <- vec(p$time.series[, "remainder"])
    v[[paste0("stl_swin13", tag, "_seasonal")]] <- vec(w$time.series[, "seasonal"])
    v[[paste0("stl_swin13", tag, "_trend")]] <- vec(w$time.series[, "trend"])
    v[[paste0("stl_swin13", tag, "_remainder")]] <- vec(w$time.series[, "remainder"])
  }
  v$stl_sd <- sd(y)
  list(values = v, dataset = "validation/datasets/generated/ts_utils_t4.csv",
       reference = "stats::ARMAacf; stats::diffinv; stats::runmed(endrule = 'median') (smoothEnds); stats::arima.sim(innov, n.start, start.innov); stats::ar(aic = FALSE, order.max = 20, method = 'yule-walker' | 'burg' | 'ols'); stats::stl(s.window = 'periodic' | 13)",
       call = "ARMAacf(ar = 0.5, ma = 0.3, lag.max = 6); diffinv(c(2, 3, 4), differences = 2, xi = c(1, 3)); runmed(z, 7, endrule = 'median'); arima.sim(list(ar = c(0.5, -0.3), ma = 0.4), 20, innov, n.start = 13, start.innov); ar(x50, aic = FALSE, order.max = 20); stl(ts(y, frequency = 12), s.window = 'periodic')",
       packages = character(0))
})
