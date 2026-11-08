# validation/r/cases/ts_holtwinters.R -- Holt-Winters fits and forecasts (T3-WP37: X-5, TS-22; planner move-in TS-23;
# CLI-23's `ts holt-winters --horizon` output is T3-WP61's). stats::HoltWinters with fixed smoothing parameters (the
# optimiser is not pinned: R's L-BFGS-B start rules differ, and p2a's clamps are TS-24, Tier 4) and
# predict(prediction.interval = TRUE): co2 to 1969 (additive), AirPassengers (multiplicative; additive without trend;
# Holt's linear trend; simple exponential smoothing). The opt_* fits record R's optimised alpha, beta, gamma for
# AirPassengers (multiplicative and additive), the legacy synthetic quarterly series and the legacy non-seasonal series
# (regenerated here exactly as validation/scripts/holtwinters_validation.R draws them); Rust refits at R's optimum,
# which supersedes validation/expected/holtwinters_*.csv.
case("ts_holtwinters_forecast", "forecasting/holtwinters", "HoltWinters additive / multiplicative / non-seasonal with fixed alpha, beta, gamma; 12-step forecasts with 95% prediction intervals", function(ctx) {
  ap <- as.numeric(AirPassengers); co <- as.numeric(window(co2, end = c(1969, 12)))
  set.seed(42); q <- (100 + 2 * (1:40)) * rep(c(0.9, 1.0, 1.2, 0.9), 10) + rnorm(40, sd = 3)   # legacy synthetic quarterly
  set.seed(123); ns <- 50 + 0.5 * (1:30) + rnorm(30, sd = 2)                                      # legacy non-seasonal
  pad <- function(v) c(v, rep(NA, 144 - length(v)))
  d <- ctx$gen_dataset("holtwinters_series", data.frame(t = 1:144, airpass = ap, co2 = pad(co), q = pad(q), ns = pad(ns)))
  apt <- ts(d$airpass, frequency = 12, start = c(1949, 1)); cot <- ts(d$co2[!is.na(d$co2)], frequency = 12, start = c(1959, 1))
  qt <- ts(d$q[!is.na(d$q)], frequency = 4); nst <- ts(d$ns[!is.na(d$ns)])
  v <- list()
  fits <- list(co2_add = HoltWinters(cot, alpha = 0.5, beta = 0.01, gamma = 0.3, seasonal = "additive"),
               airpass_mult = HoltWinters(apt, alpha = 0.3, beta = 0.05, gamma = 0.4, seasonal = "multiplicative"),
               airpass_holt = HoltWinters(apt, alpha = 0.6, beta = 0.1, gamma = FALSE),
               airpass_notrend = HoltWinters(apt, alpha = 0.3, beta = FALSE, gamma = 0.2, seasonal = "additive"),
               airpass_ses = HoltWinters(apt, alpha = 0.4, beta = FALSE, gamma = FALSE))
  for (nm in names(fits)) {
    f <- fits[[nm]]; p <- predict(f, n.ahead = 12, prediction.interval = TRUE, level = 0.95)
    v[[paste0(nm, "_sse")]] <- f$SSE; v[[paste0(nm, "_coef")]] <- named(f$coefficients)
    v[[paste0(nm, "_fit")]] <- vec(p[, "fit"]); v[[paste0(nm, "_upr")]] <- vec(p[, "upr"]); v[[paste0(nm, "_lwr")]] <- vec(p[, "lwr"])
    v[[paste0(nm, "_nres")]] <- length(residuals(f)); v[[paste0(nm, "_resvar")]] <- var(residuals(f))
  }
  p80 <- predict(fits$co2_add, n.ahead = 24, prediction.interval = TRUE, level = 0.8)
  v$co2_add_upr80_h24 <- vec(p80[, "upr"])                                                       # level and h > period
  opt <- list(opt_mult = HoltWinters(apt, seasonal = "multiplicative"), opt_add = HoltWinters(apt, seasonal = "additive"),
              opt_quarterly = HoltWinters(qt, seasonal = "multiplicative"), opt_nonseasonal = HoltWinters(nst, gamma = FALSE))
  for (nm in names(opt)) {
    f <- opt[[nm]]
    v[[paste0(nm, "_par")]] <- named(c(alpha = unname(f$alpha), beta = if (is.logical(f$beta)) NA else unname(f$beta),
                                        gamma = if (is.logical(f$gamma)) NA else unname(f$gamma)))
    v[[paste0(nm, "_sse")]] <- f$SSE; v[[paste0(nm, "_coef")]] <- named(f$coefficients)
  }
  v$opt_mult_fit <- vec(predict(opt$opt_mult, n.ahead = 12))
  list(values = v, reference = "stats::HoltWinters(alpha, beta, gamma fixed or optimised); stats::predict.HoltWinters(n.ahead = 12, prediction.interval = TRUE)",
       call = "HoltWinters(AirPassengers, alpha = 0.3, beta = 0.05, gamma = 0.4, seasonal = 'multiplicative')", packages = character(0))
})
