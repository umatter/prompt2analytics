# validation/r/cases/forecasting_arima.R -- stats::arima (CSS) fits and predict / forecast::forecast
# references (T2-WP16: TS-1, TS-17), and lmtest::grangertest on an integer series (X-2).
#
# forecast::Arima minimises the CSS objective with optim's BFGS and numerical gradients
# (ndeps = 1e-3), so even with reltol = 1e-14 its coefficients sit up to 2.3e-4 (relative) away
# from the exact CSS optimum (ARIMA(1,2,1): max |gradient| 0.43). Each fit is therefore re-solved
# by Gauss-Newton on R's own CSS residuals (stats::arima(fixed = ...)) with numDeriv's Richardson
# Jacobian. The package values stay in the case as cross-checks (untagged keys); the re-solved
# ones carry the suffix _tight and are what the Rust tests pin (T2 design 2.3). Forecasts at the
# tight optimum come from forecast::forecast (stats::predict.Arima, Kalman state from
# stats::arima's arimaSS) on the refitted object, with forecast::Arima's sigma2 = SSR / (n - d - k).
suppressPackageStartupMessages({ library(forecast); library(numDeriv); library(lmtest) })

.arima_series <- function(ctx) {
  set.seed(20261021)
  n <- 300
  a <- 100 + as.numeric(arima.sim(list(ar = 0.6), n))                       # AR(1) around 100
  c_rw <- 50 + cumsum(rnorm(n))                                             # random walk
  m <- 20 + as.numeric(arima.sim(list(ar = 0.5, ma = 0.4), n))              # ARMA(1,1) around 20
  b <- 10 + as.numeric(arima.sim(list(ar = c(0.5, -0.3)), n))               # AR(2)
  dr <- 5 + cumsum(0.3 + as.numeric(arima.sim(list(ma = -0.4), n)))         # drift + MA(1) in differences
  a8 <- round(a, 8)
  # Columns for the extraction tests (no further random draws, so a..dr are unchanged):
  # ai  whole numbers, written without a decimal point -> Int64 in polars, integer in R;
  # a_na, ai_na  missing cells at rows 150 and 151 (empty fields: null in polars, NA in R);
  # lab  text.
  ai <- round(10 * a8)
  a_na <- a8; a_na[c(150, 151)] <- NA
  ai_na <- ai; ai_na[c(150, 151)] <- NA
  ctx$gen_dataset("arima_series", data.frame(t = seq_len(n), a = a8, c = round(c_rw, 8),
                                             m = round(m, 8), b = round(b, 8), dr = round(dr, 8),
                                             ai = ai, a_na = a_na, ai_na = ai_na, lab = sprintf("q%03d", seq_len(n)),
                                             stringsAsFactors = FALSE))
}

.ctl <- list(reltol = 1e-14, maxit = 10000)

# CSS residuals for given coefficients (ar, ma, [intercept], [drift]) from R's own C_ARIMA_CSS.
.css_resid <- function(par, x, order, include.mean, drift) {
  xr <- if (drift) cbind(drift = seq_along(x)) else NULL
  f <- suppressWarnings(arima(x, order = order, fixed = par, method = "CSS", transform.pars = FALSE,
                              include.mean = include.mean, xreg = xr))
  as.numeric(residuals(f))
}
# Gauss-Newton on the CSS residuals with numDeriv's Richardson Jacobian, started at the package optimum.
.css_tight <- function(x, order, start, include.mean, drift) {
  f <- function(p) .css_resid(p, x, order, include.mean, drift)
  if (!length(start)) return(list(par = numeric(0), ssr = sum(f(numeric(0))^2), max_abs_grad = 0))
  par <- start
  for (i in 1:40) {
    J <- numDeriv::jacobian(f, par)
    step <- drop(solve(crossprod(J), -crossprod(J, f(par))))
    par <- par + step
    if (max(abs(step) / pmax(abs(par), 1)) < 1e-12) break
  }
  e <- f(par)
  J <- numDeriv::jacobian(f, par)
  list(par = par, ssr = sum(e^2), max_abs_grad = max(abs(crossprod(J, e))))
}

.fc_values <- function(fc, prefix, suffix) {
  out <- list()
  out[[paste0(prefix, "_mean", suffix)]] <- vec(as.numeric(fc$mean))
  out[[paste0(prefix, "_se", suffix)]] <- vec(as.numeric(fc$upper[, 2] - fc$mean) / qnorm(0.975))   # = predict.Arima se
  out[[paste0(prefix, "_lower80", suffix)]] <- vec(as.numeric(fc$lower[, 1])); out[[paste0(prefix, "_upper80", suffix)]] <- vec(as.numeric(fc$upper[, 1]))
  out[[paste0(prefix, "_lower95", suffix)]] <- vec(as.numeric(fc$lower[, 2])); out[[paste0(prefix, "_upper95", suffix)]] <- vec(as.numeric(fc$upper[, 2]))
  out
}

.fit_values <- function(x, order, prefix, h = 5, include.mean = TRUE, drift = FALSE) {
  if (drift) {
    f <- forecast::Arima(x, order = order, include.drift = TRUE, method = "CSS", optim.control = .ctl)
  } else {
    f <- forecast::Arima(x, order = order, include.mean = include.mean, method = "CSS", optim.control = .ctl)
  }
  fc <- forecast::forecast(f, h = h, level = c(80, 95))
  out <- list()
  if (length(coef(f))) out[[paste0(prefix, "_coef")]] <- named(coef(f))
  out[[paste0(prefix, "_sigma2")]] <- f$sigma2
  out[[paste0(prefix, "_loglik")]] <- f$loglik
  out <- c(out, .fc_values(fc, prefix, ""))
  out[[paste0(prefix, "_n_cond")]] <- order[2] + order[1]
  # The exact CSS optimum (see the header) and everything derived from it.
  mean_used <- include.mean && order[2] == 0                       # forecast::Arima drops the mean when d > 0
  tt <- .css_tight(x, order, unname(coef(f)), mean_used, drift)
  k <- length(coef(f)); n <- length(x)
  ft <- suppressWarnings(forecast::Arima(x, order = order, include.mean = mean_used, include.drift = drift,
                                         method = "CSS", fixed = tt$par, transform.pars = FALSE))
  ft$sigma2 <- tt$ssr / (n - order[2] - k)                         # forecast::Arima's sigma2 at the tight optimum
  fct <- forecast::forecast(ft, h = h, level = c(80, 95))
  if (k) out[[paste0(prefix, "_coef_tight")]] <- named(setNames(tt$par, names(coef(f))))
  out[[paste0(prefix, "_ssr_tight")]] <- tt$ssr
  out[[paste0(prefix, "_sigma2_tight")]] <- ft$sigma2
  out[[paste0(prefix, "_loglik_tight")]] <- ft$loglik
  out[[paste0(prefix, "_aic_css_tight")]] <- -2 * ft$loglik + 2 * (k + 1)   # R reports NA for CSS fits
  out <- c(out, .fc_values(fct, prefix, "_tight"))
  out[[paste0(prefix, "_resid_head_tight")]] <- vec(head(as.numeric(residuals(ft)), order[2] + order[1] + 2))
  out[[paste0(prefix, "_resid_tail_tight")]] <- vec(tail(as.numeric(residuals(ft)), 3))
  out[[paste0(prefix, "_max_abs_grad_tight")]] <- tt$max_abs_grad
  out[[paste0(prefix, "_n_coef")]] <- k
  out
}

case("arima_css_forecast", "forecasting/arima", "ARIMA CSS fits and forecasts with prediction intervals (forecast::Arima, method = 'CSS')", function(ctx) {
  d <- .arima_series(ctx)
  v <- c(.fit_values(d$a, c(1, 0, 0), "a_100"),
         .fit_values(d$a, c(1, 0, 1), "a_101"),
         .fit_values(d$b, c(2, 0, 0), "b_200"),
         .fit_values(d$m, c(1, 0, 1), "m_101"),
         .fit_values(d$m, c(0, 0, 2), "m_002"),
         .fit_values(d$c, c(0, 1, 1), "c_011", include.mean = FALSE),
         .fit_values(d$c, c(1, 1, 0), "c_110", include.mean = FALSE),
         .fit_values(d$c, c(0, 1, 0), "c_010", include.mean = FALSE),
         .fit_values(d$dr, c(0, 1, 1), "dr_011_drift", drift = TRUE),
         .fit_values(d$dr, c(1, 2, 1), "dr_121", include.mean = FALSE),
         .fit_values(d$dr, c(1, 0, 0), "dr_100_drift", drift = TRUE),
         .fit_values(d$m, c(1, 0, 0), "m_100_nomean", include.mean = FALSE),
         .fit_values(d$ai, c(1, 0, 1), "ai_101"))
  stopifnot(is.integer(d$ai), is.integer(d$ai_na), sum(is.na(d$a_na)) == 2, sum(is.na(d$ai_na)) == 2, is.character(d$lab))
  v$a_last <- tail(d$a, 1); v$c_last <- tail(d$c, 1)
  list(values = v,
       reference = "forecast::Arima(method = 'CSS', optim.control = list(reltol = 1e-14)); forecast::forecast(h = 5, level = c(80, 95)); stats::predict.Arima (se); *_tight: Gauss-Newton on stats::arima(fixed = ...) CSS residuals with numDeriv::jacobian",
       call = "forecast(Arima(x, order = c(p, d, q), method = 'CSS'), h = 5)", packages = c("forecast", "numDeriv"))
})

# Forecaster isolated from the fit: R's predict() for fixed coefficients.
case("arima_fixed_forecast", "forecasting/arima", "ARIMA forecasts for given coefficients (stats::arima(fixed = ...), predict)", function(ctx) {
  d <- .arima_series(ctx)
  v <- list()
  specs <- list(a_100 = list(x = d$a, order = c(1, 0, 0), fixed = c(0.6, 100)),
                m_101 = list(x = d$m, order = c(1, 0, 1), fixed = c(0.5, 0.4, 20)),
                c_011 = list(x = d$c, order = c(0, 1, 1), fixed = c(0.05)),
                b_200 = list(x = d$b, order = c(2, 0, 0), fixed = c(0.5, -0.3, 10)),
                # 40 observations and an MA coefficient of -0.8: predict.Arima's Kalman state differs
                # from the state of the CSS recursion (innovations 0 before the sample).
                m40_101 = list(x = d$m[1:40], order = c(1, 0, 1), fixed = c(0.5, -0.8, 20)))
  for (k in names(specs)) {
    s <- specs[[k]]
    f <- arima(s$x, order = s$order, fixed = s$fixed, method = "CSS", transform.pars = FALSE,
               include.mean = s$order[2] == 0)
    p <- predict(f, n.ahead = 8)
    v[[paste0(k, "_fixed")]] <- vec(s$fixed); v[[paste0(k, "_sigma2")]] <- f$sigma2
    v[[paste0(k, "_pred")]] <- vec(as.numeric(p$pred)); v[[paste0(k, "_se")]] <- vec(as.numeric(p$se))
    v[[paste0(k, "_resid_tail")]] <- vec(tail(as.numeric(residuals(f)), 3))
    v[[paste0(k, "_n")]] <- length(s$x)
  }
  list(values = v, reference = "stats::arima(method = 'CSS', fixed = coefficients, transform.pars = FALSE); stats::predict.Arima",
       call = "predict(arima(x, order, fixed = ..., method = 'CSS'), n.ahead = 8)", packages = character(0))
})

# A seasonal series that ARIMA(1,1,1) with drift misspecifies (TS-35): the unified benchmark's
# time-series DGP (trend 0.01 t, period-12 sine of amplitude 2, uniform noise), n = 1000, and its
# first 100 values (runif draws in order, so they are the n = 100 series). The fit leaves residual
# autocorrelation (about 0.34 at lag 2, 0.4-0.5 at lag 12), so J'J underestimates the curvature of
# the CSS objective in the MA coefficient and the Gauss-Newton iteration has an eigenvalue near -1
# at the optimum: the case asserts that, so it keeps exercising the oscillation. .css_tight is
# Gauss-Newton and does not contract there, and numDeriv's gradient in the drift is off by 2.5e-8
# at n = 100 (8e-7 relative in the drift after a Newton solve), so the exact optimum comes from
# Newton's method on the analytic gradient of the CSS residuals (their derivatives follow the
# residual recursion), each pass checked against R's own residuals (stats::arima(fixed = ...)).
.css_d1_drift <- function(par, x, p, q) {
  phi <- par[seq_len(p)]; theta <- par[p + seq_len(q)]; drift <- par[p + q + 1]
  w <- diff(x) - drift; rows <- (p + 1):length(w)
  ma <- function(u) if (q) as.numeric(stats::filter(u, -theta, method = "recursive")) else u
  ar <- if (p) drop(sapply(seq_len(p), function(i) w[rows - i]) %*% phi) else 0
  e <- ma(w[rows] - ar)
  e_r <- .css_resid(par, x, c(p, 1, q), FALSE, TRUE)[-(1:(p + 1))]
  stopifnot(max(abs(e - e_r)) < 1e-12 * max(1, abs(e)))
  # Derivatives of e over the rows after the p conditioning ones (before them all are 0).
  J <- cbind(if (p) sapply(seq_len(p), function(i) ma(-w[rows - i])),
             if (q) sapply(seq_len(q), function(j) ma(-c(rep(0, j), e)[seq_along(rows)])),
             ma(rep(-(1 - sum(phi)), length(rows))))
  list(e = e, J = J, grad = 2 * drop(crossprod(J, e)))
}
.css_newton_d1_drift <- function(x, p, q, start) {
  g <- function(par) .css_d1_drift(par, x, p, q)$grad
  par <- start
  for (i in 1:30) {
    H <- numDeriv::jacobian(g, par)
    step <- -drop(solve((H + t(H)) / 2, g(par)))
    par <- par + step
    if (max(abs(step) / pmax(abs(par), 1)) < 1e-14) break
  }
  H <- numDeriv::jacobian(g, par); H <- (H + t(H)) / 2
  r <- .css_d1_drift(par, x, p, q)
  list(par = par, ssr = sum(r$e^2), max_abs_grad = max(abs(r$grad)),
       gn_eigen = Re(eigen(diag(length(par)) - solve(2 * crossprod(r$J), H), only.values = TRUE)$values),
       acf = acf(r$e, lag.max = 12, plot = FALSE)$acf[-1])
}

case("arima_misspecified_seasonal", "forecasting/arima", "ARIMA(1,1,1) with drift by CSS on a seasonal series it misspecifies (forecast::Arima, method = 'CSS')", function(ctx) {
  set.seed(42)
  n_all <- 1000; t0 <- 0:(n_all - 1)
  d <- ctx$gen_dataset("arima_misspecified_seasonal",
                       data.frame(y = 0.01 * t0 + 2 * sin(t0 * pi / 6) + runif(n_all, 0, 0.5)))
  v <- list()
  for (n in c(100, 1000)) {
    x <- d$y[seq_len(n)]; key <- paste0("n", n)
    # The call as users (and the benchmark) make it: optim's default reltol.
    f <- forecast::Arima(x, order = c(1, 1, 1), include.drift = TRUE, method = "CSS")
    tt <- .css_newton_d1_drift(x, 1, 1, unname(coef(f)))
    stopifnot(min(tt$gn_eigen) < -0.99, tt$max_abs_grad < 1e-10)
    k <- length(coef(f))
    ft <- suppressWarnings(forecast::Arima(x, order = c(1, 1, 1), include.drift = TRUE, method = "CSS",
                                           fixed = tt$par, transform.pars = FALSE))
    v[[paste0(key, "_coef")]] <- named(coef(f))
    v[[paste0(key, "_coef_tight")]] <- named(setNames(tt$par, names(coef(f))))
    v[[paste0(key, "_ssr_tight")]] <- tt$ssr
    v[[paste0(key, "_sigma2_tight")]] <- tt$ssr / (n - 1 - k)               # forecast::Arima's sigma2
    v[[paste0(key, "_loglik_tight")]] <- ft$loglik
    v[[paste0(key, "_aic_css_tight")]] <- -2 * ft$loglik + 2 * (k + 1)      # R reports NA for CSS fits
    v[[paste0(key, "_max_abs_grad_tight")]] <- tt$max_abs_grad
    v[[paste0(key, "_gn_eigen_min_tight")]] <- min(tt$gn_eigen)
    v[[paste0(key, "_resid_acf_lag2_tight")]] <- tt$acf[2]
    v[[paste0(key, "_resid_acf_lag12_tight")]] <- tt$acf[12]
    v[[paste0(key, "_n")]] <- n
  }
  list(values = v,
       reference = "forecast::Arima(order = c(1, 1, 1), include.drift = TRUE, method = 'CSS'); *_tight: Newton's method on the analytic gradient of the stats::arima(fixed = ...) CSS residuals, Hessian by numDeriv::jacobian",
       call = "Arima(y[1:n], order = c(1, 1, 1), include.drift = TRUE, method = 'CSS')", packages = c("forecast", "numDeriv"))
})

# Granger causality on an integer series (lmtest::grangertest, default method; no missing cells in
# these columns). p2a refuses missing values here: grangertest's two interfaces disagree on them
# (the formula method's model.frame drops the NA rows and then lags the spliced series; the default
# method lags first and drops every lagged row that touches an NA, including rows where only the
# contemporaneous cause is missing).
case("ts_granger_int", "econometrics/timeseries", "Granger causality F test, integer dependent series (lmtest::grangertest)", function(ctx) {
  d <- .arima_series(ctx)
  stopifnot(is.integer(d$ai))
  v <- list()
  for (ord in 2:3) {
    g <- grangertest(d$m, d$ai, order = ord)        # does m Granger-cause ai?
    df2 <- g$Res.Df[1]
    v[[paste0("lag", ord, "_f")]] <- g$F[2]; v[[paste0("lag", ord, "_p")]] <- g$`Pr(>F)`[2]
    v[[paste0("lag", ord, "_df1")]] <- -g$Df[2]; v[[paste0("lag", ord, "_df2")]] <- df2
    v[[paste0("lag", ord, "_n_used")]] <- df2 + 2 * ord + 1
  }
  list(values = v, reference = "lmtest::grangertest(x = m, y = ai, order = 2 | 3)",
       call = "grangertest(d$m, d$ai, order = 2)", packages = "lmtest")
})
