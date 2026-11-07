# validation/r/cases/ts_statespace.R -- state-space references (T3-WP36: TS-25, TS-27).
#  * stats::StructTS(type = "level" | "trend" | "BSM", fixed) on Nile and log10(UKgas). Recorded:
#    the package values (`<fit>_coef`, `<fit>_loglik`; optim L-BFGS-B stops at factr = 1e7 on a
#    finite-difference gradient) and the tight optimum of StructTS's own objective (`*_tight`): its
#    getLike written out below (checked against stats::KalmanLike), minimised by Newton's method on
#    the exact gradient (the differentiated Kalman filter; Harvey 1990, sec. 3.4.6), with the
#    Karush-Kuhn-Tucker sign checked for every variance on its bound 0. At the tight optimum: the
#    filtered states (stats::KalmanRun, = fitted()), the smoothed states (stats::KalmanSmooth with
#    nit = -1, = tsSmooth()) and the standardised residuals (= residuals()).
#    With gaps R's loglik scales the data term by length(y)/n_obs; `*_loglik_obs_tight` is the
#    observed-data Gaussian log-likelihood p2a reports, from the same objective.
#  * a Kalman filter with state intercept c and observation intercept d (local linear trend): the
#    model augmented with a constant state (T* = [T c; 0 1], Z* = (Z, d)) run through
#    stats::KalmanRun / KalmanSmooth / KalmanForecast with nit = -1 (predict first from a_0, P_0, the
#    convention of p2a's kalman_filter); the recursion written out is checked against it.
#    KFAS is not installed.
# Package copy (T3-WP36, canonical; supersedes design Appendix A.32): adds the trend fit, the tight
# optimum, states and residuals, a series with gaps (column nile_na); the Kalman case starts from
# a_0 = 0, P_0 = 10 I (predict first) and adds a singular-covariance smoother check.

case("kalman_intercepts", "forecasting/kalman", "Kalman filter, smoother and forecast with state intercept c and observation intercept d (local linear trend)", function(ctx) {
  set.seed(20261124)
  n <- 50; Tm <- matrix(c(1, 0, 1, 1), 2); Z <- c(1, 0); cvec <- c(0.1, 0); dsc <- 2; Q <- diag(c(0.3, 0.01)); H <- 0.5
  a <- c(0, 0.2); y <- numeric(n)
  for (t in 1:n) { y[t] <- dsc + sum(Z * a) + rnorm(1, 0, sqrt(H)); a <- cvec + Tm %*% a + MASS::mvrnorm(1, c(0, 0), Q) }
  d <- ctx$gen_dataset("kalman_intercepts", data.frame(t = 1:n, y = round(y, 10)))
  y <- d$y
  a0 <- c(0, 0); P0 <- diag(c(10, 10))
  # the model augmented with a constant state carries c and d through stats' C code
  aug <- list(T = rbind(cbind(Tm, cvec), c(0, 0, 1)), Z = c(Z, dsc), a = c(a0, 1), P = rbind(cbind(P0, 0), 0),
              V = rbind(cbind(Q, 0), 0), h = H, Pn = matrix(0, 3, 3))
  run <- KalmanRun(y, aug, nit = -1L, update = TRUE)
  kl <- KalmanLike(y, aug, nit = -1L)
  ll <- -0.5 * n * (kl$s2 + 2 * kl$Lik - log(kl$s2)) - 0.5 * n * log(2 * pi)
  fc <- KalmanForecast(5L, attr(run, "mod"))
  sm <- KalmanSmooth(y, aug, nit = -1L)
  # the recursion written out (predict first): innovations and their variances, and a cross-check
  at <- a0; Pt <- P0; ll2 <- 0; filt <- matrix(0, n, 2); innov <- numeric(n); fvar <- numeric(n)
  for (t in 1:n) {
    an <- cvec + drop(Tm %*% at); Pn <- Tm %*% Pt %*% t(Tm) + Q
    v_t <- y[t] - dsc - sum(Z * an); F <- drop(t(Z) %*% Pn %*% Z) + H; M <- drop(Pn %*% Z)
    at <- an + M * v_t / F; Pt <- Pn - outer(M, M) / F
    ll2 <- ll2 - 0.5 * (log(2 * pi) + log(F) + v_t^2 / F); filt[t, ] <- at; innov[t] <- v_t; fvar[t] <- F
  }
  stopifnot(abs(ll - ll2) <= 1e-10 * abs(ll), max(abs(filt - run$states[, 1:2])) <= 1e-10,
            max(abs(innov / sqrt(fvar) - run$resid)) <= 1e-10)
  # the same model without intercepts (the value p2a returned before TS-25 was fixed)
  noint <- list(T = Tm, Z = Z, a = a0, P = P0, V = Q, h = H, Pn = matrix(0, 2, 2))
  kl0 <- KalmanLike(y, noint, nit = -1L)
  ll0 <- -0.5 * n * (kl0$s2 + 2 * kl0$Lik - log(kl0$s2)) - 0.5 * n * log(2 * pi)
  # a deterministic slope (zero slope variance, zero prior slope variance): every P_{t+1|t} is singular
  fixed_slope <- list(T = Tm, Z = Z, a = c(0, -0.1), P = diag(c(10, 0)), V = diag(c(0.3, 0)), h = H, Pn = matrix(0, 2, 2))
  smf <- KalmanSmooth(y, fixed_slope, nit = -1L)
  stopifnot(max(abs(smf$smooth[, 2] + 0.1)) < 1e-12)
  v <- list(loglik = ll, filtered_level = vec(run$states[, 1]), filtered_slope = vec(run$states[, 2]),
            innovations = vec(innov), innovation_var = vec(fvar),
            smoothed_level = vec(sm$smooth[, 1]), smoothed_slope = vec(sm$smooth[, 2]),
            smoothed_level_var = vec(sm$var[, 1, 1]),
            forecast = vec(fc$pred), forecast_var = vec(fc$var), loglik_no_intercept = ll0,
            smoothed_level_fixed_slope = vec(smf$smooth[, 1]))
  list(values = v, reference = "stats::KalmanRun / KalmanLike / KalmanSmooth / KalmanForecast (nit = -1) on the model augmented with a constant state; the recursion with c and d written out (Durbin & Koopman 2012, sec. 4.3 with c_t, d_t), a_0 = 0, P_0 = 10 I",
       call = "y_t = d + Z'a_t + eps_t; a_{t+1} = c + T a_t + eta_t; KalmanRun(y, augmented, nit = -1)", packages = "MASS")
})
