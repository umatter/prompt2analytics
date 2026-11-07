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

.sts_objective <- function(y, mod) {           # StructTS's getLike: 0.5 * (ssq + sumlog) / nu
  kl <- KalmanLike(y, mod, nit = -1L)
  0.5 * (kl$s2 + 2 * kl$Lik - log(kl$s2))
}
.sts_set <- function(mod, p, vx, np) {         # p: variances in units of vx, model order
  mod$V[cbind(1:np, 1:np)] <- p[1:np] * vx; mod$h <- p[np + 1] * vx; mod
}
# Objective and exact gradient w.r.t. the free parameters: predict-first Kalman recursion and its
# derivative (dV = vx e_i e_i' for state variance i, dh = vx for epsilon); P_0 does not depend on p.
.sts_fg <- function(par, y, mod, vx, np, cf, mask) {
  p <- cf; p[mask] <- par
  if (all(p == 0)) return(list(f = 1000, g = rep(0, sum(mask))))
  m <- length(mod$a); V <- matrix(0, m, m); V[cbind(1:np, 1:np)] <- p[1:np] * vx; h <- p[np + 1] * vx
  Tm <- mod$T; Z <- mod$Z; a <- mod$a; P <- mod$P
  k <- which(mask); nk <- length(k)
  da <- matrix(0, m, nk); dP <- array(0, c(m, m, nk))
  ssq <- 0; sumlog <- 0; nu <- 0; gs <- rep(0, nk); gl <- rep(0, nk)
  for (t in seq_along(y)) {
    an <- drop(Tm %*% a); Pn <- Tm %*% P %*% t(Tm) + V
    dan <- Tm %*% da; dPn <- dP
    for (j in seq_len(nk)) {
      dPn[, , j] <- Tm %*% dP[, , j] %*% t(Tm)
      if (k[j] <= np) dPn[k[j], k[j], j] <- dPn[k[j], k[j], j] + vx
    }
    if (!is.na(y[t])) {
      nu <- nu + 1
      v <- y[t] - sum(Z * an); M <- drop(Pn %*% Z); Fv <- h + sum(Z * M)
      ssq <- ssq + v^2 / Fv; sumlog <- sumlog + log(Fv)
      for (j in seq_len(nk)) {
        dv <- -sum(Z * dan[, j]); dM <- drop(dPn[, , j] %*% Z)
        dF <- sum(Z * dM) + (if (k[j] == np + 1) vx else 0)
        gs[j] <- gs[j] + 2 * v * dv / Fv - v^2 * dF / Fv^2
        gl[j] <- gl[j] + dF / Fv
        da[, j] <- dan[, j] + (dM * v + M * dv) / Fv - M * v * dF / Fv^2
        dP[, , j] <- dPn[, , j] - (outer(dM, M) + outer(M, dM)) / Fv + outer(M, M) * dF / Fv^2
      }
      a <- an + M * v / Fv; P <- Pn - outer(M, M) / Fv
    } else { a <- an; P <- Pn; da <- dan; dP <- dPn }
  }
  list(f = 0.5 * (ssq / nu + sumlog / nu), g = 0.5 * (gs + gl) / nu)
}
.sts_fit <- function(x, type, fixed = NULL) {
  y <- as.numeric(x)
  fit <- StructTS(x, type, fixed = fixed)
  stopifnot(fit$code == 0)
  mod <- fit$model0
  np <- switch(type, level = 1L, trend = 2L, BSM = 3L)
  vx <- var(y, na.rm = TRUE) / 100
  stopifnot(all(mod$P == 1e6 * vx), mod$a[1] == y[1], all(mod$a[-1] == 0))   # the prior as p2a builds it
  if (is.null(fixed)) fixed <- rep(NA_real_, np + 1)
  mask <- is.na(fixed); cf <- fixed / vx
  fg <- function(par) .sts_fg(par, y, mod, vx, np, cf, mask)
  pR <- (fit$coef / vx)[mask]
  # the written-out objective is StructTS's: compare with stats::KalmanLike at R's estimate
  stopifnot(abs(fg(pR)$f - .sts_objective(y, .sts_set(mod, { p <- cf; p[mask] <- pR; p }, vx, np))) <= 1e-9 * abs(fg(pR)$f))
  # the exact gradient agrees with numDeriv on the KalmanLike objective at an interior point
  pc <- pR + 0.5
  gn <- numDeriv::grad(function(q) { p <- cf; p[mask] <- q; .sts_objective(y, .sts_set(mod, p, vx, np)) }, pc)
  stopifnot(max(abs(fg(pc)$g - gn)) <= 1e-4 * max(abs(gn)))
  # Newton on the free (non-bound) set, from R's estimate
  act <- pR == 0; fr <- which(!act); p <- pR
  for (i in 1:10) {
    g <- fg(p)$g; J <- matrix(0, length(fr), length(fr))
    for (jj in seq_along(fr)) {
      j <- fr[jj]; hs <- 1e-5 * max(abs(p[j]), 1e-3)
      pp <- p; pp[j] <- p[j] + hs; pm <- p; pm[j] <- p[j] - hs
      J[, jj] <- (fg(pp)$g[fr] - fg(pm)$g[fr]) / (2 * hs)
    }
    step <- solve((J + t(J)) / 2, g[fr]); p[fr] <- p[fr] - step
    if (max(abs(step) / abs(p[fr])) < 1e-13) break
  }
  e <- fg(p)
  stopifnot(all(p[fr] > 0), all(e$g[act] > 0), max(abs(e$g[fr])) < 1e-10, e$f <= fg(pR)$f + 1e-12 * abs(e$f))
  ptight <- cf; ptight[mask] <- p
  modt <- .sts_set(mod, ptight, vx, np)
  run <- KalmanRun(y, modt, nit = -1L, update = TRUE)
  sm <- KalmanSmooth(y, modt, nit = -1L)$smooth
  nm <- names(fit$coef)
  list(coef = named(fit$coef), loglik = fit$loglik,
       coef_tight = named(setNames(ptight * vx, nm)),
       loglik_tight = -length(y) * e$f - 0.5 * sum(!is.na(y)) * log(2 * pi),
       loglik_obs_tight = -sum(!is.na(y)) * e$f - 0.5 * sum(!is.na(y)) * log(2 * pi),
       filtered = run$states, smoothed = sm, std_resid = as.numeric(run$resid), fit = fit,
       final_model = attr(run, "mod"))
}
case("structts_nile_ukgas", "forecasting/structts", "stats::StructTS: local level and local linear trend on Nile, BSM on log10(UKgas), fixed level variance; package values and the tight optimum", function(ctx) {
  nile_na <- as.numeric(Nile); nile_na[c(21:25, 61)] <- NA          # interior gaps: StructTS filters through them
  d <- ctx$gen_dataset("structts_series", data.frame(t = 1:108, nile = c(as.numeric(Nile), rep(NA, 8)), ukgas = as.numeric(log10(UKgas)),
                                                     nile_na = c(nile_na, rep(NA, 8))))
  nile <- ts(d$nile[1:100], start = 1871); uk <- ts(d$ukgas, start = 1960, frequency = 4)
  f1 <- .sts_fit(nile, "level"); f2 <- .sts_fit(uk, "BSM"); f3 <- .sts_fit(nile, "level", fixed = c(1500, NA))
  f4 <- .sts_fit(nile, "trend"); f5 <- .sts_fit(ts(d$nile_na[1:100], start = 1871), "level")
  # CausalImpact's counterfactual (p2a): a local-level fit on the pre-period t = 1..90 and its Kalman
  # forecast for t = 91..100 with z = qnorm(0.975) bands; point and cumulative effects y - forecast.
  f6 <- .sts_fit(ts(d$nile[1:90], start = 1871), "level")
  fc6 <- KalmanForecast(10L, f6$final_model)
  post <- d$nile[91:100]
  v <- list(nile_coef = f1$coef, nile_loglik = f1$loglik, nile_level_last = as.numeric(tail(fitted(f1$fit), 1)),
            ukgas_coef = f2$coef, ukgas_loglik = f2$loglik,
            nile_fixed_coef = f3$coef, nile_fixed_loglik = f3$loglik,
            nile_trend_coef = f4$coef, nile_trend_loglik = f4$loglik,
            nile_coef_tight = f1$coef_tight, nile_loglik_tight = f1$loglik_tight,
            nile_filtered_level_tight = vec(f1$filtered[, 1]), nile_smoothed_level_tight = vec(f1$smoothed[, 1]),
            nile_std_resid_tight = vec(f1$std_resid),
            ukgas_coef_tight = f2$coef_tight, ukgas_loglik_tight = f2$loglik_tight,
            ukgas_filtered_level_tight = vec(f2$filtered[, 1]), ukgas_filtered_slope_tight = vec(f2$filtered[, 2]),
            ukgas_filtered_sea_tight = vec(f2$filtered[, 3]),
            ukgas_smoothed_level_tight = vec(f2$smoothed[, 1]), ukgas_smoothed_slope_tight = vec(f2$smoothed[, 2]),
            ukgas_smoothed_sea_tight = vec(f2$smoothed[, 3]), ukgas_std_resid_tight = vec(f2$std_resid),
            nile_fixed_coef_tight = f3$coef_tight, nile_fixed_loglik_tight = f3$loglik_tight,
            nile_trend_coef_tight = f4$coef_tight, nile_trend_loglik_tight = f4$loglik_tight,
            nile_trend_smoothed_level_tight = vec(f4$smoothed[, 1]), nile_trend_smoothed_slope_tight = vec(f4$smoothed[, 2]),
            nile_na_coef = f5$coef, nile_na_loglik = f5$loglik, nile_na_coef_tight = f5$coef_tight, nile_na_loglik_tight = f5$loglik_tight,
            nile_na_loglik_obs_tight = f5$loglik_obs_tight, nile_na_std_resid_tight = vec(f5$std_resid),
            nile_pre90_coef_tight = f6$coef_tight, nile_pre90_loglik_tight = f6$loglik_tight,
            nile_pre90_forecast = vec(fc6$pred), nile_pre90_forecast_var = vec(fc6$var),
            nile_pre90_upper = vec(fc6$pred + qnorm(0.975) * sqrt(fc6$var)),
            nile_pre90_point_effect = vec(post - fc6$pred), nile_pre90_cum_effect = sum(post - fc6$pred))
  list(values = v, reference = "stats::StructTS(type = 'level' | 'trend' | 'BSM', fixed); tight optimum of its objective by Newton on the exact gradient; stats::KalmanRun / KalmanSmooth (nit = -1) at the tight optimum",
       call = "StructTS(Nile, 'level'); StructTS(Nile, 'trend'); StructTS(log10(UKgas), 'BSM'); StructTS(Nile, 'level', fixed = c(1500, NA)); StructTS(nile_na, 'level')",
       packages = "numDeriv")
})
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
