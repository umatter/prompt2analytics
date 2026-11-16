# validation/r/cases/ts_statespace_t4.R -- T4-WP25: CausalImpact intervals (TS-5).
#  Reads .sts_fit and helpers from ts_statespace.R (sourced first: run.R sorts the case files).
#  The interval of the cumulative effect is the variance of the SUM of the post-period forecast
#  errors under the fitted model: Cov(e_i, e_j) = Z' T^(j-i) P_{i|T} Z (j > i), + h on the diagonal,
#  summed by the recursion g_k = T g_{k-1} + P_k Z, V_k = V_{k-1} + 2 Z'g_k - Z'P_k Z + h, from
#  stats::KalmanRun's final (a, P) at the tight optimum of StructTS. Checked against the closed form
#  of the local level (H^2 P_T + q sum_{i,j} min(i, j) + H h) and against the full H x H covariance.
#  With controls (written out): OLS on the pre-period, StructTS on its residuals, then the regression
#  as diffuse static states: stats::KalmanRun on y and on each control with the fitted model, GLS of
#  the y innovations on the control innovations (weights 1/F_t, no intercept: the level absorbs it,
#  lm.wfit), Var(beta) = (sum V_t V_t' / F_t)^-1, added as d' Var(beta) d with d = x - Z' a(x).
.ci_joint <- function(mod, H) {
  Tm <- mod$T; Z <- mod$Z; V <- mod$V; h <- mod$h; P <- mod$P
  g <- rep(0, length(Z)); Vc <- 0; out <- numeric(H)
  for (k in 1:H) { P <- Tm %*% P %*% t(Tm) + V; g <- drop(Tm %*% g) + drop(P %*% Z)
    Vc <- Vc + 2 * sum(Z * g) - drop(t(Z) %*% P %*% Z) + h; out[k] <- Vc }
  out
}
.ci_full <- function(mod, H) {                       # the H x H covariance written out, for the check
  Tm <- mod$T; Z <- mod$Z; V <- mod$V; h <- mod$h; P <- mod$P; m <- length(Z)
  Ps <- vector("list", H); for (k in 1:H) { P <- Tm %*% P %*% t(Tm) + V; Ps[[k]] <- P }
  S <- matrix(0, H, H)
  for (i in 1:H) for (j in i:H) { Tp <- diag(m); if (j > i) for (s in 1:(j - i)) Tp <- Tm %*% Tp
    S[i, j] <- S[j, i] <- drop(t(Z) %*% Tp %*% Ps[[i]] %*% Z) + (i == j) * h }
  S
}
.ci_summary <- function(post, pred, cumvar, alpha = 0.05) {
  z <- qnorm(1 - alpha / 2); H <- length(post); eff <- post - pred; cum <- cumsum(eff)
  se <- sqrt(cumvar[H]); Y <- sum(post); P <- sum(pred)
  rel <- (Y - P) / P; rel_se <- abs(Y) / P^2 * se
  list(cum_effect = cum[H], cum_se = se, cum_lower = cum[H] - z * se, cum_upper = cum[H] + z * se,
       avg_lower = (cum[H] - z * se) / H, avg_upper = (cum[H] + z * se) / H,
       cum_series_lower = cum - z * sqrt(cumvar), cum_series_upper = cum + z * sqrt(cumvar),
       rel = rel, rel_lower = rel - z * rel_se, rel_upper = rel + z * rel_se,
       prob_positive = pnorm(cum[H] / se), p_two_sided = 2 * pnorm(-abs(cum[H]) / se),
       p_one_sided = pnorm(-abs(cum[H]) / se))
}
case("causal_impact_intervals_t4", "forecasting/causal_impact", "CausalImpact (p2a): the variance of the cumulative effect from the joint covariance of the forecast errors; level (Nile), trend (Nile), BSM (log10 UKgas); a control series as diffuse regression states", function(ctx) {
  d <- ctx$dataset("datasets/generated/structts_series.csv")
  out <- list()
  add <- function(prefix, fit, y, npre, npost) {
    H <- npost; fc <- KalmanForecast(H, fit$final_model); cv <- .ci_joint(fit$final_model, H)
    S <- .ci_full(fit$final_model, H); stopifnot(abs(sum(S) - cv[H]) <= 1e-9 * cv[H], all(abs(diag(S) - fc$var) <= 1e-9 * fc$var))
    s <- .ci_summary(y[(npre + 1):(npre + npost)], fc$pred, cv)
    out[[paste0(prefix, "_cum_var")]] <<- vec(cv); out[[paste0(prefix, "_marg_var_sum")]] <<- sum(fc$var)
    for (k in names(s)) out[[paste0(prefix, "_", k)]] <<- vec(s[[k]])
    out[[paste0(prefix, "_coef_tight")]] <<- fit$coef_tight
  }
  f1 <- .sts_fit(ts(d$nile[1:90], start = 1871), "level")
  P <- f1$final_model$P[1, 1]; q <- f1$final_model$V[1, 1]; h <- f1$final_model$h; i <- 1:10
  stopifnot(abs(.ci_joint(f1$final_model, 10)[10] - (100 * P + q * sum(outer(i, i, pmin)) + 10 * h)) <= 1e-9 * (100 * P))
  add("nile_level", f1, d$nile, 90, 10)
  f2 <- .sts_fit(ts(d$nile[1:90], start = 1871), "trend"); add("nile_trend", f2, d$nile, 90, 10)
  f3 <- .sts_fit(ts(d$ukgas[1:92], start = 1960, frequency = 4), "BSM"); add("ukgas_bsm", f3, d$ukgas, 92, 16)
  # a control series: y = 2 x + level + noise, effect 3 from t = 81 (generated)
  set.seed(20261115)
  n <- 100; x <- 10 + cumsum(rnorm(n, 0, 0.7)); lev <- 50 + cumsum(0.5 * rnorm(n))
  y <- 2 * x + lev + rnorm(n) + c(rep(0, 80), rep(3, 20))
  g <- ctx$gen_dataset("causal_impact_control_t4", data.frame(t = 1:n, y = round(y, 10), x = round(x, 10)))
  y <- g$y; x <- g$x; pre <- 1:80; post <- 81:100
  ols <- lm(y[pre] ~ x[pre]); res <- unname(resid(ols))
  fr <- .sts_fit(ts(res), "level"); m <- fr$final_model
  mod0 <- fr$fit$model0; vx <- var(res) / 100
  modk <- mod0; modk$V[1, 1] <- fr$coef_tight$values[1]; modk$h <- fr$coef_tight$values[2]
  ry <- { mm <- modk; mm$a <- y[1]; KalmanRun(y[pre], mm, nit = -1L, update = TRUE) }
  rx <- { mm <- modk; mm$a <- x[1]; KalmanRun(x[pre], mm, nit = -1L, update = TRUE) }
  my <- attr(ry, "mod"); mx <- attr(rx, "mod")
  # innovations v_t and their variances F_t (predict first, local level: a_{t|t-1} = a_{t-1|t-1});
  # the first innovation is 0 for y and x alike (a_0 is the first value), and F_t does not depend on the data
  q <- modk$V[1, 1]; h <- modk$h; ay <- y[1]; ax <- x[1]; Pt <- mod0$P[1, 1]
  vy <- vxx <- Fv <- numeric(80)
  for (t in 1:80) { Pp <- Pt + q; Fv[t] <- Pp + h; K <- Pp / Fv[t]
    vy[t] <- y[t] - ay; vxx[t] <- x[t] - ax; ay <- ay + K * vy[t]; ax <- ax + K * vxx[t]; Pt <- Pp * (1 - K) }
  stopifnot(abs(ay - my$a[1]) <= 1e-9 * abs(ay), abs(ax - mx$a[1]) <= 1e-9 * abs(ax))
  w <- 1 / Fv; Vb <- 1 / sum(w * vxx^2); b <- Vb * sum(w * vxx * vy)
  stopifnot(abs(b - lm.wfit(cbind(vxx), vy, w)$coefficients[[1]]) <= 1e-10 * abs(b))
  fy <- KalmanForecast(20L, my); fx <- KalmanForecast(20L, mx)
  pred <- fy$pred - b * fx$pred + b * x[post]
  dvec <- x[post] - fx$pred
  cv <- .ci_joint(my, 20) + cumsum(dvec)^2 * Vb
  s <- .ci_summary(y[post], pred, cv)
  out$ctrl_ols_coef <- vec(coef(ols)); out$ctrl_coef_tight <- fr$coef_tight
  out$ctrl_beta <- b; out$ctrl_beta_se <- sqrt(Vb); out$ctrl_pred <- vec(pred)
  out$ctrl_point_var <- vec(fy$var + dvec^2 * Vb); out$ctrl_cum_var <- vec(cv)
  for (k in names(s)) out[[paste0("ctrl_", k)]] <- vec(s[[k]])
  list(values = out, reference = "stats::StructTS tight optimum (ts_statespace.R) + stats::KalmanRun / KalmanForecast (nit = -1); the joint covariance of the forecast errors written out and checked against the full H x H matrix; control: lm, then KalmanRun on y and x, GLS by lm.wfit on the innovations",
       call = "StructTS(Nile[1:90], 'level' | 'trend'); StructTS(log10(UKgas)[1:92], 'BSM'); KalmanForecast(10 | 16, mod)", packages = "numDeriv")
})
