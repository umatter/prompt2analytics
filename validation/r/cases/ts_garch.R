# validation/r/cases/ts_garch.R -- GARCH(p, q) references (T3-WP35: TS-11, TS-20, TS-32; moved in TS-12,
# TS-19, TS-21). Package copy: canonical, supersedes T3 design Appendix A.31. The Appendix keys are kept
# and computed by the same code, except the two tseries keys, which become the likelihood-identity
# cross-check below.
# fGarch and rugarch are not installed, and tseries::garch has no mean term and no t errors, so the
# conditional likelihood is written out (fGarch's "mci" initialisation):
#   e_t = x_t - mu (mu = 0 without a mean term), m = max(p, q), h_t = mean(e^2) for t <= m,
#   h_t = omega + sum_i alpha_i e_{t-i}^2 + sum_j beta_j h_{t-j} for t > m;
#   normal: l = -1/2 sum_t [log(2 pi) + log h_t + e_t^2 / h_t];
#   standardised t (fGarch "std", shape nu > 2): l = sum_t [lgamma((nu+1)/2) - lgamma(nu/2)
#       - log(pi (nu - 2))/2 - log(h_t)/2 - (nu+1)/2 log(1 + e_t^2 / (h_t (nu - 2)))].
# Appendix keys (norm_par .. norm_h_last): optim (BFGS, Nelder-Mead, BFGS; reltol 1e-15, finite-difference
# gradients) with numDeriv::hessian of l for the SEs, exactly as Appendix A.31. They are a cross-check only:
# optim stops about 1e-6 (relative) short of the optimum, and second differences of the 1500-term sum
# put the mean's SE about 1 % off (2.2163e-4 against 2.2462e-4).
# Pinned keys (*_tight and the keys derived from them): Newton iterations on the ANALYTIC gradient
# (recursions for dh_t / dparameter, in .ggrad) from the optim result. The Hessians are numDeriv::jacobian
# of the analytic gradient with Richardson extrapolation (r = 6) on the relative scale, and the iterations
# stop when the relative step is below 1e-15. SEs are sqrt(diag(solve(-H))), where H is the Hessian of l in
# the natural parameters (mu, omega, alpha, beta[, shape]).
# Forecasts (TS-21): sigma2_{T+k} = omega + sum_i alpha_i E[e^2_{T+k-i}] + sum_j beta_j sigma2_{T+k-j}, with
# E[e^2_s] = e_s^2 and sigma2_s = h_s for s <= T. For GARCH(1,1) this is the closed form
# uncond + (alpha + beta)^(k-1) (sigma2_{T+1} - uncond), asserted below.
# Bound (g21b: GARCH(2,1) on the GARCH(1,1) series): the maximum lies on alpha2 = 0, where dl/dalpha2 < 0.
# alpha2 is fixed at 0, and the other parameters and SEs are those of the restricted maximum.
# tseries cross-check (zero-mean fit on the demeaned series): tseries::garch minimises
# 0.5 sum_{t >= 2} [log h_t + y_t^2 / h_t] with h_1 = mean(y^2), which equals
# -l - n/2 log(2 pi) - (log h_1 + y_1^2 / h_1)/2. The identity is asserted at tseries's own estimates.
# tseries is a reference for the likelihood only: its PORT optimizer works on the unscaled parameters
# (omega ~ 1e-6 next to alpha ~ 0.07) and stops early from any start that is not the optimum. From its
# default start it reports "FALSE CONVERGENCE" at logLik 4843.03 (tseries_default_*); started at the
# tight optimum (4854.14) it stays there.
suppressPackageStartupMessages({ library(tseries); library(numDeriv) })

# ---- Appendix A.31 code, unchanged ----
.garch_ll <- function(par, x, dist) {
  mu <- par[1]; om <- par[2]; al <- par[3]; be <- par[4]
  e <- x - mu; n <- length(e); h <- numeric(n); h[1] <- mean(e^2)
  for (t in 2:n) h[t] <- om + al * e[t - 1]^2 + be * h[t - 1]
  if (any(h <= 0)) return(-1e300)
  if (dist == "norm") return(-0.5 * sum(log(2 * pi) + log(h) + e^2 / h))
  nu <- par[5]
  sum(lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * (nu - 2)) - 0.5 * log(h) - (nu + 1) / 2 * log(1 + e^2 / (h * (nu - 2))))
}
.to_nat <- function(th, dist) {
  s <- exp(th[3]) + exp(th[4]); ab <- c(exp(th[3]), exp(th[4])) / (1 + s)   # alpha, beta > 0, alpha + beta < 1
  out <- c(th[1], exp(th[2]), ab)
  if (dist == "std") out <- c(out, 2 + exp(th[5]))
  out
}
.garch_fit <- function(x, dist, start_nat) {
  ab <- start_nat[3:4]; r <- 1 - sum(ab)
  th0 <- c(start_nat[1], log(start_nat[2]), log(ab / r))
  if (dist == "std") th0 <- c(th0, log(start_nat[5] - 2))
  f <- function(th) -.garch_ll(.to_nat(th, dist), x, dist)
  o <- optim(th0, f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  o <- optim(o$par, f, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))
  o <- optim(o$par, f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  nat <- .to_nat(o$par, dist)
  # Hessian of -l in the natural parameters, computed on the relative scale p = nat * s (s = 1) so
  # that numDeriv's steps suit omega ~ 1e-6 and mu ~ 1e-4 alike: H_nat = D^-1 H_s D^-1, D = diag(nat).
  Hs <- numDeriv::hessian(function(s) -.garch_ll(nat * s, x, dist), rep(1, length(nat)), method.args = list(d = 1e-4, r = 6))
  H <- Hs / tcrossprod(nat)
  list(par = nat, se = sqrt(diag(solve(H))), ll = -o$value, conv = o$convergence)
}
.garch_h <- function(par, x) { e <- x - par[1]; n <- length(e); h <- numeric(n); h[1] <- mean(e^2)
  for (t in 2:n) h[t] <- par[2] + par[3] * e[t - 1]^2 + par[4] * h[t - 1]; list(h = h, e = e) }
.garch_sim <- function(n, om, al, be, nu = Inf, mu = 0) {
  z <- if (is.finite(nu)) rt(n + 200, nu) * sqrt((nu - 2) / nu) else rnorm(n + 200)
  e <- numeric(n + 200); h <- numeric(n + 200); h[1] <- om / (1 - al - be)
  for (t in 1:(n + 200)) { if (t > 1) h[t] <- om + al * e[t - 1]^2 + be * h[t - 1]; e[t] <- sqrt(h[t]) * z[t] }
  mu + e[-(1:200)]
}

# ---- package code: GARCH(p, q); natural parameters nat = [mu], omega, alpha_1..p, beta_1..q, [shape] ----
.gspec <- function(x, p, q, mean, dist) list(x = x, n = length(x), p = p, q = q, mean = mean, dist = dist)
.gun <- function(nat, s) {
  i <- as.integer(s$mean)
  list(mu = if (s$mean) nat[1] else 0, om = nat[i + 1], al = nat[i + 1 + seq_len(s$p)],
       be = nat[i + 1 + s$p + seq_len(s$q)], nu = if (s$dist == "std") nat[length(nat)] else Inf)
}
.grec <- function(x, be, init) if (length(be) == 0) x else as.vector(stats::filter(x, be, method = "recursive", init = init))
.gh <- function(u, e, s) {
  m <- max(s$p, s$q); hbar <- mean(e^2); h <- rep(hbar, s$n); tt <- (m + 1):s$n
  c0 <- u$om + as.vector(sapply(seq_len(s$p), function(i) e[tt - i]^2) %*% u$al)
  h[tt] <- .grec(c0, u$be, rep(hbar, length(u$be))); h
}
.gll <- function(nat, s) {
  u <- .gun(nat, s); e <- s$x - u$mu; h <- .gh(u, e, s)
  if (s$dist == "norm") return(list(ll = -0.5 * sum(log(2 * pi) + log(h) + e^2 / h), h = h, e = e))
  nu <- u$nu
  list(ll = sum(lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(pi * (nu - 2)) - 0.5 * log(h) -
                (nu + 1) / 2 * log1p(e^2 / (h * (nu - 2)))), h = h, e = e)
}
# Analytic gradient of l. For t > m: dh_t/dmu = sum_i alpha_i (-2 e_{t-i}) + sum_j beta_j dh_{t-j}/dmu,
# dh_t/domega = 1 + .., dh_t/dalpha_i = e_{t-i}^2 + .., dh_t/dbeta_j = h_{t-j} + .. ("..": the same beta
# recursion); for t <= m dh_t/dmu = -2 mean(e) and the others are 0. dl/dshape uses digamma.
.ggrad <- function(nat, s) {
  u <- .gun(nat, s); e <- s$x - u$mu; m <- max(s$p, s$q); h <- .gh(u, e, s); tt <- (m + 1):s$n
  pre <- function(v0, d) c(rep(v0, m), .grec(d, u$be, rep(v0, s$q)))
  cols <- list()
  if (s$mean) cols <- c(cols, list(pre(-2 * mean(e), -2 * as.vector(sapply(seq_len(s$p), function(i) e[tt - i]) %*% u$al))))
  cols <- c(cols, list(pre(0, rep(1, length(tt)))))
  for (i in seq_len(s$p)) cols <- c(cols, list(pre(0, e[tt - i]^2)))
  for (j in seq_len(s$q)) cols <- c(cols, list(pre(0, h[tt - j])))
  dh <- do.call(cbind, cols)
  if (s$dist == "norm") { dl_dh <- -0.5 * (1 / h - e^2 / h^2); dl_de <- -e / h
  } else { nu <- u$nu; k <- nu - 2; r <- e^2 / (h * k)
    dl_dh <- -0.5 / h + (nu + 1) / 2 * (r / h) / (1 + r); dl_de <- -(nu + 1) * e / (h * k) / (1 + r) }
  g <- colSums(dl_dh * dh)
  if (s$mean) g[1] <- g[1] - sum(dl_de)
  if (s$dist == "std") { nu <- u$nu; k <- nu - 2; r <- e^2 / (h * k)
    g <- c(g, sum(0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / k - 0.5 * log1p(r) + (nu + 1) / 2 * (r / k) / (1 + r))) }
  g
}
# Hessian of l on the free parameters (relative scale, Richardson); fixed parameters keep their value
.ghess <- function(nat, s, free) {
  q <- nat[free]
  J <- numDeriv::jacobian(function(z) { v <- nat; v[free] <- q * z; .ggrad(v, s)[free] }, rep(1, length(q)), method.args = list(r = 6))
  H <- sweep(J, 2, q, "/"); (H + t(H)) / 2
}
# optim on a softmax transform of the free parameters (start), then Newton on the analytic gradient (tight)
.gfit <- function(x, p, q, mean, dist, start, fixed = integer(0)) {
  s <- .gspec(x, p, q, mean, dist); k <- length(start); free <- setdiff(seq_len(k), fixed)
  iv <- as.integer(mean) + 1 + seq_len(p + q); ivf <- intersect(iv, free)
  nat_of <- function(th) {
    v <- start; j <- 0; nx <- function() { j <<- j + 1; th[j] }
    if (mean) v[1] <- nx()
    v[as.integer(mean) + 1] <- exp(nx())
    w <- exp(vapply(ivf, function(.) nx(), 0)); v[ivf] <- w / (1 + sum(w))
    if (dist == "std") v[k] <- 2 + exp(nx())
    v
  }
  th0 <- c(if (mean) start[1], log(start[as.integer(mean) + 1]), log(start[ivf] / (1 - sum(start[ivf]))), if (dist == "std") log(start[k] - 2))
  f <- function(th) { r <- .gll(nat_of(th), s)$ll; if (!is.finite(r)) 1e300 else -r }
  o <- optim(th0, f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  o <- optim(o$par, f, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))
  o <- optim(o$par, f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  nat <- nat_of(o$par)
  for (it in 1:50) {
    step <- solve(.ghess(nat, s, free), .ggrad(nat, s)[free]); nat[free] <- nat[free] - step
    if (max(abs(step) / abs(nat[free])) < 1e-15) break
  }
  stopifnot(max(abs(step) / abs(nat[free])) < 1e-12, -o$value <= .gll(nat, s)$ll + 1e-9 * abs(o$value))
  H <- .ghess(nat, s, free); V <- matrix(NA_real_, k, k); V[free, free] <- solve(-H)
  r <- .gll(nat, s)
  list(par = nat, se = sqrt(diag(V)), vcov = V, ll = r$ll, h = r$h, e = r$e, s = s)
}
.gforecast <- function(fit, H) {
  u <- .gun(fit$par, fit$s); n <- fit$s$n; e2 <- c(fit$e^2, rep(NA, H)); hh <- c(fit$h, rep(NA, H))
  for (k in 1:H) { t <- n + k
    ea <- vapply(seq_len(fit$s$p), function(i) if (t - i <= n) e2[t - i] else hh[t - i], 0)
    hh[t] <- u$om + sum(u$al * ea) + sum(u$be * hh[t - seq_len(fit$s$q)]) }
  hh[n + 1:H]
}
.gnames <- function(p, q, mean, dist) c(if (mean) "mu", "omega", paste0("alpha", seq_len(p)), if (q > 0) paste0("beta", seq_len(q)), if (dist == "std") "shape")
.gvals <- function(pre, fit, nm, H = 0, nh = 0) {
  v <- list(); k <- length(fit$par); n <- fit$s$n
  v[[paste0(pre, "_par_tight")]] <- named(setNames(fit$par, nm))
  se <- fit$se; keep <- !is.na(se)
  v[[paste0(pre, "_se_tight")]] <- named(setNames(se[keep], nm[keep]))   # a parameter fixed on its bound has no SE
  v[[paste0(pre, "_loglik_tight")]] <- fit$ll
  v[[paste0(pre, "_aic")]] <- -2 * fit$ll + 2 * k
  v[[paste0(pre, "_bic")]] <- -2 * fit$ll + log(n) * k
  if (nh > 0) v[[paste0(pre, "_h_first")]] <- vec(fit$h[1:nh])
  if (H > 0) v[[paste0(pre, "_forecast_var_tight")]] <- vec(.gforecast(fit, H))
  v
}
.gsim_pq <- function(n, om, al, be, mu = 0) {
  p <- length(al); q <- length(be); m <- max(p, q); N <- n + 200; z <- rnorm(N)
  e <- numeric(N); h <- rep(om / (1 - sum(al) - sum(be)), N)
  for (t in 1:N) { if (t > m) h[t] <- om + sum(al * e[t - seq_len(p)]^2) + sum(be * h[t - seq_len(q)]); e[t] <- sqrt(h[t]) * z[t] }
  mu + e[-(1:200)]
}

case("garch_normal_t", "forecasting/garch", "GARCH(p, q) with mean, normal and standardised-t errors, return scale (variance ~1e-4): (1,1) normal and t, (1,2), (2,1) interior and on the alpha2 = 0 bound, ARCH(1), zero mean against tseries", function(ctx) {
  set.seed(20261105)
  xn <- .garch_sim(1500, 2e-6, 0.08, 0.90, mu = 5e-4)
  xt <- .garch_sim(1500, 2e-6, 0.08, 0.90, nu = 6, mu = 5e-4)
  x21 <- .gsim_pq(1500, 2e-6, c(0.05, 0.06), 0.85, mu = 5e-4)   # drawn after xn and xt, which keep the Appendix values
  rn <- round(xn, 12)
  d <- ctx$gen_dataset("garch_returns", data.frame(t = seq_along(xn), r_norm = rn, r_t = round(xt, 12),
                                                    r_g21 = round(x21, 12), r_norm_dm = rn - mean(rn)))
  n <- nrow(d); v <- list()
  # ---- Appendix A.31 keys (cross-check) ----
  fn <- .garch_fit(d$r_norm, "norm", c(mean(d$r_norm), 1e-6, 0.05, 0.9))
  v$norm_par <- vec(fn$par); v$norm_se <- vec(fn$se); v$norm_loglik <- fn$ll
  ft <- .garch_fit(d$r_t, "std", c(mean(d$r_t), 1e-6, 0.05, 0.9, 8))
  v$std_par <- vec(ft$par); v$std_se <- vec(ft$se); v$std_loglik <- ft$ll
  fnt <- .garch_fit(d$r_t, "norm", c(mean(d$r_t), 1e-6, 0.05, 0.9))
  v$norm_on_t_par <- vec(fnt$par); v$norm_on_t_loglik <- fnt$ll
  hh <- .garch_h(fn$par, d$r_norm)
  s1 <- fn$par[2] + fn$par[3] * hh$e[n]^2 + fn$par[4] * hh$h[n]; un <- fn$par[2] / (1 - fn$par[3] - fn$par[4])
  v$norm_forecast_var <- vec(un + (fn$par[3] + fn$par[4])^(0:9) * (s1 - un))
  v$norm_h_last <- hh$h[n]
  # ---- pinned references ----
  tn <- .gfit(d$r_norm, 1, 1, TRUE, "norm", fn$par)
  stopifnot(abs(tn$ll - .garch_ll(tn$par, d$r_norm, "norm")) < 1e-10 * abs(tn$ll))   # general code = Appendix code
  v <- c(v, .gvals("norm", tn, .gnames(1, 1, TRUE, "norm"), H = 10, nh = 3))
  v$norm_vcov_tight <- vec(tn$vcov)                      # column-major 4 x 4, order mu, omega, alpha1, beta1
  z <- tn$par / tn$se                                    # Wald z; p-values from the normal, as fGarch and tseries
  v$norm_z_tight <- named(setNames(z, .gnames(1, 1, TRUE, "norm"))); v$norm_p_tight <- named(setNames(2 * pnorm(-abs(z)), .gnames(1, 1, TRUE, "norm")))
  v$norm_h_last_tight <- tn$h[n]
  v$norm_uncond_var_tight <- tn$par[2] / (1 - tn$par[3] - tn$par[4])
  fc <- .gforecast(tn, 10); unt <- v$norm_uncond_var_tight
  stopifnot(max(abs(fc - (unt + (tn$par[3] + tn$par[4])^(0:9) * (fc[1] - unt))) / fc) < 1e-13)   # recursion = closed form
  tt <- .gfit(d$r_t, 1, 1, TRUE, "std", ft$par)
  stopifnot(abs(tt$ll - .garch_ll(tt$par, d$r_t, "std")) < 1e-10 * abs(tt$ll))
  v <- c(v, .gvals("std", tt, .gnames(1, 1, TRUE, "std"), H = 10))
  tnt <- .gfit(d$r_t, 1, 1, TRUE, "norm", fnt$par)
  v <- c(v, .gvals("norm_on_t", tnt, .gnames(1, 1, TRUE, "norm")))
  g12 <- .gfit(d$r_norm, 1, 2, TRUE, "norm", c(tn$par[1:3], 0.6, 0.3))
  v <- c(v, .gvals("g12", g12, .gnames(1, 2, TRUE, "norm"), H = 10, nh = 3))
  g21 <- .gfit(d$r_g21, 2, 1, TRUE, "norm", c(mean(d$r_g21), 2e-6, 0.05, 0.05, 0.85))
  v <- c(v, .gvals("g21", g21, .gnames(2, 1, TRUE, "norm"), H = 10, nh = 3))
  g21b <- .gfit(d$r_norm, 2, 1, TRUE, "norm", c(tn$par[1:3], 0, tn$par[4]), fixed = 4L)   # alpha2 on its bound 0
  v$g21b_dl_dalpha2 <- .ggrad(g21b$par, g21b$s)[4]
  stopifnot(v$g21b_dl_dalpha2 < 0)
  v <- c(v, .gvals("g21b", g21b, .gnames(2, 1, TRUE, "norm")))
  a1 <- .gfit(d$r_norm, 1, 0, TRUE, "norm", c(mean(d$r_norm), 8e-5, 0.2))
  v <- c(v, .gvals("arch1", a1, .gnames(1, 0, TRUE, "norm"), H = 5))
  # zero mean on the demeaned series, and the tseries cross-check
  y <- d$r_norm_dm
  zm <- .gfit(y, 1, 1, FALSE, "norm", tn$par[2:4])
  v <- c(v, .gvals("zm", zm, .gnames(1, 1, FALSE, "norm"), H = 5))
  ts_ll <- function(g) -g$n.likeli - n / 2 * log(2 * pi) - 0.5 * (log(mean(y^2)) + y[1]^2 / mean(y^2))
  g0 <- suppressWarnings(tseries::garch(y, order = c(1, 1), control = garch.control(trace = FALSE, maxiter = 1000)))
  stopifnot(abs(ts_ll(g0) - .gll(unname(coef(g0)), zm$s)$ll) < 1e-9 * abs(zm$ll), ts_ll(g0) < zm$ll - 1)
  # started at the optimum, tseries's own optimizer finds nothing better and stays (1e-6 relative)
  g1 <- suppressWarnings(tseries::garch(y, order = c(1, 1), control = garch.control(trace = FALSE, maxiter = 1000, start = zm$par)))
  stopifnot(abs(ts_ll(g1) - .gll(unname(coef(g1)), zm$s)$ll) < 1e-9 * abs(zm$ll), ts_ll(g1) <= zm$ll + 1e-9 * abs(zm$ll),
            max(abs(unname(coef(g1)) - zm$par) / zm$par) < 1e-6)
  v$tseries_default_coef <- vec(unname(coef(g0))); v$tseries_default_loglik <- ts_ll(g0)
  list(values = v, reference = "conditional GARCH(p, q) likelihood (normal; fGarch's standardised t; h_t = mean(e^2) for t <= max(p, q)) maximised by optim, then Newton iterations on the analytic gradient with numDeriv::jacobian Hessians, which give the SEs; forecasts by the GARCH recursion; tseries::garch likelihood identity on the demeaned series",
       call = "optim + Newton on the written-out GARCH log-likelihood; tseries::garch(x - mean(x), order = c(1, 1))",
       packages = c("tseries", "numDeriv"))
})
