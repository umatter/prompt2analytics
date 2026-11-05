# validation/r/cases/ts_varma.R -- VARMA(p, q) by the Hannan-Rissanen two-step estimator (T3-WP13:
# ECON-51; the VARMA sample checks of ECON-79 and X-7 are library tests without R values).
# No R package for this estimator is installed (MTS, marima and dse are missing), so the estimator is
# written out from Hannan & Rissanen (1982) and Luetkepohl (2005, ch. 12); each OLS step is
# stats::lm.fit / stats::lm:
#   step 1 (q > 0): a long VAR(m) with intercept by OLS on t = m + 1..n, residuals ehat_t;
#   step 2: OLS of y_t on [1, y_{t-1..p}, ehat_{t-1..q}] equation by equation on
#           t = m + max(p, q) + 1..n (n_eff = n - m - max(p, q));
#   logLik = -T/2 (K log 2 pi + log det Sigma_ML + K), Sigma_ML = U'U / T;
#   AIC = -2 logLik + 2 npar, BIC = -2 logLik + log(T) npar, npar = K (1 + K p + K q)
#   (the parameter count of stats::logLik for vars::VAR, extended by the MA coefficients).
# Standard errors and p-values are recorded only for q = 0 (an OLS VAR). For q > 0 the step-2 OLS
# standard errors treat the estimated innovations as data (a generated-regressor problem), are not
# valid for inference, and p2a does not report them.
# Default long-VAR order (p2a's rule, pinned here): q = 0 -> m = 0 (no innovations needed; the fit is
# vars::VAR); q > 0 -> m = max(floor(2 log(n)), 2 max(p, q)) (an order growing like log n, the rate of
# Hannan & Kavalieris 1984; T3-PERF4 replaced floor(log(n)^2), Gomez & Maravall 2001), reduced to
# floor((n - 2) / (2K + 1)) when larger, so that the long VAR keeps at least as many residual degrees of
# freedom as it has regressors.
# The design keys (long_var_order, step2_coef, sigma_ml, loglik, n_eff) are computed exactly as in T3
# design Appendix A.8, so their values are unchanged; every other key is new.
suppressPackageStartupMessages(library(vars))

.hr_default_order <- function(n, K, p, q) {
  if (q == 0) return(0)
  m0 <- max(floor(2 * log(n)), 2 * max(p, q))
  min(m0, floor((n - 2) / (2 * K + 1)))
}

# Hannan-Rissanen VARMA(p, q) with long-VAR order m; values prefixed by `pre`.
.hr_varma <- function(Y, p, q, m, pre) {
  n <- nrow(Y); K <- ncol(Y); nm <- colnames(Y)
  lagnames <- function(l, e = FALSE) paste0(if (e) "e." else "", nm, ".l", l)
  E <- matrix(NA_real_, n, K)
  if (q > 0) {
    Z <- cbind(1, do.call(cbind, lapply(1:m, function(l) Y[(m + 1 - l):(n - l), , drop = FALSE])))
    lf <- lm.fit(Z, Y[(m + 1):n, , drop = FALSE])
    stopifnot(lf$rank == ncol(Z))
    E[(m + 1):n, ] <- lf$residuals
  }
  s <- max(p, q); tt <- (m + s + 1):n
  X <- matrix(1, length(tt), 1, dimnames = list(NULL, "const"))
  for (l in seq_len(p)) { z <- Y[tt - l, , drop = FALSE]; colnames(z) <- lagnames(l); X <- cbind(X, z) }
  for (l in seq_len(q)) { z <- E[tt - l, , drop = FALSE]; colnames(z) <- lagnames(l, TRUE); X <- cbind(X, z) }
  stopifnot(!anyNA(X))
  Tn <- length(tt); U <- matrix(NA_real_, Tn, K); v <- list()
  for (j in seq_len(K)) {
    fit <- lm(Y[tt, j] ~ 0 + X)
    stopifnot(fit$rank == ncol(X))
    sm <- summary(fit)$coefficients
    rn <- sub("^X", "", rownames(sm))
    v[[paste0(pre, "coef_", nm[j])]] <- named(setNames(sm[, 1], rn))
    if (q == 0) {                                          # SEs only for the OLS VAR (see header)
      v[[paste0(pre, "se_", nm[j])]] <- named(setNames(sm[, 2], rn))
      v[[paste0(pre, "p_", nm[j])]] <- named(setNames(sm[, 4], rn))
    }
    U[, j] <- residuals(fit)
  }
  S <- crossprod(U) / Tn
  ll <- -Tn / 2 * (K * log(2 * pi) + as.numeric(determinant(S, logarithm = TRUE)$modulus) + K)
  npar <- K * ncol(X)
  v[[paste0(pre, "sigma_ml")]] <- vec(S)
  v[[paste0(pre, "loglik")]] <- ll
  v[[paste0(pre, "aic")]] <- -2 * ll + 2 * npar
  v[[paste0(pre, "bic")]] <- -2 * ll + log(Tn) * npar
  v[[paste0(pre, "n_eff")]] <- Tn
  v[[paste0(pre, "df_residual")]] <- Tn - ncol(X)
  v[[paste0(pre, "n_params")]] <- npar
  v
}

case("varma_hr", "econometrics/timeseries", "VARMA(p, q) by Hannan-Rissanen (long VAR by OLS, then OLS on lagged y and lagged residuals), bivariate n = 400: (1,1) m = 8, (1,0) m = 8, (0,1) m = 8, (2,1) m = 8, (2,2) m = 10, default orders, q = 0 against vars::VAR", function(ctx) {
  set.seed(20261114)
  n <- 400; A <- matrix(c(0.5, 0.1, -0.2, 0.4), 2); M <- matrix(c(0.3, 0, 0.2, -0.25), 2)
  e <- matrix(rnorm(2 * (n + 100)), ncol = 2) %*% chol(matrix(c(1, 0.3, 0.3, 0.8), 2))
  y <- matrix(0, n + 100, 2)
  for (t in 2:(n + 100)) y[t, ] <- A %*% y[t - 1, ] + e[t, ] + M %*% e[t - 1, ]
  d <- ctx$gen_dataset("varma_sim", data.frame(y1 = round(y[-(1:100), 1], 10), y2 = round(y[-(1:100), 2], 10)))
  Y <- as.matrix(d); n <- nrow(Y); m <- 8
  # ---- design keys (T3 design Appendix A.8; step2_se dropped: no SEs for q > 0) ----
  Z <- cbind(1, do.call(cbind, lapply(1:m, function(l) Y[(m + 1 - l):(n - l), ])))
  B <- solve(crossprod(Z), crossprod(Z, Y[(m + 1):n, ]))
  ehat <- matrix(NA, n, 2); ehat[(m + 1):n, ] <- Y[(m + 1):n, ] - Z %*% B
  tt <- (m + 2):n
  X2 <- cbind(1, Y[tt - 1, ], ehat[tt - 1, ])
  C <- solve(crossprod(X2), crossprod(X2, Y[tt, ]))
  u <- Y[tt, ] - X2 %*% C
  S <- crossprod(u) / length(tt)
  v <- list(long_var_order = m, step2_coef = vec(C),                       # rows: const, y1_l1, y2_l1, e1_l1, e2_l1; cols: eq y1, y2
            sigma_ml = vec(S), loglik = -length(tt) / 2 * (2 * log(2 * pi) + log(det(S)) + 2), n_eff = length(tt))
  # ---- package keys ----
  h11 <- .hr_varma(Y, 1, 1, 8, "")                       # coef_y1, coef_y2, aic, bic, n_params, df_residual
  stopifnot(abs(h11$loglik - v$loglik) < 1e-9 * abs(v$loglik), h11$n_eff == 391)
  v <- c(v, h11[setdiff(names(h11), c("sigma_ml", "loglik", "n_eff"))])
  h10 <- .hr_varma(Y, 1, 0, 8, "p1q0_lv8_")              # same sample as (1,1): comparable criteria
  stopifnot(h10$p1q0_lv8_n_eff == 391, h10$p1q0_lv8_loglik < v$loglik)
  h01 <- .hr_varma(Y, 0, 1, 8, "p0q1_lv8_")              # pure VMA(1): not intercept-only
  h21 <- .hr_varma(Y, 2, 1, 8, "p2q1_lv8_")              # p > q: sample t = m + max(p, q) + 1..n
  h22 <- .hr_varma(Y, 2, 2, 10, "p2q2_lv10_")
  mdef <- .hr_default_order(n, 2, 1, 1)                  # 11 = floor(2 log(400)), cap floor(398 / 5) = 79
  hdef <- .hr_varma(Y, 1, 1, mdef, "default_")
  Ys <- Y[1:30, ]
  m0s <- max(floor(2 * log(30)), 2); mshort <- .hr_default_order(30, 2, 1, 1)   # 6 -> 5 (cap floor(28 / 5))
  stopifnot(m0s == 6, mshort == 5)
  hshort <- .hr_varma(Ys, 1, 1, mshort, "short30_")
  # q = 0 with the default order: m = 0, an OLS VAR(p) identical to vars::VAR(type = 'const')
  hq0 <- .hr_varma(Y, 2, 0, .hr_default_order(n, 2, 2, 0), "p2q0_")
  vm <- VAR(Y, p = 2, type = "const")
  vq0 <- list()
  for (eq in colnames(Y)) {
    cf <- coef(vm$varresult[[eq]]); se <- sqrt(diag(vcov(vm$varresult[[eq]])))
    vq0[[paste0("var_p2_coef_", eq)]] <- named(cf); vq0[[paste0("var_p2_se_", eq)]] <- named(se)
    hc <- hq0[[paste0("p2q0_coef_", eq)]]
    stopifnot(setequal(hc$names, names(cf)), max(abs(cf[hc$names] - hc$values)) < 1e-10)
  }
  vq0$var_p2_loglik <- as.numeric(logLik(vm)); vq0$var_p2_aic <- AIC(vm); vq0$var_p2_bic <- BIC(vm)
  vq0$var_p2_n_eff <- vm$obs
  stopifnot(abs(vq0$var_p2_loglik - hq0$p2q0_loglik) < 1e-9 * abs(vq0$var_p2_loglik), vm$obs == hq0$p2q0_n_eff,
            abs(vq0$var_p2_aic - hq0$p2q0_aic) < 1e-9 * abs(vq0$var_p2_aic))   # npar = K (1 + K p): vars' df
  # An aliased step-2 regressor (R9): ya is 1 except its last value, so ya.l1 is constant on t = 2..n and
  # lm reports NA for it (the later of two collinear columns, dqrdc2's rule); Sigma_ML stays positive definite.
  ya <- c(rep(1, n - 1), 2); tA <- 2:n
  XA <- cbind(const = 1, y1.l1 = Y[tA - 1, "y1"], ya.l1 = ya[tA - 1])
  UA <- matrix(NA_real_, length(tA), 2); va <- list()
  for (j in 1:2) {
    yj <- if (j == 1) Y[tA, "y1"] else ya[tA]; eq <- c("y1", "ya")[j]
    fit <- lm(yj ~ 0 + XA); cf <- setNames(coef(fit), sub("^XA", "", names(coef(fit))))
    stopifnot(fit$rank == 2, is.na(cf[["ya.l1"]]))
    se <- setNames(rep(NA_real_, 3), names(cf)); sm <- summary(fit)$coefficients
    se[sub("^XA", "", rownames(sm))] <- sm[, 2]
    va[[paste0("alias_coef_", eq)]] <- named(cf); va[[paste0("alias_se_", eq)]] <- named(se)
    UA[, j] <- residuals(fit)
  }
  SA <- crossprod(UA) / length(tA)
  va$alias_loglik <- -length(tA) / 2 * (2 * log(2 * pi) + as.numeric(determinant(SA)$modulus) + 2)
  va$alias_n_params <- 4; va$alias_aic <- -2 * va$alias_loglik + 2 * 4; va$alias_df_residual <- length(tA) - 2
  v <- c(v, h10, h01, h21, h22, hdef, list(default_long_var_order = mdef, short30_long_var_order = mshort,
         short30_unreduced_order = m0s), hshort, vq0, va)
  list(values = v, reference = "Hannan & Rissanen (1982) two-step VARMA estimator written out (MTS/marima not installed); each OLS step stats::lm.fit / stats::lm (summary.lm SEs and p-values for q = 0 only); logLik = -T/2 (K log 2pi + log det Sigma_ML + K); q = 0 against vars::VAR(type = 'const') and logLik.varest",
       call = "long VAR(m) by lm.fit; lm(y_t ~ 0 + [1, y_{t-1..p}, ehat_{t-1..q}]), t = m + max(p, q) + 1..n; VAR(Y, p = 2, type = 'const')", packages = "vars")
})
