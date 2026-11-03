# validation/r/cases/ts_var.R -- VAR, impulse-response and Granger references (T3-WP12: ECON-49,
# ECON-50, ECON-53, ECON-79, ECON-90, X-12).
#  * var_canada: vars::VAR(Canada, p = 2, type = "const"). Per equation the lm fit: coefficients,
#    SEs, t values and p-values (residual df T - Kp - 1 = 73). Sigma_u with divisor T - Kp - 1
#    (summary.varest()$covres, used for SEs and the orthogonalised IRF) and with divisor T (ML, used
#    in the likelihood). logLik (log det of the ML Sigma) and stats::AIC/BIC (df = K (Kp + 1) = 36).
#    Orthogonalised (Psi: Phi_h P, P = t(chol(Sigma_df))) and plain (Phi) impulse responses for
#    n.ahead = 12 (13 horizons 0..12) for every impulse. lmtest::grangertest(prod ~ e, order = 2) on
#    the same data. vars::causality (the system Wald F test, F = 6.276811226479171 with df (6, 292))
#    is not recorded: p2a has no system Granger test.
#  * granger_short: lmtest::grangertest(y ~ x, order = p) on a 20-row series, p = 1..6
#    (df2 = n - 3p - 1, from 16 down to 1). Order 7 has 15 coefficients on 13 rows and R stops with
#    "there are aliased coefficients in the model"; p2a refuses it too. The default order is p2a's
#    own (R's grangertest defaults to order = 1): Schwert's floor(12 (n / 100)^(1/4)) = 8, capped at
#    (n - 2) %/% 3 = 6 so that df2 >= 1 (written out below, not an R function).
suppressPackageStartupMessages({ library(vars); library(lmtest) })

case("var_canada", "econometrics/timeseries", "VAR(2) const on vars::Canada: coefficients, SEs, t, p, Sigma_u, logLik/AIC/BIC, orthogonalised and plain IRF, Granger F", function(ctx) {
  data(Canada, package = "vars")
  d <- ctx$gen_dataset("var_canada", as.data.frame(Canada))
  x <- as.matrix(d)
  m <- VAR(x, p = 2, type = "const")
  v <- list()
  for (eq in colnames(x)) {
    ct <- coef(summary(m$varresult[[eq]]))
    v[[paste0("coef_", eq)]] <- named(ct[, "Estimate"])
    v[[paste0("se_", eq)]] <- named(ct[, "Std. Error"])
    v[[paste0("t_", eq)]] <- named(ct[, "t value"])
    v[[paste0("p_", eq)]] <- named(ct[, "Pr(>|t|)"])
  }
  res <- resid(m); Tn <- nrow(res); K <- ncol(x); p <- 2
  stopifnot(Tn == m$obs, ncol(m$datamat) - K == K * p + 1)
  v$sigma_df <- vec(crossprod(res) / (Tn - K * p - 1))          # = summary(m)$covres
  v$sigma_ml <- vec(crossprod(res) / Tn)
  v$sigma_ml_logdet <- log(det(crossprod(res) / Tn))
  v$loglik <- as.numeric(logLik(m))
  v$aic_stats <- AIC(m); v$bic_stats <- BIC(m)
  ir <- irf(m, n.ahead = 12, ortho = TRUE, boot = FALSE)
  irn <- irf(m, n.ahead = 12, ortho = FALSE, boot = FALSE)
  for (imp in colnames(x)) {                                    # (n.ahead + 1) x K, column-major (response-major)
    v[[paste0("irf_ortho_", imp)]] <- vec(ir$irf[[imp]])
    v[[paste0("irf_plain_", imp)]] <- vec(irn$irf[[imp]])
  }
  g <- grangertest(d$prod ~ d$e, order = 2)
  v$grangertest_prod_on_e_F <- g$F[2]; v$grangertest_prod_on_e_p <- g$`Pr(>F)`[2]
  v$grangertest_prod_on_e_df2 <- g$Res.Df[1]
  v$n_rows <- nrow(x); v$n_obs <- Tn; v$df_resid <- Tn - K * p - 1
  list(values = v, reference = "vars::VAR(p = 2, type = 'const') and summary(); stats::logLik/AIC/BIC; vars::irf(n.ahead = 12, ortho = TRUE | FALSE, boot = FALSE); lmtest::grangertest(order = 2)",
       call = "VAR(Canada, p = 2, type = 'const')", packages = c("vars", "lmtest"))
})

case("granger_short", "econometrics/timeseries", "Granger tests on a short series (n = 20) at lags 1-6 (ECON-53: df2 = n - 3p - 1; default lag capped at 6)", function(ctx) {
  set.seed(20261113)
  n <- 20; x <- cumsum(rnorm(n)); y <- 0.5 * c(0, x[-n]) + rnorm(n)
  d <- ctx$gen_dataset("granger_short", data.frame(x = round(x, 10), y = round(y, 10)))
  n <- nrow(d)
  v <- list()
  for (p in 1:6) {
    g <- grangertest(d$y ~ d$x, order = p)
    v[[paste0("lag", p, "_F")]] <- g$F[2]; v[[paste0("lag", p, "_p")]] <- g$`Pr(>F)`[2]
    v[[paste0("lag", p, "_df1")]] <- -g$Df[2]; v[[paste0("lag", p, "_df2")]] <- g$Res.Df[1]
    v[[paste0("lag", p, "_n_used")]] <- n - p
    stopifnot(g$Res.Df[1] == n - 3 * p - 1)
  }
  stopifnot(inherits(tryCatch(grangertest(d$y ~ d$x, order = 7), error = function(e) e), "error"))
  v$schwert_lag <- floor(12 * (n / 100)^0.25)
  v$default_lag <- min(v$schwert_lag, (n - 2) %/% 3)
  stopifnot(v$schwert_lag == 8, v$default_lag == 6)
  list(values = v, reference = "lmtest::grangertest(y ~ x, order = 1..6); p2a's default order written out: min(floor(12 (n/100)^(1/4)), (n - 2) %/% 3)",
       call = "grangertest(y ~ x, order = p)", packages = "lmtest")
})
