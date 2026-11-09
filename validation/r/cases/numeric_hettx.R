# validation/r/cases/numeric_hettx.R -- (X-18): type-7 quantiles and IQR of an effect vector
# with ties (the hettx effect summary and IQR statistic), and lm on a subset whose design is aliased
# (dqrdc2: the later column is NA) with its predictions for all rows (hettx regression imputation).

case("qtl_hettx_summary", "econometrics/hettx", "type-7 quantiles and IQR of an effect vector with ties (hettx effect summary)", function(ctx) {
  set.seed(20261026)
  tau <- c(round(rnorm(34, 1, 2), 2), 0.5, 0.5, 2, 2)   # n = 38: (n - 1) p is fractional at 0.25, 0.5, 0.75
  d <- ctx$gen_dataset("qtl_hettx_summary", data.frame(tau = tau))
  tau <- d$tau
  probs <- c(0.10, 0.25, 0.50, 0.75, 0.90)
  q7 <- unname(quantile(tau, probs, type = 7))
  s <- sort(tau); n <- length(s)                                          # p2a's old index rule (0-based in Rust)
  old <- c(s[floor(n * 0.10) + 1], s[n %/% 4 + 1], s[n %/% 2 + 1], s[(3 * n) %/% 4 + 1], s[min(floor(n * 0.90), n - 1) + 1])
  stopifnot(sum(old != q7) >= 4, s[(3 * n) %/% 4 + 1] - s[n %/% 4 + 1] != IQR(tau))
  list(values = list(probs = vec(probs), quantiles = vec(q7), iqr = IQR(tau),
                     min = min(tau), max = max(tau), sd = sd(tau), quantiles_old_rule = vec(old),
                     iqr_old_rule = s[(3 * n) %/% 4 + 1] - s[n %/% 4 + 1]),
       reference = "stats::quantile(type = 7); stats::IQR", call = "quantile(tau, c(0.1, 0.25, 0.5, 0.75, 0.9)); IQR(tau)",
       packages = character(0))
})

case("inv_hettx_outcome_model", "econometrics/hettx", "lm on an aliased subset design and its predictions for all rows", function(ctx) {
  set.seed(20261027)
  n <- 40
  treat <- rep(c(0L, 1L), 20)
  x1 <- round(rnorm(n), 6)
  x2 <- ifelse(treat == 1L, 2 * x1 + 1, round(rnorm(n, 0, 1.5), 6))   # aliased with (1, x1) among the treated
  y <- round(1 + 0.5 * x1 - 0.3 * x2 + 0.8 * treat + 0.2 * rnorm(n), 6)
  d <- ctx$gen_dataset("inv_hettx_outcome", data.frame(y = y, treat = treat, x1 = x1, x2 = x2))
  fit <- lm(y ~ x1 + x2, data = d, subset = treat == 1L)
  stopifnot(is.na(coef(fit)["x2"]))
  pred <- suppressWarnings(predict(fit, newdata = d))                 # "prediction from rank-deficient fit"
  Xs <- cbind(1, d$x1, d$x2)[d$treat == 1L, ]; ys <- d$y[d$treat == 1L]
  bg <- MASS::ginv(crossprod(Xs)) %*% crossprod(Xs, ys)                # what a generalized inverse gives
  pred_ginv <- drop(cbind(1, d$x1, d$x2) %*% bg)
  stopifnot(max(abs(pred_ginv - pred)) > 1e-2)
  ss <- sum(residuals(fit)^2); sst <- sum((ys - mean(ys))^2)
  list(values = list(coef = named(coef(fit)), predictions = vec(unname(pred)), r2 = 1 - ss / sst,
                     predictions_ginv = vec(pred_ginv)),
       reference = "stats::lm (dqrdc2: the later column x2 is NA); predict.lm on the rank-deficient fit",
       call = "lm(y ~ x1 + x2, subset = treat == 1); predict(fit, newdata = d)", packages = c("MASS"))
})
