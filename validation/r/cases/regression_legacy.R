# validation/r/cases/regression_legacy.R -- the data-generating code of
# validation/scripts/validate_regression_diag.R (Tests 1, 2, 4-8 and 10), ported
# verbatim (same seeds, same draws, same order) so that the data are committed and
# the Rust tests can load them. The values equal the legacy CSVs in
# validation/expected/*_test.csv where those exist; see validation/methods/regression_diagnostics.json.
# Test 3 (stepwise) is step_legacy in regression.R. Tests 9 (vcovBS: R's RNG cannot
# be matched), 11 (supsmu) and 12 (line) are not ported.
# Helpers are prefixed `.legacy_` so they cannot collide with other case files.
suppressPackageStartupMessages({
  library(lmtest); library(sandwich); library(quantreg); library(nlme); library(strucchange); library(plm)
})

.legacy_se <- function(V) named(sqrt(diag(V)))
.legacy_vcov <- function(V) vec(as.vector(t(V)))   # row-major, as p2a's Vec<Vec<f64>> flattened
.legacy_ar_series <- function(n, rho) {   # e[1] <- rnorm(1); e[i] <- rho * e[i - 1] + rnorm(1), as the legacy script
  e <- numeric(n); e[1] <- rnorm(1)
  for (i in 2:n) e[i] <- rho * e[i - 1] + rnorm(1)
  e
}

# Test 4: Breusch-Godfrey, AR(1) errors with rho = 0.5.
case("diag_bgtest_ar1", "regression/diagnostics", "Breusch-Godfrey test, AR(1) errors (rho = 0.5), n = 100 (legacy Test 4 DGP)", function(ctx) {
  set.seed(42); n <- 100
  e_ar <- .legacy_ar_series(n, 0.5)
  x <- rnorm(n)
  y <- 2 + 3 * x + e_ar
  d <- ctx$gen_dataset("bgtest_ar1", data.frame(y = y, x = x))
  m <- lm(y ~ x, data = d)
  bg1 <- bgtest(m, order = 1); bg4 <- bgtest(m, order = 4)
  bg1f <- bgtest(m, order = 1, type = "F"); bg4f <- bgtest(m, order = 4, type = "F")
  lags <- function(b) named(b$coefficients[grepl("^lag\\(resid\\)", names(b$coefficients))])
  # R-squared of the auxiliary regression (lagged residuals filled with 0), from stats::lm
  e <- residuals(m); lagk <- function(k) c(rep(0, k), e[1:(n - k)])
  aux1 <- summary(lm(e ~ d$x + lagk(1)))$r.squared
  aux4 <- summary(lm(e ~ d$x + lagk(1) + lagk(2) + lagk(3) + lagk(4)))$r.squared
  stopifnot(bg1$p.value < 0.001,   # the DGP has serial correlation; the test must detect it
            isTRUE(all.equal(n * aux1, unname(bg1$statistic), tolerance = 1e-10)))
  list(values = list(
    bgtest1_chisq = unname(bg1$statistic), bgtest1_chisq_df = unname(bg1$parameter), bgtest1_chisq_p = bg1$p.value,
    bgtest4_chisq = unname(bg4$statistic), bgtest4_chisq_df = unname(bg4$parameter), bgtest4_chisq_p = bg4$p.value,
    bgtest1_f = unname(bg1f$statistic), bgtest1_f_df1 = unname(bg1f$parameter[1]), bgtest1_f_df2 = unname(bg1f$parameter[2]), bgtest1_f_p = bg1f$p.value,
    bgtest4_f = unname(bg4f$statistic), bgtest4_f_df1 = unname(bg4f$parameter[1]), bgtest4_f_df2 = unname(bg4f$parameter[2]), bgtest4_f_p = bg4f$p.value,
    bgtest1_lag_coef = lags(bg1), bgtest4_lag_coef = lags(bg4),
    bgtest1_aux_r2 = aux1, bgtest4_aux_r2 = aux4,
    n_obs = nobs(m)),
    reference = "lmtest::bgtest(order = 1 | 4, type = 'Chisq' | 'F', fill = 0)",
    call = "bgtest(lm(y ~ x, data = bgtest_ar1), order = 1)", packages = "lmtest")
})

# Test 5: RESET, quadratic truth fitted linearly.
case("diag_reset_quad", "regression/diagnostics", "RESET test (fitted and regressor, powers 2:3), quadratic DGP, n = 100 (legacy Test 5 DGP)", function(ctx) {
  set.seed(42); n <- 100
  x <- runif(n, 1, 10)
  y <- 2 + 3 * x + 0.5 * x^2 + rnorm(n, 0, 2)
  d <- ctx$gen_dataset("reset_quad", data.frame(y = y, x = x))
  m <- lm(y ~ x, data = d)
  rf <- resettest(m, power = 2:3, type = "fitted"); rr <- resettest(m, power = 2:3, type = "regressor")
  d$fit <- fitted(m)
  m_fit <- lm(y ~ x + I(fit^2) + I(fit^3), data = d); m_reg <- lm(y ~ x + I(x^2) + I(x^3), data = d)
  stopifnot(rf$p.value < 1e-10)
  list(values = list(
    reset_fitted = unname(rf$statistic), reset_fitted_df1 = unname(rf$parameter[1]), reset_fitted_df2 = unname(rf$parameter[2]), reset_fitted_p = rf$p.value,
    reset_regressor = unname(rr$statistic), reset_regressor_df1 = unname(rr$parameter[1]), reset_regressor_df2 = unname(rr$parameter[2]), reset_regressor_p = rr$p.value,
    r2_original = summary(m)$r.squared, r2_augmented_fitted = summary(m_fit)$r.squared, r2_augmented_regressor = summary(m_reg)$r.squared,
    n_obs = nobs(m)),
    reference = "lmtest::resettest(power = 2:3, type = 'fitted' | 'regressor'); R-squared of the augmented models from stats::lm",
    call = "resettest(lm(y ~ x, data = reset_quad), power = 2:3, type = 'fitted')", packages = "lmtest")
})

# Test 6: Harvey-Collier, concave truth fitted linearly.
case("diag_harvtest_nonlin", "regression/diagnostics", "Harvey-Collier test, concave DGP, n = 100 (legacy Test 6 DGP)", function(ctx) {
  set.seed(42); n <- 100
  x <- seq(1, 10, length.out = n)
  y <- 5 + 2 * x - 0.2 * x^2 + rnorm(n, 0, 0.5)
  d <- ctx$gen_dataset("harvtest_nonlin", data.frame(y = y, x = x))
  m <- lm(y ~ x, data = d)
  hc <- harvtest(m); w <- strucchange::recresid(m)
  stopifnot(hc$p.value < 1e-6, mean(w) < 0)   # concave truth: the recursive residuals have a negative mean
  list(values = list(
    harvey_collier = unname(hc$statistic), harvey_collier_df = unname(hc$parameter), harvey_collier_p = hc$p.value,
    recresid_mean = mean(w), recresid_sd = sd(w), recresid_se_mean = sd(w) / sqrt(length(w)), n_recresid = length(w),
    n_obs = nobs(m)),
    reference = "lmtest::harvtest (statistic is |t|); strucchange::recresid for the signed mean",
    call = "harvtest(lm(y ~ x, data = harvtest_nonlin))", packages = c("lmtest", "strucchange"))
})

# Test 7: Wald test of nested models (legacy: aod::wald.test with vcov(lm), i.e. the Chisq form).
case("diag_wald_sim", "regression/diagnostics", "Wald test of nested linear models, n = 100 (legacy Test 7 DGP)", function(ctx) {
  set.seed(42); n <- 100
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
  y <- 2 + 3 * x1 + 0.001 * x2 + 1 * x3 + rnorm(n, 0, 0.5)
  d <- ctx$gen_dataset("wald_sim", data.frame(y = y, x1 = x1, x2 = x2, x3 = x3))
  m <- lm(y ~ x1 + x2 + x3, data = d); m13 <- lm(y ~ x1 + x3, data = d); m1 <- lm(y ~ x1, data = d)
  w1f <- waldtest(m, m13, test = "F"); w1c <- waldtest(m, m13, test = "Chisq")
  w2f <- waldtest(m, m1, test = "F"); w2c <- waldtest(m, m1, test = "Chisq")
  a1 <- aod::wald.test(b = coef(m), Sigma = vcov(m), Terms = 3)$result$chi2
  a2 <- aod::wald.test(b = coef(m), Sigma = vcov(m), Terms = c(3, 4))$result$chi2
  stopifnot(isTRUE(all.equal(unname(a1["chi2"]), w1c$Chisq[2], tolerance = 1e-12)),
            isTRUE(all.equal(unname(a2["chi2"]), w2c$Chisq[2], tolerance = 1e-12)))
  list(values = list(
    wald_drop_x2_f = w1f$F[2], wald_drop_x2_f_df1 = abs(w1f$Df[2]), wald_drop_x2_f_df2 = w1f$Res.Df[1], wald_drop_x2_f_p = w1f$`Pr(>F)`[2],
    wald_drop_x2_chisq = w1c$Chisq[2], wald_drop_x2_chisq_p = w1c$`Pr(>Chisq)`[2],
    wald_drop_x2_x3_f = w2f$F[2], wald_drop_x2_x3_f_df1 = abs(w2f$Df[2]), wald_drop_x2_x3_f_df2 = w2f$Res.Df[1], wald_drop_x2_x3_f_p = w2f$`Pr(>F)`[2],
    wald_drop_x2_x3_chisq = w2c$Chisq[2], wald_drop_x2_x3_chisq_p = w2c$`Pr(>Chisq)`[2],
    aod_chi2_x2 = unname(a1["chi2"]), aod_chi2_x2_p = unname(a1["P"]),
    aod_chi2_x2_x3 = unname(a2["chi2"]), aod_chi2_x2_x3_p = unname(a2["P"]),
    rss_unrestricted = deviance(m), rss_drop_x2 = deviance(m13), rss_drop_x2_x3 = deviance(m1),
    k_unrestricted = length(coef(m)), k_drop_x2 = length(coef(m13)), k_drop_x2_x3 = length(coef(m1)),
    n_obs = nobs(m)),
    reference = "lmtest::waldtest(test = 'F' | 'Chisq') (vcov = classical); aod::wald.test (the legacy reference, equal to the Chisq form)",
    call = "waldtest(lm(y ~ x1 + x2 + x3), lm(y ~ x1 + x3), test = 'F')", packages = c("lmtest", "aod"))
})
