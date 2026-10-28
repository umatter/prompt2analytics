# validation/r/cases/regression.R -- lm / sandwich / lmtest / car references for the OLS exhibits.
suppressPackageStartupMessages({ library(sandwich); library(lmtest); library(car); library(tseries) })

.ols_values <- function(m) {
  s <- summary(m); b <- coef(m); df <- m$df.residual; out <- list()
  for (t in c("classical", "HC0", "HC1", "HC2", "HC3")) {
    V <- if (t == "classical") vcov(m) else vcovHC(m, type = t)
    se <- sqrt(diag(V)); tag <- tolower(t)
    out[[paste0("se_", tag)]] <- named(se)
    out[[paste0("t_", tag)]] <- named(b / se)
    out[[paste0("p_", tag)]] <- named(2 * pt(abs(b / se), df, lower.tail = FALSE))
    out[[paste0("ci95_lower_", tag)]] <- named(b - qt(0.975, df) * se)
    out[[paste0("ci95_upper_", tag)]] <- named(b + qt(0.975, df) * se)
  }
  c(list(coef = named(b)), out, list(
    r2 = s$r.squared, adj_r2 = s$adj.r.squared, sigma = s$sigma,
    f_stat = unname(s$fstatistic["value"]), f_df1 = unname(s$fstatistic["numdf"]), f_df2 = unname(s$fstatistic["dendf"]),
    f_p = unname(pf(s$fstatistic["value"], s$fstatistic["numdf"], s$fstatistic["dendf"], lower.tail = FALSE)),
    log_lik = as.numeric(logLik(m)), aic = AIC(m), bic = BIC(m),
    extract_aic = unname(extractAIC(m)[2]), extract_bic = unname(extractAIC(m, k = log(nobs(m)))[2]),
    n_obs = nobs(m), df_resid = df))
}

case("ols_grunfeld", "regression", "OLS with classical, HC0-HC3 and clustered SEs, Grunfeld pooled", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = g)
  v <- .ols_values(m)
  v$se_cluster_firm_hc1 <- named(sqrt(diag(vcovCL(m, cluster = ~firm, type = "HC1"))))
  v$se_cluster_firm_hc0 <- named(sqrt(diag(vcovCL(m, cluster = ~firm, type = "HC0"))))
  v$se_cluster_year_hc1 <- named(sqrt(diag(vcovCL(m, cluster = ~year, type = "HC1"))))
  v$se_cluster_firm_year_hc1 <- named(sqrt(diag(vcovCL(m, cluster = ~firm + year, type = "HC1"))))
  b <- coef(m); se <- sqrt(diag(vcovCL(m, cluster = ~firm, type = "HC1")))
  v$p_cluster_firm_hc1 <- named(2 * pt(abs(b / se), m$df.residual, lower.tail = FALSE))
  list(values = v, reference = "stats::lm; sandwich::vcovHC; sandwich::vcovCL(type = 'HC1')",
       call = "lm(inv ~ value + capital, data = grunfeld)", packages = c("sandwich"))
})

case("ols_longley", "regression", "OLS on the ill-conditioned Longley data", function(ctx) {
  d <- ctx$gen_dataset("longley", datasets::longley)
  m <- lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces + Population + Year, data = d)
  list(values = .ols_values(m), reference = "stats::lm; sandwich::vcovHC",
       call = "lm(Employed ~ ., data = longley)", packages = c("sandwich"))
})

case("ols_mtcars", "regression", "OLS with HC0-HC3 SEs, mtcars mpg ~ wt + hp + disp", function(ctx) {
  mt <- cbind(model = rownames(datasets::mtcars), datasets::mtcars)
  d <- ctx$gen_dataset("mtcars", mt)
  m <- lm(mpg ~ wt + hp + disp, data = d)
  list(values = .ols_values(m), reference = "stats::lm; sandwich::vcovHC",
       call = "lm(mpg ~ wt + hp + disp, data = mtcars)", packages = c("sandwich"))
})

case("ols_diagnostics_grunfeld", "regression", "Regression diagnostics printed by `p2a reg diagnostics`, Grunfeld pooled", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = g)
  jb <- tseries::jarque.bera.test(residuals(m)); bp <- bptest(m); bpn <- bptest(m, studentize = FALSE)
  dw <- dwtest(m); bg1 <- bgtest(m, order = 1); bg1f <- bgtest(m, order = 1, type = "F")
  rs <- resettest(m, power = 2:3, type = "fitted"); hc <- harvtest(m)
  rsr <- resettest(m, power = 2:3, type = "regressor"); bg4 <- bgtest(m, order = 4)
  m_r <- lm(inv ~ value, data = g)
  wf <- waldtest(m, m_r, test = "F"); wc <- waldtest(m, m_r, test = "Chisq")
  list(dataset = "validation/datasets/grunfeld.csv", values = list(
    jarque_bera = unname(jb$statistic), jarque_bera_p = unname(jb$p.value),
    breusch_pagan = unname(bp$statistic), breusch_pagan_df = unname(bp$parameter), breusch_pagan_p = unname(bp$p.value),
    breusch_pagan_nonstudentized = unname(bpn$statistic), breusch_pagan_nonstudentized_p = unname(bpn$p.value),
    durbin_watson = unname(dw$statistic), durbin_watson_p_greater = unname(dw$p.value),
    vif = named(car::vif(m)), condition_number = kappa(model.matrix(m), exact = TRUE),
    bgtest1_chisq = unname(bg1$statistic), bgtest1_chisq_p = unname(bg1$p.value),
    bgtest1_f = unname(bg1f$statistic), bgtest1_f_p = unname(bg1f$p.value),
    reset_fitted = unname(rs$statistic), reset_fitted_p = unname(rs$p.value),
    harvey_collier = unname(hc$statistic), harvey_collier_p = unname(hc$p.value),
    reset_regressor = unname(rsr$statistic), reset_regressor_p = unname(rsr$p.value),
    bgtest4_chisq = unname(bg4$statistic), bgtest4_chisq_p = unname(bg4$p.value),
    wald_drop_capital_f = wf$F[2], wald_drop_capital_f_p = wf$`Pr(>F)`[2],
    wald_drop_capital_chisq = wc$Chisq[2], wald_drop_capital_chisq_p = wc$`Pr(>Chisq)`[2]),
    reference = "tseries::jarque.bera.test; lmtest::bptest (studentized); lmtest::dwtest; car::vif; base::kappa(exact = TRUE); lmtest::bgtest; lmtest::resettest; lmtest::harvtest",
    call = "lm(inv ~ value + capital, data = grunfeld)", packages = c("lmtest", "car", "tseries"))
})

# Stepwise selection (REG-20): the legacy DGP of validate_regression_diag.R and a
# borderline case where R's step() keeps x2 (|t| between sqrt(2) and sqrt(3)).
.step_values <- function(d) {
  full <- lm(y ~ x1 + x2 + x3, data = d); null <- lm(y ~ 1, data = d); n <- nrow(d)
  fw <- step(null, scope = list(lower = ~1, upper = ~ x1 + x2 + x3), direction = "forward", trace = 0)
  bw <- step(full, direction = "backward", trace = 0)
  bo <- step(full, direction = "both", trace = 0)
  bic <- step(full, direction = "backward", trace = 0, k = log(n))
  sel <- function(m) as.numeric(c("x1", "x2", "x3") %in% attr(terms(m), "term.labels"))
  list(forward_vars = vec(sel(fw)), forward_aic = extractAIC(fw)[2],
       backward_vars = vec(sel(bw)), backward_aic = extractAIC(bw)[2],
       both_vars = vec(sel(bo)), both_aic = extractAIC(bo)[2],
       backward_bic_vars = vec(sel(bic)), backward_bic = extractAIC(bic, k = log(n))[2],
       full_t = named(coef(full) / sqrt(diag(vcov(full)))),
       full_log_lik = as.numeric(logLik(full)), full_aic = AIC(full), full_bic = BIC(full))
}
case("step_legacy", "regression", "Stepwise AIC/BIC (validate_regression_diag.R DGP)", function(ctx) {
  set.seed(42); n <- 100
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); y <- 2 + 3 * x1 + 1.5 * x2 + rnorm(n, 0, 0.5)
  d <- ctx$gen_dataset("step_legacy", data.frame(y = y, x1 = x1, x2 = x2, x3 = x3))
  list(values = .step_values(d), reference = "stats::step; stats::extractAIC", call = "step(lm(y ~ x1 + x2 + x3), ...)", packages = character(0))
})
case("step_borderline", "regression", "Stepwise AIC where R keeps a regressor with |t| ~ 1.55", function(ctx) {
  set.seed(5); n <- 120
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); y <- 1 + 0.8 * x1 + 0.14 * x2 + rnorm(n)
  d <- ctx$gen_dataset("step_borderline", data.frame(y = y, x1 = x1, x2 = x2, x3 = x3))
  v <- .step_values(d)
  tx2 <- abs(v$full_t$values[v$full_t$names == "x2"])
  stopifnot(tx2 > sqrt(2), tx2 < sqrt(3), v$backward_vars[2] == 1)
  list(values = v, reference = "stats::step; stats::extractAIC", call = "step(lm(y ~ x1 + x2 + x3), direction = 'backward')", packages = character(0))
})
