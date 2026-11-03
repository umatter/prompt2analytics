# validation/r/cases/cli_flags.R -- references for the p2a CLI flags that T3-WP06 makes settable
# (CLI-5: boolean flags whose default is true; CLI-18: `reg clustered --cluster2`) and for the
# no-intercept fit statistics that those flags expose (REG-19, moved in from Tier 4).
#
# ols_no_intercept_grunfeld and ols_twoway_cluster_grunfeld read T1's committed
# validation/datasets/grunfeld.csv (plm::Grunfeld: firm, year, inv, value, capital; n = 200) and
# generate nothing. ts_ar_yw_aic generates ts_flags_sim.csv (an AR(2) series and a trending monthly
# series with two outliers).

# Regression through the origin, as summary.lm computes it when attr(terms, "intercept") == 0:
# R^2 = sum(fitted^2) / (sum(fitted^2) + RSS), adjusted R^2 with n / (n - k), F on k numerator df.
# Plus HC1, one- and two-way CR1 and Newey-West SEs, backward step, and rq (recorded only).
case("ols_no_intercept_grunfeld", "regression", "lm(inv ~ 0 + value + capital): uncentered R^2, HC1, CR1 (firm, firm + year), NeweyWest(lag = 4), step", function(ctx) {
  suppressPackageStartupMessages({ library(sandwich); library(quantreg) })
  g <- ctx$dataset("datasets/grunfeld.csv")
  stopifnot(nrow(g) == 200, length(unique(g$firm)) == 10, length(unique(g$year)) == 20)
  n <- nrow(g)
  m0 <- lm(inv ~ 0 + value + capital, data = g)
  stopifnot(attr(terms(m0), "intercept") == 0)
  s0 <- summary(m0); b <- coef(m0); df <- m0$df.residual
  # the uncentered definitions written out agree with summary.lm
  r2 <- sum(fitted(m0)^2) / (sum(fitted(m0)^2) + sum(resid(m0)^2))
  stopifnot(abs(r2 - s0$r.squared) < 1e-15, abs(1 - (1 - r2) * n / df - s0$adj.r.squared) < 1e-15)
  tp <- function(V) { se <- sqrt(diag(V)); list(se = named(se), t = named(b / se), p = named(2 * pt(-abs(b / se), df))) }
  cl <- tp(vcov(m0)); hc1 <- tp(vcovHC(m0, type = "HC1"))
  c1 <- tp(vcovCL(m0, cluster = ~firm, type = "HC1")); c2 <- tp(vcovCL(m0, cluster = ~firm + year, type = "HC1"))
  nw <- tp(NeweyWest(m0, lag = 4, prewhite = FALSE))   # p2a's Bartlett bandwidth 4 (T1 case hac_ar1)
  f <- s0$fstatistic
  st <- step(m0, direction = "backward", trace = 0)    # both terms stay: dropping either raises the AIC
  d1 <- drop1(m0)
  list(values = list(
    coef = named(b),
    se_classical = cl$se, t_classical = cl$t, p_classical = cl$p,
    se_hc1 = hc1$se, t_hc1 = hc1$t, p_hc1 = hc1$p,
    se_cluster_firm_hc1 = c1$se, t_cluster_firm_hc1 = c1$t, p_cluster_firm_hc1 = c1$p,
    se_cluster_firm_year_hc1 = c2$se, t_cluster_firm_year_hc1 = c2$t, p_cluster_firm_year_hc1 = c2$p,
    se_nw_lag4 = nw$se, t_nw_lag4 = nw$t, p_nw_lag4 = nw$p,
    r2 = s0$r.squared, adj_r2 = s0$adj.r.squared, sigma = s0$sigma,
    f_stat = unname(f["value"]), f_df1 = unname(f["numdf"]), f_df2 = unname(f["dendf"]),
    f_p = unname(pf(f["value"], f["numdf"], f["dendf"], lower.tail = FALSE)),
    log_lik = as.numeric(logLik(m0)), aic = AIC(m0),
    extract_aic = unname(extractAIC(m0)[2]), extract_bic = unname(extractAIC(m0, k = log(n))[2]),
    drop1_aic = named(setNames(d1$AIC, c("none", "value", "capital"))),
    step_backward_coef = named(coef(st)),
    rq_coef = named(coef(rq(inv ~ 0 + value + capital, tau = 0.5, data = g))),   # recorded only (REG-52, T3-WP10)
    n_obs = n, df_resid = df),
    reference = "stats::lm/summary.lm without intercept (uncentered R^2, n/(n - k), F on k df); sandwich 3.1.3 vcovHC(type = 'HC1'), vcovCL(cluster = ~firm and ~firm + year, type = 'HC1'), NeweyWest(lag = 4, prewhite = FALSE); stats::step(direction = 'backward'), drop1, extractAIC; quantreg::rq(tau = 0.5) recorded",
    call = "m0 <- lm(inv ~ 0 + value + capital, data = grunfeld); summary(m0); vcovCL(m0, cluster = ~firm + year, type = 'HC1'); NeweyWest(m0, lag = 4, prewhite = FALSE); step(m0, direction = 'backward')",
    packages = c("sandwich", "quantreg"))
})

# The t and p columns that `reg clustered --cluster firm --cluster2 year` prints for the model with
# intercept. The SEs are T1's ols_grunfeld se_cluster_firm_year_hc1 (the CLI test reads them there).
case("ols_twoway_cluster_grunfeld", "regression", "lm(inv ~ value + capital) with vcovCL(~firm + year, HC1): t, p and cluster counts", function(ctx) {
  suppressPackageStartupMessages(library(sandwich))
  g <- ctx$dataset("datasets/grunfeld.csv")
  m <- lm(inv ~ value + capital, data = g)
  se <- sqrt(diag(vcovCL(m, cluster = ~firm + year, type = "HC1")))
  t <- coef(m) / se
  list(values = list(
    t_cluster_firm_year_hc1 = named(t), p_cluster_firm_year_hc1 = named(2 * pt(-abs(t), m$df.residual)),
    n_clusters_firm = length(unique(g$firm)), n_clusters_year = length(unique(g$year)), n_obs = nrow(g)),
    reference = "sandwich 3.1.3 vcovCL(cluster = ~firm + year, type = 'HC1'): t = coef / se, p = 2 pt(-|t|, 197)",
    call = "vcovCL(lm(inv ~ value + capital, data = grunfeld), cluster = ~firm + year, type = 'HC1')",
    packages = c("sandwich"))
})

# ar(method = "yule-walker") with AIC order selection (the CLI default), with aic = FALSE (order.max =
# floor(10 log10 n) = 20 at n = 120: what `ts ar --aic false` must fit) and with a fixed order 2.
# var.pred is recorded only: p2a omits ar.yw's n / (n - (order + 1)) factor (TS-15, Tier 4).
# y_seas (trend, monthly season, two outliers) feeds the `ts stl --robust` and `ts spectrum --detrend`
# CLI tests, which compare with p2a-core in process: R's stl and spec.pgram differ from p2a for
# reasons outside T3-WP06 (TS-26: f32 STL and its default windows; STAT-68: periodogram scale and taper).
case("ts_ar_yw_aic", "timeseries", "ar(method = 'yule-walker'): aic = TRUE (order 2), aic = FALSE (order.max 20), order.max = 2", function(ctx) {
  set.seed(20261022); n <- 120
  e <- rnorm(n + 50); x <- numeric(n + 50)
  for (t in 3:(n + 50)) x[t] <- 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
  tt <- 1:n
  y_seas <- 10 + 0.05 * tt + 2 * sin(2 * pi * tt / 12) + rnorm(n, sd = 0.4)
  y_seas[c(6, 75)] <- y_seas[c(6, 75)] + c(8, -6)
  d <- ctx$gen_dataset("ts_flags_sim", data.frame(t = tt, y_ar = round(x[51:(n + 50)] + 5, 10), y_seas = round(y_seas, 10)))
  y <- d$y_ar
  a <- ar(y, method = "yule-walker")
  a0 <- ar(y, aic = FALSE, method = "yule-walker")
  a2 <- ar(y, aic = FALSE, order.max = 2, method = "yule-walker")
  stopifnot(a$order == 2, a0$order == floor(10 * log10(n)), a0$order == 20)
  list(values = list(
    n_obs = n, x_mean = a$x.mean,
    aic_order = a$order, aic_ar = vec(a$ar), aic_rel = vec(a$aic),
    noaic_order = a0$order, noaic_ar = vec(a0$ar),
    order2_ar = vec(a2$ar),
    aic_var_pred = a$var.pred, noaic_var_pred = a0$var.pred),   # recorded only (TS-15)
    reference = "stats::ar(method = 'yule-walker') with aic = TRUE, aic = FALSE (order.max = floor(10 log10 n)), and aic = FALSE, order.max = 2",
    call = "ar(y_ar, method = 'yule-walker'); ar(y_ar, aic = FALSE, method = 'yule-walker'); ar(y_ar, aic = FALSE, order.max = 2, method = 'yule-walker')",
    packages = character(0))
})
