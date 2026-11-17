# validation/r/cases/cli_stats.R -- references for the p2a CLI stats/viz commands that have no
# dataset wrapper in p2a-core (T2-WP21: CLI-8, CLI-9). They read the committed datasets of the
# T2-WP06 (two_sample_na) and T2-WP07 (friedman_quade) cases; no dataset is generated here.

# One-sample t-test on an integer column with missing cells, Tukey's five numbers, IQR and MAD
# with missing cells (all computed on the present values, as fivenum does by default and the
# CLI documents for IQR/MAD), and the complete-case correlation matrix behind `viz heatmap`.
case("cli_stat_descriptive_na", "cli/stats", "CLI t-test-one on Int64, fivenum/IQR/MAD and complete-case correlations with NA cells", function(ctx) {
  d <- ctx$dataset("datasets/generated/two_sample_na.csv")
  stopifnot(is.integer(d$xi), sum(is.na(d$xi)) == 2, sum(is.na(d$x)) == 15, nrow(d) == 22)
  ti <- t.test(d$xi, mu = 48, conf.level = 0.9)             # t.test drops NA itself
  cc <- complete.cases(d$x, d$y, d$xi)
  cm <- cor(d[, c("x", "y", "xi")], use = "complete.obs")   # symmetric: row- and column-major agree
  list(values = list(
    int_one_t = unname(ti$statistic), int_one_df = unname(ti$parameter), int_one_p = ti$p.value,
    int_one_ci90 = vec(ti$conf.int), int_one_mean = unname(ti$estimate), int_one_se = ti$stderr,
    int_one_n = sum(!is.na(d$xi)),
    fivenum_x = vec(fivenum(d$x)), iqr_x = IQR(d$x, na.rm = TRUE), mad_x = mad(d$x, na.rm = TRUE),
    n_x = sum(!is.na(d$x)),
    fivenum_xi = vec(fivenum(d$xi)), iqr_xi = IQR(d$xi, na.rm = TRUE), mad_xi = mad(d$xi, na.rm = TRUE),
    n_xi = sum(!is.na(d$xi)),
    cor_complete = vec(cm), n_complete = sum(cc), n_rows = nrow(d)),
    reference = "stats::t.test(mu = 48, conf.level = 0.9); stats::fivenum; stats::IQR(na.rm = TRUE); stats::mad(na.rm = TRUE); stats::cor(use = 'complete.obs')",
    call = "t.test(xi, mu = 48, conf.level = 0.9); fivenum(x); IQR(x, na.rm = TRUE); mad(x, na.rm = TRUE); cor(d[, c('x', 'y', 'xi')], use = 'complete.obs')",
    packages = character(0))
})

# ACF and PACF of an integer series (polars reads it as Int64); acf and pacf refuse a series with a
# missing value (na.action = na.fail), which is what the CLI does.
case("cli_stat_acf_int", "cli/stats", "CLI ACF/PACF of an Int64 series; NA refused as acf(na.action = na.fail)", function(ctx) {
  d <- ctx$dataset("datasets/generated/friedman_quade.csv")
  stopifnot(is.integer(d$y), !anyNA(d$y), sum(is.na(d$y_na)) == 1)
  a <- acf(d$y, lag.max = 5, plot = FALSE)
  p <- pacf(d$y, lag.max = 5, plot = FALSE)
  e_acf <- tryCatch({ acf(d$y_na, lag.max = 5, plot = FALSE); "no error" }, error = conditionMessage)
  e_pacf <- tryCatch({ pacf(d$y_na, lag.max = 5, plot = FALSE); "no error" }, error = conditionMessage)
  stopifnot(e_acf == "missing values in object", e_pacf == "missing values in object")
  # `acf` is R's estimator (denominator n), which `stats acf` reports.
  list(values = list(acf = vec(a$acf[-1]), pacf = vec(p$acf), n = length(d$y)),
       reference = "stats::acf(lag.max = 5); stats::pacf(lag.max = 5) (na.action = na.fail)",
       call = "acf(y, lag.max = 5, plot = FALSE); pacf(y, lag.max = 5, plot = FALSE)", packages = character(0))
})

# chisq.test on a column of counts (the CLI's `stats chisq-gof --counts`), uniform and with
# probabilities, plus the reviewer's die example. A counts vector with NA or a negative entry is
# refused ("all entries of 'x' must be nonnegative and finite").
case("cli_stat_chisq_gof_counts", "cli/stats", "chisq.test on a counts vector, uniform and with given probabilities", function(ctx) {
  set.seed(20261117)
  pr <- c(0.1, 0.15, 0.2, 0.25, 0.2, 0.1)
  counts <- as.vector(rmultinom(1, 120, pr))
  d <- ctx$gen_dataset("cli_chisq_counts", data.frame(label = letters[1:6], n = as.integer(counts), p = pr))
  stopifnot(is.integer(d$n), identical(d$n, as.integer(c(12, 17, 25, 34, 25, 7))))
  g0 <- chisq.test(d$n); g1 <- chisq.test(d$n, p = d$p)
  die <- chisq.test(c(12, 12, 12, 12, 12, 0))
  e_na <- tryCatch({ chisq.test(c(1, NA, 3)); "no error" }, error = conditionMessage)
  e_neg <- tryCatch({ chisq.test(c(1, -1, 3)); "no error" }, error = conditionMessage)
  stopifnot(e_na == "all entries of 'x' must be nonnegative and finite", e_neg == e_na)
  list(values = list(
    counts = vec(d$n), probs = vec(d$p),
    uniform_statistic = unname(g0$statistic), uniform_df = unname(g0$parameter), uniform_p = g0$p.value,
    uniform_expected = vec(g0$expected), uniform_residuals = vec(g0$residuals),
    probs_statistic = unname(g1$statistic), probs_df = unname(g1$parameter), probs_p = g1$p.value,
    probs_expected = vec(g1$expected), probs_residuals = vec(g1$residuals),
    die_statistic = unname(die$statistic), die_df = unname(die$parameter), die_p = die$p.value),
    reference = "stats::chisq.test(x) and chisq.test(x, p = p) on a counts vector",
    call = "chisq.test(n); chisq.test(n, p = p)", packages = character(0),
    dataset = "validation/datasets/generated/cli_chisq_counts.csv")
})

# R's default lag ranges: acf/pacf floor(10 log10 n) capped at n - 1, ccf floor(10 log10(n / 2));
# Box.test's default lag 1. `cli_ts_short` is the first 10 values of friedman_quade's y, where the
# n - 1 cap binds.
case("cli_stat_ts_defaults", "cli/stats", "Default lag.max of acf, pacf and ccf on long and short series; Box.test's default lag", function(ctx) {
  fq <- ctx$dataset("datasets/generated/friedman_quade.csv")
  y24 <- fq$y
  stopifnot(is.integer(y24), length(y24) == 24)
  sh <- ctx$gen_dataset("cli_ts_short", data.frame(y = y24[1:10]))
  y10 <- sh$y
  d <- ctx$dataset("datasets/generated/ts_acf_spline_t3.csv")
  x <- d$x; y <- d$y; n <- length(x)
  cc <- ccf(x, y, plot = FALSE)
  stopifnot(max(cc$lag) == floor(10 * log10(n / 2)))
  a24 <- acf(y24, plot = FALSE); p24 <- pacf(y24, plot = FALSE)
  a10 <- acf(y10, plot = FALSE); p10 <- pacf(y10, plot = FALSE)
  stopifnot(max(a24$lag) == 13, max(a10$lag) == 9)
  bp <- Box.test(x, type = "Box-Pierce"); lb <- Box.test(x, type = "Ljung-Box")
  lb10 <- Box.test(x, lag = 10, type = "Ljung-Box")
  list(values = list(
    ccf_lag_max = max(cc$lag), ccf_lags = vec(cc$lag), ccf = vec(cc$acf),
    acf24_lag_max = max(a24$lag), acf24 = vec(a24$acf[-1]), pacf24 = vec(p24$acf),
    acf10_lag_max = max(a10$lag), acf10 = vec(a10$acf[-1]), pacf10 = vec(p10$acf),
    box_bp_statistic = unname(bp$statistic), box_bp_df = unname(bp$parameter), box_bp_p = bp$p.value,
    box_lb_statistic = unname(lb$statistic), box_lb_df = unname(lb$parameter), box_lb_p = lb$p.value,
    box_lb10_statistic = unname(lb10$statistic), box_lb10_df = unname(lb10$parameter), box_lb10_p = lb10$p.value,
    n_long = n, n24 = length(y24), n10 = length(y10)),
    reference = "stats::acf, pacf, ccf at their default lag.max; stats::Box.test at its default lag (1) and lag 10",
    call = "ccf(x, y); acf(y24); pacf(y24); acf(y10); pacf(y10); Box.test(x); Box.test(x, type = 'Ljung-Box')",
    packages = character(0), dataset = "validation/datasets/generated/ts_acf_spline_t3.csv")
})

# The intervals `viz coefplot` draws: lm(y ~ x + w) with classical (confint) and HC1 (coefci) intervals.
case("cli_viz_coefplot_ci", "cli/stats", "Coefficient intervals of lm(y ~ x + w): confint and coefci with HC1 at 90% and 95%", function(ctx) {
  suppressPackageStartupMessages({library(sandwich); library(lmtest)})
  d <- ctx$dataset("datasets/generated/iv_weak_controls.csv")
  o <- lm(y ~ x + w, data = d)
  ci <- confint(o, level = 0.9)
  h90 <- coefci(o, vcov. = vcovHC(o, type = "HC1"), level = 0.9)
  h95 <- coefci(o, vcov. = vcovHC(o, type = "HC1"))
  list(values = list(coef = named(coef(o)),
    ci90_classical_lower = named(ci[, 1]), ci90_classical_upper = named(ci[, 2]),
    ci90_hc1_lower = named(h90[, 1]), ci90_hc1_upper = named(h90[, 2]),
    ci95_hc1_lower = named(h95[, 1]), ci95_hc1_upper = named(h95[, 2]), n = nobs(o)),
    reference = "stats::confint(lm, level = 0.9); lmtest::coefci(vcov. = sandwich::vcovHC(type = 'HC1'), level = 0.9 and 0.95)",
    call = "lm(y ~ x + w, data = iv_weak_controls)", packages = c("sandwich", "lmtest"))
})
