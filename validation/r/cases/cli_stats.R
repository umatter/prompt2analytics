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
  # `acf` is R's estimator (denominator n). The CLI's `stats acf` passes adjusted = true (n - k
  # denominators, CLI-22, Tier 4), so `acf` is recorded for that package and not pinned here.
  list(values = list(acf = vec(a$acf[-1]), pacf = vec(p$acf), n = length(d$y)),
       reference = "stats::acf(lag.max = 5); stats::pacf(lag.max = 5) (na.action = na.fail)",
       call = "acf(y, lag.max = 5, plot = FALSE); pacf(y, lag.max = 5, plot = FALSE)", packages = character(0))
})
