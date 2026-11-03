# validation/r/cases/stats_quantile.R -- Hyndman-Fan sample quantiles, types 1-9 (T3-WP02: STAT-36;
# X-18 part), IQR(type), fivenum (Tukey hinges), boxplot.stats (coef 1.5 and 0.5), the bootstrap
# percentile rule (type 7 at (1 -/+ level) / 2), and the distribution helpers of traits::estimator
# (X-17 and X-14 parts): two-sided t, chi-square and F upper tails, normal tails and quantiles,
# t and chi-square quantiles.  The package copy (T3-WP02 plan) supersedes Appendix A.2 of the
# Tier 3 design: it adds the percentile, boxplot(coef = 0.5), DATA-62 and distribution keys.
case("stat_quantile_types", "stats/robust", "quantile() types 1-9, IQR(type), fivenum, boxplot.stats, percentile intervals; pt/pchisq/pf/pnorm upper tails and qnorm/qt/qchisq", function(ctx) {
  set.seed(20261106)
  x <- c(round(rnorm(23, 10, 3), 3), 7.5, 7.5, 12, 12, 12)            # ties; n = 28
  y <- c(3.1, 9.4, 2.2, 5.0, 7.7, 1.9, 6.4)                            # n = 7
  d <- ctx$gen_dataset("quantile_sample", data.frame(i = seq_along(x), x = x, y = c(y, rep(NA, length(x) - length(y)))))
  x <- d$x; y <- d$y[!is.na(d$y)]
  stopifnot(length(x) == 28L, length(y) == 7L, anyDuplicated(x) > 0L)
  probs <- c(0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.75, 0.9, 0.975, 1)
  v <- list(probs = vec(probs))
  for (ty in 1:9) {
    v[[paste0("x_type", ty)]] <- vec(unname(quantile(x, probs, type = ty)))
    v[[paste0("y_type", ty)]] <- vec(unname(quantile(y, probs, type = ty)))
    v[[paste0("x_iqr_type", ty)]] <- IQR(x, type = ty)
  }
  # the finding's own scenario: quantile(1:10, 0.25, type = t), t = 1..9
  v$seq10_q25 <- vec(vapply(1:9, function(t) unname(quantile(1:10, 0.25, type = t)), 0))
  # types 4-9 at p = (k - a) / (n + 1 - a - b), where nppm = a + p (n + 1 - a - b) lands on k up to
  # rounding; compared exactly.  Type 9 at k = 8 has nppm just below 8, so R's fuzz (4 eps) takes j = 8.
  ab <- list(c(4, 0, 1), c(5, 0.5, 0.5), c(6, 0, 0), c(8, 1/3, 1/3), c(9, 3/8, 3/8))
  for (t in ab) {
    pk <- ((1:28) - t[2]) / (28 + 1 - t[2] - t[3]); pk <- pk[pk >= 0 & pk <= 1]
    v[[paste0("fuzz_type", t[1], "_probs")]] <- vec(pk)
    v[[paste0("x_fuzz_type", t[1])]] <- vec(unname(quantile(x, pk, type = t[1])))
  }
  stopifnot(v$x_fuzz_type9[8] == sort(x)[8], 3/8 + v$fuzz_type9_probs[8] * (28 + 1 - 3/8 - 3/8) < 8)
  # the same p on a sample with a 1e20 gap below x_(8): with the fuzz R returns x_(8) = 1 exactly,
  # without it (j = 7, h = 1 - 9e-16) the result would be -88816.84...
  gap <- c(rep(-1e20, 7), 1:21)
  v$gap_type9_p <- v$fuzz_type9_probs[8]; v$gap_type9_q <- unname(quantile(gap, v$fuzz_type9_probs[8], type = 9))
  stopifnot(v$gap_type9_q == 1)
  v$x_fivenum <- vec(fivenum(x)); v$y_fivenum <- vec(fivenum(y))
  b <- boxplot.stats(x); v$x_boxplot_stats <- vec(b$stats); v$x_boxplot_n <- b$n; v$x_boxplot_n_out <- length(b$out)
  b5 <- boxplot.stats(x, coef = 0.5); stopifnot(length(b5$out) > 0L)
  v$x_boxplot05_stats <- vec(b5$stats); v$x_boxplot05_out <- vec(b5$out)
  by <- boxplot.stats(y); v$y_boxplot_stats <- vec(by$stats); v$y_boxplot_n_out <- length(by$out)
  b62 <- boxplot.stats(c(1, 7, 12, 15, 15, 16, 18, 18)); v$data62_boxplot_stats <- vec(b62$stats); v$data62_boxplot_n_out <- length(b62$out)
  # bootstrap percentile rule: quantile(draws, c((1 - level) / 2, (1 + level) / 2), type = 7)
  for (lv in c(0.95, 0.9, 0.5)) {
    v[[paste0("x_pct_", lv * 100)]] <- vec(unname(quantile(x, c((1 - lv) / 2, (1 + lv) / 2), type = 7)))
    v[[paste0("y_pct_", lv * 100)]] <- vec(unname(quantile(y, c((1 - lv) / 2, (1 + lv) / 2), type = 7)))
  }
  # distribution helpers (traits::estimator)
  pt_t  <- c(16.967101614364047, 13.5, 9, 12, 8, 2, 0.5, -3.690321590546858, 4.720678231125446, 2, 5, 2, 10, 1, 40)
  pt_df <- c(197, 197, 500, 500, 500, 6, 10, 197, 197, 1e6, 1e6, 1e9, 4e5, 1, 3)
  v$pt2_t <- vec(pt_t); v$pt2_df <- vec(pt_df); v$pt2_p <- vec(2 * pt(-abs(pt_t), pt_df))
  cs_x  <- c(100, 150, 3.2, 1e-16, 1e-16, 0.5, 1500, 1000)
  cs_df <- c(1, 3, 2, 0.01, 1, 10, 1000, 100)
  v$pchisq_x <- vec(cs_x); v$pchisq_df <- vec(cs_df); v$pchisq_p <- vec(pchisq(cs_x, cs_df, lower.tail = FALSE))
  f_x <- c(100, 60, 2.5, 1e8, 2.5, 60, 0.5); f_d1 <- c(1, 3, 3, 3, 10, 10, 2); f_d2 <- c(500, 40, 20, 40, 1e6, 1e6, 2)
  v$pf_x <- vec(f_x); v$pf_d1 <- vec(f_d1); v$pf_d2 <- vec(f_d2); v$pf_p <- vec(pf(f_x, f_d1, f_d2, lower.tail = FALSE))
  z <- c(9, 1.959963984540054, 0.3, 40, 0)
  v$pnorm2_z <- vec(z); v$pnorm2_p <- vec(2 * pnorm(-abs(z)))
  qn_p <- c(0.975, 0.025, 0.995, 0.95, 1e-10, 0.5, 1 - 1e-12)
  v$qnorm_p <- vec(qn_p); v$qnorm_q <- vec(qnorm(qn_p))
  lv <- c(0.95, 0.99, 0.9, 0.5); v$zcrit_level <- vec(lv); v$zcrit_z <- vec(qnorm((1 + lv) / 2))
  qt_p  <- c(0.975, 0.975, 0.025, 0.995, 0.975, 0.975, 0.975, 1e-10, 0.6, 0.95, 1e-10, 0.975, 0.9, 0.999, 0.025, 1e-12, 0.4)
  qt_df <- c(10, 197, 5, 2.5, 1e6, 2e7, 1e9, 4, 30, 3, 0.5, 1, 100, 1e4, 3e4, 20, 999)
  v$qt_p <- vec(qt_p); v$qt_df <- vec(qt_df); v$qt_q <- vec(qt(qt_p, qt_df))
  qc_p  <- c(0.95, 0.95, 0.05, 0.975, 1e-10, 0.999, 1e-10, 0.025, 0.975, 0.5, 0.6, 1 - 1e-10)
  qc_df <- c(1, 10, 4, 2.5, 0.5, 1e5, 3, 318, 320, 1e6, 0.01, 10)
  v$qchisq_p <- vec(qc_p); v$qchisq_df <- vec(qc_df); v$qchisq_q <- vec(qchisq(qc_p, qc_df))
  list(values = v, reference = "stats::quantile(type = 1..9); stats::IQR(type); stats::fivenum; grDevices::boxplot.stats; stats::pt, pchisq, pf, pnorm (lower.tail = FALSE), qnorm, qt, qchisq",
       call = "quantile(x, probs, type = t); boxplot.stats(x, coef); 2 * pt(-|t|, df); qt(p, df)", packages = character(0))
})
