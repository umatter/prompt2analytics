# validation/r/cases/numeric_tails.R -- (X-17): upper tails far below 1e-16, where `1 - cdf` returns
# exactly 0 (mcnemar.test, oneway.test with both variance assumptions, prop.trend.test, var.test), and F
# quantiles at large df. R's own var.test computes 1 - pf and returns 0 at an extreme ratio; the accurate
# tail pf(lower.tail = FALSE) is recorded next to it. Above df 4e5 R's qf returns qchisq(p, df1) / df1,
# which ignores the denominator's variance (2.6e-4 off at df 2e7, where pf of its 0.975 point is 0.083);
# the exact quantile is taken from qbeta, F = x / (1 - x) for x ~ Beta(df / 2, df / 2).

case("tail_mcnemar", "stats/mcnemar", "McNemar chi-square upper tail below 1e-16", function(ctx) {
  d <- ctx$gen_dataset("tail_mcnemar", data.frame(a = 25L, b = 400L, c = 10L, d = 30L))
  tab <- matrix(c(d$a, d$c, d$b, d$d), 2)            # tab[1, 2] = b, tab[2, 1] = c
  m1 <- mcnemar.test(tab, correct = TRUE); m0 <- mcnemar.test(tab, correct = FALSE)
  stopifnot(1 - pchisq(unname(m1$statistic), 1) == 0, 1 - pchisq(unname(m0$statistic), 1) == 0)
  list(values = list(stat_correct = unname(m1$statistic), p_correct = m1$p.value,
                     stat_plain = unname(m0$statistic), p_plain = m0$p.value),
       reference = "stats::mcnemar.test", call = "mcnemar.test(matrix(c(25, 10, 400, 30), 2), correct = TRUE | FALSE)",
       packages = character(0))
})

case("tail_oneway", "stats/oneway", "oneway.test upper tail below 1e-16 (equal and Welch variances)", function(ctx) {
  set.seed(20261023)
  g <- rep(c("A", "B", "C"), each = 12)
  y <- round(c(rnorm(12, 0, 0.5), rnorm(12, 3, 0.7), rnorm(12, 6, 0.4)), 6)
  d <- ctx$gen_dataset("tail_oneway", data.frame(g = g, y = y))
  e <- oneway.test(y ~ g, data = d, var.equal = TRUE); w <- oneway.test(y ~ g, data = d, var.equal = FALSE)
  stopifnot(1 - pf(unname(e$statistic), 2, 33) == 0, 1 - pf(unname(w$statistic), 2, unname(w$parameter[2])) == 0)
  list(values = list(equal_f = unname(e$statistic), equal_df = vec(unname(e$parameter)), equal_p = e$p.value,
                     welch_f = unname(w$statistic), welch_df = vec(unname(w$parameter)), welch_p = w$p.value),
       reference = "stats::oneway.test", call = "oneway.test(y ~ g, var.equal = TRUE | FALSE)", packages = character(0))
})

case("tail_prop_trend", "stats/prop_trend", "prop.trend.test upper tail below 1e-16", function(ctx) {
  d <- ctx$gen_dataset("tail_prop_trend", data.frame(x = c(10L, 60L, 140L, 240L), n = rep(300L, 4)))
  t <- prop.trend.test(d$x, d$n)
  stopifnot(1 - pchisq(unname(t$statistic), 1) == 0)
  list(values = list(statistic = unname(t$statistic), p = t$p.value),
       reference = "stats::prop.trend.test", call = "prop.trend.test(c(10, 60, 140, 240), rep(300, 4))", packages = character(0))
})

case("tail_var_test", "stats/var_test", "var.test with an extreme variance ratio; F quantiles at large df", function(ctx) {
  set.seed(20261024)
  x <- round(rnorm(40, 0, 30), 6); y <- round(rnorm(45, 0, 1), 6)
  d <- ctx$gen_dataset("tail_var_test", data.frame(i = 1:45, x = c(x, rep(NA, 5)), y = y))
  x <- d$x[!is.na(d$x)]; y <- d$y
  v2 <- var.test(x, y); vg <- var.test(x, y, alternative = "greater"); vl <- var.test(x, y, alternative = "less")
  f <- unname(v2$statistic); df1 <- length(x) - 1; df2 <- length(y) - 1
  upper <- pf(f, df1, df2, lower.tail = FALSE); lower <- pf(f, df1, df2)
  stopifnot(v2$p.value == 0, vg$p.value == 0, upper > 0, upper < 1e-30)
  big <- 2e7
  qb <- qbeta(c(0.025, 0.975), big / 2, big / 2)
  stopifnot(abs(pbeta(qb[2], big / 2, big / 2, lower.tail = FALSE) / 0.025 - 1) < 1e-12)
  list(values = list(statistic = f, df = vec(c(df1, df2)),
                     var_test_p_two_sided = v2$p.value, var_test_p_greater = vg$p.value, var_test_p_less = vl$p.value,
                     p_upper = upper, p_lower = lower, p_two_sided_accurate = 2 * upper,
                     conf_int_two_sided = vec(v2$conf.int), conf_int_greater = vec(c(vg$conf.int[1])),
                     conf_int_less = vec(c(vl$conf.int[2])),
                     qf_big = vec(qf(c(0.025, 0.975), big, big)),
                     ci_big_estimate_1 = vec(c(1 / qf(0.975, big, big), 1 / qf(0.025, big, big))),
                     qf_big_exact = vec(qb / (1 - qb)),
                     ci_big_exact_estimate_1 = vec(c((1 - qb[2]) / qb[2], (1 - qb[1]) / qb[1]))),
       reference = "stats::var.test; stats::pf(lower.tail = FALSE); stats::qf; stats::qbeta",
       call = "var.test(x, y, alternative = 'two.sided' | 'greater' | 'less'); qf(c(0.025, 0.975), 2e7, 2e7); qbeta(c(0.025, 0.975), 1e7, 1e7)",
       packages = character(0))
})
