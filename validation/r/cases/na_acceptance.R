# validation/r/cases/na_acceptance.R -- the X-2 / X-8 reviewer repro (T2-WP22): empty cells and
# integer columns through the user paths (MCP create_dataset + tools, the p2a CLI). The data are the
# reviewer's (X-2 failure scenario: x, y, g, i) plus integer survival columns (X-8: time, event).
suppressPackageStartupMessages(library(survival))

case("na_x2_x8_reviewer", "data/na_acceptance",
     "X-2/X-8 reviewer data: t-tests, cor.test, one-way ANOVA, Wilcoxon, Kaplan-Meier with empty cells and integer columns",
     function(ctx) {
  d <- ctx$gen_dataset("na_reviewer", data.frame(
    id = 1:8,
    x = c(5.1, 6.2, NA, 7.3, 5.9, 6.8, 7.7, 6.1),
    y = c(4.0, NA, 5.5, 6.1, 4.8, 5.2, 6.6, 5.0),
    g = c(1, 2, 1, 2, 1, 2, 1, 2),
    i = c(3, 4, 5, 6, 7, 8, 9, 10),
    time = c(5, 8, 3, 9, 12, 7, 15, 4),
    event = c(1, 0, 1, 1, 0, 1, 1, 0)))
  # polars reads x, y as Float64 and g, i, time, event as Int64 (written without a decimal point)
  stopifnot(is.double(d$x), is.double(d$y), is.integer(d$g), is.integer(d$i),
            is.integer(d$time), is.integer(d$event), sum(is.na(d$x)) == 1, sum(is.na(d$y)) == 1)
  one <- t.test(d$x)
  pr <- t.test(d$x, d$y, paired = TRUE)
  we <- t.test(d$x, d$y)
  ct <- cor.test(d$x, d$y)
  av <- summary(aov(y ~ factor(g), data = d))[[1]]          # "1 observation deleted due to missingness"
  ti <- t.test(d$i)
  ai <- summary(aov(i ~ factor(g), data = d))[[1]]
  wi <- wilcox.test(d$i)                                     # n = 8, no ties: exact
  ci <- cor.test(d$i, d$y)
  km <- summary(survfit(Surv(time, event) ~ 1, data = d))   # conf.type "log" (R default)
  list(values = list(
    # X-2: missing cells dropped as R does
    one_t = unname(one$statistic), one_df = unname(one$parameter), one_p = one$p.value,
    one_mean = unname(one$estimate), one_ci = vec(one$conf.int), one_n = sum(!is.na(d$x)),
    paired_t = unname(pr$statistic), paired_df = unname(pr$parameter), paired_p = pr$p.value,
    paired_mean_diff = unname(pr$estimate), paired_n = sum(complete.cases(d$x, d$y)),
    welch_t = unname(we$statistic), welch_df = unname(we$parameter), welch_p = we$p.value,
    cor_r = unname(ct$estimate), cor_t = unname(ct$statistic), cor_df = unname(ct$parameter), cor_p = ct$p.value,
    anova_y_f = av[["F value"]][1], anova_y_df = vec(av[["Df"]]), anova_y_p = av[["Pr(>F)"]][1],
    anova_y_ss = vec(av[["Sum Sq"]]), anova_y_n = sum(complete.cases(d$y, d$g)),
    # X-8: integer columns are numeric
    int_t = unname(ti$statistic), int_df = unname(ti$parameter), int_p = ti$p.value, int_mean = unname(ti$estimate),
    anova_i_f = ai[["F value"]][1], anova_i_df = vec(ai[["Df"]]), anova_i_p = ai[["Pr(>F)"]][1],
    wilcox_i_v = unname(wi$statistic), wilcox_i_p = wi$p.value,
    cor_iy_r = unname(ci$estimate), cor_iy_df = unname(ci$parameter), cor_iy_p = ci$p.value,
    km_time = vec(km$time), km_surv = vec(km$surv), km_std_err = vec(km$std.err),
    km_lower = vec(km$lower), km_upper = vec(km$upper), km_n_event = vec(km$n.event), km_n_risk = vec(km$n.risk),
    km_median = unname(summary(survfit(Surv(time, event) ~ 1, data = d))$table["median"])),
    reference = "stats::t.test, cor.test, aov (na.omit), wilcox.test; survival::survfit (conf.type = 'log')",
    call = "t.test(x); t.test(x, y, paired = TRUE); t.test(x, y); cor.test(x, y); aov(y ~ factor(g)); t.test(i); aov(i ~ factor(g)); wilcox.test(i); cor.test(i, y); survfit(Surv(time, event) ~ 1)",
    packages = c("survival"))
})
