# validation/r/cases/stats_anova.R -- aov / car::Anova / kruskal / oneway / friedman / quade /
# wilcox / manova references (T2-WP07: STAT-4, STAT-11, STAT-25, STAT-51).
suppressPackageStartupMessages(library(car))

.ss_table <- function(tab, terms) {
  out <- list()
  for (t in terms) {
    key <- gsub(":", "_x_", t)
    out[[paste0(key, "_ss")]] <- tab[t, "Sum Sq"]
    out[[paste0(key, "_df")]] <- tab[t, "Df"]
    out[[paste0(key, "_f")]] <- tab[t, "F value"]
    out[[paste0(key, "_p")]] <- tab[t, "Pr(>F)"]
  }
  out
}

# Unbalanced, non-proportional 2x2 (cells 2/4/4/2) and a 3x2 design with one empty cell.
case("stat_anova_two_way_unbalanced", "stats/anova", "Two-way ANOVA, unbalanced cells: Type I (aov), II and III (car::Anova)", function(ctx) {
  set.seed(101)
  fa <- c(rep("A", 6), rep("B", 6))
  fb <- c("L", "L", "H", "H", "H", "H", "L", "L", "L", "L", "H", "H")
  mu <- c(A_L = 10, A_H = 14, B_L = 11, B_H = 21)
  y <- round(mu[paste(fa, fb, sep = "_")] + rnorm(12, 0, 1.2), 3)
  # 3x2 with the (C, L) cell empty
  fa3 <- c(rep("A", 5), rep("B", 5), rep("C", 3))
  fb3 <- c("L", "L", "H", "H", "H", "L", "H", "H", "L", "L", "H", "H", "H")
  y3 <- round(5 + (fa3 == "B") * 2 + (fa3 == "C") * 4 + (fb3 == "H") * 1.5 + rnorm(13, 0, 0.8), 3)
  d <- ctx$gen_dataset("anova_two_way_unbalanced", data.frame(
    y = c(y, rep(NA, 1)), fa = c(fa, NA), fb = c(fb, "L"),
    y3 = c(y3[1:12], y3[13]), fa3 = fa3, fb3 = fb3))
  # row 13 has a missing y (and fa) for the 2x2 model: aov drops it
  m_int <- aov(y ~ fa * fb, data = d); m_add <- aov(y ~ fa + fb, data = d)
  m3 <- aov(y3 ~ fa3 * fb3, data = d)
  lm_int <- lm(y ~ fa * fb, data = d, contrasts = list(fa = contr.sum, fb = contr.sum))
  v <- c(.ss_table(summary(m_int)[[1]] |> (\(t) { rownames(t) <- trimws(rownames(t)); t })(), c("fa", "fb", "fa:fb", "Residuals")[1:3]),
         list(resid_ss = sum(residuals(m_int)^2), resid_df = df.residual(m_int)))
  names(v) <- paste0("int_", names(v))
  a <- summary(m_add)[[1]]; rownames(a) <- trimws(rownames(a))
  v2 <- .ss_table(a, c("fa", "fb")); names(v2) <- paste0("add_", names(v2))
  v2$add_resid_ss <- sum(residuals(m_add)^2); v2$add_resid_df <- df.residual(m_add)
  t3 <- summary(m3)[[1]]; rownames(t3) <- trimws(rownames(t3))
  v3 <- .ss_table(t3, c("fa3", "fb3", "fa3:fb3")); names(v3) <- paste0("empty_", names(v3))
  v3$empty_resid_df <- df.residual(m3); v3$empty_resid_ss <- sum(residuals(m3)^2)
  a2 <- car::Anova(lm(y ~ fa * fb, data = d), type = 2)
  a3 <- car::Anova(lm_int, type = 3)
  v4 <- list(type2_fa_ss = a2["fa", "Sum Sq"], type2_fb_ss = a2["fb", "Sum Sq"], type2_int_ss = a2["fa:fb", "Sum Sq"],
             type2_fa_f = a2["fa", "F value"], type2_fb_f = a2["fb", "F value"], type2_int_f = a2["fa:fb", "F value"],
             type3_fa_ss = a3["fa", "Sum Sq"], type3_fb_ss = a3["fb", "Sum Sq"], type3_int_ss = a3["fa:fb", "Sum Sq"],
             type3_fa_f = a3["fa", "F value"], type3_fb_f = a3["fb", "F value"], type3_int_f = a3["fa:fb", "F value"],
             n_used = nobs(m_int), n_dropped = nrow(d) - nobs(lm(y ~ fa * fb, data = d)))
  # T2-WP07 additions to the design appendix (the keys above are unchanged): Type II/III p-values
  # and residual df; Type II on the additive model; Type II by model comparison on the 3x2 layout
  # with the empty (C, L) cell (car notes the aliased coefficient) and the Type III refusal there;
  # Type I on the 2x2 layout without its (B, H) cell, where aov has no interaction row.
  a2a <- car::Anova(lm(y ~ fa + fb, data = d), type = 2)
  a3a <- car::Anova(lm(y ~ fa + fb, data = d, contrasts = list(fa = contr.sum, fb = contr.sum)), type = 3)
  a2e <- suppressMessages(car::Anova(lm(y3 ~ fa3 * fb3, data = d), type = 2))
  e3 <- tryCatch({ car::Anova(lm(y3 ~ fa3 * fb3, data = d, contrasts = list(fa3 = contr.sum, fb3 = contr.sum)), type = 3); "no error" },
                 error = conditionMessage)
  stopifnot(e3 == "there are aliased coefficients in the model")
  s <- d[!(d$fa %in% "B" & d$fb == "H"), ]                       # keeps row 13 (fa NA); aov drops it
  ts <- summary(aov(y ~ fa * fb, data = s))[[1]]; rownames(ts) <- trimws(rownames(ts))
  stopifnot(!("fa:fb" %in% rownames(ts)), nrow(s) == 11)
  v5 <- list(type2_fa_p = a2["fa", "Pr(>F)"], type2_fb_p = a2["fb", "Pr(>F)"], type2_int_p = a2["fa:fb", "Pr(>F)"],
             type3_fa_p = a3["fa", "Pr(>F)"], type3_fb_p = a3["fb", "Pr(>F)"], type3_int_p = a3["fa:fb", "Pr(>F)"],
             type2_resid_ss = a2["Residuals", "Sum Sq"], type2_resid_df = a2["Residuals", "Df"],
             add_type2_fa_ss = a2a["fa", "Sum Sq"], add_type2_fb_ss = a2a["fb", "Sum Sq"],
             add_type2_fa_f = a2a["fa", "F value"], add_type2_fb_f = a2a["fb", "F value"],
             add_type2_fa_p = a2a["fa", "Pr(>F)"], add_type2_fb_p = a2a["fb", "Pr(>F)"],
             add_type3_fa_ss = a3a["fa", "Sum Sq"], add_type3_fb_ss = a3a["fb", "Sum Sq"],
             add_type3_fa_f = a3a["fa", "F value"], add_type3_fb_f = a3a["fb", "F value"],
             empty_type2_fa3_ss = a2e["fa3", "Sum Sq"], empty_type2_fa3_df = a2e["fa3", "Df"], empty_type2_fa3_f = a2e["fa3", "F value"],
             empty_type2_fa3_p = a2e["fa3", "Pr(>F)"], empty_type2_fb3_ss = a2e["fb3", "Sum Sq"], empty_type2_int_ss = a2e["fa3:fb3", "Sum Sq"],
             empty_type2_int_df = a2e["fa3:fb3", "Df"],
             nocell_fa_ss = ts["fa", "Sum Sq"], nocell_fa_f = ts["fa", "F value"], nocell_fa_p = ts["fa", "Pr(>F)"],
             nocell_fb_ss = ts["fb", "Sum Sq"], nocell_fb_f = ts["fb", "F value"], nocell_fb_p = ts["fb", "Pr(>F)"],
             nocell_resid_ss = ts["Residuals", "Sum Sq"], nocell_resid_df = ts["Residuals", "Df"],
             nocell_n_used = sum(complete.cases(s$y, s$fa, s$fb)))
  list(values = c(v, v2, v3, v4, v5),
       reference = "stats::aov (sequential Type I SS); car::Anova(type = 2); car::Anova(type = 3, contr.sum)",
       call = "aov(y ~ fa * fb); aov(y ~ fa + fb); aov(y3 ~ fa3 * fb3); car::Anova(lm(...), type = 2 | 3)", packages = "car")
})

# One-way ANOVA, Kruskal-Wallis and Welch's oneway.test with missing responses and labels
# (STAT-11, STAT-25); string labels must be printed without quotes.
case("stat_oneway_na_labels", "stats/anova", "One-way ANOVA, Kruskal-Wallis and oneway.test with NA responses and NA group labels", function(ctx) {
  y <- c(1, 2, NA, 4, 5, 6, 7, NA)
  g <- c("a", "a", "a", "b", "b", NA, "c", "c")
  v9 <- c(1:8, 9.5); g9 <- c("a", "a", "a", "b", "b", "b", NA, NA, NA)
  d <- ctx$gen_dataset("oneway_na_labels", data.frame(
    id = 1:9, y = c(y, NA), g = c(g, NA), v9 = v9, g9 = g9))
  a <- summary(aov(y ~ g, data = d))[[1]]
  kw <- kruskal.test(v9 ~ g9, data = d)
  ow <- oneway.test(v9 ~ g9, data = d)
  owe <- oneway.test(v9 ~ g9, data = d, var.equal = TRUE)
  cc <- complete.cases(d$y, d$g); cc9 <- complete.cases(d$v9, d$g9)
  stopifnot(is.integer(d$y), is.double(d$v9))                # y is read as Int64 by polars
  list(values = list(
    aov_f = a[1, "F value"], aov_p = a[1, "Pr(>F)"], aov_df_between = a[1, "Df"], aov_df_within = a[2, "Df"],
    aov_ss_between = a[1, "Sum Sq"], aov_ss_within = a[2, "Sum Sq"], aov_n_used = 5,
    kw_chisq = unname(kw$statistic), kw_df = unname(kw$parameter), kw_p = kw$p.value,
    welch_f = unname(ow$statistic), welch_df1 = unname(ow$parameter[1]), welch_df2 = unname(ow$parameter[2]), welch_p = ow$p.value,
    classic_f = unname(owe$statistic), classic_p = owe$p.value,
    # T2-WP07 additions: group labels, sizes and means of the rows each test keeps
    classic_df1 = unname(owe$parameter[1]), classic_df2 = unname(owe$parameter[2]),
    aov_group_n = named(c(table(d$g[cc]))), aov_group_means = named(tapply(d$y[cc], d$g[cc], mean)),
    aov_grand_mean = mean(d$y[cc]), kw_group_n = named(c(table(d$g9[cc9])))),
    reference = "stats::aov, kruskal.test, oneway.test (rows with NA response or label dropped)",
    call = "aov(y ~ g); kruskal.test(v9 ~ g9); oneway.test(v9 ~ g9)", packages = character(0))
})

# Wilcoxon signed-rank (paired) and rank-sum with missing cells (STAT-11).
case("stat_wilcoxon_na", "stats/rank", "Wilcoxon paired and rank-sum tests with NA cells", function(ctx) {
  set.seed(17)
  x <- round(rnorm(10, 5, 1), 2); yy <- round(x - 0.6 + rnorm(10, 0, 0.4), 2)
  x[3] <- NA; yy[8] <- NA
  d <- ctx$gen_dataset("wilcoxon_na", data.frame(id = 1:10, x = x, y = yy))
  p <- wilcox.test(d$x, d$y, paired = TRUE, exact = FALSE, correct = TRUE)
  pe <- wilcox.test(d$x, d$y, paired = TRUE)
  r <- wilcox.test(d$x, d$y, exact = FALSE, correct = TRUE)
  # T2-WP07 additions: the rank sum of x (p2a's `statistic` for the rank-sum test; R reports
  # W = R1 - n_x (n_x + 1) / 2, STAT-72) and the normal z of both tests, recovered from R's p
  xx <- d$x[!is.na(d$x)]; yy2 <- d$y[!is.na(d$y)]; np <- sum(complete.cases(d$x, d$y))
  stopifnot(!anyDuplicated(c(xx, yy2)), !anyDuplicated(abs((d$x - d$y)[complete.cases(d$x, d$y)])))   # no ties (STAT-42 not involved)
  list(values = list(paired_v = unname(p$statistic), paired_p_normal = p$p.value, paired_p_exact = pe$p.value,
                     ranksum_w = unname(r$statistic), ranksum_p_normal = r$p.value,
                     n_pairs = sum(complete.cases(d$x, d$y)), n_x = sum(!is.na(d$x)), n_y = sum(!is.na(d$y)),
                     ranksum_r1 = sum(rank(c(xx, yy2))[seq_along(xx)]),
                     paired_z = -qnorm(p$p.value / 2) * sign(unname(p$statistic) - np * (np + 1) / 4),
                     ranksum_z = -qnorm(r$p.value / 2) * sign(unname(r$statistic) - length(xx) * length(yy2) / 2)),
       reference = "stats::wilcox.test(paired = TRUE) on complete pairs; rank-sum drops NA per sample",
       call = "wilcox.test(x, y, paired = TRUE); wilcox.test(x, y)", packages = character(0))
})

# Friedman and Quade on a complete block design with integer responses, plus the cases R
# rejects (duplicate cell, NA block label) and drops (NA response -> block dropped) (STAT-51).
case("stat_friedman_quade", "stats/rank", "Friedman and Quade tests: complete design, NA response, and the designs R refuses", function(ctx) {
  set.seed(23)
  b <- rep(paste0("S", 1:8), each = 3); tr <- rep(c("A", "B", "C"), 8)
  y <- round(10 + (tr == "B") * 1.5 + (tr == "C") * 3 + rep(rnorm(8, 0, 2), each = 3) + rnorm(24, 0, 1.5))
  y_na <- y; y_na[5] <- NA                        # block S2 becomes incomplete -> dropped by R
  d <- ctx$gen_dataset("friedman_quade", data.frame(y = y, y_na = y_na, treat = tr, block = b))
  stopifnot(is.integer(d$y))
  f <- friedman.test(d$y, d$treat, d$block); q <- quade.test(d$y, d$treat, d$block)
  fn <- friedman.test(d$y_na, d$treat, d$block); qn <- quade.test(d$y_na, d$treat, d$block)
  dup_err <- tryCatch({ friedman.test(c(d$y, 99), c(d$treat, "A"), c(d$block, "S1")); "no error" }, error = conditionMessage)
  na_err <- tryCatch({ friedman.test(d$y, d$treat, replace(d$block, 4, NA)); "no error" }, error = conditionMessage)
  stopifnot(dup_err == "not an unreplicated complete block design", grepl("NA's are not allowed", na_err))
  # T2-WP07 additions: a block without one treatment (row absent) is refused too, by both tests;
  # quade.test refuses the duplicate and the NA label like friedman.test. quade.test with an NA
  # response removes the block (its help page) but keeps b = nlevels(blocks) = 8 in B, (b - 1)
  # and the denominator df (R 4.3.3 stats:::quade.test.default), so na_quade_* above are R's
  # NA-path values; the statistic of the 7 complete blocks is quade.test on those blocks.
  inc_err <- tryCatch({ friedman.test(d$y[-5], d$treat[-5], d$block[-5]); "no error" }, error = conditionMessage)
  inc_err_q <- tryCatch({ quade.test(d$y[-5], d$treat[-5], d$block[-5]); "no error" }, error = conditionMessage)
  dup_err_q <- tryCatch({ quade.test(c(d$y, 99), c(d$treat, "A"), c(d$block, "S1")); "no error" }, error = conditionMessage)
  na_err_q <- tryCatch({ quade.test(d$y, d$treat, replace(d$block, 4, NA)); "no error" }, error = conditionMessage)
  stopifnot(inc_err == "not an unreplicated complete block design", inc_err_q == inc_err, dup_err_q == dup_err,
            grepl("NA's are not allowed", na_err_q))
  keep <- d$block != "S2"
  qc <- quade.test(d$y_na[keep], d$treat[keep], d$block[keep]); fc <- friedman.test(d$y_na[keep], d$treat[keep], d$block[keep])
  stopifnot(isTRUE(all.equal(unname(fc$statistic), unname(fn$statistic), tolerance = 1e-14)), unname(qn$parameter[2]) == 14)
  list(values = list(friedman_q = unname(f$statistic), friedman_df = unname(f$parameter), friedman_p = f$p.value,
                     quade_f = unname(q$statistic), quade_df1 = unname(q$parameter[1]), quade_df2 = unname(q$parameter[2]), quade_p = q$p.value,
                     na_friedman_q = unname(fn$statistic), na_friedman_p = fn$p.value,
                     na_quade_f = unname(qn$statistic), na_quade_p = qn$p.value, na_n_blocks = 7,
                     na_quade_df2 = unname(qn$parameter[2]),
                     na_quade_complete_f = unname(qc$statistic), na_quade_complete_df1 = unname(qc$parameter[1]),
                     na_quade_complete_df2 = unname(qc$parameter[2]), na_quade_complete_p = qc$p.value),
       reference = "stats::friedman.test, stats::quade.test (incomplete blocks from NA responses dropped; replicated cells and NA labels are errors)",
       call = "friedman.test(y, treat, block); quade.test(y, treat, block)", packages = character(0))
})
