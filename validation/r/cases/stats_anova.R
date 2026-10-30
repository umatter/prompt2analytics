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
