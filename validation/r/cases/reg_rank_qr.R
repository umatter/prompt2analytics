# validation/r/cases/reg_rank_qr.R -- OLS rank detection and QR accuracy (T3-WP09: REG-4, REG-18,
# REG-36; the QR primitive itself is T3-WP01's linalg_qr_glm), RESET princomp (REG-35).
#  * lm drops aliased columns with NA (dqrdc2 limited pivoting keeps the earlier column), reports
#    rank and df = n - rank; summary.lm, logLik, sandwich and anova all work on the identified columns;
#  * an ill-conditioned but full-rank design (kappa(X) ~ 1.7e8) where the normal equations lose
#    digits; the reference is lm (Householder QR);
#  * lmtest::resettest(type = 'princomp'): Z = the regressors, v1 = first eigenvector of cov(Z),
#    pc1 = Z %*% v1 (NOT centred), augmentation pc1^power; df2 = n - (ncol(X) + length(power)).
# The T3-WP09 package copy is canonical; it extends Appendix A.4 of plans/T3/00-design.md
# (keys added, two columns added to ols_aliased.csv; every Appendix B value is unchanged).
suppressPackageStartupMessages({ library(lmtest); library(sandwich) })

case("ols_aliased", "regression", "lm with aliased regressors: dummy trap, x2 = 2 x1 and a constant column (NA coefficients, rank, df, sandwich, nested F)", function(ctx) {
  set.seed(20261118)
  n <- 120
  g <- sample(c("a", "b", "c"), n, replace = TRUE)
  x1 <- rnorm(n); x3 <- rnorm(n)
  y <- 1 + 0.5 * x1 + 0.3 * (g == "b") - 0.4 * (g == "c") + 0.2 * x3 + rnorm(n, 0, 0.5)
  d <- ctx$gen_dataset("ols_aliased", data.frame(y = round(y, 10), x1 = round(x1, 10), x2 = round(2 * x1, 10), x3 = round(x3, 10),
                                                 da = as.integer(g == "a"), db = as.integer(g == "b"), dc = as.integer(g == "c"),
                                                 k5 = 5L, cl = rep(1:12, length.out = n)))
  m1 <- lm(y ~ x1 + da + db + dc, data = d)            # dummy trap: dc aliased
  m2 <- lm(y ~ x1 + x2 + x3, data = d)                 # x2 aliased (the later of two collinear columns)
  m3 <- lm(y ~ x1 + k5, data = d)                      # constant regressor next to the intercept: k5 aliased
  m0 <- lm(y ~ x1, data = d)
  stopifnot(identical(as.numeric(is.na(coef(m1))), c(0, 0, 0, 0, 1)), identical(as.numeric(is.na(coef(m2))), c(0, 0, 1, 0)),
            m1$rank == 4L, m2$rank == 3L, is.na(coef(m3)[["k5"]]), max(abs(d$x2 - 2 * d$x1)) > 0)  # x2 is 2 x1 only to 1e-10
  s1 <- summary(m1); s2 <- summary(m2)
  # Nested F with an aliased unrestricted model: q = rank 3 - rank 2 = 1 (x3 only), not 3 - 1 = 2 names.
  # lmtest::waldtest refuses aliased models ("there are aliased coefficients in the model"); anova.lm does not.
  w <- anova(m0, m2)
  # Diagnostics on the aliased model (auxiliary regressions by QR): bptest's df is aux$rank - 1;
  # bgtest's and resettest's df2 count ncol(X), aliased columns included.
  bg <- bgtest(m1, order = 1); bp <- bptest(m1); rs <- resettest(m1, power = 2:3, type = "fitted")
  # The same RESET augmentation as a nested F by anova.lm: df2 = n - rank of the augmented fit (114),
  # the rank-based rule p2a follows on aliased models (T3-WP09 re-scope WP09-10).
  rs_aov <- anova(m1, lm(y ~ x1 + da + db + dc + I(fitted(m1)^2) + I(fitted(m1)^3), data = d))
  v <- list(trap_coef = named(coef(m1)[!is.na(coef(m1))]), trap_aliased = vec(as.numeric(is.na(coef(m1)))),
            trap_se = named(sqrt(diag(vcov(m1)))), trap_rank = m1$rank, trap_df = m1$df.residual,
            trap_r2 = s1$r.squared, trap_sigma = s1$sigma,
            trap_adj_r2 = s1$adj.r.squared, trap_fstat = unname(s1$fstatistic[1]), trap_fstat_df = vec(unname(s1$fstatistic[2:3])),
            trap_loglik = as.numeric(logLik(m1)), trap_aic = AIC(m1), trap_bic = BIC(m1),
            trap_hc1_se = named(sqrt(diag(vcovHC(m1, type = "HC1")))),
            trap_bgtest1_chisq = unname(bg$statistic), trap_bgtest1_p = bg$p.value,
            trap_bptest = unname(bp$statistic), trap_bptest_df = unname(bp$parameter), trap_bptest_p = bp$p.value,
            trap_reset_fitted = unname(rs$statistic), trap_reset_fitted_df = vec(unname(rs$parameter)), trap_reset_fitted_p = rs$p.value,
            trap_reset_fitted_anova_f = rs_aov$F[2], trap_reset_fitted_anova_df = vec(c(rs_aov$Df[2], rs_aov$Res.Df[2])),
            trap_reset_fitted_anova_p = rs_aov$`Pr(>F)`[2],
            dup_coef = named(coef(m2)[!is.na(coef(m2))]), dup_aliased = vec(as.numeric(is.na(coef(m2)))),
            dup_se = named(sqrt(diag(vcov(m2)))), dup_rank = m2$rank, dup_r2 = s2$r.squared,
            dup_df = m2$df.residual, dup_sigma = s2$sigma, dup_fstat = unname(s2$fstatistic[1]),
            dup_hc1_se = named(sqrt(diag(sandwich::vcovHC(m2, type = "HC1")))),
            dup_hc0_se = named(sqrt(diag(vcovHC(m2, type = "HC0")))),
            dup_hc3_se = named(sqrt(diag(vcovHC(m2, type = "HC3")))),
            dup_cl_hc1_se = named(sqrt(diag(vcovCL(m2, cluster = ~cl, type = "HC1")))),
            dup_nw_lag4_se = named(sqrt(diag(NeweyWest(m2, lag = 4, prewhite = FALSE)))),
            const_coef = named(coef(m3)[!is.na(coef(m3))]), const_aliased = vec(as.numeric(is.na(coef(m3)))),
            wald_alias_f = w$F[2], wald_alias_df1 = w$Df[2], wald_alias_df2 = w$Res.Df[2], wald_alias_p = w$`Pr(>F)`[2])
  list(values = v, reference = "stats::lm (dqrdc2 pivoting, tol = 1e-7): aliased coefficients NA, rank, df.residual, summary.lm, logLik; sandwich::vcovHC/vcovCL/NeweyWest on the identified columns; stats::anova for the nested F with an aliased unrestricted model and for the RESET augmentation by rank; lmtest::bgtest/bptest/resettest on the aliased model",
       call = "lm(y ~ x1 + da + db + dc); lm(y ~ x1 + x2 + x3); lm(y ~ x1 + k5); anova(lm(y ~ x1), lm(y ~ x1 + x2 + x3))",
       packages = c("sandwich", "lmtest"))
})
case("ols_illcond", "regression", "lm on a full-rank but ill-conditioned design (kappa(X) ~ 1.7e8; normal equations lose ~5 digits)", function(ctx) {
  set.seed(20261119)
  n <- 80
  for (sc in c(1e2, 1e3)) { x1 <- sc + rnorm(n); x2 <- x1 + rnorm(n, 0, 0.05); x3 <- rnorm(n) }   # burn the probe draws (same stream as the probe)
  x1 <- 1e4 + rnorm(n); x2 <- x1 + rnorm(n, 0, 0.05); x3 <- rnorm(n)
  y <- 1 + 2 * x1 - 1.5 * x2 + 0.5 * x3 + rnorm(n, 0, 0.3)
  d <- ctx$gen_dataset("ols_illcond", data.frame(y = y, x1 = x1, x2 = x2, x3 = x3))
  m <- lm(y ~ x1 + x2 + x3, data = d)
  X <- model.matrix(m)
  L <- chol(crossprod(X)); b_ne <- backsolve(L, forwardsolve(t(L), crossprod(X, d$y)))
  b_lapack <- qr.coef(qr(X, LAPACK = TRUE), d$y)
  s <- summary(m)
  v <- list(coef = named(coef(m)), se = named(sqrt(diag(vcov(m)))), r2 = s$r.squared, sigma = s$sigma,
            adj_r2 = s$adj.r.squared, fstat = unname(s$fstatistic[1]), loglik = as.numeric(logLik(m)),
            kappa_x = kappa(X, exact = TRUE), rank = m$rank,
            normal_equations_max_rel_err = max(abs((b_ne - coef(m)) / coef(m))),
            lapack_qr_max_rel_diff = max(abs((b_lapack - coef(m)) / coef(m))))
  list(values = v, reference = "stats::lm (Householder QR); the Cholesky normal-equation error and the spread to LAPACK dgeqp3 are recorded for the record",
       call = "lm(y ~ x1 + x2 + x3, data = ols_illcond)", packages = character(0))
})
