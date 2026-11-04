# validation/r/cases/stats_power_loglin.R -- power analysis, loglin IPF, Mauchly and the statistics
# test pins of T3-WP38 (STAT-61, STAT-62, STAT-28, STAT-56; moved in STAT-33, STAT-34, STAT-82, and the
# planner move-ins STAT-30, STAT-31, STAT-23). Package copy (T3-WP38 plan): supersedes Appendix A.35 of
# the Tier 3 design. Adds strict, default-tolerance, extendInt and finding-scenario keys to
# stat_power_solve, iteration counts and a 2x2x2 table to stat_loglin_nodecomp, the X = ~f, Sigma-with-
# contrasts and two-column designs plus chi-square, first-order p and epsilons to stat_mauchly_sigma,
# and seven planner cases (Fisher, ANOVA, p.adjust, pairwise, factanal, constrOptim legacy pins).
# stat_fisher_legacy calls T2-WP05's .fisher_values (validation/r/cases/stats_contingency.R).

case("stat_power_solve", "stats/power", "power.t.test / power.prop.test / power.anova.test solved for each parameter (tol = 1e-12)", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")    # no data needed; the case records closed-form/uniroot results
  tol <- 1e-12
  v <- list()
  for (ty in c("two.sample", "one.sample", "paired")) for (alt in c("two.sided", "one.sided")) {
    tag <- paste0(gsub("\\.", "", ty), "_", gsub("\\.", "", alt))
    v[[paste0("t_", tag, "_power")]] <- power.t.test(n = 20, delta = 0.8, sd = 1.2, sig.level = 0.05, type = ty, alternative = alt)$power
    v[[paste0("t_", tag, "_n")]] <- power.t.test(power = 0.9, delta = 0.8, sd = 1.2, sig.level = 0.05, type = ty, alternative = alt, tol = tol)$n
    v[[paste0("t_", tag, "_delta")]] <- power.t.test(n = 20, power = 0.9, sd = 1.2, sig.level = 0.05, type = ty, alternative = alt, tol = tol)$delta
    v[[paste0("t_", tag, "_sd")]] <- power.t.test(n = 20, power = 0.9, delta = 0.8, sd = NULL, sig.level = 0.05, type = ty, alternative = alt, tol = tol)$sd
    v[[paste0("t_", tag, "_sig")]] <- suppressWarnings(  # R's own pnt warns at sig.level -> 1 - 1e-10 (one-sided)
      power.t.test(n = 20, power = 0.9, delta = 0.8, sd = 1.2, sig.level = NULL, type = ty, alternative = alt, tol = tol)$sig.level)
  }
  v$t_default_sd_sig_power <- power.t.test(n = 15, delta = 1)$power            # sd = 1 and sig.level = 0.05 by default
  # STAT-61 / STAT-33 failure scenarios
  v$t_n30_d05_power <- power.t.test(n = 30, delta = 0.5)$power
  v$t_onesample_n10_d1_power <- power.t.test(n = 10, delta = 1, type = "one.sample")$power
  v$t_d05_p08_n <- power.t.test(delta = 0.5, power = 0.8, tol = tol)$n
  # strict = TRUE (both tails) and R's default tolerance (package values, cross-checks)
  v$t_twosample_twosided_power_strict <- power.t.test(n = 20, delta = 0.8, sd = 1.2, strict = TRUE)$power
  v$t_twosample_twosided_n_strict <- power.t.test(power = 0.9, delta = 0.8, sd = 1.2, strict = TRUE, tol = tol)$n
  v$t_d05_p08_n_pkg <- power.t.test(delta = 0.5, power = 0.8)$n
  v$t_twosample_twosided_sd_pkg <- power.t.test(n = 20, power = 0.9, delta = 0.8, sd = NULL)$sd
  # uniroot extendInt = "upX" moves the lower end below n = 2 (power at n = 2 already exceeds 0.8)
  ext <- power.t.test(delta = 10, sd = 1, power = 0.8, tol = tol)
  stopifnot(ext$n < 2)
  v$t_extend_n <- ext$n
  pb <- function(n) pt(qt(0.05 / 2, pmax(1e-07, n - 1) * 2, lower.tail = FALSE), pmax(1e-07, n - 1) * 2, ncp = sqrt(n / 2) * 10, lower.tail = FALSE)
  v$t_extend_init_it <- uniroot(function(n) pb(n) - 0.8, c(2, 1e+07), tol = tol, extendInt = "upX")$init.it   # R's own extension count
  # two-sided: delta is replaced by |delta|
  neg <- power.t.test(n = 20, delta = -0.8, sd = 1.2)
  v$t_negdelta_power <- neg$power; v$t_negdelta_delta <- neg$delta
  for (alt in c("two.sided", "one.sided")) {
    tag <- gsub("\\.", "", alt)
    v[[paste0("prop_", tag, "_power")]] <- power.prop.test(n = 50, p1 = 0.5, p2 = 0.75, alternative = alt)$power
    v[[paste0("prop_", tag, "_n")]] <- power.prop.test(p1 = 0.5, p2 = 0.75, power = 0.9, alternative = alt, tol = tol)$n
    v[[paste0("prop_", tag, "_p1")]] <- power.prop.test(n = 50, p2 = 0.75, power = 0.9, alternative = alt, tol = tol)$p1
    v[[paste0("prop_", tag, "_p2")]] <- power.prop.test(n = 50, p1 = 0.5, power = 0.9, alternative = alt, tol = tol)$p2
    v[[paste0("prop_", tag, "_sig")]] <- power.prop.test(n = 50, p1 = 0.5, p2 = 0.75, power = 0.9, sig.level = NULL, alternative = alt, tol = tol)$sig.level
  }
  v$prop_twosided_power_strict <- power.prop.test(n = 50, p1 = 0.5, p2 = 0.75, strict = TRUE)$power
  v$prop_n100_power <- power.prop.test(n = 100, p1 = 0.3, p2 = 0.5)$power                    # STAT-33 scenario
  v$prop_p03_p05_n <- power.prop.test(p1 = 0.3, p2 = 0.5, power = 0.8, tol = tol)$n
  v$prop_p03_p05_n_pkg <- power.prop.test(p1 = 0.3, p2 = 0.5, power = 0.8)$n
  # power.anova.test has no tol argument in R 4.3 (uniroot default tol), so the solved values are
  # re-solved here with R's own power function and uniroot(tol = 1e-12) over R's intervals; the
  # package values are recorded as *_pkg cross-checks.
  pa <- function(groups, n, b, w, sig) { df1 <- groups - 1; df2 <- (n - 1) * groups; lam <- (groups - 1) * n * (b / w)
    pf(qf(sig, df1, df2, lower.tail = FALSE), df1, df2, lam, lower.tail = FALSE) }
  v$anova_power <- power.anova.test(groups = 4, n = 5, between.var = 1, within.var = 3)$power
  v$anova_n <- uniroot(function(n) pa(4, n, 1, 3, 0.05) - 0.8, c(2, 1e5), tol = tol)$root
  v$anova_groups <- uniroot(function(g) pa(g, 5, 1, 3, 0.05) - 0.8, c(2, 100), tol = tol)$root
  v$anova_between <- uniroot(function(b) pa(4, 5, b, 3, 0.05) - 0.8, 3 * c(1e-07, 1e+07), tol = tol)$root
  v$anova_within <- uniroot(function(w) pa(4, 5, 1, w, 0.05) - 0.8, 1 * c(1e-07, 1e+07), tol = tol)$root
  v$anova_sig <- uniroot(function(s) pa(4, 5, 1, 3, s) - 0.8, c(1e-10, 1 - 1e-10), tol = tol)$root
  v$anova_n_pkg <- power.anova.test(groups = 4, between.var = 1, within.var = 3, power = 0.8)$n
  v$anova_groups_pkg <- power.anova.test(n = 5, between.var = 1, within.var = 3, power = 0.8)$groups
  v$anova_between_pkg <- power.anova.test(groups = 4, n = 5, within.var = 3, power = 0.8)$between.var
  v$anova_within_pkg <- power.anova.test(groups = 4, n = 5, between.var = 1, power = 0.8)$within.var
  v$anova_sig_pkg <- power.anova.test(groups = 4, n = 5, between.var = 1, within.var = 3, power = 0.8, sig.level = NULL)$sig.level
  # STAT-34 failure scenario: groups 3, between 0.25, within 1
  v$anova_g3_n20_power <- power.anova.test(groups = 3, n = 20, between.var = 0.25, within.var = 1)$power
  v$anova_g3_n <- uniroot(function(n) pa(3, n, 0.25, 1, 0.05) - 0.8, c(2, 1e5), tol = tol)$root
  v$anova_g3_n_pkg <- power.anova.test(groups = 3, between.var = 0.25, within.var = 1, power = 0.8)$n
  # the noncentral t and F distributions (R's pt / pf with ncp), one point per branch of nmath pnt.c / pnf.c
  gt <- rbind(c(2.1, 38, 1.79, 0), c(2.1, 38, 1.79, 1), c(-2.1, 38, 1.79, 1), c(-2.1, 38, 1.79, 0), c(1.5, 5, -0.8, 0),
              c(0, 10, 1, 1), c(3, 1e6, 2.5, 0), c(40, 10, 38.5, 0), c(2, 30, 45, 1), c(-3, 30, 45, 1), c(2.5, 0.76, 1.2, 0),
              c(1.7, 19, 0, 0), c(5, 100, 3, 0), c(0.5, 3, 10, 0), c(12, 20, 10, 1), c(-1, 4, -2, 0))
  v$pt_t <- vec(gt[, 1]); v$pt_df <- vec(gt[, 2]); v$pt_ncp <- vec(gt[, 3]); v$pt_lower <- vec(gt[, 4])
  v$pt_value <- vec(suppressWarnings(mapply(function(t, df, ncp, lo) pt(t, df, ncp, lower.tail = lo == 1), gt[, 1], gt[, 2], gt[, 3], gt[, 4])))
  gf <- rbind(c(3.2, 3, 16, 5, 0), c(3.2, 3, 16, 5, 1), c(1.5, 1, 30, 0.5, 0), c(0.01, 2, 50, 3, 1), c(4, 99, 4e5, 60, 0),
              c(2.5, 3, 2e8, 4, 0), c(10, 5, 20, 200, 0), c(0, 3, 16, 5, 0), c(2, 3, 16, 0, 0), c(0.7, 12, 7, 1.5, 1))
  v$pf_x <- vec(gf[, 1]); v$pf_df1 <- vec(gf[, 2]); v$pf_df2 <- vec(gf[, 3]); v$pf_ncp <- vec(gf[, 4]); v$pf_lower <- vec(gf[, 5])
  v$pf_value <- vec(suppressWarnings(mapply(function(x, a, b, ncp, lo) pf(x, a, b, ncp, lower.tail = lo == 1), gf[, 1], gf[, 2], gf[, 3], gf[, 4], gf[, 5])))
  list(dataset = "validation/datasets/grunfeld.csv", values = v,
       reference = "stats::power.t.test / power.prop.test / power.anova.test (uniroot tol = 1e-12 when solving)",
       call = "power.t.test(n, delta, sd, sig.level, power, type, alternative, strict, tol)", packages = character(0))
})

case("stat_loglin_nodecomp", "stats/loglin", "loglin no-three-way-interaction model (non-decomposable) on a 2x3x4 table, IPF to eps = 1e-12", function(ctx) {
  set.seed(20261116)
  tab <- array(rpois(24, 30), dim = c(2, 3, 4))
  d <- ctx$gen_dataset("loglin_2x3x4", as.data.frame(as.table(tab)))
  tb <- xtabs(Freq ~ Var1 + Var2 + Var3, data = d)
  m3 <- list(c(1, 2), c(1, 3), c(2, 3))
  nit <- function(out) as.numeric(sub(" iterations.*$", "", out))      # R's own iteration count (print = TRUE line)
  o12 <- capture.output(f <- loglin(tb, m3, eps = 1e-12, iter = 1000, fit = TRUE))
  o01 <- capture.output(f01 <- loglin(tb, m3, fit = TRUE))                 # R defaults eps = 0.1, iter = 20
  f1 <- suppressWarnings(loglin(tb, m3, iter = 1, fit = TRUE, print = FALSE))
  f3 <- withCallingHandlers(loglin(tb, m3, eps = 1e-12, iter = 3, print = FALSE),
                            warning = function(w) { stopifnot(conditionMessage(w) == "algorithm did not converge"); invokeRestart("muffleWarning") })
  fd <- loglin(tb, list(c(1, 2), c(1, 3)), eps = 1e-12, iter = 1000, fit = TRUE, print = FALSE)
  # Rust passes the table row-major (last index fastest)
  rowmajor <- function(a) as.numeric(aperm(a, rev(seq_along(dim(a)))))
  # the finding's 2x2x2 table (STAT-28 repro), row-major counts
  obs <- c(10, 20, 30, 40, 15, 25, 35, 5)
  B <- aperm(array(obs, c(2, 2, 2)), 3:1)
  stopifnot(all.equal(rowmajor(B), obs))
  oB <- capture.output(fB <- loglin(B, m3, fit = TRUE))
  oB10 <- capture.output(fB10 <- loglin(B, m3, eps = 1e-10, iter = 1000, fit = TRUE))
  fB1 <- suppressWarnings(loglin(B, m3, iter = 1, print = FALSE))
  v <- list(lrt = f$lrt, pearson = f$pearson, df = f$df, fit = vec(as.numeric(f$fit)),
            fit_rowmajor = vec(rowmajor(f$fit)), counts_rowmajor = vec(rowmajor(as.array(tb))),
            iterations = nit(o12), p_lrt = pchisq(f$lrt, f$df, lower.tail = FALSE),
            p_pearson = pchisq(f$pearson, f$df, lower.tail = FALSE),
            default_lrt = f01$lrt, default_pearson = f01$pearson, default_iterations = nit(o01),
            iter3_lrt = f3$lrt, iter3_pearson = f3$pearson,
            one_cycle_lrt = f1$lrt, one_cycle_pearson = f1$pearson,
            decomposable_lrt = fd$lrt, decomposable_df = fd$df,
            t222_counts = vec(obs), t222_lrt = fB$lrt, t222_pearson = fB$pearson, t222_iterations = nit(oB),
            t222_eps10_lrt = fB10$lrt, t222_eps10_pearson = fB10$pearson, t222_eps10_iterations = nit(oB10),
            t222_p_lrt = pchisq(fB$lrt, fB$df, lower.tail = FALSE), t222_df = fB$df,
            t222_one_cycle_lrt = fB1$lrt, t222_one_cycle_pearson = fB1$pearson,
            t222_eps10_p_lrt = pchisq(fB10$lrt, fB10$df, lower.tail = FALSE),
            t222_eps10_p_pearson = pchisq(fB10$pearson, fB10$df, lower.tail = FALSE),
            t222_eps10_fit_rowmajor = vec(rowmajor(fB10$fit)))
  list(values = v, reference = "stats::loglin(margins = list(c(1,2), c(1,3), c(2,3)), eps = 1e-12, iter = 1000)",
       call = "loglin(xtabs(Freq ~ Var1 + Var2 + Var3), list(c(1,2), c(1,3), c(2,3)), eps = 1e-12)", packages = character(0))
})

case("stat_mauchly_sigma", "stats/mauchly", "mauchly.test against a non-identity structure: T = I, X = ~1, X = ~f and Sigma = Psi; Pr2 (w2) correction", function(ctx) {
  set.seed(20261117)
  n <- 15; p <- 4
  S <- matrix(0.5, p, p); diag(S) <- 1 + (1:p) / 4
  Y <- MASS::mvrnorm(n, mu = 1:p, Sigma = S)
  d <- ctx$gen_dataset("mauchly_sigma", as.data.frame(round(Y, 10)))
  fit <- lm(as.matrix(d) ~ 1)
  s <- SSD(fit); nd <- s$df
  idata <- data.frame(f = factor(c("a", "a", "b", "b")))
  Psi <- diag(c(1, 2, 3, 4))
  designs <- list(identity = list(), contrasts = list(X = ~1), within_f = list(X = ~f, idata = idata),
                  psi_identity = list(T = diag(p), Sigma = Psi), psi_contrasts = list(X = ~1, Sigma = Psi))
  v <- list()
  for (nm in names(designs)) {
    a <- designs[[nm]]
    m <- do.call(mauchly.test, c(list(fit), a))
    sp <- do.call(stats:::sphericity, c(list(s), a))
    # R's own intermediate quantities (stats:::mauchly.test.SSD), recomputed for the first-order p
    Tm <- if (!is.null(a$T)) a$T else {
      X <- if (is.null(a$X)) matrix(0, p, 0) else model.matrix(a$X, if (is.null(a$idata)) data.frame(index = 1:p) else a$idata)
      stats:::Thin.row(stats:::Proj(diag(p)) - stats:::Proj(X)) }
    Sg <- if (is.null(a$Sigma)) diag(p) else a$Sigma
    U <- solve(Tm %*% Sg %*% t(Tm), Tm %*% s$SSD %*% t(Tm)); pp <- nrow(Tm)
    logW <- log(det(U)) - pp * log(sum(diag(U)) / pp)
    rho <- 1 - (2 * pp^2 + pp + 2) / (6 * pp * nd); z <- -nd * rho * logW; f <- pp * (pp + 1) / 2 - 1
    stopifnot(isTRUE(all.equal(exp(logW), unname(m$statistic), tolerance = 1e-12)))
    v[[paste0(nm, "_W")]] <- unname(m$statistic); v[[paste0(nm, "_p")]] <- m$p.value
    v[[paste0(nm, "_chisq")]] <- z; v[[paste0(nm, "_df")]] <- f; v[[paste0(nm, "_pp")]] <- pp
    v[[paste0(nm, "_p_first_order")]] <- pchisq(z, f, lower.tail = FALSE)
    v[[paste0(nm, "_gg")]] <- sp$GG.eps; v[[paste0(nm, "_hf")]] <- sp$HF.eps
    v[[paste0(nm, "_T")]] <- vec(as.numeric(t(Tm)))                         # row-major contrast matrix R used
    # R's p-value read back from its own w2 (3 * p), and Anderson's w2 with pp throughout (WP38-9)
    Pr1 <- pchisq(z, f, lower.tail = FALSE); Pr2 <- pchisq(z, f + 4, lower.tail = FALSE)
    w2R <- (pp + 2) * (pp - 1) * (pp - 2) * (2 * pp^3 + 6 * pp^2 + 3 * p + 2) / (288 * (nd * pp * rho)^2)   # R's code
    stopifnot(abs(m$p.value - (Pr1 + w2R * (Pr2 - Pr1))) < 1e-12)                                             # R's p read correctly
    w2A <- (pp + 2) * (pp - 1) * (pp - 2) * (2 * pp^3 + 6 * pp^2 + 3 * pp + 2) / (288 * (nd * pp * rho)^2)  # Anderson: pp throughout
    v[[paste0(nm, "_p_anderson")]] <- if (f == 0) 1 else Pr1 + w2A * (Pr2 - Pr1)
  }
  v$n <- nrow(d); v$psi_diag <- vec(diag(Psi))
  # two repeated measures, X = ~1: one contrast, W = 1
  m2 <- mauchly.test(lm(as.matrix(d[, 1:2]) ~ 1), X = ~1)
  v$two_col_W <- unname(m2$statistic); v$two_col_p <- m2$p.value
  list(values = v, reference = "stats::mauchly.test.mlm (T, X, idata, Sigma); stats:::sphericity; p-value with Anderson's second-order term",
       call = "mauchly.test(lm(Y ~ 1), ...)", packages = "MASS")
})

case("stat_fisher_legacy", "stats/contingency", "fisher.test on the four 2x2 tables of fisher.rs's legacy tests (STAT-62)", function(ctx) {
  stopifnot(exists(".fisher_values"))                                       # T2-WP05, stats_contingency.R
  tabs <- list(basic = c(1, 11, 9, 3), tea = c(3, 1, 1, 3), one_sided = c(6, 2, 1, 7), ci = c(10, 5, 5, 10))
  m <- do.call(rbind, tabs)
  d <- ctx$gen_dataset("fisher_legacy", data.frame(table = names(tabs), a = m[, 1], b = m[, 2], c = m[, 3], d = m[, 4]))
  v <- list()
  for (i in seq_len(nrow(d))) {
    fv <- .fisher_values(matrix(c(d$a[i], d$b[i], d$c[i], d$d[i]), 2, 2, byrow = TRUE))
    for (k in names(fv)) v[[paste0(d$table[i], "_", k)]] <- fv[[k]]
  }
  list(values = v, reference = "stats::fisher.test(alternative = 'two.sided' | 'less' | 'greater'); limits re-solved with uniroot(tol = 1e-15) (T2-WP05)",
       call = "fisher.test(matrix(c(a, b, c, d), 2, byrow = TRUE), alternative = ...)", packages = character(0))
})

.aov_rows <- function(fit) {
  s <- summary(fit)[[1]]; rn <- trimws(rownames(s))
  out <- list()
  for (i in seq_along(rn)) {
    key <- gsub(":", "_x_", rn[i])
    out[[paste0(key, "_ss")]] <- s[i, "Sum Sq"]; out[[paste0(key, "_df")]] <- s[i, "Df"]; out[[paste0(key, "_ms")]] <- s[i, "Mean Sq"]
    if (rn[i] != "Residuals") { out[[paste0(key, "_f")]] <- s[i, "F value"]; out[[paste0(key, "_p")]] <- s[i, "Pr(>F)"] }
  }
  out
}

case("stat_anova_oneway_legacy", "stats/anova", "aov on anova.rs's one-way fertilizer data (legacy test, STAT-62)", function(ctx) {
  d <- ctx$gen_dataset("anova_oneway_legacy", data.frame(
    yield = c(9.5, 10.2, 10.8, 9.8, 10.5, 14.2, 15.5, 14.8, 15.2, 15.8, 19.5, 20.2, 19.8, 20.5, 20.8),
    fertilizer = rep(c("A", "B", "C"), each = 5)))
  fit <- aov(yield ~ fertilizer, data = d)
  v <- .aov_rows(fit)
  ss <- summary(fit)[[1]][, "Sum Sq"]
  v$eta_squared <- ss[1] / sum(ss); v$ss_total <- sum(ss); v$grand_mean <- mean(d$yield); v$n <- nrow(d)
  ms <- summary(fit)[[1]][, "Mean Sq"]; dfs <- summary(fit)[[1]][, "Df"]
  v$omega_squared <- max(0, (ss[1] - dfs[1] * ms[2]) / (sum(ss) + ms[2]))   # (SSB - (k-1) MSW) / (SST + MSW), cross-check
  for (s in c("length", "mean", "sd", "min", "max")) v[[paste0("group_", s)]] <- vec(as.numeric(tapply(d$yield, d$fertilizer, get(s))))
  list(values = v, reference = "stats::aov; summary.aov; tapply(mean, sd, min, max)", call = "aov(yield ~ fertilizer, data = d)", packages = character(0))
})

case("stat_anova_twoway_legacy", "stats/anova", "aov on anova.rs's balanced 2x2 fertilizer x water data (legacy tests, STAT-62)", function(ctx) {
  d <- ctx$gen_dataset("anova_twoway_legacy", data.frame(
    yield = c(10, 11, 12, 15, 16, 17, 20, 21, 22, 30, 31, 32),
    fertilizer = rep(rep(c("A", "B"), each = 3), 2), water = rep(c("Low", "High"), each = 6)))
  d$water <- factor(d$water, levels = c("Low", "High"))
  vi <- .aov_rows(aov(yield ~ fertilizer * water, data = d))
  va <- .aov_rows(aov(yield ~ fertilizer + water, data = d))
  v <- c(setNames(vi, paste0("int_", names(vi))), setNames(va, paste0("add_", names(va))))
  v$ss_total <- sum((d$yield - mean(d$yield))^2); v$grand_mean <- mean(d$yield); v$n <- nrow(d)
  list(values = v, reference = "stats::aov; summary.aov (balanced, Type I = II = III)",
       call = "aov(yield ~ fertilizer * water); aov(yield ~ fertilizer + water)", packages = character(0))
})

case("stat_p_adjust", "stats/multiple_comparisons", "p.adjust, all methods, on the legacy, Hommel (STAT-31), tied and NA (STAT-30) vectors", function(ctx) {
  vecs <- list(legacy_holm = c(0.01, 0.04, 0.03, 0.02), legacy_validate = c(0.001, 0.01, 0.05, 0.1),
               legacy_five = c(0.01, 0.02, 0.03, 0.04, 0.05),
               hommel_two = c(0.01, 0.04), hommel_six = c(0.001, 0.2, 0.03, 0.04, 0.012, 0.5),
               hommel_three = c(0.02, 0.03, 0.04), hommel_five = c(0.01, 0.02, 0.03, 0.04, 0.5),
               ties = c(0.03, 0.01, 0.03, 0.2, 0.01, 0.6, 0.0004),
               na_three = c(0.01, NA, 0.03), na_four = c(0.04, NA, 0.001, 0.5),
               single = 0.3, ones = c(1, 0.9, 1))
  d <- ctx$gen_dataset("p_adjust_vectors", data.frame(vector = rep(names(vecs), lengths(vecs)), p = unlist(vecs, use.names = FALSE)))
  v <- list()
  for (nm in names(vecs)) {
    p <- d$p[d$vector == nm]
    v[[paste0(nm, "_p")]] <- vec(p)
    for (meth in c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "none"))
      v[[paste0(nm, "_", tolower(meth))]] <- vec(p.adjust(p, meth))
  }
  list(values = v, reference = "stats::p.adjust (n = number of non-NA p-values)", call = "p.adjust(p, method)", packages = character(0))
})

case("stat_pairwise_legacy", "stats/multiple_comparisons", "pairwise.t.test and pairwise.wilcox.test on pairwise.rs's legacy data (STAT-62)", function(ctx) {
  a <- c(1.0, 2.0, 3.0, 2.5, 1.5, 10.0, 11.0, 12.0, 10.5, 11.5, 20.0, 21.0, 22.0, 20.5, 21.5)
  b <- c(1, 2, 3, 4, 5, 6)
  cc <- c(1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24)
  d <- ctx$gen_dataset("pairwise_legacy", data.frame(set = c(rep("a", 15), rep("b", 6), rep("c", 15)), x = c(a, b, cc),
                                                     g = c(rep(c("A", "B", "C"), each = 5), rep(c("A", "B"), each = 3), rep(c("A", "B", "C"), each = 5))))
  lt <- function(m) vec(m[lower.tri(m, diag = TRUE)])     # (B,A), (C,A), (C,B): R's column-major lower triangle
  sa <- d[d$set == "a", ]; sb <- d[d$set == "b", ]; sc <- d[d$set == "c", ]
  v <- list(a_t_pooled_none = lt(pairwise.t.test(sa$x, sa$g, pool.sd = TRUE, p.adjust.method = "none")$p.value),
            a_t_pooled_holm = lt(pairwise.t.test(sa$x, sa$g, pool.sd = TRUE, p.adjust.method = "holm")$p.value),
            a_t_welch_none = lt(pairwise.t.test(sa$x, sa$g, pool.sd = FALSE, p.adjust.method = "none")$p.value),
            a_t_pooled_hommel = lt(pairwise.t.test(sa$x, sa$g, pool.sd = TRUE, p.adjust.method = "hommel")$p.value),
            a_w_normal_none = lt(pairwise.wilcox.test(sa$x, sa$g, p.adjust.method = "none", exact = FALSE)$p.value),
            b_w_exact_none = lt(pairwise.wilcox.test(sb$x, sb$g, p.adjust.method = "none", exact = TRUE)$p.value),
            c_w_holm = lt(pairwise.wilcox.test(sc$x, sc$g, p.adjust.method = "holm")$p.value),
            c_w_none = lt(pairwise.wilcox.test(sc$x, sc$g, p.adjust.method = "none")$p.value))
  list(values = v, reference = "stats::pairwise.t.test; stats::pairwise.wilcox.test", call = "pairwise.t.test(x, g, pool.sd, p.adjust.method)", packages = character(0))
})

case("stat_factanal_promax", "stats/factanal", "factanal (none, varimax, promax) on 12 variables / 3 correlated factors, n = 500, tight L-BFGS-B; varimax and promax of R's unrotated loadings", function(ctx) {
  set.seed(20261118)
  n <- 500
  Phi <- matrix(c(1, 0.3, 0.4, 0.3, 1, 0.5, 0.4, 0.5, 1), 3)
  L <- matrix(0, 12, 3)
  L[1:4, 1] <- c(0.8, 0.7, 0.75, 0.6); L[5:8, 2] <- c(0.8, 0.65, 0.7, 0.75); L[9:12, 3] <- c(0.7, 0.8, 0.6, 0.75)
  L[1, 2] <- 0.2; L[6, 3] <- 0.25; L[10, 1] <- 0.15
  psi <- 1 - rowSums((L %*% Phi) * L)
  Fs <- MASS::mvrnorm(n, rep(0, 3), Phi)
  X <- Fs %*% t(L) + matrix(rnorm(n * 12), n, 12) %*% diag(sqrt(psi))
  colnames(X) <- paste0("v", 1:12)
  d <- ctx$gen_dataset("factanal_promax", as.data.frame(round(X, 10)))
  X <- as.matrix(d)
  tight <- list(opt = list(factr = 1, pgtol = 0, maxit = 10000))
  f0 <- factanal(X, 3, rotation = "none", control = tight)
  fv <- factanal(X, 3, rotation = "varimax", control = tight)
  fp <- factanal(X, 3, rotation = "promax", control = tight)
  fpd <- factanal(X, 3, rotation = "promax")                                 # R defaults (factr 1e7): cross-check
  L0 <- unclass(f0$loadings)
  vm <- varimax(L0); pm <- promax(L0)
  vt <- varimax(L0, eps = 1e-14)                                          # the varimax optimum, not R's eps 1e-5 iterate
  Lv <- unclass(vt$loadings); Q <- Lv * abs(Lv)^3                         # promax's own four lines, m = 4, on it
  U <- lm.fit(Lv, Q)$coefficients; U <- U %*% diag(sqrt(diag(solve(t(U) %*% U))))
  Lp <- Lv %*% U; Rp <- vt$rotmat %*% U
  srt <- function(L, R) { o <- order(-colSums(L^2)); s <- ifelse(colSums(L[, o]) < 0, -1, 1)
    list(L = L[, o] %*% diag(s), R = R[, o] %*% diag(s)) }                 # sortLoadings' order and signs
  sv <- srt(Lv, vt$rotmat); sp <- srt(Lp, Rp)
  vcrit <- function(L) { Z <- (L / sqrt(rowSums(L^2)))^2; sum(colSums(Z^2) - colSums(Z)^2 / nrow(L)) }    # normalised varimax criterion
  # promax factor correlations in the order and sign of factanal's sorted loadings (sortLoadings rule)
  ord <- order(-colSums(unclass(pm$loadings)^2)); Ls <- unclass(pm$loadings)[, ord]
  neg <- colSums(Ls) < 0; Ls[, neg] <- -Ls[, neg]; u <- ifelse(neg, -1, 1)
  phi_unsorted <- solve(t(pm$rotmat) %*% pm$rotmat)
  phi_sorted <- diag(u) %*% phi_unsorted[ord, ord] %*% diag(u)
  stopifnot(isTRUE(all.equal(Ls, unclass(fp$loadings), check.attributes = FALSE, tolerance = 1e-8)))
  rm <- function(m) vec(as.numeric(t(unclass(m))))                          # row-major p x k
  ssq <- function(m) vec(colSums(unclass(m)^2) / nrow(m))
  v <- list(uniquenesses = vec(f0$uniquenesses), loadings_none = rm(f0$loadings), loadings_varimax = rm(fv$loadings),
            loadings_promax = rm(fp$loadings), promax_phi = rm(phi_sorted),
            communalities = vec(rowSums(L0^2)), prop_var_none = ssq(f0$loadings), prop_var_varimax = ssq(fv$loadings),
            prop_var_promax = ssq(fp$loadings),
            statistic = unname(f0$STATISTIC), dof = unname(f0$dof), pval = unname(f0$PVAL), objective = unname(f0$criteria["objective"]),
            uniquenesses_default = vec(fpd$uniquenesses), loadings_promax_default = rm(fpd$loadings),
            rot_input = rm(L0), rot_varimax = rm(vm$loadings), rot_varimax_rotmat = rm(vm$rotmat),
            rot_promax = rm(pm$loadings), rot_promax_rotmat = rm(pm$rotmat), rot_promax_phi = rm(phi_unsorted),
            rot_varimax_tight = rm(sv$L), rot_varimax_rotmat_tight = rm(sv$R), rot_varimax_criterion_tight = vcrit(Lv),
            rot_promax_tight = rm(sp$L), rot_promax_rotmat_tight = rm(sp$R), rot_promax_phi_tight = rm(solve(t(sp$R) %*% sp$R)),
            n = n, p = 12, k = 3)
  list(values = v, reference = "stats::factanal(rotation = 'none' | 'varimax' | 'promax', control = list(opt = list(factr = 1, pgtol = 0))); stats::varimax; stats::promax",
       call = "factanal(X, 3, rotation = 'promax')", packages = "MASS")
})

case("stat_constroptim_legacy", "stats/constroptim", "constrOptim on constroptim.rs's four legacy problems (STAT-62)", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")    # no data needed
  q <- constrOptim(c(1, 1), function(x) x[1]^2 + x[2]^2, grad = function(x) c(2 * x[1], 2 * x[2]), ui = matrix(c(1, 1), nrow = 1), ci = 1)
  l <- constrOptim(c(0.3, 0.3), function(x) -x[1] - x[2], grad = function(x) c(-1, -1),
                   ui = matrix(c(1, 0, 0, 1, -1, -1), nrow = 3, byrow = TRUE), ci = c(0, 0, -1))
  nm <- constrOptim(c(0.5, 0.5), function(x) (x[1] - 1)^2 + (x[2] - 2)^2, grad = NULL, ui = diag(2), ci = c(0, 0))
  b <- constrOptim(2, function(x) x^2, grad = function(x) 2 * x, ui = matrix(1), ci = 1)
  v <- list()
  for (k in c("quadratic", "lp3", "nelder_mead", "binding")) {
    r <- switch(k, quadratic = q, lp3 = l, nelder_mead = nm, binding = b)
    v[[paste0(k, "_par")]] <- vec(r$par); v[[paste0(k, "_value")]] <- r$value
    v[[paste0(k, "_convergence")]] <- r$convergence; v[[paste0(k, "_outer_iterations")]] <- r$outer.iterations
  }
  list(dataset = "validation/datasets/grunfeld.csv", values = v, reference = "stats::constrOptim (mu = 1e-4, outer.eps = 1e-5; BFGS with grad, Nelder-Mead without)",
       call = "constrOptim(theta, f, grad, ui, ci)", packages = character(0))
})
