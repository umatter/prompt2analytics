# validation/r/cases/stats_misc_t3.R -- references: KS with distribution parameters,
# PP.test internals, cancor, prop.test, poisson.test, Mahalanobis (full rank and singular),
# two-way model tables with an empty cell, t.test at other confidence levels, acf/pacf/ccf,
# spline's default n, spec.pgram's grid and spec.ci (STAT-26, STAT-38, STAT-54, STAT-83, STAT-67,
# X-3, STAT-6, STAT-35, STAT-60; STAT-62, TOOL-11 and TOOL-25 core parts).
#
# Some keys are not pinned by the tests: they are registry deviations of open findings
# (STAT-16, STAT-27, STAT-39, STAT-58, STAT-59, STAT-63, STAT-68), named in a
# `// deviation <id>` comment next to the Check. Their R values are recorded here so a later fix
# can pin them without recomputing.

# One column per sample; NA padding keeps a rectangular CSV (Rust drops it with NaPolicy::Omit,
# the KS wrappers do the same since T2-WP06).
.pad <- function(cols) {
  len <- max(lengths(cols))
  df <- data.frame(id = seq_len(len))
  for (k in names(cols)) { v <- rep(NA_real_, len); v[seq_along(cols[[k]])] <- cols[[k]]; df[[k]] <- v }
  df
}

case("stat_ks_t3", "stats/ks", "ks.test: one-sample against pnorm/punif/pexp with parameters, two-sample; D, D+, D-, asymptotic and exact p", function(ctx) {
  set.seed(20261022)
  cols <- list(xn = round(rnorm(30, 3, 2), 4), xu = round(runif(25, -1, 3), 4), xe = round(rexp(40, 0.5), 4),
               x5 = c(1.2, 1.5, 1.8, 2.1, 2.4), y5 = c(2.0, 2.5, 3.0, 3.5, 4.0))
  set.seed(77)
  cols$xa <- round(rnorm(60), 5); cols$ya <- round(rnorm(45, 0.4, 1.2), 5)
  d <- ctx$gen_dataset("ks_t3", .pad(cols))
  g <- function(k) d[[k]][!is.na(d[[k]])]
  xn <- g("xn"); xu <- g("xu"); xe <- g("xe"); x5 <- g("x5"); y5 <- g("y5"); xa <- g("xa"); ya <- g("ya")
  stopifnot(!anyDuplicated(xn), !anyDuplicated(xu), !anyDuplicated(xe), !anyDuplicated(c(xa, ya)), !anyDuplicated(c(x5, y5)))
  one <- function(x, ...) {
    a <- ks.test(x, ..., exact = FALSE); e <- ks.test(x, ...)
    gt <- ks.test(x, ..., alternative = "greater", exact = FALSE); lt <- ks.test(x, ..., alternative = "less", exact = FALSE)
    stopifnot(sqrt(length(x)) * a$statistic >= 1)   # R's pKS2 (tol 1e-6) is exact to ~1e-15 only for sqrt(n) D >= 1
    list(d = unname(a$statistic), p_asym = a$p.value, p_exact = e$p.value,
         dplus = unname(gt$statistic), p_greater_asym = gt$p.value, dminus = unname(lt$statistic), p_less_asym = lt$p.value)
  }
  two <- function(x, y) {
    a <- ks.test(x, y, exact = FALSE); e <- ks.test(x, y)
    gt <- ks.test(x, y, alternative = "greater", exact = FALSE); lt <- ks.test(x, y, alternative = "less", exact = FALSE)
    list(d = unname(a$statistic), p_asym = a$p.value, p_exact = e$p.value,
         dplus = unname(gt$statistic), p_greater_asym = gt$p.value, dminus = unname(lt$statistic), p_less_asym = lt$p.value)
  }
  v <- list()
  add <- function(prefix, l) for (k in names(l)) v[[paste0(prefix, "_", k)]] <<- l[[k]]
  add("norm", one(xn, "pnorm", 2, 2))
  add("unif", one(xu, "punif", -1.5, 3))
  add("exp", one(xe, "pexp", 0.8))
  add("two5", two(x5, y5))
  add("twol", two(xa, ya))
  v$n_xn <- length(xn); v$n_xu <- length(xu); v$n_xe <- length(xe); v$n_xa <- length(xa); v$n_ya <- length(ya)
  list(values = v, reference = "stats::ks.test (exact = FALSE: C_pKS2 tol 1e-6 and exp(-2 n D^2); exact: pkolmogorov2x / psmirnov)",
       call = "ks.test(x, 'pnorm', 2, 2[, exact = FALSE, alternative]); ks.test(x, y[, exact = FALSE, alternative])",
       packages = character(0))
})
