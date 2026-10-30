# validation/r/cases/stats_contingency.R -- fisher.test / chisq.test / mantelhaen.test references
# for the contingency-table package (T2-WP05: STAT-2, STAT-3, STAT-5, STAT-10, STAT-17, STAT-24,
# STAT-45, STAT-50, TOOL-33).

.fisher_values <- function(tab) {
  out <- list()
  for (alt in c("two.sided", "less", "greater")) {
    f <- fisher.test(tab, alternative = alt)
    tag <- sub("\\.", "_", alt)
    out[[paste0("p_", tag)]] <- f$p.value
    out[[paste0("ci_lower_", tag)]] <- f$conf.int[1]
    # an infinite upper bound is stored as null (the harness refuses Inf); the Rust test
    # checks f64::INFINITY where R reports Inf
    out[[paste0("ci_upper_", tag)]] <- if (is.finite(f$conf.int[2])) f$conf.int[2] else NA_real_
  }
  f <- fisher.test(tab)
  out$estimate_cond_mle <- unname(f$estimate)
  out$sample_odds_ratio <- (tab[1, 1] * tab[2, 2]) / (tab[1, 2] * tab[2, 1])
  out$n <- sum(tab)
  # fisher.test solves the conditional MLE and the limits with uniroot(tol = eps^0.25), so its
  # values carry ~1e-4 error; the same equations re-solved with tol = 1e-15 are the tight reference.
  tt <- .fisher_tight(tab)
  for (k in names(tt)) out[[paste0(k, "_tight")]] <- tt[[k]]
  out
}

# fisher.test's 2x2 internals (R 4.3.3 stats::fisher.test) with a tight root-finding tolerance.
.fisher_tight <- function(tab, conf.level = 0.95) {
  x <- tab[1, 1]; m <- sum(tab[, 1]); n <- sum(tab[, 2]); k <- sum(tab[1, ])
  lo <- max(0L, k - n); hi <- min(k, m); support <- lo:hi
  logdc <- dhyper(support, m, n, k, log = TRUE)
  dnhyper <- function(ncp) { d <- logdc + log(ncp) * support; d <- exp(d - max(d)); d / sum(d) }
  mnhyper <- function(ncp) { if (ncp == 0) return(lo); if (ncp == Inf) return(hi); sum(support * dnhyper(ncp)) }
  pnhyper <- function(q, ncp = 1, upper.tail = FALSE) {
    if (ncp == 1) return(if (upper.tail) phyper(x - 1, m, n, k, lower.tail = FALSE) else phyper(x, m, n, k))
    if (ncp == 0) return(as.numeric(if (upper.tail) q <= lo else q >= lo))
    if (ncp == Inf) return(as.numeric(if (upper.tail) q <= hi else q >= hi))
    sum(dnhyper(ncp)[if (upper.tail) support >= q else support <= q])
  }
  ur <- function(f, int) uniroot(f, int, tol = 1e-15, maxiter = 10000)$root
  mle <- if (x == lo) 0 else if (x == hi) Inf else {
    mu <- mnhyper(1)
    if (mu > x) ur(function(t) mnhyper(t) - x, c(0, 1)) else if (mu < x) 1 / ur(function(t) mnhyper(1 / t) - x, c(.Machine$double.eps, 1)) else 1
  }
  ncp.U <- function(alpha) {
    if (x == hi) return(Inf); p <- pnhyper(x, 1)
    if (p < alpha) ur(function(t) pnhyper(x, t) - alpha, c(0, 1)) else if (p > alpha) 1 / ur(function(t) pnhyper(x, 1 / t) - alpha, c(.Machine$double.eps, 1)) else 1
  }
  ncp.L <- function(alpha) {
    if (x == lo) return(0); p <- pnhyper(x, 1, upper.tail = TRUE)
    if (p > alpha) ur(function(t) pnhyper(x, t, upper.tail = TRUE) - alpha, c(0, 1)) else if (p < alpha) 1 / ur(function(t) pnhyper(x, 1 / t, upper.tail = TRUE) - alpha, c(.Machine$double.eps, 1)) else 1
  }
  a2 <- (1 - conf.level) / 2
  fin <- function(v) if (is.finite(v)) v else NA_real_
  list(estimate_cond_mle = fin(mle), ci_lower_two_sided = ncp.L(a2), ci_upper_two_sided = fin(ncp.U(a2)),
       ci_upper_less = fin(ncp.U(1 - conf.level)), ci_lower_greater = ncp.L(1 - conf.level))
}

# Tables given as counts [[a, b], [c, d]] (row-major), fed to fisher_exact_test directly.
.fisher_tables <- list(
  tea        = c(3, 1, 1, 3),
  n1000      = c(260, 240, 240, 260),
  n1030      = c(265, 250, 250, 265),
  n1040      = c(270, 250, 250, 270),
  n2000      = c(520, 480, 480, 520),
  n2000_sig  = c(600, 400, 400, 600),
  tiny_p     = c(100, 10, 10, 100),
  rare       = c(10, 1990, 20, 1980),
  unbal      = c(100, 900, 120, 880),
  zero_cell  = c(0, 5, 3, 2),
  null_2400  = c(600, 600, 600, 600)
)

case("stat_fisher_tables", "stats/contingency", "Fisher's exact test on 2x2 count tables (N = 8 to 4000)", function(ctx) {
  ids <- names(.fisher_tables)
  m <- do.call(rbind, .fisher_tables)
  d <- ctx$gen_dataset("fisher_tables", data.frame(table = ids, a = m[, 1], b = m[, 2], c = m[, 3], d = m[, 4]))
  v <- list()
  for (i in seq_len(nrow(d))) {
    tab <- matrix(c(d$a[i], d$b[i], d$c[i], d$d[i]), 2, 2, byrow = TRUE)
    fv <- .fisher_values(tab)
    for (k in names(fv)) v[[paste0(d$table[i], "_", k)]] <- fv[[k]]
  }
  list(values = v, reference = "stats::fisher.test(alternative = 'two.sided' | 'less' | 'greater')",
       call = "fisher.test(matrix(c(a, b, c, d), 2, byrow = TRUE), alternative = ...)", packages = character(0))
})

# Row-level data for the N = 2000 table: exercises run_fisher_test (and, through MCP, the
# polars group_by that panicked inside the tokio runtime, STAT-10).
case("stat_fisher_rows_2000", "stats/contingency", "Fisher's exact test from two columns, 2000 rows", function(ctx) {
  set.seed(20261021)
  treat <- rep(c("control", "treated"), each = 1000)
  outcome <- c(rep(c("no", "yes"), c(520, 480)), rep(c("no", "yes"), c(480, 520)))
  o <- sample(2000)
  d <- ctx$gen_dataset("fisher_rows_2000", data.frame(treat = treat[o], outcome = outcome[o]))
  tab <- table(d$treat, d$outcome)
  stopifnot(identical(as.vector(t(tab)), c(520L, 480L, 480L, 520L)))
  v <- .fisher_values(tab)
  ch <- chisq.test(tab)
  v$chisq_yates <- unname(ch$statistic); v$chisq_yates_p <- ch$p.value
  list(values = v, reference = "stats::fisher.test(table(treat, outcome)); stats::chisq.test (Yates)",
       call = "fisher.test(table(d$treat, d$outcome))", packages = character(0))
})

# Missing values and multi-digit numeric codes (STAT-17, STAT-50): R's table() drops NA and
# orders numeric levels numerically (2 < 10).
case("stat_contingency_na_codes", "stats/contingency", "chisq.test / fisher.test with NA cells and numeric codes 2/10", function(ctx) {
  g <- c("a", "a", "a", "b", "b", "b", "a", "b", "a", "b", "a", "b")
  h <- c("u", "u", "v", "v", "v", "u", NA, NA, "u", "v", NA, "v")
  code <- c(10, 2, 10, 2, 2, 10, 2, 10, 10, 2, NA, 2)       # numeric levels 2 and 10
  out <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1)
  d <- ctx$gen_dataset("contingency_na_codes", data.frame(g = g, h = h, code = code, out = out))
  t1 <- table(d$g, d$h)
  c1 <- chisq.test(t1)
  c1u <- chisq.test(t1, correct = FALSE)
  t2 <- table(d$code, d$out)
  stopifnot(identical(rownames(t2), c("2", "10")))
  f2 <- fisher.test(t2)
  g1 <- chisq.test(table(d$h))
  f2t <- .fisher_tight(t2)
  list(values = list(
    gh_chisq_yates = unname(c1$statistic), gh_df = unname(c1$parameter), gh_p_yates = c1$p.value,
    gh_chisq = unname(c1u$statistic), gh_p = c1u$p.value, gh_n = sum(t1),
    gh_observed = vec(as.vector(t(t1))), gh_expected = vec(as.vector(t(c1u$expected))),
    gh_residuals = vec(as.vector(t(c1u$residuals))), gh_stdres = vec(as.vector(t(c1u$stdres))),
    code_out_table = vec(as.vector(t(t2))), code_out_fisher_p = f2$p.value,
    code_out_fisher_estimate = unname(f2$estimate),
    code_out_fisher_ci_lower = f2$conf.int[1], code_out_fisher_ci_upper = f2$conf.int[2],
    code_out_sample_or = (t2[1, 1] * t2[2, 2]) / (t2[1, 2] * t2[2, 1]),
    code_out_fisher_estimate_tight = f2t$estimate_cond_mle,
    code_out_fisher_ci_lower_tight = f2t$ci_lower_two_sided, code_out_fisher_ci_upper_tight = f2t$ci_upper_two_sided,
    h_gof_chisq = unname(g1$statistic), h_gof_df = unname(g1$parameter), h_gof_p = g1$p.value,
    h_gof_observed = vec(as.vector(table(d$h)))),
    reference = "stats::chisq.test(table(...)); stats::fisher.test(table(...)) (table() drops NA, numeric levels in numeric order)",
    call = "chisq.test(table(g, h)); fisher.test(table(code, out)); chisq.test(table(h))", packages = character(0))
})

# Goodness of fit with user probabilities (STAT-3) and a category that never occurs (STAT-45).
case("stat_chisq_gof", "stats/contingency", "Chi-squared goodness of fit with named probabilities and an unobserved level", function(ctx) {
  set.seed(7)
  cat <- sample(c(rep("a", 5), rep("b", 30), rep("c", 15)))
  cat2 <- c(cat, NA, NA)                                     # two missing cells, dropped by table()
  d <- ctx$gen_dataset("chisq_gof", data.frame(id = seq_along(cat2), cat = cat2))   # id: a row of only NA would be a blank line
  tb <- table(d$cat)
  stopifnot(identical(names(tb), c("a", "b", "c")))
  p1 <- c(a = 0.1, b = 0.6, c = 0.3); p2 <- c(a = 0.2, b = 0.5, c = 0.3)
  g0 <- chisq.test(tb); g1 <- chisq.test(tb, p = p1); g2 <- chisq.test(tb, p = p2)
  # die with face 6 never rolled: categories 1..6 named explicitly
  face <- factor(c(rep(1, 9), rep(2, 11), rep(3, 8), rep(4, 12), rep(5, 10)), levels = 1:6)
  gd <- chisq.test(table(face))
  # a named map that declares a category ("d") the data never contains (STAT-45 on the dataset path)
  p4 <- c(a = 0.1, b = 0.5, c = 0.3, d = 0.1)
  g4 <- chisq.test(table(factor(d$cat, levels = names(p4))), p = p4)
  list(dataset = "validation/datasets/generated/chisq_gof.csv", values = list(
    uniform_chisq = unname(g0$statistic), uniform_df = unname(g0$parameter), uniform_p = g0$p.value,
    p1_chisq = unname(g1$statistic), p1_p = g1$p.value, p1_expected = vec(unname(g1$expected)),
    p2_chisq = unname(g2$statistic), p2_p = g2$p.value, p2_expected = vec(unname(g2$expected)),
    p2_residuals = vec(unname(g2$residuals)), observed = vec(as.vector(tb)), n_used = sum(tb),
    die_chisq = unname(gd$statistic), die_df = unname(gd$parameter), die_p = gd$p.value,
    die_observed = vec(as.vector(table(face))),
    named4_chisq = unname(g4$statistic), named4_df = unname(g4$parameter), named4_p = g4$p.value,
    named4_observed = vec(as.vector(g4$observed)), named4_expected = vec(unname(g4$expected)),
    named4_residuals = vec(unname(g4$residuals))),
    reference = "stats::chisq.test(table(x), p = c(a = ., b = ., c = .)) (levels in sorted order); unobserved level via factor(levels =)",
    call = "chisq.test(table(cat), p = c(0.1, 0.6, 0.3))", packages = character(0))
})

# Small expected counts: R warns "Chi-squared approximation may be incorrect" (TOOL-33).
case("stat_chisq_small_expected", "stats/contingency", "Chi-squared independence with expected counts below 5", function(ctx) {
  x <- c(rep("p", 6), rep("q", 4), rep("r", 3))
  y <- c("s", "s", "t", "s", "t", "s", "t", "t", "s", "t", "s", "t", "t")
  d <- ctx$gen_dataset("chisq_small_expected", data.frame(x = x, y = y))
  w <- NULL
  r <- withCallingHandlers(chisq.test(table(d$x, d$y)), warning = function(m) { w <<- conditionMessage(m); invokeRestart("muffleWarning") })
  stopifnot(identical(w, "Chi-squared approximation may be incorrect"))
  list(values = list(chisq = unname(r$statistic), df = unname(r$parameter), p = r$p.value,
                     min_expected = min(r$expected), warned = 1,
                     observed = vec(as.vector(t(r$observed))), expected = vec(as.vector(t(r$expected))),
                     residuals = vec(as.vector(t(r$residuals))), stdres = vec(as.vector(t(r$stdres)))),
       reference = "stats::chisq.test (warning when any expected count < 5)", call = "chisq.test(table(x, y))",
       packages = character(0))
})

# Mantel-Haenszel with a missing stratum label (STAT-17 duplicate): mantelhaen.test drops NA rows.
case("stat_mantelhaen_na", "stats/contingency", "Mantel-Haenszel test with NA strata", function(ctx) {
  set.seed(11)
  n <- 240
  s <- sample(c("s1", "s2", "s3"), n, replace = TRUE)
  tr <- sample(c("ctl", "trt"), n, replace = TRUE)
  pr <- ifelse(tr == "trt", 0.55, 0.35) + ifelse(s == "s2", 0.1, 0)
  y <- ifelse(runif(n) < pr, "yes", "no")
  s[c(5, 17, 99)] <- NA
  d <- ctx$gen_dataset("mantelhaen_na", data.frame(treat = tr, outcome = y, stratum = s))
  m <- mantelhaen.test(table(d$treat, d$outcome, d$stratum))
  mc <- mantelhaen.test(table(d$treat, d$outcome, d$stratum), correct = FALSE)
  list(values = list(statistic = unname(m$statistic), p = m$p.value, statistic_nocorrect = unname(mc$statistic),
                     p_nocorrect = mc$p.value, common_or = unname(m$estimate),
                     ci_lower = m$conf.int[1], ci_upper = m$conf.int[2], n_used = sum(!is.na(d$stratum))),
       reference = "stats::mantelhaen.test(table(treat, outcome, stratum)) (NA strata dropped)",
       call = "mantelhaen.test(table(treat, outcome, stratum))", packages = character(0))
})
