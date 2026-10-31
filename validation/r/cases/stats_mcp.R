# validation/r/cases/stats_mcp.R -- references for the MCP statistics handlers that extract their
# own columns: loglin, weighted.mean, cov.wt, mauchly.test, fivenum/IQR/mad/ecdf, spline/approx,
# isoreg, medpolish and prop.trend.test (T2-WP20: TOOL-8, TOOL-9, TOOL-24, TOOL-17 and the
# X-2/X-8/TOOL-3 sites in crates/p2a-mcp/src/tools/handlers/{stats,hypothesis}.rs).

# A 3 x 2 x 2 count table in long form with one duplicated cell (xtabs sums it), a row with a
# missing f1 label, a row with a missing f2 code and a row with a missing count. f2 holds the
# integer codes 2 and 10 (Int64 in polars), so its levels must be ordered numerically (2 < 10).
case("stat_loglin_na", "stats/loglin", "Log-linear models (IPF) on a long-form count table with missing labels and counts", function(ctx) {
  set.seed(211)
  grid <- expand.grid(f1 = c("a", "b", "c"), f2 = c(2, 10), f3 = c("x", "y"), stringsAsFactors = FALSE)
  lam <- with(grid, 20 * (1 + (f1 == "b") + 0.5 * (f1 == "c")) * ifelse(f2 == 10, 1.6, 1) *
                    ifelse(f3 == "y", 0.7, 1) * ifelse(f1 == "a" & f2 == 10, 1.8, 1) *
                    ifelse(f1 == "c" & f3 == "y", 1.5, 1))
  grid$cnt <- rpois(nrow(grid), lam)
  extra <- data.frame(f1 = c("b", NA, "c", "a"), f2 = c(10, 2, NA, 2), f3 = c("y", "x", "x", "y"),
                      cnt = c(4, 9, 6, NA))
  d0 <- rbind(grid, extra)
  d <- ctx$gen_dataset("loglin_na", data.frame(id = seq_len(nrow(d0)), d0))
  stopifnot(is.integer(d$f2), is.integer(d$cnt), sum(!complete.cases(d$cnt, d$f1, d$f2, d$f3)) == 3)
  tab <- xtabs(cnt ~ f1 + f2 + f3, data = d)            # drops the rows with a missing label or count
  stopifnot(identical(dimnames(tab)$f2, c("2", "10")))
  models <- list(indep = list(1, 2, 3), m12_3 = list(c(1, 2), 3), m12_13 = list(c(1, 2), c(1, 3)))
  v <- list()
  for (k in names(models)) {
    f <- loglin(tab, models[[k]], fit = TRUE, print = FALSE)
    v[[paste0(k, "_lrt")]] <- f$lrt; v[[paste0(k, "_pearson")]] <- f$pearson; v[[paste0(k, "_df")]] <- f$df
    # loglin reports no p-values; p2a reports the chi-square upper tails of the two statistics
    v[[paste0(k, "_p_lrt")]] <- pchisq(f$lrt, f$df, lower.tail = FALSE)
    v[[paste0(k, "_p_pearson")]] <- pchisq(f$pearson, f$df, lower.tail = FALSE)
  }
  cc <- complete.cases(d$cnt, d$f1, d$f2, d$f3)
  v$total <- sum(tab); v$dims <- vec(dim(tab)); v$n_used <- sum(cc); v$n_total <- nrow(d)
  v$n_missing_cnt <- sum(is.na(d$cnt)); v$n_missing_f1 <- sum(is.na(d$f1)); v$n_missing_f2 <- sum(is.na(d$f2))
  list(values = v, reference = "stats::loglin(xtabs(cnt ~ f1 + f2 + f3), margin, fit = TRUE); eps = 0.1, iter = 20 (defaults)",
       call = "loglin(xtabs(cnt ~ f1 + f2 + f3, data = d), list(1, 2, 3) | list(c(1, 2), 3) | list(c(1, 2), c(1, 3)))",
       packages = character(0))
})

# Chi-squared test for trend in proportions on integer count columns. Group 4 has a missing
# success count; R's prop.trend.test refuses it ("0 (non-NA) cases"), the MCP tool drops the
# group and keeps every other group's score (default: its row position, 1, 2, 3, 5, 6, 7).
# trials_neg and successes_frac exist only for the tool's count checks.
case("stat_prop_trend_na", "stats/proportions", "prop.trend.test on integer count columns with a missing group", function(ctx) {
  d <- ctx$gen_dataset("prop_trend_na", data.frame(
    group = 1:7, dose = c(0, 1, 2, 4, 8, 16, 32),
    successes = c(3, 5, 9, NA, 14, 18, 23), trials = c(40, 42, 41, 39, 40, 43, 41),
    trials_neg = c(40, 42, 41, 39, -1, 43, 41), successes_frac = c(3, 5.5, 9, 11, 14, 18, 23)))
  stopifnot(is.integer(d$successes), is.integer(d$trials), is.integer(d$trials_neg), is.double(d$successes_frac))
  e <- tryCatch({ prop.trend.test(d$successes, d$trials); "no error" }, error = conditionMessage)
  stopifnot(e == "0 (non-NA) cases")
  cc <- complete.cases(d$successes, d$trials)
  pt <- prop.trend.test(d$successes[cc], d$trials[cc], score = which(cc))
  pd <- prop.trend.test(d$successes[cc], d$trials[cc], score = d$dose[cc])
  list(values = list(chisq = unname(pt$statistic), df = unname(pt$parameter), p = pt$p.value,
                     scores = vec(which(cc)),
                     dose_chisq = unname(pd$statistic), dose_p = pd$p.value, dose_scores = vec(d$dose[cc]),
                     n_used = sum(cc), n_total = nrow(d)),
       reference = "stats::prop.trend.test on the complete groups, score = the groups' own scores",
       call = "prop.trend.test(successes[cc], trials[cc], score = which(cc) | dose[cc])", packages = character(0))
})
