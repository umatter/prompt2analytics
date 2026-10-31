# validation/r/cases/stats_mcp.R -- references for the MCP statistics handlers that extract their
# own columns: loglin, weighted.mean, cov.wt, mauchly.test, fivenum/IQR/mad/ecdf, spline/approx,
# isoreg, medpolish and prop.trend.test (T2-WP20: TOOL-8, TOOL-9, TOOL-24, TOOL-17 and the
# X-2/X-8/TOOL-3 sites in crates/p2a-mcp/src/tools/handlers/{stats,hypothesis}.rs).

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
