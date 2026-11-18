# validation/r/cases/data_viz_t4.R -- exporters keep small coefficients (DATA-77): lm on Grunfeld with value in dollars.
case("export_ols_grunfeld_usd", "export", "CSV and table exporters on a dollar-scaled regressor (stats::lm, classical SEs)", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  g$value_usd <- g$value * 1e6
  m <- lm(inv ~ value_usd + capital, data = g)
  s <- summary(m); cf <- coef(s); ci <- confint(m)
  list(values = list(
         n_obs = nobs(m),
         coef = named(cf[, 1]), se = named(cf[, 2]), t = named(cf[, 3]), p = named(cf[, 4]),
         ci_lower = named(ci[, 1]), ci_upper = named(ci[, 2]),
         r2 = s$r.squared, adj_r2 = s$adj.r.squared, f_stat = unname(s$fstatistic["value"]),
         sigma = s$sigma),
       reference = "stats::lm; summary.lm; confint.lm (classical SEs)",
       call = "lm(inv ~ value_usd + capital, grunfeld) with value_usd = value * 1e6",
       packages = character(0))
})
