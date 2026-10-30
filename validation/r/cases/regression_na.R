# validation/r/cases/regression_na.R -- OLS with missing cluster identifiers, supsmu / line with
# missing cells (T2-WP15: REG-13, REG-21). fixest drops rows whose cluster id is NA and re-estimates;
# sandwich::vcovCL refuses NA clusters, so the reference is lm + vcovCL on the complete rows.
suppressPackageStartupMessages({ library(sandwich); library(fixest) })

case("reg_cluster_na", "regression", "OLS with clustered SEs when some cluster ids are missing (rows dropped, model re-estimated)", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  g$firm_label <- sprintf("firm_%02d", g$firm)
  g$firm_label[c(3, 44, 101)] <- NA
  g$year_na <- g$year; g$year_na[c(10, 190)] <- NA
  d <- ctx$gen_dataset("grunfeld_cluster_na", g[, c("firm", "year", "firm_label", "year_na", "inv", "value", "capital")])
  cc <- d[!is.na(d$firm_label), ]
  m <- lm(inv ~ value + capital, data = cc)
  f <- feols(inv ~ value + capital, data = d, cluster = ~firm_label)
  cc2 <- d[!is.na(d$firm_label) & !is.na(d$year_na), ]
  m2 <- lm(inv ~ value + capital, data = cc2)
  list(values = list(coef = named(coef(m)), se_cluster_hc1 = named(sqrt(diag(vcovCL(m, cluster = ~firm_label, type = "HC1")))),
                     se_cluster_fixest = named(se(f)), n_used = nobs(m), n_dropped = nrow(d) - nobs(m),
                     coef_twoway = named(coef(m2)),
                     se_cluster_twoway_hc1 = named(sqrt(diag(vcovCL(m2, cluster = ~firm_label + year_na, type = "HC1")))),
                     n_used_twoway = nobs(m2)),
       reference = "stats::lm + sandwich::vcovCL(type = 'HC1') on rows with a cluster id (fixest::feols(cluster = ) drops them likewise)",
       call = "lm(inv ~ value + capital, data = d[!is.na(d$firm_label), ]); vcovCL(m, cluster = ~firm_label, type = 'HC1')",
       packages = c("sandwich", "fixest"))
})

case("reg_smoothers_na", "regression", "supsmu and Tukey's resistant line on data with missing cells (R drops incomplete pairs)", function(ctx) {
  set.seed(8)
  x <- round(runif(60, 0, 10), 4); y <- round(sin(x) + 0.1 * x + rnorm(60, 0, 0.3), 4)
  x[c(5, 33)] <- NA; y[c(12, 48)] <- NA
  d <- ctx$gen_dataset("smoothers_na", data.frame(id = 1:60, x = x, y = y))
  s <- suppressWarnings(supsmu(d$x, d$y))
  l <- line(d$x, d$y)
  list(values = list(supsmu_x = vec(s$x), supsmu_y = vec(s$y), line_coef = vec(coef(l)),
                     n_complete = sum(complete.cases(d$x, d$y))),
       reference = "stats::supsmu (NA pairs deleted with a warning); stats::line (complete pairs)",
       call = "supsmu(x, y); line(x, y)", packages = character(0))
})
