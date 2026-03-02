# shapiro_wilk.R - Shapiro-Wilk normality test using stats::shapiro.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  x <- data[[dep_var]]

  # R's shapiro.test limits n to 5000; subsample if larger
  if (length(x) > 5000) {
    set.seed(seed)
    x <- sample(x, 5000)
  }

  # Run Shapiro-Wilk test
  result <- shapiro.test(x)

  list(
    statistic = as.numeric(result$statistic),
    p_value = result$p.value,
    n_obs = length(x),
    reject_normality_05 = result$p.value < 0.05,
    reject_normality_01 = result$p.value < 0.01
  )
}
