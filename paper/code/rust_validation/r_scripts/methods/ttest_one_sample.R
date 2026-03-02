# ttest_one_sample.R - One-sample t-test using stats::t.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       mu = 0, alternative = "two.sided", conf_level = 0.95) {

  x <- data[[dep_var]]

  # Run one-sample t-test
  result <- t.test(x, mu = mu, alternative = alternative, conf.level = conf_level)

  list(
    statistic = as.numeric(result$statistic),
    p_value = result$p.value,
    df = as.numeric(result$parameter),
    mean = as.numeric(result$estimate),
    conf_int_lower = result$conf.int[1],
    conf_int_upper = result$conf.int[2],
    conf_level = conf_level,
    mu = mu,
    alternative = alternative,
    n_obs = length(x),
    std_error = as.numeric(result$statistic) / (as.numeric(result$estimate) - mu) * sd(x) / sqrt(length(x))
  )
}
