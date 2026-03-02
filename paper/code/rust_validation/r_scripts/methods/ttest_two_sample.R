# ttest_two_sample.R - Two-sample t-test using stats::t.test

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       alternative = "two.sided", var_equal = FALSE, conf_level = 0.95) {

  # Two modes:
  # 1. dep_var=values, indep_vars[1]=group variable (binary groups)
  # 2. dep_var=sample1, indep_vars[1]=sample2 (both continuous — compare directly)
  x_col <- dep_var
  y_col <- indep_vars[1]

  x_vals <- data[[x_col]]
  y_vals <- data[[y_col]]

  # If the "group" column is categorical with exactly 2 levels, do grouped t-test
  if (is.character(y_vals) || is.factor(y_vals) || length(unique(y_vals)) == 2) {
    groups <- unique(y_vals)
    x <- x_vals[y_vals == groups[1]]
    y <- x_vals[y_vals == groups[2]]
  } else {
    # Both columns are continuous — compare them directly as two independent samples
    x <- x_vals
    y <- y_vals
  }

  # Run two-sample t-test
  result <- t.test(x, y, alternative = alternative, var.equal = var_equal,
                   conf.level = conf_level)

  list(
    statistic = as.numeric(result$statistic),
    p_value = result$p.value,
    df = as.numeric(result$parameter),
    mean_x = as.numeric(result$estimate[1]),
    mean_y = as.numeric(result$estimate[2]),
    mean_difference = as.numeric(result$estimate[1] - result$estimate[2]),
    conf_int_lower = result$conf.int[1],
    conf_int_upper = result$conf.int[2],
    conf_level = conf_level,
    alternative = alternative,
    var_equal = var_equal,
    n_x = length(x),
    n_y = length(y),
    n_obs = length(x) + length(y)
  )
}
