# wilcoxon.R - Wilcoxon rank-sum test using stats::wilcox.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       alternative = "two.sided", paired = FALSE, exact = FALSE,
                       correct = TRUE, mu = 0) {

  x <- data[[dep_var]]

  if (!is.null(indep_vars) && length(indep_vars) >= 1) {
    # Two-sample or paired test
    y <- data[[indep_vars[1]]]

    result <- wilcox.test(x, y, alternative = alternative, paired = paired,
                          exact = exact, correct = correct, mu = mu)

    res <- list(
      statistic = as.numeric(result$statistic),
      p_value = result$p.value,
      alternative = alternative,
      paired = paired,
      n_x = length(x),
      n_y = length(y),
      n_obs = length(x) + length(y)
    )
  } else {
    # One-sample signed rank test
    result <- wilcox.test(x, mu = mu, alternative = alternative,
                          exact = exact, correct = correct)

    res <- list(
      statistic = as.numeric(result$statistic),
      p_value = result$p.value,
      alternative = alternative,
      mu = mu,
      n_obs = length(x)
    )
  }

  # Add method name
  res$method <- result$method

  res
}
