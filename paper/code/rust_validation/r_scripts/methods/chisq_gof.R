# chisq_gof.R - Chi-squared goodness of fit test using stats::chisq.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       factor_var = NULL, expected_probs = NULL) {

  # Use factor_var if provided for computing frequency table, otherwise dep_var
  col <- if (!is.null(factor_var)) factor_var else dep_var
  x <- data[[col]]

  # Always compute frequency table (goodness-of-fit tests work on counts)
  observed <- as.numeric(table(x))

  # Run chi-squared goodness of fit test
  if (!is.null(expected_probs)) {
    result <- chisq.test(observed, p = expected_probs)
  } else {
    result <- chisq.test(observed)
  }

  list(
    statistic = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    observed = as.list(result$observed),
    expected = as.list(result$expected),
    residuals = as.list(result$residuals),
    stdres = as.list(result$stdres),
    n_obs = sum(observed)
  )
}
