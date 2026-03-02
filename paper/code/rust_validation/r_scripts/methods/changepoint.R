# changepoint.R - Changepoint detection using changepoint package

suppressPackageStartupMessages({
  library(changepoint)
})

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       method = "PELT", test_stat = "Normal", penalty = "SIC",
                       type = "mean") {

  # Extract time series
  y <- data[[dep_var]]

  # Detect changepoints based on type
  if (type == "mean") {
    result <- cpt.mean(y, method = method, test.stat = test_stat, penalty = penalty)
  } else if (type == "variance") {
    result <- cpt.var(y, method = method, test.stat = test_stat, penalty = penalty)
  } else if (type == "meanvar") {
    result <- cpt.meanvar(y, method = method, test.stat = test_stat, penalty = penalty)
  } else {
    stop(paste("Unknown type:", type))
  }

  # Extract changepoint locations
  cpts <- cpts(result)

  # Segment parameters
  seg_means <- param.est(result)$mean
  seg_vars <- param.est(result)$variance

  list(
    changepoints = as.list(cpts),
    n_changepoints = length(cpts),
    segment_means = if (!is.null(seg_means)) as.list(seg_means) else NULL,
    segment_variances = if (!is.null(seg_vars)) as.list(seg_vars) else NULL,
    method = method,
    test_stat = test_stat,
    penalty = penalty,
    type = type,
    n_obs = length(y)
  )
}
