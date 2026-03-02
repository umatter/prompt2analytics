# acf_pacf.R - ACF and PACF computation using stats::acf

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       lag_max = NULL, type = "correlation") {

  # Extract time series
  y <- data[[dep_var]]

  # Compute ACF
  if (is.null(lag_max)) {
    lag_max <- min(floor(10 * log10(length(y))), length(y) - 1)
  }

  acf_result <- acf(y, lag.max = lag_max, type = type, plot = FALSE)
  acf_values <- as.numeric(acf_result$acf)
  acf_lags <- as.numeric(acf_result$lag)

  # Compute PACF
  pacf_result <- pacf(y, lag.max = lag_max, plot = FALSE)
  pacf_values <- as.numeric(pacf_result$acf)
  pacf_lags <- as.numeric(pacf_result$lag)

  # Confidence interval bounds (approximate 95%)
  ci_bound <- qnorm(0.975) / sqrt(length(y))

  result <- list(
    acf_values = as.list(acf_values),
    acf_lags = as.list(acf_lags),
    pacf_values = as.list(pacf_values),
    pacf_lags = as.list(pacf_lags),
    ci_upper = ci_bound,
    ci_lower = -ci_bound,
    lag_max = lag_max,
    n_obs = length(y)
  )

  # Cross-correlation if second variable provided
  if (!is.null(indep_vars) && length(indep_vars) >= 1) {
    x <- data[[indep_vars[1]]]
    ccf_result <- ccf(y, x, lag.max = lag_max, plot = FALSE)
    result$ccf_values <- as.list(as.numeric(ccf_result$acf))
    result$ccf_lags <- as.list(as.numeric(ccf_result$lag))
  }

  result
}
