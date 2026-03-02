# stl.R - STL decomposition using stats::stl

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       frequency = 12, s_window = "periodic") {

  # Extract time series
  y <- data[[dep_var]]

  # Create ts object
  ts_data <- ts(y, frequency = frequency)

  # Fit STL decomposition
  if (s_window == "periodic") {
    decomp <- stl(ts_data, s.window = "periodic")
  } else {
    decomp <- stl(ts_data, s.window = as.numeric(s_window))
  }

  # Extract components
  seasonal <- as.numeric(decomp$time.series[, "seasonal"])
  trend <- as.numeric(decomp$time.series[, "trend"])
  remainder <- as.numeric(decomp$time.series[, "remainder"])

  list(
    seasonal = as.list(seasonal),
    trend = as.list(trend),
    remainder = as.list(remainder),
    frequency = frequency,
    s_window = s_window,
    n_obs = length(y)
  )
}
