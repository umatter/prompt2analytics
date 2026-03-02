# holt_winters.R - Holt-Winters exponential smoothing using stats::HoltWinters

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       frequency = 12, seasonal = "additive", h = 12) {

  # Extract time series
  y <- data[[dep_var]]

  # Create ts object
  ts_data <- ts(y, frequency = frequency)

  # Fit Holt-Winters (try seasonal first, fall back to non-seasonal)
  model <- tryCatch({
    if (frequency > 1) {
      HoltWinters(ts_data, seasonal = seasonal)
    } else {
      HoltWinters(ts_data, gamma = FALSE)
    }
  }, error = function(e) {
    # If seasonal model fails, try non-seasonal (double exponential smoothing)
    HoltWinters(ts(y, frequency = 1), gamma = FALSE)
  })

  # Forecasting
  fc <- predict(model, n.ahead = h)

  # Extract fitted values
  fitted_vals <- as.numeric(model$fitted[, "xhat"])

  result <- list(
    alpha = model$alpha,
    beta = model$beta,
    sse = model$SSE,
    level = as.numeric(model$coefficients["a"]),
    trend = as.numeric(model$coefficients["b"]),
    fitted = as.list(fitted_vals),
    forecast = as.list(as.numeric(fc)),
    n_obs = length(y),
    frequency = frequency,
    seasonal_type = seasonal
  )

  # Add gamma and seasonal coefficients if seasonal model
  if (frequency > 1) {
    result$gamma <- model$gamma
    seasonal_coefs <- model$coefficients[grep("^s", names(model$coefficients))]
    result$seasonal_coefficients <- as.list(seasonal_coefs)
  }

  result
}
