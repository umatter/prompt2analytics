# ar.R - AR model using stats::ar

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       method = "yule-walker", order_max = NULL) {

  # Extract time series
  y <- data[[dep_var]]

  # Fit AR model
  if (!is.null(order_max)) {
    model <- ar(y, method = method, order.max = order_max)
  } else {
    model <- ar(y, method = method)
  }

  # Extract AR coefficients
  ar_coefs <- as.numeric(model$ar)
  ar_names <- paste0("ar", seq_along(ar_coefs))

  list(
    order = model$order,
    ar_coefficients = setNames(as.list(ar_coefs), ar_names),
    intercept = if (!is.null(model$x.intercept)) model$x.intercept else NA,
    innovation_variance = model$var.pred,
    aic = as.list(model$aic),
    method = method,
    n_obs = length(y)
  )
}
