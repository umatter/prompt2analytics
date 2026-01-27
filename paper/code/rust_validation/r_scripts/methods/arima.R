# arima.R - ARIMA time series using forecast::Arima

suppressPackageStartupMessages({
  library(forecast)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = c(1, 1, 1), robust = NULL, seed = 42) {

  # Extract time series
  y <- data[[dep_var]]

  # Fit ARIMA model
  model <- Arima(y, order = arima_order)
  summary_model <- summary(model)

  # Extract coefficients
  coef <- model$coef
  coef_names <- names(coef)

  # Get standard errors from variance-covariance matrix
  se <- sqrt(diag(model$var.coef))

  list(
    coefficients = setNames(as.list(coef), coef_names),
    std_errors = setNames(as.list(se), coef_names),
    order = list(p = arima_order[1], d = arima_order[2], q = arima_order[3]),
    log_likelihood = model$loglik,
    aic = model$aic,
    aicc = model$aicc,
    bic = model$bic,
    sigma2 = model$sigma2,
    n_obs = length(y)
  )
}
