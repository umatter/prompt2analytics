# loess.R - Local polynomial regression (LOESS) using stats::loess

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       span = 0.75, degree = 2, family = "gaussian") {

  y <- data[[dep_var]]
  x <- data[[indep_vars[1]]]

  # Fit LOESS model
  model <- loess(y ~ x, span = span, degree = degree, family = family)

  # Extract results
  fitted_vals <- fitted(model)
  resid <- residuals(model)
  rss <- sum(resid^2)

  list(
    fitted = as.list(fitted_vals),
    residuals = as.list(resid),
    rss = rss,
    enp = model$enp,
    span = span,
    degree = degree,
    family = family,
    n_obs = length(y)
  )
}
