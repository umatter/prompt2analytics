# var_model.R - Vector Autoregression using vars::VAR

suppressPackageStartupMessages({
  library(vars)
})

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       p = 1, type = "const") {

  # Select columns for VAR model
  # dep_var and indep_vars together form the multivariate time series
  if (!is.null(indep_vars)) {
    var_cols <- c(dep_var, indep_vars)
  } else {
    # Use all numeric columns
    var_cols <- names(data)[sapply(data, is.numeric)]
  }

  var_data <- data[, var_cols, drop = FALSE]

  # Fit VAR model
  model <- VAR(var_data, p = p, type = type)

  # Extract coefficients for each equation
  equations <- list()
  for (eq_name in names(coef(model))) {
    eq_coefs <- coef(model)[[eq_name]]
    equations[[eq_name]] <- list(
      coefficients = setNames(as.list(eq_coefs[, "Estimate"]), rownames(eq_coefs)),
      std_errors = setNames(as.list(eq_coefs[, "Std. Error"]), rownames(eq_coefs)),
      t_values = setNames(as.list(eq_coefs[, "t value"]), rownames(eq_coefs)),
      p_values = setNames(as.list(eq_coefs[, "Pr(>|t|)"]), rownames(eq_coefs))
    )
  }

  # Model selection criteria
  aic_val <- AIC(model)
  bic_val <- BIC(model)

  list(
    equations = equations,
    p = p,
    type = type,
    aic = aic_val,
    bic = bic_val,
    n_vars = length(var_cols),
    n_obs = nrow(var_data) - p
  )
}
