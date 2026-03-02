# gls_regression.R - Generalized Least Squares using nlme::gls with corAR1

suppressPackageStartupMessages({
  library(nlme)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       correlation_type = "ar1") {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Determine time ordering variable
  if (!is.null(time_var)) {
    order_var <- time_var
  } else {
    # Use row ordering
    data$.row_order <- seq_len(nrow(data))
    order_var <- ".row_order"
  }

  # Fit GLS with AR(1) correlation structure
  if (correlation_type == "ar1") {
    cor_struct <- corAR1(form = as.formula(paste("~ ", order_var)))
    model <- gls(formula, data = data, correlation = cor_struct)
  } else {
    # No correlation structure (equivalent to OLS)
    model <- gls(formula, data = data)
  }

  summary_model <- summary(model)
  coef_table <- summary_model$tTable
  coef_names <- rownames(coef_table)

  result <- list(
    coefficients = setNames(as.list(coef_table[, "Value"]), coef_names),
    std_errors = setNames(as.list(coef_table[, "Std.Error"]), coef_names),
    t_values = setNames(as.list(coef_table[, "t-value"]), coef_names),
    p_values = setNames(as.list(coef_table[, "p-value"]), coef_names),
    sigma = model$sigma,
    log_likelihood = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    n_obs = nrow(data)
  )

  # Extract AR(1) parameter if applicable
  if (correlation_type == "ar1" && !is.null(model$modelStruct$corStruct)) {
    rho <- coef(model$modelStruct$corStruct, unconstrained = FALSE)
    result$rho = as.numeric(rho)
  }

  result
}
