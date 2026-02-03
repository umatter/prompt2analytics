# probit.R - Probit regression using stats::glm

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit probit model
  model <- glm(formula, data = data, family = binomial(link = "probit"))
  summary_model <- summary(model)

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    z_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    log_likelihood = as.numeric(logLik(model)),
    aic = AIC(model),
    bic = BIC(model),
    null_deviance = model$null.deviance,
    residual_deviance = model$deviance,
    df_null = model$df.null,
    df_residual = model$df.residual,
    n_obs = nrow(data)
  )
}
