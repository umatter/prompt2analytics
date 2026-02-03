# ols.R - OLS regression using stats::lm

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit model
  model <- lm(formula, data = data)
  summary_model <- summary(model)

  # Extract results
  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    r_squared = summary_model$r.squared,
    adj_r_squared = summary_model$adj.r.squared,
    f_statistic = as.numeric(summary_model$fstatistic[1]),
    f_statistic_df1 = as.numeric(summary_model$fstatistic[2]),
    f_statistic_df2 = as.numeric(summary_model$fstatistic[3]),
    residual_std_error = summary_model$sigma,
    df = summary_model$df[2],
    n_obs = nrow(data)
  )
}
