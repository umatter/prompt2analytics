# panel_hdfe.R - High-Dimensional Fixed Effects using fixest

suppressPackageStartupMessages({
  library(fixest)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  if (is.null(entity_var)) {
    stop("entity_var is required for HDFE")
  }

  # Build formula with fixed effects
  fe_vars <- entity_var
  if (!is.null(time_var)) {
    fe_vars <- c(fe_vars, time_var)
  }

  formula_str <- paste(dep_var, "~", paste(indep_vars, collapse = " + "), "|",
                       paste(fe_vars, collapse = " + "))
  formula <- as.formula(formula_str)

  # Fit HDFE model
  model <- feols(formula, data = data)
  summary_model <- summary(model)

  coef_table <- summary_model$coeftable
  coef_names <- rownames(coef_table)

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    r_squared = fitstat(model, "r2"),
    adj_r_squared = fitstat(model, "ar2"),
    within_r_squared = fitstat(model, "wr2"),
    fe_vars = fe_vars,
    n_obs = nrow(data)
  )
}
