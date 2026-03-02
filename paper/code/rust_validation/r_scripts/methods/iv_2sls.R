# iv_2sls.R - Two-Stage Least Squares using AER::ivreg

suppressPackageStartupMessages({
  library(AER)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       endog_vars = NULL) {

  if (is.null(instrument_vars)) {
    stop("instrument_vars is required for 2SLS")
  }

  # Identify endogenous vs exogenous variables
  if (!is.null(endog_vars)) {
    all_regressors <- unique(c(endog_vars, indep_vars))
    exogenous_vars <- setdiff(all_regressors, endog_vars)
  } else {
    # Fallback: assume first variable in indep_vars is endogenous
    endog_vars <- indep_vars[1]
    all_regressors <- indep_vars
    exogenous_vars <- indep_vars[-1]
  }

  # Build IV formula: y ~ endog + exog | instruments + exog
  formula_str <- paste(dep_var, "~",
                       paste(all_regressors, collapse = " + "), "|",
                       paste(c(instrument_vars, exogenous_vars), collapse = " + "))
  formula <- as.formula(formula_str)

  # Fit 2SLS model
  model <- ivreg(formula, data = data)
  summary_model <- summary(model, diagnostics = TRUE)

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  # Extract diagnostic tests
  diag <- summary_model$diagnostics

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    r_squared = summary_model$r.squared,
    adj_r_squared = summary_model$adj.r.squared,
    residual_std_error = summary_model$sigma,
    weak_instruments_stat = if (!is.null(diag)) diag["Weak instruments", "statistic"] else NA,
    weak_instruments_p = if (!is.null(diag)) diag["Weak instruments", "p-value"] else NA,
    wu_hausman_stat = if (!is.null(diag)) diag["Wu-Hausman", "statistic"] else NA,
    wu_hausman_p = if (!is.null(diag)) diag["Wu-Hausman", "p-value"] else NA,
    sargan_stat = if (!is.null(diag) && "Sargan" %in% rownames(diag)) diag["Sargan", "statistic"] else NA,
    sargan_p = if (!is.null(diag) && "Sargan" %in% rownames(diag)) diag["Sargan", "p-value"] else NA,
    n_obs = nrow(data)
  )
}
