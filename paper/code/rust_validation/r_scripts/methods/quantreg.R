# quantreg.R - Quantile regression using quantreg::rq

suppressPackageStartupMessages({
  library(quantreg)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       tau = 0.5) {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit quantile regression
  model <- rq(formula, data = data, tau = tau)
  summary_model <- summary(model, se = "nid")

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  result <- list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    tau = tau,
    n_obs = nrow(data)
  )

  # If multiple quantiles requested as a vector
  if (length(tau) > 1) {
    multi_model <- rq(formula, data = data, tau = tau)
    result$multi_coefficients <- lapply(seq_along(tau), function(i) {
      setNames(as.list(coef(multi_model)[, i]), rownames(coef(multi_model)))
    })
    names(result$multi_coefficients) <- paste0("tau_", tau)
  }

  result
}
