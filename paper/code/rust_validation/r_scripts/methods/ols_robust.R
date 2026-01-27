# ols_robust.R - OLS with robust standard errors (HC0, HC1, HC2, HC3) using sandwich

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = "HC1", seed = 42) {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit model
  model <- lm(formula, data = data)

  # Determine robust type (default to HC1)
  robust_type <- if (is.null(robust)) "HC1" else toupper(robust)

  # Compute robust standard errors
  robust_vcov <- vcovHC(model, type = robust_type)
  robust_test <- coeftest(model, vcov = robust_vcov)

  coef_names <- rownames(robust_test)

  list(
    coefficients = setNames(as.list(robust_test[, 1]), coef_names),
    std_errors = setNames(as.list(robust_test[, 2]), coef_names),
    t_values = setNames(as.list(robust_test[, 3]), coef_names),
    p_values = setNames(as.list(robust_test[, 4]), coef_names),
    robust_type = robust_type,
    n_obs = nrow(data)
  )
}
