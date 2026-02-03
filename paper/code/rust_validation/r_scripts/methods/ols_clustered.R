# ols_clustered.R - OLS with clustered standard errors using sandwich

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  if (is.null(cluster_var)) {
    stop("cluster_var is required for clustered standard errors")
  }

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit model
  model <- lm(formula, data = data)

  # Compute clustered standard errors
  cluster_vcov <- vcovCL(model, cluster = data[[cluster_var]], type = "HC1")
  robust_test <- coeftest(model, vcov = cluster_vcov)

  coef_names <- rownames(robust_test)

  list(
    coefficients = setNames(as.list(robust_test[, 1]), coef_names),
    std_errors = setNames(as.list(robust_test[, 2]), coef_names),
    t_values = setNames(as.list(robust_test[, 3]), coef_names),
    p_values = setNames(as.list(robust_test[, 4]), coef_names),
    cluster_var = cluster_var,
    n_clusters = length(unique(data[[cluster_var]])),
    n_obs = nrow(data)
  )
}
