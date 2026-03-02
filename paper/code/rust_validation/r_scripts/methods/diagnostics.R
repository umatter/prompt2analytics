# diagnostics.R - Regression diagnostics (JB test, BP test, DW test, VIF)

suppressPackageStartupMessages({
  library(lmtest)
  library(car)
  library(tseries)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Fit model
  model <- lm(formula, data = data)
  resid <- residuals(model)

  # Jarque-Bera test for normality of residuals
  jb <- jarque.bera.test(resid)

  # Breusch-Pagan test for heteroskedasticity
  bp <- bptest(model)

  # Durbin-Watson test for autocorrelation
  dw <- dwtest(model)

  # VIF (only if more than one predictor)
  vif_values <- NULL
  if (length(indep_vars) > 1) {
    vif_values <- vif(model)
  }

  result <- list(
    jb_statistic = as.numeric(jb$statistic),
    jb_p_value = jb$p.value,
    bp_statistic = as.numeric(bp$statistic),
    bp_p_value = bp$p.value,
    bp_df = as.numeric(bp$parameter),
    dw_statistic = as.numeric(dw$statistic),
    dw_p_value = dw$p.value,
    n_obs = nrow(data)
  )

  if (!is.null(vif_values)) {
    result$vif = setNames(as.list(vif_values), names(vif_values))
  }

  result
}
