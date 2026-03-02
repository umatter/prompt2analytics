# hausman.R - Hausman test for FE vs RE using plm

suppressPackageStartupMessages({
  library(plm)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  if (is.null(entity_var) || is.null(time_var)) {
    stop("entity_var and time_var are required for Hausman test")
  }

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Create panel data frame
  pdata <- pdata.frame(data, index = c(entity_var, time_var))

  # Fit fixed effects model
  fe_model <- plm(formula, data = pdata, model = "within", effect = "individual")

  # Fit random effects model
  re_model <- plm(formula, data = pdata, model = "random", effect = "individual")

  # Hausman test
  ht <- phtest(fe_model, re_model)

  list(
    statistic = as.numeric(ht$statistic),
    p_value = ht$p.value,
    df = as.numeric(ht$parameter),
    method = ht$method,
    fe_coefficients = setNames(as.list(coef(fe_model)), names(coef(fe_model))),
    re_coefficients = setNames(as.list(coef(re_model)), names(coef(re_model))),
    n_obs = nrow(data)
  )
}
