# panel_re.R - Panel Random Effects using plm

suppressPackageStartupMessages({
  library(plm)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  if (is.null(entity_var) || is.null(time_var)) {
    stop("entity_var and time_var are required for panel data")
  }

  # Build formula
  formula <- as.formula(paste(dep_var, "~", paste(indep_vars, collapse = " + ")))

  # Create panel data frame
  pdata <- pdata.frame(data, index = c(entity_var, time_var))

  # Fit random effects model
  model <- plm(formula, data = pdata, model = "random", effect = "individual")
  summary_model <- summary(model)

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  # Extract variance components
  ercomp <- model$ercomp

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    r_squared = summary_model$r.squared["rsq"],
    adj_r_squared = summary_model$r.squared["adjrsq"],
    sigma_entity = ercomp$sigma2["id"],
    sigma_error = ercomp$sigma2["idios"],
    theta = ercomp$theta,
    entity_var = entity_var,
    time_var = time_var,
    n_entities = length(unique(data[[entity_var]])),
    n_time_periods = length(unique(data[[time_var]])),
    n_obs = nrow(data)
  )
}
