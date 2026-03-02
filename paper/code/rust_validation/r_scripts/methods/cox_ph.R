# cox_ph.R - Cox proportional hazards model using survival package

suppressPackageStartupMessages({
  library(survival)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       event_var = NULL) {

  if (is.null(event_var)) {
    stop("event_var is required for Cox proportional hazards")
  }

  # Build survival formula
  formula <- as.formula(paste("Surv(", dep_var, ",", event_var, ") ~",
                              paste(indep_vars, collapse = " + ")))

  # Fit Cox PH model
  model <- coxph(formula, data = data)
  summary_model <- summary(model)

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  list(
    coefficients = setNames(as.list(coef_table[, "coef"]), coef_names),
    exp_coef = setNames(as.list(coef_table[, "exp(coef)"]), coef_names),
    std_errors = setNames(as.list(coef_table[, "se(coef)"]), coef_names),
    z_values = setNames(as.list(coef_table[, "z"]), coef_names),
    p_values = setNames(as.list(coef_table[, "Pr(>|z|)"]), coef_names),
    log_likelihood = as.numeric(model$loglik[2]),
    null_log_likelihood = as.numeric(model$loglik[1]),
    concordance = summary_model$concordance["C"],
    concordance_se = summary_model$concordance["se(C)"],
    likelihood_ratio_test = summary_model$logtest["test"],
    lr_p_value = summary_model$logtest["pvalue"],
    wald_test = summary_model$waldtest["test"],
    wald_p_value = summary_model$waldtest["pvalue"],
    score_test = summary_model$sctest["test"],
    score_p_value = summary_model$sctest["pvalue"],
    n_obs = model$n,
    n_events = model$nevent
  )
}
