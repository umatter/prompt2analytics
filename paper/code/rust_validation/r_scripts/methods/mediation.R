# mediation.R - Causal mediation analysis using mediation package

suppressPackageStartupMessages({
  library(mediation)
})

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       treat_var = NULL, mediator_var = NULL, sims = 500) {

  set.seed(seed)

  # treat_var is the treatment indicator
  if (is.null(treat_var)) treat_var <- indep_vars[1]
  # mediator_var is the mediating variable
  if (is.null(mediator_var)) mediator_var <- indep_vars[2]

  # Covariates are the remaining variables
  covariates <- setdiff(indep_vars, c(treat_var, mediator_var))

  # Step 1: Mediator model (mediator ~ treatment + covariates)
  if (length(covariates) > 0) {
    med_formula <- as.formula(paste(mediator_var, "~", treat_var, "+",
                                    paste(covariates, collapse = " + ")))
    out_formula <- as.formula(paste(dep_var, "~", treat_var, "+", mediator_var, "+",
                                    paste(covariates, collapse = " + ")))
  } else {
    med_formula <- as.formula(paste(mediator_var, "~", treat_var))
    out_formula <- as.formula(paste(dep_var, "~", treat_var, "+", mediator_var))
  }

  med_fit <- lm(med_formula, data = data)
  out_fit <- lm(out_formula, data = data)

  # Step 2: Run mediation analysis
  med_result <- mediate(med_fit, out_fit,
                        treat = treat_var,
                        mediator = mediator_var,
                        boot = TRUE,
                        boot.ci.type = "perc",
                        sims = sims)

  list(
    total_effect = med_result$tau.coef,
    direct_effect = med_result$z0,
    indirect_effect = med_result$d0,
    proportion_mediated = med_result$n0,
    se_total = sqrt(med_result$tau.var),
    se_direct = sqrt(med_result$z0.var),
    se_indirect = sqrt(med_result$d0.var),
    p_total = med_result$tau.p,
    p_direct = med_result$z0.p,
    p_indirect = med_result$d0.p,
    ci_total = c(med_result$tau.ci[1], med_result$tau.ci[2]),
    ci_direct = c(med_result$z0.ci[1], med_result$z0.ci[2]),
    ci_indirect = c(med_result$d0.ci[1], med_result$d0.ci[2]),
    n_obs = nrow(data)
  )
}
