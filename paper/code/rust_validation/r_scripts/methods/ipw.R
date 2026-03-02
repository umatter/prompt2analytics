# ipw.R - Inverse probability weighting

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       treat_var = NULL) {

  # treat_var is the treatment indicator column
  if (is.null(treat_var)) treat_var <- indep_vars[1]

  # Covariates are the remaining indep_vars (exclude treat_var)
  covariates <- setdiff(indep_vars, treat_var)
  if (length(covariates) == 0) {
    stop("At least one covariate is required for IPW besides the treatment variable")
  }

  # Step 1: Estimate propensity scores via logistic regression
  ps_formula <- as.formula(paste(treat_var, "~", paste(covariates, collapse = " + ")))
  ps_model <- glm(ps_formula, data = data, family = binomial(link = "logit"))
  ps_hat <- predict(ps_model, type = "response")

  treatment <- data[[treat_var]]
  outcome <- data[[dep_var]]

  # Step 2: Compute IPW weights
  w1 <- treatment / ps_hat
  w0 <- (1 - treatment) / (1 - ps_hat)

  # Horvitz-Thompson estimator
  ate_ht <- mean(w1 * outcome) - mean(w0 * outcome)

  # Hajek (normalized) estimator
  ate_hajek <- sum(w1 * outcome) / sum(w1) - sum(w0 * outcome) / sum(w0)

  # ATT estimator
  att <- mean(outcome[treatment == 1]) -
    sum((1 - treatment) * ps_hat / (1 - ps_hat) * outcome) /
    sum((1 - treatment) * ps_hat / (1 - ps_hat))

  list(
    ate_ht = ate_ht,
    ate_hajek = ate_hajek,
    att = att,
    ps_mean = mean(ps_hat),
    ps_sd = sd(ps_hat),
    ps_min = min(ps_hat),
    ps_max = max(ps_hat),
    n_treated = sum(treatment == 1),
    n_control = sum(treatment == 0),
    n_obs = nrow(data)
  )
}
