# doubly_robust.R - Doubly robust estimation (AIPW)

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       treat_var = NULL) {

  # treat_var is the treatment indicator column
  if (is.null(treat_var)) treat_var <- indep_vars[1]

  # Covariates are the remaining indep_vars (exclude treat_var)
  covariates <- setdiff(indep_vars, treat_var)
  if (length(covariates) == 0) {
    stop("At least one covariate is required for doubly robust estimation")
  }

  treatment <- data[[treat_var]]
  outcome <- data[[dep_var]]

  # Step 1: Estimate propensity scores
  ps_formula <- as.formula(paste(treat_var, "~", paste(covariates, collapse = " + ")))
  ps_model <- glm(ps_formula, data = data, family = binomial(link = "logit"))
  ps_hat <- predict(ps_model, type = "response")

  # Step 2: Estimate outcome models for each treatment group
  out_formula <- as.formula(paste(dep_var, "~", paste(covariates, collapse = " + ")))

  out_model_1 <- lm(out_formula, data = data[treatment == 1, ])
  out_model_0 <- lm(out_formula, data = data[treatment == 0, ])

  # Predict counterfactuals for all observations
  mu1_hat <- predict(out_model_1, newdata = data)
  mu0_hat <- predict(out_model_0, newdata = data)

  # Step 3: AIPW estimator
  aipw_1 <- mean(treatment * (outcome - mu1_hat) / ps_hat + mu1_hat)
  aipw_0 <- mean((1 - treatment) * (outcome - mu0_hat) / (1 - ps_hat) + mu0_hat)
  ate_aipw <- aipw_1 - aipw_0

  # Influence function for standard error
  phi <- (treatment * (outcome - mu1_hat) / ps_hat + mu1_hat) -
         ((1 - treatment) * (outcome - mu0_hat) / (1 - ps_hat) + mu0_hat) - ate_aipw
  se_aipw <- sqrt(mean(phi^2) / length(phi))

  # Simple IPW for comparison
  w1 <- treatment / ps_hat
  w0 <- (1 - treatment) / (1 - ps_hat)
  ate_ipw <- mean(w1 * outcome) - mean(w0 * outcome)

  # Simple outcome regression for comparison
  ate_reg <- mean(mu1_hat) - mean(mu0_hat)

  list(
    ate_aipw = ate_aipw,
    se_aipw = se_aipw,
    ate_ipw = ate_ipw,
    ate_regression = ate_reg,
    mean_mu1 = mean(mu1_hat),
    mean_mu0 = mean(mu0_hat),
    ps_mean = mean(ps_hat),
    n_obs = nrow(data)
  )
}
