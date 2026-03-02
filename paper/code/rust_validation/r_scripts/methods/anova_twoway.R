# anova_twoway.R - Two-way ANOVA using stats::aov

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       factor_var = NULL, factor2_var = NULL, interaction = TRUE) {

  # Use factor_var/factor2_var if provided, otherwise indep_vars
  factor_a <- if (!is.null(factor_var)) factor_var else indep_vars[1]
  factor_b <- if (!is.null(factor2_var)) factor2_var else indep_vars[2]

  # Ensure grouping variables are factors
  data[[factor_a]] <- as.factor(data[[factor_a]])
  data[[factor_b]] <- as.factor(data[[factor_b]])

  # Build formula
  if (interaction) {
    formula <- as.formula(paste(dep_var, "~", factor_a, "*", factor_b))
  } else {
    formula <- as.formula(paste(dep_var, "~", factor_a, "+", factor_b))
  }

  # Fit two-way ANOVA
  model <- aov(formula, data = data)
  result <- summary(model)

  anova_table <- result[[1]]

  # Extract statistics for each factor
  interaction_name <- paste0(factor_a, ":", factor_b)

  result_list <- list(
    ss_a = anova_table[factor_a, "Sum Sq"],
    ss_b = anova_table[factor_b, "Sum Sq"],
    ss_error = anova_table["Residuals", "Sum Sq"],
    df_a = anova_table[factor_a, "Df"],
    df_b = anova_table[factor_b, "Df"],
    df_error = anova_table["Residuals", "Df"],
    ms_a = anova_table[factor_a, "Mean Sq"],
    ms_b = anova_table[factor_b, "Mean Sq"],
    ms_error = anova_table["Residuals", "Mean Sq"],
    f_a = anova_table[factor_a, "F value"],
    f_b = anova_table[factor_b, "F value"],
    p_a = anova_table[factor_a, "Pr(>F)"],
    p_b = anova_table[factor_b, "Pr(>F)"],
    n_obs = nrow(data)
  )

  if (interaction && interaction_name %in% rownames(anova_table)) {
    result_list$ss_ab <- anova_table[interaction_name, "Sum Sq"]
    result_list$df_ab <- anova_table[interaction_name, "Df"]
    result_list$ms_ab <- anova_table[interaction_name, "Mean Sq"]
    result_list$f_ab <- anova_table[interaction_name, "F value"]
    result_list$p_ab <- anova_table[interaction_name, "Pr(>F)"]
  }

  result_list
}
