# anova_oneway.R - One-way ANOVA using stats::aov

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       factor_var = NULL) {

  # The grouping factor: use factor_var if provided, otherwise indep_vars[1]
  group_var <- if (!is.null(factor_var)) factor_var else indep_vars[1]

  # Ensure grouping variable is a factor
  data[[group_var]] <- as.factor(data[[group_var]])

  # Build formula
  formula <- as.formula(paste(dep_var, "~", group_var))

  # Fit one-way ANOVA
  model <- aov(formula, data = data)
  result <- summary(model)

  anova_table <- result[[1]]

  # Extract key statistics
  ss_between <- anova_table[group_var, "Sum Sq"]
  ss_within <- anova_table["Residuals", "Sum Sq"]
  ss_total <- ss_between + ss_within

  # Effect size (eta-squared)
  eta_squared <- ss_between / ss_total

  # Group means
  group_means <- aggregate(as.formula(paste(dep_var, "~", group_var)), data = data, mean)
  grand_mean <- mean(data[[dep_var]])

  list(
    ss_between = ss_between,
    ss_within = ss_within,
    ss_total = ss_total,
    df_between = anova_table[group_var, "Df"],
    df_within = anova_table["Residuals", "Df"],
    ms_between = anova_table[group_var, "Mean Sq"],
    ms_within = anova_table["Residuals", "Mean Sq"],
    f_statistic = anova_table[group_var, "F value"],
    p_value = anova_table[group_var, "Pr(>F)"],
    eta_squared = eta_squared,
    grand_mean = grand_mean,
    group_means = setNames(as.list(group_means[[dep_var]]),
                           as.character(group_means[[group_var]])),
    n_groups = nlevels(data[[group_var]]),
    n_obs = nrow(data)
  )
}
