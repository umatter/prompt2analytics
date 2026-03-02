# kruskal_wallis.R - Kruskal-Wallis test using stats::kruskal.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       factor_var = NULL) {

  # Use factor_var if provided, otherwise indep_vars[1]
  group_var <- if (!is.null(factor_var)) factor_var else indep_vars[1]

  # Ensure grouping variable is a factor
  data[[group_var]] <- as.factor(data[[group_var]])

  # Build formula
  formula <- as.formula(paste(dep_var, "~", group_var))

  # Run Kruskal-Wallis test
  result <- kruskal.test(formula, data = data)

  # Group medians and sizes
  group_medians <- aggregate(as.formula(paste(dep_var, "~", group_var)),
                             data = data, FUN = median)
  group_sizes <- table(data[[group_var]])

  list(
    statistic = as.numeric(result$statistic),
    df = as.numeric(result$parameter),
    p_value = result$p.value,
    group_medians = setNames(as.list(group_medians[[dep_var]]),
                             as.character(group_medians[[group_var]])),
    group_sizes = as.list(as.numeric(group_sizes)),
    n_groups = nlevels(data[[group_var]]),
    n_obs = nrow(data)
  )
}
