# did.R - Difference-in-differences (simple 2x2 regression)

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       treat_var = NULL, post_var = NULL) {

  # Identify treat and post variables
  # By convention: indep_vars[1] = treat, indep_vars[2] = post
  if (is.null(treat_var)) treat_var <- indep_vars[1]
  if (is.null(post_var)) post_var <- indep_vars[2]

  # Build DiD formula: y ~ treat + post + treat:post
  formula <- as.formula(paste(dep_var, "~", treat_var, "+", post_var, "+",
                              treat_var, ":", post_var))

  # Fit DiD regression
  model <- lm(formula, data = data)
  summary_model <- summary(model)

  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  # Extract group means
  treat_col <- data[[treat_var]]
  post_col <- data[[post_var]]
  y_col <- data[[dep_var]]

  control_pre <- mean(y_col[treat_col == 0 & post_col == 0])
  control_post <- mean(y_col[treat_col == 0 & post_col == 1])
  treated_pre <- mean(y_col[treat_col == 1 & post_col == 0])
  treated_post <- mean(y_col[treat_col == 1 & post_col == 1])

  # Manual DiD estimate
  manual_did <- (treated_post - treated_pre) - (control_post - control_pre)

  # ATT is the interaction coefficient
  interaction_name <- paste0(treat_var, ":", post_var)
  att <- coef(model)[interaction_name]

  list(
    coefficients = setNames(as.list(coef_table[, 1]), coef_names),
    std_errors = setNames(as.list(coef_table[, 2]), coef_names),
    t_values = setNames(as.list(coef_table[, 3]), coef_names),
    p_values = setNames(as.list(coef_table[, 4]), coef_names),
    att = as.numeric(att),
    manual_did = manual_did,
    control_pre = control_pre,
    control_post = control_post,
    treated_pre = treated_pre,
    treated_post = treated_post,
    r_squared = summary_model$r.squared,
    n_obs = nrow(data)
  )
}
