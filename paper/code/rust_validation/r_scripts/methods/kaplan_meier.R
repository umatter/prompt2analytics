# kaplan_meier.R - Kaplan-Meier estimator using survival package

suppressPackageStartupMessages({
  library(survival)
})

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       event_var = NULL, group_var = NULL) {

  if (is.null(event_var)) {
    stop("event_var is required for Kaplan-Meier estimation")
  }

  # Create survival object
  surv_obj <- Surv(data[[dep_var]], data[[event_var]])

  # Fit Kaplan-Meier
  if (!is.null(group_var)) {
    formula <- as.formula(paste("surv_obj ~", group_var))
    km_fit <- survfit(formula, data = data)

    # Log-rank test
    logrank <- survdiff(formula, data = data)
    logrank_p <- 1 - pchisq(logrank$chisq, length(logrank$n) - 1)
  } else {
    km_fit <- survfit(surv_obj ~ 1, data = data)
    logrank <- NULL
    logrank_p <- NA
  }

  # Extract summary
  km_summary <- summary(km_fit)

  result <- list(
    n_obs = km_fit$n,
    n_events = sum(km_fit$n.event),
    median_survival = as.numeric(median(km_fit)),
    time = as.list(km_summary$time),
    n_risk = as.list(km_summary$n.risk),
    n_event = as.list(km_summary$n.event),
    survival = as.list(km_summary$surv),
    std_err = as.list(km_summary$std.err),
    lower_ci = as.list(km_summary$lower),
    upper_ci = as.list(km_summary$upper)
  )

  if (!is.null(logrank)) {
    result$logrank_chisq <- logrank$chisq
    result$logrank_p_value <- logrank_p
    result$logrank_df <- length(logrank$n) - 1
  }

  result
}
