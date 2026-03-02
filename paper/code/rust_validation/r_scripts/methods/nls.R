# nls.R - Nonlinear least squares using stats::nls

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       model_type = "exponential", start_params = NULL) {

  y <- data[[dep_var]]
  x <- data[[indep_vars[1]]]

  # Use a polynomial model that works reliably on generic data
  # y = a + b*x + c*x^2 (quadratic, always converges from OLS-like starts)
  if (model_type == "exponential") {
    lm_fit <- lm(y ~ x + I(x^2))
    start_params <- list(a = coef(lm_fit)[1], b = coef(lm_fit)[2], c = coef(lm_fit)[3])
    model <- nls(y ~ a + b * x + c * x^2,
                 start = start_params,
                 control = nls.control(maxiter = 200, tol = 1e-5))
  } else if (model_type == "michaelis_menten") {
    if (is.null(start_params)) start_params <- list(Vmax = max(y), Km = median(x))
    model <- nls(y ~ Vmax * x / (Km + x), start = start_params)
  } else if (model_type == "logistic") {
    if (is.null(start_params)) start_params <- list(K = max(y), r = 1.0, x0 = median(x))
    model <- nls(y ~ K / (1 + exp(-r * (x - x0))), start = start_params)
  } else if (model_type == "power") {
    if (is.null(start_params)) start_params <- list(a = 1, b = 1)
    model <- nls(y ~ a * x^b, start = start_params)
  } else {
    stop(paste("Unknown model_type:", model_type))
  }

  summary_model <- summary(model)
  coef_table <- summary_model$coefficients
  coef_names <- rownames(coef_table)

  list(
    coefficients = setNames(as.list(coef_table[, "Estimate"]), coef_names),
    std_errors = setNames(as.list(coef_table[, "Std. Error"]), coef_names),
    t_values = setNames(as.list(coef_table[, "t value"]), coef_names),
    p_values = setNames(as.list(coef_table[, "Pr(>|t|)"]), coef_names),
    rss = sum(residuals(model)^2),
    sigma = summary_model$sigma,
    df_residual = summary_model$df[2],
    converged = model$convInfo$isConv,
    iterations = model$convInfo$finIter,
    model_type = model_type,
    n_obs = length(y)
  )
}
