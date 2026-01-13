#!/usr/bin/env Rscript
# FEGLM Validation: R vs Rust Comparison
# This script generates reference values for validating the p2a Rust implementation
# Reference: alpaca package (Stammann 2018)

# Install packages if needed
if (!require("alpaca")) install.packages("alpaca")
if (!require("lfe")) install.packages("lfe")

library(alpaca)
library(lfe)

cat("FEGLM Validation\n")
cat("================\n")
cat("Reference implementation: alpaca (R)\n")
cat("Documentation: https://cran.r-project.org/package=alpaca\n\n")

# =============================================================================
# Test Case 1: Logit with Single Fixed Effect
# This matches crates/p2a-core/src/econometrics/feglm.rs::tests::test_feglm_logit_basic
# =============================================================================
run_test_logit_basic <- function() {
  cat("\n--- Test 1: Logit with Single Fixed Effect ---\n")

  # Use larger dataset to avoid separation issues
  set.seed(42)
  n <- 200
  id <- factor(rep(1:20, each = 10))
  x <- rnorm(n)

  # Generate with true coefficient ~1.0
  id_eff <- rnorm(20, sd = 0.5)[as.numeric(id)]
  latent <- 1.0 * x + id_eff
  prob <- 1 / (1 + exp(-latent))
  y <- rbinom(n, 1, prob)

  df <- data.frame(y = y, x = x, id = id)

  cat("Data summary:\n")
  cat(sprintf("  n = %d, n_id = %d, mean(y) = %.3f\n", n, 20, mean(y)))
  cat("  True coefficient: beta_x = 1.0\n\n")

  est <- feglm(y ~ x | id, data = df, family = binomial())

  cat("FEGLM Results:\n")
  print(summary(est))

  # Handle potential vcov issues
  se_x <- tryCatch({
    v <- vcov(est)
    if (!is.null(dim(v))) sqrt(v["x", "x"]) else NA
  }, error = function(e) NA)

  cat(sprintf("\nCoefficient (x): %.6f (true = 1.0)\n", coef(est)["x"]))
  if (!is.na(se_x)) cat(sprintf("Std. Error: %.6f\n", se_x))

  # Get deviance instead of loglik (alpaca doesn't export logLik method)
  cat(sprintf("Deviance: %.4f\n", est$deviance))

  list(
    test = "logit_basic",
    beta_x = coef(est)["x"],
    se_x = se_x,
    deviance = est$deviance,
    n_obs = nrow(df)
  )
}

# =============================================================================
# Test Case 2: Probit with Two-Way Fixed Effects
# =============================================================================
run_test_probit_twoway <- function() {
  cat("\n--- Test 2: Probit with Two-Way Fixed Effects ---\n")

  set.seed(42)
  n <- 100
  id <- factor(rep(1:10, each = 10))
  time <- factor(rep(1:10, times = 10))
  x <- rnorm(n)

  # Generate data from probit model
  id_eff <- rnorm(10, sd = 0.5)[as.numeric(id)]
  time_eff <- rnorm(10, sd = 0.5)[as.numeric(time)]
  latent <- 0.5 * x + id_eff + time_eff
  y <- as.integer(pnorm(latent) > runif(n))

  df <- data.frame(y = y, x = x, id = id, time = time)

  cat("Data summary:\n")
  cat(sprintf("  n = %d, n_id = %d, n_time = %d\n", n, length(unique(id)), length(unique(time))))
  cat(sprintf("  mean(y) = %.3f\n", mean(y)))
  cat("\n")

  est <- tryCatch({
    feglm(y ~ x | id + time, data = df, family = binomial("probit"))
  }, error = function(e) {
    cat("Error in estimation:", e$message, "\n")
    NULL
  })

  if (!is.null(est)) {
    cat("FEGLM Results:\n")
    print(summary(est))

    # Handle potential vcov issues when observations are deleted
    se_x <- tryCatch({
      v <- vcov(est)
      if (!is.null(dim(v))) sqrt(v["x", "x"]) else NA
    }, error = function(e) NA)

    cat(sprintf("\nCoefficient (x): %.6f (true = 0.5)\n", coef(est)["x"]))
    if (!is.na(se_x)) cat(sprintf("Std. Error: %.6f\n", se_x))

    list(
      test = "probit_twoway",
      beta_x = coef(est)["x"],
      se_x = se_x,
      converged = TRUE
    )
  } else {
    list(test = "probit_twoway", converged = FALSE)
  }
}

# =============================================================================
# Test Case 3: Poisson with Two-Way Fixed Effects (Gravity Model)
# =============================================================================
run_test_poisson_gravity <- function() {
  cat("\n--- Test 3: Poisson with Fixed Effects (Gravity) ---\n")

  df <- data.frame(
    y = c(5, 10, 3, 8, 12, 4, 6, 15, 2, 9, 11, 5),
    x = c(0.5, 1.0, 0.2, 0.8, 1.2, 0.3, 0.6, 1.5, 0.1, 0.9, 1.1, 0.5),
    exporter = factor(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
    importer = factor(c(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4))
  )

  cat("Data summary:\n")
  print(summary(df))
  cat("\n")

  est <- feglm(y ~ x | exporter + importer, data = df, family = poisson())

  cat("FEGLM Results:\n")
  print(summary(est))

  # Handle potential vcov issues
  se_x <- tryCatch({
    v <- vcov(est)
    if (!is.null(dim(v))) sqrt(v["x", "x"]) else NA
  }, error = function(e) NA)

  cat(sprintf("\nCoefficient (x): %.6f\n", coef(est)["x"]))
  if (!is.na(se_x)) cat(sprintf("Std. Error: %.6f\n", se_x))
  cat(sprintf("Deviance: %.6f\n", deviance(est)))

  list(
    test = "poisson_gravity",
    beta_x = coef(est)["x"],
    se_x = se_x,
    deviance = deviance(est)
  )
}

# =============================================================================
# Test Case 4: Linear HDFE Reference (for Gaussian FEGLM validation)
# Note: alpaca doesn't support Gaussian family; use lfe::felm as reference
# =============================================================================
run_test_gaussian_vs_hdfe <- function() {
  cat("\n--- Test 4: Linear HDFE Reference (for Gaussian FEGLM) ---\n")
  cat("Note: alpaca doesn't support Gaussian; using lfe::felm as reference\n")
  cat("Rust FEGLM with Gaussian should match felm results\n\n")

  set.seed(42)
  n <- 200
  id <- factor(rep(1:10, 20))
  firm <- factor(rep(1:5, 40))
  x1 <- rnorm(n)
  x2 <- rnorm(n)

  id_eff <- rnorm(10)[as.numeric(id)]
  firm_eff <- rnorm(5)[as.numeric(firm)]
  y <- 2.0 * x1 + 1.0 * x2 + id_eff + firm_eff + rnorm(n, sd = 0.5)

  df <- data.frame(y = y, x1 = x1, x2 = x2, id = id, firm = firm)

  cat("Data summary:\n")
  cat(sprintf("  n = %d, n_id = %d, n_firm = %d\n", n, 10, 5))
  cat("  True coefficients: beta_x1 = 2.0, beta_x2 = 1.0\n\n")

  # Linear HDFE reference values
  est_felm <- felm(y ~ x1 + x2 | id + firm, data = df)

  cat("FELM (Linear HDFE) Results:\n")
  print(summary(est_felm))

  cat("\nReference Values for Rust validation:\n")
  cat(sprintf("  beta_x1: %.6f (true = 2.0)\n", coef(est_felm)["x1"]))
  cat(sprintf("  beta_x2: %.6f (true = 1.0)\n", coef(est_felm)["x2"]))

  list(
    test = "gaussian_vs_hdfe",
    felm_x1 = coef(est_felm)["x1"],
    felm_x2 = coef(est_felm)["x2"],
    diff_x1 = abs(coef(est_felm)["x1"] - 2.0),
    diff_x2 = abs(coef(est_felm)["x2"] - 1.0)
  )
}

# =============================================================================
# Test Case 5: Coefficient Recovery (Logit)
# =============================================================================
run_test_coefficient_recovery <- function() {
  cat("\n--- Test 5: Coefficient Recovery (Logit) ---\n")

  set.seed(123)
  n <- 1000
  id <- factor(rep(1:50, 20))
  x1 <- rnorm(n)
  x2 <- rnorm(n)

  id_eff <- rnorm(50, sd = 0.5)[as.numeric(id)]
  latent <- 2.0 * x1 - 1.0 * x2 + id_eff
  prob <- 1 / (1 + exp(-latent))
  y <- rbinom(n, 1, prob)

  df <- data.frame(y = y, x1 = x1, x2 = x2, id = id)

  cat("Data summary:\n")
  cat(sprintf("  n = %d, n_id = %d\n", n, 50))
  cat(sprintf("  mean(y) = %.3f\n", mean(y)))
  cat("  True coefficients: beta_x1 = 2.0, beta_x2 = -1.0\n")
  cat("\n")

  est <- feglm(y ~ x1 + x2 | id, data = df, family = binomial())

  cat("FEGLM Results:\n")
  print(summary(est))

  cat("\nCoefficient Recovery:\n")
  cat(sprintf("  beta_x1: estimated = %.4f, true = 2.0, error = %.4f\n",
              coef(est)["x1"], coef(est)["x1"] - 2.0))
  cat(sprintf("  beta_x2: estimated = %.4f, true = -1.0, error = %.4f\n",
              coef(est)["x2"], coef(est)["x2"] + 1.0))

  list(
    test = "coefficient_recovery",
    beta_x1 = coef(est)["x1"],
    beta_x2 = coef(est)["x2"],
    error_x1 = coef(est)["x1"] - 2.0,
    error_x2 = coef(est)["x2"] + 1.0
  )
}

# =============================================================================
# Test Case 6: Large Panel with Many Fixed Effects
# =============================================================================
run_test_large_panel <- function() {
  cat("\n--- Test 6: Large Panel with Many Fixed Effects ---\n")

  set.seed(456)
  n_id <- 100
  n_time <- 20
  n <- n_id * n_time

  id <- factor(rep(1:n_id, each = n_time))
  time <- factor(rep(1:n_time, times = n_id))
  x <- rnorm(n)

  id_eff <- rnorm(n_id, sd = 0.5)[as.numeric(id)]
  time_eff <- rnorm(n_time, sd = 0.3)[as.numeric(time)]
  latent <- 1.5 * x + id_eff + time_eff
  prob <- 1 / (1 + exp(-latent))
  y <- rbinom(n, 1, prob)

  df <- data.frame(y = y, x = x, id = id, time = time)

  cat("Data summary:\n")
  cat(sprintf("  n = %d, n_id = %d, n_time = %d\n", n, n_id, n_time))
  cat(sprintf("  Total FE levels = %d\n", n_id + n_time))
  cat("  True coefficient: beta_x = 1.5\n")
  cat("\n")

  start_time <- Sys.time()
  est <- feglm(y ~ x | id + time, data = df, family = binomial())
  end_time <- Sys.time()

  cat("FEGLM Results:\n")
  print(summary(est))

  cat(sprintf("\nCoefficient (x): %.4f (true = 1.5)\n", coef(est)["x"]))
  cat(sprintf("Estimation time: %.3f seconds\n", as.numeric(end_time - start_time)))

  list(
    test = "large_panel",
    beta_x = coef(est)["x"],
    n_obs = n,
    n_fe_levels = n_id + n_time,
    time_seconds = as.numeric(end_time - start_time)
  )
}

# =============================================================================
# Run All Tests
# =============================================================================
results <- list()
results$logit_basic <- run_test_logit_basic()
results$probit_twoway <- run_test_probit_twoway()
results$poisson_gravity <- run_test_poisson_gravity()
results$gaussian_hdfe <- run_test_gaussian_vs_hdfe()
results$coefficient_recovery <- run_test_coefficient_recovery()
results$large_panel <- run_test_large_panel()

# =============================================================================
# Export Results for Rust Comparison
# =============================================================================
cat("\n\n=== Exporting Validation Data ===\n")

# Create validation output directory
dir.create("output", showWarnings = FALSE)

# Save summary
summary_df <- data.frame(
  test = c("logit_basic", "poisson_gravity", "coefficient_recovery", "coefficient_recovery", "large_panel"),
  metric = c("beta_x", "beta_x", "beta_x1", "beta_x2", "beta_x"),
  r_value = c(
    results$logit_basic$beta_x,
    results$poisson_gravity$beta_x,
    results$coefficient_recovery$beta_x1,
    results$coefficient_recovery$beta_x2,
    results$large_panel$beta_x
  ),
  expected = c(NA, NA, 2.0, -1.0, 1.5),
  tolerance = c(0.3, 0.3, 0.3, 0.3, 0.3)
)

write.csv(summary_df, "output/feglm_validation_results.csv", row.names = FALSE)
cat("Results saved to output/feglm_validation_results.csv\n")

# Print summary table
cat("\n=== Validation Summary ===\n")
cat("Test Case              | Metric     | R Value    | Expected   | Tolerance\n")
cat("-----------------------|------------|------------|------------|----------\n")
cat(sprintf("logit_basic            | beta_x     | %.6f   | ~1.0       | 0.3\n", results$logit_basic$beta_x))
if (results$probit_twoway$converged) {
  cat(sprintf("probit_twoway          | beta_x     | %.6f   | ~0.5       | 0.5\n", results$probit_twoway$beta_x))
} else {
  cat("probit_twoway          | beta_x     | N/A (separated data)\n")
}
cat(sprintf("poisson_gravity        | beta_x     | %.6f   | varies     | 0.3\n", results$poisson_gravity$beta_x))
cat(sprintf("gaussian_vs_hdfe       | diff_x1    | %.2e   | ~0.0       | 1e-5\n", results$gaussian_hdfe$diff_x1))
cat(sprintf("gaussian_vs_hdfe       | diff_x2    | %.2e   | ~0.0       | 1e-5\n", results$gaussian_hdfe$diff_x2))
cat(sprintf("coefficient_recovery   | beta_x1    | %.6f   | 2.0        | 0.3\n", results$coefficient_recovery$beta_x1))
cat(sprintf("coefficient_recovery   | beta_x2    | %.6f   | -1.0       | 0.3\n", results$coefficient_recovery$beta_x2))
cat(sprintf("large_panel            | beta_x     | %.6f   | 1.5        | 0.3\n", results$large_panel$beta_x))
cat(sprintf("large_panel            | time_s     | %.3f   | varies     | -\n", results$large_panel$time_seconds))

cat("\nValidation complete.\n")
