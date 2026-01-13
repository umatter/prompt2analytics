#!/usr/bin/env Rscript
# Survival Analysis Validation: R vs Rust Comparison
# This script generates reference values for validating the p2a Rust implementation

library(survival)

cat("Survival Analysis Validation\n")
cat("============================\n")
cat("Reference implementation: survival (R) + cmprsk\n\n")

# =============================================================================
# Test Case 1: Kaplan-Meier Basic
# Matches crates/p2a-core/src/econometrics/survival.rs::tests::test_kaplan_meier_basic
# =============================================================================
run_test_km_basic <- function() {
  cat("\n--- Test 1: Kaplan-Meier Basic ---\n")

  # Simple dataset with censoring
  time <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  event <- c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1)  # 1=event, 0=censored

  surv_obj <- Surv(time, event)
  km_fit <- survfit(surv_obj ~ 1, conf.type = "log-log")

  # Extract results at event times only
  summ <- summary(km_fit)

  cat("Kaplan-Meier Results:\n")
  cat("Time | N.risk | N.event | Survival | SE\n")
  for (i in seq_along(summ$time)) {
    cat(sprintf("%4.0f | %6d | %7d | %8.6f | %8.6f\n",
                summ$time[i], summ$n.risk[i], summ$n.event[i],
                summ$surv[i], summ$std.err[i]))
  }

  # Median survival
  med <- median(km_fit)
  cat(sprintf("\nMedian survival: %.1f\n", med))

  list(
    test = "km_basic",
    times = summ$time,
    survival = summ$surv,
    std_err = summ$std.err,
    median = med
  )
}

# =============================================================================
# Test Case 2: Log-Rank Test
# Matches crates/p2a-core/src/econometrics/survival.rs::tests::test_log_rank
# =============================================================================
run_test_logrank <- function() {
  cat("\n--- Test 2: Log-Rank Test ---\n")

  # Two groups with different survival
  time <- c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9)
  event <- c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
  group <- c(rep(0, 6), rep(1, 6))

  surv_obj <- Surv(time, event)
  logrank <- survdiff(surv_obj ~ group)

  cat(sprintf("Chi-squared: %.6f\n", logrank$chisq))
  cat(sprintf("df: %d\n", 1))

  p_value <- 1 - pchisq(logrank$chisq, df = 1)
  cat(sprintf("p-value: %.6f\n", p_value))

  cat("\nExpected/Observed by group:\n")
  cat(sprintf("Group 0: Observed=%d, Expected=%.4f\n", logrank$obs[1], logrank$exp[1]))
  cat(sprintf("Group 1: Observed=%d, Expected=%.4f\n", logrank$obs[2], logrank$exp[2]))

  list(
    test = "logrank",
    chi_sq = logrank$chisq,
    df = 1,
    p_value = p_value,
    observed = logrank$obs,
    expected = logrank$exp
  )
}

# =============================================================================
# Test Case 3: Cox PH Basic
# Matches crates/p2a-core/src/econometrics/survival.rs::tests::test_cox_ph_basic
# =============================================================================
run_test_cox_basic <- function() {
  cat("\n--- Test 3: Cox PH Basic ---\n")

  # Data with clear covariate effect
  set.seed(42)
  n <- 50
  x <- rnorm(n, 0, 1)

  # Generate Weibull times with known effect
  # hazard proportional to exp(0.5 * x)
  shape <- 1.5
  scale <- 10
  u <- runif(n)
  time <- scale * (-log(u))^(1/shape) * exp(-0.5 * x / shape)

  # Light censoring
  censor_time <- runif(n, 5, 30)
  observed_time <- pmin(time, censor_time)
  event <- as.integer(time <= censor_time)

  cat(sprintf("Censoring rate: %.1f%%\n", 100 * (1 - mean(event))))

  # Fit Cox model (Efron ties - R default)
  cox_fit <- coxph(Surv(observed_time, event) ~ x, ties = "efron")
  summ <- summary(cox_fit)

  cat(sprintf("\nCoefficient (beta): %.6f\n", coef(cox_fit)))
  cat(sprintf("Standard error: %.6f\n", sqrt(diag(vcov(cox_fit)))))
  cat(sprintf("Hazard ratio: %.6f\n", exp(coef(cox_fit))))
  cat(sprintf("z-value: %.6f\n", summ$coefficients[, "z"]))
  cat(sprintf("p-value: %.6f\n", summ$coefficients[, "Pr(>|z|)"]))
  cat(sprintf("Concordance: %.6f (se=%.6f)\n", summ$concordance["C"], summ$concordance["se(C)"]))
  cat(sprintf("Log-likelihood: %.6f\n", cox_fit$loglik[2]))

  # Also Breslow for comparison
  cox_breslow <- coxph(Surv(observed_time, event) ~ x, ties = "breslow")
  cat(sprintf("\nBreslow coefficient: %.6f\n", coef(cox_breslow)))

  list(
    test = "cox_basic",
    coef = coef(cox_fit),
    se = sqrt(diag(vcov(cox_fit))),
    hr = exp(coef(cox_fit)),
    concordance = summ$concordance["C"],
    loglik = cox_fit$loglik[2],
    breslow_coef = coef(cox_breslow)
  )
}

# =============================================================================
# Test Case 4: Cox PH with Ties
# Validates tie-handling methods
# =============================================================================
run_test_cox_ties <- function() {
  cat("\n--- Test 4: Cox PH with Heavy Ties ---\n")

  # Data with many ties
  time <- c(1, 1, 2, 2, 2, 3, 4, 4, 5, 5)
  event <- c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1)
  x <- c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1)

  cox_efron <- coxph(Surv(time, event) ~ x, ties = "efron")
  cox_breslow <- coxph(Surv(time, event) ~ x, ties = "breslow")

  cat(sprintf("Efron: coef=%.6f, se=%.6f\n", coef(cox_efron), sqrt(diag(vcov(cox_efron)))))
  cat(sprintf("Breslow: coef=%.6f, se=%.6f\n", coef(cox_breslow), sqrt(diag(vcov(cox_breslow)))))

  list(
    test = "cox_ties",
    efron_coef = coef(cox_efron),
    efron_se = sqrt(diag(vcov(cox_efron))),
    breslow_coef = coef(cox_breslow),
    breslow_se = sqrt(diag(vcov(cox_breslow)))
  )
}

# =============================================================================
# Test Case 5: AFT Weibull
# Matches crates/p2a-core/src/econometrics/survival.rs::tests::test_aft_weibull
# =============================================================================
run_test_aft_weibull <- function() {
  cat("\n--- Test 5: AFT Weibull ---\n")

  set.seed(42)
  n <- 100
  x <- rnorm(n, 0, 1)

  # True model: log(T) = 2 + 0.5*x + sigma*epsilon
  true_intercept <- 2
  true_beta <- 0.5
  true_scale <- 0.5  # sigma in AFT parameterization

  # Generate Weibull times (AFT parameterization)
  epsilon <- -log(runif(n))  # Standard extreme value
  log_time <- true_intercept + true_beta * x + true_scale * log(epsilon)
  time <- exp(log_time)

  # Random censoring
  censor_time <- rexp(n, rate = 0.05)
  observed_time <- pmin(time, censor_time)
  event <- as.integer(time <= censor_time)

  cat(sprintf("Censoring rate: %.1f%%\n", 100 * (1 - mean(event))))

  # Fit AFT Weibull
  aft_fit <- survreg(Surv(observed_time, event) ~ x, dist = "weibull")

  cat(sprintf("\nIntercept: %.6f (true: %.1f)\n", coef(aft_fit)[1], true_intercept))
  cat(sprintf("Beta(x): %.6f (true: %.1f)\n", coef(aft_fit)[2], true_beta))
  cat(sprintf("Scale: %.6f (true: %.1f)\n", aft_fit$scale, true_scale))
  cat(sprintf("Log-likelihood: %.6f\n", aft_fit$loglik[2]))
  cat(sprintf("AIC: %.6f\n", AIC(aft_fit)))

  list(
    test = "aft_weibull",
    intercept = coef(aft_fit)[1],
    beta = coef(aft_fit)[2],
    scale = aft_fit$scale,
    loglik = aft_fit$loglik[2],
    aic = AIC(aft_fit)
  )
}

# =============================================================================
# Test Case 6: AFT Log-Normal
# =============================================================================
run_test_aft_lognormal <- function() {
  cat("\n--- Test 6: AFT Log-Normal ---\n")

  set.seed(42)
  n <- 100
  x <- rnorm(n, 0, 1)

  # True model: log(T) = 3 + 0.3*x + sigma*epsilon, epsilon ~ N(0,1)
  true_intercept <- 3
  true_beta <- 0.3
  true_sigma <- 0.8

  log_time <- true_intercept + true_beta * x + true_sigma * rnorm(n)
  time <- exp(log_time)

  # Random censoring
  censor_time <- rexp(n, rate = 0.03)
  observed_time <- pmin(time, censor_time)
  event <- as.integer(time <= censor_time)

  cat(sprintf("Censoring rate: %.1f%%\n", 100 * (1 - mean(event))))

  # Fit AFT Log-Normal
  aft_fit <- survreg(Surv(observed_time, event) ~ x, dist = "lognormal")

  cat(sprintf("\nIntercept: %.6f (true: %.1f)\n", coef(aft_fit)[1], true_intercept))
  cat(sprintf("Beta(x): %.6f (true: %.1f)\n", coef(aft_fit)[2], true_beta))
  cat(sprintf("Scale (sigma): %.6f (true: %.1f)\n", aft_fit$scale, true_sigma))
  cat(sprintf("Log-likelihood: %.6f\n", aft_fit$loglik[2]))

  list(
    test = "aft_lognormal",
    intercept = coef(aft_fit)[1],
    beta = coef(aft_fit)[2],
    scale = aft_fit$scale,
    loglik = aft_fit$loglik[2]
  )
}

# =============================================================================
# Test Case 7: Competing Risks
# Matches crates/p2a-core/src/econometrics/survival.rs::tests::test_competing_risks
# =============================================================================
run_test_competing_risks <- function() {
  cat("\n--- Test 7: Competing Risks (Aalen-Johansen) ---\n")

  # Using survival package's multi-state capability
  time <- c(1, 2, 3, 4, 5, 6, 7, 8)
  status <- c(1, 2, 0, 1, 2, 1, 0, 2)  # 0=censored, 1=type1, 2=type2

  # Fit cumulative incidence using survfit with multi-state
  fit <- survfit(Surv(time, factor(status)) ~ 1)

  cat("Aalen-Johansen CIF estimates:\n")
  print(summary(fit))

  # Extract CIF at specific times for validation
  # This gives P(event type j by time t)

  list(
    test = "competing_risks",
    times = fit$time,
    states = fit$states
  )
}

# =============================================================================
# Run All Tests
# =============================================================================
results <- list()
results$km_basic <- run_test_km_basic()
results$logrank <- run_test_logrank()
results$cox_basic <- run_test_cox_basic()
results$cox_ties <- run_test_cox_ties()
results$aft_weibull <- run_test_aft_weibull()
results$aft_lognormal <- run_test_aft_lognormal()
results$competing_risks <- run_test_competing_risks()

# =============================================================================
# Export Summary for Rust Tests
# =============================================================================
cat("\n\n=== Validation Summary for Rust Tests ===\n")
cat("Copy these expected values to survival.rs tests:\n\n")

cat("// Kaplan-Meier at event times\n")
cat(sprintf("// times: %s\n", paste(results$km_basic$times, collapse=", ")))
cat(sprintf("// survival: %s\n", paste(sprintf("%.6f", results$km_basic$survival), collapse=", ")))
cat(sprintf("// median: %.1f\n\n", results$km_basic$median))

cat("// Log-Rank Test\n")
cat(sprintf("// chi_sq: %.6f\n", results$logrank$chi_sq))
cat(sprintf("// p_value: %.6f\n\n", results$logrank$p_value))

cat("// Cox PH (Efron ties)\n")
cat(sprintf("// coef: %.6f, se: %.6f\n", results$cox_basic$coef, results$cox_basic$se))
cat(sprintf("// concordance: %.6f\n\n", results$cox_basic$concordance))

cat("// Cox PH with heavy ties\n")
cat(sprintf("// efron: coef=%.6f, se=%.6f\n", results$cox_ties$efron_coef, results$cox_ties$efron_se))
cat(sprintf("// breslow: coef=%.6f, se=%.6f\n\n", results$cox_ties$breslow_coef, results$cox_ties$breslow_se))

cat("// AFT Weibull\n")
cat(sprintf("// intercept: %.6f, beta: %.6f, scale: %.6f\n\n",
            results$aft_weibull$intercept, results$aft_weibull$beta, results$aft_weibull$scale))

cat("// AFT Log-Normal\n")
cat(sprintf("// intercept: %.6f, beta: %.6f, scale: %.6f\n",
            results$aft_lognormal$intercept, results$aft_lognormal$beta, results$aft_lognormal$scale))

# Save to CSV
dir.create("output", showWarnings = FALSE)
summary_df <- data.frame(
  test = c("km_basic", "logrank", "cox_basic", "cox_ties_efron", "cox_ties_breslow",
           "aft_weibull", "aft_lognormal"),
  metric = c("median_survival", "chi_sq", "concordance", "coef", "coef", "scale", "scale"),
  r_value = c(
    results$km_basic$median,
    results$logrank$chi_sq,
    results$cox_basic$concordance,
    results$cox_ties$efron_coef,
    results$cox_ties$breslow_coef,
    results$aft_weibull$scale,
    results$aft_lognormal$scale
  )
)
write.csv(summary_df, "output/survival_validation_results.csv", row.names = FALSE)
cat("\n\nResults saved to output/survival_validation_results.csv\n")
