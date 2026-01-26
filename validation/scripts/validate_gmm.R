#!/usr/bin/env Rscript
# GMM (Arellano-Bond / Blundell-Bond) Validation Script
# Compares R plm package results with p2a-core Rust implementation
#
# Reference: Arellano, M., & Bond, S. (1991). Some tests of specification for panel data.
#            Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions.

library(plm)

set.seed(42)

# ==============================================================================
# Test Case 1: Dynamic panel data with known DGP
# y_it = 0.5 * y_{i,t-1} + 1.5 * x_it + alpha_i + u_it
# ==============================================================================

cat("=== GMM Validation Test Cases ===\n\n")

# Generate panel data
n_entities <- 10
n_periods <- 8
n_obs <- n_entities * n_periods

# Entity effects
entity_effects <- c(0.0, 1.0, -0.5, 0.5, -1.0, 2.0, 0.3, -0.3, 0.8, -0.8)

# Generate data
data_list <- list()
for (i in 1:n_entities) {
  alpha <- entity_effects[i]
  y_prev <- 2.0 + alpha

  for (t in 1:n_periods) {
    x <- runif(1, 0.5, 3.0)
    noise <- rnorm(1, 0, 0.2)

    # True DGP: y = 0.5 * y_lag + 1.5 * x + alpha + noise
    y <- 0.5 * y_prev + 1.5 * x + alpha + noise

    data_list[[length(data_list) + 1]] <- data.frame(
      entity = paste0("E", i),
      time = t,
      x = x,
      y = y
    )

    y_prev <- y
  }
}

panel_data <- do.call(rbind, data_list)

# Convert to pdata.frame
pdata <- pdata.frame(panel_data, index = c("entity", "time"))

cat("Test 1: Arellano-Bond (Difference GMM) - One-Step\n")
cat("------------------------------------------------\n")
cat("True parameters: rho = 0.5, beta = 1.5\n\n")

# Arellano-Bond one-step
ab_onestep <- pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdata,
  effect = "individual",
  model = "onestep",
  transformation = "d"
)

cat("R plm one-step estimates:\n")
print(summary(ab_onestep))

cat("\n\nTest 2: Arellano-Bond (Difference GMM) - Two-Step\n")
cat("--------------------------------------------------\n")

# Arellano-Bond two-step
ab_twostep <- pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdata,
  effect = "individual",
  model = "twostep",
  transformation = "d"
)

cat("R plm two-step estimates:\n")
print(summary(ab_twostep))

cat("\n\nTest 3: System GMM (Blundell-Bond) - Two-Step\n")
cat("----------------------------------------------\n")

# System GMM
sys_gmm <- pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdata,
  effect = "individual",
  model = "twostep",
  transformation = "ld"  # System GMM uses level + difference
)

cat("R plm system GMM estimates:\n")
print(summary(sys_gmm))

# ==============================================================================
# Test Case 2: Wage dynamics (classic Arellano-Bond example)
# Using similar structure to EmplUK data
# ==============================================================================

cat("\n\n=== Test Case 2: Employment Dynamics ===\n\n")

# Simulate employment-type data
set.seed(123)
n_firms <- 50
n_years <- 7

firms <- list()
for (i in 1:n_firms) {
  firm_effect <- rnorm(1, 0, 0.5)
  emp_prev <- exp(rnorm(1, 4, 0.5))
  wage_prev <- exp(rnorm(1, 2, 0.3))

  for (t in 1:n_years) {
    wage <- wage_prev * exp(rnorm(1, 0.02, 0.1))
    capital <- exp(rnorm(1, 5, 0.5))

    # Employment dynamics: log(emp) = 0.7 * log(emp_lag) - 0.3 * log(wage) + 0.1 * log(capital)
    log_emp <- 0.7 * log(emp_prev) - 0.3 * log(wage) + 0.1 * log(capital) + firm_effect + rnorm(1, 0, 0.15)
    emp <- exp(log_emp)

    firms[[length(firms) + 1]] <- data.frame(
      firm = i,
      year = 1975 + t,
      emp = emp,
      wage = wage,
      capital = capital,
      log_emp = log_emp,
      log_wage = log(wage),
      log_capital = log(capital)
    )

    emp_prev <- emp
    wage_prev <- wage
  }
}

emp_data <- do.call(rbind, firms)
emp_pdata <- pdata.frame(emp_data, index = c("firm", "year"))

cat("Employment dynamics model: log(emp) ~ lag(log(emp)) + log(wage) + log(capital)\n")
cat("True parameters: rho = 0.7, beta_wage = -0.3, beta_capital = 0.1\n\n")

# Arellano-Bond two-step
emp_gmm <- pgmm(
  log_emp ~ lag(log_emp, 1) + log_wage + log_capital | lag(log_emp, 2:99),
  data = emp_pdata,
  effect = "individual",
  model = "twostep",
  transformation = "d"
)

cat("R plm Arellano-Bond estimates:\n")
print(summary(emp_gmm))

# ==============================================================================
# Save expected results for Rust comparison
# ==============================================================================

cat("\n\n=== Expected Values for Rust Validation ===\n\n")

# Extract coefficients and standard errors
ab_coefs <- coef(ab_twostep)
ab_se <- sqrt(diag(vcov(ab_twostep)))

cat("Test 1 (two-step) coefficients:\n")
cat(sprintf("  lag(y,1): %.6f (SE: %.6f)\n", ab_coefs[1], ab_se[1]))
cat(sprintf("  x:        %.6f (SE: %.6f)\n", ab_coefs[2], ab_se[2]))

cat("\nTest statistics:\n")
cat(sprintf("  Sargan test: stat=%.4f, df=%d\n",
    ab_twostep$sargan$statistic, ab_twostep$sargan$parameter))
cat(sprintf("  AR(1) test:  z=%.4f\n",
    ab_twostep$ar1$statistic))
cat(sprintf("  AR(2) test:  z=%.4f\n",
    ab_twostep$ar2$statistic))

# Write results to CSV for programmatic comparison
results_df <- data.frame(
  method = c("AB-twostep", "AB-twostep"),
  variable = c("lag(y,1)", "x"),
  coefficient = ab_coefs,
  std_error = ab_se
)

write.csv(results_df, "validation/expected/gmm_test1.csv", row.names = FALSE)

cat("\nResults saved to validation/expected/gmm_test1.csv\n")
