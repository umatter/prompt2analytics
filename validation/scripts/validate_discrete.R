#!/usr/bin/env Rscript
#
# R Validation Script for Discrete Choice Models (Logit, Probit)
# Generates reference values for p2a-core validation tests
#

cat("=== Discrete Choice Validation Script ===\n\n")

# =============================================================================
# Test 1: Simple Logistic Regression
# =============================================================================
cat("--- Test 1: Simple Logit (P(Y=1) = Λ(-1 + 2x)) ---\n")

set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -1 + 2 * x
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data_logit <- data.frame(y = y, x = x)

logit_fit <- glm(y ~ x, data = data_logit, family = binomial(link = "logit"))

cat("\nLogit Coefficients:\n")
print(coef(logit_fit))

cat("\nLogit Standard Errors:\n")
print(sqrt(diag(vcov(logit_fit))))

cat("\nLogit Summary:\n")
print(summary(logit_fit))

cat("\nLog-Likelihood:", logLik(logit_fit), "\n")
cat("AIC:", AIC(logit_fit), "\n")
cat("BIC:", BIC(logit_fit), "\n")

# =============================================================================
# Test 2: Simple Probit Regression
# =============================================================================
cat("\n--- Test 2: Simple Probit (P(Y=1) = Φ(-0.5 + 1.5x)) ---\n")

set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -0.5 + 1.5 * x
prob <- pnorm(latent)
y <- rbinom(n, 1, prob)

data_probit <- data.frame(y = y, x = x)

probit_fit <- glm(y ~ x, data = data_probit, family = binomial(link = "probit"))

cat("\nProbit Coefficients:\n")
print(coef(probit_fit))

cat("\nProbit Standard Errors:\n")
print(sqrt(diag(vcov(probit_fit))))

cat("\nProbit Summary:\n")
print(summary(probit_fit))

cat("\nLog-Likelihood:", logLik(probit_fit), "\n")

# =============================================================================
# Test 3: Multiple Predictors - Logit
# =============================================================================
cat("\n--- Test 3: Multiple Predictors Logit ---\n")

set.seed(42)
n <- 1000

x1 <- rnorm(n)
x2 <- rnorm(n)
latent <- 0.5 + 1.5 * x1 - 0.8 * x2
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data_multi <- data.frame(y = y, x1 = x1, x2 = x2)

logit_multi <- glm(y ~ x1 + x2, data = data_multi, family = binomial)

cat("\nMultiple Predictor Logit Coefficients:\n")
print(coef(logit_multi))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(logit_multi))))

cat("\nTrue values: (0.5, 1.5, -0.8)\n")

# =============================================================================
# Test 4: Multiple Predictors - Probit
# =============================================================================
cat("\n--- Test 4: Multiple Predictors Probit ---\n")

set.seed(42)
n <- 1000

x1 <- rnorm(n)
x2 <- rnorm(n)
latent <- 0.3 + 1.0 * x1 - 0.5 * x2
prob <- pnorm(latent)
y <- rbinom(n, 1, prob)

data_multi_probit <- data.frame(y = y, x1 = x1, x2 = x2)

probit_multi <- glm(y ~ x1 + x2, data = data_multi_probit, family = binomial(link = "probit"))

cat("\nMultiple Predictor Probit Coefficients:\n")
print(coef(probit_multi))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(probit_multi))))

cat("\nTrue values: (0.3, 1.0, -0.5)\n")

# =============================================================================
# Test 5: Odds Ratios (Logit)
# =============================================================================
cat("\n--- Test 5: Odds Ratios ---\n")

set.seed(42)
n <- 500

x <- rnorm(n)
latent <- -0.5 + 1 * x  # OR for x should be e^1 ≈ 2.718
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data_or <- data.frame(y = y, x = x)

logit_or <- glm(y ~ x, data = data_or, family = binomial)

cat("\nLogit Coefficients:\n")
print(coef(logit_or))

cat("\nOdds Ratios (exp(coefficients)):\n")
print(exp(coef(logit_or)))

cat("\nExpected OR for x: e^1 ≈", exp(1), "\n")

# =============================================================================
# Test 6: Marginal Effects
# =============================================================================
cat("\n--- Test 6: Marginal Effects at Mean ---\n")

# Using data_logit from Test 1
# Marginal effect = dP/dx = β * P(1-P)

# Average predicted probability
avg_prob <- mean(predict(logit_fit, type = "response"))
cat("Average predicted probability:", avg_prob, "\n")

# Average marginal effect (AME)
# For logit: AME ≈ β * P(1-P) averaged over observations
probs <- predict(logit_fit, type = "response")
ame <- mean(coef(logit_fit)["x"] * probs * (1 - probs))
cat("Average Marginal Effect (AME):", ame, "\n")

# Marginal effect at mean (MEM)
mean_x <- mean(x)
pred_at_mean <- predict(logit_fit, newdata = data.frame(x = mean_x), type = "response")
mem <- coef(logit_fit)["x"] * pred_at_mean * (1 - pred_at_mean)
cat("Marginal Effect at Mean (MEM):", mem, "\n")

# =============================================================================
# Test 7: Model Fit Statistics
# =============================================================================
cat("\n--- Test 7: Model Fit Statistics ---\n")

# McFadden R-squared
null_model <- glm(y ~ 1, data = data_logit, family = binomial)
mcfadden_r2 <- 1 - logLik(logit_fit) / logLik(null_model)
cat("McFadden R-squared:", mcfadden_r2, "\n")

# Confusion matrix (at threshold 0.5)
pred_class <- ifelse(predict(logit_fit, type = "response") > 0.5, 1, 0)
accuracy <- mean(pred_class == data_logit$y)
cat("Accuracy (at 0.5 threshold):", accuracy, "\n")

# =============================================================================
# Test 8: Perfect Separation (Edge Case)
# =============================================================================
cat("\n--- Test 8: Perfect Separation (Edge Case) ---\n")

# This should produce warnings about fitted probabilities
data_sep <- data.frame(
  y = c(rep(0, 10), rep(1, 10)),
  x = c(rep(-1, 10), rep(1, 10))
)

# Try fitting - should warn
tryCatch({
  sep_fit <- glm(y ~ x, data = data_sep, family = binomial)
  cat("\nCoefficients (may be extreme):\n")
  print(coef(sep_fit))
  cat("\nNote: With perfect separation, coefficients diverge to +/- infinity\n")
}, warning = function(w) {
  cat("\nWarning detected:", conditionMessage(w), "\n")
})

# =============================================================================
# Test 9: Large Sample for Precision Testing
# =============================================================================
cat("\n--- Test 9: Large Sample Precision Test ---\n")

set.seed(42)
n <- 5000

x <- rnorm(n)
latent <- -0.5 + 1.0 * x
prob <- 1 / (1 + exp(-latent))
y <- rbinom(n, 1, prob)

data_large <- data.frame(y = y, x = x)

logit_large <- glm(y ~ x, data = data_large, family = binomial)

cat("\nLarge Sample Logit (n=5000):\n")
cat("True: (-0.5, 1.0)\n")
cat("Estimated:", coef(logit_large), "\n")
cat("Std Errors:", sqrt(diag(vcov(logit_large))), "\n")

# With large n, estimates should be very close to true values
cat("Difference from true: (", coef(logit_large)[1] - (-0.5), ",",
    coef(logit_large)[2] - 1.0, ")\n")

# =============================================================================
# Write expected values to CSV
# =============================================================================
cat("\n--- Writing expected values to CSV ---\n")

# Simple logit results
logit_simple <- data.frame(
  variable = names(coef(logit_fit)),
  coefficient = coef(logit_fit),
  std_error = sqrt(diag(vcov(logit_fit)))
)
write.csv(logit_simple, "validation/expected/logit_simple.csv", row.names = FALSE)

# Simple probit results
probit_simple <- data.frame(
  variable = names(coef(probit_fit)),
  coefficient = coef(probit_fit),
  std_error = sqrt(diag(vcov(probit_fit)))
)
write.csv(probit_simple, "validation/expected/probit_simple.csv", row.names = FALSE)

# Multiple predictor logit
logit_multiple <- data.frame(
  variable = names(coef(logit_multi)),
  coefficient = coef(logit_multi),
  std_error = sqrt(diag(vcov(logit_multi)))
)
write.csv(logit_multiple, "validation/expected/logit_multiple.csv", row.names = FALSE)

# Multiple predictor probit
probit_multiple <- data.frame(
  variable = names(coef(probit_multi)),
  coefficient = coef(probit_multi),
  std_error = sqrt(diag(vcov(probit_multi)))
)
write.csv(probit_multiple, "validation/expected/probit_multiple.csv", row.names = FALSE)

cat("\nExpected values written to validation/expected/\n")
cat("=== Discrete Choice Validation Script Complete ===\n")
