#!/usr/bin/env Rscript
#
# R Validation Script for Difference-in-Differences (DiD)
# Generates reference values for p2a-core validation tests
#

cat("=== DiD Validation Script ===\n\n")

# =============================================================================
# Test 1: Classic 2x2 DiD - Known ATT
# =============================================================================
cat("--- Test 1: Classic 2x2 DiD (ATT = 5) ---\n")

set.seed(42)
n_per_group <- 100

# Create balanced 2x2 design
data <- expand.grid(
  id = 1:(2*n_per_group),
  time = c("pre", "post")
)

# Treatment assignment (first n_per_group are treated)
data$treat <- ifelse(data$id <= n_per_group, 1, 0)
data$post <- ifelse(data$time == "post", 1, 0)

# Outcome with known parameters:
# Baseline = 10
# Treatment group level effect = 2
# Time trend = 3
# Treatment effect (ATT) = 5
data$y <- 10 +                            # Baseline
          2 * data$treat +                # Treated group higher
          3 * data$post +                 # Time trend
          5 * data$treat * data$post +    # Treatment effect (ATT)
          rnorm(nrow(data), 0, 1)         # Noise

# DiD regression
did_fit <- lm(y ~ treat + post + treat:post, data = data)

cat("\nDiD Coefficients:\n")
print(coef(did_fit))

cat("\nDiD Standard Errors:\n")
print(sqrt(diag(vcov(did_fit))))

cat("\nDiD Summary:\n")
print(summary(did_fit))

# Group means
cat("\nGroup Means:\n")
control_pre <- mean(data$y[data$treat == 0 & data$post == 0])
control_post <- mean(data$y[data$treat == 0 & data$post == 1])
treat_pre <- mean(data$y[data$treat == 1 & data$post == 0])
treat_post <- mean(data$y[data$treat == 1 & data$post == 1])

cat("  Control Pre:", control_pre, "\n")
cat("  Control Post:", control_post, "\n")
cat("  Treated Pre:", treat_pre, "\n")
cat("  Treated Post:", treat_post, "\n")

# Manual DiD calculation
manual_did <- (treat_post - treat_pre) - (control_post - control_pre)
cat("\nManual DiD Estimate:", manual_did, "\n")
cat("Regression DiD Estimate (treat:post):", coef(did_fit)["treat:post"], "\n")

# =============================================================================
# Test 2: DiD with Covariates
# =============================================================================
cat("\n--- Test 2: DiD with Covariates ---\n")

set.seed(42)
n_per_group <- 200

data_cov <- expand.grid(
  id = 1:(2*n_per_group),
  time = c("pre", "post")
)

data_cov$treat <- ifelse(data_cov$id <= n_per_group, 1, 0)
data_cov$post <- ifelse(data_cov$time == "post", 1, 0)

# Add covariate
data_cov$x <- rnorm(nrow(data_cov))

# Outcome with covariate effect
data_cov$y <- 10 +
              2 * data_cov$treat +
              3 * data_cov$post +
              5 * data_cov$treat * data_cov$post +  # ATT = 5
              1.5 * data_cov$x +                     # Covariate effect
              rnorm(nrow(data_cov), 0, 1)

# DiD regression with covariate
did_cov <- lm(y ~ treat + post + treat:post + x, data = data_cov)

cat("\nDiD with Covariate Coefficients:\n")
print(coef(did_cov))

cat("\nATT (treat:post):", coef(did_cov)["treat:post"], "\n")
cat("Covariate effect (x):", coef(did_cov)["x"], "\n")

# =============================================================================
# Test 3: No Treatment Effect (Null Case)
# =============================================================================
cat("\n--- Test 3: No Treatment Effect (Null Case) ---\n")

set.seed(42)
n <- 400

data_null <- data.frame(
  treat = rep(c(0, 1), each = n/2),
  post = rep(c(0, 1), n/2),
  y = 10 + rnorm(n)  # Pure noise, no actual treatment effect
)

did_null <- lm(y ~ treat + post + treat:post, data = data_null)

cat("\nDiD Null Case Coefficients:\n")
print(coef(did_null))

cat("\nDiD Null Case Summary:\n")
print(summary(did_null))

cat("\nATT should not be significant (p > 0.05):\n")
cat("  ATT:", coef(did_null)["treat:post"], "\n")
cat("  p-value:", summary(did_null)$coefficients["treat:post", "Pr(>|t|)"], "\n")

# =============================================================================
# Test 4: DiD with Heterogeneous Treatment Effects
# =============================================================================
cat("\n--- Test 4: DiD with Heterogeneous Treatment Effects ---\n")

set.seed(42)
n_per_group <- 150

data_het <- expand.grid(
  id = 1:(2*n_per_group),
  time = c("pre", "post")
)

data_het$treat <- ifelse(data_het$id <= n_per_group, 1, 0)
data_het$post <- ifelse(data_het$time == "post", 1, 0)

# Individual-level treatment effect heterogeneity
# ATT varies by individual, average = 5
individual_att <- 5 + rnorm(n_per_group, 0, 2)
treatment_effect <- rep(0, nrow(data_het))
treatment_effect[data_het$treat == 1 & data_het$post == 1] <-
  rep(individual_att, length.out = sum(data_het$treat == 1 & data_het$post == 1))

data_het$y <- 10 +
              2 * data_het$treat +
              3 * data_het$post +
              treatment_effect +
              rnorm(nrow(data_het), 0, 1)

did_het <- lm(y ~ treat + post + treat:post, data = data_het)

cat("\nDiD with Heterogeneous Effects:\n")
print(coef(did_het))

cat("\nATT (average of heterogeneous effects):", coef(did_het)["treat:post"], "\n")
cat("True average ATT:", mean(individual_att), "\n")

# =============================================================================
# Test 5: Panel DiD (Repeated Cross-Sections Approximation)
# =============================================================================
cat("\n--- Test 5: Panel DiD Simulation ---\n")

set.seed(42)
n_entities <- 50
n_periods <- 4
treatment_period <- 3

# Create panel structure
panel_data <- expand.grid(
  entity = 1:n_entities,
  t = 1:n_periods
)

# Half are treated (after period 3)
panel_data$treat <- ifelse(panel_data$entity <= 25, 1, 0)
panel_data$post <- ifelse(panel_data$t >= treatment_period, 1, 0)

# Entity fixed effects
entity_fe <- rnorm(n_entities, 0, 2)

# Time fixed effects
time_fe <- c(0, 1, 2, 3)

# Outcome with panel structure
panel_data$y <- entity_fe[panel_data$entity] +
                time_fe[panel_data$t] +
                5 * panel_data$treat * panel_data$post +  # ATT = 5
                rnorm(nrow(panel_data), 0, 0.5)

# Simple DiD (ignoring panel structure)
did_panel <- lm(y ~ treat + post + treat:post, data = panel_data)

cat("\nSimple DiD on Panel Data:\n")
print(coef(did_panel))

# DiD with entity fixed effects
did_panel_fe <- lm(y ~ treat:post + factor(entity) + factor(t), data = panel_data)
cat("\nDiD with Entity and Time FE:\n")
cat("ATT:", coef(did_panel_fe)["treat:post"], "\n")

# =============================================================================
# Write expected values to CSV
# =============================================================================
cat("\n--- Writing expected values to CSV ---\n")

# Classic DiD results
did_classic <- data.frame(
  variable = names(coef(did_fit)),
  coefficient = coef(did_fit),
  std_error = sqrt(diag(vcov(did_fit)))
)
write.csv(did_classic, "validation/expected/did_classic_2x2.csv", row.names = FALSE)

# DiD with covariates
did_with_cov <- data.frame(
  variable = names(coef(did_cov)),
  coefficient = coef(did_cov),
  std_error = sqrt(diag(vcov(did_cov)))
)
write.csv(did_with_cov, "validation/expected/did_with_covariates.csv", row.names = FALSE)

# Group means for manual calculation
group_means <- data.frame(
  control_pre = control_pre,
  control_post = control_post,
  treated_pre = treat_pre,
  treated_post = treat_post,
  manual_did = manual_did
)
write.csv(group_means, "validation/expected/did_group_means.csv", row.names = FALSE)

cat("\nExpected values written to validation/expected/\n")
cat("=== DiD Validation Script Complete ===\n")
