# Validation script for Natural Effect Models (medflex)
# Compares Rust implementation against R medflex package

library(medflex)
library(boot)

set.seed(42)

# =============================================================================
# Test Case 1: Simple Mediation (no interaction)
# =============================================================================

cat("\n=== Test Case 1: Simple Mediation ===\n")

n <- 200
# Confounders
x1 <- rnorm(n)
x2 <- rnorm(n)

# Treatment (binary)
a <- rbinom(n, 1, plogis(0.3*x1 + 0.2*x2))

# Mediator: M = 0.5 + 0.6*A + 0.3*X1 + noise
m <- 0.5 + 0.6*a + 0.3*x1 + rnorm(n, sd=0.4)

# Outcome: Y = 1.0 + 0.4*A + 0.5*M + 0.3*X1 + 0.2*X2 + noise
# No interaction: beta_3 = 0
y <- 1.0 + 0.4*a + 0.5*m + 0.3*x1 + 0.2*x2 + rnorm(n, sd=0.5)

data1 <- data.frame(y=y, a=a, m=m, x1=x1, x2=x2)

# Product of coefficients method (no interaction)
# Mediator model
med_model <- lm(m ~ a + x1 + x2, data=data1)
alpha_1 <- coef(med_model)["a"]

# Outcome model (no interaction)
out_model <- lm(y ~ a + m + x1 + x2, data=data1)
beta_1 <- coef(out_model)["a"]
beta_2 <- coef(out_model)["m"]

# Effects
nde_simple <- beta_1
nie_simple <- alpha_1 * beta_2
te_simple <- nde_simple + nie_simple

cat("Product method (no interaction):\n")
cat(sprintf("  alpha_1 (A->M): %.4f\n", alpha_1))
cat(sprintf("  beta_1 (A->Y):  %.4f\n", beta_1))
cat(sprintf("  beta_2 (M->Y):  %.4f\n", beta_2))
cat(sprintf("  NDE = beta_1:   %.4f\n", nde_simple))
cat(sprintf("  NIE = a1*b2:    %.4f\n", nie_simple))
cat(sprintf("  TE = NDE+NIE:   %.4f\n", te_simple))

# =============================================================================
# Test Case 2: Mediation with Interaction
# =============================================================================

cat("\n=== Test Case 2: Mediation with Interaction ===\n")

# Same data but with interaction
# Y = 1.0 + 0.4*A + 0.5*M + 0.2*A*M + 0.3*X1 + 0.2*X2 + noise
y_int <- 1.0 + 0.4*a + 0.5*m + 0.2*a*m + 0.3*x1 + 0.2*x2 + rnorm(n, sd=0.5)

data2 <- data.frame(y=y_int, a=a, m=m, x1=x1, x2=x2)

# Outcome model with interaction
out_model_int <- lm(y ~ a + m + a:m + x1 + x2, data=data2)
beta_1_int <- coef(out_model_int)["a"]
beta_2_int <- coef(out_model_int)["m"]
beta_3_int <- coef(out_model_int)["a:m"]

# E[M|A=0]
m_mean_control <- mean(m[a==0])

# VanderWeele formulas with interaction
nde_int <- beta_1_int + beta_3_int * m_mean_control
nie_int <- alpha_1 * (beta_2_int + beta_3_int)
te_int <- nde_int + nie_int

cat("With interaction:\n")
cat(sprintf("  alpha_1 (A->M):     %.4f\n", alpha_1))
cat(sprintf("  beta_1 (A->Y):      %.4f\n", beta_1_int))
cat(sprintf("  beta_2 (M->Y):      %.4f\n", beta_2_int))
cat(sprintf("  beta_3 (A*M->Y):    %.4f\n", beta_3_int))
cat(sprintf("  E[M|A=0]:           %.4f\n", m_mean_control))
cat(sprintf("  NDE:                %.4f\n", nde_int))
cat(sprintf("  NIE:                %.4f\n", nie_int))
cat(sprintf("  TE:                 %.4f\n", te_int))

# =============================================================================
# Test Case 3: Using medflex package
# =============================================================================

cat("\n=== Test Case 3: medflex Package ===\n")

# Create expanded dataset for medflex
tryCatch({
  # Note: medflex uses a different parameterization
  # We use neWeight for weighting-based approach
  expData <- neWeight(a ~ x1 + x2, data = data2)

  # Fit NEM
  neMod <- neModel(y ~ a0 + a1 + x1 + x2, expData = expData, se = "robust")

  cat("medflex neModel summary:\n")
  print(summary(neMod))

  # Effect decomposition
  cat("\nEffect decomposition:\n")
  print(neEffdecomp(neMod))

}, error = function(e) {
  cat("medflex package analysis failed (may need additional setup):\n")
  cat(e$message, "\n")
})

# =============================================================================
# Bootstrap for SEs (manual implementation matching Rust)
# =============================================================================

cat("\n=== Bootstrap Standard Errors ===\n")

compute_effects <- function(data, indices, interaction=TRUE) {
  d <- data[indices, ]

  # Mediator model
  med_mod <- lm(m ~ a + x1 + x2, data=d)
  a1 <- coef(med_mod)["a"]

  if (interaction) {
    # Outcome with interaction
    out_mod <- lm(y ~ a + m + a:m + x1 + x2, data=d)
    b1 <- coef(out_mod)["a"]
    b2 <- coef(out_mod)["m"]
    b3 <- coef(out_mod)["a:m"]

    m_ctrl <- mean(d$m[d$a==0])

    nde <- b1 + b3 * m_ctrl
    nie <- a1 * (b2 + b3)
  } else {
    out_mod <- lm(y ~ a + m + x1 + x2, data=d)
    b1 <- coef(out_mod)["a"]
    b2 <- coef(out_mod)["m"]

    nde <- b1
    nie <- a1 * b2
  }

  te <- nde + nie
  return(c(nde=nde, nie=nie, te=te))
}

# Bootstrap with interaction
boot_int <- boot(data2, compute_effects, R=1000, interaction=TRUE)

cat("Bootstrap results (with interaction):\n")
cat(sprintf("  NDE: %.4f (SE: %.4f)\n", mean(boot_int$t[,1]), sd(boot_int$t[,1])))
cat(sprintf("  NIE: %.4f (SE: %.4f)\n", mean(boot_int$t[,2]), sd(boot_int$t[,2])))
cat(sprintf("  TE:  %.4f (SE: %.4f)\n", mean(boot_int$t[,3]), sd(boot_int$t[,3])))

# 95% CIs
cat("\n95% Confidence Intervals (percentile):\n")
cat(sprintf("  NDE: [%.4f, %.4f]\n",
            quantile(boot_int$t[,1], 0.025),
            quantile(boot_int$t[,1], 0.975)))
cat(sprintf("  NIE: [%.4f, %.4f]\n",
            quantile(boot_int$t[,2], 0.025),
            quantile(boot_int$t[,2], 0.975)))
cat(sprintf("  TE:  [%.4f, %.4f]\n",
            quantile(boot_int$t[,3], 0.025),
            quantile(boot_int$t[,3], 0.975)))

# =============================================================================
# Export data for Rust validation
# =============================================================================

cat("\n=== Exporting Data for Rust Validation ===\n")

# Export test data
write.csv(data2, "validation/expected/medflex_test_data.csv", row.names=FALSE)

# Export expected results
expected <- data.frame(
  statistic = c("nde", "nie", "te", "nde_se", "nie_se", "te_se",
                "alpha_1", "beta_1", "beta_2", "beta_3", "m_mean_control"),
  value = c(nde_int, nie_int, te_int,
            sd(boot_int$t[,1]), sd(boot_int$t[,2]), sd(boot_int$t[,3]),
            alpha_1, beta_1_int, beta_2_int, beta_3_int, m_mean_control)
)
write.csv(expected, "validation/expected/medflex_expected.csv", row.names=FALSE)

cat("Data exported to validation/expected/medflex_*.csv\n")

cat("\n=== Validation Complete ===\n")
