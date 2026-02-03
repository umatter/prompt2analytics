# Phase 4: Causal Inference - Design-Based Methods Validation
# Generates expected values for RD, Matching, IPW, TMLE, and related methods

library(rdrobust)
library(MatchIt)
library(WeightIt)

set.seed(42)

# ==============================================================================
# 1. Regression Discontinuity (Sharp RD)
# ==============================================================================
cat("=== Sharp RD (rdrobust) ===\n")

# Generate data with treatment effect at cutoff = 0
n <- 500
x <- runif(n, -1, 1)  # Running variable
treatment <- as.numeric(x >= 0)
# Outcome: linear trend + treatment effect of 0.5 at cutoff
y <- 0.5 + 0.3 * x + 0.5 * treatment + rnorm(n, 0, 0.2)

rd_result <- rdrobust(y, x, c = 0)
print(summary(rd_result))

# Extract key values
cat("\n--- Key Values for test_validate_rd_vs_r ---\n")
cat(sprintf("tau_conventional: %.6f\n", rd_result$coef[1]))
cat(sprintf("tau_bc: %.6f\n", rd_result$coef[2]))
cat(sprintf("tau_robust: %.6f\n", rd_result$coef[3]))
cat(sprintf("se_conventional: %.6f\n", rd_result$se[1]))
cat(sprintf("se_bc: %.6f\n", rd_result$se[2]))
cat(sprintf("se_robust: %.6f\n", rd_result$se[3]))
cat(sprintf("h_left: %.6f\n", rd_result$bws[1,1]))
cat(sprintf("h_right: %.6f\n", rd_result$bws[1,2]))
cat(sprintf("n_eff_left: %d\n", rd_result$N_h[1]))
cat(sprintf("n_eff_right: %d\n", rd_result$N_h[2]))

# Save for validation
rd_sharp <- data.frame(
    metric = c("tau_conventional", "tau_bc", "tau_robust",
               "se_conventional", "se_bc", "se_robust",
               "h_left", "h_right", "n_eff_left", "n_eff_right"),
    value = c(rd_result$coef[1], rd_result$coef[2], rd_result$coef[3],
              rd_result$se[1], rd_result$se[2], rd_result$se[3],
              rd_result$bws[1,1], rd_result$bws[1,2],
              rd_result$N_h[1], rd_result$N_h[2])
)
write.csv(rd_sharp, "validation/expected/rd_sharp.csv", row.names = FALSE)

# ==============================================================================
# 2. Sharp RD with Different Polynomial Orders
# ==============================================================================
cat("\n=== Sharp RD with p=2 (quadratic) ===\n")

rd_quad <- rdrobust(y, x, c = 0, p = 2)
cat(sprintf("tau_conventional (p=2): %.6f\n", rd_quad$coef[1]))

# ==============================================================================
# 3. Propensity Score Matching (MatchIt)
# ==============================================================================
cat("\n=== Propensity Score Matching (MatchIt) ===\n")

# Generate observational data with confounding
n <- 500
x1 <- rnorm(n)
x2 <- rnorm(n)
# Propensity score model
ps_true <- plogis(-0.5 + 0.8 * x1 + 0.4 * x2)
treatment <- rbinom(n, 1, ps_true)
# Outcome model with treatment effect
y <- 1 + 0.5 * x1 - 0.3 * x2 + 0.75 * treatment + rnorm(n, 0, 0.5)

match_data <- data.frame(y = y, treatment = treatment, x1 = x1, x2 = x2)

# Nearest neighbor matching (1:1)
m_nn <- matchit(treatment ~ x1 + x2, data = match_data, method = "nearest",
                distance = "logit", replace = FALSE)
summary_nn <- summary(m_nn)
print(summary_nn)

cat("\n--- Key Values for test_validate_matching_vs_r ---\n")
cat(sprintf("n_matched_treated: %d\n", sum(match_data$treatment == 1)))
cat(sprintf("n_matched_control: %d\n", sum(match_data$treatment == 0)))
# Before matching SMD
smd_before_x1 <- summary_nn$sum.all["x1", "Std. Mean Diff."]
smd_before_x2 <- summary_nn$sum.all["x2", "Std. Mean Diff."]
cat(sprintf("smd_before_x1: %.6f\n", smd_before_x1))
cat(sprintf("smd_before_x2: %.6f\n", smd_before_x2))
# After matching SMD
smd_after_x1 <- summary_nn$sum.matched["x1", "Std. Mean Diff."]
smd_after_x2 <- summary_nn$sum.matched["x2", "Std. Mean Diff."]
cat(sprintf("smd_after_x1: %.6f\n", smd_after_x1))
cat(sprintf("smd_after_x2: %.6f\n", smd_after_x2))

# Save for validation
match_nn <- data.frame(
    metric = c("n_treated", "n_control",
               "smd_before_x1", "smd_before_x2",
               "smd_after_x1", "smd_after_x2"),
    value = c(sum(match_data$treatment == 1), sum(match_data$treatment == 0),
              smd_before_x1, smd_before_x2, smd_after_x1, smd_after_x2)
)
write.csv(match_nn, "validation/expected/match_nn.csv", row.names = FALSE)

# ==============================================================================
# 4. Coarsened Exact Matching
# ==============================================================================
cat("\n=== Coarsened Exact Matching ===\n")

m_cem <- matchit(treatment ~ x1 + x2, data = match_data, method = "cem")
summary_cem <- summary(m_cem)
print(summary_cem)

# ==============================================================================
# 5. Full/Optimal Matching
# ==============================================================================
cat("\n=== Full Matching ===\n")

m_full <- matchit(treatment ~ x1 + x2, data = match_data, method = "full",
                  distance = "logit")
summary_full <- summary(m_full)
print(summary_full)

# ==============================================================================
# 6. IPW with WeightIt
# ==============================================================================
cat("\n=== IPW Weighting (WeightIt) ===\n")

# Use same data as matching
W <- weightit(treatment ~ x1 + x2, data = match_data, method = "ps",
              estimand = "ATE")
print(summary(W))

cat("\n--- Key Values for test_validate_ipw_vs_r ---\n")
cat(sprintf("ps_mean: %.6f\n", mean(W$ps)))
cat(sprintf("ps_sd: %.6f\n", sd(W$ps)))
cat(sprintf("ps_min: %.6f\n", min(W$ps)))
cat(sprintf("ps_max: %.6f\n", max(W$ps)))
cat(sprintf("ess_treated: %.2f\n", W$ess[2]))
cat(sprintf("ess_control: %.2f\n", W$ess[1]))

# Compute ATE manually with IPW
weights <- W$weights
ipw_ate <- sum(weights * match_data$treatment * match_data$y) / sum(weights * match_data$treatment) -
           sum(weights * (1 - match_data$treatment) * match_data$y) / sum(weights * (1 - match_data$treatment))
cat(sprintf("ipw_ate: %.6f\n", ipw_ate))

# Save for validation
ipw_result <- data.frame(
    metric = c("ps_mean", "ps_sd", "ps_min", "ps_max", "ess_treated", "ess_control", "ipw_ate"),
    value = c(mean(W$ps), sd(W$ps), min(W$ps), max(W$ps), W$ess[2], W$ess[1], ipw_ate)
)
write.csv(ipw_result, "validation/expected/ipw_weightit.csv", row.names = FALSE)

# ==============================================================================
# 7. Entropy Balancing
# ==============================================================================
cat("\n=== Entropy Balancing ===\n")

W_ebal <- weightit(treatment ~ x1 + x2, data = match_data, method = "ebal",
                   estimand = "ATT")
print(summary(W_ebal))

# Save for validation
ebal_result <- data.frame(
    metric = c("ess_control", "weight_mean", "weight_sd"),
    value = c(W_ebal$ess[1], mean(W_ebal$weights), sd(W_ebal$weights))
)
write.csv(ebal_result, "validation/expected/ebal_weightit.csv", row.names = FALSE)

# ==============================================================================
# 8. CBPS (Covariate Balancing Propensity Score)
# ==============================================================================
cat("\n=== CBPS ===\n")

# Using WeightIt with CBPS
W_cbps <- weightit(treatment ~ x1 + x2, data = match_data, method = "cbps",
                   estimand = "ATE")
print(summary(W_cbps))

cat("\n--- Key Values for CBPS ---\n")
cat(sprintf("cbps_ps_mean: %.6f\n", mean(W_cbps$ps)))
cat(sprintf("cbps_ps_sd: %.6f\n", sd(W_cbps$ps)))

cbps_result <- data.frame(
    metric = c("ps_mean", "ps_sd", "ess_treated", "ess_control"),
    value = c(mean(W_cbps$ps), sd(W_cbps$ps), W_cbps$ess[2], W_cbps$ess[1])
)
write.csv(cbps_result, "validation/expected/cbps_weightit.csv", row.names = FALSE)

# ==============================================================================
# Summary statistics saved
# ==============================================================================
cat("\n=== Validation Data Saved ===\n")
cat("Files created:\n")
cat("  - validation/expected/rd_sharp.csv\n")
cat("  - validation/expected/match_nn.csv\n")
cat("  - validation/expected/ipw_weightit.csv\n")
cat("  - validation/expected/ebal_weightit.csv\n")
cat("  - validation/expected/cbps_weightit.csv\n")
