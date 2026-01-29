# Phase 5: Treatment Effects Validation
# Generates expected values for IPW, AIPW, TMLE, Mediation, etc.

library(tmle)
library(mediation)

set.seed(42)

# ==============================================================================
# 1. Generate common dataset for treatment effect estimation
# ==============================================================================
cat("=== Generating Common Dataset ===\n")

n <- 1000
# Covariates
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rbinom(n, 1, 0.5)

# Propensity score model (true)
ps_true <- plogis(-0.5 + 0.6 * x1 + 0.3 * x2 - 0.2 * x3)
treatment <- rbinom(n, 1, ps_true)

# Outcome model (binary outcome for TMLE)
y_latent <- -1 + 0.5 * x1 - 0.3 * x2 + 0.2 * x3 + 0.8 * treatment + rnorm(n, 0, 0.5)
# For binary outcome
y_binary <- as.numeric(plogis(y_latent) > runif(n))
# For continuous outcome
y_continuous <- y_latent

df_treatment <- data.frame(
    y_binary = y_binary,
    y_continuous = y_continuous,
    treatment = treatment,
    x1 = x1,
    x2 = x2,
    x3 = x3
)

cat(sprintf("N: %d, Treated: %d, Control: %d\n", n, sum(treatment), n - sum(treatment)))

# ==============================================================================
# 2. TMLE for Binary Outcome (ATE)
# ==============================================================================
cat("\n=== TMLE (Binary Outcome) ===\n")

# Prepare data for tmle
W <- data.frame(x1 = x1, x2 = x2, x3 = x3)
A <- treatment
Y <- y_binary

# Run TMLE
tmle_result <- tmle(Y, A, W,
                    Q.SL.library = "SL.glm",
                    g.SL.library = "SL.glm")
print(tmle_result)

cat("\n--- Key Values for test_validate_tmle_vs_r ---\n")
cat(sprintf("ate: %.6f\n", tmle_result$estimates$ATE$psi))
cat(sprintf("ate_se: %.6f\n", sqrt(tmle_result$estimates$ATE$var.psi)))
cat(sprintf("ate_ci_lower: %.6f\n", tmle_result$estimates$ATE$CI[1]))
cat(sprintf("ate_ci_upper: %.6f\n", tmle_result$estimates$ATE$CI[2]))
cat(sprintf("ate_p_value: %.6f\n", tmle_result$estimates$ATE$pvalue))

# Save for validation
tmle_result_df <- data.frame(
    metric = c("ate", "ate_se", "ate_ci_lower", "ate_ci_upper", "ate_p_value"),
    value = c(tmle_result$estimates$ATE$psi,
              sqrt(tmle_result$estimates$ATE$var.psi),
              tmle_result$estimates$ATE$CI[1],
              tmle_result$estimates$ATE$CI[2],
              tmle_result$estimates$ATE$pvalue)
)
write.csv(tmle_result_df, "validation/expected/tmle_binary.csv", row.names = FALSE)

# ==============================================================================
# 3. TMLE for Continuous Outcome
# ==============================================================================
cat("\n=== TMLE (Continuous Outcome) ===\n")

# Prepare family for continuous outcome
tmle_cont <- tmle(y_continuous, A, W,
                  Q.SL.library = "SL.glm",
                  g.SL.library = "SL.glm",
                  family = "gaussian")
print(tmle_cont)

cat("\n--- Key Values for continuous TMLE ---\n")
cat(sprintf("ate: %.6f\n", tmle_cont$estimates$ATE$psi))
cat(sprintf("ate_se: %.6f\n", sqrt(tmle_cont$estimates$ATE$var.psi)))

tmle_cont_df <- data.frame(
    metric = c("ate", "ate_se", "ate_ci_lower", "ate_ci_upper"),
    value = c(tmle_cont$estimates$ATE$psi,
              sqrt(tmle_cont$estimates$ATE$var.psi),
              tmle_cont$estimates$ATE$CI[1],
              tmle_cont$estimates$ATE$CI[2])
)
write.csv(tmle_cont_df, "validation/expected/tmle_continuous.csv", row.names = FALSE)

# ==============================================================================
# 4. Causal Mediation Analysis
# ==============================================================================
cat("\n=== Causal Mediation Analysis ===\n")

# Generate data with mediation structure
set.seed(42)
n <- 500
x <- rnorm(n)
treatment <- rbinom(n, 1, plogis(0.5 * x))

# Mediator affected by treatment and x
mediator <- 0.3 + 0.6 * treatment + 0.4 * x + rnorm(n, 0, 0.5)

# Outcome affected by treatment, mediator, and x
y <- 1 + 0.4 * treatment + 0.5 * mediator + 0.3 * x + rnorm(n, 0, 0.5)

med_data <- data.frame(y = y, treatment = treatment, mediator = mediator, x = x)

# Fit mediation models
med_fit <- lm(mediator ~ treatment + x, data = med_data)
out_fit <- lm(y ~ treatment + mediator + x, data = med_data)

# Run mediation analysis
med_result <- mediate(med_fit, out_fit,
                      treat = "treatment",
                      mediator = "mediator",
                      boot = TRUE,
                      boot.ci.type = "perc",
                      sims = 500)
print(summary(med_result))

cat("\n--- Key Values for test_validate_mediation_vs_r ---\n")
cat(sprintf("total_effect (ATE): %.6f\n", med_result$tau.coef))
cat(sprintf("direct_effect (ADE): %.6f\n", med_result$z0))
cat(sprintf("indirect_effect (ACME): %.6f\n", med_result$d0))
cat(sprintf("proportion_mediated: %.6f\n", med_result$n0))
cat(sprintf("se_total: %.6f\n", sqrt(med_result$tau.var)))
cat(sprintf("se_direct: %.6f\n", sqrt(med_result$z0.var)))
cat(sprintf("se_indirect: %.6f\n", sqrt(med_result$d0.var)))

# Save for validation
mediation_df <- data.frame(
    metric = c("total_effect", "direct_effect", "indirect_effect",
               "proportion_mediated", "se_total", "se_direct", "se_indirect",
               "p_total", "p_direct", "p_indirect"),
    value = c(med_result$tau.coef, med_result$z0, med_result$d0,
              med_result$n0, sqrt(med_result$tau.var), sqrt(med_result$z0.var),
              sqrt(med_result$d0.var),
              med_result$tau.p, med_result$z0.p, med_result$d0.p)
)
write.csv(mediation_df, "validation/expected/mediation.csv", row.names = FALSE)

# ==============================================================================
# 5. Basic IPW/AIPW without external packages
# ==============================================================================
cat("\n=== Manual IPW/AIPW Calculation ===\n")

# Use the binary outcome data
ps_model <- glm(treatment ~ x1 + x2 + x3, data = df_treatment, family = binomial)
ps_hat <- predict(ps_model, type = "response")

# IPW ATE (Horvitz-Thompson)
w1 <- df_treatment$treatment / ps_hat
w0 <- (1 - df_treatment$treatment) / (1 - ps_hat)
ipw_ate <- mean(w1 * df_treatment$y_binary) - mean(w0 * df_treatment$y_binary)

# IPW ATE (Hajek/normalized)
ipw_ate_hajek <- sum(w1 * df_treatment$y_binary) / sum(w1) -
                 sum(w0 * df_treatment$y_binary) / sum(w0)

cat(sprintf("IPW ATE (HT): %.6f\n", ipw_ate))
cat(sprintf("IPW ATE (Hajek): %.6f\n", ipw_ate_hajek))

# AIPW with outcome regression
out_model_1 <- glm(y_binary ~ x1 + x2 + x3, data = df_treatment[df_treatment$treatment == 1,], family = binomial)
out_model_0 <- glm(y_binary ~ x1 + x2 + x3, data = df_treatment[df_treatment$treatment == 0,], family = binomial)

# Predict counterfactuals
mu1_hat <- predict(out_model_1, newdata = df_treatment, type = "response")
mu0_hat <- predict(out_model_0, newdata = df_treatment, type = "response")

# AIPW estimator
aipw_ate <- mean(
    (df_treatment$treatment * df_treatment$y_binary - (df_treatment$treatment - ps_hat) * mu1_hat) / ps_hat -
    ((1 - df_treatment$treatment) * df_treatment$y_binary + (df_treatment$treatment - ps_hat) * mu0_hat) / (1 - ps_hat)
)

cat(sprintf("AIPW ATE: %.6f\n", aipw_ate))

# Save for validation
ipw_aipw_df <- data.frame(
    metric = c("ipw_ate_ht", "ipw_ate_hajek", "aipw_ate", "ps_mean", "ps_sd"),
    value = c(ipw_ate, ipw_ate_hajek, aipw_ate, mean(ps_hat), sd(ps_hat))
)
write.csv(ipw_aipw_df, "validation/expected/ipw_aipw_manual.csv", row.names = FALSE)

# ==============================================================================
# Summary
# ==============================================================================
cat("\n=== Validation Data Saved ===\n")
cat("Files created:\n")
cat("  - validation/expected/tmle_binary.csv\n")
cat("  - validation/expected/tmle_continuous.csv\n")
cat("  - validation/expected/mediation.csv\n")
cat("  - validation/expected/ipw_aipw_manual.csv\n")
