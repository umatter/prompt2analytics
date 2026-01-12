#!/usr/bin/env Rscript
# Synthetic Control Validation: R vs Rust Comparison
# This script generates reference values for validating the p2a Rust implementation

library(tidysynth)
library(dplyr)

cat("Synthetic Control Validation\n")
cat("=============================\n")
cat("Reference implementation: tidysynth (R)\n\n")

# =============================================================================
# Test Case 1: Basic Synth (Exact Match)
# This matches crates/p2a-core/src/econometrics/synth.rs::tests::test_basic_synth
# =============================================================================
run_test_basic <- function() {
  cat("\n--- Test 1: Basic Synthetic Control ---\n")

  # Panel data where Unit A is exactly 0.5*B + 0.5*C in pre-treatment
  panel <- tibble(
    unit = rep(c("A", "B", "C"), each = 6),
    time = rep(1:6, 3),
    outcome = c(
      10.0, 11.0, 12.0, 18.0, 19.0, 20.0,  # A (treated)
      8.0, 10.0, 12.0, 14.0, 16.0, 18.0,   # B
      12.0, 12.0, 12.0, 12.0, 12.0, 12.0   # C
    )
  )

  synth_out <- panel %>%
    synthetic_control(
      outcome = outcome,
      unit = unit,
      time = time,
      i_unit = "A",
      i_time = 4,
      generate_placebos = FALSE
    ) %>%
    generate_predictor(time_window = 1:3, outcome = mean(outcome)) %>%
    generate_weights(optimization_window = 1:3) %>%
    generate_control()

  weights <- synth_out %>% grab_unit_weights()
  synth_vals <- synth_out %>% grab_synthetic_control()

  # Calculate metrics
  weight_b <- weights$weight[weights$unit == "B"]
  weight_c <- weights$weight[weights$unit == "C"]

  effects <- synth_vals %>% mutate(effect = real_y - synth_y)
  pre_effects <- effects %>% filter(time_unit < 4)
  post_effects <- effects %>% filter(time_unit >= 4)

  pre_rmspe <- sqrt(mean(pre_effects$effect^2))
  avg_post_effect <- mean(post_effects$effect)

  cat(sprintf("Weight B: %.6f (expected ~0.5)\n", weight_b))
  cat(sprintf("Weight C: %.6f (expected ~0.5)\n", weight_c))
  cat(sprintf("Pre-treatment RMSPE: %.6f (expected ~0)\n", pre_rmspe))
  cat(sprintf("Avg post-treatment effect: %.6f (expected ~5.0)\n", avg_post_effect))

  list(
    test = "basic_synth",
    weight_b = weight_b,
    weight_c = weight_c,
    pre_rmspe = pre_rmspe,
    avg_post_effect = avg_post_effect
  )
}

# =============================================================================
# Test Case 2: With Predictor Variable
# This matches crates/p2a-core/src/econometrics/synth.rs::tests::test_synth_with_predictors
# =============================================================================
run_test_with_predictors <- function() {
  cat("\n--- Test 2: Synthetic Control with Predictors ---\n")

  panel <- tibble(
    unit = rep(c("A", "B", "C", "D"), each = 8),
    time = rep(1:8, 4),
    outcome = c(
      10, 11, 12, 13, 18, 19, 20, 21,  # A (treated at t=5, effect=5)
      8, 10, 12, 14, 16, 18, 20, 22,   # B
      12, 12, 12, 12, 12, 12, 12, 12,  # C
      6, 8, 10, 12, 14, 16, 18, 20     # D
    ),
    x1 = c(
      5, 5, 5, 5, 5, 5, 5, 5,  # A
      4, 4, 4, 4, 4, 4, 4, 4,  # B
      6, 6, 6, 6, 6, 6, 6, 6,  # C
      3, 3, 3, 3, 3, 3, 3, 3   # D
    )
  )

  synth_out <- panel %>%
    synthetic_control(
      outcome = outcome,
      unit = unit,
      time = time,
      i_unit = "A",
      i_time = 5,
      generate_placebos = FALSE
    ) %>%
    generate_predictor(time_window = 1:4, x1 = mean(x1)) %>%
    generate_predictor(time_window = 1:4, outcome = mean(outcome)) %>%
    generate_weights(optimization_window = 1:4) %>%
    generate_control()

  weights <- synth_out %>% grab_unit_weights()
  synth_vals <- synth_out %>% grab_synthetic_control()

  cat("Donor weights:\n")
  print(weights %>% filter(weight > 0.01))

  effects <- synth_vals %>% mutate(effect = real_y - synth_y)
  post_effects <- effects %>% filter(time_unit >= 5)
  avg_effect <- mean(post_effects$effect)

  cat(sprintf("Avg post-treatment effect: %.6f\n", avg_effect))

  list(
    test = "with_predictors",
    weights = weights,
    avg_effect = avg_effect
  )
}

# =============================================================================
# Test Case 3: Placebo Tests
# This matches crates/p2a-core/src/econometrics/synth.rs::tests::test_synth_placebo
# =============================================================================
run_test_placebo <- function() {
  cat("\n--- Test 3: Placebo Tests ---\n")

  # Create panel where treated unit has clear effect
  set.seed(42)

  panel <- tibble(
    unit = rep(c("Treated", "D1", "D2", "D3", "D4", "D5"), each = 10),
    time = rep(1:10, 6),
    outcome = c(
      # Treated: stable at ~10 pre, then jumps to ~15 post
      10, 10.5, 9.8, 10.2, 10.1, 15.1, 15.3, 14.9, 15.0, 15.2,
      # D1: gradual increase
      8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5,
      # D2: stable
      10, 10.1, 9.9, 10.0, 10.1, 10.0, 9.9, 10.1, 10.0, 10.0,
      # D3: declining
      12, 11.8, 11.6, 11.4, 11.2, 11.0, 10.8, 10.6, 10.4, 10.2,
      # D4: stable higher
      14, 14.1, 13.9, 14.0, 14.1, 14.0, 13.9, 14.1, 14.0, 14.0,
      # D5: stable lower
      6, 6.1, 5.9, 6.0, 6.1, 6.0, 5.9, 6.1, 6.0, 6.0
    )
  )

  synth_out <- panel %>%
    synthetic_control(
      outcome = outcome,
      unit = unit,
      time = time,
      i_unit = "Treated",
      i_time = 6,
      generate_placebos = TRUE
    ) %>%
    generate_predictor(time_window = 1:5, outcome = mean(outcome)) %>%
    generate_weights(optimization_window = 1:5) %>%
    generate_control()

  significance <- synth_out %>% grab_significance()
  cat("Significance results:\n")
  print(significance)

  treated_rank <- significance$rank[significance$unit_name == "Treated"]
  n_units <- nrow(significance)
  p_value <- treated_rank / n_units

  cat(sprintf("\nTreated unit rank: %d / %d\n", treated_rank, n_units))
  cat(sprintf("Fisher's exact p-value: %.4f\n", p_value))

  list(
    test = "placebo",
    significance = significance,
    treated_rank = treated_rank,
    p_value = p_value
  )
}

# =============================================================================
# Test Case 4: Edge Case - Single Donor
# =============================================================================
run_test_single_donor <- function() {
  cat("\n--- Test 4: Single Donor ---\n")
  cat("Note: tidysynth requires at least 2 control units.\n")
  cat("Rust implementation handles single donor case (weight = 1.0).\n")
  cat("Skipping R validation for this edge case.\n")

  list(
    test = "single_donor",
    donor_weight = NA  # Cannot test with tidysynth
  )
}

# =============================================================================
# Run All Tests
# =============================================================================
results <- list()
results$basic <- run_test_basic()
results$predictors <- run_test_with_predictors()
results$placebo <- run_test_placebo()
results$single <- run_test_single_donor()

# =============================================================================
# Export Results for Rust Comparison
# =============================================================================
cat("\n\n=== Exporting Validation Data ===\n")

# Create validation output directory
dir.create("output", showWarnings = FALSE)

# Save summary
summary_df <- data.frame(
  test = c("basic_synth", "basic_synth", "basic_synth", "basic_synth",
           "placebo"),
  metric = c("weight_b", "weight_c", "pre_rmspe", "avg_post_effect",
             "p_value"),
  r_value = c(
    results$basic$weight_b,
    results$basic$weight_c,
    results$basic$pre_rmspe,
    results$basic$avg_post_effect,
    results$placebo$p_value
  )
)

write.csv(summary_df, "output/synth_validation_results.csv", row.names = FALSE)
cat("Results saved to output/synth_validation_results.csv\n")

# Print summary table
cat("\n=== Validation Summary ===\n")
cat("Test Case          | Metric           | R Value    | Expected   | Tolerance\n")
cat("-------------------|------------------|------------|------------|----------\n")
cat(sprintf("basic_synth        | weight_b         | %.6f   | ~0.5       | 0.1\n", results$basic$weight_b))
cat(sprintf("basic_synth        | weight_c         | %.6f   | ~0.5       | 0.1\n", results$basic$weight_c))
cat(sprintf("basic_synth        | pre_rmspe        | %.6f   | ~0.0       | 0.5\n", results$basic$pre_rmspe))
cat(sprintf("basic_synth        | avg_post_effect  | %.6f   | ~5.0       | 0.5\n", results$basic$avg_post_effect))
cat(sprintf("with_predictors    | avg_post_effect  | %.6f   | ~5.0       | 1.0\n", results$predictors$avg_effect))
cat("single_donor       | donor_weight     | N/A (tidysynth requires 2+ donors)\n")
cat(sprintf("placebo            | p_value          | %.6f   | varies     | -\n", results$placebo$p_value))

cat("\nValidation complete.\n")
