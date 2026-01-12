#!/usr/bin/env Rscript
# Synthetic Control Method Benchmarks for Cross-Language Comparison
# Compares R packages (Synth, tidysynth) against p2a Rust implementation

library(microbenchmark)

# Attempt to load synth packages
synth_available <- suppressWarnings(require(Synth, quietly = TRUE))
tidysynth_available <- suppressWarnings(require(tidysynth, quietly = TRUE))

if (!synth_available && !tidysynth_available) {
  stop("Neither Synth nor tidysynth packages are available.
        Install with: install.packages(c('Synth', 'tidysynth'))")
}

library(dplyr)

# Ensure reproducibility
set.seed(42)

# =============================================================================
# Test Case 1: Perfect Synthetic Match (Simple)
# Unit A is exactly 0.5*B + 0.5*C in pre-treatment
# =============================================================================
run_simple_test <- function() {
  cat("\n=== Test 1: Perfect Synthetic Match ===\n")

  if (!tidysynth_available) {
    cat("Skipping: tidysynth not available\n")
    return(NULL)
  }

  # Create panel data where treated unit is exact combination
  panel <- tibble(
    unit = rep(c("A", "B", "C"), each = 6),
    time = rep(1:6, 3),
    outcome = c(
      # Unit A: pre = 0.5*B + 0.5*C, post = pre + 5 (treatment effect)
      10, 11, 12, 18, 19, 20,  # A (treatment effect = 5 at t>=4)
      8, 10, 12, 14, 16, 18,   # B
      12, 12, 12, 12, 12, 12   # C
    ),
    x1 = c(
      5, 5, 5, 5, 5, 5,  # A: mean = 5 = 0.5*4 + 0.5*6
      4, 4, 4, 4, 4, 4,  # B: mean = 4
      6, 6, 6, 6, 6, 6   # C: mean = 6
    )
  )

  # Run synthetic control
  synth_out <- panel %>%
    synthetic_control(
      outcome = outcome,
      unit = unit,
      time = time,
      i_unit = "A",
      i_time = 4,
      generate_placebos = FALSE
    ) %>%
    generate_predictor(time_window = 1:3, x1 = mean(x1)) %>%
    generate_predictor(time_window = 1:3, outcome = mean(outcome)) %>%
    generate_weights(optimization_window = 1:3) %>%
    generate_control()

  # Extract weights
  weights <- synth_out %>% grab_unit_weights()
  cat("Donor weights:\n")
  print(weights)

  # Extract synthetic control values
  synth_vals <- synth_out %>% grab_synthetic_control()
  cat("\nSynthetic control vs actual:\n")
  print(synth_vals)

  # Calculate treatment effects
  effects <- synth_vals %>%
    mutate(effect = real_y - synth_y)
  cat("\nTreatment effects:\n")
  print(effects)

  # Expected results
  cat("\n--- Validation Summary ---\n")
  weight_b <- weights$weight[weights$unit == "B"]
  weight_c <- weights$weight[weights$unit == "C"]
  cat(sprintf("Weight B: %.3f (expected ~0.5)\n", weight_b))
  cat(sprintf("Weight C: %.3f (expected ~0.5)\n", weight_c))

  post_effects <- effects %>% filter(time_unit >= 4)
  avg_effect <- mean(post_effects$effect)
  cat(sprintf("Avg post-treatment effect: %.3f (expected ~5.0)\n", avg_effect))

  pre_rmspe <- sqrt(mean((effects %>% filter(time_unit < 4))$effect^2))
  cat(sprintf("Pre-treatment RMSPE: %.3f (expected ~0)\n", pre_rmspe))

  list(
    weight_b = weight_b,
    weight_c = weight_c,
    avg_effect = avg_effect,
    pre_rmspe = pre_rmspe
  )
}

# =============================================================================
# Test Case 2: Larger Panel with More Donors
# =============================================================================
run_larger_panel_test <- function() {
  cat("\n=== Test 2: Larger Panel (10 donors) ===\n")

  if (!tidysynth_available) {
    cat("Skipping: tidysynth not available\n")
    return(NULL)
  }

  n_donors <- 10
  n_periods <- 15
  treatment_time <- 8

  # Generate donor data
  set.seed(42)
  panel_list <- list()

  # Generate donors with varying characteristics
  for (i in 1:n_donors) {
    donor_trend <- runif(1, -0.5, 0.5)
    donor_level <- runif(1, 5, 15)

    outcome <- donor_level + (1:n_periods) * donor_trend + rnorm(n_periods, 0, 0.5)
    x1 <- runif(n_periods, 0, 10)
    x2 <- rnorm(n_periods, 5, 1)

    panel_list[[i]] <- tibble(
      unit = paste0("D", i),
      time = 1:n_periods,
      outcome = outcome,
      x1 = x1,
      x2 = x2
    )
  }

  donors <- bind_rows(panel_list)

  # Create treated unit as weighted combination of first 3 donors + treatment effect
  weights_true <- c(0.4, 0.35, 0.25)  # True weights for D1, D2, D3

  d1 <- panel_list[[1]]$outcome
  d2 <- panel_list[[2]]$outcome
  d3 <- panel_list[[3]]$outcome

  treated_pre <- weights_true[1] * d1 + weights_true[2] * d2 + weights_true[3] * d3
  treatment_effect <- c(rep(0, treatment_time - 1), rep(3, n_periods - treatment_time + 1))
  treated_outcome <- treated_pre + treatment_effect + rnorm(n_periods, 0, 0.2)

  treated <- tibble(
    unit = "Treated",
    time = 1:n_periods,
    outcome = treated_outcome,
    x1 = 0.4 * panel_list[[1]]$x1 + 0.35 * panel_list[[2]]$x1 + 0.25 * panel_list[[3]]$x1,
    x2 = 0.4 * panel_list[[1]]$x2 + 0.35 * panel_list[[2]]$x2 + 0.25 * panel_list[[3]]$x2
  )

  panel <- bind_rows(treated, donors)

  # Run synthetic control
  tryCatch({
    synth_out <- panel %>%
      synthetic_control(
        outcome = outcome,
        unit = unit,
        time = time,
        i_unit = "Treated",
        i_time = treatment_time,
        generate_placebos = FALSE
      ) %>%
      generate_predictor(time_window = 1:(treatment_time - 1), x1 = mean(x1)) %>%
      generate_predictor(time_window = 1:(treatment_time - 1), x2 = mean(x2)) %>%
      generate_predictor(time_window = 1:(treatment_time - 1), outcome = mean(outcome)) %>%
      generate_weights(optimization_window = 1:(treatment_time - 1)) %>%
      generate_control()

    weights <- synth_out %>% grab_unit_weights()
    cat("Donor weights:\n")
    print(weights %>% filter(weight > 0.01))

    synth_vals <- synth_out %>% grab_synthetic_control()
    effects <- synth_vals %>% mutate(effect = real_y - synth_y)

    post_effects <- effects %>% filter(time_unit >= treatment_time)
    avg_effect <- mean(post_effects$effect)
    cat(sprintf("\nAvg post-treatment effect: %.3f (true effect = 3.0)\n", avg_effect))

    pre_rmspe <- sqrt(mean((effects %>% filter(time_unit < treatment_time))$effect^2))
    cat(sprintf("Pre-treatment RMSPE: %.3f\n", pre_rmspe))

    # Check if D1, D2, D3 have highest weights
    top_donors <- weights %>% arrange(desc(weight)) %>% head(3)
    cat("\nTop 3 donors (should be D1, D2, D3):\n")
    print(top_donors)

    list(
      weights = weights,
      avg_effect = avg_effect,
      pre_rmspe = pre_rmspe,
      top_donors = top_donors
    )
  }, error = function(e) {
    cat(sprintf("Error: %s\n", e$message))
    NULL
  })
}

# =============================================================================
# Test Case 3: QP Solver Validation
# =============================================================================
run_qp_test <- function() {
  cat("\n=== Test 3: QP Solver Validation ===\n")

  # Verify simplex-constrained QP solver
  # Minimize: ||x||² subject to Σx = 1, x ≥ 0
  # Expected: x = [0.5, 0.5] for n=2

  if (!require(quadprog, quietly = TRUE)) {
    cat("Skipping: quadprog not available\n")
    return(NULL)
  }

  # QP: min 0.5 x'Dx + d'x
  # s.t. A'x >= b

  D <- diag(2)
  d <- rep(0, 2)

  # Equality: sum(x) = 1 → Aeq'x = 1
  # Inequality: x >= 0

  Amat <- cbind(c(1, 1), diag(2))  # Equality + bounds
  bvec <- c(1, 0, 0)

  sol <- solve.QP(D, d, Amat, bvec, meq = 1)

  cat(sprintf("Solution: [%.4f, %.4f] (expected [0.5, 0.5])\n",
              sol$solution[1], sol$solution[2]))
  cat(sprintf("Sum of weights: %.4f (expected 1.0)\n", sum(sol$solution)))

  # Test with 4 variables
  D4 <- diag(4)
  d4 <- rep(0, 4)
  Amat4 <- cbind(rep(1, 4), diag(4))
  bvec4 <- c(1, 0, 0, 0, 0)

  sol4 <- solve.QP(D4, d4, Amat4, bvec4, meq = 1)
  cat(sprintf("4-var solution: [%.4f, %.4f, %.4f, %.4f] (expected [0.25, 0.25, 0.25, 0.25])\n",
              sol4$solution[1], sol4$solution[2], sol4$solution[3], sol4$solution[4]))

  list(
    solution_2 = sol$solution,
    solution_4 = sol4$solution
  )
}

# =============================================================================
# Test Case 4: Placebo Test / Inference
# =============================================================================
run_placebo_test <- function() {
  cat("\n=== Test 4: Placebo Tests ===\n")

  if (!tidysynth_available) {
    cat("Skipping: tidysynth not available\n")
    return(NULL)
  }

  # Create data with clear treatment effect
  n_units <- 8
  n_periods <- 12
  treatment_time <- 7

  set.seed(42)
  panel_list <- list()

  for (i in 1:n_units) {
    trend <- runif(1, -0.3, 0.3)
    level <- runif(1, 8, 12)
    outcome <- level + (1:n_periods) * trend + rnorm(n_periods, 0, 0.3)

    panel_list[[i]] <- tibble(
      unit = paste0("U", i),
      time = 1:n_periods,
      outcome = outcome,
      x1 = rnorm(n_periods, 5, 1)
    )
  }

  # Add treated unit with clear effect
  treated_base <- 10 + (1:n_periods) * 0.1 + rnorm(n_periods, 0, 0.3)
  treatment_effect <- c(rep(0, treatment_time - 1), rep(4, n_periods - treatment_time + 1))

  panel_list[[n_units + 1]] <- tibble(
    unit = "Treated",
    time = 1:n_periods,
    outcome = treated_base + treatment_effect,
    x1 = rnorm(n_periods, 5, 1)
  )

  panel <- bind_rows(panel_list)

  # Run with placebos
  tryCatch({
    synth_out <- panel %>%
      synthetic_control(
        outcome = outcome,
        unit = unit,
        time = time,
        i_unit = "Treated",
        i_time = treatment_time,
        generate_placebos = TRUE
      ) %>%
      generate_predictor(time_window = 1:(treatment_time - 1),
                         x1 = mean(x1),
                         outcome = mean(outcome)) %>%
      generate_weights(optimization_window = 1:(treatment_time - 1)) %>%
      generate_control()

    # Get significance
    significance <- synth_out %>% grab_significance()
    cat("Significance results:\n")
    print(significance)

    # Check that treated unit has low rank (high RMSPE ratio)
    treated_rank <- significance$rank[significance$unit_name == "Treated"]
    n_total <- nrow(significance)
    cat(sprintf("\nTreated unit rank: %d out of %d\n", treated_rank, n_total))
    cat(sprintf("Fisher exact p-value: %.3f\n", treated_rank / n_total))

    list(
      significance = significance,
      treated_rank = treated_rank,
      p_value = treated_rank / n_total
    )
  }, error = function(e) {
    cat(sprintf("Error: %s\n", e$message))
    NULL
  })
}

# =============================================================================
# Benchmark Performance
# =============================================================================
benchmark_synth <- function() {
  cat("\n=== Performance Benchmarks ===\n")

  if (!tidysynth_available) {
    cat("Skipping: tidysynth not available\n")
    return(NULL)
  }

  results <- list()

  # Small panel
  cat("\nSmall panel (5 donors, 10 periods):\n")
  set.seed(42)
  small_panel <- generate_test_panel(5, 10, 6)

  bm_small <- microbenchmark(
    {
      small_panel %>%
        synthetic_control(
          outcome = outcome, unit = unit, time = time,
          i_unit = "Treated", i_time = 6, generate_placebos = FALSE
        ) %>%
        generate_predictor(time_window = 1:5, outcome = mean(outcome)) %>%
        generate_weights(optimization_window = 1:5) %>%
        generate_control()
    },
    times = 20,
    unit = "milliseconds"
  )
  print(summary(bm_small))
  results[["small"]] <- summary(bm_small)

  # Medium panel
  cat("\nMedium panel (15 donors, 20 periods):\n")
  medium_panel <- generate_test_panel(15, 20, 12)

  bm_medium <- microbenchmark(
    {
      medium_panel %>%
        synthetic_control(
          outcome = outcome, unit = unit, time = time,
          i_unit = "Treated", i_time = 12, generate_placebos = FALSE
        ) %>%
        generate_predictor(time_window = 1:11, outcome = mean(outcome)) %>%
        generate_weights(optimization_window = 1:11) %>%
        generate_control()
    },
    times = 10,
    unit = "milliseconds"
  )
  print(summary(bm_medium))
  results[["medium"]] <- summary(bm_medium)

  # Large panel (with placebos)
  cat("\nMedium panel with placebos (10 donors, 15 periods):\n")
  large_panel <- generate_test_panel(10, 15, 8)

  bm_large <- microbenchmark(
    {
      large_panel %>%
        synthetic_control(
          outcome = outcome, unit = unit, time = time,
          i_unit = "Treated", i_time = 8, generate_placebos = TRUE
        ) %>%
        generate_predictor(time_window = 1:7, outcome = mean(outcome)) %>%
        generate_weights(optimization_window = 1:7) %>%
        generate_control()
    },
    times = 5,
    unit = "milliseconds"
  )
  print(summary(bm_large))
  results[["large_placebo"]] <- summary(bm_large)

  results
}

# Helper function to generate test panels
generate_test_panel <- function(n_donors, n_periods, treatment_time) {
  panel_list <- list()

  for (i in 1:n_donors) {
    trend <- runif(1, -0.3, 0.3)
    level <- runif(1, 8, 12)
    outcome <- level + (1:n_periods) * trend + rnorm(n_periods, 0, 0.5)

    panel_list[[i]] <- tibble(
      unit = paste0("D", i),
      time = 1:n_periods,
      outcome = outcome
    )
  }

  # Treated unit
  treated_base <- 10 + (1:n_periods) * 0.1 + rnorm(n_periods, 0, 0.3)
  treatment_effect <- c(rep(0, treatment_time - 1), rep(3, n_periods - treatment_time + 1))

  panel_list[[n_donors + 1]] <- tibble(
    unit = "Treated",
    time = 1:n_periods,
    outcome = treated_base + treatment_effect
  )

  bind_rows(panel_list)
}

# =============================================================================
# Export Results for Rust Comparison
# =============================================================================
export_test_data <- function() {
  cat("\n=== Exporting Test Data for Rust Comparison ===\n")

  # Simple test case
  simple_data <- tibble(
    unit = rep(c("A", "B", "C"), each = 6),
    time = rep(1:6, 3),
    outcome = c(
      10, 11, 12, 18, 19, 20,
      8, 10, 12, 14, 16, 18,
      12, 12, 12, 12, 12, 12
    ),
    x1 = c(
      5, 5, 5, 5, 5, 5,
      4, 4, 4, 4, 4, 4,
      6, 6, 6, 6, 6, 6
    )
  )

  dir.create("test_data", showWarnings = FALSE)
  write.csv(simple_data, "test_data/synth_simple.csv", row.names = FALSE)
  cat("Exported: test_data/synth_simple.csv\n")

  # Larger test case
  set.seed(42)
  larger_data <- generate_test_panel(10, 15, 8)
  write.csv(larger_data, "test_data/synth_larger.csv", row.names = FALSE)
  cat("Exported: test_data/synth_larger.csv\n")
}

# =============================================================================
# Main
# =============================================================================
cat("Synthetic Control Method Validation\n")
cat("====================================\n")
cat(sprintf("Synth package available: %s\n", synth_available))
cat(sprintf("tidysynth package available: %s\n", tidysynth_available))

# Run validation tests
simple_results <- run_simple_test()
larger_results <- run_larger_panel_test()
qp_results <- run_qp_test()
placebo_results <- run_placebo_test()

# Run benchmarks
benchmark_results <- benchmark_synth()

# Export test data
export_test_data()

# Save results
dir.create("results", showWarnings = FALSE)

if (!is.null(benchmark_results)) {
  df <- do.call(rbind, lapply(names(benchmark_results), function(name) {
    r <- benchmark_results[[name]]
    data.frame(
      test_case = name,
      mean_ms = r$mean,
      median_ms = r$median,
      min_ms = r$min,
      max_ms = r$max,
      n_eval = r$neval
    )
  }))

  write.csv(df, "results/synth_benchmarks.csv", row.names = FALSE)
  cat("\nBenchmark results saved to results/synth_benchmarks.csv\n")
}

# Summary table for validation
cat("\n=== Validation Summary ===\n")
if (!is.null(simple_results)) {
  cat(sprintf("Simple test - Weight B: %.3f, Weight C: %.3f, Avg Effect: %.3f\n",
              simple_results$weight_b, simple_results$weight_c, simple_results$avg_effect))
}
if (!is.null(qp_results)) {
  cat(sprintf("QP test - Solution: [%.4f, %.4f]\n",
              qp_results$solution_2[1], qp_results$solution_2[2]))
}
if (!is.null(placebo_results)) {
  cat(sprintf("Placebo test - Treated rank: %d, p-value: %.3f\n",
              placebo_results$treated_rank, placebo_results$p_value))
}
