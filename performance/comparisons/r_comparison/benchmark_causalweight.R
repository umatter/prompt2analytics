#!/usr/bin/env Rscript
# Causal Weight Benchmarks for Cross-Language Comparison
# Compares R causalweight package against p2a Rust implementation
#
# Required packages:
#   install.packages(c("causalweight", "microbenchmark"))

library(microbenchmark)
library(causalweight)

# Ensure reproducibility
set.seed(42)

# ============================================================================
# Data Generation Functions
# ============================================================================

#' Generate treatment effects data (matching Rust benchmark DGP)
generate_treatment_data <- function(n, seed = 42) {
  set.seed(seed)

  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)

  # Propensity score: logit(0.5 + 0.3*x1 + 0.2*x2)
  ps <- plogis(0.5 + 0.3*x1 + 0.2*x2)
  treatment <- rbinom(n, 1, ps)

  # Outcome: true ATE = 2.0
  outcome <- 2.0*treatment + 1.0*x1 + 0.5*x2 + runif(n, -0.5, 0.5)

  data.frame(
    outcome = outcome,
    treatment = treatment,
    x1 = x1,
    x2 = x2
  )
}

#' Generate mediation data (matching Rust benchmark DGP)
generate_mediation_data <- function(n, seed = 42) {
  set.seed(seed)

  x <- runif(n, -1, 1)

  # Random treatment (50/50)
  treatment <- rbinom(n, 1, 0.5)

  # Mediator: M = 0.5*D + 0.3*X + noise
  mediator <- 0.5*treatment + 0.3*x + runif(n, -0.3, 0.3)

  # Outcome: Y = 0.4*D + 0.6*M + 0.2*X + noise
  outcome <- 0.4*treatment + 0.6*mediator + 0.2*x + runif(n, -0.5, 0.5)

  data.frame(
    outcome = outcome,
    treatment = treatment,
    mediator = mediator,
    x = x
  )
}

# ============================================================================
# Benchmarking Functions
# ============================================================================

#' Benchmark IPW treatment effects (treatweight)
benchmark_ipw <- function() {
  cat("=== IPW Treatment Effects Benchmarks ===\n")
  results <- list()

  for (n in c(200, 500, 1000, 2000)) {
    cat(sprintf("Benchmarking IPW with n=%d...\n", n))

    data <- generate_treatment_data(n)

    bm <- microbenchmark(
      treatweight(
        y = data$outcome,
        d = data$treatment,
        x = data[, c("x1", "x2")],
        boot = 99,  # Match Rust benchmark
        trim = 0.05
      ),
      times = 10,  # Fewer iterations due to bootstrap
      unit = "milliseconds"
    )

    results[[paste0("ipw_", n)]] <- summary(bm)

    # Also get point estimate for validation
    result <- treatweight(
      y = data$outcome,
      d = data$treatment,
      x = data[, c("x1", "x2")],
      boot = 99,
      trim = 0.05
    )
    cat(sprintf("  ATE estimate: %.4f (true: 2.0)\n", result$effect))
  }

  results
}

#' Benchmark mediation analysis (medweight)
benchmark_mediation <- function() {
  cat("\n=== Mediation Analysis Benchmarks ===\n")
  results <- list()

  for (n in c(200, 500, 1000)) {
    cat(sprintf("Benchmarking mediation with n=%d...\n", n))

    data <- generate_mediation_data(n)

    bm <- microbenchmark(
      medweight(
        y = data$outcome,
        d = data$treatment,
        m = data$mediator,
        x = as.matrix(data$x),
        boot = 99,  # Match Rust benchmark
        trim = 0.05
      ),
      times = 10,
      unit = "milliseconds"
    )

    results[[paste0("mediation_", n)]] <- summary(bm)

    # Get point estimates for validation
    result <- medweight(
      y = data$outcome,
      d = data$treatment,
      m = data$mediator,
      x = as.matrix(data$x),
      boot = 99,
      trim = 0.05
    )
    cat(sprintf("  Total effect: %.4f (expected ~0.7)\n", result$total))
    cat(sprintf("  Direct effect: %.4f (expected ~0.4)\n", result$dir0))
    cat(sprintf("  Indirect effect: %.4f (expected ~0.3)\n", result$indir0))
  }

  results
}

# ============================================================================
# Validation Functions
# ============================================================================

#' Generate and save test data for cross-validation with Rust
generate_validation_data <- function(output_dir = "validation_data") {
  dir.create(output_dir, showWarnings = FALSE)

  # Treatment effects data
  cat("Generating treatment effects validation data...\n")
  data_treatment <- generate_treatment_data(1000, seed = 42)
  write.csv(data_treatment, file.path(output_dir, "treatment_data.csv"), row.names = FALSE)

  result_ipw <- treatweight(
    y = data_treatment$outcome,
    d = data_treatment$treatment,
    x = data_treatment[, c("x1", "x2")],
    boot = 999,
    trim = 0.05
  )

  cat(sprintf("IPW ATE: %.6f, SE: %.6f\n", result_ipw$effect, result_ipw$se))

  # Mediation data
  cat("\nGenerating mediation validation data...\n")
  data_mediation <- generate_mediation_data(1000, seed = 42)
  write.csv(data_mediation, file.path(output_dir, "mediation_data.csv"), row.names = FALSE)

  result_med <- medweight(
    y = data_mediation$outcome,
    d = data_mediation$treatment,
    m = data_mediation$mediator,
    x = as.matrix(data_mediation$x),
    boot = 999,
    trim = 0.05
  )

  cat(sprintf("Mediation Total: %.6f\n", result_med$total))
  cat(sprintf("Mediation Direct (NDE): %.6f\n", result_med$dir0))
  cat(sprintf("Mediation Indirect (NIE): %.6f\n", result_med$indir0))

  # Save R results for comparison
  r_results <- data.frame(
    method = c("ipw_ate", "mediation_total", "mediation_direct", "mediation_indirect"),
    estimate = c(result_ipw$effect, result_med$total, result_med$dir0, result_med$indir0),
    se = c(result_ipw$se, NA, NA, NA)  # medweight doesn't return component SEs directly
  )
  write.csv(r_results, file.path(output_dir, "r_reference_results.csv"), row.names = FALSE)

  cat(sprintf("\nValidation data saved to %s/\n", output_dir))
}

# ============================================================================
# Main Execution
# ============================================================================

main <- function() {
  cat("========================================\n")
  cat("p2a Causalweight Comparison Benchmarks\n")
  cat("========================================\n\n")

  # Run benchmarks
  ipw_results <- benchmark_ipw()
  med_results <- benchmark_mediation()

  # Save timing results
  dir.create("results", showWarnings = FALSE)

  save_results <- function(results, filename) {
    df <- do.call(rbind, lapply(names(results), function(name) {
      r <- results[[name]]
      data.frame(
        method = name,
        mean_ms = r$mean,
        median_ms = r$median,
        min_ms = r$min,
        max_ms = r$max,
        n_eval = r$neval
      )
    }))
    write.csv(df, filename, row.names = FALSE)
    cat(sprintf("Results saved to %s\n", filename))
  }

  save_results(ipw_results, "results/causalweight_ipw.csv")
  save_results(med_results, "results/causalweight_mediation.csv")

  # Print summary
  cat("\n=== Timing Summary ===\n")
  cat("\nIPW Treatment Effects:\n")
  print(do.call(rbind, ipw_results))
  cat("\nMediation Analysis:\n")
  print(do.call(rbind, med_results))

  # Generate validation data
  cat("\n")
  generate_validation_data()
}

# Run if executed directly
if (interactive() || !exists("sourced_as_library")) {
  main()
}
