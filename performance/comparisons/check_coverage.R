#!/usr/bin/env Rscript
# Check benchmark and validation coverage
# Reports methods with missing Rust or R benchmarks

suppressPackageStartupMessages(library(dplyr))

# Paths
script_dir <- tryCatch(
  dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
  error = function(e) getwd()
)
results_dir <- file.path(script_dir, "r_comparison/results")

speed_file <- file.path(results_dir, "comparison_speed.csv")
coverage_file <- file.path(results_dir, "validation_coverage.csv")

cat("=== Benchmark & Validation Coverage Check ===\n\n")

# --------------------------------------------------------------------------
# 1. Speed comparison coverage
# --------------------------------------------------------------------------
if (file.exists(speed_file)) {
  speed <- read.csv(speed_file, stringsAsFactors = FALSE)

  matched <- speed %>% filter(!is.na(speedup_median) & is.finite(speedup_median))
  r_only <- speed %>% filter(!is.na(r_median_us) & (is.na(rust_median_us) | !is.finite(rust_median_us)))
  rust_only <- speed %>% filter(!is.na(rust_median_us) & (is.na(r_median_us) | !is.finite(r_median_us)))

  cat(sprintf("Speed comparison: %d total entries, %d matched\n", nrow(speed), nrow(matched)))
  cat(sprintf("  Unique matched methods: %d\n", n_distinct(matched$method_norm)))

  if (nrow(r_only) > 0) {
    cat(sprintf("\n  Methods with R benchmark but NO Rust benchmark (%d):\n", n_distinct(r_only$method_norm)))
    for (m in sort(unique(r_only$method_norm))) {
      cat(sprintf("    - %s\n", m))
    }
  }

  if (nrow(rust_only) > 0) {
    cat(sprintf("\n  Methods with Rust benchmark but NO R benchmark (%d):\n", n_distinct(rust_only$method_norm)))
    for (m in sort(unique(rust_only$method_norm))) {
      cat(sprintf("    - %s\n", m))
    }
  }

  # Module-level coverage
  cat("\n  Coverage by module:\n")
  matched %>%
    group_by(module) %>%
    summarise(
      methods = n_distinct(method_norm),
      benchmarks = n(),
      .groups = "drop"
    ) %>%
    arrange(desc(methods)) %>%
    {for (i in 1:nrow(.)) {
      cat(sprintf("    %-15s: %2d methods, %3d benchmarks\n",
                  .$module[i], .$methods[i], .$benchmarks[i]))
    }}
} else {
  cat("  WARNING: comparison_speed.csv not found at", speed_file, "\n")
}

# --------------------------------------------------------------------------
# 2. Validation coverage
# --------------------------------------------------------------------------
cat("\n")
if (file.exists(coverage_file)) {
  coverage <- read.csv(coverage_file, stringsAsFactors = FALSE)

  cat(sprintf("Validation coverage: %d methods\n", nrow(coverage)))

  if ("has_benchmark" %in% names(coverage) && "has_validation" %in% names(coverage)) {
    no_bench <- coverage %>% filter(!has_benchmark)
    no_val <- coverage %>% filter(!has_validation)

    if (nrow(no_bench) > 0) {
      cat(sprintf("\n  Methods in validation but NOT benchmarked (%d):\n", nrow(no_bench)))
      for (m in sort(no_bench$method)) {
        cat(sprintf("    - %s\n", m))
      }
    }

    if (nrow(no_val) > 0) {
      cat(sprintf("\n  Methods benchmarked but NOT validated (%d):\n", nrow(no_val)))
      for (m in sort(no_val$method)) {
        cat(sprintf("    - %s\n", m))
      }
    }
  } else {
    # Just report what columns exist
    cat("  Columns:", paste(names(coverage), collapse = ", "), "\n")
    cat("  (Adjust coverage check logic for actual column names)\n")
  }
} else {
  cat("  WARNING: validation_coverage.csv not found at", coverage_file, "\n")
}

cat("\nDone.\n")
