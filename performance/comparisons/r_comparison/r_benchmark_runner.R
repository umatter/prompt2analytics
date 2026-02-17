#!/usr/bin/env Rscript
# R Benchmark Runner — Orchestrator
#
# Runs all benchmark_*.R scripts that use bench::mark and collects results
# into a unified results/r_benchmarks_all.csv file.
#
# Usage: Rscript r_benchmark_runner.R [--skip-run] [--verbose]
#
# Options:
#   --skip-run   Skip running benchmarks, only merge existing results
#   --verbose    Print detailed output from each benchmark script

args <- commandArgs(trailingOnly = TRUE)
skip_run <- "--skip-run" %in% args
verbose <- "--verbose" %in% args

cat("=== R Benchmark Runner ===\n")
cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("Date: %s\n", Sys.time()))
cat(sprintf("Working directory: %s\n\n", getwd()))

# Ensure we're in the r_comparison directory
if (!file.exists("benchmark_comprehensive.R")) {
  # Try to cd to the right directory
  possible_dirs <- c(
    "performance/comparisons/r_comparison",
    "../performance/comparisons/r_comparison"
  )
  found <- FALSE
  for (d in possible_dirs) {
    if (dir.exists(d)) {
      setwd(d)
      found <- TRUE
      break
    }
  }
  if (!found) {
    stop("Cannot find benchmark directory. Run from project root or r_comparison/")
  }
}

# Ensure results directory exists
dir.create("results", showWarnings = FALSE)

# ============================================
# Phase 1: Run Benchmark Scripts
# ============================================

if (!skip_run) {
  # Find all benchmark scripts
  all_scripts <- sort(list.files(".", pattern = "^benchmark_.*\\.R$"))

  # Exclude the runner itself and any helper scripts
  exclude <- c("r_benchmark_runner.R", "merge_results.R", "analyze_results.R",
               "generate_latex_tables.R")
  scripts <- setdiff(all_scripts, exclude)

  cat(sprintf("Found %d benchmark scripts to run\n\n", length(scripts)))

  # Track success/failure
  results_log <- data.frame(
    script = character(),
    status = character(),
    duration_s = numeric(),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(scripts)) {
    script <- scripts[i]
    cat(sprintf("[%d/%d] Running %s ... ", i, length(scripts), script))

    start_time <- proc.time()

    tryCatch({
      if (verbose) {
        cat("\n")
        source(script, local = TRUE)
        cat("\n")
      } else {
        # Capture output to suppress verbose benchmark output
        capture.output(source(script, local = TRUE), type = "output")
      }

      duration <- (proc.time() - start_time)["elapsed"]
      cat(sprintf("OK (%.1fs)\n", duration))
      results_log <- rbind(results_log, data.frame(
        script = script, status = "success", duration_s = duration,
        stringsAsFactors = FALSE
      ))
    }, error = function(e) {
      duration <- (proc.time() - start_time)["elapsed"]
      cat(sprintf("FAILED (%.1fs): %s\n", duration, e$message))
      results_log <<- rbind(results_log, data.frame(
        script = script, status = "failed", duration_s = duration,
        stringsAsFactors = FALSE
      ))
    })
  }

  # Print run summary
  cat("\n=== Run Summary ===\n")
  n_success <- sum(results_log$status == "success")
  n_failed <- sum(results_log$status == "failed")
  cat(sprintf("Succeeded: %d / %d\n", n_success, nrow(results_log)))
  if (n_failed > 0) {
    cat(sprintf("Failed: %d\n", n_failed))
    failed_scripts <- results_log$script[results_log$status == "failed"]
    for (s in failed_scripts) {
      cat(sprintf("  - %s\n", s))
    }
  }
  cat(sprintf("Total time: %.1fs\n", sum(results_log$duration_s)))
}

# ============================================
# Phase 2: Merge All Results
# ============================================

cat("\n=== Merging Results ===\n")

# Find all result CSVs
csv_files <- list.files("results", pattern = "^r_.*\\.csv$", full.names = TRUE)

# Exclude merged files to avoid double-counting
csv_files <- csv_files[!grepl("r_benchmarks_all|comparison_|validation_", csv_files)]

if (length(csv_files) == 0) {
  cat("No result CSV files found.\n")
  quit(status = 0)
}

cat(sprintf("Found %d result CSV files\n", length(csv_files)))

# Standard columns
standard_cols <- c("method", "n", "iterations", "time_min_us", "time_p25_us",
                   "time_median_us", "time_p75_us", "time_max_us",
                   "time_mean_us", "time_std_us", "itr_per_sec",
                   "mem_alloc_bytes")

all_dfs <- list()
for (f in csv_files) {
  tryCatch({
    df <- read.csv(f, stringsAsFactors = FALSE)

    # Only include if it has the core columns
    if ("method" %in% names(df) && "time_median_us" %in% names(df)) {
      # Add missing columns
      for (col in standard_cols) {
        if (!col %in% names(df)) df[[col]] <- NA
      }
      df$source_file <- basename(f)
      all_dfs[[length(all_dfs) + 1]] <- df[, c(standard_cols, "source_file")]
    }
  }, error = function(e) {
    cat(sprintf("  Warning: Could not read %s: %s\n", basename(f), e$message))
  })
}

if (length(all_dfs) > 0) {
  merged <- do.call(rbind, all_dfs)

  # Remove rows with NA median times
  merged <- merged[!is.na(merged$time_median_us), ]

  # For duplicate method+n combinations, keep the most recent
  merged <- merged[order(merged$source_file, decreasing = TRUE), ]
  merged <- merged[!duplicated(paste(merged$method, merged$n)), ]

  # Sort by method name
  merged <- merged[order(merged$method, merged$n), ]

  # Save merged results
  output_file <- "results/r_benchmarks_all.csv"
  write.csv(merged, output_file, row.names = FALSE)

  cat(sprintf("\nMerged results saved to: %s\n", output_file))
  cat(sprintf("Total unique benchmarks: %d\n", nrow(merged)))
  cat(sprintf("Unique methods: %d\n", length(unique(merged$method))))

  # Print per-method summary
  cat("\n=== Methods Summary ===\n")
  cat(sprintf("%-30s  %5s  %12s  %12s\n", "Method", "N", "Median (us)", "Memory"))
  cat(paste(rep("-", 65), collapse = ""), "\n")

  for (m in sort(unique(merged$method))) {
    subset <- merged[merged$method == m, ]
    # Show the largest N for each method
    row <- subset[which.max(subset$n), ]
    mem_str <- if (is.na(row$mem_alloc_bytes)) {
      "N/A"
    } else if (row$mem_alloc_bytes < 1024) {
      sprintf("%d B", row$mem_alloc_bytes)
    } else if (row$mem_alloc_bytes < 1024^2) {
      sprintf("%.1f KB", row$mem_alloc_bytes / 1024)
    } else {
      sprintf("%.2f MB", row$mem_alloc_bytes / 1024^2)
    }
    cat(sprintf("%-30s  %5d  %12.1f  %12s\n", m, row$n, row$time_median_us, mem_str))
  }
} else {
  cat("No valid data to merge.\n")
}

cat("\n=== Runner Complete ===\n")
