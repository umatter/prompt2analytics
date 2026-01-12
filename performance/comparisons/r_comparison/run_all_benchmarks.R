#!/usr/bin/env Rscript
# Master script to run all R benchmarks
# Usage: Rscript run_all_benchmarks.R

cat("======================================\n")
cat("  p2a Cross-Language Benchmark Suite  \n")
cat("         R Reference Benchmarks       \n")
cat("======================================\n\n")

# Record start time
start_time <- Sys.time()

# Get script directory
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
script_dir <- if (length(script_path) > 0) dirname(script_path) else "."
if (script_dir == "") script_dir <- "."

# Change to script directory
setwd(script_dir)

# Check required packages
required_packages <- c("microbenchmark", "sandwich", "plm", "lfe", "data.table", "dplyr", "tidyr")
optional_packages <- c("forecast", "dbscan", "changepoint")

cat("Checking required packages...\n")
missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  stop(sprintf("Missing required packages: %s\nInstall with: install.packages(c('%s'))",
               paste(missing, collapse = ", "),
               paste(missing, collapse = "', '")))
}

cat("Checking optional packages...\n")
for (pkg in optional_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  [OK] %s\n", pkg))
  } else {
    cat(sprintf("  [MISSING] %s (some benchmarks will be skipped)\n", pkg))
  }
}

# Create results directory
dir.create("results", showWarnings = FALSE)

# Run benchmark scripts
scripts <- c(
  "benchmark_regression.R",
  "benchmark_econometrics.R",
  "benchmark_ml.R",
  "benchmark_forecasting.R",
  "benchmark_munging.R"
)

for (script in scripts) {
  cat(sprintf("\n\n========== Running %s ==========\n", script))
  if (file.exists(script)) {
    tryCatch(
      source(script),
      error = function(e) {
        cat(sprintf("Error in %s: %s\n", script, e$message))
      }
    )
  } else {
    cat(sprintf("Script not found: %s\n", script))
  }
}

# Combine all results
cat("\n\n========== Combining Results ==========\n")

csv_files <- list.files("results", pattern = "\\.csv$", full.names = TRUE)
if (length(csv_files) > 0) {
  all_results <- do.call(rbind, lapply(csv_files, function(f) {
    df <- read.csv(f)
    df$source_file <- basename(f)
    df
  }))

  write.csv(all_results, "results/combined_results.csv", row.names = FALSE)
  cat("Combined results saved to results/combined_results.csv\n")

  cat("\nAll results:\n")
  print(all_results)
}

# Record end time and print summary
end_time <- Sys.time()
duration <- difftime(end_time, start_time, units = "secs")

cat(sprintf("\n\n========== Benchmark Complete ==========\n"))
cat(sprintf("Total time: %.2f seconds\n", as.numeric(duration)))
cat(sprintf("Results saved to: %s/results/\n", getwd()))

# Print system info
cat("\nSystem Information:\n")
cat(sprintf("  R version: %s\n", R.version.string))
cat(sprintf("  Platform: %s\n", R.version$platform))
cat(sprintf("  Date: %s\n", Sys.Date()))
