#!/usr/bin/env Rscript
# Unified R Validation Runner
# Runs all validation scripts and generates structured output

# ============================================================================
# CONFIGURATION
# ============================================================================

cat("============================================================\n")
cat("p2a-core R Validation Runner\n")
cat("============================================================\n\n")

# Get script directory
args <- commandArgs(trailingOnly = TRUE)
script_dir <- if (length(args) > 0) args[1] else "."

# Track results
results <- data.frame(
  script = character(),
  status = character(),
  duration_sec = numeric(),
  error = character(),
  stringsAsFactors = FALSE
)

# Required packages check
required_packages <- c(
  "lmtest", "sandwich", "MASS", "nnet", "survival", "plm",
  "pscl", "AER", "car", "tseries", "forecast"
)

cat("Checking required packages...\n")
missing_packages <- c()
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}

if (length(missing_packages) > 0) {
  cat("Warning: Missing packages:", paste(missing_packages, collapse = ", "), "\n")
  cat("Install with: install.packages(c('", paste(missing_packages, collapse = "', '"), "'))\n\n")
} else {
  cat("All required packages available.\n\n")
}

# ============================================================================
# FIND VALIDATION SCRIPTS
# ============================================================================

validation_scripts <- list.files(
  path = script_dir,
  pattern = ".*validation\\.R$|^validate_.*\\.R$",
  full.names = TRUE,
  recursive = FALSE
)

# Exclude this runner script
validation_scripts <- validation_scripts[!grepl("run_all_validations\\.R$", validation_scripts)]

cat(sprintf("Found %d validation scripts\n\n", length(validation_scripts)))

# ============================================================================
# RUN EACH SCRIPT
# ============================================================================

for (script_path in validation_scripts) {
  script_name <- basename(script_path)
  cat(sprintf("Running: %s ... ", script_name))

  start_time <- Sys.time()
  error_msg <- ""
  status <- "PASS"

  tryCatch({
    # Run script in isolated environment
    source(script_path, local = new.env())
  }, error = function(e) {
    status <<- "FAIL"
    error_msg <<- conditionMessage(e)
  }, warning = function(w) {
    # Warnings don't fail the test
  })

  end_time <- Sys.time()
  duration <- as.numeric(difftime(end_time, start_time, units = "secs"))

  if (status == "PASS") {
    cat(sprintf("OK (%.2fs)\n", duration))
  } else {
    cat(sprintf("FAILED (%.2fs)\n", duration))
    cat(sprintf("  Error: %s\n", substr(error_msg, 1, 100)))
  }

  results <- rbind(results, data.frame(
    script = script_name,
    status = status,
    duration_sec = round(duration, 2),
    error = substr(error_msg, 1, 200),
    stringsAsFactors = FALSE
  ))
}

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n============================================================\n")
cat("VALIDATION SUMMARY\n")
cat("============================================================\n\n")

passed <- sum(results$status == "PASS")
failed <- sum(results$status == "FAIL")
total <- nrow(results)

cat(sprintf("Passed: %d/%d (%.1f%%)\n", passed, total, 100 * passed / max(total, 1)))
cat(sprintf("Failed: %d/%d\n", failed, total))
cat(sprintf("Total time: %.1f seconds\n\n", sum(results$duration_sec)))

if (failed > 0) {
  cat("Failed scripts:\n")
  failed_scripts <- results[results$status == "FAIL", ]
  for (i in seq_len(nrow(failed_scripts))) {
    cat(sprintf("  - %s: %s\n", failed_scripts$script[i], failed_scripts$error[i]))
  }
  cat("\n")
}

# Save results to CSV
output_file <- file.path(dirname(script_dir), "reports",
                         paste0("r_validation_", format(Sys.time(), "%Y-%m-%d"), ".csv"))
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))

# Exit with appropriate code
if (failed > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}
