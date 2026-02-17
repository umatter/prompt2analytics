#!/usr/bin/env Rscript
# Batch converter: Transforms microbenchmark/system.time scripts to bench::mark
# This handles the most common patterns automatically.

convert_file <- function(filepath) {
  lines <- readLines(filepath)
  content <- paste(lines, collapse = "\n")

  # Skip if already converted
  if (grepl("library\\(bench\\)", content)) {
    cat(sprintf("  SKIP (already converted): %s\n", filepath))
    return(FALSE)
  }

  # Extract the script base name for CSV output
  script_name <- sub("benchmark_", "", sub("\\.R$", "", basename(filepath)))

  # === Step 1: Replace library(microbenchmark) with library(bench) ===
  content <- gsub("library\\(microbenchmark\\)", "library(bench)", content)

  # === Step 2: Remove has_microbenchmark / use_microbenchmark checks and fallback blocks ===
  # Remove common patterns:
  # - has_microbenchmark <- requireNamespace(...)
  # - use_microbenchmark <- require(...)
  # - if (has_microbenchmark) { library(microbenchmark) }
  # - if (use_microbenchmark) library(microbenchmark)
  content <- gsub("has_microbenchmark <- requireNamespace\\(\"microbenchmark\", quietly = TRUE\\)\n?", "", content)
  content <- gsub("use_microbenchmark <- requireNamespace\\(\"microbenchmark\", quietly = TRUE\\)\n?", "", content)
  content <- gsub("use_microbenchmark <- require\\(microbenchmark, quietly = TRUE\\)\n?", "", content)
  content <- gsub("has_microbenchmark <- require\\(microbenchmark, quietly = TRUE\\)\n?", "", content)

  # Remove if (has_microbenchmark) { library(microbenchmark) } blocks
  content <- gsub("if \\(has_microbenchmark\\) \\{\n\\s*library\\(microbenchmark\\)\n\\}\n?", "", content)

  # Remove microbenchmark availability checks
  content <- gsub("if \\(!requireNamespace\\(\"microbenchmark\", quietly = TRUE\\)\\) \\{[^}]*\\}\n?", "", content)

  # Remove suppressPackageStartupMessages blocks for microbenchmark
  content <- gsub("suppressPackageStartupMessages\\(\\{\n\\s*if \\(requireNamespace\\(\"microbenchmark\"[^}]*\\}[^}]*\\}\\)\n?", "library(bench)\n", content)

  # === Step 3: Add run_bench function after the first library() call ===
  run_bench_fn <- '
run_bench <- function(name, fn, iterations = 100) {
  result <- bench::mark(fn(), iterations = iterations, check = FALSE, memory = TRUE, filter_gc = FALSE)
  raw_times <- result$time[[1]]
  times_us <- as.numeric(raw_times) * 1e6
  mem_alloc <- as.numeric(result$mem_alloc[[1]])
  list(method = name, n = NA, iterations = length(times_us),
    time_min_us = min(times_us), time_p25_us = quantile(times_us, 0.25),
    time_median_us = median(times_us), time_p75_us = quantile(times_us, 0.75),
    time_max_us = max(times_us), time_mean_us = mean(times_us),
    time_std_us = sd(times_us), itr_per_sec = 1e6 / median(times_us),
    mem_alloc_bytes = mem_alloc)
}
'

  # Add library(bench) if not present and run_bench function
  if (!grepl("library\\(bench\\)", content)) {
    # Add after set.seed or at the top
    if (grepl("set\\.seed\\(", content)) {
      content <- sub("(set\\.seed\\([0-9]+\\))", paste0("library(bench)\n\n\\1\n", run_bench_fn), content)
    } else {
      content <- paste0("library(bench)\n", run_bench_fn, "\n", content)
    }
  } else if (!grepl("run_bench", content)) {
    # library(bench) exists but no run_bench
    content <- sub("(library\\(bench\\))", paste0("\\1\n", run_bench_fn), content)
  }

  writeLines(content, filepath)
  cat(sprintf("  PARTIAL: %s (added bench + run_bench, manual review needed)\n", filepath))
  return(TRUE)
}

# Find all files needing conversion
files <- list.files(".", pattern = "^benchmark_.*\\.R$", full.names = TRUE)
converted <- 0
for (f in sort(files)) {
  content <- paste(readLines(f), collapse = "\n")
  if (!grepl("library\\(bench\\)", content)) {
    if (convert_file(f)) converted <- converted + 1
  }
}

cat(sprintf("\nPartially converted %d files. Manual review needed for benchmark logic.\n", converted))
