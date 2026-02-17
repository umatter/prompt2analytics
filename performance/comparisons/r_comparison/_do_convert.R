#!/usr/bin/env Rscript
# Automated conversion script: converts microbenchmark/system.time scripts to bench::mark
# This reads each unconverted script, applies pattern transformations, and writes back.

library(stringr)

run_bench_template <- '
# Standardized benchmark runner (bench::mark with memory tracking)
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

csv_output_template <- function(script_name) {
  sprintf('
# Save results to standardized CSV
results_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(method = r$method, n = r$n, iterations = r$iterations,
    time_min_us = r$time_min_us, time_p25_us = r$time_p25_us,
    time_median_us = r$time_median_us, time_p75_us = r$time_p75_us,
    time_max_us = r$time_max_us, time_mean_us = r$time_mean_us,
    time_std_us = r$time_std_us, itr_per_sec = r$itr_per_sec,
    mem_alloc_bytes = r$mem_alloc_bytes)
}))
dir.create("results", showWarnings = FALSE)
timestamp <- format(Sys.time(), "%%Y%%m%%d_%%H%%M%%S")
write.csv(results_df, sprintf("results/r_%s_%%s.csv", timestamp), row.names = FALSE)
', script_name)
}

# Just list which files need conversion
files <- list.files(".", pattern = "^benchmark_.*\\.R$")
to_convert <- c()
for (f in sort(files)) {
  content <- paste(readLines(f), collapse = "\n")
  has_bench <- grepl("library\\(bench\\)", content)
  if (!has_bench) {
    to_convert <- c(to_convert, f)
  }
}

cat("Files needing conversion:\n")
for (f in to_convert) {
  cat(sprintf("  %s\n", f))
}
cat(sprintf("\nTotal: %d files\n", length(to_convert)))
