#!/usr/bin/env Rscript
# Helper script to verify which files still need conversion
# Run: Rscript _convert_remaining.R

files <- list.files(".", pattern = "^benchmark_.*\\.R$")
for (f in sort(files)) {
  content <- readLines(f)
  has_bench <- any(grepl("library\\(bench\\)", content))
  has_micro <- any(grepl("microbenchmark", content))
  has_systime <- any(grepl("system\\.time|Sys\\.time", content)) && !has_bench && !has_micro

  status <- if (has_bench) "DONE (bench)"
            else if (has_micro) "NEEDS CONVERSION (microbenchmark)"
            else "NEEDS CONVERSION (system.time/Sys.time)"

  cat(sprintf("%-40s %s\n", f, status))
}
