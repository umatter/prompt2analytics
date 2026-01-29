# ============================================================================
# tab_benchmark_summary.R
# Performance benchmark summary table (LaTeX)
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
library(xtable)
source("helpers.R")

## DATA IMPORT AND PREPARATION ----
benchmark_df <- load_benchmark_summary(paste0(INPUT_BENCHMARK, "benchmark_summary.json"))

# Filter to n=100,000 and format for table
table_data <- benchmark_df %>%
  filter(n == 100000 | n == max(n)) %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  mutate(
    Category = categorize_method(method),
    Method = clean_method_label(method),
    # Format columns
    `$n$` = format(n, big.mark = ","),
    `R (ms)` = sprintf("%.1f", r_median_us / 1000),
    `Rust (ms)` = sprintf("%.1f", rust_median_us / 1000),
    Speedup = sprintf("%.1f$\\times$", speedup)
  ) %>%
  arrange(Category, desc(speedup)) %>%
  select(Category, Method, `$n$`, `R (ms)`, `Rust (ms)`, Speedup)

## CREATE LATEX TABLE ----
latex_table <- xtable(
  table_data,
  caption = "Performance benchmark results at $n = 100{,}000$ (median execution time). Speedup = R time / Rust time. Methods with speedup $>1\\times$ are faster in Rust; methods $<1\\times$ are faster in R, typically due to CLI startup overhead dominating trivial computations.",
  label = "tab:benchmark-full",
  align = c("l", "l", "l", "r", "r", "r", "r")
)

## WRITE TO DISK ----
dir.create(OUTPUT_TABLES, showWarnings = FALSE, recursive = TRUE)

print(
  latex_table,
  file = paste0(OUTPUT_TABLES, "tab_benchmark_summary.tex"),
  include.rownames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = identity,
  caption.placement = "top",
  table.placement = "htbp",
  floating = TRUE
)

message("Created: ", OUTPUT_TABLES, "tab_benchmark_summary.tex")

## ALSO SAVE CSV FOR REFERENCE ----
write_csv(table_data, paste0(INPUT_BENCHMARK, "benchmark_table.csv"))
message("Created: ", INPUT_BENCHMARK, "benchmark_table.csv")
