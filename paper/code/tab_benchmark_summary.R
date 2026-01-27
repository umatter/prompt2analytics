# ============================================================================
# tab_benchmark_summary.R
# Performance benchmark summary table (LaTeX)
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
library(xtable)

INPUT <- "rust_validation/results/summaries/"
OUTPUT <- "../tables/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "benchmark_summary.json"), simplifyVector = FALSE)

# Extract methods data and flatten (handle type inconsistencies)
methods_list <- summary_data$methods
benchmark_df <- map_dfr(names(methods_list), function(method) {
  method_results <- methods_list[[method]]
  map_dfr(method_results, function(r) {
    tibble(
      method = method,
      n = r$n,
      r_median_us = r$r_median_us,
      rust_median_us = r$rust_median_us,
      speedup = r$speedup
    )
  })
})

# Filter to n=100,000 and format for table
table_data <- benchmark_df %>%
  filter(n == 100000 | n == max(n)) %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  mutate(
    # Add category labels
    Category = case_when(
      grepl("^ols", method, ignore.case = TRUE) ~ "Regression",
      grepl("^panel", method, ignore.case = TRUE) ~ "Panel",
      grepl("logit|probit", method, ignore.case = TRUE) ~ "Discrete",
      grepl("kmeans|pca|dbscan|hierarchical", method, ignore.case = TRUE) ~ "ML",
      grepl("sort|filter|group|select|standardize|lag|lead|diff", method, ignore.case = TRUE) ~ "Munging",
      TRUE ~ "Other"
    ),
    # Clean method labels
    Method = method %>%
      str_replace_all("_", " ") %>%
      str_replace_all("hc([0-3])", "HC\\1") %>%
      str_replace("^ols$", "OLS") %>%
      str_replace("^ols ", "OLS+") %>%
      str_replace("panel fe", "Panel FE") %>%
      str_replace("panel re", "Panel RE") %>%
      str_to_title(),
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
dir.create(OUTPUT, showWarnings = FALSE, recursive = TRUE)

print(
  latex_table,
  file = paste0(OUTPUT, "tab_benchmark_summary.tex"),
  include.rownames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = identity,
  caption.placement = "top",
  table.placement = "htbp",
  floating = TRUE
)

message("Created: ", OUTPUT, "tab_benchmark_summary.tex")

## ALSO SAVE CSV FOR REFERENCE ----
write_csv(table_data, paste0(INPUT, "benchmark_table.csv"))
message("Created: ", INPUT, "benchmark_table.csv")
