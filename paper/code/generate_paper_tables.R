#!/usr/bin/env Rscript
# ============================================================================
# generate_paper_tables.R
# Generate all benchmark tables for the paper from unified benchmark data
#
# Reads from: performance/comparisons/r_comparison/results/
#   - comparison_unified.csv (speed + validation)
#   - unified_r_*.json (R memory data)
#   - rust_unified_*.json (Rust memory data)
# Outputs to: paper/tables/
#
# Usage: cd paper/code && Rscript generate_paper_tables.R
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(jsonlite)
})

# Paths
script_dir <- tryCatch(
  dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
  error = function(e) getwd()
)
paper_dir <- dirname(script_dir)
project_root <- dirname(paper_dir)
results_dir <- file.path(project_root, "performance/comparisons/r_comparison/results")
tables_dir <- file.path(paper_dir, "tables")

cat("=== Generating Paper Tables from Unified Benchmark Data ===\n")
cat("Results dir:", results_dir, "\n")
cat("Output dir:", tables_dir, "\n\n")

dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Load data
# ============================================================================

# Primary: unified CSV
unified_df <- read.csv(file.path(results_dir, "comparison_unified.csv"),
                       stringsAsFactors = FALSE)

matched <- unified_df %>%
  filter(!is.na(speedup) & is.finite(speedup))

# Standardize to n=1000 for clean apples-to-apples comparison
largest_n <- matched %>%
  filter(n == 1000) %>%
  group_by(method) %>%
  slice(1) %>%
  ungroup()

cat(sprintf("Unified data: %d entries, %d matched, %d unique methods\n",
            nrow(unified_df), nrow(matched), n_distinct(matched$method)))

# Memory: load from JSON files (latest of each)
load_latest_json <- function(pattern) {
  files <- sort(list.files(results_dir, pattern = pattern, full.names = TRUE),
                decreasing = TRUE)
  if (length(files) == 0) {
    warning(sprintf("No JSON files matching pattern '%s' found", pattern))
    return(data.frame())
  }
  cat(sprintf("  Loading: %s\n", basename(files[1])))
  raw <- fromJSON(files[1], flatten = TRUE)
  raw %>%
    select(method, any_of(c("variant", "n", "mem_alloc_bytes"))) %>%
    filter(!is.na(mem_alloc_bytes))
}

cat("Loading memory data from JSON files...\n")
r_mem <- load_latest_json("^unified_r_.*\\.json$") %>%
  rename(r_mem_bytes = mem_alloc_bytes)
rust_mem <- load_latest_json("^rust_unified_.*\\.json$") %>%
  rename(rust_mem_bytes = mem_alloc_bytes)

if (nrow(r_mem) > 0 && nrow(rust_mem) > 0) {
  join_cols <- intersect(c("method", "variant", "n"), intersect(names(r_mem), names(rust_mem)))
  mem_df <- inner_join(r_mem, rust_mem, by = join_cols) %>%
    filter(rust_mem_bytes > 0, n == 1000) %>%
    mutate(mem_ratio = r_mem_bytes / rust_mem_bytes)
  cat(sprintf("Memory data: %d matched entries (n=1000)\n\n", nrow(mem_df)))
} else {
  mem_df <- data.frame()
  cat("WARNING: No memory data available\n\n")
}

# Pretty module names
module_labels <- c(
  "regression" = "Regression", "panel" = "Panel", "discrete" = "Discrete",
  "timeseries" = "Time Series", "ml" = "ML", "causal" = "Causal",
  "stats" = "Stats", "survival" = "Survival", "treatment" = "Treatment"
)

# ============================================================================
# Helper functions
# ============================================================================
fmt_speedup <- function(x) {
  if (is.na(x) || !is.finite(x)) return("---")
  if (x > 1000) return(sprintf("$>$1000$\\times$"))
  if (x >= 100) return(sprintf("%.0f$\\times$", x))
  if (x >= 10) return(sprintf("%.1f$\\times$", x))
  return(sprintf("%.1f$\\times$", x))
}

fmt_n <- function(n) format(n, big.mark = ",")

fmt_us <- function(us) {
  if (is.na(us) || !is.finite(us)) return("---")
  if (us < 1) return("$<$1")
  return(format(round(us), big.mark = ","))
}

fmt_mem <- function(bytes) {
  if (is.na(bytes) || !is.finite(bytes)) return("---")
  if (bytes < 1024) return(sprintf("%d B", round(bytes)))
  if (bytes < 1024 * 1024) return(sprintf("%.1f KB", bytes / 1024))
  return(sprintf("%.1f MB", bytes / (1024 * 1024)))
}

# ============================================================================
# Table 1: Benchmark Summary (tab_benchmark_summary.tex)
# ============================================================================
cat("Generating tab_benchmark_summary.tex...\n")

# Select ~25 representative methods across all modules
representative_methods <- c(
  # Regression
  "OLS", "OLS_HC3", "NLS", "LOESS", "GLS",
  # Panel
  "FixedEffects", "RandomEffects", "Hausman", "HDFE",
  # Discrete
  "Logit", "Probit", "Poisson",
  # Causal
  "DiD", "IV_2SLS",
  # ML
  "K-Means", "PCA", "DBSCAN",
  # Time Series
  "ARIMA", "MSTL", "Holt_Winters",
  # Survival
  "CoxPH", "KM",
  # Stats
  "t_test", "ANOVA",
  # Treatment
  "IPW", "Matching"
)

tab <- matched %>%
  filter(method %in% representative_methods) %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  slice(1) %>%
  ungroup()

# Join memory data if available
if (nrow(mem_df) > 0) {
  mem_largest <- mem_df %>%
    group_by(method) %>%
    filter(n == max(n)) %>%
    slice(1) %>%
    ungroup() %>%
    select(method, n, r_mem_bytes, rust_mem_bytes)

  tab <- tab %>%
    left_join(mem_largest, by = c("method", "n"))
} else {
  tab$r_mem_bytes <- NA_real_
  tab$rust_mem_bytes <- NA_real_
}

tab <- tab %>% arrange(module, desc(speedup))

missing_rep <- setdiff(representative_methods, tab$method)
if (length(missing_rep) > 0) {
  warning(sprintf("Missing %d representative methods from benchmark data: %s",
                  length(missing_rep), paste(missing_rep, collapse = ", ")))
}

# Build LaTeX table
tex <- c(
  "% Benchmark summary table -- auto-generated by generate_paper_tables.R",
  sprintf("%% Generated: %s", Sys.time()),
  "\\begin{table}[htbp]",
  "\\centering",
  paste0("\\caption{Performance benchmark results across representative methods ",
         "(median execution time). Speedup = R time / Rust time. Sample sizes ",
         "vary by method to reflect typical use cases ($n = 100$--$10{,}000$). ",
         "Memory columns show per-call allocation from R and Rust respectively.}"),
  "\\label{tab:benchmark-full}",
  "\\renewcommand{\\arraystretch}{1.15}",
  "\\small",
  "\\begin{tabular}{llrrrrrr}",
  "  \\toprule",
  "Category & Method & $n$ & R ($\\mu$s) & Rust ($\\mu$s) & Speedup & R Mem & Rust Mem \\\\",
  "  \\midrule"
)

# Group by module
last_module <- ""
for (i in 1:nrow(tab)) {
  r <- tab[i, ]

  # Add midrule between modules
  if (r$module != last_module && last_module != "") {
    tex <- c(tex, "\\midrule")
  }
  last_module <- r$module

  # Clean method name for display
  method_clean <- gsub("_", " ", r$method)
  method_clean <- gsub("HDFE", "HDFE (2-way)", method_clean)
  method_clean <- gsub("CoxPH", "Cox PH", method_clean)
  method_clean <- gsub("OLS HC3", "OLS + HC3", method_clean)
  method_clean <- gsub("K-Means", "K-Means ($k=3$)", method_clean)
  method_clean <- gsub("t test", "$t$-test", method_clean)
  method_clean <- gsub("IV 2SLS", "IV/2SLS", method_clean)
  method_clean <- gsub("Holt Winters", "Holt--Winters", method_clean)
  method_clean <- gsub("ARIMA", "ARIMA(1,1,1)", method_clean)
  method_clean <- gsub("KM", "Kaplan--Meier", method_clean)

  # Pretty module name
  mod_pretty <- ifelse(r$module %in% names(module_labels),
                       module_labels[r$module], r$module)

  tex <- c(tex, sprintf("%s & %s & %s & %s & %s & %s & %s & %s \\\\",
                        mod_pretty,
                        method_clean,
                        fmt_n(r$n),
                        fmt_us(r$r_median_us),
                        fmt_us(r$rust_median_us),
                        fmt_speedup(r$speedup),
                        fmt_mem(r$r_mem_bytes),
                        fmt_mem(r$rust_mem_bytes)))
}

tex <- c(tex,
  "   \\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(tex, file.path(tables_dir, "tab_benchmark_summary.tex"))
cat("  Saved tab_benchmark_summary.tex\n")

# ============================================================================
# Table 2: Speedup Summary by Module (tab_speedup_by_module.tex)
# ============================================================================
cat("Generating tab_speedup_by_module.tex...\n")

summary_mod <- largest_n %>%
  mutate(module_pretty = ifelse(module %in% names(module_labels),
                                module_labels[module], module)) %>%
  group_by(module_pretty) %>%
  summarise(
    Methods = n_distinct(method),
    Benchmarks = n(),
    Min = min(speedup),
    Median = median(speedup),
    Mean = mean(speedup),
    Max = max(speedup),
    .groups = "drop"
  ) %>%
  arrange(desc(Median))

tex2 <- c(
  "% Speedup summary by module -- auto-generated by generate_paper_tables.R",
  sprintf("%% Generated: %s", Sys.time()),
  "\\begin{table}[htbp]",
  "\\centering",
  "\\caption{Speedup summary by method category (at largest matched sample size per method). Speedup factor = R median time / Rust median time.}",
  "\\label{tab:speedup-summary}",
  "\\renewcommand{\\arraystretch}{1.15}",
  "\\small",
  "\\begin{tabular}{lrrcccc}",
  "\\toprule",
  "Module & Methods & Benchmarks & Min & Median & Mean & Max \\\\",
  "\\midrule"
)

for (i in 1:nrow(summary_mod)) {
  r <- summary_mod[i, ]
  tex2 <- c(tex2, sprintf("%s & %d & %d & %s & %s & %s & %s \\\\",
                          r$module_pretty, r$Methods, r$Benchmarks,
                          fmt_speedup(r$Min), fmt_speedup(r$Median),
                          fmt_speedup(r$Mean), fmt_speedup(r$Max)))
}

# Overall row
all_s <- largest_n$speedup
tex2 <- c(tex2,
  "\\midrule",
  sprintf("\\textbf{Overall} & \\textbf{%d} & \\textbf{%d} & %s & \\textbf{%s} & %s & %s \\\\",
          n_distinct(largest_n$method), nrow(largest_n),
          fmt_speedup(min(all_s)), fmt_speedup(median(all_s)),
          fmt_speedup(mean(all_s)), fmt_speedup(max(all_s))),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(tex2, file.path(tables_dir, "tab_speedup_by_module.tex"))
cat("  Saved tab_speedup_by_module.tex\n")

# ============================================================================
# Table 3: Validation Summary (tab_validation_summary.tex)
# ============================================================================
cat("Generating tab_validation_summary.tex...\n")

# Compute validation status per method at largest n
val_data <- unified_df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    module_pretty = ifelse(module %in% names(module_labels),
                           module_labels[module], module),
    status = case_when(
      is.na(outputs_agree) ~ "speed_only",
      outputs_agree == TRUE ~ "agree",
      outputs_agree == FALSE ~ "disagree"
    )
  )

# Also get median speedup from matched data for each module
mod_speedup <- largest_n %>%
  mutate(module_pretty = ifelse(module %in% names(module_labels),
                                module_labels[module], module)) %>%
  group_by(module_pretty) %>%
  summarise(median_speedup = median(speedup), .groups = "drop")

val_summary <- val_data %>%
  group_by(module_pretty) %>%
  summarise(
    Methods = n_distinct(method),
    Comparisons = sum(status != "speed_only"),
    Agree = sum(status == "agree"),
    Disagree = sum(status == "disagree"),
    Speed_only = sum(status == "speed_only"),
    .groups = "drop"
  ) %>%
  left_join(mod_speedup, by = "module_pretty") %>%
  arrange(desc(Agree))

tex3 <- c(
  "% Validation summary by module -- auto-generated by generate_paper_tables.R",
  sprintf("%% Generated: %s", Sys.time()),
  "\\begin{table}[htbp]",
  "\\centering",
  paste0("\\caption{Output validation summary by module. ``Agree'' indicates methods where ",
         "Rust and R outputs match within tolerance. ``Speed-only'' indicates methods ",
         "where only timing was compared (no output comparison available).}"),
  "\\label{tab:validation-summary}",
  "\\renewcommand{\\arraystretch}{1.15}",
  "\\small",
  "\\begin{tabular}{lrrrrrr}",
  "\\toprule",
  "Module & Methods & Comparisons & Agree & Disagree & Speed-only & Median Speedup \\\\",
  "\\midrule"
)

for (i in 1:nrow(val_summary)) {
  r <- val_summary[i, ]
  tex3 <- c(tex3, sprintf("%s & %d & %d & %d & %d & %d & %s \\\\",
                          r$module_pretty, r$Methods, r$Comparisons,
                          r$Agree, r$Disagree, r$Speed_only,
                          fmt_speedup(r$median_speedup)))
}

# Overall row
tex3 <- c(tex3,
  "\\midrule",
  sprintf("\\textbf{Overall} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%s} \\\\",
          sum(val_summary$Methods), sum(val_summary$Comparisons),
          sum(val_summary$Agree), sum(val_summary$Disagree),
          sum(val_summary$Speed_only),
          fmt_speedup(median(largest_n$speedup))),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}"
)

writeLines(tex3, file.path(tables_dir, "tab_validation_summary.tex"))
cat("  Saved tab_validation_summary.tex\n")

# ============================================================================
# Table 4: Memory Summary by Module (tab_memory_summary.tex)
# ============================================================================
cat("Generating tab_memory_summary.tex...\n")

if (nrow(mem_df) > 0) {
  # Join module info from unified CSV
  method_modules <- unified_df %>%
    select(method, module) %>%
    distinct()

  mem_with_module <- mem_df %>%
    group_by(method) %>%
    filter(n == max(n)) %>%
    slice(1) %>%
    ungroup() %>%
    left_join(method_modules, by = "method") %>%
    filter(!is.na(module)) %>%
    mutate(module_pretty = ifelse(module %in% names(module_labels),
                                  module_labels[module], module))

  mem_summary <- mem_with_module %>%
    group_by(module_pretty) %>%
    summarise(
      Methods = n_distinct(method),
      Median_R = median(r_mem_bytes),
      Median_Rust = median(rust_mem_bytes),
      Median_Ratio = median(mem_ratio),
      .groups = "drop"
    ) %>%
    arrange(desc(Median_Ratio))

  tex4 <- c(
    "% Memory summary by module -- auto-generated by generate_paper_tables.R",
    sprintf("%% Generated: %s", Sys.time()),
    "\\begin{table}[htbp]",
    "\\centering",
    paste0("\\caption{Memory allocation comparison by module (at largest sample size per method). ",
           "Shows median per-call allocation for R and Rust, and the ratio (R / Rust).}"),
    "\\label{tab:memory-summary}",
    "\\renewcommand{\\arraystretch}{1.15}",
    "\\small",
    "\\begin{tabular}{lrrrr}",
    "\\toprule",
    "Module & Methods & Median R Alloc & Median Rust Alloc & Median Ratio \\\\",
    "\\midrule"
  )

  for (i in 1:nrow(mem_summary)) {
    r <- mem_summary[i, ]
    tex4 <- c(tex4, sprintf("%s & %d & %s & %s & %s \\\\",
                            r$module_pretty, r$Methods,
                            fmt_mem(r$Median_R), fmt_mem(r$Median_Rust),
                            fmt_speedup(r$Median_Ratio)))
  }

  # Overall row
  tex4 <- c(tex4,
    "\\midrule",
    sprintf("\\textbf{Overall} & \\textbf{%d} & %s & %s & \\textbf{%s} \\\\",
            sum(mem_summary$Methods),
            fmt_mem(median(mem_with_module$r_mem_bytes)),
            fmt_mem(median(mem_with_module$rust_mem_bytes)),
            fmt_speedup(median(mem_with_module$mem_ratio))),
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(tex4, file.path(tables_dir, "tab_memory_summary.tex"))
  cat("  Saved tab_memory_summary.tex\n")
} else {
  cat("  WARNING: No memory data available, skipping memory table\n")
}

# ============================================================================
# Summary
# ============================================================================
cat("\n=== Summary Statistics ===\n")
cat(sprintf("Matched methods: %d\n", n_distinct(matched$method)))
cat(sprintf("Overall median speedup: %.1fx\n", median(largest_n$speedup)))
cat(sprintf("Validation: %d agree, %d disagree, %d speed-only\n",
            sum(val_data$status == "agree"),
            sum(val_data$status == "disagree"),
            sum(val_data$status == "speed_only")))
if (nrow(mem_df) > 0) {
  cat(sprintf("Memory: %d matched, median ratio %.0fx\n",
              nrow(mem_with_module),
              median(mem_with_module$mem_ratio, na.rm = TRUE)))
}

cat("\n=== Tables Generated ===\n")
tex_files <- list.files(tables_dir, pattern = "\\.tex$")
for (f in sort(tex_files)) {
  cat(sprintf("  %s\n", f))
}
cat("\nDone.\n")
