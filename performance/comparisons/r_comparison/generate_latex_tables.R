#!/usr/bin/env Rscript
# Generate LaTeX Tables for Paper
#
# Produces publication-ready LaTeX table fragments from benchmark results.
#
# Output: results/figures/latex_*.tex
#
# Usage: Rscript generate_latex_tables.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

cat("=== Generating LaTeX Tables ===\n\n")

fig_dir <- "results/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================
# Helper: Format number for LaTeX
# ============================================

fmt_speedup <- function(x) {
  if (is.na(x) || !is.finite(x)) return("---")
  if (x >= 100) return(sprintf("%.0f$\\times$", x))
  if (x >= 10) return(sprintf("%.1f$\\times$", x))
  return(sprintf("%.2f$\\times$", x))
}

fmt_time <- function(us) {
  if (is.na(us) || !is.finite(us)) return("---")
  if (us < 1) return(sprintf("%.2f~ns", us * 1000))
  if (us < 1000) return(sprintf("%.1f~\\textmu s", us))
  if (us < 1e6) return(sprintf("%.2f~ms", us / 1000))
  return(sprintf("%.2f~s", us / 1e6))
}

fmt_mem <- function(bytes) {
  if (is.na(bytes) || !is.finite(bytes)) return("---")
  bytes <- abs(bytes)
  if (bytes < 1024) return(sprintf("%d~B", bytes))
  if (bytes < 1024^2) return(sprintf("%.1f~KB", bytes / 1024))
  return(sprintf("%.2f~MB", bytes / 1024^2))
}

# ============================================
# Table A: Speedup Summary by Module
# ============================================

speed_file <- "results/comparison_speed.csv"
if (file.exists(speed_file)) {
  cat("Generating Table A: Speedup Summary...\n")

  speed_df <- read.csv(speed_file, stringsAsFactors = FALSE)

  summary <- speed_df %>%
    filter(!is.na(speedup_median) & is.finite(speedup_median)) %>%
    group_by(module) %>%
    summarise(
      Methods = n_distinct(method_norm),
      Benchmarks = n(),
      Min = min(speedup_median),
      Median = median(speedup_median),
      Mean = mean(speedup_median),
      Max = max(speedup_median),
      .groups = "drop"
    ) %>%
    arrange(desc(Median))

  # Generate LaTeX
  tex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Speedup summary by method category. Speedup factor = R median time / Rust median time.}",
    "\\label{tab:speedup-summary}",
    "\\begin{tabular}{lrrcccc}",
    "\\toprule",
    "Module & Methods & Benchmarks & Min & Median & Mean & Max \\\\",
    "\\midrule"
  )

  for (i in 1:nrow(summary)) {
    r <- summary[i, ]
    tex <- c(tex, sprintf("%s & %d & %d & %s & %s & %s & %s \\\\",
                          r$module, r$Methods, r$Benchmarks,
                          fmt_speedup(r$Min), fmt_speedup(r$Median),
                          fmt_speedup(r$Mean), fmt_speedup(r$Max)))
  }

  # Overall
  all_s <- speed_df$speedup_median[!is.na(speed_df$speedup_median) & is.finite(speed_df$speedup_median)]
  tex <- c(tex,
    "\\midrule",
    sprintf("\\textbf{Overall} & \\textbf{%d} & \\textbf{%d} & %s & \\textbf{%s} & %s & %s \\\\",
            n_distinct(speed_df$method_norm[!is.na(speed_df$speedup_median)]),
            length(all_s),
            fmt_speedup(min(all_s)), fmt_speedup(median(all_s)),
            fmt_speedup(mean(all_s)), fmt_speedup(max(all_s))),
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(tex, file.path(fig_dir, "latex_speedup_summary.tex"))
  cat("  Saved latex_speedup_summary.tex\n")
}

# ============================================
# Table B: Detailed Method Comparison
# ============================================

if (file.exists(speed_file)) {
  cat("Generating Table B: Detailed Method Comparison...\n")

  detail <- speed_df %>%
    filter(!is.na(speedup_median) & is.finite(speedup_median)) %>%
    # Keep the largest n per method for the table
    group_by(method_norm, module) %>%
    filter(n == max(n)) %>%
    ungroup() %>%
    arrange(module, desc(speedup_median)) %>%
    select(module, method_norm, n, r_median_us, rust_median_us, speedup_median)

  tex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Detailed speed comparison at largest sample size per method.}",
    "\\label{tab:detailed-comparison}",
    "\\footnotesize",
    "\\begin{tabular}{llrccc}",
    "\\toprule",
    "Module & Method & $n$ & R Time & Rust Time & Speedup \\\\",
    "\\midrule"
  )

  last_module <- ""
  for (i in 1:nrow(detail)) {
    r <- detail[i, ]
    module_str <- if (r$module != last_module) r$module else ""
    last_module <- r$module

    if (module_str != "" && i > 1) {
      tex <- c(tex, "\\addlinespace")
    }

    tex <- c(tex, sprintf("%s & %s & %s & %s & %s & %s \\\\",
                          gsub("_", "\\\\_", module_str),
                          gsub("_", "\\\\_", r$method_norm),
                          format(r$n, big.mark = ","),
                          fmt_time(r$r_median_us),
                          fmt_time(r$rust_median_us),
                          fmt_speedup(r$speedup_median)))
  }

  tex <- c(tex,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(tex, file.path(fig_dir, "latex_detailed_comparison.tex"))
  cat("  Saved latex_detailed_comparison.tex\n")
}

# ============================================
# Table C: Memory Comparison
# ============================================

memory_file <- "results/comparison_memory.csv"
if (file.exists(memory_file)) {
  cat("Generating Table C: Memory Comparison...\n")

  mem_df <- read.csv(memory_file, stringsAsFactors = FALSE)

  mem_table <- mem_df %>%
    filter(!is.na(mem_ratio) & is.finite(mem_ratio)) %>%
    group_by(method_norm, module) %>%
    filter(n == max(n)) %>%
    ungroup() %>%
    arrange(module, desc(mem_ratio)) %>%
    select(module, method_norm, n, r_mem_bytes, rust_mem_bytes, mem_ratio)

  if (nrow(mem_table) > 0) {
    tex <- c(
      "\\begin{table}[htbp]",
      "\\centering",
      "\\caption{Memory usage comparison. R = heap allocations (\\texttt{bench::mark}); Rust = RSS delta.}",
      "\\label{tab:memory-comparison}",
      "\\footnotesize",
      "\\begin{tabular}{llrccc}",
      "\\toprule",
      "Module & Method & $n$ & R Memory & Rust Memory & Ratio \\\\",
      "\\midrule"
    )

    last_module <- ""
    for (i in 1:nrow(mem_table)) {
      r <- mem_table[i, ]
      module_str <- if (r$module != last_module) r$module else ""
      last_module <- r$module

      if (module_str != "" && i > 1) {
        tex <- c(tex, "\\addlinespace")
      }

      tex <- c(tex, sprintf("%s & %s & %s & %s & %s & %s \\\\",
                            gsub("_", "\\\\_", module_str),
                            gsub("_", "\\\\_", r$method_norm),
                            format(r$n, big.mark = ","),
                            fmt_mem(r$r_mem_bytes),
                            fmt_mem(r$rust_mem_bytes),
                            fmt_speedup(r$mem_ratio)))
    }

    tex <- c(tex,
      "\\bottomrule",
      "\\end{tabular}",
      "\\end{table}"
    )

    writeLines(tex, file.path(fig_dir, "latex_memory_comparison.tex"))
    cat("  Saved latex_memory_comparison.tex\n")
  } else {
    cat("  No memory data for table\n")
  }
}

# ============================================
# Table D: Validation Coverage
# ============================================

coverage_file <- "results/validation_coverage.csv"
if (file.exists(coverage_file)) {
  cat("Generating Table D: Validation Coverage...\n")

  cov_df <- read.csv(coverage_file, stringsAsFactors = FALSE)

  cov_summary <- cov_df %>%
    group_by(module) %>%
    summarise(
      Total = n(),
      Implemented = sum(rust_impl),
      Validated = sum(r_validation),
      `Speed Bench` = sum(speed_bench),
      `Mem Bench` = sum(mem_bench),
      `Valid \\%` = sprintf("%.0f\\%%", 100 * sum(r_validation) / n()),
      .groups = "drop"
    )

  tex <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    "\\caption{Validation and benchmark coverage by method category.}",
    "\\label{tab:coverage}",
    "\\begin{tabular}{lrrrrrr}",
    "\\toprule",
    "Module & Total & Impl. & Valid. & Speed & Memory & Valid.~\\% \\\\",
    "\\midrule"
  )

  for (i in 1:nrow(cov_summary)) {
    r <- cov_summary[i, ]
    tex <- c(tex, sprintf("%s & %d & %d & %d & %d & %d & %s \\\\",
                          r$module, r$Total, r$Implemented, r$Validated,
                          r$`Speed Bench`, r$`Mem Bench`, r$`Valid \\%`))
  }

  # Totals
  tex <- c(tex,
    "\\midrule",
    sprintf("\\textbf{Total} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%d} & \\textbf{%.0f\\%%} \\\\",
            nrow(cov_df), sum(cov_df$rust_impl), sum(cov_df$r_validation),
            sum(cov_df$speed_bench), sum(cov_df$mem_bench),
            100 * sum(cov_df$r_validation) / nrow(cov_df)),
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}"
  )

  writeLines(tex, file.path(fig_dir, "latex_coverage.tex"))
  cat("  Saved latex_coverage.tex\n")
}

# ============================================
# Summary
# ============================================

cat("\n=== LaTeX Tables Generated ===\n")
tex_files <- list.files(fig_dir, pattern = "\\.tex$")
for (f in sort(tex_files)) {
  cat(sprintf("  %s\n", f))
}
cat("\nInclude in paper with: \\input{results/figures/latex_*.tex}\n")
cat("Done.\n")
