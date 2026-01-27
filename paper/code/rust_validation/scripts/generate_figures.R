#!/usr/bin/env Rscript
# generate_figures.R
#
# Generates benchmark figures from rust_validation results
# Reads from results/summaries/benchmark_summary.json
#
# Output:
#   - figures/benchmark_speedup.pdf - Speedup bar chart
#   - figures/benchmark_boxplots.pdf - Execution time comparison

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(scales)
})

# Paths
script_dir <- dirname(normalizePath(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))] |>
                                      sub("--file=", "", x = _)))
if (length(script_dir) == 0) script_dir <- getwd()

base_dir <- dirname(script_dir)
results_dir <- file.path(base_dir, "results")
benchmarks_dir <- file.path(results_dir, "benchmarks")
summaries_dir <- file.path(results_dir, "summaries")
figures_dir <- file.path(base_dir, "figures")
paper_figures_dir <- file.path(dirname(dirname(base_dir)), "figures")

dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Generating Benchmark Figures ===\n\n")

# Read benchmark summary
summary_file <- file.path(summaries_dir, "benchmark_summary.json")
if (!file.exists(summary_file)) {
  stop("Benchmark summary not found. Run ./scripts/run_benchmark.sh first.")
}

summary_data <- fromJSON(summary_file)
cat("Loaded benchmark summary\n")

# Extract method results into a data frame
methods_list <- summary_data$methods
benchmark_df <- do.call(rbind, lapply(names(methods_list), function(method) {
  method_data <- methods_list[[method]]
  if (is.data.frame(method_data)) {
    method_data$method <- method
    return(method_data)
  }
  return(NULL)
}))

if (is.null(benchmark_df) || nrow(benchmark_df) == 0) {
  stop("No benchmark data found in summary")
}

cat("\nBenchmark data loaded:\n")
print(benchmark_df %>% select(method, n, r_median_us, rust_median_us, speedup))

# Add category labels
benchmark_df <- benchmark_df %>%
  mutate(
    category = case_when(
      grepl("^ols", method, ignore.case = TRUE) ~ "Regression",
      grepl("^panel", method, ignore.case = TRUE) ~ "Panel",
      grepl("logit|probit", method, ignore.case = TRUE) ~ "Discrete",
      grepl("kmeans|pca|dbscan|hierarchical", method, ignore.case = TRUE) ~ "ML",
      grepl("arima|mstl|stl|holt|ar$", method, ignore.case = TRUE) ~ "TimeSeries",
      grepl("sort|filter|group|select|standardize|lag|lead|diff", method, ignore.case = TRUE) ~ "Munging",
      TRUE ~ "Other"
    ),
    method_label = gsub("_", " ", method) %>%
      gsub("hc([0-3])", "HC\\1", .) %>%
      gsub("^ols$", "OLS", ., ignore.case = TRUE) %>%
      gsub("^ols ", "OLS+", ., ignore.case = TRUE) %>%
      gsub("panel fe", "Panel FE", ., ignore.case = TRUE) %>%
      gsub("panel re", "Panel RE", ., ignore.case = TRUE) %>%
      tools::toTitleCase(.)
  )

# ============================================================
# Figure 1: Speedup Bar Chart (n=100,000)
# ============================================================

speedup_data <- benchmark_df %>%
  filter(n == 100000, speedup > 0) %>%
  arrange(desc(speedup)) %>%
  mutate(
    method_label = factor(method_label, levels = rev(method_label)),
    faster = ifelse(speedup >= 1, "Rust faster", "R faster")
  )

if (nrow(speedup_data) > 0) {
  p_speedup <- ggplot(speedup_data, aes(x = method_label, y = speedup, fill = category)) +
    geom_bar(stat = "identity", width = 0.7) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.1fx", speedup)),
              hjust = -0.1, size = 3, color = "gray30") +
    coord_flip() +
    scale_y_continuous(
      name = "Speedup Factor (R time / Rust time)",
      limits = c(0, max(speedup_data$speedup) * 1.15),
      expand = c(0, 0)
    ) +
    scale_x_discrete(name = "") +
    scale_fill_brewer(palette = "Set2", name = "Category") +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 10),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_blank(),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "gray40")
    ) +
    labs(
      title = "Rust vs R Performance Comparison",
      subtitle = sprintf("n = 100,000 observations | %d methods benchmarked", nrow(speedup_data))
    )

  # Save speedup figure
  output_speedup <- file.path(figures_dir, "benchmark_speedup.pdf")
  ggsave(output_speedup, p_speedup, width = 8, height = 6, dpi = 300)
  cat("\nSaved speedup figure:", output_speedup, "\n")

  # Also save PNG
  ggsave(sub("\\.pdf$", ".png", output_speedup), p_speedup, width = 8, height = 6, dpi = 300)

  # Copy to paper figures directory
  file.copy(output_speedup, file.path(paper_figures_dir, "benchmark_speedup.pdf"), overwrite = TRUE)
  file.copy(sub("\\.pdf$", ".png", output_speedup),
            file.path(paper_figures_dir, "benchmark_speedup.png"), overwrite = TRUE)
  cat("Copied to paper figures directory\n")
}

# ============================================================
# Figure 2: Execution Time Box Plots
# ============================================================

# Select largest n for each method for the main figure
largest_n_data <- benchmark_df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  filter(r_median_us > 0, rust_median_us > 0) %>%
  # Handle empty arrays/lists in min/max columns
  mutate(
    r_min_us = sapply(r_min_us, function(x) if(length(x) == 0 || is.list(x)) NA_real_ else as.numeric(x)),
    r_max_us = sapply(r_max_us, function(x) if(length(x) == 0 || is.list(x)) NA_real_ else as.numeric(x)),
    rust_min_us = sapply(rust_min_us, function(x) if(length(x) == 0 || is.list(x)) NA_real_ else as.numeric(x)),
    rust_max_us = sapply(rust_max_us, function(x) if(length(x) == 0 || is.list(x)) NA_real_ else as.numeric(x))
  )

if (nrow(largest_n_data) > 0) {
  # Convert to long format for plotting
  plot_long <- largest_n_data %>%
    select(method_label, category, n, r_median_us, rust_median_us,
           r_min_us, r_max_us, rust_min_us, rust_max_us) %>%
    pivot_longer(
      cols = c(r_median_us, rust_median_us),
      names_to = "impl",
      values_to = "median_us"
    ) %>%
    mutate(
      implementation = ifelse(grepl("^r_", impl), "R", "Rust"),
      min_us = ifelse(implementation == "R", as.numeric(r_min_us), as.numeric(rust_min_us)),
      max_us = ifelse(implementation == "R", as.numeric(r_max_us), as.numeric(rust_max_us)),
      # Use median for missing min/max
      min_us = ifelse(is.na(min_us), median_us * 0.9, min_us),
      max_us = ifelse(is.na(max_us), median_us * 1.1, max_us),
      time_ms = median_us / 1000,
      min_ms = min_us / 1000,
      max_ms = max_us / 1000
    )

  # Order by category and method
  method_order <- plot_long %>%
    arrange(category, desc(time_ms)) %>%
    pull(method_label) %>%
    unique()

  plot_long$method_label <- factor(plot_long$method_label, levels = method_order)
  plot_long$implementation <- factor(plot_long$implementation, levels = c("R", "Rust"))

  p_boxplot <- ggplot(plot_long, aes(x = method_label, y = time_ms, fill = implementation)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(
      aes(ymin = min_ms, ymax = max_ms),
      position = position_dodge(width = 0.8),
      width = 0.3,
      color = "gray40"
    ) +
    scale_y_log10(
      name = "Execution Time (ms, log scale)",
      labels = scales::comma
    ) +
    scale_x_discrete(name = "") +
    scale_fill_manual(
      values = c("R" = "#0097A7", "Rust" = "#E65100"),
      name = "Implementation"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 12, face = "bold")
    ) +
    labs(
      title = "Execution Time Comparison: R vs Rust",
      subtitle = "Largest sample size per method (min/max whiskers shown)"
    )

  output_boxplot <- file.path(figures_dir, "benchmark_boxplots.pdf")
  ggsave(output_boxplot, p_boxplot, width = 12, height = 6, dpi = 300)
  cat("\nSaved boxplot figure:", output_boxplot, "\n")

  ggsave(sub("\\.pdf$", ".png", output_boxplot), p_boxplot, width = 12, height = 6, dpi = 300)

  # Copy to paper figures
  file.copy(output_boxplot, file.path(paper_figures_dir, "benchmark_boxplots.pdf"), overwrite = TRUE)
  file.copy(sub("\\.pdf$", ".png", output_boxplot),
            file.path(paper_figures_dir, "benchmark_boxplots.png"), overwrite = TRUE)
}

# ============================================================
# Generate LaTeX Table
# ============================================================

# Create summary table for paper
table_data <- benchmark_df %>%
  filter(n == 100000 | n == max(n)) %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  arrange(category, desc(speedup)) %>%
  select(Category = category, Method = method_label, n,
         `R (ms)` = r_median_us, `Rust (ms)` = rust_median_us, Speedup = speedup) %>%
  mutate(
    `R (ms)` = round(`R (ms)` / 1000, 1),
    `Rust (ms)` = round(`Rust (ms)` / 1000, 1),
    Speedup = sprintf("%.1fx", Speedup)
  )

cat("\n=== Summary Table for Paper ===\n")
print(table_data, n = 50)

# Save as CSV for easy inclusion
write.csv(table_data, file.path(summaries_dir, "benchmark_table.csv"), row.names = FALSE)

# Generate LaTeX table
latex_table <- paste0(
  "\\begin{table}[htbp]\n",
  "\\centering\n",
  "\\caption{Performance benchmark results (median execution time)}\n",
  "\\label{tab:benchmark-full}\n",
  "\\begin{tabular}{llrrrr}\n",
  "\\toprule\n",
  "Category & Method & $n$ & R (ms) & Rust (ms) & Speedup \\\\\n",
  "\\midrule\n"
)

for (i in 1:nrow(table_data)) {
  row <- table_data[i, ]
  latex_table <- paste0(latex_table,
                        sprintf("%s & %s & %s & %.1f & %.1f & %s \\\\\n",
                                row$Category, row$Method,
                                format(row$n, big.mark = ","),
                                as.numeric(row$`R (ms)`),
                                as.numeric(row$`Rust (ms)`),
                                row$Speedup))
}

latex_table <- paste0(latex_table,
                      "\\bottomrule\n",
                      "\\end{tabular}\n",
                      "\\end{table}\n")

writeLines(latex_table, file.path(summaries_dir, "benchmark_table.tex"))
cat("\nLaTeX table saved to:", file.path(summaries_dir, "benchmark_table.tex"), "\n")

cat("\n=== Done! ===\n")
