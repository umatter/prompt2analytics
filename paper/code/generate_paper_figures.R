#!/usr/bin/env Rscript
# ============================================================================
# generate_paper_figures.R
# Generate all benchmark figures for the paper from comparison CSVs
#
# Reads from: performance/comparisons/r_comparison/results/
# Outputs to: paper/figures/
#
# Usage: cd paper/code && Rscript generate_paper_figures.R
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(viridis)
  library(gridExtra)
})

# Paths
script_dir <- tryCatch(
  dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
  error = function(e) getwd()
)
paper_dir <- dirname(script_dir)
project_root <- dirname(paper_dir)
results_dir <- file.path(project_root, "performance/comparisons/r_comparison/results")
fig_dir <- file.path(paper_dir, "figures")

cat("=== Generating Paper Figures from Benchmark Data ===\n")
cat("Results dir:", results_dir, "\n")
cat("Output dir:", fig_dir, "\n\n")

# Load data
speed_df <- read.csv(file.path(results_dir, "comparison_speed.csv"), stringsAsFactors = FALSE)
mem_df <- read.csv(file.path(results_dir, "comparison_memory.csv"), stringsAsFactors = FALSE)

cat(sprintf("Speed data: %d entries, %d matched\n", nrow(speed_df),
            sum(!is.na(speed_df$speedup_median))))
cat(sprintf("Memory data: %d entries\n", nrow(mem_df)))

# Filter to matched data only
matched <- speed_df %>% filter(!is.na(speedup_median) & is.finite(speedup_median))
cat(sprintf("Matched methods: %d\n\n", n_distinct(matched$method_norm)))

if (nrow(matched) < 50) {
  message(sprintf("WARNING: Only %d matched rows found (expected >= 50). Figures may be incomplete.", nrow(matched)))
}

# Check for modules with 0 matches
all_modules <- c("Regression", "Panel", "Discrete", "Time Series", "ML",
                 "Causal", "Spatial", "Stats", "Survival")
missing_modules <- setdiff(all_modules, unique(matched$module))
if (length(missing_modules) > 0) {
  message(sprintf("WARNING: No matched benchmarks for module(s): %s",
                  paste(missing_modules, collapse = ", ")))
}

# Use largest-n per method for summary figures
largest_n <- matched %>%
  group_by(method_norm) %>%
  filter(n == max(n)) %>%
  ungroup()

# ============================================================================
# Figure: Speedup Histogram (benchmark_histogram)
# ============================================================================
cat("Generating benchmark_histogram...\n")

p_hist <- ggplot(largest_n, aes(x = speedup_median, fill = speedup_median >= 1)) +
  geom_histogram(bins = 25, color = "white", linewidth = 0.3) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  geom_vline(xintercept = median(largest_n$speedup_median),
             linetype = "solid", color = "#1B5E20", linewidth = 0.7) +
  annotate("text",
           x = median(largest_n$speedup_median) * 1.3,
           y = Inf, vjust = 2,
           label = sprintf("Median: %.1fx", median(largest_n$speedup_median)),
           color = "#1B5E20", size = 3.5, fontface = "bold") +
  scale_x_log10(
    name = "Speedup Factor (Rust / R)",
    breaks = c(0.01, 0.1, 1, 10, 100, 1000, 10000),
    labels = c("0.01x", "0.1x", "1x", "10x", "100x", "1000x", "10000x")
  ) +
  scale_y_continuous(name = "Number of Methods") +
  scale_fill_manual(
    values = c("FALSE" = "#0097A7", "TRUE" = "#E65100"),
    labels = c("FALSE" = "R faster", "TRUE" = "Rust faster"),
    name = NULL
  ) +
  annotate("text", x = 0.03, y = Inf, label = "R faster",
           vjust = 2, hjust = 0.5, color = "#0097A7", fontface = "bold", size = 3.5) +
  annotate("text", x = 500, y = Inf, label = "Rust faster",
           vjust = 2, hjust = 0.5, color = "#E65100", fontface = "bold", size = 3.5) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.margin = margin(15, 15, 10, 10)
  )

ggsave(file.path(fig_dir, "benchmark_histogram.pdf"), p_hist, width = 7, height = 4)
ggsave(file.path(fig_dir, "benchmark_histogram.png"), p_hist, width = 7, height = 4, dpi = 300)
cat("  Saved benchmark_histogram.pdf/png\n")

# ============================================================================
# Figure: Speedup Bar Chart (benchmark_speedup)
# ============================================================================
cat("Generating benchmark_speedup...\n")

# Select representative methods across categories for a readable figure
# Use largest_n, pick top methods per module
representative <- largest_n %>%
  mutate(category = module) %>%
  group_by(category) %>%
  # Take top 3 per category by absolute speedup
  arrange(desc(abs(log10(speedup_median)))) %>%
  slice_head(n = 3) %>%
  ungroup() %>%
  arrange(desc(speedup_median)) %>%
  # Take top 25 overall for readability
  slice_head(n = 30)

representative <- representative %>%
  mutate(
    method_label = paste0(method_norm, " (n=", format(n, big.mark = ","), ")"),
    method_label = factor(method_label, levels = rev(method_label)),
    speedup_label = ifelse(speedup_median >= 1,
                           sprintf("%.0fx", speedup_median),
                           sprintf("%.2fx", speedup_median))
  )

# Module color palette
module_colors <- c(
  "Regression" = "#2196F3", "Panel" = "#4CAF50", "Discrete" = "#FF9800",
  "Time Series" = "#9C27B0", "ML" = "#F44336", "Causal" = "#00BCD4",
  "Spatial" = "#795548", "Stats" = "#607D8B", "Diagnostics" = "#CDDC39",
  "Survival" = "#E91E63", "Econometrics" = "#3F51B5", "Other" = "#9E9E9E"
)

p_speedup <- ggplot(representative, aes(x = method_label, y = speedup_median, fill = category)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_text(aes(label = speedup_label), hjust = -0.1, size = 2.5, color = "gray30") +
  coord_flip() +
  scale_y_log10(
    name = "Speedup Factor (R time / Rust time)",
    breaks = c(0.01, 0.1, 1, 10, 100, 1000, 10000),
    labels = c("0.01x", "0.1x", "1x", "10x", "100x", "1000x", "10000x")
  ) +
  scale_x_discrete(name = NULL) +
  scale_fill_manual(values = module_colors, name = "Category") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 9),
    plot.margin = margin(10, 20, 10, 10)
  )

ggsave(file.path(fig_dir, "benchmark_speedup.pdf"), p_speedup, width = 9, height = 8)
ggsave(file.path(fig_dir, "benchmark_speedup.png"), p_speedup, width = 9, height = 8, dpi = 300)
cat("  Saved benchmark_speedup.pdf/png\n")

# ============================================================================
# Figure: Execution Time Boxplots (benchmark_boxplots)
# ============================================================================
cat("Generating benchmark_boxplots...\n")

# Select representative methods for boxplots
selected_methods <- c("OLS", "OLS_HC1", "Fixed_Effects", "HDFE", "Logit", "Probit",
                       "PCA", "K-Means", "ARIMA", "MSTL", "t_test", "DiD",
                       "Cox_PH", "Kaplan_Meier", "IV_2SLS", "Random_Forest")

# Get data for selected methods at largest n
box_data <- matched %>%
  filter(method_norm %in% selected_methods) %>%
  group_by(method_norm) %>%
  filter(n == max(n)) %>%
  ungroup()

if (nrow(box_data) > 0) {
  # Pivot to long format for R vs Rust comparison
  box_long <- box_data %>%
    select(method_norm, module, n, r_median_us, r_p25_us, r_p75_us,
           rust_median_us, rust_p25_us, rust_p75_us) %>%
    pivot_longer(
      cols = c(r_median_us, rust_median_us),
      names_to = "impl_median",
      values_to = "median_us"
    ) %>%
    mutate(
      implementation = ifelse(grepl("^r_", impl_median), "R", "p2a (Rust)"),
      p25_us = ifelse(implementation == "R", r_p25_us, rust_p25_us),
      p75_us = ifelse(implementation == "R", r_p75_us, rust_p75_us),
      label = paste0(method_norm, "\n(n=", format(n, big.mark = ","), ")"),
      median_ms = median_us / 1000,
      p25_ms = p25_us / 1000,
      p75_ms = p75_us / 1000
    ) %>%
    select(method_norm, module, n, implementation, label, median_ms, p25_ms, p75_ms)

  # Order by module then method
  method_order <- box_long %>%
    arrange(module, method_norm) %>%
    pull(label) %>%
    unique()
  box_long$label <- factor(box_long$label, levels = method_order)
  box_long$implementation <- factor(box_long$implementation, levels = c("R", "p2a (Rust)"))

  p_box <- ggplot(box_long, aes(x = label, y = median_ms, fill = implementation)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    scale_y_log10(
      name = "Median Execution Time (ms, log scale)",
      labels = scales::comma
    ) +
    scale_x_discrete(name = NULL) +
    scale_fill_manual(values = c("R" = "#0097A7", "p2a (Rust)" = "#E65100"), name = NULL) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )

  ggsave(file.path(fig_dir, "benchmark_boxplots.pdf"), p_box, width = 10, height = 5, dpi = 300)
  ggsave(file.path(fig_dir, "benchmark_boxplots.png"), p_box, width = 10, height = 5, dpi = 300)
  cat("  Saved benchmark_boxplots.pdf/png\n")
}

# ============================================================================
# Figure: Memory Comparison (benchmark_memory) -- NEW
# ============================================================================
cat("Generating benchmark_memory...\n")

mem_matched <- mem_df %>%
  filter(!is.na(mem_ratio) & is.finite(mem_ratio) & rust_mem_bytes > 0) %>%
  group_by(method_norm) %>%
  filter(n == max(n)) %>%
  ungroup()

if (nrow(mem_matched) > 0) {
  # Select top 25 methods by memory ratio for readability
  mem_top <- mem_matched %>%
    arrange(desc(mem_ratio)) %>%
    slice_head(n = 25) %>%
    mutate(
      r_mem_kb = r_mem_bytes / 1024,
      rust_mem_kb = rust_mem_bytes / 1024,
      method_label = paste0(method_norm, " (n=", format(n, big.mark = ","), ")"),
      method_label = reorder(method_label, mem_ratio)
    )

  mem_long <- mem_top %>%
    pivot_longer(cols = c(r_mem_kb, rust_mem_kb),
                 names_to = "language",
                 values_to = "mem_kb") %>%
    mutate(language = ifelse(language == "r_mem_kb", "R", "Rust"))

  p_mem <- ggplot(mem_long, aes(x = method_label, y = mem_kb, fill = language)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("R" = "#0097A7", "Rust" = "#E65100"), name = NULL) +
    scale_y_log10(
      name = "Memory Allocation (log scale)",
      labels = function(x) {
        ifelse(x < 1, sprintf("%.0f B", x * 1024),
        ifelse(x < 1024, sprintf("%.0f KB", x),
               sprintf("%.1f MB", x / 1024)))
      }
    ) +
    coord_flip() +
    labs(x = NULL) +
    theme_minimal(base_size = 11) +
    theme(
      legend.position = "top",
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )

  ggsave(file.path(fig_dir, "benchmark_memory.pdf"), p_mem,
         width = 9, height = 7, limitsize = FALSE)
  ggsave(file.path(fig_dir, "benchmark_memory.png"), p_mem,
         width = 9, height = 7, dpi = 300, limitsize = FALSE)
  cat("  Saved benchmark_memory.pdf/png\n")
}

# ============================================================================
# Figure: Speedup by Module (violin) -- NEW for appendix
# ============================================================================
cat("Generating benchmark_speedup_violin...\n")

p_violin <- ggplot(matched, aes(x = module, y = speedup_median, fill = module)) +
  geom_violin(alpha = 0.6, draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_jitter(width = 0.15, size = 1.2, alpha = 0.4) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  scale_y_log10(
    name = "Speedup Factor (log scale)",
    labels = function(x) sprintf("%.0fx", x)
  ) +
  scale_fill_manual(values = module_colors, guide = "none") +
  labs(x = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(fig_dir, "benchmark_speedup_violin.pdf"), p_violin, width = 10, height = 5)
ggsave(file.path(fig_dir, "benchmark_speedup_violin.png"), p_violin, width = 10, height = 5, dpi = 300)
cat("  Saved benchmark_speedup_violin.pdf/png\n")

# ============================================================================
# Print Summary Stats
# ============================================================================
cat("\n=== Summary Statistics ===\n")
cat(sprintf("Total matched benchmarks: %d\n", nrow(matched)))
cat(sprintf("Unique methods: %d\n", n_distinct(matched$method_norm)))
cat(sprintf("Overall median speedup: %.1fx\n", median(matched$speedup_median)))
cat(sprintf("Methods where Rust faster: %d (%.0f%%)\n",
            sum(largest_n$speedup_median >= 1),
            100 * mean(largest_n$speedup_median >= 1)))
cat(sprintf("Methods where R faster: %d (%.0f%%)\n",
            sum(largest_n$speedup_median < 1),
            100 * mean(largest_n$speedup_median < 1)))

cat("\nMedian memory ratio (R/Rust): ",
    sprintf("%.0fx", median(mem_matched$mem_ratio, na.rm = TRUE)), "\n")

cat("\n--- Per Module (largest n) ---\n")
largest_n %>%
  group_by(module) %>%
  summarise(
    methods = n_distinct(method_norm),
    median_speedup = median(speedup_median),
    .groups = "drop"
  ) %>%
  arrange(desc(median_speedup)) %>%
  {for (i in 1:nrow(.)) {
    cat(sprintf("  %-15s: %2d methods, median %.1fx\n",
                .$module[i], .$methods[i], .$median_speedup[i]))
  }}

cat("\nDone.\n")
