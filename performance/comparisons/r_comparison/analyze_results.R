#!/usr/bin/env Rscript
# Publication-Quality Figures for p2a Benchmark Analysis
#
# Reads comparison CSVs from results/ and produces:
#   - Figure 1: Speedup factor heatmap (methods x sample sizes)
#   - Figure 2: Memory efficiency comparison (bar chart)
#   - Figure 3: Validation coverage matrix
#   - Figure 4: Speedup distribution (violin/histogram)
#   - Table 1: Summary statistics by module
#
# Output: results/figures/ (PDF + PNG)
#
# Usage: Rscript analyze_results.R
#
# Required packages: ggplot2, dplyr, tidyr, scales, viridis, gridExtra, ggtext

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(viridis)
  library(gridExtra)
})

cat("=== Generating Publication-Quality Figures ===\n\n")

# Setup output directory
fig_dir <- "results/figures"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================
# Theme
# ============================================

theme_pub <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = ""),
      plot.title = element_text(size = base_size + 2, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size, color = "gray40"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 1),
      legend.title = element_text(size = base_size - 1, face = "bold"),
      legend.text = element_text(size = base_size - 2),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )
}

# Module color palette
module_colors <- c(
  "Regression" = "#2196F3",
  "Panel" = "#4CAF50",
  "Discrete" = "#FF9800",
  "Time Series" = "#9C27B0",
  "ML" = "#F44336",
  "Causal" = "#00BCD4",
  "Spatial" = "#795548",
  "Stats" = "#607D8B",
  "Diagnostics" = "#CDDC39",
  "Survival" = "#E91E63",
  "Other" = "#9E9E9E"
)

# ============================================
# Load Data
# ============================================

speed_file <- "results/comparison_speed.csv"
memory_file <- "results/comparison_memory.csv"
coverage_file <- "results/validation_coverage.csv"

has_speed <- file.exists(speed_file)
has_memory <- file.exists(memory_file)
has_coverage <- file.exists(coverage_file)

if (has_speed) {
  speed_df <- read.csv(speed_file, stringsAsFactors = FALSE)
  cat(sprintf("Loaded speed comparison: %d entries\n", nrow(speed_df)))
} else {
  cat("WARNING: No speed comparison data. Run merge_results.R first.\n")
}

if (has_memory) {
  mem_df <- read.csv(memory_file, stringsAsFactors = FALSE)
  cat(sprintf("Loaded memory comparison: %d entries\n", nrow(mem_df)))
} else {
  cat("WARNING: No memory comparison data.\n")
}

if (has_coverage) {
  cov_df <- read.csv(coverage_file, stringsAsFactors = FALSE)
  cat(sprintf("Loaded validation coverage: %d methods\n", nrow(cov_df)))
} else {
  cat("WARNING: No validation coverage data. Run merge_results.R first.\n")
}

# ============================================
# Figure 1: Speedup Heatmap
# ============================================

if (has_speed) {
  cat("\nGenerating Figure 1: Speedup Heatmap...\n")

  heatmap_data <- speed_df %>%
    filter(!is.na(speedup_median) & is.finite(speedup_median)) %>%
    mutate(
      n_label = factor(n, levels = sort(unique(n))),
      method_norm = reorder(method_norm, speedup_median, FUN = median, na.rm = TRUE),
      log_speedup = log10(pmax(speedup_median, 0.1))
    )

  if (nrow(heatmap_data) > 0) {
    p1 <- ggplot(heatmap_data, aes(x = n_label, y = method_norm, fill = log_speedup)) +
      geom_tile(color = "white", linewidth = 0.3) +
      geom_text(aes(label = sprintf("%.0fx", speedup_median)),
                size = 2.5, color = ifelse(heatmap_data$log_speedup > 1.5, "white", "black")) +
      scale_fill_viridis(
        name = "Speedup\n(log10)",
        option = "D",
        labels = function(x) sprintf("%.0fx", 10^x)
      ) +
      facet_grid(module ~ ., scales = "free_y", space = "free_y") +
      labs(
        title = "Rust vs R Speedup by Method and Sample Size",
        subtitle = "Speedup factor = R median time / Rust median time",
        x = "Sample Size (n)",
        y = ""
      ) +
      theme_pub() +
      theme(
        strip.text.y = element_text(angle = 0, hjust = 0, face = "bold"),
        panel.spacing = unit(0.3, "lines")
      )

    ggsave(file.path(fig_dir, "fig1_speedup_heatmap.pdf"), p1,
           width = 10, height = max(6, nrow(heatmap_data) * 0.3 + 2), limitsize = FALSE)
    ggsave(file.path(fig_dir, "fig1_speedup_heatmap.png"), p1,
           width = 10, height = max(6, nrow(heatmap_data) * 0.3 + 2), dpi = 300, limitsize = FALSE)
    cat("  Saved fig1_speedup_heatmap.pdf/png\n")
  } else {
    cat("  No matched speed data for heatmap\n")
  }
}

# ============================================
# Figure 2: Memory Comparison
# ============================================

if (has_memory) {
  cat("Generating Figure 2: Memory Comparison...\n")

  mem_plot_data <- mem_df %>%
    filter(!is.na(r_mem_bytes) & !is.na(rust_mem_bytes) & rust_mem_bytes > 0) %>%
    # Use the largest n for each method
    group_by(method_norm) %>%
    filter(n == max(n)) %>%
    ungroup() %>%
    mutate(
      r_mem_kb = r_mem_bytes / 1024,
      rust_mem_kb = abs(rust_mem_bytes) / 1024,
      method_norm = reorder(method_norm, mem_ratio)
    ) %>%
    pivot_longer(cols = c(r_mem_kb, rust_mem_kb),
                 names_to = "language",
                 values_to = "mem_kb") %>%
    mutate(
      language = ifelse(language == "r_mem_kb", "R (heap alloc)", "Rust (RSS delta)")
    )

  if (nrow(mem_plot_data) > 0) {
    p2 <- ggplot(mem_plot_data, aes(x = method_norm, y = mem_kb, fill = language)) +
      geom_col(position = "dodge", width = 0.7) +
      scale_fill_manual(values = c("R (heap alloc)" = "#2196F3", "Rust (RSS delta)" = "#FF5722")) +
      scale_y_log10(labels = function(x) {
        ifelse(x < 1, sprintf("%.0f B", x * 1024),
        ifelse(x < 1024, sprintf("%.0f KB", x),
               sprintf("%.1f MB", x / 1024)))
      }) +
      coord_flip() +
      labs(
        title = "Memory Usage: R vs Rust",
        subtitle = "R = heap allocations (bench::mark); Rust = RSS delta (process-level)",
        x = "",
        y = "Memory (log scale)",
        fill = ""
      ) +
      theme_pub() +
      theme(legend.position = "top")

    ggsave(file.path(fig_dir, "fig2_memory_comparison.pdf"), p2,
           width = 10, height = max(5, length(unique(mem_plot_data$method_norm)) * 0.4 + 2),
           limitsize = FALSE)
    ggsave(file.path(fig_dir, "fig2_memory_comparison.png"), p2,
           width = 10, height = max(5, length(unique(mem_plot_data$method_norm)) * 0.4 + 2),
           dpi = 300, limitsize = FALSE)
    cat("  Saved fig2_memory_comparison.pdf/png\n")
  } else {
    cat("  No matched memory data for comparison\n")
  }
}

# ============================================
# Figure 3: Validation Coverage Matrix
# ============================================

if (has_coverage) {
  cat("Generating Figure 3: Validation Coverage Matrix...\n")

  cov_long <- cov_df %>%
    select(method, module, rust_impl, r_validation, speed_bench, mem_bench) %>%
    pivot_longer(cols = c(rust_impl, r_validation, speed_bench, mem_bench),
                 names_to = "check_type",
                 values_to = "status") %>%
    mutate(
      check_label = case_when(
        check_type == "rust_impl" ~ "Rust\nImpl",
        check_type == "r_validation" ~ "R\nValidation",
        check_type == "speed_bench" ~ "Speed\nBench",
        check_type == "mem_bench" ~ "Memory\nBench"
      ),
      check_label = factor(check_label,
                           levels = c("Rust\nImpl", "R\nValidation", "Speed\nBench", "Memory\nBench")),
      status_label = ifelse(status, "\u2713", ""),
      module = factor(module, levels = c("Regression", "Diagnostics", "Panel", "Discrete",
                                          "Time Series", "Econometrics", "Causal", "ML",
                                          "Stats", "Spatial", "Survival"))
    )

  p3 <- ggplot(cov_long, aes(x = check_label, y = method, fill = status)) +
    geom_tile(color = "white", linewidth = 0.2) +
    geom_text(aes(label = status_label), size = 3, color = "white") +
    scale_fill_manual(
      values = c("TRUE" = "#4CAF50", "FALSE" = "#FFCDD2"),
      labels = c("TRUE" = "Yes", "FALSE" = "No"),
      name = "Status"
    ) +
    facet_grid(module ~ ., scales = "free_y", space = "free_y") +
    labs(
      title = "Method Coverage Matrix",
      subtitle = "Implementation, validation, and benchmark status for each method",
      x = "",
      y = ""
    ) +
    theme_pub(base_size = 9) +
    theme(
      strip.text.y = element_text(angle = 0, hjust = 0, face = "bold", size = 8),
      axis.text.y = element_text(size = 6),
      panel.spacing = unit(0.2, "lines"),
      legend.position = "bottom"
    )

  fig_height <- max(12, nrow(cov_df) * 0.2 + 3)
  ggsave(file.path(fig_dir, "fig3_validation_coverage.pdf"), p3,
         width = 8, height = fig_height, limitsize = FALSE)
  ggsave(file.path(fig_dir, "fig3_validation_coverage.png"), p3,
         width = 8, height = fig_height, dpi = 300, limitsize = FALSE)
  cat("  Saved fig3_validation_coverage.pdf/png\n")
}

# ============================================
# Figure 4: Speedup Distribution
# ============================================

if (has_speed) {
  cat("Generating Figure 4: Speedup Distribution...\n")

  speedup_data <- speed_df %>%
    filter(!is.na(speedup_median) & is.finite(speedup_median) & speedup_median > 0) %>%
    mutate(module = factor(module))

  if (nrow(speedup_data) > 0) {
    # Overall distribution
    p4a <- ggplot(speedup_data, aes(x = speedup_median)) +
      geom_histogram(aes(y = after_stat(density)),
                     bins = 30, fill = "#2196F3", alpha = 0.7, color = "white") +
      geom_density(linewidth = 0.8, color = "#1565C0") +
      geom_vline(xintercept = 1, linetype = "dashed", color = "red", linewidth = 0.5) +
      geom_vline(xintercept = median(speedup_data$speedup_median),
                 linetype = "solid", color = "#1B5E20", linewidth = 0.7) +
      annotate("text",
               x = median(speedup_data$speedup_median) * 1.1,
               y = Inf, vjust = 2,
               label = sprintf("Median: %.1fx", median(speedup_data$speedup_median)),
               color = "#1B5E20", size = 3.5, fontface = "bold") +
      scale_x_log10(labels = function(x) sprintf("%.0fx", x)) +
      labs(
        title = "Distribution of Rust vs R Speedup Factors",
        subtitle = "Across all benchmarked methods and sample sizes",
        x = "Speedup Factor (log scale)",
        y = "Density"
      ) +
      theme_pub()

    # By module violin
    p4b <- ggplot(speedup_data, aes(x = module, y = speedup_median, fill = module)) +
      geom_violin(alpha = 0.6, draw_quantiles = c(0.25, 0.5, 0.75)) +
      geom_jitter(width = 0.1, size = 1.5, alpha = 0.4) +
      geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
      scale_y_log10(labels = function(x) sprintf("%.0fx", x)) +
      scale_fill_manual(values = module_colors, guide = "none") +
      labs(
        title = "Speedup by Module Category",
        subtitle = "Violin plots with quartile lines; red dashed = parity (1x)",
        x = "",
        y = "Speedup Factor (log scale)"
      ) +
      theme_pub() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))

    p4 <- gridExtra::arrangeGrob(p4a, p4b, ncol = 1, heights = c(1, 1))

    ggsave(file.path(fig_dir, "fig4_speedup_distribution.pdf"), p4,
           width = 10, height = 10)
    ggsave(file.path(fig_dir, "fig4_speedup_distribution.png"), p4,
           width = 10, height = 10, dpi = 300)
    cat("  Saved fig4_speedup_distribution.pdf/png\n")
  } else {
    cat("  No speedup data available\n")
  }
}

# ============================================
# Table 1: Summary Statistics
# ============================================

if (has_speed) {
  cat("Generating Table 1: Summary Statistics...\n")

  summary_stats <- speed_df %>%
    filter(!is.na(speedup_median) & is.finite(speedup_median)) %>%
    group_by(module) %>%
    summarise(
      n_methods = n_distinct(method_norm),
      n_benchmarks = n(),
      speedup_min = min(speedup_median, na.rm = TRUE),
      speedup_p25 = quantile(speedup_median, 0.25, na.rm = TRUE),
      speedup_median = median(speedup_median, na.rm = TRUE),
      speedup_p75 = quantile(speedup_median, 0.75, na.rm = TRUE),
      speedup_max = max(speedup_median, na.rm = TRUE),
      speedup_mean = mean(speedup_median, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(speedup_median))

  # Add memory stats if available
  if (has_memory && nrow(mem_df) > 0) {
    mem_summary <- mem_df %>%
      filter(!is.na(mem_ratio) & is.finite(mem_ratio)) %>%
      group_by(module) %>%
      summarise(
        mem_ratio_median = median(mem_ratio, na.rm = TRUE),
        mem_ratio_mean = mean(mem_ratio, na.rm = TRUE),
        .groups = "drop"
      )
    summary_stats <- left_join(summary_stats, mem_summary, by = "module")
  }

  # Save as CSV
  write.csv(summary_stats, file.path(fig_dir, "table1_summary_stats.csv"), row.names = FALSE)

  # Print formatted table
  cat("\n")
  cat(sprintf("%-15s  %5s  %5s  %8s  %8s  %8s  %8s\n",
              "Module", "Meth", "Bench", "Min", "Median", "Mean", "Max"))
  cat(paste(rep("-", 70), collapse = ""), "\n")
  for (i in 1:nrow(summary_stats)) {
    r <- summary_stats[i, ]
    cat(sprintf("%-15s  %5d  %5d  %8.1fx  %8.1fx  %8.1fx  %8.1fx\n",
                r$module, r$n_methods, r$n_benchmarks,
                r$speedup_min, r$speedup_median, r$speedup_mean, r$speedup_max))
  }

  # Overall row
  all_speedups <- speed_df$speedup_median[!is.na(speed_df$speedup_median) &
                                           is.finite(speed_df$speedup_median)]
  cat(paste(rep("-", 70), collapse = ""), "\n")
  cat(sprintf("%-15s  %5d  %5d  %8.1fx  %8.1fx  %8.1fx  %8.1fx\n",
              "OVERALL",
              n_distinct(speed_df$method_norm[!is.na(speed_df$speedup_median)]),
              length(all_speedups),
              min(all_speedups), median(all_speedups),
              mean(all_speedups), max(all_speedups)))
  cat("\n")
  cat("  Saved table1_summary_stats.csv\n")
}

# ============================================
# Validation coverage bar chart (bonus)
# ============================================

if (has_coverage) {
  cat("Generating Bonus: Coverage Bar Chart...\n")

  cov_summary <- cov_df %>%
    group_by(module) %>%
    summarise(
      total = n(),
      implemented = sum(rust_impl),
      validated = sum(r_validation),
      speed_benched = sum(speed_bench),
      mem_benched = sum(mem_bench),
      .groups = "drop"
    ) %>%
    pivot_longer(cols = c(implemented, validated, speed_benched, mem_benched),
                 names_to = "category",
                 values_to = "count") %>%
    mutate(
      pct = count / total * 100,
      category = factor(category,
                        levels = c("implemented", "validated", "speed_benched", "mem_benched"),
                        labels = c("Implemented", "R Validated", "Speed Benchmark", "Memory Benchmark"))
    )

  p_cov <- ggplot(cov_summary, aes(x = module, y = pct, fill = category)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c(
      "Implemented" = "#4CAF50",
      "R Validated" = "#2196F3",
      "Speed Benchmark" = "#FF9800",
      "Memory Benchmark" = "#9C27B0"
    )) +
    scale_y_continuous(labels = function(x) paste0(x, "%"), limits = c(0, 105)) +
    geom_hline(yintercept = 100, linetype = "dashed", color = "gray50", linewidth = 0.3) +
    labs(
      title = "Coverage by Module",
      subtitle = "Percentage of methods with implementation, validation, and benchmarks",
      x = "",
      y = "Coverage (%)",
      fill = ""
    ) +
    theme_pub() +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1),
      legend.position = "top"
    )

  ggsave(file.path(fig_dir, "coverage_bar_chart.pdf"), p_cov, width = 10, height = 6)
  ggsave(file.path(fig_dir, "coverage_bar_chart.png"), p_cov, width = 10, height = 6, dpi = 300)
  cat("  Saved coverage_bar_chart.pdf/png\n")
}

# ============================================
# Summary
# ============================================

cat("\n=== Figures Generated ===\n")
figures <- list.files(fig_dir, pattern = "\\.(pdf|png|csv)$")
for (f in sort(figures)) {
  size <- file.info(file.path(fig_dir, f))$size
  size_str <- if (size < 1024) sprintf("%d B", size)
              else if (size < 1024^2) sprintf("%.1f KB", size / 1024)
              else sprintf("%.1f MB", size / 1024^2)
  cat(sprintf("  %-45s %8s\n", f, size_str))
}
cat("\nDone.\n")
