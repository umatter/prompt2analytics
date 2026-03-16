#!/usr/bin/env Rscript
# ============================================================================
# generate_paper_figures.R
# Generate all benchmark figures for the paper from unified benchmark data
#
# Reads from: performance/comparisons/r_comparison/results/
#   - comparison_unified.csv (speed + validation)
#   - unified_r_*.json (R memory data)
#   - rust_unified_*.json (Rust memory data)
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
fig_dir <- file.path(paper_dir, "figures")

cat("=== Generating Paper Figures from Unified Benchmark Data ===\n")
cat("Results dir:", results_dir, "\n")
cat("Output dir:", fig_dir, "\n\n")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Load data
# ============================================================================

# Primary: unified CSV
unified_df <- read.csv(file.path(results_dir, "comparison_unified.csv"),
                       stringsAsFactors = FALSE)

cat(sprintf("Unified data: %d entries\n", nrow(unified_df)))

# Filter to matched speed data (both R and Rust have times)
matched <- unified_df %>%
  filter(!is.na(speedup) & is.finite(speedup))

cat(sprintf("Matched benchmarks: %d entries, %d unique methods\n",
            nrow(matched), n_distinct(matched$method)))

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
  # Extract only the fields we need (method, variant, n, mem_alloc_bytes)
  raw %>%
    select(method, any_of(c("variant", "n", "mem_alloc_bytes"))) %>%
    filter(!is.na(mem_alloc_bytes))
}

cat("Loading memory data from JSON files...\n")
r_mem <- load_latest_json("^unified_r_.*\\.json$") %>%
  rename(r_mem_bytes = mem_alloc_bytes)
rust_mem <- load_latest_json("^rust_unified_.*\\.json$") %>%
  rename(rust_mem_bytes = mem_alloc_bytes)

# Merge memory data on method + n (and variant if available)
if (nrow(r_mem) > 0 && nrow(rust_mem) > 0) {
  join_cols <- intersect(c("method", "variant", "n"), intersect(names(r_mem), names(rust_mem)))
  mem_df <- inner_join(r_mem, rust_mem, by = join_cols) %>%
    filter(rust_mem_bytes > 0) %>%
    mutate(mem_ratio = r_mem_bytes / rust_mem_bytes)
  cat(sprintf("Memory data: %d matched entries\n", nrow(mem_df)))
} else {
  mem_df <- data.frame()
  cat("WARNING: No memory data available\n")
}

# Check for modules with 0 matches
all_modules <- c("regression", "panel", "discrete", "timeseries", "ml",
                 "causal", "stats", "survival", "treatment")
missing_modules <- setdiff(all_modules, unique(matched$module))
if (length(missing_modules) > 0) {
  message(sprintf("WARNING: No matched benchmarks for module(s): %s",
                  paste(missing_modules, collapse = ", ")))
}

# Use largest-n per method for summary figures
largest_n <- matched %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  slice(1) %>%
  ungroup()

# Module color palette (consistent across all figures)
module_colors <- c(
  "regression" = "#2196F3", "panel" = "#4CAF50", "discrete" = "#FF9800",
  "timeseries" = "#9C27B0", "ml" = "#F44336", "causal" = "#00BCD4",
  "stats" = "#607D8B", "survival" = "#E91E63", "treatment" = "#3F51B5",
  "Other" = "#9E9E9E"
)

# Pretty module names for display
module_labels <- c(
  "regression" = "Regression", "panel" = "Panel", "discrete" = "Discrete",
  "timeseries" = "Time Series", "ml" = "ML", "causal" = "Causal",
  "stats" = "Stats", "survival" = "Survival", "treatment" = "Treatment"
)

# ============================================================================
# Figure 1: Speedup Histogram (benchmark_histogram)
# ============================================================================
cat("\nGenerating benchmark_histogram...\n")

med_speedup <- median(largest_n$speedup)

p_hist <- ggplot(largest_n, aes(x = speedup, fill = speedup >= 1)) +
  geom_histogram(bins = 25, color = "white", linewidth = 0.3) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  geom_vline(xintercept = med_speedup,
             linetype = "solid", color = "#1B5E20", linewidth = 0.7) +
  annotate("text",
           x = med_speedup * 1.3,
           y = Inf, vjust = 2,
           label = sprintf("Median: %.1fx", med_speedup),
           color = "#1B5E20", size = 4.5, fontface = "bold") +
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
           vjust = 2, hjust = 0.5, color = "#0097A7", fontface = "bold", size = 4.5) +
  annotate("text", x = 500, y = Inf, label = "Rust faster",
           vjust = 2, hjust = 0.5, color = "#E65100", fontface = "bold", size = 4.5) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.margin = margin(15, 15, 10, 10)
  )

ggsave(file.path(fig_dir, "benchmark_histogram.pdf"), p_hist, width = 7, height = 4)
ggsave(file.path(fig_dir, "benchmark_histogram.png"), p_hist, width = 7, height = 4, dpi = 300)
cat("  Saved benchmark_histogram.pdf/png\n")

# ============================================================================
# Figure 2: Speedup Bar Chart - Top 30 (benchmark_speedup)
# ============================================================================
cat("Generating benchmark_speedup...\n")

# Select top 3 per module by |log10(speedup)|, then top 30 overall
representative <- largest_n %>%
  group_by(module) %>%
  arrange(desc(abs(log10(speedup)))) %>%
  slice_head(n = 3) %>%
  ungroup() %>%
  arrange(desc(speedup)) %>%
  slice_head(n = 30)

representative <- representative %>%
  mutate(
    method_label = paste0(method, " (n=", format(n, big.mark = ","), ")"),
    method_label = factor(method_label, levels = rev(method_label)),
    speedup_label = ifelse(speedup >= 1,
                           sprintf("%.0fx", speedup),
                           sprintf("%.2fx", speedup)),
    module_pretty = ifelse(module %in% names(module_labels),
                           module_labels[module], module)
  )

p_speedup <- ggplot(representative, aes(x = method_label, y = speedup, fill = module)) +
  geom_col(width = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.5) +
  geom_text(aes(label = speedup_label), hjust = -0.1, size = 3, color = "gray30") +
  coord_flip() +
  scale_y_log10(
    name = "Speedup Factor (R time / Rust time)",
    breaks = c(0.01, 0.1, 1, 10, 100, 1000, 10000),
    labels = c("0.01x", "0.1x", "1x", "10x", "100x", "1000x", "10000x")
  ) +
  scale_x_discrete(name = NULL) +
  scale_fill_manual(values = module_colors, name = "Category",
                    labels = module_labels) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 12),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 10),
    plot.margin = margin(10, 20, 10, 10)
  )

ggsave(file.path(fig_dir, "benchmark_speedup.pdf"), p_speedup, width = 9, height = 8)
ggsave(file.path(fig_dir, "benchmark_speedup.png"), p_speedup, width = 9, height = 8, dpi = 300)
cat("  Saved benchmark_speedup.pdf/png\n")

# ============================================================================
# Figure 3: R vs Rust Execution Time Comparison (benchmark_boxplots)
# ============================================================================
cat("Generating benchmark_boxplots...\n")

# Select ~16 representative methods spanning all modules
selected_methods <- c("OLS", "FixedEffects", "HDFE", "Logit", "Probit",
                      "K-Means", "PCA", "ARIMA", "DiD", "IV_2SLS",
                      "CoxPH", "KM", "t_test", "DBSCAN", "Poisson", "GARCH")

# Also try alternative names that may exist in the data
alt_methods <- c("Holt_Winters", "VAR", "STL", "Matching", "IPW",
                 "NLS", "MSTL", "Hausman", "RandomEffects")

# Use what we can find
available <- union(
  intersect(selected_methods, matched$method),
  intersect(alt_methods, matched$method)
)

# Prefer selected_methods, fill remaining slots from alt_methods
box_methods <- intersect(selected_methods, available)
if (length(box_methods) < 16) {
  extras <- setdiff(intersect(alt_methods, available), box_methods)
  box_methods <- c(box_methods, head(extras, 16 - length(box_methods)))
}

box_data <- matched %>%
  filter(method %in% box_methods) %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  slice(1) %>%
  ungroup()

if (nrow(box_data) > 0) {
  # Pivot to long format for R vs Rust comparison
  box_long <- box_data %>%
    select(method, module, n, r_median_us, rust_median_us) %>%
    pivot_longer(
      cols = c(r_median_us, rust_median_us),
      names_to = "impl_col",
      values_to = "median_us"
    ) %>%
    mutate(
      implementation = ifelse(grepl("^r_", impl_col), "R", "p2a (Rust)"),
      label = paste0(method, "\n(n=", format(n, big.mark = ","), ")"),
      median_ms = median_us / 1000
    )

  # Order by module then method
  method_order <- box_long %>%
    arrange(module, method) %>%
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
    theme_bw(base_size = 14) +
    theme(
      legend.position = "top",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )

  ggsave(file.path(fig_dir, "benchmark_boxplots.pdf"), p_box, width = 10, height = 5, dpi = 300)
  ggsave(file.path(fig_dir, "benchmark_boxplots.png"), p_box, width = 10, height = 5, dpi = 300)
  cat("  Saved benchmark_boxplots.pdf/png\n")
} else {
  cat("  WARNING: No data for boxplot methods, skipping\n")
}

# ============================================================================
# Figure 4: Memory Comparison (benchmark_memory)
# ============================================================================
cat("Generating benchmark_memory...\n")

if (nrow(mem_df) > 0) {
  mem_largest <- mem_df %>%
    group_by(method) %>%
    filter(n == max(n)) %>%
    slice(1) %>%
    ungroup()

  # Select top 25 methods by memory ratio for readability
  mem_top <- mem_largest %>%
    arrange(desc(mem_ratio)) %>%
    slice_head(n = 25) %>%
    mutate(
      r_mem_kb = r_mem_bytes / 1024,
      rust_mem_kb = rust_mem_bytes / 1024,
      method_label = paste0(method, " (n=", format(n, big.mark = ","), ")"),
      method_label = reorder(method_label, mem_ratio)
    )

  mem_long <- mem_top %>%
    pivot_longer(cols = c(r_mem_kb, rust_mem_kb),
                 names_to = "language",
                 values_to = "mem_kb") %>%
    mutate(language = ifelse(language == "r_mem_kb", "R", "Rust"))

  # Use bytes directly so log scale baseline is 1 byte (all bars go right)
  mem_long <- mem_long %>%
    mutate(mem_bytes = mem_kb * 1024)

  p_mem <- ggplot(mem_long, aes(x = method_label, y = mem_bytes, fill = language)) +
    geom_col(position = "dodge", width = 0.7) +
    scale_fill_manual(values = c("R" = "#0097A7", "Rust" = "#E65100"), name = NULL) +
    scale_y_log10(
      name = "Memory Allocation per Call (log scale)",
      breaks = c(1, 1e3, 1e6, 1e9),
      labels = c("1 B", "1 KB", "1 MB", "1 GB"),
      limits = c(1, NA)
    ) +
    coord_flip() +
    labs(x = NULL) +
    theme_minimal(base_size = 14) +
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
} else {
  cat("  WARNING: No memory data available, skipping memory figure\n")
}

# ============================================================================
# Figure 5: Speedup by Module - Violin (benchmark_speedup_violin)
# ============================================================================
cat("Generating benchmark_speedup_violin...\n")

# Use all matched data (not just largest-n)
violin_data <- matched %>%
  mutate(module_pretty = ifelse(module %in% names(module_labels),
                                module_labels[module], module))

p_violin <- ggplot(violin_data, aes(x = module_pretty, y = speedup, fill = module)) +
  geom_violin(alpha = 0.6, draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_jitter(width = 0.15, size = 1.2, alpha = 0.4) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  scale_y_log10(
    name = "Speedup Factor (log scale)",
    labels = function(x) sprintf("%.0fx", x)
  ) +
  scale_fill_manual(values = module_colors, guide = "none") +
  labs(x = NULL) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 11))

ggsave(file.path(fig_dir, "benchmark_speedup_violin.pdf"), p_violin, width = 10, height = 5)
ggsave(file.path(fig_dir, "benchmark_speedup_violin.png"), p_violin, width = 10, height = 5, dpi = 300)
cat("  Saved benchmark_speedup_violin.pdf/png\n")

# ============================================================================
# Figure 6: Validation / Correctness Agreement (benchmark_validation)
# ============================================================================
cat("Generating benchmark_validation...\n")

# Compute validation status per method at largest n
validation_data <- unified_df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(
    module_pretty = ifelse(module %in% names(module_labels),
                           module_labels[module], module),
    status = case_when(
      is.na(outputs_agree) ~ "Speed-only",
      outputs_agree == TRUE ~ "Agree",
      outputs_agree == FALSE ~ "Disagree"
    )
  )

# Summarize by module
val_summary <- validation_data %>%
  count(module_pretty, status) %>%
  mutate(status = factor(status, levels = c("Agree", "Disagree", "Speed-only")))

status_colors <- c("Agree" = "#4CAF50", "Disagree" = "#F44336", "Speed-only" = "#9E9E9E")

p_val <- ggplot(val_summary, aes(x = reorder(module_pretty, -n, sum), y = n, fill = status)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = status_colors, name = "Output Comparison") +
  labs(x = NULL, y = "Number of Methods") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 30, hjust = 1, size = 11),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "benchmark_validation.pdf"), p_val, width = 9, height = 5)
ggsave(file.path(fig_dir, "benchmark_validation.png"), p_val, width = 9, height = 5, dpi = 300)
cat("  Saved benchmark_validation.pdf/png\n")

# ============================================================================
# Print Summary Stats
# ============================================================================
cat("\n=== Summary Statistics ===\n")
cat(sprintf("Total unified entries: %d\n", nrow(unified_df)))
cat(sprintf("Matched speed benchmarks: %d\n", nrow(matched)))
cat(sprintf("Unique matched methods: %d\n", n_distinct(matched$method)))
cat(sprintf("Overall median speedup: %.1fx\n", median(matched$speedup)))
cat(sprintf("Methods where Rust faster: %d (%.0f%%)\n",
            sum(largest_n$speedup >= 1),
            100 * mean(largest_n$speedup >= 1)))
cat(sprintf("Methods where R faster: %d (%.0f%%)\n",
            sum(largest_n$speedup < 1),
            100 * mean(largest_n$speedup < 1)))

if (nrow(mem_df) > 0) {
  mem_largest_all <- mem_df %>%
    group_by(method) %>%
    filter(n == max(n)) %>%
    slice(1) %>%
    ungroup()
  cat(sprintf("\nMemory: %d matched methods\n", nrow(mem_largest_all)))
  cat(sprintf("Median memory ratio (R/Rust): %.0fx\n",
              median(mem_largest_all$mem_ratio, na.rm = TRUE)))
}

# Validation summary
cat(sprintf("\nValidation: %d agree, %d disagree, %d speed-only\n",
            sum(validation_data$status == "Agree"),
            sum(validation_data$status == "Disagree"),
            sum(validation_data$status == "Speed-only")))

cat("\n--- Per Module (largest n) ---\n")
largest_n %>%
  group_by(module) %>%
  summarise(
    methods = n_distinct(method),
    median_speedup = median(speedup),
    .groups = "drop"
  ) %>%
  arrange(desc(median_speedup)) %>%
  {for (i in 1:nrow(.)) {
    cat(sprintf("  %-15s: %2d methods, median %.1fx\n",
                .$module[i], .$methods[i], .$median_speedup[i]))
  }}

cat("\nDone.\n")
