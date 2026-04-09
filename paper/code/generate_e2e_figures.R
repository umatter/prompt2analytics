#!/usr/bin/env Rscript
# ============================================================================
# generate_e2e_figures.R
# Generate end-to-end evaluation figures and tables for the paper
#
# Reads from: paper/code/e2e_eval/results/
#   - summary.json (single-turn API results for 6 models)
#   - chrome_results_claude-sonnet-4.6.json (Chrome multi-turn results)
# Outputs to: paper/figures/ and paper/tables/
#
# Usage: cd paper/code && Rscript generate_e2e_figures.R
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(scales)
})

# Paths
script_dir <- tryCatch(
  dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))),
  error = function(e) getwd()
)
paper_dir <- dirname(script_dir)
project_root <- dirname(paper_dir)
results_dir <- file.path(paper_dir, "code/e2e_eval/results")
fig_dir <- file.path(paper_dir, "figures")
tables_dir <- file.path(paper_dir, "tables")

cat("=== Generating E2E Evaluation Figures and Tables ===\n")
cat("Results dir:", results_dir, "\n")
cat("Output dir (figs):", fig_dir, "\n")
cat("Output dir (tabs):", tables_dir, "\n\n")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Load data
# ============================================================================

# Single-turn API results
summary_data <- fromJSON(file.path(results_dir, "summary.json"))
cat(sprintf("Single-turn data: %d models\n", nrow(summary_data)))

# Chrome multi-turn results
chrome_data <- fromJSON(file.path(results_dir, "chrome_results_claude-sonnet-4.6.json"))
cat(sprintf("Chrome MT data: %d conversations, score %d/%d (%.1f%%)\n",
            length(chrome_data$multi_turn),
            chrome_data$summary$multi_turn_total_score,
            chrome_data$summary$multi_turn_max_score,
            chrome_data$summary$multi_turn_accuracy * 100))

# ============================================================================
# Theme
# ============================================================================

theme_paper <- theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 15),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 12),
    legend.text = element_text(size = 11),
    strip.text = element_text(face = "bold", size = 12)
  )

# Model display names (valid models only — excludes Opus 4.6, Sonnet 4, Qwen 2.5
# due to incomplete runs from API credit exhaustion / context overflow)
model_labels <- c(
  "gpt-4.1-mini" = "GPT-4.1 Mini",
  "claude-sonnet-4.6" = "Sonnet 4.6",
  "claude-haiku-4.5" = "Haiku 4.5",
  "ministral-3b" = "Ministral 3B",
  "gemini-2.5-flash" = "Gemini 2.5 Flash",
  "llama-4-scout" = "Llama 4 Scout"
)

# Filter summary data to valid models only
summary_data <- summary_data %>%
  filter(model %in% names(model_labels))

# ============================================================================
# Figure 1: Adequate rate by model (bar chart)
# ============================================================================

model_df <- summary_data %>%
  mutate(
    label = model_labels[model],
    label = factor(label, levels = model_labels[order(-summary_data$adequate_rate)])
  )

p1 <- ggplot(model_df, aes(x = reorder(label, adequate_rate), y = adequate_rate * 100)) +
  geom_col(fill = "#2C7BB6", width = 0.65) +
  geom_text(aes(label = sprintf("%.1f%%", adequate_rate * 100)),
            hjust = -0.15, size = 4.5) +
  coord_flip(ylim = c(0, 100)) +
  labs(x = NULL, y = "Adequate rate (%)") +
  theme_paper +
  theme(panel.grid.major.y = element_blank())

ggsave(file.path(fig_dir, "e2e_adequate_rate.pdf"), p1, width = 7, height = 4)
ggsave(file.path(fig_dir, "e2e_adequate_rate.png"), p1, width = 7, height = 4, dpi = 300)
cat("Saved: e2e_adequate_rate.pdf/png\n")

# ============================================================================
# Table: Combined model info + adequate rates (LaTeX)
# ============================================================================

version_ids <- c(
  "gpt-4.1-mini" = "openai/gpt-4.1-mini",
  "claude-sonnet-4.6" = "claude-sonnet-4-6",
  "claude-haiku-4.5" = "claude-haiku-4-5-20251001",
  "ministral-3b" = "mistralai/ministral-3b-2512",
  "gemini-2.5-flash" = "google/gemini-2.5-flash",
  "llama-4-scout" = "meta-llama/llama-4-scout"
)

providers <- c(
  "gpt-4.1-mini" = "OpenRouter (OpenAI)",
  "claude-sonnet-4.6" = "Anthropic API",
  "claude-haiku-4.5" = "Anthropic API",
  "ministral-3b" = "OpenRouter",
  "gemini-2.5-flash" = "OpenRouter (Google)",
  "llama-4-scout" = "OpenRouter (Meta)"
)

param_counts <- c(
  "gpt-4.1-mini" = "---",
  "claude-sonnet-4.6" = "---",
  "claude-haiku-4.5" = "---",
  "ministral-3b" = "3B",
  "gemini-2.5-flash" = "---",
  "llama-4-scout" = "109B (17B active)"
)

combined_df <- summary_data %>%
  arrange(desc(adequate_rate)) %>%
  mutate(
    label = model_labels[model],
    version_id = version_ids[model],
    provider = providers[model],
    params = param_counts[model],
    rate_pct = sprintf("%.1f", adequate_rate * 100)
  )

combined_tex <- paste0(
  "\\begin{table}[htbp]\n",
  "\\centering\n",
  "\\begin{threeparttable}\n",
  "\\caption{Models evaluated and single-turn adequate rates}\n",
  "\\label{tab:model-results}\n",
  "\\small\n",
  "\\begin{tabular}{lllr}\n",
  "\\toprule\n",
  "Model & Version ID & Parameters & Adequate (\\%) \\\\\n",
  "\\midrule\n"
)

for (i in seq_len(nrow(combined_df))) {
  r <- combined_df[i, ]
  combined_tex <- paste0(combined_tex, sprintf(
    "%s & \\code{%s} & %s & %s \\\\\n",
    r$label, r$version_id, r$params, r$rate_pct
  ))
}

combined_tex <- paste0(combined_tex,
  "\\bottomrule\n",
  "\\end{tabular}\n",
  "\\begin{tablenotes}[flushleft]\n",
  "\\item \\emph{Note:} 96~single-turn prompts (32~test cases $\\times$\n",
  "3~clarity levels), temperature~$= 0$. Accessed March~2026.\n",
  "Cloud-served model weights may be updated without notice;\n",
  "for reproducible local deployment, pin Ollama models by digest.\n",
  "Parameter counts marked ``---'' are not publicly disclosed.\n",
  "\\end{tablenotes}\n",
  "\\end{threeparttable}\n",
  "\\end{table}\n"
)

writeLines(combined_tex, file.path(tables_dir, "tab_model_results.tex"))
cat("Saved: tab_model_results.tex\n")

# ============================================================================
# Figure 2: Adequate rate by clarity level (grouped bar)
# ============================================================================

clarity_df <- summary_data %>%
  rowwise() %>%
  mutate(
    precise = by_clarity$precise$adequate / by_clarity$precise$total * 100,
    moderate = by_clarity$moderate$adequate / by_clarity$moderate$total * 100,
    vague = by_clarity$vague$adequate / by_clarity$vague$total * 100,
    label = model_labels[model]
  ) %>%
  ungroup() %>%
  pivot_longer(cols = c(precise, moderate, vague),
               names_to = "clarity", values_to = "rate") %>%
  mutate(
    clarity = factor(clarity, levels = c("precise", "moderate", "vague"),
                     labels = c("Precise", "Moderate", "Vague")),
    label = factor(label, levels = rev(model_labels[order(-summary_data$adequate_rate)]))
  )

p2 <- ggplot(clarity_df, aes(x = label, y = rate, fill = clarity)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  scale_fill_manual(values = c("Precise" = "#2C7BB6", "Moderate" = "#ABD9E9",
                                "Vague" = "#FDAE61"),
                    name = "Prompt clarity") +
  coord_flip() +
  labs(x = NULL, y = "Adequate rate (%)") +
  theme_paper

ggsave(file.path(fig_dir, "e2e_clarity_effect.pdf"), p2, width = 8, height = 4.5)
ggsave(file.path(fig_dir, "e2e_clarity_effect.png"), p2, width = 8, height = 4.5, dpi = 300)
cat("Saved: e2e_clarity_effect.pdf/png\n")

# ============================================================================
# Figure 3: Category heatmap (adequate rate by category x model)
# ============================================================================

# Map from summary category names to display labels
cat_labels <- c(
  "Regression" = "Regression", "Panel data" = "Panel",
  "Causal inference" = "Causal", "Time series" = "Time series",
  "Hypothesis testing" = "Hypothesis", "Discrete choice" = "Discrete",
  "Machine learning" = "ML", "Messy data" = "Messy data"
)

cat_df <- summary_data %>%
  rowwise() %>%
  mutate(label = model_labels[model]) %>%
  ungroup()

cat_long <- do.call(rbind, lapply(seq_len(nrow(cat_df)), function(i) {
  row <- cat_df[i, ]
  cats <- row$by_category
  do.call(rbind, lapply(names(cats), function(cname) {
    cat_data <- cats[[cname]]
    adequate <- if (is.data.frame(cat_data)) cat_data$adequate[1] else cat_data$adequate
    total <- if (is.data.frame(cat_data)) cat_data$total[1] else cat_data$total
    rate <- if (total > 0) adequate / total * 100 else 0
    disp <- if (cname %in% names(cat_labels)) cat_labels[[cname]] else cname
    data.frame(
      model = row$label,
      category = disp,
      rate = rate,
      stringsAsFactors = FALSE
    )
  }))
}))

cat_long$model <- factor(cat_long$model,
                         levels = model_labels[order(-summary_data$adequate_rate)])
cat_long$category <- factor(cat_long$category,
                            levels = rev(unique(unname(cat_labels))))

p3 <- ggplot(cat_long, aes(x = model, y = category, fill = rate)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", rate)), size = 4) +
  scale_fill_gradient2(low = "#D73027", mid = "#FFFFBF", high = "#1A9850",
                       midpoint = 50, limits = c(0, 100),
                       name = "Adequate %") +
  labs(x = NULL, y = NULL) +
  theme_paper +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        panel.grid = element_blank())

ggsave(file.path(fig_dir, "e2e_category_heatmap.pdf"), p3, width = 8, height = 5)
ggsave(file.path(fig_dir, "e2e_category_heatmap.png"), p3, width = 8, height = 5, dpi = 300)
cat("Saved: e2e_category_heatmap.pdf/png\n")

# ============================================================================
# Figure 4: Chrome multi-turn scores by conversation
# ============================================================================

mt_scores <- chrome_data$summary$per_conversation_scores
mt_names <- names(mt_scores)
mt_df <- data.frame(
  conversation = mt_names,
  score = sapply(mt_names, function(n) mt_scores[[n]]$score),
  max_score = sapply(mt_names, function(n) mt_scores[[n]]$max),
  pct = sapply(mt_names, function(n) mt_scores[[n]]$pct * 100),
  stringsAsFactors = FALSE,
  row.names = NULL
)

# Add descriptions
mt_descriptions <- c(
  MT1 = "OLS + robust SE\n+ diagnostics",
  MT2 = "Panel FE/RE\n+ Hausman",
  MT3 = "Time series\nARIMA + forecast",
  MT4 = "Causal: OLS +\nDiD + IPW",
  MT5 = "Data cleaning\n+ regression",
  MT6 = "Discrete:\nlogit/ordered/NB",
  MT7 = "OLS + diagnostics\n+ robust SE",
  MT8 = "Cross-dataset\ncomparison"
)
mt_df$desc <- mt_descriptions[mt_df$conversation]
mt_df$conversation <- factor(mt_df$conversation, levels = paste0("MT", 1:8))

p4 <- ggplot(mt_df, aes(x = conversation, y = pct)) +
  geom_col(fill = "#2C7BB6", width = 0.65) +
  geom_text(aes(label = sprintf("%d/%d", score, max_score)),
            vjust = -0.5, size = 4) +
  geom_hline(yintercept = 96.7, linetype = "dashed", color = "#D73027",
             linewidth = 0.5) +
  annotate("text", x = 8.4, y = 96.7, label = "Mean: 96.7%",
           hjust = 1, vjust = -0.5, size = 3.5, color = "#D73027") +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, 25)) +
  scale_x_discrete(labels = mt_df$desc) +
  labs(x = NULL, y = "Score (%)") +
  theme_paper +
  theme(axis.text.x = element_text(size = 9, lineheight = 0.9))

ggsave(file.path(fig_dir, "e2e_chrome_mt_scores.pdf"), p4, width = 9, height = 5)
ggsave(file.path(fig_dir, "e2e_chrome_mt_scores.png"), p4, width = 9, height = 5, dpi = 300)
cat("Saved: e2e_chrome_mt_scores.pdf/png\n")

# ============================================================================
# Figure 5: Infrastructure improvement trajectory
# ============================================================================

fix_history <- chrome_data$summary$infrastructure_fix_history
trajectory_df <- data.frame(
  stage = c("Pre-fix\n(API tests)", "Round 1\n(router fix)", "Round 2\n(tool defs)"),
  score = c(fix_history$score_before[1],
            fix_history$score_after[1],
            fix_history$score_after[2]),
  stringsAsFactors = FALSE
)
trajectory_df$stage <- factor(trajectory_df$stage, levels = trajectory_df$stage)
trajectory_df$pct <- trajectory_df$score / 240 * 100

p5 <- ggplot(trajectory_df, aes(x = stage, y = pct, group = 1)) +
  geom_line(color = "#2C7BB6", linewidth = 1.2) +
  geom_point(color = "#2C7BB6", size = 4) +
  geom_text(aes(label = sprintf("%.1f%%\n(%d/240)", pct, score)),
            vjust = -1, size = 4) +
  scale_y_continuous(limits = c(60, 105), breaks = seq(60, 100, 10)) +
  labs(x = NULL, y = "Multi-turn accuracy (%)") +
  theme_paper

ggsave(file.path(fig_dir, "e2e_improvement_trajectory.pdf"), p5, width = 6, height = 4.5)
ggsave(file.path(fig_dir, "e2e_improvement_trajectory.png"), p5, width = 6, height = 4.5, dpi = 300)
cat("Saved: e2e_improvement_trajectory.pdf/png\n")

# ============================================================================
# Figure 6: Per-dimension scores (Chrome MT, Sonnet 4.6)
# ============================================================================

dim_df <- data.frame(
  dimension = c("Tool\nselection", "Parameter\nextraction",
                "Numerical\ncorrectness", "Interpretation"),
  score = c(
    chrome_data$summary$per_dimension_averages$tool_selection,
    chrome_data$summary$per_dimension_averages$parameter_extraction,
    chrome_data$summary$per_dimension_averages$numerical_correctness,
    chrome_data$summary$per_dimension_averages$interpretation
  ),
  stringsAsFactors = FALSE
)
dim_df$dimension <- factor(dim_df$dimension, levels = dim_df$dimension)

p6 <- ggplot(dim_df, aes(x = dimension, y = score)) +
  geom_col(fill = "#2C7BB6", width = 0.6) +
  geom_text(aes(label = sprintf("%.2f", score)), vjust = -0.5, size = 4.5) +
  geom_hline(yintercept = 2.0, linetype = "dashed", color = "#999999") +
  scale_y_continuous(limits = c(0, 2.3), breaks = c(0, 0.5, 1.0, 1.5, 2.0)) +
  labs(x = NULL, y = "Average score (max 2.0)") +
  theme_paper

ggsave(file.path(fig_dir, "e2e_dimension_scores.pdf"), p6, width = 7, height = 4)
ggsave(file.path(fig_dir, "e2e_dimension_scores.png"), p6, width = 7, height = 4, dpi = 300)
cat("Saved: e2e_dimension_scores.pdf/png\n")

# ============================================================================
# Table 1: Chrome multi-turn results (LaTeX)
# ============================================================================

mt_tex <- paste0(
  "\\begin{table}[htbp]\n",
  "\\centering\n",
  "\\begin{threeparttable}\n",
  "\\caption{Chrome multi-turn evaluation: Claude Sonnet~4.6}\n",
  "\\label{tab:e2e-chrome-mt}\n",
  "\\small\n",
  "\\begin{tabular}{llrrr}\n",
  "\\toprule\n",
  "ID & Workflow & Turns & Score & \\% \\\\\n",
  "\\midrule\n"
)

for (i in seq_len(nrow(mt_df))) {
  # Clean description for LaTeX (remove newlines)
  desc <- gsub("\n", " ", mt_descriptions[paste0("MT", i)])
  n_turns <- chrome_data$multi_turn$total_turns[i]
  mt_tex <- paste0(mt_tex, sprintf("MT%d & %s & %d & %d/%d & %.1f \\\\\n",
                                    i, desc, n_turns,
                                    mt_df$score[i], mt_df$max_score[i],
                                    mt_df$pct[i]))
}

mt_tex <- paste0(mt_tex,
  "\\midrule\n",
  sprintf("\\textbf{Total} & & \\textbf{30} & \\textbf{%d/%d} & \\textbf{%.1f} \\\\\n",
          chrome_data$summary$multi_turn_total_score,
          chrome_data$summary$multi_turn_max_score,
          chrome_data$summary$multi_turn_accuracy * 100),
  "\\bottomrule\n",
  "\\end{tabular}\n",
  "\\begin{tablenotes}[flushleft]\n",
  "\\item \\emph{Note:} Evaluated through the Dioxus web frontend using\n",
  "automated Chrome browser interaction. Each turn scored on tool selection,\n",
  "parameter extraction, numerical correctness, and interpretation (max 8\n",
  "points). Context retention: 100\\% across all conversations.\n",
  "\\end{tablenotes}\n",
  "\\end{threeparttable}\n",
  "\\end{table}\n"
)

writeLines(mt_tex, file.path(tables_dir, "tab_e2e_chrome_mt.tex"))
cat("Saved: tab_e2e_chrome_mt.tex\n")

# ============================================================================
# Table 2: Infrastructure improvement summary (LaTeX)
# ============================================================================

imp_tex <- paste0(
  "\\begin{table}[htbp]\n",
  "\\centering\n",
  "\\begin{threeparttable}\n",
  "\\caption{Infrastructure improvement trajectory}\n",
  "\\label{tab:e2e-improvement}\n",
  "\\small\n",
  "\\begin{tabular}{lp{6.5cm}rr}\n",
  "\\toprule\n",
  "Stage & Fix description & Score & Accuracy \\\\\n",
  "\\midrule\n",
  sprintf("Pre-fix & API-only single-turn evaluation (baseline) & %d/240 & %.1f\\%% \\\\\n",
          fix_history$score_before[1], fix_history$score_before[1] / 240 * 100),
  sprintf("Round~1 & %s & %d/240 & %.1f\\%% \\\\\n",
          fix_history$description[1], fix_history$score_after[1],
          fix_history$score_after[1] / 240 * 100),
  sprintf("Round~2 & %s & %d/240 & %.1f\\%% \\\\\n",
          gsub("_", "\\\\_", fix_history$description[2]),
          fix_history$score_after[2],
          fix_history$score_after[2] / 240 * 100),
  "\\bottomrule\n",
  "\\end{tabular}\n",
  "\\begin{tablenotes}[flushleft]\n",
  "\\item \\emph{Note:} Scores are for 8 multi-turn conversations\n",
  "(30 total turns, max 240 points) with Claude Sonnet~4.6 via Chrome.\n",
  "Round~1 replaced a manual 70-tool HTTP match with router dispatch\n",
  "(268 tools). Round~2 fixed LLM tool definitions and added missing tools.\n",
  "\\end{tablenotes}\n",
  "\\end{threeparttable}\n",
  "\\end{table}\n"
)

writeLines(imp_tex, file.path(tables_dir, "tab_e2e_improvement.tex"))
cat("Saved: tab_e2e_improvement.tex\n")

# ============================================================================
# Table 3: Per-dimension averages (LaTeX)
# ============================================================================

dim_tex <- paste0(
  "\\begin{table}[htbp]\n",
  "\\centering\n",
  "\\begin{threeparttable}\n",
  "\\caption{Per-dimension scoring averages: Chrome multi-turn evaluation}\n",
  "\\label{tab:e2e-dimensions}\n",
  "\\small\n",
  "\\begin{tabular}{lrrr}\n",
  "\\toprule\n",
  "Dimension & Average & Max & Perfect (\\%) \\\\\n",
  "\\midrule\n",
  sprintf("Tool selection & %.2f & 2.00 & %.0f \\\\\n",
          dim_df$score[1], dim_df$score[1] / 2 * 100),
  sprintf("Parameter extraction & %.2f & 2.00 & %.0f \\\\\n",
          dim_df$score[2], dim_df$score[2] / 2 * 100),
  sprintf("Numerical correctness & %.2f & 2.00 & %.0f \\\\\n",
          dim_df$score[3], dim_df$score[3] / 2 * 100),
  sprintf("Interpretation & %.2f & 2.00 & %.0f \\\\\n",
          dim_df$score[4], dim_df$score[4] / 2 * 100),
  "\\bottomrule\n",
  "\\end{tabular}\n",
  "\\begin{tablenotes}[flushleft]\n",
  "\\item \\emph{Note:} Averages across 30 turns in 8 multi-turn\n",
  "conversations. Claude Sonnet~4.6 via Chrome frontend.\n",
  "\\end{tablenotes}\n",
  "\\end{threeparttable}\n",
  "\\end{table}\n"
)

writeLines(dim_tex, file.path(tables_dir, "tab_e2e_dimensions.tex"))
cat("Saved: tab_e2e_dimensions.tex\n")

cat("\n=== Done. Generated 6 figures and 3 tables. ===\n")
