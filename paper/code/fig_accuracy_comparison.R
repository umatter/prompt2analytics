# ============================================================================
# fig_accuracy_comparison.R
# Bar chart comparing overall accuracy across models (all evaluations)
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

INPUT <- "llm_eval/results/"
OUTPUT <- "../figures/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "comparison_summary.json"))

# Include all models, mark partial evaluations
all_evals <- summary_data %>%
  mutate(
    model_display = case_when(
      model == "claude-3-5-haiku-20241022" ~ "Claude 3.5 Haiku",
      model == "qwen-2.5-72b-instruct" ~ "Qwen 2.5 72B",
      model == "llama-3.3-70b-instruct" ~ "Llama 3.3 70B",
      model == "gpt-4o-mini" ~ "GPT-4o Mini",
      model == "gpt-5-nano-2025-08-07" ~ "GPT-5 Nano",
      model == "gpt-4.1-nano-2025-04-14" ~ "GPT-4.1 Nano",
      model == "ministral-3b" ~ "Ministral 3B",
      model == "nemotron-nano-9b-v2" ~ "Nemotron 9B",
      model == "qwen3-8b" ~ "Qwen3 8B",
      model == "deepseek-chat" ~ "DeepSeek V3",
      TRUE ~ model
    ),
    provider_display = case_when(
      provider == "openai" ~ "OpenAI",
      provider == "anthropic" ~ "Anthropic",
      provider == "openrouter" ~ "Open Source",
      TRUE ~ provider
    ),
    is_partial = total < 87,
    accuracy_label = paste0(format(accuracy, nsmall = 1), "%")
  ) %>%
  arrange(desc(accuracy), desc(total)) %>%
  mutate(model_display = factor(model_display, levels = model_display))

## PLOT ----
p <- ggplot(all_evals, aes(x = model_display, y = accuracy, fill = provider_display)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = accuracy_label), vjust = -0.3, size = 4.5) +
  scale_y_continuous(
    limits = c(0, 108),
    breaks = seq(0, 100, 20),
    labels = function(x) paste0(x, "%"),
    expand = c(0, 0)
  ) +
  scale_fill_manual(
    values = c(
      "OpenAI" = "#74aa9c",
      "Anthropic" = "#d4a27f",
      "Open Source" = "#7facd4"
    ),
    name = "Provider"
  ) +
  labs(
    x = NULL,
    y = "Accuracy",
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    axis.title.y = element_text(size = 15),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    plot.margin = margin(10, 20, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "fig_accuracy_comparison.pdf"),
  plot = p,
  width = 8.5,
  height = 5.5
)

message("Created: ", OUTPUT, "fig_accuracy_comparison.pdf")
