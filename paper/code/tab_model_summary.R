# ============================================================================
# tab_model_summary.R
# Comprehensive model comparison table (LaTeX)
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)
library(xtable)

INPUT <- "llm_eval/results/"
OUTPUT <- "../tables/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "comparison_summary.json"))

# Format all models for table
table_data <- summary_data %>%
  mutate(
    # Clean model names for display
    Model = case_when(
      model == "claude-3-5-haiku-20241022" ~ "Claude 3.5 Haiku",
      model == "qwen-2.5-72b-instruct" ~ "Qwen 2.5 72B",
      model == "llama-3.3-70b-instruct" ~ "Llama 3.3 70B",
      model == "gpt-4o-mini" ~ "GPT-4o Mini",
      model == "gpt-5-nano-2025-08-07" ~ "GPT-5 Nano",
      model == "gpt-4.1-nano-2025-04-14" ~ "GPT-4.1 Nano",
      model == "ministral-3b" ~ "Ministral 3B",
      model == "nemotron-nano-9b-v2" ~ "Nemotron 9B$^\\dagger$",
      model == "qwen3-8b" ~ "Qwen3 8B$^\\dagger$",
      model == "deepseek-chat" ~ "DeepSeek$^\\dagger$",
      TRUE ~ model
    ),
    Size = ifelse(is.na(size) | size == "null", "---", size),
    Provider = case_when(
      provider == "openai" ~ "OpenAI",
      provider == "anthropic" ~ "Anthropic",
      provider == "openrouter" ~ "OpenRouter",
      TRUE ~ provider
    ),
    N = total,
    Exact = exact,
    Acceptable = acceptable,
    Category = category,
    Failed = failed,
    `Accuracy (\\%)` = sprintf("%.1f", accuracy),
    `Latency (ms)` = format(avg_latency_ms, big.mark = ",")
  ) %>%
  arrange(desc(accuracy), desc(total)) %>%
  select(Model, Size, Provider, N, Exact, Acceptable, Category, Failed,
         `Accuracy (\\%)`, `Latency (ms)`)

## CREATE LATEX TABLE ----
latex_table <- xtable(
  table_data,
  caption = "LLM tool selection accuracy comparison. Models marked with $\\dagger$ had partial evaluations due to API errors. Exact = exact tool match; Acceptable = valid alternative tool; Category = correct category, wrong specific tool; Failed = invalid tool selection. Latency reflects cloud API response time.",
  label = "tab:model-summary",
  align = c("l", "l", "c", "l", "r", "r", "r", "r", "r", "r", "r")
)

## WRITE TO DISK ----
print(
  latex_table,
  file = paste0(OUTPUT, "tab_model_summary.tex"),
  include.rownames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = identity,
  caption.placement = "top",
  table.placement = "htbp",
  floating = TRUE
)

message("Created: ", OUTPUT, "tab_model_summary.tex")
