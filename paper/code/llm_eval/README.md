# LLM Tool Selection Evaluation Framework

A comprehensive script-based evaluation framework to test whether LLMs correctly select the appropriate p2a-mcp tool for natural language econometric prompts.

## Overview

This framework evaluates LLM performance across five evaluation dimensions:

1. **Single-Turn Tool Selection**: Baseline accuracy on synthetic prompts
2. **Multi-Turn Conversations**: Context retention across conversation turns
3. **Naturalistic Prompts**: Robustness to informal, verbose, and typo-ridden prompts
4. **Parameter Extraction**: Precision and recall for tool parameter extraction
5. **Result Interpretation**: Accuracy of econometric output interpretation
6. **Out-of-Scope Detection**: Recognition of unsupported methods

## Supported Providers

### Cloud Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1-nano`
- **Anthropic**: `claude-3-5-sonnet`, `claude-3-5-haiku`
- **OpenRouter**: Various Llama, Mistral, Qwen, Gemma models

### Local Models (Ollama)
- `llama3.2`, `mistral`, `qwen2.5`

## Prerequisites

- `jq` - JSON processor
- `curl` - HTTP client
- `bc` - Calculator for percentages

Set API keys as needed:
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
export OPENROUTER_API_KEY="your-api-key"
```

For Ollama models:
```bash
ollama serve
ollama pull llama3.2
```

## Usage

### Single-Turn Evaluation (Baseline)

```bash
# Run all categories
./scripts/run_eval.sh gpt-4o openai all

# Run specific category
./scripts/run_eval.sh claude-3-5-sonnet anthropic regression

# Dry run
./scripts/run_eval.sh gpt-4o openai all --dry-run
```

### Multi-Turn Evaluation

Tests context retention across conversation turns (20 conversations, ~72 turns).

```bash
# All multi-turn categories
./scripts/run_multi_turn_eval.sh gpt-4o openai all

# Specific category
./scripts/run_multi_turn_eval.sh gpt-4o openai regression
```

### Naturalistic Prompts Evaluation

Tests robustness to informal, verbose, and error-prone prompts (~100 tests).

```bash
# All naturalistic tests
./scripts/run_naturalistic_eval.sh gpt-4o openai all

# Specific category
./scripts/run_naturalistic_eval.sh gpt-4o openai panel
```

### Parameter Extraction Evaluation

Tests ability to correctly extract tool parameters (~50 tests).

```bash
./scripts/run_parameter_eval.sh gpt-4o openai
```

### Interpretation Evaluation

Tests ability to correctly interpret econometric output (~26 tests).

```bash
# All interpretation categories
./scripts/run_interpretation_eval.sh gpt-4o openai all

# Specific category
./scripts/run_interpretation_eval.sh gpt-4o openai regression
```

### Out-of-Scope Detection

Tests recognition of unsupported methods (20 tests).

```bash
./scripts/run_oos_eval.sh gpt-4o openai
```

### Generate Report from Existing Results

```bash
./scripts/generate_report.sh results/gpt-4o_all_20260123_120000.jsonl
```

## Test Categories

### Single-Turn Tests (87 total)

| Category | Tests | Description |
|----------|-------|-------------|
| `regression` | 10 | OLS, robust SE, clustered SE, NLS |
| `panel` | 10 | Fixed/Random effects, Hausman, HDFE |
| `causal` | 11 | 2SLS, DiD, RD, IPW, synthetic control |
| `discrete` | 10 | Logit, Probit, FEGLM |
| `timeseries` | 12 | ARIMA, VAR, decomposition, unit root |
| `hypothesis` | 12 | t-test, ANOVA, chi-squared, normality |
| `ml` | 12 | Clustering, PCA, t-SNE, Random Forest |
| `viz` | 10 | Histograms, scatter, heatmaps |

### Multi-Turn Conversations (20 conversations, 72 turns)

| Category | Conversations | Example Workflow |
|----------|---------------|------------------|
| `regression` | 5 | OLS → robust SE → diagnostics |
| `panel` | 5 | FE vs RE → Hausman → HDFE |
| `causal` | 5 | IV setup → first stage → overid test |
| `timeseries` | 5 | Stationarity → ARIMA → forecast |

### Naturalistic Prompts (~100 total)

| Prompt Type | Description | Example |
|-------------|-------------|---------|
| `informal` | Casual language, abbreviations | "hey can u run a quick reg" |
| `verbose` | Over-explained, hedging | "I've been wondering if perhaps..." |
| `typos` | Misspellings, errors | "esitmate teh efect" |
| `domain_jargon` | Technical terminology | "Mincerian regression", "LATE" |
| `ambiguous` | Context-dependent meaning | "cluster these by company" |

### Parameter Extraction (50 tests)

Tests extraction of:
- Column names (`y_col`, `x_cols`, `cluster_var`)
- Numeric parameters (lags, k, horizon)
- Method specifications (`robust`, `kernel`, `linkage`)
- Categorical parameters (`family`, `estimand`)

### Interpretation (26 tests)

| Category | Tests | Example Tasks |
|----------|-------|---------------|
| `regression` | 8 | Log coefficient, interaction, R² |
| `hypothesis` | 10 | p-value, CI, power, multiple testing |
| `causal` | 8 | IV LATE, DiD assumptions, RD local effect |

### Out-of-Scope (20 tests)

Tests for methods NOT in p2a:
- Bayesian inference (MCMC, priors)
- GARCH/volatility models
- Quantile regression
- Spatial econometrics
- Mixed-effects GLM
- Deep learning
- Ordered/multinomial choice

## Scoring Methodology

### Tool Selection (Single-Turn, Multi-Turn, Naturalistic)

| Match Type | Score | Condition |
|------------|-------|-----------|
| exact | 1.0 | Selected tool matches expected exactly |
| acceptable | 0.7 | Selected tool in acceptable_tools list |
| category | 0.3 | Selected tool has correct category prefix |
| none | 0.0 | Wrong tool selected |

**Accuracy** = (exact + acceptable) / total

### Multi-Turn Additional Metrics

- **Turn Accuracy**: Per-turn success rate
- **Conversation Completion**: % of conversations with all turns correct
- **Context Sensitivity**: Improvement from having conversation context

### Naturalistic Additional Metrics

- **Robustness Score**: Minimum accuracy across prompt types
- **Accuracy Gap**: Single-turn accuracy - naturalistic accuracy

### Parameter Extraction Metrics

- **Precision**: Correct parameters / extracted parameters
- **Recall**: Correct parameters / expected parameters
- **F1**: Harmonic mean of precision and recall

### Interpretation Metrics

- **Element Coverage**: % of required interpretation elements mentioned
- **Error Rate**: % with critical misinterpretations
- **Accuracy**: Coverage × (1 - Error Rate)

### Out-of-Scope Metrics

- **OOS Detection Rate**: % correctly identified as out of scope
- **False Tool Rate**: % that incorrectly selected a tool
- **Graceful Degradation**: % that suggested alternatives

## Output Format

### Results (JSONL)

Single-turn result:
```json
{
  "timestamp": "2026-01-23T10:15:00Z",
  "model": "gpt-4o",
  "test_id": "reg_001",
  "category": "regression",
  "prompt": "Estimate the effect of firm value on investment using OLS",
  "selected": "regression_ols",
  "expected": ["regression_ols"],
  "match_type": "exact",
  "score": 1.0,
  "latency_ms": 450,
  "difficulty": "basic"
}
```

Multi-turn result:
```json
{
  "timestamp": "2026-01-23T10:20:00Z",
  "model": "gpt-4o",
  "conversation_id": "mt_reg_001",
  "turn": 2,
  "category": "regression",
  "prompt": "Use robust standard errors instead",
  "selected": "regression_ols",
  "expected": "regression_ols",
  "match_type": "exact",
  "context_from_previous": true
}
```

Parameter extraction result:
```json
{
  "test_id": "param_reg_001",
  "tool_match": true,
  "param_precision": 0.857,
  "param_recall": 1.0,
  "param_f1": 0.923
}
```

## Directory Structure

```
llm_eval/
├── README.md
├── config/
│   ├── models.json                # Model configurations
│   └── tools.json                 # MCP tool definitions
├── test_cases/
│   ├── regression.json            # Single-turn tests
│   ├── panel.json
│   ├── causal.json
│   ├── discrete.json
│   ├── timeseries.json
│   ├── hypothesis.json
│   ├── ml.json
│   ├── viz.json
│   ├── multi_turn/                # Multi-turn conversations
│   │   ├── regression_conversations.json
│   │   ├── panel_conversations.json
│   │   ├── causal_conversations.json
│   │   └── timeseries_conversations.json
│   ├── naturalistic/              # Naturalistic prompt variations
│   │   ├── regression_natural.json
│   │   ├── panel_natural.json
│   │   ├── causal_natural.json
│   │   ├── discrete_natural.json
│   │   ├── timeseries_natural.json
│   │   ├── hypothesis_natural.json
│   │   ├── ml_natural.json
│   │   └── viz_natural.json
│   ├── parameter_extraction/      # Parameter extraction tests
│   │   └── parameter_tests.json
│   ├── interpretation/            # Interpretation tests
│   │   ├── regression_interpretation.json
│   │   ├── hypothesis_interpretation.json
│   │   └── causal_interpretation.json
│   └── out_of_scope.json          # OOS detection tests
├── scripts/
│   ├── run_eval.sh                # Single-turn evaluation
│   ├── run_multi_turn_eval.sh     # Multi-turn evaluation
│   ├── run_naturalistic_eval.sh   # Naturalistic evaluation
│   ├── run_parameter_eval.sh      # Parameter extraction evaluation
│   ├── run_interpretation_eval.sh # Interpretation evaluation
│   ├── run_oos_eval.sh            # Out-of-scope evaluation
│   ├── build_prompt.sh            # Single-turn prompt construction
│   ├── build_multi_turn_prompt.sh # Multi-turn prompt construction
│   ├── call_openai.sh             # OpenAI API wrapper
│   ├── call_anthropic.sh          # Anthropic API wrapper
│   ├── call_openrouter.sh         # OpenRouter API wrapper
│   ├── call_ollama.sh             # Ollama API wrapper
│   ├── score_response.sh          # Tool selection scoring
│   ├── score_parameters.sh        # Parameter extraction scoring
│   ├── score_interpretation.sh    # Interpretation scoring
│   ├── score_oos_response.sh      # OOS detection scoring
│   └── generate_report.sh         # Report generation
└── results/
    ├── *.jsonl                    # Single-turn results
    ├── multi_turn/                # Multi-turn results
    ├── naturalistic/              # Naturalistic results
    ├── parameter_extraction/      # Parameter extraction results
    ├── interpretation/            # Interpretation results
    └── out_of_scope/              # OOS detection results
```

## Test Case Counts

| Type | New Tests | Description |
|------|-----------|-------------|
| Single-Turn (baseline) | 87 | Original test suite |
| Multi-Turn | ~72 turns (20 conversations) | Context retention |
| Naturalistic | ~100 | Prompt robustness |
| Parameter Extraction | 50 | Tool parameter accuracy |
| Interpretation | 26 | Output interpretation |
| Out-of-Scope | 20 | Method boundary detection |
| **Total** | **~355** | |

## Adding New Test Cases

### Single-Turn Tests

Add to the appropriate category file in `test_cases/`:
```json
{
  "id": "reg_011",
  "prompt": "Your prompt here",
  "expected_tools": ["regression_ols"],
  "acceptable_tools": ["regression_ols", "regression_clustered"],
  "dataset_context": "dataset description",
  "difficulty": "intermediate"
}
```

### Multi-Turn Conversations

Add to `test_cases/multi_turn/{category}_conversations.json`:
```json
{
  "id": "mt_reg_006",
  "description": "Conversation description",
  "dataset_context": "dataset info",
  "turns": [
    {"turn": 1, "user_prompt": "...", "expected_tool": "...", "context_from_previous": false},
    {"turn": 2, "user_prompt": "...", "expected_tool": "...", "context_from_previous": true}
  ]
}
```

### Naturalistic Prompts

Add to `test_cases/naturalistic/{category}_natural.json`:
```json
{
  "id": "nat_reg_014",
  "base_test_id": "reg_001",
  "prompt_type": "informal|verbose|typos|domain_jargon|ambiguous",
  "prompt": "Naturalistic version of prompt",
  "expected_tools": ["regression_ols"],
  "linguistic_features": ["informal_register", "abbreviations"]
}
```

## R Scripts for Paper

Generate figures and tables for the paper:

```bash
Rscript paper/code/fig_extended_eval_comparison.R  # Multi-panel comparison
Rscript paper/code/tab_extended_eval_summary.R     # Summary table
Rscript paper/code/fig_robustness_radar.R          # Radar chart
```

## Troubleshooting

### API Errors
- Verify API keys are set correctly
- Check rate limits (0.5s delay between requests)
- For Ollama: ensure `ollama serve` is running

### Tool Extraction for Non-Tool-Calling Models
Some models don't support native tool calling. Scripts attempt regex extraction from response content.

### Missing Results
- Check `results/` subdirectories for each evaluation type
- Ensure test files exist in `test_cases/`

## License

Part of the p2a-paper project.
