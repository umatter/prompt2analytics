//! LLM tool definitions auto-generated from the MCP router.
//!
//! Tool definitions for LLM function calling are derived from the router's
//! `JsonSchema` derives on request structs, ensuring a single source of truth.
//! Tools are organized into tiers for priority-based filtering.

use super::ToolDefinition;

/// Tier 1: Core analytical tools always exposed to LLM (~119 tools).
/// Stays under OpenAI's 128-tool limit while covering all common workflows.
pub const TIER1_TOOLS: &[&str] = &[
    // Data management (6)
    "load_dataset",
    "create_dataset",
    "list_datasets",
    "describe_dataset",
    "head_dataset",
    "export_dataset",
    // Regression (10)
    "regression_ols",
    "regression_clustered",
    "regression_diagnostics",
    "regression_hac",
    "regression_quantreg",
    "regression_bootstrap_cov",
    "regression_driscoll_kraay",
    "regression_bgtest",
    "regression_loess",
    "regression_step",
    // Panel (10)
    "panel_fixed_effects",
    "panel_random_effects",
    "panel_hdfe",
    "panel_gmm",
    "panel_gls",
    "panel_pvcm",
    "panel_pmg",
    "panel_unit_root",
    "hausman_test",
    "feglm",
    // Causal inference (18)
    "iv_2sls",
    "iv_first_stage",
    "iv_sargan_test",
    "diff_in_diff",
    "staggered_did",
    "etwfe",
    "bacon_decomp",
    "rd_estimate",
    "rd_bw",
    "rd_fuzzy",
    "synthetic_control",
    "scpi",
    "gsynth",
    "treatment_ipw",
    "treatment_doubly_robust",
    "propensity_matching",
    "treatment_cbps",
    "treatment_weightit",
    // Hypothesis testing (15)
    "hypothesis_t_test",
    "hypothesis_wilcoxon",
    "hypothesis_kruskal_wallis",
    "hypothesis_chisq_independence",
    "hypothesis_chisq_gof",
    "hypothesis_fisher_exact",
    "hypothesis_shapiro_wilk",
    "hypothesis_ks_test",
    "hypothesis_bartlett_test",
    "hypothesis_cor_test",
    "hypothesis_friedman",
    "anova_one_way",
    "anova_two_way",
    "anova_tukey_hsd",
    "anova_manova",
    // Discrete choice (5)
    "logit",
    "probit",
    "ordered_model",
    "negbin",
    "mlogit",
    // Time series (12)
    "ts_arima_fit",
    "ts_arima_forecast",
    "ts_var",
    "ts_vecm",
    "ts_var_irf",
    "ts_mstl",
    "ts_holt_winters",
    "ts_garch_fit",
    "ts_changepoint",
    "timeseries_pp_test",
    "timeseries_decompose",
    "timeseries_acf",
    // ML (7)
    "ml_kmeans",
    "ml_pca",
    "ml_random_forest",
    "ml_dbscan",
    "ml_hierarchical",
    "ml_tsne",
    "ml_svm",
    // Data munging (16)
    "munge_filter",
    "munge_rename",
    "munge_mutate",
    "munge_cast",
    "munge_sort",
    "munge_group_by",
    "munge_join",
    "munge_pivot",
    "munge_melt",
    "munge_select",
    "munge_drop_columns",
    "munge_drop_na",
    "munge_fill_na",
    "munge_deduplicate",
    "munge_concat",
    "str_replace_value",
    // Visualization (10)
    "viz_histogram",
    "viz_scatter",
    "viz_line",
    "viz_boxplot",
    "viz_heatmap",
    "viz_coefficient",
    "viz_residual_diagnostics",
    "viz_event_study",
    "viz_irf",
    "viz_dendrogram",
    // Statistics & other (4)
    "compute_correlation",
    "marginal_effects",
    "stats_model_tables",
    "generate_random_data",
    // Database (6)
    "db_sqlite_query",
    "db_sqlite_tables",
    "db_sqlite_schema",
    "db_duckdb_query",
    "db_duckdb_tables",
    "db_duckdb_schema",
];

/// Internal tools never exposed to LLM.
pub const INTERNAL_TOOLS: &[&str] = &[
    "server_stats",
    "set_seed",
    "get_seed",
    "import_session",
    "export_session",
    "batch_process",
    "search_tools",
    "list_tool_categories",
    "tool_info",
    "suggest_cleaning",
    "preview_cleaning",
    "verify_cleaning",
    "cleaning_session_start",
    "cleaning_session_status",
    "cleaning_session_apply",
    "cleaning_session_checkpoints",
    "cleaning_rollback",
    "list_cleaning_sessions",
    "data_quality_profile",
    "compare_datasets",
    "upload_dataset",
    "generate_report",
];

/// Returns the base system prompt for the data analytics assistant.
fn get_base_system_prompt() -> &'static str {
    r#"You are a data analytics assistant for prompt2analytics. You help users analyze data by invoking specialized Rust-powered tools.

## CRITICAL RULES

1. **ALWAYS use tools for any statistical computation or analysis.** You have access to high-performance Rust implementations - USE THEM.

2. **NEVER perform calculations yourself.** Do not:
   - Calculate means, standard deviations, or any statistics manually
   - Estimate regression coefficients or p-values in your head
   - Make up or approximate numerical results
   - Describe what an analysis "would show" without running it

3. **When in doubt, use a tool.** If a user asks anything that could be answered by a tool, call that tool.

4. **PROBLEM DESCRIPTIONS ARE ACTION REQUESTS.** When a user describes a problem or goal, DO the analysis - don't just explain what to do:
   - "I'm concerned about endogeneity" → Call `iv_2sls` (don't just explain IV)
   - "I want to evaluate a policy" → Call `diff_in_diff` or `staggered_did`
   - "I'm worried about heteroskedasticity" → Call `regression_ols` with robust SE
   - "There might be serial correlation" → Call `regression_bgtest`
   The user wants RESULTS, not explanations of methodology.

5. **ALWAYS USE EXISTING DATASETS.** Before creating a new dataset:
   - Check the "Currently Loaded Datasets" section below (if present) to see what data is already available
   - If a dataset with the data you need already exists, USE IT - do NOT call `create_dataset` again
   - Only call `create_dataset` if the user explicitly asks to create NEW data or no suitable dataset exists
   - When referencing a dataset in a tool call, use the EXACT name shown in the loaded datasets list

6. **SKIP EXPLORATION — CALL THE ANALYTICAL TOOL DIRECTLY.**
   - Do NOT call `head_dataset` or `describe_dataset` before cleaning, munging, or analysis — the dataset context below already provides column names, types, and sample values
   - Do NOT call `timeseries_acf` before `ts_arima_fit` — if the user doesn't specify ARIMA order, use reasonable defaults like ARIMA(1,0,1)
   - Do NOT call `iv_first_stage` before `iv_2sls` — 2SLS already includes first-stage diagnostics
   - Do NOT call `munge_filter` to split groups before a statistical test — pass column names directly to the test tool
   - When the user says "clean", "replace", or "fix values", call `str_replace_value` or `munge_filter` immediately — do NOT explore first

7. **TOOL CALL BUDGET.** Aim for at most 10 tool calls per request. Plan efficiently:
   - Single-step analyses: 1 tool call
   - Multi-step tasks (e.g., "clean then analyze"): 2–3 tool calls
   - Do NOT call the same tool with the same arguments twice
   - If a tool errors, try a different approach rather than retrying

8. **Be aware of conversation context.** The user may be continuing a previous analysis:
   - Refer back to datasets, analyses, or results from earlier in the conversation
   - Don't repeat tool calls unnecessarily if the result is already available
   - Build on previous work rather than starting over

## WORKFLOW

1. **Check existing datasets first** → Look at the "Currently Loaded Datasets" section to see what's available
2. **No dataset?** → Help user load one with `load_dataset` OR create sample data with `create_dataset`
3. **Dataset exists?** → USE IT directly - do NOT recreate it
4. **User asks for analysis?** → Find the matching tool and call it with the existing dataset
5. **Got results?** → Explain them in plain language, suggest follow-up analyses
6. **User wants visualization?** → Use the appropriate viz_* tool

## EXAMPLES OF CORRECT BEHAVIOR

User: "What's the average income in my dataset?"
✓ CORRECT: Call `describe_dataset` to get statistics
✗ WRONG: Try to calculate or estimate the average yourself

User: "Run a regression of price on sqft and bedrooms"
✓ CORRECT: Call `regression_ols` with y="price", x=["sqft", "bedrooms"]
✗ WRONG: Describe what regression would do without calling the tool

User: "Generate some test data for regression"
✓ CORRECT: Call `create_dataset` with actual CSV content
✗ WRONG: Just describe what data would look like

User: "Now run OLS on that data" (after data was already created)
✓ CORRECT: Call `regression_ols` using the EXISTING dataset name
✗ WRONG: Call `create_dataset` again to recreate the same data

User: "Is there heteroskedasticity in my model?"
✓ CORRECT: Call `regression_diagnostics` which includes Breusch-Pagan test
✗ WRONG: Speculate about heteroskedasticity without testing

### CRITICAL: Problem descriptions = ACTION requests

User: "I'm concerned that education is endogenous"
✓ CORRECT: Call `iv_2sls` with appropriate instruments
✗ WRONG: Explain what endogeneity is and suggest IV

User: "I want to evaluate a policy implemented in 2015"
✓ CORRECT: Call `diff_in_diff` or `staggered_did`
✗ WRONG: Explain DiD methodology without running it

User: "Students above a threshold get a scholarship - evaluate its effect"
✓ CORRECT: Call `rd_estimate` (this is a regression discontinuity design)
✗ WRONG: Describe RD methodology

User: "Evaluate California's policy using other states as controls"
✓ CORRECT: Call `synthetic_control` (this is synthetic control method)
✗ WRONG: Explain synthetic control without running it

## TECHNICAL VOCABULARY MAPPING

When users mention these terms, map to the correct tool:
- "Johansen procedure/test" → `ts_vecm` (cointegration)
- "Cointegration" → `ts_vecm`
- "Unit root test" → `timeseries_pp_test` or `timeseries_acf`
- "Granger causality" → `ts_var`
- "Robust standard errors" / "HC0-HC3" → `regression_ols` (NOT `regression_clustered`)
- "Clustered standard errors" → `regression_clustered`
- "Driscoll-Kraay" → `regression_driscoll_kraay`
- "HDFE" / "high-dimensional fixed effects" → `panel_hdfe`
- "Arellano-Bond" / "dynamic panel" → `panel_gmm`
- "Event study" → `staggered_did` or `etwfe`
- "Parallel trends" → `staggered_did` or `viz_event_study`
- "SCPI" / "prediction intervals for synth" → `scpi`

## Vocabulary → Tool Mapping
- "Compare group means" / "ANOVA" → anova_one_way
- "Matching" / "matched sample" → propensity_matching
- "IPW" / "inverse probability weighting" → treatment_ipw
- "Multiple fixed effects" / "two-way FE" → panel_hdfe
- "Staggered treatment" / "event study" → staggered_did
- "Cast" / "convert type" / "change column type" → munge_cast

## FEW-SHOT EXAMPLES OF CORRECT TOOL CALLS

These examples show the exact format for common multi-turn scenarios:

### Example 1: Loading data and running regression

**Turn 1 - User**: "Load the file sales.csv"
**Turn 1 - Tool Call**:
```json
{"name": "load_dataset", "arguments": {"path": "sales.csv"}}
```
**Result**: Dataset loaded as "sales" with columns: price, sqft, bedrooms, location

**Turn 2 - User**: "Run a regression of price on the other variables"
**Turn 2 - Tool Call**:
```json
{"name": "regression_ols", "arguments": {"dataset": "sales", "y": "price", "x": ["sqft", "bedrooms", "location"]}}
```

### Example 2: Follow-up analysis referencing previous results

**Turn 1**: User loads "housing" dataset
**Turn 2**: User runs OLS regression on "housing"
**Turn 3 - User**: "Check for heteroskedasticity in that model"
**Turn 3 - Tool Call**:
```json
{"name": "regression_diagnostics", "arguments": {"dataset": "housing", "y": "price", "x": ["sqft", "bedrooms"]}}
```
Note: Use the SAME dataset and variables from the previous regression.

### Example 3: Referencing "those results" or "that data"

**Previous context**: User ran `describe_dataset` on "survey_data"
**User**: "Now show me a histogram of the income column from that data"
**Tool Call**:
```json
{"name": "viz_histogram", "arguments": {"dataset": "survey_data", "column": "income"}}
```
Note: "that data" refers to "survey_data" from the previous tool call.

### Example 4: Multi-step analysis workflow

**User**: "I want to analyze my panel data. First describe it, then run fixed effects."
**Tool Calls** (in sequence):
1. `{"name": "describe_dataset", "arguments": {"dataset": "panel_data"}}`
2. After seeing the columns, call: `{"name": "panel_fixed_effects", "arguments": {"dataset": "panel_data", "y": "y", "x": ["x1", "x2"], "entity_var": "firm_id"}}`

## CRITICAL REMINDERS

1. **Dataset names must match exactly** - Use the name shown in "Currently Loaded Datasets"
2. **Column names are case-sensitive** - Use the exact column names from the dataset
3. **Reference previous context** - When user says "those results" or "that dataset", look at previous tool calls
4. **Don't recreate data** - If a dataset exists, USE IT - don't call create_dataset again

Remember: Your value is in orchestrating these powerful Rust tools, not in doing mental math. The tools are fast, accurate, and provide publication-quality output. USE THEM. Always check what datasets are already loaded before creating new ones."#
}

/// Categorize a tool by its name prefix and return the category label.
fn tool_category(name: &str) -> &'static str {
    if name.starts_with("load_") || name.starts_with("create_") || name.starts_with("list_datasets")
        || name.starts_with("describe_") || name.starts_with("head_") || name.starts_with("export_")
        || name.starts_with("upload_")
    {
        "Data Management"
    } else if name.starts_with("regression_") {
        "Regression"
    } else if name.starts_with("panel_") || name == "hausman_test" || name == "feglm" {
        "Panel Data"
    } else if name.starts_with("iv_") || name.starts_with("diff_") || name.starts_with("staggered_")
        || name == "etwfe" || name.starts_with("bacon_") || name.starts_with("rd_")
        || name.starts_with("synthetic_") || name == "scpi" || name == "gsynth"
        || name.starts_with("treatment_") || name.starts_with("propensity_")
        || name == "marginal_effects"
    {
        "Causal Inference"
    } else if name.starts_with("hypothesis_") || name.starts_with("anova_") {
        "Hypothesis Testing"
    } else if name == "logit" || name == "probit" || name.starts_with("ordered_")
        || name == "negbin" || name == "mlogit" || name == "poisson"
        || name.starts_with("zeroinfl") || name.starts_with("hurdle")
    {
        "Discrete Choice"
    } else if name.starts_with("ts_") || name.starts_with("timeseries_") {
        "Time Series"
    } else if name.starts_with("ml_") {
        "Machine Learning"
    } else if name.starts_with("munge_") || name.starts_with("str_") {
        "Data Munging"
    } else if name.starts_with("viz_") {
        "Visualization"
    } else if name.starts_with("db_") {
        "Database"
    } else if name.starts_with("survival_") || name.starts_with("km_") || name.starts_with("cox_") {
        "Survival Analysis"
    } else if name.starts_with("spatial_") || name.starts_with("moran") {
        "Spatial Econometrics"
    } else {
        "Other"
    }
}

/// Generate a categorized tool listing for the system prompt from actual tool definitions.
fn generate_tool_listing(tools: &[ToolDefinition]) -> String {
    use std::collections::BTreeMap;

    let mut categories: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();

    for tool in tools {
        let cat = tool_category(&tool.name);
        // Take the first sentence of the description
        let desc = tool.description.split(". ").next().unwrap_or(&tool.description);
        categories.entry(cat).or_default().push((&tool.name, desc));
    }

    // Define display order
    let order = [
        "Data Management", "Regression", "Panel Data", "Causal Inference",
        "Hypothesis Testing", "Discrete Choice", "Time Series", "Machine Learning",
        "Data Munging", "Visualization", "Database", "Survival Analysis",
        "Spatial Econometrics", "Other",
    ];

    let mut out = String::from("## AVAILABLE TOOLS BY CATEGORY\n\n");

    for &cat_name in &order {
        if let Some(tools_in_cat) = categories.get(cat_name) {
            out.push_str(&format!("### {}\n", cat_name));
            for (name, desc) in tools_in_cat {
                out.push_str(&format!("- `{}` - {}\n", name, desc));
            }
            out.push('\n');
        }
    }

    // Any remaining categories not in the order
    for (cat_name, tools_in_cat) in &categories {
        if !order.contains(cat_name) {
            out.push_str(&format!("### {}\n", cat_name));
            for (name, desc) in tools_in_cat {
                out.push_str(&format!("- `{}` - {}\n", name, desc));
            }
            out.push('\n');
        }
    }

    out
}

/// Returns the system prompt for the data analytics assistant.
pub fn get_system_prompt() -> String {
    get_base_system_prompt().to_string()
}

/// Returns the system prompt with dataset context and dynamically generated tool listing.
pub fn get_system_prompt_with_context(
    dataset_context: Option<&str>,
    tools: &[ToolDefinition],
) -> String {
    let base = get_base_system_prompt();
    let tool_listing = generate_tool_listing(tools);

    let mut prompt = format!("{}\n\n{}", base, tool_listing);

    if let Some(context) = dataset_context {
        if !context.is_empty() {
            prompt.push_str(&format!("\n## Currently Loaded Datasets\n\n{}", context));
        }
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_not_empty() {
        let prompt = get_system_prompt();
        assert!(!prompt.is_empty());
        assert!(prompt.contains("data analytics"));
    }

    #[test]
    fn test_tool_listing_generation() {
        let tools = vec![
            ToolDefinition {
                name: "regression_ols".to_string(),
                description: "Run OLS regression. Returns coefficients.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "munge_filter".to_string(),
                description: "Filter rows in a dataset.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let listing = generate_tool_listing(&tools);
        assert!(listing.contains("### Regression"));
        assert!(listing.contains("`regression_ols`"));
        assert!(listing.contains("### Data Munging"));
        assert!(listing.contains("`munge_filter`"));
    }

    #[test]
    fn test_system_prompt_with_context_includes_tools() {
        let tools = vec![
            ToolDefinition {
                name: "regression_ols".to_string(),
                description: "Run OLS regression.".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ];
        let prompt = get_system_prompt_with_context(Some("dataset: test"), &tools);
        assert!(prompt.contains("AVAILABLE TOOLS BY CATEGORY"));
        assert!(prompt.contains("`regression_ols`"));
        assert!(prompt.contains("Currently Loaded Datasets"));
        assert!(prompt.contains("dataset: test"));
    }

    #[test]
    fn test_tier1_tools_under_128() {
        assert!(
            TIER1_TOOLS.len() <= 128,
            "Tier 1 has {} tools, exceeds OpenAI limit of 128",
            TIER1_TOOLS.len()
        );
    }

    #[test]
    fn test_no_overlap_between_tiers() {
        for tool in INTERNAL_TOOLS {
            assert!(
                !TIER1_TOOLS.contains(tool),
                "Tool {} is in both TIER1 and INTERNAL",
                tool
            );
        }
    }
}
