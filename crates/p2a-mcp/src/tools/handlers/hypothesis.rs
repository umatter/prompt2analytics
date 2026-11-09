//! Hypothesis testing tool handlers.
//!
//! This module implements MCP tools for statistical hypothesis tests:
//! - T-tests (one-sample, two-sample, paired)
//! - Wilcoxon rank tests
//! - Chi-squared tests (goodness-of-fit, independence)
//! - Fisher's exact test
//! - Kolmogorov-Smirnov tests
//! - Kruskal-Wallis test
//! - Friedman test
//! - Shapiro-Wilk normality test
//! - And more

use crate::server::AnalyticsServer;
use crate::tools::common::{
    MissingSummary, NaPolicy, budget_group_count, budget_level_count, check_named_budget,
    insert_missing, missing_json, numeric_columns, parse_option,
};
use crate::tools::limits::{Family, echo};
use crate::tools::requests::hypothesis::*;
use p2a_core::stats::{
    Alternative, CmhAlternative, GofExpected, PValueAdjustMethod, PoissonAlternative, mcnemar_test,
    poisson_test, run_bartlett_test, run_chisq_gof_with, run_chisq_independence, run_friedman_test,
    run_kruskal_test, run_mantelhaen_test, run_mood_test, run_oneway_test, run_pairwise_t_test,
    run_pairwise_wilcox_test, run_quade_test, t_test,
};
use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use crate::tools::handlers::stats::parse_choice;
use p2a_core::options::Choices;
use p2a_core::stats::FisherAlternative;

/// `hypothesis_fisher_test` alternative (R's fisher.test `alternative`).
static FISHER_ALTERNATIVE: Choices<FisherAlternative> = Choices {
    option: "alternative",
    values: &[
        (&["two.sided", "twosided"], FisherAlternative::TwoSided),
        (&["greater", "right", "gt"], FisherAlternative::Greater),
        (&["less", "left", "lt"], FisherAlternative::Less),
    ],
};

/// `hypothesis_mantelhaen_test` alternative (R's mantelhaen.test `alternative`, 2 x 2 x K).
static CMH_ALTERNATIVE: Choices<CmhAlternative> = Choices {
    option: "alternative",
    values: &[
        (&["two.sided", "twosided"], CmhAlternative::TwoSided),
        (&["greater", "right", "gt"], CmhAlternative::Greater),
        (&["less", "left", "lt"], CmhAlternative::Less),
    ],
};

#[tool_router(router = hypothesis_router, vis = "pub")]
impl AnalyticsServer {
    /// Run Bartlett's test for homogeneity of variances.
    #[tool(
        description = "Run Bartlett's test for homogeneity of variances across groups. Tests H₀: all group variances are equal. Suitable for checking the equal variance assumption before ANOVA. Note: Sensitive to non-normality; use Levene's test if data may be non-normal. Returns: K-squared statistic, df, p-value, and group-wise variance estimates."
    )]
    pub async fn hypothesis_bartlett_test(
        &self,
        Parameters(request): Parameters<BartlettTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match run_bartlett_test(dataset, &request.response, &request.factor) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bartlett test failed: {}",
                    e
                ))]));
            }
        };

        // Build group statistics array
        let group_stats: Vec<serde_json::Value> = result
            .group_stats
            .iter()
            .map(|gs| {
                serde_json::json!({
                    "group": gs.group,
                    "n": gs.n,
                    "variance": gs.variance,
                    "std_dev": gs.std_dev
                })
            })
            .collect();

        let mut output = serde_json::json!({
            "method": "Bartlett's Test for Homogeneity of Variances",
            "statistic": result.statistic,
            "statistic_name": "K-squared",
            "df": result.df,
            "p_value": result.p_value,
            "significance": result.significance.to_string(),
            "n_groups": result.n_groups,
            "n_obs": result.n_obs,
            "pooled_variance": result.pooled_variance,
            "group_statistics": group_stats,
            "hypothesis": {
                "null": "All group variances are equal",
                "alternative": "At least two group variances differ"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Variances are significantly different across groups."
            } else {
                "Fail to reject H₀: No significant difference in variances across groups."
            },
            "note": "Bartlett's test is sensitive to non-normality. If data are non-normal, consider using Levene's test instead."
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run chi-squared goodness-of-fit test.
    #[tool(
        description = "Run Pearson's chi-squared goodness-of-fit test (R chisq.test(table(x), p = ...)) to check if observed category frequencies match expected probabilities. Tests H₀: observed frequencies follow the expected distribution. Each distinct non-missing value of the column is a category; rows with a missing value are dropped and reported under missing. Give the expected probabilities either as probs, one per category in the order of the sorted category labels (numbers numerically, text in C-locale order; the result's categories lists that order), or as expected, a category -> probability map in which categories that never occur in the data count as 0 (e.g. all six faces of a die); omit both for the uniform distribution. Returns chi-squared statistic, p-value, degrees of freedom, the categories with observed and expected counts and Pearson residuals, and R's warning when an expected count is below 5."
    )]
    pub async fn hypothesis_chisq_gof(
        &self,
        Parameters(request): Parameters<ChiSquaredGofRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let expected = match (request.probs, request.expected) {
            (Some(_), Some(_)) => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Chi-squared goodness-of-fit test refused: give either probs (in the sorted \
                     category order) or expected (a category -> probability map), not both",
                )]));
            }
            (Some(probs), None) => GofExpected::Probs(probs),
            (None, Some(map)) => {
                // The core copies each key several times, so the map is held to the length
                // budget of an array (probs has an array rule; a map is checked here).
                if let Err(msg) = check_named_budget(
                    "hypothesis_chisq_gof",
                    "array_length",
                    map.len() as u64,
                    crate::tools::limits::current().cap(Family::ArrayLength),
                ) {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Chi-squared goodness-of-fit test refused: expected has {} categories. \
                         {msg}",
                        map.len()
                    ))]));
                }
                GofExpected::Named(map.into_iter().collect())
            }
            (None, None) => GofExpected::Uniform,
        };

        match run_chisq_gof_with(dataset, &request.column, &expected, false) {
            Ok(result) => {
                let categories = result.row_labels.clone().unwrap_or_default();
                let k = result.observed.len();
                let label_bytes = label_budget_bytes(&categories);
                let mut out = serde_json::Map::new();
                out.insert("test".into(), serde_json::json!(result.test_name));
                out.insert("statistic".into(), json_number(result.statistic));
                out.insert("df".into(), serde_json::json!(result.df));
                out.insert("p_value".into(), json_number(result.p_value));
                out.insert(
                    "significance".into(),
                    serde_json::json!(result.significance.to_string()),
                );
                out.insert("n_categories".into(), serde_json::json!(k));
                out.insert("n".into(), json_number(result.observed.iter().sum::<f64>()));
                if fits_response_budget((k as u64).saturating_mul(3), label_bytes) {
                    out.insert("categories".into(), serde_json::json!(categories));
                    out.insert("observed".into(), serde_json::json!(result.observed));
                    out.insert("expected".into(), serde_json::json!(result.expected));
                    out.insert("residuals".into(), serde_json::json!(result.residuals));
                } else {
                    out.insert(
                        "per_category_arrays".into(),
                        serde_json::json!(format!(
                            "omitted: {k} categories are too many to return; the statistic, df \
                             and p-value are complete"
                        )),
                    );
                }
                out.insert("warnings".into(), serde_json::json!(result.warnings));
                if let Some(missing) = missing_json(&result.missing) {
                    out.insert("missing".into(), missing);
                }
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&serde_json::Value::Object(out)).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Chi-squared goodness-of-fit test failed: {}",
                e
            ))])),
        }
    }

    /// Run chi-squared test of independence.
    #[tool(
        description = "Run Pearson's chi-squared test of independence (R chisq.test(table(x, y))) to check if two categorical variables are independent. Creates a contingency table from two columns and tests H₀: row and column variables are independent. Rows with a missing value in either column are dropped (R table()) and reported under missing; the levels are sorted (numbers numerically, text in C-locale order) and returned as row_levels and col_levels. For 2×2 tables, Yates' continuity correction is applied by default. Returns chi-squared statistic, p-value, degrees of freedom, observed and expected counts and residuals in row_levels x col_levels order, and R's warning 'Chi-squared approximation may be incorrect' when an expected count is below 5."
    )]
    pub async fn hypothesis_chisq_independence(
        &self,
        Parameters(request): Parameters<ChiSquaredIndependenceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let correct = request.correct.unwrap_or(true);

        // Size the table before any work, when a budget depends on its size: both counts include
        // null as a level, an upper bound on the table the core builds after dropping missing
        // values. A missing column is left to the core to report; a count that fails is a
        // refusal, since the budget could not be checked.
        let limits = crate::tools::limits::current();
        let mut cells: Option<usize> = None;
        if limits.table_cells.is_some() || limits.response_bytes.is_some() {
            let df = dataset.df();
            let (r, c) = match (
                budget_level_count(df, &request.row_var),
                budget_level_count(df, &request.col_var),
            ) {
                (Ok(r), Ok(c)) => (r, c),
                (Err(msg), _) | (_, Err(msg)) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Chi-squared test of independence refused: {msg}"
                    ))]));
                }
            };
            if let (Some(r), Some(c)) = (r, c) {
                let (row_var, col_var) = (echo(&request.row_var), echo(&request.col_var));
                let Some(n) = r.checked_mul(c) else {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Chi-squared test of independence refused: `{row_var}` has {r} levels \
                         and `{col_var}` has {c}, a table that is too large to count."
                    ))]));
                };
                if let Err(msg) = check_named_budget(
                    "hypothesis_chisq_independence",
                    "table_cells",
                    n as u64,
                    limits.table_cells,
                ) {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Chi-squared test of independence refused: `{row_var}` has {r} levels \
                         and `{col_var}` has {c}, a table of {n} cells. {msg} Use columns with \
                         fewer distinct values."
                    ))]));
                }
                cells = Some(n);
            }
        }

        match run_chisq_independence(dataset, &request.row_var, &request.col_var, correct) {
            Ok(result) => {
                // The four per-cell arrays take about 4 x 32 bytes of JSON per cell, and the
                // level labels their own length; leave them out together when they could not
                // fit the response budget.
                let n_cells = cells.unwrap_or(result.observed.len());
                let row_levels = result.row_labels.clone().unwrap_or_default();
                let col_levels = result.col_labels.clone().unwrap_or_default();
                let label_bytes =
                    label_budget_bytes(&row_levels).saturating_add(label_budget_bytes(&col_levels));
                let fits = fits_response_budget((n_cells as u64).saturating_mul(4), label_bytes);
                let mut out = serde_json::Map::new();
                out.insert("test".into(), serde_json::json!(result.test_name));
                out.insert("statistic".into(), json_number(result.statistic));
                out.insert("df".into(), serde_json::json!(result.df));
                out.insert("p_value".into(), json_number(result.p_value));
                out.insert(
                    "significance".into(),
                    serde_json::json!(result.significance.to_string()),
                );
                out.insert("n_rows".into(), serde_json::json!(result.n_rows));
                out.insert("n_cols".into(), serde_json::json!(result.n_cols));
                if fits {
                    out.insert("row_levels".into(), serde_json::json!(row_levels));
                    out.insert("col_levels".into(), serde_json::json!(col_levels));
                    out.insert("observed".into(), serde_json::json!(result.observed));
                    out.insert("expected".into(), serde_json::json!(result.expected));
                    out.insert("residuals".into(), serde_json::json!(result.residuals));
                    out.insert(
                        "std_residuals".into(),
                        serde_json::json!(result.std_residuals),
                    );
                } else {
                    out.insert(
                        "per_cell_arrays".into(),
                        serde_json::json!(format!(
                            "omitted: a table of {n_cells} cells is too large to return; the \
                             statistic, df and p-value are complete"
                        )),
                    );
                }
                out.insert(
                    "yates_correction".into(),
                    serde_json::json!(result.yates_correction),
                );
                out.insert("warnings".into(), serde_json::json!(result.warnings));
                if let Some(missing) = missing_json(&result.missing) {
                    out.insert("missing".into(), missing);
                }
                let json_output = serde_json::Value::Object(out);
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Chi-squared test of independence failed: {}",
                e
            ))])),
        }
    }

    /// Test for association between paired samples using correlation.
    #[tool(
        description = "Test for association between paired samples using Pearson, Spearman, or Kendall correlation. Returns correlation coefficient, test statistic, p-value, and confidence interval (for Pearson). Pearson measures linear association; Spearman and Kendall measure monotonic association and are robust to outliers."
    )]
    pub async fn hypothesis_cor_test(
        &self,
        Parameters(request): Parameters<CorTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::cortest::run_cor_test_dataset;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let method = request.method.as_deref().unwrap_or("pearson");
        let alternative = request.alternative.as_deref().unwrap_or("two.sided");
        let conf_level = request.conf_level.unwrap_or(0.95);

        match run_cor_test_dataset(
            dataset,
            &request.x,
            &request.y,
            method,
            alternative,
            conf_level,
        ) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.method_name,
                    "estimate": result.estimate,
                    "estimate_name": result.estimate_name,
                    "statistic": result.statistic,
                    "statistic_name": result.statistic_name,
                    "p_value": result.p_value,
                    "alternative": result.alternative.to_string(),
                    "n": result.n,
                    "null_value": result.null_value,
                    "conf_level": result.conf_level,
                    "interpretation": if result.p_value < 0.05 {
                        format!("Significant {} correlation (p < 0.05)",
                            if result.estimate > 0.0 { "positive" } else { "negative" })
                    } else {
                        "No significant correlation (p >= 0.05)".to_string()
                    }
                });

                if let Some(df) = result.df {
                    json_output["df"] = serde_json::json!(df);
                }

                if let (Some(low), Some(high)) = (result.conf_low, result.conf_high) {
                    json_output["confidence_interval"] = serde_json::json!({
                        "lower": low,
                        "upper": high,
                        "conf_level": result.conf_level,
                    });
                }
                insert_missing(&mut json_output, &result.missing);

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Correlation test failed: {}",
                e
            ))])),
        }
    }

    /// Fisher's exact test for 2×2 contingency tables.
    #[tool(
        description = "Run Fisher's exact test for a 2×2 contingency table built from two columns, as R fisher.test. Tests independence between two binary categorical variables using exact probability calculations (hypergeometric distribution). More accurate than chi-squared test for small samples. Rows with a missing value in either column are dropped (R table()); the levels are sorted (numbers numerically, text in C-locale order) and returned as row_levels and col_levels. Returns the p-value, the conditional MLE odds ratio and (with conf_level) the exact conditional CI as R fisher.test (one-sided for 'less'/'greater'), the sample odds ratio and the level labels; infinite values are the string \"Inf\". Use when expected cell counts are small (<5) or when exact p-values are needed."
    )]
    pub async fn hypothesis_fisher_exact(
        &self,
        Parameters(request): Parameters<FisherExactRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::fisher::{FisherAlternative, run_fisher_test};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alternative = match parse_choice(
            &FISHER_ALTERNATIVE,
            request.alternative.as_deref(),
            FisherAlternative::TwoSided,
        ) {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        match run_fisher_test(
            dataset,
            &request.row_var,
            &request.col_var,
            alternative,
            request.conf_level,
        ) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.test_name,
                    "p_value": json_number(result.p_value),
                    "significance": result.significance.to_string(),
                    "alternative": result.alternative.to_string(),
                    "odds_ratio": json_number(result.odds_ratio),
                    "sample_odds_ratio": json_number(result.odds_ratio),
                    "conditional_mle": json_number(result.conditional_mle),
                    "table": {
                        "a": json_number(result.table[0]),
                        "b": json_number(result.table[1]),
                        "c": json_number(result.table[2]),
                        "d": json_number(result.table[3]),
                    },
                    "row_levels": result.row_levels,
                    "col_levels": result.col_levels,
                    "row_totals": result.row_totals.map(json_number),
                    "col_totals": result.col_totals.map(json_number),
                    "n": json_number(result.n),
                    "prob_observed": json_number(result.prob_observed),
                });

                if let (Some((lo, hi)), Some(level)) = (result.odds_ratio_ci, result.conf_level) {
                    json_output["odds_ratio_ci"] = serde_json::json!({
                        "lower": json_number(lo),
                        "upper": json_number(hi),
                        "conf_level": json_number(level),
                        "method": "exact conditional (as R fisher.test)",
                    });
                }
                insert_missing(&mut json_output, &result.missing);

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Fisher's exact test failed: {}",
                e
            ))])),
        }
    }

    /// Run Friedman rank sum test.
    #[tool(
        description = "Run Friedman rank sum test for unreplicated blocked data. Non-parametric alternative to one-way repeated measures ANOVA. Tests whether treatments have equal effects across blocks. Returns Q statistic, degrees of freedom, p-value, and rank statistics per treatment."
    )]
    pub async fn hypothesis_friedman(
        &self,
        Parameters(request): Parameters<FriedmanTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result =
            match run_friedman_test(dataset, &request.value, &request.treatment, &request.block) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Friedman test failed: {}",
                        e
                    ))]));
                }
            };

        let treatment_stats: Vec<serde_json::Value> = result
            .treatment_names
            .iter()
            .zip(result.rank_sums.iter())
            .zip(result.mean_ranks.iter())
            .map(|((name, rank_sum), mean_rank)| {
                serde_json::json!({
                    "treatment": name,
                    "rank_sum": rank_sum,
                    "mean_rank": mean_rank
                })
            })
            .collect();

        let mut output = serde_json::json!({
            "method": "Friedman rank sum test",
            "statistic": result.statistic,
            "statistic_name": "Q (chi-squared approximation)",
            "df": result.df,
            "p_value": result.p_value,
            "n_blocks": result.n_blocks,
            "n_treatments": result.n_treatments,
            "has_ties": result.has_ties,
            "tie_correction": result.tie_correction,
            "treatment_statistics": treatment_stats,
            "hypothesis": {
                "null": "All treatments have the same effect",
                "alternative": "At least one treatment has a different effect"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant difference in treatment effects."
            } else {
                "Fail to reject H₀: No significant difference in treatment effects."
            },
            "references": {
                "method": "Friedman (1937), JASA 32(200):675-701",
                "text": "Hollander & Wolfe (1973), Nonparametric Statistical Methods, pp. 139-146"
            }
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run Kruskal-Wallis rank sum test.
    #[tool(
        description = "Run Kruskal-Wallis rank sum test - the non-parametric alternative to one-way ANOVA. Tests whether samples from two or more groups have the same median. Uses chi-squared approximation with tie correction. Returns H statistic, degrees of freedom, p-value, and rank statistics per group."
    )]
    pub async fn hypothesis_kruskal_wallis(
        &self,
        Parameters(request): Parameters<KruskalWallisRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match run_kruskal_test(dataset, &request.value, &request.group) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Kruskal-Wallis test failed: {}",
                    e
                ))]));
            }
        };

        let group_stats: Vec<serde_json::Value> = result
            .group_names
            .iter()
            .zip(result.group_sizes.iter())
            .zip(result.rank_sums.iter())
            .zip(result.mean_ranks.iter())
            .map(|(((name, size), rank_sum), mean_rank)| {
                serde_json::json!({
                    "group": name,
                    "n": size,
                    "rank_sum": rank_sum,
                    "mean_rank": mean_rank
                })
            })
            .collect();

        let mut output = serde_json::json!({
            "method": "Kruskal-Wallis rank sum test",
            "statistic": result.statistic,
            "statistic_name": "H (chi-squared approximation)",
            "df": result.df,
            "p_value": result.p_value,
            "n_groups": result.n_groups,
            "n_total": result.n_total,
            "has_ties": result.has_ties,
            "tie_correction": result.tie_correction,
            "group_statistics": group_stats,
            "hypothesis": {
                "null": "All groups have the same median",
                "alternative": "At least one group has a different median"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant difference in medians across groups."
            } else {
                "Fail to reject H₀: No significant difference in medians."
            },
            "references": {
                "method": "Kruskal & Wallis (1952), JASA 47(260):583-621",
                "text": "Hollander & Wolfe (1973), Nonparametric Statistical Methods, pp. 115-120"
            }
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Kolmogorov-Smirnov test for comparing distributions.
    #[tool(
        description = "Run the Kolmogorov-Smirnov test. Two-sample test: Tests if two samples come from the same distribution. One-sample test: Tests if a sample comes from a specified theoretical distribution (normal, uniform, exponential). Returns D statistic (maximum absolute difference between CDFs), p-value and, for a one-sample test, the reference distribution tested in R's notation (e.g. punif(min = -1.5, max = 3)). A small p-value suggests the distributions differ."
    )]
    pub async fn hypothesis_ks_test(
        &self,
        Parameters(request): Parameters<KsTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::ks::{KsDistribution, TheoreticalDistribution, ks_test};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // The theoretical distribution of a one-sample test (none for two samples)
        let params = [
            ("mean", request.mean, KsDistribution::Normal),
            ("sd", request.sd, KsDistribution::Normal),
            ("a", request.a, KsDistribution::Uniform),
            ("b", request.b, KsDistribution::Uniform),
            ("rate", request.rate, KsDistribution::Exponential),
        ];
        let distribution = if request.y.is_some() {
            if request.distribution.is_some() || params.iter().any(|(_, v, _)| v.is_some()) {
                return Ok(CallToolResult::error(vec![Content::text(
                    "distribution and its parameters apply only to the one-sample test (omit y)",
                )]));
            }
            None
        } else {
            let kind = match parse_option(request.distribution.as_deref(), KsDistribution::Normal) {
                Ok(k) => k,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };
            if let Some((param, _, owner)) = params
                .iter()
                .find(|(_, v, owner)| v.is_some() && *owner != kind)
            {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "{param} applies only to distribution = '{owner}'; drop it or change distribution"
                ))]));
            }
            match TheoreticalDistribution::from_kind(
                kind,
                request.mean,
                request.sd,
                request.a,
                request.b,
                request.rate,
            ) {
                Ok(d) => Some(d),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "KS test failed: {e}"
                    ))]));
                }
            }
        };
        let distribution_label = distribution.as_ref().map(|d| d.label());

        match ks_test(
            dataset,
            &request.x,
            request.y.as_deref(),
            distribution,
            alternative,
        ) {
            Ok(result) => {
                let alt_description = match alternative {
                    Alternative::TwoSided => "two-sided",
                    Alternative::Greater => "greater (CDF of x not below CDF of y)",
                    Alternative::Less => "less (CDF of x not above CDF of y)",
                };

                let mut json_output = serde_json::json!({
                    "test": result.test_name,
                    "statistic_D": result.statistic,
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "alternative": alt_description,
                    "exact": result.exact,
                    "n": result.n,
                    "reject_null": result.reject_null,
                    "interpretation": if result.reject_null {
                        "Evidence that distributions differ (reject H₀ at α = 0.05)"
                    } else {
                        "No evidence that distributions differ (fail to reject H₀ at α = 0.05)"
                    }
                });

                if let Some(label) = &distribution_label {
                    json_output["distribution"] = serde_json::json!(label);
                }
                if let Some(n2) = result.n_2 {
                    json_output["n_2"] = serde_json::json!(n2);
                }
                insert_missing(&mut json_output, &result.missing);

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Kolmogorov-Smirnov test failed: {}",
                e
            ))])),
        }
    }

    /// Run Cochran-Mantel-Haenszel test for stratified 2×2 tables.
    #[tool(
        description = "Run Cochran-Mantel-Haenszel test for conditional independence in stratified 2×2 tables (R mantelhaen.test(table(x, y, stratum))). Tests whether two binary variables are associated while controlling for a third (stratum) variable. Rows with a missing value in any of the three columns are dropped (R table()) and reported under missing; levels are sorted (numbers numerically, text in C-locale order) and returned as row_levels and col_levels. Returns CMH chi-squared statistic, p-value, common odds ratio estimate with confidence interval, and per-stratum statistics."
    )]
    pub async fn hypothesis_mantelhaen(
        &self,
        Parameters(request): Parameters<MantelhaenTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let correct = request.correct.unwrap_or(true);

        let alternative = match parse_choice(
            &CMH_ALTERNATIVE,
            request.alternative.as_deref(),
            CmhAlternative::TwoSided,
        ) {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let result = match run_mantelhaen_test(
            dataset,
            &request.row_var,
            &request.col_var,
            &request.stratum_var,
            correct,
            alternative,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cochran-Mantel-Haenszel test failed: {}",
                    e
                ))]));
            }
        };

        // The per-stratum table is left out when it would not fit the response budget, as
        // the GOF and independence tools do, rather than losing the whole result.
        let labels: Vec<String> = result
            .stratum_stats
            .iter()
            .map(|s| s.stratum.clone())
            .collect();
        let with_strata = fits_response_budget(
            (result.n_strata as u64).saturating_mul(9),
            label_budget_bytes(&labels),
        );
        let stratum_stats: Vec<serde_json::Value> = result
            .stratum_stats
            .iter()
            .take(if with_strata { usize::MAX } else { 0 })
            .map(|s| {
                serde_json::json!({
                    "stratum": s.stratum,
                    "counts": {
                        "a": json_number(s.counts[0]),
                        "b": json_number(s.counts[1]),
                        "c": json_number(s.counts[2]),
                        "d": json_number(s.counts[3])
                    },
                    "n": json_number(s.n),
                    "expected_a": json_number(s.expected_a),
                    "variance_a": json_number(s.variance_a),
                    "odds_ratio": json_number(s.odds_ratio)
                })
            })
            .collect();

        let alt_desc = match result.alternative {
            CmhAlternative::TwoSided => "true common odds ratio is not equal to 1",
            CmhAlternative::Greater => "true common odds ratio is greater than 1",
            CmhAlternative::Less => "true common odds ratio is less than 1",
        };

        let mut output = serde_json::json!({
            "method": result.test_name,
            "statistic": json_number(result.statistic),
            "statistic_name": "Mantel-Haenszel X-squared",
            "df": result.df,
            "p_value": json_number(result.p_value),
            "n_strata": result.n_strata,
            "total_n": result.total_n,
            "common_odds_ratio": json_number(result.common_odds_ratio),
            "odds_ratio_ci": {
                "lower": json_number(result.odds_ratio_ci.0),
                "upper": json_number(result.odds_ratio_ci.1),
                "conf_level": 0.95
            },
            "continuity_correction": result.continuity_correction,
            "hypothesis": {
                "null": "Row and column variables are conditionally independent given stratum",
                "alternative": alt_desc
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant conditional association between variables."
            } else {
                "Fail to reject H₀: No significant conditional association detected."
            },
            "references": {
                "method": "Cochran (1954), Biometrics 10(4):417-451; Mantel & Haenszel (1959), JNCI 22(4):719-748",
                "variance": "Robins et al. (1986) for odds ratio CI"
            },
            "row_levels": result.row_levels,
            "col_levels": result.col_levels,
        });
        if with_strata {
            output["stratum_statistics"] = serde_json::json!(stratum_stats);
        } else {
            output["per_stratum_statistics"] = serde_json::json!(format!(
                "omitted: {} strata are too many to return; the statistic, p-value and common \
                 odds ratio are complete",
                result.n_strata
            ));
        }
        if let Some(missing) = missing_json(&result.missing) {
            output["missing"] = missing;
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run McNemar's chi-squared test.
    #[tool(
        description = "Run McNemar's chi-squared test for paired nominal data. Tests symmetry in a 2x2 contingency table. Commonly used for comparing two classifiers or before/after studies. Only requires the discordant cells (b and c)."
    )]
    pub async fn hypothesis_mcnemar(
        &self,
        Parameters(request): Parameters<McnemarTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let correct = request.correct.unwrap_or(true);

        let result = match mcnemar_test(request.b, request.c, correct) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "McNemar test failed: {}",
                    e
                ))]));
            }
        };

        let method = if result.continuity_correction {
            "McNemar's Chi-squared test with continuity correction"
        } else {
            "McNemar's Chi-squared test"
        };

        let output = serde_json::json!({
            "method": method,
            "statistic": result.statistic,
            "statistic_name": "Chi-squared",
            "df": result.df,
            "p_value": result.p_value,
            "b": result.b,
            "c": result.c,
            "n_discordant": result.n_discordant,
            "continuity_correction": result.continuity_correction,
            "hypothesis": {
                "null": "P(b) = P(c) (marginal homogeneity)",
                "alternative": "P(b) ≠ P(c)"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant asymmetry in the contingency table."
            } else {
                "Fail to reject H₀: No significant asymmetry detected."
            },
            "references": {
                "method": "McNemar (1947), Psychometrika 12(2):153-157",
                "correction": "Edwards (1948), Psychometrika 13(3):185-187"
            }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run Mood's two-sample test of scale.
    #[tool(
        description = "Run Mood's two-sample test for comparing scale parameters between two independent samples. Tests H₀: scale ratio = 1 (equal scales) using squared rank deviations from the mean rank. Non-parametric alternative to F-test for variance comparison. Handles ties using Mielke (1967) variance correction. Returns Z-statistic, p-value, and sample sizes."
    )]
    pub async fn hypothesis_mood_test(
        &self,
        Parameters(request): Parameters<MoodTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let result = match run_mood_test(dataset, &request.x, &request.y, Some(alternative)) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mood test failed: {}",
                    e
                ))]));
            }
        };

        let alt_description = match result.alternative {
            Alternative::TwoSided => "true ratio of scales is not equal to 1",
            Alternative::Greater => "true ratio of scales is greater than 1 (x more spread than y)",
            Alternative::Less => "true ratio of scales is less than 1 (x less spread than y)",
        };

        let mut output = serde_json::json!({
            "method": "Mood two-sample test of scale",
            "statistic": result.statistic,
            "statistic_name": "T (sum of squared rank deviations)",
            "z_score": result.z_score,
            "p_value": result.p_value,
            "significance": result.significance.to_string(),
            "n_x": result.n_x,
            "n_y": result.n_y,
            "n_total": result.n_total,
            "has_ties": result.has_ties,
            "hypothesis": {
                "null": "Scale ratio = 1 (equal scale parameters)",
                "alternative": alt_description
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Scales are significantly different between the two samples."
            } else {
                "Fail to reject H₀: No significant difference in scales between samples."
            },
            "references": {
                "method": "Conover (1971), Practical Nonparametric Statistics, pp. 234-235",
                "ties": "Mielke (1967), Technometrics 9(2):312-314"
            }
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run Welch's one-way ANOVA test.
    #[tool(
        description = "Run Welch's one-way ANOVA test - compares means of multiple groups without assuming equal variances. This is more robust than standard ANOVA when variances differ. Returns F statistic, numerator and denominator degrees of freedom, p-value, and group statistics."
    )]
    pub async fn hypothesis_oneway(
        &self,
        Parameters(request): Parameters<OnewayTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let var_equal = request.var_equal.unwrap_or(false);

        let result = match run_oneway_test(dataset, &request.value, &request.group, var_equal) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Welch's ANOVA failed: {}",
                    e
                ))]));
            }
        };

        let group_stats: Vec<serde_json::Value> = result
            .group_names
            .iter()
            .zip(result.group_sizes.iter())
            .zip(result.group_means.iter())
            .zip(result.group_variances.iter())
            .map(|(((name, size), mean), variance)| {
                serde_json::json!({
                    "group": name,
                    "n": size,
                    "mean": mean,
                    "variance": variance,
                    "std_dev": variance.sqrt()
                })
            })
            .collect();

        let method = if result.var_equal {
            "One-way analysis of means"
        } else {
            "One-way analysis of means (not assuming equal variances)"
        };

        let mut output = serde_json::json!({
            "method": method,
            "statistic": result.statistic,
            "statistic_name": "F",
            "df_numerator": result.df_num,
            "df_denominator": result.df_denom,
            "p_value": result.p_value,
            "n_groups": result.n_groups,
            "n_total": result.n_total,
            "var_equal": result.var_equal,
            "group_statistics": group_stats,
            "hypothesis": {
                "null": "All groups have the same mean",
                "alternative": "At least one group has a different mean"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant difference in means across groups."
            } else {
                "Fail to reject H₀: No significant difference in means."
            },
            "references": {
                "method": "Welch (1951), Biometrika 38(3/4):330-336"
            }
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run pairwise t-tests between all group levels.
    #[tool(
        description = "Run pairwise t-tests between all group levels with p-value adjustment for multiple comparisons. Post-hoc analysis after ANOVA. Options: pool_sd=true uses pooled variance (Student's), false uses Welch's (default). Adjustment methods: 'holm' (default, FWER), 'bonferroni', 'hochberg', 'hommel', 'BH' (FDR), 'BY', 'none'. Returns matrix of adjusted p-values for all pairs."
    )]
    pub async fn hypothesis_pairwise_t_test(
        &self,
        Parameters(request): Parameters<PairwiseTTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let p_adjust_method =
            match parse_option(request.p_adjust_method.as_deref(), PValueAdjustMethod::Holm) {
                Ok(m) => m,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };

        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let pool_sd = request.pool_sd.unwrap_or(false);

        if let Err(msg) = check_pairwise_groups(
            dataset,
            "hypothesis_pairwise_t_test",
            "t-test",
            &request.factor,
        ) {
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }

        match run_pairwise_t_test(
            dataset,
            &request.response,
            &request.factor,
            pool_sd,
            alternative,
            p_adjust_method,
        ) {
            Ok(result) => {
                let p_matrix = match lower_triangular_p_matrix(result.group_names.len(), |i, j| {
                    result.p_values_adj[result.index(i, j)]
                }) {
                    Ok(m) => m,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Pairwise t-test failed: {e}"
                        ))]));
                    }
                };

                let mut output = serde_json::json!({
                    "test": result.test_name,
                    "p_adjust_method": result.p_adjust_method.to_string(),
                    "pool_sd": result.pool_sd,
                    "alternative": result.alternative.to_string(),
                    "group_names": result.group_names,
                    "group_sizes": result.group_sizes,
                    "group_means": result.group_means,
                    "n_comparisons": result.n_comparisons,
                    "pooled_sd": result.pooled_sd,
                    "p_values_adjusted": p_matrix,
                    "p_values_raw": result.p_values_raw,
                    "t_statistics": result.t_statistics,
                    "degrees_of_freedom": result.df,
                    "interpretation": {
                        "how_to_read": "p_values_adjusted is a lower-triangular matrix. Row i, column j gives the adjusted p-value for comparing group_names[i] vs group_names[j].",
                        "significance": "p < 0.05 indicates significant difference between groups at 5% level."
                    },
                    "references": {
                        "method": "R stats::pairwise.t.test",
                        "p_adjust": format!("{} adjustment for multiple comparisons", result.p_adjust_method.name())
                    }
                });
                insert_missing(&mut output, &result.missing);

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&output).unwrap_or_else(|_| result.to_string()),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Pairwise t-test failed: {}",
                e
            ))])),
        }
    }

    /// Run pairwise Wilcoxon rank sum tests between all group levels.
    #[tool(
        description = "Run pairwise Wilcoxon rank sum tests between all group levels with p-value adjustment. Non-parametric post-hoc analysis after Kruskal-Wallis test. Does not assume normality. Adjustment methods: 'holm' (default, FWER), 'bonferroni', 'hochberg', 'hommel', 'BH' (FDR), 'BY', 'none'. Returns matrix of adjusted p-values for all pairs."
    )]
    pub async fn hypothesis_pairwise_wilcox(
        &self,
        Parameters(request): Parameters<PairwiseWilcoxRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let p_adjust_method =
            match parse_option(request.p_adjust_method.as_deref(), PValueAdjustMethod::Holm) {
                Ok(m) => m,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };

        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        if let Err(msg) = check_pairwise_groups(
            dataset,
            "hypothesis_pairwise_wilcox",
            "Wilcoxon test",
            &request.factor,
        ) {
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }

        match run_pairwise_wilcox_test(
            dataset,
            &request.response,
            &request.factor,
            alternative,
            p_adjust_method,
            request.exact,
        ) {
            Ok(result) => {
                let p_matrix = match lower_triangular_p_matrix(result.group_names.len(), |i, j| {
                    result.p_values_adj[result.index(i, j)]
                }) {
                    Ok(m) => m,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Pairwise Wilcoxon test failed: {e}"
                        ))]));
                    }
                };

                let mut output = serde_json::json!({
                    "test": result.test_name,
                    "p_adjust_method": result.p_adjust_method.to_string(),
                    "alternative": result.alternative.to_string(),
                    "group_names": result.group_names,
                    "group_sizes": result.group_sizes,
                    "group_medians": result.group_medians,
                    "n_comparisons": result.n_comparisons,
                    "exact": result.exact,
                    "p_values_adjusted": p_matrix,
                    "p_values_raw": result.p_values_raw,
                    "w_statistics": result.w_statistics,
                    "warning": result.warning,
                    "interpretation": {
                        "how_to_read": "p_values_adjusted is a lower-triangular matrix. Row i, column j gives the adjusted p-value for comparing group_names[i] vs group_names[j].",
                        "significance": "p < 0.05 indicates significant difference in location between groups at 5% level."
                    },
                    "references": {
                        "method": "R stats::pairwise.wilcox.test",
                        "p_adjust": format!("{} adjustment for multiple comparisons", result.p_adjust_method.name())
                    }
                });
                insert_missing(&mut output, &result.missing);

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&output).unwrap_or_else(|_| result.to_string()),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Pairwise Wilcoxon test failed: {}",
                e
            ))])),
        }
    }

    /// Run exact Poisson test.
    #[tool(
        description = "Run exact Poisson test for rate parameters. One-sample test: tests whether the rate equals a hypothesized value. Two-sample test: compares the ratio of two rates. Returns test statistic, p-value, rate estimate (or ratio), and confidence interval."
    )]
    pub async fn hypothesis_poisson(
        &self,
        Parameters(request): Parameters<PoissonTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let r = request.r.unwrap_or(1.0);
        let conf_level = request.conf_level.unwrap_or(0.95);

        let alternative =
            match parse_option(request.alternative.as_deref(), PoissonAlternative::TwoSided) {
                Ok(a) => a,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };

        let result = match poisson_test(&request.x, &request.t, r, alternative, conf_level) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Poisson test failed: {}",
                    e
                ))]));
            }
        };

        let alt_desc = match result.alternative {
            PoissonAlternative::TwoSided => {
                if result.n_samples == 1 {
                    format!("true event rate is not equal to {:.4}", result.null_value)
                } else {
                    format!("true rate ratio is not equal to {:.4}", result.null_value)
                }
            }
            PoissonAlternative::Greater => {
                if result.n_samples == 1 {
                    format!("true event rate is greater than {:.4}", result.null_value)
                } else {
                    format!("true rate ratio is greater than {:.4}", result.null_value)
                }
            }
            PoissonAlternative::Less => {
                if result.n_samples == 1 {
                    format!("true event rate is less than {:.4}", result.null_value)
                } else {
                    format!("true rate ratio is less than {:.4}", result.null_value)
                }
            }
        };

        let estimate_name = if result.n_samples == 1 {
            "event rate"
        } else {
            "rate ratio"
        };

        let output = serde_json::json!({
            "method": result.method,
            "statistic": result.statistic,
            "statistic_name": if result.n_samples == 1 { "number of events" } else { "count1" },
            "expected": result.parameter,
            "p_value": result.p_value,
            "estimate": {
                "name": estimate_name,
                "value": result.estimate
            },
            "confidence_interval": {
                "lower": result.conf_int.0,
                "upper": result.conf_int.1,
                "conf_level": result.conf_level
            },
            "n_samples": result.n_samples,
            "hypothesis": {
                "null": if result.n_samples == 1 {
                    format!("event rate = {:.4}", result.null_value)
                } else {
                    format!("rate ratio = {:.4}", result.null_value)
                },
                "alternative": alt_desc
            },
            "conclusion": if result.p_value < 0.05 {
                if result.n_samples == 1 {
                    "Reject H₀: Event rate significantly differs from hypothesized value."
                } else {
                    "Reject H₀: Rate ratio significantly differs from hypothesized value."
                }
            } else if result.n_samples == 1 {
                "Fail to reject H₀: No significant difference from hypothesized rate."
            } else {
                "Fail to reject H₀: No significant difference in rates."
            },
            "references": "Przyborowski & Wilenski (1940), Biometrika 31(3/4):313-323"
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Cochran-Armitage test for trend in proportions.
    #[tool(
        description = "Test for trend in proportions across ordered groups (Cochran-Armitage test). Tests whether proportions increase or decrease linearly with group scores. Commonly used in dose-response studies. Returns chi-squared statistic with 1 df and p-value. Successes and trials are whole non-negative counts, one row per group; a group with a missing count is dropped and reported under missing, and the other groups keep their scores."
    )]
    pub async fn hypothesis_prop_trend_test(
        &self,
        Parameters(request): Parameters<PropTrendTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::proptrendtest::run_prop_trend_test;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // A group (row) with a missing count is dropped from both columns
        let names = [request.successes.as_str(), request.trials.as_str()];
        let cols = match numeric_columns(dataset, &names, NaPolicy::Omit) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Trend test failed: {e}"
                ))]));
            }
        };
        let mut counts = Vec::with_capacity(names.len());
        for name in names {
            match whole_counts(name, cols.numeric(name), cols.rows()) {
                Ok(c) => counts.push(c),
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            }
        }

        // Each kept group keeps its own score: the user's score of its dataset row, or by
        // default its row position (1, 2, 3, ...) when a group was dropped
        let n_total = cols.missing().n_total;
        let scores = match &request.scores {
            Some(s) if s.len() != n_total => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Trend test failed: scores must have one value per dataset row ({n_total}); \
                     got {}",
                    s.len()
                ))]));
            }
            Some(s) => Some(cols.rows().iter().map(|&r| s[r]).collect()),
            None if cols.missing().is_empty() => None,
            None => Some(cols.rows().iter().map(|&r| (r + 1) as f64).collect()),
        };

        match run_prop_trend_test(&counts[0], &counts[1], scores) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.method,
                    "statistic": result.statistic,
                    "df": result.df,
                    "p_value": result.p_value,
                    "scores": result.scores,
                    "n_groups": result.n_groups,
                    "interpretation": if result.p_value < 0.05 {
                        "Significant trend in proportions (p < 0.05)"
                    } else {
                        "No significant trend in proportions (p >= 0.05)"
                    }
                });
                insert_missing(&mut json_output, cols.missing());

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Trend test failed: {}",
                e
            ))])),
        }
    }

    /// Run Quade test for unreplicated blocked data.
    #[tool(
        description = "Run Quade test for unreplicated blocked data. Similar to Friedman test but uses weighted rankings based on block ranges, making it more powerful when block effects vary. Returns F statistic, degrees of freedom, p-value, and treatment statistics. Requires complete blocks (one observation per treatment per block)."
    )]
    pub async fn hypothesis_quade(
        &self,
        Parameters(request): Parameters<QuadeTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result =
            match run_quade_test(dataset, &request.value, &request.treatment, &request.block) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Quade test failed: {}",
                        e
                    ))]));
                }
            };

        let treatment_stats: Vec<serde_json::Value> = result
            .treatment_names
            .iter()
            .zip(result.treatment_sums.iter())
            .zip(result.mean_weighted_ranks.iter())
            .map(|((name, sum), mean_rank)| {
                serde_json::json!({
                    "treatment": name,
                    "weighted_rank_sum": sum,
                    "mean_weighted_rank": mean_rank
                })
            })
            .collect();

        let mut output = serde_json::json!({
            "method": "Quade test",
            "statistic": result.statistic,
            "statistic_name": "F",
            "df1": result.df1,
            "df2": result.df2,
            "p_value": result.p_value,
            "n_blocks": result.n_blocks,
            "n_treatments": result.n_treatments,
            "a_statistic": result.a_statistic,
            "b_statistic": result.b_statistic,
            "block_ranges": result.block_ranges,
            "treatment_statistics": treatment_stats,
            "hypothesis": {
                "null": "All treatments have the same effect",
                "alternative": "At least one treatment has a different effect"
            },
            "conclusion": if result.p_value < 0.05 {
                "Reject H₀: Significant difference in treatment effects."
            } else {
                "Fail to reject H₀: No significant difference in treatment effects."
            },
            "references": {
                "method": "Quade (1979), JASA 74(367):680-683",
                "text": "Conover (1999), Practical Nonparametric Statistics, pp. 373-380"
            }
        });
        insert_missing(&mut output, &result.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Shapiro-Wilk test for normality.
    #[tool(
        description = "Run the Shapiro-Wilk test for normality. Tests the null hypothesis that a sample came from a normally distributed population. Returns W statistic (values close to 1 indicate normality) and p-value. Sample size must be between 3 and 5000. A small p-value (e.g., < 0.05) suggests data is not normally distributed."
    )]
    pub async fn hypothesis_shapiro_wilk(
        &self,
        Parameters(request): Parameters<ShapiroWilkRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::shapiro::run_shapiro_wilk;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        match run_shapiro_wilk(dataset, &request.column) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.test_name,
                    "w_statistic": result.w_statistic,
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "n": result.n,
                    "reject_normality": result.reject_normality,
                    "interpretation": if result.reject_normality {
                        "Evidence against normality (reject H₀ at α = 0.05)"
                    } else {
                        "No evidence against normality (fail to reject H₀ at α = 0.05)"
                    }
                });
                insert_missing(&mut json_output, &result.missing);
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Shapiro-Wilk test failed: {}",
                e
            ))])),
        }
    }

    /// Run t-test for comparing means.
    #[tool(
        description = "Run Student's t-test for comparing means. Supports: (1) One-sample t-test: compare sample mean to hypothesized value, (2) Two-sample t-test: pass x and y as two numeric columns to compare (Welch's by default), (3) Paired t-test: compare matched pairs. Returns t-statistic, p-value, confidence interval, and effect estimate. Note: x and y must both be numeric columns. To compare a numeric variable across groups (e.g., salary by gender), first filter into two datasets or use two numeric columns. Missing values are dropped as R's t.test does (each sample on its own; complete pairs when paired) and reported in a 'Missing values' line. Integer columns are accepted."
    )]
    pub async fn hypothesis_t_test(
        &self,
        Parameters(request): Parameters<TTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let mu = request.mu.unwrap_or(0.0);
        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        let paired = request.paired.unwrap_or(false);
        let var_equal = request.var_equal.unwrap_or(false);
        let conf_level = request.conf_level.unwrap_or(0.95);

        if paired && request.y.is_none() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Paired t-test requires both x and y columns.".to_string(),
            )]));
        }

        match t_test(
            dataset,
            &request.x,
            request.y.as_deref(),
            mu,
            alternative,
            paired,
            var_equal,
            conf_level,
        ) {
            Ok(r) => Ok(CallToolResult::success(vec![Content::text(r.to_string())])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "t-test failed: {}",
                e
            ))])),
        }
    }

    /// Run Wilcoxon rank sum or signed rank test.
    #[tool(
        description = "Run Wilcoxon non-parametric test for location. Supports: (1) One-sample signed rank test: test if median differs from hypothesized value, (2) Two-sample rank sum test (Mann-Whitney U): compare distributions between two groups, (3) Paired signed rank test: compare matched pairs. Does not assume normality. Returns W/V statistic, p-value, and optionally confidence interval and location estimate."
    )]
    pub async fn hypothesis_wilcoxon(
        &self,
        Parameters(request): Parameters<WilcoxonTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::wilcoxon::{WilcoxonConfig, wilcoxon_test};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alternative = match parse_option(request.alternative.as_deref(), Alternative::TwoSided)
        {
            Ok(a) => a,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let mu = request.mu.unwrap_or(0.0);
        let paired = request.paired.unwrap_or(false);
        let mut config = WilcoxonConfig::default();
        config.exact = request.exact;
        config.correct = request.correct.unwrap_or(true);
        config.conf_int = request.conf_int.unwrap_or(false);
        config.conf_level = request.conf_level.unwrap_or(0.95);

        match wilcoxon_test(
            dataset,
            &request.x,
            request.y.as_deref(),
            mu,
            alternative,
            paired,
            &config,
        ) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.test_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "alternative": result.alternative.to_string(),
                    "null_value": result.null_value,
                    "exact": result.exact,
                    "continuity_correction": result.continuity_correction,
                    "n1": result.n,
                });
                insert_missing(&mut json_output, &result.missing);

                if let Some(n2) = result.n_2 {
                    json_output["n2"] = serde_json::json!(n2);
                }
                if let Some(u) = result.u_statistic {
                    json_output["u_statistic"] = serde_json::json!(u);
                }
                if let Some(z) = result.z_score {
                    json_output["z_score"] = serde_json::json!(z);
                }
                if let Some(est) = result.estimate {
                    json_output["estimate"] = serde_json::json!(est);
                }
                if let (Some(cl), Some(lo), Some(hi)) = (
                    result.conf_level,
                    result.conf_int_lower,
                    result.conf_int_upper,
                ) {
                    json_output["confidence_interval"] = serde_json::json!({
                        "level": cl,
                        "lower": lo,
                        "upper": hi
                    });
                }
                if result.n_ties > 0 {
                    json_output["n_ties"] = serde_json::json!(result.n_ties);
                }
                if let Some(ref warn) = result.warning {
                    json_output["warning"] = serde_json::json!(warn);
                }

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{}\n\n{}",
                    result,
                    serde_json::to_string_pretty(&json_output).unwrap_or_default()
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Wilcoxon test failed: {}",
                e
            ))])),
        }
    }
}

/// The values of the count column `name` (the kept `rows` of the dataset, 0-based) as `usize`.
/// Each must be a finite, non-negative whole number of at most 2^53, beyond which an f64 no
/// longer represents every integer, and their total must fit in a `usize`. The error names the column and
/// the first offending dataset row (1-based).
fn whole_counts(name: &str, values: &[f64], rows: &[usize]) -> Result<Vec<usize>, String> {
    const MAX_EXACT: f64 = 9_007_199_254_740_992.0;
    let mut counts = p2a_core::resource::try_with_capacity(values.len(), "count values")
        .map_err(|e| format!("Trend test failed: {e}"))?;
    let mut total = 0usize;
    for (&v, &row) in values.iter().zip(rows) {
        let refusal = || {
            format!(
                "Trend test failed: Column '{name}' must hold non-negative whole numbers (counts); \
                 row {} is {v}",
                row + 1
            )
        };
        let whole = v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= MAX_EXACT;
        if !whole {
            return Err(refusal());
        }
        let count = usize::try_from(v as u64).map_err(|_| refusal())?;
        total = total.checked_add(count).ok_or_else(|| {
            format!(
                "Trend test failed: the counts of column '{name}' sum past {}",
                usize::MAX
            )
        })?;
        counts.push(count);
    }
    Ok(counts)
}

/// Refuses a pairwise test whose factor has more groups than the `max_groups` budget, before any
/// of the k(k-1)/2 tests run.
///
/// The count is the factor's distinct values minus one for null, an upper bound on the core's k
/// (the core also drops rows whose response is missing). Nothing is counted when `max_groups` is
/// off. A missing column is left to the core to report; a count that fails is a refusal.
fn check_pairwise_groups(
    dataset: &p2a_core::data::Dataset,
    tool: &str,
    label: &str,
    factor: &str,
) -> Result<(), String> {
    let Some(max_groups) = crate::tools::limits::current().max_groups else {
        return Ok(());
    };
    let Some(k) = budget_group_count(dataset.df(), factor)
        .map_err(|msg| format!("Pairwise {label} refused: {msg}"))?
    else {
        return Ok(());
    };
    check_named_budget(tool, "max_groups", k, Some(max_groups as u64)).map_err(|msg| {
        let comparisons = k.saturating_mul(k.saturating_sub(1)) / 2;
        format!(
            "Pairwise {label} refused: factor `{}` has {k} groups, which needs \
             {comparisons} comparisons. {msg} Use a factor with fewer levels or filter the data \
             first.",
            echo(factor)
        )
    })
}

/// A number for a JSON result: finite values as numbers, others as the strings `"Inf"`,
/// `"-Inf"` or `"NaN"` (JSON has no infinity, and `null` would hide which it is).
fn json_number(v: f64) -> serde_json::Value {
    if v.is_finite() {
        serde_json::json!(v)
    } else if v == f64::INFINITY {
        serde_json::json!("Inf")
    } else if v == f64::NEG_INFINITY {
        serde_json::json!("-Inf")
    } else {
        serde_json::json!("NaN")
    }
}

/// Whether `numbers` JSON numbers (about 32 bytes each) and labels taking `label_bytes` fit
/// this server's `response_bytes` budget (always, when it is off).
fn fits_response_budget(numbers: u64, label_bytes: u64) -> bool {
    crate::tools::limits::current()
        .response_bytes
        .is_none_or(|cap| numbers.saturating_mul(32).saturating_add(label_bytes) <= cap)
}

/// The JSON bytes a list of labels takes for [`fits_response_budget`]: each label's length
/// plus 8 for its quotes, comma and indentation.
fn label_budget_bytes(labels: &[String]) -> u64 {
    labels
        .iter()
        .fold(0u64, |acc, l| acc.saturating_add(l.len() as u64 + 8))
}

/// The k x k lower-triangular matrix of adjusted p-values (row i, column j < i), with its size
/// checked before it is built.
fn lower_triangular_p_matrix(
    k: usize,
    p_value: impl Fn(usize, usize) -> f64,
) -> Result<Vec<Vec<Option<f64>>>, String> {
    let cells =
        p2a_core::resource::checked_len(&[k, k], "p-value matrix").map_err(|e| e.to_string())?;
    p2a_core::resource::check_alloc::<Option<f64>>(cells, "p-value matrix")
        .map_err(|e| e.to_string())?;
    let mut p_matrix: Vec<Vec<Option<f64>>> = vec![vec![None; k]; k];
    for (i, row) in p_matrix.iter_mut().enumerate().skip(1) {
        for (j, cell) in row.iter_mut().enumerate().take(i) {
            *cell = Some(p_value(i, j));
        }
    }
    Ok(p_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::level_count_or_refusal;
    use crate::tools::common::test_support::{hosted_with, server_with, text};
    use crate::tools::limits::{self, Limits};
    use p2a_core::polars::prelude::*;
    use p2a_validation::{Case, Check, Tol};

    #[test]
    fn test_hypothesis_option_tables() {
        use p2a_core::stats::{CmhAlternative, fisher::FisherAlternative};
        assert!(FISHER_ALTERNATIVE.validate().is_ok() && CMH_ALTERNATIVE.validate().is_ok());
        for s in ["two.sided", "two-sided", "TWOSIDED"] {
            assert_eq!(
                parse_choice(&FISHER_ALTERNATIVE, Some(s), FisherAlternative::Less),
                Ok(FisherAlternative::TwoSided)
            );
        }
        assert_eq!(
            parse_choice(&CMH_ALTERNATIVE, Some("Right"), CmhAlternative::TwoSided),
            Ok(CmhAlternative::Greater)
        );
        assert_eq!(
            parse_choice(&CMH_ALTERNATIVE, Some("bigger"), CmhAlternative::TwoSided).unwrap_err(),
            "unknown alternative 'bigger'; expected one of: two.sided, greater, less"
        );
    }

    /// `r{i}`, `c{i}`: an n x n table
    fn pair_ids(n: usize) -> DataFrame {
        df!(
            "id1" => (0..n).map(|i| format!("r{i}")).collect::<Vec<_>>(),
            "id2" => (0..n).map(|i| format!("c{i}")).collect::<Vec<_>>()
        )
        .unwrap()
    }

    fn chisq_req(row: &str, col: &str) -> ChiSquaredIndependenceRequest {
        ChiSquaredIndependenceRequest {
            dataset: "d".into(),
            row_var: row.into(),
            col_var: col.into(),
            correct: None,
        }
    }

    #[test]
    fn chisq_independence_refuses_table_above_table_cells() {
        // 1001 x 1001 = 1,002,001 > 1,000,000 cells
        let (rt, server) = server_with(vec![("d", pair_ids(1001))]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_chisq_independence(Parameters(chisq_req("id1", "id2"))))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(t.contains("1002001") && t.contains("table_cells"), "{t}");
    }

    #[test]
    fn chisq_independence_refusal_bounds_the_names_it_echoes() {
        let (row, col) = ("r".repeat(300), "c".repeat(300));
        let mut df = pair_ids(1001);
        df.rename("id1", row.as_str().into()).unwrap();
        df.rename("id2", col.as_str().into()).unwrap();
        let (rt, server) = server_with(vec![("d", df)]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_chisq_independence(Parameters(chisq_req(&row, &col))))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("table_cells") && t.contains(&"r".repeat(64)),
            "{t}"
        );
        assert!(
            !t.contains(&"r".repeat(65)) && !t.contains(&"c".repeat(65)),
            "{t}"
        );
    }

    #[test]
    fn chisq_independence_with_limits_off_returns_every_array() {
        // No budget: no level pre-count, and the per-cell arrays are always returned
        let (rt, server) = server_with(vec![("d", pair_ids(50))]);
        let r = limits::with_limits(Limits::off(), || {
            rt.block_on(server.hypothesis_chisq_independence(Parameters(chisq_req("id1", "id2"))))
        })
        .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let v: serde_json::Value = serde_json::from_str(&text(&r)).unwrap();
        assert_eq!(v["observed"].as_array().map(Vec::len), Some(2500));
        assert_eq!(v["std_residuals"].as_array().map(Vec::len), Some(2500));
    }

    #[test]
    fn level_count_failure_is_a_refusal_naming_the_column() {
        // Pre-fix a failed n_unique() skipped the budget check and the core ran unbounded
        let long = "g".repeat(100);
        let e = level_count_or_refusal(&long, Err(PolarsError::ComputeError("boom".into())))
            .unwrap_err();
        assert!(
            e.contains(&"g".repeat(64)) && !e.contains(&"g".repeat(65)),
            "{e}"
        );
        assert!(
            e.contains("could not be counted") && e.contains("boom"),
            "{e}"
        );
        assert_eq!(level_count_or_refusal("g", Ok(7)), Ok(7));
    }

    #[test]
    fn chisq_independence_omits_per_cell_arrays_above_response_budget() {
        // 400 x 400 = 160,000 cells: within table_cells, but 4 arrays x 32 bytes x 160,000 > 16 MiB
        let (rt, server) = server_with(vec![("d", pair_ids(400))]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_chisq_independence(Parameters(chisq_req("id1", "id2"))))
        })
        .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        assert!(
            t.contains("\"statistic\"") && t.contains("omitted") && !t.contains("\"expected\""),
            "{}",
            t.chars().take(400).collect::<String>()
        );
    }

    #[test]
    fn chisq_independence_small_table_matches_r_under_hosted_limits() {
        // same 40 rows as the core test; pre-fix this panicked inside the runtime (STAT-10)
        let mut h: Vec<&str> = Vec::new();
        for (lvl, k) in [("x", 7), ("y", 5), ("z", 8), ("x", 4), ("y", 9), ("z", 7)] {
            h.extend(std::iter::repeat_n(lvl, k));
        }
        let g: Vec<&str> = (0..40).map(|i| if i < 20 { "a" } else { "b" }).collect();
        let (rt, server) = server_with(vec![("d", df!("g" => g, "h" => h).unwrap())]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_chisq_independence(Parameters(chisq_req("g", "h"))))
        })
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&text(&r)).unwrap();
        assert!((v["statistic"].as_f64().unwrap() - 2.0277056277056276).abs() < 1e-10);
        assert!(v["expected"].is_array());
    }

    fn grouped(n_rows: usize, rows_per_group: usize) -> DataFrame {
        df!(
            "y" => (0..n_rows)
                .map(|i| (i as f64 * 0.37).sin() + (i % 7) as f64 * 0.1)
                .collect::<Vec<_>>(),
            "g" => (0..n_rows)
                .map(|i| format!("g{}", i / rows_per_group))
                .collect::<Vec<_>>()
        )
        .unwrap()
    }

    fn ttest_req() -> PairwiseTTestRequest {
        PairwiseTTestRequest {
            dataset: "d".into(),
            response: "y".into(),
            factor: "g".into(),
            p_adjust_method: None,
            pool_sd: None,
            alternative: None,
        }
    }

    #[test]
    fn pairwise_t_test_refuses_more_groups_than_max_groups() {
        let (rt, server) = server_with(vec![("d", grouped(202, 2))]); // 101 groups > 100
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_pairwise_t_test(Parameters(ttest_req())))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(
            text(&r).contains("max_groups") && text(&r).contains("101"),
            "{}",
            text(&r)
        );
    }

    #[test]
    fn pairwise_wilcox_refuses_more_groups_than_max_groups() {
        let (rt, server) = server_with(vec![("d", grouped(101, 1))]); // 101 singleton groups
        let req = PairwiseWilcoxRequest {
            dataset: "d".into(),
            response: "y".into(),
            factor: "g".into(),
            p_adjust_method: None,
            alternative: None,
            exact: None,
        };
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_pairwise_wilcox(Parameters(req)))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("max_groups"), "{}", text(&r));
    }

    #[test]
    fn pairwise_refusal_bounds_the_factor_name_it_echoes() {
        let long = "f".repeat(300);
        let mut df = grouped(202, 2); // 101 groups > 100
        df.rename("g", long.as_str().into()).unwrap();
        let (rt, server) = server_with(vec![("d", df)]);
        let req = PairwiseTTestRequest {
            factor: long.clone(),
            ..ttest_req()
        };
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_pairwise_t_test(Parameters(req)))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("max_groups") && t.contains(&"f".repeat(64)),
            "{t}"
        );
        assert!(!t.contains(&"f".repeat(65)), "{t}");
    }

    #[test]
    fn pairwise_group_cap_boundary() {
        let l = hosted_with(&["max_groups=3"]);
        let (rt, server) = server_with(vec![("ok", grouped(15, 5)), ("over", grouped(20, 5))]);
        let ok = PairwiseTTestRequest {
            dataset: "ok".into(),
            ..ttest_req()
        };
        let over = PairwiseTTestRequest {
            dataset: "over".into(),
            ..ttest_req()
        };
        let r = limits::with_limits(l.clone(), || {
            rt.block_on(server.hypothesis_pairwise_t_test(Parameters(ok)))
        })
        .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let r = limits::with_limits(l, || {
            rt.block_on(server.hypothesis_pairwise_t_test(Parameters(over)))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
    }

    #[test]
    fn pairwise_p_matrix_layout_is_lower_triangular() {
        let m = lower_triangular_p_matrix(3, |i, j| (i * 10 + j) as f64).unwrap();
        assert_eq!(
            m,
            vec![
                vec![None, None, None],
                vec![Some(10.0), None, None],
                vec![Some(20.0), Some(21.0), None],
            ]
        );
    }

    async fn case_server(id: &str) -> (Case, AnalyticsServer, String) {
        let case = Case::load(id);
        let df = p2a_core::data::DataLoader::load_csv(case.dataset_path()).unwrap();
        let name = std::path::Path::new(&case.dataset)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let server = AnalyticsServer::new();
        server
            .datasets
            .write()
            .await
            .insert(name.clone(), p2a_core::data::Dataset::new(df));
        (case, server, name)
    }

    fn json_of(r: &CallToolResult) -> serde_json::Value {
        assert_ne!(r.is_error, Some(true), "{}", text(r));
        serde_json::from_str(&text(r)).unwrap()
    }

    async fn gof(server: &AnalyticsServer, args: serde_json::Value) -> CallToolResult {
        let req: ChiSquaredGofRequest = serde_json::from_value(args).unwrap();
        server.hypothesis_chisq_gof(Parameters(req)).await.unwrap()
    }

    /// The tight re-solve of fisher.test's equations (validation/stats/fisher.md, "Tight
    /// limits, not R's printed ones, are the reference").
    const TIGHT: Tol = Tol::new(1e-9, 1e-12);

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_fisher_exact_rows_2000() {
        // STAT-10: the Fisher table no longer goes through polars' lazy engine, so the tool
        // answers inside the server's async runtime; STAT-2: p is R's value, not 0.
        // R: fisher.test(table(treat, outcome)).
        let (case, server, name) = case_server("stat_fisher_rows_2000").await;
        let req: FisherExactRequest = serde_json::from_value(serde_json::json!(
            {"dataset": name, "row_var": "treat", "col_var": "outcome", "conf_level": 0.95}))
        .unwrap();
        let v = json_of(
            &server
                .hypothesis_fisher_exact(Parameters(req))
                .await
                .unwrap(),
        );
        Check::new(&case)
            .scalar(
                "p_two_sided",
                v["p_value"].as_f64().unwrap(),
                Tol::p_value(2000),
            )
            .scalar(
                "estimate_cond_mle_tight",
                v["conditional_mle"].as_f64().unwrap(),
                TIGHT,
            )
            .scalar(
                "ci_lower_two_sided_tight",
                v["odds_ratio_ci"]["lower"].as_f64().unwrap(),
                TIGHT,
            )
            .scalar(
                "ci_upper_two_sided_tight",
                v["odds_ratio_ci"]["upper"].as_f64().unwrap(),
                TIGHT,
            )
            .scalar(
                "sample_odds_ratio",
                v["sample_odds_ratio"].as_f64().unwrap(),
                Tol::new(1e-15, 0.0),
            )
            .scalar(
                "sample_odds_ratio",
                v["odds_ratio"].as_f64().unwrap(),
                Tol::new(1e-15, 0.0),
            )
            .scalar("n", v["n"].as_f64().unwrap(), Tol::exact())
            .finish();
        assert_eq!(v["row_levels"], serde_json::json!(["control", "treated"]));
        assert_eq!(v["col_levels"], serde_json::json!(["no", "yes"]));
        assert_eq!(
            v["table"],
            serde_json::json!({"a": 520.0, "b": 480.0, "c": 480.0, "d": 520.0})
        );
        assert_eq!(
            v["odds_ratio_ci"]["method"],
            "exact conditional (as R fisher.test)"
        );
        assert!(v.get("missing").is_none());
        // greater: the upper limit is +inf, shown as "Inf" (JSON has no infinity)
        let req: FisherExactRequest = serde_json::from_value(serde_json::json!(
            {"dataset": name, "row_var": "treat", "col_var": "outcome", "conf_level": 0.95,
             "alternative": "greater"}))
        .unwrap();
        let g = json_of(
            &server
                .hypothesis_fisher_exact(Parameters(req))
                .await
                .unwrap(),
        );
        assert_eq!(g["odds_ratio_ci"]["upper"], "Inf");
        Check::new(&case)
            .scalar(
                "p_greater",
                g["p_value"].as_f64().unwrap(),
                Tol::p_value(2000),
            )
            .scalar(
                "ci_lower_greater_tight",
                g["odds_ratio_ci"]["lower"].as_f64().unwrap(),
                TIGHT,
            )
            .finish();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_chisq_independence_rows_2000() {
        // R: chisq.test(table(treat, outcome)) with Yates: 3.042, p 0.08113589702211416 (no
        // panic since T0-WP19).
        let (case, server, name) = case_server("stat_fisher_rows_2000").await;
        let req: ChiSquaredIndependenceRequest = serde_json::from_value(serde_json::json!(
            {"dataset": name, "row_var": "treat", "col_var": "outcome"}))
        .unwrap();
        let v = json_of(
            &server
                .hypothesis_chisq_independence(Parameters(req))
                .await
                .unwrap(),
        );
        Check::new(&case)
            .scalar(
                "chisq_yates",
                v["statistic"].as_f64().unwrap(),
                Tol::stat(2000),
            )
            .scalar(
                "chisq_yates_p",
                v["p_value"].as_f64().unwrap(),
                Tol::p_value(2000),
            )
            .finish();
        assert_eq!(v["row_levels"], serde_json::json!(["control", "treated"]));
        assert_eq!(v["col_levels"], serde_json::json!(["no", "yes"]));
        assert_eq!(v["warnings"], serde_json::json!([]));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_chisq_independence_small_expected() {
        // TOOL-33: R warns "Chi-squared approximation may be incorrect"; the JSON had neither
        // warning nor labels.
        let (case, server, name) = case_server("stat_chisq_small_expected").await;
        let req: ChiSquaredIndependenceRequest = serde_json::from_value(serde_json::json!(
            {"dataset": name, "row_var": "x", "col_var": "y"}))
        .unwrap();
        let v = json_of(
            &server
                .hypothesis_chisq_independence(Parameters(req))
                .await
                .unwrap(),
        );
        let f = |k: &str| {
            v[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        Check::new(&case)
            .scalar("chisq", v["statistic"].as_f64().unwrap(), Tol::stat(13))
            .scalar("p", v["p_value"].as_f64().unwrap(), Tol::p_value(13))
            .vector("observed", &f("observed"), Tol::exact())
            .vector("expected", &f("expected"), Tol::stat(13))
            .vector("residuals", &f("residuals"), Tol::stat(13))
            .vector("stdres", &f("std_residuals"), Tol::stat(13))
            .finish();
        assert_eq!(v["row_levels"], serde_json::json!(["p", "q", "r"]));
        assert_eq!(v["col_levels"], serde_json::json!(["s", "t"]));
        assert!(
            v["warnings"][0]
                .as_str()
                .unwrap()
                .starts_with("Chi-squared approximation may be incorrect")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_chisq_gof_expected() {
        // STAT-3/STAT-45: R chisq.test(table(cat), p = c(a = .1, b = .6, c = .3)) -> X2 0, p 1;
        // the named form with an unobserved "d" -> 6, df 3, p 0.11161022509471258; the 2 NA
        // cells are reported. Fails before: `expected` was ignored (uniform, with the null as
        // a 4th category: 36.769, df 3).
        let (case, server, name) = case_server("stat_chisq_gof").await;
        let v = json_of(
            &gof(
                &server,
                serde_json::json!({"dataset": name, "column": "cat",
                                   "expected": {"a": 0.1, "b": 0.6, "c": 0.3}}),
            )
            .await,
        );
        let w = json_of(
            &gof(
                &server,
                serde_json::json!({"dataset": name, "column": "cat",
                                   "expected": {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.1}}),
            )
            .await,
        );
        let p = json_of(
            &gof(
                &server,
                serde_json::json!({"dataset": name, "column": "cat", "probs": [0.2, 0.5, 0.3]}),
            )
            .await,
        );
        Check::new(&case)
            .scalar("p1_chisq", v["statistic"].as_f64().unwrap(), Tol::stat(50))
            .scalar("p1_p", v["p_value"].as_f64().unwrap(), Tol::p_value(50))
            .scalar(
                "named4_chisq",
                w["statistic"].as_f64().unwrap(),
                Tol::stat(50),
            )
            .scalar("named4_df", w["df"].as_f64().unwrap(), Tol::exact())
            .scalar("named4_p", w["p_value"].as_f64().unwrap(), Tol::p_value(50))
            .scalar("p2_chisq", p["statistic"].as_f64().unwrap(), Tol::stat(50))
            .scalar("n_used", v["n"].as_f64().unwrap(), Tol::exact())
            .finish();
        assert_eq!(v["categories"], serde_json::json!(["a", "b", "c"]));
        assert_eq!(w["categories"], serde_json::json!(["a", "b", "c", "d"]));
        assert_eq!(v["observed"], serde_json::json!([5.0, 30.0, 15.0]));
        assert_eq!(v["missing"]["n_dropped"], 2);
        // both forms at once is a tool error, not a guess
        let both = gof(
            &server,
            serde_json::json!({"dataset": name, "column": "cat", "probs": [0.1, 0.6, 0.3],
                               "expected": {"a": 0.1, "b": 0.6, "c": 0.3}}),
        )
        .await;
        assert_eq!(both.is_error, Some(true));
        assert!(text(&both).contains("either"), "{}", text(&both));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_mantelhaen_na() {
        // R: mantelhaen.test(table(treat, outcome, stratum)), NA strata dropped (237 rows).
        let (case, server, name) = case_server("stat_mantelhaen_na").await;
        let req: MantelhaenTestRequest = serde_json::from_value(serde_json::json!(
            {"dataset": name, "row_var": "treat", "col_var": "outcome", "stratum_var": "stratum"}))
        .unwrap();
        let v = json_of(&server.hypothesis_mantelhaen(Parameters(req)).await.unwrap());
        Check::new(&case)
            .scalar(
                "statistic",
                v["statistic"].as_f64().unwrap(),
                Tol::stat(237),
            )
            .scalar("p", v["p_value"].as_f64().unwrap(), Tol::p_value(237))
            .scalar(
                "common_or",
                v["common_odds_ratio"].as_f64().unwrap(),
                Tol::stat(237),
            )
            .scalar("n_used", v["total_n"].as_f64().unwrap(), Tol::exact())
            .finish();
        assert_eq!(v["n_strata"], 3);
        assert_eq!(v["row_levels"], serde_json::json!(["ctl", "trt"]));
        assert_eq!(v["missing"]["by_column"]["stratum"], 3);
    }

    #[test]
    fn test_mcp_hypothesis_chisq_gof_omits_arrays_above_response_budget() {
        // per-category arrays above the hosted response budget are left out; the statistic is
        // still returned.
        let ids: Vec<String> = (0..200_000).map(|i| format!("c{i}")).collect();
        let (rt, server) = server_with(vec![("d", df!("id" => ids).unwrap())]);
        let req: ChiSquaredGofRequest =
            serde_json::from_value(serde_json::json!({"dataset": "d", "column": "id"})).unwrap();
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.hypothesis_chisq_gof(Parameters(req)))
        })
        .unwrap();
        let t = text(&r);
        assert!(
            t.contains("\"statistic\"")
                && t.contains("per_category_arrays")
                && !t.contains("\"categories\""),
            "{}",
            t.chars().take(400).collect::<String>()
        );
    }

    #[test]
    fn test_mcp_hypothesis_chisq_gof_holds_expected_to_the_array_length_budget() {
        // review: the expected map had no length bound but the request body; it now has the
        // array_length budget that bounds probs.
        let mut cat: Vec<&str> = Vec::new();
        for (lvl, k) in [("a", 7), ("b", 12), ("c", 9)] {
            cat.extend(std::iter::repeat_n(lvl, k));
        }
        let (rt, server) = server_with(vec![("d", df!("cat" => cat).unwrap())]);
        let call = |l: Limits| {
            let req: ChiSquaredGofRequest = serde_json::from_value(serde_json::json!(
                {"dataset": "d", "column": "cat", "expected": {"a": 0.2, "b": 0.5, "c": 0.3}}))
            .unwrap();
            limits::with_limits(l, || {
                rt.block_on(server.hypothesis_chisq_gof(Parameters(req)))
            })
            .unwrap()
        };
        let r = call(hosted_with(&["array_length=2"]));
        assert_eq!(r.is_error, Some(true), "{}", text(&r));
        assert!(
            text(&r).contains("array_length") && text(&r).contains("3 categories"),
            "{}",
            text(&r)
        );
        let ok = call(hosted_with(&["array_length=3"]));
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
        assert_eq!(
            json_of(&ok)["categories"],
            serde_json::json!(["a", "b", "c"])
        );
    }

    #[test]
    fn test_mcp_hypothesis_mantelhaen_spells_an_infinite_odds_ratio() {
        // review: every stratum has b * c = 0, so the common odds ratio, the upper limit and
        // each stratum's odds ratio are +inf; serde_json turned them into null.
        // s1: a = 3, b = 0, c = 2, d = 4; s2: a = 2, b = 1, c = 0, d = 5
        let mut t: Vec<&str> = Vec::new();
        let mut y: Vec<&str> = Vec::new();
        let mut s: Vec<&str> = Vec::new();
        for (stratum, cells) in [("s1", [3, 0, 2, 4]), ("s2", [2, 1, 0, 5])] {
            let kinds = [("ctl", "no"), ("ctl", "yes"), ("trt", "no"), ("trt", "yes")];
            for ((ti, yi), k) in kinds.into_iter().zip(cells) {
                t.extend(std::iter::repeat_n(ti, k));
                y.extend(std::iter::repeat_n(yi, k));
                s.extend(std::iter::repeat_n(stratum, k));
            }
        }
        let (rt, server) = server_with(vec![("d", df!("t" => t, "y" => y, "s" => s).unwrap())]);
        let req: MantelhaenTestRequest = serde_json::from_value(serde_json::json!(
            {"dataset": "d", "row_var": "t", "col_var": "y", "stratum_var": "s"}))
        .unwrap();
        let v = json_of(
            &rt.block_on(server.hypothesis_mantelhaen(Parameters(req)))
                .unwrap(),
        );
        assert_eq!(v["common_odds_ratio"], "Inf");
        assert_eq!(v["odds_ratio_ci"]["upper"], "Inf");
        // 0 is p2a's fallback when sum(a d / n) or sum(b c / n) is 0; R's mantelhaen.test
        // gives conf.int NaN NaN (and estimate Inf)
        assert_eq!(v["odds_ratio_ci"]["lower"], 0.0);
        assert_eq!(v["stratum_statistics"][0]["odds_ratio"], "Inf");
        assert_eq!(v["stratum_statistics"][1]["odds_ratio"], "Inf");
        assert_eq!(v["stratum_statistics"][0]["counts"]["a"], 3.0);
        // R: mantelhaen.test(array(c(3, 2, 0, 4, 2, 0, 1, 5), c(2, 2, 2)))$statistic
        let stat = v["statistic"].as_f64().unwrap_or(f64::NAN);
        assert!(Tol::printed(6).allows(stat, 4.533679), "{v}");
    }

    /// 2 x 2 x `k` strata labelled s01, s02, ...: ctl/trt by no/yes, every cell non-empty so
    /// each odds ratio is finite.
    fn mantelhaen_strata(k: usize) -> DataFrame {
        let (mut t, mut y, mut s): (Vec<&str>, Vec<&str>, Vec<String>) = Default::default();
        for i in 0..k {
            let cells = [2 + i % 3, 1 + i % 2, 1 + i % 4, 3];
            let kinds = [("ctl", "no"), ("ctl", "yes"), ("trt", "no"), ("trt", "yes")];
            for ((ti, yi), n) in kinds.into_iter().zip(cells) {
                t.extend(std::iter::repeat_n(ti, n));
                y.extend(std::iter::repeat_n(yi, n));
                s.extend(std::iter::repeat_n(format!("s{:02}", i + 1), n));
            }
        }
        df!("t" => t, "y" => y, "s" => s).unwrap()
    }

    #[test]
    fn test_mcp_hypothesis_mantelhaen_omits_strata_over_the_response_budget() {
        let (rt, server) = server_with(vec![("d", mantelhaen_strata(40))]);
        let run = || {
            let req: MantelhaenTestRequest = serde_json::from_value(serde_json::json!(
                {"dataset": "d", "row_var": "t", "col_var": "y", "stratum_var": "s"}))
            .unwrap();
            json_of(
                &rt.block_on(server.hypothesis_mantelhaen(Parameters(req)))
                    .unwrap(),
            )
        };
        let full = limits::with_limits(Limits::off(), run);
        let small = limits::with_limits(hosted_with(&["response_bytes=2000"]), run);
        assert_eq!(
            full["stratum_statistics"].as_array().map(Vec::len),
            Some(40)
        );
        assert!(small.get("stratum_statistics").is_none(), "{small}");
        assert!(
            small["per_stratum_statistics"]
                .as_str()
                .is_some_and(|m| m.contains("40 strata are too many to return")),
            "{small}"
        );
        for key in ["statistic", "p_value", "common_odds_ratio"] {
            assert_eq!(small[key], full[key], "{key}");
        }
    }

    #[test]
    fn test_mcp_prop_trend_whole_counts_message() {
        let e = whole_counts("x", &[2.0, 1.5], &[0, 1]).unwrap_err();
        assert!(
            e.starts_with(
                "Trend test failed: Column 'x' must hold non-negative whole numbers (counts); row 2 is 1.5"
            ),
            "{e}"
        );
    }

    #[test]
    fn test_mcp_json_number_spells_non_finite_values() {
        assert_eq!(json_number(1.5), serde_json::json!(1.5));
        assert_eq!(json_number(f64::INFINITY), "Inf");
        assert_eq!(json_number(f64::NEG_INFINITY), "-Inf");
        assert_eq!(json_number(f64::NAN), "NaN");
        assert!(limits::with_limits(Limits::off(), || fits_response_budget(
            u64::MAX,
            1
        )));
        let small = hosted_with(&["response_bytes=100"]);
        assert!(limits::with_limits(small.clone(), || fits_response_budget(
            3, 4
        )));
        assert!(!limits::with_limits(small, || fits_response_budget(3, 5)));
    }

    /// Calls the hypothesis tool named `tool` (without its `hypothesis_` prefix) with `args`,
    /// deserialised into the tool's request type.
    async fn hyp(server: &AnalyticsServer, tool: &str, args: serde_json::Value) -> CallToolResult {
        fn req<T: serde::de::DeserializeOwned>(v: serde_json::Value) -> Parameters<T> {
            Parameters(serde_json::from_value(v).unwrap())
        }
        match tool {
            "t_test" => server.hypothesis_t_test(req(args)).await,
            "cor_test" => server.hypothesis_cor_test(req(args)).await,
            "ks_test" => server.hypothesis_ks_test(req(args)).await,
            "mood_test" => server.hypothesis_mood_test(req(args)).await,
            "bartlett_test" => server.hypothesis_bartlett_test(req(args)).await,
            "kruskal_wallis" => server.hypothesis_kruskal_wallis(req(args)).await,
            "oneway" => server.hypothesis_oneway(req(args)).await,
            "friedman" => server.hypothesis_friedman(req(args)).await,
            "quade" => server.hypothesis_quade(req(args)).await,
            "wilcoxon" => server.hypothesis_wilcoxon(req(args)).await,
            "pairwise_t_test" => server.hypothesis_pairwise_t_test(req(args)).await,
            "pairwise_wilcox" => server.hypothesis_pairwise_wilcox(req(args)).await,
            "shapiro_wilk" => server.hypothesis_shapiro_wilk(req(args)).await,
            "prop_trend_test" => server.hypothesis_prop_trend_test(req(args)).await,
            other => panic!("no hypothesis tool {other}"),
        }
        .unwrap()
    }

    /// The text of a successful tool result.
    fn ok_text(r: &CallToolResult) -> String {
        assert_ne!(r.is_error, Some(true), "{}", text(r));
        text(r)
    }

    /// The text of a tool error.
    fn err_text(r: &CallToolResult) -> String {
        assert_eq!(
            r.is_error,
            Some(true),
            "expected a tool error, got:\n{}",
            text(r)
        );
        text(r)
    }

    /// The number after the first `label` in `text`.
    fn number_after(text: &str, label: &str) -> f64 {
        let i = text
            .find(label)
            .unwrap_or_else(|| panic!("`{label}` not found in:\n{text}"))
            + label.len();
        let s: String = text[i..]
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
            .collect();
        s.parse()
            .unwrap_or_else(|_| panic!("no number after `{label}` in:\n{text}"))
    }

    /// "(lo, hi)" on the line after "confidence interval:" (the t-test's text).
    fn ci_line(text: &str) -> [f64; 2] {
        let line = text
            .lines()
            .skip_while(|l| !l.contains("confidence interval:"))
            .nth(1)
            .unwrap_or_else(|| panic!("no confidence interval in:\n{text}"));
        let (lo, hi) = line
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split_once(", ")
            .unwrap();
        [lo.parse().unwrap(), hi.parse().unwrap()]
    }

    /// A JSON number as f64 (NaN when it is not a number).
    fn num(v: &serde_json::Value) -> f64 {
        v.as_f64().unwrap_or(f64::NAN)
    }

    fn nums(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap_or_else(|| panic!("not an array: {v}"))
            .iter()
            .map(num)
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_t_test_two_sample_na() {
        // X-2/TOOL-9: nulls were read as 0 (one-sample t 3.093880 on 21 df) and Int64 columns
        // were refused. R: t.test drops NA from each sample, and incomplete pairs when paired.
        let (case, server, name) = case_server("stat_two_sample_na").await;
        let t = |args: serde_json::Value| {
            let server = &server;
            async move { ok_text(&hyp(server, "t_test", args).await) }
        };
        let one = t(serde_json::json!({"dataset": name, "x": "x"})).await;
        let pr = t(serde_json::json!({"dataset": name, "x": "x", "y": "y", "paired": true})).await;
        let we = t(serde_json::json!({"dataset": name, "x": "x", "y": "y"})).await;
        let it = t(serde_json::json!({"dataset": name, "x": "xi", "y": "yi"})).await;
        let (p6, p2) = (Tol::printed(6), Tol::printed(2));
        let mut c = Check::new(&case);
        for (k, out) in [
            ("one", &one),
            ("paired", &pr),
            ("welch", &we),
            ("int_welch", &it),
        ] {
            c.scalar(&format!("{k}_t"), number_after(out, "t = "), p6)
                .scalar(&format!("{k}_df"), number_after(out, "df = "), p2)
                .scalar(&format!("{k}_p"), number_after(out, "p-value = "), p6)
                .vector(&format!("{k}_ci"), &ci_line(out), p6);
        }
        c.scalar("one_mean", number_after(&one, "mean: "), p6)
            .scalar("paired_est", number_after(&pr, "mean: "), p6)
            .vector(
                "welch_est",
                &[
                    number_after(&we, "mean of x: "),
                    number_after(&we, "mean of y: "),
                ],
                p6,
            )
            .vector(
                "int_welch_est",
                &[
                    number_after(&it, "mean of x: "),
                    number_after(&it, "mean of y: "),
                ],
                p6,
            )
            .finish();
        assert!(
            one.contains("Missing values: 15 of 22 rows dropped because of missing values (x: 15)"),
            "{one}"
        );
        assert!(
            pr.contains(
                "Missing values: 16 of 22 rows dropped because of missing values (x: 15, y: 15)"
            ),
            "{pr}"
        );
        assert!(
            we.lines().any(|l| l.starts_with("Missing values: ")),
            "{we}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_cor_test_two_sample_na() {
        // X-2: nulls were read as 0 (r 0.7955101 on 22 pairs). R: cor.test(x, y) keeps the
        // complete pairs.
        let (case, server, name) = case_server("stat_two_sample_na").await;
        let v = json_of(
            &hyp(
                &server,
                "cor_test",
                serde_json::json!({"dataset": name, "x": "x", "y": "y"}),
            )
            .await,
        );
        let ci = &v["confidence_interval"];
        Check::new(&case)
            .scalar("cor_r", num(&v["estimate"]), Tol::coef(6))
            .scalar("cor_t", num(&v["statistic"]), Tol::stat(6))
            .scalar("cor_df", num(&v["df"]), Tol::exact())
            .scalar("cor_p", num(&v["p_value"]), Tol::p_value(6))
            .vector(
                "cor_ci",
                &[num(&ci["lower"]), num(&ci["upper"])],
                Tol::stat(6),
            )
            .scalar("n_pairs", num(&v["n"]), Tol::exact())
            .scalar("n_pairs", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            serde_json::json!({"n_total": 22, "n_used": 6, "n_dropped": 16,
                               "by_column": {"x": 15, "y": 15}})
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_ks_test_two_sample_na() {
        // X-8: the Int64 columns xi, yi were refused ("Column 'xi' is not numeric."). R:
        // ks.test(xi, yi) drops NA from each sample.
        let (case, server, name) = case_server("stat_two_sample_na").await;
        let v = json_of(
            &hyp(
                &server,
                "ks_test",
                serde_json::json!({"dataset": name, "x": "xi", "y": "yi"}),
            )
            .await,
        );
        Check::new(&case)
            .scalar("ks_d", num(&v["statistic_D"]), Tol::stat(38))
            .scalar("n_xi", num(&v["n"]), Tol::exact())
            .scalar("n_yi", num(&v["n_2"]), Tol::exact())
            .finish();
        // deviation STAT-27: ks_p (R's exact p 0.1565386776017454; p2a's asymptotic p
        // 0.19563779265722031) is not pinned.
        assert_eq!(
            v["missing"]["by_column"],
            serde_json::json!({"xi": 2, "yi": 4})
        );
        assert_eq!(v["missing"]["n_dropped"], serde_json::json!(6));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_mood_test_two_sample_na() {
        // X-8: the Int64 columns mi, mj were refused ("Column 'mi' must be numeric."). R:
        // mood.test(mi, mj) drops NA from each sample; the pair has no ties.
        let (case, server, name) = case_server("stat_two_sample_na").await;
        let v = json_of(
            &hyp(
                &server,
                "mood_test",
                serde_json::json!({"dataset": name, "x": "mi", "y": "mj"}),
            )
            .await,
        );
        Check::new(&case)
            .scalar("mood_nt_T", num(&v["statistic"]), Tol::stat(27))
            .scalar("mood_nt_z", num(&v["z_score"]), Tol::stat(27))
            .scalar("mood_nt_p", num(&v["p_value"]), Tol::p_value(27))
            .scalar("n_mi", num(&v["n_x"]), Tol::exact())
            .scalar("n_mj", num(&v["n_y"]), Tol::exact())
            .finish();
        assert_eq!(v["has_ties"], serde_json::json!(false));
        assert_eq!(
            v["missing"]["by_column"],
            serde_json::json!({"mi": 7, "mj": 10})
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mcp_hypothesis_two_sample_columns() {
        // Int64 columns give the results of their Float64 copies; a text column and an absent
        // column are refused with the core's error text, in either role. Before: Int64 was
        // refused and the errors named no reason.
        let df = p2a_core::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/generated/two_sample_na.csv",
        ))
        .unwrap();
        let mut floats = df.clone();
        for (from, to) in [("xi", "xi_f"), ("yi", "yi_f")] {
            let copy = floats
                .column(from)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .with_name(to.into());
            floats.with_column(copy).unwrap();
        }
        let server = AnalyticsServer::new();
        {
            let mut map = server.datasets.write().await;
            map.insert("d".into(), p2a_core::data::Dataset::new(df));
            map.insert("two_sample_f".into(), p2a_core::data::Dataset::new(floats));
        }
        let ints = json_of(
            &hyp(
                &server,
                "cor_test",
                serde_json::json!({"dataset": "d", "x": "xi", "y": "yi"}),
            )
            .await,
        );
        let fl = json_of(
            &hyp(
                &server,
                "cor_test",
                serde_json::json!({"dataset": "two_sample_f", "x": "xi_f", "y": "yi_f"}),
            )
            .await,
        );
        assert_eq!(ints["n"], serde_json::json!(18));
        assert_eq!(fl["n"], serde_json::json!(18));
        for key in ["estimate", "statistic", "p_value"] {
            assert_eq!(ints[key], fl[key], "{key}");
        }
        for tool in ["t_test", "cor_test", "ks_test", "mood_test"] {
            for (x, y) in [("g", "x"), ("x", "g")] {
                let t = err_text(
                    &hyp(
                        &server,
                        tool,
                        serde_json::json!({"dataset": "d", "x": x, "y": y}),
                    )
                    .await,
                );
                assert!(
                    t.contains("Column 'g' contains non-numeric values"),
                    "{tool} ({x}, {y}): {t}"
                );
            }
            for (x, y) in [("nope", "x"), ("x", "nope")] {
                let t = err_text(
                    &hyp(
                        &server,
                        tool,
                        serde_json::json!({"dataset": "d", "x": x, "y": y}),
                    )
                    .await,
                );
                assert!(
                    t.contains("Column 'nope' not found in dataset."),
                    "{tool} ({x}, {y}): {t}"
                );
            }
        }
        let t = err_text(
            &hyp(
                &server,
                "t_test",
                serde_json::json!({"dataset": "d", "x": "x", "paired": true}),
            )
            .await,
        );
        assert!(
            t.contains("Paired t-test requires both x and y columns."),
            "{t}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_handlers_report_missing_rows() {
        // X-2: the core wrappers drop incomplete rows (T2-WP06, T2-WP07), but the JSON did not
        // say so. The rows used are R's complete cases over the columns each R function uses.
        let summary = |n_total: usize, n_used: usize, by_column: serde_json::Value| {
            serde_json::json!({"n_total": n_total, "n_used": n_used,
                               "n_dropped": n_total - n_used, "by_column": by_column})
        };

        // bartlett.test(xi ~ g)
        let (case, server, name) = case_server("stat_two_sample_na").await;
        let v = json_of(
            &hyp(
                &server,
                "bartlett_test",
                serde_json::json!({"dataset": name, "response": "xi", "factor": "g"}),
            )
            .await,
        );
        Check::new(&case)
            .scalar("n_xi_g", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            summary(22, 20, serde_json::json!({"xi": 2, "g": 1}))
        );

        // kruskal.test(v9 ~ g9), oneway.test(v9 ~ g9)
        let (case, server, name) = case_server("stat_oneway_na_labels").await;
        for tool in ["kruskal_wallis", "oneway"] {
            let v = json_of(
                &hyp(
                    &server,
                    tool,
                    serde_json::json!({"dataset": name, "value": "v9", "group": "g9"}),
                )
                .await,
            );
            let sizes: Vec<f64> = v["group_statistics"]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| num(&g["n"]))
                .collect();
            Check::new(&case)
                .vector("kw_group_n", &sizes, Tol::exact())
                .finish();
            assert_eq!(
                num(&v["missing"]["n_used"]),
                case.vector("kw_group_n").iter().sum::<f64>(),
                "{tool}"
            );
            assert_eq!(
                v["missing"],
                summary(9, 6, serde_json::json!({"g9": 3})),
                "{tool}"
            );
        }

        // friedman.test / quade.test(y_na, treat, block): the block with the missing cell goes
        let (case, server, name) = case_server("stat_friedman_quade").await;
        for tool in ["friedman", "quade"] {
            let v = json_of(
                &hyp(
                    &server,
                    tool,
                    serde_json::json!({"dataset": name, "value": "y_na", "treatment": "treat",
                                       "block": "block"}),
                )
                .await,
            );
            Check::new(&case)
                .scalar("na_n_blocks", num(&v["n_blocks"]), Tol::exact())
                // the rows used: three treatments per kept block
                .scalar(
                    "na_n_blocks",
                    num(&v["missing"]["n_used"]) / 3.0,
                    Tol::exact(),
                )
                .finish();
            assert_eq!(
                v["missing"],
                summary(24, 21, serde_json::json!({"y_na": 1})),
                "{tool}"
            );
            let complete = json_of(
                &hyp(
                    &server,
                    tool,
                    serde_json::json!({"dataset": name, "value": "y", "treatment": "treat",
                                       "block": "block"}),
                )
                .await,
            );
            assert!(complete.get("missing").is_none(), "{tool}: {complete}");
        }

        // wilcox.test(x, y, paired = TRUE): the JSON follows the Display text
        let (case, server, name) = case_server("stat_wilcoxon_na").await;
        let t = ok_text(
            &hyp(
                &server,
                "wilcoxon",
                serde_json::json!({"dataset": name, "x": "x", "y": "y", "paired": true}),
            )
            .await,
        );
        let v: serde_json::Value = serde_json::from_str(&t[t.find("\n{").unwrap() + 1..]).unwrap();
        Check::new(&case)
            .scalar("n_pairs", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            summary(10, 8, serde_json::json!({"x": 1, "y": 1}))
        );

        // pairwise.t.test / pairwise.wilcox.test(y, g)
        let (case, server, name) = case_server("stat_pairwise_mahalanobis_na").await;
        for tool in ["pairwise_t_test", "pairwise_wilcox"] {
            let v = json_of(
                &hyp(
                    &server,
                    tool,
                    serde_json::json!({"dataset": name, "response": "y", "factor": "g"}),
                )
                .await,
            );
            Check::new(&case)
                .vector("pairwise_group_n", &nums(&v["group_sizes"]), Tol::exact())
                .finish();
            assert_eq!(
                v["missing"],
                summary(30, 28, serde_json::json!({"y": 2})),
                "{tool}"
            );
        }

        // shapiro.test(n3_124): 3 values of 5000 rows
        let (case, server, name) = case_server("stat_shapiro_wilk").await;
        let v = json_of(
            &hyp(
                &server,
                "shapiro_wilk",
                serde_json::json!({"dataset": name, "column": "n3_124"}),
            )
            .await,
        );
        Check::new(&case)
            .scalar("n3_124_n", num(&v["n"]), Tol::exact())
            .scalar("n3_124_n", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            summary(5000, 3, serde_json::json!({"n3_124": 4997}))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_mcp_hypothesis_prop_trend_test_na() {
        // TOOL-17/X-2: the missing success count of group 4 was read as 0 (X-squared 37.388 on
        // scores 1..7). R refuses the data; the tool drops the group and keeps the other
        // groups' scores: prop.trend.test(x[cc], n[cc], score = which(cc)).
        let (case, server, name) = case_server("stat_prop_trend_na").await;
        let call = |extra: serde_json::Value| {
            let mut args = serde_json::json!({"dataset": name, "successes": "successes",
                                              "trials": "trials"});
            if let serde_json::Value::Object(m) = extra {
                args.as_object_mut().unwrap().extend(m);
            }
            let server = &server;
            async move { hyp(server, "prop_trend_test", args).await }
        };
        let v = json_of(&call(serde_json::json!({})).await);
        Check::new(&case)
            .scalar("chisq", num(&v["statistic"]), Tol::stat(6))
            .scalar("df", num(&v["df"]), Tol::exact())
            .scalar("p", num(&v["p_value"]), Tol::p_value(6))
            .vector("scores", &nums(&v["scores"]), Tol::exact())
            .scalar("n_used", num(&v["missing"]["n_used"]), Tol::exact())
            .scalar("n_total", num(&v["missing"]["n_total"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            serde_json::json!({"n_total": 7, "n_used": 6, "n_dropped": 1,
                               "by_column": {"successes": 1}})
        );
        // user scores, one per dataset row; the dropped group's score goes with it
        let d = json_of(&call(serde_json::json!({"scores": [0, 1, 2, 4, 8, 16, 32]})).await);
        Check::new(&case)
            .scalar("dose_chisq", num(&d["statistic"]), Tol::stat(6))
            .scalar("dose_p", num(&d["p_value"]), Tol::p_value(6))
            .vector("dose_scores", &nums(&d["scores"]), Tol::exact())
            .finish();
        let e = err_text(&call(serde_json::json!({"scores": [0, 1, 2, 8, 16, 32]})).await);
        assert!(
            e.contains("scores must have one value per dataset row (7)"),
            "{e}"
        );
        // Int32 counts give the same result
        let mut df = server
            .datasets
            .read()
            .await
            .get(&name)
            .unwrap()
            .df()
            .clone();
        for c in ["successes", "trials"] {
            let cast = df.column(c).unwrap().cast(&DataType::Int32).unwrap();
            df.with_column(cast).unwrap();
        }
        server
            .datasets
            .write()
            .await
            .insert("prop_trend_i32".into(), p2a_core::data::Dataset::new(df));
        let i = json_of(
            &hyp(
                &server,
                "prop_trend_test",
                serde_json::json!({"dataset": "prop_trend_i32", "successes": "successes",
                                   "trials": "trials"}),
            )
            .await,
        );
        assert_eq!(i["statistic"], v["statistic"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mcp_hypothesis_prop_trend_test_refuses_non_counts() {
        // TOOL-17: a negative count wrapped to a huge usize (debug builds overflowed the sums)
        // and a fractional count was truncated. Both are refused with the column and the row.
        let df = p2a_core::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/generated/prop_trend_na.csv",
        ))
        .unwrap();
        let server = AnalyticsServer::new();
        server
            .datasets
            .write()
            .await
            .insert("d".into(), p2a_core::data::Dataset::new(df));
        let e = err_text(
            &hyp(
                &server,
                "prop_trend_test",
                serde_json::json!({"dataset": "d", "successes": "successes",
                                   "trials": "trials_neg"}),
            )
            .await,
        );
        assert!(
            e.contains(
                "Column 'trials_neg' must hold non-negative whole numbers (counts); row 5 is -1"
            ),
            "{e}"
        );
        let e = err_text(
            &hyp(
                &server,
                "prop_trend_test",
                serde_json::json!({"dataset": "d", "successes": "successes_frac",
                                   "trials": "trials"}),
            )
            .await,
        );
        assert!(
            e.contains(
                "Column 'successes_frac' must hold non-negative whole numbers (counts); row 2 is 5.5"
            ),
            "{e}"
        );
    }
}
