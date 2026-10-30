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
use crate::tools::common::{check_named_budget, missing_json};
use crate::tools::limits::{Family, echo};
use crate::tools::requests::hypothesis::*;
use p2a_core::stats::{
    Alternative, CmhAlternative, GofExpected, PValueAdjustMethod, PoissonAlternative, mcnemar_test,
    mood_test, one_sample_t_test, paired_t_test, poisson_test, run_bartlett_test,
    run_chisq_gof_with, run_chisq_independence, run_friedman_test, run_kruskal_test,
    run_mantelhaen_test, run_oneway_test, run_pairwise_t_test, run_pairwise_wilcox_test,
    run_quade_test, two_sample_t_test,
};
use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
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

        let output = serde_json::json!({
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
        use p2a_core::stats::cortest::run_cor_test;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let df = dataset.df();
        let x: Vec<f64> = match df.column(&request.x) {
            Ok(col) => match col.f64() {
                Ok(ca) => ca.into_no_null_iter().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' is not numeric: {}",
                        request.x, e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.x, e
                ))]));
            }
        };

        let y: Vec<f64> = match df.column(&request.y) {
            Ok(col) => match col.f64() {
                Ok(ca) => ca.into_no_null_iter().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' is not numeric: {}",
                        request.y, e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.y, e
                ))]));
            }
        };

        let method = request.method.as_deref().unwrap_or("pearson");
        let alternative = request.alternative.as_deref().unwrap_or("two.sided");
        let conf_level = request.conf_level.unwrap_or(0.95);

        match run_cor_test(&x, &y, method, alternative, conf_level) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "test": result.method_name,
                    "estimate": result.estimate,
                    "estimate_name": result.estimate_name,
                    "statistic": result.statistic,
                    "statistic_name": result.statistic_name,
                    "p_value": result.p_value,
                    "alternative": format!("{:?}", result.alternative),
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

        let alternative = match request.alternative.as_deref() {
            Some("greater") => FisherAlternative::Greater,
            Some("less") => FisherAlternative::Less,
            Some("two.sided") | None => FisherAlternative::TwoSided,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Invalid alternative '{}'. Must be 'two.sided', 'greater', or 'less'.",
                    other
                ))]));
            }
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
                if let Some(missing) = missing_json(&result.missing) {
                    json_output["missing"] = missing;
                }

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

        let output = serde_json::json!({
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

        let output = serde_json::json!({
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

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Kolmogorov-Smirnov test for comparing distributions.
    #[tool(
        description = "Run the Kolmogorov-Smirnov test. Two-sample test: Tests if two samples come from the same distribution. One-sample test: Tests if a sample comes from a specified theoretical distribution (normal, uniform, exponential). Returns D statistic (maximum absolute difference between CDFs) and p-value. A small p-value suggests the distributions differ."
    )]
    pub async fn hypothesis_ks_test(
        &self,
        Parameters(request): Parameters<KsTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::ks::{
            TheoreticalDistribution, ks_test_one_sample, ks_test_two_sample,
        };

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let alternative = match request.alternative.as_deref() {
            Some(alt) => match Alternative::from_str(alt) {
                Some(a) => a,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid alternative '{}'. Use 'two.sided', 'greater', or 'less'.",
                        alt
                    ))]));
                }
            },
            None => Alternative::TwoSided,
        };

        let df = dataset.df();

        let x_series = match df.column(&request.x) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found in dataset.",
                    request.x
                ))]));
            }
        };
        let x: Vec<f64> = match x_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' is not numeric.",
                    request.x
                ))]));
            }
        };

        let result = if let Some(y_col) = &request.y {
            let y_series = match df.column(y_col) {
                Ok(s) => s,
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found in dataset.",
                        y_col
                    ))]));
                }
            };
            let y: Vec<f64> = match y_series.f64() {
                Ok(ca) => ca.into_no_null_iter().collect(),
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' is not numeric.",
                        y_col
                    ))]));
                }
            };

            ks_test_two_sample(&x, &y, alternative)
        } else {
            let distribution = match request.distribution.as_deref() {
                Some("uniform") => {
                    let a = request.a.unwrap_or(0.0);
                    let b = request.b.unwrap_or(1.0);
                    TheoreticalDistribution::UniformParams { a, b }
                }
                Some("exponential") => {
                    let rate = request.rate.unwrap_or(1.0);
                    TheoreticalDistribution::Exponential { rate }
                }
                _ => {
                    let mean = request.mean.unwrap_or(0.0);
                    let sd = request.sd.unwrap_or(1.0);
                    if (mean - 0.0).abs() < 1e-10 && (sd - 1.0).abs() < 1e-10 {
                        TheoreticalDistribution::Normal
                    } else {
                        TheoreticalDistribution::NormalParams { mean, sd }
                    }
                }
            };

            ks_test_one_sample(&x, distribution, alternative)
        };

        match result {
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

                if let Some(n2) = result.n_2 {
                    json_output["n_2"] = serde_json::json!(n2);
                }

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

        let alternative = match request.alternative.as_deref() {
            Some("greater") => CmhAlternative::Greater,
            Some("less") => CmhAlternative::Less,
            _ => CmhAlternative::TwoSided,
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

        let stratum_stats: Vec<serde_json::Value> = result
            .stratum_stats
            .iter()
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
            "stratum_statistics": stratum_stats,
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

        let x_series = match dataset.df().column(&request.x) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found in dataset.",
                    request.x
                ))]));
            }
        };
        let x: Vec<f64> = match x_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' must be numeric.",
                    request.x
                ))]));
            }
        };

        let y_series = match dataset.df().column(&request.y) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found in dataset.",
                    request.y
                ))]));
            }
        };
        let y: Vec<f64> = match y_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' must be numeric.",
                    request.y
                ))]));
            }
        };

        let alternative = match request.alternative.as_deref() {
            Some("greater") | Some("gt") => Alternative::Greater,
            Some("less") | Some("lt") => Alternative::Less,
            _ => Alternative::TwoSided,
        };

        let result = match mood_test(&x, &y, alternative) {
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

        let output = serde_json::json!({
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

        let output = serde_json::json!({
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

        let p_adjust_method = match request.p_adjust_method.as_deref() {
            Some(m) => match PValueAdjustMethod::from_str(m) {
                Some(method) => method,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid p-value adjustment method '{}'. Use 'holm', 'bonferroni', 'hochberg', 'hommel', 'BH', 'BY', or 'none'.",
                        m
                    ))]));
                }
            },
            None => PValueAdjustMethod::Holm,
        };

        let alternative = match request.alternative.as_deref() {
            Some(alt) => match Alternative::from_str(alt) {
                Some(a) => a,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid alternative '{}'. Use 'two.sided', 'greater', or 'less'.",
                        alt
                    ))]));
                }
            },
            None => Alternative::TwoSided,
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

                let output = serde_json::json!({
                    "test": result.test_name,
                    "p_adjust_method": result.p_adjust_method.name(),
                    "pool_sd": result.pool_sd,
                    "alternative": format!("{:?}", result.alternative).to_lowercase(),
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

        let p_adjust_method = match request.p_adjust_method.as_deref() {
            Some(m) => match PValueAdjustMethod::from_str(m) {
                Some(method) => method,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid p-value adjustment method '{}'. Use 'holm', 'bonferroni', 'hochberg', 'hommel', 'BH', 'BY', or 'none'.",
                        m
                    ))]));
                }
            },
            None => PValueAdjustMethod::Holm,
        };

        let alternative = match request.alternative.as_deref() {
            Some(alt) => match Alternative::from_str(alt) {
                Some(a) => a,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid alternative '{}'. Use 'two.sided', 'greater', or 'less'.",
                        alt
                    ))]));
                }
            },
            None => Alternative::TwoSided,
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

                let output = serde_json::json!({
                    "test": result.test_name,
                    "p_adjust_method": result.p_adjust_method.name(),
                    "alternative": format!("{:?}", result.alternative).to_lowercase(),
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

        let alternative = match request.alternative.as_deref() {
            Some("greater") => PoissonAlternative::Greater,
            Some("less") => PoissonAlternative::Less,
            _ => PoissonAlternative::TwoSided,
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
        description = "Test for trend in proportions across ordered groups (Cochran-Armitage test). Tests whether proportions increase or decrease linearly with group scores. Commonly used in dose-response studies. Returns chi-squared statistic with 1 df and p-value."
    )]
    pub async fn hypothesis_prop_trend_test(
        &self,
        Parameters(request): Parameters<PropTrendTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::proptrendtest::run_prop_trend_test;

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let df = dataset.df();
        let successes: Vec<usize> = match df.column(&request.successes) {
            Ok(col) => {
                if let Ok(ca) = col.u64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else if let Ok(ca) = col.i64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else if let Ok(ca) = col.f64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' must be numeric (integers)",
                        request.successes
                    ))]));
                }
            }
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.successes, e
                ))]));
            }
        };

        let trials: Vec<usize> = match df.column(&request.trials) {
            Ok(col) => {
                if let Ok(ca) = col.u64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else if let Ok(ca) = col.i64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else if let Ok(ca) = col.f64() {
                    ca.into_no_null_iter().map(|v| v as usize).collect()
                } else {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' must be numeric (integers)",
                        request.trials
                    ))]));
                }
            }
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.trials, e
                ))]));
            }
        };

        match run_prop_trend_test(&successes, &trials, request.scores.clone()) {
            Ok(result) => {
                let json_output = serde_json::json!({
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

        let output = serde_json::json!({
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
                let json_output = serde_json::json!({
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
        description = "Run Student's t-test for comparing means. Supports: (1) One-sample t-test: compare sample mean to hypothesized value, (2) Two-sample t-test: pass x and y as two numeric columns to compare (Welch's by default), (3) Paired t-test: compare matched pairs. Returns t-statistic, p-value, confidence interval, and effect estimate. Note: x and y must both be numeric columns. To compare a numeric variable across groups (e.g., salary by gender), first filter into two datasets or use two numeric columns."
    )]
    pub async fn hypothesis_t_test(
        &self,
        Parameters(request): Parameters<TTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let x_series = match dataset.df().column(&request.x) {
            Ok(s) => s,
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found in dataset.",
                    request.x
                ))]));
            }
        };
        let x: Vec<f64> = match x_series.f64() {
            Ok(ca) => ca.into_no_null_iter().collect(),
            Err(_) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' must be numeric.",
                    request.x
                ))]));
            }
        };

        let mu = request.mu.unwrap_or(0.0);
        let alternative = match request.alternative.as_deref() {
            Some("greater") | Some("gt") => Alternative::Greater,
            Some("less") | Some("lt") => Alternative::Less,
            _ => Alternative::TwoSided,
        };
        let paired = request.paired.unwrap_or(false);
        let var_equal = request.var_equal.unwrap_or(false);
        let conf_level = request.conf_level.unwrap_or(0.95);

        let result = match &request.y {
            Some(y_col) => {
                let y_series = match dataset.df().column(y_col) {
                    Ok(s) => s,
                    Err(_) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' not found in dataset.",
                            y_col
                        ))]));
                    }
                };
                let y: Vec<f64> = match y_series.f64() {
                    Ok(ca) => ca.into_no_null_iter().collect(),
                    Err(_) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' must be numeric.",
                            y_col
                        ))]));
                    }
                };

                if paired {
                    paired_t_test(&x, &y, mu, alternative, conf_level)
                } else {
                    two_sample_t_test(&x, &y, mu, alternative, var_equal, conf_level)
                }
            }
            None => {
                if paired {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "Paired t-test requires both x and y columns.".to_string(),
                    )]));
                }
                one_sample_t_test(&x, mu, alternative, conf_level)
            }
        };

        match result {
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

        let alternative = match request.alternative.as_deref() {
            Some(alt) => match Alternative::from_str(alt) {
                Some(a) => a,
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid alternative '{}'. Use 'two.sided', 'greater', or 'less'.",
                        alt
                    ))]));
                }
            },
            None => Alternative::TwoSided,
        };

        let mu = request.mu.unwrap_or(0.0);
        let paired = request.paired.unwrap_or(false);
        let config = WilcoxonConfig {
            exact: request.exact,
            correct: request.correct.unwrap_or(true),
            conf_int: request.conf_int.unwrap_or(false),
            conf_level: request.conf_level.unwrap_or(0.95),
        };

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
                    "alternative": format!("{:?}", result.alternative).to_lowercase(),
                    "null_value": result.null_value,
                    "exact": result.exact,
                    "continuity_correction": result.continuity_correction,
                    "n1": result.n,
                });

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
    let Ok(col) = dataset.df().column(factor) else {
        return Ok(());
    };
    let n_unique = level_count_or_refusal(factor, col.n_unique())
        .map_err(|msg| format!("Pairwise {label} refused: {msg}"))?;
    let k = n_unique.saturating_sub(usize::from(col.null_count() > 0)) as u64;
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

/// The distinct values of column `name` for a size budget, null counted as one: an upper bound
/// on the levels the core keeps after dropping missing values. `Ok(None)` when the column is
/// missing, which the core reports itself.
fn budget_level_count(
    df: &p2a_core::polars::prelude::DataFrame,
    name: &str,
) -> Result<Option<usize>, String> {
    let Ok(col) = df.column(name) else {
        return Ok(None);
    };
    level_count_or_refusal(name, col.n_unique()).map(Some)
}

/// A distinct-value count a budget depends on, or the refusal when it could not be taken (the
/// budget cannot be checked, so the request must not run unbounded).
fn level_count_or_refusal(
    name: &str,
    count: p2a_core::polars::prelude::PolarsResult<usize>,
) -> Result<usize, String> {
    count.map_err(|e| {
        format!(
            "the distinct values of `{}` could not be counted to check this server's limits \
             ({e}).",
            echo(name)
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
    use crate::tools::common::test_support::{hosted_with, server_with, text};
    use crate::tools::limits::{self, Limits};
    use p2a_core::polars::prelude::*;
    use p2a_validation::{Case, Check, Tol};

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
        // decision 8: 200,000 categories x (3 numbers + a label) exceed the hosted 16 MiB
        // response budget; the statistic is still returned (before: the full arrays, which
        // the governor would replace by an error).
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
        assert_eq!(v["odds_ratio_ci"]["lower"], 0.0);
        assert_eq!(v["stratum_statistics"][0]["odds_ratio"], "Inf");
        assert_eq!(v["stratum_statistics"][1]["odds_ratio"], "Inf");
        assert_eq!(v["stratum_statistics"][0]["counts"]["a"], 3.0);
        assert!(v["statistic"].as_f64().is_some_and(f64::is_finite), "{v}");
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
}
