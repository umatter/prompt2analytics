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
use crate::tools::requests::hypothesis::*;
use p2a_core::stats::{
    run_bartlett_test, run_chisq_gof, run_chisq_independence, run_friedman_test,
    run_kruskal_test, run_mantelhaen_test, CmhAlternative, mcnemar_test, mood_test,
    run_oneway_test, run_pairwise_t_test, PValueAdjustMethod, run_pairwise_wilcox_test,
    poisson_test, PoissonAlternative, run_quade_test, one_sample_t_test, paired_t_test,
    two_sample_t_test, Alternative,
};
use rmcp::{
    handler::server::wrapper::Parameters, model::*, tool, tool_router, ErrorData as McpError,
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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
        description = "Run Pearson's chi-squared goodness-of-fit test to check if observed category frequencies match expected probabilities. Tests H₀: observed frequencies follow the expected distribution. Returns chi-squared statistic, p-value, degrees of freedom, and residuals. Use for categorical data to test if a distribution is uniform or matches specific probabilities."
    )]
    pub async fn hypothesis_chisq_gof(
        &self,
        Parameters(request): Parameters<ChiSquaredGofRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let probs: Option<Vec<f64>> = request.probs;
        let probs_ref: Option<&[f64]> = probs.as_deref();

        match run_chisq_gof(dataset, &request.column, probs_ref) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "test": result.test_name,
                    "statistic": result.statistic,
                    "df": result.df,
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "observed": result.observed,
                    "expected": result.expected,
                    "residuals": result.residuals,
                    "n_categories": result.observed.len(),
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
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
        description = "Run Pearson's chi-squared test of independence to check if two categorical variables are independent. Creates a contingency table from two columns and tests H₀: row and column variables are independent. For 2×2 tables, Yates' continuity correction is applied by default. Returns chi-squared statistic, p-value, degrees of freedom, expected values, and residuals."
    )]
    pub async fn hypothesis_chisq_independence(
        &self,
        Parameters(request): Parameters<ChiSquaredIndependenceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let correct = request.correct.unwrap_or(true);

        match run_chisq_independence(dataset, &request.row_var, &request.col_var, correct) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "test": result.test_name,
                    "statistic": result.statistic,
                    "df": result.df,
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "n_rows": result.n_rows,
                    "n_cols": result.n_cols,
                    "observed": result.observed,
                    "expected": result.expected,
                    "residuals": result.residuals,
                    "std_residuals": result.std_residuals,
                    "yates_correction": result.yates_correction,
                });
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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
        description = "Run Fisher's exact test for a 2×2 contingency table. Tests independence between two binary categorical variables using exact probability calculations (hypergeometric distribution). More accurate than chi-squared test for small samples. Returns p-value, odds ratio, and optionally a confidence interval. Use when expected cell counts are small (<5) or when exact p-values are needed."
    )]
    pub async fn hypothesis_fisher_exact(
        &self,
        Parameters(request): Parameters<FisherExactRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::fisher::{run_fisher_test, FisherAlternative};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
                    "p_value": result.p_value,
                    "significance": result.significance.to_string(),
                    "alternative": result.alternative.to_string(),
                    "odds_ratio": result.odds_ratio,
                    "table": {
                        "a": result.table[0],
                        "b": result.table[1],
                        "c": result.table[2],
                        "d": result.table[3],
                    },
                    "row_totals": result.row_totals,
                    "col_totals": result.col_totals,
                    "n": result.n,
                    "prob_observed": result.prob_observed,
                });

                if let (Some((lo, hi)), Some(level)) = (result.odds_ratio_ci, result.conf_level) {
                    json_output["odds_ratio_ci"] = serde_json::json!({
                        "lower": lo,
                        "upper": hi,
                        "conf_level": level,
                    });
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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
        use p2a_core::stats::ks::{ks_test_one_sample, ks_test_two_sample, TheoreticalDistribution};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
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
        description = "Run Cochran-Mantel-Haenszel test for conditional independence in stratified 2×2 tables. Tests whether two binary variables are associated while controlling for a third (stratum) variable. Returns CMH chi-squared statistic, p-value, common odds ratio estimate with confidence interval, and per-stratum statistics."
    )]
    pub async fn hypothesis_mantelhaen(
        &self,
        Parameters(request): Parameters<MantelhaenTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
                        "a": s.counts[0],
                        "b": s.counts[1],
                        "c": s.counts[2],
                        "d": s.counts[3]
                    },
                    "n": s.n,
                    "expected_a": s.expected_a,
                    "variance_a": s.variance_a,
                    "odds_ratio": s.odds_ratio
                })
            })
            .collect();

        let alt_desc = match result.alternative {
            CmhAlternative::TwoSided => "true common odds ratio is not equal to 1",
            CmhAlternative::Greater => "true common odds ratio is greater than 1",
            CmhAlternative::Less => "true common odds ratio is less than 1",
        };

        let output = serde_json::json!({
            "method": result.test_name,
            "statistic": result.statistic,
            "statistic_name": "Mantel-Haenszel X-squared",
            "df": result.df,
            "p_value": result.p_value,
            "n_strata": result.n_strata,
            "total_n": result.total_n,
            "common_odds_ratio": result.common_odds_ratio,
            "odds_ratio_ci": {
                "lower": result.odds_ratio_ci.0,
                "upper": result.odds_ratio_ci.1,
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
            }
        });

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        match run_pairwise_t_test(
            dataset,
            &request.response,
            &request.factor,
            pool_sd,
            alternative,
            p_adjust_method,
        ) {
            Ok(result) => {
                let k = result.group_names.len();
                let mut p_matrix: Vec<Vec<Option<f64>>> = vec![vec![None; k]; k];
                for i in 1..k {
                    for j in 0..i {
                        p_matrix[i][j] = Some(result.p_values_adj[result.index(i, j)]);
                    }
                }

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        match run_pairwise_wilcox_test(
            dataset,
            &request.response,
            &request.factor,
            alternative,
            p_adjust_method,
            request.exact,
        ) {
            Ok(result) => {
                let k = result.group_names.len();
                let mut p_matrix: Vec<Vec<Option<f64>>> = vec![vec![None; k]; k];
                for i in 1..k {
                    for j in 0..i {
                        p_matrix[i][j] = Some(result.p_values_adj[result.index(i, j)]);
                    }
                }

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
        description = "Run Student's t-test for comparing means. Supports: (1) One-sample t-test: compare sample mean to hypothesized value, (2) Two-sample t-test: compare means between two groups (Welch's by default), (3) Paired t-test: compare matched pairs. Returns t-statistic, p-value, confidence interval, and effect estimate."
    )]
    pub async fn hypothesis_t_test(
        &self,
        Parameters(request): Parameters<TTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

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
        use p2a_core::stats::wilcoxon::{wilcoxon_test, WilcoxonConfig};

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
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
