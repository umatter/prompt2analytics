//! Statistical tools handlers.
//!
//! This module provides MCP tool handlers for statistical analysis:
//! - Log-linear models
//! - ANOVA model tables and contrasts
//! - Weighted statistics
//! - Sphericity tests
//! - Robust statistics (fivenum, IQR, MAD, ECDF, density)
//! - Spline/interpolation tools
//! - MANOVA
//! - Factor analysis
//! - Power analysis
//! - Canonical correlation
//! - Mahalanobis distance
//! - Tukey HSD

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

/// Maximum number of elements allowed in a single matrix allocation.
const MAX_MATRIX_ELEMENTS: usize = 10_000_000;

use crate::server::AnalyticsServer;
use crate::tools::common::{extract_column_f64, extract_numeric_matrix};
use crate::tools::requests::stats::{
    ApproxRequest, CancorRequest, CorrelationRequest, CovWtRequest, DensityRequest, EcdfRequest,
    FactorAnalysisRequest, FivenumRequest, IqrRequest, IsoregRequest, LoglinRequest, MadRequest,
    MahalanobisRequest, ManovaRequest, MauchlyTestRequest, MedpolishRequest, ModelTablesRequest,
    OneWayAnovaRequest, PowerAnovaTestRequest, PowerPropTestRequest, PowerTTestRequest,
    SeContrastRequest, SplineRequest, TukeyHsdRequest, TwoWayAnovaRequest, WeightedMeanRequest,
};

use p2a_core::stats::{
    ApproxMethod, ApproxRule, ContrastType, SplineMethod, TableType, approx, correlation_matrix,
    fivenum, generate_contrasts, iqr, isoreg, loglin, model_tables, run_cov_wt, run_density,
    run_ecdf, run_mad, run_mahalanobis, run_manova, run_mauchly_test, run_medpolish,
    run_one_way_anova, run_tukey_hsd, run_two_way_anova, se_contrast, spline, weighted_mean,
};

#[tool_router(router = stats_router, vis = "pub")]
impl AnalyticsServer {
    /// Log-linear model fitting for contingency tables.
    #[tool(
        description = "Fit log-linear models to multi-way contingency tables using iterative proportional fitting (IPF). Tests association and independence patterns. Equivalent to R's loglin()."
    )]
    pub async fn stats_loglin(
        &self,
        Parameters(request): Parameters<LoglinRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;
        use std::collections::HashMap;

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

        // Extract count column
        let count_col = match df.column(&request.count_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Count column '{}' not found: {}",
                    request.count_column, e
                ))]));
            }
        };

        let counts: Vec<f64> = match count_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_no_null_iter().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Count column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot cast count column: {}",
                    e
                ))]));
            }
        };

        // Extract factor columns and determine dimensions
        let mut factor_levels: Vec<Vec<String>> = Vec::new();
        let mut factor_data: Vec<Vec<usize>> = Vec::new();

        for col_name in &request.factor_columns {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Factor column '{}' not found: {}",
                        col_name, e
                    ))]));
                }
            };

            // Get unique levels by casting to string series
            let str_col = match col.cast(&DataType::String) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot cast factor column '{}' to string: {}",
                        col_name, e
                    ))]));
                }
            };

            let unique_col = match str_col.unique() {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Error getting unique values for '{}': {}",
                        col_name, e
                    ))]));
                }
            };

            let levels: Vec<String> = match unique_col.str() {
                Ok(s) => s.into_no_null_iter().map(|v| v.to_string()).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Error reading unique levels: {}",
                        e
                    ))]));
                }
            };

            // Map values to indices
            let level_map: HashMap<String, usize> = levels
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i))
                .collect();

            // Get string values and map to indices
            let str_series = match str_col.str() {
                Ok(s) => s,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot read string series: {}",
                        e
                    ))]));
                }
            };

            let indices: Vec<usize> = str_series
                .into_no_null_iter()
                .map(|v| *level_map.get(v).unwrap_or(&0))
                .collect();

            factor_levels.push(levels);
            factor_data.push(indices);
        }

        // Build dimensions
        let dimensions: Vec<usize> = factor_levels.iter().map(|l| l.len()).collect();

        // Build flattened contingency table
        let n_cells: usize = dimensions.iter().product();
        let mut table = vec![0.0; n_cells];

        for (i, &count) in counts.iter().enumerate() {
            // Compute cell index from factor indices
            let mut cell_idx = 0;
            let mut multiplier = 1;
            for dim_idx in (0..factor_data.len()).rev() {
                cell_idx += factor_data[dim_idx][i] * multiplier;
                multiplier *= dimensions[dim_idx];
            }
            if cell_idx < n_cells {
                table[cell_idx] += count;
            }
        }

        // Determine margins
        let margins = if let Some(m) = request.margins {
            m
        } else {
            // Default: independence model (main effects only)
            (0..dimensions.len()).map(|i| vec![i]).collect()
        };

        let result = match loglin(&table, &dimensions, &margins, request.eps, request.max_iter) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Log-linear model fitting failed: {}",
                    e
                ))]));
            }
        };

        // Build JSON output
        let json_output = serde_json::json!({
            "model": {
                "margins": result.margins,
                "factor_columns": request.factor_columns,
                "dimensions": result.dimensions,
                "factor_levels": factor_levels,
                "n_cells": result.n_cells,
                "total_count": result.total
            },
            "goodness_of_fit": {
                "likelihood_ratio_test": {
                    "statistic": format!("{:.4}", result.lrt),
                    "df": result.df,
                    "p_value": format!("{:.4}", result.p_value_lrt)
                },
                "pearson_chi_squared": {
                    "statistic": format!("{:.4}", result.pearson),
                    "df": result.df,
                    "p_value": format!("{:.4}", result.p_value_pearson)
                }
            },
            "convergence": {
                "converged": result.converged,
                "iterations": result.n_iter
            },
            "interpretation": format!(
                "Log-linear model with margins {:?} on a {} table. G²={:.2}, df={}, p={:.4}. {}",
                result.margins,
                result.dimensions.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×"),
                result.lrt, result.df, result.p_value_lrt,
                if result.p_value_lrt >= 0.05 {
                    "The model fits adequately (p≥0.05)."
                } else {
                    "The model does not fit well (p<0.05); consider adding interaction terms."
                }
            ),
            "references": "Haberman (1972). Algorithm AS 51; Agresti (2002). Categorical Data Analysis."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Model tables (means/effects) from ANOVA.
    #[tool(
        description = "Compute cell means or effects tables from one-way or two-way ANOVA. Returns group means or deviations from grand mean with optional standard errors. Equivalent to R's model.tables()."
    )]
    pub async fn stats_model_tables(
        &self,
        Parameters(request): Parameters<ModelTablesRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

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

        let table_type = match request.table_type.as_deref() {
            Some("effects") => TableType::Effects,
            Some("means") | None => TableType::Means,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown table type '{}'. Use 'means' or 'effects'.",
                    other
                ))]));
            }
        };

        let compute_se = request.se.unwrap_or(true);

        if request.factors.len() == 1 {
            // One-way ANOVA
            let anova = match run_one_way_anova(dataset, &request.response, &request.factors[0]) {
                Ok(a) => a,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "One-way ANOVA failed: {}",
                        e
                    ))]));
                }
            };

            let result = match model_tables(&anova, table_type, compute_se) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Model tables failed: {}",
                        e
                    ))]));
                }
            };

            let json_output = serde_json::json!({
                "table_type": result.table_type,
                "grand_mean": result.grand_mean,
                "one_way_table": {
                    "factor": &request.factors[0],
                    "levels": result.group_names,
                    "values": result.values,
                    "se": result.se,
                    "n": result.n
                },
                "mse": result.mse,
                "df_mse": result.df_mse
            });

            Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&json_output).unwrap(),
            )]))
        } else if request.factors.len() == 2 {
            // Two-way ANOVA
            let anova = match run_two_way_anova(
                dataset,
                &request.response,
                &request.factors[0],
                &request.factors[1],
                true, // include interaction term
            ) {
                Ok(a) => a,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Two-way ANOVA failed: {}",
                        e
                    ))]));
                }
            };

            let df = dataset.df();

            // Extract response and factors
            let y: Vec<f64> = match df.column(&request.response) {
                Ok(c) => match c.cast(&DataType::Float64) {
                    Ok(c) => c.f64().unwrap().into_no_null_iter().collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Response not numeric: {}",
                            e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Response column not found: {}",
                        e
                    ))]));
                }
            };

            let get_factor = |name: &str| -> Result<Vec<String>, CallToolResult> {
                match df.column(name) {
                    Ok(c) => match c.cast(&DataType::String) {
                        Ok(c) => Ok(c
                            .str()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|s| s.to_string())
                            .collect()),
                        Err(e) => Err(CallToolResult::error(vec![Content::text(format!(
                            "Cannot read factor '{}': {}",
                            name, e
                        ))])),
                    },
                    Err(e) => Err(CallToolResult::error(vec![Content::text(format!(
                        "Factor column '{}' not found: {}",
                        name, e
                    ))])),
                }
            };

            let factor_a = match get_factor(&request.factors[0]) {
                Ok(f) => f,
                Err(e) => return Ok(e),
            };

            let factor_b = match get_factor(&request.factors[1]) {
                Ok(f) => f,
                Err(e) => return Ok(e),
            };

            // Compute grand mean
            let grand_mean = y.iter().sum::<f64>() / y.len() as f64;

            // Get unique levels
            let levels_a: Vec<String> = {
                let mut v: Vec<_> = factor_a.to_vec();
                v.sort();
                v.dedup();
                v
            };

            let levels_b: Vec<String> = {
                let mut v: Vec<_> = factor_b.to_vec();
                v.sort();
                v.dedup();
                v
            };

            // Compute cell means
            let mut cell_means: Vec<Vec<f64>> = vec![vec![0.0; levels_b.len()]; levels_a.len()];
            let mut cell_counts: Vec<Vec<usize>> = vec![vec![0; levels_b.len()]; levels_a.len()];

            for (i, yi) in y.iter().enumerate() {
                let a_idx = levels_a.iter().position(|x| x == &factor_a[i]).unwrap();
                let b_idx = levels_b.iter().position(|x| x == &factor_b[i]).unwrap();
                cell_means[a_idx][b_idx] += yi;
                cell_counts[a_idx][b_idx] += 1;
            }

            for i in 0..levels_a.len() {
                for j in 0..levels_b.len() {
                    if cell_counts[i][j] > 0 {
                        cell_means[i][j] /= cell_counts[i][j] as f64;
                    }
                }
            }

            // Convert to effects if requested
            let values = if table_type == TableType::Effects {
                cell_means
                    .iter()
                    .map(|row| row.iter().map(|&m| m - grand_mean).collect())
                    .collect()
            } else {
                cell_means
            };

            let json_output = serde_json::json!({
                "table_type": format!("{:?}", table_type),
                "grand_mean": grand_mean,
                "two_way_table": {
                    "factor_a": &request.factors[0],
                    "factor_b": &request.factors[1],
                    "levels_a": levels_a,
                    "levels_b": levels_b,
                    "values": values,
                    "counts": cell_counts
                },
                "anova_summary": {
                    "ss_a": anova.ss_a,
                    "ss_b": anova.ss_b,
                    "ss_ab": anova.ss_ab,
                    "ss_residual": anova.ss_error
                }
            });

            Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&json_output).unwrap(),
            )]))
        } else {
            Ok(CallToolResult::error(vec![Content::text(
                "Please provide 1 factor for one-way or 2 factors for two-way ANOVA.",
            )]))
        }
    }

    /// Standard errors of contrasts.
    #[tool(
        description = "Compute standard errors for linear contrasts of group means from one-way ANOVA. Can use custom contrasts or generate standard ones (treatment, Helmert, sum, polynomial). Equivalent to R's se.contrast()."
    )]
    pub async fn stats_se_contrast(
        &self,
        Parameters(request): Parameters<SeContrastRequest>,
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

        // Run one-way ANOVA first
        let anova = match run_one_way_anova(dataset, &request.response, &request.factor) {
            Ok(a) => a,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "One-way ANOVA failed: {}",
                    e
                ))]));
            }
        };

        let k = anova.groups.len();

        // Get contrasts - either from request or generate
        let contrasts = if let Some(ref c) = request.contrasts {
            c.clone()
        } else {
            let contrast_type = match request.contrast_type.as_deref() {
                Some("treatment") | None => ContrastType::Treatment,
                Some("helmert") => ContrastType::Helmert,
                Some("sum") => ContrastType::Sum,
                Some("poly") | Some("polynomial") => ContrastType::Poly,
                Some(other) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Unknown contrast type '{}'. Use 'treatment', 'helmert', 'sum', or 'poly'.",
                        other
                    ))]));
                }
            };
            generate_contrasts(k, contrast_type)
        };

        let result = match se_contrast(&anova, &contrasts) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Contrast SE computation failed: {}",
                    e
                ))]));
            }
        };

        // Extract group info from AnovaResult
        let group_names: Vec<&str> = anova.groups.iter().map(|g| g.group.as_str()).collect();
        let group_means: Vec<f64> = anova.groups.iter().map(|g| g.mean).collect();
        let group_sizes: Vec<usize> = anova.groups.iter().map(|g| g.n).collect();

        let json_output = serde_json::json!({
            "groups": group_names,
            "group_means": group_means,
            "group_sizes": group_sizes,
            "contrasts": contrasts,
            "standard_errors": result.se,
            "mse": result.mse,
            "df_mse": result.df_mse
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Weighted mean.
    #[tool(
        description = "Compute the weighted arithmetic mean of a numeric column. Equivalent to R's weighted.mean()."
    )]
    pub async fn stats_weighted_mean(
        &self,
        Parameters(request): Parameters<WeightedMeanRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        let values: Vec<f64> = match df.column(&request.column) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c.f64().unwrap().into_iter().flatten().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column not found: {}",
                    e
                ))]));
            }
        };

        let weights: Vec<f64> = match df.column(&request.weights) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c.f64().unwrap().into_iter().flatten().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Weights not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weights column not found: {}",
                    e
                ))]));
            }
        };

        let na_rm = request.na_rm.unwrap_or(true);
        let result = match weighted_mean(&values, &weights, na_rm) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted mean failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "weighted_mean": result,
            "na_rm": na_rm,
            "n": values.len(),
            "sum_weights": weights.iter().sum::<f64>()
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Weighted covariance matrix.
    #[tool(
        description = "Compute the weighted covariance matrix for a set of numeric columns. Can use unbiased (reliability) or ML weighting. Equivalent to R's cov.wt()."
    )]
    pub async fn stats_cov_wt(
        &self,
        Parameters(request): Parameters<CovWtRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        use p2a_core::polars::prelude::*;
        let df = dataset.df();
        let weights: Vec<f64> = match df.column(&request.weights) {
            Ok(c) => match c.cast(&DataType::Float64) {
                Ok(c) => c.f64().unwrap().into_iter().flatten().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Weights not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weights column not found: {}",
                    e
                ))]));
            }
        };

        let method = request.method.as_deref();
        let compute_cor = request.center.unwrap_or(true); // Use center flag to also compute correlation

        // Flatten matrix data to row-major slice
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let flat_data: Vec<f64> = data.iter().copied().collect();

        let result = match run_cov_wt(
            &flat_data,
            n_rows,
            n_cols,
            Some(&weights),
            compute_cor,
            method,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted covariance failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "covariance_matrix": result.cov,
            "center": result.center,
            "n_obs": result.n_obs,
            "weights": result.wt,
            "correlation_matrix": result.cor,
            "columns": request.columns,
            "method": method.unwrap_or("unbiased")
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Mauchly's sphericity test.
    #[tool(
        description = "Mauchly's test for sphericity in repeated measures designs. Tests whether the variances of the differences between all combinations of conditions are equal. Includes epsilon corrections (Greenhouse-Geisser, Huynh-Feldt). Equivalent to R's mauchly.test()."
    )]
    pub async fn stats_mauchly_test(
        &self,
        Parameters(request): Parameters<MauchlyTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Flatten to row-major and call run_mauchly_test
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let flat_data: Vec<f64> = data.iter().copied().collect();

        let result = match run_mauchly_test(&flat_data, n_rows, n_cols) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mauchly's test failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "w_statistic": result.w,
            "chi_squared": result.chi_squared,
            "p_value": result.p_value,
            "df": result.df,
            "sphericity_violated": result.p_value < 0.05,
            "epsilon_corrections": {
                "greenhouse_geisser": result.epsilon_gg,
                "huynh_feldt": result.epsilon_hf,
                "lower_bound": result.epsilon_lb
            },
            "n_subjects": result.n,
            "n_conditions": result.p_levels
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap(),
        )]))
    }

    /// Tukey's five-number summary.
    #[tool(
        description = "Compute Tukey's five-number summary: minimum, lower-hinge (Q1), median, upper-hinge (Q3), and maximum. Equivalent to R's fivenum()."
    )]
    pub async fn stats_fivenum(
        &self,
        Parameters(request): Parameters<FivenumRequest>,
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

        let values = match extract_column_f64(dataset, &request.column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        match fivenum(&values) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "minimum": result.minimum,
                    "lower_hinge": result.lower_hinge,
                    "median": result.median,
                    "upper_hinge": result.upper_hinge,
                    "maximum": result.maximum,
                    "n": result.n,
                    "iqr": result.upper_hinge - result.lower_hinge
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Five-number summary failed: {}",
                e
            ))])),
        }
    }

    /// Interquartile range.
    #[tool(
        description = "Compute the interquartile range (Q3 - Q1) of a numeric column. Supports different quantile types (1-9). Equivalent to R's IQR()."
    )]
    pub async fn stats_iqr(
        &self,
        Parameters(request): Parameters<IqrRequest>,
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

        let values = match extract_column_f64(dataset, &request.column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        match iqr(&values, request.qtype) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "iqr": result,
                    "quantile_type": request.qtype.unwrap_or(7)
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "IQR computation failed: {}",
                e
            ))])),
        }
    }

    /// Median absolute deviation.
    #[tool(
        description = "Compute the median absolute deviation (MAD), a robust measure of dispersion. By default, scaled by 1.4826 for consistency with the standard deviation of normal distributions. Equivalent to R's mad()."
    )]
    pub async fn stats_mad(
        &self,
        Parameters(request): Parameters<MadRequest>,
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

        let values = match extract_column_f64(dataset, &request.column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // If center is "mean", compute mean; otherwise use None (which defaults to median in mad())
        let center = if request.center.as_deref() == Some("mean") {
            let sum: f64 = values.iter().filter(|x| !x.is_nan()).sum();
            let count = values.iter().filter(|x| !x.is_nan()).count();
            if count > 0 {
                Some(sum / count as f64)
            } else {
                None
            }
        } else {
            None // Will use median by default
        };

        let constant = request.constant;

        match run_mad(&values, center, constant) {
            Ok(mad_value) => {
                let json_output = serde_json::json!({
                    "mad": mad_value,
                    "center_used": center,
                    "constant": constant.unwrap_or(1.4826),
                    "n": values.len()
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "MAD computation failed: {}",
                e
            ))])),
        }
    }

    /// Empirical cumulative distribution function.
    #[tool(
        description = "Compute the empirical cumulative distribution function (ECDF) for a numeric column. Returns step function values at data points or specified evaluation points. Equivalent to R's ecdf()."
    )]
    pub async fn stats_ecdf(
        &self,
        Parameters(request): Parameters<EcdfRequest>,
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

        let values = match extract_column_f64(dataset, &request.column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        match run_ecdf(&values) {
            Ok(result) => {
                // If specific evaluation points requested, use those
                let (x_out, y_out) = if let Some(ref at) = request.at {
                    let y_eval = result.evaluate_many(at);
                    (at.clone(), y_eval)
                } else {
                    (result.x.clone(), result.y.clone())
                };

                let json_output = serde_json::json!({
                    "x": x_out,
                    "y": y_out,
                    "n": result.n,
                    "n_unique": result.x.len()
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "ECDF computation failed: {}",
                e
            ))])),
        }
    }

    /// Kernel density estimation.
    #[tool(
        description = "Estimate the probability density function using kernel density estimation. Supports multiple kernel functions and automatic bandwidth selection. Equivalent to R's density()."
    )]
    pub async fn stats_density(
        &self,
        Parameters(request): Parameters<DensityRequest>,
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

        let values = match extract_column_f64(dataset, &request.column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let kernel = request.kernel.as_deref().unwrap_or("gaussian");

        // Validate kernel
        if ![
            "gaussian",
            "normal",
            "epanechnikov",
            "rectangular",
            "uniform",
            "triangular",
            "biweight",
            "quartic",
            "cosine",
        ]
        .contains(&kernel)
        {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Unknown kernel '{}'. Supported: gaussian, epanechnikov, rectangular, triangular, biweight, cosine.",
                kernel
            ))]));
        }

        let n = request.n.unwrap_or(512);

        match run_density(&values, request.bw, kernel, Some(n)) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "bw": result.bw,
                    "kernel": result.kernel,
                    "n": result.n,
                    "data_n": values.len(),
                    "range": [result.x.first(), result.x.last()]
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Density estimation failed: {}",
                e
            ))])),
        }
    }

    /// Cubic spline interpolation.
    #[tool(
        description = "Fit a cubic spline through data points for smooth interpolation. Supports natural, FMM (Forsythe-Malcolm-Moler), periodic, and monotone (Hyman) splines. Equivalent to R's spline()."
    )]
    pub async fn stats_spline(
        &self,
        Parameters(request): Parameters<SplineRequest>,
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

        let x = match extract_column_f64(dataset, &request.x) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let y = match extract_column_f64(dataset, &request.y) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let method = match request.method.as_deref().unwrap_or("fmm") {
            "fmm" => SplineMethod::Fmm,
            "natural" => SplineMethod::Natural,
            "periodic" => SplineMethod::Periodic,
            "hyman" | "monotone" | "monotonefc" => SplineMethod::MonotoneFC,
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown spline method '{}'. Supported: fmm, natural, periodic, hyman.",
                    other
                ))]));
            }
        };

        match spline(&x, &y, request.xout.as_deref(), None, method) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "method": format!("{:?}", method),
                    "n_knots": x.len()
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Spline interpolation failed: {}",
                e
            ))])),
        }
    }

    /// Linear/constant approximation.
    #[tool(
        description = "Linear or constant (step function) interpolation between data points. Handles points outside the data range with NA or nearest value. Equivalent to R's approx()."
    )]
    pub async fn stats_approx(
        &self,
        Parameters(request): Parameters<ApproxRequest>,
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

        let x = match extract_column_f64(dataset, &request.x) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let y = match extract_column_f64(dataset, &request.y) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let method = match request.method.as_deref().unwrap_or("linear") {
            "linear" => ApproxMethod::Linear,
            "constant" | "step" => ApproxMethod::Constant,
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown approx method '{}'. Supported: linear, constant.",
                    other
                ))]));
            }
        };

        let rule = match request.rule.as_deref().unwrap_or("na") {
            "na" | "NA" => ApproxRule::Na,
            "nearest" | "boundary" | "const" => ApproxRule::Nearest,
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown rule '{}'. Supported: na, nearest.",
                    other
                ))]));
            }
        };

        // f parameter: for constant interpolation, 0 = left value, 1 = right value, 0.5 = average
        let f = 0.0; // Default: use left value for step interpolation

        match approx(&x, &y, Some(&request.xout), None, method, rule, f) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "method": format!("{:?}", method),
                    "rule": format!("{:?}", rule),
                    "n_interpolated": result.x.len()
                });
                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Approximation failed: {}",
                e
            ))])),
        }
    }

    /// Run MANOVA (Multivariate Analysis of Variance).
    #[tool(
        description = "Run Multivariate Analysis of Variance (MANOVA) to test whether group means differ across multiple response variables simultaneously. Returns four test statistics: Wilks' Lambda (most popular), Pillai's Trace (most robust, default), Hotelling-Lawley Trace, and Roy's Largest Root. Each with approximate F-statistic and p-value. Use when you have 2+ continuous response variables and want to test group differences while accounting for correlations between responses."
    )]
    pub async fn anova_manova(
        &self,
        Parameters(request): Parameters<ManovaRequest>,
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

        if request.response_vars.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "MANOVA requires at least 2 response variables.".to_string(),
            )]));
        }

        let response_refs: Vec<&str> = request.response_vars.iter().map(|s| s.as_str()).collect();

        let result = match run_manova(dataset, &response_refs, &request.factor) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "MANOVA failed: {}",
                    e
                ))]));
            }
        };

        let output = serde_json::json!({
            "n_obs": result.n_obs,
            "n_groups": result.n_groups,
            "n_responses": result.n_responses,
            "response_vars": result.response_vars,
            "factor_var": result.factor_var,
            "df_hypothesis": result.df_hypothesis,
            "df_error": result.df_error,
            "eigenvalues": result.eigenvalues,
            "tests": {
                "pillai": {
                    "statistic": result.pillai.statistic,
                    "f_value": result.pillai.f_value,
                    "df1": result.pillai.df1,
                    "df2": result.pillai.df2,
                    "p_value": result.pillai.p_value,
                    "is_exact": result.pillai.is_exact
                },
                "wilks": {
                    "statistic": result.wilks.statistic,
                    "f_value": result.wilks.f_value,
                    "df1": result.wilks.df1,
                    "df2": result.wilks.df2,
                    "p_value": result.wilks.p_value,
                    "is_exact": result.wilks.is_exact
                },
                "hotelling_lawley": {
                    "statistic": result.hotelling_lawley.statistic,
                    "f_value": result.hotelling_lawley.f_value,
                    "df1": result.hotelling_lawley.df1,
                    "df2": result.hotelling_lawley.df2,
                    "p_value": result.hotelling_lawley.p_value,
                    "is_exact": result.hotelling_lawley.is_exact
                },
                "roy": {
                    "statistic": result.roy.statistic,
                    "f_value": result.roy.f_value,
                    "df1": result.roy.df1,
                    "df2": result.roy.df2,
                    "p_value": result.roy.p_value,
                    "is_exact": result.roy.is_exact,
                    "note": "Roy's test p-value is a lower bound"
                }
            },
            "group_means": result.group_means,
            "group_sizes": result.group_sizes,
            "grand_mean": result.grand_mean,
            "interpretation": format!(
                "Pillai's Trace = {:.4} (F = {:.2}, p = {:.4}). {}",
                result.pillai.statistic,
                result.pillai.f_value,
                result.pillai.p_value,
                if result.pillai.p_value < 0.05 {
                    "Groups differ significantly on the combined response variables."
                } else {
                    "No significant difference between groups on combined response variables."
                }
            )
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", result)),
        )]))
    }

    /// Run one-way ANOVA.
    #[tool(
        description = "Run one-way Analysis of Variance (ANOVA) to test whether means differ across groups. Returns F-statistic, p-value, effect sizes (eta-squared, omega-squared), and group statistics. Use when comparing a continuous response variable across 2+ categorical groups."
    )]
    pub async fn anova_one_way(
        &self,
        Parameters(request): Parameters<OneWayAnovaRequest>,
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

        let result = match run_one_way_anova(dataset, &request.response, &request.factor) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "One-way ANOVA failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Run Tukey's HSD (Honest Significant Differences) test.
    #[tool(
        description = "Run Tukey's HSD post-hoc test after one-way ANOVA to perform pairwise comparisons between all group means. Controls for family-wise error rate using the Studentized range distribution. Returns: difference in means, confidence interval, and adjusted p-value for each pair. Use this after finding a significant ANOVA result to identify which specific groups differ."
    )]
    pub async fn anova_tukey_hsd(
        &self,
        Parameters(request): Parameters<TukeyHsdRequest>,
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

        let conf_level = request.conf_level.unwrap_or(0.95);

        if conf_level <= 0.0 || conf_level >= 1.0 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Confidence level must be between 0 and 1 (e.g., 0.95 for 95% CI).".to_string(),
            )]));
        }

        let (anova, tukey) =
            match run_tukey_hsd(dataset, &request.response, &request.factor, conf_level) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Tukey HSD failed: {}",
                        e
                    ))]));
                }
            };

        let comparisons: Vec<serde_json::Value> = tukey
            .comparisons
            .iter()
            .map(|c| {
                serde_json::json!({
                    "group1": c.group1,
                    "group2": c.group2,
                    "diff": c.diff,
                    "ci_lower": c.ci_lower,
                    "ci_upper": c.ci_upper,
                    "p_adj": c.p_adj,
                    "significant": c.p_adj < (1.0 - conf_level)
                })
            })
            .collect();

        let output = serde_json::json!({
            "method": "Tukey HSD (Honest Significant Differences)",
            "response_var": tukey.response_var,
            "factor_var": tukey.factor_var,
            "conf_level": tukey.conf_level,
            "n_groups": tukey.n_groups,
            "df": tukey.df,
            "mse": tukey.mse,
            "anova_summary": {
                "f_statistic": anova.f_statistic,
                "p_value": anova.p_value,
                "significant": anova.p_value < 0.05
            },
            "comparisons": comparisons,
            "interpretation": format!(
                "{} pairwise comparisons at {:.0}% family-wise confidence level. {} pairs show significant differences (p < {:.2}).",
                comparisons.len(),
                conf_level * 100.0,
                comparisons.iter().filter(|c| c["significant"].as_bool().unwrap_or(false)).count(),
                1.0 - conf_level
            )
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", tukey)),
        )]))
    }

    /// Run two-way ANOVA.
    #[tool(
        description = "Run two-way Analysis of Variance (ANOVA) to test effects of two factors and their interaction on a response variable. Returns F-statistics and p-values for each factor and the interaction term. Use for factorial experimental designs."
    )]
    pub async fn anova_two_way(
        &self,
        Parameters(request): Parameters<TwoWayAnovaRequest>,
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

        let interaction = request.interaction.unwrap_or(true);

        let result = match run_two_way_anova(
            dataset,
            &request.response,
            &request.factor_a,
            &request.factor_b,
            interaction,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Two-way ANOVA failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            result.to_string(),
        )]))
    }

    /// Compute correlation matrix for numeric columns.
    #[tool(
        description = "Compute the Pearson correlation matrix for all numeric columns in a dataset."
    )]
    pub async fn compute_correlation(
        &self,
        Parameters(request): Parameters<CorrelationRequest>,
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

        let corr = match correlation_matrix(dataset) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to compute correlation matrix: {}",
                    e
                ))]));
            }
        };

        if corr.columns.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No numeric columns found in dataset.",
            )]));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Correlation Matrix for '{}':\n\n{}",
            request.dataset,
            corr.to_string_table()
        ))]))
    }

    /// Run isotonic regression (monotonic constraint).
    #[tool(
        description = "Run isotonic (monotonically increasing) least squares regression using the Pool Adjacent Violators Algorithm (PAVA). Returns piecewise constant fitted values that are monotonically non-decreasing. Useful for calibration, dose-response modeling, and trend analysis where monotonicity is assumed."
    )]
    pub async fn descriptive_isoreg(
        &self,
        Parameters(request): Parameters<IsoregRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

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

        let y_col = match df.column(&request.y_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Y column '{}' not found: {}",
                    request.y_column, e
                ))]));
            }
        };

        let y: Vec<f64> = match y_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_no_null_iter().collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot cast Y column: {}",
                    e
                ))]));
            }
        };

        let x: Vec<f64> = if let Some(ref x_col_name) = request.x_column {
            let x_col = match df.column(x_col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column '{}' not found: {}",
                        x_col_name, e
                    ))]));
                }
            };
            match x_col.cast(&DataType::Float64) {
                Ok(c) => match c.f64() {
                    Ok(f) => f.into_no_null_iter().collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "X column not numeric: {}",
                            e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot cast X column: {}",
                        e
                    ))]));
                }
            }
        } else {
            (1..=y.len()).map(|i| i as f64).collect()
        };

        let result = match isoreg(&x, &y) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Isotonic regression failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "n_observations": result.n,
            "n_knots": result.i_knots.len(),
            "was_sorted": result.is_ordered,
            "knots": result.i_knots.iter().map(|&k| serde_json::json!({
                "index": k,
                "x": result.x[k],
                "fitted": result.yf[k]
            })).collect::<Vec<_>>(),
            "fitted_values": &result.yf[..result.n.min(50)],
            "note": if result.n > 50 { Some(format!("Showing first 50 of {} fitted values", result.n)) } else { None },
            "interpretation": format!(
                "Isotonic regression on {} observations produced {} knots (level changes). The fitted values are monotonically non-decreasing and minimize squared error subject to this constraint.",
                result.n, result.i_knots.len()
            ),
            "references": "Barlow et al. (1972). Statistical Inference Under Order Restrictions."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap_or_default(),
        )]))
    }

    /// Run Tukey's Median Polish for robust two-way decomposition.
    #[tool(
        description = "Run Tukey's Median Polish for robust two-way decomposition of a matrix. Fits an additive model (constant + row effects + column effects + residuals) iteratively using medians instead of means, making it resistant to outliers. Returns the overall effect, row effects, column effects, and residuals. Useful for exploratory data analysis of two-way tables."
    )]
    pub async fn descriptive_medpolish(
        &self,
        Parameters(request): Parameters<MedpolishRequest>,
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

        let df = dataset.df();
        let n_rows = df.height();

        if request.columns.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "No columns specified for median polish. Provide at least 2 columns.",
            )]));
        }

        let n_cols = request.columns.len();
        if n_rows
            .checked_mul(n_cols)
            .is_none_or(|total| total > MAX_MATRIX_ELEMENTS)
        {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Matrix dimensions too large: {} rows x {} columns exceeds limit of {} elements.",
                n_rows, n_cols, MAX_MATRIX_ELEMENTS
            ))]));
        }

        let mut matrix: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut row_values: Vec<f64> = Vec::with_capacity(request.columns.len());
            for col_name in &request.columns {
                let col = match df.column(col_name) {
                    Ok(c) => c,
                    Err(_) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' not found in dataset.",
                            col_name
                        ))]));
                    }
                };
                let val = match col.f64() {
                    Ok(ca) => ca.get(row_idx).unwrap_or(f64::NAN),
                    Err(_) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' is not numeric.",
                            col_name
                        ))]));
                    }
                };
                row_values.push(val);
            }
            matrix.push(row_values);
        }

        let na_rm = request.na_rm.unwrap_or(false);

        let result = match run_medpolish(&matrix, request.eps, request.max_iter, na_rm) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Median polish failed: {}",
                    e
                ))]));
            }
        };

        let json_output = serde_json::json!({
            "overall": format!("{:.6}", result.overall),
            "row_effects": result.row.iter()
                .enumerate()
                .map(|(i, &r)| serde_json::json!({
                    "row": i + 1,
                    "effect": format!("{:.6}", r)
                }))
                .collect::<Vec<_>>(),
            "column_effects": request.columns.iter()
                .zip(result.col.iter())
                .map(|(name, &c)| serde_json::json!({
                    "column": name,
                    "effect": format!("{:.6}", c)
                }))
                .collect::<Vec<_>>(),
            "dimensions": {
                "n_rows": result.n_rows,
                "n_cols": result.n_cols
            },
            "convergence": {
                "converged": result.converged,
                "iterations": result.iterations,
                "final_sum_abs_residuals": format!("{:.6}", result.final_sum)
            },
            "residuals_summary": {
                "max_abs_residual": format!("{:.6}", result.residuals.iter()
                    .flat_map(|row| row.iter())
                    .map(|r| r.abs())
                    .fold(0.0f64, f64::max)),
                "sum_abs_residuals": format!("{:.6}", result.final_sum)
            },
            "interpretation": format!(
                "Median polish decomposed a {}×{} matrix into: overall={:.4}, {} row effects, {} column effects. {} after {} iterations. The additive model is: value = overall + row_effect + column_effect + residual.",
                result.n_rows, result.n_cols, result.overall,
                result.n_rows, result.n_cols,
                if result.converged { "Converged" } else { "Did not converge" },
                result.iterations
            ),
            "references": "Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap_or_default(),
        )]))
    }

    /// Canonical Correlation Analysis.
    #[tool(
        description = "Run Canonical Correlation Analysis (CCA) to find linear combinations of two sets of variables that have maximum correlation with each other. Returns canonical correlations (in decreasing order), coefficients for X variables (xcoef), and coefficients for Y variables (ycoef). The canonical variates are X*xcoef and Y*ycoef. Useful for multivariate dimensionality reduction, identifying relationships between variable sets, and understanding shared variance."
    )]
    pub async fn multivariate_cancor(
        &self,
        Parameters(request): Parameters<CancorRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::cancor::run_cancor;

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

        let xcenter = request.xcenter.unwrap_or(true);
        let ycenter = request.ycenter.unwrap_or(true);

        let x_refs: Vec<&str> = request.x_columns.iter().map(|s| s.as_str()).collect();
        let y_refs: Vec<&str> = request.y_columns.iter().map(|s| s.as_str()).collect();

        let result = match run_cancor(dataset, &x_refs, &y_refs, xcenter, ycenter) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Canonical correlation analysis failed: {}",
                    e
                ))]));
            }
        };

        let x_names: Vec<String> = result.x_names.clone().unwrap_or_else(|| {
            (0..result.n_x_vars)
                .map(|i| format!("X{}", i + 1))
                .collect()
        });
        let mut xcoef_display = Vec::new();
        for i in 0..result.n_x_vars {
            let mut row = serde_json::json!({
                "variable": x_names[i].clone()
            });
            for j in 0..result.n_canonical {
                row[format!("CC{}", j + 1)] = serde_json::json!(result.xcoef[[i, j]]);
            }
            xcoef_display.push(row);
        }

        let y_names: Vec<String> = result.y_names.clone().unwrap_or_else(|| {
            (0..result.n_y_vars)
                .map(|i| format!("Y{}", i + 1))
                .collect()
        });
        let mut ycoef_display = Vec::new();
        for i in 0..result.n_y_vars {
            let mut row = serde_json::json!({
                "variable": y_names[i].clone()
            });
            for j in 0..result.n_canonical {
                row[format!("CC{}", j + 1)] = serde_json::json!(result.ycoef[[i, j]]);
            }
            ycoef_display.push(row);
        }

        let json_output = serde_json::json!({
            "n_obs": result.n_obs,
            "n_x_vars": result.n_x_vars,
            "n_y_vars": result.n_y_vars,
            "n_canonical": result.n_canonical,
            "canonical_correlations": result.cor.to_vec(),
            "squared_correlations": result.squared_correlations().to_vec(),
            "x_coefficients": xcoef_display,
            "y_coefficients": ycoef_display,
            "x_center": result.xcenter.to_vec(),
            "y_center": result.ycenter.to_vec(),
            "interpretation": {
                "first_correlation": format!(
                    "The first canonical correlation is {:.4}, meaning {:.1}% of the variance in the first canonical variate of Y is explained by the first canonical variate of X.",
                    result.cor[0],
                    result.cor[0] * result.cor[0] * 100.0
                ),
                "total_canonical_pairs": format!(
                    "There are {} canonical correlation pairs (min of {} X variables and {} Y variables).",
                    result.n_canonical, result.n_x_vars, result.n_y_vars
                )
            }
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap_or_default(),
        )]))
    }

    /// Maximum Likelihood Factor Analysis.
    #[tool(
        description = "Run Maximum Likelihood Factor Analysis to identify latent factors underlying observed variables. Models correlation structure as x = Λf + e where Λ is the loadings matrix, f are factor scores, and e is error. Returns loadings matrix, uniquenesses (specific variances), communalities (variance explained per variable), variance proportions, chi-squared goodness-of-fit test, and optionally factor scores. Supports varimax (orthogonal) and promax (oblique) rotation."
    )]
    pub async fn multivariate_factanal(
        &self,
        Parameters(request): Parameters<FactorAnalysisRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::factanal::{RotationMethod, ScoresMethod, run_factanal};

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

        let rotation = match request.rotation.as_deref() {
            Some("none") => RotationMethod::None,
            Some("promax") => RotationMethod::Promax,
            Some("varimax") | None => RotationMethod::Varimax,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Invalid rotation method '{}'. Use 'varimax', 'promax', or 'none'.",
                    other
                ))]));
            }
        };

        let scores = match request.scores.as_deref() {
            Some("regression") => ScoresMethod::Regression,
            Some("bartlett") => ScoresMethod::Bartlett,
            Some("none") | None => ScoresMethod::None,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Invalid scores method '{}'. Use 'none', 'regression', or 'bartlett'.",
                    other
                ))]));
            }
        };

        let col_refs: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_factanal(dataset, &col_refs, request.n_factors, rotation, scores) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Factor analysis failed: {}",
                    e
                ))]));
            }
        };

        let var_names: Vec<String> = result.var_names.clone().unwrap_or_else(|| {
            (0..result.n_vars)
                .map(|i| format!("Var{}", i + 1))
                .collect()
        });

        let mut loadings_display = Vec::new();
        for i in 0..result.n_vars {
            let mut row = serde_json::json!({
                "variable": var_names[i].clone()
            });
            for j in 0..result.n_factors {
                row[format!("Factor{}", j + 1)] = serde_json::json!(result.loadings[[i, j]]);
            }
            row["communality"] = serde_json::json!(result.communalities[i]);
            row["uniqueness"] = serde_json::json!(result.uniquenesses[i]);
            loadings_display.push(row);
        }

        let mut json_output = serde_json::json!({
            "n_obs": result.n_obs,
            "n_vars": result.n_vars,
            "n_factors": result.n_factors,
            "converged": result.converged,
            "iterations": result.iterations,
            "loadings": loadings_display,
            "variance_proportions": result.variance_proportions.to_vec(),
            "cumulative_variance": result.cumulative_variance.to_vec(),
            "chi_squared_test": {
                "statistic": result.chi_squared,
                "df": result.df,
                "p_value": result.p_value,
                "interpretation": if result.p_value > 0.05 {
                    format!("The {} factor model adequately explains the correlations (p={:.4}).", result.n_factors, result.p_value)
                } else {
                    format!("The {} factor model may not fully explain the correlations (p={:.4}). Consider adding more factors.", result.n_factors, result.p_value)
                }
            }
        });

        if let Some(ref phi) = result.factor_correlation {
            let mut factor_corr = Vec::new();
            for i in 0..result.n_factors {
                let mut row = serde_json::Map::new();
                for j in 0..result.n_factors {
                    row.insert(format!("Factor{}", j + 1), serde_json::json!(phi[[i, j]]));
                }
                factor_corr.push(serde_json::Value::Object(row));
            }
            json_output["factor_correlations"] = serde_json::json!(factor_corr);
        }

        if let Some(ref scores) = result.scores {
            json_output["scores_computed"] = serde_json::json!(true);
            json_output["scores_shape"] = serde_json::json!({
                "n_obs": scores.nrows(),
                "n_factors": scores.ncols()
            });
        }

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap_or_default(),
        )]))
    }

    /// Compute Mahalanobis distance.
    #[tool(
        description = "Compute squared Mahalanobis distance for each observation. The Mahalanobis distance measures how far each observation is from the center of the distribution, accounting for correlations between variables. Useful for outlier detection, multivariate normality assessment, and cluster analysis. Returns squared distances (D²) which follow a chi-squared distribution with p degrees of freedom under multivariate normality."
    )]
    pub async fn multivariate_mahalanobis(
        &self,
        Parameters(request): Parameters<MahalanobisRequest>,
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

        let col_refs: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let center = request.center.as_deref();

        let result = match run_mahalanobis(dataset, &col_refs, center, None) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mahalanobis distance computation failed: {}",
                    e
                ))]));
            }
        };

        let n = result.distances.len();
        let mean_dist = result.distances.iter().sum::<f64>() / n as f64;
        let max_dist = result
            .distances
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_dist = result
            .distances
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let p = result.n_vars;
        let chi_sq_95 = p as f64 * 2.7;

        let n_outliers = result.distances.iter().filter(|&&d| d > chi_sq_95).count();

        let json_output = serde_json::json!({
            "n_obs": result.n_obs,
            "n_vars": result.n_vars,
            "center": result.center,
            "summary": {
                "mean_distance": format!("{:.4}", mean_dist),
                "min_distance": format!("{:.4}", min_dist),
                "max_distance": format!("{:.4}", max_dist),
            },
            "outlier_detection": {
                "chi_squared_threshold_95": format!("{:.4}", chi_sq_95),
                "n_potential_outliers": n_outliers,
                "note": format!("Under multivariate normality, D² ~ χ²({})", p)
            },
            "first_10_distances": result.distances.iter()
                .take(10)
                .enumerate()
                .map(|(i, &d)| serde_json::json!({
                    "observation": i + 1,
                    "distance_squared": format!("{:.4}", d),
                    "potential_outlier": d > chi_sq_95
                }))
                .collect::<Vec<_>>(),
            "interpretation": format!(
                "Computed Mahalanobis distances for {} observations with {} variables. {} observations exceed the χ²({}) 95th percentile threshold of {:.2}, suggesting they may be multivariate outliers.",
                n, p, n_outliers, p, chi_sq_95
            ),
            "references": "Mahalanobis, P. C. (1936). On the generalized distance in statistics."
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap_or_default(),
        )]))
    }

    /// Power analysis for one-way ANOVA.
    #[tool(
        description = "Compute power or sample size for balanced one-way ANOVA. Given 4 of {groups, n, between_var, within_var, sig_level, power}, computes the 5th. Use for study design comparing means across multiple groups."
    )]
    pub async fn power_anova_test(
        &self,
        Parameters(request): Parameters<PowerAnovaTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::power::power_anova_test;

        let groups = match request.groups {
            Some(g) if g >= 2 => g,
            Some(g) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "groups must be at least 2, got {}",
                    g
                ))]));
            }
            None => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "groups (number of groups) is required".to_string(),
                )]));
            }
        };

        let between_var = match request.between_var {
            Some(v) => v,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "between_var (between-group variance) is required".to_string(),
                )]));
            }
        };

        let within_var = request.within_var.unwrap_or(1.0);

        match power_anova_test(
            groups,
            request.n,
            between_var,
            within_var,
            request.sig_level,
            request.power,
        ) {
            Ok(result) => {
                let f = (result.between_var / result.within_var).sqrt();

                let json_output = serde_json::json!({
                    "method": result.method,
                    "groups": result.groups,
                    "n": result.n,
                    "between_var": result.between_var,
                    "within_var": result.within_var,
                    "sig_level": result.sig_level,
                    "power": result.power,
                    "effect_size_f": f,
                    "note": result.note,
                    "interpretation": format!(
                        "With {} groups, n={:.0} per group, and f={:.3}, power is {:.1}%",
                        result.groups, result.n, f, result.power * 100.0
                    )
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Power analysis failed: {}",
                e
            ))])),
        }
    }

    /// Power analysis for proportion tests.
    #[tool(
        description = "Compute power or sample size for two-sample proportion tests. Given 4 of {n, p1, p2, sig_level, power}, computes the 5th. Use for study design comparing proportions between two groups."
    )]
    pub async fn power_prop_test(
        &self,
        Parameters(request): Parameters<PowerPropTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::power::{PowerAlternative, power_prop_test};

        let alternative = match request.alternative.as_deref() {
            Some("one.sided") | Some("one_sided") | Some("onesided") => PowerAlternative::OneSided,
            _ => PowerAlternative::TwoSided,
        };

        let p1 = match request.p1 {
            Some(p) => p,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "p1 (proportion in first group) is required".to_string(),
                )]));
            }
        };

        match power_prop_test(
            request.n,
            p1,
            request.p2,
            request.sig_level,
            request.power,
            alternative,
        ) {
            Ok(result) => {
                let h = 2.0 * (result.p1.sqrt().asin() - result.p2.sqrt().asin());

                let json_output = serde_json::json!({
                    "method": result.method,
                    "n": result.n,
                    "p1": result.p1,
                    "p2": result.p2,
                    "sig_level": result.sig_level,
                    "power": result.power,
                    "alternative": format!("{:?}", result.alternative),
                    "effect_size_h": h,
                    "note": result.note,
                    "interpretation": format!(
                        "With n={:.0} per group, comparing p1={:.3} vs p2={:.3}, power is {:.1}%",
                        result.n, result.p1, result.p2, result.power * 100.0
                    )
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Power analysis failed: {}",
                e
            ))])),
        }
    }

    /// Power analysis for t-tests.
    #[tool(
        description = "Compute power or sample size for t-tests. Given 4 of {n, delta, sd, sig_level, power}, computes the 5th. Use for study design to determine required sample size for desired power, or to compute power for a given sample size. Supports one-sample, two-sample, and paired t-tests."
    )]
    pub async fn power_t_test(
        &self,
        Parameters(request): Parameters<PowerTTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::power::{PowerAlternative, TTestType, power_t_test};

        let test_type = match request.test_type.as_deref() {
            Some("one.sample") | Some("one_sample") | Some("onesample") => TTestType::OneSample,
            Some("paired") => TTestType::Paired,
            _ => TTestType::TwoSample,
        };

        let alternative = match request.alternative.as_deref() {
            Some("one.sided") | Some("one_sided") | Some("onesided") => PowerAlternative::OneSided,
            _ => PowerAlternative::TwoSided,
        };

        match power_t_test(
            request.n,
            request.delta,
            request.sd,
            request.sig_level,
            request.power,
            test_type,
            alternative,
        ) {
            Ok(result) => {
                let json_output = serde_json::json!({
                    "method": result.method,
                    "n": result.n,
                    "delta": result.delta,
                    "sd": result.sd,
                    "sig_level": result.sig_level,
                    "power": result.power,
                    "alternative": format!("{:?}", result.alternative),
                    "test_type": format!("{:?}", result.test_type),
                    "note": result.note,
                    "effect_size_d": result.delta / result.sd,
                    "interpretation": format!(
                        "With n={:.0} per group, effect d={:.3}, and α={:.3}, power is {:.1}%",
                        result.n, result.delta / result.sd, result.sig_level, result.power * 100.0
                    )
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string_pretty(&json_output).unwrap(),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Power analysis failed: {}",
                e
            ))])),
        }
    }
}
