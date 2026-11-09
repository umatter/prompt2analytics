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

/// Inline factor scores of `multivariate_factanal` without output_name: at most this many
/// rows, and at most FACTANAL_INLINE_VALUES values in all.
const FACTANAL_INLINE_ROWS: usize = 50;
const FACTANAL_INLINE_VALUES: usize = 500;

/// Cap on the default number of points `stats_spline` evaluates when neither xout nor n is
/// given (R's default is 3 x the number of pairs, uncapped).
const SPLINE_DEFAULT_MAX_POINTS: usize = 200;

use crate::server::AnalyticsServer;
use crate::tools::common::{
    MissingSummary, NaPolicy, REFUSAL_NAMES, budget_level_count, check_named_budget, echo_list,
    insert_missing, numeric_column, numeric_columns, numeric_matrix, parse_option,
};
use crate::tools::requests::stats::{
    ApproxRequest, CancorRequest, CorrelationRequest, CovWtRequest, DensityRequest, EcdfRequest,
    FactorAnalysisRequest, FivenumRequest, IqrRequest, IsoregRequest, LoglinRequest, MadRequest,
    MahalanobisRequest, ManovaRequest, MauchlyTestRequest, MedpolishRequest, ModelTablesRequest,
    OneWayAnovaRequest, PowerAnovaTestRequest, PowerPropTestRequest, PowerTTestRequest,
    SeContrastRequest, SplineRequest, TukeyHsdRequest, TwoWayAnovaRequest, WeightedMeanRequest,
};
use p2a_core::data::columns::{ColumnRequest, Factor};

use p2a_core::stats::{
    ApproxMethod, ApproxRule, ContrastType, DensityKernel, SplineMethod, SsType, TableType,
    TwoWayAnovaConfig, approx, correlation_matrix, fivenum, generate_contrasts, iqr, isoreg,
    loglin, model_tables, run_cov_wt, run_density, run_ecdf, run_mad, run_mahalanobis, run_manova,
    run_mauchly_test, run_medpolish, run_one_way_anova, run_tukey_hsd, run_two_way_anova,
    run_two_way_anova_with, se_contrast, spline, weighted_mean,
};

use p2a_core::options::Choices;
use p2a_core::stats::CovWtMethod;

/// Parses an optional option string with a `Choices` table: None gives `default`; an unknown
/// value gives the table's message ("unknown method 'biased'; expected one of: unbiased, ML").
/// For p2a-core enums that have no `FromStr` (no `impl` is possible in this crate) and for
/// handler-local choices; enums with one use `common::parse_option`.
pub(crate) fn parse_choice<T: Copy + PartialEq + 'static>(
    choices: &Choices<T>,
    value: Option<&str>,
    default: T,
) -> Result<T, String> {
    match value {
        None => Ok(default),
        Some(s) => choices.parse(s).map_err(|e| match e {
            p2a_core::EconError::InvalidSpecification { message } => message,
            other => other.to_string(),
        }),
    }
}

/// Merges `common::notes_json(converged, warnings)` into a JSON tool output object; a
/// non-object is left alone.
pub(crate) fn insert_notes(
    output: &mut serde_json::Value,
    converged: Option<(bool, usize)>,
    warnings: &[String],
) {
    if let (Some(o), serde_json::Value::Object(n)) = (
        output.as_object_mut(),
        crate::tools::common::notes_json(converged, warnings),
    ) {
        o.extend(n);
    }
}

/// `stats_cov_wt` method (R's cov.wt `method`).
pub(crate) static COV_WT_METHOD: Choices<CovWtMethod> = Choices {
    option: "method",
    values: &[
        (&["unbiased"], CovWtMethod::Unbiased),
        (&["ML", "mle", "maximum_likelihood"], CovWtMethod::ML),
    ],
};

/// Centre of `stats_mad` (R's mad `center`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MadCenter {
    Median,
    Mean,
}

static MAD_CENTER: Choices<MadCenter> = Choices {
    option: "center",
    values: &[
        (&["median"], MadCenter::Median),
        (&["mean"], MadCenter::Mean),
    ],
};

/// Test statistic of `stats_manova` (R's summary.manova `test`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManovaTest {
    Pillai,
    Wilks,
    HotellingLawley,
    Roy,
}

static MANOVA_TEST: Choices<ManovaTest> = Choices {
    option: "test",
    values: &[
        (&["pillai"], ManovaTest::Pillai),
        (&["wilks"], ManovaTest::Wilks),
        (
            &["hotelling-lawley", "hotelling", "lawley"],
            ManovaTest::HotellingLawley,
        ),
        (&["roy"], ManovaTest::Roy),
    ],
};

#[tool_router(router = stats_router, vis = "pub")]
impl AnalyticsServer {
    /// Log-linear model fitting for contingency tables.
    #[tool(
        description = "Fit log-linear models to multi-way contingency tables using iterative proportional fitting (IPF). Tests association and independence patterns. Equivalent to R's loglin(). Takes long-form data (one row per cell or observation); rows with missing values are dropped as xtabs does."
    )]
    pub async fn stats_loglin(
        &self,
        Parameters(request): Parameters<LoglinRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        if request.factor_columns.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one factor column is required.",
            )]));
        }

        // The count and every factor in one request: a row with a missing count or a missing
        // label is dropped, as R's xtabs drops it, and the levels follow R's factor() order
        let names: Vec<&str> = request.factor_columns.iter().map(String::as_str).collect();
        let cols = match ColumnRequest::new(dataset.df())
            .numeric(&request.count_column)
            .factors(&names)
            .na(NaPolicy::Omit)
            .extract()
        {
            Ok(c) => c,
            // "Computation stopped: ...", said once
            Err(e @ p2a_core::EconError::Cancelled(_)) => {
                return Ok(CallToolResult::error(vec![Content::text(e.to_string())]));
            }
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Log-linear model failed: {e}"
                ))]));
            }
        };
        let counts = cols.numeric(&request.count_column);
        let factors: Vec<&Factor> = names.iter().map(|name| cols.factor(name)).collect();
        let factor_levels: Vec<&[String]> = factors.iter().map(|f| f.levels.as_slice()).collect();

        // Build dimensions
        let dimensions: Vec<usize> = factors.iter().map(|f| f.n_levels()).collect();

        // Size the contingency table with checked arithmetic, then hold it to the budget
        let n_cells = match p2a_core::resource::checked_len(&dimensions, "contingency table") {
            Ok(n) => n,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Log-linear model refused: {e}"
                ))]));
            }
        };
        let limits = crate::tools::limits::current();
        if let Err(msg) = check_named_budget(
            "stats_loglin",
            "table_cells",
            n_cells as u64,
            limits.table_cells,
        ) {
            let dims: Vec<String> = dimensions.iter().map(|d| d.to_string()).collect();
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Log-linear model refused: the contingency table of {} would have {n_cells} cells \
                 ({}). {msg} Use factor columns with fewer distinct values.",
                echo_list(&request.factor_columns, ", ", REFUSAL_NAMES),
                echo_list(&dims, " x ", REFUSAL_NAMES)
            ))]));
        }

        // Build flattened contingency table
        let mut table = match p2a_core::resource::try_vec(n_cells, 0.0, "contingency table") {
            Ok(t) => t,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Log-linear model refused: {e}"
                ))]));
            }
        };

        // Row-major cell strides, last factor fastest (the layout `loglin` expects); every
        // partial product is at most `n_cells`, which was checked above
        let mut strides = vec![0usize; dimensions.len()];
        let mut stride = 1usize;
        for (s, &d) in strides.iter_mut().zip(&dimensions).rev() {
            *s = stride;
            stride *= d;
        }
        for (i, &count) in counts.iter().enumerate() {
            if i % 65_536 == 0
                && let Err(e) = p2a_core::resource::checkpoint()
            {
                // "Computation stopped: ...", raised while the table is being counted
                return Ok(CallToolResult::error(vec![Content::text(e.to_string())]));
            }
            let cell: usize = factors
                .iter()
                .zip(&strides)
                .map(|(f, s)| f.codes[i] as usize * s)
                .sum();
            table[cell] += count;
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
        let mut json_output = serde_json::json!({
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
                    "statistic": result.lrt,
                    "df": result.df,
                    "p_value": result.p_value_lrt
                },
                "pearson_chi_squared": {
                    "statistic": result.pearson,
                    "df": result.df,
                    "p_value": result.p_value_pearson
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
        insert_missing(&mut json_output, cols.missing());

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
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let table_type = match parse_option(request.table_type.as_deref(), TableType::Means) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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

            let mut json_output = serde_json::json!({
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
            insert_missing(&mut json_output, &anova.missing);

            Ok(CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&json_output).unwrap(),
            )]))
        } else if request.factors.len() == 2 {
            if let Err(msg) = check_two_way_levels(
                dataset,
                "stats_model_tables",
                &request.factors[0],
                &request.factors[1],
            ) {
                return Ok(CallToolResult::error(vec![Content::text(msg)]));
            }
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

            // The response and both factors in one request, so the table uses the rows the
            // ANOVA used; the levels follow R's factor() order
            let cols = match ColumnRequest::new(dataset.df())
                .numeric(&request.response)
                .factor(&request.factors[0])
                .factor(&request.factors[1])
                .na(NaPolicy::Omit)
                .extract()
            {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Two-way model tables failed: {e}"
                    ))]));
                }
            };
            let y = cols.numeric(&request.response);
            let fa = cols.factor(&request.factors[0]);
            let fb = cols.factor(&request.factors[1]);
            let (la, lb) = (fa.n_levels(), fb.n_levels());

            // Compute grand mean
            let grand_mean = y.iter().sum::<f64>() / y.len() as f64;

            // Cell sums and counts in one pass; cell (a, b) at a * lb + b. The la x lb table is
            // held to the table_cells budget before its buffers are allocated.
            let cells = match p2a_core::resource::checked_len(&[la, lb], "cell-means table") {
                Ok(cells) => cells,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Two-way model tables refused: {e}"
                    ))]));
                }
            };
            if let Err(msg) = check_named_budget(
                "stats_model_tables",
                "table_cells",
                cells as u64,
                crate::tools::limits::current().table_cells,
            ) {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Two-way model tables refused: the cell-means table would have {cells} cells \
                     ({la} x {lb} levels). {msg} Use factors with fewer levels."
                ))]));
            }
            let buffers =
                p2a_core::resource::try_vec(cells, 0.0f64, "cell-means table").and_then(|sums| {
                    Ok((
                        sums,
                        p2a_core::resource::try_vec(cells, 0usize, "cell-means table")?,
                    ))
                });
            let (mut sums, mut counts) = match buffers {
                Ok(b) => b,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Two-way model tables refused: {e}"
                    ))]));
                }
            };
            for ((&yi, &a), &b) in y.iter().zip(&fa.codes).zip(&fb.codes) {
                let cell = a as usize * lb + b as usize;
                sums[cell] += yi;
                counts[cell] += 1;
            }
            // An empty cell keeps the mean 0
            for (sum, &count) in sums.iter_mut().zip(&counts) {
                if count > 0 {
                    *sum /= count as f64;
                }
            }
            let cell_means: Vec<Vec<f64>> = sums.chunks(lb).map(<[f64]>::to_vec).collect();
            let cell_counts: Vec<Vec<usize>> = counts.chunks(lb).map(<[usize]>::to_vec).collect();
            let (levels_a, levels_b) = (&fa.levels, &fb.levels);

            // Convert to effects if requested
            let values = if table_type == TableType::Effects {
                cell_means
                    .iter()
                    .map(|row| row.iter().map(|&m| m - grand_mean).collect())
                    .collect()
            } else {
                cell_means
            };

            let mut json_output = serde_json::json!({
                "table_type": table_type.to_string(),
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
            insert_missing(&mut json_output, cols.missing());

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
            let contrast_type =
                match parse_option(request.contrast_type.as_deref(), ContrastType::Treatment) {
                    Ok(v) => v,
                    Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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

        let mut json_output = serde_json::json!({
            "groups": group_names,
            "group_means": group_means,
            "group_sizes": group_sizes,
            "contrasts": contrasts,
            "standard_errors": result.se,
            "mse": result.mse,
            "df_mse": result.df_mse
        });
        insert_missing(&mut json_output, &anova.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Weighted mean.
    #[tool(
        description = "Compute the weighted arithmetic mean of a numeric column. Equivalent to R's weighted.mean(). With na_rm (default) a row whose value or weight is missing is dropped as a pair and reported under missing; with na_rm = false a missing value or weight is an error."
    )]
    pub async fn stats_weighted_mean(
        &self,
        Parameters(request): Parameters<WeightedMeanRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // na_rm (default): a row whose value or weight is missing is dropped as a pair;
        // otherwise a missing value or weight is an error naming the column
        let na_rm = request.na_rm.unwrap_or(true);
        let policy = if na_rm {
            NaPolicy::Omit
        } else {
            NaPolicy::Fail
        };
        let cols = match numeric_columns(
            dataset,
            &[request.column.as_str(), request.weights.as_str()],
            policy,
        ) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted mean failed: {e}"
                ))]));
            }
        };
        let values = cols.numeric(&request.column);
        let weights = cols.numeric(&request.weights);

        let result = match weighted_mean(values, weights, na_rm) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted mean failed: {}",
                    e
                ))]));
            }
        };

        let mut json_output = serde_json::json!({
            "weighted_mean": result,
            "na_rm": na_rm,
            "n": cols.n(),
            "sum_weights": weights.iter().sum::<f64>()
        });
        insert_missing(&mut json_output, cols.missing());

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Weighted covariance matrix.
    #[tool(
        description = "Compute the weighted covariance matrix for a set of numeric columns. Can use unbiased (reliability) or ML weighting. Equivalent to R's cov.wt(). Missing values are refused, as R's cov.wt does."
    )]
    pub async fn stats_cov_wt(
        &self,
        Parameters(request): Parameters<CovWtRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let p = request.columns.len();
        if let Err(msg) = check_square_matrix("stats_cov_wt", p) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Weighted covariance refused: {p} columns give a {p} x {p} matrix. {msg}"
            ))]));
        }

        // cov.wt refuses missing values and weights; with NaPolicy::Fail no row is dropped,
        // so the matrix and the weights stay aligned
        let (data, _) = match numeric_matrix(dataset, &request.columns, NaPolicy::Fail) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted covariance failed: {e}"
                ))]));
            }
        };
        let (weights, _) = match numeric_column(dataset, &request.weights, NaPolicy::Fail) {
            Ok(w) => w,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Weighted covariance failed: {e}"
                ))]));
            }
        };

        let method = match parse_choice(
            &COV_WT_METHOD,
            request.method.as_deref(),
            CovWtMethod::Unbiased,
        ) {
            Ok(m) => COV_WT_METHOD.canonical(m),
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };
        let compute_cor = request.center.unwrap_or(true); // Use center flag to also compute correlation

        // Flatten matrix data to row-major slice
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let flat_data = into_row_major(data);

        let result = match run_cov_wt(
            &flat_data,
            n_rows,
            n_cols,
            Some(&weights),
            compute_cor,
            Some(method),
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
            "method": method
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json_output).unwrap(),
        )]))
    }

    /// Mauchly's sphericity test.
    #[tool(
        description = "Mauchly's test for sphericity in repeated measures designs. Tests whether the variances of the differences between all combinations of conditions are equal. Includes epsilon corrections (Greenhouse-Geisser, Huynh-Feldt). Equivalent to R's mauchly.test(). The p-value includes Anderson's second-order correction (as R's mauchly.test, with R's typo in one coefficient corrected; see validation/stats/mauchly.md); p_value_first_order is the first-order chi-square tail. Subjects with a missing value are dropped and reported under missing; sphericity_violated is null when the p-value cannot be computed."
    )]
    pub async fn stats_mauchly_test(
        &self,
        Parameters(request): Parameters<MauchlyTestRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let p = request.columns.len();
        if p < 3 {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Mauchly's test needs at least 3 repeated-measures columns (got {p}); with 2 \
                 conditions there is one contrast and sphericity holds trivially (R reports W = \
                 1, p = 1)."
            ))]));
        }
        if let Err(msg) = check_square_matrix("stats_mauchly_test", p) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Mauchly's test refused: {p} columns give a {p} x {p} matrix. {msg}"
            ))]));
        }

        // A subject with a missing value in any condition is dropped, as R's
        // lm(cbind(...) ~ 1) drops it
        let (data, missing) = match numeric_matrix(dataset, &request.columns, NaPolicy::Omit) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mauchly's test failed: {e}"
                ))]));
            }
        };

        // Flatten to row-major and call run_mauchly_test
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let flat_data = into_row_major(data);

        let result = match run_mauchly_test(&flat_data, n_rows, n_cols) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mauchly's test failed: {}",
                    e
                ))]));
            }
        };

        let mut output = serde_json::json!({
            "w_statistic": result.w,
            "chi_squared": result.chi_squared,
            "p_value": result.p_value,
            "p_value_first_order": result.p_value_first_order,
            "df": result.df,
            "sphericity_violated": sphericity_verdict(result.p_value),
            "epsilon_corrections": {
                "greenhouse_geisser": result.epsilon_gg,
                "huynh_feldt": result.epsilon_hf,
                "lower_bound": result.epsilon_lb
            },
            "n_subjects": result.n,
            "n_conditions": result.p_levels
        });
        insert_missing(&mut output, &missing);

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Missing values are dropped (as na.rm = TRUE) and reported
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        match fivenum(&values) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "minimum": result.minimum,
                    "lower_hinge": result.lower_hinge,
                    "median": result.median,
                    "upper_hinge": result.upper_hinge,
                    "maximum": result.maximum,
                    "n": result.n,
                    "iqr": result.upper_hinge - result.lower_hinge
                });
                insert_missing(&mut json_output, &missing);
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Missing values are dropped (as na.rm = TRUE) and reported
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        match iqr(&values, request.qtype) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "iqr": result,
                    "quantile_type": request.qtype.unwrap_or(7)
                });
                insert_missing(&mut json_output, &missing);
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Missing values are dropped (as na.rm = TRUE) and reported
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // If center is "mean", the mean of the values used (no missing value is left);
        // otherwise None, which is the median in mad()
        let center_kind =
            match parse_choice(&MAD_CENTER, request.center.as_deref(), MadCenter::Median) {
                Ok(v) => v,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
            };
        let center = if center_kind == MadCenter::Mean && !values.is_empty() {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        } else {
            None
        };

        let constant = request.constant;

        match run_mad(&values, center, constant) {
            Ok(mad_value) => {
                let mut json_output = serde_json::json!({
                    "mad": mad_value,
                    "center": MAD_CENTER.canonical(center_kind),
                    "center_used": center,
                    "constant": constant.unwrap_or(1.4826),
                    "n": values.len()
                });
                insert_missing(&mut json_output, &missing);
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Missing values are dropped (as na.rm = TRUE) and reported
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
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

                let mut json_output = serde_json::json!({
                    "x": x_out,
                    "y": y_out,
                    "n": result.n,
                    "n_unique": result.x.len()
                });
                insert_missing(&mut json_output, &missing);
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // Missing values are dropped (as na.rm = TRUE) and reported
        let (values, missing) = match numeric_column(dataset, &request.column, NaPolicy::Omit) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let kernel = match parse_option(request.kernel.as_deref(), DensityKernel::Gaussian) {
            Ok(k) => k.to_string(),
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let n = request.n.unwrap_or(512);

        match run_density(&values, request.bw, &kernel, Some(n)) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "bw": result.bw,
                    "kernel": result.kernel,
                    "n": result.n,
                    "data_n": values.len(),
                    "range": [result.x.first(), result.x.last()]
                });
                insert_missing(&mut json_output, &missing);
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
        description = "Interpolating cubic spline through (x, y), as R's spline(): method 'fmm' (default), 'natural', 'periodic' or 'hyman' (monotone). Returns the evaluated points (x, y): at xout, or at n equally spaced points (default 3 times the number of complete pairs, as R, capped at 200). Rows with a missing x or y are dropped, as R does. No spline coefficients are returned."
    )]
    pub async fn stats_spline(
        &self,
        Parameters(request): Parameters<SplineRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // A row with a missing x or y is dropped from both, as R's regularize.values drops it
        let cols = match numeric_columns(
            dataset,
            &[request.x.as_str(), request.y.as_str()],
            NaPolicy::Omit,
        ) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let (x, y) = (cols.numeric(&request.x), cols.numeric(&request.y));

        let method = match parse_option(request.method.as_deref(), SplineMethod::Fmm) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // R's default n is 3 x the number of distinct x after ties are collapsed; the MCP
        // default is capped at SPLINE_DEFAULT_MAX_POINTS so that a large dataset does not
        // return thousands of points nobody asked for (an explicit n is honoured).
        let n = request.n.or_else(|| {
            request.xout.is_none().then(|| {
                let mut xs = x.to_vec();
                xs.sort_by(f64::total_cmp);
                xs.dedup();
                (3 * xs.len()).min(SPLINE_DEFAULT_MAX_POINTS)
            })
        });
        match spline(x, y, request.xout.as_deref(), n, method) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "method": method.to_string(),
                    "n_points": result.x.len(),
                    "n_pairs": x.len(),
                    "n_knots": x.len()
                });
                insert_missing(&mut json_output, cols.missing());
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // A row with a missing x or y is dropped from both, as R's regularize.values drops it
        let cols = match numeric_columns(
            dataset,
            &[request.x.as_str(), request.y.as_str()],
            NaPolicy::Omit,
        ) {
            Ok(c) => c,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let (x, y) = (cols.numeric(&request.x), cols.numeric(&request.y));

        let method = match parse_option(request.method.as_deref(), ApproxMethod::Linear) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let rule = match parse_option(request.rule.as_deref(), ApproxRule::Na) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        // f parameter: for constant interpolation, 0 = left value, 1 = right value, 0.5 = average
        let f = 0.0; // Default: use left value for step interpolation

        match approx(x, y, Some(&request.xout), None, method, rule, f) {
            Ok(result) => {
                let mut json_output = serde_json::json!({
                    "x": result.x,
                    "y": result.y,
                    "method": method.to_string(),
                    "rule": rule.to_string(),
                    "n_interpolated": result.x.len()
                });
                insert_missing(&mut json_output, cols.missing());
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        if request.response_vars.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "MANOVA requires at least 2 response variables.".to_string(),
            )]));
        }

        let test = match parse_choice(&MANOVA_TEST, request.test.as_deref(), ManovaTest::Pillai) {
            Ok(t) => t,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

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

        let (label, selected, bound) = match test {
            ManovaTest::Pillai => ("Pillai's Trace", &result.pillai, ""),
            ManovaTest::Wilks => ("Wilks' Lambda", &result.wilks, ""),
            ManovaTest::HotellingLawley => ("Hotelling-Lawley Trace", &result.hotelling_lawley, ""),
            ManovaTest::Roy => ("Roy's Largest Root", &result.roy, "; p is a lower bound"),
        };

        let mut output = serde_json::json!({
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
            "selected_test": MANOVA_TEST.canonical(test),
            "interpretation": format!(
                "{label} = {:.4} (F = {:.2}, p = {:.4}{bound}). {}",
                selected.statistic,
                selected.f_value,
                selected.p_value,
                if selected.p_value < 0.05 {
                    "Groups differ significantly on the combined response variables."
                } else {
                    "No significant difference between groups on combined response variables."
                }
            )
        });
        insert_missing(&mut output, &result.missing);

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut output = serde_json::json!({
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
        insert_missing(&mut output, &anova.missing);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| format!("{:?}", tukey)),
        )]))
    }

    /// Run two-way ANOVA.
    #[tool(
        description = "Run two-way Analysis of Variance (ANOVA) to test effects of two factors and their interaction on a response variable. Returns F-statistics and p-values for each factor and the interaction term. Use for factorial experimental designs. ss_type: 'I' (default, sequential as R aov), 'II' or 'III' (as car::Anova). Rows with missing values are dropped and reported."
    )]
    pub async fn anova_two_way(
        &self,
        Parameters(request): Parameters<TwoWayAnovaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let interaction = request.interaction.unwrap_or(true);
        let ss_type = match request.ss_type.as_deref().map(str::parse::<SsType>) {
            None => SsType::I,
            Some(Ok(t)) => t,
            Some(Err(e)) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Two-way ANOVA failed: {e}"
                ))]));
            }
        };

        if let Err(msg) = check_two_way_levels(
            dataset,
            "anova_two_way",
            &request.factor_a,
            &request.factor_b,
        ) {
            return Ok(CallToolResult::error(vec![Content::text(msg)]));
        }

        let result = match run_two_way_anova_with(
            dataset,
            &request.response,
            &request.factor_a,
            &request.factor_b,
            &TwoWayAnovaConfig {
                interaction,
                ss_type,
            },
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Run isotonic (monotonically increasing) least squares regression using the Pool Adjacent Violators Algorithm (PAVA). Returns piecewise constant fitted values that are monotonically non-decreasing. Useful for calibration, dose-response modeling, and trend analysis where monotonicity is assumed. Missing values are refused, as R's isoreg does."
    )]
    pub async fn descriptive_isoreg(
        &self,
        Parameters(request): Parameters<IsoregRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        // isoreg refuses missing values, as R's isoreg does ("missing values not allowed")
        let mut names = vec![request.y_column.as_str()];
        if let Some(x) = &request.x_column {
            names.push(x.as_str());
        }
        let cols = match numeric_columns(dataset, &names, NaPolicy::Fail) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Isotonic regression failed: {e}"
                ))]));
            }
        };
        let y = cols.numeric(&request.y_column);
        // Without an x column, x is the row index 1..=n
        let row_index: Vec<f64> = if request.x_column.is_some() {
            Vec::new()
        } else {
            match p2a_core::resource::try_with_capacity(y.len(), "row index") {
                Ok(mut v) => {
                    v.extend((1..=y.len()).map(|i| i as f64));
                    v
                }
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Isotonic regression failed: {e}"
                    ))]));
                }
            }
        };
        let x = match &request.x_column {
            Some(name) => cols.numeric(name),
            None => &row_index,
        };

        let result = match isoreg(x, y) {
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        // Each column read once; a missing cell is NaN, which run_medpolish skips with
        // na_rm = true and refuses otherwise, as R's medpolish does
        let mut columns: Vec<Vec<Option<f64>>> = Vec::with_capacity(n_cols);
        for name in &request.columns {
            match p2a_core::data::columns::numeric_options(df, name) {
                Ok(values) => columns.push(values),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
            }
        }
        let mut matrix: Vec<Vec<f64>> =
            match p2a_core::resource::try_with_capacity(n_rows, "median polish matrix") {
                Ok(m) => m,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Median polish refused: {e}"
                    ))]));
                }
            };
        for row in 0..n_rows {
            matrix.push(columns.iter().map(|c| c[row].unwrap_or(f64::NAN)).collect());
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let xcenter = request.xcenter.unwrap_or(true);
        let ycenter = request.ycenter.unwrap_or(true);

        let (p, q) = (request.x_columns.len(), request.y_columns.len());
        let side = p.max(q);
        if let Err(msg) = check_square_matrix("multivariate_cancor", side) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Canonical correlation refused: {p} X columns and {q} Y columns need {side} x \
                 {side} matrices. {msg}"
            ))]));
        }

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
        description = "Run Maximum Likelihood Factor Analysis to identify latent factors underlying observed variables. Models correlation structure as x = Λf + e where Λ is the loadings matrix, f are factor scores, and e is error. Returns loadings matrix, uniquenesses (specific variances), communalities (variance explained per variable), variance proportions, chi-squared goodness-of-fit test and, with scores = 'regression' or 'bartlett', the factor scores (the first rows inline, or all of them as columns Factor1..Factork of a new dataset with output_name). Supports varimax (orthogonal) and promax (oblique) rotation."
    )]
    pub async fn multivariate_factanal(
        &self,
        Parameters(request): Parameters<FactorAnalysisRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::stats::factanal::{RotationMethod, ScoresMethod, run_factanal};

        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let rotation = match parse_option(request.rotation.as_deref(), RotationMethod::Varimax) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let scores = match parse_option(request.scores.as_deref(), ScoresMethod::None) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        if scores == ScoresMethod::None && request.output_name.is_some() {
            return Ok(CallToolResult::error(vec![Content::text(
                "output_name stores factor scores; set scores = 'regression' or 'bartlett'",
            )]));
        }

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

        if let Some(ref sc) = result.scores {
            let (n, k) = (sc.nrows(), sc.ncols());
            let columns: Vec<String> = (1..=k).map(|j| format!("Factor{j}")).collect();
            let mut block = serde_json::json!({
                "method": scores.to_string(),
                "columns": columns,
                "n_rows": n,
            });
            if let Some(name) = request.output_name.as_deref() {
                let df = dataset.df();
                if let Some(c) = columns.iter().find(|c| {
                    df.get_column_names()
                        .iter()
                        .any(|h| h.as_str() == c.as_str())
                }) {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "dataset '{}' already has a column '{c}'; the scores cannot be added",
                        request.dataset
                    ))]));
                }
                if df.height() != n {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "factor scores have {n} rows but dataset '{}' has {}",
                        request.dataset,
                        df.height()
                    ))]));
                }
                let new_cols: Vec<p2a_core::polars::prelude::Column> = columns
                    .iter()
                    .enumerate()
                    .map(|(j, c)| {
                        p2a_core::polars::prelude::Column::new(
                            c.as_str().into(),
                            sc.column(j).to_vec(),
                        )
                    })
                    .collect();
                let out = match df.hstack(&new_cols) {
                    Ok(out) => out,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Factor analysis failed: cannot add the score columns: {e}"
                        ))]));
                    }
                };
                block["saved_as"] = serde_json::json!(name);
                drop(datasets);
                self.datasets
                    .write()
                    .await
                    .insert(name.to_string(), p2a_core::data::Dataset::new(out));
            } else {
                let r = n
                    .min(FACTANAL_INLINE_ROWS)
                    .min((FACTANAL_INLINE_VALUES / k.max(1)).max(1));
                let rows: Vec<Vec<f64>> = (0..r).map(|i| sc.row(i).to_vec()).collect();
                block["rows_returned"] = serde_json::json!(r);
                block["rows"] = serde_json::json!(rows);
                if r < n {
                    block["note"] = serde_json::json!(format!(
                        "showing rows 1-{r} ({r} of {n}); set output_name to store all scores as dataset columns"
                    ));
                }
            }
            json_output["scores"] = block;
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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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

        let mut json_output = serde_json::json!({
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
                .zip(&result.rows)
                .take(10)
                .map(|(&d, &row)| serde_json::json!({
                    "observation": row + 1,
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
        insert_missing(&mut json_output, &result.missing);

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

        let alternative =
            match parse_option(request.alternative.as_deref(), PowerAlternative::TwoSided) {
                Ok(v) => v,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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
                    "alternative": result.alternative.to_string(),
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

        let test_type = match parse_option(request.test_type.as_deref(), TTestType::TwoSample) {
            Ok(v) => v,
            Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
        };

        let alternative =
            match parse_option(request.alternative.as_deref(), PowerAlternative::TwoSided) {
                Ok(v) => v,
                Err(msg) => return Ok(CallToolResult::error(vec![Content::text(msg)])),
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
                    "alternative": result.alternative.to_string(),
                    "test_type": result.test_type.to_string(),
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

/// `true`/`false` from p < 0.05 when p is finite, JSON null otherwise: a p-value that could not
/// be computed is no evidence for sphericity.
fn sphericity_verdict(p_value: f64) -> serde_json::Value {
    if p_value.is_finite() {
        serde_json::json!(p_value < 0.05)
    } else {
        serde_json::Value::Null
    }
}

/// Refuses a two-way ANOVA whose two factors together have more levels than the `max_groups`
/// budget of the installed limits, before the fit: the least-squares fit's cost grows with the
/// square of the level count. Each factor's distinct values are counted, an upper bound on the
/// levels the fit keeps. Nothing is counted when `max_groups` is off; a missing column is left
/// to the core to report.
fn check_two_way_levels(
    dataset: &p2a_core::data::Dataset,
    tool: &str,
    factor_a: &str,
    factor_b: &str,
) -> Result<(), String> {
    let Some(max_groups) = crate::tools::limits::current().max_groups else {
        return Ok(());
    };
    let refused = |msg: String| format!("Two-way ANOVA refused: {msg}");
    let df = dataset.df();
    let (Some(la), Some(lb)) = (
        budget_level_count(df, factor_a).map_err(refused)?,
        budget_level_count(df, factor_b).map_err(refused)?,
    ) else {
        return Ok(());
    };
    let levels = la.saturating_add(lb) as u64;
    check_named_budget(tool, "max_groups", levels, Some(max_groups as u64)).map_err(|msg| {
        refused(format!(
            "factors `{}` and `{}` have {la} and {lb} levels, {levels} together, and the fit's \
             cost grows with the square of that count. {msg} Use factors with fewer levels or \
             filter the data first.",
            crate::tools::limits::echo(factor_a),
            crate::tools::limits::echo(factor_b)
        ))
    })
}

/// Checks a `side x side` matrix against the `table_cells` budget of the installed limits (a
/// product past `u64` counts as `u64::MAX`, so it passes only when the budget is unlimited).
fn check_square_matrix(tool: &str, side: usize) -> Result<(), String> {
    let cells = (side as u64).saturating_mul(side as u64);
    check_named_budget(
        tool,
        "table_cells",
        cells,
        crate::tools::limits::current().table_cells,
    )
}

/// The elements of `data` in row-major order, without a copy when it already has that layout
/// (a matrix fresh from `numeric_matrix` does).
fn into_row_major(data: ndarray::Array2<f64>) -> Vec<f64> {
    if !data.is_standard_layout() {
        return data.iter().copied().collect();
    }
    let n = data.len();
    let (raw, offset) = data.into_raw_vec_and_offset();
    let start = offset.unwrap_or(0);
    if start == 0 && raw.len() == n {
        raw
    } else {
        raw[start..start + n].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::common::test_support::{hosted_with, server_with, text};
    use crate::tools::limits::{self, Limits};
    use p2a_core::polars::prelude::*;
    use p2a_validation::{Case, Check, Tol};

    #[test]
    fn test_parse_choice_and_stats_option_tables() {
        for (name, r) in [
            ("cov_wt method", COV_WT_METHOD.validate()),
            ("mad center", MAD_CENTER.validate()),
            ("manova test", MANOVA_TEST.validate()),
        ] {
            assert!(r.is_ok(), "{name}: {r:?}");
        }
        use p2a_core::stats::CovWtMethod;
        assert_eq!(
            parse_choice(&COV_WT_METHOD, None, CovWtMethod::Unbiased),
            Ok(CovWtMethod::Unbiased)
        );
        assert_eq!(
            parse_choice(&COV_WT_METHOD, Some("mle"), CovWtMethod::Unbiased),
            Ok(CovWtMethod::ML)
        );
        assert_eq!(
            parse_choice(
                &COV_WT_METHOD,
                Some(" Maximum-Likelihood "),
                CovWtMethod::Unbiased
            ),
            Ok(CovWtMethod::ML)
        );
        assert_eq!(
            parse_choice(&COV_WT_METHOD, Some("biased"), CovWtMethod::Unbiased).unwrap_err(),
            "unknown method 'biased'; expected one of: unbiased, ML"
        );
        assert_eq!(
            parse_choice(&MANOVA_TEST, Some("Hotelling-Lawley"), ManovaTest::Pillai),
            Ok(ManovaTest::HotellingLawley)
        );
        assert_eq!(
            parse_choice(&MAD_CENTER, Some("trimmed"), MadCenter::Median).unwrap_err(),
            "unknown center 'trimmed'; expected one of: median, mean"
        );
        let mut out = serde_json::json!({"a": 1});
        insert_notes(
            &mut out,
            Some((false, 3)),
            &["Warning: loglin did not converge".to_string()],
        );
        assert_eq!(
            out,
            serde_json::json!({"a": 1, "converged": false, "iterations": 3, "warnings": ["loglin did not converge"]})
        );
        let mut arr = serde_json::json!([1]);
        insert_notes(&mut arr, None, &[]);
        assert_eq!(arr, serde_json::json!([1]));
    }

    /// One unique string level per row and column, and a count column `n` = 1.0.
    fn ids(n: usize, cols: &[&str]) -> DataFrame {
        let mut columns: Vec<Column> = cols
            .iter()
            .map(|c| {
                Column::new(
                    (*c).into(),
                    (0..n).map(|i| format!("{c}{i}")).collect::<Vec<_>>(),
                )
            })
            .collect();
        columns.push(Column::new("n".into(), vec![1.0f64; n]));
        DataFrame::new(columns).unwrap()
    }

    /// `n` rows of a noisy response and two factors with `la` and `lb` levels (row i in level
    /// i % la and i % lb).
    fn two_factors(n: usize, la: usize, lb: usize) -> DataFrame {
        let y: Vec<f64> = (0..n)
            .map(|i| (i % 7) as f64 * 0.8 + ((i * 13) % 5) as f64 * 0.3)
            .collect();
        let a: Vec<String> = (0..n).map(|i| format!("a{}", i % la)).collect();
        let b: Vec<String> = (0..n).map(|i| format!("b{}", i % lb)).collect();
        df! { "y" => y, "a" => a, "b" => b }.unwrap()
    }

    fn two_way_req() -> TwoWayAnovaRequest {
        TwoWayAnovaRequest {
            dataset: "d".into(),
            response: "y".into(),
            factor_a: "a".into(),
            factor_b: "b".into(),
            interaction: None,
            ss_type: None,
        }
    }

    fn tables_req() -> ModelTablesRequest {
        ModelTablesRequest {
            dataset: "d".into(),
            response: "y".into(),
            factors: vec!["a".into(), "b".into()],
            table_type: None,
            se: None,
        }
    }

    #[test]
    fn anova_two_way_refuses_more_levels_than_max_groups() {
        // 51 + 50 = 101 levels > 100 (hosted max_groups): refused before the fit.
        let (rt, server) = server_with(vec![("d", two_factors(102, 51, 50))]);
        for r in limits::with_limits(Limits::hosted(), || {
            [
                rt.block_on(server.anova_two_way(Parameters(two_way_req()))),
                rt.block_on(server.stats_model_tables(Parameters(tables_req()))),
            ]
        }) {
            let r = r.unwrap();
            assert_eq!(r.is_error, Some(true), "{}", text(&r));
            let t = text(&r);
            assert!(
                t.contains("max_groups") && t.contains("101") && t.contains("51 and 50"),
                "{t}"
            );
        }
    }

    #[test]
    fn anova_two_way_level_budget_boundary() {
        // 3 + 2 = 5 levels pass max_groups = 5, 3 + 3 = 6 do not; off counts nothing.
        let l = hosted_with(&["max_groups=5"]);
        let (rt, server) = server_with(vec![
            ("ok", two_factors(36, 3, 2)),
            ("over", two_factors(36, 3, 3)),
        ]);
        let run = |limits: Limits, dataset: &str| {
            let req = TwoWayAnovaRequest {
                dataset: dataset.into(),
                ..two_way_req()
            };
            limits::with_limits(limits, || {
                rt.block_on(server.anova_two_way(Parameters(req))).unwrap()
            })
        };
        let ok = run(l.clone(), "ok");
        assert_eq!(ok.is_error, Some(false), "{}", text(&ok));
        let over = run(l, "over");
        assert_eq!(over.is_error, Some(true), "{}", text(&over));
        assert!(text(&over).contains("max_groups"), "{}", text(&over));
        let off = run(Limits::off(), "over");
        assert_eq!(off.is_error, Some(false), "{}", text(&off));
    }

    #[test]
    fn model_tables_two_way_cells_are_held_to_table_cells() {
        // a 3 x 3 cell-means table (9 cells) against table_cells = 8, before its buffers.
        let (rt, server) = server_with(vec![("d", two_factors(36, 3, 3))]);
        let r = limits::with_limits(hosted_with(&["table_cells=8"]), || {
            rt.block_on(server.stats_model_tables(Parameters(tables_req())))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true), "{}", text(&r));
        assert!(
            text(&r).contains("table_cells") && text(&r).contains('9'),
            "{}",
            text(&r)
        );
        let fits = limits::with_limits(hosted_with(&["table_cells=9"]), || {
            rt.block_on(server.stats_model_tables(Parameters(tables_req())))
        })
        .unwrap();
        assert_eq!(fits.is_error, Some(false), "{}", text(&fits));
    }

    fn loglin_req(cols: &[&str]) -> LoglinRequest {
        LoglinRequest {
            dataset: "d".into(),
            count_column: "n".into(),
            factor_columns: cols.iter().map(|c| c.to_string()).collect(),
            margins: None,
            eps: None,
            max_iter: None,
        }
    }

    #[test]
    fn loglin_refuses_table_above_table_cells() {
        // 120 x 120 x 120 = 1,728,000 cells > 1,000,000 (hosted). Pre-fix: allocates and fits it.
        let (rt, server) = server_with(vec![("d", ids(120, &["a", "b", "c"]))]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.stats_loglin(Parameters(loglin_req(&["a", "b", "c"]))))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("1728000") && t.contains("table_cells") && t.contains("1000000"),
            "{t}"
        );
    }

    #[test]
    fn loglin_refusal_bounds_the_names_it_echoes() {
        // 7 factors of 10 levels: 10^7 cells > 1,000,000. Pre-fix the refusal listed every
        // factor name in full.
        let names: Vec<String> = (0..7).map(|i| format!("{}{i}", "f".repeat(200))).collect();
        let cols: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let (rt, server) = server_with(vec![("d", ids(10, &cols))]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.stats_loglin(Parameters(loglin_req(&cols))))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("table_cells") && t.contains("(and 2 more)"),
            "{t}"
        );
        assert!(!t.contains(&"f".repeat(65)), "{t}");
    }

    #[test]
    fn loglin_cancelled_while_counting_says_so_once() {
        // Pre-fix: "Log-linear model fitting stopped: Computation stopped: ...", raised while the
        // table was still being counted
        let (rt, server) = server_with(vec![("d", ids(4, &["a", "b"]))]);
        let token = p2a_core::resource::CancelToken::new();
        token.cancel();
        let r =
            p2a_core::resource::with_budget(p2a_core::resource::Budget::new(None, token), || {
                rt.block_on(server.stats_loglin(Parameters(loglin_req(&["a", "b"]))))
            })
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.starts_with("Computation stopped") && t.matches("stopped").count() == 1,
            "{t}"
        );
    }

    #[test]
    fn loglin_reports_overflowing_table_instead_of_panicking() {
        // 1000^7 overflows usize. Pre-fix: debug panic "attempt to multiply with overflow".
        let cols = ["a", "b", "c", "d", "e", "f", "g"];
        let (rt, server) = server_with(vec![("d", ids(1000, &cols))]);
        let r = limits::with_limits(Limits::off(), || {
            rt.block_on(server.stats_loglin(Parameters(loglin_req(&cols))))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("too large"), "{}", text(&r));
    }

    #[test]
    fn loglin_small_table_still_runs_under_hosted_limits() {
        let df = df!(
            "a" => ["a1", "a1", "a1", "a2", "a2", "a2"],
            "b" => ["b1", "b2", "b3", "b1", "b2", "b3"],
            "n" => [12.0, 7.0, 9.0, 8.0, 11.0, 14.0]
        )
        .unwrap();
        let (rt, server) = server_with(vec![("d", df)]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.stats_loglin(Parameters(loglin_req(&["a", "b"]))))
        })
        .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).contains("\"df\": 2"), "{}", text(&r));
    }

    fn xyw(n: usize) -> DataFrame {
        let f = |k: f64| {
            (0..n)
                .map(|i| (i as f64 * k).sin() + i as f64 * 0.01)
                .collect::<Vec<f64>>()
        };
        df!(
            "x1" => f(0.37),
            "x2" => f(1.13),
            "x3" => f(2.07),
            "y1" => f(0.71),
            "w" => (0..n).map(|i| 1.0 + (i % 5) as f64 * 0.1).collect::<Vec<f64>>()
        )
        .unwrap()
    }

    fn cov_wt_req() -> CovWtRequest {
        CovWtRequest {
            dataset: "d".into(),
            columns: vec!["x1".into(), "x2".into()],
            weights: "w".into(),
            center: None,
            method: None,
        }
    }

    #[test]
    fn cancor_refuses_matrices_above_table_cells() {
        let (rt, server) = server_with(vec![("d", xyw(30))]);
        let req = CancorRequest {
            dataset: "d".into(),
            x_columns: vec!["x1".into(), "x2".into()],
            y_columns: vec!["y1".into()],
            xcenter: None,
            ycenter: None,
        };
        let r = limits::with_limits(hosted_with(&["table_cells=3"]), || {
            rt.block_on(server.multivariate_cancor(Parameters(req)))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("table_cells"), "{}", text(&r));
    }

    #[test]
    fn cov_wt_refuses_matrix_above_table_cells() {
        let (rt, server) = server_with(vec![("d", xyw(30))]);
        let r = limits::with_limits(hosted_with(&["table_cells=3"]), || {
            rt.block_on(server.stats_cov_wt(Parameters(cov_wt_req())))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("table_cells"), "{}", text(&r));
    }

    #[test]
    fn mauchly_refuses_matrix_above_table_cells() {
        let (rt, server) = server_with(vec![("d", xyw(30))]);
        let req = MauchlyTestRequest {
            dataset: "d".into(),
            columns: vec!["x1".into(), "x2".into(), "x3".into()],
        };
        let r = limits::with_limits(hosted_with(&["table_cells=8"]), || {
            rt.block_on(server.stats_mauchly_test(Parameters(req)))
        })
        .unwrap();
        assert_eq!(r.is_error, Some(true));
    }

    #[test]
    fn cov_wt_still_runs_under_hosted_limits() {
        let (rt, server) = server_with(vec![("d", xyw(30))]);
        let r = limits::with_limits(Limits::hosted(), || {
            rt.block_on(server.stats_cov_wt(Parameters(cov_wt_req())))
        })
        .unwrap();
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
    }

    #[test]
    fn into_row_major_keeps_logical_order() {
        let m = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(
            into_row_major(m.clone()),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            into_row_major(m.reversed_axes()),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    /// The R case `id`, a runtime and a server holding its dataset under the dataset's file
    /// stem (returned as the last element).
    fn case_server(id: &str) -> (Case, tokio::runtime::Runtime, AnalyticsServer, String) {
        let case = Case::load(id);
        let df = p2a_core::data::DataLoader::load_csv(case.dataset_path()).unwrap();
        let name = std::path::Path::new(&case.dataset)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (rt, server) = server_with(vec![(name.as_str(), df)]);
        (case, rt, server, name)
    }

    /// The JSON of a successful tool result.
    fn json_of(r: &CallToolResult) -> serde_json::Value {
        assert_ne!(r.is_error, Some(true), "{}", text(r));
        serde_json::from_str(&text(r)).unwrap()
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

    /// Handlers that format numbers as strings ("38.019531").
    fn str_num(v: &serde_json::Value) -> f64 {
        v.as_str()
            .unwrap_or_else(|| panic!("not a string: {v}"))
            .parse()
            .unwrap()
    }

    /// A request deserialised from JSON, as the server does.
    fn req<T: serde::de::DeserializeOwned>(v: serde_json::Value) -> Parameters<T> {
        Parameters(serde_json::from_value(v).unwrap())
    }

    /// The numbers on the first line whose first whitespace token is `label` (significance
    /// stars and other words skipped). Panics with the text when no such line exists.
    fn row(text: &str, label: &str) -> Vec<f64> {
        let line = text
            .lines()
            .find(|l| l.split_whitespace().next() == Some(label))
            .unwrap_or_else(|| panic!("no line starting with {label:?} in:\n{text}"));
        line.split_whitespace()
            .skip(1)
            .filter_map(|t| t.parse::<f64>().ok())
            .collect()
    }

    /// `{"n_total", "n_used", "n_dropped", "by_column"}` as the tools report it.
    fn summary(n_total: usize, n_used: usize, by_column: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"n_total": n_total, "n_used": n_used,
                           "n_dropped": n_total - n_used, "by_column": by_column})
    }

    #[test]
    fn test_validate_mcp_stats_loglin_na() {
        // TOOL-8: a null label became the level "" (a 4 x 3 x 2 table, df 17) and null counts
        // were read as 0. R: loglin(xtabs(cnt ~ f1 + f2 + f3)), xtabs dropping those rows.
        let (case, rt, server, name) = case_server("stat_loglin_na");
        for (k, margins) in [
            ("indep", None),
            ("m12_3", Some(serde_json::json!([[0, 1], [2]]))),
            ("m12_13", Some(serde_json::json!([[0, 1], [0, 2]]))),
        ] {
            let mut a = serde_json::json!({"dataset": name, "count_column": "cnt",
                                           "factor_columns": ["f1", "f2", "f3"]});
            if let Some(m) = margins {
                a["margins"] = m;
            }
            let v = json_of(&rt.block_on(server.stats_loglin(req(a))).unwrap());
            let (g, model, missing) = (&v["goodness_of_fit"], &v["model"], &v["missing"]);
            Check::new(&case)
                .scalar(
                    &format!("{k}_lrt"),
                    num(&g["likelihood_ratio_test"]["statistic"]),
                    Tol::stat(448),
                )
                .scalar(
                    &format!("{k}_pearson"),
                    num(&g["pearson_chi_squared"]["statistic"]),
                    Tol::stat(448),
                )
                .scalar(
                    &format!("{k}_df"),
                    num(&g["likelihood_ratio_test"]["df"]),
                    Tol::exact(),
                )
                .scalar(
                    &format!("{k}_p_lrt"),
                    num(&g["likelihood_ratio_test"]["p_value"]),
                    Tol::p_value(448),
                )
                .scalar(
                    &format!("{k}_p_pearson"),
                    num(&g["pearson_chi_squared"]["p_value"]),
                    Tol::p_value(448),
                )
                .scalar("total", num(&model["total_count"]), Tol::exact())
                .vector("dims", &nums(&model["dimensions"]), Tol::exact())
                .scalar("n_total", num(&missing["n_total"]), Tol::exact())
                .scalar("n_used", num(&missing["n_used"]), Tol::exact())
                .scalar(
                    "n_missing_cnt",
                    num(&missing["by_column"]["cnt"]),
                    Tol::exact(),
                )
                .scalar(
                    "n_missing_f1",
                    num(&missing["by_column"]["f1"]),
                    Tol::exact(),
                )
                .scalar(
                    "n_missing_f2",
                    num(&missing["by_column"]["f2"]),
                    Tol::exact(),
                )
                .finish();
            assert_eq!(
                model["factor_levels"],
                serde_json::json!([["a", "b", "c"], ["2", "10"], ["x", "y"]])
            );
            assert_eq!(
                v["missing"],
                summary(16, 13, serde_json::json!({"cnt": 1, "f1": 1, "f2": 1}))
            );
        }
        let e = err_text(
            &rt.block_on(server.stats_loglin(req(serde_json::json!(
                {"dataset": name, "count_column": "f1", "factor_columns": ["f2", "f3"]}))))
                .unwrap(),
        );
        assert!(e.contains("Column 'f1' contains non-numeric values"), "{e}");
    }

    #[test]
    fn test_validate_mcp_stats_weighted_mean_na() {
        // TOOL-9: value and weight dropped their nulls each on their own and were paired by
        // position (10.4408 on 11 rows). R: weighted.mean on complete.cases(v, w).
        let (case, rt, server, name) = case_server("stat_weighted_na");
        let call = |extra: serde_json::Value| {
            let mut a = serde_json::json!({"dataset": name, "column": "v", "weights": "w"});
            if let serde_json::Value::Object(m) = extra {
                a.as_object_mut().unwrap().extend(m);
            }
            rt.block_on(server.stats_weighted_mean(req(a))).unwrap()
        };
        let v = json_of(&call(serde_json::json!({})));
        let i = json_of(&call(serde_json::json!({"column": "a", "weights": "wc"})));
        Check::new(&case)
            .scalar("wmean", num(&v["weighted_mean"]), Tol::coef(10))
            .scalar("wmean_n", num(&v["n"]), Tol::exact())
            .scalar("wmean_n", num(&v["missing"]["n_used"]), Tol::exact())
            .scalar("wmean_sum_w", num(&v["sum_weights"]), Tol::coef(10))
            .scalar("wmean_int", num(&i["weighted_mean"]), Tol::coef(12))
            .scalar("wmean_int_sum_w", num(&i["sum_weights"]), Tol::coef(12))
            .finish();
        assert_eq!(
            v["missing"],
            summary(12, 10, serde_json::json!({"v": 1, "w": 1}))
        );
        assert!(i.get("missing").is_none(), "{i}");
        let e = err_text(&call(serde_json::json!({"na_rm": false})));
        assert!(e.contains("Column 'v' contains 1 missing value(s)"), "{e}");
    }

    #[test]
    fn test_validate_mcp_stats_cov_wt_int() {
        // TOOL-9/TOOL-24: a null cell became NaN (null covariance entries) and a null weight
        // shortened the weights ("Weights length 11 doesn't match row count 12"). R: cov.wt
        // on the Int64 columns a, b; it refuses missing values and weights.
        let (case, rt, server, name) = case_server("stat_weighted_na");
        let call = |extra: serde_json::Value| {
            let mut a = serde_json::json!({"dataset": name, "columns": ["a", "b"],
                                           "weights": "wc"});
            if let serde_json::Value::Object(m) = extra {
                a.as_object_mut().unwrap().extend(m);
            }
            rt.block_on(server.stats_cov_wt(req(a))).unwrap()
        };
        let flat = |m: &serde_json::Value| -> Vec<f64> {
            m.as_array()
                .unwrap_or_else(|| panic!("not a matrix: {m}"))
                .iter()
                .flat_map(nums)
                .collect()
        };
        // deviation TOOL-23: center = false switches the correlation matrix off but still
        // centres the data (R's cov.wt(center = FALSE) does not), so center is left out.
        let v = json_of(&call(serde_json::json!({})));
        let ml = json_of(&call(serde_json::json!({"method": "ML"})));
        Check::new(&case)
            .vector("cov", &flat(&v["covariance_matrix"]), Tol::stat(12))
            .vector("cor", &flat(&v["correlation_matrix"]), Tol::stat(12))
            .named("center", &["a", "b"], &nums(&v["center"]), Tol::coef(12))
            .vector("wt", &nums(&v["weights"]), Tol::coef(12))
            .scalar("n_obs", num(&v["n_obs"]), Tol::exact())
            .vector("cov_ml", &flat(&ml["covariance_matrix"]), Tol::stat(12))
            .vector("cor_ml", &flat(&ml["correlation_matrix"]), Tol::stat(12))
            .finish();
        let e = err_text(&call(serde_json::json!({"columns": ["a", "b_na"]})));
        assert!(
            e.contains("Column 'b_na' contains 1 missing value(s)"),
            "{e}"
        );
        let e = err_text(&call(serde_json::json!({"weights": "w"})));
        assert!(e.contains("Column 'w' contains 1 missing value(s)"), "{e}");
    }

    #[test]
    fn test_validate_mcp_stats_mauchly_test_na() {
        // TOOL-24: missing cells reached the core test as NaN: W, chi-squared and p were null,
        // sphericity_violated was false and both epsilons 1, on 16 subjects. R:
        // mauchly.test(lm(cbind(t1, t2, t3, t4) ~ 1), X = ~1) on the 14 complete subjects.
        let (case, rt, server, name) = case_server("stat_mauchly_na");
        let v = json_of(
            &rt.block_on(server.stats_mauchly_test(req(serde_json::json!(
                {"dataset": name, "columns": ["t1", "t2", "t3", "t4"]}))))
                .unwrap(),
        );
        let eps = &v["epsilon_corrections"];
        // deviation STAT-82: Anderson's w2 with pp (R's 3 * p), so p_value is left out; the
        // first-order p is pinned to R's intermediate Pr1 (p_first_order).
        Check::new(&case)
            .scalar("w", num(&v["w_statistic"]), Tol::stat(14))
            .scalar("chisq", num(&v["chi_squared"]), Tol::stat(14))
            .scalar("df", num(&v["df"]), Tol::exact())
            .scalar(
                "p_first_order",
                num(&v["p_value_first_order"]),
                Tol::p_value(14),
            )
            .scalar("gg", num(&eps["greenhouse_geisser"]), Tol::stat(14))
            .scalar("hf", num(&eps["huynh_feldt"]), Tol::stat(14))
            .scalar("lb", num(&eps["lower_bound"]), Tol::stat(14))
            .scalar("n_used", num(&v["n_subjects"]), Tol::exact())
            .scalar("n_used", num(&v["missing"]["n_used"]), Tol::exact())
            .scalar("n_total", num(&v["missing"]["n_total"]), Tol::exact())
            .finish();
        assert_eq!(v["sphericity_violated"], serde_json::json!(true));
        assert_eq!(
            v["missing"],
            summary(16, 14, serde_json::json!({"t2": 1, "t4": 1}))
        );
        let e = err_text(
            &rt.block_on(server.stats_mauchly_test(req(serde_json::json!(
                {"dataset": name, "columns": ["t1", "t3"]}))))
                .unwrap(),
        );
        assert!(e.contains("at least 3 repeated-measures columns"), "{e}");
    }

    #[test]
    fn test_mcp_stats_mauchly_verdict_is_null_for_non_finite_p() {
        // TOOL-24: the verdict was `p < 0.05`, so a NaN p-value read as "sphericity holds"
        assert_eq!(sphericity_verdict(f64::NAN), serde_json::Value::Null);
        assert_eq!(sphericity_verdict(f64::INFINITY), serde_json::Value::Null);
        assert_eq!(sphericity_verdict(0.01), serde_json::json!(true));
        assert_eq!(sphericity_verdict(0.2), serde_json::json!(false));
    }

    #[test]
    fn test_validate_mcp_anova_two_way_ss_type() {
        // STAT-4: ss_type was not a field, so serde ignored it and Type I was computed (fa SS
        // 7.7072 where car::Anova(type = 2) gives 40.1011).
        let (case, rt, server, name) = case_server("stat_anova_two_way_unbalanced");
        let call = |extra: serde_json::Value| {
            let mut a = serde_json::json!({"dataset": name, "response": "y", "factor_a": "fa",
                                           "factor_b": "fb"});
            if let serde_json::Value::Object(m) = extra {
                a.as_object_mut().unwrap().extend(m);
            }
            rt.block_on(server.anova_two_way(req(a))).unwrap()
        };
        let (p4, x) = (Tol::printed(4), Tol::exact());
        let mut check = Check::new(&case);
        for (ss_type, prefix, line) in [
            ("II", "type2", "Sums of squares: Type II"),
            ("III", "type3", "Sums of squares: Type III"),
        ] {
            let t = ok_text(&call(serde_json::json!({"ss_type": ss_type})));
            assert!(t.contains(line), "{t}");
            for (label, key) in [("fa", "fa"), ("fb", "fb"), ("fa:fb", "int")] {
                let v = row(&t, label);
                assert_eq!(v.len(), 5, "{label}: {v:?}\n{t}");
                assert_eq!(v[1], 1.0, "{label} df\n{t}");
                check
                    .scalar(&format!("{prefix}_{key}_ss"), v[0], p4)
                    .scalar(&format!("{prefix}_{key}_f"), v[3], p4)
                    .scalar(&format!("{prefix}_{key}_p"), v[4], p4);
            }
            if ss_type == "II" {
                let resid = row(&t, "Residuals");
                check
                    .scalar("type2_resid_ss", resid[0], p4)
                    .scalar("type2_resid_df", resid[1], x);
            }
        }
        let t = ok_text(&call(
            serde_json::json!({"ss_type": "2", "interaction": false}),
        ));
        assert!(t.contains("Sums of squares: Type II"), "{t}");
        for label in ["fa", "fb"] {
            let v = row(&t, label);
            assert_eq!(v.len(), 5, "{label}: {v:?}\n{t}");
            check
                .scalar(&format!("add_type2_{label}_ss"), v[0], p4)
                .scalar(&format!("add_type2_{label}_f"), v[3], p4)
                .scalar(&format!("add_type2_{label}_p"), v[4], p4);
        }
        check.finish();
        let e = err_text(&call(serde_json::json!({"ss_type": "IV"})));
        assert!(e.contains("II"), "{e}");
        let t = ok_text(&call(serde_json::json!({})));
        assert!(t.contains("Sums of squares: Type I (sequential"), "{t}");
    }

    #[test]
    fn test_validate_mcp_anova_handlers_report_missing_rows() {
        // X-2: the ANOVA-family core functions drop incomplete rows (T2-WP07), but the JSON
        // did not say so; the two-way model table sorted Int64 levels as strings ("10" < "2").
        // R: manova and aov on the complete cases, factor() level order.
        let (case, rt, server, name) = case_server("stat_manova_na");
        let v = json_of(
            &rt.block_on(server.anova_manova(req(serde_json::json!(
                {"dataset": name, "response_vars": ["y1", "y2"], "factor": "g"}))))
                .unwrap(),
        );
        Check::new(&case)
            .scalar("n_used", num(&v["n_obs"]), Tol::exact())
            .scalar("n_used", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(
            v["missing"],
            summary(36, 33, serde_json::json!({"y1": 2, "y2": 1}))
        );

        let (case, rt, server, name) = case_server("stat_oneway_na_labels");
        let one_way = summary(9, 5, serde_json::json!({"y": 3, "g": 2}));
        let tukey = json_of(
            &rt.block_on(server.anova_tukey_hsd(req(serde_json::json!(
                {"dataset": name, "response": "y", "factor": "g"}))))
                .unwrap(),
        );
        let contrast = json_of(
            &rt.block_on(server.stats_se_contrast(req(serde_json::json!(
                {"dataset": name, "response": "y", "factor": "g"}))))
                .unwrap(),
        );
        let table = json_of(
            &rt.block_on(server.stats_model_tables(req(serde_json::json!(
                {"dataset": name, "response": "y", "factors": ["g"]}))))
                .unwrap(),
        );
        for (tool, v) in [
            ("anova_tukey_hsd", &tukey),
            ("stats_se_contrast", &contrast),
            ("stats_model_tables", &table),
        ] {
            Check::new(&case)
                .scalar("aov_n_used", num(&v["missing"]["n_used"]), Tol::exact())
                .finish();
            assert_eq!(v["missing"], one_way, "{tool}");
        }

        let (case, rt, server, name) = case_server("stat_loglin_na");
        let v = json_of(
            &rt.block_on(server.stats_model_tables(req(serde_json::json!(
                {"dataset": name, "response": "cnt", "factors": ["f2", "f3"]}))))
                .unwrap(),
        );
        let two = &v["two_way_table"];
        assert_eq!(two["levels_a"], serde_json::json!(["2", "10"]));
        assert_eq!(two["levels_b"], serde_json::json!(["x", "y"]));
        let counts: f64 = two["counts"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(nums)
            .sum();
        assert_eq!(counts, 14.0, "{two}");
        Check::new(&case)
            .scalar(
                "n_missing_cnt",
                num(&v["missing"]["by_column"]["cnt"]),
                Tol::exact(),
            )
            .scalar(
                "n_missing_f2",
                num(&v["missing"]["by_column"]["f2"]),
                Tol::exact(),
            )
            .finish();
        assert_eq!(
            v["missing"],
            summary(16, 14, serde_json::json!({"cnt": 1, "f2": 1}))
        );
        // the model table and the two-way ANOVA use the same rows
        let t = ok_text(
            &rt.block_on(server.anova_two_way(req(serde_json::json!(
                {"dataset": name, "response": "cnt", "factor_a": "f2", "factor_b": "f3"}))))
                .unwrap(),
        );
        let ss_f2 = row(&t, "f2")[0];
        assert!(
            Tol::printed(4).allows(ss_f2, num(&v["anova_summary"]["ss_a"])),
            "{ss_f2} vs {}\n{t}",
            v["anova_summary"]["ss_a"]
        );
    }

    #[test]
    fn test_validate_mcp_multivariate_mahalanobis_rows() {
        // X-2: since the core drops incomplete rows (T2-WP06) the tool numbered the distances
        // 1..10 by kept row; row 7 (1-based) has a missing `a`. R: mahalanobis on
        // complete.cases(a, b), which(complete.cases(a, b)).
        let (case, rt, server, name) = case_server("stat_pairwise_mahalanobis_na");
        let v = json_of(
            &rt.block_on(server.multivariate_mahalanobis(req(serde_json::json!(
                {"dataset": name, "columns": ["a", "b"]}))))
                .unwrap(),
        );
        let first = v["first_10_distances"].as_array().unwrap();
        let rows: Vec<f64> = first.iter().map(|d| num(&d["observation"])).collect();
        let want_rows: Vec<f64> = case.vector("mahalanobis_rows")[..10]
            .iter()
            .map(|r| r + 1.0)
            .collect();
        assert_eq!(rows, want_rows);
        assert_eq!(
            rows,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0]
        );
        let r = case.vector("mahalanobis_complete");
        for (i, d) in first.iter().enumerate() {
            let got = str_num(&d["distance_squared"]);
            assert!(
                Tol::printed(4).allows(got, r[i]),
                "distance {i}: {got} vs R {}",
                r[i]
            );
        }
        // deviation TOOL-10: chi_squared_threshold_95 and n_potential_outliers use 2.7 p, not
        // qchisq(0.95, p); not pinned.
        Check::new(&case)
            .scalar("n_complete", num(&v["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(v["missing"], summary(30, 29, serde_json::json!({"a": 1})));
    }

    #[test]
    fn test_validate_mcp_stats_robust_summaries_na() {
        // TOOL-3/X-2: the legacy extraction read nulls as NaN: stats_mad and stats_density
        // counted the missing cells in n, nothing reported the drop, and text columns gave the
        // legacy "is not numeric (type str)" message. R: the functions with na.rm = TRUE.
        let (case, rt, server, name) = case_server("stat_descriptive_na");
        let call = |tool: &str, extra: serde_json::Value| {
            let mut a = serde_json::json!({"dataset": name});
            if let serde_json::Value::Object(m) = extra {
                a.as_object_mut().unwrap().extend(m);
            }
            rt.block_on(async {
                match tool {
                    "fivenum" => server.stats_fivenum(req(a)).await,
                    "iqr" => server.stats_iqr(req(a)).await,
                    "mad" => server.stats_mad(req(a)).await,
                    "ecdf" => server.stats_ecdf(req(a)).await,
                    "density" => server.stats_density(req(a)).await,
                    other => panic!("no tool {other}"),
                }
            })
            .unwrap()
        };
        let five = |v: &serde_json::Value| -> Vec<f64> {
            ["minimum", "lower_hinge", "median", "upper_hinge", "maximum"]
                .iter()
                .map(|k| num(&v[*k]))
                .collect()
        };
        let mut check = Check::new(&case);
        for (col, sfx, n_key, missing) in [
            (
                "v",
                "v",
                "n_v",
                summary(15, 12, serde_json::json!({"v": 3})),
            ),
            (
                "vi",
                "vi",
                "n_vi",
                summary(15, 13, serde_json::json!({"vi": 2})),
            ),
        ] {
            let c = serde_json::json!({"column": col});
            let f = json_of(&call("fivenum", c.clone()));
            let q = json_of(&call("iqr", c.clone()));
            let m = json_of(&call("mad", c.clone()));
            let e = json_of(&call(
                "ecdf",
                serde_json::json!({"column": col, "at": [15, 20, 25]}),
            ));
            let d = json_of(&call("density", c.clone()));
            // deviation STAT-36: IQR types other than 7 are computed as type 7; only the
            // default is pinned.
            // deviation STAT-37: the density grid is mislabelled when n is not a power of
            // two.
            // deviation STAT-65: the default bandwidth is not bw.nrd0; only the density's n
            // is pinned.
            // deviation STAT-87: fivenum's `iqr` field is the hinge spread, not R's IQR,
            // and is not pinned.
            check
                .vector(&format!("fivenum_{sfx}"), &five(&f), Tol::coef(12))
                .scalar(n_key, num(&f["n"]), Tol::exact())
                .scalar(&format!("iqr_{sfx}"), num(&q["iqr"]), Tol::stat(12))
                .scalar(&format!("mad_{sfx}"), num(&m["mad"]), Tol::stat(12))
                .scalar(n_key, num(&m["n"]), Tol::exact())
                .scalar(n_key, num(&e["n"]), Tol::exact())
                .scalar(n_key, num(&d["data_n"]), Tol::exact());
            if col == "v" {
                let mm = json_of(&call(
                    "mad",
                    serde_json::json!({"column": col, "center": "mean"}),
                ));
                check
                    .scalar("mad_mean_v", num(&mm["mad"]), Tol::stat(12))
                    .scalar("mean_v", num(&mm["center_used"]), Tol::coef(12))
                    .vector("ecdf_v_at", &nums(&e["y"]), Tol::coef(12));
                assert_eq!(mm["missing"], missing, "mad center = mean");
            }
            for (tool, v) in [
                ("fivenum", &f),
                ("iqr", &q),
                ("mad", &m),
                ("ecdf", &e),
                ("density", &d),
            ] {
                assert_eq!(v["missing"], missing, "{tool} on {col}");
            }
        }
        check.finish();
        for tool in ["fivenum", "iqr", "mad", "ecdf", "density"] {
            let e = err_text(&call(tool, serde_json::json!({"column": "s"})));
            assert!(
                e.contains("Column 's' contains non-numeric values"),
                "{tool}: {e}"
            );
            let e = err_text(&call(tool, serde_json::json!({"column": "nope"})));
            assert!(
                e.contains("Column 'nope' not found in dataset."),
                "{tool}: {e}"
            );
        }
    }

    #[test]
    fn test_validate_mcp_stats_spline_approx_na() {
        // TOOL-25 (missing values)/X-2: stats_spline failed with "non-finite values" on any
        // incomplete pair, and nothing reported the drop. R: approx and spline drop the
        // incomplete (x, y) pairs (regularize.values).
        let (case, rt, server, name) = case_server("stat_descriptive_na");
        let args = serde_json::json!({"dataset": name, "x": "xs", "y": "ys",
                                      "xout": [1.5, 4, 6.5, 8]});
        let a = json_of(&rt.block_on(server.stats_approx(req(args.clone()))).unwrap());
        let mut sargs = args;
        sargs["method"] = serde_json::json!("natural");
        // deviation STAT-69: the default method fmm is not R's FMM spline; only natural is pinned
        let s = json_of(&rt.block_on(server.stats_spline(req(sargs))).unwrap());
        Check::new(&case)
            .vector("approx_y", &nums(&a["y"]), Tol::stat(13))
            .vector("spline_natural_y", &nums(&s["y"]), Tol::stat(13))
            .scalar("n_pairs", num(&s["n_knots"]), Tol::exact())
            .scalar("n_pairs", num(&a["missing"]["n_used"]), Tol::exact())
            .finish();
        assert_eq!(a["n_interpolated"], serde_json::json!(4));
        let missing = summary(15, 13, serde_json::json!({"xs": 1, "ys": 1}));
        assert_eq!(a["missing"], missing);
        assert_eq!(s["missing"], missing);
    }

    #[test]
    fn test_validate_mcp_descriptive_isoreg_int() {
        // X-2: a missing response was read through into_no_null_iter after a cast and entered
        // the fit. R: isoreg(x, y) on an Int64 response; isoreg refuses missing values.
        let (case, rt, server, name) = case_server("stat_isoreg_medpolish_int");
        let v = json_of(
            &rt.block_on(server.descriptive_isoreg(req(serde_json::json!(
                {"dataset": name, "x_column": "x", "y_column": "y"}))))
                .unwrap(),
        );
        // deviation STAT-52: the knot positions (p2a: first index of each block, R: the last)
        // are not pinned, only their number.
        Check::new(&case)
            .vector("isoreg_yf", &nums(&v["fitted_values"]), Tol::coef(10))
            .scalar("isoreg_n_knots", num(&v["n_knots"]), Tol::exact())
            .finish();
        assert_eq!(v["n_observations"], serde_json::json!(10));
        let e = err_text(
            &rt.block_on(server.descriptive_isoreg(req(serde_json::json!(
                {"dataset": name, "x_column": "x", "y_column": "y_na"}))))
                .unwrap(),
        );
        assert!(
            e.contains("Column 'y_na' contains 1 missing value(s)"),
            "{e}"
        );
        let v = json_of(
            &rt.block_on(server.descriptive_isoreg(req(serde_json::json!(
                {"dataset": name, "y_column": "y"}))))
                .unwrap(),
        );
        assert_eq!(v["n_observations"], serde_json::json!(10));
    }

    #[test]
    fn test_validate_mcp_descriptive_medpolish_int() {
        // X-8: every Int64 column was refused ("Column 'c1' is not numeric."). R: medpolish on
        // the integer matrix, na.rm = TRUE with the missing cell of c3, and on c1, c2, c4.
        let (case, rt, server, name) = case_server("stat_isoreg_medpolish_int");
        let call = |columns: serde_json::Value, na_rm: Option<bool>| {
            let mut a = serde_json::json!({"dataset": name, "columns": columns});
            if let Some(b) = na_rm {
                a["na_rm"] = serde_json::json!(b);
            }
            rt.block_on(server.descriptive_medpolish(req(a))).unwrap()
        };
        let p6 = Tol::printed(6);
        let mut check = Check::new(&case);
        for (prefix, columns, na_rm) in [
            (
                "mp",
                serde_json::json!(["c1", "c2", "c3", "c4"]),
                Some(true),
            ),
            ("mp3", serde_json::json!(["c1", "c2", "c4"]), None),
        ] {
            let v = json_of(&call(columns, na_rm));
            let rows: Vec<f64> = v["row_effects"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| str_num(&r["effect"]))
                .collect();
            let cols = v["column_effects"].as_array().unwrap();
            let names: Vec<&str> = cols.iter().map(|c| c["column"].as_str().unwrap()).collect();
            let effects: Vec<f64> = cols.iter().map(|c| str_num(&c["effect"])).collect();
            check
                .scalar(&format!("{prefix}_overall"), str_num(&v["overall"]), p6)
                .vector(&format!("{prefix}_row"), &rows, p6)
                .named(&format!("{prefix}_col"), &names, &effects, p6);
        }
        check.finish();
        let e = err_text(&call(serde_json::json!(["c1", "c2", "c3", "c4"]), None));
        assert!(e.contains("na_rm"), "{e}");
    }
}
