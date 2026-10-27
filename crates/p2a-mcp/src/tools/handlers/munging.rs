//! Data munging tool handlers.
//!
//! This module defines data munging tool handlers using the `#[tool_router(router = munging_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};
use std::collections::HashMap;

use crate::server::AnalyticsServer;
use crate::tools::limits::{self, Family, LimitViolation};
use crate::tools::requests::munging::*;

use p2a_core::{
    data::{
        Dataset,
        munging::{
            AggFn,
            AggSpec,
            ArithOp,
            BinStrategy,
            FillStrategy,
            JoinType,
            MungeError,
            MutateExpr,
            bin,
            concat,
            deduplicate,
            diff,
            drop_columns,
            // Clean operations
            drop_na,
            estimate_join_rows,
            fill_na,
            // Transform operations
            filter,
            // Aggregate operations
            group_by,
            // Join operations
            join,
            // Feature engineering
            lag,
            lead,
            melt,
            mutate,
            normalize,
            one_hot_encode,
            pct_change,
            // Reshape operations
            pivot,
            regex_count,
            regex_extract,
            regex_replace,
            regex_replace_bytes,
            rename,
            replace,
            sample,
            select,
            sort,
            standardize,
            str_concat,
            str_length,
            str_split,
            str_split_part_count,
            str_substring,
            to_lowercase,
            to_uppercase,
            // String operations
            trim,
            value_counts,
        },
    },
    regression::{CovarianceType, run_ols},
    stats::correlation_matrix,
};

#[tool_router(router = munging_router, vis = "pub")]
impl AnalyticsServer {
    /// Batch process multiple datasets with the same operation.
    #[tool(
        description = "Run the same analysis (describe, correlation, or OLS regression) on multiple datasets at once. Useful for comparing results across datasets or processing survey waves."
    )]
    pub async fn batch_process(
        &self,
        Parameters(request): Parameters<BatchProcessRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        if request.datasets.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one dataset must be specified".to_string(),
            )]));
        }

        let datasets = self.datasets.read().await;
        let mut results = Vec::new();
        let mut combined_stats: Option<Vec<(String, Vec<f64>)>> =
            if request.combine_results.unwrap_or(false) {
                Some(Vec::new())
            } else {
                None
            };

        for ds_name in &request.datasets {
            let dataset = match datasets.get(ds_name) {
                Some(ds) => ds,
                None => {
                    results.push(format!("Dataset '{}': NOT FOUND", ds_name));
                    continue;
                }
            };

            let df = dataset.df();
            let result = match request.operation.to_lowercase().as_str() {
                "describe" => {
                    // Get summary statistics
                    let columns: Vec<String> = if let Some(ref cols) = request.columns {
                        cols.clone()
                    } else {
                        // Get all numeric columns
                        df.get_columns()
                            .iter()
                            .filter(|c| c.dtype().is_primitive_numeric())
                            .map(|c| c.name().to_string())
                            .collect()
                    };

                    let mut stats_output = format!("Dataset: {}\n", ds_name);
                    stats_output.push_str(&format!("{:-<60}\n", ""));

                    for col_name in &columns {
                        if let Ok(col) = df.column(col_name) {
                            if let Ok(casted) = col.cast(&DataType::Float64) {
                                if let Ok(arr) = casted.f64() {
                                    let values: Vec<f64> = arr.into_iter().flatten().collect();
                                    if !values.is_empty() {
                                        let n = values.len();
                                        let sum: f64 = values.iter().sum();
                                        let mean = sum / n as f64;
                                        let variance: f64 =
                                            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                                / (n - 1).max(1) as f64;
                                        let std_dev = variance.sqrt();
                                        let min =
                                            values.iter().copied().fold(f64::INFINITY, f64::min);
                                        let max = values
                                            .iter()
                                            .copied()
                                            .fold(f64::NEG_INFINITY, f64::max);

                                        stats_output.push_str(&format!(
                                            "  {}: n={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}\n",
                                            col_name, n, mean, std_dev, min, max
                                        ));

                                        if let Some(ref mut combined) = combined_stats {
                                            combined.push((
                                                format!("{}:{}", ds_name, col_name),
                                                vec![n as f64, mean, std_dev, min, max],
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    stats_output
                }
                "correlation" => {
                    // Get correlation matrix
                    match correlation_matrix(dataset) {
                        Ok(corr) => {
                            format!("Dataset: {}\n{:?}", ds_name, corr)
                        }
                        Err(e) => format!("Dataset '{}': Error - {}", ds_name, e),
                    }
                }
                "ols" => {
                    // Run OLS regression
                    if let Some(ref cols) = request.columns {
                        if cols.len() < 2 {
                            format!(
                                "Dataset '{}': OLS requires at least 2 columns (dependent + independent)",
                                ds_name
                            )
                        } else {
                            let y_col = &cols[0];
                            let x_cols: Vec<&str> = cols[1..].iter().map(|s| s.as_str()).collect();

                            match run_ols(dataset, y_col, &x_cols, true, CovarianceType::HC1) {
                                Ok(ols_result) => {
                                    let mut output = format!("Dataset: {}\n{:-<60}\n", ds_name, "");
                                    output.push_str(&format!(
                                        "R²: {:.4}, Adj R²: {:.4}\n",
                                        ols_result.r_squared, ols_result.adj_r_squared
                                    ));
                                    output.push_str(&format!(
                                        "F-stat: {:.4}\n",
                                        ols_result.f_statistic
                                    ));
                                    for coef in &ols_result.coefficients {
                                        output.push_str(&format!(
                                            "  {}: coef={:.4}, se={:.4}, t={:.4}, p={:.4}\n",
                                            coef.name,
                                            coef.estimate,
                                            coef.std_error,
                                            coef.t_value,
                                            coef.p_value
                                        ));
                                    }
                                    output
                                }
                                Err(e) => format!("Dataset '{}': Error - {}", ds_name, e),
                            }
                        }
                    } else {
                        format!(
                            "Dataset '{}': OLS requires columns to be specified",
                            ds_name
                        )
                    }
                }
                other => format!(
                    "Unknown operation: '{}'. Use 'describe', 'correlation', or 'ols'.",
                    other
                ),
            };

            results.push(result);
        }

        let mut output = format!("Batch Processing Results\n{}\n\n", "=".repeat(40));
        output.push_str(&format!("Datasets processed: {}\n", request.datasets.len()));
        output.push_str(&format!("Operation: {}\n\n", request.operation));

        for result in results {
            output.push_str(&result);
            output.push_str("\n\n");
        }

        // Add combined summary if requested
        if let Some(combined) = combined_stats {
            if !combined.is_empty() {
                output.push_str(&format!("Combined Summary\n{}\n", "-".repeat(40)));
                for (name, stats) in combined {
                    output.push_str(&format!(
                        "{}: n={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}\n",
                        name, stats[0] as usize, stats[1], stats[2], stats[3], stats[4]
                    ));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Compare the same columns across multiple datasets.
    #[tool(
        description = "Compare statistics for specific columns across multiple datasets. Useful for comparing distributions, means, and correlations between different datasets (e.g., treatment vs control, before vs after)."
    )]
    pub async fn compare_datasets(
        &self,
        Parameters(request): Parameters<CompareDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        if request.datasets.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least two datasets must be specified for comparison".to_string(),
            )]));
        }

        if request.columns.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one column must be specified for comparison".to_string(),
            )]));
        }

        let datasets = self.datasets.read().await;
        let comparison_type = request.comparison_type.as_deref().unwrap_or("summary");

        let mut output = format!("Dataset Comparison\n{}\n\n", "=".repeat(40));
        output.push_str(&format!("Datasets: {:?}\n", request.datasets));
        output.push_str(&format!("Columns: {:?}\n", request.columns));
        output.push_str(&format!("Comparison type: {}\n\n", comparison_type));

        // Collect statistics for each dataset and column
        let mut all_stats: HashMap<String, HashMap<String, (usize, f64, f64, f64, f64)>> =
            HashMap::new();

        for ds_name in &request.datasets {
            let dataset = match datasets.get(ds_name) {
                Some(ds) => ds,
                None => {
                    output.push_str(&format!("Warning: Dataset '{}' not found\n", ds_name));
                    continue;
                }
            };

            let df = dataset.df();
            let mut ds_stats: HashMap<String, (usize, f64, f64, f64, f64)> = HashMap::new();

            for col_name in &request.columns {
                if let Ok(col) = df.column(col_name) {
                    if let Ok(casted) = col.cast(&DataType::Float64) {
                        if let Ok(arr) = casted.f64() {
                            let values: Vec<f64> = arr.into_iter().flatten().collect();
                            if !values.is_empty() {
                                let n = values.len();
                                let sum: f64 = values.iter().sum();
                                let mean = sum / n as f64;
                                let variance: f64 =
                                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                        / (n - 1).max(1) as f64;
                                let std_dev = variance.sqrt();
                                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                ds_stats.insert(col_name.clone(), (n, mean, std_dev, min, max));
                            }
                        }
                    }
                }
            }

            all_stats.insert(ds_name.clone(), ds_stats);
        }

        match comparison_type {
            "summary" => {
                // Side-by-side comparison table
                for col_name in &request.columns {
                    output.push_str(&format!("\nColumn: {}\n{}\n", col_name, "-".repeat(60)));
                    output.push_str(&format!(
                        "{:<20} {:>10} {:>12} {:>12} {:>12} {:>12}\n",
                        "Dataset", "N", "Mean", "Std Dev", "Min", "Max"
                    ));
                    output.push_str(&format!("{}\n", "-".repeat(80)));

                    for ds_name in &request.datasets {
                        if let Some(ds_stats) = all_stats.get(ds_name) {
                            if let Some((n, mean, std, min, max)) = ds_stats.get(col_name) {
                                output.push_str(&format!(
                                    "{:<20} {:>10} {:>12.4} {:>12.4} {:>12.4} {:>12.4}\n",
                                    ds_name, n, mean, std, min, max
                                ));
                            } else {
                                output.push_str(&format!(
                                    "{:<20} Column not found or not numeric\n",
                                    ds_name
                                ));
                            }
                        }
                    }

                    // Calculate and show differences between first two datasets
                    if request.datasets.len() >= 2 {
                        let ds1 = &request.datasets[0];
                        let ds2 = &request.datasets[1];
                        if let (Some(stats1), Some(stats2)) = (
                            all_stats.get(ds1).and_then(|s| s.get(col_name)),
                            all_stats.get(ds2).and_then(|s| s.get(col_name)),
                        ) {
                            let mean_diff = stats2.1 - stats1.1;
                            let pct_diff = if stats1.1.abs() > 1e-10 {
                                (mean_diff / stats1.1) * 100.0
                            } else {
                                f64::NAN
                            };
                            output.push_str(&format!(
                                "\nDifference ({} - {}): mean diff = {:.4} ({:.2}%)\n",
                                ds2, ds1, mean_diff, pct_diff
                            ));
                        }
                    }
                }
            }
            "distribution" => {
                // Distribution comparison (basic)
                for col_name in &request.columns {
                    output.push_str(&format!(
                        "\nColumn: {} - Distribution Comparison\n{}\n",
                        col_name,
                        "-".repeat(60)
                    ));

                    for ds_name in &request.datasets {
                        if let Some(ds_stats) = all_stats.get(ds_name) {
                            if let Some((n, mean, std, min, max)) = ds_stats.get(col_name) {
                                let range = max - min;
                                let cv = if mean.abs() > 1e-10 {
                                    std / mean.abs()
                                } else {
                                    f64::NAN
                                };
                                output.push_str(&format!(
                                    "{}: n={}, range={:.4}, CV={:.4}\n",
                                    ds_name, n, range, cv
                                ));
                            }
                        }
                    }
                }
            }
            "correlation" => {
                // Correlation comparison (if multiple columns)
                if request.columns.len() < 2 {
                    output.push_str("Correlation comparison requires at least 2 columns\n");
                } else {
                    for ds_name in &request.datasets {
                        if let Some(dataset) = datasets.get(ds_name) {
                            match correlation_matrix(dataset) {
                                Ok(corr) => {
                                    output.push_str(&format!("\n{}\n{:?}\n", ds_name, corr));
                                }
                                Err(e) => {
                                    output.push_str(&format!("\n{}: Error - {}\n", ds_name, e));
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                output.push_str(&format!("Unknown comparison type: '{}'. Use 'summary', 'distribution', or 'correlation'.\n", comparison_type));
            }
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Filter rows in a dataset based on a condition.
    #[tool(
        description = "Filter rows in a dataset based on a column condition. Supports operators: 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'contains', 'starts_with', 'ends_with'. The value is parsed based on the column type."
    )]
    pub async fn munge_filter(
        &self,
        Parameters(request): Parameters<FilterDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let result = match filter(dataset, &request.column, &request.op, &request.value) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Filter failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Filtered dataset saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Select specific columns from a dataset.
    #[tool(description = "Select (keep) specific columns from a dataset, dropping all others.")]
    pub async fn munge_select(
        &self,
        Parameters(request): Parameters<SelectColumnsRequest>,
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

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let result = match select(dataset, &cols) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Select failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Selected {} columns, saved as '{}' ({} rows)",
            n_cols, result_name, n_rows
        ))]))
    }

    /// Drop columns from a dataset.
    #[tool(description = "Drop (remove) specific columns from a dataset.")]
    pub async fn munge_drop_columns(
        &self,
        Parameters(request): Parameters<DropColumnsRequest>,
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

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let result = match drop_columns(dataset, &cols) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Drop columns failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Dropped {} columns, saved as '{}' ({} rows, {} columns remaining)",
            request.columns.len(),
            result_name,
            n_rows,
            n_cols
        ))]))
    }

    /// Rename columns in a dataset.
    #[tool(description = "Rename columns in a dataset. Provide pairs of [old_name, new_name].")]
    pub async fn munge_rename(
        &self,
        Parameters(request): Parameters<RenameColumnsRequest>,
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

        let renames: Vec<(&str, &str)> = request
            .renames
            .iter()
            .filter_map(|pair| {
                if pair.len() >= 2 {
                    Some((pair[0].as_str(), pair[1].as_str()))
                } else {
                    None
                }
            })
            .collect();

        let result = match rename(dataset, &renames) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Rename failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Renamed {} columns, saved as '{}'",
            renames.len(),
            result_name
        ))]))
    }

    /// Sort a dataset by one or more columns.
    #[tool(description = "Sort a dataset by one or more columns in ascending or descending order.")]
    pub async fn munge_sort(
        &self,
        Parameters(request): Parameters<SortDatasetRequest>,
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

        let by_cols: Vec<&str> = request.by.iter().map(|s| s.as_str()).collect();
        let descending = request.descending.unwrap_or(false);
        let descending_flags: Vec<bool> = vec![descending; by_cols.len()];

        let result = match sort(dataset, &by_cols, &descending_flags) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sort failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Sorted by {:?} ({}), saved as '{}' ({} rows)",
            request.by,
            if descending {
                "descending"
            } else {
                "ascending"
            },
            result_name,
            n_rows
        ))]))
    }

    /// Join two datasets on key columns.
    #[tool(
        description = "Join two datasets on key columns. Supports 'left', 'right', 'inner', and 'full' join types."
    )]
    pub async fn munge_join(
        &self,
        Parameters(request): Parameters<JoinDatasetsRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Owned copies (polars columns are reference-counted): the read lock is
        // released before the join is estimated and computed.
        let (left_ds, right_ds) = {
            let datasets = self.datasets.read().await;
            (
                datasets.get(&request.left).cloned(),
                datasets.get(&request.right).cloned(),
            )
        };

        let left_ds = match left_ds {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Left dataset '{}' not found.",
                    request.left
                ))]));
            }
        };

        let right_ds = match right_ds {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Right dataset '{}' not found.",
                    request.right
                ))]));
            }
        };

        let left_on: Vec<&str> = request.left_on.iter().map(|s| s.as_str()).collect();
        let right_on_vec: Option<Vec<&str>> = request
            .right_on
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let right_on: Option<&[&str]> = right_on_vec.as_deref();
        let suffix: Option<&str> = request.suffix.as_deref();

        let join_type_name = request.join_type.as_deref().unwrap_or("left");
        let join_type = match join_type_name {
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "inner" => JoinType::Inner,
            "full" => JoinType::Full,
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown join type: '{}'. Use 'left', 'right', 'inner', or 'full'.",
                    join_type_name
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| format!("{}_{}", request.left, request.right));

        // With a join_output_rows budget or dataset quotas, predict the
        // result size from per-key counts before joining: its rows, and what
        // the rows of both datasets' columns take.
        let limits = limits::current();
        if limits.join_output_rows.is_some() || bound::active() {
            let rows = match estimate_join_rows(&left_ds, &right_ds, &left_on, right_on, join_type)
            {
                Ok(rows) => rows,
                Err(e) => return Ok(join_failed(&e)),
            };
            if let Err(v) = limits.check_budget(
                "munge_join",
                "join_output_rows",
                rows,
                limits.join_output_rows,
            ) {
                return Ok(refused(v));
            }
            if bound::active() {
                let bytes = bound::join(
                    left_ds.df(),
                    right_ds.df(),
                    rows,
                    suffix.unwrap_or("_right"),
                );
                bound::admit(
                    "munge_join",
                    &result_name,
                    bytes,
                    || format!("{rows} joined rows (`left_on`, `join_type` {join_type_name})"),
                    "join on keys that repeat less, select fewer columns first or start a new \
                     session",
                )
                .await?;
            }
        }

        let result = match join(&left_ds, &right_ds, &left_on, right_on, join_type, suffix) {
            Ok(ds) => ds,
            Err(e) => return Ok(join_failed(&e)),
        };
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} join completed, saved as '{}' ({} rows, {} columns)",
            join_type_name, result_name, n_rows, n_cols
        ))]))
    }

    /// Concatenate multiple datasets vertically.
    #[tool(
        description = "Concatenate (row-bind) multiple datasets vertically. All datasets must have the same columns."
    )]
    pub async fn munge_concat(
        &self,
        Parameters(request): Parameters<ConcatDatasetsRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Owned copies (polars columns are reference-counted): the read lock
        // is released before the result is bounded and built.
        let mut owned: Vec<Dataset> = Vec::with_capacity(request.datasets.len());
        {
            let datasets = self.datasets.read().await;
            for name in &request.datasets {
                match datasets.get(name) {
                    Some(ds) => owned.push(ds.clone()),
                    None => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Dataset '{}' not found.",
                            name
                        ))]));
                    }
                }
            }
        }
        let ds_list: Vec<&Dataset> = owned.iter().collect();

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| "concatenated".to_string());

        // A dataset may be listed many times, and a column one dataset
        // lacks is filled with nulls in it.
        if bound::active() {
            let frames: Vec<_> = ds_list.iter().map(|ds| ds.df()).collect();
            let bytes = bound::concat(&frames);
            bound::admit(
                "munge_concat",
                &result_name,
                bytes,
                || {
                    let rows: usize = ds_list.iter().map(|ds| ds.nrows()).sum();
                    format!(
                        "row-binding the {} `datasets` ({rows} rows)",
                        request.datasets.len()
                    )
                },
                "list fewer datasets, filter their rows first or start a new session",
            )
            .await?;
        }

        let result = match concat(&ds_list) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Concat failed: {}",
                    e
                ))]));
            }
        };

        let n_rows = result.nrows();
        let n_cols = result.ncols();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Concatenated {} datasets, saved as '{}' ({} rows, {} columns)",
            request.datasets.len(),
            result_name,
            n_rows,
            n_cols
        ))]))
    }

    /// Group by columns and compute aggregations.
    #[tool(
        description = "Group a dataset by columns and compute aggregations. Supported functions: 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'first', 'last'."
    )]
    pub async fn munge_group_by(
        &self,
        Parameters(request): Parameters<GroupByRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let by_cols: Vec<&str> = request.by.iter().map(|s| s.as_str()).collect();

        // Parse aggregation specs
        let mut agg_specs: Vec<AggSpec> = Vec::new();
        for spec in &request.aggs {
            if spec.len() >= 2 {
                let col = &spec[0];
                let func_str = spec[1].to_lowercase();
                let agg_fn = match func_str.as_str() {
                    "count" => AggFn::Count,
                    "sum" => AggFn::Sum,
                    "mean" => AggFn::Mean,
                    "median" => AggFn::Median,
                    "min" => AggFn::Min,
                    "max" => AggFn::Max,
                    "std" => AggFn::Std,
                    "var" => AggFn::Var,
                    "first" => AggFn::First,
                    "last" => AggFn::Last,
                    _ => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Unknown aggregation function: '{}'. Use: count, sum, mean, median, min, max, std, var, first, last.",
                            func_str
                        ))]));
                    }
                };
                agg_specs.push(AggSpec::new(col, agg_fn));
            }
        }

        if agg_specs.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one aggregation spec is required. Format: [[\"column\", \"function\"], ...]".to_string()
            )]));
        }

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| format!("{}_grouped", request.dataset));

        // One column of every group per aggregation, which `aggs` may list
        // many of.
        if bound::active()
            && let Some((bytes, groups)) = bound::group_by(dataset.df(), &by_cols, &agg_specs)
        {
            bound::admit(
                "munge_group_by",
                &result_name,
                bytes,
                || format!("{} `aggs` over up to {groups} groups", agg_specs.len()),
                "list fewer aggregations, group by fewer distinct values, filter the rows first \
                 or start a new session",
            )
            .await?;
        }

        let result = match group_by(&dataset, &by_cols, &agg_specs) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Group by failed: {}",
                    e
                ))]));
            }
        };

        let n_rows = result.nrows();
        let n_cols = result.ncols();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Grouped by {:?} with {} aggregations, saved as '{}' ({} groups, {} columns)",
            request.by,
            agg_specs.len(),
            result_name,
            n_rows,
            n_cols
        ))]))
    }

    /// Compute value counts for a column.
    #[tool(
        description = "Compute frequency counts for unique values in a column. Optionally normalize to percentages."
    )]
    pub async fn munge_value_counts(
        &self,
        Parameters(request): Parameters<ValueCountsRequest>,
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

        let result = match value_counts(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Value counts failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_value_counts", request.column));
        let n_rows = result.nrows();

        // Format output for display
        let mut output = format!(
            "Value Counts for '{}'\n{}\n",
            request.column,
            "=".repeat(40)
        );

        // Show first few rows
        let show_n = 10.min(n_rows);
        output.push_str(&format!(
            "Showing top {} of {} unique values:\n\n",
            show_n, n_rows
        ));

        let df = result.df();
        for i in 0..show_n {
            let val_col = df.column(&request.column).ok();
            let count_col = df.column("count").ok();

            if let (Some(v), Some(c)) = (val_col, count_col) {
                if let (Ok(val), Ok(cnt)) = (v.get(i), c.get(i)) {
                    output.push_str(&format!("  {:?}: {}\n", val, cnt));
                }
            }
        }

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        output.push_str(&format!("\nFull result saved as '{}'", result_name));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Pivot a dataset from long to wide format.
    #[tool(
        description = "Pivot a dataset from long to wide format. Index columns remain as rows, 'on' column values become new column names, and 'values' column fills those columns."
    )]
    pub async fn munge_pivot(
        &self,
        Parameters(request): Parameters<PivotDatasetRequest>,
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

        let index: Vec<&str> = request.index.iter().map(|s| s.as_str()).collect();

        // The wide table's columns and cells were checked against the pivot budgets before this
        // handler ran (`Limits::check_pivot`, `PIVOT_RULES`), on every governed path and stdio,
        // so that a refusal is a 422 `limit_exceeded` like the other request limits.
        let result = match pivot(dataset, &index, &request.on, &request.values) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Pivot failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_pivoted", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Pivoted to wide format, saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Melt a dataset from wide to long format.
    #[tool(
        description = "Melt a dataset from wide to long format. ID variables remain as-is, value variables are unpivoted into rows."
    )]
    pub async fn munge_melt(
        &self,
        Parameters(request): Parameters<MeltDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let id_vars: Vec<&str> = request.id_vars.iter().map(|s| s.as_str()).collect();
        let value_vars: Vec<&str> = request.value_vars.iter().map(|s| s.as_str()).collect();
        let variable_name = request.variable_name.as_deref().unwrap_or("variable");
        let value_name = request.value_name.as_deref().unwrap_or("value");
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| format!("{}_melted", request.dataset));

        // Every row once per value variable, its id columns repeated.
        if bound::active()
            && let Some(bytes) = bound::melt(
                dataset.df(),
                &id_vars,
                &value_vars,
                variable_name,
                value_name,
            )
        {
            bound::admit(
                "munge_melt",
                &result_name,
                bytes,
                || {
                    format!(
                        "{} rows times {} `value_vars`, each with {} `id_vars`",
                        dataset.nrows(),
                        value_vars.len(),
                        id_vars.len()
                    )
                },
                "list fewer value_vars or id_vars, filter the rows first or start a new session",
            )
            .await?;
        }

        let result = match melt(&dataset, &id_vars, &value_vars, variable_name, value_name) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Melt failed: {}",
                    e
                ))]));
            }
        };

        let n_rows = result.nrows();
        let n_cols = result.ncols();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Melted to long format, saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Drop rows with null values.
    #[tool(
        description = "Drop rows containing null values. Use 'any' to drop if any column is null, 'all' to drop only if all columns are null."
    )]
    pub async fn munge_drop_na(
        &self,
        Parameters(request): Parameters<DropNaRequest>,
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

        let columns: Option<Vec<&str>> = request
            .columns
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let how = request.how.as_deref().unwrap_or("any");

        let orig_rows = dataset.nrows();
        let result = match drop_na(dataset, columns.as_deref(), how) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Drop NA failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let dropped = orig_rows - n_rows;

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Dropped {} rows with null values, saved as '{}' ({} rows remaining)",
            dropped, result_name, n_rows
        ))]))
    }

    /// Fill null values using a strategy.
    #[tool(
        description = "Fill null values using a strategy: 'mean', 'median', 'mode', 'forward', 'backward', or a constant value."
    )]
    pub async fn munge_fill_na(
        &self,
        Parameters(request): Parameters<FillNaRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Option<Vec<&str>> = request
            .columns
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let strategy = match request.strategy.to_lowercase().as_str() {
            "mean" => FillStrategy::Mean,
            "median" => FillStrategy::Median,
            "forward" => FillStrategy::Forward,
            "backward" => FillStrategy::Backward,
            "zero" => FillStrategy::Zero,
            val => {
                // Try to use as a constant value string
                FillStrategy::Constant(val.to_string())
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // A constant written into every null of a String column, or a mean
        // that widens a column to Float64, can make the result far larger.
        if bound::active()
            && let Some(bytes) = bound::fill_na(dataset.df(), columns.as_deref(), &strategy)
        {
            bound::admit(
                "munge_fill_na",
                &result_name,
                bytes,
                || {
                    let df = dataset.df();
                    let nulls: usize = match columns.as_deref() {
                        Some(names) => names
                            .iter()
                            .filter_map(|n| df.column(n).ok())
                            .map(|c| c.null_count())
                            .sum(),
                        None => df.get_columns().iter().map(|c| c.null_count()).sum(),
                    };
                    match &strategy {
                        FillStrategy::Constant(value) => format!(
                            "filling {nulls} nulls with the {}-byte constant `strategy`",
                            value.len()
                        ),
                        FillStrategy::Mean | FillStrategy::Median => format!(
                            "filling {nulls} nulls with the `strategy` {}, which makes each \
                             numeric column Float64",
                            request.strategy.to_lowercase()
                        ),
                        _ => format!(
                            "filling {nulls} nulls with the `strategy` {}",
                            request.strategy.to_lowercase()
                        ),
                    }
                },
                "fill fewer columns, use a shorter constant, filter the rows first or start a \
                 new session",
            )
            .await?;
        }

        let result = match fill_na(&dataset, columns.as_deref(), strategy) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fill NA failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Filled null values with strategy '{}', saved as '{}'",
            request.strategy, result_name
        ))]))
    }

    /// Remove duplicate rows.
    #[tool(
        description = "Remove duplicate rows from a dataset. Specify which duplicate to keep: 'first', 'last', or 'none'."
    )]
    pub async fn munge_deduplicate(
        &self,
        Parameters(request): Parameters<DeduplicateRequest>,
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

        let columns: Option<Vec<&str>> = request
            .columns
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let keep = request.keep.as_deref().unwrap_or("first");

        let orig_rows = dataset.nrows();
        let result = match deduplicate(dataset, columns.as_deref(), keep) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Deduplicate failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let removed = orig_rows - n_rows;

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Removed {} duplicate rows, saved as '{}' ({} rows remaining)",
            removed, result_name, n_rows
        ))]))
    }

    /// Trim whitespace from string columns.
    #[tool(
        description = "Trim leading and trailing whitespace from string columns. If no columns specified, trims all string columns."
    )]
    pub async fn str_trim(
        &self,
        Parameters(request): Parameters<TrimRequest>,
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

        let columns: Option<Vec<&str>> = request
            .columns
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = match trim(dataset, columns.as_deref()) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Trim failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        let cols_desc = request
            .columns
            .as_ref()
            .map(|c| c.join(", "))
            .unwrap_or_else(|| "all string columns".to_string());

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Trimmed whitespace from {}, saved as '{}'",
            cols_desc, result_name
        ))]))
    }

    /// Convert string column to lowercase.
    #[tool(description = "Convert all characters in a string column to lowercase.")]
    pub async fn str_to_lowercase(
        &self,
        Parameters(request): Parameters<ToLowercaseRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());
        if let Err(refused) =
            admit_case_mapped(&dataset, &request.column, &result_name, false).await
        {
            return refused;
        }

        let result = match to_lowercase(&dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To lowercase failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Converted '{}' to lowercase, saved as '{}'",
            request.column, result_name
        ))]))
    }

    /// Convert string column to uppercase.
    #[tool(description = "Convert all characters in a string column to uppercase.")]
    pub async fn str_to_uppercase(
        &self,
        Parameters(request): Parameters<ToUppercaseRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());
        if let Err(refused) = admit_case_mapped(&dataset, &request.column, &result_name, true).await
        {
            return refused;
        }

        let result = match to_uppercase(&dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To uppercase failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Converted '{}' to uppercase, saved as '{}'",
            request.column, result_name
        ))]))
    }

    /// Replace exact values in a column.
    #[tool(
        description = "Replace exact values in a column with a new value. Use to clean sentinel values like -99, 'N/A', or 'missing' by replacing them (use null as new_value to convert to missing). For pattern-based replacement, use str_regex_replace. Call this directly — do NOT call head_dataset or describe_dataset first."
    )]
    pub async fn str_replace_value(
        &self,
        Parameters(request): Parameters<ReplaceValueRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // Each matching value of a String column becomes `new_value`.
        if bound::active() {
            let bounded = match bound::replace_value(
                dataset.df(),
                &request.column,
                &request.old_value,
                &request.new_value,
            ) {
                Ok(bounded) => bounded,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Replace failed: {e}"
                    ))]));
                }
            };
            if let Some((bytes, matches)) = bounded {
                bound::admit(
                    "str_replace_value",
                    &result_name,
                    bytes,
                    || {
                        format!(
                            "replacing {matches} values of `column` with the {}-byte `new_value`",
                            request.new_value.len()
                        )
                    },
                    "use a shorter new_value, filter the rows first or start a new session",
                )
                .await?;
            }
        }

        let result = match replace(
            &dataset,
            &request.column,
            &request.old_value,
            &request.new_value,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Replace failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Replaced '{}' with '{}' in column '{}', saved as '{}'",
            request.old_value, request.new_value, request.column, result_name
        ))]))
    }

    /// Replace substrings matching a regex pattern.
    #[tool(
        description = "Replace substrings matching a regex pattern with a replacement string. Supports capture groups ($1, $2, etc.) in the replacement."
    )]
    pub async fn str_regex_replace(
        &self,
        Parameters(request): Parameters<RegexReplaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // Each match becomes the replacement, its group references
        // expanded: the text the replaced column will hold is counted first.
        if bound::active() {
            let text = match regex_replace_bytes(
                &dataset,
                &request.column,
                &request.pattern,
                &request.replacement,
            ) {
                Ok(text) => text,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Regex replace failed: {}",
                        e
                    ))]));
                }
            };
            let bytes = bound::regex_replace(dataset.df(), &request.column, text);
            bound::admit(
                "str_regex_replace",
                &result_name,
                bytes,
                || {
                    format!(
                        "replacing the matches of `pattern` with the {}-byte `replacement`",
                        request.replacement.len()
                    )
                },
                "use a shorter replacement or fewer group references, a pattern that matches \
                 less, or start a new session",
            )
            .await?;
        }

        let result = match regex_replace(
            &dataset,
            &request.column,
            &request.pattern,
            &request.replacement,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex replace failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Replaced pattern '{}' in '{}', saved as '{}'",
            request.pattern, request.column, result_name
        ))]))
    }

    /// Extract substrings matching a regex pattern into a new column.
    #[tool(
        description = "Extract substrings matching a regex pattern into a new column. Use capture groups () to specify what to extract, or extract the whole match."
    )]
    pub async fn str_regex_extract(
        &self,
        Parameters(request): Parameters<RegexExtractRequest>,
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

        let group = request.group.unwrap_or(1);

        let result = match regex_extract(
            dataset,
            &request.column,
            &request.pattern,
            &request.new_column,
            group,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex extract failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Extracted pattern '{}' from '{}' into '{}', saved as '{}'",
            request.pattern, request.column, request.new_column, result_name
        ))]))
    }

    /// Count regex pattern matches in each row.
    #[tool(
        description = "Count the number of times a regex pattern matches in each row, creating a new integer column with the counts."
    )]
    pub async fn str_regex_count(
        &self,
        Parameters(request): Parameters<RegexCountRequest>,
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

        let result = match regex_count(
            dataset,
            &request.column,
            &request.pattern,
            &request.new_column,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex count failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Counted pattern '{}' matches from '{}' into '{}', saved as '{}'",
            request.pattern, request.column, request.new_column, result_name
        ))]))
    }

    /// Split a string column into multiple columns.
    #[tool(
        description = "Split a string column by a pattern (supports regex) into multiple columns named prefix_0, prefix_1, etc."
    )]
    pub async fn str_split(
        &self,
        Parameters(request): Parameters<StrSplitRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before splitting.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // With a str_split_parts budget or dataset quotas, count the output
        // columns first.
        let limits = limits::current();
        if limits.str_split_parts.is_some() || bound::active() {
            let parts = match str_split_part_count(
                &dataset,
                &request.column,
                &request.pattern,
                request.max_splits,
            ) {
                Ok(parts) => parts,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "String split failed: {}",
                        e
                    ))]));
                }
            };
            if let Err(v) = limits.check_budget(
                "str_split",
                "str_split_parts",
                parts as u64,
                limits.str_split_parts.map(|v| v as u64),
            ) {
                return Ok(refused(v));
            }
            if bound::active()
                && let Some(bytes) =
                    bound::split(dataset.df(), &request.column, &request.prefix, parts)
            {
                bound::admit(
                    "str_split",
                    &result_name,
                    bytes,
                    || format!("splitting `column` by `pattern` into {parts} columns"),
                    "set max_splits, split a smaller dataset, filter the rows first or start a \
                     new session",
                )
                .await?;
            }
        }

        let result = match str_split(
            &dataset,
            &request.column,
            &request.pattern,
            request.max_splits,
            &request.prefix,
        ) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String split failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Split '{}' by '{}' into columns with prefix '{}', saved as '{}'",
            request.column, request.pattern, request.prefix, result_name
        ))]))
    }

    /// Concatenate multiple string columns.
    #[tool(
        description = "Concatenate multiple string columns into a new column, optionally with a separator between values."
    )]
    pub async fn str_concat(
        &self,
        Parameters(request): Parameters<StrConcatRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let separator = request.separator.as_deref();
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // Every row gets a separator between two listed values: the result
        // can be far larger than the dataset.
        if bound::active()
            && let Some(bytes) = bound::str_concat(
                dataset.df(),
                &columns,
                &request.new_column,
                separator.unwrap_or(""),
            )
        {
            bound::admit(
                "str_concat",
                &result_name,
                bytes,
                || {
                    format!(
                        "concatenating {} `columns` with a {}-byte `separator` on {} rows",
                        columns.len(),
                        separator.map_or(0, str::len),
                        dataset.nrows()
                    )
                },
                "concatenate fewer or shorter columns, use a shorter separator, filter the rows \
                 first or start a new session",
            )
            .await?;
        }

        let result = match str_concat(&dataset, &columns, &request.new_column, separator) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String concat failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Concatenated {} columns into '{}', saved as '{}'",
            request.columns.len(),
            request.new_column,
            result_name
        ))]))
    }

    /// Get string lengths.
    #[tool(
        description = "Create a new column containing the length (number of characters) of each string in the source column."
    )]
    pub async fn str_length(
        &self,
        Parameters(request): Parameters<StrLengthRequest>,
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

        let result = match str_length(dataset, &request.column, &request.new_column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String length failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created length column '{}' from '{}', saved as '{}'",
            request.new_column, request.column, result_name
        ))]))
    }

    /// Extract a substring from a string column.
    #[tool(
        description = "Extract a substring from each string in a column. Supports negative indices to count from end."
    )]
    pub async fn str_substring(
        &self,
        Parameters(request): Parameters<StrSubstringRequest>,
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

        let result = match str_substring(dataset, &request.column, request.start, request.length) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String substring failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        let length_desc = request
            .length
            .map(|l| format!(", length {}", l))
            .unwrap_or_else(|| " to end".to_string());

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Extracted substring from '{}' (start: {}{}), saved as '{}'",
            request.column, request.start, length_desc, result_name
        ))]))
    }

    /// Create lag or lead columns for time series data.
    #[tool(
        description = "Create lag or lead columns for time series or panel data. Lag shifts values forward (past values), lead shifts values backward (future values)."
    )]
    pub async fn munge_lag_lead(
        &self,
        Parameters(request): Parameters<LagLeadRequest>,
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

        let periods = request.periods.unsigned_abs() as usize;
        let group_by_cols: Option<Vec<&str>> = request
            .group_by
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let direction = request.direction.as_deref().unwrap_or("lag");
        let result = match direction {
            "lag" => lag(dataset, &request.column, periods, group_by_cols.as_deref()),
            "lead" => lead(dataset, &request.column, periods, group_by_cols.as_deref()),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown direction: '{}'. Use 'lag' or 'lead'.",
                    direction
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Lag/lead failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let new_col = format!("{}_{}{}", request.column, direction, periods);

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created '{}' column, saved as '{}'",
            new_col, result_name
        ))]))
    }

    /// Standardize or normalize columns.
    #[tool(
        description = "Standardize (z-score) or normalize (0-1 range) numeric columns. Standardize subtracts mean and divides by std. Normalize scales to [0, 1]."
    )]
    pub async fn munge_standardize(
        &self,
        Parameters(request): Parameters<StandardizeRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let method = request.method.as_deref().unwrap_or("standardize");
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // One new Float64 column per listed column, whatever its width.
        let suffix = match method {
            "standardize" => Some("standardized"),
            "normalize" => Some("normalized"),
            _ => None,
        };
        if bound::active()
            && let Some(suffix) = suffix
            && let Some(bytes) = bound::standardize(dataset.df(), &cols, suffix)
        {
            bound::admit(
                "munge_standardize",
                &result_name,
                bytes,
                || {
                    format!(
                        "{} new columns of `columns` on {} rows",
                        cols.len(),
                        dataset.nrows()
                    )
                },
                "list fewer columns, filter the rows first or start a new session",
            )
            .await?;
        }

        let result = match method {
            "standardize" => standardize(&dataset, &cols),
            "normalize" => normalize(&dataset, &cols),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown method: '{}'. Use 'standardize' or 'normalize'.",
                    method
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "{} failed: {}",
                    method, e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Applied {} to {} columns, saved as '{}'",
            method,
            request.columns.len(),
            result_name
        ))]))
    }

    /// Bin a continuous variable into discrete categories.
    #[tool(
        description = "Bin a continuous variable into discrete categories. Strategies: 'uniform' (equal width), 'quantile' (equal frequency), or 'custom' (specify break points)."
    )]
    pub async fn munge_bin(
        &self,
        Parameters(request): Parameters<BinColumnRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy (polars columns are reference-counted): the read lock
        // is released at the end of this statement, not held while binning.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let strategy = match request.strategy.to_lowercase().as_str() {
            name @ ("uniform" | "equal_width" | "quantile") => {
                let n_bins = match bin_count(&request.bins, &request.strategy) {
                    Ok(n) => n,
                    Err(message) => return Ok(CallToolResult::error(vec![Content::text(message)])),
                };
                let limits = limits::current();
                if let Err(v) = limits.check_budget(
                    "munge_bin",
                    Family::Bins.key(),
                    n_bins as u64,
                    limits.cap(Family::Bins),
                ) {
                    return Ok(refused(v));
                }
                if name == "quantile" {
                    BinStrategy::Quantile(n_bins)
                } else {
                    BinStrategy::EqualWidth(n_bins)
                }
            }
            "custom" => BinStrategy::Custom(request.bins.clone()),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown strategy: '{}'. Use 'uniform', 'quantile', or 'custom'.",
                    request.strategy
                ))]));
            }
        };

        let labels = request
            .labels
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>());

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // Every row gets its bin's label, a `labels` string or `bin_<i>`.
        if bound::active() {
            let n_bins = match &strategy {
                BinStrategy::EqualWidth(n) | BinStrategy::Quantile(n) => *n,
                BinStrategy::Custom(edges) => edges.len().saturating_sub(1),
            };
            let longest = match &labels {
                Some(labels) => labels.iter().map(|l| l.len()).max().unwrap_or(0),
                None => format!("bin_{}", n_bins.saturating_sub(1)).len(),
            };
            if let Some(bytes) = bound::bin(dataset.df(), &request.column, longest) {
                bound::admit(
                    "munge_bin",
                    &result_name,
                    bytes,
                    || {
                        format!(
                            "a label of up to {longest} bytes (`labels`) on each of {} rows",
                            dataset.nrows()
                        )
                    },
                    "use shorter labels, filter the rows first or start a new session",
                )
                .await?;
            }
        }

        let result = match bin(&dataset, &request.column, strategy, labels.as_deref()) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bin failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Binned '{}' using {} strategy, saved as '{}'",
            request.column, request.strategy, result_name
        ))]))
    }

    /// One-hot encode a categorical column.
    #[tool(
        description = "One-hot encode a categorical column, creating binary indicator columns for each category. Use drop_first=true to avoid multicollinearity in regression."
    )]
    pub async fn munge_one_hot_encode(
        &self,
        Parameters(request): Parameters<OneHotEncodeRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before encoding.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        // With an onehot_levels budget, count the distinct values (a null is a
        // level, as in one_hot_encode) before encoding.
        let limits = limits::current();
        if limits.onehot_levels.is_some()
            && let Ok(column) = dataset.df().column(&request.column)
        {
            let levels = match column.n_unique() {
                Ok(levels) => levels,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "One-hot encode failed: {}",
                        e
                    ))]));
                }
            };
            if let Err(v) = limits.check_budget(
                "munge_one_hot_encode",
                "onehot_levels",
                levels as u64,
                limits.onehot_levels.map(|v| v as u64),
            ) {
                return Ok(refused(v));
            }
        }

        let drop_first = request.drop_first.unwrap_or(false);
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // One Int32 column of every row per distinct value.
        if bound::active() {
            let bounded = match bound::one_hot(dataset.df(), &request.column) {
                Ok(bounded) => bounded,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "One-hot encode failed: {}",
                        e
                    ))]));
                }
            };
            if let Some((bytes, levels)) = bounded {
                bound::admit(
                    "munge_one_hot_encode",
                    &result_name,
                    bytes,
                    || {
                        format!(
                            "up to {levels} dummy columns of `column` on {} rows",
                            dataset.nrows()
                        )
                    },
                    "encode a column with fewer distinct values, filter the rows first or start \
                     a new session",
                )
                .await?;
            }
        }

        let result = match one_hot_encode(&dataset, &request.column, drop_first) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "One-hot encode failed: {}",
                    e
                ))]));
            }
        };

        let n_cols = result.ncols();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "One-hot encoded '{}' (drop_first={}), saved as '{}' ({} total columns)",
            request.column, drop_first, result_name, n_cols
        ))]))
    }

    /// Compute differences or percent changes.
    #[tool(
        description = "Compute differences or percent changes for a column. Useful for time series and panel data analysis."
    )]
    pub async fn munge_diff(
        &self,
        Parameters(request): Parameters<DiffRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let periods = request.periods.unwrap_or(1) as usize;
        let diff_type = request.diff_type.as_deref().unwrap_or("diff");
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());
        let new_col = format!("{}_{}{}", request.column, diff_type, periods);

        // The new column can be wider than the one it is computed from.
        if bound::active()
            && matches!(diff_type, "diff" | "pct_change")
            && let Some(bytes) = bound::diff(dataset.df(), &request.column, &new_col)
        {
            bound::admit(
                "munge_diff",
                &result_name,
                bytes,
                || format!("the new column of `column` on {} rows", dataset.nrows()),
                "filter the rows first or start a new session",
            )
            .await?;
        }

        let result = match diff_type {
            "diff" => diff(&dataset, &request.column, periods),
            "pct_change" => pct_change(&dataset, &request.column, periods),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown diff type: '{}'. Use 'diff' or 'pct_change'.",
                    diff_type
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Diff/pct_change failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created '{}' column, saved as '{}'",
            new_col, result_name
        ))]))
    }

    /// Sample rows from a dataset.
    #[tool(
        description = "Randomly sample rows from a dataset. Useful for creating training/test splits or working with large datasets."
    )]
    pub async fn munge_sample(
        &self,
        Parameters(request): Parameters<SampleDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let replace = request.replace.unwrap_or(false);
        let seed = request.seed;
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| format!("{}_sample", request.dataset));

        // With replacement, `n` can be far above the dataset's rows (without,
        // polars refuses that).
        if bound::active() {
            let rows = if replace {
                request.n
            } else {
                request.n.min(dataset.nrows())
            };
            let bytes = bound::sample(dataset.df(), rows as u64, replace);
            bound::admit(
                "munge_sample",
                &result_name,
                bytes,
                || format!("{} sampled rows (`n`)", request.n),
                "sample fewer rows, sample fewer columns or start a new session",
            )
            .await?;
        }

        let result = match sample(&dataset, Some(request.n), None, replace, seed) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sample failed: {}",
                    e
                ))]));
            }
        };

        let n_rows = result.nrows();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Sampled {} rows (replace={}), saved as '{}'",
            n_rows, replace, result_name
        ))]))
    }

    /// Create a new column by computation.
    #[tool(
        description = "Create a new column by applying arithmetic operations or functions. Supports: arithmetic (+, -, *, /), functions (log, exp, sqrt, abs, square), or constant values."
    )]
    pub async fn munge_mutate(
        &self,
        Parameters(request): Parameters<MutateColumnRequest>,
    ) -> Result<CallToolResult, McpError> {
        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let expr = match request.expr_type.as_str() {
            "arithmetic" => {
                let op = match request.operator.as_deref() {
                    Some("+") => ArithOp::Add,
                    Some("-") => ArithOp::Sub,
                    Some("*") => ArithOp::Mul,
                    Some("/") => ArithOp::Div,
                    Some(other) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Unknown operator: '{}'. Use '+', '-', '*', or '/'.",
                            other
                        ))]));
                    }
                    None => {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "Arithmetic expressions require an 'operator' field.".to_string(),
                        )]));
                    }
                };

                let right = request.right.as_deref().ok_or_else(|| {
                    McpError::invalid_request(
                        "Arithmetic expressions require a 'right' field",
                        None,
                    )
                })?;

                MutateExpr::Arithmetic(request.left.clone(), op, right.to_string())
            }
            "function" => {
                let func = request.operator.as_deref().unwrap_or("log");
                MutateExpr::Function(func.to_string(), request.left.clone())
            }
            "constant" => MutateExpr::Constant(request.left.clone()),
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown expression type: '{}'. Use 'arithmetic', 'function', or 'constant'.",
                    other
                ))]));
            }
        };

        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // A constant is written on every row, a computed column is Float64:
        // either can be far larger than the column it is computed from.
        if bound::active()
            && let Some(bytes) = bound::mutate(dataset.df(), &request.new_column, &expr)
        {
            bound::admit(
                "munge_mutate",
                &result_name,
                bytes,
                || {
                    format!(
                        "the new column (`expr_type` {}, a {}-byte `left`) on {} rows",
                        request.expr_type,
                        request.left.len(),
                        dataset.nrows()
                    )
                },
                "use a shorter constant, filter the rows first or start a new session",
            )
            .await?;
        }

        let result = match mutate(&dataset, &request.new_column, expr) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mutate failed: {}",
                    e
                ))]));
            }
        };

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created column '{}', saved as '{}'",
            request.new_column, result_name
        ))]))
    }

    /// Cast column(s) to a different data type.
    #[tool(
        description = "Cast column(s) to a different data type (int, float, string, bool). Use after cleaning sentinel values to convert string columns to numeric for analysis."
    )]
    pub async fn munge_cast(
        &self,
        Parameters(request): Parameters<CastColumnRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::data::munging::{cast, cast_columns};

        // An owned copy: the read lock is released before the result is
        // bounded and built.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();

        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        // Build list of (column, dtype) pairs from either single or batch mode
        let cast_pairs: Vec<(String, String)> = if let Some(ref casts) = request.casts {
            // Batch mode
            let mut pairs = Vec::new();
            for pair in casts {
                if pair.len() != 2 {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "Each entry in 'casts' must be a [column, dtype] pair.".to_string(),
                    )]));
                }
                pairs.push((pair[0].clone(), pair[1].clone()));
            }
            pairs
        } else if let (Some(col), Some(dt)) = (&request.column, &request.dtype) {
            // Single column mode
            vec![(col.clone(), dt.clone())]
        } else {
            return Ok(CallToolResult::error(vec![Content::text(
                "Provide either 'column' + 'dtype' for single-column casting, or 'casts' for batch casting.".to_string(),
            )]));
        };

        let cast_refs: Vec<(&str, &str)> = cast_pairs
            .iter()
            .map(|(c, d)| (c.as_str(), d.as_str()))
            .collect();
        let result_name = request
            .result_name
            .clone()
            .unwrap_or_else(|| request.dataset.clone());

        // A cast can widen every value: a boolean to a string, a category to
        // its label on every row.
        if bound::active() {
            let bounded = match bound::cast(dataset.df(), &cast_refs) {
                Ok(bounded) => bounded,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cast failed: {}",
                        e
                    ))]));
                }
            };
            if let Some(bytes) = bounded {
                let parameters = if request.casts.is_some() {
                    "`casts`"
                } else {
                    "`column` to `dtype`"
                };
                bound::admit(
                    "munge_cast",
                    &result_name,
                    bytes,
                    || format!("casting {parameters} on {} rows", dataset.nrows()),
                    "cast fewer columns, to a narrower type, filter the rows first or start a \
                     new session",
                )
                .await?;
            }
        }

        let result = if cast_refs.len() == 1 {
            match cast(&dataset, cast_refs[0].0, cast_refs[0].1) {
                Ok(ds) => ds,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cast failed: {}",
                        e
                    ))]));
                }
            }
        } else {
            match cast_columns(&dataset, &cast_refs) {
                Ok(ds) => ds,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cast failed: {}",
                        e
                    ))]));
                }
            }
        };

        let cast_desc: Vec<String> = cast_pairs
            .iter()
            .map(|(c, d)| format!("{} -> {}", c, d))
            .collect();

        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Cast columns [{}], saved as '{}'",
            cast_desc.join(", "),
            result_name
        ))]))
    }
}

/// The bin count of the `uniform` and `quantile` strategies: `bins[0]`, which
/// must be a whole number of at least 1 (5 when `bins` is empty).
fn bin_count(bins: &[f64], strategy: &str) -> Result<usize, String> {
    let Some(&v) = bins.first() else {
        return Ok(5);
    };
    let invalid = || {
        format!(
            "`bins` must be a whole number of at least 1 for the '{strategy}' strategy (got {v})"
        )
    };
    if !(v.is_finite() && v >= 1.0 && v.fract() == 0.0) {
        return Err(invalid());
    }
    // `as u64` saturates above u64::MAX; the budget and the core refuse such counts.
    usize::try_from(v as u64).map_err(|_| invalid())
}

/// A join the core refused, as an in-band tool error that says "Join failed:"
/// once: the core's join refusals are [`MungeError::JoinError`]s, which already
/// read that way, so only another error variant would get the prefix here.
fn join_failed(e: &MungeError) -> CallToolResult {
    let text = match e {
        MungeError::JoinError(_) => e.to_string(),
        _ => format!("Join failed: {e}"),
    };
    CallToolResult::error(vec![Content::text(text)])
}

/// A request refused by a resource limit: logged, and returned as an in-band
/// tool error whose text names the parameter, the requested size and the limit.
fn refused(v: LimitViolation) -> CallToolResult {
    tracing::info!(
        tool = %v.tool,
        knob = v.override_key,
        requested = %v.requested,
        limit = %v.limit,
        "tool request refused by a resource limit"
    );
    CallToolResult::error(vec![Content::text(v.message)])
}

/// Admit the result of `str_to_uppercase` (`upper`) or `str_to_lowercase`
/// against the dataset quotas before it is built (a case mapping can take up
/// to three times a character's bytes). The `Err` is the handler's answer.
async fn admit_case_mapped(
    dataset: &Dataset,
    column: &str,
    result_name: &str,
    upper: bool,
) -> Result<(), Result<CallToolResult, McpError>> {
    if !bound::active() {
        return Ok(());
    }
    let (tool, case) = if upper {
        ("str_to_uppercase", "uppercase")
    } else {
        ("str_to_lowercase", "lowercase")
    };
    match bound::case_mapped(dataset.df(), column, upper) {
        Err(e) => Err(Ok(CallToolResult::error(vec![Content::text(format!(
            "To {case} failed: {e}"
        ))]))),
        Ok(None) => Ok(()),
        Ok(Some(bytes)) => bound::admit(
            tool,
            result_name,
            bytes,
            || format!("converting `column` to {case}"),
            "filter the rows first or start a new session",
        )
        .await
        .map_err(Err),
    }
}

/// Upper bounds of what the dataset quotas will charge for the dataset a
/// munging tool is about to store, and the admission of that dataset against
/// the quotas before the tool builds it.
///
/// The governor checks the quotas after a call, on what the call stored. A
/// tool whose result can be far larger than its input (a separator or a
/// replacement written into every row, a list that repeats rows or columns,
/// a type that widens every value) would allocate all of it before that
/// check. Such a tool therefore predicts, from the row count, the input's
/// column widths and the request, an upper bound of what the quotas will
/// charge for the dataset it stores (`session::datasets_bytes_of`), and is
/// refused before it builds anything when the session's datasets, with that
/// dataset in place of any of the same name, would pass the session quota,
/// or the server's would pass the total (`session::admit_charge`). The bound
/// covers what is stored, not the working memory of the kernel that builds
/// it. Without installed quotas (stdio, the off profile, HTTP without
/// quotas) nothing is predicted: [`bound::active`] is false.
mod bound {
    use std::collections::HashSet;

    use p2a_core::data::munging::{AggFn, AggSpec, FillStrategy, MutateExpr};
    use p2a_core::polars::prelude::*;
    use rmcp::ErrorData as McpError;

    use crate::session::{self, VIEW_CHARGE_BYTES};

    /// The longest String or Binary value polars' `View` holds inline: a
    /// longer one keeps its bytes in a buffer as well.
    const VIEW_INLINE_BYTES: u64 = 12;

    /// Rows between two cooperative-cancellation checkpoints in the passes
    /// over a column below.
    const CHECKPOINT_EVERY: usize = 65_536;

    /// Whether the dataset quotas of a governed call are installed on this
    /// thread, so that a tool bounds its result before building it.
    pub(super) fn active() -> bool {
        session::dataset_admission_active()
    }

    /// Admit the dataset `tool` is about to store as `result_name`, charged
    /// at most `bytes`, or refuse it. The refusal says what makes the result
    /// that large (`cause`, naming the request parameters), which quota it
    /// would pass, and what to change (`advice`); the governor answers it as
    /// `quota_exceeded`.
    pub(super) async fn admit(
        tool: &str,
        result_name: &str,
        bytes: u64,
        cause: impl FnOnce() -> String,
        advice: &str,
    ) -> Result<(), McpError> {
        let Err(shortfall) = session::admit_charge(result_name, bytes).await else {
            return Ok(());
        };
        tracing::info!(
            tool,
            requested = bytes,
            limit = shortfall.limit,
            per_session = shortfall.per_session,
            "munging result refused by a dataset quota before it was built"
        );
        Err(session::quota_refusal(format!(
            "{tool} was not run: {} would make its result take up to {}, so {}. Nothing was \
             built; {advice}.",
            cause(),
            session::fmt_bytes(bytes),
            shortfall.outcome()
        )))
    }

    /// A cooperative-cancellation checkpoint every [`CHECKPOINT_EVERY`] rows
    /// of a pass: the call's deadline or cancellation, as an error text.
    pub(super) fn checkpoint(row: usize) -> Result<(), String> {
        if row.is_multiple_of(CHECKPOINT_EVERY) {
            p2a_core::resource::checkpoint().map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// What the dataset a tool stores will be charged, at most, built up
    /// column by column: the dataset's structure, each column's structure
    /// (its name included) and each column's values.
    #[derive(Debug)]
    pub(super) struct Bound {
        /// The result's rows.
        rows: u64,
        /// Arrays a column of the result is split into, at most (a validity
        /// bitmap costs up to two bytes more per array).
        chunks: u64,
        bytes: u64,
    }

    impl Bound {
        /// A result of `rows` rows, built from frames whose columns are split
        /// into at most `chunks` arrays, with no column yet.
        pub(super) fn new(rows: u64, chunks: u64) -> Self {
            Self {
                rows,
                // A derived column can combine the arrays of two inputs.
                chunks: chunks.saturating_mul(2).saturating_add(1),
                bytes: session::structure_bytes(std::iter::empty()),
            }
        }

        /// A result of `df`'s rows that keeps every column of `df` but those
        /// named in `replaced` (polars puts a new column whose name `df` has
        /// in that column's place), each charged as the quotas charge it now.
        pub(super) fn keeping(df: &DataFrame, replaced: &[&str]) -> Self {
            let replaced: HashSet<&str> = replaced.iter().copied().collect();
            let mut bound = Self::new(df.height() as u64, df.max_n_chunks() as u64);
            for column in df.get_columns() {
                if !replaced.contains(column.name().as_str()) {
                    bound.column(
                        column.name(),
                        session::series_bytes(column.as_materialized_series()),
                    );
                }
            }
            bound
        }

        /// What the dataset is charged at most.
        pub(super) fn bytes(&self) -> u64 {
            self.bytes
        }

        /// The result's rows.
        pub(super) fn rows(&self) -> u64 {
            self.rows
        }

        /// Bytes that belong to no single column (text that the result's
        /// columns share out, or the values of a column added in parts).
        pub(super) fn add(&mut self, bytes: u64) {
            self.bytes = self.bytes.saturating_add(bytes);
        }

        /// A column named `name` whose values are charged at most `values`.
        pub(super) fn column(&mut self, name: &str, values: u64) {
            self.add(session::column_structure_bytes(name).saturating_add(values));
        }

        /// A column of the result's rows of fixed-width values of `width`
        /// bytes each, nulls allowed.
        pub(super) fn fixed(&mut self, name: &str, width: u64) {
            let values = width
                .saturating_mul(self.rows)
                .saturating_add(validity(self.rows, self.chunks));
            self.column(name, values);
        }

        /// A String column of the result's rows holding at most `text` bytes
        /// of text, nulls allowed.
        pub(super) fn string(&mut self, name: &str, text: u64) {
            let values = string_values(self.rows, text, self.chunks);
            self.column(name, values);
        }

        /// A column of the result's rows gathered from `source` (taken by row
        /// index) as `how` takes them, see [`gathered_values`].
        pub(super) fn gathered(&mut self, name: &str, source: &Column, how: Gather) {
            let values =
                gathered_values(source.as_materialized_series(), self.rows, self.chunks, how);
            self.column(name, values);
        }
    }

    /// How a tool gathers a source column's rows into its result, which
    /// bounds the bytes of the String or Binary values it gathers and the
    /// inner values of the lists: polars copies every value it gathers from
    /// a String column of several arrays, and shares a single array's
    /// buffers with each of the arrays it splits a gather into.
    #[derive(Debug, Clone, Copy)]
    pub(super) enum Gather {
        /// Each source row at most once (a sample without replacement, a
        /// group's key or first value).
        Once,
        /// Each source row exactly this many times (a melt).
        Repeat(u64),
        /// Each source row once, and each of this many nulls filled with
        /// another row's value (a forward or backward fill).
        Fill(u64),
        /// Any rows, any number of times (a join, a sample with
        /// replacement).
        Any,
    }

    /// A validity bitmap of `rows` values in at most `chunks` arrays: a byte
    /// per eight values, and up to two more per array (a slice that starts
    /// inside a byte).
    fn validity(rows: u64, chunks: u64) -> u64 {
        rows.div_ceil(8).saturating_add(chunks.saturating_mul(2))
    }

    /// `rows` String values holding at most `text` bytes of text: a view
    /// each (as the quotas charge it), the text and a validity bitmap.
    fn string_values(rows: u64, text: u64, chunks: u64) -> u64 {
        VIEW_CHARGE_BYTES
            .saturating_mul(rows)
            .saturating_add(text)
            .saturating_add(validity(rows, chunks))
    }

    /// The bytes a String value of `len` bytes keeps out of its view.
    fn out_of_line(len: usize) -> u64 {
        let len = len as u64;
        if len > VIEW_INLINE_BYTES { len } else { 0 }
    }

    /// The width in bytes of one value of a fixed-width type (not Boolean).
    fn fixed_width(dtype: &DataType) -> Option<u64> {
        Some(match dtype {
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date => 4,
            DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time => 8,
            DataType::Int128 | DataType::UInt128 | DataType::Decimal(_, _) => 16,
            _ => return None,
        })
    }

    /// What `rows` values of a fixed-width type or Boolean are charged at
    /// most, nulls allowed; `None` for other types.
    fn fixed_values(dtype: &DataType, rows: u64, chunks: u64) -> Option<u64> {
        let validity = validity(rows, chunks);
        if *dtype == DataType::Boolean {
            // The values are a bitmap too.
            return Some(validity.saturating_mul(2));
        }
        fixed_width(dtype).map(|w| w.saturating_mul(rows).saturating_add(validity))
    }

    /// The bytes of each code of a Categorical or Enum type.
    fn code_width(dtype: &DataType) -> u64 {
        match dtype.cat_physical() {
            Ok(CategoricalPhysical::U8) => 1,
            Ok(CategoricalPhysical::U16) => 2,
            _ => 4,
        }
    }

    /// The text of a String or Binary column: every value's bytes, inline
    /// values and null slots included. The quotas charge an array's text as
    /// the smaller of its buffers and its values' bytes, so an array built
    /// from these values, or sharing their buffers, is charged no more.
    /// Zero for other types.
    pub(super) fn text_bytes(s: &Series) -> u64 {
        let mut text = 0u64;
        match s.dtype() {
            DataType::String => {
                if let Ok(ca) = s.str() {
                    for arr in ca.downcast_iter() {
                        text = text.saturating_add(arr.total_bytes_len() as u64);
                    }
                }
            }
            DataType::Binary => {
                if let Ok(ca) = s.binary() {
                    for arr in ca.downcast_iter() {
                        text = text.saturating_add(arr.total_bytes_len() as u64);
                    }
                }
            }
            _ => {}
        }
        text
    }

    /// The bytes of the longest value of a String or Binary column (0 for
    /// other types).
    fn longest_value(s: &Series) -> u64 {
        let longest = match s.dtype() {
            DataType::String => s.str().ok().and_then(|ca| {
                ca.downcast_iter()
                    .flat_map(|a| a.views().iter().map(|v| v.length))
                    .max()
            }),
            DataType::Binary => s.binary().ok().and_then(|ca| {
                ca.downcast_iter()
                    .flat_map(|a| a.views().iter().map(|v| v.length))
                    .max()
            }),
            _ => None,
        };
        u64::from(longest.unwrap_or(0))
    }

    /// The text of `rows` values gathered from the String or Binary column
    /// `s` as `how` takes them, at most: the bytes of the values gathered.
    /// That bounds the charge whether the gather copies them or shares `s`'s
    /// buffers, also when polars splits it into several arrays that each
    /// share all of them.
    fn gathered_text(s: &Series, rows: u64, how: Gather) -> u64 {
        let text = text_bytes(s);
        match how {
            Gather::Once => text,
            Gather::Repeat(times) => text.saturating_mul(times),
            Gather::Fill(nulls) => text.saturating_add(nulls.saturating_mul(longest_value(s))),
            Gather::Any => rows.saturating_mul(longest_value(s)),
        }
    }

    /// The inner values `rows` lists gathered from a list column take, at
    /// most, given its `inner` value count and `longest` list.
    fn gathered_inner(inner: u64, longest: u64, rows: u64, how: Gather) -> (u64, Gather) {
        match how {
            Gather::Once => (inner, Gather::Once),
            Gather::Repeat(times) => (inner.saturating_mul(times), Gather::Repeat(times)),
            Gather::Fill(nulls) => (
                inner.saturating_add(nulls.saturating_mul(longest)),
                Gather::Any,
            ),
            Gather::Any => (rows.saturating_mul(longest), Gather::Any),
        }
    }

    /// What `rows` values gathered from `s` as `how` takes them are charged
    /// at most: a fixed-width value its width; a String or Binary value its
    /// view and the text of [`gathered_text`] (polars' compaction of a
    /// gathered array only shrinks its buffers); a Categorical or Enum value
    /// its code, with `s`'s labels; nested values recursively; and validity.
    /// A type a dataset rarely holds is charged its whole column for every
    /// row.
    fn gathered_values(s: &Series, rows: u64, chunks: u64, how: Gather) -> u64 {
        let dtype = s.dtype();
        if let Some(values) = fixed_values(dtype, rows, chunks) {
            return values;
        }
        let whole_column_per_row = || {
            session::series_bytes(s)
                .saturating_add(16)
                .saturating_mul(rows.max(1))
        };
        let validity = validity(rows, chunks);
        match dtype {
            DataType::String | DataType::Binary => {
                string_values(rows, gathered_text(s, rows, how), chunks)
            }
            DataType::Categorical(_, _) | DataType::Enum(_, _) => {
                let labels = session::series_bytes(s).saturating_sub(s.estimated_size() as u64);
                code_width(dtype)
                    .saturating_mul(rows)
                    .saturating_add(labels)
                    .saturating_add(validity)
            }
            DataType::List(_) => {
                let Ok(ca) = s.list() else {
                    return whole_column_per_row();
                };
                let longest = ca
                    .downcast_iter()
                    .flat_map(|a| a.offsets().lengths())
                    .max()
                    .unwrap_or(0) as u64;
                let inner = ca.get_inner();
                let (values, inner_how) = gathered_inner(inner.len() as u64, longest, rows, how);
                rows.saturating_add(chunks)
                    .saturating_mul(8)
                    .saturating_add(validity)
                    .saturating_add(gathered_values(&inner, values, chunks, inner_how))
            }
            DataType::Array(_, width) => match s.array() {
                Ok(ca) => validity.saturating_add(gathered_values(
                    &ca.get_inner(),
                    rows.saturating_mul(*width as u64),
                    chunks,
                    match how {
                        Gather::Fill(nulls) => Gather::Fill(nulls.saturating_mul(*width as u64)),
                        other => other,
                    },
                )),
                Err(_) => whole_column_per_row(),
            },
            DataType::Struct(_) => match s.struct_() {
                Ok(ca) => ca.fields_as_series().iter().fold(validity, |acc, field| {
                    acc.saturating_add(gathered_values(field, rows, chunks, how))
                }),
                Err(_) => whole_column_per_row(),
            },
            _ => whole_column_per_row(),
        }
    }

    /// What `rows` null values of type `dtype` are charged at most: the
    /// column polars fills with nulls where a row-bound dataset lacks it.
    fn null_values(dtype: &DataType, rows: u64, chunks: u64) -> u64 {
        if let Some(values) = fixed_values(dtype, rows, chunks) {
            return values;
        }
        let values = match dtype {
            DataType::String | DataType::Binary => VIEW_CHARGE_BYTES.saturating_mul(rows),
            DataType::Categorical(_, _) => code_width(dtype).saturating_mul(rows),
            DataType::Enum(frozen, _) => code_width(dtype)
                .saturating_mul(rows)
                .saturating_add(session::frozen_label_bytes(frozen)),
            DataType::List(_) => rows.saturating_add(chunks).saturating_mul(8),
            DataType::Array(inner, width) => {
                null_values(inner, rows.saturating_mul(*width as u64), chunks)
            }
            DataType::Struct(fields) => fields.iter().fold(0u64, |acc, field| {
                acc.saturating_add(null_values(field.dtype(), rows, chunks))
            }),
            _ => 16u64.saturating_mul(rows),
        };
        values.saturating_add(validity(rows, chunks))
    }

    // Each tool's bound below is `None` when the request names a column the
    // tool refuses (missing, or of a type it does not take): the tool then
    // reports its own error and builds nothing.

    /// `str_concat`: `columns` (repeats allowed) joined by `separator`,
    /// nulls skipped, into `new_column`. A row holds at most every listed
    /// value and one separator between two of them.
    pub(super) fn str_concat(
        df: &DataFrame,
        columns: &[&str],
        new_column: &str,
        separator: &str,
    ) -> Option<u64> {
        let mut text = 0u64;
        for name in columns {
            let column = df.column(name).ok()?;
            if column.dtype() != &DataType::String {
                return None;
            }
            text = text.saturating_add(text_bytes(column.as_materialized_series()));
        }
        let mut bound = Bound::keeping(df, &[new_column]);
        let separators = (columns.len() as u64)
            .saturating_sub(1)
            .saturating_mul(separator.len() as u64)
            .saturating_mul(bound.rows());
        bound.string(new_column, text.saturating_add(separators));
        Some(bound.bytes())
    }

    /// `str_replace_value` on a String column: each value equal to `old`
    /// becomes `new`, every value is copied. Returns the bound and the
    /// number of values replaced. (A numeric column keeps its width, and a
    /// column of another type is kept: `None`.)
    pub(super) fn replace_value(
        df: &DataFrame,
        column: &str,
        old: &str,
        new: &str,
    ) -> Result<Option<(u64, u64)>, String> {
        let Some(source) = df.column(column).ok() else {
            return Ok(None);
        };
        let Ok(ca) = source.str() else {
            return Ok(None);
        };
        let mut matches = 0u64;
        for (row, value) in ca.into_iter().enumerate() {
            checkpoint(row)?;
            matches += u64::from(value == Some(old));
        }
        let text = text_bytes(source.as_materialized_series())
            .saturating_sub(matches.saturating_mul(old.len() as u64))
            .saturating_add(matches.saturating_mul(new.len() as u64));
        let mut bound = Bound::keeping(df, &[column]);
        bound.string(column, text);
        Ok(Some((bound.bytes(), matches)))
    }

    /// `str_regex_replace`: the replaced column holds at most `text` bytes
    /// (`p2a_core::data::munging::regex_replace_bytes`).
    pub(super) fn regex_replace(df: &DataFrame, column: &str, text: u64) -> u64 {
        let mut bound = Bound::keeping(df, &[column]);
        bound.string(column, text);
        bound.bytes()
    }

    /// `str_to_uppercase` (`upper`) or `str_to_lowercase`: the column with
    /// its text after the case mapping, counted exactly (a mapping can take
    /// up to three times a character's bytes).
    pub(super) fn case_mapped(
        df: &DataFrame,
        column: &str,
        upper: bool,
    ) -> Result<Option<u64>, String> {
        let Some(ca) = df.column(column).ok().and_then(|c| c.str().ok()) else {
            return Ok(None);
        };
        let mut text = 0u64;
        for (row, value) in ca.into_iter().enumerate() {
            checkpoint(row)?;
            if let Some(s) = value {
                text = text.saturating_add(mapped_len(s, upper));
            }
        }
        let mut bound = Bound::keeping(df, &[column]);
        bound.string(column, text);
        Ok(Some(bound.bytes()))
    }

    /// The bytes of `s` mapped to upper (or lower) case character by
    /// character, as Rust's and polars' case mappings map it.
    fn mapped_len(s: &str, upper: bool) -> u64 {
        if s.is_ascii() {
            return s.len() as u64;
        }
        let bytes: usize = if upper {
            s.chars()
                .map(|c| c.to_uppercase().map(char::len_utf8).sum::<usize>())
                .sum()
        } else {
            s.chars()
                .map(|c| c.to_lowercase().map(char::len_utf8).sum::<usize>())
                .sum()
        };
        bytes as u64
    }

    /// `str_split` of `column` into `parts` columns `{prefix}_{i}`: each a
    /// String column of every row, together holding at most the column's
    /// text.
    pub(super) fn split(df: &DataFrame, column: &str, prefix: &str, parts: usize) -> Option<u64> {
        let source = df.column(column).ok()?;
        if source.dtype() != &DataType::String {
            return None;
        }
        let names: Vec<String> = (0..parts).map(|i| format!("{prefix}_{i}")).collect();
        let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut bound = Bound::keeping(df, &replaced);
        for name in &names {
            bound.string(name, 0);
        }
        bound.add(text_bytes(source.as_materialized_series()));
        Some(bound.bytes())
    }

    /// `munge_fill_na` of `columns` (every column when `None`): a String
    /// column filled with a constant is copied with each null replaced, a
    /// numeric column filled with its mean or median becomes Float64, a
    /// forward or backward fill gathers, and any other fill keeps the
    /// column's type and width.
    pub(super) fn fill_na(
        df: &DataFrame,
        columns: Option<&[&str]>,
        strategy: &FillStrategy,
    ) -> Option<u64> {
        let names: Vec<&str> = match columns {
            Some(names) => names.to_vec(),
            None => df
                .get_column_names()
                .into_iter()
                .map(|n| n.as_str())
                .collect(),
        };
        let mut bound = Bound::keeping(df, &names);
        for name in names {
            let column = df.column(name).ok()?;
            let dtype = column.dtype();
            match strategy {
                FillStrategy::Constant(_) | FillStrategy::Zero if dtype == &DataType::String => {
                    let fill = match strategy {
                        FillStrategy::Constant(value) => value.len() as u64,
                        _ => 0,
                    };
                    let nulls = column.null_count() as u64;
                    let text = text_bytes(column.as_materialized_series())
                        .saturating_add(nulls.saturating_mul(fill));
                    bound.string(name, text);
                }
                FillStrategy::Mean | FillStrategy::Median if dtype.is_primitive_numeric() => {
                    bound.fixed(name, 8);
                }
                FillStrategy::Forward | FillStrategy::Backward => {
                    let nulls = column.null_count() as u64;
                    bound.gathered(name, column, Gather::Fill(nulls))
                }
                _ => bound.column(name, session::series_bytes(column.as_materialized_series())),
            }
        }
        Some(bound.bytes())
    }

    /// `munge_mutate`: the column `new_column`, a constant on every row
    /// (Float64 when it parses as a number, else String) or a Float64
    /// computed from columns.
    pub(super) fn mutate(df: &DataFrame, new_column: &str, expr: &MutateExpr) -> Option<u64> {
        let mut bound = Bound::keeping(df, &[new_column]);
        match expr {
            MutateExpr::Constant(value) if value.parse::<f64>().is_err() => {
                let text = out_of_line(value.len()).saturating_mul(bound.rows());
                bound.string(new_column, text);
            }
            MutateExpr::Copy(source) => {
                let s = df.column(source).ok()?.as_materialized_series();
                bound.column(new_column, session::series_bytes(s));
            }
            MutateExpr::Constant(_)
            | MutateExpr::Arithmetic(..)
            | MutateExpr::ArithmeticConst(..)
            | MutateExpr::Function(..) => bound.fixed(new_column, 8),
        }
        Some(bound.bytes())
    }

    /// `munge_bin`: the String column `{column}_binned`, one label of at
    /// most `longest_label` bytes per row.
    pub(super) fn bin(df: &DataFrame, column: &str, longest_label: usize) -> Option<u64> {
        df.column(column).ok()?;
        let name = format!("{column}_binned");
        let mut bound = Bound::keeping(df, &[&name]);
        let text = out_of_line(longest_label).saturating_mul(bound.rows());
        bound.string(&name, text);
        Some(bound.bytes())
    }

    /// `munge_one_hot_encode`: one Int32 column `{column}_{value}` per
    /// distinct value (each of them: which one `drop_first` drops depends
    /// on the order polars lists them in). Returns the bound and the number
    /// of distinct values.
    pub(super) fn one_hot(df: &DataFrame, column: &str) -> Result<Option<(u64, usize)>, String> {
        let Ok(source) = df.column(column) else {
            return Ok(None);
        };
        // polars refuses the column here as the tool will: it reports that.
        let Ok(unique) = source.unique() else {
            return Ok(None);
        };
        let mut names = Vec::with_capacity(unique.len());
        for i in 0..unique.len() {
            checkpoint(i)?;
            let value = unique
                .get(i)
                .map(|v| format!("{v}"))
                .unwrap_or_else(|_| "null".to_string());
            names.push(format!("{column}_{value}"));
        }
        let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut bound = Bound::keeping(df, &replaced);
        for name in &names {
            bound.fixed(name, 4);
        }
        Ok(Some((bound.bytes(), names.len())))
    }

    /// `munge_concat`: the frames row-bound, each column of the union of
    /// their columns as each frame holds it, or filled with nulls in a frame
    /// that lacks it.
    pub(super) fn concat(frames: &[&DataFrame]) -> u64 {
        let mut schema: Vec<(&str, &DataType)> = Vec::new();
        let mut seen = HashSet::new();
        for df in frames {
            for column in df.get_columns() {
                if seen.insert(column.name().as_str()) {
                    schema.push((column.name().as_str(), column.dtype()));
                }
            }
        }
        let rows = frames
            .iter()
            .fold(0u64, |acc, df| acc.saturating_add(df.height() as u64));
        let chunks = frames
            .iter()
            .fold(0u64, |acc, df| acc.saturating_add(df.max_n_chunks() as u64));
        let mut bound = Bound::new(rows, chunks);
        for (name, _) in &schema {
            bound.column(name, 0);
        }
        for df in frames {
            for (name, dtype) in &schema {
                match df.column(name) {
                    Ok(column) => bound.add(session::series_bytes(column.as_materialized_series())),
                    Err(_) => bound.add(null_values(dtype, df.height() as u64, 1)),
                }
            }
        }
        bound.bytes()
    }

    /// `munge_melt`: every row once per value variable, the id columns
    /// gathered, the variable names as String and the values as Float64.
    pub(super) fn melt(
        df: &DataFrame,
        id_vars: &[&str],
        value_vars: &[&str],
        variable_name: &str,
        value_name: &str,
    ) -> Option<u64> {
        let height = df.height() as u64;
        let times = value_vars.len() as u64;
        let rows = height.saturating_mul(times);
        let mut bound = Bound::new(rows, df.max_n_chunks() as u64);
        for id in id_vars {
            bound.gathered(id, df.column(id).ok()?, Gather::Repeat(times));
        }
        let names = value_vars
            .iter()
            .fold(0u64, |acc, v| acc.saturating_add(out_of_line(v.len())));
        bound.string(variable_name, names.saturating_mul(height));
        bound.fixed(value_name, 8);
        Some(bound.bytes())
    }

    /// `munge_sample` of `rows` rows, with `replacement` or without: every
    /// column gathered.
    pub(super) fn sample(df: &DataFrame, rows: u64, replacement: bool) -> u64 {
        let how = if replacement {
            Gather::Any
        } else {
            Gather::Once
        };
        let mut bound = Bound::new(rows, df.max_n_chunks() as u64);
        for column in df.get_columns() {
            bound.gathered(column.name(), column, how);
        }
        bound.bytes()
    }

    /// `munge_join` of `rows` predicted rows: every column of both frames
    /// gathered (the right key columns too, which polars may drop), a right
    /// column whose name the left has renamed with `suffix`.
    pub(super) fn join(left: &DataFrame, right: &DataFrame, rows: u64, suffix: &str) -> u64 {
        let chunks = (left.max_n_chunks() as u64).saturating_add(right.max_n_chunks() as u64);
        let mut bound = Bound::new(rows, chunks);
        for column in left.get_columns() {
            bound.gathered(column.name(), column, Gather::Any);
        }
        for column in right.get_columns() {
            if left.column(column.name()).is_ok() {
                bound.gathered(&format!("{}{suffix}", column.name()), column, Gather::Any);
            } else {
                bound.gathered(column.name(), column, Gather::Any);
            }
        }
        bound.bytes()
    }

    /// `munge_group_by`: one row per group (at most the rows, and at most
    /// the product of the `by` columns' distinct counts), the `by` columns
    /// gathered and one column per aggregation. Returns the bound and the
    /// group count it assumed.
    pub(super) fn group_by(df: &DataFrame, by: &[&str], aggs: &[AggSpec]) -> Option<(u64, u64)> {
        let mut groups = 1u64;
        for name in by {
            let column = df.column(name).ok()?;
            // A key polars cannot count may still group (or the tool reports
            // its own error): bound it by the rows.
            let distinct = column.n_unique().unwrap_or(df.height()) as u64;
            groups = groups.saturating_mul(distinct);
        }
        let groups = groups.min(df.height() as u64);
        let mut bound = Bound::new(groups, df.max_n_chunks() as u64);
        for name in by {
            if let Ok(column) = df.column(name) {
                bound.gathered(name, column, Gather::Once);
            }
        }
        for spec in aggs {
            let column = df.column(&spec.column).ok()?;
            let name = spec.result_name();
            let dtype = column.dtype();
            let scalar = dtype == &DataType::Boolean
                || dtype == &DataType::String
                || fixed_width(dtype).is_some();
            match spec.agg_fn {
                AggFn::Count | AggFn::NUnique => bound.fixed(&name, 8),
                // A number of up to 16 bytes, or a null String.
                AggFn::Sum | AggFn::Mean | AggFn::Median | AggFn::Std | AggFn::Var if scalar => {
                    bound.fixed(&name, VIEW_CHARGE_BYTES)
                }
                // One value of the group (each source value at most once),
                // or a type a dataset rarely holds.
                _ => bound.gathered(&name, column, Gather::Once),
            }
        }
        Some((bound.bytes(), groups))
    }

    /// A type `munge_cast` takes (the core's names for it), or `None`.
    fn cast_target(name: &str) -> Option<DataType> {
        Some(match name.to_lowercase().as_str() {
            "int" | "int64" | "integer" => DataType::Int64,
            "int32" => DataType::Int32,
            "float" | "float64" | "double" => DataType::Float64,
            "float32" => DataType::Float32,
            "string" | "str" | "text" => DataType::String,
            "bool" | "boolean" => DataType::Boolean,
            _ => return None,
        })
    }

    /// The longest text polars writes for one value of `dtype` cast to
    /// String: integers in decimal, floats in their shortest form (ryu, at
    /// most 24 bytes), temporal values as ISO text, generously for the rest.
    fn cast_text_width(dtype: &DataType) -> usize {
        match dtype {
            DataType::Boolean => 5,
            DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16 => 6,
            DataType::Int32 | DataType::UInt32 => 11,
            DataType::Int64 | DataType::UInt64 => 20,
            DataType::Int128 | DataType::UInt128 => 40,
            DataType::Float32 => 16,
            DataType::Float64 => 24,
            DataType::Date => 16,
            DataType::Time => 32,
            _ => 64,
        }
    }

    /// `munge_cast`: each `(column, type)` pair cast in turn; a column cast
    /// more than once is charged for its largest version, and a cast to the
    /// type a column already has keeps it.
    pub(super) fn cast(df: &DataFrame, casts: &[(&str, &str)]) -> Result<Option<u64>, String> {
        /// A cast column: its type now, the bound of its current version
        /// and of its largest one.
        struct Cast<'a> {
            name: &'a str,
            dtype: DataType,
            now: u64,
            largest: u64,
        }
        let rows = df.height() as u64;
        let chunks = df.max_n_chunks() as u64;
        let mut cast: Vec<Cast> = Vec::new();
        for (name, target) in casts {
            let (Ok(source), Some(target)) = (df.column(name), cast_target(target)) else {
                return Ok(None);
            };
            let index = match cast.iter().position(|c| c.name == *name) {
                Some(i) => i,
                None => {
                    let charged = session::series_bytes(source.as_materialized_series());
                    cast.push(Cast {
                        name,
                        dtype: source.dtype().clone(),
                        now: charged,
                        largest: charged,
                    });
                    cast.len() - 1
                }
            };
            let current = &cast[index];
            let next = if current.dtype == target {
                current.now
            } else if let Some(values) = fixed_values(&target, rows, chunks) {
                values
            } else {
                // To String: the column's own text while it has its own
                // type, else the widest text of the type it was cast to.
                let text = if &current.dtype == source.dtype() {
                    cast_source_text(source.as_materialized_series())?
                } else {
                    out_of_line(cast_text_width(&current.dtype)).saturating_mul(rows)
                };
                string_values(rows, text, chunks)
            };
            let entry = &mut cast[index];
            entry.dtype = target;
            entry.now = next;
            entry.largest = entry.largest.max(next);
        }
        let replaced: Vec<&str> = cast.iter().map(|c| c.name).collect();
        let mut bound = Bound::keeping(df, &replaced);
        for c in &cast {
            bound.column(c.name, c.largest);
        }
        Ok(Some(bound.bytes()))
    }

    /// The text a cast of the column `s` to String writes, at most: a String
    /// or Binary column's own text, a Categorical or Enum column's label on
    /// every row, and for other types the widest text of a value (when a
    /// view cannot hold it) on every row.
    fn cast_source_text(s: &Series) -> Result<u64, String> {
        match s.dtype() {
            DataType::String | DataType::Binary => Ok(text_bytes(s)),
            DataType::Categorical(_, mapping) => {
                let mapping = mapping.clone();
                codes_text(s, |code| mapping.cat_to_str(code).map_or(0, str::len))
            }
            DataType::Enum(frozen, _) => {
                let labels = frozen.categories().clone();
                codes_text(s, |code| {
                    let code = code as usize;
                    if code < labels.len() {
                        labels.value(code).len()
                    } else {
                        0
                    }
                })
            }
            dtype => {
                let rows = s.len() as u64;
                Ok(out_of_line(cast_text_width(dtype)).saturating_mul(rows))
            }
        }
    }

    /// The bytes of `label(code)` summed over the non-null codes of the
    /// Categorical or Enum column `s`.
    fn codes_text(s: &Series, label: impl Fn(u32) -> usize) -> Result<u64, String> {
        let physical = s.to_physical_repr();
        let mut text = 0u64;
        let mut add = |row: usize, code: Option<u32>| -> Result<(), String> {
            checkpoint(row)?;
            if let Some(code) = code {
                text = text.saturating_add(label(code) as u64);
            }
            Ok(())
        };
        let err = |e: PolarsError| e.to_string();
        match physical.dtype() {
            DataType::UInt8 => {
                for (row, code) in physical.u8().map_err(err)?.into_iter().enumerate() {
                    add(row, code.map(u32::from))?;
                }
            }
            DataType::UInt16 => {
                for (row, code) in physical.u16().map_err(err)?.into_iter().enumerate() {
                    add(row, code.map(u32::from))?;
                }
            }
            DataType::UInt32 => {
                for (row, code) in physical.u32().map_err(err)?.into_iter().enumerate() {
                    add(row, code)?;
                }
            }
            _ => {
                let rows = s.len() as u64;
                return Ok(session::series_bytes(s).saturating_mul(rows.max(1)));
            }
        }
        Ok(text)
    }

    /// `munge_standardize` (`suffix` `standardized`) or normalize
    /// (`normalized`): one Float64 column `{c}_{suffix}` per listed column.
    pub(super) fn standardize(df: &DataFrame, columns: &[&str], suffix: &str) -> Option<u64> {
        for c in columns {
            df.column(c).ok()?;
        }
        let names: Vec<String> = columns.iter().map(|c| format!("{c}_{suffix}")).collect();
        let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut bound = Bound::keeping(df, &replaced);
        for name in &names {
            bound.fixed(name, 8);
        }
        Some(bound.bytes())
    }

    /// `munge_diff`: the column `new_column`, of `column`'s width or a
    /// Float64 (a percent change, a difference of dates), whichever is
    /// wider.
    pub(super) fn diff(df: &DataFrame, column: &str, new_column: &str) -> Option<u64> {
        let width = fixed_width(df.column(column).ok()?.dtype())
            .unwrap_or(8)
            .max(8);
        let mut bound = Bound::keeping(df, &[new_column]);
        bound.fixed(new_column, width);
        Some(bound.bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SessionConfig;
    use crate::session::{DATASET_QUOTA_EXCEEDED, DatasetAdmission, Session, SessionManager};
    use crate::tools::limits::{self, Limits};
    use p2a_core::data::munging::{JoinType, MungeResult, cast_columns, concat};
    use p2a_core::polars::prelude::*;
    use rmcp::model::RawContent;
    use std::sync::Arc;

    fn server_with(datasets: Vec<(&str, DataFrame)>) -> AnalyticsServer {
        let server = AnalyticsServer::new();
        for (name, df) in datasets {
            server
                .datasets
                .blocking_write()
                .insert(name.to_string(), Dataset::new(df));
        }
        server
    }

    fn run(
        limits: Limits,
        fut: impl std::future::Future<Output = Result<CallToolResult, McpError>>,
    ) -> CallToolResult {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        limits::with_limits(limits, || rt.block_on(fut)).expect("refusals are in-band tool errors")
    }

    fn text(r: &CallToolResult) -> String {
        match &r.content[0].raw {
            RawContent::Text(t) => t.text.clone(),
            _ => String::new(),
        }
    }

    fn hosted_with(o: &str) -> Limits {
        Limits::hosted().with_overrides(&[o.to_string()]).unwrap()
    }

    fn noise(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 0.73).sin() * 5.0 + i as f64 * 0.1)
            .collect()
    }

    #[test]
    fn munge_bin_validates_the_bin_count() {
        let server = server_with(vec![("d", df!("x" => noise(40)).unwrap())]);
        let req = |bins: Vec<f64>| BinColumnRequest {
            dataset: "d".into(),
            column: "x".into(),
            strategy: "uniform".into(),
            bins,
            labels: None,
            result_name: Some("b".into()),
        };
        let r = run(
            Limits::hosted(),
            server.munge_bin(Parameters(req(vec![1e18]))),
        );
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("needs 1000000000000000000 (bins)") && t.contains("limit of 1000"),
            "{t}"
        );
        for bad in [2.5, -1.0, 0.0] {
            let r = run(Limits::off(), server.munge_bin(Parameters(req(vec![bad]))));
            assert_eq!(r.is_error, Some(true));
            assert!(text(&r).contains("whole number"), "{}", text(&r));
        }
        // No profile: the core refuses the impossible allocation; the process survives.
        assert_eq!(
            run(Limits::off(), server.munge_bin(Parameters(req(vec![1e18])))).is_error,
            Some(true)
        );
        let ok = run(
            Limits::hosted(),
            server.munge_bin(Parameters(req(vec![4.0]))),
        );
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
    }

    #[test]
    fn munge_join_refuses_output_above_budget() {
        let v: Vec<f64> = (0..20).map(|i| i as f64 * 0.37 + 0.11).collect();
        let server = server_with(vec![("a", df!("k" => vec![1i64; 20], "v" => v).unwrap())]);
        let req = || JoinDatasetsRequest {
            left: "a".into(),
            right: "a".into(),
            left_on: vec!["k".into()],
            right_on: None,
            join_type: Some("inner".into()),
            suffix: None,
            result_name: Some("j".into()),
        };
        let r = run(
            hosted_with("join_output_rows=100"),
            server.munge_join(Parameters(req())),
        );
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("needs 400 (join_output_rows)") && t.contains("limit of 100"),
            "{t}"
        );
        assert!(server.datasets.blocking_read().get("j").is_none());
        let ok = run(Limits::off(), server.munge_join(Parameters(req())));
        assert!(text(&ok).contains("(400 rows"), "{}", text(&ok));
    }

    /// A refused join says "Join failed:" once, whether the hosted size
    /// estimate or the join itself refuses it (the core's join errors already
    /// carry that prefix).
    #[test]
    fn munge_join_errors_say_join_failed_once() {
        let server = server_with(vec![
            (
                "p",
                df!("id" => [1i64, 2, 3], "x" => [0.4, 1.3, 2.1]).unwrap(),
            ),
            (
                "s",
                df!("id" => [1i64, 2, 4], "y" => [3.2, 0.7, 1.9], "code" => ["a", "b", "c"])
                    .unwrap(),
            ),
        ]);
        let req = |left_on: &[&str], right_on: Option<&[&str]>| JoinDatasetsRequest {
            left: "p".into(),
            right: "s".into(),
            left_on: left_on.iter().map(|s| s.to_string()).collect(),
            right_on: right_on.map(|r| r.iter().map(|s| s.to_string()).collect()),
            join_type: Some("inner".into()),
            suffix: None,
            result_name: Some("o".into()),
        };
        let cases: [(&[&str], Option<&[&str]>, &str); 6] = [
            (
                &[],
                None,
                "Join failed: a join needs at least one key column, but left_on is empty",
            ),
            (
                &["id"],
                Some(&["id", "y"]),
                "Join failed: left_on has 1 columns but right_on has 2 columns",
            ),
            // A missing key names its parameter and the dataset that lacks it.
            (
                &["nope"],
                None,
                "Join failed: left_on column 'nope' is not in the left dataset",
            ),
            (
                &["id"],
                Some(&["nope"]),
                "Join failed: right_on column 'nope' is not in the right dataset",
            ),
            (
                &["x"],
                None,
                "Join failed: left_on column 'x' is not in the right dataset (right_on is not \
                 given, so the left_on names are used for both datasets)",
            ),
            // polars refuses keys of different types; the core wraps its error.
            (&["id"], Some(&["code"]), "Join failed: "),
        ];
        for (profile, limits) in [("off", Limits::off()), ("hosted", Limits::hosted())] {
            for (left_on, right_on, want) in cases {
                let r = run(
                    limits.clone(),
                    server.munge_join(Parameters(req(left_on, right_on))),
                );
                assert_eq!(r.is_error, Some(true), "{profile}: {}", text(&r));
                let t = text(&r);
                assert!(t.starts_with(want), "{profile}: {t}");
                assert_eq!(t.matches("Join failed").count(), 1, "{profile}: {t}");
            }
        }
        assert!(server.datasets.blocking_read().get("o").is_none());
    }

    #[test]
    fn one_hot_refuses_too_many_levels() {
        let cat: Vec<String> = (0..24).map(|i| format!("c{}", i % 8)).collect();
        let server = server_with(vec![("d", df!("cat" => cat, "y" => noise(24)).unwrap())]);
        let req = || OneHotEncodeRequest {
            dataset: "d".into(),
            column: "cat".into(),
            drop_first: None,
            result_name: Some("o".into()),
        };
        let r = run(
            hosted_with("onehot_levels=5"),
            server.munge_one_hot_encode(Parameters(req())),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("needs 8 (onehot_levels)"), "{}", text(&r));
        assert_ne!(
            run(
                Limits::off(),
                server.munge_one_hot_encode(Parameters(req()))
            )
            .is_error,
            Some(true)
        );
    }

    #[test]
    fn str_split_refuses_too_many_parts() {
        let server = server_with(vec![("d", df!("s" => ["a,b,c,d", "e"]).unwrap())]);
        let req = || StrSplitRequest {
            dataset: "d".into(),
            column: "s".into(),
            pattern: ",".into(),
            max_splits: None,
            prefix: "p".into(),
            result_name: Some("o".into()),
        };
        let r = run(
            hosted_with("str_split_parts=3"),
            server.str_split(Parameters(req())),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(
            text(&r).contains("needs 4 (str_split_parts)"),
            "{}",
            text(&r)
        );
        let ok = run(Limits::off(), server.str_split(Parameters(req())));
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
        assert!(
            server
                .datasets
                .blocking_read()
                .get("o")
                .unwrap()
                .df()
                .column("p_3")
                .is_ok()
        );
    }

    // ---- Results bounded by the dataset quotas before they are built ------

    /// A session holding some datasets, a server that shares them (as a
    /// governed call's session-scoped server does) and a current-thread
    /// runtime, so that a test can run a handler with the dataset quotas
    /// installed as the governor installs them for a call.
    struct QuotaSession {
        rt: tokio::runtime::Runtime,
        manager: Arc<SessionManager>,
        session: Arc<Session>,
        server: AnalyticsServer,
    }

    impl QuotaSession {
        fn new(datasets: Vec<(&str, DataFrame)>) -> Self {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let manager = Arc::new(SessionManager::new(SessionConfig {
                ttl_minutes: 60,
                max_sessions: 4,
                evict_idle_minutes: None,
            }));
            let session = rt.block_on(async {
                let id = manager.create_session(None).await.unwrap();
                manager.get_session(&id).await.unwrap()
            });
            {
                let mut held = session.datasets.blocking_write();
                for (name, df) in datasets {
                    held.insert(name.to_string(), Dataset::new(df));
                }
            }
            let mut server = AnalyticsServer::new();
            server.datasets = Arc::clone(&session.datasets);
            Self {
                rt,
                manager,
                session,
                server,
            }
        }

        /// What the session's datasets are charged now.
        fn held(&self) -> u64 {
            self.rt.block_on(self.session.refresh_data_bytes())
        }

        /// Run a handler's call under the hosted limits with a session quota
        /// of `quota` bytes installed (`None`: no quota, as in stdio and the
        /// off profile).
        fn call(
            &self,
            quota: Option<u64>,
            call: impl std::future::Future<Output = Result<CallToolResult, McpError>>,
        ) -> Result<CallToolResult, McpError> {
            let _admission = quota.map(|q| {
                DatasetAdmission::install(
                    Some(q),
                    None,
                    Arc::clone(&self.session),
                    Arc::clone(&self.manager),
                )
            });
            limits::with_limits(Limits::hosted(), || self.rt.block_on(call))
        }

        fn has(&self, name: &str) -> bool {
            self.session.datasets.blocking_read().contains_key(name)
        }

        fn frame(&self, name: &str) -> DataFrame {
            self.session.datasets.blocking_read()[name].df().clone()
        }
    }

    /// The text of a dataset-quota refusal a handler made before building
    /// anything: its `DATASET_QUOTA_EXCEEDED` error (a handler that built the
    /// result returns it, and only the governor's check after the call could
    /// refuse it then).
    fn refused_before_building(outcome: Result<CallToolResult, McpError>) -> String {
        match outcome {
            Err(e) => {
                assert_eq!(e.code, DATASET_QUOTA_EXCEEDED, "{}", e.message);
                e.message.to_string()
            }
            Ok(r) => panic!("built instead of being refused: {}", text(&r)),
        }
    }

    /// `big` is refused before it builds anything when the session quota
    /// leaves `headroom` bytes above what the session holds, naming each of
    /// `mentions` and the limit, and the session's datasets stay as they
    /// were; `small`, under the same quota, runs and stores `small_out`.
    fn refuses_then_fits(
        qs: &QuotaSession,
        headroom: u64,
        big: impl std::future::Future<Output = Result<CallToolResult, McpError>>,
        mentions: &[&str],
        small: impl std::future::Future<Output = Result<CallToolResult, McpError>>,
        small_out: &str,
    ) {
        let held = qs.held();
        let names: Vec<String> = qs
            .session
            .datasets
            .blocking_read()
            .keys()
            .cloned()
            .collect();
        let quota = held + headroom;
        let message = refused_before_building(qs.call(Some(quota), big));
        for needle in mentions
            .iter()
            .chain(&["was not run", "per session", "Nothing was built"])
        {
            assert!(message.contains(needle), "missing {needle:?}: {message}");
        }
        let mut after: Vec<String> = qs
            .session
            .datasets
            .blocking_read()
            .keys()
            .cloned()
            .collect();
        let mut before = names;
        before.sort();
        after.sort();
        assert_eq!(after, before, "the refusal stored a dataset: {message}");
        assert_eq!(
            qs.held(),
            held,
            "the refusal changed the datasets: {message}"
        );
        let ok = qs
            .call(Some(quota), small)
            .unwrap_or_else(|e| panic!("an ordinary call is refused: {}", e.message));
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
        assert!(qs.has(small_out));
    }

    fn short_strings(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("v{}", i % 7)).collect()
    }

    /// A small dataset for the ordinary calls: ASCII strings, numbers and a
    /// four-level category.
    fn small_frame() -> DataFrame {
        let s: Vec<String> = (0..60).map(|i| format!("p{},q{}", i % 5, i % 3)).collect();
        let g: Vec<String> = (0..60).map(|i| format!("g{}", i % 4)).collect();
        df!("s" => s, "g" => g, "x" => noise(60), "y" => noise(60).iter().map(|v| v * 0.5 + 1.0).collect::<Vec<_>>())
            .unwrap()
    }

    #[test]
    fn str_concat_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![(
            "d",
            df!("s" => short_strings(2_000), "x" => noise(2_000)).unwrap(),
        )]);
        let req = |separator: &str, result: &str| StrConcatRequest {
            dataset: "d".into(),
            columns: vec!["s".into(), "s".into()],
            new_column: "c".into(),
            separator: Some(separator.into()),
            result_name: Some(result.into()),
        };
        // 2,000 rows, each with a 1,000-byte separator: about 2 MB of text.
        let wide = "|".repeat(1_000);
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.str_concat(Parameters(req(&wide, "big"))),
            &["`columns`", "1000-byte `separator`", "2000 rows"],
            qs.server.str_concat(Parameters(req("-", "small"))),
            "small",
        );
        // No quota installed (stdio, the off profile): the same call runs.
        let ran = qs
            .call(None, qs.server.str_concat(Parameters(req(&wide, "big"))))
            .unwrap();
        assert_ne!(ran.is_error, Some(true), "{}", text(&ran));
        let built = qs.frame("big");
        let first = built
            .column("c")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap()
            .len();
        assert_eq!(first, 2 + 1_000 + 2);
    }

    #[test]
    fn str_replace_value_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![("d", df!("s" => short_strings(2_000)).unwrap())]);
        let req = |new_value: &str, result: &str| ReplaceValueRequest {
            dataset: "d".into(),
            column: "s".into(),
            old_value: "v3".into(),
            new_value: new_value.into(),
            result_name: Some(result.into()),
        };
        // 286 of 2,000 rows replaced by 2,000 bytes each: about 570 kB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .str_replace_value(Parameters(req(&"w".repeat(2_000), "big"))),
            &["286 values", "2000-byte `new_value`"],
            qs.server.str_replace_value(Parameters(req("w", "small"))),
            "small",
        );
    }

    #[test]
    fn str_regex_replace_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![("d", df!("s" => short_strings(2_000)).unwrap())]);
        let req = |pattern: &str, replacement: &str, result: &str| RegexReplaceRequest {
            dataset: "d".into(),
            column: "s".into(),
            pattern: pattern.into(),
            replacement: replacement.into(),
            result_name: Some(result.into()),
        };
        // Every row's "v" becomes 1,000 bytes: about 2 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .str_regex_replace(Parameters(req("v", &"y".repeat(1_000), "big"))),
            &["`pattern`", "1000-byte `replacement`"],
            qs.server
                .str_regex_replace(Parameters(req("v", "w", "small"))),
            "small",
        );
        // Group references count: 500 copies of each two-byte match.
        let refs = "$0".repeat(500);
        refused_before_building(
            qs.call(
                Some(qs.held() + (256 << 10)),
                qs.server
                    .str_regex_replace(Parameters(req(r"v\d", &refs, "big"))),
            ),
        );
    }

    /// 2,000 rows of 86 bytes whose case mapping takes 246 (upper) or 126
    /// (lower) bytes each, and a small ASCII dataset.
    fn case_frames(upper: bool) -> QuotaSession {
        let grows = if upper { "ΐ" } else { "İ" };
        let u: Vec<String> = (0..2_000)
            .map(|i| format!("{}{i:06}", grows.repeat(40)))
            .collect();
        QuotaSession::new(vec![("d", df!("u" => u).unwrap()), ("e", small_frame())])
    }

    #[test]
    fn str_to_uppercase_is_refused_before_building_a_result_above_the_quota() {
        let qs = case_frames(true);
        let req = |dataset: &str, column: &str, result: &str| ToUppercaseRequest {
            dataset: dataset.into(),
            column: column.into(),
            result_name: Some(result.into()),
        };
        refuses_then_fits(
            &qs,
            200 << 10,
            qs.server.str_to_uppercase(Parameters(req("d", "u", "big"))),
            &["converting `column`"],
            qs.server
                .str_to_uppercase(Parameters(req("e", "s", "small"))),
            "small",
        );
    }

    #[test]
    fn str_to_lowercase_is_refused_before_building_a_result_above_the_quota() {
        let qs = case_frames(false);
        let req = |dataset: &str, column: &str, result: &str| ToLowercaseRequest {
            dataset: dataset.into(),
            column: column.into(),
            result_name: Some(result.into()),
        };
        refuses_then_fits(
            &qs,
            200 << 10,
            qs.server.str_to_lowercase(Parameters(req("d", "u", "big"))),
            &["converting `column`"],
            qs.server
                .str_to_lowercase(Parameters(req("e", "s", "small"))),
            "small",
        );
    }

    #[test]
    fn str_split_is_refused_before_building_a_result_above_the_quota() {
        let parts: Vec<String> = (0..2_000)
            .map(|i| {
                (0..40)
                    .map(|p| format!("{}", (i + p) % 10))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        let qs = QuotaSession::new(vec![
            ("d", df!("s" => parts).unwrap()),
            ("e", small_frame()),
        ]);
        let req = |dataset: &str, result: &str| StrSplitRequest {
            dataset: dataset.into(),
            column: "s".into(),
            pattern: ",".into(),
            max_splits: None,
            prefix: "p".into(),
            result_name: Some(result.into()),
        };
        // 40 String columns of 2,000 rows: about 1.4 MB of views.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.str_split(Parameters(req("d", "big"))),
            &["40 columns", "`pattern`"],
            qs.server.str_split(Parameters(req("e", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_fill_na_is_refused_before_building_a_result_above_the_quota() {
        let s: Vec<Option<&str>> = (0..2_000)
            .map(|i| (i % 10 == 0).then_some("kept"))
            .collect();
        let qs = QuotaSession::new(vec![("d", df!("s" => s, "x" => noise(2_000)).unwrap())]);
        let req = |strategy: &str, result: &str| FillNaRequest {
            dataset: "d".into(),
            columns: Some(vec!["s".into()]),
            strategy: strategy.into(),
            result_name: Some(result.into()),
        };
        // 1,800 nulls filled with 1,000 bytes each: about 1.8 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .munge_fill_na(Parameters(req(&"z".repeat(1_000), "big"))),
            &["1800 nulls", "`strategy`"],
            qs.server.munge_fill_na(Parameters(req("none", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_mutate_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![("d", df!("x" => noise(2_000)).unwrap())]);
        let req = |left: &str, result: &str| MutateColumnRequest {
            dataset: "d".into(),
            new_column: "c".into(),
            expr_type: "constant".into(),
            left: left.into(),
            operator: None,
            right: None,
            result_name: Some(result.into()),
        };
        // A 1,000-byte constant on 2,000 rows: about 2 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .munge_mutate(Parameters(req(&"q".repeat(1_000), "big"))),
            &["`expr_type` constant", "`left`"],
            qs.server.munge_mutate(Parameters(req("1.5", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_bin_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![("d", df!("x" => noise(2_000)).unwrap())]);
        let req = |strategy: &str, bins: Vec<f64>, labels: Option<Vec<String>>, result: &str| {
            BinColumnRequest {
                dataset: "d".into(),
                column: "x".into(),
                strategy: strategy.into(),
                bins,
                labels,
                result_name: Some(result.into()),
            }
        };
        // A 1,000-byte label on each of 2,000 rows: about 2 MB.
        let labels = Some(vec!["L".repeat(1_000), "R".repeat(1_000)]);
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_bin(Parameters(req(
                "custom",
                vec![-1e9, 0.0, 1e9],
                labels,
                "big",
            ))),
            &["`labels`"],
            qs.server
                .munge_bin(Parameters(req("uniform", vec![4.0], None, "small"))),
            "small",
        );
    }

    #[test]
    fn munge_one_hot_encode_is_refused_before_building_a_result_above_the_quota() {
        let levels: Vec<String> = (0..4_000).map(|i| format!("c{}", i % 400)).collect();
        let qs = QuotaSession::new(vec![
            ("d", df!("c" => levels).unwrap()),
            ("e", small_frame()),
        ]);
        let req = |dataset: &str, column: &str, result: &str| OneHotEncodeRequest {
            dataset: dataset.into(),
            column: column.into(),
            drop_first: Some(true),
            result_name: Some(result.into()),
        };
        // 400 Int32 columns of 4,000 rows: about 6.4 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .munge_one_hot_encode(Parameters(req("d", "c", "big"))),
            &["400 dummy columns", "`column`"],
            qs.server
                .munge_one_hot_encode(Parameters(req("e", "g", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_concat_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![("d", df!("x" => noise(2_000)).unwrap())]);
        let req = |copies: usize, result: &str| ConcatDatasetsRequest {
            datasets: vec!["d".to_string(); copies],
            result_name: Some(result.into()),
        };
        // 100 copies of 16 kB of values.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_concat(Parameters(req(100, "big"))),
            &["100 `datasets`"],
            qs.server.munge_concat(Parameters(req(2, "small"))),
            "small",
        );
    }

    #[test]
    fn munge_melt_is_refused_before_building_a_result_above_the_quota() {
        let mut df = df!(
            "id" => noise(1_000),
            "name" => (0..1_000).map(|i| format!("a name long enough for a buffer {i}")).collect::<Vec<_>>()
        )
        .unwrap();
        for v in 0..40 {
            df.with_column(Column::new(format!("v{v}").into(), noise(1_000)))
                .unwrap();
        }
        let qs = QuotaSession::new(vec![("d", df), ("e", small_frame())]);
        let req = |dataset: &str, value_vars: Vec<String>, result: &str| MeltDatasetRequest {
            dataset: dataset.into(),
            id_vars: vec!["id".into(), "name".into()],
            value_vars,
            variable_name: None,
            value_name: None,
            result_name: Some(result.into()),
        };
        // 40,000 long rows with both id columns: about 2 MB.
        let all: Vec<String> = (0..40).map(|v| format!("v{v}")).collect();
        let mut small = small_frame();
        small.rename("s", "name".into()).unwrap();
        small.rename("x", "id".into()).unwrap();
        qs.session
            .datasets
            .blocking_write()
            .insert("e".into(), Dataset::new(small));
        refuses_then_fits(
            &qs,
            512 << 10,
            qs.server.munge_melt(Parameters(req("d", all, "big"))),
            &["40 `value_vars`"],
            qs.server
                .munge_melt(Parameters(req("e", vec!["y".into()], "small"))),
            "small",
        );
    }

    #[test]
    fn munge_sample_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![(
            "d",
            df!("x" => noise(100), "y" => noise(100)).unwrap(),
        )]);
        let req = |n: usize, result: &str| SampleDatasetRequest {
            dataset: "d".into(),
            n,
            replace: Some(true),
            seed: Some(3),
            result_name: Some(result.into()),
        };
        // 100,000 sampled rows of two Float64 columns: 1.6 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_sample(Parameters(req(100_000, "big"))),
            &["100000 sampled rows (`n`)"],
            qs.server.munge_sample(Parameters(req(50, "small"))),
            "small",
        );
    }

    #[test]
    fn munge_join_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![
            ("a", df!("k" => vec![1i64; 300], "v" => noise(300)).unwrap()),
            (
                "b",
                df!("k" => (0..300i64).collect::<Vec<_>>(), "w" => noise(300)).unwrap(),
            ),
        ]);
        let req = |right: &str, result: &str| JoinDatasetsRequest {
            left: "a".into(),
            right: right.into(),
            left_on: vec!["k".into()],
            right_on: None,
            join_type: Some("inner".into()),
            suffix: None,
            result_name: Some(result.into()),
        };
        // 90,000 joined rows (every key equal) of three columns: 2.2 MB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_join(Parameters(req("a", "big"))),
            &["90000 joined rows", "`left_on`"],
            qs.server.munge_join(Parameters(req("b", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_group_by_is_refused_before_building_a_result_above_the_quota() {
        let g: Vec<String> = (0..4_000).map(|i| format!("g{}", i % 4)).collect();
        let qs = QuotaSession::new(vec![(
            "d",
            df!("id" => (0..4_000i64).collect::<Vec<_>>(), "g" => g, "x" => noise(4_000)).unwrap(),
        )]);
        let fns = [
            "count", "sum", "mean", "median", "min", "max", "std", "var", "first", "last",
        ];
        let req = |by: &str, result: &str| GroupByRequest {
            dataset: "d".into(),
            by: vec![by.into()],
            aggs: fns
                .iter()
                .map(|f| vec!["x".to_string(), f.to_string()])
                .collect(),
            result_name: Some(result.into()),
        };
        // 4,000 groups of ten aggregations: about 720 kB.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_group_by(Parameters(req("id", "big"))),
            &["10 `aggs`", "4000 groups"],
            qs.server.munge_group_by(Parameters(req("g", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_cast_is_refused_before_building_a_result_above_the_quota() {
        let b: Vec<bool> = (0..40_000).map(|i| i % 3 == 0).collect();
        let qs = QuotaSession::new(vec![("d", df!("b" => b).unwrap())]);
        let req = |dtype: &str, result: &str| CastColumnRequest {
            dataset: "d".into(),
            column: Some("b".into()),
            dtype: Some(dtype.into()),
            casts: None,
            result_name: Some(result.into()),
        };
        // 40,000 booleans (5 kB) as strings: 720 kB of views.
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server.munge_cast(Parameters(req("string", "big"))),
            &["`column`", "`dtype`"],
            qs.server.munge_cast(Parameters(req("int32", "small"))),
            "small",
        );
    }

    #[test]
    fn munge_standardize_is_refused_before_building_a_result_above_the_quota() {
        let mut df = df!("x0" => noise(2_000)).unwrap();
        for c in 1..50 {
            df.with_column(Column::new(format!("x{c}").into(), noise(2_000)))
                .unwrap();
        }
        let qs = QuotaSession::new(vec![("d", df)]);
        let req = |columns: Vec<String>, method: &str| StandardizeRequest {
            dataset: "d".into(),
            columns,
            method: Some(method.into()),
            result_name: None,
        };
        // In place: 50 new Float64 columns of 2,000 rows add 800 kB.
        let all: Vec<String> = (0..50).map(|c| format!("x{c}")).collect();
        for method in ["standardize", "normalize"] {
            let held = qs.held();
            refused_before_building(
                qs.call(
                    Some(held + (256 << 10)),
                    qs.server
                        .munge_standardize(Parameters(req(all.clone(), method))),
                ),
            );
            assert_eq!(qs.held(), held);
        }
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .munge_standardize(Parameters(req(all.clone(), "standardize"))),
            &["50 new columns", "`columns`"],
            qs.server
                .munge_standardize(Parameters(req(vec!["x0".into()], "normalize"))),
            "d",
        );
    }

    #[test]
    fn munge_diff_is_refused_before_building_a_result_above_the_quota() {
        let qs = QuotaSession::new(vec![
            ("d", df!("x" => noise(40_000)).unwrap()),
            ("e", small_frame()),
        ]);
        let req = |dataset: &str, diff_type: &str, result: &str| DiffRequest {
            dataset: dataset.into(),
            column: "x".into(),
            periods: Some(1),
            diff_type: Some(diff_type.into()),
            group_by: None,
            result_name: Some(result.into()),
        };
        // In a new dataset: 320 kB of values and 320 kB more.
        refused_before_building(qs.call(
            Some(qs.held() + (256 << 10)),
            qs.server.munge_diff(Parameters(req("d", "diff", "big"))),
        ));
        refuses_then_fits(
            &qs,
            256 << 10,
            qs.server
                .munge_diff(Parameters(req("d", "pct_change", "big"))),
            &["the new column", "`column`"],
            qs.server.munge_diff(Parameters(req("e", "diff", "small"))),
            "small",
        );
    }

    /// A frame of two chunks with nulls, inline and out-of-line strings,
    /// characters whose case mapping grows, numbers, booleans, repeated keys
    /// and a Categorical column.
    fn mixed_frame() -> DataFrame {
        let part = |offset: usize, n: usize| {
            let rows = offset..offset + n;
            let s: Vec<Option<String>> = rows
                .clone()
                .map(|i| match i % 4 {
                    0 => None,
                    1 => Some(format!("short{}", i % 3)),
                    _ => Some(format!("a fairly long value number {i}")),
                })
                .collect();
            let t: Vec<String> = rows
                .clone()
                .map(|i| format!("{}-ΐİ-{}", "x".repeat(i % 20), i % 9))
                .collect();
            let x: Vec<Option<f64>> = rows
                .clone()
                .map(|i| (i % 5 != 0).then(|| (i as f64 * 0.37).sin() * 3.0 + 0.1))
                .collect();
            let k: Vec<i64> = rows.clone().map(|i| (i % 6) as i64).collect();
            let i32s: Vec<i32> = rows.clone().map(|i| i as i32 * 3 - 7).collect();
            let b: Vec<bool> = rows.map(|i| i % 3 == 0).collect();
            let mut df =
                df!("s" => s, "t" => t, "x" => x, "k" => k, "i" => i32s, "b" => b).unwrap();
            let cat = df
                .column("t")
                .unwrap()
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap()
                .with_name("cat".into());
            df.with_column(cat).unwrap();
            df
        };
        let mut df = part(0, 40);
        df.vstack_mut(&part(40, 25)).unwrap();
        assert!(df.max_n_chunks() > 1);
        df
    }

    /// Assert that `bound` is at least what the quotas charge for the
    /// dataset the tool built (stored under an empty name: the admission
    /// adds the name's charge itself).
    #[track_caller]
    fn covers(what: &str, bound: Option<u64>, built: MungeResult<Dataset>) {
        let built = built.unwrap_or_else(|e| panic!("{what}: {e}"));
        let bound = bound.unwrap_or_else(|| panic!("{what}: no bound"));
        let charged = crate::session::datasets_bytes_of([("", &built)]);
        let columns: Vec<String> = built
            .df()
            .get_columns()
            .iter()
            .map(|c| {
                let charge = crate::session::series_bytes(c.as_materialized_series());
                format!("{} {:?} {charge}", c.name(), c.dtype())
            })
            .collect();
        assert!(
            bound >= charged,
            "{what}: bound {bound} < charged {charged} ({})",
            columns.join(", ")
        );
    }

    #[test]
    fn result_bounds_never_undercount_what_the_quotas_charge() {
        // polars shares a String column's buffers when it gathers from one
        // array and copies each value it gathers from several.
        let mut single = mixed_frame();
        single.as_single_chunk();
        assert_eq!(single.max_n_chunks(), 1);
        for df in [mixed_frame(), single] {
            bounds_cover(&df);
        }
    }

    fn bounds_cover(df: &DataFrame) {
        let df = df.clone();
        let ds = Dataset::new(df.clone());
        let long = "#".repeat(40);
        for sep in ["", "-", long.as_str()] {
            covers(
                "str_concat",
                bound::str_concat(&df, &["s", "t", "s"], "c", sep),
                str_concat(&ds, &["s", "t", "s"], "c", Some(sep)),
            );
            covers(
                "str_concat of one column",
                bound::str_concat(&df, &["s"], "s", sep),
                str_concat(&ds, &["s"], "s", Some(sep)),
            );
        }
        for (old, new) in [("short1", long.as_str()), ("short2", ""), ("none", "x")] {
            covers(
                "str_replace_value",
                bound::replace_value(&df, "s", old, new)
                    .unwrap()
                    .map(|(bytes, _)| bytes),
                replace(&ds, "s", old, new),
            );
        }
        for (pattern, replacement) in [("o", "$0$0$0"), ("", "_"), (r"\d+", long.as_str())] {
            let text = regex_replace_bytes(&ds, "t", pattern, replacement).unwrap();
            covers(
                "str_regex_replace",
                Some(bound::regex_replace(&df, "t", text)),
                regex_replace(&ds, "t", pattern, replacement),
            );
        }
        covers(
            "str_to_uppercase",
            bound::case_mapped(&df, "t", true).unwrap(),
            to_uppercase(&ds, "t"),
        );
        covers(
            "str_to_lowercase",
            bound::case_mapped(&df, "t", false).unwrap(),
            to_lowercase(&ds, "t"),
        );
        let parts = str_split_part_count(&ds, "t", "-", None).unwrap();
        covers(
            "str_split",
            bound::split(&df, "t", "p", parts),
            str_split(&ds, "t", "-", None, "p"),
        );
        for strategy in [
            FillStrategy::Constant(long.clone()),
            FillStrategy::Constant("7".into()),
            FillStrategy::Mean,
            FillStrategy::Median,
            FillStrategy::Forward,
            FillStrategy::Backward,
            FillStrategy::Zero,
        ] {
            for columns in [None, Some(&["s", "x"][..])] {
                let built = fill_na(&ds, columns, strategy.clone());
                if built.is_err() {
                    continue; // a constant that does not parse for a column
                }
                covers(
                    &format!("munge_fill_na {strategy:?} {columns:?}"),
                    bound::fill_na(&df, columns, &strategy),
                    built,
                );
            }
        }
        for expr in [
            MutateExpr::Constant(long.clone()),
            MutateExpr::Constant("short".into()),
            MutateExpr::Constant("2.5".into()),
            MutateExpr::Arithmetic("x".into(), ArithOp::Mul, "i".into()),
            MutateExpr::Function("log".into(), "k".into()),
        ] {
            covers(
                "munge_mutate",
                bound::mutate(&df, "x", &expr),
                mutate(&ds, "x", expr.clone()),
            );
        }
        let labels = [long.as_str(), "lo"];
        covers(
            "munge_bin with labels",
            bound::bin(&df, "x", long.len()),
            bin(
                &ds,
                "x",
                BinStrategy::Custom(vec![-10.0, 0.0, 10.0]),
                Some(&labels),
            ),
        );
        covers(
            "munge_bin",
            bound::bin(&df, "x", "bin_9".len()),
            bin(&ds, "x", BinStrategy::EqualWidth(10), None),
        );
        for (column, drop_first) in [("k", false), ("s", true), ("cat", false)] {
            covers(
                "munge_one_hot_encode",
                bound::one_hot(&df, column).unwrap().map(|(bytes, _)| bytes),
                one_hot_encode(&ds, column, drop_first),
            );
        }
        let other = Dataset::new(df.select(["k", "s"]).unwrap());
        let extra = Dataset::new(df!("z" => noise(9), "k" => vec![3i64; 9]).unwrap());
        covers(
            "munge_concat",
            Some(bound::concat(&[&df, other.df(), extra.df(), &df])),
            concat(&[&ds, &other, &extra, &ds]),
        );
        covers(
            "munge_melt",
            bound::melt(
                &df,
                &["k", "s", "cat"],
                &["x", "i", "b"],
                "variable",
                "value",
            ),
            melt(
                &ds,
                &["k", "s", "cat"],
                &["x", "i", "b"],
                "variable",
                "value",
            ),
        );
        for (n, replace) in [(500usize, true), (30, false)] {
            covers(
                "munge_sample",
                Some(bound::sample(&df, n as u64, replace)),
                sample(&ds, Some(n), None, replace, Some(7)),
            );
        }
        for join_type in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
        ] {
            let rows = estimate_join_rows(&ds, &other, &["k"], None, join_type).unwrap();
            covers(
                "munge_join",
                Some(bound::join(&df, other.df(), rows, "_right")),
                join(&ds, &other, &["k"], None, join_type, None),
            );
        }
        let aggs: Vec<AggSpec> = [
            ("x", AggFn::Count),
            ("x", AggFn::Sum),
            ("i", AggFn::Mean),
            ("x", AggFn::Median),
            ("s", AggFn::Min),
            ("t", AggFn::Max),
            ("x", AggFn::Std),
            ("i", AggFn::Var),
            ("s", AggFn::First),
            ("cat", AggFn::Last),
            ("t", AggFn::NUnique),
        ]
        .into_iter()
        .map(|(c, f)| AggSpec::new(c, f))
        .collect();
        for by in [&["k"][..], &["k", "b"], &["s"]] {
            covers(
                "munge_group_by",
                bound::group_by(&df, by, &aggs).map(|(bytes, _)| bytes),
                group_by(&ds, by, &aggs),
            );
        }
        for casts in [
            &[("cat", "string")][..],
            &[("b", "string"), ("x", "int")],
            &[("i", "float"), ("i", "string"), ("i", "int32")],
            &[("k", "string"), ("s", "string"), ("x", "bool")],
        ] {
            covers(
                "munge_cast",
                bound::cast(&df, casts).unwrap(),
                cast_columns(&ds, casts),
            );
        }
        covers(
            "munge_standardize",
            bound::standardize(&df, &["x", "i", "k"], "standardized"),
            standardize(&ds, &["x", "i", "k"]),
        );
        covers(
            "munge_standardize normalize",
            bound::standardize(&df, &["x", "i"], "normalized"),
            normalize(&ds, &["x", "i"]),
        );
        covers(
            "munge_diff",
            bound::diff(&df, "i", "i_diff1"),
            diff(&ds, "i", 1),
        );
        covers(
            "munge_diff pct_change",
            bound::diff(&df, "x", "x_pct_change2"),
            pct_change(&ds, "x", 2),
        );
    }

    /// Through the governor (the HTTP route's and the chat's path), a
    /// result above the session quota is refused by the check before it is
    /// built, not by the quota check after the call; one that fits runs.
    #[cfg(feature = "http")]
    #[tokio::test]
    async fn a_governed_call_is_refused_before_it_builds() {
        use crate::exec::{CallOrigin, ExecConfig, ToolCallError, ToolGovernor};
        let sessions = Arc::new(SessionManager::new(SessionConfig {
            ttl_minutes: 60,
            max_sessions: 4,
            evict_idle_minutes: None,
        }));
        let id = sessions.create_session(None).await.unwrap();
        let session = sessions.get_session(&id).await.unwrap();
        session.datasets.write().await.insert(
            "d".into(),
            Dataset::new(df!("s" => short_strings(2_000)).unwrap()),
        );
        let governor = ToolGovernor::new(
            ExecConfig {
                session_data_bytes: Some(1 << 20),
                ..Default::default()
            },
            sessions,
        );
        let server = Arc::new(AnalyticsServer::new());
        let request = |separator: String, result: &str| {
            serde_json::json!({
                "dataset": "d", "columns": ["s", "s"], "new_column": "c",
                "separator": separator, "result_name": result,
            })
        };
        let refused = governor
            .call(
                Arc::clone(&server),
                Arc::clone(&session),
                "str_concat",
                request("|".repeat(1_000), "big"),
                CallOrigin::Http,
            )
            .await;
        match refused {
            Err(ToolCallError::QuotaExceeded(message)) => {
                assert!(message.starts_with("str_concat was not run"), "{message}");
                assert!(message.contains("limit 1.00 MB per session"), "{message}");
            }
            other => panic!("expected a quota refusal, got {other:?}"),
        }
        assert!(!session.datasets.read().await.contains_key("big"));
        let ran = governor
            .call(
                server,
                Arc::clone(&session),
                "str_concat",
                request(" ".into(), "small"),
                CallOrigin::Http,
            )
            .await
            .unwrap();
        assert!(ran.success);
        assert!(session.datasets.read().await.contains_key("small"));
    }
}
