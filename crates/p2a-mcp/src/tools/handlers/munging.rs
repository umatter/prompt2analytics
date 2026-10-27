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

        // With a join_output_rows budget, predict the result size from per-key
        // counts before joining.
        let limits = limits::current();
        if limits.join_output_rows.is_some() {
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
        }

        let result = match join(&left_ds, &right_ds, &left_on, right_on, join_type, suffix) {
            Ok(ds) => ds,
            Err(e) => return Ok(join_failed(&e)),
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_{}", request.left, request.right));
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
        let datasets = self.datasets.read().await;

        let mut ds_list: Vec<&Dataset> = Vec::new();
        for name in &request.datasets {
            match datasets.get(name) {
                Some(ds) => ds_list.push(ds),
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Dataset '{}' not found.",
                        name
                    ))]));
                }
            }
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| "concatenated".to_string());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
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

        let result = match group_by(dataset, &by_cols, &agg_specs) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Group by failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_grouped", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
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

        let id_vars: Vec<&str> = request.id_vars.iter().map(|s| s.as_str()).collect();
        let value_vars: Vec<&str> = request.value_vars.iter().map(|s| s.as_str()).collect();
        let variable_name = request.variable_name.as_deref().unwrap_or("variable");
        let value_name = request.value_name.as_deref().unwrap_or("value");

        let result = match melt(dataset, &id_vars, &value_vars, variable_name, value_name) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Melt failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_melted", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
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

        let result = match fill_na(dataset, columns.as_deref(), strategy) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fill NA failed: {}",
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

        let result = match to_lowercase(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To lowercase failed: {}",
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

        let result = match to_uppercase(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To uppercase failed: {}",
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

        let result = match replace(
            dataset,
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
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

        let result = match regex_replace(
            dataset,
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
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

        // With a str_split_parts budget, count the output columns first.
        let limits = limits::current();
        if limits.str_split_parts.is_some() {
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

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

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let separator = request.separator.as_deref();

        let result = match str_concat(dataset, &columns, &request.new_column, separator) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String concat failed: {}",
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
        let method = request.method.as_deref().unwrap_or("standardize");

        let result = match method {
            "standardize" => standardize(dataset, &cols),
            "normalize" => normalize(dataset, &cols),
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
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

        let result = match bin(&dataset, &request.column, strategy, labels.as_deref()) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bin failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

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

        let result = match one_hot_encode(&dataset, &request.column, drop_first) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "One-hot encode failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
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

        let periods = request.periods.unwrap_or(1) as usize;
        let diff_type = request.diff_type.as_deref().unwrap_or("diff");

        let result = match diff_type {
            "diff" => diff(dataset, &request.column, periods),
            "pct_change" => pct_change(dataset, &request.column, periods),
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

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());
        let new_col = format!("{}_{}{}", request.column, diff_type, periods);

        drop(datasets);
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

        let replace = request.replace.unwrap_or(false);
        let seed = request.seed;

        let result = match sample(dataset, Some(request.n), None, replace, seed) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sample failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| format!("{}_sample", request.dataset));
        let n_rows = result.nrows();

        drop(datasets);
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

        let result = match mutate(dataset, &request.new_column, expr) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mutate failed: {}",
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

        let result = if cast_refs.len() == 1 {
            match cast(dataset, cast_refs[0].0, cast_refs[0].1) {
                Ok(ds) => ds,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cast failed: {}",
                        e
                    ))]));
                }
            }
        } else {
            match cast_columns(dataset, &cast_refs) {
                Ok(ds) => ds,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cast failed: {}",
                        e
                    ))]));
                }
            }
        };

        let result_name = request
            .result_name
            .unwrap_or_else(|| request.dataset.clone());

        let cast_desc: Vec<String> = cast_pairs
            .iter()
            .map(|(c, d)| format!("{} -> {}", c, d))
            .collect();

        drop(datasets);
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
/// once: the core's [`MungeError::JoinError`] already reads that way, so only
/// its other errors (a key column missing from a dataset) get the prefix here.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::limits::{self, Limits};
    use p2a_core::polars::prelude::*;
    use rmcp::model::RawContent;

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
        let cases: [(&[&str], Option<&[&str]>, &str); 4] = [
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
            (
                &["nope"],
                None,
                "Join failed: Column 'nope' not found in dataset",
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
}
