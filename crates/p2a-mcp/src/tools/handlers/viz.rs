//! Visualization tool handlers.
//!
//! This module defines visualization tools using the `#[tool_router(router = viz_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::common::{append_notes, limit_refusal};
use crate::tools::handlers::data::format_significant;
use crate::tools::limits;
use crate::tools::requests::viz::*;

use p2a_core::{
    ChartConfig,
    visualization::{
        box_plot, coefficient_plot, correlation_heatmap, dendrogram, event_study_plot, histogram,
        irf_plot, line_chart, residual_diagnostics, scatter_plot,
    },
};

/// 2^53: above it a JSON number does not hold every integer exactly.
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;

/// How the cluster ids of a linkage row are numbered, and how to convert R's.
const ID_RULE: &str = "Cluster ids are 0-based: 0..n-1 are the n samples and row i creates cluster \
                       n + i (scipy's linkage format). R's hclust()$merge numbers samples -1..-n \
                       and merges 1..n-1: use j - 1 for -j and n + k - 1 for k.";

/// A request number as an error message shows it ("1e300", not 301 digits).
fn shown(v: f64) -> String {
    if v.is_finite() && v.abs() < 1e15 {
        format!("{v}")
    } else {
        format!("{v:e}")
    }
}

/// The request's linkage rows as the core takes them: four numbers per row, the two cluster ids
/// finite non-negative whole numbers, the size a whole number from 2 to the number of samples.
/// The core checks the rest (ids that exist and were not merged before, heights).
fn linkage_rows(matrix: &[Vec<f64>]) -> Result<Vec<(usize, usize, f64, usize)>, String> {
    if matrix.is_empty() {
        return Err(
            "linkage_matrix is empty: give one [cluster1, cluster2, height, size] row per merge"
                .to_string(),
        );
    }
    let n_samples = matrix.len() + 1;
    let mut rows = p2a_core::resource::try_with_capacity(matrix.len(), "linkage rows")
        .map_err(|e| e.to_string())?;
    let id = |i: usize, what: &str, v: f64| -> Result<usize, String> {
        if !v.is_finite() {
            return Err(format!(
                "linkage_matrix row {i}: {what} must be a finite number, not {}",
                shown(v)
            ));
        }
        if v < 0.0 {
            return Err(format!(
                "linkage_matrix row {i}: {what} {} is negative. {ID_RULE}",
                shown(v)
            ));
        }
        if v.fract() != 0.0 {
            return Err(format!(
                "linkage_matrix row {i}: {what} {} is not a whole number. {ID_RULE}",
                shown(v)
            ));
        }
        if v > MAX_EXACT_INTEGER {
            return Err(format!(
                "linkage_matrix row {i}: {what} {} is too large",
                shown(v)
            ));
        }
        // A whole number in 0..=2^53.
        usize::try_from(v as u64)
            .map_err(|_| format!("linkage_matrix row {i}: {what} {} is too large", shown(v)))
    };
    for (i, row) in matrix.iter().enumerate() {
        let &[c1, c2, height, size] = row.as_slice() else {
            return Err(format!(
                "linkage_matrix row {i} has {} numbers; every row is [cluster1, cluster2, height, \
                 size]",
                row.len()
            ));
        };
        let c1 = id(i, "cluster1", c1)?;
        let c2 = id(i, "cluster2", c2)?;
        let whole_size = size.is_finite() && size.fract() == 0.0;
        if !whole_size || size < 2.0 || size > n_samples as f64 {
            return Err(format!(
                "linkage_matrix row {i}: size {} must be a whole number from 2 to {n_samples} (the \
                 number of samples in the new cluster)",
                shown(size)
            ));
        }
        rows.push((c1, c2, height, size as usize));
    }
    Ok(rows)
}

/// Outliers listed per column in the viz_boxplot reply (in the core's order); the rest are
/// counted, so a heavy-tailed column cannot push later columns past the chat loop's cut.
const BOXPLOT_OUTLIERS_LISTED: usize = 20;
#[tool_router(router = viz_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Static Chart Tools (PNG output)
    // ========================================================================

    /// Generate a histogram for a numeric column.
    #[tool(
        description = "Histogram of a numeric column (missing and non-finite values left out): equal-width bins from the minimum to the maximum, right-closed with the lowest value included, values within 1e-7 bin widths of a break assigned as R's hist() does; the bin count defaults to R's nclass.Sturges, ceiling(log2(n) + 1) (R's own hist() default also rounds the breaks with pretty(), which p2a does not). Returns a PNG image, the number of bins and the count per bin."
    )]
    pub async fn viz_histogram(
        &self,
        Parameters(request): Parameters<HistogramRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        // An owned copy (polars columns are reference-counted): the read lock is
        // released at the end of the statement, not held while the chart is
        // computed and rendered. Every handler below does the same.
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let col = match df.column(&request.column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.column, e
                ))]));
            }
        };

        let values: Vec<f64> = match col.cast(&DataType::Float64) {
            // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().flatten().filter(|v| v.is_finite()).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Not a numeric column: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert to numeric: {}",
                    e
                ))]));
            }
        };

        if values.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "No valid numeric values in column",
            )]));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.column.clone());
        config.y_label = Some("Frequency".to_string());

        match histogram(&values, request.bins, config) {
            Ok(result) => {
                let counts: Vec<String> = result.counts.iter().map(u32::to_string).collect();
                let description = format!(
                    "Histogram of '{}' generated successfully.\nBins: {}\nCounts per bin: {}\nData points: {}",
                    request.column,
                    result.bins,
                    counts.join(", "),
                    values.len()
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate histogram: {}",
                e
            ))])),
        }
    }

    /// Generate a scatter plot for two numeric columns.
    #[tool(
        description = "Generate a scatter plot visualization showing the relationship between two numeric columns. Returns a base64-encoded PNG image."
    )]
    pub async fn viz_scatter(
        &self,
        Parameters(request): Parameters<ScatterPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract X values
        let x_col = match df.column(&request.x_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x_column, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract Y values
        let y_col = match df.column(&request.y_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Y column '{}' not found: {}",
                    request.y_column, e
                ))]));
            }
        };

        let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
            // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert Y to numeric: {}",
                    e
                ))]));
            }
        };

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());
        config.y_label = Some(request.y_column.clone());

        match scatter_plot(&x_values, &y_values, config) {
            Ok(result) => {
                let description = format!(
                    "Scatter plot of {} vs {} generated successfully.\nPoints: {}\nCorrelation: {:.4}",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.correlation.unwrap_or(f64::NAN)
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate scatter plot: {}",
                e
            ))])),
        }
    }

    /// Generate a line chart for time series or sequential data.
    #[tool(
        description = "Generate a line chart visualization for time series or sequential data. Supports multiple Y series. Returns a base64-encoded PNG image."
    )]
    pub async fn viz_line(
        &self,
        Parameters(request): Parameters<LineChartRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract X values (shared across all series)
        let x_col = match df.column(&request.x_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x_column, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract Y series - API expects (name, x_vals, y_vals) tuples
        let mut series: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut series_names = Vec::new();
        for y_col_name in &request.y_columns {
            let y_col = match df.column(y_col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column '{}' not found: {}",
                        y_col_name, e
                    ))]));
                }
            };

            let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
                // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
                Ok(c) => match c.f64() {
                    Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Y column '{}' not numeric: {}",
                            y_col_name, e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot convert Y column '{}' to numeric: {}",
                        y_col_name, e
                    ))]));
                }
            };
            series_names.push(y_col_name.clone());
            series.push((y_col_name.clone(), x_values.clone(), y_values));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());

        match line_chart(&series, config) {
            Ok(result) => {
                let description = format!(
                    "Line chart generated successfully.\nX: {}\nSeries: {}\nPoints: {}",
                    request.x_column,
                    series_names.join(", "),
                    result.n_points
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate line chart: {}",
                e
            ))])),
        }
    }

    /// Generate a box plot for comparing distributions.
    #[tool(
        description = "Box plots of one or more numeric columns with Tukey hinges and whiskers to the most extreme values within 1.5 IQR, as R's boxplot.stats; values beyond them are listed as outliers (at most 20 per column, then the count of the rest). Missing values are left out; ±Inf count as values and are listed as outliers but not drawn. Returns a PNG image and the statistics with n per column."
    )]
    pub async fn viz_boxplot(
        &self,
        Parameters(request): Parameters<BoxPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract data for each column
        let mut groups = Vec::new();
        for col_name in &request.columns {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found: {}",
                        col_name, e
                    ))]));
                }
            };

            let values: Vec<f64> = match col.cast(&DataType::Float64) {
                // na-legacy(T4): per-column read (nulls dropped or NaN, no missing-value note)
                Ok(c) => match c.f64() {
                    // NaN is missing; ±Inf are values, as in R's boxplot.stats.
                    Ok(f) => f.into_iter().flatten().filter(|v| !v.is_nan()).collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' not numeric: {}",
                            col_name, e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot convert '{}' to numeric: {}",
                        col_name, e
                    ))]));
                }
            };
            groups.push((col_name.clone(), values));
        }

        if groups.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "No valid columns specified",
            )]));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.y_label = Some("Value".to_string());

        match box_plot(&groups, config) {
            Ok(result) => {
                let mut description = String::from(
                    "Box plot generated successfully (Tukey hinges, whiskers to the most extreme \
                     values within 1.5 IQR, as R's boxplot.stats).\n\nStatistics:",
                );
                for stat in &result.statistics {
                    if stat.n == 0 {
                        description.push_str(&format!("\n{} (n = 0): no values", stat.label));
                        continue;
                    }
                    let outliers = if stat.outliers.is_empty() {
                        "none".to_string()
                    } else {
                        let shown = stat
                            .outliers
                            .iter()
                            .take(BOXPLOT_OUTLIERS_LISTED)
                            .map(|v| format!("{v:.4}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        match stat.outliers.len().saturating_sub(BOXPLOT_OUTLIERS_LISTED) {
                            0 => shown,
                            more => format!("{shown}, and {more} more"),
                        }
                    };
                    description.push_str(&format!(
                        "\n{} (n = {}):\n  Lower whisker: {:.4}, Q1 (lower hinge): {:.4}, Median: {:.4}, \
                         Q3 (upper hinge): {:.4}, Upper whisker: {:.4}\n  Outliers: {outliers}",
                        stat.label, stat.n, stat.min, stat.q1, stat.median, stat.q3, stat.max
                    ));
                }
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate box plot: {}",
                e
            ))])),
        }
    }

    /// Generate a correlation heatmap.
    #[tool(
        description = "Generate a correlation heatmap visualization for numeric columns. Uses a diverging blue-white-red colormap. Returns a base64-encoded PNG image."
    )]
    pub async fn viz_heatmap(
        &self,
        Parameters(request): Parameters<HeatmapRequest>,
    ) -> Result<CallToolResult, McpError> {
        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        // Compute correlation matrix
        let corr_result = match p2a_core::stats::correlation_matrix(&dataset) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to compute correlation: {}",
                    e
                ))]));
            }
        };

        if corr_result.columns.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Need at least 2 numeric columns for correlation heatmap",
            )]));
        }

        // Filter to specified columns if provided
        let (matrix, columns) = if let Some(ref selected_cols) = request.columns {
            // Find indices of requested columns
            let indices: Vec<usize> = selected_cols
                .iter()
                .filter_map(|name| corr_result.columns.iter().position(|c| c == name))
                .collect();

            if indices.len() < 2 {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Need at least 2 valid numeric columns for correlation heatmap",
                )]));
            }

            // Build filtered matrix
            let filtered_matrix: Vec<Vec<f64>> = indices
                .iter()
                .map(|&i| indices.iter().map(|&j| corr_result.matrix[i][j]).collect())
                .collect();
            let filtered_cols: Vec<String> = indices
                .iter()
                .map(|&i| corr_result.columns[i].clone())
                .collect();

            (filtered_matrix, filtered_cols)
        } else {
            (corr_result.matrix.clone(), corr_result.columns.clone())
        };

        match correlation_heatmap(
            &matrix,
            &columns,
            &columns,
            request.title.as_deref(),
            None,
            None,
        ) {
            Ok(result) => {
                let description = format!(
                    "Correlation heatmap generated successfully.\nVariables: {}",
                    columns.join(", ")
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate heatmap: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // Interactive Chart Tools (HTML/Plotly output)
    // ========================================================================

    /// Generate an interactive scatter plot with Plotly.js.
    #[tool(
        description = "Generate an interactive scatter plot visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Supports grouping data by a categorical column for colored traces. Interactive features include zoom, pan, and hover."
    )]
    pub async fn viz_scatter_interactive(
        &self,
        Parameters(request): Parameters<ScatterInteractiveRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::visualization::{InteractiveConfig, scatter_interactive};

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let config = InteractiveConfig {
            title: request.title.clone(),
            x_label: Some(request.x_column.clone()),
            y_label: Some(request.y_column.clone()),
            ..Default::default()
        };

        match scatter_interactive(
            dataset.df(),
            &request.x_column,
            &request.y_column,
            request.group_column.as_deref(),
            config,
        ) {
            Ok(result) => {
                let description = format!(
                    "Interactive scatter plot generated successfully.\nX: {}\nY: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.html.len()
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::text(format!("```html\n{}\n```", result.html)),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate interactive scatter plot: {}",
                e
            ))])),
        }
    }

    /// Generate an interactive histogram with Plotly.js.
    #[tool(
        description = "Generate an interactive histogram visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Supports grouping data by a categorical column for overlaid histograms. Interactive features include zoom, pan, and hover."
    )]
    pub async fn viz_histogram_interactive(
        &self,
        Parameters(request): Parameters<HistogramInteractiveRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::visualization::{InteractiveConfig, histogram_interactive};

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let config = InteractiveConfig {
            title: request.title.clone(),
            x_label: Some(request.column.clone()),
            y_label: Some("Frequency".to_string()),
            ..Default::default()
        };

        match histogram_interactive(
            dataset.df(),
            &request.column,
            request.group_column.as_deref(),
            config,
        ) {
            Ok(result) => {
                let description = format!(
                    "Interactive histogram generated successfully.\nColumn: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.column,
                    result.n_points,
                    result.html.len()
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::text(format!("```html\n{}\n```", result.html)),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate interactive histogram: {}",
                e
            ))])),
        }
    }

    /// Generate an interactive line chart with Plotly.js.
    #[tool(
        description = "Generate an interactive line chart visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Interactive features include zoom, pan, and hover."
    )]
    pub async fn viz_line_interactive(
        &self,
        Parameters(request): Parameters<LineInteractiveRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::visualization::{InteractiveConfig, line_interactive};

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let config = InteractiveConfig {
            title: request.title.clone(),
            x_label: Some(request.x_column.clone()),
            y_label: Some(request.y_column.clone()),
            ..Default::default()
        };

        match line_interactive(dataset.df(), &request.x_column, &request.y_column, config) {
            Ok(result) => {
                let description = format!(
                    "Interactive line chart generated successfully.\nX: {}\nY: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.html.len()
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::text(format!("```html\n{}\n```", result.html)),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate interactive line chart: {}",
                e
            ))])),
        }
    }

    // ========================================================================
    // Specialized Chart Tools
    // ========================================================================

    /// Generate an event study plot for treatment effect visualization.
    #[tool(
        description = "Generate an event study plot showing treatment effects over time with confidence intervals. Used for visualizing DiD or panel event study results. Shows point estimates with CI bands and reference lines at t=0 and y=0."
    )]
    pub async fn viz_event_study(
        &self,
        Parameters(request): Parameters<EventStudyRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df
                .column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col
                .cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted
                // na-legacy(T4): null read as NaN (no missing-value note)
                .f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let time = match extract_numeric(&request.time_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let estimates = match extract_numeric(&request.estimate_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_lower = match extract_numeric(&request.ci_lower_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_upper = match extract_numeric(&request.ci_upper_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some("Time Relative to Treatment".to_string());
        config.y_label = Some("Effect".to_string());

        match event_study_plot(&time, &estimates, &ci_lower, &ci_upper, config) {
            Ok(result) => {
                let description = format!(
                    "Event study plot generated successfully.\nPeriods: {}",
                    result.n_periods
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate event study plot: {}",
                e
            ))])),
        }
    }

    /// Generate a coefficient plot with confidence intervals.
    #[tool(
        description = "Generate a coefficient plot showing regression coefficients with confidence intervals (error bars). Useful for visualizing regression results. Shows vertical zero line for reference."
    )]
    pub async fn viz_coefficient(
        &self,
        Parameters(request): Parameters<CoefficientPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract name column
        let name_col = match df.column(&request.name_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Name column '{}' not found: {}",
                    request.name_column, e
                ))]));
            }
        };
        let names: Vec<String> = match name_col.str() {
            Ok(s) => s.into_iter().map(|v| v.unwrap_or("").to_string()).collect(),
            Err(_) => (0..name_col.len()).map(|i| format!("Var_{}", i)).collect(),
        };

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df
                .column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col
                .cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted
                // na-legacy(T4): null read as NaN (no missing-value note)
                .f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let estimates = match extract_numeric(&request.estimate_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_lower = match extract_numeric(&request.ci_lower_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_upper = match extract_numeric(&request.ci_upper_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mut config = ChartConfig::default();
        config.title = request.title;

        let horizontal = request.horizontal.unwrap_or(true);

        match coefficient_plot(&names, &estimates, &ci_lower, &ci_upper, config, horizontal) {
            Ok(result) => {
                let description = format!(
                    "Coefficient plot generated successfully.\nCoefficients: {}\nOrientation: {}",
                    result.n_coefficients,
                    if horizontal { "horizontal" } else { "vertical" }
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate coefficient plot: {}",
                e
            ))])),
        }
    }

    /// Generate an IRF (Impulse Response Function) plot.
    #[tool(
        description = "Generate an Impulse Response Function (IRF) plot from VAR models. Shows how a variable responds to a shock over time. Optionally includes confidence bands."
    )]
    pub async fn viz_irf(
        &self,
        Parameters(request): Parameters<IrfPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df
                .column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col
                .cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted
                // na-legacy(T4): null read as NaN (no missing-value note)
                .f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let horizon = match extract_numeric(&request.horizon_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let response = match extract_numeric(&request.response_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract optional CI columns
        let ci_lower: Option<Vec<f64>> = if let Some(ref col_name) = request.ci_lower_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let ci_upper: Option<Vec<f64>> = if let Some(ref col_name) = request.ci_upper_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let shock_label = request.shock_label.as_deref();
        let response_label = request.response_label.as_deref();
        let config = ChartConfig {
            title: request.title,
            ..ChartConfig::default()
        };

        let has_ci = ci_lower.is_some() && ci_upper.is_some();

        match irf_plot(
            &horizon,
            &response,
            ci_lower.as_deref(),
            ci_upper.as_deref(),
            shock_label,
            response_label,
            config,
        ) {
            Ok(result) => {
                let description = format!(
                    "IRF plot generated successfully.\nHorizons: {}\nHas CI bands: {}\nShock: {}\nResponse: {}",
                    result.n_horizons,
                    has_ci,
                    result.shock.as_deref().unwrap_or("unnamed"),
                    result.response.as_deref().unwrap_or("unnamed")
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate IRF plot: {}",
                e
            ))])),
        }
    }

    /// Generate residual diagnostic plots for regression model validation.
    #[tool(
        description = "Four diagnostic plots of a fitted regression (Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage), with standardized residuals and Cook's distance as R's rstandard() and cooks.distance(). Exact when the model's hat values are given (leverage_column; R: hatvalues(model)); without them the hat values of a regression on one predictor are used and a warning says so. Rows with a missing fitted value or residual are left out. Reports the residual standard error and the dataset rows with Cook's distance above 0.5. Returns four PNG images."
    )]
    pub async fn viz_residual_diagnostics(
        &self,
        Parameters(request): Parameters<ResidualDiagnosticsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let dataset = self.datasets.read().await.get(&request.dataset).cloned();
        let dataset = match dataset {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df
                .column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col
                .cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted
                // na-legacy(T4): null read as NaN (no missing-value note)
                .f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let fitted = match extract_numeric(&request.fitted_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let residuals = match extract_numeric(&request.residuals_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract optional leverage column
        let leverage: Option<Vec<f64>> = if let Some(ref col_name) = request.leverage_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let config = ChartConfig::default();

        match residual_diagnostics(&fitted, &residuals, leverage.as_deref(), config) {
            Ok(result) => {
                // Observations with high Cook's distance, as dataset rows.
                let high_influence: Vec<usize> = result
                    .cooks_distance
                    .iter()
                    .zip(&result.rows)
                    .filter(|(d, _)| **d > 0.5)
                    .map(|(_, &row)| row)
                    .collect();

                let mut description = format!(
                    "Residual diagnostics generated (4 plots).\n\
                     Observations: {}\n\
                     Residual standard error: {} ({} coefficients)\n\
                     High influence points (Cook's D > 0.5; dataset rows, 0-based): {}\n\
                     Plots: Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage",
                    result.n_observations,
                    format_significant(result.sigma),
                    result.n_params,
                    if high_influence.is_empty() {
                        "None".to_string()
                    } else {
                        format!("{:?}", high_influence)
                    }
                );
                append_notes(&mut description, None, &result.warnings);
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.residuals_vs_fitted, "image/png"),
                    Content::image(result.qq_plot, "image/png"),
                    Content::image(result.scale_location, "image/png"),
                    Content::image(result.residuals_vs_leverage, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate residual diagnostics: {}",
                e
            ))])),
        }
    }

    /// Generate a dendrogram visualization from hierarchical clustering results.
    #[tool(
        description = "Draw a dendrogram (tree diagram) of a hierarchical clustering from its linkage matrix in scipy's format: one row [cluster1, cluster2, height, size] per merge, in merge order; a matrix with r rows has r + 1 samples. Cluster ids are 0-based: 0..n-1 are the samples and row i creates cluster n + i; size is the number of samples in the new cluster. ml_hierarchical prints its merge history in this format. For R's hclust() use row i = [m(merge[i, 1]), m(merge[i, 2]), height[i], size] with m(-j) = j - 1 and m(k) = n + k - 1. Malformed rows are refused."
    )]
    pub async fn viz_dendrogram(
        &self,
        Parameters(request): Parameters<DendrogramRequest>,
    ) -> Result<CallToolResult, McpError> {
        let linkage = match linkage_rows(&request.linkage_matrix) {
            Ok(rows) => rows,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        if let Some(labels) = &request.labels
            && labels.len() != linkage.len() + 1
        {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "labels has {} entries, but a linkage matrix with {} rows has {} samples; give one \
                 label per sample, in sample order (R's plot.hclust refuses this too)",
                labels.len(),
                linkage.len(),
                linkage.len() + 1
            ))]));
        }

        let width = request.width.unwrap_or(800);
        let height = request.height.unwrap_or(600);
        let limits = limits::current();
        if let Err(v) = limits.check_budget(
            "viz_dendrogram",
            "image_pixels",
            u64::from(width) * u64::from(height),
            limits.image_pixels,
        ) {
            return Ok(limit_refusal(v));
        }

        let config = ChartConfig {
            width,
            height,
            title: request.title,
            x_label: None,
            y_label: Some("Distance".to_string()),
            ..Default::default()
        };

        match dendrogram(&linkage, request.labels.as_deref(), config) {
            Ok(result) => {
                let description = format!(
                    "Dendrogram generated successfully.\nSamples: {}\nMerge steps: {}\nMax distance: {:.4}",
                    result.n_samples, result.n_merges, result.max_distance
                );
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    Content::image(result.image_base64, "image/png"),
                ]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate dendrogram: {}",
                e
            ))])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::limits::{self, Limits};
    use p2a_core::Dataset;
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

    #[test]
    fn linkage_rows_refuses_malformed_rows() {
        // DATA-25: `row[i] as usize` turned -1 and NaN into 0 and 0.5 into 0, and short rows were dropped.
        assert_eq!(
            linkage_rows(&[vec![0.0, 1.0, 1.5, 2.0], vec![2.0, 3.0, 2.5, 3.0]]).unwrap(),
            vec![(0, 1, 1.5, 2), (2, 3, 2.5, 3)]
        );
        for (m, needle) in [
            (vec![], "linkage_matrix is empty"),
            (
                vec![vec![-1.0, 1.0, 1.0, 2.0]],
                "linkage_matrix row 0: cluster1 -1 is negative",
            ),
            (
                vec![vec![0.0, 0.5, 1.0, 2.0]],
                "linkage_matrix row 0: cluster2 0.5 is not a whole number",
            ),
            (
                vec![vec![0.0, 1.0, 1.0, 2.0], vec![2.0, 3.0]],
                "linkage_matrix row 1 has 2 numbers",
            ),
            (
                vec![vec![0.0, 1.0, 1.0, 2.0, 7.0]],
                "linkage_matrix row 0 has 5 numbers",
            ),
            (
                vec![vec![0.0, f64::NAN, 1.0, 2.0]],
                "linkage_matrix row 0: cluster2 must be a finite number, not NaN",
            ),
            (
                vec![vec![1e300, 1.0, 1.0, 2.0]],
                "linkage_matrix row 0: cluster1 1e300 is too large",
            ),
            (
                vec![vec![0.0, 1.0, 1.0, 1.5]],
                "linkage_matrix row 0: size 1.5 must be a whole number from 2 to 2",
            ),
            (
                vec![vec![0.0, 1.0, 1.0, 3.0]],
                "linkage_matrix row 0: size 3 must be a whole number from 2 to 2",
            ),
        ] {
            let e = linkage_rows(&m).unwrap_err();
            assert!(e.contains(needle), "{needle}: {e}");
        }
        assert!(
            linkage_rows(&[vec![-1.0, 1.0, 1.0, 2.0]])
                .unwrap_err()
                .contains("R's hclust()$merge")
        );
    }

    #[test]
    fn dendrogram_pixels_are_budgeted() {
        let server = AnalyticsServer::new();
        let req = |w: u32, h: u32| DendrogramRequest {
            linkage_matrix: vec![vec![0.0, 1.0, 1.0, 2.0], vec![2.0, 3.0, 1.5, 3.0]],
            labels: None,
            width: Some(w),
            height: Some(h),
            title: None,
        };
        let r = run(
            hosted_with("image_pixels=100000"),
            server.viz_dendrogram(Parameters(req(400, 300))),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(
            text(&r).contains("needs 120000 (image_pixels)"),
            "{}",
            text(&r)
        );
        let ok = run(
            hosted_with("image_pixels=100000"),
            server.viz_dendrogram(Parameters(req(300, 300))),
        );
        assert_ne!(ok.is_error, Some(true), "{}", text(&ok));
    }

    #[test]
    fn histogram_reports_the_bins_it_used() {
        let v: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.41).sin() * 3.0 + i as f64 * 0.05)
            .collect();
        let server = server_with(vec![("d", df!("v" => v).unwrap())]);
        let req = |bins| HistogramRequest {
            dataset: "d".into(),
            column: "v".into(),
            bins,
            title: None,
        };
        // Sturges' rule for n = 100 gives 8 bins.
        let t = text(&run(
            Limits::off(),
            server.viz_histogram(Parameters(req(None))),
        ));
        assert!(t.contains("Bins: 8"), "{t}");
        let t = text(&run(
            Limits::off(),
            server.viz_histogram(Parameters(req(Some(4)))),
        ));
        assert!(t.contains("Bins: 4"), "{t}");
    }

    #[test]
    fn charts_of_overflowing_data_are_tool_errors() {
        let server = server_with(vec![(
            "h",
            df!("v" => [-1e308, 0.0, 1e308], "w" => [1.0, 2.0, 3.0]).unwrap(),
        )]);
        let r = run(
            Limits::off(),
            server.viz_histogram(Parameters(HistogramRequest {
                dataset: "h".into(),
                column: "v".into(),
                bins: None,
                title: None,
            })),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("axis"), "{}", text(&r));
    }
}
