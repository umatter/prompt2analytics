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
use crate::tools::common::limit_refusal;
use crate::tools::limits;
use crate::tools::requests::viz::*;

use p2a_core::{
    ChartConfig,
    visualization::{
        box_plot, coefficient_plot, correlation_heatmap, dendrogram, event_study_plot, histogram,
        irf_plot, line_chart, residual_diagnostics, scatter_plot,
    },
};

#[tool_router(router = viz_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Static Chart Tools (PNG output)
    // ========================================================================

    /// Generate a histogram for a numeric column.
    #[tool(
        description = "Generate a histogram visualization for a numeric column. Returns a base64-encoded PNG image along with bin statistics."
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
                let description = format!(
                    "Histogram of '{}' generated successfully.\nBins: {}\nData points: {}",
                    request.column,
                    result.bins,
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
        description = "Generate a box plot visualization comparing the distributions of one or more numeric columns. Shows median, quartiles, and outliers. Returns a base64-encoded PNG image."
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
                Ok(c) => match c.f64() {
                    Ok(f) => f.into_iter().flatten().filter(|v| v.is_finite()).collect(),
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
                let mut description =
                    String::from("Box plot generated successfully.\n\nStatistics:");
                for stat in &result.statistics {
                    description.push_str(&format!(
                        "\n{}:\n  Min: {:.4}, Q1: {:.4}, Median: {:.4}, Q3: {:.4}, Max: {:.4}",
                        stat.label, stat.min, stat.q1, stat.median, stat.q3, stat.max
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
        description = "Generate four diagnostic plots for regression analysis: (1) Residuals vs Fitted, (2) Normal Q-Q plot, (3) Scale-Location, (4) Residuals vs Leverage. Also calculates Cook's distance for identifying influential observations. Returns four base64-encoded PNG images."
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
                // Find observations with high Cook's distance
                let high_influence: Vec<usize> = result
                    .cooks_distance
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0.5)
                    .map(|(i, _)| i)
                    .collect();

                let description = format!(
                    "Residual diagnostics generated (4 plots).\n\
                     Observations: {}\n\
                     High influence points (Cook's D > 0.5): {}\n\
                     Plots: Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage",
                    result.n_observations,
                    if high_influence.is_empty() {
                        "None".to_string()
                    } else {
                        format!("{:?}", high_influence)
                    }
                );
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
        description = "Generate a dendrogram (tree diagram) from hierarchical clustering results. Shows how clusters are merged at each level with merge distances. Takes a linkage matrix from hierarchical clustering output."
    )]
    pub async fn viz_dendrogram(
        &self,
        Parameters(request): Parameters<DendrogramRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Convert linkage matrix from Vec<Vec<f64>> to Vec<(usize, usize, f64, usize)>
        let linkage: Vec<(usize, usize, f64, usize)> = request
            .linkage_matrix
            .iter()
            .filter_map(|row| {
                if row.len() >= 4 {
                    Some((row[0] as usize, row[1] as usize, row[2], row[3] as usize))
                } else {
                    None
                }
            })
            .collect();

        if linkage.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid linkage matrix: must be array of [cluster1, cluster2, distance, size] tuples".to_string()
            )]));
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
