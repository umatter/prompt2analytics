//! Request types for visualization tools.

use schemars::JsonSchema;
use serde::Deserialize;

// ============================================================================
// Static Chart Requests (PNG output)
// ============================================================================

/// Request to generate a histogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HistogramRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name to plot
    #[schemars(description = "Name of the numeric column to create histogram from.")]
    pub column: String,

    /// Number of bins (optional, auto-calculated if not specified)
    #[schemars(
        description = "Number of bins for the histogram. If not specified, uses Sturges' rule."
    )]
    pub bins: Option<usize>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a scatter plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScatterPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values.")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a line chart.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineChartRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values (e.g., time index).")]
    pub x_column: String,

    /// Y-axis column names (one or more series)
    #[schemars(
        description = "Names of the columns to plot as lines (can be multiple for multi-series)."
    )]
    pub y_columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a box plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BoxPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include in box plot
    #[schemars(description = "Names of numeric columns to create box plots for.")]
    pub columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a correlation heatmap.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeatmapRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include (optional, uses all numeric if not specified)
    #[schemars(
        description = "Names of numeric columns to include. If not specified, uses all numeric columns."
    )]
    pub columns: Option<Vec<String>>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the heatmap.")]
    pub title: Option<String>,
}

// ============================================================================
// Interactive Chart Requests (HTML/Plotly output)
// ============================================================================

/// Request to generate an interactive scatter plot (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScatterInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values.")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Group column for separate traces (optional)
    #[schemars(
        description = "Optional column for grouping data points into separate traces with different colors."
    )]
    pub group_column: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate an interactive histogram (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HistogramInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name to plot
    #[schemars(description = "Name of the numeric column to create histogram from.")]
    pub column: String,

    /// Group column for separate traces (optional)
    #[schemars(
        description = "Optional column for grouping data into separate overlaid histograms."
    )]
    pub group_column: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate an interactive line chart (HTML/Plotly output).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineInteractiveRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values (e.g., time index).")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

// ============================================================================
// Specialized Chart Requests
// ============================================================================

/// Request to generate an event study plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EventStudyRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time/period relative to treatment
    #[schemars(
        description = "Column with time periods relative to treatment (e.g., -3, -2, -1, 0, 1, 2, 3)."
    )]
    pub time_column: String,

    /// Column name for point estimates
    #[schemars(description = "Column with coefficient estimates at each time period.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a coefficient plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CoefficientPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for variable/coefficient names
    #[schemars(description = "Column with variable names or coefficient labels.")]
    pub name_column: String,

    /// Column name for coefficient estimates
    #[schemars(description = "Column with coefficient point estimates.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,

    /// Horizontal orientation (optional, default: true)
    #[schemars(
        description = "If true, draw horizontal error bars (default). If false, draw vertical."
    )]
    pub horizontal: Option<bool>,
}

/// Request to generate an IRF (Impulse Response Function) plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IrfPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time horizon
    #[schemars(description = "Column with time horizon (0, 1, 2, ...).")]
    pub horizon_column: String,

    /// Column name for response values
    #[schemars(description = "Column with impulse response values.")]
    pub response_column: String,

    /// Column name for lower confidence interval bound (optional)
    #[schemars(description = "Optional column with lower bound of confidence interval.")]
    pub ci_lower_column: Option<String>,

    /// Column name for upper confidence interval bound (optional)
    #[schemars(description = "Optional column with upper bound of confidence interval.")]
    pub ci_upper_column: Option<String>,

    /// Label for the shock (optional)
    #[schemars(description = "Optional label for the shock variable.")]
    pub shock_label: Option<String>,

    /// Label for the response (optional)
    #[schemars(description = "Optional label for the response variable.")]
    pub response_label: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate residual diagnostic plots.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResidualDiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(
        description = "Name or ID of a previously loaded dataset containing regression results."
    )]
    pub dataset: String,

    /// Column name for fitted/predicted values
    #[schemars(description = "Column with fitted (predicted) values from regression.")]
    pub fitted_column: String,

    /// Column name for residual values
    #[schemars(description = "Column with residual values (observed - fitted).")]
    pub residuals_column: String,

    /// Column name for leverage (hat) values (optional)
    #[schemars(
        description = "Optional column with leverage (hat) values. If not provided, will be estimated."
    )]
    pub leverage_column: Option<String>,
}

/// Request to visualize hierarchical clustering results as a dendrogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DendrogramRequest {
    /// Linkage matrix from hierarchical clustering (JSON array of arrays)
    #[schemars(
        description = "Linkage matrix from hierarchical clustering. Array of [cluster1, cluster2, distance, size] tuples."
    )]
    pub linkage_matrix: Vec<Vec<f64>>,

    /// Optional labels for leaf nodes
    #[schemars(
        description = "Optional labels for leaf nodes (original samples). If not provided, uses indices."
    )]
    pub labels: Option<Vec<String>>,

    /// Chart width
    #[schemars(description = "Width of the chart in pixels (default: 800).")]
    pub width: Option<u32>,

    /// Chart height
    #[schemars(description = "Height of the chart in pixels (default: 600).")]
    pub height: Option<u32>,

    /// Chart title
    #[schemars(description = "Title for the dendrogram (default: 'Dendrogram').")]
    pub title: Option<String>,
}
