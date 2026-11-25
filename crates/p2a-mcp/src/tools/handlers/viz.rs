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
use crate::tools::common::{append_missing_note, append_notes, chart_content, limit_refusal};
use crate::tools::handlers::data::format_significant;
use crate::tools::limits;
use crate::tools::requests::viz::*;

use p2a_core::{
    ChartConfig,
    data::columns::{ColumnRequest, Columns, MissingSummary, NaPolicy},
    data::dates::as_date_column,
    polars::prelude::{DataFrame, DataType},
    visualization::{
        Band, box_plot, coefficient_plot, correlation_heatmap, dendrogram, event_study_plot,
        histogram, irf_plot, line_chart_with_band, residual_diagnostics, scatter_plot,
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

/// How a chart reads a plotted column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A number (integers, floats, Decimal; Boolean as 0/1).
    Numeric,
    /// An axis: dates (a `Date` or `Datetime` column, or text such as "1949-01") as days since
    /// 1970-01-01, anything else as a number.
    Axis,
}

/// The plotted columns of one chart, read together: a row with a missing value (null, or NaN
/// in a float column) in any of them is dropped and counted; ±Inf are kept for the core, which
/// leaves them out of the drawing.
struct Plotted {
    cols: Columns,
    days: Vec<String>,
    /// Whether the axis is in days.
    dates: bool,
}

impl Plotted {
    /// The kept values of `name` as numbers (a `Date` axis as days).
    fn values(&self, name: &str) -> Vec<f64> {
        if self.days.iter().any(|d| d == name) {
            self.cols
                .time_index(name)
                .iter()
                .map(|&d| d as f64)
                .collect()
        } else {
            self.cols.numeric(name).to_vec()
        }
    }

    /// The dataset rows kept.
    fn rows(&self) -> &[usize] {
        self.cols.rows()
    }

    fn missing(&self) -> &MissingSummary {
        self.cols.missing()
    }
}

/// Read `columns` of `df` jointly through `data::columns` (missing rows dropped, ±Inf kept),
/// with at least `min_rows` complete rows. Errors name the column (not found, not numeric).
fn plotted_columns(
    df: &DataFrame,
    columns: &[(&str, Role)],
    min_rows: usize,
    context: &str,
) -> Result<Plotted, String> {
    let mut df = df.clone();
    let mut dates = false;
    for &(name, _) in columns.iter().filter(|(_, role)| *role == Role::Axis) {
        let Some(i) = df.get_column_index(name) else {
            continue;
        };
        let column = &df.get_columns()[i];
        if let Some(days) = as_date_column(column) {
            dates = true;
            df.replace_column(i, days.take_materialized_series())
                .map_err(|e| e.to_string())?;
        } else if column.dtype() == &DataType::String {
            return Err(text_axis_error(&df, name));
        }
    }
    let df = &df;
    let mut request = ColumnRequest::new(df)
        .na(NaPolicy::Omit)
        .allow_infinite(true)
        .min_rows(min_rows, context);
    let mut days = Vec::new();
    for &(name, role) in columns {
        let is_date = role == Role::Axis
            && df
                .column(name)
                .is_ok_and(|c| matches!(c.dtype(), DataType::Date));
        request = if is_date {
            days.push(name.to_string());
            request.time_index(name)
        } else {
            request.numeric(name)
        };
    }
    let cols = request.extract().map_err(|e| e.to_string())?;
    Ok(Plotted { cols, days, dates })
}

/// The refusal of a text x axis that does not hold dates, with the columns that would do.
fn text_axis_error(df: &DataFrame, name: &str) -> String {
    let usable: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|c| c.dtype().is_integer() || c.dtype().is_float() || as_date_column(c).is_some())
        .map(|c| c.name().to_string())
        .collect();
    format!(
        "Column '{name}' contains non-numeric values that are not all dates, so it cannot be the \
         x axis. Text is read as dates when every value looks like 1949-01, 1949-01-31 or 1949Q1. \
         Do not repeat this call: use another x column (numeric or date columns: {}), or \
         viz_boxplot to compare groups.",
        crate::tools::common::echo_list(&usable, ", ", 20)
    )
}

/// A numeric column over every row, a missing value as NaN (for an optional band whose gaps
/// are drawn as gaps); ±Inf kept.
fn numeric_with_gaps(df: &DataFrame, name: &str) -> Result<Vec<f64>, String> {
    with_gaps(df, name, Role::Numeric)
}

/// [`numeric_with_gaps`] for a column in any role.
fn with_gaps(df: &DataFrame, name: &str, role: Role) -> Result<Vec<f64>, String> {
    let plotted = plotted_columns(df, &[(name, role)], 0, name)?;
    let mut out = vec![f64::NAN; df.height()];
    for (&row, v) in plotted.rows().iter().zip(plotted.values(name)) {
        out[row] = v;
    }
    Ok(out)
}

/// "Pairwise complete observations: 35 to 38 of 40 rows" when some pair of the correlation
/// matrix dropped rows (the sentence `CorrelationMatrix`'s table ends with).
fn pairwise_note(corr: &p2a_core::stats::CorrelationMatrix) -> Option<String> {
    let min = corr.pair_n.iter().flatten().min().copied()?;
    let max = corr.pair_n.iter().flatten().max().copied()?;
    if min >= corr.n_rows {
        return None;
    }
    let range = if min == max {
        min.to_string()
    } else {
        format!("{min} to {max}")
    };
    Some(format!(
        "Pairwise complete observations: {range} of {} rows",
        corr.n_rows
    ))
}

/// The viz_heatmap reply (text, PNG): the selection is checked first (every name exists and is
/// numeric, and no more variables than the image holds), then only the selected columns are
/// correlated (all numeric columns without a selection).
fn heatmap_reply(
    dataset: &p2a_core::Dataset,
    selection: Option<&[String]>,
    title: Option<&str>,
) -> Result<(String, String, Option<String>), String> {
    use p2a_core::visualization::heatmap::heatmap_capacity;
    let df = dataset.df();
    // The dtypes stats::correlation_matrix correlates.
    let correlated = |dtype: &DataType| dtype.is_integer() || dtype.is_float();
    let numeric_names: Vec<String> = df
        .get_columns()
        .iter()
        .filter(|c| correlated(c.dtype()))
        .map(|c| c.name().to_string())
        .collect();
    let capacity = heatmap_capacity(None, None);
    let refuse_size = |k: usize| {
        format!(
            "{k} variables do not fit the heatmap image: at most {capacity} do. Select fewer \
             with `columns`"
        )
    };
    let corr = match selection {
        Some(names) => {
            let mut problems = Vec::new();
            for name in names {
                match df.column(name) {
                    Err(_) => problems.push(format!("column '{name}' not found")),
                    Ok(c) if !correlated(c.dtype()) => {
                        problems.push(format!("'{name}' is not numeric ({})", c.dtype()))
                    }
                    Ok(_) => {}
                }
            }
            if !problems.is_empty() {
                return Err(format!(
                    "Cannot draw the heatmap: {}. Numeric columns: {}",
                    problems.join("; "),
                    crate::tools::common::echo_list(&numeric_names, ", ", 20)
                ));
            }
            let mut unique: Vec<&str> = Vec::new();
            for name in names {
                if !unique.contains(&name.as_str()) {
                    unique.push(name);
                }
            }
            if unique.len() < 2 {
                return Err(
                    "Need at least 2 distinct numeric columns for a correlation heatmap"
                        .to_string(),
                );
            }
            if unique.len() > capacity {
                return Err(refuse_size(unique.len()));
            }
            let selected = dataset
                .select_columns(&unique)
                .map_err(|e| format!("Failed to select the columns: {e}"))?;
            let corr = p2a_core::stats::correlation_matrix(&selected)
                .map_err(|e| format!("Failed to compute correlation: {e}"))?;
            if let Some(name) = unique
                .iter()
                .find(|n| !corr.columns.iter().any(|c| c == *n))
            {
                return Err(format!("'{name}' is not numeric"));
            }
            corr
        }
        None => {
            if numeric_names.len() > capacity {
                return Err(refuse_size(numeric_names.len()));
            }
            let corr = p2a_core::stats::correlation_matrix(dataset)
                .map_err(|e| format!("Failed to compute correlation: {e}"))?;
            if corr.columns.len() < 2 {
                return Err("Need at least 2 numeric columns for correlation heatmap".to_string());
            }
            corr
        }
    };
    let result = correlation_heatmap(
        &corr.matrix,
        &corr.columns,
        &corr.columns,
        title,
        None,
        None,
    )
    .map_err(|e| format!("Failed to generate heatmap: {e}"))?;
    let mut description = format!(
        "Correlation heatmap generated successfully.\nVariables: {}",
        corr.columns.join(", ")
    );
    if let Some(note) = pairwise_note(&corr) {
        description.push('\n');
        description.push_str(&note);
    }
    append_notes(&mut description, None, &result.warnings);
    Ok((description, result.image_base64, result.svg))
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
        description = "Histogram of a numeric column (missing and non-finite values left out): equal-width bins from the minimum to the maximum, right-closed with the lowest value included, values within 1e-7 bin widths of a break assigned as R's hist() does; the bin count defaults to R's nclass.Sturges, ceiling(log2(n) + 1) (R's own hist() default also rounds the breaks with pretty(), which p2a does not). Rows with a missing value are left out and counted. Returns a PNG image, the number of bins and the count per bin."
    )]
    pub async fn viz_histogram(
        &self,
        Parameters(request): Parameters<HistogramRequest>,
    ) -> Result<CallToolResult, McpError> {
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

        let plotted = match plotted_columns(
            dataset.df(),
            &[(&request.column, Role::Numeric)],
            0,
            "viz_histogram",
        ) {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let values: Vec<f64> = plotted
            .values(&request.column)
            .into_iter()
            .filter(|v| v.is_finite())
            .collect();

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
                let mut description = format!(
                    "Histogram of '{}' generated successfully.\nBins: {}\nCounts per bin: {}\nData points: {}",
                    request.column,
                    result.bins,
                    counts.join(", "),
                    values.len()
                );
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Generate a scatter plot visualization showing the relationship between two numeric columns (a date x column, or text such as 1949-01, plots as dates). Rows with a missing value in a plotted column are left out and counted. Returns a base64-encoded PNG image."
    )]
    pub async fn viz_scatter(
        &self,
        Parameters(request): Parameters<ScatterPlotRequest>,
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

        let plotted = match plotted_columns(
            dataset.df(),
            &[
                (&request.x_column, Role::Axis),
                (&request.y_column, Role::Numeric),
            ],
            1,
            "viz_scatter",
        ) {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let x_values = plotted.values(&request.x_column);
        let y_values = plotted.values(&request.y_column);

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());
        config.y_label = Some(request.y_column.clone());
        config.x_dates = plotted.dates;

        match scatter_plot(&x_values, &y_values, config) {
            Ok(result) => {
                let mut description = format!(
                    "Scatter plot of {} vs {} generated successfully.\nPoints: {}\nCorrelation: {:.4}",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.correlation.unwrap_or(f64::NAN)
                );
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Generate a line chart visualization for time series or sequential data. Supports multiple Y series. The x column is numeric or holds dates (date columns, or text such as 1949-01, 1949-01-31 or 1949Q1). lower_column and upper_column shade a band, such as the intervals of a saved forecast. Rows with a missing value in a plotted column are left out and counted, each series on its own. Returns a base64-encoded PNG image."
    )]
    pub async fn viz_line(
        &self,
        Parameters(request): Parameters<LineChartRequest>,
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

        // Each series drops its own rows with a missing x or y value.
        let mut series: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut series_names = Vec::new();
        let mut missing = Vec::new();
        let mut x_dates = false;
        for y_col_name in &request.y_columns {
            let plotted = match plotted_columns(
                dataset.df(),
                &[(&request.x_column, Role::Axis), (y_col_name, Role::Numeric)],
                1,
                &format!("viz_line series '{y_col_name}'"),
            ) {
                Ok(p) => p,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };
            x_dates = plotted.dates;
            series_names.push(y_col_name.clone());
            series.push((
                y_col_name.clone(),
                plotted.values(&request.x_column),
                plotted.values(y_col_name),
            ));
            missing.push(plotted.missing().clone());
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());
        config.x_dates = x_dates;
        if let [y] = request.y_columns.as_slice() {
            config.y_label = Some(y.clone());
        }

        let band = match (&request.lower_column, &request.upper_column) {
            (None, None) => None,
            (Some(lower), Some(upper)) => {
                let df = dataset.df();
                let read = || -> Result<Band, String> {
                    Ok(Band {
                        label: format!("{lower} to {upper}"),
                        x: with_gaps(df, &request.x_column, Role::Axis)?,
                        lower: numeric_with_gaps(df, lower)?,
                        upper: numeric_with_gaps(df, upper)?,
                    })
                };
                match read() {
                    Ok(band) => Some(band),
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
                }
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "lower_column and upper_column go together: give both or neither",
                )]));
            }
        };

        match line_chart_with_band(&series, band.as_ref(), config) {
            Ok(result) => {
                let mut description = format!(
                    "Line chart generated successfully.\nX: {}\nSeries: {}\nPoints: {}",
                    request.x_column,
                    series_names.join(", "),
                    result.n_points
                );
                for m in &missing {
                    append_missing_note(&mut description, m);
                }
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Box plots of one or more numeric columns with Tukey hinges and whiskers to the most extreme values within 1.5 IQR, as R's boxplot.stats; values beyond them are listed as outliers (at most 20 per column, then the count of the rest). Missing values are left out and counted per column; ±Inf count as values and are listed as outliers but not drawn. Returns a PNG image and the statistics with n per column."
    )]
    pub async fn viz_boxplot(
        &self,
        Parameters(request): Parameters<BoxPlotRequest>,
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

        let df = dataset.df();

        // Each column on its own: its missing values are dropped and counted; ±Inf are values,
        // as in R's boxplot.stats. A column with no type at all (every cell empty) has n = 0.
        let mut groups = Vec::new();
        let mut missing = Vec::new();
        for col_name in &request.columns {
            if df
                .column(col_name)
                .is_ok_and(|c| matches!(c.dtype(), DataType::Null))
            {
                let n = df.height();
                groups.push((col_name.clone(), Vec::new()));
                missing.push(MissingSummary {
                    n_total: n,
                    n_used: 0,
                    n_dropped: n,
                    by_column: if n > 0 {
                        vec![(col_name.clone(), n)]
                    } else {
                        Vec::new()
                    },
                });
                continue;
            }
            let plotted = match plotted_columns(df, &[(col_name, Role::Numeric)], 0, "viz_boxplot")
            {
                Ok(p) => p,
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            };
            groups.push((col_name.clone(), plotted.values(col_name)));
            missing.push(plotted.missing().clone());
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
                for m in &missing {
                    append_missing_note(&mut description, m);
                }
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Correlation heatmap of numeric columns: Pearson correlations, each pair over the rows where both are present (R's cor(use = \"pairwise.complete.obs\")), on a diverging colour scale. At most 127 variables fit the image (labels are hidden above about 40); unknown or non-numeric names in columns are refused. Returns a base64-encoded PNG image."
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

        match heatmap_reply(
            &dataset,
            request.columns.as_deref(),
            request.title.as_deref(),
        ) {
            Ok((description, image, svg)) => Ok(CallToolResult::success(vec![
                Content::text(description),
                chart_content(image, svg),
            ])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    // ========================================================================
    // Interactive Chart Tools (HTML/Plotly output)
    // ========================================================================

    /// Generate an interactive scatter plot with Plotly.js.
    #[tool(
        description = "Generate an interactive scatter plot visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Supports grouping data by a categorical column for colored traces. Interactive features include zoom, pan, and hover. The HTML loads plotly.js 3.0.1 and MathJax from cdn.plot.ly and cdn.jsdelivr.net when opened, so viewing needs network access; values are kept in double precision; rows with a missing value in a plotted column are left out and counted."
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
                let mut description = format!(
                    "Interactive scatter plot generated successfully.\nX: {}\nY: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.html.len()
                );
                append_notes(&mut description, None, &result.warnings);
                append_missing_note(&mut description, &result.missing);
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
        description = "Generate an interactive histogram visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Supports grouping data by a categorical column for overlaid histograms. Interactive features include zoom, pan, and hover. The HTML loads plotly.js 3.0.1 and MathJax from cdn.plot.ly and cdn.jsdelivr.net when opened, so viewing needs network access; values are kept in double precision; rows with a missing value in a plotted column are left out and counted."
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
                let mut description = format!(
                    "Interactive histogram generated successfully.\nColumn: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.column,
                    result.n_points,
                    result.html.len()
                );
                append_notes(&mut description, None, &result.warnings);
                append_missing_note(&mut description, &result.missing);
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
        description = "Generate an interactive line chart visualization using Plotly.js. Returns HTML that can be saved to a file and opened in a browser. Interactive features include zoom, pan, and hover. The HTML loads plotly.js 3.0.1 and MathJax from cdn.plot.ly and cdn.jsdelivr.net when opened, so viewing needs network access; values are kept in double precision; rows with a missing value in a plotted column are left out and counted."
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
                let mut description = format!(
                    "Interactive line chart generated successfully.\nX: {}\nY: {}\nPoints: {}\n\nHTML output ({} bytes) - save to a .html file and open in browser.",
                    request.x_column,
                    request.y_column,
                    result.n_points,
                    result.html.len()
                );
                append_notes(&mut description, None, &result.warnings);
                append_missing_note(&mut description, &result.missing);
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
        description = "Generate an event study plot showing treatment effects over time with confidence intervals. Used for visualizing DiD or panel event study results. Shows point estimates with CI bands and reference lines at t=0 and y=0. Rows with a missing value in a plotted column are left out and counted."
    )]
    pub async fn viz_event_study(
        &self,
        Parameters(request): Parameters<EventStudyRequest>,
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

        let plotted = match plotted_columns(
            dataset.df(),
            &[
                (&request.time_column, Role::Numeric),
                (&request.estimate_column, Role::Numeric),
                (&request.ci_lower_column, Role::Numeric),
                (&request.ci_upper_column, Role::Numeric),
            ],
            1,
            "viz_event_study",
        ) {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let time = plotted.values(&request.time_column);
        let estimates = plotted.values(&request.estimate_column);
        let ci_lower = plotted.values(&request.ci_lower_column);
        let ci_upper = plotted.values(&request.ci_upper_column);

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some("Time Relative to Treatment".to_string());
        config.y_label = Some("Effect".to_string());

        match event_study_plot(&time, &estimates, &ci_lower, &ci_upper, config) {
            Ok(result) => {
                let mut description = format!(
                    "Event study plot generated successfully.\nPeriods: {}",
                    result.n_periods
                );
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Generate a coefficient plot showing regression coefficients with confidence intervals (error bars). After regression_ols, pass the coefficient table it saved (named in its reply) as the dataset; the column arguments default to that table's columns. Shows vertical zero line for reference. Rows with a missing value in a plotted column are left out and counted."
    )]
    pub async fn viz_coefficient(
        &self,
        Parameters(request): Parameters<CoefficientPlotRequest>,
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

        let df = dataset.df();
        let name_column = request.name_column.as_deref().unwrap_or("term");
        let estimate_column = request.estimate_column.as_deref().unwrap_or("estimate");
        let ci_lower_column = request.ci_lower_column.as_deref().unwrap_or("ci_lower");
        let ci_upper_column = request.ci_upper_column.as_deref().unwrap_or("ci_upper");

        // Extract name column
        let name_col = match df.column(name_column) {
            Ok(c) => c,
            Err(_) => {
                let available: Vec<String> = df
                    .get_column_names()
                    .iter()
                    .map(|c| c.to_string())
                    .collect();
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Name column '{name_column}' not found in dataset '{}' (columns: {}). \
                     viz_coefficient plots a table with one row per coefficient: after \
                     regression_ols, pass the coefficient table it saved as the dataset.",
                    request.dataset,
                    available.join(", ")
                ))]));
            }
        };
        let names: Vec<String> = match name_col.str() {
            Ok(s) => s.into_iter().map(|v| v.unwrap_or("").to_string()).collect(),
            Err(_) => (0..name_col.len()).map(|i| format!("Var_{}", i)).collect(),
        };

        let plotted = match plotted_columns(
            df,
            &[
                (estimate_column, Role::Numeric),
                (ci_lower_column, Role::Numeric),
                (ci_upper_column, Role::Numeric),
            ],
            1,
            "viz_coefficient",
        ) {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let names: Vec<String> = plotted.rows().iter().map(|&i| names[i].clone()).collect();
        let estimates = plotted.values(estimate_column);
        let ci_lower = plotted.values(ci_lower_column);
        let ci_upper = plotted.values(ci_upper_column);

        let mut config = ChartConfig::default();
        config.title = request.title;

        let horizontal = request.horizontal.unwrap_or(true);

        match coefficient_plot(&names, &estimates, &ci_lower, &ci_upper, config, horizontal) {
            Ok(result) => {
                let mut description = format!(
                    "Coefficient plot generated successfully.\nCoefficients: {}\nOrientation: {}",
                    result.n_coefficients,
                    if horizontal { "horizontal" } else { "vertical" }
                );
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Generate an Impulse Response Function (IRF) plot from VAR models. Shows how a variable responds to a shock over time. Optionally includes confidence bands. Rows with a missing horizon or response are left out and counted; a missing band bound is a gap in the band."
    )]
    pub async fn viz_irf(
        &self,
        Parameters(request): Parameters<IrfPlotRequest>,
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

        let df = dataset.df();

        let plotted = match plotted_columns(
            df,
            &[
                (&request.horizon_column, Role::Numeric),
                (&request.response_column, Role::Numeric),
            ],
            1,
            "viz_irf",
        ) {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let horizon = plotted.values(&request.horizon_column);
        let response = plotted.values(&request.response_column);

        // Optional CI bands on the kept rows; a missing bound is a gap in the band.
        let band = |name: &Option<String>| -> Result<Option<Vec<f64>>, String> {
            let Some(name) = name else { return Ok(None) };
            let all = numeric_with_gaps(df, name)?;
            Ok(Some(plotted.rows().iter().map(|&i| all[i]).collect()))
        };
        let ci_lower = match band(&request.ci_lower_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_upper = match band(&request.ci_upper_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
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
                let mut description = format!(
                    "IRF plot generated successfully.\nHorizons: {}\nHas CI bands: {}\nShock: {}\nResponse: {}",
                    result.n_horizons,
                    has_ci,
                    result.shock.as_deref().unwrap_or("unnamed"),
                    result.response.as_deref().unwrap_or("unnamed")
                );
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.image_base64, result.svg),
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
        description = "Four diagnostic plots of a fitted regression (Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage), with standardized residuals and Cook's distance as R's rstandard() and cooks.distance(). Exact when the model's hat values are given (leverage_column; R: hatvalues(model)); without them the hat values of a regression on one predictor are used and a warning says so. Rows with a missing fitted value, residual or leverage are left out and counted. Reports the residual standard error and the dataset rows with Cook's distance above 0.5. Returns four PNG images."
    )]
    pub async fn viz_residual_diagnostics(
        &self,
        Parameters(request): Parameters<ResidualDiagnosticsRequest>,
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

        let mut columns = vec![
            (request.fitted_column.as_str(), Role::Numeric),
            (request.residuals_column.as_str(), Role::Numeric),
        ];
        if let Some(name) = &request.leverage_column {
            columns.push((name.as_str(), Role::Numeric));
        }
        let plotted = match plotted_columns(dataset.df(), &columns, 1, "viz_residual_diagnostics") {
            Ok(p) => p,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let fitted = plotted.values(&request.fitted_column);
        let residuals = plotted.values(&request.residuals_column);
        let leverage: Option<Vec<f64>> = request
            .leverage_column
            .as_ref()
            .map(|name| plotted.values(name));
        // A hat value lies in [0, 1]; name the dataset row of one that does not.
        if let (Some(name), Some(lev)) = (&request.leverage_column, &leverage)
            && let Some((&row, &h)) = plotted
                .rows()
                .iter()
                .zip(lev)
                .find(|(_, h)| !(0.0..=1.0).contains(*h))
        {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Leverage column '{name}' has {h} at dataset row {row} (0-based): hat values lie \
                 in [0, 1] (R: hatvalues(model))"
            ))]));
        }

        let config = ChartConfig::default();

        match residual_diagnostics(&fitted, &residuals, leverage.as_deref(), config) {
            Ok(result) => {
                // Observations with high Cook's distance, as dataset rows.
                let high_influence: Vec<usize> = result
                    .cooks_distance
                    .iter()
                    .zip(&result.rows)
                    .filter(|(d, _)| **d > 0.5)
                    .map(|(_, &row)| plotted.rows()[row])
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
                append_missing_note(&mut description, plotted.missing());
                Ok(CallToolResult::success(vec![
                    Content::text(description),
                    chart_content(result.residuals_vs_fitted, result.residuals_vs_fitted_svg),
                    chart_content(result.qq_plot, result.qq_plot_svg),
                    chart_content(result.scale_location, result.scale_location_svg),
                    chart_content(
                        result.residuals_vs_leverage,
                        result.residuals_vs_leverage_svg,
                    ),
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
                    chart_content(result.image_base64, result.svg),
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

    /// Six rows: an Int64 `x` with a null (row 1), a Float64 `y` with a NaN (row 3).
    fn missing_frame() -> DataFrame {
        let mut df = df!(
            "x" => [Some(1i64), None, Some(3), Some(4), Some(5), Some(6)],
            "y" => [1.2, 2.1, 2.9, f64::NAN, 5.3, 5.8],
            "t" => [1i64, 2, 3, 4, 5, 6],
            "y1" => [1.1, 1.9, f64::NAN, 4.2, 4.8, 6.1],
            "y2" => [Some(2.0), Some(2.6), Some(3.1), Some(3.5), None, Some(4.4)],
            "lo" => [0.7, 1.6, 2.4, 3.4, 4.8, 5.3],
            "hi" => [1.7, 2.6, 3.4, 4.4, 5.8, 6.3],
            "w" => [1.0, 2.0, 3.0, 4.0, 5.0, f64::INFINITY],
            "names" => ["a", "b", "c", "d", "e", "f"],
            "f" => [1.0, 2.1, 2.9, 4.2, 5.1, 5.8],
            "r" => [Some(0.3), None, Some(-0.25), Some(0.2), Some(0.5), Some(-0.35)],
            "h" => [0.3, 0.2, 0.3, 0.3, 0.8, 0.3],
            "h_bad" => [0.3, 0.2, 0.3, 0.3, 0.8, 1.3]
        )
        .unwrap();
        df.with_column(Column::full_null("z".into(), 6, &DataType::Null))
            .unwrap();
        df
    }

    /// The chart tools read their columns NA-aware: dropped rows are counted in a note, an
    /// Int64 column with a null is read without a cast, and the drawn points exclude the
    /// dropped rows.
    #[test]
    fn viz_reads_report_dropped_rows() {
        let server = server_with(vec![("d", missing_frame())]);
        let both = "Missing values: 2 of 6 rows dropped because of missing values (x: 1, y: 1)";
        let ok = |r: CallToolResult| -> String {
            assert_ne!(r.is_error, Some(true), "{}", text(&r));
            text(&r)
        };
        let t = ok(run(
            Limits::off(),
            server.viz_scatter(Parameters(ScatterPlotRequest {
                dataset: "d".into(),
                x_column: "x".into(),
                y_column: "y".into(),
                title: None,
            })),
        ));
        assert!(t.contains("Points: 4") && t.ends_with(both), "{t}");

        let t = ok(run(
            Limits::off(),
            server.viz_histogram(Parameters(HistogramRequest {
                dataset: "d".into(),
                column: "x".into(),
                bins: None,
                title: None,
            })),
        ));
        assert!(
            t.contains("Data points: 5")
                && t.ends_with(
                    "Missing values: 1 of 6 rows dropped because of missing values (x: 1)"
                ),
            "{t}"
        );

        let t = ok(run(
            Limits::off(),
            server.viz_line(Parameters(LineChartRequest {
                dataset: "d".into(),
                x_column: "t".into(),
                y_columns: vec!["y1".into(), "y2".into()],
                lower_column: None,
                upper_column: None,
                title: None,
            })),
        ));
        assert!(
            t.contains("Missing values: 1 of 6 rows dropped because of missing values (y1: 1)")
                && t.ends_with(
                    "Missing values: 1 of 6 rows dropped because of missing values (y2: 1)"
                ),
            "{t}"
        );

        let t = ok(run(
            Limits::off(),
            server.viz_boxplot(Parameters(BoxPlotRequest {
                dataset: "d".into(),
                columns: vec!["x".into(), "w".into(), "z".into()],
                title: None,
            })),
        ));
        assert!(t.contains("x (n = 5):") && t.contains("w (n = 6):"), "{t}");
        assert!(t.contains("Outliers: inf"), "{t}");
        assert!(t.contains("z (n = 0): no values"), "{t}");
        assert!(
            t.contains("Missing values: 1 of 6 rows dropped because of missing values (x: 1)")
                && t.ends_with(
                    "Missing values: 6 of 6 rows dropped because of missing values (z: 6)"
                ),
            "{t}"
        );

        let t = ok(run(
            Limits::off(),
            server.viz_event_study(Parameters(EventStudyRequest {
                dataset: "d".into(),
                time_column: "x".into(),
                estimate_column: "y".into(),
                ci_lower_column: "lo".into(),
                ci_upper_column: "hi".into(),
                title: None,
            })),
        ));
        assert!(t.contains("Periods: 4") && t.ends_with(both), "{t}");

        let t = ok(run(
            Limits::off(),
            server.viz_coefficient(Parameters(CoefficientPlotRequest {
                dataset: "d".into(),
                name_column: Some("names".into()),
                estimate_column: Some("y".into()),
                ci_lower_column: Some("lo".into()),
                ci_upper_column: Some("hi".into()),
                title: None,
                horizontal: None,
            })),
        ));
        assert!(
            t.contains("Coefficients: 5")
                && t.ends_with(
                    "Missing values: 1 of 6 rows dropped because of missing values (y: 1)"
                ),
            "{t}"
        );

        let t = ok(run(
            Limits::off(),
            server.viz_irf(Parameters(IrfPlotRequest {
                dataset: "d".into(),
                horizon_column: "x".into(),
                response_column: "y".into(),
                ci_lower_column: Some("lo".into()),
                ci_upper_column: Some("hi".into()),
                shock_label: None,
                response_label: None,
                title: None,
            })),
        ));
        assert!(t.contains("Horizons: 4") && t.ends_with(both), "{t}");

        let diag = |leverage: &str| {
            run(
                Limits::off(),
                server.viz_residual_diagnostics(Parameters(ResidualDiagnosticsRequest {
                    dataset: "d".into(),
                    fitted_column: "f".into(),
                    residuals_column: "r".into(),
                    leverage_column: Some(leverage.into()),
                })),
            )
        };
        let t = ok(diag("h"));
        assert!(
            t.contains("Observations: 5")
                && t.contains("(Cook's D > 0.5; dataset rows, 0-based): [4]")
                && t.ends_with(
                    "Missing values: 1 of 6 rows dropped because of missing values (r: 1)"
                ),
            "{t}"
        );
        let r = diag("h_bad");
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("1.3 at dataset row 5"), "{}", text(&r));

        // A text column is refused, naming it.
        let r = run(
            Limits::off(),
            server.viz_scatter(Parameters(ScatterPlotRequest {
                dataset: "d".into(),
                x_column: "x".into(),
                y_column: "names".into(),
                title: None,
            })),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("Column 'names'"), "{}", text(&r));
    }

    /// A `Date` x axis is plotted as days since 1970-01-01, a missing date dropping its row;
    /// so are a `Datetime` and text dates, while other text is refused with the usable columns.
    #[test]
    fn date_axes_are_plotted_and_other_text_refused() {
        let days = [
            Some(19_000i32),
            Some(19_001),
            None,
            Some(19_003),
            Some(19_004),
        ];
        let mut df = df!(
            "y" => [1.3, 1.9, 3.2, 3.8, 5.1],
            "ms" => [1_000i64, 2_000, 3_000, 4_000, 5_000],
            "month" => ["1949-01", "1949-02", "1949-03", "1949-04", "1949-05"],
            "name" => ["a", "b", "c", "1949-04", "1949-05"]
        )
        .unwrap();
        let date = Series::new("day".into(), days.as_slice())
            .cast(&DataType::Date)
            .unwrap();
        let stamp = df
            .column("ms")
            .unwrap()
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap()
            .with_name("stamp".into());
        df.with_column(date).unwrap();
        df.with_column(stamp).unwrap();
        let server = server_with(vec![("d", df)]);
        let note = "Missing values: 1 of 5 rows dropped because of missing values (day: 1)";

        let scatter = |x: &str| {
            run(
                Limits::off(),
                server.viz_scatter(Parameters(ScatterPlotRequest {
                    dataset: "d".into(),
                    x_column: x.into(),
                    y_column: "y".into(),
                    title: None,
                })),
            )
        };
        let r = scatter("day");
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        assert!(t.contains("Points: 4") && t.ends_with(note), "{t}");

        let r = run(
            Limits::off(),
            server.viz_line(Parameters(LineChartRequest {
                dataset: "d".into(),
                x_column: "day".into(),
                y_columns: vec!["y".into()],
                lower_column: None,
                upper_column: None,
                title: None,
            })),
        );
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).ends_with(note), "{}", text(&r));

        let r = scatter("stamp");
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).contains("Points: 5"), "{}", text(&r));

        let r = scatter("month");
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        assert!(text(&r).contains("Points: 5"), "{}", text(&r));

        let r = scatter("name");
        assert_eq!(r.is_error, Some(true), "{}", text(&r));
        let t = text(&r);
        assert!(
            t.contains("Column 'name' contains non-numeric")
                && t.contains("y, ms, month, day, stamp"),
            "{t}"
        );
    }

    fn heatmap(server: &AnalyticsServer, columns: Option<Vec<&str>>) -> CallToolResult {
        run(
            Limits::off(),
            server.viz_heatmap(Parameters(HeatmapRequest {
                dataset: "d".into(),
                columns: columns.map(|c| c.into_iter().map(String::from).collect()),
                title: None,
            })),
        )
    }

    /// 40 rows: numeric a, b (b with 2 nulls, c with 5), a text and a Boolean column.
    fn heatmap_frame() -> DataFrame {
        let a: Vec<f64> = (0..40)
            .map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.1)
            .collect();
        let b: Vec<Option<f64>> = (0..40)
            .map(|i| (i % 17 != 3).then(|| (i as f64 * 1.3).cos() * 2.0 + i as f64 * 0.05))
            .collect();
        let c: Vec<Option<i64>> = (0..40)
            .map(|i| (i % 8 != 5).then_some((i * 7 % 11) as i64))
            .collect();
        let name: Vec<String> = (0..40).map(|i| format!("n{i}")).collect();
        let flag: Vec<bool> = (0..40).map(|i| i % 3 == 0).collect();
        df!("a" => a, "b" => b, "c" => c, "name" => name, "flag" => flag).unwrap()
    }

    /// TOOL-38: unknown and non-numeric names were dropped without a word.
    #[test]
    fn heatmap_refuses_unknown_and_non_numeric_columns() {
        let server = server_with(vec![("d", heatmap_frame())]);
        let r = heatmap(&server, Some(vec!["a", "b", "zzz", "name"]));
        assert_eq!(r.is_error, Some(true));
        let t = text(&r);
        assert!(
            t.contains("column 'zzz' not found")
                && t.contains("'name' is not numeric")
                && t.contains("Numeric columns: a, b, c"),
            "{t}"
        );
        let r = heatmap(&server, Some(vec!["a", "flag"]));
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("'flag' is not numeric"), "{}", text(&r));
    }

    #[test]
    fn heatmap_correlates_only_the_selection() {
        let server = server_with(vec![("d", heatmap_frame())]);
        let t = text(&heatmap(&server, Some(vec!["a", "b"])));
        assert!(t.contains("Variables: a, b\n"), "{t}");
        assert!(
            t.ends_with("Pairwise complete observations: 37 to 40 of 40 rows"),
            "{t}"
        );
        let all = text(&heatmap(&server, None));
        assert!(all.contains("Variables: a, b, c\n"), "{all}");
        // The handler's sentence equals the last line of CorrelationMatrix's table.
        let corr = p2a_core::stats::correlation_matrix(&Dataset::new(heatmap_frame())).unwrap();
        assert_eq!(
            pairwise_note(&corr).as_deref(),
            corr.to_string_table().lines().last()
        );
        assert!(
            all.ends_with(corr.to_string_table().lines().last().unwrap()),
            "{all}"
        );
    }

    #[test]
    fn heatmap_refuses_more_variables_than_fit() {
        let columns: Vec<Column> = (0..130)
            .map(|j| {
                let v: Vec<f64> = (0..20)
                    .map(|i| ((i * (j + 3)) % 13) as f64 + 0.1 * j as f64)
                    .collect();
                Column::new(format!("v{j:03}").into(), v)
            })
            .collect();
        let server = server_with(vec![("d", DataFrame::new(columns).unwrap())]);
        let r = heatmap(&server, None);
        assert_eq!(r.is_error, Some(true));
        assert!(
            text(&r).contains("130 variables do not fit") && text(&r).contains("at most 127"),
            "{}",
            text(&r)
        );
        let t = text(&heatmap(&server, Some(vec!["v000", "v001", "v002"])));
        assert!(t.contains("Variables: v000, v001, v002"), "{t}");
    }

    /// The interactive charts count the rows they leave out.
    #[test]
    fn interactive_charts_report_dropped_rows() {
        let server = server_with(vec![("d", missing_frame())]);
        let note = "Missing values: 2 of 6 rows dropped because of missing values (x: 1, y: 1)";
        let r = run(
            Limits::off(),
            server.viz_scatter_interactive(Parameters(ScatterInteractiveRequest {
                dataset: "d".into(),
                x_column: "x".into(),
                y_column: "y".into(),
                group_column: None,
                title: None,
            })),
        );
        assert!(
            text(&r).contains("Points: 4") && text(&r).ends_with(note),
            "{}",
            text(&r)
        );
        let r = run(
            Limits::off(),
            server.viz_line_interactive(Parameters(LineInteractiveRequest {
                dataset: "d".into(),
                x_column: "x".into(),
                y_column: "y".into(),
                title: None,
            })),
        );
        assert!(text(&r).ends_with(note), "{}", text(&r));
        let r = run(
            Limits::off(),
            server.viz_histogram_interactive(Parameters(HistogramInteractiveRequest {
                dataset: "d".into(),
                column: "y".into(),
                group_column: None,
                title: None,
            })),
        );
        assert!(
            text(&r)
                .ends_with("Missing values: 1 of 6 rows dropped because of missing values (y: 1)"),
            "{}",
            text(&r)
        );
    }
}
