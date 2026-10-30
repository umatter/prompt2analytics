//! Interactive visualization module using Plotlars (Plotly-based).
//!
//! Provides functions for creating interactive HTML charts from Polars DataFrames.

use super::VisualizationError;
use super::colors::PLOTLY_PALETTE;
use plotlars::{Histogram, LinePlot, Plot, Rgb, ScatterPlot};
use polars::prelude::*;
use std::borrow::Cow;

/// Configuration for interactive charts.
#[derive(Debug, Clone, Default)]
pub struct InteractiveConfig {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub legend_title: Option<String>,
    pub opacity: Option<f64>,
    pub colors: Option<Vec<(u8, u8, u8)>>,
}

/// Result of interactive plot generation.
#[derive(Debug, Clone)]
pub struct InteractivePlotResult {
    /// Full HTML page with embedded Plotly
    pub html: String,
    /// Number of data points plotted
    pub n_points: usize,
    /// Chart type
    pub chart_type: String,
}

impl std::fmt::Display for InteractivePlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Interactive {} Plot", self.chart_type)?;
        writeln!(f, "====================")?;
        writeln!(f, "Points: {}", self.n_points)?;
        writeln!(f)?;
        writeln!(f, "HTML output: {} bytes", self.html.len())
    }
}

// Helper function to validate column exists
fn validate_column(df: &DataFrame, col: &str) -> Result<(), VisualizationError> {
    if df.column(col).is_err() {
        return Err(VisualizationError::InvalidData(format!(
            "Column '{}' not found in DataFrame",
            col
        )));
    }
    Ok(())
}

/// The frame plotlars sees for a chart grouped by `gc`: the group column as
/// text, a missing label shown as the group "NA" (ggplot2's convention).
/// plotlars unwraps every group label, so a null label would panic. A String
/// column without nulls is used as it is.
fn with_group_labels<'a>(
    df: &'a DataFrame,
    gc: &str,
) -> Result<Cow<'a, DataFrame>, VisualizationError> {
    let invalid = |e: PolarsError| VisualizationError::InvalidData(e.to_string());
    let col = df.column(gc).map_err(invalid)?;
    if col.dtype() == &DataType::String && col.null_count() == 0 {
        return Ok(Cow::Borrowed(df));
    }
    let text = col.cast(&DataType::String).map_err(invalid)?;
    let labels: StringChunked = text
        .str()
        .map_err(invalid)?
        .into_iter()
        .map(|v| Some(v.unwrap_or("NA")))
        .collect();
    let mut out = df.clone();
    out.with_column(labels.with_name(col.name().clone()).into_column())
        .map_err(invalid)?;
    Ok(Cow::Owned(out))
}

// Helper to convert color tuples to Rgb
fn to_rgb_colors(colors: &[(u8, u8, u8)]) -> Vec<Rgb> {
    colors.iter().map(|(r, g, b)| Rgb(*r, *g, *b)).collect()
}

// Get colors to use (custom or brand palette)
fn get_colors(custom_colors: &Option<Vec<(u8, u8, u8)>>) -> Vec<Rgb> {
    match custom_colors {
        Some(colors) => to_rgb_colors(colors),
        None => to_rgb_colors(PLOTLY_PALETTE.as_ref()),
    }
}

/// Create an interactive scatter plot from a DataFrame.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `x_col` - Column name for x-axis values
/// * `y_col` - Column name for y-axis values
/// * `group_col` - Optional column name for grouping (creates separate traces;
///   a missing label is the group "NA")
/// * `config` - Chart configuration
pub fn scatter_interactive(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    group_col: Option<&str>,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    validate_column(df, x_col)?;
    validate_column(df, y_col)?;
    if let Some(gc) = group_col {
        validate_column(df, gc)?;
    }
    let grouped = match group_col {
        Some(gc) => with_group_labels(df, gc)?,
        None => Cow::Borrowed(df),
    };
    let df: &DataFrame = &grouped;

    let n_points = df.height();
    let colors = get_colors(&config.colors);

    // Build the plot using chained method calls
    let plot = match (group_col, &config.title, &config.x_label, &config.y_label) {
        (Some(gc), Some(title), Some(xl), Some(yl)) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .group(gc)
            .plot_title(title)
            .x_title(xl)
            .y_title(yl)
            .colors(colors)
            .build(),
        (Some(gc), Some(title), _, _) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .group(gc)
            .plot_title(title)
            .colors(colors)
            .build(),
        (Some(gc), None, _, _) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .group(gc)
            .colors(colors)
            .build(),
        (None, Some(title), Some(xl), Some(yl)) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .plot_title(title)
            .x_title(xl)
            .y_title(yl)
            .colors(colors)
            .build(),
        (None, Some(title), _, _) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .plot_title(title)
            .colors(colors)
            .build(),
        (None, None, _, _) => ScatterPlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .colors(colors)
            .build(),
    };

    let html = plot.to_html();

    Ok(InteractivePlotResult {
        html,
        n_points,
        chart_type: "Scatter".to_string(),
    })
}

/// Create an interactive histogram from a DataFrame column.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `col` - Column name for histogram values
/// * `group_col` - Optional column name for grouping (a missing label is the
///   group "NA")
/// * `config` - Chart configuration
pub fn histogram_interactive(
    df: &DataFrame,
    col: &str,
    group_col: Option<&str>,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    validate_column(df, col)?;
    if let Some(gc) = group_col {
        validate_column(df, gc)?;
    }
    let grouped = match group_col {
        Some(gc) => with_group_labels(df, gc)?,
        None => Cow::Borrowed(df),
    };
    let df: &DataFrame = &grouped;

    let n_points = df.height();

    let plot = match (group_col, &config.title, &config.x_label, &config.y_label) {
        (Some(gc), Some(title), Some(xl), Some(yl)) => Histogram::builder()
            .data(df)
            .x(col)
            .group(gc)
            .plot_title(title)
            .x_title(xl)
            .y_title(yl)
            .build(),
        (Some(gc), Some(title), _, _) => Histogram::builder()
            .data(df)
            .x(col)
            .group(gc)
            .plot_title(title)
            .build(),
        (Some(gc), None, _, _) => Histogram::builder().data(df).x(col).group(gc).build(),
        (None, Some(title), Some(xl), Some(yl)) => Histogram::builder()
            .data(df)
            .x(col)
            .plot_title(title)
            .x_title(xl)
            .y_title(yl)
            .build(),
        (None, Some(title), _, _) => Histogram::builder()
            .data(df)
            .x(col)
            .plot_title(title)
            .build(),
        (None, None, _, _) => Histogram::builder().data(df).x(col).build(),
    };

    let html = plot.to_html();

    Ok(InteractivePlotResult {
        html,
        n_points,
        chart_type: "Histogram".to_string(),
    })
}

/// Create an interactive line chart from a DataFrame.
///
/// # Arguments
///
/// * `df` - Polars DataFrame containing the data
/// * `x_col` - Column name for x-axis values
/// * `y_col` - Column name for y-axis values
/// * `config` - Chart configuration
pub fn line_interactive(
    df: &DataFrame,
    x_col: &str,
    y_col: &str,
    config: InteractiveConfig,
) -> Result<InteractivePlotResult, VisualizationError> {
    validate_column(df, x_col)?;
    validate_column(df, y_col)?;

    let n_points = df.height();

    let plot = match (&config.title, &config.x_label, &config.y_label) {
        (Some(title), Some(xl), Some(yl)) => LinePlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .plot_title(title)
            .x_title(xl)
            .y_title(yl)
            .build(),
        (Some(title), _, _) => LinePlot::builder()
            .data(df)
            .x(x_col)
            .y(y_col)
            .plot_title(title)
            .build(),
        (None, _, _) => LinePlot::builder().data(df).x(x_col).y(y_col).build(),
    };

    let html = plot.to_html();

    Ok(InteractivePlotResult {
        html,
        n_points,
        chart_type: "Line".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn make_test_df() -> DataFrame {
        df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [2.1, 3.9, 6.2, 7.8, 10.1],
            "group" => ["A", "A", "B", "B", "B"],
        }
        .unwrap()
    }

    #[test]
    fn test_scatter_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig {
            title: Some("Test Scatter".to_string()),
            ..Default::default()
        };

        let result = scatter_interactive(&df, "x", "y", None, config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Scatter");
        assert!(result.html.contains("plotly"));
    }

    #[test]
    fn test_scatter_with_group() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = scatter_interactive(&df, "x", "y", Some("group"), config).unwrap();
        assert_eq!(result.n_points, 5);
        assert!(result.html.contains("plotly"));
    }

    #[test]
    fn test_histogram_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig {
            title: Some("Test Histogram".to_string()),
            ..Default::default()
        };

        let result = histogram_interactive(&df, "x", None, config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Histogram");
    }

    #[test]
    fn test_line_interactive() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = line_interactive(&df, "x", "y", config).unwrap();
        assert_eq!(result.n_points, 5);
        assert_eq!(result.chart_type, "Line");
    }

    #[test]
    fn test_invalid_column() {
        let df = make_test_df();
        let config = InteractiveConfig::default();

        let result = scatter_interactive(&df, "nonexistent", "y", None, config);
        assert!(result.is_err());
    }

    /// The HTML without whitespace, so trace names compare as `"name":"<label>"`.
    fn squeezed(html: &str) -> String {
        html.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn test_interactive_charts_label_a_missing_group_na() {
        // DATA-27: plotlars 0.11.4 (common/polar.rs:24) unwraps every unique group label, so a
        // null group panicked. ggplot2 shows a missing discrete value as "NA"; p2a does the same.
        let df = df!(
            "x" => [1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 3.9, 6.2, 7.8],
            "g" => [Some("a"), None, Some("b"), Some("a")],
            "k" => [Some(1i64), Some(2), None, Some(1)]
        )
        .unwrap();
        for gc in ["g", "k"] {
            let s =
                scatter_interactive(&df, "x", "y", Some(gc), InteractiveConfig::default()).unwrap();
            assert!(squeezed(&s.html).contains("\"name\":\"NA\""), "{gc}");
            assert_eq!(s.n_points, 4);
            let h =
                histogram_interactive(&df, "x", Some(gc), InteractiveConfig::default()).unwrap();
            assert!(squeezed(&h.html).contains("\"name\":\"NA\""), "{gc}");
        }
        let s =
            scatter_interactive(&df, "x", "y", Some("g"), InteractiveConfig::default()).unwrap();
        assert!(squeezed(&s.html).contains("\"name\":\"a\""));
        let k =
            scatter_interactive(&df, "x", "y", Some("k"), InteractiveConfig::default()).unwrap();
        assert!(squeezed(&k.html).contains("\"name\":\"1\""));
    }
}
