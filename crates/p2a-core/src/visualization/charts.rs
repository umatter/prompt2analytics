//! Chart generation: histograms, scatter plots, box plots, line charts.

use super::{VisualizationError, DEFAULT_WIDTH, DEFAULT_HEIGHT};
use super::colors::{
    CHART_PALETTE, DEFAULT_SERIES_COLOR, OUTLIER_COLOR, TREND_LINE_COLOR,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use plotters::prelude::*;

/// Configuration for chart generation.
#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub width: u32,
    pub height: u32,
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    /// Font size for chart title (default: 32)
    pub title_font_size: u32,
    /// Font size for axis labels (default: 20)
    pub label_font_size: u32,
    /// Font size for tick labels (default: 16)
    pub tick_font_size: u32,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            title: None,
            x_label: None,
            y_label: None,
            title_font_size: 32,
            label_font_size: 20,
            tick_font_size: 16,
        }
    }
}

/// Result of histogram generation.
pub struct HistogramResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of bins used
    pub bins: usize,
    /// Minimum value in data
    pub min: f64,
    /// Maximum value in data
    pub max: f64,
    /// Mean value
    pub mean: f64,
}

impl std::fmt::Display for HistogramResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Histogram")?;
        writeln!(f, "=========")?;
        writeln!(f, "Bins: {}", self.bins)?;
        writeln!(f, "Range: [{:.4}, {:.4}]", self.min, self.max)?;
        writeln!(f, "Mean: {:.4}", self.mean)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a histogram from numeric data.
///
/// # Arguments
/// * `data` - Numeric values to plot
/// * `bins` - Number of bins (default: auto-calculated)
/// * `config` - Chart configuration
pub fn histogram(
    data: &[f64],
    bins: Option<usize>,
    config: ChartConfig,
) -> Result<HistogramResult, VisualizationError> {
    if data.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data array".to_string()));
    }

    // Filter out NaN values
    let clean_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if clean_data.is_empty() {
        return Err(VisualizationError::InvalidData("No finite values in data".to_string()));
    }

    let min_val = clean_data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = clean_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_val: f64 = clean_data.iter().sum::<f64>() / clean_data.len() as f64;

    // Auto-calculate bins using Sturges' rule if not specified
    let n_bins = bins.unwrap_or_else(|| {
        let n = clean_data.len() as f64;
        (1.0 + 3.322 * n.log10()).ceil() as usize
    }).max(1);

    // Calculate bin edges and counts
    let bin_width = if (max_val - min_val).abs() < 1e-10 {
        1.0
    } else {
        (max_val - min_val) / n_bins as f64
    };

    let mut counts = vec![0u32; n_bins];

    for &val in &clean_data {
        let bin_idx = if bin_width > 0.0 {
            ((val - min_val) / bin_width).floor() as usize
        } else {
            0
        };
        let bin_idx = bin_idx.min(n_bins - 1);
        counts[bin_idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Histogram");
        let x_label = config.x_label.as_deref().unwrap_or("Value");
        let y_label = config.y_label.as_deref().unwrap_or("Frequency");

        let x_range = min_val..(max_val + bin_width * 0.01);
        let y_range = 0u32..(max_count + max_count / 10 + 1);

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range, y_range)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .label_style(("sans-serif", config.tick_font_size))
            .axis_desc_style(("sans-serif", config.label_font_size))
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw histogram bars as rectangles
        for (i, &count) in counts.iter().enumerate() {
            let x0 = min_val + i as f64 * bin_width;
            let x1 = x0 + bin_width;

            chart.draw_series(std::iter::once(
                Rectangle::new([(x0, 0), (x1, count)], DEFAULT_SERIES_COLOR.mix(0.7).filled())
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    // Encode as PNG then base64
    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(HistogramResult {
        image_base64,
        bins: n_bins,
        min: min_val,
        max: max_val,
        mean: mean_val,
    })
}

/// Result of scatter plot generation.
pub struct ScatterResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of points plotted
    pub n_points: usize,
    /// Correlation coefficient (if calculated)
    pub correlation: Option<f64>,
}

impl std::fmt::Display for ScatterResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Scatter Plot")?;
        writeln!(f, "============")?;
        writeln!(f, "Points: {}", self.n_points)?;
        if let Some(r) = self.correlation {
            writeln!(f, "Correlation: {:.4}", r)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a scatter plot from two numeric arrays.
///
/// # Arguments
/// * `x` - X-axis values
/// * `y` - Y-axis values
/// * `config` - Chart configuration
pub fn scatter_plot(
    x: &[f64],
    y: &[f64],
    config: ChartConfig,
) -> Result<ScatterResult, VisualizationError> {
    if x.len() != y.len() {
        return Err(VisualizationError::InvalidData(
            format!("X and Y must have same length: {} vs {}", x.len(), y.len())
        ));
    }

    if x.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    // Filter out pairs with NaN
    let points: Vec<(f64, f64)> = x.iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(xi, yi)| (*xi, *yi))
        .collect();

    if points.is_empty() {
        return Err(VisualizationError::InvalidData("No finite value pairs".to_string()));
    }

    let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    // Add 5% padding
    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.05;

    // Calculate correlation
    let correlation = calculate_correlation(&points);

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Scatter Plot");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .label_style(("sans-serif", config.tick_font_size))
            .axis_desc_style(("sans-serif", config.label_font_size))
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        chart
            .draw_series(
                points.iter().map(|(xi, yi)| {
                    Circle::new((*xi, *yi), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled())
                }),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(ScatterResult {
        image_base64,
        n_points: points.len(),
        correlation,
    })
}

/// Result of box plot generation.
pub struct BoxPlotResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Statistics for each group
    pub statistics: Vec<BoxPlotStats>,
}

/// Statistics for a single box in the box plot.
#[derive(Debug, Clone)]
pub struct BoxPlotStats {
    pub label: String,
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    pub outliers: Vec<f64>,
}

impl std::fmt::Display for BoxPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Box Plot")?;
        writeln!(f, "========")?;
        for stat in &self.statistics {
            writeln!(f, "{}: min={:.2}, Q1={:.2}, med={:.2}, Q3={:.2}, max={:.2}",
                stat.label, stat.min, stat.q1, stat.median, stat.q3, stat.max)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a box plot from multiple data groups.
///
/// # Arguments
/// * `groups` - Vector of (label, data) pairs
/// * `config` - Chart configuration
pub fn box_plot(
    groups: &[(String, Vec<f64>)],
    config: ChartConfig,
) -> Result<BoxPlotResult, VisualizationError> {
    if groups.is_empty() {
        return Err(VisualizationError::InvalidData("No groups provided".to_string()));
    }

    // Calculate statistics for each group
    let mut statistics = Vec::new();
    let mut all_values = Vec::new();

    for (label, data) in groups {
        let clean: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
        if clean.is_empty() {
            continue;
        }

        let stats = calculate_box_stats(label.clone(), &clean);
        all_values.extend(clean.iter().copied());
        statistics.push(stats);
    }

    if statistics.is_empty() {
        return Err(VisualizationError::InvalidData("No valid data in groups".to_string()));
    }

    let y_min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Box Plot");
        let y_label = config.y_label.as_deref().unwrap_or("Value");

        // Build x-axis with continuous coordinates (0 to n_groups)
        let labels: Vec<String> = statistics.iter().map(|s| s.label.clone()).collect();
        let n_groups = labels.len() as f64;

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                -0.5..(n_groups - 0.5),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .y_desc(y_label)
            .x_label_formatter(&|x| {
                let idx = x.round() as usize;
                labels.get(idx).cloned().unwrap_or_default()
            })
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw box plots manually
        for (i, stat) in statistics.iter().enumerate() {
            let box_half_width = 0.3;
            let x_center = i as f64;
            let x_left = x_center - box_half_width;
            let x_right = x_center + box_half_width;

            // Draw box (Q1 to Q3)
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_left, stat.q1), (x_right, stat.q3)],
                DEFAULT_SERIES_COLOR.mix(0.5).filled(),
            ))).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw box border
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_left, stat.q1), (x_right, stat.q3)],
                BLACK.stroke_width(1),
            ))).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw median line
            chart.draw_series(LineSeries::new(
                vec![(x_left, stat.median), (x_right, stat.median)],
                BLACK.stroke_width(2),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw whiskers
            // Lower whisker (vertical line)
            chart.draw_series(LineSeries::new(
                vec![(x_center, stat.min), (x_center, stat.q1)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            // Lower whisker cap
            chart.draw_series(LineSeries::new(
                vec![(x_left + 0.1, stat.min), (x_right - 0.1, stat.min)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Upper whisker (vertical line)
            chart.draw_series(LineSeries::new(
                vec![(x_center, stat.q3), (x_center, stat.max)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            // Upper whisker cap
            chart.draw_series(LineSeries::new(
                vec![(x_left + 0.1, stat.max), (x_right - 0.1, stat.max)],
                BLACK.stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw outliers
            chart.draw_series(
                stat.outliers.iter().map(|&outlier| {
                    Circle::new((x_center, outlier), 3, OUTLIER_COLOR.filled())
                })
            ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(BoxPlotResult {
        image_base64,
        statistics,
    })
}

/// Result of line chart generation.
pub struct LineChartResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of data points
    pub n_points: usize,
    /// Number of series
    pub n_series: usize,
}

impl std::fmt::Display for LineChartResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Line Chart")?;
        writeln!(f, "==========")?;
        writeln!(f, "Series: {}", self.n_series)?;
        writeln!(f, "Points: {}", self.n_points)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a line chart from one or more time series.
///
/// # Arguments
/// * `series` - Vector of (label, x_values, y_values) tuples
/// * `config` - Chart configuration
pub fn line_chart(
    series: &[(String, Vec<f64>, Vec<f64>)],
    config: ChartConfig,
) -> Result<LineChartResult, VisualizationError> {
    if series.is_empty() {
        return Err(VisualizationError::InvalidData("No series provided".to_string()));
    }

    // Find global bounds
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut total_points = 0;

    for (_, x_vals, y_vals) in series {
        for (xi, yi) in x_vals.iter().zip(y_vals.iter()) {
            if xi.is_finite() && yi.is_finite() {
                x_min = x_min.min(*xi);
                x_max = x_max.max(*xi);
                y_min = y_min.min(*yi);
                y_max = y_max.max(*yi);
                total_points += 1;
            }
        }
    }

    if total_points == 0 {
        return Err(VisualizationError::InvalidData("No valid data points".to_string()));
    }

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.05;

    // Colors for multiple series (brand palette)
    let colors = CHART_PALETTE;

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Line Chart");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .label_style(("sans-serif", config.tick_font_size))
            .axis_desc_style(("sans-serif", config.label_font_size))
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw each series
        for (idx, (label, x_vals, y_vals)) in series.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let points: Vec<(f64, f64)> = x_vals.iter()
                .zip(y_vals.iter())
                .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
                .map(|(xi, yi)| (*xi, *yi))
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2)));
        }

        // Draw legend if multiple series
        if series.len() > 1 {
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(LineChartResult {
        image_base64,
        n_points: total_points,
        n_series: series.len(),
    })
}

/// Result of event study plot generation.
pub struct EventStudyResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of time periods
    pub n_periods: usize,
    /// Treatment period (if identified)
    pub treatment_period: Option<f64>,
}

impl std::fmt::Display for EventStudyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Event Study Plot")?;
        writeln!(f, "================")?;
        writeln!(f, "Time periods: {}", self.n_periods)?;
        if let Some(t) = self.treatment_period {
            writeln!(f, "Treatment period: {}", t)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate an event study plot showing treatment effects over time with confidence intervals.
///
/// # Arguments
/// * `time` - Time periods (relative to treatment, e.g., -3, -2, -1, 0, 1, 2, 3)
/// * `estimates` - Point estimates for each period
/// * `ci_lower` - Lower bound of confidence interval
/// * `ci_upper` - Upper bound of confidence interval
/// * `config` - Chart configuration
pub fn event_study_plot(
    time: &[f64],
    estimates: &[f64],
    ci_lower: &[f64],
    ci_upper: &[f64],
    config: ChartConfig,
) -> Result<EventStudyResult, VisualizationError> {
    if time.is_empty() || estimates.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    let n = time.len();
    if estimates.len() != n || ci_lower.len() != n || ci_upper.len() != n {
        return Err(VisualizationError::InvalidData(
            "All arrays must have the same length".to_string()
        ));
    }

    // Filter out invalid values and collect valid points
    let mut valid_points: Vec<(f64, f64, f64, f64)> = Vec::new();
    for i in 0..n {
        if time[i].is_finite() && estimates[i].is_finite()
            && ci_lower[i].is_finite() && ci_upper[i].is_finite() {
            valid_points.push((time[i], estimates[i], ci_lower[i], ci_upper[i]));
        }
    }

    if valid_points.is_empty() {
        return Err(VisualizationError::InvalidData("No valid data points".to_string()));
    }

    // Sort by time
    valid_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate bounds
    let x_min = valid_points.first().unwrap().0;
    let x_max = valid_points.last().unwrap().0;
    let y_min = valid_points.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
    let y_max = valid_points.iter().map(|p| p.3).fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(1.0) * 0.1;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Detect treatment period (typically 0)
    let treatment_period = if valid_points.iter().any(|p| (p.0 - 0.0).abs() < 0.01) {
        Some(0.0)
    } else {
        None
    };

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Event Study");
        let x_label = config.x_label.as_deref().unwrap_or("Time Relative to Treatment");
        let y_label = config.y_label.as_deref().unwrap_or("Treatment Effect");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .label_style(("sans-serif", config.tick_font_size))
            .axis_desc_style(("sans-serif", config.label_font_size))
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line (dashed)
        chart.draw_series(LineSeries::new(
            vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
            BLACK.stroke_width(1),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw vertical treatment line at t=0 if present
        if treatment_period.is_some() {
            chart.draw_series(LineSeries::new(
                vec![(0.0, y_min - y_pad), (0.0, y_max + y_pad)],
                RGBColor(128, 128, 128).stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        // Draw confidence interval as shaded region
        let ci_polygon: Vec<(f64, f64)> = {
            let mut points = Vec::new();
            // Upper bound (forward)
            for &(t, _, _, upper) in &valid_points {
                points.push((t, upper));
            }
            // Lower bound (backward)
            for &(t, _, lower, _) in valid_points.iter().rev() {
                points.push((t, lower));
            }
            points
        };

        chart.draw_series(std::iter::once(
            Polygon::new(ci_polygon, DEFAULT_SERIES_COLOR.mix(0.2).filled())
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point estimates as line
        let estimate_points: Vec<(f64, f64)> = valid_points.iter()
            .map(|&(t, est, _, _)| (t, est))
            .collect();

        chart.draw_series(LineSeries::new(
            estimate_points.clone(),
            DEFAULT_SERIES_COLOR.stroke_width(2),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point markers
        chart.draw_series(
            estimate_points.iter().map(|&(x, y)| {
                Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.filled())
            })
        ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(EventStudyResult {
        image_base64,
        n_periods: valid_points.len(),
        treatment_period,
    })
}

/// Result of coefficient plot generation.
pub struct CoefficientPlotResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of coefficients plotted
    pub n_coefficients: usize,
}

impl std::fmt::Display for CoefficientPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Coefficient Plot")?;
        writeln!(f, "================")?;
        writeln!(f, "Coefficients: {}", self.n_coefficients)?;
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a coefficient plot showing regression coefficients with confidence intervals.
///
/// # Arguments
/// * `names` - Variable names
/// * `estimates` - Coefficient estimates
/// * `ci_lower` - Lower bound of confidence interval
/// * `ci_upper` - Upper bound of confidence interval
/// * `config` - Chart configuration
/// * `horizontal` - If true, draw horizontal bars (default)
pub fn coefficient_plot(
    names: &[String],
    estimates: &[f64],
    ci_lower: &[f64],
    ci_upper: &[f64],
    config: ChartConfig,
    horizontal: bool,
) -> Result<CoefficientPlotResult, VisualizationError> {
    if names.is_empty() || estimates.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    let n = names.len();
    if estimates.len() != n || ci_lower.len() != n || ci_upper.len() != n {
        return Err(VisualizationError::InvalidData(
            "All arrays must have the same length".to_string()
        ));
    }

    // Collect valid coefficients
    let mut valid_coefs: Vec<(String, f64, f64, f64)> = Vec::new();
    for i in 0..n {
        if estimates[i].is_finite() && ci_lower[i].is_finite() && ci_upper[i].is_finite() {
            valid_coefs.push((names[i].clone(), estimates[i], ci_lower[i], ci_upper[i]));
        }
    }

    if valid_coefs.is_empty() {
        return Err(VisualizationError::InvalidData("No valid coefficients".to_string()));
    }

    // Calculate bounds
    let x_min = valid_coefs.iter().map(|c| c.2).fold(f64::INFINITY, f64::min);
    let x_max = valid_coefs.iter().map(|c| c.3).fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min).max(0.1) * 0.1;

    // Ensure zero is visible
    let x_range_min = x_min.min(0.0) - x_pad;
    let x_range_max = x_max.max(0.0) + x_pad;

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Coefficient Plot");
        let x_label = config.x_label.as_deref().unwrap_or("Estimate");

        let n_coefs = valid_coefs.len() as f64;

        if horizontal {
            // Horizontal layout: coefficients on Y-axis, values on X-axis
            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", config.title_font_size))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(120) // More space for variable names
                .build_cartesian_2d(
                    x_range_min..x_range_max,
                    -0.5..(n_coefs - 0.5),
                )
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            chart
                .configure_mesh()
                .x_desc(x_label)
                .y_label_formatter(&|y| {
                    let idx = y.round() as usize;
                    valid_coefs.get(idx).map(|c| c.0.clone()).unwrap_or_default()
                })
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw vertical zero line
            chart.draw_series(LineSeries::new(
                vec![(0.0, -0.5), (0.0, n_coefs - 0.5)],
                RGBColor(128, 128, 128).stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw each coefficient with error bar
            for (i, (_name, est, lower, upper)) in valid_coefs.iter().enumerate() {
                let y_pos = i as f64;

                // Draw error bar (horizontal line)
                chart.draw_series(LineSeries::new(
                    vec![(*lower, y_pos), (*upper, y_pos)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw caps on error bar
                chart.draw_series(LineSeries::new(
                    vec![(*lower, y_pos - 0.1), (*lower, y_pos + 0.1)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                chart.draw_series(LineSeries::new(
                    vec![(*upper, y_pos - 0.1), (*upper, y_pos + 0.1)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw point estimate
                chart.draw_series(std::iter::once(
                    Circle::new((*est, y_pos), 5, DEFAULT_SERIES_COLOR.filled())
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        } else {
            // Vertical layout: coefficients on X-axis, values on Y-axis
            let y_range_min = x_range_min; // Swap x and y ranges
            let y_range_max = x_range_max;

            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", config.title_font_size))
                .margin(10)
                .x_label_area_size(60)
                .y_label_area_size(60)
                .build_cartesian_2d(
                    -0.5..(n_coefs - 0.5),
                    y_range_min..y_range_max,
                )
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            chart
                .configure_mesh()
                .y_desc(x_label)
                .x_label_formatter(&|x| {
                    let idx = x.round() as usize;
                    valid_coefs.get(idx).map(|c| c.0.clone()).unwrap_or_default()
                })
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw horizontal zero line
            chart.draw_series(LineSeries::new(
                vec![(-0.5, 0.0), (n_coefs - 0.5, 0.0)],
                RGBColor(128, 128, 128).stroke_width(1),
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw each coefficient with error bar
            for (i, (_name, est, lower, upper)) in valid_coefs.iter().enumerate() {
                let x_pos = i as f64;

                // Draw error bar (vertical line)
                chart.draw_series(LineSeries::new(
                    vec![(x_pos, *lower), (x_pos, *upper)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw caps on error bar
                chart.draw_series(LineSeries::new(
                    vec![(x_pos - 0.1, *lower), (x_pos + 0.1, *lower)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                chart.draw_series(LineSeries::new(
                    vec![(x_pos - 0.1, *upper), (x_pos + 0.1, *upper)],
                    DEFAULT_SERIES_COLOR.stroke_width(2),
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw point estimate
                chart.draw_series(std::iter::once(
                    Circle::new((x_pos, *est), 5, DEFAULT_SERIES_COLOR.filled())
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(CoefficientPlotResult {
        image_base64,
        n_coefficients: valid_coefs.len(),
    })
}

/// Result of IRF plot generation.
pub struct IrfPlotResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of horizons
    pub n_horizons: usize,
    /// Shock variable name
    pub shock: Option<String>,
    /// Response variable name
    pub response: Option<String>,
}

impl std::fmt::Display for IrfPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Impulse Response Function Plot")?;
        writeln!(f, "===============================")?;
        writeln!(f, "Horizons: {}", self.n_horizons)?;
        if let Some(ref shock) = self.shock {
            writeln!(f, "Shock: {}", shock)?;
        }
        if let Some(ref response) = self.response {
            writeln!(f, "Response: {}", response)?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate an IRF (Impulse Response Function) plot.
///
/// # Arguments
/// * `horizons` - Time horizons (0, 1, 2, ...)
/// * `responses` - Response values at each horizon
/// * `ci_lower` - Optional lower bound of confidence interval
/// * `ci_upper` - Optional upper bound of confidence interval
/// * `shock_label` - Optional label for shock variable
/// * `response_label` - Optional label for response variable
/// * `config` - Chart configuration
pub fn irf_plot(
    horizons: &[f64],
    responses: &[f64],
    ci_lower: Option<&[f64]>,
    ci_upper: Option<&[f64]>,
    shock_label: Option<&str>,
    response_label: Option<&str>,
    config: ChartConfig,
) -> Result<IrfPlotResult, VisualizationError> {
    if horizons.is_empty() || responses.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    if horizons.len() != responses.len() {
        return Err(VisualizationError::InvalidData(
            "Horizons and responses must have same length".to_string()
        ));
    }

    let n = horizons.len();
    let has_ci = ci_lower.is_some() && ci_upper.is_some();

    if has_ci {
        let lower = ci_lower.unwrap();
        let upper = ci_upper.unwrap();
        if lower.len() != n || upper.len() != n {
            return Err(VisualizationError::InvalidData(
                "CI arrays must have same length as horizons".to_string()
            ));
        }
    }

    // Collect valid points
    let mut valid_points: Vec<(f64, f64, Option<f64>, Option<f64>)> = Vec::new();
    for i in 0..n {
        if horizons[i].is_finite() && responses[i].is_finite() {
            let lower = if has_ci {
                let l = ci_lower.unwrap()[i];
                if l.is_finite() { Some(l) } else { None }
            } else { None };
            let upper = if has_ci {
                let u = ci_upper.unwrap()[i];
                if u.is_finite() { Some(u) } else { None }
            } else { None };
            valid_points.push((horizons[i], responses[i], lower, upper));
        }
    }

    if valid_points.is_empty() {
        return Err(VisualizationError::InvalidData("No valid data points".to_string()));
    }

    // Sort by horizon
    valid_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate bounds
    let x_min = valid_points.first().unwrap().0;
    let x_max = valid_points.last().unwrap().0;

    let mut y_min = valid_points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let mut y_max = valid_points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    // Include CI bounds if present
    if has_ci {
        for p in &valid_points {
            if let Some(l) = p.2 { y_min = y_min.min(l); }
            if let Some(u) = p.3 { y_max = y_max.max(u); }
        }
    }

    let x_pad = (x_max - x_min).max(1.0) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Ensure zero is visible on y-axis
    y_min = y_min.min(0.0);
    y_max = y_max.max(0.0);

    // Create image buffer
    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or_else(|| {
            if shock_label.is_some() && response_label.is_some() {
                "Impulse Response Function"
            } else {
                "IRF"
            }
        });
        let x_label = config.x_label.as_deref().unwrap_or("Horizon");
        let y_label = config.y_label.as_deref().unwrap_or("Response");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc(x_label)
            .y_desc(y_label)
            .label_style(("sans-serif", config.tick_font_size))
            .axis_desc_style(("sans-serif", config.label_font_size))
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line
        chart.draw_series(LineSeries::new(
            vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
            BLACK.stroke_width(1),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw confidence interval as shaded region if present
        if has_ci {
            let ci_points_with_bounds: Vec<_> = valid_points.iter()
                .filter(|p| p.2.is_some() && p.3.is_some())
                .collect();

            if ci_points_with_bounds.len() > 1 {
                let ci_polygon: Vec<(f64, f64)> = {
                    let mut points = Vec::new();
                    // Upper bound (forward)
                    for p in &ci_points_with_bounds {
                        points.push((p.0, p.3.unwrap()));
                    }
                    // Lower bound (backward)
                    for p in ci_points_with_bounds.iter().rev() {
                        points.push((p.0, p.2.unwrap()));
                    }
                    points
                };

                chart.draw_series(std::iter::once(
                    Polygon::new(ci_polygon, DEFAULT_SERIES_COLOR.mix(0.2).filled())
                )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        // Draw response line
        let response_points: Vec<(f64, f64)> = valid_points.iter()
            .map(|p| (p.0, p.1))
            .collect();

        chart.draw_series(LineSeries::new(
            response_points.clone(),
            DEFAULT_SERIES_COLOR.stroke_width(2),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point markers
        chart.draw_series(
            response_points.iter().map(|&(x, y)| {
                Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.filled())
            })
        ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(IrfPlotResult {
        image_base64,
        n_horizons: valid_points.len(),
        shock: shock_label.map(String::from),
        response: response_label.map(String::from),
    })
}

/// Result of residual diagnostics plot generation.
pub struct ResidualDiagnosticsResult {
    /// Residuals vs Fitted values plot (base64 PNG)
    pub residuals_vs_fitted: String,
    /// Q-Q plot for normality (base64 PNG)
    pub qq_plot: String,
    /// Scale-Location plot (base64 PNG)
    pub scale_location: String,
    /// Residuals vs Leverage plot (base64 PNG)
    pub residuals_vs_leverage: String,
    /// Number of observations
    pub n_observations: usize,
    /// Standardized residuals
    pub standardized_residuals: Vec<f64>,
    /// Cook's distances
    pub cooks_distance: Vec<f64>,
}

impl std::fmt::Display for ResidualDiagnosticsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Residual Diagnostics")?;
        writeln!(f, "====================")?;
        writeln!(f, "Observations: {}", self.n_observations)?;
        writeln!(f)?;
        writeln!(f, "Plots Generated:")?;
        writeln!(f, "  - Residuals vs Fitted: {} bytes", self.residuals_vs_fitted.len())?;
        writeln!(f, "  - Q-Q Plot: {} bytes", self.qq_plot.len())?;
        writeln!(f, "  - Scale-Location: {} bytes", self.scale_location.len())?;
        writeln!(f, "  - Residuals vs Leverage: {} bytes", self.residuals_vs_leverage.len())
    }
}

/// Generate residual diagnostic plots for regression analysis.
///
/// Creates four diagnostic plots:
/// 1. Residuals vs Fitted - checks linearity and homoscedasticity
/// 2. Normal Q-Q - checks normality of residuals
/// 3. Scale-Location - checks homoscedasticity (√|standardized residuals| vs fitted)
/// 4. Residuals vs Leverage - identifies influential observations
///
/// # Arguments
/// * `fitted` - Fitted/predicted values from the regression
/// * `residuals` - Residual values (y - fitted)
/// * `leverage` - Optional leverage (hat) values for each observation
/// * `config` - Chart configuration
pub fn residual_diagnostics(
    fitted: &[f64],
    residuals: &[f64],
    leverage: Option<&[f64]>,
    config: ChartConfig,
) -> Result<ResidualDiagnosticsResult, VisualizationError> {
    if fitted.is_empty() || residuals.is_empty() {
        return Err(VisualizationError::InvalidData("Empty data arrays".to_string()));
    }

    if fitted.len() != residuals.len() {
        return Err(VisualizationError::InvalidData(
            "Fitted and residuals must have same length".to_string()
        ));
    }

    let n = fitted.len();

    // Filter out invalid values
    let valid_indices: Vec<usize> = (0..n)
        .filter(|&i| fitted[i].is_finite() && residuals[i].is_finite())
        .collect();

    if valid_indices.is_empty() {
        return Err(VisualizationError::InvalidData("No valid data points".to_string()));
    }

    let fitted_valid: Vec<f64> = valid_indices.iter().map(|&i| fitted[i]).collect();
    let residuals_valid: Vec<f64> = valid_indices.iter().map(|&i| residuals[i]).collect();

    let n_valid = fitted_valid.len();

    // Calculate standardized residuals
    let mean_residual: f64 = residuals_valid.iter().sum::<f64>() / n_valid as f64;
    let variance: f64 = residuals_valid.iter()
        .map(|r| (r - mean_residual).powi(2))
        .sum::<f64>() / (n_valid - 1) as f64;
    let std_dev = variance.sqrt();

    let standardized_residuals: Vec<f64> = if std_dev > 1e-10 {
        residuals_valid.iter().map(|r| (r - mean_residual) / std_dev).collect()
    } else {
        vec![0.0; n_valid]
    };

    // Calculate or use provided leverage values
    let leverage_valid: Vec<f64> = if let Some(lev) = leverage {
        if lev.len() != n {
            return Err(VisualizationError::InvalidData(
                "Leverage must have same length as fitted".to_string()
            ));
        }
        valid_indices.iter().map(|&i| lev[i]).collect()
    } else {
        // Simple approximation: 1/n + (x - mean(x))^2 / sum((x - mean(x))^2)
        let mean_fitted: f64 = fitted_valid.iter().sum::<f64>() / n_valid as f64;
        let ss_fitted: f64 = fitted_valid.iter().map(|f| (f - mean_fitted).powi(2)).sum();
        if ss_fitted > 1e-10 {
            fitted_valid.iter()
                .map(|f| 1.0 / n_valid as f64 + (f - mean_fitted).powi(2) / ss_fitted)
                .collect()
        } else {
            vec![1.0 / n_valid as f64; n_valid]
        }
    };

    // Calculate Cook's distance: D_i = (r_i^2 / p) * (h_i / (1 - h_i))
    // where p is number of parameters (approximate with 2 for simple regression)
    let p = 2.0;
    let cooks_distance: Vec<f64> = standardized_residuals.iter()
        .zip(leverage_valid.iter())
        .map(|(r, h)| {
            let h_clamped = h.min(0.9999);
            (r.powi(2) / p) * (h_clamped / (1.0 - h_clamped))
        })
        .collect();

    // Generate the four diagnostic plots
    let residuals_vs_fitted = generate_residuals_vs_fitted(&fitted_valid, &residuals_valid, &config)?;
    let qq_plot = generate_qq_plot(&standardized_residuals, &config)?;
    let scale_location = generate_scale_location(&fitted_valid, &standardized_residuals, &config)?;
    let residuals_vs_leverage = generate_residuals_vs_leverage(
        &leverage_valid, &standardized_residuals, &cooks_distance, &config
    )?;

    Ok(ResidualDiagnosticsResult {
        residuals_vs_fitted,
        qq_plot,
        scale_location,
        residuals_vs_leverage,
        n_observations: n_valid,
        standardized_residuals,
        cooks_distance,
    })
}

/// Generate Residuals vs Fitted plot.
fn generate_residuals_vs_fitted(
    fitted: &[f64],
    residuals: &[f64],
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    let x_min = fitted.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = residuals.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Ensure zero is visible on y-axis
    let y_min_adj = y_min.min(0.0) - y_pad;
    let y_max_adj = y_max.max(0.0) + y_pad;

    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Residuals vs Fitted", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                y_min_adj..y_max_adj,
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Fitted values")
            .y_desc("Residuals")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line
        chart.draw_series(LineSeries::new(
            vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
            RGBColor(128, 128, 128).stroke_width(1),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        let points: Vec<(f64, f64)> = fitted.iter().zip(residuals.iter()).map(|(&f, &r)| (f, r)).collect();
        chart.draw_series(
            points.iter().map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled()))
        ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Calculate and draw LOWESS-like smooth (simple moving average for demonstration)
        let smoothed = calculate_smooth(&points, 0.3);
        if smoothed.len() > 1 {
            chart.draw_series(LineSeries::new(smoothed, TREND_LINE_COLOR.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Generate Q-Q plot for normality check.
fn generate_qq_plot(
    standardized_residuals: &[f64],
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    let n = standardized_residuals.len();

    // Sort residuals
    let mut sorted_residuals: Vec<f64> = standardized_residuals.to_vec();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate theoretical quantiles (standard normal)
    let theoretical: Vec<f64> = (1..=n)
        .map(|i| normal_quantile((i as f64 - 0.5) / n as f64))
        .collect();

    let x_min = theoretical.first().copied().unwrap_or(-3.0);
    let x_max = theoretical.last().copied().unwrap_or(3.0);
    let y_min = sorted_residuals.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = sorted_residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let range = (x_max - x_min).max(y_max - y_min);
    let x_pad = range * 0.1;
    let y_pad = range * 0.1;

    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Normal Q-Q", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Theoretical Quantiles")
            .y_desc("Standardized Residuals")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw reference line (y = x for standardized residuals)
        let line_min = x_min.min(y_min);
        let line_max = x_max.max(y_max);
        chart.draw_series(LineSeries::new(
            vec![(line_min, line_min), (line_max, line_max)],
            RGBColor(128, 128, 128).stroke_width(1),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw Q-Q points
        let points: Vec<(f64, f64)> = theoretical.iter().zip(sorted_residuals.iter())
            .map(|(&t, &s)| (t, s))
            .collect();
        chart.draw_series(
            points.iter().map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled()))
        ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Generate Scale-Location plot.
fn generate_scale_location(
    fitted: &[f64],
    standardized_residuals: &[f64],
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    // Calculate sqrt of absolute standardized residuals
    let sqrt_abs_resid: Vec<f64> = standardized_residuals.iter()
        .map(|r| r.abs().sqrt())
        .collect();

    let x_min = fitted.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_max = sqrt_abs_resid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = y_max * 0.1;

    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Scale-Location", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                (x_min - x_pad)..(x_max + x_pad),
                0.0..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Fitted values")
            .y_desc("√|Standardized residuals|")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        let points: Vec<(f64, f64)> = fitted.iter().zip(sqrt_abs_resid.iter())
            .map(|(&f, &r)| (f, r))
            .collect();
        chart.draw_series(
            points.iter().map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled()))
        ).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw smooth line
        let smoothed = calculate_smooth(&points, 0.3);
        if smoothed.len() > 1 {
            chart.draw_series(LineSeries::new(smoothed, TREND_LINE_COLOR.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Generate Residuals vs Leverage plot.
fn generate_residuals_vs_leverage(
    leverage: &[f64],
    standardized_residuals: &[f64],
    cooks_distance: &[f64],
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    let x_min = 0.0;
    let x_max = leverage.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = standardized_residuals.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = standardized_residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = x_max * 0.1;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Calculate Cook's distance threshold (typically 0.5 or 1.0)
    let cooks_threshold = 0.5;
    let has_influential = cooks_distance.iter().any(|&c| c > cooks_threshold);

    let mut buffer = vec![0u8; (config.width * config.height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Residuals vs Leverage", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                x_min..(x_max + x_pad),
                (y_min - y_pad)..(y_max + y_pad),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Leverage")
            .y_desc("Standardized residuals")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line
        chart.draw_series(LineSeries::new(
            vec![(x_min, 0.0), (x_max + x_pad, 0.0)],
            RGBColor(128, 128, 128).stroke_width(1),
        )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw Cook's distance contour lines (approximate)
        if has_influential {
            // D = r^2 * h / (p * (1-h)) => r = sqrt(D * p * (1-h) / h)
            let p = 2.0;
            for &d in &[0.5, 1.0] {
                let contour_points: Vec<(f64, f64)> = (1..100)
                    .map(|i| {
                        let h = (i as f64) * x_max / 100.0;
                        let h_clamped = h.max(0.01).min(0.99);
                        let r = (d * p * (1.0 - h_clamped) / h_clamped).sqrt();
                        (h, r)
                    })
                    .filter(|&(_, r)| r.is_finite() && r <= y_max + y_pad)
                    .collect();

                if contour_points.len() > 1 {
                    chart.draw_series(LineSeries::new(
                        contour_points.clone(),
                        OUTLIER_COLOR.stroke_width(1),
                    )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                    // Draw negative contour
                    let neg_contour: Vec<(f64, f64)> = contour_points.iter()
                        .map(|&(h, r)| (h, -r))
                        .collect();
                    chart.draw_series(LineSeries::new(
                        neg_contour,
                        OUTLIER_COLOR.stroke_width(1),
                    )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }
            }
        }

        // Draw points (color by Cook's distance)
        for i in 0..leverage.len() {
            let color = if cooks_distance[i] > cooks_threshold {
                OUTLIER_COLOR
            } else {
                DEFAULT_SERIES_COLOR
            };
            chart.draw_series(std::iter::once(
                Circle::new((leverage[i], standardized_residuals[i]), 4, color.mix(0.7).filled())
            )).map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present().map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Calculate a simple smoothing line (local regression approximation).
fn calculate_smooth(points: &[(f64, f64)], span: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return Vec::new();
    }

    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_points.len();
    let window = ((n as f64 * span) as usize).max(3).min(n);
    let half_window = window / 2;

    let mut smoothed = Vec::new();

    for i in 0..n {
        let start = if i >= half_window { i - half_window } else { 0 };
        let end = (i + half_window + 1).min(n);

        let local_points = &sorted_points[start..end];
        let mean_x: f64 = local_points.iter().map(|p| p.0).sum::<f64>() / local_points.len() as f64;
        let mean_y: f64 = local_points.iter().map(|p| p.1).sum::<f64>() / local_points.len() as f64;

        smoothed.push((mean_x, mean_y));
    }

    // Remove duplicates in x
    smoothed.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);
    smoothed
}

/// Approximation of the standard normal quantile function (inverse CDF).
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation by Abramowitz and Stegun
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }

    let p_clamped = p.max(1e-10).min(1.0 - 1e-10);

    if p_clamped < 0.5 {
        -rational_approximation((-2.0 * p_clamped.ln()).sqrt())
    } else {
        rational_approximation((-2.0 * (1.0 - p_clamped).ln()).sqrt())
    }
}

fn rational_approximation(t: f64) -> f64 {
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}

// Helper functions

fn calculate_correlation(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|p| p.0).sum();
    let sum_y: f64 = points.iter().map(|p| p.1).sum();
    let sum_xy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let sum_x2: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sum_y2: f64 = points.iter().map(|p| p.1 * p.1).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn calculate_box_stats(label: String, data: &[f64]) -> BoxPlotStats {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let q1_idx = n / 4;
    let median_idx = n / 2;
    let q3_idx = 3 * n / 4;

    let q1 = sorted[q1_idx];
    let median = if n % 2 == 0 && n > 1 {
        (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
    } else {
        sorted[median_idx]
    };
    let q3 = sorted[q3_idx.min(n - 1)];

    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let min = sorted.iter().copied().find(|&x| x >= lower_fence).unwrap_or(sorted[0]);
    let max = sorted.iter().rev().copied().find(|&x| x <= upper_fence).unwrap_or(sorted[n - 1]);

    let outliers: Vec<f64> = sorted.iter()
        .copied()
        .filter(|&x| x < lower_fence || x > upper_fence)
        .collect();

    BoxPlotStats { label, min, q1, median, q3, max, outliers }
}

fn encode_rgb_to_png_base64(rgb_buffer: &[u8], width: u32, height: u32) -> Result<String, VisualizationError> {
    use image::{ImageBuffer, Rgb, ImageEncoder, codecs::png::PngEncoder, ColorType};
    use std::io::Cursor;

    // Create image from RGB buffer
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgb_buffer.to_vec())
        .ok_or_else(|| VisualizationError::EncodingError("Failed to create image buffer".to_string()))?;

    // Encode to PNG
    let mut png_data = Vec::new();
    let encoder = PngEncoder::new(Cursor::new(&mut png_data));
    encoder.write_image(
        img.as_raw(),
        width,
        height,
        ColorType::Rgb8,
    ).map_err(|e: image::ImageError| VisualizationError::EncodingError(e.to_string()))?;

    Ok(BASE64.encode(&png_data))
}

/// Result from dendrogram visualization
#[derive(Debug, Clone)]
pub struct DendrogramResult {
    /// Base64-encoded PNG image
    pub image_base64: String,
    /// Number of original samples
    pub n_samples: usize,
    /// Number of merges shown
    pub n_merges: usize,
    /// Maximum merge distance
    pub max_distance: f64,
}

/// Create a dendrogram visualization from hierarchical clustering linkage matrix.
///
/// # Arguments
/// * `linkage_matrix` - Vec of (cluster1_idx, cluster2_idx, distance, size) tuples
/// * `labels` - Optional labels for leaf nodes (original samples)
/// * `config` - Chart configuration
///
/// # Returns
/// DendrogramResult with the base64-encoded image
pub fn dendrogram(
    linkage_matrix: &[(usize, usize, f64, usize)],
    labels: Option<&[String]>,
    config: ChartConfig,
) -> Result<DendrogramResult, VisualizationError> {
    if linkage_matrix.is_empty() {
        return Err(VisualizationError::InvalidData(
            "Linkage matrix is empty".to_string(),
        ));
    }

    let n_merges = linkage_matrix.len();
    let n_samples = n_merges + 1;

    let max_distance = linkage_matrix
        .iter()
        .map(|(_, _, d, _)| *d)
        .fold(0.0_f64, |a, b| a.max(b));

    let width = config.width;
    let height = config.height;
    let title = config.title.unwrap_or_else(|| "Dendrogram".to_string());

    // Create drawing area
    let mut buffer = vec![0u8; (width * height * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width, height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let margin_left = 80;
        let margin_right = 40;
        let margin_top = 60;
        let margin_bottom = 100;

        let plot_width = width as i32 - margin_left - margin_right;
        let plot_height = height as i32 - margin_top - margin_bottom;

        // Draw title
        root.draw(&Text::new(
            title,
            (width as i32 / 2, 20),
            ("sans-serif", 20).into_font().color(&BLACK),
        ))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Calculate leaf positions (x-coordinates for each original sample)
        // We need to determine the order of leaves in the dendrogram
        let leaf_order = compute_leaf_order(linkage_matrix, n_samples);

        // Map each sample to its x position
        let mut leaf_x: Vec<f64> = vec![0.0; n_samples];
        for (pos, &sample_idx) in leaf_order.iter().enumerate() {
            leaf_x[sample_idx] = pos as f64;
        }

        // Track cluster positions: each cluster has (x_center, y_height)
        // First n_samples are original samples at height 0
        let mut cluster_pos: Vec<(f64, f64)> = Vec::with_capacity(n_samples + n_merges);
        for i in 0..n_samples {
            cluster_pos.push((leaf_x[i], 0.0));
        }

        // Process merges and record positions of new clusters
        for (c1, c2, dist, _size) in linkage_matrix.iter() {
            let (x1, _y1) = cluster_pos[*c1];
            let (x2, _y2) = cluster_pos[*c2];
            let new_x = (x1 + x2) / 2.0;
            cluster_pos.push((new_x, *dist));
        }

        // Scaling functions
        let x_scale = |x: f64| -> i32 {
            margin_left + ((x / (n_samples - 1) as f64) * plot_width as f64) as i32
        };
        let y_scale = |y: f64| -> i32 {
            margin_top + plot_height - ((y / max_distance) * plot_height as f64) as i32
        };

        // Draw the dendrogram links
        for (i, (c1, c2, dist, _size)) in linkage_matrix.iter().enumerate() {
            let (x1, y1) = cluster_pos[*c1];
            let (x2, y2) = cluster_pos[*c2];
            let new_cluster_idx = n_samples + i;
            let (_, new_y) = cluster_pos[new_cluster_idx];

            // Draw U-shaped connector:
            // Vertical line from c1 to merge height
            root.draw(&PathElement::new(
                vec![
                    (x_scale(x1), y_scale(y1)),
                    (x_scale(x1), y_scale(new_y)),
                ],
                &DEFAULT_SERIES_COLOR,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Vertical line from c2 to merge height
            root.draw(&PathElement::new(
                vec![
                    (x_scale(x2), y_scale(y2)),
                    (x_scale(x2), y_scale(new_y)),
                ],
                &DEFAULT_SERIES_COLOR,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Horizontal line connecting c1 and c2 at merge height
            root.draw(&PathElement::new(
                vec![
                    (x_scale(x1), y_scale(*dist)),
                    (x_scale(x2), y_scale(*dist)),
                ],
                &DEFAULT_SERIES_COLOR,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        // Draw leaf labels
        for (pos, &sample_idx) in leaf_order.iter().enumerate() {
            let label = if let Some(lbls) = labels {
                if sample_idx < lbls.len() {
                    lbls[sample_idx].clone()
                } else {
                    format!("{}", sample_idx)
                }
            } else {
                format!("{}", sample_idx)
            };

            let x = x_scale(pos as f64);
            let y = y_scale(0.0) + 15;

            // Rotate labels if too many samples
            if n_samples > 20 {
                // Draw rotated text (approximate with vertical positioning)
                for (char_idx, c) in label.chars().take(8).enumerate() {
                    root.draw(&Text::new(
                        c.to_string(),
                        (x - 3, y + char_idx as i32 * 10),
                        ("sans-serif", 9).into_font().color(&BLACK),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }
            } else {
                root.draw(&Text::new(
                    label,
                    (x - 10, y),
                    ("sans-serif", 10).into_font().color(&BLACK),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        // Draw y-axis (distance scale)
        root.draw(&PathElement::new(
            vec![
                (margin_left, margin_top),
                (margin_left, margin_top + plot_height),
            ],
            &BLACK,
        ))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Y-axis label
        root.draw(&Text::new(
            "Distance",
            (15, margin_top + plot_height / 2),
            ("sans-serif", 12).into_font().color(&BLACK),
        ))
        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Y-axis ticks
        let n_ticks = 5;
        for i in 0..=n_ticks {
            let dist = (i as f64 / n_ticks as f64) * max_distance;
            let y = y_scale(dist);

            root.draw(&PathElement::new(
                vec![(margin_left - 5, y), (margin_left, y)],
                &BLACK,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            root.draw(&Text::new(
                format!("{:.2}", dist),
                (margin_left - 45, y - 5),
                ("sans-serif", 10).into_font().color(&BLACK),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    let image_base64 = encode_rgb_to_png_base64(&buffer, width, height)?;

    Ok(DendrogramResult {
        image_base64,
        n_samples,
        n_merges,
        max_distance,
    })
}

/// Compute the optimal leaf ordering for dendrogram visualization.
/// Returns the order in which leaves should be placed (left to right).
fn compute_leaf_order(linkage_matrix: &[(usize, usize, f64, usize)], n_samples: usize) -> Vec<usize> {
    // Build a tree structure from linkage matrix
    // Each cluster has left and right children (or is a leaf)

    let n_merges = linkage_matrix.len();

    // For each cluster, store its leaves in order
    let mut cluster_leaves: Vec<Vec<usize>> = Vec::with_capacity(n_samples + n_merges);

    // Initialize leaves (original samples)
    for i in 0..n_samples {
        cluster_leaves.push(vec![i]);
    }

    // Process merges
    for (c1, c2, _, _) in linkage_matrix.iter() {
        let left_leaves = cluster_leaves[*c1].clone();
        let right_leaves = cluster_leaves[*c2].clone();
        let mut merged = left_leaves;
        merged.extend(right_leaves);
        cluster_leaves.push(merged);
    }

    // The last cluster contains all leaves in order
    cluster_leaves.last().cloned().unwrap_or_else(|| (0..n_samples).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = histogram(&data, Some(10), ChartConfig::default()).unwrap();
        assert_eq!(result.bins, 10);
        assert!(!result.image_base64.is_empty());
    }

    #[test]
    fn test_scatter() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + 1.0).collect();
        let result = scatter_plot(&x, &y, ChartConfig::default()).unwrap();
        assert_eq!(result.n_points, 50);
        assert!(result.correlation.unwrap() > 0.99);
    }

    #[test]
    fn test_line_chart() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let series = vec![("sin(x)".to_string(), x, y)];
        let result = line_chart(&series, ChartConfig::default()).unwrap();
        assert_eq!(result.n_series, 1);
    }

    #[test]
    fn test_residual_diagnostics() {
        // Generate synthetic data: y = 2*x + noise
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let fitted: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 10.0).collect();
        // Simulate residuals with some noise
        let residuals: Vec<f64> = (0..50).map(|i| {
            let noise = ((i * 7) % 11) as f64 - 5.0; // Deterministic "noise" for reproducibility
            noise
        }).collect();

        let result = residual_diagnostics(&fitted, &residuals, None, ChartConfig::default()).unwrap();

        assert_eq!(result.n_observations, 50);
        assert!(!result.residuals_vs_fitted.is_empty());
        assert!(!result.qq_plot.is_empty());
        assert!(!result.scale_location.is_empty());
        assert!(!result.residuals_vs_leverage.is_empty());
        assert_eq!(result.standardized_residuals.len(), 50);
        assert_eq!(result.cooks_distance.len(), 50);
    }

    #[test]
    fn test_dendrogram() {
        // Create a simple linkage matrix for 4 samples
        // Format: (cluster1, cluster2, distance, size)
        let linkage_matrix = vec![
            (0, 1, 1.0, 2),   // Merge samples 0 and 1 -> cluster 4
            (2, 3, 1.5, 2),   // Merge samples 2 and 3 -> cluster 5
            (4, 5, 3.0, 4),   // Merge clusters 4 and 5 -> cluster 6
        ];

        let labels = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];

        let result = dendrogram(&linkage_matrix, Some(&labels), ChartConfig::default()).unwrap();

        assert_eq!(result.n_samples, 4);
        assert_eq!(result.n_merges, 3);
        assert!((result.max_distance - 3.0).abs() < 1e-10);
        assert!(!result.image_base64.is_empty());
    }
}
