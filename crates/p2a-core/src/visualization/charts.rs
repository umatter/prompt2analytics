//! Chart generation: histograms, scatter plots, box plots, line charts.

use super::colors::{CHART_PALETTE, DEFAULT_SERIES_COLOR, OUTLIER_COLOR, TREND_LINE_COLOR};
use super::{DEFAULT_HEIGHT, DEFAULT_WIDTH, VisualizationError};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
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
    /// Values per bin; bin i is `(breaks[i], breaks[i + 1]]`, the first also holding
    /// the minimum, with values within 1e-7 bin widths of a break assigned as R's
    /// `hist()` does.
    pub counts: Vec<u32>,
    /// The `bins + 1` edges: `min + i * width` for `i < bins` and `max` last (R's
    /// `seq(min, max, length.out = bins + 1)`); `min + i` (width 1) for constant data.
    pub breaks: Vec<f64>,
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

/// R's `nclass.Sturges`, `ceiling(log2(n) + 1)`, in integers; `n >= 1`.
fn sturges_bins(n: usize) -> usize {
    let log2 = n.ilog2() as usize;
    if n.is_power_of_two() {
        log2 + 1
    } else {
        log2 + 2
    }
}

/// Generate a histogram from numeric data.
///
/// `k` equal-width bins from min to max, right-closed with the lowest value included, values
/// within 1e-7 bin widths of a break assigned as R's `hist()` does (R:
/// `hist(x, breaks = seq(min, max, length.out = k + 1))` with its defaults); default
/// `k` = `nclass.Sturges`; R's default `hist()` breaks (`pretty()`) differ. Constant data
/// (a range of at most 4 eps times the data's magnitude) get `k` bins of width 1 from the
/// value.
///
/// # Arguments
/// * `data` - Numeric values to plot
/// * `bins` - Number of bins (default: `nclass.Sturges`)
/// * `config` - Chart configuration
pub fn histogram(
    data: &[f64],
    bins: Option<usize>,
    config: ChartConfig,
) -> Result<HistogramResult, VisualizationError> {
    if data.is_empty() {
        return Err(VisualizationError::InvalidData(
            "Empty data array".to_string(),
        ));
    }

    // Filter out NaN values
    let clean_data: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if clean_data.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No finite values in data".to_string(),
        ));
    }

    let min_val = clean_data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = clean_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_val: f64 = clean_data.iter().sum::<f64>() / clean_data.len() as f64;

    if bins == Some(0) {
        return Err(VisualizationError::InvalidData(
            "bins must be at least 1".to_string(),
        ));
    }

    if clean_data.len() > u32::MAX as usize {
        return Err(VisualizationError::InvalidData(format!(
            "a histogram counts at most {} values",
            u32::MAX
        )));
    }

    let n_bins = bins.unwrap_or_else(|| sturges_bins(clean_data.len()));

    // Bin width; the breaks exist before anything is counted.
    let range = max_val - min_val;
    let constant = range <= 4.0 * f64::EPSILON * min_val.abs().max(max_val.abs());
    let bin_width = if constant { 1.0 } else { range / n_bins as f64 };

    let mut counts = resource::try_vec(n_bins, 0u32, "histogram bins").map_err(invalid_data)?;
    let n_edges = n_bins
        .checked_add(1)
        .ok_or_else(|| VisualizationError::InvalidData(format!("bins is too large: {n_bins}")))?;
    let mut breaks =
        resource::try_vec(n_edges, 0.0f64, "histogram bins (edges)").map_err(invalid_data)?;
    for (i, edge) in breaks.iter_mut().enumerate() {
        *edge = min_val + i as f64 * bin_width;
    }
    breaks[n_bins] = if constant {
        min_val + n_bins as f64
    } else {
        max_val
    };

    let x_range = checked_axis_range(min_val, breaks[n_bins] + bin_width * 0.01, "x")?;

    // R's hist(): right-closed bins compared with breaks shifted by this fuzz.
    let fuzz = 1e-7 * if n_bins <= 2 { range } else { bin_width };
    for &val in &clean_data {
        let mut bin = (((val - min_val) / bin_width).ceil() as usize)
            .saturating_sub(1)
            .min(n_bins - 1);
        if bin > 0 && val <= breaks[bin] + fuzz {
            bin -= 1;
        } else if bin + 1 < n_bins && val > breaks[bin + 1] + fuzz {
            bin += 1;
        }
        counts[bin] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Histogram");
        let x_label = config.x_label.as_deref().unwrap_or("Value");
        let y_label = config.y_label.as_deref().unwrap_or("Frequency");

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
            resource::checkpoint().map_err(invalid_data)?;
            let (x0, x1) = (breaks[i], breaks[i + 1]);

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x0, 0), (x1, count)],
                    DEFAULT_SERIES_COLOR.mix(0.7).filled(),
                )))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    // Encode as PNG then base64
    let image_base64 = encode_rgb_to_png_base64(&buffer, config.width, config.height)?;

    Ok(HistogramResult {
        image_base64,
        bins: n_bins,
        min: min_val,
        max: max_val,
        mean: mean_val,
        counts,
        breaks,
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
        return Err(VisualizationError::InvalidData(format!(
            "X and Y must have same length: {} vs {}",
            x.len(),
            y.len()
        )));
    }

    if x.is_empty() {
        return Err(VisualizationError::InvalidData(
            "Empty data arrays".to_string(),
        ));
    }

    // Filter out pairs with NaN
    let points: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(xi, yi)| (*xi, *yi))
        .collect();

    if points.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No finite value pairs".to_string(),
        ));
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

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Scatter Plot");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
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

        // Draw points
        chart
            .draw_series(
                points.iter().map(|(xi, yi)| {
                    Circle::new((*xi, *yi), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled())
                }),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
    /// Lower whisker end (`boxplot.stats()$stats[1]`).
    pub min: f64,
    /// Lower hinge (Tukey, `fivenum`).
    pub q1: f64,
    pub median: f64,
    /// Upper hinge (Tukey, `fivenum`).
    pub q3: f64,
    /// Upper whisker end (`boxplot.stats()$stats[5]`).
    pub max: f64,
    /// Values beyond 1.5 times the hinge spread, in data order.
    pub outliers: Vec<f64>,
    /// Non-NaN values of the group (`boxplot.stats()$n`); 0 for a group without
    /// values, whose statistics are NaN.
    pub n: usize,
}

impl BoxPlotStats {
    /// The statistics and outliers a chart can draw.
    fn finite_elements(&self) -> impl Iterator<Item = f64> + '_ {
        [self.min, self.q1, self.median, self.q3, self.max]
            .into_iter()
            .chain(self.outliers.iter().copied())
            .filter(|x| x.is_finite())
    }
}

impl std::fmt::Display for BoxPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Box Plot")?;
        writeln!(f, "========")?;
        for stat in &self.statistics {
            if stat.n == 0 {
                writeln!(f, "{}: no values", stat.label)?;
                continue;
            }
            writeln!(
                f,
                "{}: min={:.2}, Q1={:.2}, med={:.2}, Q3={:.2}, max={:.2}",
                stat.label, stat.min, stat.q1, stat.median, stat.q3, stat.max
            )?;
        }
        writeln!(f)?;
        writeln!(f, "Image (base64): {} bytes", self.image_base64.len())
    }
}

/// Generate a box plot from multiple data groups, with the statistics of
/// `grDevices::boxplot.stats` (coef 1.5). A group without values keeps its slot and label,
/// as in R's `boxplot`; infinite values count as in R and are not drawn.
///
/// # Arguments
/// * `groups` - Vector of (label, data) pairs
/// * `config` - Chart configuration
pub fn box_plot(
    groups: &[(String, Vec<f64>)],
    config: ChartConfig,
) -> Result<BoxPlotResult, VisualizationError> {
    if groups.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No groups provided".to_string(),
        ));
    }

    // Calculate statistics for each group
    let statistics = groups
        .iter()
        .map(|(label, data)| calculate_box_stats(label.clone(), data))
        .collect::<Result<Vec<_>, _>>()?;

    if !statistics
        .iter()
        .any(|s| s.finite_elements().next().is_some())
    {
        return Err(VisualizationError::InvalidData(
            "No valid data in groups".to_string(),
        ));
    }

    let finite = || statistics.iter().flat_map(BoxPlotStats::finite_elements);
    let y_min = finite().fold(f64::INFINITY, f64::min);
    let y_max = finite().fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // x-axis with continuous coordinates (0 to n_groups)
    let n_groups = statistics.len() as f64;
    let x_range = checked_axis_range(-0.5, n_groups - 0.5, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Box Plot");
        let y_label = config.y_label.as_deref().unwrap_or("Value");

        let labels: Vec<String> = statistics.iter().map(|s| s.label.clone()).collect();

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
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

            // Only finite elements are drawn: an empty group draws nothing, and an
            // infinite hinge or whisker end leaves out the parts that reach it.
            let drawn = |a: f64, b: f64| a.is_finite() && b.is_finite();

            if drawn(stat.q1, stat.q3) {
                // Draw box (Q1 to Q3)
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [(x_left, stat.q1), (x_right, stat.q3)],
                        DEFAULT_SERIES_COLOR.mix(0.5).filled(),
                    )))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw box border
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [(x_left, stat.q1), (x_right, stat.q3)],
                        BLACK.stroke_width(1),
                    )))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }

            // Draw median line
            if stat.median.is_finite() {
                chart
                    .draw_series(LineSeries::new(
                        vec![(x_left, stat.median), (x_right, stat.median)],
                        BLACK.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }

            // Draw whiskers: the vertical line from the hinge, then the cap
            for (whisker, hinge) in [(stat.min, stat.q1), (stat.max, stat.q3)] {
                if drawn(whisker, hinge) {
                    chart
                        .draw_series(LineSeries::new(
                            vec![(x_center, whisker), (x_center, hinge)],
                            BLACK.stroke_width(1),
                        ))
                        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                    chart
                        .draw_series(LineSeries::new(
                            vec![(x_left + 0.1, whisker), (x_right - 0.1, whisker)],
                            BLACK.stroke_width(1),
                        ))
                        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
                }
            }

            // Draw outliers
            chart
                .draw_series(
                    stat.outliers
                        .iter()
                        .filter(|outlier| outlier.is_finite())
                        .map(|&outlier| {
                            Circle::new((x_center, outlier), 3, OUTLIER_COLOR.filled())
                        }),
                )
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
        return Err(VisualizationError::InvalidData(
            "No series provided".to_string(),
        ));
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
        return Err(VisualizationError::InvalidData(
            "No valid data points".to_string(),
        ));
    }

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.05;

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    // Colors for multiple series (brand palette)
    let colors = CHART_PALETTE;

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Line Chart");
        let x_label = config.x_label.as_deref().unwrap_or("X");
        let y_label = config.y_label.as_deref().unwrap_or("Y");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
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

        // Draw each series
        for (idx, (label, x_vals, y_vals)) in series.iter().enumerate() {
            let color = colors[idx % colors.len()];
            let points: Vec<(f64, f64)> = x_vals
                .iter()
                .zip(y_vals.iter())
                .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
                .map(|(xi, yi)| (*xi, *yi))
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?
                .label(label)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
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

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
        return Err(VisualizationError::InvalidData(
            "Empty data arrays".to_string(),
        ));
    }

    let n = time.len();
    if estimates.len() != n || ci_lower.len() != n || ci_upper.len() != n {
        return Err(VisualizationError::InvalidData(
            "All arrays must have the same length".to_string(),
        ));
    }

    // Filter out invalid values and collect valid points
    let mut valid_points: Vec<(f64, f64, f64, f64)> = Vec::new();
    for i in 0..n {
        if time[i].is_finite()
            && estimates[i].is_finite()
            && ci_lower[i].is_finite()
            && ci_upper[i].is_finite()
        {
            valid_points.push((time[i], estimates[i], ci_lower[i], ci_upper[i]));
        }
    }

    if valid_points.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No valid data points".to_string(),
        ));
    }

    // Sort by time
    valid_points.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Calculate bounds
    let x_min = valid_points.first().unwrap().0;
    let x_max = valid_points.last().unwrap().0;
    let y_min = valid_points
        .iter()
        .map(|p| p.2)
        .fold(f64::INFINITY, f64::min);
    let y_max = valid_points
        .iter()
        .map(|p| p.3)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(1.0) * 0.1;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;
    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    // Detect treatment period (typically 0)
    let treatment_period = if valid_points.iter().any(|p| (p.0 - 0.0).abs() < 0.01) {
        Some(0.0)
    } else {
        None
    };

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Event Study");
        let x_label = config
            .x_label
            .as_deref()
            .unwrap_or("Time Relative to Treatment");
        let y_label = config.y_label.as_deref().unwrap_or("Treatment Effect");

        let mut chart = ChartBuilder::on(&root)
            .caption(title, ("sans-serif", config.title_font_size))
            .margin(10)
            .x_label_area_size(40)
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

        // Draw horizontal zero line (dashed)
        chart
            .draw_series(LineSeries::new(
                vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
                BLACK.stroke_width(1),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw vertical treatment line at t=0 if present
        if treatment_period.is_some() {
            chart
                .draw_series(LineSeries::new(
                    vec![(0.0, y_min - y_pad), (0.0, y_max + y_pad)],
                    RGBColor(128, 128, 128).stroke_width(1),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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

        chart
            .draw_series(std::iter::once(Polygon::new(
                ci_polygon,
                DEFAULT_SERIES_COLOR.mix(0.2).filled(),
            )))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point estimates as line
        let estimate_points: Vec<(f64, f64)> = valid_points
            .iter()
            .map(|&(t, est, _, _)| (t, est))
            .collect();

        chart
            .draw_series(LineSeries::new(
                estimate_points.clone(),
                DEFAULT_SERIES_COLOR.stroke_width(2),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point markers
        chart
            .draw_series(
                estimate_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.filled())),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
        return Err(VisualizationError::InvalidData(
            "Empty data arrays".to_string(),
        ));
    }

    let n = names.len();
    if estimates.len() != n || ci_lower.len() != n || ci_upper.len() != n {
        return Err(VisualizationError::InvalidData(
            "All arrays must have the same length".to_string(),
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
        return Err(VisualizationError::InvalidData(
            "No valid coefficients".to_string(),
        ));
    }

    // Calculate bounds
    let x_min = valid_coefs
        .iter()
        .map(|c| c.2)
        .fold(f64::INFINITY, f64::min);
    let x_max = valid_coefs
        .iter()
        .map(|c| c.3)
        .fold(f64::NEG_INFINITY, f64::max);
    let x_pad = (x_max - x_min).max(0.1) * 0.1;

    // Ensure zero is visible
    let x_range_min = x_min.min(0.0) - x_pad;
    let x_range_max = x_max.max(0.0) + x_pad;

    // The estimates run along x (horizontal layout) or y (vertical layout),
    // the coefficients along the other axis.
    let n_coefs = valid_coefs.len() as f64;
    let (value_axis, coef_axis) = if horizontal { ("x", "y") } else { ("y", "x") };
    let value_range = checked_axis_range(x_range_min, x_range_max, value_axis)?;
    let coef_range = checked_axis_range(-0.5, n_coefs - 0.5, coef_axis)?;

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let title = config.title.as_deref().unwrap_or("Coefficient Plot");
        let x_label = config.x_label.as_deref().unwrap_or("Estimate");

        if horizontal {
            // Horizontal layout: coefficients on Y-axis, values on X-axis
            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", config.title_font_size))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(120) // More space for variable names
                .build_cartesian_2d(value_range, coef_range)
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            chart
                .configure_mesh()
                .x_desc(x_label)
                .y_label_formatter(&|y| {
                    let idx = y.round() as usize;
                    valid_coefs
                        .get(idx)
                        .map(|c| c.0.clone())
                        .unwrap_or_default()
                })
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw vertical zero line
            chart
                .draw_series(LineSeries::new(
                    vec![(0.0, -0.5), (0.0, n_coefs - 0.5)],
                    RGBColor(128, 128, 128).stroke_width(1),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw each coefficient with error bar
            for (i, (_name, est, lower, upper)) in valid_coefs.iter().enumerate() {
                let y_pos = i as f64;

                // Draw error bar (horizontal line)
                chart
                    .draw_series(LineSeries::new(
                        vec![(*lower, y_pos), (*upper, y_pos)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw caps on error bar
                chart
                    .draw_series(LineSeries::new(
                        vec![(*lower, y_pos - 0.1), (*lower, y_pos + 0.1)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                chart
                    .draw_series(LineSeries::new(
                        vec![(*upper, y_pos - 0.1), (*upper, y_pos + 0.1)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw point estimate
                chart
                    .draw_series(std::iter::once(Circle::new(
                        (*est, y_pos),
                        5,
                        DEFAULT_SERIES_COLOR.filled(),
                    )))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        } else {
            // Vertical layout: coefficients on X-axis, values on Y-axis
            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", config.title_font_size))
                .margin(10)
                .x_label_area_size(60)
                .y_label_area_size(60)
                .build_cartesian_2d(coef_range, value_range)
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            chart
                .configure_mesh()
                .y_desc(x_label)
                .x_label_formatter(&|x| {
                    let idx = x.round() as usize;
                    valid_coefs
                        .get(idx)
                        .map(|c| c.0.clone())
                        .unwrap_or_default()
                })
                .draw()
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw horizontal zero line
            chart
                .draw_series(LineSeries::new(
                    vec![(-0.5, 0.0), (n_coefs - 0.5, 0.0)],
                    RGBColor(128, 128, 128).stroke_width(1),
                ))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Draw each coefficient with error bar
            for (i, (_name, est, lower, upper)) in valid_coefs.iter().enumerate() {
                let x_pos = i as f64;

                // Draw error bar (vertical line)
                chart
                    .draw_series(LineSeries::new(
                        vec![(x_pos, *lower), (x_pos, *upper)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw caps on error bar
                chart
                    .draw_series(LineSeries::new(
                        vec![(x_pos - 0.1, *lower), (x_pos + 0.1, *lower)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                chart
                    .draw_series(LineSeries::new(
                        vec![(x_pos - 0.1, *upper), (x_pos + 0.1, *upper)],
                        DEFAULT_SERIES_COLOR.stroke_width(2),
                    ))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                // Draw point estimate
                chart
                    .draw_series(std::iter::once(Circle::new(
                        (x_pos, *est),
                        5,
                        DEFAULT_SERIES_COLOR.filled(),
                    )))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
        return Err(VisualizationError::InvalidData(
            "Empty data arrays".to_string(),
        ));
    }

    if horizons.len() != responses.len() {
        return Err(VisualizationError::InvalidData(
            "Horizons and responses must have same length".to_string(),
        ));
    }

    let n = horizons.len();
    let has_ci = ci_lower.is_some() && ci_upper.is_some();

    if has_ci {
        let lower = ci_lower.unwrap();
        let upper = ci_upper.unwrap();
        if lower.len() != n || upper.len() != n {
            return Err(VisualizationError::InvalidData(
                "CI arrays must have same length as horizons".to_string(),
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
            } else {
                None
            };
            let upper = if has_ci {
                let u = ci_upper.unwrap()[i];
                if u.is_finite() { Some(u) } else { None }
            } else {
                None
            };
            valid_points.push((horizons[i], responses[i], lower, upper));
        }
    }

    if valid_points.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No valid data points".to_string(),
        ));
    }

    // Sort by horizon
    valid_points.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Calculate bounds
    let x_min = valid_points.first().unwrap().0;
    let x_max = valid_points.last().unwrap().0;

    let mut y_min = valid_points
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min);
    let mut y_max = valid_points
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    // Include CI bounds if present
    if has_ci {
        for p in &valid_points {
            if let Some(l) = p.2 {
                y_min = y_min.min(l);
            }
            if let Some(u) = p.3 {
                y_max = y_max.max(u);
            }
        }
    }

    let x_pad = (x_max - x_min).max(1.0) * 0.05;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Ensure zero is visible on y-axis
    y_min = y_min.min(0.0);
    y_max = y_max.max(0.0);

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    // Create image buffer
    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

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

        // Draw horizontal zero line
        chart
            .draw_series(LineSeries::new(
                vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
                BLACK.stroke_width(1),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw confidence interval as shaded region if present
        if has_ci {
            let ci_points_with_bounds: Vec<_> = valid_points
                .iter()
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

                chart
                    .draw_series(std::iter::once(Polygon::new(
                        ci_polygon,
                        DEFAULT_SERIES_COLOR.mix(0.2).filled(),
                    )))
                    .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
            }
        }

        // Draw response line
        let response_points: Vec<(f64, f64)> = valid_points.iter().map(|p| (p.0, p.1)).collect();

        chart
            .draw_series(LineSeries::new(
                response_points.clone(),
                DEFAULT_SERIES_COLOR.stroke_width(2),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw point markers
        chart
            .draw_series(
                response_points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.filled())),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
    /// Standardized residuals (R's `rstandard`); NaN at leverage one
    pub standardized_residuals: Vec<f64>,
    /// Cook's distances (R's `cooks.distance`); NaN at leverage one
    pub cooks_distance: Vec<f64>,
    /// Hat values used: the given leverage or the one-predictor formula.
    pub leverage: Vec<f64>,
    /// Number of coefficients p (given, or the rounded sum of the hat values).
    pub n_params: usize,
    /// Residual standard error `sqrt(sum e^2 / (n - p))` (R's `summary.lm` sigma).
    pub sigma: f64,
    /// 0-based input row of each element of the vectors above.
    pub rows: Vec<usize>,
    /// What the caller should know: the leverage fallback, observations not plotted.
    pub warnings: Vec<String>,
}

impl ResultNotes for ResidualDiagnosticsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl std::fmt::Display for ResidualDiagnosticsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Residual Diagnostics")?;
        writeln!(f, "====================")?;
        writeln!(f, "Observations: {}", self.n_observations)?;
        writeln!(f, "Coefficients (p): {}", self.n_params)?;
        writeln!(
            f,
            "Residual standard error: {:.4} on {} degrees of freedom",
            self.sigma,
            self.n_observations - self.n_params
        )?;
        writeln!(f)?;
        writeln!(f, "Plots Generated:")?;
        writeln!(
            f,
            "  - Residuals vs Fitted: {} bytes",
            self.residuals_vs_fitted.len()
        )?;
        writeln!(f, "  - Q-Q Plot: {} bytes", self.qq_plot.len())?;
        writeln!(f, "  - Scale-Location: {} bytes", self.scale_location.len())?;
        writeln!(
            f,
            "  - Residuals vs Leverage: {} bytes",
            self.residuals_vs_leverage.len()
        )?;
        write!(f, "{}", notes_text(self))
    }
}

/// Generate residual diagnostic plots for regression analysis: R's `plot.lm`
/// (`which = c(1:3, 5)`).
///
/// Creates four diagnostic plots:
/// 1. Residuals vs Fitted - checks linearity and homoscedasticity
/// 2. Normal Q-Q - checks normality of residuals (`qnorm(ppoints(n))`)
/// 3. Scale-Location - checks homoscedasticity (√|standardized residuals| vs fitted)
/// 4. Residuals vs Leverage - identifies influential observations
///
/// The standardized residuals are R's `rstandard` and the Cook's distances R's
/// `cooks.distance`, both NaN and not plotted at leverage one. They are exact only with the
/// model's hat values: without `leverage` the one-predictor formula is used, with a warning,
/// which is exact for a model with one regressor and an intercept. Leverage that is not a
/// number in `[0, 1]` at a used row is an error naming the row.
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
    residual_diagnostics_with(fitted, residuals, leverage, None, config)
}

/// As [`residual_diagnostics`], with the model size. `n_params`: R's model rank; `None` takes
/// it from the hat values (their rounded sum) or, without them, from the one-predictor
/// formula.
pub fn residual_diagnostics_with(
    fitted: &[f64],
    residuals: &[f64],
    leverage: Option<&[f64]>,
    n_params: Option<usize>,
    config: ChartConfig,
) -> Result<ResidualDiagnosticsResult, VisualizationError> {
    if fitted.is_empty() || residuals.is_empty() {
        return Err(VisualizationError::InvalidData(
            "Empty data arrays".to_string(),
        ));
    }

    if fitted.len() != residuals.len() {
        return Err(VisualizationError::InvalidData(
            "Fitted and residuals must have same length".to_string(),
        ));
    }

    if leverage.is_some_and(|lev| lev.len() != fitted.len()) {
        return Err(VisualizationError::InvalidData(
            "Leverage must have same length as fitted".to_string(),
        ));
    }

    // Filter out invalid values
    let rows: Vec<usize> = (0..fitted.len())
        .filter(|&i| fitted[i].is_finite() && residuals[i].is_finite())
        .collect();

    if rows.is_empty() {
        return Err(VisualizationError::InvalidData(
            "No valid data points".to_string(),
        ));
    }

    let fitted_valid: Vec<f64> = rows.iter().map(|&i| fitted[i]).collect();
    let residuals_valid: Vec<f64> = rows.iter().map(|&i| residuals[i]).collect();
    let n_valid = rows.len();
    let mut warnings = Vec::new();

    // Leverage and the number of coefficients
    let (leverage_valid, p) = match leverage {
        Some(lev) => {
            let hat = given_leverage(lev, &rows)?;
            let p = match n_params {
                Some(p) => p,
                None => params_from_hat_values(&hat)?,
            };
            (hat, p)
        }
        None => {
            let (hat, p0, note) = one_predictor_leverage(&fitted_valid);
            if let Some(p) = n_params
                && p != p0
            {
                return Err(VisualizationError::InvalidData(format!(
                    "n_params = {p} needs the model's hat values (leverage); without them only a \
                     model with one regressor and an intercept (p = 2) can be diagnosed"
                )));
            }
            warnings.push(note.to_string());
            (hat, p0)
        }
    };

    if p == 0 || p >= n_valid {
        return Err(VisualizationError::InvalidData(format!(
            "n_params is {p}; it must be at least 1 and below the number of observations \
             ({n_valid})"
        )));
    }
    let df = n_valid - p;

    // sigma, scaled so that it cannot overflow
    let largest = residuals_valid.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    if largest == 0.0 {
        return Err(VisualizationError::InvalidData(
            "all residuals are zero (a perfect fit): standardized residuals and Cook's distance \
             are undefined"
                .to_string(),
        ));
    }
    let scaled_rss: f64 = residuals_valid.iter().map(|e| (e / largest).powi(2)).sum();
    let sigma = largest * (scaled_rss / df as f64).sqrt();

    // R's rstandard.lm and cooks.distance.lm; leverage one as lm.influence
    let mut standardized_residuals = Vec::with_capacity(n_valid);
    let mut cooks_distance = Vec::with_capacity(n_valid);
    let mut leverage_one = Vec::new();
    for ((e, h), row) in residuals_valid.iter().zip(&leverage_valid).zip(&rows) {
        if *h >= 1.0 - 10.0 * f64::EPSILON {
            standardized_residuals.push(f64::NAN);
            cooks_distance.push(f64::NAN);
            leverage_one.push(*row);
        } else {
            let r = e / (sigma * (1.0 - h).sqrt());
            standardized_residuals.push(r);
            cooks_distance.push(r * r * h / (p as f64 * (1.0 - h)));
        }
    }
    if !leverage_one.is_empty() {
        const LISTED: usize = 20;
        let count = leverage_one.len();
        let (listed, rows_label) = if count > LISTED {
            (&leverage_one[..LISTED], "first rows")
        } else {
            (&leverage_one[..], "rows")
        };
        warnings.push(format!(
            "observations with leverage one have NaN standardized residuals and Cook's \
             distances and are not plotted ({count} observations, {rows_label} {listed:?})"
        ));
    }

    // Generate the four diagnostic plots: all rows against the fitted values, the rest
    // only where the standardized residual is a number.
    let plotted: Vec<usize> = (0..n_valid)
        .filter(|&i| standardized_residuals[i].is_finite())
        .collect();
    if plotted.is_empty() {
        return Err(VisualizationError::InvalidData(
            "every observation has leverage one, so standardized residuals and Cook's distance \
             are undefined and nothing can be plotted"
                .to_string(),
        ));
    }
    let pick = |v: &[f64]| -> Vec<f64> { plotted.iter().map(|&i| v[i]).collect() };
    let fitted_plotted = pick(&fitted_valid);
    let standardized_plotted = pick(&standardized_residuals);

    let residuals_vs_fitted =
        generate_residuals_vs_fitted(&fitted_valid, &residuals_valid, &config)?;
    let qq_plot = generate_qq_plot(&standardized_plotted, &config)?;
    let scale_location = generate_scale_location(&fitted_plotted, &standardized_plotted, &config)?;
    let residuals_vs_leverage = generate_residuals_vs_leverage(
        &pick(&leverage_valid),
        &standardized_plotted,
        &pick(&cooks_distance),
        p,
        &config,
    )?;

    Ok(ResidualDiagnosticsResult {
        residuals_vs_fitted,
        qq_plot,
        scale_location,
        residuals_vs_leverage,
        n_observations: n_valid,
        standardized_residuals,
        cooks_distance,
        leverage: leverage_valid,
        n_params: p,
        sigma,
        rows,
        warnings,
    })
}

/// The hat values of the used rows, clamped to `[0, 1]`; a value that is not a number in
/// that range (up to rounding) is an error naming the row.
fn given_leverage(lev: &[f64], rows: &[usize]) -> Result<Vec<f64>, VisualizationError> {
    rows.iter()
        .map(|&i| {
            let v = lev[i];
            if v.is_finite() && (-1e-12..=1.0 + 1e-12).contains(&v) {
                Ok(v.clamp(0.0, 1.0))
            } else {
                Err(VisualizationError::InvalidData(format!(
                    "leverage at row {i} is {v}; hat values must be finite numbers between 0 and 1"
                )))
            }
        })
        .collect()
}

/// The number of coefficients of a linear model from its hat values, whose sum is its rank.
fn params_from_hat_values(hat: &[f64]) -> Result<usize, VisualizationError> {
    let sum: f64 = hat.iter().sum();
    let p = sum.round();
    if (sum - p).abs() > 0.05 || p < 1.0 {
        return Err(VisualizationError::InvalidData(format!(
            "the leverage values sum to {sum:.4}, but the hat values of a linear model sum to its \
             number of coefficients; pass the model's hat values (R: hatvalues(model)) or n_params"
        )));
    }
    Ok(p as usize)
}

/// Hat values without the design: `1/n + (f - mean f)^2 / sum (f - mean f)^2` (p = 2), exact
/// for a model with one regressor and an intercept; 1/n (p = 1) for constant fitted values.
/// Also the number of coefficients and the warning that says so.
fn one_predictor_leverage(fitted: &[f64]) -> (Vec<f64>, usize, &'static str) {
    let n = fitted.len() as f64;
    let mean = corrected_mean(fitted.iter().copied());
    let largest = fitted.iter().fold(0.0f64, |m, f| m.max(f.abs()));
    let spread = fitted.iter().fold(0.0f64, |m, f| m.max((f - mean).abs()));
    if !spread.is_finite() || spread <= 4.0 * f64::EPSILON * largest {
        return (
            vec![1.0 / n; fitted.len()],
            1,
            "leverage not supplied and the fitted values are constant: hat values 1/n as for an \
             intercept-only model (p = 1)",
        );
    }
    let squares: Vec<f64> = fitted
        .iter()
        .map(|f| ((f - mean) / spread).powi(2))
        .collect();
    let sum: f64 = squares.iter().sum();
    (
        squares.iter().map(|q| 1.0 / n + q / sum).collect(),
        2,
        "leverage not supplied: hat values computed as for a regression on one predictor (1/n + \
         (f - mean f)^2 / sum (f - mean f)^2, p = 2); exact only for a model with one regressor \
         and an intercept; pass the model's hat values (R: hatvalues(model)) for any other model",
    )
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

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min_adj, y_max_adj, "y")?;

    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Residuals vs Fitted", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Fitted values")
            .y_desc("Residuals")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line
        chart
            .draw_series(LineSeries::new(
                vec![(x_min - x_pad, 0.0), (x_max + x_pad, 0.0)],
                RGBColor(128, 128, 128).stroke_width(1),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        let points: Vec<(f64, f64)> = fitted
            .iter()
            .zip(residuals.iter())
            .map(|(&f, &r)| (f, r))
            .collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled())),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Calculate and draw LOWESS-like smooth (simple moving average for demonstration)
        let smoothed = calculate_smooth(&points, 0.3);
        if smoothed.len() > 1 {
            chart
                .draw_series(LineSeries::new(smoothed, TREND_LINE_COLOR.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// The abscissae of a normal Q-Q plot: R's `qnorm(ppoints(n))`.
fn qq_theoretical_quantiles(n: usize) -> Result<Vec<f64>, VisualizationError> {
    let a = if n <= 10 { 0.375 } else { 0.5 };
    (1..=n)
        .map(|i| {
            let p = (i as f64 - a) / (n as f64 + 1.0 - 2.0 * a);
            crate::traits::normal_quantile(p).map_err(invalid_data)
        })
        .collect()
}

/// Generate Q-Q plot for normality check.
fn generate_qq_plot(
    standardized_residuals: &[f64],
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    // Sort residuals
    let mut sorted_residuals: Vec<f64> = standardized_residuals.to_vec();
    sorted_residuals.sort_by(f64::total_cmp);

    let theoretical = qq_theoretical_quantiles(sorted_residuals.len())?;

    let x_min = theoretical.first().copied().unwrap_or(-3.0);
    let x_max = theoretical.last().copied().unwrap_or(3.0);
    let y_min = sorted_residuals
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let y_max = sorted_residuals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let range = (x_max - x_min).max(y_max - y_min);
    let x_pad = range * 0.1;
    let y_pad = range * 0.1;

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Normal Q-Q", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
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
        chart
            .draw_series(LineSeries::new(
                vec![(line_min, line_min), (line_max, line_max)],
                RGBColor(128, 128, 128).stroke_width(1),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw Q-Q points
        let points: Vec<(f64, f64)> = theoretical
            .iter()
            .zip(sorted_residuals.iter())
            .map(|(&t, &s)| (t, s))
            .collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled())),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
    let sqrt_abs_resid: Vec<f64> = standardized_residuals
        .iter()
        .map(|r| r.abs().sqrt())
        .collect();

    let x_min = fitted.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = fitted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_max = sqrt_abs_resid
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(0.1) * 0.05;
    let y_pad = y_max * 0.1;

    let x_range = checked_axis_range(x_min - x_pad, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(0.0, y_max + y_pad, "y")?;

    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Scale-Location", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Fitted values")
            .y_desc("√|Standardized residuals|")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw points
        let points: Vec<(f64, f64)> = fitted
            .iter()
            .zip(sqrt_abs_resid.iter())
            .map(|(&f, &r)| (f, r))
            .collect();
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 4, DEFAULT_SERIES_COLOR.mix(0.7).filled())),
            )
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw smooth line
        let smoothed = calculate_smooth(&points, 0.3);
        if smoothed.len() > 1 {
            chart
                .draw_series(LineSeries::new(smoothed, TREND_LINE_COLOR.stroke_width(2)))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Generate Residuals vs Leverage plot.
fn generate_residuals_vs_leverage(
    leverage: &[f64],
    standardized_residuals: &[f64],
    cooks_distance: &[f64],
    n_params: usize,
    config: &ChartConfig,
) -> Result<String, VisualizationError> {
    let x_min = 0.0;
    let x_max = leverage.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = standardized_residuals
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let y_max = standardized_residuals
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let x_pad = x_max * 0.1;
    let y_pad = (y_max - y_min).max(0.1) * 0.1;

    // Calculate Cook's distance threshold (typically 0.5 or 1.0)
    let cooks_threshold = 0.5;
    let has_influential = cooks_distance.iter().any(|&c| c > cooks_threshold);

    let x_range = checked_axis_range(x_min, x_max + x_pad, "x")?;
    let y_range = checked_axis_range(y_min - y_pad, y_max + y_pad, "y")?;

    let mut buffer = rgb_buffer(config.width, config.height)?;

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (config.width, config.height))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Residuals vs Leverage", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range)
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        chart
            .configure_mesh()
            .x_desc("Leverage")
            .y_desc("Standardized residuals")
            .draw()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw horizontal zero line
        chart
            .draw_series(LineSeries::new(
                vec![(x_min, 0.0), (x_max + x_pad, 0.0)],
                RGBColor(128, 128, 128).stroke_width(1),
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

        // Draw Cook's distance contour lines (approximate)
        if has_influential {
            // D = r^2 * h / (p * (1-h)) => r = sqrt(D * p * (1-h) / h)
            let p = n_params as f64;
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
                    chart
                        .draw_series(LineSeries::new(
                            contour_points.clone(),
                            OUTLIER_COLOR.stroke_width(1),
                        ))
                        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

                    // Draw negative contour
                    let neg_contour: Vec<(f64, f64)> =
                        contour_points.iter().map(|&(h, r)| (h, -r)).collect();
                    chart
                        .draw_series(LineSeries::new(neg_contour, OUTLIER_COLOR.stroke_width(1)))
                        .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
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
            chart
                .draw_series(std::iter::once(Circle::new(
                    (leverage[i], standardized_residuals[i]),
                    4,
                    color.mix(0.7).filled(),
                )))
                .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
        }

        root.present()
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;
    }

    encode_rgb_to_png_base64(&buffer, config.width, config.height)
}

/// Calculate a simple smoothing line (local regression approximation).
fn calculate_smooth(points: &[(f64, f64)], span: f64) -> Vec<(f64, f64)> {
    if points.len() < 3 {
        return Vec::new();
    }

    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n = sorted_points.len();
    let window = ((n as f64 * span) as usize).max(3).min(n);
    let half_window = window / 2;

    let mut smoothed = Vec::new();

    for i in 0..n {
        let start = i.saturating_sub(half_window);
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

// Helper functions

/// The mean with R's correction: `m = sum / n; m += sum(x - m) / n`. The sum runs over
/// `x / n` when the plain sum overflows.
fn corrected_mean(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = x.clone().count() as f64;
    let mean = x.clone().sum::<f64>() / n;
    let mean = if mean.is_finite() {
        mean
    } else {
        x.clone().map(|v| v / n).sum::<f64>()
    };
    mean + x.map(|v| v - mean).sum::<f64>() / n
}

/// Pearson r of finite pairs as R's `cor`: means with R's correction, centred deviations
/// scaled by their largest magnitude (no overflow or underflow at any scale), and None
/// (R: NA) for fewer than two pairs or a zero spread, judged relative to the data's
/// magnitude (`max |x - m| <= 4 eps max |x|`, where the centred values are rounding noise).
fn calculate_correlation(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let mx = corrected_mean(points.iter().map(|p| p.0));
    let my = corrected_mean(points.iter().map(|p| p.1));
    let spread = |centre: f64, value: fn(&(f64, f64)) -> f64| -> Option<f64> {
        let (mut largest, mut deviation) = (0.0f64, 0.0f64);
        for p in points {
            largest = largest.max(value(p).abs());
            deviation = deviation.max((value(p) - centre).abs());
        }
        (deviation.is_finite() && deviation > 4.0 * f64::EPSILON * largest).then_some(deviation)
    };
    let sx = spread(mx, |p| p.0)?;
    let sy = spread(my, |p| p.1)?;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in points {
        let dx = (p.0 - mx) / sx;
        let dy = (p.1 - my) / sy;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    Some((sxy / (sxx.sqrt() * syy.sqrt())).clamp(-1.0, 1.0))
}

fn calculate_box_stats(label: String, data: &[f64]) -> Result<BoxPlotStats, VisualizationError> {
    // grDevices::boxplot.stats: NaN is dropped; +-Inf count in n and the hinges and are
    // outliers, as in R; infinite elements are not drawn.
    if data.iter().all(|x| x.is_nan()) {
        return Ok(BoxPlotStats {
            label,
            min: f64::NAN,
            q1: f64::NAN,
            median: f64::NAN,
            q3: f64::NAN,
            max: f64::NAN,
            outliers: Vec::new(),
            n: 0,
        });
    }
    let b = crate::stats::robust::boxplot_stats(data, 1.5).map_err(invalid_data)?;
    let [min, q1, median, q3, max] = b.stats;
    Ok(BoxPlotStats {
        label,
        min,
        q1,
        median,
        q3,
        max,
        outliers: b.out,
        n: b.n,
    })
}

/// Map a core resource error (allocation, size arithmetic, cancellation) to
/// this module's invalid-data error, keeping its text.
fn invalid_data(e: crate::errors::EconError) -> VisualizationError {
    VisualizationError::InvalidData(e.to_string())
}

/// The widest f64 axis a chart may have.
///
/// plotters 0.3.7's f64 key-point search (`coord/ranged1d/types/numeric.rs:157`)
/// never terminates when the axis width is infinite, which finite data near
/// `f64::MAX` produce through their spread or the padding, and with a single
/// key point (a very small axis) its scale also overflows once the width
/// reaches 1e308. Below this width the search terminates for any number of
/// key points.
const MAX_AXIS_WIDTH: f64 = 1e308;

/// The axis range `lo..hi` if plotters can lay it out: both ends finite,
/// `hi >= lo` and a width below [`MAX_AXIS_WIDTH`]. A zero-width range is
/// accepted (plotters returns early on it, so constant data still render).
/// Anything else is `InvalidData` naming the axis, so that no chart reaches
/// `build_cartesian_2d` with a range its key-point search cannot handle.
fn checked_axis_range(
    lo: f64,
    hi: f64,
    axis: &str,
) -> Result<std::ops::Range<f64>, VisualizationError> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(VisualizationError::InvalidData(format!(
            "the {axis} axis range is not finite; the data contain values too large to plot"
        )));
    }
    if hi < lo {
        return Err(VisualizationError::InvalidData(format!(
            "the {axis} axis range is empty"
        )));
    }
    // Both ends are finite and hi >= lo, so the width is finite and >= 0, or +inf.
    if hi - lo >= MAX_AXIS_WIDTH {
        return Err(VisualizationError::InvalidData(format!(
            "the {axis} axis would span from {lo:e} to {hi:e}, which is too wide to draw; \
             rescale the data (for example divide by a power of ten)"
        )));
    }
    Ok(lo..hi)
}

/// A zeroed RGB buffer for a `width` x `height` image, sized with checked
/// arithmetic and allocated fallibly, so an impossible size is an error
/// instead of a wrapped `u32` product, a panic or a process abort.
pub(super) fn rgb_buffer(width: u32, height: u32) -> Result<Vec<u8>, VisualizationError> {
    if width == 0 || height == 0 {
        return Err(VisualizationError::InvalidData(format!(
            "image size {width} x {height}: width and height must be at least 1"
        )));
    }
    let len = resource::checked_len(&[width as usize, height as usize, 3], "image buffer")
        .map_err(invalid_data)?;
    resource::try_vec(len, 0u8, "image buffer").map_err(invalid_data)
}

/// Encode an RGB buffer of `width` x `height` pixels as a base64 PNG, without
/// copying the buffer.
pub(super) fn encode_rgb_to_png_base64(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<String, VisualizationError> {
    use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};
    use std::io::Cursor;

    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3));
    if expected != Some(rgb.len()) {
        return Err(VisualizationError::EncodingError(
            "Failed to create image buffer".to_string(),
        ));
    }

    // Encode to PNG
    let mut png_data = Vec::new();
    let encoder = PngEncoder::new(Cursor::new(&mut png_data));
    encoder
        .write_image(rgb, width, height, ColorType::Rgb8)
        .map_err(|e: image::ImageError| VisualizationError::EncodingError(e.to_string()))?;

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
    validate_linkage(linkage_matrix, n_samples)?;

    let max_distance = linkage_matrix
        .iter()
        .map(|(_, _, d, _)| *d)
        .fold(0.0_f64, |a, b| a.max(b));

    let width = config.width;
    let height = config.height;
    let title = config.title.unwrap_or_else(|| "Dendrogram".to_string());

    // Create drawing area
    let mut buffer = rgb_buffer(width, height)?;
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width, height)).into_drawing_area();
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
            resource::checkpoint().map_err(invalid_data)?;
            let (x1, y1) = cluster_pos[*c1];
            let (x2, y2) = cluster_pos[*c2];
            let new_cluster_idx = n_samples + i;
            let (_, new_y) = cluster_pos[new_cluster_idx];

            // Draw U-shaped connector:
            // Vertical line from c1 to merge height
            root.draw(&PathElement::new(
                vec![(x_scale(x1), y_scale(y1)), (x_scale(x1), y_scale(new_y))],
                DEFAULT_SERIES_COLOR,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Vertical line from c2 to merge height
            root.draw(&PathElement::new(
                vec![(x_scale(x2), y_scale(y2)), (x_scale(x2), y_scale(new_y))],
                DEFAULT_SERIES_COLOR,
            ))
            .map_err(|e| VisualizationError::PlottingError(e.to_string()))?;

            // Horizontal line connecting c1 and c2 at merge height
            root.draw(&PathElement::new(
                vec![(x_scale(x1), y_scale(*dist)), (x_scale(x2), y_scale(*dist))],
                DEFAULT_SERIES_COLOR,
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
            BLACK,
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
                BLACK,
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

/// Check a linkage matrix before it is used to index the cluster table.
///
/// Merge `i` creates cluster `n_samples + i`, so it may only refer to the
/// samples and the clusters created by earlier merges; it must join two
/// different clusters, neither of which was merged before, at a finite,
/// non-negative height.
fn validate_linkage(
    linkage: &[(usize, usize, f64, usize)],
    n_samples: usize,
) -> Result<(), VisualizationError> {
    let n_clusters = n_samples
        .checked_add(linkage.len())
        .ok_or_else(|| VisualizationError::InvalidData("linkage matrix is too large".into()))?;
    let mut merged = vec![false; n_clusters];
    for (i, &(c1, c2, height, _)) in linkage.iter().enumerate() {
        let existing = n_samples + i;
        for c in [c1, c2] {
            if c >= existing {
                return Err(VisualizationError::InvalidData(format!(
                    "linkage row {i} refers to cluster {c}, which does not exist yet"
                )));
            }
        }
        if c1 == c2 {
            return Err(VisualizationError::InvalidData(format!(
                "linkage row {i} merges cluster {c1} with itself"
            )));
        }
        for c in [c1, c2] {
            if merged[c] {
                return Err(VisualizationError::InvalidData(format!(
                    "linkage row {i} merges cluster {c}, which an earlier row already merged"
                )));
            }
            merged[c] = true;
        }
        if !height.is_finite() || height < 0.0 {
            return Err(VisualizationError::InvalidData(format!(
                "linkage row {i} has merge height {height}; heights must be finite and \
                 non-negative"
            )));
        }
    }
    Ok(())
}

/// Compute the optimal leaf ordering for dendrogram visualization.
/// Returns the order in which leaves should be placed (left to right).
///
/// The leaves of the last cluster, left subtree before right subtree, found
/// by one depth-first traversal of the merge tree (O(n) time and memory).
/// The linkage must have passed [`validate_linkage`].
fn compute_leaf_order(
    linkage_matrix: &[(usize, usize, f64, usize)],
    n_samples: usize,
) -> Vec<usize> {
    let n_clusters = n_samples + linkage_matrix.len();
    let Some(root) = n_clusters.checked_sub(1) else {
        return Vec::new();
    };

    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if id < n_samples {
            order.push(id);
        } else {
            // Merge `id - n_samples` created cluster `id`; visit its left
            // child first, so push the right child first.
            let (left, right, _, _) = linkage_matrix[id - n_samples];
            stack.push(right);
            stack.push(left);
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wavy(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 10.0 + i as f64)
            .collect()
    }

    #[test]
    fn histogram_rejects_zero_and_impossible_bin_counts() {
        let data = wavy(50);
        assert!(matches!(histogram(&data, Some(0), ChartConfig::default()),
            Err(VisualizationError::InvalidData(m)) if m.contains("bins")));
        assert!(matches!(
            histogram(&data, Some(usize::MAX), ChartConfig::default()),
            Err(VisualizationError::InvalidData(_))
        ));
        let small = ChartConfig {
            width: 100,
            height: 100,
            ..ChartConfig::default()
        };
        let r = crate::resource::with_alloc_ceiling(Some(1 << 20), || {
            histogram(&data, Some(300_000), small)
        });
        assert!(matches!(r, Err(VisualizationError::InvalidData(m)) if m.contains("bins")));
    }

    #[test]
    fn image_buffers_use_checked_sizes() {
        assert!(rgb_buffer(0, 10).is_err());
        assert!(rgb_buffer(10, 0).is_err());
        assert_eq!(rgb_buffer(10, 20).unwrap().len(), 600);
        assert!(rgb_buffer(u32::MAX, u32::MAX).is_err()); // product overflows usize
        let link = [(0usize, 1usize, 1.0f64, 2usize)];
        let huge = ChartConfig {
            width: u32::MAX,
            height: u32::MAX,
            ..ChartConfig::default()
        };
        assert!(matches!(
            dendrogram(&link, None, huge),
            Err(VisualizationError::InvalidData(_))
        ));
        let big = ChartConfig {
            width: 65_536,
            height: 65_536,
            ..ChartConfig::default()
        };
        let r =
            crate::resource::with_alloc_ceiling(Some(256 << 20), || dendrogram(&link, None, big));
        assert!(matches!(r, Err(VisualizationError::InvalidData(_))));
    }

    #[test]
    fn axis_range_refuses_non_finite_and_overflowing_widths() {
        assert!(checked_axis_range(-1e308, f64::INFINITY, "x").is_err());
        assert!(checked_axis_range(f64::NEG_INFINITY, f64::INFINITY, "y").is_err());
        assert!(checked_axis_range(f64::NAN, 1.0, "x").is_err());
        assert!(checked_axis_range(-1.1e308, 1.1e308, "x").is_err()); // finite ends, infinite width
        assert!(checked_axis_range(0.0, 1.5e308, "x").is_err()); // finite width >= 1e308
        assert!(checked_axis_range(2.0, 1.0, "x").is_err()); // reversed
        assert!(matches!(checked_axis_range(-1e308, 1e308, "x"),
            Err(VisualizationError::InvalidData(m)) if m.contains("x axis")));
        assert_eq!(checked_axis_range(0.0, 1.0, "x").unwrap(), 0.0..1.0);
        assert_eq!(checked_axis_range(5.0, 5.0, "y").unwrap(), 5.0..5.0);
        assert_eq!(
            checked_axis_range(-4.9e307, 4.9e307, "y").unwrap(),
            -4.9e307..4.9e307
        );
    }

    #[test]
    fn charts_refuse_ranges_that_overflow() {
        let cfg = ChartConfig::default;
        let v = [-1e308, 0.0, 1e308];
        let bad = |r: Result<(), VisualizationError>| {
            matches!(r, Err(VisualizationError::InvalidData(_)))
        };
        assert!(bad(histogram(&v, Some(3), cfg()).map(|_| ())));
        assert!(histogram(&[-1e300, 0.0, 1e300], Some(3), cfg()).is_ok()); // control
        assert!(bad(scatter_plot(&v, &[1.0, 2.0, 3.0], cfg()).map(|_| ())));
        // padding overflow
        assert!(bad(scatter_plot(
            &[0.0, 1.0, 1.75e308],
            &[1.0, 2.0, 3.0],
            cfg()
        )
        .map(|_| ())));
        assert!(bad(line_chart(
            &[("s".into(), vec![0.0, 1.0, 2.0], v.to_vec())],
            cfg()
        )
        .map(|_| ())));
        assert!(bad(box_plot(&[("g".into(), v.to_vec())], cfg()).map(|_| ())));
        assert!(bad(event_study_plot(
            &[-1.0, 0.0, 1.0],
            &[0.0; 3],
            &[-1e308, 0.0, 0.0],
            &[0.0, 0.0, 1e308],
            cfg()
        )
        .map(|_| ())));
        let names = ["a".to_string(), "b".to_string()];
        for horizontal in [true, false] {
            assert!(bad(coefficient_plot(
                &names,
                &[0.0, 0.0],
                &[-1e308, 0.0],
                &[0.0, 1e308],
                cfg(),
                horizontal
            )
            .map(|_| ())));
        }
        assert!(bad(irf_plot(
            &[0.0, 1.0, 2.0],
            &v,
            None,
            None,
            None,
            None,
            cfg()
        )
        .map(|_| ())));
        assert!(bad(residual_diagnostics(
            &[-1e308, 0.0, 1e308, 1.0],
            &[0.1, -0.2, 0.3, -0.1],
            None,
            cfg()
        )
        .map(|_| ())));
    }

    #[test]
    fn dendrogram_rejects_malformed_linkage() {
        let cfg = || ChartConfig {
            width: 200,
            height: 150,
            ..ChartConfig::default()
        };
        for bad in [
            vec![(0usize, 7usize, 1.0f64, 2usize)], // cluster 7 does not exist yet
            vec![(0, 1, 1.0, 2), (0, 2, 2.0, 2)],   // cluster 0 merged twice
            vec![(1, 1, 1.0, 2)],                   // self-merge
            vec![(0, 1, f64::NAN, 2)],              // non-finite height
            vec![(0, 1, -1.0, 2)],                  // negative height
        ] {
            assert!(
                matches!(
                    dendrogram(&bad, None, cfg()),
                    Err(VisualizationError::InvalidData(_))
                ),
                "{bad:?}"
            );
        }
    }

    /// The pre-fix leaf order (one cloned leaf list per merge), kept as the
    /// reference.
    fn leaf_order_reference(linkage: &[(usize, usize, f64, usize)], n: usize) -> Vec<usize> {
        let mut cluster_leaves: Vec<Vec<usize>> = Vec::with_capacity(n + linkage.len());
        for i in 0..n {
            cluster_leaves.push(vec![i]);
        }
        for (c1, c2, _, _) in linkage.iter() {
            let mut merged = cluster_leaves[*c1].clone();
            merged.extend(cluster_leaves[*c2].clone());
            cluster_leaves.push(merged);
        }
        cluster_leaves
            .last()
            .cloned()
            .unwrap_or_else(|| (0..n).collect())
    }

    #[test]
    fn leaf_order_matches_reference_on_random_trees() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let n = 300;
        let (mut active, mut next, mut linkage) = ((0..n).collect::<Vec<usize>>(), n, Vec::new());
        while active.len() > 1 {
            let a = active.swap_remove(rng.gen_range(0..active.len()));
            let b = active.swap_remove(rng.gen_range(0..active.len()));
            linkage.push((a, b, linkage.len() as f64 + rng.gen_range(0.0..0.5), 2));
            active.push(next);
            next += 1;
        }
        assert_eq!(
            compute_leaf_order(&linkage, n),
            leaf_order_reference(&linkage, n)
        );
        // A forest: the order of the last cluster's leaves.
        let partial = &linkage[..n / 2];
        assert_eq!(
            compute_leaf_order(partial, n),
            leaf_order_reference(partial, n)
        );
    }

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
        let residuals: Vec<f64> = (0..50)
            .map(|i| {
                // Deterministic "noise" for reproducibility
                ((i * 7) % 11) as f64 - 5.0
            })
            .collect();

        let result =
            residual_diagnostics(&fitted, &residuals, None, ChartConfig::default()).unwrap();

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
            (0, 1, 1.0, 2), // Merge samples 0 and 1 -> cluster 4
            (2, 3, 1.5, 2), // Merge samples 2 and 3 -> cluster 5
            (4, 5, 3.0, 4), // Merge clusters 4 and 5 -> cluster 6
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

    fn small() -> ChartConfig {
        ChartConfig {
            width: 240,
            height: 180,
            ..ChartConfig::default()
        }
    }

    #[test]
    fn test_validate_scatter_correlation_large_offset() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("viz_scatter_cor");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let r = |x: &str, y: &str| scatter_plot(&col(x), &col(y), small()).unwrap().correlation;
        let n = case.scalar("n") as usize;
        Check::new(&case)
            .scalar("cor_x0", r("x0", "y").unwrap_or(f64::NAN), Tol::stat(n))
            .scalar(
                "cor_epoch",
                r("x_epoch", "y").unwrap_or(f64::NAN),
                Tol::stat(n),
            )
            .scalar("cor_big", r("x_big", "y").unwrap_or(f64::NAN), Tol::stat(n))
            .finish();
        // R: NA with "the standard deviation is zero"
        assert!(case.scalar("cor_const").is_nan());
        assert_eq!(r("x0", "y_const"), None);
    }

    #[test]
    fn test_validate_scatter_correlation_tiny_scale() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("viz_scatter_cor");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let r = scatter_plot(&col("x_tiny"), &col("y_tiny"), small())
            .unwrap()
            .correlation;
        Check::new(&case)
            .scalar(
                "cor_tiny",
                r.unwrap_or(f64::NAN),
                Tol::stat(case.scalar("n") as usize),
            )
            .finish();
    }

    #[test]
    fn test_validate_boxplot_stats_chart() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("viz_boxplot_groups");
        let labels = ds.df().column("group").unwrap().str().unwrap().clone();
        let values = crate::data::columns::numeric_options(ds.df(), "value").unwrap();
        let order = ["data62", "n4", "n12", "n7", "norm37", "const3", "allna"];
        let mut groups: Vec<(String, Vec<f64>)> = order
            .iter()
            .map(|g| {
                let v = labels
                    .iter()
                    .zip(&values)
                    .filter(|(l, _)| *l == Some(*g))
                    .map(|(_, v)| v.unwrap_or(f64::NAN))
                    .collect();
                (g.to_string(), v)
            })
            .collect();
        // The case JSON cannot hold Inf, so the outlier is asserted below.
        groups.push((
            "inf6".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, f64::INFINITY],
        ));
        let r = box_plot(&groups, small()).unwrap();
        // the all-missing group keeps its slot, as in R
        assert_eq!(r.statistics.len(), order.len() + 1);
        let mut check = Check::new(&case);
        for (g, s) in order.iter().zip(&r.statistics) {
            assert_eq!(s.label, *g);
            check
                .scalar(&format!("{g}_n"), s.n as f64, Tol::exact())
                .scalar(&format!("{g}_n_out"), s.outliers.len() as f64, Tol::exact());
            if s.n > 0 {
                check.vector(
                    &format!("{g}_stats"),
                    &[s.min, s.q1, s.median, s.q3, s.max],
                    Tol::stat(s.n),
                );
            }
            if !s.outliers.is_empty() {
                check.vector(&format!("{g}_out"), &s.outliers, Tol::stat(s.n));
            }
        }
        let inf = &r.statistics[7];
        check
            .scalar("inf6_n", inf.n as f64, Tol::exact())
            .scalar("inf6_n_out", inf.outliers.len() as f64, Tol::exact())
            .vector(
                "inf6_stats",
                &[inf.min, inf.q1, inf.median, inf.q3, inf.max],
                Tol::stat(inf.n),
            );
        check.finish();
        assert_eq!(inf.outliers, [f64::INFINITY]);
        let empty = &r.statistics[6];
        assert!(
            empty.n == 0
                && empty.median.is_nan()
                && empty.min.is_nan()
                && empty.outliers.is_empty()
        );
        assert!(format!("{r}").contains("allna: no values"));
    }

    #[test]
    fn test_validate_histogram_sturges_and_counts() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("viz_histogram_bins");
        for (n, k) in case
            .vector("sturges_n")
            .iter()
            .zip(case.vector("sturges_k"))
        {
            let h = histogram(&wavy(*n as usize), None, small()).unwrap();
            assert_eq!(h.bins, k as usize, "n = {n}");
        }
        let labels = ds.df().column("sample").unwrap().str().unwrap().clone();
        let values = crate::data::columns::numeric_options(ds.df(), "value").unwrap();
        let mut check = Check::new(&case);
        for s in ["norm100", "small50", "grid13"] {
            let x: Vec<f64> = labels
                .iter()
                .zip(&values)
                .filter(|(l, _)| *l == Some(s))
                .map(|(_, v)| v.unwrap())
                .collect();
            let h = histogram(&x, None, small()).unwrap();
            let counts: Vec<f64> = h.counts.iter().map(|&c| f64::from(c)).collect();
            check
                .scalar(&format!("{s}_k"), h.bins as f64, Tol::exact())
                .vector(&format!("{s}_breaks"), &h.breaks, Tol::stat(x.len()))
                .vector(&format!("{s}_counts"), &counts, Tol::exact());
        }
        check.finish();
        // Constant data: Sturges bins of width 1 from the value (p2a's rule; R's hist() uses
        // pretty() here).
        let h = histogram(&[5.0; 10], None, small()).unwrap();
        assert_eq!(
            (h.bins, h.counts.clone(), h.breaks.clone()),
            (5, vec![10, 0, 0, 0, 0], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        );
    }

    #[test]
    fn test_validate_cooks_distance_rstandard() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("viz_residual_diag");
        let col = |c: &str| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0;
        let n = case.scalar("n") as usize;
        // three regressors, R's hat values given: p is their (rounded) sum, 4
        let m3 = residual_diagnostics(&col("fitted3"), &col("resid3"), Some(&col("hat3")), small())
            .unwrap();
        assert_eq!(m3.n_params, case.scalar("p3") as usize);
        assert!(m3.warnings.is_empty(), "{:?}", m3.warnings);
        assert_eq!(m3.rows, (0..n).collect::<Vec<_>>());
        let m3p = residual_diagnostics_with(
            &col("fitted3"),
            &col("resid3"),
            Some(&col("hat3")),
            Some(4),
            small(),
        )
        .unwrap();
        assert_eq!(
            (&m3p.standardized_residuals, &m3p.cooks_distance),
            (&m3.standardized_residuals, &m3.cooks_distance)
        );
        // one regressor, no leverage: the one-predictor hat values are exact here
        let m1 = residual_diagnostics(&col("fitted1"), &col("resid1"), None, small()).unwrap();
        assert_eq!(m1.n_params, case.scalar("p1") as usize);
        assert!(
            m1.warnings.iter().any(|w| w.contains("one predictor")),
            "{:?}",
            m1.warnings
        );
        for (a, b) in m1.leverage.iter().zip(col("hat1")) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
        // observation 0 has leverage one: NaN as R, not plotted, warned
        let ml = residual_diagnostics(
            &col("fitted_l1"),
            &col("resid_l1"),
            Some(&col("hat_l1")),
            small(),
        )
        .unwrap();
        assert_eq!(ml.n_params, case.scalar("p_l1") as usize);
        assert!(
            ml.warnings.iter().any(|w| w.contains("leverage one")),
            "{:?}",
            ml.warnings
        );
        assert!(format!("{ml}").contains("Warning: observations with leverage one"));
        Check::new(&case)
            .scalar("sigma3", m3.sigma, Tol::stat(n))
            .vector("rstandard3", &m3.standardized_residuals, Tol::stat(n))
            .vector("cooks3", &m3.cooks_distance, Tol::stat(n))
            .scalar("sigma1", m1.sigma, Tol::stat(n))
            .vector("rstandard1", &m1.standardized_residuals, Tol::stat(n))
            .vector("cooks1", &m1.cooks_distance, Tol::stat(n))
            .scalar("sigma_l1", ml.sigma, Tol::stat(n))
            .vector("rstandard_l1", &ml.standardized_residuals, Tol::stat(n))
            .vector("cooks_l1", &ml.cooks_distance, Tol::stat(n))
            .finish();
        // R flags nothing above 0.5 for the three-regressor model, and only observation 16
        // for mpg ~ wt
        assert_eq!(m3.cooks_distance.iter().filter(|d| **d > 0.5).count(), 0);
        let flagged: Vec<usize> = m1
            .cooks_distance
            .iter()
            .enumerate()
            .filter(|(_, d)| **d > 0.5)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(flagged, vec![16]);
    }

    #[test]
    fn test_validate_qq_theoretical_quantiles() {
        use p2a_validation::{Check, Tol};
        let (case, _) = crate::test_support::load_case("viz_residual_diag");
        let mut check = Check::new(&case);
        for n in [1usize, 8, 11, 32] {
            // statrs' normal quantile agrees with R's qnorm to a few ulp
            check.vector(
                &format!("qq_ppoints_{n}"),
                &qq_theoretical_quantiles(n).unwrap(),
                Tol::new(1e-11, 1e-15),
            );
        }
        check.finish();
    }

    #[test]
    fn test_residual_diagnostics_rejects_bad_leverage() {
        use crate::data::columns::{NaPolicy, numeric_column};
        let df = crate::data::DataLoader::load_csv(p2a_validation::repo_path(
            "validation/datasets/generated/viz_residual_diag.csv",
        ))
        .unwrap();
        let col = |c: &str| numeric_column(&df, c, NaPolicy::Fail).unwrap().0;
        let (f, e, h) = (col("fitted3"), col("resid3"), col("hat3"));
        let refused = |fitted: &[f64],
                       resid: &[f64],
                       hat: Option<&[f64]>,
                       n_params: Option<usize>,
                       needles: &[&str]| {
            match residual_diagnostics_with(fitted, resid, hat, n_params, small()) {
                Err(VisualizationError::InvalidData(m)) => {
                    for needle in needles {
                        assert!(m.contains(needle), "{m}");
                    }
                }
                Err(other) => panic!("{other:?}"),
                Ok(_) => panic!("accepted: {needles:?}"),
            }
        };
        let with = |i: usize, v: f64| {
            let mut h = h.clone();
            h[i] = v;
            h
        };
        refused(
            &f,
            &e,
            Some(&with(3, f64::NAN)),
            None,
            &["leverage at row 3 is NaN"],
        );
        refused(
            &f,
            &e,
            Some(&with(0, 1.5)),
            None,
            &["row 0", "between 0 and 1"],
        );
        refused(&f, &e, Some(&with(5, -0.2)), None, &["row 5"]);
        let scaled: Vec<f64> = h.iter().map(|v| v * 1.1).collect();
        refused(&f, &e, Some(&scaled), None, &["sum to 4.4000"]);
        refused(&f, &e, Some(&h), Some(0), &["n_params is 0"]);
        refused(
            &f,
            &e,
            Some(&h),
            Some(32),
            &["below the number of observations (32)"],
        );
        refused(&f, &e, None, Some(4), &["needs the model's hat values"]);
        refused(&f, &vec![0.0; 32], Some(&h), None, &["perfect fit"]);
        refused(
            &f,
            &e,
            Some(&[1.0; 32]),
            Some(2),
            &[
                "every observation has leverage one",
                "nothing can be plotted",
            ],
        );
        // The warning about unit hat values stays short however many rows have them.
        let (f60, e60) = (
            wavy(60),
            wavy(60).iter().rev().map(|v| v - 30.0).collect::<Vec<_>>(),
        );
        let h60: Vec<f64> = (0..60).map(|i| if i < 30 { 1.0 } else { 0.1 }).collect();
        let r = residual_diagnostics_with(&f60, &e60, Some(&h60), Some(31), small()).unwrap();
        let w = r
            .warnings
            .iter()
            .find(|w| w.contains("leverage one"))
            .unwrap();
        assert!(w.contains("30 observations, first rows [0, 1, "), "{w}");
        assert!(w.contains(", 19]") && !w.contains("20"), "{w}");
        // A row dropped for its fitted value is not checked for its leverage.
        let mut f4 = f.clone();
        f4[4] = f64::NAN;
        let r =
            residual_diagnostics_with(&f4, &e, Some(&with(4, f64::NAN)), None, small()).unwrap();
        assert!(!r.rows.contains(&4));
        assert_eq!(r.rows.len(), 31);
        assert_eq!(r.n_params, 4);
    }

    /// Merge 0 joins samples 0 and 1; merge i joins the cluster built so far (n + i - 1) with
    /// sample i + 1.
    fn chain_linkage(n: usize) -> Vec<(usize, usize, f64, usize)> {
        let mut l = vec![(0, 1, 1.0, 2)];
        l.extend((1..n - 1).map(|i| (n + i - 1, i + 1, (i + 1) as f64, i + 2)));
        l
    }

    #[test]
    fn test_charts_accept_utf8_labels_and_reject_empty_inputs() {
        let cfg = || ChartConfig {
            title: Some("Größe_kg 収入合計額".into()),
            x_label: Some("収入合計額".into()),
            y_label: Some("R²".into()),
            ..small()
        };
        let x = wavy(30);
        let y: Vec<f64> = x.iter().rev().copied().collect();
        assert!(histogram(&x, None, cfg()).is_ok());
        assert!(scatter_plot(&x, &y, cfg()).is_ok());
        assert!(line_chart(&[("収入".into(), x.clone(), y.clone())], cfg()).is_ok());
        let b = box_plot(
            &[
                ("R²".into(), x.clone()),
                ("Größe".into(), wavy(7)),
                ("収入合計額".into(), wavy(12)),
            ],
            cfg(),
        )
        .unwrap();
        assert_eq!(
            b.statistics
                .iter()
                .map(|s| s.label.as_str())
                .collect::<Vec<_>>(),
            ["R²", "Größe", "収入合計額"]
        );
        let labels: Vec<String> = (0..25).map(|i| format!("収入合計額_ラベル{i}")).collect();
        assert!(dendrogram(&chain_linkage(4), Some(&labels[..4]), cfg()).is_ok()); // whole labels
        assert!(dendrogram(&chain_linkage(25), Some(&labels), cfg()).is_ok()); // > 20 leaves
        assert!(
            event_study_plot(
                &[-1.0, 0.0, 1.0],
                &[0.1, 0.0, 0.3],
                &[-0.2, -0.1, 0.0],
                &[0.4, 0.1, 0.6],
                cfg()
            )
            .is_ok()
        );
        // empty or all-missing inputs are errors
        assert!(histogram(&[f64::NAN; 3], None, cfg()).is_err());
        assert!(scatter_plot(&[f64::NAN; 2], &[1.0, 2.0], cfg()).is_err());
        assert!(box_plot(&[("a".into(), vec![f64::NAN])], cfg()).is_err());
        assert!(residual_diagnostics(&[f64::NAN; 3], &[0.1, 0.2, 0.3], None, cfg()).is_err());
    }

    #[test]
    fn test_dendrogram_leaf_order_chain_linear_memory() {
        // A leaf list per merge would hold n^2 / 2 indices here.
        let n = 100_000;
        let link = chain_linkage(n);
        validate_linkage(&link, n).unwrap();
        let order = compute_leaf_order(&link, n);
        assert_eq!(order, (0..n).collect::<Vec<_>>());
        let r = dendrogram(&chain_linkage(1_000), None, small()).unwrap();
        assert_eq!((r.n_samples, r.n_merges), (1_000, 999));
        assert_eq!(r.max_distance, 999.0);
    }

    #[test]
    fn test_dendrogram_rejects_bad_linkage() {
        for (bad, needle) in [
            (
                vec![(0usize, 2usize, 1.0f64, 2usize)], // the cluster this row creates
                "linkage row 0 refers to cluster 2",
            ),
            (
                vec![(0, 1, 1.0, 2), (usize::MAX, 2, 2.0, 3)],
                "linkage row 1 refers to cluster 18446744073709551615",
            ),
            (
                vec![(0, 1, 1.0, 2), (2, 2, 2.0, 4)],
                "linkage row 1 merges cluster 2 with itself",
            ),
            (
                vec![(0, 1, f64::INFINITY, 2)],
                "linkage row 0 has merge height inf",
            ),
        ] {
            match dendrogram(&bad, None, small()) {
                Err(VisualizationError::InvalidData(m)) => assert!(m.contains(needle), "{m}"),
                Err(other) => panic!("{bad:?}: {other:?}"),
                Ok(_) => panic!("{bad:?} was accepted"),
            }
        }
    }
}
